#include "kernel/guestWait.h"

#include "common/profiler.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::LibKernel::GuestWait {

namespace {

constexpr size_t PrimCount   = static_cast<size_t>(Prim::Count);
constexpr size_t EventCount  = static_cast<size_t>(Event::Count);
constexpr size_t SubmitCount = static_cast<size_t>(Submit::Count);

constexpr const char* PrimNames[PrimCount] = {
    "equeue",         "cond",          "mutex",          "rwlock",       "sema",       "event-flag",
    "sync-on-address", "sleep",        "yield",          "vblank-wait",  "suspend-point", "mem-lock",
    "gpu-sync",       "fault-stale",   "fault-clean-read", "fault-clean-lift", "fault-write", "fault-read", "fault-other",
    "readback-queue", "readback-exec", "readback-copy",  "readback-tick"};
constexpr const char* SubmitNames[SubmitCount] = {"gfx", "compute", "flip-prep", "suspend-point", "command"};
static_assert(PrimNames[PrimCount - 1] != nullptr && SubmitNames[SubmitCount - 1] != nullptr);

// The events a submission's turnaround is measured from: the global ones, then this thread's own
// readback.
constexpr size_t      SinceCount            = EventCount + 1;
constexpr size_t      SinceReadback         = EventCount;
constexpr const char* SinceNames[SinceCount] = {"label", "eop", "flip", "vblank", "readback"};
// A vblank comes every 16.7 ms whatever the game does: it is printed, never taken as the event
// a submission answers.
constexpr bool SinceCandidate[SinceCount] = {true, true, true, false, true};

constexpr size_t      HistCount             = 8;
constexpr uint64_t    HistLimitNs[HistCount] = {250000, 500000, 1000000, 2000000,
                                                4000000, 8000000, 16000000, UINT64_MAX};
constexpr const char* HistNames[HistCount]   = {"<0.25", "<0.5", "<1", "<2", "<4", "<8", "<16", ">=16"};

constexpr size_t ObjectSlots     = 512; // a power of two
constexpr size_t ObjectSlotsFull = 384;
constexpr size_t MaxThreads      = 256;

struct SubmitStats {
	std::atomic<uint64_t> count {0};
	// The latest candidate event before the submission: which it was, and how long before.
	std::atomic<uint64_t> latest_n[SinceCount] {};
	std::atomic<uint64_t> latest_ns[SinceCount] {};
	std::atomic<uint64_t> latest_max_ns {0};
	std::atomic<uint64_t> none {0}; // no event yet
	std::atomic<uint64_t> hist[HistCount] {};
	// Since each event, whichever was the latest.
	std::atomic<uint64_t> since_n[SinceCount] {};
	std::atomic<uint64_t> since_ns[SinceCount] {};
	// Since this thread last came back from an emulator wait.
	std::atomic<uint64_t> wake_n {0};
	std::atomic<uint64_t> wake_ns {0};
	std::atomic<uint32_t> wake_prim[PrimCount] {};
	// GPU-thread "empty" waits this ended: all of the wait, and its part before the latest event
	// (the GPU thread and the game both waited for it) and after it (the game's turnaround).
	std::atomic<uint64_t> empty_n {0};
	std::atomic<uint64_t> empty_ns {0};
	std::atomic<uint64_t> empty_before_ns {0};
};

} // namespace

namespace Detail {
struct ObjectSlot {
	std::atomic<uint64_t> key {0}; // prim << 56 | object address (48 bits); 0: free
	std::atomic<uint32_t> count {0};
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> max_ns {0};
	char                  name[24] {};
};
} // namespace Detail

namespace {

using Detail::ObjectSlot;

// One thread's counters. The owner adds, the report takes (exchange): no lock. The tables are
// made once, before main, when the switch is on: claiming one (in the fault handler too)
// allocates nothing.
struct ThreadTable {
	std::atomic<bool>     claimed {false};
	std::atomic<bool>     dirty {false}; // something was recorded since the last report
	std::atomic<uint32_t> guest_id {0};
	std::atomic<bool>     guest {false};
	char                  name[32] {};
	std::atomic<uint64_t> prim_n[PrimCount] {};
	std::atomic<uint64_t> prim_ns[PrimCount] {};
	std::atomic<uint64_t> prim_max_ns[PrimCount] {};
	std::atomic<uint32_t> objects_used {0};
	std::atomic<uint64_t> objects_dropped {0}; // scopes whose object found no slot
	ObjectSlot            objects[ObjectSlots];
	SubmitStats           submits[SubmitCount];
	// The owner's own.
	uint64_t child_ns         = 0; // of the scopes nested in the open one
	uint64_t last_wake_ns     = 0;
	Prim     last_wake_prim   = Prim::Count;
	uint64_t last_readback_ns = 0;
	uint32_t records          = 0;
	bool     suppress_gpu_sync = false;
};

ThreadTable*          g_tables = nullptr;
std::atomic<uint32_t> g_tables_used {0};
std::atomic<uint64_t> g_no_table {0}; // records lost: every table was taken

std::atomic<uint64_t> g_event_ns[EventCount] {};
std::atomic<uint64_t> g_event_n[EventCount] {};

// The GPU thread's wait for work (written under its queue mutex).
std::atomic<uint64_t> g_empty_since_ns {0};  // the wait in progress; 0: none, or its end was noted
uint64_t              g_empty_begin_ns = 0;  // the same, kept by the GPU thread for the total
std::atomic<uint64_t> g_empty_signal_ns {0}; // when the waker signalled
std::atomic<uint64_t> g_empty_waits {0};
std::atomic<uint64_t> g_empty_total_ns {0};
std::atomic<uint64_t> g_empty_latency_ns {0};  // signal -> the GPU thread running again
std::atomic<uint64_t> g_empty_unnoted_ns {0};  // waits that ended without a noted waker
std::atomic<uint64_t> g_empty_unnoted_n {0};

std::atomic<uint64_t> g_frames {0};
std::atomic<uint64_t> g_report_ns {0};

void Add(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
	counter.fetch_add(value, std::memory_order_relaxed);
}
uint64_t Take(std::atomic<uint64_t>& counter) noexcept {
	return counter.exchange(0, std::memory_order_relaxed);
}
// Owner only.
void Max(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
	if (value > counter.load(std::memory_order_relaxed)) {
		counter.store(value, std::memory_order_relaxed);
	}
}

void NameTable(ThreadTable& table) noexcept {
	char name[64] = "";
	bool guest    = false;
	if (auto self = PthreadSelfOrNull(); self != nullptr) {
		guest = true;
		if (PthreadGetname(self, name) != 0) {
			name[0] = '\0';
		}
	} else {
		std::snprintf(name, sizeof(name), "%s", Profiler::t_gpu_thread ? "(GPU thread)" : "(host thread)");
	}
	name[sizeof(table.name) - 1] = '\0';
	std::memcpy(table.name, name, sizeof(table.name));
	table.guest_id.store(static_cast<uint32_t>(PthreadGetthreadid()), std::memory_order_relaxed);
	table.guest.store(guest, std::memory_order_relaxed);
}

thread_local ThreadTable* t_table = nullptr;
// Gives the table back when the thread ends (its counts stay for the next report).
struct TableHolder {
	ThreadTable* table = nullptr;
	~TableHolder() {
		if (table != nullptr) {
			t_table = nullptr;
			table->claimed.store(false, std::memory_order_release);
		}
	}
};
thread_local TableHolder t_holder;

ThreadTable* ClaimTable() noexcept {
	const auto used = std::min<uint32_t>(g_tables_used.load(std::memory_order_acquire), MaxThreads);
	for (uint32_t pass = 0; pass < 3; pass++) {
		// A table an ended thread gave back, once the report has taken its counts; then a new
		// one; then any free one (its last owner's counts of this window go to the new owner).
		const uint32_t begin = pass == 1 ? used : 0;
		const uint32_t end   = pass == 0 ? used : MaxThreads;
		for (uint32_t i = begin; i < end; i++) {
			if (pass == 0 && g_tables[i].dirty.load(std::memory_order_relaxed)) {
				continue;
			}
			bool expected = false;
			if (g_tables[i].claimed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
				auto& table             = g_tables[i];
				table.child_ns          = 0;
				table.last_wake_ns      = 0;
				table.last_wake_prim    = Prim::Count;
				table.last_readback_ns  = 0;
				table.records           = 0;
				table.suppress_gpu_sync = false;
				NameTable(table);
				auto current = g_tables_used.load(std::memory_order_relaxed);
				while (current < i + 1 &&
				       !g_tables_used.compare_exchange_weak(current, i + 1, std::memory_order_acq_rel)) {
				}
				return &table;
			}
		}
	}
	return nullptr;
}

ThreadTable* Table() noexcept {
	auto* table = t_table;
	if (table == nullptr) [[unlikely]] {
		table = ClaimTable();
		if (table == nullptr) {
			g_no_table.fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		t_table        = table;
		t_holder.table = table;
	}
	return table;
}

void Touch(ThreadTable& table) noexcept {
	if (!table.dirty.load(std::memory_order_relaxed)) {
		table.dirty.store(true, std::memory_order_relaxed);
	}
	// A thread named after its first wait gets its name here.
	if ((++table.records & 1023u) == 0) {
		NameTable(table);
	}
}

ObjectSlot* FindObject(ThreadTable& table, uint64_t key, const char* name) noexcept {
	auto index = static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> 40u) & (ObjectSlots - 1u);
	for (size_t probe = 0; probe < 32; probe++, index = (index + 1u) & (ObjectSlots - 1u)) {
		auto&      slot    = table.objects[index];
		const auto current = slot.key.load(std::memory_order_relaxed);
		if (current == key) {
			return &slot;
		}
		if (current == 0) {
			if (table.objects_used.load(std::memory_order_relaxed) >= ObjectSlotsFull) {
				return nullptr;
			}
			table.objects_used.fetch_add(1, std::memory_order_relaxed);
			std::memset(slot.name, 0, sizeof(slot.name));
			if (name != nullptr) {
				std::strncpy(slot.name, name, sizeof(slot.name) - 1);
			}
			slot.count.store(0, std::memory_order_relaxed);
			slot.ns.store(0, std::memory_order_relaxed);
			slot.max_ns.store(0, std::memory_order_relaxed);
			slot.key.store(key, std::memory_order_release);
			return &slot;
		}
	}
	return nullptr;
}

bool EnvOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

size_t TopCount() {
	static const size_t top = [] {
		const char* value = std::getenv("KYTY_GUEST_WAIT_TOP");
		const auto  count = value != nullptr ? std::strtoull(value, nullptr, 10) : 12u;
		return static_cast<size_t>(std::clamp<unsigned long long>(count, 1, 64));
	}();
	return top;
}

std::string ThreadLabel(const ThreadTable& table) {
	char name[sizeof(table.name)];
	std::memcpy(name, table.name, sizeof(name));
	name[sizeof(name) - 1] = '\0';
	char text[80];
	if (table.guest.load(std::memory_order_relaxed)) {
		std::snprintf(text, sizeof(text), "thread %u \"%s\"", table.guest_id.load(std::memory_order_relaxed), name);
	} else {
		std::snprintf(text, sizeof(text), "%s", name);
	}
	return text;
}

void Report(double seconds, uint64_t frames) {
	const double per_frame = frames != 0 ? 1.0 / static_cast<double>(frames) : 0.0;
	const auto   ms_f      = [per_frame](uint64_t ns) { return static_cast<double>(ns) * 1e-6 * per_frame; };
	const auto   n_f       = [per_frame](uint64_t n) { return static_cast<double>(n) * per_frame; };
	const size_t top       = TopCount();
	const auto   used      = std::min<uint32_t>(g_tables_used.load(std::memory_order_acquire), MaxThreads);

	struct ThreadRow {
		const ThreadTable* table;
		uint64_t           n[PrimCount];
		uint64_t           ns[PrimCount];
		uint64_t           total_ns;
	};
	struct ObjectRow {
		const ThreadTable* table;
		Prim               prim;
		uint64_t           address;
		char               name[24];
		uint64_t           count, ns, max_ns;
	};
	struct SubmitRow {
		const ThreadTable* table;
		size_t             kind;
		uint64_t           count, latest_n[SinceCount], latest_ns[SinceCount], latest_max_ns, none, hist[HistCount];
		uint64_t           since_n[SinceCount], since_ns[SinceCount], wake_n, wake_ns;
		uint32_t           wake_prim[PrimCount];
		uint64_t           empty_n, empty_ns, empty_before_ns;
	};
	std::vector<ThreadRow> threads;
	std::vector<ObjectRow> objects;
	std::vector<SubmitRow> submits;
	uint64_t               guest_total_ns = 0, host_total_ns = 0, objects_dropped = 0;
	uint64_t               prim_all_ns[PrimCount] {}, prim_all_n[PrimCount] {};
	for (uint32_t t = 0; t < used; t++) {
		auto& table = g_tables[t];
		if (!table.dirty.exchange(false, std::memory_order_relaxed)) {
			continue;
		}
		ThreadRow row {&table, {}, {}, 0};
		for (size_t p = 0; p < PrimCount; p++) {
			row.n[p]  = Take(table.prim_n[p]);
			row.ns[p] = Take(table.prim_ns[p]);
			(void)Take(table.prim_max_ns[p]);
			row.total_ns += row.ns[p];
			if (table.guest.load(std::memory_order_relaxed)) {
				prim_all_ns[p] += row.ns[p];
				prim_all_n[p] += row.n[p];
			}
		}
		(table.guest.load(std::memory_order_relaxed) ? guest_total_ns : host_total_ns) += row.total_ns;
		if (row.total_ns != 0) {
			threads.push_back(row);
		}
		objects_dropped += Take(table.objects_dropped);
		const bool full = table.objects_used.load(std::memory_order_relaxed) >= ObjectSlotsFull / 2u;
		for (auto& slot: table.objects) {
			const auto key = slot.key.load(std::memory_order_acquire);
			if (key == 0) {
				continue;
			}
			const auto count = slot.count.exchange(0, std::memory_order_relaxed);
			ObjectRow  object {&table, static_cast<Prim>(key >> 56u), key & ((uint64_t {1} << 56u) - 1u), {}, count,
			                   Take(slot.ns), Take(slot.max_ns)};
			std::memcpy(object.name, slot.name, sizeof(object.name));
			object.name[sizeof(object.name) - 1] = '\0';
			if (count != 0) {
				objects.push_back(object);
			}
			// A table half full starts again (objects come and go); a scope open across this
			// finds its slot's key changed and drops its item (the primitive's total keeps it).
			if (full) {
				slot.key.store(0, std::memory_order_release);
			}
		}
		if (full) {
			table.objects_used.store(0, std::memory_order_relaxed);
		}
		for (size_t k = 0; k < SubmitCount; k++) {
			auto&     s = table.submits[k];
			SubmitRow sr {};
			sr.table = &table;
			sr.kind  = k;
			sr.count = Take(s.count);
			for (size_t i = 0; i < SinceCount; i++) {
				sr.latest_n[i]  = Take(s.latest_n[i]);
				sr.latest_ns[i] = Take(s.latest_ns[i]);
				sr.since_n[i]   = Take(s.since_n[i]);
				sr.since_ns[i]  = Take(s.since_ns[i]);
			}
			sr.latest_max_ns = Take(s.latest_max_ns);
			sr.none          = Take(s.none);
			for (size_t i = 0; i < HistCount; i++) {
				sr.hist[i] = Take(s.hist[i]);
			}
			sr.wake_n  = Take(s.wake_n);
			sr.wake_ns = Take(s.wake_ns);
			for (size_t p = 0; p < PrimCount; p++) {
				sr.wake_prim[p] = s.wake_prim[p].exchange(0, std::memory_order_relaxed);
			}
			sr.empty_n         = Take(s.empty_n);
			sr.empty_ns        = Take(s.empty_ns);
			sr.empty_before_ns = Take(s.empty_before_ns);
			if (sr.count != 0) {
				submits.push_back(sr);
			}
		}
	}

	std::string text;
	char        line[1024];
	// [guest-wait]
	std::snprintf(line, sizeof(line),
	              "[guest-wait] %.1f s, %llu frames (%.1f fps): ms per frame blocked inside the emulator, self "
	              "times | all guest threads %.2f, host threads %.2f | %zu threads",
	              seconds, static_cast<unsigned long long>(frames), static_cast<double>(frames) / seconds,
	              ms_f(guest_total_ns), ms_f(host_total_ns), threads.size());
	text += line;
	if (const auto lost = Take(g_no_table); lost != 0) {
		std::snprintf(line, sizeof(line), " | %llu records lost (over %zu threads)",
		              static_cast<unsigned long long>(lost), MaxThreads);
		text += line;
	}
	text += "\n[guest-wait]   all guest threads by primitive (ms/f, calls/f):";
	{
		std::array<size_t, PrimCount> order {};
		for (size_t p = 0; p < PrimCount; p++) {
			order[p] = p;
		}
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return prim_all_ns[a] > prim_all_ns[b]; });
		for (const auto p: order) {
			if (prim_all_n[p] == 0) {
				continue;
			}
			std::snprintf(line, sizeof(line), " %s %.2f (%.1f) |", PrimNames[p], ms_f(prim_all_ns[p]),
			              n_f(prim_all_n[p]));
			text += line;
		}
	}
	text += "\n";
	std::sort(threads.begin(), threads.end(),
	          [](const ThreadRow& a, const ThreadRow& b) { return a.total_ns > b.total_ns; });
	uint64_t rest_ns = 0;
	for (size_t i = 0; i < threads.size(); i++) {
		const auto& row = threads[i];
		if (i >= top) {
			rest_ns += row.total_ns;
			continue;
		}
		std::snprintf(line, sizeof(line), "[guest-wait]   %s: %.2f ms/f |", ThreadLabel(*row.table).c_str(),
		              ms_f(row.total_ns));
		text += line;
		std::array<size_t, PrimCount> order {};
		for (size_t p = 0; p < PrimCount; p++) {
			order[p] = p;
		}
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return row.ns[a] > row.ns[b]; });
		for (const auto p: order) {
			if (row.n[p] == 0) {
				continue;
			}
			std::snprintf(line, sizeof(line), " %s %.2f (%.1f) |", PrimNames[p], ms_f(row.ns[p]), n_f(row.n[p]));
			text += line;
		}
		text += "\n";
	}
	if (threads.size() > top) {
		std::snprintf(line, sizeof(line), "[guest-wait]   + %zu more threads: %.2f ms/f\n", threads.size() - top,
		              ms_f(rest_ns));
		text += line;
	}
	std::sort(objects.begin(), objects.end(), [](const ObjectRow& a, const ObjectRow& b) { return a.ns > b.ns; });
	for (size_t i = 0; i < objects.size() && i < top; i++) {
		const auto& object = objects[i];
		std::snprintf(line, sizeof(line),
		              "[guest-wait]   #%zu %s %s \"%s\" 0x%llx: %.2f ms/f, %.1f/f, avg %.1f us, max %.2f ms\n", i + 1,
		              ThreadLabel(*object.table).c_str(), PrimNames[static_cast<size_t>(object.prim) % PrimCount],
		              object.name, static_cast<unsigned long long>(object.address), ms_f(object.ns),
		              n_f(object.count), static_cast<double>(object.ns) * 1e-3 / static_cast<double>(object.count),
		              static_cast<double>(object.max_ns) * 1e-6);
		text += line;
	}
	if (objects_dropped != 0) {
		std::snprintf(line, sizeof(line),
		              "[guest-wait]   %llu waits not itemised (a thread's object table was full; the totals have them)\n",
		              static_cast<unsigned long long>(objects_dropped));
		text += line;
	}

	// [guest-submit]
	const uint64_t empty_waits   = Take(g_empty_waits);
	const uint64_t empty_total   = Take(g_empty_total_ns);
	const uint64_t empty_latency = Take(g_empty_latency_ns);
	const uint64_t unnoted_ns    = Take(g_empty_unnoted_ns);
	const uint64_t unnoted_n     = Take(g_empty_unnoted_n);
	std::snprintf(line, sizeof(line),
	              "[guest-submit] %.1f s, %llu frames: events/f label %.1f, eop %.1f, flip %.1f, vblank %.1f | GPU "
	              "thread empty %.2f ms/f in %.1f waits/f: wake-up after the signal %.2f, ended by nothing noted "
	              "%.2f (%.1f/f), the rest by the submissions and commands below\n",
	              seconds, static_cast<unsigned long long>(frames), n_f(Take(g_event_n[0])), n_f(Take(g_event_n[1])),
	              n_f(Take(g_event_n[2])), n_f(Take(g_event_n[3])), ms_f(empty_total), n_f(empty_waits),
	              ms_f(empty_latency), ms_f(unnoted_ns), n_f(unnoted_n));
	text += line;
	std::sort(submits.begin(), submits.end(), [](const SubmitRow& a, const SubmitRow& b) {
		return a.empty_ns != b.empty_ns ? a.empty_ns > b.empty_ns : a.count > b.count;
	});
	uint64_t rest_empty_ns = 0, rest_count = 0;
	for (size_t i = 0; i < submits.size(); i++) {
		const auto& s = submits[i];
		if (i >= top) {
			rest_empty_ns += s.empty_ns;
			rest_count += s.count;
			continue;
		}
		std::snprintf(line, sizeof(line), "[guest-submit]   %s %s: %.2f/f | ended GPU-thread empty waits %.2f/f, "
		                                  "%.3f ms/f = %.3f before the event + %.3f after it",
		              ThreadLabel(*s.table).c_str(), SubmitNames[s.kind], n_f(s.count), n_f(s.empty_n),
		              ms_f(s.empty_ns), ms_f(s.empty_before_ns), ms_f(s.empty_ns - (std::min)(s.empty_ns, s.empty_before_ns)));
		text += line;
		if (s.kind != static_cast<size_t>(Submit::Command)) {
			uint64_t latest_n = 0, latest_ns = 0;
			for (size_t e = 0; e < SinceCount; e++) {
				latest_n += s.latest_n[e];
				latest_ns += s.latest_ns[e];
			}
			std::snprintf(line, sizeof(line), " | event->submit avg %.3f ms, max %.2f (the latest event was",
			              latest_n != 0 ? static_cast<double>(latest_ns) * 1e-6 / static_cast<double>(latest_n) : 0.0,
			              static_cast<double>(s.latest_max_ns) * 1e-6);
			text += line;
			for (size_t e = 0; e < SinceCount; e++) {
				if (s.latest_n[e] == 0) {
					continue;
				}
				std::snprintf(line, sizeof(line), " %s %.0f%% avg %.3f ms;", SinceNames[e],
				              100.0 * static_cast<double>(s.latest_n[e]) / static_cast<double>(s.count),
				              static_cast<double>(s.latest_ns[e]) * 1e-6 / static_cast<double>(s.latest_n[e]));
				text += line;
			}
			if (s.none != 0) {
				std::snprintf(line, sizeof(line), " none yet %.0f%%;",
				              100.0 * static_cast<double>(s.none) / static_cast<double>(s.count));
				text += line;
			}
			text += ") | event->submit ms:";
			for (size_t h = 0; h < HistCount; h++) {
				std::snprintf(line, sizeof(line), " %s:%llu", HistNames[h], static_cast<unsigned long long>(s.hist[h]));
				text += line;
			}
			text += " | avg ms since";
			for (size_t e = 0; e < SinceCount; e++) {
				if (s.since_n[e] == 0) {
					continue;
				}
				std::snprintf(line, sizeof(line), " %s %.2f", SinceNames[e],
				              static_cast<double>(s.since_ns[e]) * 1e-6 / static_cast<double>(s.since_n[e]));
				text += line;
			}
			std::snprintf(line, sizeof(line), " | wake->submit avg %.3f ms (last wait:",
			              s.wake_n != 0 ? static_cast<double>(s.wake_ns) * 1e-6 / static_cast<double>(s.wake_n) : 0.0);
			text += line;
			for (size_t p = 0; p < PrimCount; p++) {
				if (s.wake_prim[p] == 0) {
					continue;
				}
				std::snprintf(line, sizeof(line), " %s %.0f%%", PrimNames[p],
				              100.0 * static_cast<double>(s.wake_prim[p]) / static_cast<double>(s.count));
				text += line;
			}
			text += ")";
		}
		text += "\n";
	}
	if (submits.size() > top) {
		std::snprintf(line, sizeof(line), "[guest-submit]   + %zu more rows: %.2f/f, ended empty waits %.3f ms/f\n",
		              submits.size() - top, n_f(rest_count), ms_f(rest_empty_ns));
		text += line;
	}
	std::fputs(text.c_str(), stderr);
}

} // namespace

namespace Detail {

bool g_enabled = [] {
	if (!EnvOn("KYTY_GUEST_WAIT_STATS")) {
		return false;
	}
	g_tables = new ThreadTable[MaxThreads]();
	return true;
}();

void BeginSlow(Prim prim, const volatile void* object, const char* name, uint64_t& begin, uint64_t& saved_child,
               ObjectSlot*& slot, uint64_t& key) noexcept {
	auto* table = Table();
	if (table == nullptr) {
		return;
	}
	if (object != nullptr) {
		key  = (static_cast<uint64_t>(prim) << 56u) | (reinterpret_cast<uintptr_t>(object) & ((uint64_t {1} << 56u) - 1u));
		slot = FindObject(*table, key, name);
		if (slot == nullptr) {
			table->objects_dropped.fetch_add(1, std::memory_order_relaxed);
		}
	}
	saved_child     = table->child_ns;
	table->child_ns = 0;
	begin           = NowNs();
}

void EndSlow(Prim prim, uint64_t begin, uint64_t saved_child, ObjectSlot* slot, uint64_t key) noexcept {
	const auto now   = NowNs();
	auto*      table = t_table;
	if (table == nullptr || prim >= Prim::Count) {
		return;
	}
	const auto total = now > begin ? now - begin : 0;
	const auto child = table->child_ns;
	const auto self  = total > child ? total - child : 0;
	const auto p     = static_cast<size_t>(prim);
	Add(table->prim_n[p], 1);
	Add(table->prim_ns[p], self);
	Max(table->prim_max_ns[p], self);
	if (slot != nullptr && slot->key.load(std::memory_order_relaxed) == key) {
		slot->count.fetch_add(1, std::memory_order_relaxed);
		Add(slot->ns, self);
		Max(slot->max_ns, self);
	}
	table->child_ns       = saved_child + total;
	table->last_wake_ns   = now;
	table->last_wake_prim = prim;
	Touch(*table);
}

void NoteEventSlow(Event event) noexcept {
	const auto e = static_cast<size_t>(event);
	g_event_ns[e].store(NowNs(), std::memory_order_relaxed);
	g_event_n[e].fetch_add(1, std::memory_order_relaxed);
}

void NoteSubmitSlow(Submit kind, bool wakes_gpu_thread) noexcept {
	const auto now   = NowNs();
	auto*      table = Table();
	// The GPU thread's wait this ends: taken whether or not the thread has a table.
	uint64_t empty_since = 0;
	if (wakes_gpu_thread) {
		empty_since = g_empty_since_ns.exchange(0, std::memory_order_relaxed);
		if (empty_since != 0) {
			g_empty_signal_ns.store(now, std::memory_order_relaxed);
		}
	}
	if (table == nullptr) {
		if (empty_since != 0 && now > empty_since) {
			Add(g_empty_unnoted_ns, now - empty_since);
			Add(g_empty_unnoted_n, 1);
		}
		return;
	}
	auto& s = table->submits[static_cast<size_t>(kind)];
	Add(s.count, 1);
	uint64_t latest_ns = 0;
	if (kind != Submit::Command) {
		size_t latest = SinceCount;
		for (size_t e = 0; e < SinceCount; e++) {
			const auto at = e == SinceReadback ? table->last_readback_ns : g_event_ns[e].load(std::memory_order_relaxed);
			if (at == 0 || at > now) {
				continue;
			}
			Add(s.since_n[e], 1);
			Add(s.since_ns[e], now - at);
			if (SinceCandidate[e] && at > latest_ns) {
				latest_ns = at;
				latest    = e;
			}
		}
		if (latest == SinceCount) {
			Add(s.none, 1);
		} else {
			const auto since = now - latest_ns;
			Add(s.latest_n[latest], 1);
			Add(s.latest_ns[latest], since);
			Max(s.latest_max_ns, since);
			for (size_t h = 0; h < HistCount; h++) {
				if (since < HistLimitNs[h]) {
					Add(s.hist[h], 1);
					break;
				}
			}
		}
		if (table->last_wake_ns != 0 && table->last_wake_prim < Prim::Count && now >= table->last_wake_ns) {
			Add(s.wake_n, 1);
			Add(s.wake_ns, now - table->last_wake_ns);
			s.wake_prim[static_cast<size_t>(table->last_wake_prim)].fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (empty_since != 0 && now > empty_since) {
		Add(s.empty_n, 1);
		Add(s.empty_ns, now - empty_since);
		if (latest_ns > empty_since) {
			Add(s.empty_before_ns, latest_ns - empty_since);
		}
	}
	Touch(*table);
}

void GpuEmptyBeginSlow() noexcept {
	g_empty_begin_ns = NowNs();
	g_empty_signal_ns.store(0, std::memory_order_relaxed);
	g_empty_since_ns.store(g_empty_begin_ns, std::memory_order_relaxed);
}

void GpuEmptyEndSlow() noexcept {
	const auto now    = NowNs();
	const auto since  = g_empty_since_ns.exchange(0, std::memory_order_relaxed);
	const auto signal = g_empty_signal_ns.exchange(0, std::memory_order_relaxed);
	Add(g_empty_waits, 1);
	if (g_empty_begin_ns != 0 && now > g_empty_begin_ns) {
		Add(g_empty_total_ns, now - g_empty_begin_ns);
	}
	g_empty_begin_ns = 0;
	if (since != 0) {
		// Nothing noted its end: a timed wait ran out, the other role's signal, a shutdown.
		if (now > since) {
			Add(g_empty_unnoted_ns, now - since);
		}
		Add(g_empty_unnoted_n, 1);
	} else if (signal != 0 && now > signal) {
		Add(g_empty_latency_ns, now - signal);
	}
}

void OnGuestFlipSlow() {
	g_frames.fetch_add(1, std::memory_order_relaxed);
	const auto now  = NowNs();
	auto       last = g_report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_report_ns.compare_exchange_strong(last, now);
		return;
	}
	if (now - last < 5000000000ull || !g_report_ns.compare_exchange_strong(last, now)) {
		return;
	}
	Report(static_cast<double>(now - last) * 1e-9, g_frames.exchange(0, std::memory_order_relaxed));
}

} // namespace Detail

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void AddTime(Prim prim, uint64_t ns) noexcept {
	auto* table = Table();
	if (table == nullptr || prim >= Prim::Count) {
		return;
	}
	const auto p = static_cast<size_t>(prim);
	Add(table->prim_n[p], 1);
	Add(table->prim_ns[p], ns);
	Max(table->prim_max_ns[p], ns);
	table->child_ns += ns;
	Touch(*table);
}

void BeginGpuSync(Scope& scope) noexcept {
	const auto* table = Table();
	if (table != nullptr && !table->suppress_gpu_sync) {
		scope.Begin(Prim::GpuSync);
	}
}

SuppressGpuSync::SuppressGpuSync(bool on) noexcept: m_on(on) {
	if (m_on) {
		if (auto* table = Table(); table != nullptr) {
			m_previous               = table->suppress_gpu_sync;
			table->suppress_gpu_sync = true;
		} else {
			m_on = false;
		}
	}
}

SuppressGpuSync::~SuppressGpuSync() {
	if (m_on && t_table != nullptr) {
		t_table->suppress_gpu_sync = m_previous;
	}
}

void NoteReadbackEnd() noexcept {
	if (auto* table = Table(); table != nullptr) {
		table->last_readback_ns = NowNs();
	}
}

} // namespace Libs::LibKernel::GuestWait
