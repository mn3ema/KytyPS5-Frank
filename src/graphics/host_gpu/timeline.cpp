#include "graphics/host_gpu/timeline.h"

#include "common/logging/log.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include "common/profiler.h"
#include <string_view>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::Timeline {

namespace {

// KYTY_TIMELINE=<first frame>:<frames>, or t<seconds>:<frames> to open at the first present
// that many seconds after the emulator started.
struct Window {
	uint64_t first   = 0;
	uint64_t count   = 0;
	uint64_t seconds = 0;
};

const auto g_process_start = std::chrono::steady_clock::now();

const Window& GetWindow() {
	static const Window window = [] {
		Window result;
		const char* value = std::getenv("KYTY_TIMELINE");
		if (value != nullptr) {
			char* end = nullptr;
			if (value[0] == 't') {
				result.seconds = std::strtoull(value + 1, &end, 10);
				result.first   = UINT64_MAX;
			} else {
				result.first = std::strtoull(value, &end, 10);
			}
			result.count = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 1;
		}
		return result;
	}();
	return window;
}

uint64_t NowMicros() {
	static const auto start = std::chrono::steady_clock::now();
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now() - start)
	                                 .count());
}

uint32_t ThreadId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return static_cast<uint32_t>(GetCurrentThreadId());
#else
	return 0;
#endif
}

// Events are kept in memory while the window is open and written out when it closes: written as
// they happened (unbuffered stderr), the marks slowed the frames they measured ~10x.
struct Event {
	uint64_t    micros;
	uint32_t    thread;
	const char* name;
	uint64_t    a;
	uint64_t    b;
};

constexpr size_t MaxEvents = size_t {1} << 21u;

struct Store {
	std::unique_ptr<Event[]>                       events {new Event[MaxEvents]};
	std::atomic<uint64_t>                          next {0};
	std::mutex                                     names_mutex;
	std::vector<std::pair<uint32_t, std::string>> names;
};

Store& GetStore() {
	static Store store;
	return store;
}

void Flush() {
	auto&      store = GetStore();
	const auto count = std::min<uint64_t>(store.next.exchange(0), MaxEvents);
	std::string text;
	text.reserve(1u << 20u);
	{
		std::lock_guard lock(store.names_mutex);
		for (const auto& [thread, name]: store.names) {
			text += "TLN " + std::to_string(thread) + " " + name + "\n";
		}
	}
	char line[192];
	for (uint64_t i = 0; i < count; i++) {
		const auto& e = store.events[i];
		std::snprintf(line, sizeof(line), "TL %" PRIu64 " %u %s 0x%" PRIx64 " 0x%" PRIx64 "\n",
		              e.micros, e.thread, e.name, e.a, e.b);
		text += line;
		if (text.size() > (1u << 20u)) {
			std::fwrite(text.data(), 1, text.size(), stderr);
			text.clear();
		}
	}
	std::fwrite(text.data(), 1, text.size(), stderr);
	std::fprintf(stderr, "TLEND %" PRIu64 " events\n", count);
	std::fflush(stderr);
}

} // namespace

namespace Detail {

void MarkSlow(const char* event, uint64_t a, uint64_t b) {
	auto& store = GetStore();
	// Each thread names itself once (guest thread name, or host).
	thread_local bool named = false;
	if (!named) {
		named         = true;
		char name[64] = "(host)";
		if (auto self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
			if (Libs::LibKernel::PthreadGetname(self, name) != 0) {
				std::snprintf(name, sizeof(name), "(guest)");
			}
		}
		std::lock_guard lock(store.names_mutex);
		store.names.emplace_back(ThreadId(), name);
	}
	const auto index = store.next.fetch_add(1, std::memory_order_relaxed);
	if (index < MaxEvents) {
		store.events[index] = {NowMicros(), ThreadId(), event, a, b};
	}
}

} // namespace Detail

void Present(uint64_t frame) {
	const auto& window = GetWindow();
	if (window.count == 0) {
		return;
	}
	static uint64_t first = window.first;
	if (first == UINT64_MAX &&
	    std::chrono::steady_clock::now() - g_process_start >= std::chrono::seconds(window.seconds)) {
		first = frame;
	}
	const bool active      = first != UINT64_MAX && frame >= first && frame - first < window.count;
	const bool was_active  = Detail::g_active.load(std::memory_order_relaxed);
	if (active || was_active) {
		Detail::MarkSlow("present", frame, 0);
	}
	Detail::g_active.store(active, std::memory_order_relaxed);
	if (was_active && !active) {
		Flush();
	}
}

} // namespace Libs::Graphics::Timeline

namespace Libs::Graphics::CpuCost {

namespace {

int64_t HitchCostMs() {
	static const int64_t ms = [] {
		const char* value = std::getenv("KYTY_HITCH_COST_MS");
		return value != nullptr ? static_cast<int64_t>(std::strtoll(value, nullptr, 10)) : int64_t {0};
	}();
	return ms;
}

// The guest frame count and the last frame's length, written by the flip (the presenter's
// thread) and read by the recording threads.
std::atomic<uint64_t> g_frame_count {0};
std::atomic<uint64_t> g_frame_us {0};
std::atomic<uint32_t> g_hitches_printed {0};

// The calling thread's user and kernel CPU time so far, in ns.
void ThreadCpuNs(uint64_t out[2]) {
	out[0] = out[1] = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FILETIME creation {}, exited {}, kernel {}, user {};
	if (GetThreadTimes(GetCurrentThread(), &creation, &exited, &kernel, &user) != 0) {
		out[0] = ((static_cast<uint64_t>(user.dwHighDateTime) << 32u) | user.dwLowDateTime) * 100u;
		out[1] = ((static_cast<uint64_t>(kernel.dwHighDateTime) << 32u) | kernel.dwLowDateTime) * 100u;
	}
#else
	timespec ts {};
	if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
		out[0] = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
	}
#endif
}

struct Row {
	uint64_t count = 0;
	uint64_t ticks = 0;
	uint64_t self  = 0;
};
using RowKey = std::pair<char, std::string_view>;

// [hitch-cost]: the frame that just ended, from the frame fields, which are then cleared.
void FrameReport(Detail::ThreadTable& table, uint64_t frame_count, double ns_per_tick) {
	table.frame_seen     = frame_count;
	const double frame_ms = static_cast<double>(g_frame_us.load(std::memory_order_relaxed)) / 1e3;
	std::string  text;
	if (frame_ms >= static_cast<double>(HitchCostMs()) &&
	    g_hitches_printed.fetch_add(1, std::memory_order_relaxed) < 400u) {
		// One literal can have several addresses (one per file that names the phase): merged here.
		std::map<RowKey, Row> merged;
		for (const auto& slot: table.phases) {
			if (slot.name != nullptr && slot.frame_count != 0) {
				auto& m = merged[{slot.kind, std::string_view(slot.name)}];
				m.count += slot.frame_count;
				m.self  += slot.frame_self;
			}
		}
		std::vector<std::pair<RowKey, Row>> self(merged.begin(), merged.end());
		std::sort(self.begin(), self.end(), [](const auto& a, const auto& b) { return a.second.self > b.second.self; });
		uint64_t total = 0;
		for (const auto& entry: self) {
			total += entry.second.self;
		}
		const double ms_per_tick = ns_per_tick / 1e6;
		char         line[224];
		std::snprintf(line, sizeof(line), "[hitch-cost] frame %.0f ms: GPU-thread phases %.0f ms (self time, ms x calls):",
		              frame_ms, static_cast<double>(total) * ms_per_tick);
		text += line;
		for (size_t i = 0; i < self.size() && i < 16 && static_cast<double>(self[i].second.self) * ms_per_tick >= 0.5;
		     i++) {
			std::snprintf(line, sizeof(line), " %c %.*s %.1f x%llu |", self[i].first.first,
			              static_cast<int>(self[i].first.second.size()), self[i].first.second.data(),
			              static_cast<double>(self[i].second.self) * ms_per_tick,
			              static_cast<unsigned long long>(self[i].second.count));
			text += line;
		}
		text += "\n";
		std::vector<Detail::CostSlot> list;
		for (const auto& slot: table.costs) {
			if (slot.key != 0 && slot.frame_count != 0) {
				list.push_back(slot);
			}
		}
		std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.frame_ticks > b.frame_ticks; });
		text += "[hitch-cost]   heaviest draws (A..) and dispatches (X, I) by shader:";
		for (size_t i = 0; i < list.size() && i < 10 && static_cast<double>(list[i].frame_ticks) * ms_per_tick >= 0.5;
		     i++) {
			std::snprintf(line, sizeof(line), " %c %014llx %.1f x%llu |", static_cast<char>(list[i].key >> 56u),
			              static_cast<unsigned long long>(list[i].key & ((uint64_t {1} << 56u) - 1u)),
			              static_cast<double>(list[i].frame_ticks) * ms_per_tick,
			              static_cast<unsigned long long>(list[i].frame_count));
			text += line;
		}
		text += "\n";
	}
	for (auto& slot: table.phases) {
		slot.frame_count = slot.frame_self = 0;
	}
	table.phase_overflow.frame_count = table.phase_overflow.frame_self = 0;
	for (auto& slot: table.costs) {
		slot.frame_count = slot.frame_ticks = 0;
	}
	if (!text.empty()) {
		std::fputs(text.c_str(), stderr);
	}
}

// ns per tick of the window so far (the time-stamp counter against the steady clock).
double NsPerTick(const Detail::ThreadTable& table, std::chrono::steady_clock::time_point now, uint64_t ticks_now) {
	const auto ticks = ticks_now - table.window_ticks;
	const auto ns    = std::chrono::duration<double, std::nano>(now - table.window_time).count();
	return ticks != 0 && ns > 0.0 ? ns / static_cast<double>(ticks) : 1.0;
}

} // namespace

namespace Detail {

ThreadTable* CreateTable() {
	auto* table        = new ThreadTable(); // one per recording thread, never freed
	table->window_time  = std::chrono::steady_clock::now();
	table->window_ticks = Ticks();
	table->gpu_thread   = OnGpuThread();
	table->frame_seen   = g_frame_count.load(std::memory_order_relaxed);
	ThreadCpuNs(table->window_cpu_ns);
	t_table = table;
	return table;
}

void Report(ThreadTable& table) {
	const auto   now       = std::chrono::steady_clock::now();
	const auto   ticks_now = Ticks();
	const double s         = std::chrono::duration<double>(now - table.window_time).count();
	if (s <= 0.0) {
		return;
	}
	const double ns_per_tick = NsPerTick(table, now, ticks_now);
	const auto   ms_per_s    = [&](uint64_t ticks) { return static_cast<double>(ticks) * ns_per_tick / 1e6 / s; };
	const auto   avg_us      = [&](uint64_t ticks, uint64_t count) {
        return count != 0 ? static_cast<double>(ticks) * ns_per_tick / 1e3 / static_cast<double>(count) : 0.0;
	};
	const char* const thread = table.gpu_thread ? "" : " (not the GPU thread)";
	std::string       text   = std::string("[cpu-cost] by kind") + thread + ":";
	uint64_t          scope_ticks = 0;
	char              line[256];
	for (size_t i = 0; i < 128; i++) {
		if (table.kind_count[i] == 0) continue;
		std::snprintf(line, sizeof(line), " %c %.0f/s %.1f ms/s |", static_cast<char>(i),
		              static_cast<double>(table.kind_count[i]) / s, ms_per_s(table.kind_ticks[i]));
		text += line;
		scope_ticks += table.kind_ticks[i];
		table.kind_count[i] = table.kind_ticks[i] = 0;
	}
	text += "\n";
	{
		std::vector<CostSlot> list;
		for (const auto& slot: table.costs) {
			if (slot.key != 0 && slot.count != 0) {
				list.push_back(slot);
			}
		}
		if (table.cost_overflow.count != 0) {
			list.push_back(table.cost_overflow);
		}
		std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.ticks > b.ticks; });
		for (size_t i = 0; i < list.size() && i < 30; i++) {
			std::snprintf(line, sizeof(line), "[cpu-cost]   %c %014llx %7.0f/s %6.2f ms/s avg %6.1f us\n",
			              list[i].key != 0 ? static_cast<char>(list[i].key >> 56u) : '?',
			              static_cast<unsigned long long>(list[i].key & ((uint64_t {1} << 56u) - 1u)),
			              static_cast<double>(list[i].count) / s, ms_per_s(list[i].ticks),
			              avg_us(list[i].ticks, list[i].count));
			text += line;
		}
		// Programs change from window to window: the table starts empty again.
		for (auto& slot: table.costs) {
			slot = {};
		}
		table.cost_overflow = {};
		table.costs_used    = 0;
	}
	// One literal can have several addresses (one per file that names the phase): merged by text.
	std::map<RowKey, Row> merged;
	const auto            merge = [&](PhaseSlot& slot, const char* name) {
        if (slot.count == 0) {
            return;
        }
        auto& m = merged[{slot.kind != 0 ? slot.kind : '?', std::string_view(name)}];
        m.count += slot.count;
        m.ticks += slot.ticks;
        m.self  += slot.self_ticks;
        slot.count = slot.ticks = slot.self_ticks = 0;
	};
	for (auto& slot: table.phases) {
		if (slot.name != nullptr) {
			merge(slot, slot.name);
		}
	}
	const bool overflowed = table.phase_overflow.count != 0;
	merge(table.phase_overflow, "(phase table full)");
	std::vector<std::pair<RowKey, Row>> rows(merged.begin(), merged.end());
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ticks > b.second.ticks; });
	for (const auto& [key, row]: rows) {
		std::snprintf(line, sizeof(line), "[cpu-phase]   %c %-40.*s %8.0f/s %7.2f ms/s avg %6.1f us\n", key.first,
		              static_cast<int>(key.second.size()), key.second.data(), static_cast<double>(row.count) / s,
		              ms_per_s(row.ticks), avg_us(row.ticks, row.count));
		text += line;
	}
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.self > b.second.self; });
	uint64_t self_sum = 0, self_outside = 0, self_wait = 0;
	for (const auto& [key, row]: rows) {
		std::snprintf(line, sizeof(line), "[cpu-self]    %c %-40.*s %8.0f/s %7.2f ms/s avg %6.1f us\n", key.first,
		              static_cast<int>(key.second.size()), key.second.data(), static_cast<double>(row.count) / s,
		              ms_per_s(row.self), avg_us(row.self, row.count));
		text += line;
		self_sum += row.self;
		if (key.first == '.') {
			self_outside += row.self;
		}
		// The GPU thread's own waits for work (graphicsRun.cpp, "Thread::Wait...").
		if (key.second.substr(0, 12) == "Thread::Wait") {
			self_wait += row.self;
		}
	}
	// What one record costs: two counter reads and a slot lookup, measured here.
	double record_ns = 0.0;
	{
		constexpr uint32_t Rounds = 2048;
		uint64_t           sink   = 0;
		const auto         begin  = Ticks();
		for (uint32_t i = 0; i < Rounds; i++) {
			const auto a = Ticks();
			sink += FindPhase(table, ScopeSelfName, 'D').count;
			sink += Ticks() - a;
		}
		record_ns = static_cast<double>(Ticks() - begin) * ns_per_tick / Rounds + static_cast<double>(sink & 1u) * 0.0;
	}
	uint64_t cpu_ns[2];
	ThreadCpuNs(cpu_ns);
	const double user_ms   = static_cast<double>(cpu_ns[0] - table.window_cpu_ns[0]) / 1e6 / s;
	const double kernel_ms = static_cast<double>(cpu_ns[1] - table.window_cpu_ns[1]) / 1e6 / s;
	const double sum_ms    = ms_per_s(self_sum);
	const double wait_ms   = ms_per_s(self_wait);
	std::snprintf(line, sizeof(line),
	              "[cpu-total]%s window %.2f s: wall 1000.0 ms/s | thread CPU %.1f ms/s (user %.1f, kernel %.1f) | "
	              "self times sum %.1f ms/s = %.1f%% of wall",
	              thread, s, user_ms + kernel_ms, user_ms, kernel_ms, sum_ms, sum_ms / 10.0);
	text += line;
	std::snprintf(line, sizeof(line),
	              " (waiting for work %.1f, working %.1f: outside draws and dispatches '.' %.1f, inside %.1f) | "
	              "draws and dispatches by kind %.1f ms/s",
	              wait_ms, sum_ms - wait_ms, ms_per_s(self_outside) - wait_ms, sum_ms - ms_per_s(self_outside),
	              ms_per_s(scope_ticks));
	text += line;
	std::snprintf(line, sizeof(line),
	              " | rows %zu%s | profiler: %.0f records/s x %.0f ns = about %.1f ms/s inside these numbers\n",
	              rows.size(), overflowed ? " (PHASE TABLE FULL)" : "", static_cast<double>(table.records) / s, record_ns,
	              static_cast<double>(table.records) / s * record_ns / 1e6);
	text += line;
	table.records      = 0;
	table.window_time  = now;
	table.window_ticks = ticks_now;
	table.window_cpu_ns[0] = cpu_ns[0];
	table.window_cpu_ns[1] = cpu_ns[1];
	std::fputs(text.c_str(), stderr);
}

} // namespace Detail

void FrameEnd(double frame_ms) {
	if (!Enabled() || HitchCostMs() <= 0) {
		return;
	}
	g_frame_us.store(static_cast<uint64_t>(frame_ms * 1e3), std::memory_order_relaxed);
	g_frame_count.fetch_add(1, std::memory_order_relaxed);
}

bool OnGpuThread() {
	return Profiler::t_gpu_thread;
}

void Record(char kind, uint64_t hash, uint64_t ticks) {
	auto&      table = Detail::Table();
	const auto key   = (static_cast<uint64_t>(static_cast<uint8_t>(kind)) << 56u) | (hash & ((uint64_t {1} << 56u) - 1u));
	Detail::CostSlot* found = &table.cost_overflow;
	auto index = static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> 40u) & (Detail::CostSlotCount - 1u);
	for (size_t probe = 0; probe < 64; probe++, index = (index + 1u) & (Detail::CostSlotCount - 1u)) {
		auto& slot = table.costs[index];
		if (slot.key == key) {
			found = &slot;
			break;
		}
		if (slot.key == 0) {
			if (table.costs_used < Detail::CostSlotCount / 2u) {
				table.costs_used++;
				slot.key = key;
				found    = &slot;
			}
			break;
		}
	}
	found->count++;
	found->ticks += ticks;
	found->frame_count++;
	found->frame_ticks += ticks;
	table.kind_count[static_cast<uint8_t>(kind) & 127u]++;
	table.kind_ticks[static_cast<uint8_t>(kind) & 127u] += ticks;
	if (HitchCostMs() > 0 && table.gpu_thread) {
		if (const auto frames = g_frame_count.load(std::memory_order_relaxed); frames != table.frame_seen) {
			FrameReport(table, frames, NsPerTick(table, std::chrono::steady_clock::now(), Detail::Ticks()));
		}
	}
	// The clock is read every 64th draw or dispatch.
	static thread_local uint32_t calls = 0;
	if ((++calls & 63u) != 0) {
		return;
	}
	if (std::chrono::steady_clock::now() - table.window_time >= std::chrono::seconds(5)) {
		Detail::Report(table);
	}
}

} // namespace Libs::Graphics::CpuCost
