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

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CPU_COST");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

namespace {
struct CostEntry {
	uint64_t count = 0;
	uint64_t ns    = 0;
};
std::mutex& CostMutex() {
	static std::mutex mutex;
	return mutex;
}
// (kind, phase name) -> time; the names are string literals.
std::map<std::pair<char, std::string_view>, CostEntry>& Phases() {
	static std::map<std::pair<char, std::string_view>, CostEntry> phases;
	return phases;
}
std::map<std::pair<char, std::string_view>, CostEntry>& SelfPhases() {
	static std::map<std::pair<char, std::string_view>, CostEntry> phases;
	return phases;
}
// KYTY_HITCH_COST_MS: the same two tables for the guest frame in progress (FrameEnd). The
// phases by the address of their name (a string literal) and the kind: one hash lookup a phase.
int64_t HitchCostMs() {
	static const int64_t ms = [] {
		const char* value = std::getenv("KYTY_HITCH_COST_MS");
		return value != nullptr ? static_cast<int64_t>(std::strtoll(value, nullptr, 10)) : int64_t {0};
	}();
	return ms;
}
struct FramePhase {
	const char* name  = nullptr;
	char        kind  = 0;
	uint64_t    count = 0;
	uint64_t    ns    = 0;
};
std::unordered_map<uint64_t, FramePhase>& FrameSelfPhases() {
	static std::unordered_map<uint64_t, FramePhase> phases;
	return phases;
}
std::unordered_map<uint64_t, CostEntry>& FrameEntries() {
	static std::unordered_map<uint64_t, CostEntry> entries;
	return entries;
}
} // namespace

void FrameEnd(double frame_ms) {
	if (!Enabled() || HitchCostMs() <= 0) {
		return;
	}
	std::string text;
	{
		std::lock_guard lock(CostMutex());
		static uint32_t printed = 0;
		if (frame_ms >= static_cast<double>(HitchCostMs()) && printed < 400) {
			printed++;
			// One literal can have several addresses (one per file that names the phase): merged here.
			std::map<std::pair<char, std::string_view>, CostEntry> merged;
			for (const auto& entry: FrameSelfPhases()) {
				auto& m = merged[{entry.second.kind, std::string_view(entry.second.name)}];
				m.count += entry.second.count;
				m.ns    += entry.second.ns;
			}
			std::vector<std::pair<std::pair<char, std::string_view>, CostEntry>> self(merged.begin(), merged.end());
			std::sort(self.begin(), self.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
			uint64_t total_ns = 0;
			for (const auto& entry: self) {
				total_ns += entry.second.ns;
			}
			char line[224];
			std::snprintf(line, sizeof(line), "[hitch-cost] frame %.0f ms: GPU-thread phases %.0f ms (self time, ms x calls):",
			              frame_ms, static_cast<double>(total_ns) / 1e6);
			text += line;
			for (size_t i = 0; i < self.size() && i < 16 && self[i].second.ns >= 500000u; i++) {
				std::snprintf(line, sizeof(line), " %c %.*s %.1f x%llu |", self[i].first.first,
				              static_cast<int>(self[i].first.second.size()), self[i].first.second.data(),
				              static_cast<double>(self[i].second.ns) / 1e6,
				              static_cast<unsigned long long>(self[i].second.count));
				text += line;
			}
			text += "\n";
			std::vector<std::pair<uint64_t, CostEntry>> list(FrameEntries().begin(), FrameEntries().end());
			std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
			text += "[hitch-cost]   heaviest draws (A..) and dispatches (X, I) by shader:";
			for (size_t i = 0; i < list.size() && i < 10 && list[i].second.ns >= 500000u; i++) {
				std::snprintf(line, sizeof(line), " %c %014llx %.1f x%llu |", static_cast<char>(list[i].first >> 56u),
				              static_cast<unsigned long long>(list[i].first & ((uint64_t {1} << 56u) - 1u)),
				              static_cast<double>(list[i].second.ns) / 1e6,
				              static_cast<unsigned long long>(list[i].second.count));
				text += line;
			}
			text += "\n";
		}
		FrameSelfPhases().clear();
		FrameEntries().clear();
	}
	if (!text.empty()) {
		std::fputs(text.c_str(), stderr);
	}
}

bool OnGpuThread() {
	return Profiler::t_gpu_thread;
}

void RecordPhase(char kind, const char* name, uint64_t ns) {
	std::lock_guard lock(CostMutex());
	auto& e = Phases()[{kind, std::string_view(name)}];
	e.count++;
	e.ns += ns;
}

void RecordPhaseSelf(char kind, const char* name, uint64_t ns) {
	std::lock_guard lock(CostMutex());
	auto& e = SelfPhases()[{kind, std::string_view(name)}];
	e.count++;
	e.ns += ns;
	if (HitchCostMs() > 0) {
		auto& f = FrameSelfPhases()[(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(name)) << 8u) |
		                            static_cast<uint8_t>(kind)];
		f.name = name;
		f.kind = kind;
		f.count++;
		f.ns += ns;
	}
}

void Record(char kind, uint64_t hash, uint64_t ns) {
	using Entry = CostEntry;
	static std::unordered_map<uint64_t, Entry>          entries;
	static std::array<Entry, 128>                       kinds {};
	static auto                                         window = std::chrono::steady_clock::now();
	std::lock_guard                                     lock(CostMutex());
	const auto key = (static_cast<uint64_t>(static_cast<uint8_t>(kind)) << 56u) ^ (hash & ((uint64_t {1} << 56u) - 1u));
	auto&      e   = entries[key];
	e.count++;
	e.ns += ns;
	auto& k = kinds[static_cast<uint8_t>(kind) & 127u];
	k.count++;
	k.ns += ns;
	if (HitchCostMs() > 0) {
		auto& f = FrameEntries()[key];
		f.count++;
		f.ns += ns;
	}
	const auto now = std::chrono::steady_clock::now();
	const auto s   = std::chrono::duration<double>(now - window).count();
	if (s < 5.0) {
		return;
	}
	window = now;
	std::string text = "[cpu-cost] by kind:";
	for (size_t i = 0; i < kinds.size(); i++) {
		if (kinds[i].count == 0) continue;
		char line[128];
		std::snprintf(line, sizeof(line), " %c %.0f/s %.1f ms/s |", static_cast<char>(i),
		              static_cast<double>(kinds[i].count) / s, static_cast<double>(kinds[i].ns) / 1e6 / s);
		text += line;
		kinds[i] = {};
	}
	std::vector<std::pair<uint64_t, Entry>> list(entries.begin(), entries.end());
	std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	text += "\n";
	for (size_t i = 0; i < list.size() && i < 30; i++) {
		char line[192];
		std::snprintf(line, sizeof(line), "[cpu-cost]   %c %014llx %7.0f/s %6.2f ms/s avg %6.1f us\n",
		              static_cast<char>(list[i].first >> 56u),
		              static_cast<unsigned long long>(list[i].first & ((uint64_t {1} << 56u) - 1u)),
		              static_cast<double>(list[i].second.count) / s,
		              static_cast<double>(list[i].second.ns) / 1e6 / s,
		              static_cast<double>(list[i].second.ns) / 1e3 / static_cast<double>(list[i].second.count));
		text += line;
	}
	entries.clear();
	{
		std::vector<std::pair<std::pair<char, std::string_view>, Entry>> phases(Phases().begin(),
		                                                                       Phases().end());
		std::sort(phases.begin(), phases.end(),
		          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (size_t i = 0; i < phases.size() && i < 60; i++) {
			char line[192];
			std::snprintf(line, sizeof(line), "[cpu-phase]   %c %-34.*s %8.0f/s %7.2f ms/s avg %6.1f us\n",
			              phases[i].first.first, static_cast<int>(phases[i].first.second.size()),
			              phases[i].first.second.data(), static_cast<double>(phases[i].second.count) / s,
			              static_cast<double>(phases[i].second.ns) / 1e6 / s,
			              static_cast<double>(phases[i].second.ns) / 1e3 /
			                  static_cast<double>(phases[i].second.count));
			text += line;
		}
		Phases().clear();
		std::vector<std::pair<std::pair<char, std::string_view>, CostEntry>> self(SelfPhases().begin(),
		                                                                         SelfPhases().end());
		std::sort(self.begin(), self.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (size_t i = 0; i < self.size() && i < 60; i++) {
			char line[192];
			std::snprintf(line, sizeof(line), "[cpu-self]    %c %-34.*s %8.0f/s %7.2f ms/s avg %6.1f us\n",
			              self[i].first.first, static_cast<int>(self[i].first.second.size()),
			              self[i].first.second.data(), static_cast<double>(self[i].second.count) / s,
			              static_cast<double>(self[i].second.ns) / 1e6 / s,
			              static_cast<double>(self[i].second.ns) / 1e3 / static_cast<double>(self[i].second.count));
			text += line;
		}
		SelfPhases().clear();
	}
	std::fputs(text.c_str(), stderr);
}

} // namespace Libs::Graphics::CpuCost
