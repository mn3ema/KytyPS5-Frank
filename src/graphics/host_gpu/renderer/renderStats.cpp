#include "graphics/host_gpu/renderer/renderStats.h"

#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/renderer/syncToggle.h"

#include <algorithm>
#include <unordered_map>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::RenderStats {

namespace {

// The last instance ended on this thread, for the restart comparison.
thread_local const void* t_last_buffer = nullptr;
thread_local bool        t_have_last   = false;
thread_local RenderState t_last_state {};
thread_local const char* t_last_file = nullptr;
thread_local uint32_t    t_last_line = 0;

struct Stats {
	std::mutex                                           mutex;
	int64_t                                              window_start = 0;
	uint64_t                                             begins       = 0;
	uint64_t                                             identical    = 0;
	uint64_t                                             same_targets = 0;
	std::map<std::pair<const char*, uint32_t>, uint64_t> identical_sites;
	std::map<std::pair<uint32_t, uint32_t>, uint64_t>    extents; // instances with a color target
	uint32_t                                             mrt_width  = 0;
	uint32_t                                             mrt_height = 0;
	uint32_t                                             mrt_count  = 0; // color targets of it
	uint64_t                                             flip_wait_ns = 0;
	uint64_t                                             flip_waits   = 0;
};

Stats& GetStats() {
	static Stats stats;
	return stats;
}

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

std::string BaseName(const char* path) {
	std::string s = path != nullptr ? path : "?";
	if (const auto slash = s.find_last_of("\\/"); slash != std::string::npos) {
		s.erase(0, slash + 1);
	}
	return s;
}

bool SameTargets(const RenderState& a, const RenderState& b) {
	if (a.width != b.width || a.height != b.height || a.num_layers != b.num_layers ||
	    a.num_color_attachments != b.num_color_attachments) {
		return false;
	}
	for (uint32_t i = 0; i < a.num_color_attachments; i++) {
		if (a.color_attachments[i].image_view != b.color_attachments[i].image_view) {
			return false;
		}
	}
	const auto& da = a.depth_stencil_attachment;
	const auto& db = b.depth_stencil_attachment;
	return da.image_view == db.image_view && da.has_depth == db.has_depth &&
	       da.has_stencil == db.has_stencil;
}

// Prints and resets the window. Called with the mutex held.
void Report(Stats& s, int64_t now) {
	const double seconds = (std::max)(1.0, static_cast<double>(now - s.window_start) / 1000.0);
	std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> extents(s.extents.begin(),
	                                                                        s.extents.end());
	std::sort(extents.begin(), extents.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	std::string out = "[res]";
	if (s.mrt_count != 0) {
		out += " gbuffer=" + std::to_string(s.mrt_width) + "x" + std::to_string(s.mrt_height) +
		       " (" + std::to_string(s.mrt_count) + " targets)";
	}
	out += " top:";
	for (size_t i = 0; i < extents.size() && i < 6; i++) {
		char text[64];
		std::snprintf(text, sizeof(text), " %ux%u=%.0f/s", extents[i].first.first,
		              extents[i].first.second, static_cast<double>(extents[i].second) / seconds);
		out += text;
	}
	out += "\n";
	char line[256];
	std::snprintf(line, sizeof(line),
	              "[render-restart] begins=%.0f/s identical=%.0f/s same-targets=%.0f/s over %.1f s;"
	              " identical by end site:",
	              static_cast<double>(s.begins) / seconds,
	              static_cast<double>(s.identical) / seconds,
	              static_cast<double>(s.same_targets) / seconds, seconds);
	out += line;
	std::vector<std::pair<std::pair<const char*, uint32_t>, uint64_t>> sites(
	    s.identical_sites.begin(), s.identical_sites.end());
	std::sort(sites.begin(), sites.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < sites.size() && i < 10; i++) {
		std::snprintf(line, sizeof(line), " %s:%u=%.0f", BaseName(sites[i].first.first).c_str(),
		              sites[i].first.second, static_cast<double>(sites[i].second) / seconds);
		out += line;
	}
	out += "\n";
	std::snprintf(line, sizeof(line), "[flip-wait] %.1f ms/s in %.1f waits/s (%.2f ms each)\n",
	              static_cast<double>(s.flip_wait_ns) / 1e6 / seconds,
	              static_cast<double>(s.flip_waits) / seconds,
	              s.flip_waits != 0 ? static_cast<double>(s.flip_wait_ns) / 1e6 /
	                                      static_cast<double>(s.flip_waits)
	                                : 0.0);
	out += line;
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
	s.begins       = 0;
	s.identical    = 0;
	s.same_targets = 0;
	s.identical_sites.clear();
	s.extents.clear();
	s.mrt_width    = 0;
	s.mrt_height   = 0;
	s.mrt_count    = 0;
	s.flip_wait_ns = 0;
	s.flip_waits   = 0;
	s.window_start = now;
}

void MaybeReport(Stats& s) {
	const auto now = NowMs();
	if (s.window_start == 0) {
		s.window_start = now;
	} else if (now - s.window_start >= 5000) {
		Report(s, now);
	}
}

thread_local uint64_t t_last_commit_hash = 0;
thread_local uint64_t t_probe_acc        = 0;

struct MultiStats {
	std::mutex                         mutex;
	int64_t                            window_start = 0;
	uint64_t                           packets      = 0;
	uint64_t                           draws        = 0;
	uint64_t                           same         = 0;
	uint64_t                           all_same     = 0;
	std::map<std::pair<uint32_t, bool>, uint64_t> registers; // (draw index register, indexed)
	std::map<uint64_t, uint64_t>       big_packets;          // first hash -> draws (>= 16 draws)
};

MultiStats& GetMultiStats() {
	static MultiStats stats;
	return stats;
}

} // namespace

bool MultiDrawProbeEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MULTI_DRAW_PROBE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void NoteCommitHash(uint64_t hash) {
	t_last_commit_hash = hash == 0 ? 0 : hash ^ (t_probe_acc * 0x9e3779b97f4a7c15ull);
	t_probe_acc        = 0;
}

void MixProbe(uint64_t hash) {
	t_probe_acc = t_probe_acc * 31u + hash;
}

uint64_t LastCommitHash() {
	return t_last_commit_hash;
}

void NoteMultiDrawPacket(uint32_t draws, uint32_t same_as_first, uint32_t draw_index_register,
                         bool indexed, uint64_t first_hash) {
	auto&           s = GetMultiStats();
	std::lock_guard lock(s.mutex);
	s.packets++;
	s.draws += draws;
	s.same += same_as_first;
	s.all_same += same_as_first + 1u == draws ? 1u : 0u;
	s.registers[{draw_index_register, indexed}] += draws;
	if (draws >= 16u) {
		s.big_packets[first_hash] += draws;
	}
	const auto now = NowMs();
	if (s.window_start == 0) {
		s.window_start = now;
		return;
	}
	if (now - s.window_start < 5000) {
		return;
	}
	const double seconds = static_cast<double>(now - s.window_start) / 1000.0;
	std::string  out     = "[multi-draw] ";
	char         line[256];
	std::snprintf(line, sizeof(line),
	              "packets=%.1f/s draws=%.0f/s same-as-first=%.0f/s (%.1f%%) packets-all-same=%.1f/s;"
	              " by draw-index register/indexed:",
	              static_cast<double>(s.packets) / seconds, static_cast<double>(s.draws) / seconds,
	              static_cast<double>(s.same) / seconds,
	              100.0 * static_cast<double>(s.same) /
	                  static_cast<double>((std::max)(s.draws - s.packets, uint64_t {1})),
	              static_cast<double>(s.all_same) / seconds);
	out += line;
	for (const auto& [key, count]: s.registers) {
		std::snprintf(line, sizeof(line), " r%u/%s=%.0f/s", key.first, key.second ? "idx" : "auto",
		              static_cast<double>(count) / seconds);
		out += line;
	}
	std::snprintf(line, sizeof(line), "; big packets: %zu kinds", s.big_packets.size());
	out += line;
	out += "\n";
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
	s.packets = s.draws = s.same = s.all_same = 0;
	s.registers.clear();
	s.big_packets.clear();
	s.window_start = now;
}

bool MultiDrawReplayConfigured() {
	static const bool configured = [] {
		const char* value = std::getenv("KYTY_MULTI_DRAW_REPLAY");
		return (value != nullptr && value[0] == '1') ||
		       (SyncToggle::PeriodSeconds() != 0 &&
		        (SyncToggle::ModeMask() & SyncToggle::ModeReplay) != 0);
	}();
	return configured;
}

bool MultiDrawReplayActive() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_MULTI_DRAW_REPLAY");
		return value != nullptr && value[0] == '1';
	}();
	return always || SyncToggle::ReplayWindow();
}

bool IndirectReplayConfigured() {
	static const bool configured = [] {
		const char* value = std::getenv("KYTY_INDIRECT_REPLAY");
		return (value != nullptr && value[0] == '1') ||
		       (SyncToggle::PeriodSeconds() != 0 &&
		        (SyncToggle::ModeMask() & SyncToggle::ModeIndirectReplay) != 0);
	}();
	return configured;
}

bool IndirectReplayActive() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_INDIRECT_REPLAY");
		return value != nullptr && value[0] == '1';
	}();
	return always || SyncToggle::IndirectReplayWindow();
}

static std::unordered_map<uint32_t, uint64_t>& IndirectReplayStops() {
	static std::unordered_map<uint32_t, uint64_t> stops;
	return stops;
}

void NoteIndirectReplayStop(uint32_t key) {
	IndirectReplayStops()[key]++;
}

void NoteIndirectReplay(IndirectReplayNote note) {
	static std::array<std::atomic<uint64_t>, static_cast<size_t>(IndirectReplayNote::Count)> counts {};
	static std::atomic<int64_t> window_start {0};
	counts[static_cast<size_t>(note)].fetch_add(1, std::memory_order_relaxed);
	const auto now   = NowMs();
	auto       start = window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 10000 ||
	    !window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
		return;
	}
	static const char* const names[] = {
	    "replayed", "no-base", "packet-boundary", "packet-state", "args-run", "args-buffer",
	    "buffer-cache-changed", "start-instance", "command-changed", "commit-changed", "ps-user-data", "vs-program",
	    "plan-reads", "diagnostics", "base-writes", "base-not-fast-launch", "base-args-run",
	    "base-user-data", "base-plan-unknown", "base-captured", "context-changed"};
	static_assert(std::size(names) == static_cast<size_t>(IndirectReplayNote::Count));
	const double seconds = static_cast<double>(now - start) / 1000.0;
	std::string  out     = "[indirect-replay]";
	for (size_t i = 0; i < std::size(names); i++) {
		const auto n = counts[i].exchange(0, std::memory_order_relaxed);
		if (n == 0) continue;
		char text[64];
		std::snprintf(text, sizeof(text), " %s=%.0f/s", names[i], static_cast<double>(n) / seconds);
		out += text;
	}
	if (auto& stops = IndirectReplayStops(); !stops.empty()) {
		std::vector<std::pair<uint32_t, uint64_t>> list(stops.begin(), stops.end());
		std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		out += " stops=";
		for (size_t i = 0; i < list.size() && i < 8; i++) {
			char text[48];
			std::snprintf(text, sizeof(text), "%s%02x:%04x=%.0f/s", i == 0 ? "" : ",", list[i].first >> 16u,
			              list[i].first & 0xffffu, static_cast<double>(list[i].second) / seconds);
			out += text;
		}
		stops.clear();
	}
	out += IndirectReplayActive() || !SyncToggle::PeriodSeconds() ? "\n" : " (toggle off)\n";
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

void NoteReplay(ReplayNote note) {
	static std::array<std::atomic<uint64_t>, static_cast<size_t>(ReplayNote::Count)> counts {};
	static std::atomic<int64_t> window_start {0};
	counts[static_cast<size_t>(note)].fetch_add(1, std::memory_order_relaxed);
	if (!Enabled()) {
		return;
	}
	const auto now   = NowMs();
	auto       start = window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 5000 ||
	    !window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
		return;
	}
	static const char* const names[] = {
	    "replayed",     "same-data",  "empty",       "no-base",     "command-changed",
	    "commit-changed", "ps-user-data", "vs-program", "plan-reads", "limits",
	    "upload-moved", "base-not-mesh", "base-writes", "base-diagnostics", "base-user-data",
	    "base-plan-unknown", "base-captured"};
	static_assert(std::size(names) == static_cast<size_t>(ReplayNote::Count));
	const double seconds = static_cast<double>(now - start) / 1000.0;
	std::string  out     = "[multi-replay]";
	for (size_t i = 0; i < std::size(names); i++) {
		const auto n = counts[i].exchange(0, std::memory_order_relaxed);
		if (n == 0) {
			continue;
		}
		char text[64];
		std::snprintf(text, sizeof(text), " %s=%.0f/s", names[i], static_cast<double>(n) / seconds);
		out += text;
	}
	out += SyncToggle::ReplayWindow() || !SyncToggle::PeriodSeconds() ? "\n" : " (toggle off)\n";
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

void NoteMultiPacket(MultiPacketNote note, uint64_t count) {
	if (!Enabled()) {
		return;
	}
	static std::array<std::atomic<uint64_t>, static_cast<size_t>(MultiPacketNote::Count)> counts {};
	static std::atomic<int64_t> window_start {0};
	counts[static_cast<size_t>(note)].fetch_add(count, std::memory_order_relaxed);
	const auto now   = NowMs();
	auto       start = window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 5000 ||
	    !window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
		return;
	}
	static const char* const names[] = {
	    "packets",        "sub-draws",     "single",         "indexed-packets",
	    "indexed-rest",   "indexed-mesh",  "indexed-vertex", "indexed-writes",
	    "replay-packets", "first-draws",   "first-no-base",  "first-other-commit",
	    "first-state-changed", "first-same-state"};
	static_assert(std::size(names) == static_cast<size_t>(MultiPacketNote::Count));
	const double seconds = static_cast<double>(now - start) / 1000.0;
	std::string  out     = "[multi-packets]";
	for (size_t i = 0; i < std::size(names); i++) {
		const auto n = counts[i].exchange(0, std::memory_order_relaxed);
		if (n == 0) {
			continue;
		}
		char text[64];
		std::snprintf(text, sizeof(text), " %s=%.0f/s", names[i], static_cast<double>(n) / seconds);
		out += text;
	}
	out += "\n";
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_RENDER_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void NoteBegin(const void* buffer, const RenderState& state) {
	bool identical = false;
	bool same      = false;
	if (t_have_last && t_last_buffer == buffer) {
		identical = t_last_state == state;
		same      = identical || SameTargets(t_last_state, state);
	}
	t_have_last = false;
	auto&           s = GetStats();
	std::lock_guard lock(s.mutex);
	s.begins++;
	if (identical) {
		s.identical++;
		s.identical_sites[{t_last_file, t_last_line}]++;
	}
	if (same) {
		s.same_targets++;
	}
	if (state.num_color_attachments != 0) {
		s.extents[{state.width, state.height}]++;
		const uint64_t area     = static_cast<uint64_t>(state.width) * state.height;
		const uint64_t mrt_area = static_cast<uint64_t>(s.mrt_width) * s.mrt_height;
		if (state.num_color_attachments >= 2 &&
		    (state.num_color_attachments > s.mrt_count ||
		     (state.num_color_attachments == s.mrt_count && area > mrt_area))) {
			s.mrt_width  = state.width;
			s.mrt_height = state.height;
			s.mrt_count  = state.num_color_attachments;
		}
	}
	MaybeReport(s);
}

void NoteEnd(const void* buffer, const RenderState& state, const std::source_location& site) {
	t_last_buffer = buffer;
	t_last_state  = state;
	t_have_last   = true;
	t_last_file   = site.file_name();
	t_last_line   = site.line();
}

void NoteFlipWait(uint64_t ns) {
	auto&           s = GetStats();
	std::lock_guard lock(s.mutex);
	s.flip_wait_ns += ns;
	s.flip_waits++;
	MaybeReport(s);
}

} // namespace Libs::Graphics::RenderStats
