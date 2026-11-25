#include "graphics/host_gpu/renderer/gpuTiming.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderStats.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::GpuTiming {

namespace {

constexpr uint32_t BlockQueries = 2048;
constexpr uint32_t BlockCount   = 32;

constexpr uint32_t QueueSlots = 4;

thread_local uint32_t t_queue = 0;
// The thread that times guest commands (BeforeSlow ran on it): only it may add to a block.
thread_local bool t_timed_thread = false;

constexpr const char* WorkNames[Work::WorkCount] = {"copy-buf", "copy-img", "buf-to-img",
                                                    "img-to-buf", "fill", "dispatch"};
constexpr const char* WorkUnits[Work::WorkCount] = {"B", "texels", "texels", "texels", "B", "groups"};

struct WorkSite {
	std::string name; // kind and file:line
	uint32_t    work    = 0;
	uint64_t    count   = 0;
	uint64_t    amount  = 0;
	uint64_t    inside  = 0; // of count: recorded inside a timed guest draw or dispatch
};

struct Pair {
	uint32_t before = 0;
	uint32_t after  = 0;
	uint64_t key    = 0;
	uint32_t kind   = 0;
	uint32_t queue  = 0;
	uint32_t gap_mask     = 0; // what was recorded between the previous timed command and this one
	uint32_t gap_barriers = 0;
};

struct Block {
	uint64_t          tick = 0;
	uint32_t          used = 0;
	int64_t           open = -1; // query index of a Before without its After yet
	uint32_t          open_mask     = 0;
	uint32_t          open_barriers = 0;
	std::vector<Pair> pairs;
	std::atomic<bool> in_flight {false};
};

struct Stat {
	double   ms    = 0.0;
	uint64_t count = 0;
	uint32_t kind  = 0;
};

// v445m, the whole window ("GpuWindow", "GpuGapPair", "GpuKey" lines when it closes): every
// gap of 2 us or more between two timed commands goes to the pair (command before, command
// after), within a command buffer and across command buffers; and every key's time over the
// window, not only each frame's heaviest.
struct GapPair {
	uint64_t prev = 0, next = 0;
	uint32_t prev_kind = 0, next_kind = 0;
	bool     cross = false; // across two command buffers
	double   ms    = 0.0;
	uint64_t count = 0;
	uint32_t masks = 0; // GapNote bits recorded in any of its gaps
};

struct State {
	vk::Device                           device;
	vk::QueryPool                        pool;
	double                               period_ns = 1.0;
	std::array<Block, BlockCount>        blocks;
	uint32_t                             next_block = 0;
	Block*                               current    = nullptr;
	std::mutex                           stats_mutex;
	std::unordered_map<uint64_t, Stat>   stats; // key 0: GPU time between timed commands
	double                               total_ms = 0.0;
	std::array<double, QueueSlots>       queue_ms {};
	std::array<uint64_t, QueueSlots>     queue_n {};
	std::unordered_map<uint64_t, uint32_t> lanes; // compute shader hash -> lanes per invocation
	// Gaps by what was recorded in them (GapNote mask), by length, and the barriers in them.
	std::unordered_map<uint32_t, Stat>     gap_by_mask;
	std::array<double, 4>                  gap_len_ms {}; // <2 us, 2-20 us, 20-200 us, >=200 us
	std::array<uint64_t, 4>                gap_len_n {};
	uint64_t                               gap_barriers = 0;
	// Across command buffers ("GpuCross"): from the last timed command of one buffer to the first
	// timed command of a later one. Not part of the span or of "between" (both are per buffer).
	// [0]: nothing was recorded in between (the GPU waited for the submit, or ran another
	// thread's buffer); [1]: something was (GapNote bits), so work and waiting are mixed.
	uint64_t                               cross_tick = 0;
	uint64_t                               cross_last = 0;
	std::array<double, 2>                  cross_ms {};
	std::array<uint64_t, 2>                cross_n {};
	// The work census (gpuTiming.h, Work): by site, and its report window.
	std::unordered_map<uint64_t, WorkSite> work;
	uint64_t                               work_frames = 0;
	std::chrono::steady_clock::time_point  work_last {};
	// The window's summary (GapPair).
	std::unordered_map<uint64_t, GapPair>  window_pairs;
	std::unordered_map<uint64_t, Stat>     window_stats; // by key; key 0: all gaps inside command buffers
	double                                 window_span_ms    = 0.0;
	double                                 window_short_ms   = 0.0; // gaps under 2 us: not attributed
	uint64_t                               window_short_n    = 0;
	double                                 window_cross_ms   = 0.0;
	uint64_t                               window_cross_n    = 0;
	uint64_t                               cross_last_key    = 0;
	uint32_t                               cross_last_kind   = 0;
	uint64_t                               window_frames     = 0;
	bool                                   window_printed    = false;
};

void AddGapPair(State& state, uint64_t prev, uint32_t prev_kind, uint64_t next, uint32_t next_kind, bool cross,
                double ms, uint32_t mask) {
	const uint64_t id = (prev * 0x9e3779b97f4a7c15ull) ^ (next * 0xc2b2ae3d27d4eb4full) ^
	                    (uint64_t {prev_kind} << 1u) ^ (uint64_t {next_kind} << 9u) ^ (cross ? 1u : 0u);
	auto& pair     = state.window_pairs[id];
	pair.prev      = prev;
	pair.next      = next;
	pair.prev_kind = prev_kind;
	pair.next_kind = next_kind;
	pair.cross     = cross;
	pair.ms += ms;
	pair.count++;
	pair.masks |= mask;
}

State& GetState() {
	static State state;
	return state;
}

struct Window {
	uint64_t first = 0;
	uint64_t count = 0;
};

const Window& GetWindow() {
	static const Window window = [] {
		Window result;
		const char* value = std::getenv("KYTY_GPU_TIMING");
		if (value != nullptr) {
			char* end    = nullptr;
			result.first = std::strtoull(value, &end, 10);
			result.count = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 1;
		}
		return result;
	}();
	return window;
}

void ReadBlock(State& state, Block& block) {
	if (block.used != 0) {
		std::vector<uint64_t> values(block.used);
		const auto result = state.device.getQueryPoolResults(
		    state.pool, static_cast<uint32_t>(&block - state.blocks.data()) * BlockQueries,
		    block.used, values.size() * sizeof(uint64_t), values.data(), sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eSuccess) {
			std::lock_guard lock(state.stats_mutex);
			uint64_t        previous_after = 0;
			const Pair*     previous_pair  = nullptr;
			for (const auto& pair: block.pairs) {
				const auto before = values[pair.before];
				const auto after  = values[pair.after];
				if (after >= before) {
					auto& stat = state.stats[pair.key];
					stat.ms += static_cast<double>(after - before) * state.period_ns / 1e6;
					stat.count++;
					stat.kind = pair.kind;
					auto& window = state.window_stats[pair.key];
					window.ms += static_cast<double>(after - before) * state.period_ns / 1e6;
					window.count++;
					window.kind = pair.kind;
					if (pair.kind < WorkKindBase) {
						const auto q = (std::min)(pair.queue, QueueSlots - 1);
						state.queue_ms[q] += static_cast<double>(after - before) * state.period_ns / 1e6;
						state.queue_n[q]++;
					}
				}
				if (previous_after != 0 && before >= previous_after) {
					auto&      gap = state.stats[0];
					const auto ms  = static_cast<double>(before - previous_after) * state.period_ns / 1e6;
					gap.ms += ms;
					gap.count++;
					auto& by_mask = state.gap_by_mask[pair.gap_mask];
					by_mask.ms += ms;
					by_mask.count++;
					const size_t len = ms < 0.002 ? 0 : ms < 0.02 ? 1 : ms < 0.2 ? 2 : 3;
					state.gap_len_ms[len] += ms;
					state.gap_len_n[len]++;
					state.gap_barriers += pair.gap_barriers;
					auto& window = state.window_stats[0];
					window.ms += ms;
					window.count++;
					if (ms >= 0.002 && previous_pair != nullptr) {
						AddGapPair(state, previous_pair->key, previous_pair->kind, pair.key, pair.kind, false, ms,
						           pair.gap_mask);
					} else {
						state.window_short_ms += ms;
						state.window_short_n++;
					}
				}
				previous_after = after;
				previous_pair  = &pair;
			}
			if (!block.pairs.empty()) {
				const auto first = values[block.pairs.front().before];
				const auto last  = values[block.pairs.back().after];
				if (last >= first) {
					state.total_ms += static_cast<double>(last - first) * state.period_ns / 1e6;
					state.window_span_ms += static_cast<double>(last - first) * state.period_ns / 1e6;
				}
				if (state.cross_last != 0 && block.tick > state.cross_tick && first >= state.cross_last) {
					const size_t kind = block.pairs.front().gap_mask == 0 ? 0 : 1;
					const double ms   = static_cast<double>(first - state.cross_last) * state.period_ns / 1e6;
					state.cross_ms[kind] += ms;
					state.cross_n[kind]++;
					state.window_cross_ms += ms;
					state.window_cross_n++;
					if (ms >= 0.002) {
						AddGapPair(state, state.cross_last_key, state.cross_last_kind, block.pairs.front().key,
						           block.pairs.front().kind, true, ms, block.pairs.front().gap_mask);
					}
				}
				if (block.tick >= state.cross_tick) {
					state.cross_tick      = block.tick;
					state.cross_last      = last;
					state.cross_last_key  = block.pairs.back().key;
					state.cross_last_kind = block.pairs.back().kind;
				}
			}
		}
	}
	block.pairs.clear();
	block.used = 0;
	block.open = -1;
	block.in_flight.store(false, std::memory_order_release);
}

// The block for the recording in progress, or null when none is free.
Block* CurrentBlock(RenderContext& context, CommandBuffer& buffer) {
	auto&      state     = GetState();
	auto&      scheduler = context.GetCommandScheduler();
	const auto tick      = scheduler.CurrentTick();
	if (state.current != nullptr && state.current->tick == tick &&
	    state.current->used + 2 <= BlockQueries) {
		return state.current;
	}
	if (!state.pool) {
		const auto& graphics = context.GetGraphics();
		state.device         = graphics.device;
		state.period_ns      = graphics.physical_device_properties.limits.timestampPeriod;
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = BlockQueries * BlockCount;
		if (state.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
			Detail::g_active.store(false);
			return nullptr;
		}
	}
	auto& block = state.blocks[state.next_block % BlockCount];
	if (block.in_flight.load(std::memory_order_acquire)) {
		state.current = nullptr;
		return nullptr;
	}
	state.next_block++;
	block.tick = tick;
	block.used = 0;
	block.open = -1;
	block.pairs.clear();
	block.in_flight.store(true, std::memory_order_release);
	buffer.EndRendering();
	buffer.Handle().resetQueryPool(state.pool,
	                               static_cast<uint32_t>(&block - state.blocks.data()) * BlockQueries,
	                               BlockQueries);
	// Runs once this tick has executed, so every query of the block is written.
	scheduler.DeferPriorityOperation([&state, &block] { ReadBlock(state, block); });
	state.current = &block;
	return &block;
}

} // namespace

namespace Detail {

// Counts one internal command at its site. When it can be timed, reserves its two timestamp
// queries in the current block and returns the first one's index in `pool` (the caller records
// timestamp, command, timestamp); -1 otherwise.
int64_t WorkBegin(uint32_t work, uint32_t gap_bit, uint64_t amount, const std::source_location& site,
                  uint64_t tick, vk::QueryPool& pool) {
	const bool inside = t_inside;
	if (inside && work == WorkDispatch) {
		return -1; // the guest's own dispatch
	}
	auto& state = GetState();
	// The site's key: the file name's address (a literal), the line and the kind.
	const uint64_t key = ((reinterpret_cast<uint64_t>(site.file_name()) * 0x9e3779b97f4a7c15ull) ^
	                      (uint64_t {site.line()} << 8u) ^ work) | (uint64_t {1} << 63u);
	{
		std::lock_guard lock(state.stats_mutex);
		auto&           entry = state.work[key];
		if (entry.name.empty()) {
			std::string file = site.file_name();
			if (const auto slash = file.find_last_of("/\\"); slash != std::string::npos) {
				file.erase(0, slash + 1);
			}
			entry.name = std::string(WorkNames[work]) + " " + file + ":" + std::to_string(site.line());
			entry.work = work;
		}
		entry.count++;
		entry.amount += amount;
		entry.inside += inside ? 1u : 0u;
	}
	if (inside) {
		return -1; // part of the timed guest command around it
	}
	auto* block = state.current;
	if (!g_active.load(std::memory_order_relaxed) || !t_timed_thread || tick == 0 || block == nullptr ||
	    block->tick != tick || block->open >= 0 || block->used + 2 > BlockQueries) {
		Note(gap_bit);
		return -1;
	}
	const auto base  = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	const auto first = block->used;
	block->used += 2;
	block->pairs.push_back({first, first + 1u, key, WorkKindBase + work, t_queue,
	                        std::exchange(t_gap_mask, 0u), std::exchange(t_gap_barriers, 0u)});
	pool = state.pool;
	return static_cast<int64_t>(base) + first;
}

void BeforeSlow(RenderContext& context, CommandBuffer& buffer) {
	t_timed_thread = true;
	auto* block = CurrentBlock(context, buffer);
	if (block == nullptr) {
		return;
	}
	auto&      state = GetState();
	const auto base  = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	buffer.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                               base + block->used);
	block->open          = block->used++;
	block->open_mask     = std::exchange(t_gap_mask, 0u);
	block->open_barriers = std::exchange(t_gap_barriers, 0u);
}

void AfterSlow(RenderContext& context, CommandBuffer& buffer, uint64_t key, uint32_t kind) {
	auto& state = GetState();
	// The command's own recording (its draw or dispatch, barriers inside it) is not a gap.
	struct Reset {
		~Reset() {
			t_gap_mask     = 0;
			t_gap_barriers = 0;
		}
	} reset_;
	auto* block = state.current;
	if (block == nullptr || block->open < 0 ||
	    block->tick != context.GetCommandScheduler().CurrentTick() ||
	    block->used >= BlockQueries) {
		return;
	}
	const auto base = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	buffer.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                               base + block->used);
	block->pairs.push_back({static_cast<uint32_t>(block->open), block->used, key, kind, t_queue,
	                        block->open_mask, block->open_barriers});
	block->used++;
	block->open = -1;
}

} // namespace Detail

// v445m: the window's totals, its gaps by (command before, command after) and every key.
// KYTY_GPU_GAP_PAIRS_TOP=<n>: the pairs printed of each list (default 60; the key table is always whole).
static void ReportWindow(State& state, uint64_t first_frame, uint64_t frames) {
	static constexpr const char* names[GapNoteCount] = {"barrier", "guest-barrier", "copy-buf", "copy-img", "clear",
	                                                    "fill",    "rendering",     "dispatch", "draw"};
	std::vector<GapPair>                      pairs;
	std::vector<std::pair<uint64_t, Stat>>    keys;
	std::unordered_map<uint64_t, std::string> work_names;
	std::unordered_map<uint64_t, uint32_t>    lanes;
	double                                    span = 0.0, short_ms = 0.0, cross_ms = 0.0;
	uint64_t                                  short_n = 0, cross_n = 0;
	{
		std::lock_guard lock(state.stats_mutex);
		pairs.reserve(state.window_pairs.size());
		for (const auto& entry: state.window_pairs) {
			pairs.push_back(entry.second);
		}
		keys.assign(state.window_stats.begin(), state.window_stats.end());
		for (const auto& [key, site]: state.work) {
			work_names[key] = site.name;
		}
		lanes    = state.lanes;
		span     = state.window_span_ms;
		short_ms = state.window_short_ms;
		short_n  = state.window_short_n;
		cross_ms = state.window_cross_ms;
		cross_n  = state.window_cross_n;
		state.window_pairs.clear();
		state.window_stats.clear();
	}
	const auto name_of = [&](uint64_t key, uint32_t kind) -> std::string {
		char text[96];
		if (kind >= WorkKindBase) {
			const auto it = work_names.find(key);
			return "internal " + (it != work_names.end() ? it->second : std::string("?"));
		}
		const auto it    = lanes.find(key);
		const bool split = kind != Draw && it != lanes.end() && it->second == 2u;
		std::snprintf(text, sizeof(text), "%s 0x%016" PRIx64, kind == Draw ? "draw" : (split ? "cs_split" : "cs_nativ"),
		              key);
		return text;
	};
	const double per = frames != 0 ? 1.0 / static_cast<double>(frames) : 0.0;
	double       commands = 0.0, internal = 0.0, between = 0.0, attributed = 0.0;
	uint64_t     between_n = 0, attributed_n = 0;
	for (const auto& [key, stat]: keys) {
		if (key == 0) {
			between   = stat.ms;
			between_n = stat.count;
		} else if (stat.kind >= WorkKindBase) {
			internal += stat.ms;
		} else {
			commands += stat.ms;
		}
	}
	for (const auto& pair: pairs) {
		if (!pair.cross) {
			attributed += pair.ms;
			attributed_n += pair.count;
		}
	}
	std::string out;
	char        line[512];
	std::snprintf(line, sizeof(line),
	              "GpuWindow frames %" PRIu64 "..%" PRIu64 " (%" PRIu64 " frames), ms per frame: span=%.2f commands=%.2f "
	              "internal=%.2f between=%.2f/%.0f (gaps >=2us, attributed below: %.2f/%.0f; shorter: %.2f/%.0f) | across "
	              "command buffers (not in the span): %.2f/%.0f\n",
	              first_frame, first_frame + frames - 1u, frames, span * per, commands * per, internal * per,
	              between * per, static_cast<double>(between_n) * per, attributed * per,
	              static_cast<double>(attributed_n) * per, short_ms * per, static_cast<double>(short_n) * per,
	              cross_ms * per, static_cast<double>(cross_n) * per);
	out += line;
	static const size_t top = [] {
		const char* value = std::getenv("KYTY_GPU_GAP_PAIRS_TOP");
		return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t {60};
	}();
	std::sort(pairs.begin(), pairs.end(), [](const GapPair& a, const GapPair& b) { return a.ms > b.ms; });
	// v445n: two lists, the top <n> of each. The gaps inside a command buffer are the GPU's own
	// time (the window's "between"); the ones across command buffers are mostly the GPU waiting
	// for the thread while the window is on, and in one list they crowded the others out (g450b).
	for (int cross = 0; cross < 2; cross++) {
		size_t total = 0;
		for (const auto& pair: pairs) {
			total += (pair.cross ? 1 : 0) == cross ? 1u : 0u;
		}
		std::snprintf(line, sizeof(line),
		              "GpuGapPairs %s: %zu pairs with a gap of 2 us or more, the top %zu by total ms (ms per frame, "
		              "gaps per frame, average us; after -> before; what was recorded in such gaps)\n",
		              cross != 0 ? "across command buffers" : "inside a command buffer", total, (std::min)(top, total));
		out += line;
		size_t printed = 0;
		for (size_t i = 0; i < pairs.size() && printed < top; i++) {
			const auto& pair = pairs[i];
			if ((pair.cross ? 1 : 0) != cross) {
				continue;
			}
			printed++;
			std::string recorded;
			for (uint32_t bit = 0; bit < GapNoteCount; bit++) {
				if ((pair.masks & (1u << bit)) != 0) {
					recorded += recorded.empty() ? "" : "+";
					recorded += names[bit];
				}
			}
			std::snprintf(line, sizeof(line), "GpuGapPair  %7.3fms/f %7.1f/f avg %7.1fus | %s -> %s%s | %s\n",
			              pair.ms * per, static_cast<double>(pair.count) * per,
			              pair.ms * 1e3 / static_cast<double>(pair.count), name_of(pair.prev, pair.prev_kind).c_str(),
			              name_of(pair.next, pair.next_kind).c_str(), pair.cross ? " (across command buffers)" : "",
			              recorded.empty() ? "nothing" : recorded.c_str());
			out += line;
			if (out.size() > (1u << 16u)) {
				std::fputs(out.c_str(), stderr);
				out.clear();
			}
		}
	}
	std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
	std::snprintf(line, sizeof(line), "GpuKeys: all %zu keys of the window (ms per frame, count per frame, average us)\n",
	              keys.size());
	out += line;
	for (const auto& [key, stat]: keys) {
		std::snprintf(line, sizeof(line), "GpuKey  %8.3fms/f %8.1f/f avg %8.1fus | %s\n", stat.ms * per,
		              static_cast<double>(stat.count) * per,
		              stat.count != 0 ? stat.ms * 1e3 / static_cast<double>(stat.count) : 0.0,
		              key == 0 ? "between (all gaps inside command buffers)" : name_of(key, stat.kind).c_str());
		out += line;
		if (out.size() > (1u << 16u)) {
			std::fputs(out.c_str(), stderr);
			out.clear();
		}
	}
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

// [gpu-work]: the census of the last 5 s, per presented frame, the largest sites first.
static void ReportWork(State& state) {
	std::vector<WorkSite> sites;
	uint64_t              frames = 0;
	{
		std::lock_guard lock(state.stats_mutex);
		const auto      now = std::chrono::steady_clock::now();
		state.work_frames++;
		if (state.work_last == std::chrono::steady_clock::time_point {}) {
			state.work_last = now;
		}
		if (now - state.work_last < std::chrono::seconds(5)) {
			return;
		}
		state.work_last = now;
		frames          = std::exchange(state.work_frames, 0);
		for (auto& [key, site]: state.work) {
			if (site.count != 0) {
				sites.push_back(site);
				site.count = site.amount = site.inside = 0;
			}
		}
	}
	std::array<uint64_t, Work::WorkCount> count {}, amount {};
	for (const auto& site: sites) {
		count[site.work] += site.count;
		amount[site.work] += site.amount;
	}
	const double per = frames != 0 ? 1.0 / static_cast<double>(frames) : 0.0;
	std::string  out = "[gpu-work] 5 s, " + std::to_string(frames) + " frames, per frame:";
	for (uint32_t work = 0; work < Work::WorkCount; work++) {
		char text[96];
		std::snprintf(text, sizeof(text), " %s %.1f (%.0f %s)", WorkNames[work],
		              static_cast<double>(count[work]) * per, static_cast<double>(amount[work]) * per,
		              WorkUnits[work]);
		out += text;
	}
	out += "\n";
	std::sort(sites.begin(), sites.end(), [](const auto& a, const auto& b) {
		return a.work != b.work ? a.work < b.work : (a.amount != b.amount ? a.amount > b.amount : a.count > b.count);
	});
	std::array<uint32_t, Work::WorkCount> listed {};
	for (const auto& site: sites) {
		if (listed[site.work]++ >= 10u) {
			continue; // ten sites a kind
		}
		char text[256];
		std::snprintf(text, sizeof(text), "[gpu-work]   %-40s %8.1f a frame, %12.0f %s, %.1f inside a timed command\n",
		              site.name.c_str(), static_cast<double>(site.count) * per,
		              static_cast<double>(site.amount) * per, WorkUnits[site.work],
		              static_cast<double>(site.inside) * per);
		out += text;
	}
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

void Present(uint64_t frame) {
	BarrierReason::NoteFrame(); // KYTY_BARRIER_REASON_STATS
	const auto& window = GetWindow();
	static const bool census = [&window] {
		const bool on = window.count != 0 || RenderStats::Enabled();
		Detail::g_census.store(on, std::memory_order_relaxed);
		return on;
	}();
	if (census) {
		ReportWork(GetState());
	}
	if (window.count == 0) {
		return;
	}
	auto& state = GetState();
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		std::vector<std::pair<uint64_t, Stat>> sorted;
		double                                 timed = 0.0;
		double                                 total = 0.0;
		// Buckets: draws, native compute, two-lane (64 on 32) compute, gaps.
		double   bucket_ms[5] {}; // [4]: the renderer's own copies, fills and dispatches, timed
		uint64_t bucket_n[5] {};
		std::unordered_map<uint64_t, std::string> work_names;
		std::vector<uint64_t> split_keys;
		std::array<double, QueueSlots>   queue_ms {};
		std::array<uint64_t, QueueSlots> queue_n {};
		std::vector<std::pair<uint32_t, Stat>> gap_masks;
		std::array<double, 4>                  gap_len_ms {};
		std::array<uint64_t, 4>                gap_len_n {};
		uint64_t                               gap_barriers = 0;
		std::array<double, 2>                  cross_ms {};
		std::array<uint64_t, 2>                cross_n {};
		{
			std::lock_guard lock(state.stats_mutex);
			cross_ms = std::exchange(state.cross_ms, {});
			cross_n  = std::exchange(state.cross_n, {});
			sorted.assign(state.stats.begin(), state.stats.end());
			state.stats.clear();
			total          = state.total_ms;
			state.total_ms = 0.0;
			queue_ms       = std::exchange(state.queue_ms, {});
			queue_n        = std::exchange(state.queue_n, {});
			gap_masks.assign(state.gap_by_mask.begin(), state.gap_by_mask.end());
			state.gap_by_mask.clear();
			gap_len_ms   = std::exchange(state.gap_len_ms, {});
			gap_len_n    = std::exchange(state.gap_len_n, {});
			gap_barriers = std::exchange(state.gap_barriers, 0);
			for (const auto& [key, stat]: sorted) {
				size_t b = 3;
				if (key != 0) {
					if (stat.kind >= WorkKindBase) {
						b = 4;
						if (const auto it = state.work.find(key); it != state.work.end()) {
							work_names[key] = it->second.name;
						}
					} else if (stat.kind == Draw) {
						b = 0;
					} else {
						const auto it = state.lanes.find(key);
						b             = it != state.lanes.end() && it->second == 2u ? 2 : 1;
					}
				}
				bucket_ms[b] += stat.ms;
				bucket_n[b] += stat.count;
				if (b == 2) {
					split_keys.push_back(key);
				}
			}
		}
		std::fprintf(stderr,
		             "GpuBuckets frame %" PRIu64 ": span=%.1f draws=%.2f/%" PRIu64
		             " cs_native=%.2f/%" PRIu64 " cs_split=%.2f/%" PRIu64 " gaps=%.2f/%" PRIu64
		             " internal=%.2f/%" PRIu64 "\n",
		             frame, total, bucket_ms[0], bucket_n[0], bucket_ms[1], bucket_n[1], bucket_ms[2],
		             bucket_n[2], bucket_ms[3], bucket_n[3], bucket_ms[4], bucket_n[4]);
		std::fprintf(stderr,
		             "GpuQueues frame %" PRIu64 ": q0=%.2f/%" PRIu64 " q1=%.2f/%" PRIu64
		             " q2=%.2f/%" PRIu64 " other=%.2f/%" PRIu64 "\n",
		             frame, queue_ms[0], queue_n[0], queue_ms[1], queue_n[1], queue_ms[2], queue_n[2],
		             queue_ms[3], queue_n[3]);
		{
			// GpuGaps: the "between" time split by what the emulator recorded in each gap
			// (exclusive signatures), by gap length, and the barriers recorded in gaps.
			static constexpr const char* names[GapNoteCount] = {
			    "barrier", "guest-barrier", "copy-buf", "copy-img", "clear",
			    "fill", "rendering", "dispatch", "draw"};
			// Per kind, the gap time where it appears at all (not exclusive).
			std::array<double, GapNoteCount> any_ms {};
			for (const auto& [mask, stat]: gap_masks) {
				for (uint32_t bit = 0; bit < GapNoteCount; bit++) {
					if ((mask & (1u << bit)) != 0) {
						any_ms[bit] += stat.ms;
					}
				}
			}
			std::string any_text;
			for (uint32_t bit = 0; bit < GapNoteCount; bit++) {
				char item[48];
				std::snprintf(item, sizeof(item), " %s=%.2f", names[bit], any_ms[bit]);
				any_text += item;
			}
			std::fprintf(stderr,
			             "GpuGaps frame %" PRIu64 ": barriers=%" PRIu64 " | by length <2us %.2f/%" PRIu64
			             " 2-20us %.2f/%" PRIu64 " 20-200us %.2f/%" PRIu64 " >=200us %.2f/%" PRIu64
			             " | containing:%s\n",
			             frame, gap_barriers, gap_len_ms[0], gap_len_n[0], gap_len_ms[1], gap_len_n[1],
			             gap_len_ms[2], gap_len_n[2], gap_len_ms[3], gap_len_n[3], any_text.c_str());
			std::sort(gap_masks.begin(), gap_masks.end(),
			          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
			std::string top_text;
			for (size_t i = 0; i < gap_masks.size() && i < 10; i++) {
				const auto& [mask, stat] = gap_masks[i];
				std::string name;
				for (uint32_t bit = 0; bit < GapNoteCount; bit++) {
					if ((mask & (1u << bit)) != 0) {
						name += name.empty() ? "" : "+";
						name += names[bit];
					}
				}
				char item[96];
				std::snprintf(item, sizeof(item), " %s=%.2f/%" PRIu64,
				              name.empty() ? "nothing" : name.c_str(), stat.ms, stat.count);
				top_text += item;
			}
			std::fprintf(stderr, "GpuGaps frame %" PRIu64 ": top:%s\n", frame, top_text.c_str());
			// Across command buffers; neither the span nor "between" holds this time.
			std::fprintf(stderr,
			             "GpuCross frame %" PRIu64 ": nothing recorded=%.2f/%" PRIu64
			             " something recorded=%.2f/%" PRIu64 "\n",
			             frame, cross_ms[0], cross_n[0], cross_ms[1], cross_n[1]);
		}
		std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
		for (const auto& [key, stat]: sorted) {
			if (key != 0 && stat.kind < WorkKindBase) {
				timed += stat.ms; // the guest's commands, as before; the internal ones are their own bucket
			}
		}
		// stderr: test runs keep the log silent.
		std::fprintf(stderr, "GpuTiming frame %" PRIu64 ": span=%.1fms commands=%.1fms\n", frame,
		             total, timed);
		static const size_t top = [] {
			const char* value = std::getenv("KYTY_GPU_TIMING_TOP");
			return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t {20};
		}();
		for (size_t i = 0; i < sorted.size() && i < top; i++) {
			const auto& [key, stat] = sorted[i];
			const bool split = std::find(split_keys.begin(), split_keys.end(), key) != split_keys.end();
			if (stat.kind >= WorkKindBase && key != 0) {
				const auto name = work_names.find(key);
				std::fprintf(stderr, "GpuTiming   internal %-40s %7.2fms x%" PRIu64 "\n",
				             name != work_names.end() ? name->second.c_str() : "?", stat.ms, stat.count);
				continue;
			}
			std::fprintf(stderr, "GpuTiming   %s 0x%016" PRIx64 " %7.2fms x%" PRIu64 "\n",
			             key == 0 ? "between " : (stat.kind == Draw ? "draw    " : (split ? "cs_split" : "cs_nativ")),
			             key, stat.ms, stat.count);
		}
		std::fflush(stderr);
	}
	const bool active = frame >= window.first && frame - window.first < window.count;
	Detail::g_active.store(active, std::memory_order_relaxed);
	if (active) {
		state.window_frames++;
	}
	// The window's summary, a few frames after its last one so that their blocks have been read.
	if (!state.window_printed && frame >= window.first + window.count + 8u && state.window_frames != 0) {
		state.window_printed = true;
		ReportWindow(state, window.first, state.window_frames);
	}
}

void SetQueue(uint32_t queue) {
	t_queue = queue;
}

void NoteComputeLanes(uint64_t hash, uint32_t lanes) {
	if (GetWindow().count == 0) {
		return;
	}
	auto&           state = GetState();
	std::lock_guard lock(state.stats_mutex);
	state.lanes[hash] = lanes;
}

} // namespace Libs::Graphics::GpuTiming
