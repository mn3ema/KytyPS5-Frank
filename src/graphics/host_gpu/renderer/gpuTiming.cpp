#include "graphics/host_gpu/renderer/gpuTiming.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
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
};

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
			for (const auto& pair: block.pairs) {
				const auto before = values[pair.before];
				const auto after  = values[pair.after];
				if (after >= before) {
					auto& stat = state.stats[pair.key];
					stat.ms += static_cast<double>(after - before) * state.period_ns / 1e6;
					stat.count++;
					stat.kind = pair.kind;
					const auto q = (std::min)(pair.queue, QueueSlots - 1);
					state.queue_ms[q] += static_cast<double>(after - before) * state.period_ns / 1e6;
					state.queue_n[q]++;
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
				}
				previous_after = after;
			}
			if (!block.pairs.empty()) {
				const auto first = values[block.pairs.front().before];
				const auto last  = values[block.pairs.back().after];
				if (last >= first) {
					state.total_ms += static_cast<double>(last - first) * state.period_ns / 1e6;
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

void BeforeSlow(RenderContext& context, CommandBuffer& buffer) {
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

void Present(uint64_t frame) {
	const auto& window = GetWindow();
	if (window.count == 0) {
		return;
	}
	auto& state = GetState();
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		std::vector<std::pair<uint64_t, Stat>> sorted;
		double                                 timed = 0.0;
		double                                 total = 0.0;
		// Buckets: draws, native compute, two-lane (64 on 32) compute, gaps.
		double   bucket_ms[4] {};
		uint64_t bucket_n[4] {};
		std::vector<uint64_t> split_keys;
		std::array<double, QueueSlots>   queue_ms {};
		std::array<uint64_t, QueueSlots> queue_n {};
		std::vector<std::pair<uint32_t, Stat>> gap_masks;
		std::array<double, 4>                  gap_len_ms {};
		std::array<uint64_t, 4>                gap_len_n {};
		uint64_t                               gap_barriers = 0;
		{
			std::lock_guard lock(state.stats_mutex);
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
					if (stat.kind == Draw) {
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
		             " cs_native=%.2f/%" PRIu64 " cs_split=%.2f/%" PRIu64 " gaps=%.2f/%" PRIu64 "\n",
		             frame, total, bucket_ms[0], bucket_n[0], bucket_ms[1], bucket_n[1], bucket_ms[2],
		             bucket_n[2], bucket_ms[3], bucket_n[3]);
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
		}
		std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
		for (const auto& [key, stat]: sorted) {
			if (key != 0) {
				timed += stat.ms;
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
			std::fprintf(stderr, "GpuTiming   %s 0x%016" PRIx64 " %7.2fms x%" PRIu64 "\n",
			             key == 0 ? "between " : (stat.kind == Draw ? "draw    " : (split ? "cs_split" : "cs_nativ")),
			             key, stat.ms, stat.count);
		}
		std::fflush(stderr);
	}
	const bool active = frame >= window.first && frame - window.first < window.count;
	Detail::g_active.store(active, std::memory_order_relaxed);
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
