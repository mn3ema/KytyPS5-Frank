#include "graphics/host_gpu/renderer/drawStats.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace Libs::Graphics::DrawStats {

namespace {

constexpr uint32_t BlockQueries = 1024;
constexpr uint32_t BlockCount   = 16;
constexpr uint32_t Values       = 3; // clipping invocations, clipping primitives, fragments

struct Entry {
	Info     info;
	uint64_t flip  = 0;
	uint32_t index = 0;
};

struct Block {
	uint64_t           tick = 0;
	uint32_t           used = 0;
	bool               open = false;
	std::vector<Entry> entries;
	std::atomic<bool>  in_flight {false};
};

struct Window {
	uint64_t first = 0;
	uint64_t flips = 1;
	uint64_t every = 0;
	bool     on    = false;
};

const Window& GetWindow() {
	static const Window window = [] {
		Window      result;
		const char* value = std::getenv("KYTY_DRAW_STATS");
		if (value == nullptr) {
			return result;
		}
		unsigned long long first = 0, flips = 1, every = 0;
		const int          n = std::sscanf(value, "%llu:%llu:%llu", &first, &flips, &every);
		if (n >= 1) {
			result = {first, flips == 0 ? 1 : flips, every, true};
			std::fprintf(stderr, "[draw-stats] flips %llu (+%llu each, every %llu)\n", first, flips,
			             every);
		}
		return result;
	}();
	return window;
}

bool InWindow(uint64_t flip) {
	const auto& w = GetWindow();
	if (!w.on || flip < w.first) {
		return false;
	}
	const auto offset = flip - w.first;
	if (w.every == 0) {
		return offset < w.flips;
	}
	return offset % w.every < w.flips;
}

struct State {
	vk::Device                    device;
	vk::QueryPool                 pool;
	bool                          failed = false;
	std::array<Block, BlockCount> blocks;
	uint32_t                      next_block = 0;
	Block*                        current    = nullptr;
	uint64_t                      flip       = ~0ull;
	uint32_t                      index      = 0;
};

State& GetState() {
	static State state;
	return state;
}

void ReadBlock(State& state, Block& block) {
	if (block.used != 0) {
		std::vector<uint64_t> values(static_cast<size_t>(block.used) * Values);
		const auto            result = state.device.getQueryPoolResults(
            state.pool, static_cast<uint32_t>(&block - state.blocks.data()) * BlockQueries,
            block.used, values.size() * sizeof(uint64_t), values.data(),
            sizeof(uint64_t) * Values, vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eSuccess) {
			for (uint32_t i = 0; i < block.entries.size(); i++) {
				const auto& e = block.entries[i];
				const auto* v = values.data() + static_cast<size_t>(i) * Values;
				std::fprintf(stderr,
				             "[draw-stats] flip=%llu n=%u vs=%016llx ps=%016llx %s count=%u inst=%u "
				             "gpu_args=%d clip_in=%llu clip_out=%llu frag=%llu rt0=%llx:%ux%u blend=%d:%u/%u\n",
				             static_cast<unsigned long long>(e.flip), e.index,
				             static_cast<unsigned long long>(e.info.vs),
				             static_cast<unsigned long long>(e.info.ps), e.info.kind, e.info.count,
				             e.info.instances, e.info.gpu_args ? 1 : 0,
				             static_cast<unsigned long long>(v[0]),
				             static_cast<unsigned long long>(v[1]),
				             static_cast<unsigned long long>(v[2]),
				             static_cast<unsigned long long>(e.info.rt0), e.info.rt0_w, e.info.rt0_h,
				             e.info.blend ? 1 : 0, e.info.blend_src, e.info.blend_dst);
			}
		} else {
			std::fprintf(stderr, "[draw-stats] results unavailable (%d) for %u draws\n",
			             static_cast<int>(result), block.used);
		}
	}
	block.entries.clear();
	block.used = 0;
	block.open = false;
	block.in_flight.store(false, std::memory_order_release);
}

} // namespace

bool Enabled() {
	return GetWindow().on;
}

void Prepare(RenderContext& context, CommandBuffer& buffer) {
	if (!Enabled() || !InWindow(Sync::GuestFlipCount())) {
		return;
	}
	auto&      state     = GetState();
	auto&      scheduler = context.GetCommandScheduler();
	const auto tick      = scheduler.CurrentTick();
	if (state.failed) {
		return;
	}
	if (state.current != nullptr && state.current->tick == tick &&
	    state.current->used < BlockQueries) {
		return;
	}
	if (!state.pool) {
		state.device = context.GetGraphics().device;
		vk::QueryPoolCreateInfo info {};
		info.queryType          = vk::QueryType::ePipelineStatistics;
		info.queryCount         = BlockQueries * BlockCount;
		info.pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
		                          vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
		                          vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
		if (state.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
			std::fprintf(stderr, "[draw-stats] no pipeline statistics query pool\n");
			state.failed = true;
			return;
		}
	}
	auto& block = state.blocks[state.next_block % BlockCount];
	if (block.in_flight.load(std::memory_order_acquire)) {
		state.current = nullptr;
		return;
	}
	state.next_block++;
	block.tick = tick;
	block.used = 0;
	block.open = false;
	block.entries.clear();
	block.in_flight.store(true, std::memory_order_release);
	buffer.EndRendering();
	buffer.Handle().resetQueryPool(
	    state.pool, static_cast<uint32_t>(&block - state.blocks.data()) * BlockQueries,
	    BlockQueries);
	scheduler.DeferPriorityOperation([&state, &block] { ReadBlock(state, block); });
	state.current = &block;
}

void Begin(RenderContext& context, CommandBuffer& buffer) {
	if (!Enabled()) {
		return;
	}
	auto& state = GetState();
	auto* block = state.current;
	if (block == nullptr || block->open || block->used >= BlockQueries ||
	    block->tick != context.GetCommandScheduler().CurrentTick() ||
	    !InWindow(Sync::GuestFlipCount())) {
		return;
	}
	const auto base = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	buffer.Handle().beginQuery(state.pool, base + block->used, {});
	block->open = true;
}

void End(RenderContext& context, CommandBuffer& buffer, const Info& info) {
	if (!Enabled()) {
		return;
	}
	auto& state = GetState();
	auto* block = state.current;
	if (block == nullptr || !block->open ||
	    block->tick != context.GetCommandScheduler().CurrentTick()) {
		return;
	}
	const auto base = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	buffer.Handle().endQuery(state.pool, base + block->used);
	const auto flip = Sync::GuestFlipCount();
	if (flip != state.flip) {
		state.flip  = flip;
		state.index = 0;
	}
	block->entries.push_back({info, flip, state.index++});
	block->used++;
	block->open = false;
}

} // namespace Libs::Graphics::DrawStats
