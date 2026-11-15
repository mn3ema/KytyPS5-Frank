#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include "common/fpsLog.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::Occlusion {

namespace {

constexpr uint32_t BlockQueries = 1024;
constexpr uint32_t BlockCount   = 64;
constexpr uint32_t NoBlock      = ~0u;
constexpr uint64_t ReadyBit     = 1ull << 63u;
constexpr uint64_t CounterMask  = ReadyBit - 1u;
constexpr uint32_t Dbs          = 16;

struct Block {
	uint32_t used     = 0;     // queries recorded (CP thread)
	uint32_t consumed = 0;     // queries read back (priority thread)
	bool     closed   = true;  // no more queries will be recorded in it
	bool     busy     = false; // reset and not yet fully read back
};

struct Pending {
	uint32_t block = 0;
	uint32_t index = 0;
};

struct Stats {
	uint64_t dumps    = 0;
	uint64_t counted  = 0;
	uint64_t draws    = 0;
	uint64_t starved  = 0;
	uint64_t visible  = 0; // dumps whose running count moved since the dump before
	std::map<uint32_t, uint64_t> controls;
};

struct State {
	vk::Device                    device;
	vk::QueryPool                 pool;
	vk::QueryControlFlags         flags;
	bool                          failed = false;
	std::mutex                    mutex;
	std::array<Block, BlockCount> blocks;
	uint32_t                      next_block = 0;
	uint32_t                      current    = NoBlock;
	bool                          open       = false;
	bool                          counted_draw = false; // Prepare chose to count this draw
	uint64_t                      recorded   = 0; // queries recorded so far (CP thread, under mutex)
	uint64_t                      consumed   = 0; // queries read back so far (priority thread)
	uint64_t                      count      = 0; // running sample count (priority thread)
	std::deque<Pending>           pending;
	uint64_t                      last_dump_flip = ~0ull;
	Stats                         stats;
	// KYTY_OCCLUSION_LOG
	uint64_t                      draws_since_dump   = 0;
	uint64_t                      guest_since_dump   = 0;
	uint64_t                      counted_since_dump = 0;
	std::map<uint64_t, uint64_t>  begin_values; // begin slot -> running count (priority thread)
	std::map<uint64_t, uint64_t>  last_samples; // begin slot -> samples last time (GPU thread)
	uint32_t                      queries_logged = 0;
};

State& GetState() {
	static State state;
	return state;
}

// KYTY_OCCLUSION_LOG=1: the [occl-real] stats every 5 s even with KYTY_FPS_ONLY, and the first
// 400 query results ([occl-query]: begin/end dump addresses, samples, draws between them).
bool LogEnabled() {
	static const bool enabled = Common::FpsLogDiagnostics() || std::getenv("KYTY_OCCLUSION_LOG") != nullptr;
	return enabled;
}

bool QueryLogEnabled() {
	static const bool enabled = std::getenv("KYTY_OCCLUSION_LOG") != nullptr;
	return enabled;
}

// ZPASS_INCREMENT_DISABLE clear and a ZPASS_ENABLE (or PERFECT_ZPASS_COUNTS) set: the game writes
// 0x11000102 around its queries and 0 otherwise.
bool CountingControl(uint32_t control) {
	return (control & 1u) == 0 && (control & 0x00000f02u) != 0;
}

// Only while the game queries: a dump in this guest flip or the one before.
bool QueryingNow() {
	const auto& state = GetState();
	return state.last_dump_flip != ~0ull && state.last_dump_flip + 1u >= Sync::GuestFlipCount();
}

bool Counting(const CommandBuffer& buffer) {
	auto&      state   = GetState();
	const auto control = buffer.GetRegisters().GetDbCountControl();
	state.draws_since_dump++;
	if (LogEnabled()) {
		std::lock_guard lock(state.mutex);
		state.stats.draws++;
		state.stats.controls[control]++;
	}
	return CountingControl(control) && QueryingNow();
}

bool CreatePool(RenderContext& context) {
	auto& state = GetState();
	if (state.pool) {
		return true;
	}
	if (state.failed) {
		return false;
	}
	state.device = context.GetGraphics().device;
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eOcclusion;
	info.queryCount = BlockQueries * BlockCount;
	if (state.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
		std::fprintf(stderr, "[occl-real] no occlusion query pool; queries stay always visible\n");
		state.failed = true;
		return false;
	}
	if (context.GetGraphics().occlusion_query_precise) {
		state.flags = vk::QueryControlFlagBits::ePrecise;
	}
	std::fprintf(stderr, "[occl-real] host occlusion queries (%s)\n",
	             state.flags ? "precise" : "not precise");
	return true;
}

// Under the mutex.
void CloseCurrent(State& state) {
	if (state.current == NoBlock) {
		return;
	}
	auto& block  = state.blocks[state.current];
	block.closed = true;
	if (block.consumed == block.used) {
		block.busy = false;
	}
	state.current = NoBlock;
}

// Priority thread: adds the samples of the queries recorded before <upto> to the running count.
void Consume(State& state, uint64_t upto) {
	std::lock_guard lock(state.mutex);
	std::vector<uint64_t> values;
	while (state.consumed < upto && !state.pending.empty()) {
		const auto first = state.pending.front();
		uint32_t   n     = 1;
		while (n < state.pending.size() && state.consumed + n < upto &&
		       state.pending[n].block == first.block && state.pending[n].index == first.index + n) {
			n++;
		}
		values.assign(n, 0);
		const auto result = state.device.getQueryPoolResults(
		    state.pool, first.block * BlockQueries + first.index, n, n * sizeof(uint64_t),
		    values.data(), sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
		if (result == vk::Result::eSuccess) {
			for (const auto v: values) {
				state.count = (state.count + v) & CounterMask;
			}
		} else {
			std::fprintf(stderr, "[occl-real] results unavailable (%d) for %u draws\n",
			             static_cast<int>(result), n);
		}
		auto& block = state.blocks[first.block];
		block.consumed += n;
		if (block.closed && block.consumed == block.used) {
			block.busy = false;
		}
		state.pending.erase(state.pending.begin(), state.pending.begin() + n);
		state.consumed += n;
	}
}

void Log(State& state) {
	static auto start = std::chrono::steady_clock::now();
	const auto  now   = std::chrono::steady_clock::now();
	const auto  s     = std::chrono::duration<double>(now - start).count();
	if (s < 5.0) {
		return;
	}
	start = now;
	Stats stats;
	{
		std::lock_guard lock(state.mutex);
		stats       = std::move(state.stats);
		state.stats = {};
	}
	std::string controls;
	for (const auto& [control, n]: stats.controls) {
		char text[48];
		std::snprintf(text, sizeof(text), " %08x:%.0f", control, static_cast<double>(n) / s);
		controls += text;
	}
	std::fprintf(stderr,
	             "[occl-real] %.1f dumps/s (%.1f moved), %.0f counted of %.0f draws/s, %.1f "
	             "uncounted (no free block)/s, DB_COUNT_CONTROL/s:%s\n",
	             static_cast<double>(stats.dumps) / s, static_cast<double>(stats.visible) / s,
	             static_cast<double>(stats.counted) / s, static_cast<double>(stats.draws) / s,
	             static_cast<double>(stats.starved) / s, controls.c_str());
}

} // namespace

bool CountsDraw(const CommandBuffer& buffer) {
	return Enabled() && !GetState().failed &&
	       CountingControl(buffer.GetRegisters().GetDbCountControl()) && QueryingNow();
}

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OCCLUSION_REAL");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void Prepare(RenderContext& context, CommandBuffer& buffer) {
	if (!Enabled()) {
		return;
	}
	auto& state        = GetState();
	state.open         = false;
	state.counted_draw = false;
	if (!Counting(buffer) || !CreatePool(context)) {
		return;
	}
	std::lock_guard lock(state.mutex);
	if (state.current != NoBlock && state.blocks[state.current].used < BlockQueries) {
		state.counted_draw = true;
		return;
	}
	CloseCurrent(state);
	auto& block = state.blocks[state.next_block % BlockCount];
	if (block.busy) {
		state.stats.starved++;
		return;
	}
	const auto index = state.next_block % BlockCount;
	state.next_block++;
	block = {.used = 0, .consumed = 0, .closed = false, .busy = true};
	buffer.EndRendering();
	buffer.Recorder().resetQueryPool(state.pool, index * BlockQueries, BlockQueries);
	state.current      = index;
	state.counted_draw = true;
}

void Begin(RenderContext& context, CommandBuffer& buffer) {
	(void)context;
	if (!Enabled()) {
		return;
	}
	auto& state = GetState();
	if (state.current == NoBlock || !state.counted_draw) {
		return;
	}
	state.counted_draw = false;
	const auto& block = state.blocks[state.current];
	if (block.closed || block.used >= BlockQueries) {
		return;
	}
	buffer.Recorder().beginQuery(state.pool, state.current * BlockQueries + block.used,
	                             state.flags);
	state.open = true;
}

void End(RenderContext& context, CommandBuffer& buffer) {
	(void)context;
	auto& state = GetState();
	if (!state.open) {
		return;
	}
	state.open = false;
	std::lock_guard lock(state.mutex);
	auto&           block = state.blocks[state.current];
	buffer.Recorder().endQuery(state.pool, state.current * BlockQueries + block.used);
	state.pending.push_back({state.current, block.used});
	block.used++;
	state.recorded++;
	state.stats.counted++;
	state.counted_since_dump++;
}

void NoteGuestDraw() {
	GetState().guest_since_dump++;
}

void Dump(RenderContext& context, CommandBuffer& buffer, uint64_t address) {
	auto&      state   = GetState();
	const auto control = buffer.GetRegisters().GetDbCountControl();
	const auto guest   = state.guest_since_dump;
	state.guest_since_dump = 0;
	const auto draws   = state.draws_since_dump;
	const auto counted = state.counted_since_dump;
	state.draws_since_dump   = 0;
	state.counted_since_dump = 0;
	state.last_dump_flip     = Sync::GuestFlipCount();
	// A query is a dump to a 16-byte aligned begin slot and one to the end slot 8 bytes on. Both
	// are written at once, ready, so a reader in the same frame (the GPU, or the CPU right after
	// the frame's label) sees a result: the begin as zero, the end as the samples this query had
	// the last time the GPU finished it. Once the GPU has run the draws, the end slot gets this
	// time's samples.
	const bool is_end  = (address & 8u) != 0;
	const auto begin   = is_end ? address - 8u : address;
	auto*      results = reinterpret_cast<volatile uint64_t*>(address);
	uint64_t   shown   = 0;
	if (is_end) {
		const auto last = state.last_samples.find(begin);
		shown           = last != state.last_samples.end() ? last->second : 0;
	}
	for (uint32_t db = 1; db < Dbs; db++) {
		results[db * 2u] = ReadyBit;
	}
	results[0] = ReadyBit | shown;
	uint64_t upto = 0;
	{
		std::lock_guard lock(state.mutex);
		upto = state.recorded;
		state.stats.dumps++;
	}
	auto& gpu = context.GetGpu();
	context.GetCommandScheduler().DeferPriorityOperation(
	    [&state, &gpu, upto, address, begin, is_end, shown, control, guest, draws, counted] {
		    if (state.pool) {
			    Consume(state, upto);
		    }
		    const auto value = state.count;
		    if (!is_end) {
			    state.begin_values[begin] = value;
			    if (state.begin_values.size() > 65536) {
				    state.begin_values.clear();
				    state.begin_values[begin] = value;
			    }
			    return;
		    }
		    const auto start = state.begin_values.find(begin);
		    if (start == state.begin_values.end()) {
			    return;
		    }
		    const auto samples = (value - start->second) & CounterMask;
		    state.begin_values.erase(start);
		    if (samples != 0) {
			    std::lock_guard lock(state.mutex);
			    state.stats.visible++;
		    }
		    // Guest memory is written on the GPU thread, as deferred labels are.
		    (void)gpu.TrySendCommand([&state, address, begin, samples, shown, control, guest,
		                              draws, counted] {
			    state.last_samples[begin] = samples;
			    if (state.last_samples.size() > 65536) {
				    state.last_samples.clear();
			    }
			    auto* results = reinterpret_cast<volatile uint64_t*>(address);
			    results[0]    = ReadyBit | samples;
			    if (QueryLogEnabled() && state.queries_logged < 400) {
				    state.queries_logged++;
				    std::fprintf(stderr,
				                 "[occl-query] %016llx samples=%llu shown=%llu guest=%llu draws=%llu "
				                 "counted=%llu ctrl=%08x flip=%llu\n",
				                 static_cast<unsigned long long>(begin),
				                 static_cast<unsigned long long>(samples),
				                 static_cast<unsigned long long>(shown),
				                 static_cast<unsigned long long>(guest),
				                 static_cast<unsigned long long>(draws),
				                 static_cast<unsigned long long>(counted), control,
				                 static_cast<unsigned long long>(Sync::GuestFlipCount()));
			    }
		    });
	    });
	if (LogEnabled()) {
		Log(state);
	}
}

} // namespace Libs::Graphics::Occlusion
