#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHRUN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHRUN_H_

#include <array>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

// Dispatch runs (KYTY_DISPATCH_RUNS=1). A queue's command processor launches back-to-back
// dispatches without waiting for one another; the game orders them only where its command
// stream asks for it (EVENT_WRITE CS_PARTIAL_FLUSH, ACQUIRE_MEM, RELEASE_MEM, WAIT_REG_MEM, ...).
// The renderer put a full barrier before and after every dispatch instead, so the async compute
// chains of hundreds of tiny dispatches ran one at a time with the GPU draining between them
// (the title: ~640 indirect dispatches of 0xbad108e74fb72e9f, 28 ms of GPU time, plus ~25 ms of
// gaps, a frame). A run is a sequence of dispatches separated only by SET_SH_REG and debug-marker
// packets: inside it the barriers between the dispatches are left out, the thread counts of its
// indirect dispatches are converted together, and the barrier after the run's last dispatch
// orders the whole run for what follows.
struct DispatchRunIndirect {
	const uint32_t*         packet = nullptr;
	uint64_t                args   = 0;
	std::array<uint32_t, 3> local {}; // 0: the value current at the run's first dispatch
};

struct DispatchRunConverted {
	vk::Buffer        groups_buffer;
	vk::DeviceSize    groups_offset = 0;
	vk::DeviceAddress threads       = 0;
};

struct DispatchRunState {
	const uint32_t* packet    = nullptr; // the dispatch being recorded
	bool            continues = false;   // an earlier dispatch of the same run was recorded last
	bool            more      = false;   // another dispatch of the run follows
	// KYTY_BARRIER_REASON_STATS (statistics only): why this dispatch does not continue the run of
	// the dispatch before it (BarrierReason::Why; 0: it does).
	uint32_t        why       = 0;
	// The run's upcoming thread-dimension indirect dispatches (from this one), collected by the
	// command processor when this one has no converted counts yet.
	std::vector<DispatchRunIndirect>                                  upcoming;
	std::unordered_map<const uint32_t*, DispatchRunConverted>         converted;
	bool pending_post = false; // a dispatch of the run left out its barrier after it
	uint64_t runs = 0, members = 0, batched = 0, skipped_barriers = 0; // stats
	uint64_t stale_hits = 0; // run starts that found entries of an earlier instance (cumulative)
};

inline bool DispatchRunsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DISPATCH_RUNS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Per command-processor thread (KYTY_ASYNC_THREAD runs the compute queues on their own).
inline DispatchRunState& DispatchRun() {
	thread_local DispatchRunState state;
	return state;
}

// Mesh-args runs (KYTY_MESH_ARGS_RUNS=1). Each DRAW_INDIRECT of a fast-launch mesh draw had its
// counts converted into a mesh-tasks command by a one-thread dispatch of its own: the rendering
// instance ended, a barrier, the dispatch, a barrier, and rendering began again (~290 a frame in
// the jungle, about 38 % of the GPU's drains). The arguments of the DRAW_INDIRECTs that follow,
// up to the command stream's next sync or memory-writing packet, cannot change before those draws
// read them (on the console their writers are ordered by such packets too), so the command
// processor collects them at the first draw and the renderer converts them all there, between one
// barrier before and one after. The later draws use their converted commands and keep their
// rendering instance.
struct MeshArgsRunEntry {
	const uint32_t* packet = nullptr;
	uint64_t        args   = 0;
	vk::Buffer      buffer;
	vk::DeviceSize  offset = 0;
};

struct MeshArgsRunState {
	const uint32_t*               packet = nullptr; // the DRAW_INDIRECT being recorded
	std::vector<MeshArgsRunEntry> upcoming;         // collected when packet has no conversion
	std::vector<MeshArgsRunEntry> converted;        // the current batch, in command order
	size_t                        next = 0;         // first batch entry not used yet
	uint64_t                      generation = 0;   // changes whenever a new batch replaces it
	const uint32_t*               last_used_packet = nullptr;
	uint64_t                      last_used_generation = 0;
	uint64_t batches = 0, batched = 0, hits = 0, single = 0; // stats

	void Invalidate() {
		packet = nullptr;
		upcoming.clear();
		converted.clear();
		next = 0;
		generation++;
		last_used_packet = nullptr;
		last_used_generation = 0;
	}

	[[nodiscard]] bool Contains(const uint32_t* p) const {
		for (size_t i = next; i < converted.size(); i++) {
			if (converted[i].packet == p) return true;
		}
		return false;
	}
	[[nodiscard]] const MeshArgsRunEntry* Peek(const uint32_t* p) const {
		for (size_t i = next; i < converted.size(); i++) {
			if (converted[i].packet == p) return &converted[i];
		}
		return nullptr;
	}
	// The batch entry of `p`; entries before it are skipped (their draws were not mesh draws).
	const MeshArgsRunEntry* Take(const uint32_t* p) {
		for (size_t i = next; i < converted.size(); i++) {
			if (converted[i].packet == p) {
				next = i + 1;
				return &converted[i];
			}
		}
		return nullptr;
	}
};

inline bool MeshArgsRunsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MESH_ARGS_RUNS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// GPU thread only.
inline MeshArgsRunState& MeshArgsRun() {
	static MeshArgsRunState state;
	return state;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHRUN_H_
