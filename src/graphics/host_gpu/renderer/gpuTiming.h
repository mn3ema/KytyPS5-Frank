#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMING_H_

#include <atomic>
#include <cstdint>

// Diagnostics: KYTY_GPU_TIMING=<first frame>:<frames> times every draw and dispatch on the host
// GPU (timestamps before and after, which the per-command barriers make exact) and logs, per
// presented frame in that window, the GPU time of the costliest shaders and of everything
// between commands. Off by default.
namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;

namespace GpuTiming {

namespace Detail {
inline std::atomic<bool> g_active {false};
// What the recording thread put between the last timed command and the next (GapNote bits).
inline thread_local uint32_t t_gap_mask     = 0;
inline thread_local uint32_t t_gap_barriers = 0;
void BeforeSlow(RenderContext& context, CommandBuffer& buffer);
void AfterSlow(RenderContext& context, CommandBuffer& buffer, uint64_t key, uint32_t kind);
} // namespace Detail

enum Kind : uint32_t { Draw = 1, Dispatch = 2 };

// KYTY_GPU_TIMING: what fills the GPU time between timed commands ("GpuGaps" lines).
enum GapNote : uint32_t {
	GapBarrier      = 1u << 0u, // an emulator barrier
	GapGuestBarrier = 1u << 1u, // the guest's own barrier (EmitGlobalBarrier)
	GapCopyBuffer   = 1u << 2u,
	GapCopyImage    = 1u << 3u, // image<->image, buffer<->image, resolve
	GapClear        = 1u << 4u, // clear colour / depth-stencil image
	GapFill         = 1u << 5u, // fill / update buffer
	GapRendering    = 1u << 6u, // begin or end of dynamic rendering (load/store ops)
	GapDispatch     = 1u << 7u, // a dispatch nobody timed (internal compute)
	GapDraw         = 1u << 8u, // a draw nobody timed
	GapNoteCount    = 9u,
};

inline void Note(uint32_t bit) {
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		Detail::t_gap_mask |= bit;
		if ((bit & (GapBarrier | GapGuestBarrier)) != 0) {
			Detail::t_gap_barriers++;
		}
	}
}

inline void Before(RenderContext& context, CommandBuffer& buffer) {
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		Detail::BeforeSlow(context, buffer);
	}
}

inline void After(RenderContext& context, CommandBuffer& buffer, uint64_t key, uint32_t kind) {
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		Detail::AfterSlow(context, buffer, key, kind);
	}
}

// Called once per presented frame.
void Present(uint64_t frame);

// The guest queue whose commands the calling thread translates (0 graphics, 1-2 async compute):
// the timed commands are also summed per queue ("GpuQueues" line).
void SetQueue(uint32_t queue);

// Profiling buckets: a compute shader that emulates its 64-lane wave as two lanes per host
// invocation (32-lane host subgroups) is reported apart from the native ones.
void NoteComputeLanes(uint64_t hash, uint32_t lanes);

} // namespace GpuTiming
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMING_H_
