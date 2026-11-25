#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPAD_H_

#include "graphics/host_gpu/renderer/syncToggle.h"

// KYTY_SYNC_TOGGLE=<seconds> with KYTY_SYNC_TOGGLE_MODE=gpupad (v445m): a MEASUREMENT ONLY. In
// the on windows the GPU thread records one extra compute dispatch at every guest flip, between
// two barriers so that it runs by itself: a fixed amount of arithmetic on the GPU's own local
// memory (the local-memory warm-up shader, 64 threads a group, 256 loop steps a thread), writing
// 64 dwords of a scratch buffer of its own. It touches no guest memory and no cache state. If
// the frame rate of the on windows equals the off windows', the GPU had at least that much time
// to spare in a frame; if it drops by the pad's share of the frame, the GPU binds.
// KYTY_GPU_PAD_US=<n> (default 3000): the GPU time aimed for. The dispatch is timed with two
// timestamps and its size (rows of 64 groups, about one million loop steps a row) is adjusted
// over the first on window, then kept for the run. KYTY_GPU_PAD_ROWS=<n>: that many rows, no
// adjustment (the iteration count: n x 64 groups x 64 threads x 256 steps).
// "[gpu-pad]" every 5 s while on: the rows and the GPU time measured. Under KYTY_GPU_TIMING the
// dispatch has its own key ("internal dispatch localMemoryWarmup.cpp:<line>").
namespace Libs::Graphics {

class CommandBuffer;

namespace GpuPad {

void RecordSlow(CommandBuffer& buffer); // localMemoryWarmup.cpp

// Once per guest flip on the GPU thread, with the scheduler's current command buffer.
inline void AtFlip(CommandBuffer& buffer) {
	if (SyncToggle::GpuPadWindow()) {
		RecordSlow(buffer);
	}
}

} // namespace GpuPad
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPAD_H_
