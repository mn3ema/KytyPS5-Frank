#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERMERGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERMERGE_H_

namespace Libs::Graphics {

// Barrier merging (KYTY_BARRIER_MERGE=1, or in-run with KYTY_SYNC_TOGGLE_MODE=barriermerge).
//
// The renderer records ~120,000 barriers a second, ~3/4 of them directly behind another barrier
// with no command in between (KYTY_BARRIER_STATS, 11-09). A barrier recorded right after a full
// memory barrier (ALL_COMMANDS / MEMORY_WRITE -> ALL_COMMANDS / MEMORY_READ | MEMORY_WRITE, no
// image barriers) with no command in between orders nothing that barrier did not: every command
// before it has finished at every stage and its writes are visible to every later access. Such a
// barrier is left out when it carries no image layout transition, no queue-family transfer and no
// host stage or host access (host is outside ALL_COMMANDS).
//
// Implemented at the Vulkan dispatch level (VULKAN_HPP_DEFAULT_DISPATCHER), which sees the
// commands of a command buffer in recording order whichever path records them. Every command that
// does work (draws, dispatches, copies, clears, blits, resolves, rendering begin/end, queries,
// timestamps, events, secondary command buffers, markers) and every begin/end of a command buffer
// clears the "behind a full barrier" state of its command buffer; state commands (binds, push
// constants, dynamic state) do not. Barriers are left out only on the renderer's own threads
// (the GPU thread and the Vulkan recording thread, SetBarrierMergeThread).
void InstallBarrierMerge();
// The calling thread records renderer command buffers (barriers may be left out there).
void SetBarrierMergeThread() noexcept;

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERMERGE_H_ */
