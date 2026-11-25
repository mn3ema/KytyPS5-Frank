#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERMERGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERMERGE_H_

#include "graphics/host_gpu/vulkanCommon.h"

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
//
// Barrier coalescing (KYTY_BARRIER_COALESCE=1, or in-run with KYTY_SYNC_TOGGLE_MODE=coalesce;
// v438zg, off by default). A frame records ~3,200 barriers in ~720 groups with no command between
// the barriers of a group, and every barrier that reaches the driver costs GPU time whether or
// not it has anything to wait for (merge on: -800 barriers, -1.2 ms a frame; KYTY_PRECISE_FLUSHES:
// +800, +1.6 ms). The barriers of a group are held back and recorded as one vkCmdPipelineBarrier2
// before the next command that does work (the same commands that clear the merge's state):
// - their memory and buffer barriers, and image barriers that change no layout, become one global
//   memory barrier whose source and destination stage and access masks are the union of theirs.
//   It waits for at least what each of them waited for and holds back at least what each held
//   back, at the same place relative to every command, so it orders everything they ordered;
// - image barriers that change a layout stay image barriers, with their own masks, in the same
//   call (a second transition of the same image records the group so far first).
// Recorded at once, after the group so far: barriers with a host stage or access, a queue-family
// transfer, dependency flags or an extension structure, and barriers from other threads. Off with
// KYTY_BARRIER_SINK=1 (the sink reads the guest's barriers one by one below this level).
// [barrier-merge] prints the calls saved.
void InstallBarrierMerge();
// The calling thread records renderer command buffers (barriers may be left out there).
void SetBarrierMergeThread() noexcept;
// Work is about to be recorded into `command` through pointers the dispatcher hooks don't see
// (FidelityFX, NGX): held-back barriers are recorded first, and the next barrier is not "behind a
// full barrier". Call it before such commands and again before the first barrier after them.
void NoteUnhookedCommands(vk::CommandBuffer command);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERMERGE_H_ */
