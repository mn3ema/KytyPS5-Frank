#ifndef GRAPHICS_HOST_GPU_RENDERER_DRAW_PIPE_H
#define GRAPHICS_HOST_GPU_RENDERER_DRAW_PIPE_H

// Part 1, the two-stage draw pipeline (KYTY_DRAW_PIPE=1, docs/draw_pipe_plan.md).
//
// Stage A, the command-processor thread, decodes the command stream and keeps the register
// state. A direct draw (DRAW_INDEX_2, DRAW_INDEX_OFFSET_2, DRAW_INDEX_AUTO) becomes a packet: a
// copy of the registers, the draw's arguments, and its programs, selected on stage A
// (GetGraphicsPrograms with the SRT refresh) with copies of their resource snapshots.
// Stage B, its own thread, runs the rest of the draw in order exactly as the command processor
// did (render targets, bindings, descriptors, recording) on the copied registers.
//
// Everything else that touches the renderer runs on stage A after stage B has run dry (Drain):
// every packet but register writes and direct draws, the commands other threads send, the end of
// each execution slice, a fault on stage A, and a draw that writes memory (its successors' SRT
// reads may read what it writes).

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics {
class CommandProcessor;
class RenderContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::DrawPipe {

// What RefreshShaders produces for a draw, made on stage A.
struct PreparedPrograms {
	bool                                  ps_active         = false;
	uint32_t                              color_output_mask = 0;
	PipelineCache::GraphicsPrograms       programs;
	std::array<ShaderVertexInputInfo, 3>  vertex_info;
	ShaderPixelInputInfo                  ps_input_info;
	// The program cache entries' snapshots, which the next lookup of a shader overwrites:
	// vertex stages 0..2, then the pixel stage. The input infos point here.
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 4> resources;
	bool                                  writes_memory = false;
	// The guest bytes the draw's buffer writes reach (base + write extent, or the whole V#);
	// writes_unbounded: storage images, atomics on images or raw-pointer stores, which have none.
	std::vector<std::pair<uint64_t, uint64_t>> write_ranges;
	bool                                       writes_unbounded = false;
};

enum class Kind : uint8_t { DrawIndex, DrawAuto, Event };

struct Packet {
	Kind              kind      = Kind::DrawIndex;
	uint64_t          submit_id = 0;
	CommandProcessor* cp        = nullptr;
	HW::Context       ctx;
	HW::UserConfig    ucfg;
	HW::Shader        sh;
	DrawIndexArgs     index_args;
	DrawAutoArgs      auto_args;
	PreparedPrograms  prepared;
	// Kind::Event: a guest EVENT_WRITE barrier (CommandProcessor::TriggerEvent), in order.
	uint32_t          event_type  = 0;
	uint32_t          event_index = 0;
};

enum class Reason : uint8_t {
	Packet,   // a packet other than register writes and direct draws
	Commands, // commands from other threads
	Slice,    // the end of an execution slice
	Fault,    // a fault on stage A
	GpuState, // stage A asked the buffer cache about GPU-written memory
	Writes,   // a draw that writes memory where no range bounds it
	PendingRead, // stage A reads bytes a queued draw writes
	Unready,  // a direct draw that could not be prepared
	Count
};

// KYTY_DRAW_PIPE=1, without the async-compute thread and indirect replay.
[[nodiscard]] bool Enabled();
// Packets handed to stage B that it has not finished, and this thread is stage A.
[[nodiscard]] bool InFlightHere();
[[nodiscard]] bool IsStageB();
// Stage A: a free packet (waits while all are in flight), started lazily.
Packet& Acquire(RenderContext& context);
// Stage A: hands the packet Acquire returned to stage B.
void Publish();
// Stage A: gives the packet Acquire returned back unused.
void Cancel();
// Stage A: an EVENT_WRITE of this type only records a barrier (or nothing): it can run on the draw
// thread in order instead of draining it.
[[nodiscard]] bool DeferrableEvent(uint32_t event_type);
// Stage A, with draws in flight: queues the event for the draw thread.
void QueueEvent(CommandProcessor& cp, uint32_t event_type, uint32_t event_index);
// Stage A: waits until stage B has run every packet. Nothing when none are in flight.
void Drain(Reason reason, uint32_t opcode = 0);
// Any thread: [address, address + size) overlaps the writes of a queued draw stage B has not
// recorded yet (so the GPU-written tracking does not know them). Cheap when none is queued.
[[nodiscard]] bool PendingWriteOverlaps(uint64_t address, uint64_t size);
// Stage A: drains when [address, address + size) overlaps a queued draw's writes (its next read
// of those bytes must see them recorded). Nothing when no ranged writer is queued.
void DrainIfPendingWrite(uint64_t address, uint64_t size);
[[nodiscard]] bool HasPendingWrites();
// KYTY_DRAW_PIPE_RANGES=0: every memory-writing draw drains as in v355 (default: ranged ones queue).
[[nodiscard]] bool RangesEnabled();
// Stage B, while it runs a packet: the prepared programs (RefreshShaders uses them).
[[nodiscard]] const PreparedPrograms* CurrentPrepared();
// Stage A: the opcode is a direct draw stage B can take (set while its handler runs).
void SetDrawPacket(bool draw_packet);
[[nodiscard]] bool DrawPacket();

// Diagnostics: per 5 s with KYTY_FPS_LOG ([draw-pipe] lines), from stage A.
void NoteSyncDraw();

} // namespace Libs::Graphics::DrawPipe

#endif // GRAPHICS_HOST_GPU_RENDERER_DRAW_PIPE_H
