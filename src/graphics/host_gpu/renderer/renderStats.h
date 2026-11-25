#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERSTATS_H_

#include <atomic>
#include <cstdint>
#include <source_location>

// Diagnostics, KYTY_RENDER_STATS=1 (off by default), printed every 5 s:
// - [res]: the render-area sizes the game draws at, by count, and the largest multi-target one
//   (the G-buffer: its size is the dynamic resolution the game picked), so an fps change can be
//   told apart from a resolution change.
// - [render-restart]: dynamic-rendering instances begun, and those that begin with exactly the
//   attachments the previous instance on the same command buffer ended with (a restart that
//   changed nothing), by the call site that ended it; "same targets" also counts restarts
//   whose only difference is a load op or a layout.
// - [flip-wait]: time the command-processor thread spends in WaitFlipDone, which blocks every
//   guest queue it translates.
namespace Libs::Graphics {

struct RenderState;

namespace RenderStats {

bool Enabled();
void NoteBegin(const void* buffer, const RenderState& state);
void NoteEnd(const void* buffer, const RenderState& state, const std::source_location& site);
void NoteFlipWait(uint64_t ns);

// KYTY_MULTI_DRAW_PROBE=1 (diagnostics): whether the draws a DRAW_INDIRECT_MULTI packet unrolls
// commit the same pipeline and descriptors as its first draw (so they could be replayed with
// only their arguments and draw index changed). [multi-draw] every 5 s.
bool MultiDrawProbeEnabled();
// CommitBindings: a hash of the pipeline and the descriptors it wrote (not the push data).
void NoteCommitHash(uint64_t hash);
// Folded into the next NoteCommitHash (the flattened SRT contents of the draw's stages).
void MixProbe(uint64_t hash);
uint64_t LastCommitHash();
void NoteMultiDrawPacket(uint32_t draws, uint32_t same_as_first, uint32_t draw_index_register,
                         bool indexed, uint64_t first_hash);

// KYTY_MULTI_DRAW_REPLAY=1 (experiment, off by default): the draws after the first of a
// non-indexed DRAW_INDIRECT_MULTI of mesh draws are recorded again with only what differs
// (the user data the CP wrote, as shader data, and the arguments) when nothing their resources
// are resolved from changed (RenderExecutor::ReplayMultiDraw). KYTY_SYNC_TOGGLE_MODE=replay
// turns it on in the toggle's on windows only (an A/B inside one run).
bool MultiDrawReplayConfigured();
// Configured and, with the toggle, in an on window (constant within a frame).
bool MultiDrawReplayActive();

// KYTY_INDIRECT_REPLAY=1 (experiment): consecutive non-indexed fast-launch mesh DRAW_INDIRECT
// packets can reuse the last prepared draw when the command processor saw only validated user
// data writes or NOPs and the mesh-argument conversion belongs to the same run.
bool IndirectReplayConfigured();
bool IndirectReplayActive();
enum class IndirectReplayNote : uint32_t {
	Replayed,
	NoBase,
	PacketBoundary,
	PacketState,
	ArgsRun,
	ArgsBuffer,
	BufferCacheChanged,
	StartInstance,
	CommandChanged,
	CommitChanged,
	PsUserData,
	VsProgram,
	PlanReads,
	Diagnostics,
	BaseWrites,
	BaseNotFastLaunch,
	BaseArgsRun,
	BaseUserData,
	BasePlanUnknown,
	BaseCaptured,
	ContextChanged,
	Count
};
void NoteIndirectReplay(IndirectReplayNote note);
// GPU thread: a packet that ended an indirect replay base (opcode << 16 | first register).
void NoteIndirectReplayStop(uint32_t key);
// Set when a replay base with GPU arguments is captured, cleared when it is invalidated: the command
// processor skips the renderer lock for packets while no such base can exist (most of the time).
inline std::atomic<bool> g_indirect_base_maybe {false};
enum class ReplayNote : uint32_t {
	Replayed,
	ReplayedSameData, // replayed without a new shader-data upload
	Empty,
	NoBase,
	CommandChanged, // another command buffer, or the rendering ended
	CommitChanged,
	PsUserData,
	VsProgram,
	PlanReads, // the resources depend on user data that changed
	Limits,
	UploadMoved,
	BaseNotMesh,
	BaseWrites,
	BaseDiagnostics,
	BaseUserData, // the snapshot's user data did not match the registers
	BasePlanUnknown,
	BaseCaptured,
	Count
};
// Counted, and printed as [multi-replay] every 5 s with KYTY_RENDER_STATS=1.
void NoteReplay(ReplayNote note);

// Multi-draw packets by class (item 2, 11-20), printed as [multi-packets] every 5 s with
// KYTY_RENDER_STATS=1. Counted only then. With [multi-replay] it says how many sub-draws a
// coverage change could reach:
// - replay for indexed packets: at most indexed-rest (sub-draws after the first), split by
//   pipeline kind in indexed-mesh / indexed-vertex (every indexed sub-draw that was drawn) and
//   indexed-writes (those that write memory: they would need the barrier change as well);
// - replay for non-mesh packets and the write barrier at the packet's end: at most
//   [multi-replay] no-base (sub-draws after the first with no base), of which base-not-mesh and
//   base-writes say which kind;
// - first-draws: the first sub-draws of replayed packets, always drawn in full. Of those:
//   first-same-state would have been replayed from the previous packet's base if a base lived
//   across packets (same command buffer, rendering and commit, same registers, and what
//   ReplayMultiDraw checks; an upper bound, nothing is replayed), first-other-commit had other
//   work or a submit in between, first-state-changed had other registers, pixel user data, vertex
//   program or plan-read user data, first-no-base followed a packet that ended without a base.
enum class MultiPacketNote : uint32_t {
	Packets,
	SubDraws,
	Single, // packets with one sub-draw
	IndexedPackets,
	IndexedRest,
	IndexedMesh,
	IndexedVertex,
	IndexedWrites,
	ReplayPackets,
	FirstDraws,
	FirstNoBase,
	FirstOtherCommit,
	FirstStateChanged,
	FirstSameState,
	Count
};
void NoteMultiPacket(MultiPacketNote note, uint64_t count = 1);

} // namespace RenderStats
} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERSTATS_H_ */
