#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_

#include <cstdint>

// Occlusion queries (ZPASS_DONE, EVENT_WRITE 0x39) from host occlusion queries.
//
// On the PS5 every DB keeps a running count of the samples that passed the depth/stencil tests
// while DB_COUNT_CONTROL.ZPASS_INCREMENT_DISABLE is clear; a ZPASS_DONE event writes each DB's
// count, with bit 63 set, to <address> + 16 * db. A query is a ZPASS_DONE before its draws and one
// after (8 bytes further on); the game sums end - begin over the DBs (CPU, or SET_PREDICATION).
//
// Here every draw recorded while the counter runs (the game sets ZPASS_ENABLE around its queries)
// is wrapped in a host occlusion query. Both dumps are written at once, ready: the begin as zero,
// the end as the samples the same query had the last time the GPU finished it, so a reader in the
// same frame sees a result. Once the GPU has run the draws, the end slot gets this time's samples
// (DB 0; DBs 1..15 read zero).
//
// KYTY_OCCLUSION_REAL=0 keeps the old result (always visible, a synthetic counter).
// With KYTY_FPS_LOG, [occl-real] prints dumps, counted draws and DB_COUNT_CONTROL values.
namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;

namespace Occlusion {

bool Enabled();
// Whether a draw recorded now would be counted (no side effects).
bool CountsDraw(const CommandBuffer& buffer);
// Before the draw's rendering begins (may end the current rendering to reset queries).
void Prepare(RenderContext& context, CommandBuffer& buffer);
// Right around the draw command, inside the rendering.
void Begin(RenderContext& context, CommandBuffer& buffer);
void End(RenderContext& context, CommandBuffer& buffer);
// A guest draw packet (KYTY_OCCLUSION_LOG counts them per query).
void NoteGuestDraw();
// ZPASS_DONE to <address> (GPU thread, a command buffer active).
void Dump(RenderContext& context, CommandBuffer& buffer, uint64_t address);

} // namespace Occlusion
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
