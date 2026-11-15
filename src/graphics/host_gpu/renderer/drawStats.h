#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWSTATS_H_

#include <cstdint>

// Diagnostics: KYTY_DRAW_STATS=<first guest flip>[:<flips>[:<every>]] wraps every draw of those
// guest flips (flips first, first+every, ... ; flips of them each) in a pipeline statistics query
// and prints, once the GPU ran it, what the draw really did on the host GPU:
// [draw-stats] flip=<f> n=<draw in flip> vs=<hash> ps=<hash> <kind> count=<n> inst=<n>
//   gpu_args=<0|1> clip_in=<primitives into clipping> clip_out=<primitives out> frag=<fragment
//   shader invocations> rt0=<address>:<w>x<h> blend=<src>/<dst>
// A draw that should show on screen but rasterizes nothing (no primitives, all culled, every
// fragment rejected early) or a GPU-driven draw whose arguments came out empty shows up here.
namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;

namespace DrawStats {

struct Info {
	uint64_t    vs        = 0;
	uint64_t    ps        = 0;
	const char* kind      = "";
	uint32_t    count     = 0;
	uint32_t    instances = 0;
	bool        gpu_args  = false;
	uint64_t    rt0       = 0;
	uint32_t    rt0_w     = 0;
	uint32_t    rt0_h     = 0;
	uint32_t    blend_src = 0;
	uint32_t    blend_dst = 0;
	bool        blend     = false;
};

bool Enabled();
// Before the draw's rendering begins (may end the current rendering to reset queries).
void Prepare(RenderContext& context, CommandBuffer& buffer);
// Right around the draw command, inside the rendering.
void Begin(RenderContext& context, CommandBuffer& buffer);
void End(RenderContext& context, CommandBuffer& buffer, const Info& info);

} // namespace DrawStats
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWSTATS_H_
