#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTSNAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTSNAP_H_

#include "graphics/host_gpu/renderer/image/image.h"

#include <cstdint>
#include <span>

// KYTY_RT_SNAP=<hex guest address>:<x>+<y>+<w>+<h> (any separator) (diagnostics; adds copies, changes nothing):
// a region of the color image at that address is copied to a readback ring after every draw or
// dispatch that binds it as a storage image, and after the frame's first one that binds it as a
// texture (its state after the render-target passes). On completion [rt-snap] prints, per guest
// flip, the stage number, the shader and the xxh3 of the region: the first stage whose hash
// differs between a good and a bad frame is the pass that loses or corrupts the region.
namespace Libs::Graphics {
class RenderContext;
struct PreparedBindings;
namespace RtSnap {

[[nodiscard]] bool Enabled();
// GPU thread, right after the draw or dispatch was recorded.
void AfterCommand(RenderContext& context, const char* kind,
                  std::span<PreparedBindings* const> stages);
// GPU thread, once per guest flip: prints the snapshots whose ticks completed.
void AtFlip(RenderContext& context);

// KYTY_PIXEL_PROBE=<target w>+<target h>:<x>+<y>+<w>+<h> with KYTY_PIXEL_PROBE_TRIGGER=<file>
// (diagnostics, 11-13 hex holes; adds copies for one guest frame): an in-emulator pixel history.
// Once the file exists, every draw of the next whole guest frame whose colour or depth attachment
// is exactly <target w>x<target h> gets the region (x,y,w,h) of each such attachment copied back
// right after it. [pixel-probe] lines list, in draw order, every draw that changed a region (its
// shaders, the attachment, the centre texel's bytes, depth as a float), and the first draw per
// attachment.
[[nodiscard]] bool ProbeEnabled();
void ProbeAfterDraw(RenderContext& context, std::span<const ImageId> colors, ImageId depth,
                    uint64_t vs, uint64_t ps, const char* kind);

// KYTY_BUFFER_PROBE=<compute hash>:<range>[,<range>...][/<hash>:...] with KYTY_BUFFER_PROBE_TRIGGER=<file> and
// KYTY_BUFFER_PROBE_DIR=<folder> (diagnostics, 11-13 hex holes; adds copies for one guest frame).
// <range> is <hex guest address>+<hex bytes>, or "all" for every buffer the dispatch binds (each
// up to KYTY_BUFFER_PROBE_EACH_MB, default 64; none above KYTY_BUFFER_PROBE_SKIP_MB). Once the file exists, the next dispatches of that
// shader (KYTY_BUFFER_PROBE_DISPATCHES, default 4) get the
// GPU copies of those ranges read back right after it; they are written to
// <folder>/bp_<dispatch>_<hash>_<address>.bin, with [buffer-probe] lines (user data, bindings).
void BufferProbeAfterDispatch(RenderContext& context, uint64_t hash, const PreparedBindings& bindings,
                              const uint32_t* user_data, uint32_t user_count);

} // namespace RtSnap
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTSNAP_H_
