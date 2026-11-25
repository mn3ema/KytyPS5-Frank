#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_QUEUESHARE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_QUEUESHARE_H_

#include <cstdint>
#include <cstdlib>

// KYTY_QUEUE_STATS, [queue-share] (v445m, counters only; docs/async_compute_thread_design_20261120.md,
// section 6, check 2). The GPU thread translates every guest queue in one order of its own. This
// counts what a recording stream per queue (graphics queue 0 against the compute queues) would
// have to order: every buffer obtain and image use is noted with its queue class, a per-class
// sequence number and read or write. An access conflicts with the other class's earlier access
// to an overlapping range in the same frame when one of the two writes (read-after-write,
// write-after-read, write-after-write). A conflict whose earlier access has not been cut off
// yet forces a join: the other stream's command buffer is cut there (everything it recorded so
// far is submitted first), so later conflicts with its earlier accesses are free. Joins are
// counted twice: by themselves, and beyond the joins the guest already makes (a wait that passes
// on a label the other class wrote). Frames are sampled (KYTY_QUEUE_SHARE_EVERY, default 4:
// every fourth guest frame is tracked in full; 0 turns the counter off).
// Not seen: bindless textures (read through the heap, no per-draw use), vertex and index
// buffers a draw binds without an obtain, and the readback path.
namespace Libs::Graphics::QueueShare {

inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* every = std::getenv("KYTY_QUEUE_SHARE_EVERY");
		return std::getenv("KYTY_QUEUE_STATS") != nullptr && !(every != nullptr && every[0] == '0' && every[1] == '\0');
	}();
	return enabled;
}

namespace Detail {
void NoteSlow(uint64_t begin, uint64_t end, bool write, bool image);
void GuestLabelSlow();
void GuestWaitSlow(bool cross_queue, bool passed);
void FrameEndSlow();
} // namespace Detail

// A buffer range of guest memory the current draw or dispatch reads or writes on the GPU.
inline void NoteBuffer(uint64_t vaddr, uint64_t size, bool write) {
	if (Enabled() && size != 0) {
		Detail::NoteSlow(vaddr, vaddr + size, write, false);
	}
}
// An image (by its slot index) sampled (read) or bound as a target or written storage (write).
inline void NoteImage(uint32_t index, bool write) {
	if (Enabled()) {
		const uint64_t begin = (uint64_t {1} << 62u) | (uint64_t {index} << 12u);
		Detail::NoteSlow(begin, begin + 4096u, write, true);
	}
}
// The current queue wrote a label; evaluated a wait (on a label the other class wrote: cross_queue).
inline void NoteGuestLabel() {
	if (Enabled()) {
		Detail::GuestLabelSlow();
	}
}
inline void NoteGuestWait(bool cross_queue, bool passed) {
	if (Enabled()) {
		Detail::GuestWaitSlow(cross_queue, passed);
	}
}
// Once per guest flip, on the GPU thread.
inline void FrameEnd() {
	if (Enabled()) {
		Detail::FrameEndSlow();
	}
}

} // namespace Libs::Graphics::QueueShare

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_QUEUESHARE_H_
