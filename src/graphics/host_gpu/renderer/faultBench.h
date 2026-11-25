#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FAULTBENCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FAULTBENCH_H_

#include <atomic>
#include <cstdint>

// KYTY_FAULT_BENCH=1 (v445r, measurement only): what one page fault of the tracker costs, so
// [fault-census]'s counts can be read as time. At the first presented frame (and, with
// KYTY_FAULT_BENCH_EVERY_S=<seconds>, again every that many seconds, in whatever state the game
// is in then) a thread of its own takes KYTY_FAULT_BENCH_N (default 2000) write faults on one
// scratch page, each through the emulator's real path:
//
//   arm      the buffer cache's tracker marks the page CPU-clean, as a sync that found nothing to
//            upload does: the page manager write-protects it (the host protection call);
//   touch    a write to the page: the host's exception dispatch, the emulator's handler (fault
//            census, stale check, RenderContext::HandleFault: the buffer cache marks the page
//            CPU-dirty and the page manager lifts the protection, the texture cache and the
//            bindless table are told), and the return to the faulting instruction.
//
// The page is 64 KB of flexible memory mapped once, as the game maps its own, at the top of the
// range hint-less mappings come from (hint 0x7f0000000: far above the game's, below the PRT
// apertures), and kept for the session; every run checks that the mapping is still the bench's
// and writes back the byte it read. [fault-bench] prints microseconds per fault with the split
// dispatch / handler / resume, the arming and one bare protection change. The faulting thread is
// a host thread: a guest thread's fault takes the same path from the handler on.
// KYTY_HOT_PAGES=1 makes the page hot after a few faults (it then stays writable): the line
// counts those writes as not faulted.
//
// Off: one load of Detail::g_enabled in the fault handler and at every flip.
namespace Libs::Graphics {

class RenderContext;

namespace FaultBench {

namespace Detail {
extern bool                  g_enabled; // set once, before main
extern std::atomic<uint64_t> g_page;    // the scratch page while a run is in progress; 0: none
inline constexpr uint64_t    PageBytes = 0x1000;
[[nodiscard]] uint64_t HandlerEnter() noexcept;
void                   HandlerLeave() noexcept;
void                   OnGuestFlipSlow(RenderContext& renderer);
} // namespace Detail

// In the host exception handler, around the handling of an access violation at `vaddr`.
class HandlerScope {
public:
	explicit HandlerScope(uint64_t vaddr) noexcept {
		if (Detail::g_enabled) [[unlikely]] {
			const auto page = Detail::g_page.load(std::memory_order_relaxed);
			if (page != 0 && vaddr >= page && vaddr - page < Detail::PageBytes) {
				m_entered = Detail::HandlerEnter() != 0;
			}
		}
	}
	~HandlerScope() {
		if (m_entered) [[unlikely]] {
			Detail::HandlerLeave();
		}
	}
	HandlerScope(const HandlerScope&)            = delete;
	HandlerScope& operator=(const HandlerScope&) = delete;

private:
	bool m_entered = false;
};

// Every presented frame.
inline void OnGuestFlip(RenderContext& renderer) {
	if (Detail::g_enabled) [[unlikely]] {
		Detail::OnGuestFlipSlow(renderer);
	}
}

} // namespace FaultBench
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FAULTBENCH_H_
