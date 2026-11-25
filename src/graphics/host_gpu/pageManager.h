#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <cstddef>
#include <memory>

namespace Libs::Graphics {

// KYTY_PROTECT_STATS=1 (or KYTY_FPS_LOG's diagnostics), [fault-census] (v445m, counters only):
// every access violation on guest memory that reaches the tracker (PageManager::ResolveStale, the
// first step of the fault handler) is counted by page, guest thread, read or write, and what the
// tracker held for the page at that moment (GPU-dirty: read-watched; CPU-clean: write-watched).
// The handler writes a fixed table of atomics only. The report (every 5 s, on the GPU thread)
// prints the totals and the ten heaviest entries, each with the buffer that owns the page now.
namespace FaultCensus {
[[nodiscard]] bool Enabled() noexcept;
// Writes what owns the guest page at `address` into `out` (the buffer cache's answer).
using Describe = void (*)(void* context, uint64_t address, char* out, size_t size);
void Report(void* context, Describe describe);
} // namespace FaultCensus

enum class PageFaultAccess { Read, Write, Execute, Unknown };

enum class ProtectDrainReason : uint8_t {
	Submit,
	ReadbackSubmit,
	Fault,
	GuestWrite,
	GuestSignal,
	Mapping,
	ToggleOff,
	SynchronousProtection,
	Count
};

// Waits until the helper has applied every queued protection request. Calls are cheap when the
// helper has not been started or there is nothing queued.
void DrainProtectionHelper(ProtectDrainReason reason) noexcept;

// KYTY_FPS_LOG diagnostics: the caller a protection change is attributed to ([page-protect]).
inline thread_local const char* t_protect_reason = nullptr;
struct ProtectReasonScope {
	const char* previous;
	explicit ProtectReasonScope(const char* reason) noexcept: previous(t_protect_reason) {
		t_protect_reason = reason;
	}
	~ProtectReasonScope() { t_protect_reason = previous; }
	ProtectReasonScope(const ProtectReasonScope&)            = delete;
	ProtectReasonScope& operator=(const ProtectReasonScope&) = delete;
};

// KYTY_LAZY_READ_RELEASE=1: a page whose last read watcher goes away on this thread while the flag
// is set (the GPU write bitmap releasing pages a written binding marked but the GPU did not write)
// keeps its no-access protection, "stale", instead of going back to read-only. The same pages are
// marked again by the next frame's binding, which then needs no protection change either: in the
// sewer that pair was ~80 ms of the GPU thread a second. A CPU access to a stale page faults once
// and gets the protection the counts ask for (ResolveStale, first thing in the fault handler).
bool LazyReadReleaseEnabled() noexcept;
inline thread_local bool t_lazy_read_release = false;
struct LazyReadReleaseScope {
	bool previous;
	LazyReadReleaseScope() noexcept: previous(t_lazy_read_release) { t_lazy_read_release = true; }
	~LazyReadReleaseScope() { t_lazy_read_release = previous; }
	LazyReadReleaseScope(const LazyReadReleaseScope&)            = delete;
	LazyReadReleaseScope& operator=(const LazyReadReleaseScope&) = delete;
};

// [page-protect] stats: the line of the tracker call behind a protection change (0: none).
inline thread_local uint32_t t_protect_line = 0;
struct ProtectLineScope {
	uint32_t previous;
	explicit ProtectLineScope(uint32_t line) noexcept: previous(t_protect_line) { t_protect_line = line; }
	~ProtectLineScope() { t_protect_line = previous; }
	ProtectLineScope(const ProtectLineScope&)            = delete;
	ProtectLineScope& operator=(const ProtectLineScope&) = delete;
};

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// Whether the page holding vaddr is read-protected for tracking. Unlocked: a hint that can
	// be stale by the time the caller acts on it.
	[[nodiscard]] bool IsReadWatched(uint64_t vaddr) const noexcept;

	// The host protection of [vaddr, vaddr + size) was replaced behind the tracker (a view mapped
	// again with its own access by a partial unmap, a guest mprotect that allows writes): the
	// watched pages get the protection their counts ask for again, so CPU writes to them fault
	// (and reach the GPU) once more. Unwatched pages keep what they were given. Any thread,
	// outside the guest address-space lock.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size);

	// Fault handler, before anything else: resolves LazyReadRelease pages and retries spurious
	// faults whose current tracker permission already allows the access.
	[[nodiscard]] bool ResolveStale(uint64_t vaddr, PageFaultAccess access) noexcept;
	// The range is about to be unmapped: its stale pages get their real protection first.
	void ForgetStale(uint64_t vaddr, uint64_t size) noexcept;
	// A guest mapping or permission change makes any delayed-release fault marker obsolete.
	void ForgetProtectionFaults(uint64_t vaddr, uint64_t size) noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
