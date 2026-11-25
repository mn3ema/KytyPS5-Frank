#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/timeline.h"

#include <array>
#include <atomic>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h> // __readgsdword
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

inline void TrackingSpinPause() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
	_mm_pause();
#endif
}

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			// Wait with plain loads and a pause until the lock looks free, then retry the
			// exchange: write faults take this lock while another thread holds it across a
			// protection change, and spinning on test_and_set kept the cache line bouncing
			// between cores (ReverendOzmund, Senaxx/KytyPS5 PR #17, 33e02045).
			do {
				TrackingSpinPause();
				std::atomic_signal_fence(std::memory_order_seq_cst);
			} while (m_lock.test(std::memory_order_relaxed));
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#if defined(_M_X64)
		// v438zk: the id GetCurrentThreadId returns, read where it reads it (the thread block's
		// ClientId.UniqueThread, gs:[0x48]) without the call into kernel32: every lock and unlock
		// of every tracker walk asks (after Senaxx/KytyPS5, regionManager.h; GPL-2.0-only).
		return __readgsdword(0x48);
#else
		return GetCurrentThreadId();
#endif
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Bumped whenever pages become CPU-dirty anywhere (a write fault, an untrack, an
// invalidation). A consumer that must see every CPU write, such as the pre-dispatch sync of
// all buffers for global-memory shaders, can skip its walk while this has not moved.
inline std::atomic<uint64_t> g_cpu_dirty_epoch {1};

// Bumped whenever pages stop being GPU-dirty anywhere (a download, a release of pages the GPU
// write bitmap shows unwritten). With g_cpu_dirty_epoch, an unchanged pair means no page changed
// state in a way a repeated identical sync would act on (BufferCache::SynchronizeBuffer).
inline std::atomic<uint64_t> g_gpu_clean_epoch {1};

// Bumped by the GPU thread when it starts a guest submission (its first slice).
inline std::atomic<uint64_t> g_guest_submission_seq {1};

// Bumped by a command processor when a WAIT_REG_MEM passes (what the CPU wrote before the label
// it waited on must be visible to the commands after it) and when a WRITE_DATA writes guest memory
// on the CPU (the commands after it read those bytes): G6's RenderContext::PrepareBda walks again.
inline std::atomic<uint64_t> g_bda_resync_seq {0};

// KYTY_G6=compute (G6 stage 1): compute shaders read their read-only buffers through the GPU
// address table (ResourceTracking.cpp); the host side of it.
[[nodiscard]] inline bool G6Compute() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_G6");
		return value != nullptr && std::string_view(value).starts_with("compute");
	}();
	return enabled;
}

// Hot pages (KYTY_HOT_PAGES=1). A page the CPU writes again and again (a ring of per-frame
// constants, say) used to cost a write fault on the game thread and two protection changes per
// write: in the jungle ~120,000 faults a second over the same ~40,000 pages. A page that faults
// KYTY_HOT_PAGE_FAULTS times (default 6) within 2 s turns hot: it stays CPU-dirty and writable,
// so the CPU writes it without faults, and a sync uploads it like any CPU-dirty page, at most once
// per hot generation. The GPU thread bumps the generation whenever the game may have written
// since without the command stream noticing: when a submission starts (the game submitted after
// writing) and when the command stream reads a label the CPU wrote (a wait or a conditional on a
// CPU-dirty page), and at predication. Within a generation the commands read the bytes as they were at its start, as a
// game that writes only before submitting or before setting a label the GPU waits on expects.
// A page the GPU is about to write stops being hot (its CPU copy was just uploaded); every page
// stops being hot every 60 s and turns hot again only if it still faults.
inline bool HotPagesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOT_PAGES");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
inline uint32_t HotPageFaults() {
	static const uint32_t faults = [] {
		const char* value = std::getenv("KYTY_HOT_PAGE_FAULTS");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 6ul;
		return static_cast<uint32_t>(n < 2 ? 2 : n > 200 ? 200 : n);
	}();
	return faults;
}
inline std::atomic<uint32_t> g_hot_generation {1};
inline std::atomic<uint32_t> g_hot_epoch {1};
struct HotPageStats {
	std::atomic<uint64_t> promoted {0}, uploads {0}, repeats {0}, demoted_by_gpu {0},
	    faults {0};
	// Label reads that started a generation, and the polls that did not because the label still
	// held the value the previous read saw (KYTY_HOT_LABEL_DEDUPE).
	std::atomic<uint64_t> label_bumps {0}, label_skips {0};
};
inline HotPageStats g_hot_stats;
// GPU thread: the game may have written guest memory since the last bump.
inline void BumpHotGeneration() {
	if (HotPagesEnabled()) {
		g_hot_generation.fetch_add(1, std::memory_order_relaxed);
	}
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }

	// Lock-free "any page in this region may be dirty" summaries. A binding that spans a
	// whole heap covers hundreds of regions; without these every draw locks and walks each
	// one. False is exact (set only after a walk saw no bits); true is conservative.
	template <DirtySource source>
	[[nodiscard]] bool MaybeModified() const noexcept {
		return GetSummary<source>().load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	// Under the lock, before ChangeState<Cpu, true> for a CPU write (a fault or an invalidation):
	// counts the faults of the pages it makes CPU-dirty and turns busy ones hot.
	void NoteCpuWrite(uint64_t vaddr, uint64_t size) {
		if (!HotPagesEnabled()) {
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		if (!m_hot) {
			m_hot = std::make_unique<HotState>();
		}
		auto&      hot = *m_hot;
		SyncHotEpoch();
		const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		if (now - hot.window_start > 2000) {
			hot.faults.fill(0);
			hot.window_start = now;
		}
		const auto threshold = HotPageFaults();
		for (size_t page = start; page < end; page++) {
			if (m_cpu_dirty.Get(page) || m_gpu_dirty.Get(page)) {
				continue; // no fault: already writable, or the GPU owns it
			}
			g_hot_stats.faults.fetch_add(1, std::memory_order_relaxed);
			if (hot.faults[page] < 255) {
				hot.faults[page]++;
			}
			if (hot.faults[page] >= threshold && !hot.hot.Get(page)) {
				hot.hot.Set(page);
				hot.uploaded[page] = 0;
				g_hot_stats.promoted.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}

	// Under the lock: the hot pages of the range stop being hot and lose their CPU-dirty state
	// (the GPU writes them next; their CPU bytes were uploaded in this generation). Returns
	// whether there were any.
	bool DemoteHot(uint64_t vaddr, uint64_t size) {
		if (!m_hot) {
			return false;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits in(m_hot->hot, start, end);
		if (in.None()) {
			return false;
		}
		for (const auto [first, last]: in) {
			for (size_t page = first; page < last; page++) {
				m_hot->hot.Unset(page);
				m_cpu_dirty.Unset(page);
				g_hot_stats.demoted_by_gpu.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (m_cpu_dirty.None()) {
			StoreSummary<DirtySource::Cpu>(false);
		}
		UpdateProtection<true, false>();
		return true;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		ChangeStateBits<source, enable>(vaddr, size);
		ApplyStateProtection<source, enable>();
	}

	// v445j: ChangeState in two halves, so a caller that changes several spans of one region
	// under its lock can set all the bits first and bring the page protection up to date once
	// (UpdateProtection protects by the difference between the bits and the last protection, so
	// one pass after N bit changes gives the same pages the same protection as N passes).
	template <DirtySource source, bool enable>
	void ApplyStateProtection() {
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool enable>
	void ChangeStateBits(uint64_t vaddr, uint64_t size) {
		if constexpr (source == DirtySource::Gpu && enable) {
			DemoteHot(vaddr, size);
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
			StoreSummary<source>(true);
			if constexpr (source == DirtySource::Cpu) {
				if (m_cpu_touched_word != nullptr) {
					m_cpu_touched_word->fetch_or(m_cpu_summary_bit, std::memory_order_acq_rel);
				}
				g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
				m_version.fetch_add(1, std::memory_order_release);
			}
		} else {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				StoreSummary<source>(false);
			}
			if constexpr (source == DirtySource::Gpu) {
				g_gpu_clean_epoch.fetch_add(1, std::memory_order_release);
				m_version.fetch_add(1, std::memory_order_release);
			}
		}
	}

	// KYTY_READ_SNAPSHOT: let CPU reads of these GPU-dirty pages through without a fault (they
	// read the snapshot published there), or fault again (restore). The GPU-dirty state is kept:
	// writes still fault and wait, and the GPU thread's reads still download.
	template <bool lift>
	void SetReadLift(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (lift) {
			m_read_lift.SetRange(start, end);
			UpdateProtection<false, true>();
		} else {
			m_read_lift.UnsetRange(start, end);
			UpdateProtection<true, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (source == DirtySource::Cpu && clear) {
			if (m_hot) {
				SyncHotEpoch();
				RegionBits hot_in(m_hot->hot, start, end);
				if (hot_in.Any()) {
					// Hot pages stay CPU-dirty (and writable); each is uploaded once a generation.
					const auto generation = g_hot_generation.load(std::memory_order_relaxed);
					for (const auto [first, last]: hot_in) {
						for (size_t page = first; page < last; page++) {
							if (m_hot->uploaded[page] == generation) {
								mask.Unset(page);
								g_hot_stats.repeats.fetch_add(1, std::memory_order_relaxed);
							} else {
								m_hot->uploaded[page] = generation;
								g_hot_stats.uploads.fetch_add(1, std::memory_order_relaxed);
							}
						}
					}
					bits.UnsetRange(start, end);
					bits |= hot_in;
					UpdateProtection<true, false>();
					for (const auto [first, last]: mask) {
						func(m_cpu_addr + first * TRACKER_PAGE_SIZE,
						     (last - first) * TRACKER_PAGE_SIZE);
					}
					return;
				}
			}
		}
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				StoreSummary<source>(false);
			}
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	[[nodiscard]] uint64_t Version() const noexcept {
		return m_version.load(std::memory_order_acquire);
	}
	// A region with hot pages re-uploads them by generation, which no version tracks.
	[[nodiscard]] bool HasHotPages() const noexcept { return m_hot != nullptr; }

	// The tracker mirrors the CPU summary in one bit per region (MemoryTracker::IsRangeCpuCleanHint).
	// Set before the region is published.
	void SetCpuSummaryBit(std::atomic<uint64_t>* word, uint64_t bit) noexcept {
		m_cpu_summary_word = word;
		m_cpu_summary_bit  = bit;
	}
	// The tracker's touched bitmap (same bit): set whenever pages become CPU-dirty, cleared by the
	// BDA walk (MemoryTracker::ForEachCpuTouchedRegion).
	void SetCpuTouchedWord(std::atomic<uint64_t>* word) noexcept { m_cpu_touched_word = word; }

	TrackingSpinLock lock;

private:
	struct HotState {
		RegionBits                                   hot;
		std::array<uint8_t, TRACKER_REGION_PAGES>    faults {};
		std::array<uint32_t, TRACKER_REGION_PAGES>   uploaded {};
		int64_t                                      window_start = 0;
		uint32_t                                     epoch        = 0;
	};

	// Every page stops being hot when the global epoch moves (every 60 s); a page that still
	// faults turns hot again. The pages stay CPU-dirty until their next upload re-arms them.
	void SyncHotEpoch() {
		const auto epoch = g_hot_epoch.load(std::memory_order_relaxed);
		if (m_hot->epoch != epoch) {
			m_hot->epoch = epoch;
			m_hot->hot.Clear();
		}
	}

	// Every summary change goes through here; the CPU one is mirrored in the tracker's bitmap at
	// the same point, before any page protection changes.
	template <DirtySource source>
	void StoreSummary(bool value) {
		GetSummary<source>().store(value, std::memory_order_release);
		if constexpr (source == DirtySource::Cpu) {
			if (m_cpu_summary_word != nullptr) {
				if (value) {
					m_cpu_summary_word->fetch_or(m_cpu_summary_bit, std::memory_order_acq_rel);
				} else {
					m_cpu_summary_word->fetch_and(~m_cpu_summary_bit, std::memory_order_acq_rel);
				}
			}
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		// m_read_lift: GPU-dirty pages the CPU may read as guest memory holds them
		// (BufferCache snapshot reads, KYTY_READ_SNAPSHOT).
		const auto protection = is_read ? (~m_gpu_dirty | m_read_lift) : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		// v445l: KYTY_CPU_COST phase of the page manager's part (host protection calls).
		const CpuCost::Phase protect_phase_("RM::PageWatchers");
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	std::atomic<bool>& GetSummary() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	template <DirtySource source>
	const std::atomic<bool>& GetSummary() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	// The constructor fills m_cpu_dirty, so the CPU summary starts true.
	std::atomic<bool> m_cpu_maybe_dirty {true};
	// Bumped with g_cpu_dirty_epoch / g_gpu_clean_epoch for this region's pages only
	// (BufferCache::SynchronizeBuffer's per-region memo, KYTY_SYNC_SKIP_REGION).
	std::atomic<uint64_t> m_version {1};
	std::atomic<bool> m_gpu_maybe_dirty {false};
	std::atomic<uint64_t>* m_cpu_summary_word = nullptr;
	std::atomic<uint64_t>* m_cpu_touched_word = nullptr;
	uint64_t               m_cpu_summary_bit  = 0;
	RegionBits   m_writable;
	RegionBits   m_readable;
	RegionBits   m_read_lift;
	std::unique_ptr<HotState> m_hot;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
