#include "graphics/host_gpu/pageManager.h"
#include "common/fpsLog.h"

#include "common/alignment.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/timeline.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

// [watcher-loop] (KYTY_PROTECT_STATS, v445m): the tracker's own loop over a region's pages
// (UpdateRegionWatchers), apart from the host protection calls it makes. [0]: the GPU thread,
// [1]: every other thread (fault handlers). Times in time-stamp counter ticks (CpuCost::Detail).
struct WatcherLoopStats {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> loop_ticks {0};    // inside UpdateRegionWatchers, without lock wait and Protect
	std::atomic<uint64_t> lock_ticks {0};    // waiting for the region's spin lock
	std::atomic<uint64_t> protect_ticks {0}; // Protect() called from the loop
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> visited {0}; // pages the loop went over
	std::atomic<uint64_t> updated {0}; // pages whose watcher count it changed
	std::atomic<uint64_t> changed {0}; // pages whose host protection had to change
};
std::array<WatcherLoopStats, 2> g_watcher_loop;
thread_local uint64_t           t_watcher_protect_ticks = 0;
thread_local uint64_t           t_watcher_protect_calls = 0;
thread_local bool               t_in_watcher_loop       = false;

// [fault-census] (pageManager.h). Written from the fault handler: atomics only.
struct FaultCensusState {
	static constexpr size_t Slots = 16384;
	static constexpr uint64_t Write = 1, GpuDirty = 2, CpuClean = 4, GpuThread = 8, None = 0;
	struct Slot {
		std::atomic<uint64_t> key {0};    // bit 63 | page << 4 | flags; 0: free
		std::atomic<uint32_t> thread {0}; // guest thread id + 1 (0: not written yet)
		std::atomic<uint32_t> count {0};
	};
	struct Name {
		std::atomic<uint32_t> thread {0}; // guest thread id + 1; 0: free
		std::atomic<bool>     ready {false};
		char                  name[32] {};
	};
	std::array<Slot, Slots> slots;
	std::array<Name, 128>   names;
	std::atomic<uint64_t>   total {0}, writes {0}, gpu_dirty {0}, cpu_clean {0}, untracked {0}, gpu_thread {0};
	std::atomic<uint64_t>   dropped {0}; // the table was full
	std::atomic<uint32_t>   used {0};
	std::atomic<int64_t>    reported_ms {0};
};
FaultCensusState* g_fault_census = nullptr; // set once by the first PageManager, before any fault is counted

bool FaultCensusEnabledSlow() noexcept {
	static const bool enabled = [] {
		const char* only = std::getenv("KYTY_PROTECT_STATS");
		return Common::FpsLogDiagnostics() || (only != nullptr && only[0] == '1');
	}();
	return enabled;
}

// In the fault handler: no allocation, no lock.
void FaultCensusNote(uint64_t vaddr, bool write, bool gpu_dirty, bool cpu_clean) noexcept {
	auto* census = g_fault_census;
	if (census == nullptr) {
		return;
	}
	using State = FaultCensusState;
	const bool     gpu_thread = Profiler::t_gpu_thread;
	const uint32_t thread     = static_cast<uint32_t>(Libs::LibKernel::PthreadGetthreadid()) + 1u;
	census->total.fetch_add(1, std::memory_order_relaxed);
	census->writes.fetch_add(write ? 1u : 0u, std::memory_order_relaxed);
	census->gpu_dirty.fetch_add(gpu_dirty ? 1u : 0u, std::memory_order_relaxed);
	census->cpu_clean.fetch_add(cpu_clean && !gpu_dirty ? 1u : 0u, std::memory_order_relaxed);
	census->untracked.fetch_add(!cpu_clean && !gpu_dirty ? 1u : 0u, std::memory_order_relaxed);
	census->gpu_thread.fetch_add(gpu_thread ? 1u : 0u, std::memory_order_relaxed);
	// The thread's name, once per thread.
	static thread_local bool named = false;
	if (!named) {
		named = true;
		for (auto& entry: census->names) {
			uint32_t expected = 0;
			if (entry.thread.load(std::memory_order_relaxed) == thread) {
				break;
			}
			if (entry.thread.compare_exchange_strong(expected, thread, std::memory_order_relaxed)) {
				char name[64] = "(host thread)";
				if (auto self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
					if (Libs::LibKernel::PthreadGetname(self, name) != 0) {
						name[0] = '\0';
					}
				}
				name[31] = '\0';
				std::memcpy(entry.name, name, sizeof(entry.name));
				entry.ready.store(true, std::memory_order_release);
				break;
			}
			if (expected == thread) {
				break;
			}
		}
	}
	const uint64_t key = (uint64_t {1} << 63u) | ((vaddr / PAGE_SIZE) << 4u) | (write ? State::Write : State::None) |
	                     (gpu_dirty ? State::GpuDirty : State::None) | (cpu_clean ? State::CpuClean : State::None) |
	                     (gpu_thread ? State::GpuThread : State::None);
	auto index = static_cast<size_t>(((key ^ (uint64_t {thread} << 40u)) * 0x9e3779b97f4a7c15ull) >> 40u) &
	             (State::Slots - 1u);
	for (size_t probe = 0; probe < 32; probe++, index = (index + 1u) & (State::Slots - 1u)) {
		auto& slot    = census->slots[index];
		auto  current = slot.key.load(std::memory_order_acquire);
		if (current == 0) {
			if (census->used.load(std::memory_order_relaxed) >= State::Slots / 2u) {
				break;
			}
			if (slot.key.compare_exchange_strong(current, key, std::memory_order_acq_rel)) {
				census->used.fetch_add(1, std::memory_order_relaxed);
				slot.thread.store(thread, std::memory_order_release);
				slot.count.fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
		// A slot another thread is just claiming (its thread not written yet) reads as another
		// thread's: this fault takes a slot of its own, and the report adds equal entries up.
		if (current == key && slot.thread.load(std::memory_order_acquire) == thread) {
			slot.count.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	census->dropped.fetch_add(1, std::memory_order_relaxed);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			std::atomic_signal_fence(std::memory_order_seq_cst);
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

std::atomic<uint64_t> g_stale_resolved {0}; // pages, for [page-protect]
// [page-protect]: pages a lazy release left no-access, and stale pages a new read watcher took
// over (the host call the lazy release saved).
std::atomic<uint64_t> g_stale_made {0};
std::atomic<uint64_t> g_stale_reused {0};
// [page-protect]: pages inside a changed range whose protection did not change, by the length of
// their run (1-3, 4-15, 16-63, 64 and more), and what KYTY_PROTECT_GAP left out.
std::atomic<uint64_t> g_gap_pages[4] {};
std::atomic<uint64_t> g_gap_skipped {0};
std::atomic<uint64_t> g_gap_splits {0};

// v445h: default back to 0 (off); two in-run A/Bs (g449b, g449d) showed no fps change either way.
// v445e: default 16 (KYTY_PROTECT_GAP=0: off). M3/M4 of 11-17: GPU-thread arm time 80 -> 48-58 ms/s,
// disarm time on all threads 149 -> 41 ms/s, and KYTY_PROTECT_VERIFY found no page whose protection
// differed (94 M left-out pages checked). KYTY_SYNC_TOGGLE_MODE=protectgap: on in the toggle's on
// windows only, for an A/B inside one run.
// KYTY_PROTECT_GAP=<pages> (v438zw; was default 0: off). One host call covers the pages from the first
// to the last page of a pass whose protection changes, with every unchanged page between them.
// With this, a call is not carried across <pages> or more unchanged pages: it ends at the last
// changed page and a new one starts at the next. The same pages get the same protection; the
// pages left out already have it. Off while the protection helper is configured (its marks are
// cleared per call). At most 4096.
uint64_t ProtectGapBytes() noexcept {
	static const uint64_t bytes = [] {
		const char*    value = std::getenv("KYTY_PROTECT_GAP");
		const uint64_t pages = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
		if (pages == 0 || SyncToggle::ProtectHelperConfigured() ||
		    SyncToggle::ProtectHelperAlwaysEnabled()) {
			return uint64_t {0};
		}
		return (pages > 4096 ? 4096 : pages) * PAGE_SIZE;
	}();
	if (SyncToggle::ProtectGapToggled() && !SyncToggle::ProtectGapWindow()) {
		return 0; // the toggle's off window
	}
	return bytes;
}

// KYTY_PROTECT_VERIFY=1 (v438zw, Windows; slow): checks against the host what the tracker takes
// for granted. (1) A page a lazy release leaves stale, and a stale page a new read watcher takes
// over without a host call, must be no-access on the host at that moment. (2) The pages
// KYTY_PROTECT_GAP leaves out get the call after all, and the host's protection of each must be
// the same before and after it. A difference ends the run with the page.
bool ProtectVerify() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PROTECT_VERIFY");
		const bool  on    = value != nullptr && value[0] == '1';
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		if (on) {
			std::fprintf(stderr, "[protect-verify] not available on this host: off\n");
			return false;
		}
#endif
		if (on && (SyncToggle::ProtectHelperConfigured() || SyncToggle::ProtectHelperAlwaysEnabled())) {
			std::fprintf(stderr, "[protect-verify] off: the protection helper applies changes later\n");
			return false;
		}
		return on;
	}();
	return enabled;
}
std::atomic<uint64_t> g_verify_stale_pages {0};
std::atomic<uint64_t> g_verify_gap_pages {0};

// The host's protection of every page of the range (0 where it is not committed). False where it
// cannot be read.
bool HostProtections([[maybe_unused]] uint64_t vaddr, [[maybe_unused]] uint64_t size,
                     [[maybe_unused]] std::vector<uint32_t>& pages) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	pages.clear();
	pages.reserve(size / PAGE_SIZE);
	const uint64_t end = vaddr + size;
	for (uint64_t current = vaddr; current < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) == 0) {
			return false;
		}
		const uint64_t next = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		if (next <= current) {
			return false;
		}
		const uint32_t protect = info.State == MEM_COMMIT ? static_cast<uint32_t>(info.Protect) : 0u;
		for (uint64_t page = current; page < next && page < end; page += PAGE_SIZE) {
			pages.push_back(protect);
		}
		current = next;
	}
	return true;
#else
	return false;
#endif
}

// KYTY_PROTECT_VERIFY, part 1: the tracker relies on these pages being no-access on the host.
void VerifyNoAccess(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::vector<uint32_t> pages;
	if (!HostProtections(vaddr, size, pages)) {
		return;
	}
	for (size_t i = 0; i < pages.size(); i++) {
		if (pages[i] != 0 && (pages[i] & 0xffu) != PAGE_NOACCESS) {
			Fatal("KYTY_PROTECT_VERIFY: page 0x%016" PRIx64 " is stale (no-access by the tracker) but "
			      "the host has protection 0x%" PRIx32 " (run 0x%016" PRIx64 "+0x%" PRIx64 ")",
			      vaddr + i * PAGE_SIZE, pages[i], vaddr, size);
		}
	}
	g_verify_stale_pages.fetch_add(pages.size(), std::memory_order_relaxed);
#else
	(void)vaddr;
	(void)size;
#endif
}
thread_local const char* t_protect_kind = nullptr; // [page-protect]: the watcher change

bool QueueProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept;

} // namespace

bool LazyReadReleaseEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LAZY_READ_RELEASE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

namespace {

bool LazyReadReleaseActive() noexcept {
	return LazyReadReleaseEnabled() || SyncToggle::ProtectLazyReleaseWindow();
}

bool LazyReadReleaseMayBeActive() noexcept {
	return LazyReadReleaseEnabled() || SyncToggle::ProtectLazyReleaseConfigured();
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		// Pages left no-access when their last read watcher went (LazyReadRelease): the host
		// protection is stricter than the counts ask for until a fault or a new watcher.
		RegionBits                          stale;
		// A queued release can fault an access that tracker state already permits. Keep that mark
		// until the fault is retried or a mapping/protection change replaces the host permission.
		RegionBits                          helper_release;
		std::atomic<uint32_t>               stale_count {0}; // read unlocked as a hint

		[[nodiscard]] Common::VirtualMemory::Mode HostPerms(size_t index) const noexcept {
			return stale_count.load(std::memory_order_relaxed) != 0 && stale.Get(index) ? Common::VirtualMemory::Mode::NoAccess
			                                            : pages[index].Perms();
		}
		void SetStale(size_t index) noexcept {
			if (!stale.Get(index)) {
				stale.Set(index);
				stale_count.fetch_add(1, std::memory_order_relaxed);
			}
		}
		void ClearStale(size_t index) noexcept {
			if (stale_count.load(std::memory_order_relaxed) != 0 && stale.Get(index)) {
				stale.Unset(index);
				stale_count.fetch_sub(1, std::memory_order_relaxed);
			}
		}
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	// KYTY_FPS_LOG: host protection changes per second, arming (to read-only or no access) and
	// disarming (back to read-write), and how many pages are armed again and again ([page-protect]).
	struct ProtectStats {
		std::mutex                             mutex;
		std::chrono::steady_clock::time_point  start = std::chrono::steady_clock::now();
		uint64_t                               arm_calls = 0, arm_pages = 0, disarm_calls = 0,
		                                       disarm_pages = 0, noaccess_pages = 0;
		std::unordered_map<uint64_t, std::array<uint64_t, 3>> regions; // 2 MB -> read, none, rw
		double                                 arm_us = 0, disarm_us = 0;
		double                                 arm_us_gpu = 0, disarm_us_gpu = 0; // on the GPU thread
		std::unordered_map<std::string, std::array<double, 5>> reasons; // arm calls, arm us, disarm calls, disarm us, arm pages
		std::unordered_map<uint64_t, uint32_t> armed; // page -> arm count in the window
		uint64_t                               start_ticks = CpuCost::Detail::Ticks(); // [watcher-loop]
	};
	static ProtectStats* Stats() {
		// KYTY_PROTECT_STATS=1: these stats alone. KYTY_FPS_LOG is named in the recompiler's sources,
		// so the shader disk cache keys on it: turning it on for a play session recompiles every
		// shader; this switch is not named there.
		static ProtectStats* stats = [] {
			const char* only = std::getenv("KYTY_PROTECT_STATS");
			return Common::FpsLogDiagnostics() || (only != nullptr && only[0] == '1')
			           ? new ProtectStats
			           : nullptr;
		}();
		return stats;
	}
	// Who changed the protection: the caller's reason, the watcher change (t_protect_kind) and
	// whether the GPU thread or another thread (a fault handler) made it.
	static std::string ReasonKey() {
		std::string key = t_protect_reason != nullptr ? t_protect_reason : "other";
		key += '/';
		key += t_protect_kind != nullptr ? t_protect_kind : "resync";
		if (t_protect_line != 0) {
			key += ':' + std::to_string(t_protect_line);
		}
		// KYTY_CPU_COST: the shader of the draw or dispatch the GPU thread is doing.
		if (Profiler::t_gpu_thread && CpuCost::t_kind != 0) {
			char text[32];
			std::snprintf(text, sizeof(text), " %c%014llx", CpuCost::t_kind,
			              static_cast<unsigned long long>(CpuCost::t_hash & ((uint64_t {1} << 56u) - 1u)));
			key += text;
		}
		if (!Profiler::t_gpu_thread) {
			key += "@cpu";
		}
		return key;
	}
	static void Note(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode, double us) {
		auto* stats = Stats();
		if (stats == nullptr) {
			return;
		}
		const bool       arm   = mode != Common::VirtualMemory::Mode::ReadWrite;
		const uint64_t   pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
		std::scoped_lock lock(stats->mutex);
		auto& region = stats->regions[vaddr >> 21u];
		region[mode == Common::VirtualMemory::Mode::Read ? 0 : arm ? 1 : 2] += pages;
		if (mode == Common::VirtualMemory::Mode::NoAccess) {
			stats->noaccess_pages += pages;
		}
		if (arm) {
			stats->arm_calls++;
			stats->arm_pages += pages;
			stats->arm_us += us;
			if (Profiler::t_gpu_thread) {
				stats->arm_us_gpu += us;
			}
			{
				auto& r = stats->reasons[ReasonKey()];
				r[0] += 1;
				r[4] += static_cast<double>(pages);
				r[1] += us;
			}
			for (uint64_t p = 0; p < pages && p < 4096; p++) {
				stats->armed[vaddr / PAGE_SIZE + p]++;
			}
		} else {
			stats->disarm_calls++;
			stats->disarm_pages += pages;
			stats->disarm_us += us;
			if (Profiler::t_gpu_thread) {
				stats->disarm_us_gpu += us;
			}
			{
				auto& r = stats->reasons[ReasonKey()];
				r[2] += 1;
				r[3] += us;
			}
		}
		const auto now = std::chrono::steady_clock::now();
		const auto s   = std::chrono::duration<double>(now - stats->start).count();
		if (s < 5.0) {
			return;
		}
		uint64_t hot = 0, hot_arms = 0, total_arms = 0;
		for (const auto& [page, count]: stats->armed) {
			total_arms += count;
			if (count >= 5) {
				hot++;
				hot_arms += count;
			}
		}
		std::fprintf(stderr,
		             "[page-protect] arm %.0f calls/s %.0f pages/s %.1f ms/s | disarm %.0f calls/s "
		             "%.0f pages/s %.1f ms/s | pages armed %zu, armed >=5x: %llu pages = %.0f%% of "
		             "arms\n",
		             stats->arm_calls / s, stats->arm_pages / s, stats->arm_us / 1000.0 / s,
		             stats->disarm_calls / s, stats->disarm_pages / s,
		             stats->disarm_us / 1000.0 / s, stats->armed.size(),
		             static_cast<unsigned long long>(hot),
		             total_arms != 0 ? 100.0 * static_cast<double>(hot_arms) / total_arms : 0.0);
		std::vector<std::pair<uint64_t, std::array<uint64_t, 3>>> top(stats->regions.begin(),
		                                                               stats->regions.end());
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
			return a.second[0] + a.second[1] + a.second[2] > b.second[0] + b.second[1] + b.second[2];
		});
		std::string text;
		for (size_t i = 0; i < top.size() && i < 10; i++) {
			char line[160];
			std::snprintf(line, sizeof(line), " 0x%llx:r%.0f/n%.0f/w%.0f",
			              static_cast<unsigned long long>(top[i].first << 21u),
			              top[i].second[0] / s, top[i].second[1] / s, top[i].second[2] / s);
			text += line;
		}
		// Where pages go back to read-write (CPU write faults mostly), by 2 MB region.
		std::sort(top.begin(), top.end(),
		          [](const auto& a, const auto& b) { return a.second[2] > b.second[2]; });
		std::string rw_text;
		uint64_t    rw_total = 0;
		for (const auto& [region, counts]: top) {
			rw_total += counts[2];
		}
		for (size_t i = 0; i < top.size() && i < 16 && top[i].second[2] != 0; i++) {
			char line[96];
			std::snprintf(line, sizeof(line), " 0x%llx:%.0f",
			              static_cast<unsigned long long>(top[i].first << 21u),
			              top[i].second[2] / s);
			rw_text += line;
		}
		std::fprintf(stderr, "[page-protect] read-write %.0f pages/s, by 2MB region:%s\n",
		             rw_total / s, rw_text.c_str());
		std::fprintf(stderr, "[page-protect] GPU thread: arm %.1f ms/s disarm %.1f ms/s; stale pages resolved by faults %.0f/s\n",
		             stats->arm_us_gpu / 1000.0 / s, stats->disarm_us_gpu / 1000.0 / s,
		             static_cast<double>(g_stale_resolved.exchange(0)) / s);
		std::fprintf(stderr,
		             "[page-protect] lazy release: %.0f pages/s left no-access, %.0f pages/s taken over by "
		             "a new GPU write without a host call\n",
		             static_cast<double>(g_stale_made.exchange(0)) / s,
		             static_cast<double>(g_stale_reused.exchange(0)) / s);
		std::fprintf(stderr,
		             "[page-protect] unchanged pages inside changed ranges, by run length: 1-3: %.0f/s, "
		             "4-15: %.0f/s, 16-63: %.0f/s, 64+: %.0f/s; left out by KYTY_PROTECT_GAP: %.0f/s in "
		             "%.0f splits/s\n",
		             static_cast<double>(g_gap_pages[0].exchange(0)) / s,
		             static_cast<double>(g_gap_pages[1].exchange(0)) / s,
		             static_cast<double>(g_gap_pages[2].exchange(0)) / s,
		             static_cast<double>(g_gap_pages[3].exchange(0)) / s,
		             static_cast<double>(g_gap_skipped.exchange(0)) / s,
		             static_cast<double>(g_gap_splits.exchange(0)) / s);
		{
			// [watcher-loop]: the tracker's page loop apart from its host calls (v445m).
			const auto   ticks_now   = CpuCost::Detail::Ticks();
			const double ns_per_tick = ticks_now > stats->start_ticks
			                               ? s * 1e9 / static_cast<double>(ticks_now - stats->start_ticks)
			                               : 0.0;
			stats->start_ticks       = ticks_now;
			std::string line         = "[watcher-loop] 5 s:";
			for (size_t who = 0; who < g_watcher_loop.size(); who++) {
				auto&        w       = g_watcher_loop[who];
				const auto   take    = [](std::atomic<uint64_t>& value) {
                    return static_cast<double>(value.exchange(0, std::memory_order_relaxed));
				};
				const double calls   = take(w.calls);
				const double loop    = take(w.loop_ticks) * ns_per_tick / 1e6 / s;
				const double lock    = take(w.lock_ticks) * ns_per_tick / 1e6 / s;
				const double protect = take(w.protect_ticks) * ns_per_tick / 1e6 / s;
				const double pcalls  = take(w.protect_calls);
				const double visited = take(w.visited);
				const double updated = take(w.updated);
				const double changed = take(w.changed);
				char         text[400];
				std::snprintf(text, sizeof(text),
				              "%s %s: %.0f calls/s, loop %.2f ms/s (%.0f ns/call, %.1f ns/page), lock wait %.2f ms/s, "
				              "Protect inside %.2f ms/s in %.0f calls/s; pages/s visited %.0f, count changed %.0f, "
				              "protection changed %.0f (%.1f%% of visited)",
				              who == 0 ? "" : " |", who == 0 ? "GPU thread" : "other threads", calls / s, loop,
				              calls != 0 ? loop * 1e6 * s / calls : 0.0, visited != 0 ? loop * 1e6 * s / visited : 0.0,
				              lock, protect, pcalls / s, visited / s, updated / s, changed / s,
				              visited != 0 ? 100.0 * changed / visited : 0.0);
				line += text;
			}
			std::fprintf(stderr, "%s\n", line.c_str());
		}
		if (ProtectVerify()) {
			std::fprintf(stderr,
			             "[protect-verify] %llu stale pages and %llu left-out pages checked against the "
			             "host, no difference\n",
			             static_cast<unsigned long long>(g_verify_stale_pages.load()),
			             static_cast<unsigned long long>(g_verify_gap_pages.load()));
		}
		std::vector<std::pair<std::string, std::array<double, 5>>> reasons(stats->reasons.begin(),
		                                                                stats->reasons.end());
		std::sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) {
			return a.second[1] + a.second[3] > b.second[1] + b.second[3];
		});
		for (size_t i = 0; i < reasons.size() && i < 30; i++) {
			const auto& [reason, r] = reasons[i];
			std::fprintf(stderr, "[page-protect]   %s: arm %.0f calls/s %.1f ms/s (%.1f pages/call), disarm %.0f calls/s %.1f ms/s\n",
			             reason.c_str(), r[0] / s, r[1] / 1000.0 / s, r[0] != 0 ? r[4] / r[0] : 0.0,
			             r[2] / s, r[3] / 1000.0 / s);
		}
		stats->reasons.clear();
		std::fprintf(stderr, "[page-protect] no-access %.0f pages/s; 2MB regions (pages/s read-only/no-access/rw):%s\n",
		             stats->noaccess_pages / s, text.c_str());
		stats->regions.clear();
		stats->noaccess_pages = 0;
		stats->arm_calls = stats->arm_pages = stats->disarm_calls = stats->disarm_pages = 0;
		stats->arm_us = stats->disarm_us = 0;
		stats->arm_us_gpu = stats->disarm_us_gpu = 0;
		stats->armed.clear();
		stats->start = now;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
	            Region* region = nullptr) noexcept {
		// [watcher-loop]: this call's time is not the loop's (the stats' own bookkeeping included).
		struct WatcherProtectTime {
			const uint64_t begin = t_in_watcher_loop ? CpuCost::Detail::Ticks() : 0;
			~WatcherProtectTime() {
				if (begin != 0) {
					t_watcher_protect_ticks += CpuCost::Detail::Ticks() - begin;
					t_watcher_protect_calls++;
				}
			}
		} watcher_protect_time;
		const bool gpu_ownership_change =
		    t_protect_kind != nullptr &&
		    (std::strcmp(t_protect_kind, "gpu-dirty") == 0 ||
		     std::strcmp(t_protect_kind, "gpu-clean") == 0);
		const bool queue_change = Profiler::t_gpu_thread && gpu_ownership_change && region != nullptr &&
		                          SyncToggle::ProtectHelperEnabled();
		const bool queued_release = queue_change && std::strcmp(t_protect_kind, "gpu-clean") == 0 &&
		                            mode != Common::VirtualMemory::Mode::NoAccess;
		const auto region_base = Common::AlignDown(vaddr, REGION_SIZE);
		const auto first = static_cast<size_t>((vaddr - region_base) / PAGE_SIZE);
		const auto last = static_cast<size_t>((vaddr + size - region_base) / PAGE_SIZE);
		if (region != nullptr && !queued_release) {
			region->helper_release.UnsetRange(first, last);
		}
		if (queued_release) {
			region->helper_release.SetRange(first, last);
		}
		if (queue_change && QueueProtection(vaddr, size, mode)) {
			return;
		}
		if (queued_release) {
			region->helper_release.UnsetRange(first, last);
		}
		// A synchronous permission change must follow any queued GPU-thread change in FIFO order.
		DrainProtectionHelper(ProtectDrainReason::SynchronousProtection);
		const auto t0 = Stats() != nullptr ? std::chrono::steady_clock::now()
		                                   : std::chrono::steady_clock::time_point {};
		const bool ok = Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode);
		if (Stats() != nullptr) {
			Note(vaddr, size, mode,
			     std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
			         .count());
		}
		if (!ok) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		// [watcher-loop] (KYTY_PROTECT_STATS): this loop's own time, lock wait and Protect apart.
		const bool     loop_stats  = Stats() != nullptr;
		const uint64_t loop_begin  = loop_stats ? CpuCost::Detail::Ticks() : 0;
		SpinGuard      lock(region.lock);
		const uint64_t loop_locked = loop_stats ? CpuCost::Detail::Ticks() : 0;
		uint64_t       loop_updated = 0;
		uint64_t       loop_changed = 0;
		struct LoopNote {
			bool            on;
			uint64_t        begin, locked, pages;
			const uint64_t& updated;
			const uint64_t& changed;
			bool            outer_in_loop      = t_in_watcher_loop;
			uint64_t        outer_protect      = t_watcher_protect_ticks;
			uint64_t        outer_protect_call = t_watcher_protect_calls;
			LoopNote(bool on_, uint64_t begin_, uint64_t locked_, uint64_t pages_, const uint64_t& updated_,
			         const uint64_t& changed_)
			    : on(on_), begin(begin_), locked(locked_), pages(pages_), updated(updated_), changed(changed_) {
				if (on) {
					t_in_watcher_loop       = true;
					t_watcher_protect_ticks = 0;
					t_watcher_protect_calls = 0;
				}
			}
			~LoopNote() {
				if (!on) {
					return;
				}
				const auto end     = CpuCost::Detail::Ticks();
				const auto protect = t_watcher_protect_ticks;
				auto&      w       = g_watcher_loop[Profiler::t_gpu_thread ? 0 : 1];
				w.calls.fetch_add(1, std::memory_order_relaxed);
				w.lock_ticks.fetch_add(locked - begin, std::memory_order_relaxed);
				w.loop_ticks.fetch_add(end - locked > protect ? end - locked - protect : 0, std::memory_order_relaxed);
				w.protect_ticks.fetch_add(protect, std::memory_order_relaxed);
				w.protect_calls.fetch_add(t_watcher_protect_calls, std::memory_order_relaxed);
				w.visited.fetch_add(pages, std::memory_order_relaxed);
				w.updated.fetch_add(updated, std::memory_order_relaxed);
				w.changed.fetch_add(changed, std::memory_order_relaxed);
				t_in_watcher_loop       = outer_in_loop;
				t_watcher_protect_ticks = outer_protect;
				t_watcher_protect_calls = outer_protect_call;
			}
		} loop_note(loop_stats, loop_begin, loop_locked, last - first, loop_updated, loop_changed);
		const bool lazy = is_read && !track && t_lazy_read_release && LazyReadReleaseActive();
		t_protect_kind  = is_read ? (track ? "gpu-dirty" : "gpu-clean") : (track ? "cpu-clean" : "cpu-dirty");
		struct KindReset {
			~KindReset() { t_protect_kind = nullptr; }
		} kind_reset;
		auto       perms                 = region.HostPerms(first);
		uint64_t   range_begin           = 0;
		uint64_t   range_bytes           = 0;
		uint64_t   potential_range_bytes = 0;
		// KYTY_PROTECT_GAP: the first page a split left out, until the next changed page shows
		// that the unsplit call would have covered it.
		const uint64_t gap_bytes    = ProtectGapBytes();
		size_t         skipped_from = SIZE_MAX;
		const bool     verify       = ProtectVerify();
		const bool     counting     = verify || Stats() != nullptr;
		// KYTY_PROTECT_VERIFY: a run of pages whose host protection this pass relies on without
		// a call; checked before the pass's next host call, which could cover (and repair) them.
		size_t check_begin = 0;
		size_t check_end   = 0;
		const auto flush_check = [&] {
			if (check_end > check_begin) {
				VerifyNoAccess(base_addr + check_begin * PAGE_SIZE, (check_end - check_begin) * PAGE_SIZE);
			}
			check_begin = check_end = 0;
		};
		[[maybe_unused]] const auto note_check = [&](size_t page) {
			if (check_end > check_begin && check_end == page) {
				check_end++;
			} else {
				flush_check();
				check_begin = page;
				check_end   = page + 1;
			}
		};

		const auto release_pending = [&] {
			flush_check();
			if (range_bytes != 0) {
				Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms, &region);
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = region.HostPerms(page_index);
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			if constexpr (is_read) {
				if (update) {
					if constexpr (track) {
						// No-access again by its counts: the host already is.
						if (counting && region.stale_count.load(std::memory_order_relaxed) != 0 &&
						    region.stale.Get(page_index)) {
							g_stale_reused.fetch_add(1, std::memory_order_relaxed);
							if (verify) {
								note_check(page_index);
							}
						}
						region.ClearStale(page_index);
					} else if (new_count == 0 &&
					           old_perms == Common::VirtualMemory::Mode::NoAccess) {
						if (lazy) {
							if (counting && !(region.stale_count.load(std::memory_order_relaxed) != 0 &&
							                  region.stale.Get(page_index))) {
								g_stale_made.fetch_add(1, std::memory_order_relaxed);
								if (verify) {
									note_check(page_index);
								}
							}
							region.SetStale(page_index);
						} else {
							// A sync-toggle window can end while pages from its on-window are stale.
							region.ClearStale(page_index);
						}
					}
				}
			}
			const auto new_perms = region.HostPerms(page_index);

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms        = new_perms;
				skipped_from = SIZE_MAX; // the unsplit call would have ended here too
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
				if (gap_bytes != 0 && potential_range_bytes - range_bytes > gap_bytes) {
					// KYTY_PROTECT_GAP: this many unchanged pages since the last changed one.
					skipped_from = static_cast<size_t>(range_begin + range_bytes / PAGE_SIZE);
					release_pending();
				}
			}

			if (!update) {
				continue;
			}
			loop_updated++;

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				loop_changed++;
				if (range_bytes == 0) {
					if (skipped_from != SIZE_MAX) {
						// The unsplit call would have run on to this page: [skipped_from, here)
						// is what the split left out.
						g_gap_skipped.fetch_add(page_index - skipped_from, std::memory_order_relaxed);
						g_gap_splits.fetch_add(1, std::memory_order_relaxed);
						if (verify) {
							flush_check();
							VerifyGap(region, base_addr, skipped_from, page_index, perms);
						}
						skipped_from = SIZE_MAX;
					}
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				} else if (counting) {
					const uint64_t between = (potential_range_bytes - range_bytes) / PAGE_SIZE - 1;
					if (between != 0) {
						g_gap_pages[between <= 3 ? 0 : between <= 15 ? 1 : between <= 63 ? 2 : 3].fetch_add(
						    between, std::memory_order_relaxed);
					}
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
	}

	// KYTY_PROTECT_VERIFY, part 2: the pages a KYTY_PROTECT_GAP split left out get the host call
	// after all; the host's protection of each must be what it was.
	void VerifyGap(Region& region, uint64_t base_addr, size_t from, size_t to,
	               Common::VirtualMemory::Mode perms) {
		const uint64_t        vaddr = base_addr + from * PAGE_SIZE;
		const uint64_t        size  = (to - from) * PAGE_SIZE;
		std::vector<uint32_t> before;
		std::vector<uint32_t> after;
		if (!HostProtections(vaddr, size, before)) {
			return;
		}
		Protect(vaddr, size, perms, &region);
		if (!HostProtections(vaddr, size, after) || after.size() != before.size()) {
			return;
		}
		for (size_t i = 0; i < before.size(); i++) {
			if (before[i] != after[i]) {
				Fatal("KYTY_PROTECT_VERIFY: page 0x%016" PRIx64 " was left out by KYTY_PROTECT_GAP with "
				      "host protection 0x%" PRIx32 ", the call sets 0x%" PRIx32 " (gap 0x%016" PRIx64
				      "+0x%" PRIx64 ")",
				      vaddr + i * PAGE_SIZE, before[i], after[i], vaddr, size);
			}
		}
		g_verify_gap_pages.fetch_add(before.size(), std::memory_order_relaxed);
	}

	// The stale pages around vaddr's (a run of them with one protection by their counts, within
	// the region) get that protection; whether it allows the access.
	bool ResolveStale(uint64_t vaddr, PageFaultAccess access) noexcept {
		auto* region = FindRegion(vaddr);
		if (region == nullptr) {
			return false;
		}
		const auto base  = Common::AlignDown(vaddr, REGION_SIZE);
		const auto index = static_cast<size_t>((vaddr - base) / PAGE_SIZE);
		SpinGuard  lock(region->lock);
		const bool stale = region->stale.Get(index);
		const bool helper_release = region->helper_release.Get(index);
		if (helper_release) {
			region->helper_release.Unset(index);
		}
		const auto perms = region->pages[index].Perms();
		size_t     begin = index;
		size_t     end   = index + 1;
		if (stale) {
			while (begin > 0 && region->stale.Get(begin - 1) &&
			       region->pages[begin - 1].Perms() == perms) {
				begin--;
			}
			while (end < REGION_PAGES && region->stale.Get(end) &&
			       region->pages[end].Perms() == perms) {
				end++;
			}
			for (size_t i = begin; i < end; i++) {
				region->ClearStale(i);
			}
			g_stale_resolved.fetch_add(end - begin, std::memory_order_relaxed);
		}
		const bool allows_access = access == PageFaultAccess::Write
		                               ? perms == Common::VirtualMemory::Mode::ReadWrite
		                               : perms != Common::VirtualMemory::Mode::NoAccess;
		const bool is_cpu_access = access == PageFaultAccess::Read || access == PageFaultAccess::Write;
		if (!stale && (!helper_release || !is_cpu_access || !allows_access)) {
			return false;
		}
		t_protect_kind = "stale-fault";
		struct KindReset {
			~KindReset() { t_protect_kind = nullptr; }
		} kind_reset;
		if (perms != Common::VirtualMemory::Mode::NoAccess) {
			Protect(base + begin * PAGE_SIZE, (end - begin) * PAGE_SIZE, perms, region);
		}
		// A queued release may have changed tracker state before its host permission landed. A
		// fault that drains that release is spurious; reapply the now-permitted access and retry.
		return allows_access;
	}

	// Every stale page of the range gets the protection its counts ask for.
	void ForgetStale(uint64_t vaddr, uint64_t size) noexcept {
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region      = FindRegion(chunk_begin);
			const auto first       = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last        = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			chunk_begin            = chunk_end;
			if (region == nullptr) {
				continue;
			}
			SpinGuard lock(region->lock);
			region->helper_release.UnsetRange(first, last);
			if (region->stale_count.load(std::memory_order_relaxed) == 0) {
				continue;
			}
			for (size_t page = first; page < last;) {
				if (!region->stale.Get(page)) {
					page++;
					continue;
				}
				const auto perms = region->pages[page].Perms();
				size_t     run   = page;
				while (run < last && region->stale.Get(run) && region->pages[run].Perms() == perms) {
					region->ClearStale(run);
					run++;
				}
				if (perms != Common::VirtualMemory::Mode::NoAccess) {
					Protect(region_base + page * PAGE_SIZE, (run - page) * PAGE_SIZE, perms, region);
				}
				page = run;
			}
		}
	}

	void ForgetProtectionFaults(uint64_t vaddr, uint64_t size) noexcept {
		if (!GuestRange {vaddr, size}.Valid() || vaddr >= ADDRESS_SIZE) {
			return;
		}
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = FindRegion(chunk_begin);
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			chunk_begin = chunk_end;
			if (region == nullptr) {
				continue;
			}
			SpinGuard lock(region->lock);
			region->helper_release.UnsetRange(first, last);
		}
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

namespace {

class ProtectHelper;
std::atomic<ProtectHelper*> g_protect_helper {nullptr};

class ProtectHelper final {
public:
	struct Request {
		uint64_t                      vaddr = 0;
		uint64_t                      size = 0;
		uint64_t                      first_seq = 0;
		uint64_t                      last_seq = 0;
		Common::VirtualMemory::Mode   mode = Common::VirtualMemory::Mode::NoAccess;
	};

	// KYTY_PROTECT_DEFER: no thread; the queue is applied by whoever drains it.
	ProtectHelper() {
		if (!m_inline) {
			m_thread = std::thread([this] { Run(); });
		}
	}
	~ProtectHelper() {
		Drain(ProtectDrainReason::Mapping);
		{
			std::lock_guard lock(m_mutex);
			m_stopping = true;
		}
		m_work.notify_one();
		if (m_thread.joinable()) {
			m_thread.join();
		}
		g_protect_helper.store(nullptr, std::memory_order_release);
	}
	KYTY_CLASS_NO_COPY(ProtectHelper);

	bool Enqueue(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		try {
			std::lock_guard lock(m_mutex);
			const uint64_t  seq = m_issued + 1;
			if (!m_queue.empty()) {
				auto& previous = m_queue.back();
				const bool follows_previous = previous.vaddr + previous.size == vaddr;
				const bool precedes_previous = vaddr + size == previous.vaddr;
				if (previous.mode == mode && (follows_previous || precedes_previous)) {
					if (precedes_previous) {
						previous.vaddr = vaddr;
					}
					previous.size += size;
					previous.last_seq = seq;
					m_issued = seq;
					m_issued_atomic.store(seq, std::memory_order_release);
					m_queued++;
					m_coalesced++;
					m_work.notify_one();
					return true;
				}
			}
			m_queue.push_back({vaddr, size, seq, seq, mode});
			m_issued = seq;
			m_issued_atomic.store(seq, std::memory_order_release);
			m_queued++;
			m_work.notify_one();
			return true;
		} catch (...) {
			return false;
		}
	}

	void Drain(ProtectDrainReason reason) noexcept {
		// Most drains find nothing queued (faults, signals and guest writes run ~1.5M times a
		// second in the cinematic): answer those without the mutex the helper and the game threads
		// would otherwise contend on. m_completed_atomic only grows; a request issued after this
		// load is not one this drain has to wait for.
		if (m_completed_atomic.load(std::memory_order_acquire) >=
		    m_issued_atomic.load(std::memory_order_acquire)) {
			return;
		}
		const bool gpu_thread = Profiler::t_gpu_thread;
		const auto begin = std::chrono::steady_clock::now();
		std::unique_lock lock(m_mutex);
		m_drains[static_cast<size_t>(reason)]++;
		if (m_inline) {
			// Applied here, under the lock: concurrent drainers keep the queue's order.
			while (!m_queue.empty()) {
				const auto request = m_queue.front();
				m_queue.pop_front();
				const auto applied_at = std::chrono::steady_clock::now();
				if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(request.vaddr, request.size,
				                                                     request.mode)) {
					Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32,
					      request.vaddr, static_cast<uint32_t>(request.mode));
				}
				m_completed = request.last_seq;
				m_completed_atomic.store(request.last_seq, std::memory_order_release);
				m_applied++;
				m_helper_busy_ms += std::chrono::duration<double, std::milli>(
				                        std::chrono::steady_clock::now() - applied_at)
				                        .count();
			}
			if (gpu_thread) {
				m_gpu_wait_ms += std::chrono::duration<double, std::milli>(
				                     std::chrono::steady_clock::now() - begin)
				                     .count();
			}
			ReportStats(lock);
			return;
		}
		const auto target = m_issued;
		if (m_completed < target) {
			m_done.wait(lock, [this, target] { return m_completed >= target; });
		}
		if (gpu_thread) {
			m_gpu_wait_ms += std::chrono::duration<double, std::milli>(
			                     std::chrono::steady_clock::now() - begin)
			                     .count();
		}
	}

private:
	static bool StatsEnabled() noexcept {
		static const bool enabled = [] {
			const char* gpu = std::getenv("KYTY_GPU_THREAD_STATS");
			const char* protect = std::getenv("KYTY_PROTECT_STATS");
			return (gpu != nullptr && gpu[0] == '1') || (protect != nullptr && protect[0] == '1');
		}();
		return enabled;
	}

	void ReportStats(std::unique_lock<std::mutex>& lock) {
		if (!StatsEnabled()) {
			return;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - m_reported < std::chrono::seconds(10)) {
			return;
		}
		m_reported = now;
		const auto queued = std::exchange(m_queued, 0);
		const auto applied = std::exchange(m_applied, 0);
		const auto coalesced = std::exchange(m_coalesced, 0);
		const auto drains = m_drains;
		m_drains.fill(0);
		const auto gpu_wait_ms = std::exchange(m_gpu_wait_ms, 0.0);
		const auto helper_busy_ms = std::exchange(m_helper_busy_ms, 0.0);
		lock.unlock();
		std::fprintf(stderr,
		             "[protect-helper] queued=%llu applied=%llu coalesced=%llu drains "
		             "submit=%llu readback-submit=%llu fault=%llu guest-write=%llu "
		             "guest-signal=%llu mapping=%llu toggle-off=%llu sync-protect=%llu | "
		             "GPU-thread drain wait %.2f ms helper busy %.2f ms\n",
		             static_cast<unsigned long long>(queued), static_cast<unsigned long long>(applied),
		             static_cast<unsigned long long>(coalesced),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::Submit)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::ReadbackSubmit)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::Fault)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::GuestWrite)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::GuestSignal)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::Mapping)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::ToggleOff)]),
		             static_cast<unsigned long long>(drains[static_cast<size_t>(ProtectDrainReason::SynchronousProtection)]),
		             gpu_wait_ms, helper_busy_ms);
		lock.lock();
	}

	void Run() noexcept {
		std::unique_lock lock(m_mutex);
		for (;;) {
			if (m_queue.empty()) {
				m_work.wait_for(lock, std::chrono::seconds(10),
				                [this] { return m_stopping || !m_queue.empty(); });
				ReportStats(lock);
				if (m_stopping && m_queue.empty()) {
					return;
				}
				if (m_queue.empty()) {
					continue;
				}
			}
			const auto request = m_queue.front();
			m_queue.pop_front();
			lock.unlock();
			const auto begin = std::chrono::steady_clock::now();
			const bool ok = Libs::LibKernel::Memory::ProtectGuestHostMemory(
			    request.vaddr, request.size, request.mode);
			const auto elapsed = std::chrono::duration<double, std::milli>(
			                         std::chrono::steady_clock::now() - begin)
			                         .count();
			if (!ok) {
				Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32,
				      request.vaddr, static_cast<uint32_t>(request.mode));
			}
			lock.lock();
			m_completed = request.last_seq;
			m_completed_atomic.store(request.last_seq, std::memory_order_release);
			m_applied++;
			m_helper_busy_ms += elapsed;
			m_done.notify_all();
			ReportStats(lock);
		}
	}

	std::mutex                          m_mutex;
	std::condition_variable             m_work, m_done;
	std::deque<Request>                 m_queue;
	uint64_t                            m_issued = 0, m_completed = 0;
	std::atomic<uint64_t>               m_issued_atomic {0}, m_completed_atomic {0};
	uint64_t                            m_queued = 0, m_applied = 0, m_coalesced = 0;
	std::array<uint64_t, static_cast<size_t>(ProtectDrainReason::Count)> m_drains {};
	double                              m_gpu_wait_ms = 0.0, m_helper_busy_ms = 0.0;
	std::chrono::steady_clock::time_point m_reported = std::chrono::steady_clock::now();
	bool                                m_stopping = false;
	const bool                          m_inline = SyncToggle::ProtectDeferConfigured();
	std::thread                         m_thread;
};

ProtectHelper* GetProtectHelper() noexcept {
	static std::mutex creation_mutex;
	static std::unique_ptr<ProtectHelper> owner;
	if (auto* helper = g_protect_helper.load(std::memory_order_acquire); helper != nullptr) {
		return helper;
	}
	std::lock_guard lock(creation_mutex);
	if (auto* helper = g_protect_helper.load(std::memory_order_relaxed); helper != nullptr) {
		return helper;
	}
	try {
		owner = std::make_unique<ProtectHelper>();
		auto* helper = owner.get();
		g_protect_helper.store(helper, std::memory_order_release);
		return helper;
	} catch (...) {
		return nullptr;
	}
}

bool QueueProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
	auto* helper = GetProtectHelper();
	return helper != nullptr && helper->Enqueue(vaddr, size, mode);
}

} // namespace

void DrainProtectionHelper(ProtectDrainReason reason) noexcept {
	if (auto* helper = g_protect_helper.load(std::memory_order_acquire); helper != nullptr) {
		helper->Drain(reason);
	}
}

void DrainProtectionHelperBeforeToggleOff() noexcept {
	DrainProtectionHelper(ProtectDrainReason::ToggleOff);
}

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {
	// [fault-census]: the table exists before the first fault is counted (none is made in a handler).
	if (FaultCensusEnabledSlow() && g_fault_census == nullptr) {
		g_fault_census = new FaultCensusState;
	}
}

bool FaultCensus::Enabled() noexcept {
	return g_fault_census != nullptr;
}

void FaultCensus::Report(void* context, Describe describe) {
	auto* census = g_fault_census;
	if (census == nullptr) {
		return;
	}
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count();
	auto last = census->reported_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		census->reported_ms.compare_exchange_strong(last, ms);
		return;
	}
	if (ms - last < 5000 || !census->reported_ms.compare_exchange_strong(last, ms)) {
		return;
	}
	using State = FaultCensusState;
	const double s = static_cast<double>(ms - last) / 1000.0;
	struct Entry {
		uint64_t key;
		uint32_t thread;
		uint64_t count;
	};
	// Faults that land while the table is read are counted in the next window or lost (a count
	// taken and its slot cleared are two steps); the totals are exact.
	std::vector<Entry> entries;
	for (auto& slot: census->slots) {
		const auto key = slot.key.load(std::memory_order_acquire);
		if (key == 0) {
			continue;
		}
		const auto count = slot.count.exchange(0, std::memory_order_relaxed);
		if (count != 0) {
			entries.push_back({key, slot.thread.load(std::memory_order_acquire), count});
		}
	}
	// The slots keep their keys while the table has room (the same pages fault again); a table
	// half full starts again, which loses the faults of that moment.
	if (census->used.load(std::memory_order_relaxed) >= State::Slots / 4u) {
		for (auto& slot: census->slots) {
			slot.thread.store(0, std::memory_order_relaxed);
			slot.count.store(0, std::memory_order_relaxed);
			slot.key.store(0, std::memory_order_release);
		}
		census->used.store(0, std::memory_order_relaxed);
	}
	std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
		return a.key != b.key ? a.key < b.key : a.thread < b.thread;
	});
	std::vector<Entry> merged;
	for (const auto& entry: entries) {
		if (!merged.empty() && merged.back().key == entry.key && merged.back().thread == entry.thread) {
			merged.back().count += entry.count;
		} else {
			merged.push_back(entry);
		}
	}
	std::sort(merged.begin(), merged.end(), [](const Entry& a, const Entry& b) { return a.count > b.count; });
	// By thread, over every entry.
	std::unordered_map<uint32_t, std::array<uint64_t, 2>> by_thread; // reads, writes
	for (const auto& entry: merged) {
		by_thread[entry.thread][(entry.key & State::Write) != 0 ? 1 : 0] += entry.count;
	}
	const auto thread_name = [census](uint32_t thread) -> std::string {
		for (const auto& name: census->names) {
			if (name.thread.load(std::memory_order_relaxed) == thread && name.ready.load(std::memory_order_acquire)) {
				return name.name;
			}
		}
		return "?";
	};
	const auto take = [s](std::atomic<uint64_t>& value) {
		return static_cast<double>(value.exchange(0, std::memory_order_relaxed)) / s;
	};
	const double total = take(census->total), writes = take(census->writes);
	std::string  text;
	char         line[512];
	std::snprintf(line, sizeof(line),
	              "[fault-census] 5 s: %.0f faults/s: read %.0f, write %.0f | page GPU-dirty (read-watched) %.0f, "
	              "CPU-clean only (write-watched) %.0f, neither (stale, released or not the tracker's) %.0f | on the GPU "
	              "thread %.0f | distinct (page, thread, kind) %zu, not counted (table full) %.0f/s\n",
	              total, total - writes, writes, take(census->gpu_dirty), take(census->cpu_clean), take(census->untracked),
	              take(census->gpu_thread), merged.size(), take(census->dropped));
	text += line;
	{
		std::vector<std::pair<uint32_t, std::array<uint64_t, 2>>> threads(by_thread.begin(), by_thread.end());
		std::sort(threads.begin(), threads.end(), [](const auto& a, const auto& b) {
			return a.second[0] + a.second[1] > b.second[0] + b.second[1];
		});
		text += "[fault-census]   by guest thread (id name: reads/s writes/s):";
		for (size_t i = 0; i < threads.size() && i < 12; i++) {
			std::snprintf(line, sizeof(line), " %u %s: %.0f %.0f |", threads[i].first - 1u,
			              thread_name(threads[i].first).c_str(), static_cast<double>(threads[i].second[0]) / s,
			              static_cast<double>(threads[i].second[1]) / s);
			text += line;
		}
		text += "\n";
	}
	for (size_t i = 0; i < merged.size() && i < 10; i++) {
		const auto&    entry   = merged[i];
		const uint64_t address = ((entry.key & ~(uint64_t {1} << 63u)) >> 4u) * PAGE_SIZE;
		char           owner[160] = "?";
		if (describe != nullptr) {
			describe(context, address, owner, sizeof(owner));
		}
		std::snprintf(line, sizeof(line),
		              "[fault-census]   #%zu page 0x%012llx %s %.1f/s thread %u %s%s | at the fault: %s | now: %s\n", i + 1,
		              static_cast<unsigned long long>(address), (entry.key & State::Write) != 0 ? "write" : "read ",
		              static_cast<double>(entry.count) / s, entry.thread - 1u, thread_name(entry.thread).c_str(),
		              (entry.key & State::GpuThread) != 0 ? " (GPU thread)" : "",
		              (entry.key & State::GpuDirty) != 0
		                  ? ((entry.key & State::CpuClean) != 0 ? "GPU-dirty + CPU-clean" : "GPU-dirty")
		                  : ((entry.key & State::CpuClean) != 0 ? "CPU-clean (not GPU-dirty)" : "neither"),
		              owner);
		text += line;
	}
	std::fputs(text.c_str(), stderr);
}

PageManager::~PageManager() {
	DrainProtectionHelper(ProtectDrainReason::Mapping);
}

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

bool PageManager::IsReadWatched(uint64_t vaddr) const noexcept {
	const auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto index = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	return region->pages[index].access_watchers != 0;
}

void PageManager::ResyncHostProtection(uint64_t vaddr, uint64_t size) {
	const CpuCost::Phase cpu_phase_("ResyncHostProtection");
	DrainProtectionHelper(ProtectDrainReason::Mapping);
	if (!GuestRange {vaddr, size}.Valid() || vaddr >= ADDRESS_SIZE) {
		return;
	}
	const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto end   = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
	for (auto chunk_begin = begin; chunk_begin < end;) {
		const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
		const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
		const auto first       = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
		const auto last        = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
		auto*      region      = m_impl->FindRegion(chunk_begin);
		chunk_begin            = chunk_end;
		if (region == nullptr) {
			continue; // never watched: the host protection is not the tracker's
		}
		SpinGuard lock(region->lock);
		region->helper_release.UnsetRange(first, last);
		// Runs of watched pages with the same protection, one host call each (stale pages stay
		// no-access).
		for (size_t page = first; page < last;) {
			const auto perms = region->HostPerms(page);
			size_t     run   = page + 1;
			while (run < last && region->HostPerms(run) == perms) {
				run++;
			}
			if (perms != Common::VirtualMemory::Mode::ReadWrite) {
				m_impl->Protect(region_base + page * PAGE_SIZE, (run - page) * PAGE_SIZE, perms,
				                region);
			}
			page = run;
		}
	}
}

bool PageManager::ResolveStale(uint64_t vaddr, PageFaultAccess access) noexcept {
	if (g_fault_census != nullptr && vaddr < ADDRESS_SIZE) [[unlikely]] {
		// [fault-census]: the page's watchers as they are now, read without the region's lock.
		bool gpu_dirty = false;
		bool cpu_clean = false;
		if (const auto* region = m_impl->FindRegion(vaddr); region != nullptr) {
			const auto& page = region->pages[static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE)];
			gpu_dirty        = page.access_watchers != 0;
			cpu_clean        = page.write_watchers != 0;
		}
		FaultCensusNote(vaddr, access == PageFaultAccess::Write, gpu_dirty, cpu_clean);
	}
	if (!LazyReadReleaseMayBeActive() && !SyncToggle::ProtectHelperConfigured() &&
	    !SyncToggle::ProtectHelperAlwaysEnabled()) {
		return false;
	}
	DrainProtectionHelper(ProtectDrainReason::Fault);
	return vaddr < ADDRESS_SIZE && m_impl->ResolveStale(vaddr, access);
}

void PageManager::ForgetStale(uint64_t vaddr, uint64_t size) noexcept {
	DrainProtectionHelper(ProtectDrainReason::Mapping);
	if (GuestRange {vaddr, size}.Valid() && vaddr < ADDRESS_SIZE) {
		m_impl->ForgetStale(vaddr, size);
	}
}

void PageManager::ForgetProtectionFaults(uint64_t vaddr, uint64_t size) noexcept {
	DrainProtectionHelper(ProtectDrainReason::Mapping);
	m_impl->ForgetProtectionFaults(vaddr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
