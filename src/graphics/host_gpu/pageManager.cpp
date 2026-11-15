#include "graphics/host_gpu/pageManager.h"
#include "common/fpsLog.h"

#include "common/alignment.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/timeline.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
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
		SpinGuard  lock(region.lock);
		const bool lazy = is_read && !track && t_lazy_read_release && LazyReadReleaseActive();
		t_protect_kind  = is_read ? (track ? "gpu-dirty" : "gpu-clean") : (track ? "cpu-clean" : "cpu-dirty");
		struct KindReset {
			~KindReset() { t_protect_kind = nullptr; }
		} kind_reset;
		auto       perms                 = region.HostPerms(first);
		uint64_t   range_begin           = 0;
		uint64_t   range_bytes           = 0;
		uint64_t   potential_range_bytes = 0;

		const auto release_pending = [&] {
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
						region.ClearStale(page_index);
					} else if (new_count == 0 &&
					           old_perms == Common::VirtualMemory::Mode::NoAccess) {
						if (lazy) {
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
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
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

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

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
