#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SYNCTOGGLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SYNCTOGGLE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace Libs::Graphics {
void DrainProtectionHelperBeforeToggleOff() noexcept;
}

// KYTY_SYNC_TOGGLE=<seconds> (experiment): every that many seconds the emulator switches between
// its normal overlap of game, GPU thread and host GPU and a synchronous mode, and prints
// "[sync-toggle] flip=<n> on|off". KYTY_SYNC_TOGGLE_MODE picks what the synchronous mode does:
// "all" (default): end-of-pipe labels become visible at host GPU completion (as
// KYTY_LABELS_AT_COMPLETION) and the GPU thread waits for the host GPU to finish every frame at
// its flip; "labels": the labels only; "idle": the flip wait only; "upload": see ForceUpload;
// "drawbarrier": no barrier after draws whose shaders write buffers (KYTY_DRAW_WRITE_BARRIER=0's
// experiment) while on; "replay": multi-draw replay (KYTY_MULTI_DRAW_REPLAY) on while on;
// "indirectreplay": fast-launch GPU-argument draw replay (KYTY_INDIRECT_REPLAY) while on;
// "gfxprio": the graphics queue goes first on the GPU thread (KYTY_GFX_PRIORITY) while on;
// "async": the compute queues run on the async-compute thread (KYTY_ASYNC_THREAD) while on;
// "coalesce": back-to-back barriers recorded as one (KYTY_BARRIER_COALESCE, barrierMerge.h);
// "fastfinish": game-thread readbacks lift a copied clean page without a second GPU-thread step;
// "helper": GPU-thread GPU-dirty/gpu-clean page protections run on the protection helper.
// "rbsubmit" (v445q): copy-queue readbacks are submitted by a worker thread (ReadbackSubmitThread).
// "ffclean" (v445s): with fastfinish on, a clean-read fault on a page with no GPU-written bytes
// left lifts the page instead of emulating the load (FfCleanWindow).
// "exactrange" (v445t): a game thread's readback is copied on the copy queue when no unfinished
// GPU write reaches its pages, by the writes' own ranges (ExactRangeWindow).
// "bitmapclean" (v445t): a readback step of a page with no GPU-written bytes does not wait for
// the write bitmap's region tick (BitmapCleanWindow).
// "gpupad" (v445m): a MEASUREMENT ONLY: one extra internal compute dispatch per guest flip while
// on, a fixed amount of GPU work (KYTY_GPU_PAD_US, default 3000; gpuPad.h). It measures how much
// GPU time a frame has to spare.
// An A/B inside one run: the dropout rate of a run varies far more between runs than between
// settings.
namespace Libs::Graphics::SyncToggle {

inline uint32_t PeriodSeconds() {
	static const uint32_t period = [] {
		const char* value = std::getenv("KYTY_SYNC_TOGGLE");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return period;
}

enum Mode : uint64_t {
	ModeLabels      = 1,
	ModeIdle        = 2,
	ModeUpload      = 4,
	ModeDrawBarrier = 8,
	ModeReplay      = 16,
	ModeGfxPriority = 32,
	ModeAsync       = 64,
	ModeTrackedBatch = 128,
	ModeRegionMemo   = 256,
	ModeMergedBarrier = 512,
	ModeNoReadbackWait = 1024,
	ModePreciseFlush   = 2048,
	ModeCopyInflight   = 4096,
	ModeDmaReadback    = 8192,
	ModeEagerReadback  = 16384,
	ModeDmaInflight    = 32768,
	ModeNarrowCopy     = 65536,
	ModeProtectCoalesce = 131072,
	ModeProtectLazyRelease = 262144,
	ModeFlush4         = 1048576,
	ModeProtectDefer   = 2097152,
	ModeFastFinish     = 524288,
	ModeFlipNow        = 4194304,
	ModeGpuSlow        = 16777216,
	ModeDirtyFilter    = 67108864,
	ModeIndirectReplay = 8388608,
	ModeProtectHelper  = 33554432,
	ModeNoGuestBarrier = 134217728,
	ModeHotPath        = 268435456,
	ModeObtainMemo     = 536870912,
	ModeComputeMemo    = 1073741824,
	ModeBarrierMerge   = 2147483648u,
	ModeBarrierCoalesce = 4294967296ull,
	ModeProtectGap      = 8589934592ull,
	ModeDispatchBarrier = 17179869184ull,
	ModeUploadProtect   = 34359738368ull,
	ModeGpuPad          = 68719476736ull,
	ModeLazyReadback    = 137438953472ull,
	ModeWindowScan      = 274877906944ull,
	ModeSetReuse        = 549755813888ull,
	ModeSyncStats       = 1099511627776ull,
	ModeAllocCount      = 2199023255552ull,
	ModeTimers          = 4398046511104ull,
	ModeReadbackSubmit  = 8796093022208ull,
	ModeFfClean         = 17592186044416ull,
	ModeExactRange      = 35184372088832ull,
	ModeBitmapClean     = 70368744177664ull,
	ModeRunOverlap      = 140737488355328ull,
	ModeVariant         = 281474976710656ull
};

// KYTY_SYNC_TOGGLE_MODE: one name, or several separated by commas (v304), toggled together.
inline uint64_t ModeMask() {
	static const uint64_t mask = [] {
		const char* value = std::getenv("KYTY_SYNC_TOGGLE_MODE");
		if (value == nullptr || std::strcmp(value, "all") == 0) return static_cast<uint64_t>(ModeLabels | ModeIdle);
		const auto one = [](std::string_view name) -> uint64_t {
			if (name == "labels") return ModeLabels;
			if (name == "idle") return ModeIdle;
			if (name == "upload") return ModeUpload;
			if (name == "drawbarrier") return ModeDrawBarrier;
			if (name == "replay") return ModeReplay;
			if (name == "indirectreplay") return ModeIndirectReplay;
			if (name == "gfxprio") return ModeGfxPriority;
			if (name == "async") return ModeAsync;
			if (name == "tbatch") return ModeTrackedBatch;
			if (name == "region") return ModeRegionMemo;
			if (name == "mbarrier") return ModeMergedBarrier;
			if (name == "nrw") return ModeNoReadbackWait;
			if (name == "precise") return ModePreciseFlush;
			if (name == "inflight") return ModeCopyInflight;
			if (name == "dma") return ModeDmaReadback;
			if (name == "eager") return ModeEagerReadback;
			if (name == "dmainflight") return ModeDmaInflight;
			if (name == "narrowcopy") return ModeNarrowCopy;
			if (name == "protectcoalesce") return ModeProtectCoalesce;
			if (name == "protectlazyrelease") return ModeProtectLazyRelease;
			if (name == "flush4") return ModeFlush4;
			if (name == "fastfinish") return ModeFastFinish;
			if (name == "flipnow") return ModeFlipNow;
			if (name == "gpuslow") return ModeGpuSlow;
			if (name == "dirtyfilter") return ModeDirtyFilter;
			if (name == "helper") return ModeProtectHelper;
			if (name == "noguestbarrier") return ModeNoGuestBarrier;
			if (name == "hotpath") return ModeHotPath;
			if (name == "obtainmemo") return ModeObtainMemo;
			if (name == "computememo") return ModeComputeMemo;
			if (name == "barriermerge") return ModeBarrierMerge;
			if (name == "coalesce") return ModeBarrierCoalesce;
			if (name == "protectdefer") return ModeProtectDefer;
			if (name == "protectgap") return ModeProtectGap;
			if (name == "dispatchbarrier") return ModeDispatchBarrier;
			if (name == "uploadprotect") return ModeUploadProtect;
			if (name == "gpupad") return ModeGpuPad;
			if (name == "lazyreadback") return ModeLazyReadback;
			if (name == "windowscan") return ModeWindowScan;
			if (name == "setreuse") return ModeSetReuse;
			if (name == "syncstats") return ModeSyncStats;
			if (name == "alloccount") return ModeAllocCount;
			if (name == "timers") return ModeTimers;
			if (name == "rbsubmit") return ModeReadbackSubmit;
			if (name == "ffclean") return ModeFfClean;
			if (name == "exactrange") return ModeExactRange;
			if (name == "bitmapclean") return ModeBitmapClean;
			if (name == "runoverlap") return ModeRunOverlap;
			if (name == "variant") return ModeVariant;
			return 0;
		};
		uint64_t         result = 0;
		std::string_view list(value);
		while (!list.empty()) {
			const auto comma = list.find(',');
			result |= one(list.substr(0, comma));
			if (comma == std::string_view::npos) {
				break;
			}
			list.remove_prefix(comma + 1);
		}
		return result != 0 ? result : static_cast<uint64_t>(ModeLabels | ModeIdle);
	}();
	return mask;
}

inline std::atomic<bool>& State() {
	static std::atomic<bool> on {false};
	return on;
}

// KYTY_PROTECT_DEFER=1 (or the toggle mode "protectdefer", v368): the GPU thread's GPU-ownership
// protection changes go through the helper's queue as with "helper", but no helper thread applies
// them: they wait in the queue, adjacent ones merged, until the next drain (before every
// vkQueueSubmit, a fault, a guest signal or write, a mapping change, a synchronous protection),
// and the draining thread applies them. A GPU write cannot land before the submission's drain, so
// a CPU access before it sees the bytes as they are, as on the console.
inline bool ProtectDeferAlwaysEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PROTECT_DEFER");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline bool ProtectDeferConfigured() {
	return ProtectDeferAlwaysEnabled() ||
	       (PeriodSeconds() != 0 && (ModeMask() & ModeProtectDefer) != 0);
}

inline bool ProtectHelperAlwaysEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PROTECT_HELPER");
		return (value != nullptr && value[0] == '1') || ProtectDeferAlwaysEnabled();
	}();
	return enabled;
}

inline bool ProtectHelperConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & (ModeProtectHelper | ModeProtectDefer)) != 0;
}

inline bool ProtectHelperWindow() {
	return ProtectHelperConfigured() && State().load(std::memory_order_relaxed);
}

inline bool ProtectHelperEnabled() {
	return ProtectHelperAlwaysEnabled() || ProtectHelperWindow();
}

// Called once per guest flip on the GPU thread: updates the state (constant within a frame).
inline void AtFlip(uint64_t flip) {
	const auto period = PeriodSeconds();
	if (period == 0) {
		return;
	}
	static const auto start   = std::chrono::steady_clock::now();
	const auto        seconds = std::chrono::duration_cast<std::chrono::seconds>(
	                         std::chrono::steady_clock::now() - start)
	                         .count();
	const bool on = ((static_cast<uint64_t>(seconds) / period) % 2u) == 1u;
	const bool was_on = State().load(std::memory_order_relaxed);
	if (was_on && !on && (ModeMask() & (ModeProtectHelper | ModeProtectDefer)) != 0 &&
	    !ProtectHelperAlwaysEnabled()) {
		// Drain before the helper mode becomes observably off: later Protect calls are synchronous.
		DrainProtectionHelperBeforeToggleOff();
	}
	if (State().exchange(on, std::memory_order_relaxed) != on) {
		std::fprintf(stderr, "[sync-toggle] flip=%llu %s\n", static_cast<unsigned long long>(flip),
		             on ? "on" : "off");
	}
}

inline bool LabelsAtCompletion() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeLabels) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "upload": every buffer range a draw or dispatch binds read-only is uploaded again from guest
// memory first (unless the GPU wrote it), whatever the write tracker says. Tests whether the
// host copies miss CPU writes.
inline bool ForceUpload() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeUpload) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "dispatchbarrier": a MEASUREMENT ONLY (v445g): no write-hazard barrier before or after dispatches
// while on, races included. It gives the upper bound of what removing them could save on the GPU.
inline bool SkipDispatchBarrier() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeDispatchBarrier) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "uploadprotect" (v445j): KYTY_UPLOAD_PROTECT_BATCH's one protection pass per tracker region
// after a batched upload, instead of one per uploaded span (MemoryTracker::ForEachUploadRanges).
inline bool UploadProtectToggled() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeUploadProtect) != 0;
}
inline bool UploadProtectWindow() {
	return State().load(std::memory_order_relaxed);
}

// "gpupad" (v445m, MEASUREMENT ONLY): GpuPad's extra dispatch per guest flip in the on windows.
inline bool GpuPadConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeGpuPad) != 0;
}
inline bool GpuPadWindow() {
	return GpuPadConfigured() && State().load(std::memory_order_relaxed);
}

// "lazyreadback" (v445n): KYTY_LAZY_READBACK_RELEASE's lazy release of a readback window in the
// on windows only (bufferCache.cpp LazyReadbackRelease). Pages it leaves no-access are resolved by
// the fault handler or taken over by the next GPU write, whichever window that happens in.
inline bool LazyReadbackConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeLazyReadback) != 0;
}
inline bool LazyReadbackWindow() {
	return State().load(std::memory_order_relaxed);
}

// v445o (2): proof that each cut is taken. Counted where a cut's path runs, with a load and a
// store (no lock prefix: a lost count between threads does not matter here), and printed by
// sync.cpp at every change of the toggle's state for the window that just ended ([cut-hits]).
struct alignas(64) CutHits {
	std::atomic<uint64_t> window_scans {0};   // readback windows scanned with the bitmap
	std::atomic<uint64_t> window_by_page {0}; //   and page by page
	std::atomic<uint64_t> set_reused {0};     // DescriptorHeap::Commit: a finished set again
	std::atomic<uint64_t> set_busy {0};       //   the oldest noted set was not finished yet
	std::atomic<uint64_t> set_fresh {0};      //   a set from the pool
	std::atomic<uint64_t> sync_local {0};     // syncs counted thread-locally (at their fold)
	std::atomic<uint64_t> loop_untimed {0};   // GPU-thread loop iterations without the srt clocks
	std::atomic<uint64_t> loop_timed {0};
	std::atomic<uint64_t> program_untimed {0}; // queued dispatches without the program clock
	std::atomic<uint64_t> pair_memo {0};      // NoteGoodPair calls the last-key memo answered
	std::atomic<uint64_t> pair_set {0};       //   and the set lookup
	std::atomic<uint64_t> rb_thread {0};      // v445q: copy-queue readbacks submitted by the worker
	std::atomic<uint64_t> rb_inline {0};      //   and by the GPU thread itself
	std::atomic<uint64_t> rb_handoff_ns {0};  //   the worker's: from the hand-over to vkQueueSubmit's return
};
inline CutHits& Hits() {
	static CutHits hits;
	return hits;
}
inline void Bump(std::atomic<uint64_t>& counter, uint64_t by = 1) noexcept {
	counter.store(counter.load(std::memory_order_relaxed) + by, std::memory_order_relaxed);
}
inline bool CutsConfigured() {
	return PeriodSeconds() != 0 &&
	       (ModeMask() & (ModeWindowScan | ModeSetReuse | ModeSyncStats | ModeAllocCount | ModeTimers |
	                      ModeReadbackSubmit)) != 0;
}

// "rbsubmit" (v445q, KYTY_READBACK_SUBMIT_THREAD, default off): the vkQueueSubmit of a copy-queue
// readback (BufferCache::TryCopyQueueReadback) is made by a worker thread instead of the GPU
// thread; whoever waits for the copy waits for that submit first (bufferCache.cpp). =1: always;
// a toggled mode decides by window.
inline bool ReadbackSubmitConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeReadbackSubmit) != 0;
}
inline bool ReadbackSubmitAlways() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_SUBMIT_THREAD");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
inline bool ReadbackSubmitThread() {
	return ReadbackSubmitConfigured() ? State().load(std::memory_order_relaxed) : ReadbackSubmitAlways();
}

// v445o: five cuts of GPU-thread CPU work that change no output (g450g's 1 kHz profile of the GPU
// thread in a fight). Each has its own environment switch and a toggle mode. Configured(): the
// mode is being toggled; Window(): the toggle's state.
// v445p: on by default (g450j, in-run A/B of the five over 893 s of a fight: GPU thread -1.0 ms a
// frame without its waits, frame -0.8 ms; [cut-hits] showed every cut taken in the on windows
// only). The switch set to 0 turns a cut off. A toggled mode overrides both: the cut is on in the
// on windows and off in the off windows, whatever the switch says.
//
// "windowscan" (KYTY_WINDOW_SCAN): ReadMemoryStep finds the clean pages of a readback window with
// one walk of each range set instead of two map lookups per page (bufferCache.cpp).
// KYTY_WINDOW_SCAN_VERIFY=1 computes both and stops the emulator on a difference.
inline bool WindowScanConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeWindowScan) != 0;
}
inline bool WindowScanWindow() {
	return WindowScanConfigured() && State().load(std::memory_order_relaxed);
}
inline bool WindowScan() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WINDOW_SCAN");
		return value == nullptr || value[0] != '0';
	}();
	return WindowScanConfigured() ? State().load(std::memory_order_relaxed) : enabled;
}

// "setreuse" (KYTY_DESC_SET_REUSE): DescriptorHeap::Commit hands out a set of the layout whose
// last command buffer the host GPU has finished, instead of allocating one (descriptorHeap.cpp).
inline bool SetReuseConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeSetReuse) != 0;
}
inline bool SetReuseWindow() {
	return SetReuseConfigured() && State().load(std::memory_order_relaxed);
}
inline bool SetReuse() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DESC_SET_REUSE");
		return value == nullptr || value[0] != '0';
	}();
	return SetReuseConfigured() ? State().load(std::memory_order_relaxed) : enabled;
}

// "syncstats" (KYTY_SYNC_STATS_LOCAL): the per-sync [floor] counters are counted in plain
// thread-local numbers and added to the shared ones once per guest flip (bufferCache.cpp
// RecordSync, FoldSyncCounters).
inline bool SyncStatsLocalConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeSyncStats) != 0;
}
inline bool SyncStatsLocalWindow() {
	return SyncStatsLocalConfigured() && State().load(std::memory_order_relaxed);
}
inline bool SyncStatsLocal() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SYNC_STATS_LOCAL");
		return value == nullptr || value[0] != '0';
	}();
	return SyncStatsLocalConfigured() ? State().load(std::memory_order_relaxed) : enabled;
}

// "alloccount" (KYTY_ALLOC_COUNT_LOCAL): operator new counts into a block its thread owns
// (allocStats.cpp). That file reads the mask and the state itself: it runs before main, so its
// default is in its InitAllocMode, not in main.cpp's table of defaults (applied later).
inline bool AllocCountConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeAllocCount) != 0;
}

// "timers" (KYTY_TIMERS_CUT): no clock reads on the GPU thread for statistics nothing prints, a
// one-load test for CpuCost::Phase with KYTY_CPU_COST off, DrawPrep::NoteGoodPair's last-key memo.
inline bool TimersCutConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeTimers) != 0;
}
inline bool TimersCutWindow() {
	return TimersCutConfigured() && State().load(std::memory_order_relaxed);
}
inline bool TimersCut() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TIMERS_CUT");
		return value == nullptr || value[0] != '0';
	}();
	return TimersCutConfigured() ? State().load(std::memory_order_relaxed) : enabled;
}

inline bool SkipDrawWriteBarrier() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeDrawBarrier) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "replay": KYTY_MULTI_DRAW_REPLAY's replay of multi-draw packets on (RenderStats).
inline bool ReplayWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeReplay) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "indirectreplay": replay consecutive GPU-argument fast-launch mesh draws.
inline bool IndirectReplayWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeIndirectReplay) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "gfxprio": the GPU thread takes the graphics queue first (KYTY_GFX_PRIORITY).
inline bool GfxPriorityWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeGfxPriority) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "async": the compute queues run on the async-compute thread (KYTY_ASYNC_THREAD).
inline bool AsyncConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeAsync) != 0;
}
inline bool AsyncWindow() {
	return AsyncConfigured() && State().load(std::memory_order_relaxed);
}

// "hotpath" (v358): KYTY_HOTPATH's exact cuts of per-draw/per-dispatch work that changes no output:
// no clock reads for stats that are off, no second image pass per dispatch, no driver poll for an
// operation of the unsubmitted command buffer, no gave-up lookup after a fast-program hit.
// KYTY_HOTPATH=1: always on; =0 or unset: off (in-run A/B with KYTY_SYNC_TOGGLE_MODE=hotpath).
inline bool HotPath() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_HOTPATH");
		return value != nullptr && value[0] == '1';
	}();
	return always || (PeriodSeconds() != 0 && (ModeMask() & ModeHotPath) != 0 &&
	                  State().load(std::memory_order_relaxed));
}

// "obtainmemo" (v359): KYTY_OBTAIN_MEMO's per-recording memo of read-only buffer obtains.
// KYTY_OBTAIN_MEMO=1: always on.
inline bool ObtainMemo() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_OBTAIN_MEMO");
		return value != nullptr && value[0] == '1';
	}();
	return always || (PeriodSeconds() != 0 && (ModeMask() & ModeObtainMemo) != 0 &&
	                  State().load(std::memory_order_relaxed));
}

// "computememo" (v363): KYTY_COMPUTE_MEMO, a compute dispatch with exactly the previous lookup's
// inputs reuses its program (ProgramCache::Get). KYTY_COMPUTE_MEMO=1: always on.
inline bool ComputeMemo() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_COMPUTE_MEMO");
		return value != nullptr && value[0] == '1';
	}();
	return always || (PeriodSeconds() != 0 && (ModeMask() & ModeComputeMemo) != 0 &&
	                  State().load(std::memory_order_relaxed));
}

// "protectgap" (v445e): KYTY_PROTECT_GAP's split of page-protection calls (pageManager.cpp), in
// the on windows only; in the off windows calls span their unchanged pages as with
// KYTY_PROTECT_GAP=0. Each protection pass reads the setting once and is complete by itself.
inline bool ProtectGapToggled() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeProtectGap) != 0;
}
inline bool ProtectGapWindow() {
	return State().load(std::memory_order_relaxed);
}

// "tbatch": KYTY_TRACKED_BATCH's one-pass sync + mark of a tracked binding's predicted ranges.
inline bool TrackedBatchWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeTrackedBatch) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "region" (v304): KYTY_SYNC_SKIP_REGION's per-range memo key (the range's region versions instead
// of the global epochs any page anywhere moves).
inline bool RegionMemoWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeRegionMemo) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "mbarrier" (v305): KYTY_INDIRECT_MERGED_BARRIERS (two barriers instead of four around each indirect
// dispatch's thread-count conversion).
inline bool MergedBarrierWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeMergedBarrier) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "nrw" (v306, MEASUREMENT ONLY, wrong results): KYTY_TEST_NO_READBACK_WAIT in the on windows: game
// threads read GPU-written pages without a readback, to bound what the readback waits cost.
inline bool NoReadbackWaitWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeNoReadbackWait) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "noguestbarrier" (v332; upstream 70668fbd): the guest's cache writeback events, RELEASE_MEM cache
// actions and suspend points record no ALL_COMMANDS -> ALL_COMMANDS barrier. The renderer's own barriers
// order the work (after buffer-writing draws and dispatches, before their writes, and image
// transitions). Partial flushes keep their barrier: dispatch runs rely on them. KYTY_NO_GUEST_BARRIERS=1: always.
inline bool NoGuestBarriers() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_NO_GUEST_BARRIERS");
		return value != nullptr && value[0] == '1';
	}();
	return always || (PeriodSeconds() != 0 && (ModeMask() & ModeNoGuestBarrier) != 0 &&
	                  State().load(std::memory_order_relaxed));
}

// "precise" (v307): KYTY_PRECISE_FLUSHES (partial-flush events as stage-scoped barriers).
inline bool PreciseFlushWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModePreciseFlush) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "inflight" (v308): KYTY_COPY_QUEUE_INFLIGHT (a readback whose writer has not finished is copied on
// the second queue behind that writer's tick only, not behind everything recorded since).
inline bool CopyInflightWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeCopyInflight) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "dma" (v309, with KYTY_DMA_READBACK=1): copy-queue readbacks on the copy engine's queue in the on
// windows, on the graphics family's second queue in the off windows.
inline bool DmaReadbackWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeDmaReadback) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "eager" (v310): eager readbacks of the windows the game threads keep reading back
// (KYTY_EAGER_READBACK) in the on windows only; the hot windows are noted all the time.
inline bool EagerReadbackMode() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeEagerReadback) != 0;
}
inline bool EagerReadbackWindow() {
	return EagerReadbackMode() && State().load(std::memory_order_relaxed);
}

// "dmainflight" (v314): copy-queue readbacks whose writer is still running, on the copy engine
// behind that writer's tick, when that tick has already been submitted (KYTY_DMA_INFLIGHT=1: always).
inline bool DmaInflightWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeDmaInflight) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "narrowcopy" (v315): copy-engine readback of the faulting 1 MiB block when the 4 MiB window has
// a running writer elsewhere (KYTY_NARROW_COPY=1: always).
inline bool NarrowCopyWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeNarrowCopy) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "protectcoalesce" (v316, Codex): KYTY_PROTECT_COALESCE's one-page-gap merge of predicted tracked
// write runs, on in this window.
inline bool ProtectCoalesceWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeProtectCoalesce) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// Whether this run can leave stale pages, including while its A/B window is off.
inline bool ProtectLazyReleaseConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeProtectLazyRelease) != 0;
}

// "protectlazyrelease" (v316, Codex): KYTY_LAZY_READ_RELEASE's lazy release, on in this window.
inline bool ProtectLazyReleaseWindow() {
	return ProtectLazyReleaseConfigured() && State().load(std::memory_order_relaxed);
}

// "flush4" (v317): CompleteDraw submits every 4 draws in the on windows.
inline bool Flush4Window() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeFlush4) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "fastfinish" (v318, Codex): lift the faulting clean page after its copy readback is published.
inline bool FastFinishWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeFastFinish) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "ffclean" (v445s, KYTY_FAST_FINISH_CLEANREAD): the fault handler lifts a clean page it would
// otherwise serve by emulating the load, in the on windows. It only acts while fastfinish is on
// (KYTY_FAST_FINISH=1, or "fastfinish" in the same mode list). All listed modes share the one
// on/off state: "fastfinish,ffclean" toggles both together; ffclean alone against fastfinish
// always on is KYTY_FAST_FINISH=1 with mode "ffclean".
inline bool FfCleanConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeFfClean) != 0;
}
inline bool FfCleanWindow() {
	return FfCleanConfigured() && State().load(std::memory_order_relaxed);
}

// "exactrange" (v445t, KYTY_EXACT_WRITE_RANGES): a readback whose 4 MiB window has a running
// writer is copied on the copy queue all the same when no unfinished write's own range reaches the
// faulting pages: the part of the window around them that no such range reaches (BufferCache::
// ExactReadbackWindow). The log of unfinished writes is kept in both windows of a toggled run.
inline bool ExactRangeConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeExactRange) != 0;
}
inline bool ExactRangeWindow() {
	return ExactRangeConfigured() && State().load(std::memory_order_relaxed);
}

// "bitmapclean" (v445t, KYTY_BITMAP_CLEAN_LIFT): a game thread's readback step of a page with no
// GPU-written bytes and no download in flight does not wait for the write bitmap's region tick;
// it lifts the page as the step after that wait would.
inline bool BitmapCleanConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeBitmapClean) != 0;
}
inline bool BitmapCleanWindow() {
	return BitmapCleanConfigured() && State().load(std::memory_order_relaxed);
}

// "runoverlap" (v445w): in the on windows the dispatch-run exclusion list (KYTY_DISPATCH_RUN_EXCLUDE,
// default 06544f382c6b0850 and 3378b4a64461d6ca) is ignored, so those dispatches overlap inside
// their runs like every other; off windows keep their barriers. Scene-matched A/B of the overlap.
// "variant" (v446h): in the on windows compute dispatches use the translation variant
// KYTY_TRANSLATION_VARIANT names (ShaderComputeInputInfo::translation_variant: 1 static buffer
// formats, 2 wave-uniform branches everywhere), as separate programs; off windows the default
// translation. Scene-matched A/B of translation choices (pipelineCache.cpp TranslationVariantFor).
inline bool VariantWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeVariant) != 0 &&
	       State().load(std::memory_order_relaxed);
}

inline bool RunOverlapWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeRunOverlap) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "flipnow" (v319): KYTY_FLIP_IMMEDIATE in the on windows (a ready flip is presented at once instead
// of at the next 60 Hz vblank), to measure what vblank quantization costs.
inline bool FlipNowWindow() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeFlipNow) != 0 &&
	       State().load(std::memory_order_relaxed);
}

// "gpuslow" (v321): a causal test, not an optimization. In the on windows every draw and dispatch
// packet spins KYTY_GPU_SLOW_NS (default 1500) on the thread that processes it, adding ~10 % to the
// GPU thread's work. If frame time grows by that much, the GPU thread is on the critical path and
// trimming it pays; if not, the frame is set elsewhere (sync points, game threads).
inline void GpuSlowSpin() {
	if (PeriodSeconds() == 0 || (ModeMask() & ModeGpuSlow) == 0 ||
	    !State().load(std::memory_order_relaxed)) {
		return;
	}
	static const int64_t ns = [] {
		const char* value = std::getenv("KYTY_GPU_SLOW_NS");
		return value != nullptr ? std::strtoll(value, nullptr, 10) : int64_t {1500};
	}();
	const auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
	while (std::chrono::steady_clock::now() < end) {
	}
}

// "dirtyfilter" (v322): HasGpuDirtyBytes answers from the block filter in the on windows (the filter
// is kept up to date whenever the mode is configured).
inline bool DirtyFilterConfigured() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeDirtyFilter) != 0;
}

inline bool DirtyFilterWindow() {
	return DirtyFilterConfigured() && State().load(std::memory_order_relaxed);
}

inline bool IdleAtFlip() {
	return PeriodSeconds() != 0 && (ModeMask() & ModeIdle) != 0 &&
	       State().load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::SyncToggle

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SYNCTOGGLE_H_
