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
// "fastfinish": game-thread readbacks lift a copied clean page without a second GPU-thread step;
// "helper": GPU-thread GPU-dirty/gpu-clean page protections run on the protection helper.
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

enum Mode : uint32_t {
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
	ModeBarrierMerge   = 2147483648u
};

// KYTY_SYNC_TOGGLE_MODE: one name, or several separated by commas (v304), toggled together.
inline uint32_t ModeMask() {
	static const uint32_t mask = [] {
		const char* value = std::getenv("KYTY_SYNC_TOGGLE_MODE");
		if (value == nullptr || std::strcmp(value, "all") == 0) return ModeLabels | ModeIdle;
		const auto one = [](std::string_view name) -> uint32_t {
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
			if (name == "protectdefer") return ModeProtectDefer;
			return 0;
		};
		uint32_t         result = 0;
		std::string_view list(value);
		while (!list.empty()) {
			const auto comma = list.find(',');
			result |= one(list.substr(0, comma));
			if (comma == std::string_view::npos) {
				break;
			}
			list.remove_prefix(comma + 1);
		}
		return result != 0 ? result : static_cast<uint32_t>(ModeLabels | ModeIdle);
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
