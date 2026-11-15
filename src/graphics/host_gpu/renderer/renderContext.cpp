#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/timeline.h"
#include "common/fpsLog.h"

#include "common/profiler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <mutex>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

namespace Libs::Graphics {

// vulkanCommon.cpp: run on a device loss reported through RequireVulkanSuccess.
extern void (*g_device_lost_report)();
extern void (*g_device_lost_extra)(); // diagnosticCheckpoints.cpp
extern void (*g_device_fault_guest_scan)(uint64_t lo, uint64_t hi); // diagnosticCheckpoints.cpp

// image.cpp: run before an image's views are destroyed (KYTY_FAULT_MAP=1).
extern void (*g_image_views_destroying)(const Image& image);

// g_cpu_dirty_epoch when the guest submitted the submission the GPU thread is running (0: unknown).
std::atomic<uint64_t> g_bda_submission_cpu_epoch {0};

namespace {
thread_local uint32_t t_renderer_depth    = 0;
thread_local bool     t_renderer_released = false;
thread_local RenderContext::LockStats            t_lock_stats;
thread_local std::chrono::steady_clock::time_point t_released_at;
RenderContext*        g_render_context    = nullptr;
} // namespace

namespace {
// Graphics command-processor threads waiting for the renderer mutex (RenderContext::GraphicsWaiting).
std::atomic<uint32_t> g_graphics_waiting {0};
} // namespace

bool RenderContext::PoliteAsync() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_POLITE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool RenderContext::GraphicsWaiting() noexcept {
	return g_graphics_waiting.load(std::memory_order_acquire) != 0;
}

void RenderContext::LockRenderer() {
	t_lock_stats.acquires++;
	// The graphics thread (the frame's critical path) announces that it waits, so the async
	// thread hands the mutex over at its next packet instead of finishing its segment.
	const bool graphics = PoliteAsync() && t_renderer_depth == 0 && GuestGpu::IsGpuThread() &&
	                      !GuestGpu::IsAsyncThread();
	if (graphics && !m_mutex.TryLock()) {
		g_graphics_waiting.fetch_add(1, std::memory_order_acq_rel);
		const auto begin = std::chrono::steady_clock::now();
		m_mutex.Lock();
		g_graphics_waiting.fetch_sub(1, std::memory_order_acq_rel);
		t_lock_stats.contended++;
		t_lock_stats.wait_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin)
		        .count());
		t_renderer_depth++;
		return;
	}
	if (graphics) {
		t_renderer_depth++;
		return;
	}
	if (!m_mutex.TryLock()) {
		const auto begin = std::chrono::steady_clock::now();
		m_mutex.Lock();
		t_lock_stats.contended++;
		t_lock_stats.wait_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin)
		        .count());
	}
	t_renderer_depth++;
}

RenderContext::LockStats RenderContext::TakeLockStats() noexcept {
	return std::exchange(t_lock_stats, {});
}

void RenderContext::UnlockRenderer() {
	EXIT_IF(t_renderer_depth == 0);
	t_renderer_depth--;
	m_mutex.Unlock();
}

uint32_t RenderContext::ReleaseRenderer() {
	const auto depth = t_renderer_depth;
	if (depth == 0) {
		return 0;
	}
	for (uint32_t i = 0; i < depth; i++) {
		UnlockRenderer();
	}
	t_renderer_released = true;
	t_released_at       = std::chrono::steady_clock::now();
	LibKernel::Memory::SetGpuReadDelegate(true);
	return depth;
}

void RenderContext::ReacquireRenderer(uint32_t depth) {
	if (depth == 0) {
		return;
	}
	LibKernel::Memory::SetGpuReadDelegate(false);
	t_renderer_released = false;
	t_lock_stats.released_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
	                                                         t_released_at)
	        .count());
	for (uint32_t i = 0; i < depth; i++) {
		LockRenderer();
	}
}

RenderContext::UnlockedScope::UnlockedScope(RenderContext& context)
    : m_context(context), m_depth(context.ReleaseRenderer()) {}

RenderContext::UnlockedScope::~UnlockedScope() {
	if (m_depth != 0) {
		m_context.ReacquireRenderer(m_depth);
		GuestGpu::RebindCurrentProcessor();
	}
}

bool RenderContext::RendererReleased() noexcept {
	return t_renderer_released;
}

bool RenderContext::HoldsRenderer() noexcept {
	return t_renderer_depth != 0;
}

void RenderContext::MarkOffLock(bool off) {
	EXIT_IF(t_renderer_depth != 0);
	if (off) {
		t_released_at = std::chrono::steady_clock::now();
	} else if (t_renderer_released) {
		t_lock_stats.released_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
		                                                         t_released_at)
		        .count());
	}
	t_renderer_released = off;
	LibKernel::Memory::SetGpuReadDelegate(off);
}

RenderContext::ReacquireScope::ReacquireScope(RenderContext& context) {
	if (t_renderer_released) {
		m_context = &context;
		context.ReacquireRenderer(1);
	}
}

RenderContext::ReacquireScope::ReacquireScope() {
	if (t_renderer_released && g_render_context != nullptr) {
		m_context = g_render_context;
		m_context->ReacquireRenderer(1);
	}
}

RenderContext::ReacquireScope::~ReacquireScope() {
	if (m_context != nullptr) {
		m_context->UnlockRenderer();
		t_renderer_released = true;
		LibKernel::Memory::SetGpuReadDelegate(true);
	}
}

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler, m_page_manager) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	g_render_context = this;
	m_texture_cache.on_bindless_unregister = [this](ImageId id) {
		m_bindless_table.OnImageUnregistered(id);
	};
	m_buffer_cache.on_cpu_write_range = [this](uint64_t vaddr, uint64_t size) {
		if (!IsMapped(vaddr, size)) {
			return;
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.InvalidateMemory(vaddr, size);
		m_bindless_table.OnCpuWrite(vaddr, size);
	};
	g_device_fault_guest_scan = [](uint64_t lo, uint64_t hi) {
		if (g_render_context != nullptr && g_render_context->m_bindless_table.Enabled()) {
			g_render_context->m_bindless_table.DescribeKeysForGuestRange(lo, hi, stderr);
		}
	};
	g_device_lost_extra = [] {
		if (g_render_context != nullptr) {
			g_render_context->m_buffer_cache.TakeLoopGuardReport(true);
		}
		if (g_render_context != nullptr && g_render_context->m_bindless_table.Enabled() &&
		    [] { const char* v = std::getenv("KYTY_FAULT_MAP"); return v != nullptr && v[0] == '1'; }()) {
			g_render_context->m_bindless_table.AuditForFault();
			g_render_context->m_bindless_table.ReportSweep(0);
		}
	};
	g_device_lost_report = [] {
		if (g_render_context != nullptr) {
			DumpDeviceLossDiagnostics(g_render_context->m_graphics);
		}
	};
	g_image_views_destroying = [](const Image& image) {
		if (g_render_context == nullptr || !g_render_context->m_bindless_table.Enabled()) {
			return;
		}
		std::vector<vk::ImageView> views;
		for (const auto& cached: image.views) {
			views.push_back(cached.view);
		}
		char what[128];
		std::snprintf(what, sizeof(what), "a destroyed %ux%u image (fmt %d, guest 0x%llx)",
		              image.info.extent.width, image.info.extent.height,
		              static_cast<int>(image.info.pixel_format),
		              static_cast<unsigned long long>(image.info.data.address));
		g_render_context->m_bindless_table.ReportHeldViews(views, what);
	};
	m_texture_cache.on_image_release = [this](const Image& image) {
		if (!m_bindless_table.Enabled()) {
			return false;
		}
		std::vector<vk::ImageView> views;
		views.reserve(image.views.size());
		for (const auto& cached: image.views) {
			views.push_back(cached.view);
		}
		return m_bindless_table.ReleaseViews(views);
	};
	// vkQueueSubmit, and the queue lock that present also holds, runs on a queue thread instead
	// of the GPU thread. Frames reach the presenter only after their flip tick completed, and the
	// presenter's own scheduler stays synchronous. KYTY_ASYNC_SUBMIT=0 submits on the GPU thread.
	// KYTY_RECORD_THREAD=0: no recording thread (CommandScheduler::EnableRecordingThread), which
	// submits too.
	const char* record_thread = std::getenv("KYTY_RECORD_THREAD");
	const char* async_submit  = std::getenv("KYTY_ASYNC_SUBMIT");
	if (record_thread == nullptr || std::strcmp(record_thread, "0") != 0) {
		m_command_scheduler.EnableRecordingThread();
	} else if (async_submit == nullptr || std::strcmp(async_submit, "0") != 0) {
		m_command_scheduler.EnableAsyncSubmit();
	}
	LOGF("Recording thread: %s, async queue submit: %s\n",
	     m_command_scheduler.Threaded() ? "on" : "off",
	     m_command_scheduler.AsyncSubmit() ? "on" : "off");
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void WarmUpLocalMemory(GraphicContext& graphics, CommandScheduler& scheduler);

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	WarmUpLocalMemory(m_graphics, m_command_scheduler);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			(KYTY_SUBMIT_REASON(), m_command_scheduler.Finish());
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

namespace {
// KYTY_FAULT_GRANULE_KB (default 0 = one page): a power of two of at least 4 KB. Read once at
// start-up: HandleFault runs in the fault handler. Port of Senaxx/KytyPS5 PR #17 (aea48484).
// 64 KB: env3 18.2/18.9 -> 19.2/19.1 fps, fight 23.5 -> 24.3, grass 25.7 -> 26.1, but loads froze
// with it (the game stopped presenting, GPU at 100%) in most of the player's sessions, 2026-10-20.
uint64_t FaultGranuleFromEnv() {
	const char*    value = std::getenv("KYTY_FAULT_GRANULE_KB");
	const uint64_t bytes = (value != nullptr ? std::strtoull(value, nullptr, 10) : 0) << 10u;
	return bytes >= 4096 && (bytes & (bytes - 1)) == 0 ? bytes : 0;
}
const uint64_t g_fault_granule = FaultGranuleFromEnv();
} // namespace

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	DrainProtectionHelper(ProtectDrainReason::Fault);
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// A command-processor thread in an off-lock window handles its fault as the GPU thread does,
	// which needs the renderer.
	const ReacquireScope reacquire(*this);
	if (access == PageFaultAccess::Write) {
		// Diagnostics: KYTY_TRACE_WRITERS=<hex address>:<hex size>: guest CPU write faults in that
		// range ([cpu-write], first 200), with whether the page holds GPU-written bytes.
		static const std::pair<uint64_t, uint64_t> trace = [] {
			std::pair<uint64_t, uint64_t> range {0, 0};
			if (const char* value = std::getenv("KYTY_TRACE_WRITERS"); value != nullptr) {
				char* end    = nullptr;
				range.first  = std::strtoull(value, &end, 16);
				range.second = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 4u;
			}
			return range;
		}();
		static std::atomic<uint32_t> traced {0};
		if (trace.first != 0 && fault_vaddr >= trace.first && fault_vaddr < trace.first + trace.second &&
		    traced.fetch_add(1) < 200u) {
			std::fprintf(stderr, "[cpu-write] 0x%012llx frame=%llu gpu_dirty_page=%d\n",
			             static_cast<unsigned long long>(fault_vaddr),
			             static_cast<unsigned long long>(GetGraphics().presented_frames.load()),
			             m_buffer_cache.HasGpuDirtyBytes(fault_vaddr & ~uint64_t {0xfff}, 0x1000) ? 1 : 0);
		}
		// KYTY_FAULT_GRANULE_KB: the buffer cache takes the whole aligned granule around the written
		// byte as CPU-written, not just its page, so a guest filling a buffer page by page faults
		// (and changes protection) once per granule instead of once per page. A granule holding
		// GPU-written bytes keeps the single page, so nothing is downloaded for it. Images and the
		// bindless table still take only the page.
		uint64_t vaddr = fault_vaddr;
		uint64_t size  = fault_size;
		if (g_fault_granule != 0) {
			const uint64_t base = fault_vaddr & ~(g_fault_granule - 1);
			if (IsMapped(base, g_fault_granule) &&
			    !m_buffer_cache.IsRegionGpuModified(base, g_fault_granule)) {
				vaddr = base;
				size  = g_fault_granule;
			}
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
	} else {
		KYTY_PROFILER_BLOCK("ReadMemory: read fault");
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) const noexcept {
	static const bool stats = std::getenv("KYTY_READBACK_STATS") != nullptr;
	if (!stats) {
		return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
		       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
	}
	// Which check declines a clean-read emulation (KYTY_READBACK_STATS): [clean-read-check].
	static std::atomic<uint64_t> unmapped {0}, unwatched {0}, dirty {0}, served {0}, next {4096};
	bool ok = false;
	if (!IsMapped(vaddr, size)) {
		unmapped.fetch_add(1, std::memory_order_relaxed);
	} else if (!m_page_manager.IsReadWatched(fault_vaddr)) {
		unwatched.fetch_add(1, std::memory_order_relaxed);
	} else if (!m_buffer_cache.IsCleanForConcurrentRead(vaddr, size)) {
		dirty.fetch_add(1, std::memory_order_relaxed);
	} else {
		served.fetch_add(1, std::memory_order_relaxed);
		ok = true;
	}
	const auto total = unmapped.load() + unwatched.load() + dirty.load() + served.load();
	auto       due   = next.load(std::memory_order_relaxed);
	if (total >= due && next.compare_exchange_strong(due, due * 2)) {
		std::fprintf(stderr,
		             "[clean-read-check] served=%llu gpu-dirty=%llu not-read-watched=%llu unmapped=%llu\n",
		             static_cast<unsigned long long>(served.load()),
		             static_cast<unsigned long long>(dirty.load()),
		             static_cast<unsigned long long>(unwatched.load()),
		             static_cast<unsigned long long>(unmapped.load()));
	}
	return ok;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_bindless_table.OnCpuWrite(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

namespace {
// F4.1 (KYTY_UNMAP_DRAIN=1 restores the old behaviour): a guest unmap used to drain the whole GPU
// (CommandScheduler::Finish), 28 times a second in the Performance save, ~1.2 ms of waiting each
// plus the bubble after it. Only one thing an unmap does needs that: dropping a shared buffer,
// whose Vulkan import points into the guest memory being unmapped and is freed at once
// (BufferCache::UnmapShared). Everything else is ordered by ticks already: images and buffers
// over the range are released with DeferRelease after the GPU finished the ticks that used them,
// GPU-written bytes are downloaded by InvalidateMemory -> ReadMemory, which waits for their own
// tick, the BDA page table is cleared in command order, and downloads still in flight into the
// range are waited for (WaitDownloadsLanded, their ticks only) before the range goes.
bool UnmapDrainAlways() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_UNMAP_DRAIN");
		return value != nullptr && value[0] == '1';
	}();
	return always;
}
struct UnmapDrainStats {
	std::atomic<uint64_t> drained {0}, skipped {0};
};
UnmapDrainStats g_unmap_drain_stats;
void ReportUnmapDrains() {
	static const bool stats = Common::FpsLogDiagnostics();
	if (!stats) {
		return;
	}
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	const double s = std::chrono::duration<double>(now - last).count();
	last           = now;
	std::fprintf(stderr, "[unmap-drain] %.1f/s drained (shared buffer in range), %.1f/s without a drain\n",
	             static_cast<double>(g_unmap_drain_stats.drained.exchange(0)) / s,
	             static_cast<double>(g_unmap_drain_stats.skipped.exchange(0)) / s);
}
} // namespace

void RenderContext::DrainForUnmap(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	if (!m_command_scheduler.Active()) {
		return;
	}
	bool drain = UnmapDrainAlways();
	for (size_t i = 0; !drain && i < ranges.size(); i++) {
		drain = m_buffer_cache.HasSharedBuffersIn(ranges[i].first, ranges[i].second);
	}
	if (drain) {
		g_unmap_drain_stats.drained.fetch_add(1, std::memory_order_relaxed);
		const auto tick = m_command_scheduler.CurrentTick();
		(KYTY_SUBMIT_REASON(), m_command_scheduler.Finish());
		m_command_scheduler.WaitPriorityOperations(tick);
	} else {
		g_unmap_drain_stats.skipped.fetch_add(1, std::memory_order_relaxed);
		for (const auto& [vaddr, size]: ranges) {
			m_buffer_cache.WaitDownloadsLanded(vaddr, size);
		}
	}
	ReportUnmapDrains();
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	m_page_manager.ForgetProtectionFaults(vaddr, size);
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// The kernel unmaps every free range it is about to (re)map, most of which were never GPU
	// mapped. Buffers, images and GPU-dirty pages are only tracked inside mapped ranges (faults
	// and invalidations outside them are ignored), so such an unmap has nothing to invalidate:
	// skip the GPU-thread round trip and the full drain it forces. KYTY_UNMAP_SKIP_UNMAPPED=0
	// sends every unmap to the GPU thread again.
	static const bool skip_unmapped = [] {
		const auto* value = std::getenv("KYTY_UNMAP_SKIP_UNMAPPED");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (skip_unmapped && GuestRange {vaddr, size}.Valid()) {
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	const auto unmap = [this, vaddr, size] {
		m_page_manager.ForgetStale(vaddr, size);
		const std::pair<uint64_t, uint64_t> range {vaddr, size};
		DrainForUnmap({&range, 1});
		m_buffer_cache.UnmapShared(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_bindless_table.UnwatchRange(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	// KYTY_FPS_LOG: time the GPU-thread round trips of unmaps (each drains the GPU) every 5 s.
	static const bool stats = Common::FpsLogDiagnostics();
	const auto        begin = std::chrono::steady_clock::now();
	m_gpu->SendCommandSync(unmap);
	if (stats) {
		static std::mutex mutex;
		static double     total_ms = 0.0, max_ms = 0.0;
		static uint64_t   count    = 0;
		static auto       last     = std::chrono::steady_clock::now();
		const auto        now      = std::chrono::steady_clock::now();
		const double      ms = std::chrono::duration<double, std::milli>(now - begin).count();
		std::lock_guard   lock(mutex);
		total_ms += ms;
		max_ms = std::max(max_ms, ms);
		count++;
		if (now - last > std::chrono::seconds(5)) {
			std::fprintf(stderr, "[unmap-sync] %llu GPU-thread unmaps in %.1f s: %.1f ms total, max %.1f ms\n",
			             static_cast<unsigned long long>(count),
			             std::chrono::duration<double>(now - last).count(), total_ms, max_ms);
			total_ms = max_ms = 0.0;
			count             = 0;
			last              = now;
		}
	}
}

void RenderContext::UnmapMemoryRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion\n");
	}
	std::vector<std::pair<uint64_t, uint64_t>> mapped;
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		for (const auto& [vaddr, size]: ranges) {
			if (GuestRange {vaddr, size}.Valid() && m_mapped_ranges.Intersects(vaddr, size)) {
				mapped.emplace_back(vaddr, size);
			}
		}
	}
	if (mapped.empty()) {
		return;
	}
	const auto unmap = [this, &mapped] {
		DrainForUnmap(mapped);
		for (const auto& [vaddr, size]: mapped) {
			m_page_manager.ForgetStale(vaddr, size);
			m_buffer_cache.UnmapShared(vaddr, size);
			m_buffer_cache.InvalidateMemory(vaddr, size);
			m_texture_cache.UnmapMemory(vaddr, size);
			m_bindless_table.UnwatchRange(vaddr, size);
		}
		std::lock_guard lock(m_mapped_ranges_mutex);
		for (const auto& [vaddr, size]: mapped) {
			m_mapped_ranges.Subtract(vaddr, size);
		}
	};
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	const CpuCost::Phase cpu_phase_("PrepareBda");
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	// Every cached buffer is synchronized so a global-memory shader sees the CPU's writes.
	// That walk touched every buffer per dispatch (a quarter of the GPU thread in the
	// world); it only has to repeat once something became CPU-dirty or a buffer appeared.
	// The walk runs at most once per guest submission: what the game wrote before submitting is
	// visible to all its commands, as on the console, and the GPU thread runs behind the game
	// anyway. Per draw, the walk and the re-protection it triggers took ~38 % of the GPU thread
	// on the title menu (21 -> 32 fps).
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	if (epoch != m_bda_synced_epoch) {
		// The running submission only reads what the guest wrote before submitting it. The last
		// walk started at m_bda_synced_epoch: when the guest submitted no later than that, every
		// page it had dirtied by then was synced (and one written again since faulted, which
		// moved the epoch past the walk's). The GPU thread runs behind the game, so most
		// submissions were made before the previous walk (KYTY_BDA_SUBMIT_EPOCH=0: walk anyway).
		static const bool submit_epoch = [] {
			const char* value = std::getenv("KYTY_BDA_SUBMIT_EPOCH");
			return value == nullptr || value[0] != '0';
		}();
		// KYTY_BDA_EVERY=1: walk whenever something became CPU-dirty, not once per submission. A
		// submission can wait on a label and then read what the CPU wrote after it; per-binding
		// syncs see those writes, buffers read through the table (KYTY_GPU_RESOURCES) do not.
		static const bool every = [] {
			const char* value = std::getenv("KYTY_BDA_EVERY");
			return value != nullptr && value[0] == '1';
		}();
		// G6 (KYTY_G6): the table's buffers get no sync per binding, so a WAIT_REG_MEM passed or a
		// WRITE_DATA since the last walk walks again (g_bda_resync_seq): the CPU may have written
		// data before the label the submission waited on, after it was submitted, and WRITE_DATA
		// writes on the CPU in command order (a per-binding sync, made when the GPU thread reaches
		// the binding, sees both). KYTY_G6_WAIT_SYNC=0: once per submission only.
		static const bool wait_sync = [] {
			const char* value = std::getenv("KYTY_G6_WAIT_SYNC");
			return G6Compute() && (value == nullptr || value[0] != '0');
		}();
		const auto waits       = g_bda_resync_seq.load(std::memory_order_relaxed);
		const bool wait_passed = wait_sync && waits != m_bda_synced_waits;
		const auto needed      = g_bda_submission_cpu_epoch.load(std::memory_order_relaxed);
		if (!every && !wait_passed && submit_epoch && needed != 0 && m_bda_synced_epoch != 0 &&
		    needed <= m_bda_synced_epoch) {
			m_fault_process_pending = true;
			return;
		}
		const auto submission = g_guest_submission_seq.load(std::memory_order_relaxed);
		if (!every && !wait_passed && submission == m_bda_synced_submission) {
			m_fault_process_pending = true;
			return;
		}
		m_bda_synced_submission = submission;
		m_bda_synced_waits      = waits;
	}
	if (epoch != m_bda_synced_epoch) {
		// The guest writes somewhere nearly all the time, so the epoch moves between most
		// dispatches; walk only the regions holding CPU-dirty pages, not every buffer.
		KYTY_PROFILER_BLOCK("PrepareBda::Walk");
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (BufferCache::BdaTouchedEnabled()) {
			m_buffer_cache.SynchronizeTouchedMappedBuffers(m_mapped_ranges);
		} else {
			m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
				m_buffer_cache.SynchronizeCpuDirtyBuffersInRange(start, end - start);
			});
		}
		m_bda_synced_epoch = epoch;
	}
	m_fault_process_pending = true;
}

// Diagnostics: KYTY_DUMP_GUEST=<frame>:<address>+<size>[,<address>+<size>...] (frame decimal,
// the rest hex) writes guest memory, with the GPU's writes downloaded first, to
// dump_<frame>_<address>.bin once that many frames have been presented. A frame written +<n>
// counts from the first use of the KYTY_WATCH_SHADER shader.
static void DumpGuestMemory(GraphicContext& graphics, BufferCache& cache) {
	struct Request {
		uint64_t                                   frame    = 0;
		bool                                       relative = false;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	static const Request request = [] {
		Request     result;
		const char* value = std::getenv("KYTY_DUMP_GUEST");
		if (value == nullptr) {
			return result;
		}
		char* end       = nullptr;
		result.relative = *value == '+';
		result.frame    = std::strtoull(result.relative ? value + 1 : value, &end, 10);
		value        = end != nullptr && *end == ':' ? end + 1 : nullptr;
		while (value != nullptr && *value != '\0') {
			const auto address = std::strtoull(value, &end, 16);
			uint64_t   size    = 4;
			if (end != nullptr && *end == '+') {
				size = std::strtoull(end + 1, &end, 16);
			}
			result.ranges.emplace_back(address, size);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static bool done = false;
	if (done || request.ranges.empty()) {
		return;
	}
	const auto first = WatchedShaderFirstFrame();
	if (request.relative && first == UINT64_MAX) {
		return;
	}
	const auto frame = request.relative ? first + request.frame : request.frame;
	if (graphics.presented_frames.load(std::memory_order_relaxed) < frame) {
		return;
	}
	done = true;
	for (const auto& [address, size]: request.ranges) {
		cache.DownloadRangeForDiagnostics(address, size);
		std::vector<uint8_t> bytes(size);
		const bool read = LibKernel::Memory::TryReadBacking(address, bytes.data(), size);
		const auto name = "dump_" + std::to_string(frame) + "_" +
		                  std::to_string(address) + ".bin";
		if (read) {
			if (auto* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
				std::fwrite(bytes.data(), 1, bytes.size(), file);
				std::fclose(file);
			}
		}
		LOGF("DumpGuest: 0x%016" PRIx64 "+0x%" PRIx64 " -> %s (%s)\n", address, size, name.c_str(),
		     read ? "written" : "not readable");
	}
}

void RenderContext::RunGarbageCollector() {
	DumpGuestMemory(GetGraphics(), m_buffer_cache);
	LogWatchedRereads(m_buffer_cache, GetGraphics().presented_frames.load(std::memory_order_relaxed));
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	// v386 [mem-census] every 10 s (KYTY_MEM_CENSUS=0 turns it off): who holds the video memory, to
	// find what fills the card over a long session (fps fell to ~6 at ~20 min with 22 GB of 24 in use).
	static const bool census = [] {
		const char* value = std::getenv("KYTY_MEM_CENSUS");
		return value == nullptr || value[0] != '0';
	}();
	static const auto start = std::chrono::steady_clock::now();
	static auto       last  = start;
	if (const auto now = std::chrono::steady_clock::now(); census && now - last >= std::chrono::seconds(10)) {
		last            = now;
		auto& graphics  = GetGraphics();
		const auto gc   = m_texture_cache.GetGcTotals();
		std::fprintf(stderr,
		             "[mem-census] t=%.0f device %" PRIu64 "/%" PRIu64 " MB short %" PRIu64 " MB | textures %" PRIu64
		             " MB (pinned %" PRIu64 ", %zu images, pressure at %" PRIu64 ", evictable %zu; gc freed %" PRIu64
		             " / %" PRIu64 " MB, bindless %" PRIu64 ", written back %" PRIu64 ", trim runs %" PRIu64 " (backoffs %" PRIu64 ", GPU-written left %" PRIu64 ")"
		             ") | buffers %" PRIu64 " MB (%zu) |%s\n",
		             std::chrono::duration<double>(now - start).count(), graphics.GetDeviceMemoryUsage() >> 20u,
		             graphics.GetTotalMemoryBudget() >> 20u, graphics.MemoryShortfall() >> 20u,
		             m_texture_cache.TotalUsedMemory() >> 20u, m_texture_cache.PinnedMemory() >> 20u,
		             m_texture_cache.ImageCount(), m_texture_cache.PressureThreshold() >> 20u,
		             m_texture_cache.EvictableCount(), gc.freed, gc.freed_mb, gc.bindless, gc.written_back, gc.trim_runs, gc.trim_backoffs, gc.trim_gpu_skipped,
		             m_buffer_cache.TotalUsedMemory() >> 20u, m_buffer_cache.BufferCount(),
		             graphics.DescribeDeviceMemory().c_str());
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
