#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/timeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics {

void DumpHangWork(uint64_t first_incomplete, uint64_t waited); // diagnosticCheckpoints.cpp

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

// When the GPU tick last advanced (steady clock, ns), for device-fault reports: the GPU stops at
// the fault, seconds before the device loss surfaces.
std::atomic<int64_t>  g_gpu_progress_ns {0};
std::atomic<uint64_t> g_gpu_progress_tick {0};

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	// v438zx: after a device loss the counter reads UINT64_MAX (AMD 11-17: the report then named
	// the last work sent, not the work the GPU stopped in). Only a tick that was handed out
	// counts as progress, so the report keeps the last tick the GPU really finished.
	if (known < counter && counter < CurrentTick() &&
	    counter > g_gpu_progress_tick.load(std::memory_order_relaxed)) {
		g_gpu_progress_tick.store(counter, std::memory_order_relaxed);
		g_gpu_progress_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
		                        std::memory_order_relaxed);
	}
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

// Every blocking wait on the host GPU goes through here. Per-thread totals every 256 waits,
// and a host stack for the slow ones, name the paths that serialize the CPU on the GPU.
namespace {
struct SemaphoreWaitStats {
	uint64_t count   = 0;
	double   total_s = 0.0;
	double   max_s   = 0.0;
};
thread_local SemaphoreWaitStats g_semaphore_wait_stats;
std::atomic<uint32_t>           g_slow_wait_traces = 0;

void RecordSemaphoreWait(double seconds, uint64_t tick, uint64_t gpu_tick) {
	auto& stats = g_semaphore_wait_stats;
	stats.count++;
	stats.total_s += seconds;
	stats.max_s = std::max(stats.max_s, seconds);
	if (seconds >= 0.020 && g_slow_wait_traces.fetch_add(1) < 24) {
		LOGF("SemaphoreWait slow: %.1fms tick=%" PRIu64 " gpu_tick=%" PRIu64 " thread=%d\n%s",
		     seconds * 1000.0, tick, gpu_tick, Common::Thread::GetThreadIdUnique(),
		     Common::HostBacktrace().c_str());
	}
	if (stats.count % 256 == 0) {
		LOGF("SemaphoreWaits: thread=%d count=%" PRIu64 " total=%.1fms max=%.1fms\n",
		     Common::Thread::GetThreadIdUnique(), stats.count, stats.total_s * 1000.0,
		     stats.max_s * 1000.0);
		stats = {};
	}
}
} // namespace

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	// Profiling-only zone: this is the actual CPU-blocks-on-GPU stall. Everything above this
	// point is a fast non-blocking check; only reaching here means the GPU genuinely hasn't
	// caught up yet.
	KYTY_PROFILER_BLOCK("MasterSemaphore::Wait (blocked on GPU)");

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::Timer timer;
	timer.Start();
	// A GPU hang (work that never finishes, the GPU busy) blocks here: after 5 s, and every 20 s
	// after that, the GPU's last checkpoints name the draw or dispatch it is stuck in ([gpu-hang]).
	auto result = m_graphics.device.waitSemaphores(&wait_info, 5'000'000'000ull);
	for (uint32_t reports = 0; result == vk::Result::eTimeout; reports++) {
		if (reports < 8) {
			DumpGpuHangCheckpoints(m_graphics, tick, timer.GetTimeS());
			uint64_t done = 0;
			if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &done) == vk::Result::eSuccess) {
				DumpHangWork(done + 1, tick);
			}
		}
		result = m_graphics.device.waitSemaphores(&wait_info, 20'000'000'000ull);
	}
	const double waited = timer.GetTimeS();
	RecordSemaphoreWait(waited, tick, m_gpu_tick.load(std::memory_order_acquire));
	if (Profiler::t_gpu_thread) {
		Profiler::Add(Profiler::Counter::GpuWaits);
		Profiler::Add(Profiler::Counter::GpuWaitTime, static_cast<int64_t>(waited * 1e6));
	}
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(m_graphics);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	Refresh();
}

bool MasterSemaphore::WaitFor(uint64_t tick, uint64_t timeout_ns) {
	const CpuCost::Phase cpu_phase_("MasterSemaphore::WaitFor");
	if (IsFree(tick)) {
		return true;
	}
	Refresh();
	if (IsFree(tick)) {
		return true;
	}
	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;
	const auto result        = m_graphics.device.waitSemaphores(&wait_info, timeout_ns);
	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(m_graphics);
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: device lost\n", tick);
	}
	Refresh();
	return IsFree(tick);
}

} // namespace Libs::Graphics
