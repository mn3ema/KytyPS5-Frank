#include "common/sampler.h"
#include "common/floorStats.h"
#include "common/fpsLog.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/renderer/barrierMerge.h"
#include "graphics/host_gpu/pageManager.h"

#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/timeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

namespace Libs::Graphics {

// bufferCache.cpp (KYTY_UPLOAD_WORKER).
uint64_t UploadWorkerPushed();
void     WaitUploadWorker(uint64_t upto);

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

// KYTY_QUEUE_STATS, [op-latency] (v445m, counters only): how long after the host GPU finished a
// tick the operations deferred to that tick ran. A thread of its own waits on the timeline
// semaphore for each next value and notes the time it wakes: that is when the completion became
// observable. It reads the semaphore only (no Refresh, no scheduler state), so nothing else
// changes. Classes: the GPU thread's pending operations (PopPendingOperations, polled per draw
// and dispatch and throttled by KYTY_POP_POLL_US), the priority thread's operations (interrupts,
// flips, readback publications), and deferred end-of-pipe label writes (labels at completion:
// the priority thread hands the write to the GPU thread).
struct OpLatency {
	static constexpr uint64_t Slots = 8192;
	enum Class : uint32_t { Pending, Priority, Label, ClassCount };
	struct Stat {
		std::atomic<uint64_t> count {0};  // with a completion time
		std::atomic<uint64_t> early {0};  // ran before the watcher saw the tick (latency ~0)
		std::atomic<uint64_t> sum_ns {0};
		std::atomic<uint64_t> max_ns {0};
		std::atomic<uint64_t> buckets[5] {}; // <50 us, <200 us, <1 ms, <5 ms, >=5 ms
	};
	std::atomic<int64_t>  done_ns[Slots] {};
	std::atomic<uint64_t> seen {0};
	std::atomic<uint64_t> ticks {0}; // completions seen in the window
	Stat                  stats[ClassCount];
	std::atomic<int64_t>  reported_ms {0};
	std::atomic<bool>     stop {false};
	std::atomic<bool>     running {false};
	std::atomic<const void*> owner {nullptr};
	std::thread           thread;
};

bool OpLatencyEnabled() {
	static const bool enabled = std::getenv("KYTY_QUEUE_STATS") != nullptr;
	return enabled;
}

// Never destroyed: a watcher still running at exit must not meet its std::thread's destructor.
OpLatency& GetOpLatency() {
	static OpLatency* const state = new OpLatency;
	return *state;
}

thread_local int64_t t_operation_done_ns = 0;

int64_t SteadyNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void OpLatencyWatch(vk::Device device, vk::Semaphore semaphore) {
	Common::Sampler::RegisterCurrentThread("tick-watch");
	auto&    state = GetOpLatency();
	uint64_t seen  = 0;
	if (device.getSemaphoreCounterValue(semaphore, &seen) != vk::Result::eSuccess) {
		state.running.store(false, std::memory_order_release);
		return;
	}
	state.seen.store(seen, std::memory_order_release);
	while (!state.stop.load(std::memory_order_acquire)) {
		uint64_t              target = seen + 1;
		vk::SemaphoreWaitInfo wait_info {};
		wait_info.semaphoreCount = 1;
		wait_info.pSemaphores    = &semaphore;
		wait_info.pValues        = &target;
		const auto result        = device.waitSemaphores(&wait_info, 50'000'000ull);
		if (result == vk::Result::eTimeout) {
			continue;
		}
		if (result != vk::Result::eSuccess) {
			break; // a lost device: the scheduler's own waits report it
		}
		const auto now   = SteadyNs();
		uint64_t   value = target;
		if (device.getSemaphoreCounterValue(semaphore, &value) != vk::Result::eSuccess || value < target) {
			value = target;
		}
		const auto first = value - seen > OpLatency::Slots ? value - OpLatency::Slots + 1 : seen + 1;
		for (uint64_t tick = first; tick <= value; tick++) {
			state.done_ns[tick & (OpLatency::Slots - 1)].store(now, std::memory_order_relaxed);
		}
		state.ticks.fetch_add(value - seen, std::memory_order_relaxed);
		seen = value;
		state.seen.store(seen, std::memory_order_release);
	}
	state.running.store(false, std::memory_order_release);
}

// The time the watcher saw `tick` done; 0 when it has not (yet), or no longer knows.
int64_t OpLatencyDoneNs(uint64_t tick) {
	auto&      state = GetOpLatency();
	const auto seen  = state.seen.load(std::memory_order_acquire);
	if (tick == 0 || tick > seen || seen - tick >= OpLatency::Slots - 64) {
		return 0;
	}
	return state.done_ns[tick & (OpLatency::Slots - 1)].load(std::memory_order_relaxed);
}

void OpLatencyReport(bool force);

void OpLatencyAdd(OpLatency::Class kind, int64_t done_ns) {
	auto& stat = GetOpLatency().stats[kind];
	if (done_ns == 0) {
		stat.early.fetch_add(1, std::memory_order_relaxed);
	} else {
		const auto now = SteadyNs();
		const auto ns  = static_cast<uint64_t>(now > done_ns ? now - done_ns : 0);
		stat.count.fetch_add(1, std::memory_order_relaxed);
		stat.sum_ns.fetch_add(ns, std::memory_order_relaxed);
		auto max = stat.max_ns.load(std::memory_order_relaxed);
		while (ns > max && !stat.max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
		}
		stat.buckets[ns < 50'000 ? 0 : ns < 200'000 ? 1 : ns < 1'000'000 ? 2 : ns < 5'000'000 ? 3 : 4].fetch_add(
		    1, std::memory_order_relaxed);
	}
	OpLatencyReport(false);
}

// Printed with [pop-poll] (force), or by itself when that line has not come for 7 s.
void OpLatencyReport(bool force) {
	auto&      state = GetOpLatency();
	const auto ms    = SteadyNs() / 1'000'000;
	auto       last  = state.reported_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		state.reported_ms.compare_exchange_strong(last, ms);
		return;
	}
	if (ms - last < (force ? 1000 : 7000) || !state.reported_ms.compare_exchange_strong(last, ms)) {
		return;
	}
	const double s = static_cast<double>(ms - last) / 1000.0;
	static constexpr const char* names[OpLatency::ClassCount] = {"pending ops (GPU thread)", "priority ops",
	                                                             "deferred label writes"};
	std::string text = "[op-latency] " + std::to_string(static_cast<int>(s * 10.0 + 0.5) / 10) + " s, GPU ticks seen " +
	                   std::to_string(state.ticks.exchange(0)) + (state.running.load() ? "" : " (NO WATCHER)") +
	                   "; from a tick's completion being observable to its operation running:";
	for (uint32_t kind = 0; kind < OpLatency::ClassCount; kind++) {
		auto&      stat  = state.stats[kind];
		const auto count = stat.count.exchange(0);
		const auto early = stat.early.exchange(0);
		const auto sum   = stat.sum_ns.exchange(0);
		const auto max   = stat.max_ns.exchange(0);
		uint64_t   b[5];
		for (uint32_t i = 0; i < 5; i++) {
			b[i] = stat.buckets[i].exchange(0);
		}
		char line[320];
		std::snprintf(line, sizeof(line),
		              "%s %s: %.0f/s (+%.0f/s run before the watcher woke), avg %.0f us, max %.0f us, total %.2f ms/s, "
		              "<50us %llu, <200us %llu, <1ms %llu, <5ms %llu, >=5ms %llu",
		              kind == 0 ? "" : " |", names[kind], static_cast<double>(count) / s, static_cast<double>(early) / s,
		              count != 0 ? static_cast<double>(sum) / 1e3 / static_cast<double>(count) : 0.0,
		              static_cast<double>(max) / 1e3, static_cast<double>(sum) / 1e6 / s,
		              static_cast<unsigned long long>(b[0]), static_cast<unsigned long long>(b[1]),
		              static_cast<unsigned long long>(b[2]), static_cast<unsigned long long>(b[3]),
		              static_cast<unsigned long long>(b[4]));
		text += line;
	}
	text += "\n";
	std::fputs(text.c_str(), stderr);
}

void OpLatencyStart(const void* owner, vk::Device device, vk::Semaphore semaphore) {
	if (!OpLatencyEnabled()) {
		return;
	}
	auto&       state    = GetOpLatency();
	const void* expected = nullptr;
	if (!state.owner.compare_exchange_strong(expected, owner)) {
		return; // one scheduler only
	}
	state.stop.store(false, std::memory_order_release);
	state.running.store(true, std::memory_order_release);
	state.thread = std::thread([device, semaphore] { OpLatencyWatch(device, semaphore); });
}

void OpLatencyStop(const void* owner) {
	if (!OpLatencyEnabled()) {
		return;
	}
	auto& state = GetOpLatency();
	if (state.owner.load(std::memory_order_acquire) != owner) {
		return;
	}
	state.stop.store(true, std::memory_order_release);
	if (state.thread.joinable()) {
		state.thread.join();
	}
}

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

// KYTY_DRAW_FLUSH_INTERVAL=N overrides CompleteDraw()'s periodic non-blocking flush interval.
// Defaults to 16 -- validated against real gameplay, where it cut the fraction of the main thread
// spent in MasterSemaphore::Wait from dominating the frame to under 10%. Explicitly setting it to
// 0 disables the flush entirely, same as before this had a default.
uint32_t DrawFlushInterval() {
	static const uint32_t interval = [] {
		const char* v = std::getenv("KYTY_DRAW_FLUSH_INTERVAL");
		if (v == nullptr) {
			return 16u;
		}
		return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
	}();
	return interval;
}

// v438y: the interval of the "flush4" toggle mode's on windows (KYTY_TOGGLE_FLUSH_INTERVAL, default
// 4 as before; 0: no periodic submit). With KYTY_SYNC_TOGGLE=10;KYTY_SYNC_TOGGLE_MODE=flush4 and
// KYTY_GPU_TIME=1, one run alternates the default 16 with this value every 10 s, and [floor]'s
// gpu= and fps= show what the periodic submits cost or buy on the GPU (61 % of all submits were
// "draw interval" in s03, 11-15: 4,400 of 7,200 a second).
uint32_t ToggleFlushInterval() {
	static const uint32_t interval = [] {
		const char* v = std::getenv("KYTY_TOGGLE_FLUSH_INTERVAL");
		return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : 4u;
	}();
	return interval;
}

// v438y: [submits] by reason without the full diagnostics (KYTY_FPS_ONLY=0 changes the shader cache
// key and the timing): KYTY_SUBMIT_STATS=1.
bool SubmitStats() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_SUBMIT_STATS");
		return Common::FpsLogDiagnostics() || (v != nullptr && v[0] == '1');
	}();
	return enabled;
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit(uint64_t tick) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search = [this, &gpu_tick, tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = tick;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = tick;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

// KYTY_GPU_TIME=1 (ported from idxtri/wolverine-perf 4f42b798): every command buffer gets a
// timestamp at its start and end. When its tick completes, the GPU time it covered (overlap with
// the previous command buffer removed) is added up, and every 5 s the log reports how busy the GPU
// was: whether the GPU or the GPU thread limits the frame rate. Begin/End run on the thread that
// records (the recording thread when KYTY_RECORD_THREAD is on), Collect on the GPU thread.
struct CommandScheduler::GpuTimer {
	static constexpr uint32_t Slots = 4096;

	explicit GpuTimer(GraphicContext& graphics): graphics(graphics) {
		if (!Enabled()) return;
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = Slots * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &pool) != vk::Result::eSuccess) {
			pool = nullptr;
		}
		period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	}
	~GpuTimer() {
		if (pool) graphics.device.destroyQueryPool(pool, nullptr);
	}
	KYTY_CLASS_NO_COPY(GpuTimer);

	static bool Enabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_GPU_TIME");
			return value != nullptr && value[0] != '0';
		}();
		return enabled;
	}

	// The recording thread, a new command buffer: its slot, or UINT32_MAX.
	uint32_t Begin(vk::CommandBuffer command) {
		if (!pool) return UINT32_MAX;
		{
			std::lock_guard lock(mutex);
			if (pending.size() + open >= Slots) return UINT32_MAX;
			open++;
		}
		const auto slot = next_slot++ % Slots;
		command.resetQueryPool(pool, slot * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, pool, slot * 2);
		return slot;
	}

	// The recording thread, before the command buffer ends.
	void End(vk::CommandBuffer command, uint32_t slot) {
		if (slot != UINT32_MAX) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, pool, slot * 2 + 1);
		}
	}

	void Submitted(uint64_t tick, uint32_t slot) {
		if (slot == UINT32_MAX) return;
		std::lock_guard lock(mutex);
		open--;
		pending.push_back({tick, slot});
	}

	template <typename IsFree>
	void Collect(IsFree&& is_free) {
		if (!pool) return;
		std::lock_guard lock(mutex);
		while (!pending.empty() && is_free(pending.front().tick)) {
			const auto              slot = pending.front().slot;
			std::array<uint64_t, 2> stamps {};
			if (graphics.device.getQueryPoolResults(pool, slot * 2, 2, sizeof(stamps), stamps.data(),
			                                        sizeof(uint64_t), vk::QueryResultFlagBits::e64) ==
			        vk::Result::eSuccess &&
			    stamps[1] >= stamps[0]) {
				const auto begin = std::max(stamps[0], last_end);
				if (stamps[1] > begin) {
					const double span = static_cast<double>(stamps[1] - begin) * period_ns;
					busy_ns += span;
					if (Common::FloorStats::Enabled()) {
						Common::FloorStats::Add(Common::FloorStats::Get().gpu_busy_ns, static_cast<uint64_t>(span));
					}
				}
				last_end = std::max(last_end, stamps[1]);
				buffers++;
			}
			pending.pop_front();
		}
		const auto now     = std::chrono::steady_clock::now();
		const auto elapsed = std::chrono::duration<double>(now - report).count();
		if (elapsed >= 5.0) {
			if (buffers != 0) {
				const double busy_ms_per_s = busy_ns / 1e6 / elapsed;
				std::fprintf(stderr, "[gpu-time] %.1f s: GPU busy %.1f ms/s (%.0f%%), %llu command buffers\n",
				             elapsed, busy_ms_per_s, busy_ms_per_s / 10.0,
				             static_cast<unsigned long long>(buffers));
				std::fflush(stderr);
			}
			busy_ns = 0;
			buffers = 0;
			report  = now;
		}
	}

	struct Pending {
		uint64_t tick = 0;
		uint32_t slot = 0;
	};

	GraphicContext&                       graphics;
	vk::QueryPool                         pool      = nullptr;
	float                                 period_ns = 1.0f;
	uint32_t                              next_slot = 0; // recording thread only
	uint32_t                              open      = 0;
	std::mutex                            mutex;
	std::deque<Pending>                   pending;
	uint64_t                              last_end = 0;
	double                                busy_ns  = 0;
	uint64_t                              buffers  = 0;
	std::chrono::steady_clock::time_point report   = std::chrono::steady_clock::now();
};

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_gpu_timer(std::make_unique<GpuTimer>(graphics)), m_master(graphics), m_context(context),
      m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	OpLatencyStart(this, m_graphics.device, m_master.Handle()); // KYTY_QUEUE_STATS only
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	OpLatencyStop(this); // before the timeline semaphore goes
}

int64_t CommandScheduler::OperationDoneNs() noexcept {
	return t_operation_done_ns;
}

void CommandScheduler::NoteLabelLatency(int64_t done_ns) noexcept {
	if (done_ns != 0 && OpLatencyEnabled()) {
		OpLatencyAdd(OpLatency::Label, done_ns);
	}
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	// Every tick was waited on above, so the queue and recording threads have nothing left to
	// submit.
	StopRecordingThread();
	StopSubmitThread();
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering(std::source_location site) {
	if (Active() && !m_command.IsInvalid()) {
		const CpuCost::Phase cpu_phase_("Scheduler::EndRendering");
		Current().EndRendering(site);
	}
}

static thread_local const char* t_submit_reason = nullptr;

void SetSubmitReason(const char* reason) {
	t_submit_reason = reason;
}

// Sets the reason unless the caller already named one.
void SetSubmitReasonIfUnset(const char* reason) {
	if (t_submit_reason == nullptr) {
		t_submit_reason = reason;
	}
}
static void DefaultSubmitReason(const char* reason) {
	SetSubmitReasonIfUnset(reason);
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::CompleteReleaseMemWrite() {
	constexpr uint32_t WritesPerSubmission = 32;
	if (++m_recorded_release_mem_writes < WritesPerSubmission) {
		return;
	}
	CheckActive();
	DefaultSubmitReason("eop writes");
	Flush();
}

void CommandScheduler::CompleteReleaseMemInterrupt() {
	// Deliberately smaller than CompleteReleaseMemWrite's 32: this event has already been queued
	// for guest delivery once its tick completes (see Sync::TriggerEopEventAtEndOfPipe ->
	// DeferPriorityOperation), and a guest thread may be blocked waiting on it via an event queue.
	// Batching still defers only the vkQueueSubmit -- the event fires once that (now slightly
	// larger) submission's tick completes, same as before, just a handful of RELEASE_MEM events
	// later instead of immediately.
	constexpr uint32_t InterruptsPerSubmission = 8;
	if (++m_recorded_release_mem_interrupts < InterruptsPerSubmission) {
		return;
	}
	CheckActive();
	DefaultSubmitReason("eop interrupts");
	Flush();
}

void CommandScheduler::CompleteDraw() {
	// Toggle mode "flush4" (v317): a submit every 4 draws instead of KYTY_DRAW_FLUSH_INTERVAL's 16 in
	// the on windows, so the work a game thread then waits for reaches the GPU sooner.
	if (Common::FloorStats::Enabled()) {
		Common::FloorStats::Add(Common::FloorStats::Get().draws, 1);
	}
	const auto interval = SyncToggle::Flush4Window() ? ToggleFlushInterval() : DrawFlushInterval();
	if (interval == 0u || ++m_recorded_draws < interval) {
		return;
	}
	CheckActive();
	DefaultSubmitReason("draw interval");
	Flush();
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	if (Common::FloorStats::Enabled()) {
		auto& floor = Common::FloorStats::Get();
		Common::FloorStats::Add(floor.cmd_buffers, 1);
		// [floor] lag=: how many command buffers this thread has handed off that the host GPU has
		// not finished (recording thread + GPU), sampled at every 8th submit with a fresh GPU tick
		// (one driver call, ~500 a second). Near 0: the GPU keeps up with this thread. A large
		// share of cb/f: the frame waits behind the recording thread or the GPU.
		if ((++m_lag_sample_counter & 7u) == 0) {
			m_master.Refresh();
			const auto current = m_master.CurrentTick();
			const auto done    = m_master.KnownGpuTick();
			Common::FloorStats::Add(floor.lag_ticks, current > done + 1 ? current - done - 1 : 0);
			Common::FloorStats::Add(floor.lag_samples, 1);
		}
	}
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	KYTY_PROFILER_FUNCTION();
	DefaultSubmitReason("flush and wait");
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	KYTY_PROFILER_FUNCTION();
	CheckActive();
	const CpuCost::Phase cpu_phase_("Scheduler::Finish");
	if (!m_command.IsInvalid()) {
		DefaultSubmitReason("finish");
		Submit();
	}
	const auto wait_begin = std::chrono::steady_clock::now();
	m_master.Wait(CurrentTick() - 1);
	if (Common::FloorStats::Enabled() && GuestGpu::IsGpuThread()) {
		auto& floor = Common::FloorStats::Get();
		Common::FloorStats::Add(floor.gpu_wait_ns,
		                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                  std::chrono::steady_clock::now() - wait_begin)
		                                                  .count()));
		Common::FloorStats::Add(floor.gpu_waits, 1);
	}
	BeginNext();
	PopPendingOperations();
}

// How long the GPU thread spends drained on the host GPU, and how often. A drain that
// submits the current tick first (a CPU read of GPU-owned memory mid-frame) is counted apart
// from one that waits on already submitted work.
namespace {
struct WaitStats {
	uint64_t count       = 0;
	uint64_t drain_count = 0;
	double   total_s     = 0.0;
	double   drain_s     = 0.0;
	double   max_s       = 0.0;
};
WaitStats g_wait_stats;

void RecordWait(double seconds, bool drained) {
	if (!GuestGpu::IsGpuThread()) {
		return;
	}
	if (Common::FloorStats::Enabled()) {
		auto& floor = Common::FloorStats::Get();
		Common::FloorStats::Add(floor.gpu_wait_ns, static_cast<uint64_t>(seconds * 1e9));
		Common::FloorStats::Add(floor.gpu_waits, 1);
	}
	auto& stats = g_wait_stats;
	stats.count++;
	stats.total_s += seconds;
	stats.max_s = std::max(stats.max_s, seconds);
	if (drained) {
		stats.drain_count++;
		stats.drain_s += seconds;
	}
	if (stats.count % 256 == 0) {
		LOGF("GpuWaits: count=%" PRIu64 " total=%.1fms max=%.1fms drains=%" PRIu64
		     " drain_total=%.1fms\n",
		     stats.count, stats.total_s * 1000.0, stats.max_s * 1000.0, stats.drain_count,
		     stats.drain_s * 1000.0);
		stats = {};
	}
}
} // namespace

void CommandScheduler::Wait(uint64_t tick) {
	KYTY_PROFILER_FUNCTION();
	const CpuCost::Phase cpu_phase_("Scheduler::Wait");
	EXIT_IF(tick > CurrentTick());
	// KYTY_WAIT_STACKS=1: a host stack for every 23rd wait that has to block (up to 60), to name
	// the paths that make the command thread wait for the GPU.
	if (static const bool stacks = std::getenv("KYTY_WAIT_STACKS") != nullptr;
	    stacks && !m_master.IsFree(tick)) {
		static std::atomic<uint32_t> waits {0};
		const auto                   n = waits.fetch_add(1);
		if (n % 23 == 0 && n / 23 < 60) {
			std::fprintf(stderr, "[wait-stack] #%u tick=%llu current=%llu\n%s\n", n,
			             static_cast<unsigned long long>(tick),
			             static_cast<unsigned long long>(CurrentTick()),
			             Common::HostBacktrace().c_str());
			std::fflush(stderr);
		}
	}
	Common::Timer timer;
	timer.Start();
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		{
			KYTY_PROFILER_BLOCK("CommandScheduler::Wait (forced submit-then-wait)");
			// KYTY_DRAIN_STACKS=1: a host stack for every 37th forced drain (up to 48), to name
			// the paths that make the command thread wait for the whole GPU.
			if (static const bool stacks = std::getenv("KYTY_DRAIN_STACKS") != nullptr; stacks) {
				static std::atomic<uint32_t> drains {0};
				const auto                   n = drains.fetch_add(1);
				if (n % 37 == 0 && n / 37 < 48) {
					std::fprintf(stderr, "[drain] #%u\n%s\n", n, Common::HostBacktrace().c_str());
					std::fflush(stderr);
				}
			}
			DefaultSubmitReason("wait for the open tick");
			const auto submitted_tick = Submit();
			EXIT_IF(submitted_tick != tick);
			Timeline::Mark("drain-begin", tick);
			m_master.Wait(tick);
			Timeline::Mark("drain-end", tick);
			BeginNext();
		}
		RecordWait(timer.GetTimeS(), true);
	} else {
		Timeline::Mark("wait-begin", tick);
		m_master.Wait(tick);
		Timeline::Mark("wait-end", tick);
		RecordWait(timer.GetTimeS(), false);
	}
}

void CommandScheduler::PopPendingOperations() {
	const CpuCost::Phase cpu_phase_("PopPendingOperations");
	// Read the GPU's progress (a driver call, ~1 us, once per draw and dispatch before) only when
	// an operation waits for a tick not known to be done.
	{
		uint64_t front = 0;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			front = m_pending_operations.front().tick;
		}
		// An operation deferred while recording carries the current tick, which is not submitted
		// yet: the GPU cannot have reached it, so polling the driver would find nothing to run
		// (KYTY_HOTPATH).
		if (front >= m_master.CurrentTick() && SyncToggle::HotPath()) {
			return;
		}
		if (!m_master.IsFree(front)) {
			// v445e, KYTY_POP_POLL_US=<n> (default 50; 0: at every call, as before). This runs before
			// every draw and dispatch (~75,000 a second in a fight), and while the oldest operation's
			// tick is submitted and not finished each call asked the driver how far the GPU is (~1 us).
			// Now it asks when a command buffer was submitted since the last time it asked (the GPU got
			// new work, and whoever waits for a tick submits first) or n microseconds have passed.
			// An operation runs once its tick is done, never before; here it may run up to n us later.
			static const int64_t poll_us = [] {
				const char* value = std::getenv("KYTY_POP_POLL_US");
				return value != nullptr ? static_cast<int64_t>(std::strtoll(value, nullptr, 10)) : int64_t {50};
			}();
			static const bool poll_stats = std::getenv("KYTY_QUEUE_STATS") != nullptr;
			static std::atomic<uint64_t> calls {0}, polls {0};
			static std::atomic<int64_t>  reported {0};
			bool                         poll = true;
			if (poll_us > 0) {
				const auto now  = std::chrono::steady_clock::now();
				const auto tick = m_master.CurrentTick();
				poll = tick != m_pop_poll_tick || now - m_pop_poll_time >= std::chrono::microseconds(poll_us);
				if (poll) {
					m_pop_poll_tick = tick;
					m_pop_poll_time = now;
				}
			}
			if (poll_stats) {
				calls.fetch_add(1, std::memory_order_relaxed);
				polls.fetch_add(poll ? 1u : 0u, std::memory_order_relaxed);
				const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				                    std::chrono::steady_clock::now().time_since_epoch())
				                    .count();
				auto last = reported.load(std::memory_order_relaxed);
				if (last == 0) {
					reported.compare_exchange_strong(last, ms);
				} else if (ms - last >= 5000 && reported.compare_exchange_strong(last, ms)) {
					const double s = static_cast<double>(ms - last) / 1000.0;
					std::fprintf(stderr,
					             "[pop-poll] %.0f calls/s with an unfinished oldest operation, %.0f/s asked the "
					             "driver (KYTY_POP_POLL_US=%lld)\n",
					             static_cast<double>(calls.exchange(0)) / s, static_cast<double>(polls.exchange(0)) / s,
					             static_cast<long long>(poll_us));
					OpLatencyReport(true);
				}
			}
			if (!poll) {
				return; // nothing new is known: the loop below would find the front not free
			}
			m_master.Refresh();
		}
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		if (OpLatencyEnabled()) {
			const auto done_ns  = OpLatencyDoneNs(operation.tick);
			OpLatencyAdd(OpLatency::Pending, done_ns);
			t_operation_done_ns = done_ns != 0 ? done_ns : SteadyNs();
		}
		RunOperation(std::move(operation.callback));
		t_operation_done_ns = 0;
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	DeferRelease(std::move(operation));
}

void CommandScheduler::DeferRelease(Common::UniqueFunction<void>&& operation) {
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		DrainProtectionHelper(ProtectDrainReason::GuestSignal);
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	Common::Sampler::RegisterCurrentThread("prio-ops");
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		{
			const auto wait_begin = std::chrono::steady_clock::now();
			m_master.Wait(operation.tick);
			if (Common::FloorStats::Enabled()) {
				const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				                    std::chrono::steady_clock::now() - wait_begin)
				                    .count();
				if (ms >= 200) {
					std::fprintf(stderr,
					             "[floor-prio] priority operation waited %lld ms for tick %llu (recording tick now %llu)\n",
					             static_cast<long long>(ms), static_cast<unsigned long long>(operation.tick),
					             static_cast<unsigned long long>(CurrentTick()));
				}
			}
		}
		Timeline::Mark("tick-done", operation.tick);
		if (!stop.stop_requested()) {
			// Priority callbacks publish labels, interrupts, fences, and flips to guest waiters.
			DrainProtectionHelper(ProtectDrainReason::GuestSignal);
			if (OpLatencyEnabled()) {
				const auto done_ns  = OpLatencyDoneNs(operation.tick);
				OpLatencyAdd(OpLatency::Priority, done_ns);
				t_operation_done_ns = done_ns != 0 ? done_ns : SteadyNs();
			}
			RunOperation(std::move(operation.callback));
			t_operation_done_ns = 0;
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

bool CommandScheduler::WaitFor(uint64_t tick, uint64_t timeout_ns) {
	return IsFree(tick) || (tick < CurrentTick() && m_master.WaitFor(tick, timeout_ns));
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	const CpuCost::Phase cpu_phase_("Scheduler::BeginCommand");
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_open = true;
	m_command_serial++;       // v445d: a new command buffer:
	m_compute_state_serial++; // nothing is bound in it
	// Ticks are allocated in submit order, so this command buffer will be submitted with the
	// current one.
	const auto tick = m_master.CurrentTick();
	m_gpu_timer->Collect([this](uint64_t done) { return m_master.IsFree(done); });
	if (m_threaded) {
		m_command.m_buffer = nullptr;
		Record([this, tick](vk::CommandBuffer) {
			m_worker_buffer = m_command_pool.Commit(tick);
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(m_worker_buffer.begin(&begin_info) != vk::Result::eSuccess);
			m_timer_slot = m_gpu_timer->Begin(m_worker_buffer);
		});
		return m_command;
	}
	m_command.m_buffer = m_command_pool.Commit(tick);
	m_command.Begin();
	m_timer_slot = m_gpu_timer->Begin(m_command.Handle());
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	KYTY_PROFILER_FUNCTION();
	const CpuCost::Phase cpu_phase_("Scheduler::Submit"); // the hand-off of a finished command buffer
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	if (!m_threaded) {
		m_gpu_timer->End(m_command.Handle(), m_timer_slot);
	}
	m_command.End();
	EXIT_IF(m_graphics.queue == nullptr);
	Profiler::Add(Profiler::Counter::CommandBuffersSubmitted);
	// KYTY_SUBMIT_STACKS=1: a host stack for every 97th submission (up to 40), to name what splits
	// the frame into so many submissions. KYTY_FPS_LOG: submissions per second every 5 s.
	if (static const bool stacks = std::getenv("KYTY_SUBMIT_STACKS") != nullptr; stacks) {
		static std::atomic<uint32_t> submits {0};
		const auto                   n = submits.fetch_add(1);
		if (n % 97 == 0 && n / 97 < 40) {
			std::fprintf(stderr, "[submit-stack] #%u\n%s\n", n, Common::HostBacktrace().c_str());
			std::fflush(stderr);
		}
	}
	if (static const bool fps_log = SubmitStats(); fps_log) {
		static std::atomic<uint64_t> count {0};
		static std::atomic<int64_t>  window_start {0};
		const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		auto start = window_start.load(std::memory_order_relaxed);
		if (start == 0) {
			window_start.compare_exchange_strong(start, now);
			start = now;
		}
		const auto n = count.fetch_add(1) + 1;
		static std::mutex                                  reasons_mutex;
		static std::vector<std::pair<const char*, uint64_t>> reasons;
		{
			std::lock_guard lock(reasons_mutex);
			const char*     reason = t_submit_reason != nullptr ? t_submit_reason : "untagged";
			auto it = std::find_if(reasons.begin(), reasons.end(), [reason](const auto& r) { return r.first == reason; });
			if (it == reasons.end()) {
				reasons.emplace_back(reason, 1);
			} else {
				it->second++;
			}
		}
		if (now - start >= 5000 && window_start.compare_exchange_strong(start, now)) {
			count.store(0);
			std::string text;
			{
				std::lock_guard lock(reasons_mutex);
				std::sort(reasons.begin(), reasons.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				for (const auto& [reason, k]: reasons) {
					const char* base = std::strrchr(reason, '/');
					const char* base2 = std::strrchr(reason, '\\');
					base = base2 != nullptr && (base == nullptr || base2 > base) ? base2 : base;
					text += fmt::format(" {}={:.0f}", base != nullptr ? base + 1 : reason,
					                    static_cast<double>(k) * 1000.0 / static_cast<double>(now - start));
				}
				reasons.clear();
			}
			std::fprintf(stderr, "[submits] %.0f per second:%s\n",
			             static_cast<double>(n) * 1000.0 / static_cast<double>(now - start), text.c_str());
		}
	}
	t_submit_reason = nullptr;

	SubmitJob job {
	    .buffer       = m_command.m_buffer,
	    .submit       = submit,
	    .debug_op     = m_command.m_debug_op,
	    .debug_submit = m_command.m_debug_submit_id,
	    .debug_arg0   = m_command.m_debug_arg0,
	    .debug_arg1   = m_command.m_debug_arg1,
	    .debug_arg2   = m_command.m_debug_arg2,
	    .debug_arg3   = m_command.m_debug_arg3,
	    .debug_arg4   = m_command.m_debug_arg4,
	    .upload_reads = UploadWorkerPushed(),
	};
	m_command.m_buffer                = nullptr;
	m_command.m_open                  = false;
	m_recorded_release_mem_writes     = 0;
	m_recorded_release_mem_interrupts = 0;
	m_recorded_draws                  = 0;

	if (m_threaded) {
		// The tick is allocated now, in order; the recording thread ends the command buffer and
		// submits it once it has recorded everything queued before.
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		Record([this, job](vk::CommandBuffer command) mutable {
			m_gpu_timer->End(command, m_timer_slot);
			EXIT_NOT_IMPLEMENTED(command.end() != vk::Result::eSuccess);
			job.buffer = command;
			QueueSubmit(job);
			m_gpu_timer->Submitted(job.tick, m_timer_slot);
			m_timer_slot    = UINT32_MAX;
			m_worker_buffer = nullptr;
		});
		DispatchChunk();
		ReportRecording();
		return job.tick;
	}
	if (!m_async_submit) {
		QueueSubmit(job);
		m_gpu_timer->Submitted(job.tick, m_timer_slot);
		m_timer_slot = UINT32_MAX;
		return job.tick;
	}
	{
		// Ticks are allocated in queue order and the queue thread submits in that order, so the
		// timeline is signaled in order.
		std::lock_guard lock(m_submit_mutex);
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		m_submit_jobs.push_back(job);
	}
	m_gpu_timer->Submitted(job.tick, m_timer_slot);
	m_timer_slot = UINT32_MAX;
	m_submit_available.notify_one();
	return job.tick;
}

void CommandScheduler::QueueSubmit(SubmitJob& job) {
	const CpuCost::Phase cpu_phase_("Scheduler::QueueSubmit"); // on the GPU thread only without a recording thread
	// KYTY_UPLOAD_WORKER: the staging bytes this submission's copies read are written.
	WaitUploadWorker(job.upload_reads);
	// v438zv (KYTY_TEX_SPARSE): the sparse binds asked for so far are done before this runs.
	m_graphics.WaitSparseBinds();
	// Every guarded write recorded in this command buffer must be protected before it can execute.
	DrainProtectionHelper(ProtectDrainReason::Submit);
	auto&      graphics = m_graphics;
	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		if (!m_async_submit) {
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
		}
		const auto& submit = job.submit;

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
		if (result == vk::Result::eSuccess && job.tick > m_vk_submitted_tick.load(std::memory_order_relaxed)) {
			m_vk_submitted_tick.store(job.tick, std::memory_order_release);
		}
	}
	Timeline::Mark("vk-submit", job.tick);

	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(graphics);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit,
		                  job.debug_arg0, job.debug_arg1, job.debug_arg2, job.debug_arg3,
		                  job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	{
		std::lock_guard lock(m_after_submit_mutex);
		if (m_after_submit) {
			m_after_submit();
		}
	}
}

void CommandScheduler::EnableAsyncSubmit() {
	// The owner enables it while constructing, before any other thread can submit.
	EXIT_IF(m_async_submit);
	m_async_submit  = true;
	m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("GpuQueueSubmit");
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			// A stop request still submits the queued jobs: their ticks may already be waited on.
			if (!m_submit_available.wait(lock, stop, [this] { return !m_submit_jobs.empty(); })) {
				return;
			}
			job = m_submit_jobs.front();
			m_submit_jobs.pop_front();
		}
		QueueSubmit(job);
	}
}

void CommandScheduler::StopSubmitThread() {
	if (m_submit_thread.joinable()) {
		m_submit_thread.request_stop();
		m_submit_thread.join();
	}
	std::lock_guard lock(m_submit_mutex);
	EXIT_IF(!m_submit_jobs.empty());
}

void CommandScheduler::EnableRecordingThread() {
	// The owner enables it while constructing, before the first command buffer.
	EXIT_IF(m_record_thread.joinable() || m_async_submit || !m_command.IsInvalid());
	// QueueSubmit then takes the tick Submit() allocated, as for the queue thread.
	m_async_submit  = true;
	m_threaded      = true;
	m_record_thread = std::jthread([this](std::stop_token stop) { RecordingThread(stop); });
}

void* CommandScheduler::Allocate(size_t size, size_t alignment) {
	EXIT_IF(!m_threaded);
	if (size > Chunk::Size) {
		// v441d: a stash bigger than a chunk (a player hit 'size > Chunk::Size' here) gets its own
		// block; the next recorded command's chunk keeps it (Record), so it lives as long as the
		// data that command reads.
		auto block = std::make_unique<std::byte[]>(size + alignment);
		auto raw   = reinterpret_cast<uintptr_t>(block.get());
		auto* data = reinterpret_cast<void*>((raw + alignment - 1) & ~(uintptr_t {alignment} - 1));
		m_pending_big.push_back(std::move(block));
		return data;
	}
	const auto take = [this] {
		std::lock_guard lock(m_record_mutex);
		if (m_free_chunks.empty()) {
			return std::make_unique<Chunk>();
		}
		auto chunk = std::move(m_free_chunks.back());
		m_free_chunks.pop_back();
		return chunk;
	};
	if (m_chunk == nullptr) {
		m_chunk = take();
	}
	auto offset = (m_chunk->used + alignment - 1) & ~(alignment - 1);
	if (offset + size > Chunk::Size) {
		DispatchChunk();
		m_chunk = take();
		offset  = 0;
	}
	m_chunk->used = offset + size;
	return m_chunk->storage + offset;
}

void CommandScheduler::DispatchChunk() {
	if (m_chunk == nullptr || m_chunk->first == nullptr) {
		return;
	}
	{
		std::lock_guard lock(m_record_mutex);
		m_record_queue.push_back(std::move(m_chunk));
		m_chunks_dispatched++;
	}
	m_record_available.notify_one();
}

vk::CommandBuffer CommandScheduler::DrainRecording() {
	m_compute_state_serial++; // whoever asked records directly: the compute bind point may change
	if (!m_threaded) {
		return m_command.m_buffer;
	}
	KYTY_PROFILER_FUNCTION();
	DispatchChunk();
	std::unique_lock lock(m_record_mutex);
	m_drains++;
	Profiler::Add(Profiler::Counter::RecordingDrains);
	// KYTY_RECORD_DRAIN_STACKS=1: which raw Handle() sites make the GPU thread wait (a host stack
	// for every 4,999th drain, up to 64).
	static const bool stacks = std::getenv("KYTY_RECORD_DRAIN_STACKS") != nullptr;
	if (stacks && m_drains % 4999 == 0 && m_drains / 4999 <= 64) {
		LOGF("Recording thread drain %" PRIu64 ":\n%s", m_drains,
		     Common::HostBacktrace().c_str());
	}
	if (m_chunks_executed != m_chunks_dispatched && Common::FloorStats::Enabled()) {
		// [floor] drain=: this thread waiting for the recording thread to catch up.
		const auto wait_begin = std::chrono::steady_clock::now();
		m_record_executed.wait(lock, [this] { return m_chunks_executed == m_chunks_dispatched; });
		auto& floor = Common::FloorStats::Get();
		Common::FloorStats::Add(floor.record_drain_ns,
		                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                  std::chrono::steady_clock::now() - wait_begin)
		                                                  .count()));
		Common::FloorStats::Add(floor.record_drains, 1);
		return m_worker_buffer;
	}
	m_record_executed.wait(lock, [this] { return m_chunks_executed == m_chunks_dispatched; });
	return m_worker_buffer;
}

void CommandScheduler::RecordingThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("GpuRecord");
	Common::Sampler::RegisterCurrentThread("record");
	SetBarrierMergeThread();
	Common::PreferPerformanceCores("Vulkan recording thread");
	Common::RaiseGpuThreadPriority();
	// A command's stashed arrays may sit in up to two chunks before its own: executed chunks are
	// recycled only after the next two have run.
	std::deque<std::unique_ptr<Chunk>> retained;
	for (;;) {
		std::unique_ptr<Chunk> chunk;
		{
			std::unique_lock lock(m_record_mutex);
			// A stop request still runs the queued chunks: their ticks may already be waited on.
			if (!m_record_available.wait(lock, stop, [this] { return !m_record_queue.empty(); })) {
				return;
			}
			chunk = std::move(m_record_queue.front());
			m_record_queue.pop_front();
		}
		// [floor] rec=: the recording thread's busy time (two clock reads per chunk).
		const bool floor_stats = Common::FloorStats::Enabled();
		const auto chunk_begin = floor_stats ? std::chrono::steady_clock::now()
		                                     : std::chrono::steady_clock::time_point {};
		for (auto* command = chunk->first; command != nullptr;) {
			auto* next = command->next;
			command->Execute(m_worker_buffer);
			command->~RecordedCommand();
			command = next;
		}
		if (floor_stats) {
			Common::FloorStats::Add(Common::FloorStats::Get().record_busy_ns,
			                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                                  std::chrono::steady_clock::now() - chunk_begin)
			                                                  .count()));
		}
		chunk->first = nullptr;
		chunk->last  = nullptr;
		chunk->used  = 0;
		retained.push_back(std::move(chunk));
		std::unique_ptr<Chunk> recycled;
		if (retained.size() > 2) {
			recycled = std::move(retained.front());
			retained.pop_front();
		}
		{
			std::lock_guard lock(m_record_mutex);
			if (recycled != nullptr) {
				recycled->big.clear();
				m_free_chunks.push_back(std::move(recycled));
			}
			m_chunks_executed++;
		}
		m_record_executed.notify_all();
	}
}

// Every 5 s: command buffers submitted, chunks handed over, and drains (raw Handle() uses, which
// make the GPU thread wait for the recording thread).
void CommandScheduler::ReportRecording() {
	m_record_report_buffers++;
	const auto now = std::chrono::steady_clock::now();
	if (now - m_record_report_time < std::chrono::seconds(5)) {
		return;
	}
	uint64_t chunks = 0;
	uint64_t drains = 0;
	{
		std::lock_guard lock(m_record_mutex);
		chunks = m_chunks_dispatched;
		drains = m_drains;
	}
	LOGF("Recording thread (5 s): %" PRIu64 " command buffers, %" PRIu64 " chunks, %" PRIu64
	     " drains\n",
	     m_record_report_buffers, chunks - m_record_report_chunks, drains - m_record_report_drains);
	m_record_report_buffers = 0;
	m_record_report_chunks  = chunks;
	m_record_report_drains  = drains;
	m_record_report_time    = now;
}

void CommandScheduler::StopRecordingThread() {
	if (!m_record_thread.joinable()) {
		return;
	}
	(void)DrainRecording();
	m_record_thread.request_stop();
	m_record_available.notify_all();
	m_record_thread.join();
	LOGF("Recording thread: %" PRIu64 " chunks, %" PRIu64 " drains\n", m_chunks_dispatched,
	     m_drains);
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
