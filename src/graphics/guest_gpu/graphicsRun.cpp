#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/drawPipe.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "common/floorStats.h"
#include "common/fpsLog.h"

#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/drawPrepScanner.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/queueShare.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/imgRef.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "common/sampler.h"
#include "graphics/host_gpu/renderer/barrierMerge.h"
#include "graphics/host_gpu/renderer/barrierSink.h"
#include "graphics/host_gpu/renderer/barrierReason.h"
#include "graphics/host_gpu/renderer/dispatchRun.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompileProgress.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderStats.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/guestWait.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <map>
#include <string>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// renderContext.cpp: the CPU epoch of the submission the GPU thread is running (PrepareBda).
extern std::atomic<uint64_t> g_bda_submission_cpu_epoch;

// bufferCache.cpp (KYTY_UPLOAD_WORKER).
uint64_t UploadWorkerPushed();
void     WaitUploadWorker(uint64_t upto);
// Before the command processor writes guest memory itself: upload reads queued before must not
// see the bytes it writes (they belong to work recorded earlier).
static void SettleUploadReads() {
	const CpuCost::Phase cpu_phase_("SettleUploadReads");
	DrainProtectionHelper(ProtectDrainReason::GuestWrite);
	if (const auto pushed = UploadWorkerPushed(); pushed != 0) {
		WaitUploadWorker(pushed);
	}
}

static std::atomic<uint64_t> g_cmd_from_send {0};
static std::atomic<uint64_t> g_cmd_from_try {0};
static std::atomic<uint64_t> g_cmd_from_sync {0};

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local const uint32_t*   g_run_expected_next = nullptr; // KYTY_DISPATCH_RUNS
// KYTY_BARRIER_REASON_STATS (statistics only): why the scan after the thread's last dispatch found
// no next dispatch for its run (BarrierReason::Why).
static thread_local uint32_t          g_run_last_stop     = 0;
static std::atomic<uint64_t> g_compute_submissions {0};
static std::atomic<uint64_t> g_compute_submission_dw {0};
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;
static thread_local bool              g_draw_pipe_thread  = false; // KYTY_DRAW_PIPE stage B

// KYTY_ASYNC_THREAD=1: the compute queues run on a second command-processor thread. The renderer
// is shared: each thread takes the renderer mutex (recursive) for every packet that can touch it
// (all but register writes) and for the renderer work around its slices.
// KYTY_GFX_PRIORITY=1 (experiment, off by default): the GPU thread takes the graphics queue's next
// submission whenever it can run, before the compute queues' (which keep their round-robin among
// themselves). On the PS5 the queues run concurrently: the graphics queue never waits behind a
// compute queue's slice unless it synchronizes with it. The game paces its frames on a label the
// graphics queue writes (PushBuffer::WaitForFrame). KYTY_SYNC_TOGGLE_MODE=gfxprio: on in the
// toggle's on windows only.
static bool GfxPriority() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_GFX_PRIORITY");
		return value != nullptr && value[0] == '1';
	}();
	return always || SyncToggle::GfxPriorityWindow();
}

static bool AsyncThreadAlways() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_THREAD");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// The async-compute thread exists: KYTY_ASYNC_THREAD=1, or KYTY_SYNC_TOGGLE_MODE=async (on in the
// toggle's on windows only; in the off windows the GPU thread runs every queue as without it, still
// taking the renderer mutex, so both windows pay for the mutex).
static bool AsyncThreadEnabled() {
	static const bool enabled = AsyncThreadAlways() || SyncToggle::AsyncConfigured();
	return enabled;
}
// The compute queues run on the async-compute thread now.
static bool AsyncThreadActive() {
	return AsyncThreadAlways() || SyncToggle::AsyncWindow();
}
// G4 stage 1 (docs/g4_split_plan_20261102.md): on the async-compute thread, direct dispatches are
// prepared without the renderer mutex and queued; the mutex is taken once for each run of other
// packets (a segment), which first records the queued dispatches. KYTY_ASYNC_BATCH=0: every
// non-register packet takes the mutex, as in the prototype.
static bool AsyncBatchEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_BATCH");
		return !(value != nullptr && value[0] == '0');
	}();
	return enabled;
}
// KYTY_ASYNC_BATCH_INDIRECT=1 (with the batch): DISPATCH_INDIRECT is queued the same way.
static bool AsyncBatchIndirect() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_BATCH_INDIRECT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// The async thread holds the renderer mutex for a segment.
static thread_local bool t_segment_held = false;
// Dispatches queued before they are recorded, at most (KYTY_ASYNC_BATCH_MAX, default 128): with
// graphics right of way the queue is recorded in one go, which the graphics thread waits out.
static size_t AsyncBatchMaxValue() {
	static const size_t value = [] {
		const char* env = std::getenv("KYTY_ASYNC_BATCH_MAX");
		const size_t n  = env != nullptr ? std::strtoull(env, nullptr, 10) : 128u;
		return n == 0 ? size_t {1} : n;
	}();
	return value;
}
// The command processor whose registers the scheduler's command buffer is bound to (renderer
// mutex): a thread that takes the mutex binds its own first.
static CommandProcessor* g_bound_cp = nullptr;
// The renderer mutex while the async thread runs, nothing otherwise.
class RendererScope {
public:
	explicit RendererScope(RenderContext& renderer)
	    : m_renderer(AsyncThreadEnabled() ? &renderer : nullptr) {
		if (m_renderer != nullptr) {
			m_renderer->LockRenderer();
		}
	}
	~RendererScope() {
		if (m_renderer != nullptr) {
			m_renderer->UnlockRenderer();
		}
	}
	KYTY_CLASS_NO_COPY(RendererScope);

private:
	RenderContext* m_renderer;
};

static void InvalidateReplayBase(RenderContext& renderer, RenderStats::IndirectReplayNote reason) {
	if (!RenderStats::g_indirect_base_maybe.load(std::memory_order_relaxed)) {
		return;
	}
	RendererScope scope(renderer);
	renderer.GetRenderExecutor().InvalidateIndirectReplay(static_cast<uint32_t>(reason));
}
// Packets that only write the command processor's own registers run without the renderer mutex.
static bool RegisterOnlyPacket(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE: return true;
		default: return false;
	}
}

// KYTY_DRAW_PIPE: packets stage A runs while the draw thread still has draws (no renderer, no
// GPU-written memory): register writes, argument bases, calls and chains of command buffers, and
// NOPs that only skip or name. Everything else drains the draw thread first.
static bool DrawPipeDrawPacket(uint32_t opcode) {
	return opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 ||
	       opcode == Pm4::IT_DRAW_INDEX_AUTO;
}

static bool DrawPipeSafePacket(uint32_t opcode, const uint32_t* packet) {
	if (RegisterOnlyPacket(opcode)) {
		return true;
	}
	const auto header = packet[0];
	switch (opcode) {
		case Pm4::IT_SET_BASE: return true;
		// Memory the CP reads directly: safe unless a queued draw writes it (those bytes are not
		// marked GPU-written until the draw thread records it, so no fault would drain).
		case Pm4::IT_INDIRECT_BUFFER: {
			if (KYTY_PM4_LEN(header) != 4u) {
				return false;
			}
			const uint64_t address = packet[1] | (static_cast<uint64_t>(packet[2]) << 32u);
			return !DrawPipe::PendingWriteOverlaps(address, uint64_t {packet[3] & 0xfffffu} * 4u);
		}
		case Pm4::IT_SET_SH_REG_INDIRECT: {
			if (KYTY_PM4_LEN(header) != 5u) {
				return false;
			}
			const uint64_t address =
			    (packet[1] & 0xfffffffcu) | (static_cast<uint64_t>(packet[2]) << 32u);
			return !DrawPipe::PendingWriteOverlaps(address, uint64_t {packet[4] & 0x3fffu} * 8u);
		}
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return !DrawPipe::HasPendingWrites();
		case Pm4::IT_CLEAR_STATE: return true;
		// No-ops with dispatch sync on (dispatches carry their own barriers).
		case Pm4::IT_ACQUIRE_MEM: return DispatchSyncEnabled();
		// Barriers: queued for the draw thread (CommandProcessor::TriggerEvent).
		case Pm4::IT_EVENT_WRITE: return DrawPipe::DeferrableEvent(packet[1] & 0x3fu);
		case Pm4::IT_NOP: {
			const auto r = KYTY_PM4_R(header);
			if (r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER || r == Pm4::R_CONTEXT_STATE) {
				return true;
			}
			if (r == Pm4::R_ACQUIRE_MEM) {
				return DispatchSyncEnabled();
			}
			return r == Pm4::R_ZERO && (KYTY_PM4_LEN(header) < 2u || (packet[1] & 0xffff0000u) != 0x68750000u);
		}
		default: return false;
	}
}

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

// The Tracy source locations of the PM4 packets that get a command zone (Profiler::CommandZone):
// draws, dispatches, copies, memory writes and synchronization; not register writes and NOPs.
static const tracy::SourceLocationData* Pm4ZoneLocation(uint32_t opcode, uint32_t r) {
	struct Table {
		std::array<tracy::SourceLocationData, 256>        ops {};
		std::array<tracy::SourceLocationData, Pm4::R_NUM> custom {};
	};
	static const Table table = [] {
		Table      result;
		const auto op = [&](uint32_t code, const char* name) {
			result.ops[code] = {name, "CommandProcessor::ProcessPm4", __FILE__, 0, 0};
		};
		const auto custom = [&](uint32_t code, const char* name) {
			result.custom[code] = {name, "CommandProcessor::ProcessPm4", __FILE__, 0, 0};
		};
		op(Pm4::IT_DISPATCH_DIRECT, "PM4 DISPATCH_DIRECT");
		op(Pm4::IT_DISPATCH_INDIRECT, "PM4 DISPATCH_INDIRECT");
		op(Pm4::IT_DRAW_INDIRECT, "PM4 DRAW_INDIRECT");
		op(Pm4::IT_DRAW_INDEX_INDIRECT, "PM4 DRAW_INDEX_INDIRECT");
		op(Pm4::IT_DRAW_INDEX_2, "PM4 DRAW_INDEX_2");
		op(Pm4::IT_DRAW_INDIRECT_MULTI, "PM4 DRAW_INDIRECT_MULTI");
		op(Pm4::IT_DRAW_INDEX_AUTO, "PM4 DRAW_INDEX_AUTO");
		op(Pm4::IT_DRAW_INDEX_OFFSET_2, "PM4 DRAW_INDEX_OFFSET_2");
		op(Pm4::IT_DRAW_INDEX_INDIRECT_MULTI, "PM4 DRAW_INDEX_INDIRECT_MULTI");
		op(Pm4::IT_DISPATCH_DRAW, "PM4 DISPATCH_DRAW");
		op(Pm4::IT_WRITE_DATA, "PM4 WRITE_DATA");
		op(Pm4::IT_MEM_SEMAPHORE, "PM4 MEM_SEMAPHORE");
		op(Pm4::IT_WAIT_REG_MEM, "PM4 WAIT_REG_MEM");
		op(Pm4::IT_WAIT_REG_MEM_64, "PM4 WAIT_REG_MEM_64");
		op(Pm4::IT_COPY_DATA, "PM4 COPY_DATA");
		op(Pm4::IT_CP_DMA, "PM4 CP_DMA");
		op(Pm4::IT_DMA_DATA, "PM4 DMA_DATA");
		op(Pm4::IT_SURFACE_SYNC, "PM4 SURFACE_SYNC");
		op(Pm4::IT_EVENT_WRITE, "PM4 EVENT_WRITE");
		op(Pm4::IT_EVENT_WRITE_EOP, "PM4 EVENT_WRITE_EOP");
		op(Pm4::IT_EVENT_WRITE_EOS, "PM4 EVENT_WRITE_EOS");
		op(Pm4::IT_RELEASE_MEM, "PM4 RELEASE_MEM");
		op(Pm4::IT_ACQUIRE_MEM, "PM4 ACQUIRE_MEM");
		op(Pm4::IT_DUMP_CONST_RAM, "PM4 DUMP_CONST_RAM");
		custom(Pm4::R_DRAW_RESET, "PM4 NOP DRAW_RESET");
		custom(Pm4::R_WAIT_FLIP_DONE, "PM4 NOP WAIT_FLIP_DONE");
		custom(Pm4::R_DISPATCH_RESET, "PM4 NOP DISPATCH_RESET");
		custom(Pm4::R_ACQUIRE_MEM, "PM4 NOP ACQUIRE_MEM");
		custom(Pm4::R_WRITE_DATA, "PM4 NOP WRITE_DATA");
		custom(Pm4::R_FLIP, "PM4 NOP FLIP");
		custom(Pm4::R_RELEASE_MEM, "PM4 NOP RELEASE_MEM");
		custom(Pm4::R_DMA_DATA, "PM4 NOP DMA_DATA");
		custom(Pm4::R_CONTEXT_STATE, "PM4 NOP CONTEXT_STATE");
		return result;
	}();
	const auto& location = opcode == Pm4::IT_NOP ? table.custom[r & (Pm4::R_NUM - 1u)]
	                                             : table.ops[opcode & 0xffu];
	return location.name != nullptr ? &location : nullptr;
}

// KYTY_CPU_COST: the phase a PM4 packet's handler runs under, by opcode and, for NOPs, by the
// custom code in the header ("PM4 <name>", or the hex opcode). The time of a draw or dispatch
// packet outside its draw or dispatch (argument decoding) is this phase's self time.
static const char* Pm4PhaseName(uint32_t opcode, uint32_t r) {
	struct Names {
		char ops[256][16];
		char nops[Pm4::R_NUM][20];
	};
	static const Names* const names = [] {
		auto* result = new Names;
		for (uint32_t i = 0; i < 256; i++) {
			std::snprintf(result->ops[i], sizeof(result->ops[i]), "PM4 op 0x%02x", i);
		}
		for (uint32_t i = 0; i < Pm4::R_NUM; i++) {
			std::snprintf(result->nops[i], sizeof(result->nops[i]), "PM4 NOP r 0x%02x", i);
		}
		return result;
	}();
	if (opcode == Pm4::IT_NOP) {
		switch (r) {
			case Pm4::R_DRAW_RESET: return "PM4 NOP DRAW_RESET";
			case Pm4::R_WAIT_FLIP_DONE: return "PM4 NOP WAIT_FLIP_DONE";
			case Pm4::R_DISPATCH_RESET: return "PM4 NOP DISPATCH_RESET";
			case Pm4::R_ACQUIRE_MEM: return "PM4 NOP ACQUIRE_MEM";
			case Pm4::R_WRITE_DATA: return "PM4 NOP WRITE_DATA";
			case Pm4::R_FLIP: return "PM4 NOP FLIP";
			case Pm4::R_RELEASE_MEM: return "PM4 NOP RELEASE_MEM";
			case Pm4::R_DMA_DATA: return "PM4 NOP DMA_DATA";
			case Pm4::R_CONTEXT_STATE: return "PM4 NOP CONTEXT_STATE";
			default: return names->nops[r & (Pm4::R_NUM - 1u)];
		}
	}
	switch (opcode) {
		case Pm4::IT_SET_BASE: return "PM4 SET_BASE";
		case Pm4::IT_INDEX_BUFFER_SIZE: return "PM4 INDEX_BUFFER_SIZE";
		case Pm4::IT_DISPATCH_DIRECT: return "PM4 DISPATCH_DIRECT";
		case Pm4::IT_DISPATCH_INDIRECT: return "PM4 DISPATCH_INDIRECT";
		case Pm4::IT_SET_PREDICATION: return "PM4 SET_PREDICATION";
		case Pm4::IT_COND_EXEC: return "PM4 COND_EXEC";
		case Pm4::IT_DRAW_INDIRECT: return "PM4 DRAW_INDIRECT";
		case Pm4::IT_DRAW_INDEX_INDIRECT: return "PM4 DRAW_INDEX_INDIRECT";
		case Pm4::IT_INDEX_BASE: return "PM4 INDEX_BASE";
		case Pm4::IT_DRAW_INDEX_2: return "PM4 DRAW_INDEX_2";
		case Pm4::IT_CONTEXT_CONTROL: return "PM4 CONTEXT_CONTROL";
		case Pm4::IT_INDEX_TYPE: return "PM4 INDEX_TYPE";
		case Pm4::IT_DRAW_INDIRECT_MULTI: return "PM4 DRAW_INDIRECT_MULTI";
		case Pm4::IT_DRAW_INDEX_AUTO: return "PM4 DRAW_INDEX_AUTO";
		case Pm4::IT_NUM_INSTANCES: return "PM4 NUM_INSTANCES";
		case Pm4::IT_INDIRECT_BUFFER_CNST: return "PM4 INDIRECT_BUFFER_CNST";
		case Pm4::IT_DRAW_INDEX_OFFSET_2: return "PM4 DRAW_INDEX_OFFSET_2";
		case Pm4::IT_WRITE_DATA: return "PM4 WRITE_DATA";
		case Pm4::IT_MEM_SEMAPHORE: return "PM4 MEM_SEMAPHORE";
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return "PM4 DRAW_INDEX_INDIRECT_MULTI";
		case Pm4::IT_WAIT_REG_MEM: return "PM4 WAIT_REG_MEM";
		case Pm4::IT_INDIRECT_BUFFER: return "PM4 INDIRECT_BUFFER";
		case Pm4::IT_COPY_DATA: return "PM4 COPY_DATA";
		case Pm4::IT_CP_DMA: return "PM4 CP_DMA";
		case Pm4::IT_PFP_SYNC_ME: return "PM4 PFP_SYNC_ME";
		case Pm4::IT_SURFACE_SYNC: return "PM4 SURFACE_SYNC";
		case Pm4::IT_EVENT_WRITE: return "PM4 EVENT_WRITE";
		case Pm4::IT_EVENT_WRITE_EOP: return "PM4 EVENT_WRITE_EOP";
		case Pm4::IT_EVENT_WRITE_EOS: return "PM4 EVENT_WRITE_EOS";
		case Pm4::IT_RELEASE_MEM: return "PM4 RELEASE_MEM";
		case Pm4::IT_DMA_DATA: return "PM4 DMA_DATA";
		case Pm4::IT_ACQUIRE_MEM: return "PM4 ACQUIRE_MEM";
		case Pm4::IT_SET_SH_REG_INDIRECT: return "PM4 SET_SH_REG_INDIRECT";
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return "PM4 SET_UCONFIG_REG_INDIRECT";
		case Pm4::IT_SET_CONFIG_REG: return "PM4 SET_CONFIG_REG";
		case Pm4::IT_SET_CONTEXT_REG: return "PM4 SET_CONTEXT_REG";
		case Pm4::IT_SET_SH_REG: return "PM4 SET_SH_REG";
		case Pm4::IT_SET_QUEUE_REG: return "PM4 SET_QUEUE_REG";
		case Pm4::IT_SET_UCONFIG_REG: return "PM4 SET_UCONFIG_REG";
		case Pm4::IT_SET_UCONFIG_REG_INDEX: return "PM4 SET_UCONFIG_REG_INDEX";
		case Pm4::IT_WRITE_CONST_RAM: return "PM4 WRITE_CONST_RAM";
		case Pm4::IT_DUMP_CONST_RAM: return "PM4 DUMP_CONST_RAM";
		case Pm4::IT_INCREMENT_CE_COUNTER: return "PM4 INCREMENT_CE_COUNTER";
		case Pm4::IT_INCREMENT_DE_COUNTER: return "PM4 INCREMENT_DE_COUNTER";
		case Pm4::IT_WAIT_ON_CE_COUNTER: return "PM4 WAIT_ON_CE_COUNTER";
		case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF: return "PM4 WAIT_ON_DE_COUNTER_DIFF";
		case Pm4::IT_DISPATCH_DRAW: return "PM4 DISPATCH_DRAW";
		case Pm4::IT_WAIT_REG_MEM_64: return "PM4 WAIT_REG_MEM_64";
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT: return "PM4 SET_CONTEXT_REG_INDIRECT";
		default: return names->ops[opcode & 0xffu];
	}
}

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// KYTY_LOG_INDIRECT_DRAWS=<n>: log the arguments of the first n indirect draws, including
// multi-draws whose GPU-written count is zero. A diagnostic for draws that GPU work generation
// (culling passes) should have filled.
// KYTY_GPU_INDIRECT_DRAWS=0 reads every DRAW_INDIRECT's arguments on the CPU, as before.
static bool GpuIndirectDraws() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_INDIRECT_DRAWS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

static bool LogIndirectDraw() {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_LOG_INDIRECT_DRAWS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (budget == 0) {
		return false;
	}
	// With KYTY_LOG_DRAWS_TRIGGER=<file>, start once that file exists (checked every 256 draws).
	static const char*           trigger   = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	static std::atomic_bool      triggered = trigger == nullptr;
	static std::atomic<uint32_t> polls {0};
	if (!triggered) {
		if ((polls.fetch_add(1, std::memory_order_relaxed) & 255u) != 0u) {
			return false;
		}
		FILE* file = std::fopen(trigger, "rb");
		if (file == nullptr) {
			return false;
		}
		std::fclose(file);
		triggered = true;
	}
	static std::atomic<uint32_t> used {0};
	return used.fetch_add(1, std::memory_order_relaxed) < budget;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	if (DrawPrep::Enabled()) {
		m_draw_prep = std::make_unique<DrawPrepScanner>(renderer);
		LOGF("DrawPrep: the scanner prepares draws ahead of the GPU thread (KYTY_DRAW_PREP=0 turns it "
		     "off; verify %d)\n",
		     DrawPrep::VerifyMode());
	}
	m_thread = std::jthread(ThreadRun, this);
	if (AsyncThreadEnabled()) {
		LOGF("GPU: compute queues run on their own thread (KYTY_ASYNC_THREAD)\n");
		m_async_thread = std::jthread(ThreadRunRole, this, 1u);
	}
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
		m_async_work.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	if (m_async_thread.joinable()) {
		m_async_thread.join();
	}
	m_draw_prep.reset();
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	g_cmd_from_send.fetch_add(1, std::memory_order_relaxed);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	LibKernel::GuestWait::NoteSubmit(LibKernel::GuestWait::Submit::Command, true);
	m_work_available.Signal();
}

bool GuestGpu::TrySendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return true;
	}
	g_cmd_from_try.fetch_add(1, std::memory_order_relaxed);
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) {
		return false;
	}
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	LibKernel::GuestWait::NoteSubmit(LibKernel::GuestWait::Submit::Command, true);
	m_work_available.Signal();
	return true;
}

namespace {
std::atomic<bool> g_labels_deferred {false};
} // namespace

bool GuestGpu::LabelsDeferred() noexcept {
	return g_labels_deferred.load(std::memory_order_relaxed);
}

bool GuestGpu::LabelsAtCompletion() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LABELS_AT_COMPLETION");
		return value != nullptr && value[0] == '1';
	}();
	return enabled || SyncToggle::LabelsAtCompletion();
}

void GuestGpu::DeferLabelWrite(uint64_t address, uint64_t value, uint32_t size) {
	EXIT_IF(!IsGpuThread() || (size != 4 && size != 8));
	g_labels_deferred.store(true, std::memory_order_relaxed);
	const auto sequence        = ++m_label_sequence;
	m_pending_labels[address] = {value, size, sequence};
	// The completion runs on the scheduler's priority thread, which must not fault on guest
	// memory the GPU wrote to; the write itself goes back to the GPU thread.
	m_renderer.GetCommandScheduler().DeferPriorityOperation([this, address, value, size,
	                                                         sequence] {
		// KYTY_QUEUE_STATS ([op-latency]): when the label's tick was seen finished (0: stats off).
		const auto done_ns = CommandScheduler::OperationDoneNs();
		(void)TrySendCommand([this, address, value, size, sequence, done_ns] {
			DrainProtectionHelper(ProtectDrainReason::GuestWrite);
			Timeline::Mark("label", address, value);
			std::memcpy(reinterpret_cast<void*>(address), &value, size);
			LibKernel::GuestWait::NoteEvent(LibKernel::GuestWait::Event::Label);
			CommandScheduler::NoteLabelLatency(done_ns);
			const auto pending = m_pending_labels.find(address);
			if (pending != m_pending_labels.end() && pending->second.sequence == sequence) {
				m_pending_labels.erase(pending);
			}
		});
	});
}

template <typename T>
T GuestGpu::ReadLabel(const volatile T* address) const {
	if (!m_pending_labels.empty()) {
		const auto pending = m_pending_labels.find(reinterpret_cast<uint64_t>(address));
		if (pending != m_pending_labels.end() && pending->second.size >= sizeof(T)) {
			return static_cast<T>(pending->second.value);
		}
	}
	return *address;
}

template uint32_t GuestGpu::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t GuestGpu::ReadLabel<uint64_t>(const volatile uint64_t*) const;

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	if (m_pending_commands.load(std::memory_order_acquire) != 0) {
		DrawPipe::Drain(DrawPipe::Reason::Commands);
	}
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		const CpuCost::Phase command_phase_("Pm4::Command");
		RendererScope renderer_scope(m_renderer);
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	g_cmd_from_sync.fetch_add(1, std::memory_order_relaxed);
	// KYTY_GUEST_WAIT_STATS: the calling thread waits for the GPU thread (gpu-sync; a readback
	// times its own round trips).
	LibKernel::GuestWait::Scope guest_wait;
	if (LibKernel::GuestWait::Enabled()) [[unlikely]] {
		LibKernel::GuestWait::BeginGpuSync(guest_wait);
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

static bool CopySubmissions() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COPY_SUBMISSIONS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Timeline::Mark("submit-gfx", draw_commands.size(), constant_commands.size());
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	if (CopySubmissions()) {
		auto words = std::make_shared<std::vector<uint32_t>>();
		words->reserve(draw_commands.size() + constant_commands.size());
		words->insert(words->end(), draw_commands.begin(), draw_commands.end());
		words->insert(words->end(), constant_commands.begin(), constant_commands.end());
		submission.commands = std::span<const uint32_t>(words->data(), draw_commands.size());
		submission.constant_commands =
		    constant_commands.empty()
		        ? std::span<const uint32_t> {}
		        : std::span<const uint32_t>(words->data() + draw_commands.size(),
		                                    constant_commands.size());
		submission.owned_words = std::move(words);
	}
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Timeline::Mark("submit-cq", compute_queue, commands.size());
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	if (CopySubmissions()) {
		auto words = std::make_shared<std::vector<uint32_t>>(commands.begin(), commands.end());
		submission.commands    = std::span<const uint32_t>(words->data(), words->size());
		submission.owned_words = std::move(words);
	}
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Timeline::Mark("submit-flip", request_id);
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	{
		const LibKernel::GuestWait::Scope guest_wait(LibKernel::GuestWait::Prim::SuspendPoint);
		m_suspend_point_ready->acquire();
	}
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	if (RenderStats::IndirectReplayConfigured()) {
		InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
	}
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_args               = 0;
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	if (RenderStats::IndirectReplayConfigured()) {
		InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
	}
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
	g_bound_cp = this;
}

void CommandProcessor::BufferFlush() {
	if (RenderStats::IndirectReplayConfigured()) {
		InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
	}
	// Downloads of the windows the game threads keep reading back, behind this submission's
	// writes, in the recording about to be submitted (BufferCache::EagerDownloads).
	{
		const CpuCost::Phase eager_phase_("Flush::EagerDownloads");
		m_renderer.GetBufferCache().EagerDownloads();
	}
	SetSubmitReasonIfUnset("BufferFlush (other)");
	GetScheduler().Flush();
}

void CommandProcessor::CompleteReleaseMemWrite() {
	GetScheduler().CompleteReleaseMemWrite();
}

void CommandProcessor::CompleteReleaseMemInterrupt() {
	GetScheduler().CompleteReleaseMemInterrupt();
}

void CommandProcessor::CompleteDraw() {
	// Item 2 (11-20). Inside a replayed multi-draw packet, the draw-flush interval's submit after
	// a sub-draw drawn in full takes the base that sub-draw just left: the next sub-draw finds
	// another command buffer ([multi-replay] command-changed, one base in 16 in every log) and is
	// drawn in full too. A full sub-draw that left a base is counted at the packet's end instead
	// (DrawIndirectMulti), as the replayed sub-draws already are. At most 15 a packet, so a packet
	// that never replays is submitted as before. Only when a submit happens moves, by one packet
	// at most. KYTY_MULTI_DRAW_DEFER_FLUSH=0: as before.
	if (m_multi_draw_defer && m_multi_draw_deferred < 15u &&
	    m_renderer.GetRenderExecutor().MultiDrawBaseValid()) {
		m_multi_draw_deferred++;
		return;
	}
	const auto tick = GetScheduler().CurrentTick();
	GetScheduler().CompleteDraw();
	if (RenderStats::IndirectReplayConfigured() && GetScheduler().CurrentTick() != tick) {
		InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
	}
}

void CommandProcessor::BufferWait() {
	if (RenderStats::IndirectReplayConfigured()) {
		InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
	}
	BufferInit();
	(KYTY_SUBMIT_REASON(), GetScheduler().Finish());
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

static thread_local uint32_t g_stat_queue = 0;
// KYTY_QUEUE_STATS=1 (stage 1 of an async-compute thread): per guest queue, the GPU thread's
// processing time, its dispatches' time split into program lookup (no cache bindings) and the
// rest, and its label waits split by which queue last wrote the label ([queue-stats], 5 s).
static bool QueueStatsEnabled() {
	static const bool enabled = std::getenv("KYTY_QUEUE_STATS") != nullptr;
	return enabled;
}
constexpr uint32_t QueueStatsCount = 1 + 7 * 8; // GuestGpu::QueueCount (private)
struct QueueStats {
	uint64_t process_ns[QueueStatsCount] {};
	uint64_t calls[QueueStatsCount] {};
	uint64_t incomplete[QueueStatsCount] {};
	uint64_t dispatches[QueueStatsCount] {};
	uint64_t dispatch_ns[QueueStatsCount] {};
	uint64_t program_ns[QueueStatsCount] {};
	// waits: [queue][writer class]: 0 same queue, 1 other GPU queue, 2 never written by a queue (CPU)
	uint64_t waits[QueueStatsCount][3] {};
	uint64_t waits_blocked[QueueStatsCount][3] {};
	uint64_t label_writes[QueueStatsCount] {};
	uint64_t prepare_ns[QueueStatsCount] {}, record_ns[QueueStatsCount] {};
	uint64_t parts_count[QueueStatsCount] {}; // dispatches whose parts were timed (direct ones that recorded)
	uint64_t dispatch_on_async[QueueStatsCount] {};
	uint64_t process_async_ns[QueueStatsCount] {};
	uint64_t same_shader[QueueStatsCount] {}, same_bindings[QueueStatsCount] {}, same_all[QueueStatsCount] {};
	uint64_t last_shader[QueueStatsCount] {}, last_bindings[QueueStatsCount] {}, last_srt[QueueStatsCount] {};
	// cross-queue waits by (waiting queue, writer queue), blocked only
	std::unordered_map<uint32_t, uint64_t> cross_pairs;
};
static QueueStats                                 g_queue_stats;
static std::unordered_map<uint64_t, uint8_t>      g_label_writer;
// Both command-processor threads with the async-compute thread.
static std::mutex                                 g_queue_stats_mutex;
static void QueueStatsNoteLabelWrite(uint64_t address) {
	if (!QueueStatsEnabled()) return;
	QueueShare::NoteGuestLabel(); // [queue-share]: the guest's own join points
	std::lock_guard stats_lock(g_queue_stats_mutex);
	g_label_writer[address & ~uint64_t {3}] = static_cast<uint8_t>(g_stat_queue);
	g_queue_stats.label_writes[g_stat_queue % QueueStatsCount]++;
}
static void QueueStatsNoteWait(uint64_t address, bool blocked) {
	if (!QueueStatsEnabled()) return;
	std::lock_guard stats_lock(g_queue_stats_mutex);
	const auto q  = g_stat_queue % QueueStatsCount;
	const auto it = g_label_writer.find(address & ~uint64_t {3});
	const int  writer_class = it == g_label_writer.end() ? 2 : (it->second == q ? 0 : 1);
	QueueShare::NoteGuestWait(writer_class == 1 && ((it->second != 0) != (q != 0)), !blocked);
	g_queue_stats.waits[q][writer_class]++;
	if (blocked) {
		g_queue_stats.waits_blocked[q][writer_class]++;
		if (writer_class == 1) {
			g_queue_stats.cross_pairs[(q << 8u) | it->second]++;
		}
	}
}
// The guest queue the calling thread is processing (renderCompute.cpp's [dispatch-delta]).
uint32_t QueueStatsQueue() {
	return g_stat_queue % QueueStatsCount;
}
void QueueStatsNoteBindings(uint64_t shader_hash, uint64_t bindings_hash, uint64_t srt_hash) {
	if (!QueueStatsEnabled()) return;
	std::lock_guard stats_lock(g_queue_stats_mutex);
	auto&      st = g_queue_stats;
	const auto q  = g_stat_queue % QueueStatsCount;
	if (st.last_shader[q] == shader_hash) {
		st.same_shader[q]++;
		if (st.last_bindings[q] == bindings_hash) {
			st.same_bindings[q]++;
			if (st.last_srt[q] == srt_hash) st.same_all[q]++;
		}
	}
	st.last_shader[q]   = shader_hash;
	st.last_bindings[q] = bindings_hash;
	st.last_srt[q]      = srt_hash;
}
void QueueStatsNoteDispatch(uint64_t program_ns, uint64_t total_ns) {
	if (!QueueStatsEnabled()) return;
	std::lock_guard stats_lock(g_queue_stats_mutex);
	const auto q = g_stat_queue % QueueStatsCount;
	g_queue_stats.dispatches[q]++;
	g_queue_stats.dispatch_ns[q] += total_ns;
	g_queue_stats.program_ns[q] += program_ns;
	g_queue_stats.dispatch_on_async[q] += GuestGpu::IsAsyncThread() ? 1u : 0u;
}
// A direct dispatch's parts after the program lookup (renderCompute.cpp): preparing the bindings
// (PrepareBindings to BindSharedMemory), and binding, committing and recording.
void QueueStatsNoteDispatchParts(uint64_t prepare_ns, uint64_t record_ns) {
	if (!QueueStatsEnabled()) return;
	std::lock_guard stats_lock(g_queue_stats_mutex);
	const auto q = g_stat_queue % QueueStatsCount;
	g_queue_stats.prepare_ns[q] += prepare_ns;
	g_queue_stats.record_ns[q] += record_ns;
	g_queue_stats.parts_count[q]++;
}
// KYTY_QUEUE_STATS, [run-breaks] (v438zd, stats only): why a dispatch run ended, by the packet
// NextRunDispatch stopped at. A run's last dispatch records a barrier after it; when another
// dispatch follows a few packets later ("then dispatch"), that barrier sits between two dispatches
// and the GPU drains there (the barrier-only gaps of KYTY_GPU_TIMING). Key: kind << 24 | opcode << 16
// | sub. kind 0: a type-3 packet that is not a run member (sub: the R code of a NOP, the event
// type of an EVENT_WRITE*, the first register of a SET_*_REG, else 0); 1: the command buffer
// ended (or its end is unknown); 2: not a type-3 packet; 3: a predicated packet; 4: a bad length.
struct RunBreakCount {
	uint64_t total         = 0;
	uint64_t then_dispatch = 0;
};
static std::mutex                        g_run_break_mutex;
static std::map<uint32_t, RunBreakCount> g_run_breaks;
static void QueueStatsNoteRunBreak(uint32_t key, bool then_dispatch) {
	std::lock_guard lock(g_run_break_mutex);
	auto&           count = g_run_breaks[key];
	count.total++;
	count.then_dispatch += then_dispatch ? 1u : 0u;
}
static void RunBreaksReport(double seconds) {
	std::vector<std::pair<uint32_t, RunBreakCount>> list;
	{
		std::lock_guard lock(g_run_break_mutex);
		list.assign(g_run_breaks.begin(), g_run_breaks.end());
		g_run_breaks.clear();
	}
	if (list.empty()) return;
	uint64_t total = 0, then_dispatch = 0;
	for (const auto& [key, count]: list) {
		total += count.total;
		then_dispatch += count.then_dispatch;
	}
	std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
		return a.second.then_dispatch != b.second.then_dispatch
		           ? a.second.then_dispatch > b.second.then_dispatch
		           : a.second.total > b.second.total;
	});
	std::string out;
	char        text[96];
	std::snprintf(text, sizeof(text), "[run-breaks] %.0f/s, then dispatch %.0f/s |", total / seconds,
	              then_dispatch / seconds);
	out += text;
	static constexpr const char* Kinds[] = {"op", "end", "type", "pred", "len"};
	for (size_t i = 0; i < list.size() && i < 14; i++) {
		const auto key  = list[i].first;
		const auto kind = std::min<uint32_t>(key >> 24u, 4u);
		if (kind == 0) {
			std::snprintf(text, sizeof(text), " op=0x%02x:0x%x %.0f (%.0f)", (key >> 16u) & 0xffu,
			              key & 0xffffu, list[i].second.total / seconds,
			              list[i].second.then_dispatch / seconds);
		} else {
			std::snprintf(text, sizeof(text), " %s %.0f (%.0f)", Kinds[kind],
			              list[i].second.total / seconds, list[i].second.then_dispatch / seconds);
		}
		out += text;
	}
	std::fprintf(stderr, "%s\n", out.c_str());
}

static void QueueStatsReport(double seconds) {
	if (!QueueStatsEnabled()) return;
	RunBreaksReport(seconds);
	std::lock_guard stats_lock(g_queue_stats_mutex);
	auto& st = g_queue_stats;
	for (uint32_t q = 0; q < QueueStatsCount; q++) {
		if (st.calls[q] == 0 && st.dispatches[q] == 0 && st.label_writes[q] == 0) continue;
		std::fprintf(stderr,
		             "[queue-stats] q=%u process=%.0f ms/s calls=%.0f/s incomplete=%.0f/s | dispatches=%.0f/s "
		             "%.0f ms/s (program %.0f ms/s) | label writes=%.0f/s | waits same=%.0f/%.0f other=%.0f/%.0f "
		             "cpu=%.0f/%.0f (/s, total/blocked) | repeat of previous dispatch: shader %.0f/s, +bindings %.0f/s, "
		             "+srt %.0f/s\n",
		             q, st.process_ns[q] / 1e6 / seconds, st.calls[q] / seconds, st.incomplete[q] / seconds,
		             st.dispatches[q] / seconds, st.dispatch_ns[q] / 1e6 / seconds,
		             st.program_ns[q] / 1e6 / seconds, st.label_writes[q] / seconds,
		             st.waits[q][0] / seconds, st.waits_blocked[q][0] / seconds, st.waits[q][1] / seconds,
		             st.waits_blocked[q][1] / seconds, st.waits[q][2] / seconds,
		             st.waits_blocked[q][2] / seconds, st.same_shader[q] / seconds, st.same_bindings[q] / seconds,
		             st.same_all[q] / seconds);
		if (st.dispatches[q] != 0 || st.process_async_ns[q] != 0) {
			std::fprintf(stderr,
			             "[queue-stats2] q=%u direct dispatch parts: prepare %.0f ms/s, bind+commit+record %.0f ms/s | "
			             "on the async thread: dispatches %.0f/s, process %.0f ms/s\n",
			             q, st.prepare_ns[q] / 1e6 / seconds, st.record_ns[q] / 1e6 / seconds,
			             st.dispatch_on_async[q] / seconds, st.process_async_ns[q] / 1e6 / seconds);
			// v438z+: the same per dispatch. "rest" is what neither the program lookup nor the two
			// timed parts cover, over all of the queue's dispatches (the prologue, run bookkeeping,
			// clears, every part of an indirect dispatch after its lookup).
			const double n     = static_cast<double>(std::max<uint64_t>(st.dispatches[q], 1));
			const double timed = static_cast<double>(std::max<uint64_t>(st.parts_count[q], 1));
			std::fprintf(stderr,
			             "[queue-stats3] q=%u per dispatch: all %.2f us (program %.2f us) over %.0f/s | timed direct %.0f/s: "
			             "prepare %.2f us, bind+commit+record %.2f us | rest %.0f ms/s\n",
			             q, st.dispatch_ns[q] / 1e3 / n, st.program_ns[q] / 1e3 / n, st.dispatches[q] / seconds,
			             st.parts_count[q] / seconds, st.prepare_ns[q] / 1e3 / timed, st.record_ns[q] / 1e3 / timed,
			             (static_cast<double>(st.dispatch_ns[q]) - static_cast<double>(st.program_ns[q]) -
			              static_cast<double>(st.prepare_ns[q]) - static_cast<double>(st.record_ns[q])) /
			                 1e6 / seconds);
		}
	}
	for (const auto& [key, count]: st.cross_pairs) {
		std::fprintf(stderr, "[queue-stats] blocked cross wait: q=%u waits on q=%u %.1f/s\n", key >> 8u,
		             key & 0xffu, count / seconds);
	}
	std::fflush(stderr);
	QueueStats next;
	std::copy(std::begin(st.last_shader), std::end(st.last_shader), std::begin(next.last_shader));
	std::copy(std::begin(st.last_bindings), std::end(st.last_bindings), std::begin(next.last_bindings));
	std::copy(std::begin(st.last_srt), std::end(st.last_srt), std::begin(next.last_srt));
	st = std::move(next);
}

// KYTY_HOT_LABEL_DEDUPE=0 turns this off. A wait loop polls the same label over and over until the
// CPU changes it, and every poll of a CPU-dirty page used to start a hot generation: ~57 a frame,
// each one making every hot page upload itself again on its next use. Only the read that sees a
// value the previous read did not is a new signal, so the unchanged polls start no generation. A
// label that went back to an earlier value between two polls would be missed here, but the game
// has to submit for that, and a submission starts a generation of its own.
static bool NoteLabelRead(uint64_t address, uint64_t value) {
	static const bool dedupe = [] {
		const char* text = std::getenv("KYTY_HOT_LABEL_DEDUPE");
		return text == nullptr || text[0] != '0';
	}();
	if (!dedupe) {
		return true;
	}
	struct Entry {
		uint64_t address = 0;
		uint64_t value   = 0;
		bool     valid   = false;
	};
	// The GPU thread alone reads labels; direct-mapped, so a label never evicts itself.
	static thread_local std::array<Entry, 64> cache {};
	auto&                                     entry = cache[(address >> 2u) & 63u];
	if (entry.valid && entry.address == address && entry.value == value) {
		g_hot_stats.label_skips.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	entry = Entry {address, value, true};
	g_hot_stats.label_bumps.fetch_add(1, std::memory_order_relaxed);
	return true;
}

template <typename T>
T CommandProcessor::ReadLabel(const volatile T* addr) const {
	const T value = GuestGpu::LabelsDeferred() ? m_renderer.GetGpu().ReadLabel(addr) : *addr;
	// A label the CPU wrote (its page is CPU-dirty) may follow data it wrote (KYTY_HOT_PAGES);
	// the GPU's own labels (end-of-pipe writes) do not start a generation.
	if (HotPagesEnabled()) {
		const auto address = reinterpret_cast<uint64_t>(addr) & ~uint64_t {3};
		if (m_renderer.GetBufferCache().IsRegionCpuModified(address, sizeof(T)) &&
		    NoteLabelRead(address, static_cast<uint64_t>(value))) {
			BumpHotGeneration();
		}
	}
	return value;
}

template uint32_t CommandProcessor::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t CommandProcessor::ReadLabel<uint64_t>(const volatile uint64_t*) const;

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	const bool passed = TestWaitRegMemValue(ReadLabel(addr), ref, mask, func);
	QueueStatsNoteWait(reinterpret_cast<uint64_t>(addr), !passed);
	if (!passed) {
		SuspendPm4();
	} else {
		g_bda_resync_seq.fetch_add(1, std::memory_order_relaxed);
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

// Diagnostics: KYTY_TRACE_WRITERS=<hex address>[:<hex size>]: CP writes (DMA_DATA, WRITE_DATA)
// overlapping that range are printed with their source ([cp-writers] on stderr).
static bool CpWriteTraced(uint64_t dst, uint64_t bytes) {
	static const std::pair<uint64_t, uint64_t> target = [] {
		std::pair<uint64_t, uint64_t> range {0, 0};
		if (const char* value = std::getenv("KYTY_TRACE_WRITERS"); value != nullptr) {
			char* end    = nullptr;
			range.first  = std::strtoull(value, &end, 16);
			range.second = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 4u;
		}
		return range;
	}();
	static std::atomic<uint32_t> printed {0};
	return target.first != 0 && dst < target.first + target.second && target.first < dst + bytes &&
	       printed.fetch_add(1) < 200u;
}

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}
	QueueStatsNoteLabelWrite(reinterpret_cast<uint64_t>(write_one_address ? dst : dst + dw_num - 1));
	LibKernel::GuestWait::NoteEvent(LibKernel::GuestWait::Event::Label); // written just below
	g_bda_resync_seq.fetch_add(1, std::memory_order_relaxed);
	SettleUploadReads();
	if (CpWriteTraced(reinterpret_cast<uint64_t>(dst), static_cast<uint64_t>(dw_num) * 4u)) {
		std::fprintf(stderr, "[cp-writers] WRITE_DATA dst=0x%012" PRIx64 " dwords=%u first=%08x one_addr=%d\n",
		             reinterpret_cast<uint64_t>(dst), dw_num, src[0], write_one_address ? 1 : 0);
	}

	// Labels often share a page with bytes the GPU wrote. A plain write then faults and downloads
	// the page, draining the GPU; WriteClean writes both copies instead.
	const CpuCost::Phase write_phase_("WriteData::Write");
	auto& cache = m_renderer.GetBufferCache();
	if (write_one_address) {
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), &src[dw_num - 1], sizeof(uint32_t))) {
			return;
		}
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
	} else {
		const auto bytes = static_cast<size_t>(dw_num) * sizeof(uint32_t);
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), src, bytes)) {
			return;
		}
		memcpy(dst, src, bytes);
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadGpuTimestamp();
	SettleUploadReads();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (!dst_gds && CpWriteTraced(dst_address_or_offset, num_bytes)) {
		uint32_t words[8] {};
		const bool src_mem = src_sel == 0 || src_sel == 3;
		const bool read = src_mem && Libs::LibKernel::Memory::TryReadBacking(
		                                 src_address_or_offset_or_immediate, words, std::min<uint32_t>(num_bytes, 32u));
		std::fprintf(stderr,
		             "[cp-writers] DMA_DATA dst=0x%012" PRIx64 " bytes=0x%x src_sel=%u src=0x%012" PRIx64
		             " src_gpu_dirty=%d src_words(%d)=%08x %08x %08x %08x %08x %08x %08x %08x\n",
		             dst_address_or_offset, num_bytes, src_sel, src_address_or_offset_or_immediate,
		             src_mem && buffer_cache.HasGpuDirtyBytes(src_address_or_offset_or_immediate, num_bytes) ? 1 : 0,
		             read ? 1 : 0, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
	}
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	// Every page the guest dirtied before submitting has bumped the epoch by now (write faults
	// run on the writing thread).
	submission.cpu_epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	if (m_draw_prep != nullptr) {
		// In queue order, as the GPU thread will run them.
		if (submission.type == SubmissionType::Graphics ||
		    submission.type == SubmissionType::Compute) {
			submission.prep_seq = ++m_prep_seq;
			m_draw_prep->EnqueueSubmission(submission.queue_id, submission.prep_seq,
			                               submission.commands);
		} else if (submission.type == SubmissionType::SuspendPoint) {
			m_draw_prep->EnqueueReset(submission.queue_id);
		}
	}
	if (submission.owned_words != nullptr) {
		// The draw-prep scanner can still be reading a submission the GPU thread has finished:
		// keep the copies of the last ones alive.
		static std::deque<std::shared_ptr<std::vector<uint32_t>>> recent;
		recent.push_back(submission.owned_words);
		if (recent.size() > 1024) {
			recent.pop_front();
		}
	}
	const auto role = RoleOf(submission.queue_id);
	if (LibKernel::GuestWait::Enabled()) [[unlikely]] {
		namespace GW = LibKernel::GuestWait;
		// The second argument: the signals below wake the GPU thread (role 0).
		GW::NoteSubmit(submission.type == SubmissionType::Graphics          ? GW::Submit::Gfx
		               : submission.type == SubmissionType::Compute         ? GW::Submit::Compute
		               : submission.type == SubmissionType::FlipPreparation ? GW::Submit::FlipPrep
		                                                                    : GW::Submit::SuspendPoint,
		               role != 1 || !AsyncThreadAlways());
	}
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_role_count[role]++;
	if (role == 1) {
		m_async_work.Signal();
		if (!AsyncThreadAlways()) {
			m_work_available.Signal(); // the GPU thread runs it outside the toggle's on windows
		}
	} else {
		m_work_available.Signal();
	}
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing[0] || m_processing[1] || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

// KYTY_SRT_DEFER: the tick the dispatch just suspended waits for; GuestGpu::ThreadRun moves it
// to the submission (GPU thread only).
static thread_local uint64_t g_srt_wait_tick = 0;

struct SrtDeferStats {
	uint64_t deferred   = 0; // dispatches suspended
	uint64_t ran_other  = 0; // submissions of other queues run while one waited
	uint64_t idle_waits = 0; // nothing else to run: the GPU thread waited for the tick
	uint64_t idle_us    = 0;
	// GPU thread time by state (KYTY_FPS_LOG): no work queued, every queue blocked, processing
	// submissions, running commands.
	uint64_t empty_us   = 0;
	uint64_t blocked_us = 0;
	uint64_t process_us = 0;
	uint64_t command_us = 0;
	uint64_t lock_us    = 0; // acquiring the queue lock
	uint64_t loop_us    = 0; // whole loop iterations
	uint64_t iters      = 0;
	uint64_t prog_ns = 0, prog_calls = 0, noprog_ns = 0, noprog_calls = 0, completes = 0;
	uint64_t blocked_waits = 0, cmd_count = 0;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
static thread_local SrtDeferStats g_srt_defer_stats;
// v445v (KYTY_SRT_IDLE_STATS=1, diagnostics only): the GPU thread's waits for an SRT download when
// no other guest queue has work ("Thread::WaitSrtDownload": 59 ms/s median in g450x, hidden in
// [floor]'s proc). Per 2 ms wait slice: how many command buffers the host GPU still had to finish
// before the download's tick when the slice began (1 = only the one the deferral itself submitted),
// whether this wait had to submit the open command buffer first, and whether the tick landed.
// [srt-idle] every 5 s.
static bool SrtIdleStats() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_IDLE_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
struct SrtIdleStatsState {
	uint64_t slices[5] {};    // behind 0, 1, 2, 3-4, 5+
	uint64_t ns[5] {};
	uint64_t flushed      = 0; // the open command buffer held the download: submitted here
	uint64_t flushed_ns   = 0;
	uint64_t landed       = 0; // the tick completed within the slice
	uint64_t waits        = 0; // first slices (a dispatch's wait that ran nothing else)
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
static thread_local SrtIdleStatsState g_srt_idle;
static thread_local uint64_t          g_srt_idle_last_tick = 0;
static void SrtIdleNote(uint64_t behind, bool flushed, uint64_t ns, bool landed, uint64_t tick) {
	auto&     st     = g_srt_idle;
	const int bucket = behind == 0 ? 0 : behind == 1 ? 1 : behind == 2 ? 2 : behind <= 4 ? 3 : 4;
	st.slices[bucket]++;
	st.ns[bucket] += ns;
	if (flushed) {
		st.flushed++;
		st.flushed_ns += ns;
	}
	st.landed += landed ? 1u : 0u;
	if (tick != g_srt_idle_last_tick) {
		st.waits++;
		g_srt_idle_last_tick = tick;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - st.last < std::chrono::seconds(5)) {
		return;
	}
	const double s = std::chrono::duration<double>(now - st.last).count();
	uint64_t     n = 0, t = 0;
	for (int i = 0; i < 5; i++) {
		n += st.slices[i];
		t += st.ns[i];
	}
	std::fprintf(stderr,
	             "[srt-idle] %.1fs: %.0f waits/s, %.0f slices/s, %.1f ms/s (landed in the slice %.0f%%) | "
	             "command buffers left before the tick at the slice's start: 0 %.0f/%.1fms 1 %.0f/%.1fms "
	             "2 %.0f/%.1fms 3-4 %.0f/%.1fms 5+ %.0f/%.1fms (per s) | open command buffer submitted by "
	             "the wait %.0f/s %.1f ms/s\n",
	             s, static_cast<double>(st.waits) / s, static_cast<double>(n) / s,
	             static_cast<double>(t) / 1e6 / s, n != 0 ? 100.0 * static_cast<double>(st.landed) / static_cast<double>(n) : 0.0,
	             static_cast<double>(st.slices[0]) / s, static_cast<double>(st.ns[0]) / 1e6 / s,
	             static_cast<double>(st.slices[1]) / s, static_cast<double>(st.ns[1]) / 1e6 / s,
	             static_cast<double>(st.slices[2]) / s, static_cast<double>(st.ns[2]) / 1e6 / s,
	             static_cast<double>(st.slices[3]) / s, static_cast<double>(st.ns[3]) / 1e6 / s,
	             static_cast<double>(st.slices[4]) / s, static_cast<double>(st.ns[4]) / 1e6 / s,
	             static_cast<double>(st.flushed) / s, static_cast<double>(st.flushed_ns) / 1e6 / s);
	st      = {};
	st.last = now;
}
struct SuspendSite {
	uint64_t count = 0;
	uint32_t queue = 0;
	uint32_t dw[8] {};
};
static thread_local std::unordered_map<const uint32_t*, SuspendSite> g_suspend_sites;
static void NoteSuspendSite(const uint32_t* packet, uint32_t remaining_dw) {
	static const bool log = Common::FpsLogDiagnostics();
	if (!log) return;
	auto& site = g_suspend_sites[packet];
	if (site.count++ == 0) {
		site.queue = g_stat_queue;
		for (uint32_t i = 0; i < 8 && i < remaining_dw; i++) site.dw[i] = packet[i];
	}
}

// KYTY_GPU_THREAD_STATS=1 (v312): the [gpu-thread] lines without the rest of the KYTY_FPS_LOG
// diagnostics (KYTY_FPS_ONLY=0 also changes the shader disk cache key).
static bool SrtStatsLogged() {
	static const bool log = Common::FpsLogDiagnostics() || std::getenv("KYTY_GPU_THREAD_STATS") != nullptr;
	return log;
}

static void ReportSrtDeferStats() {
	auto&      st  = g_srt_defer_stats;
	const auto now = std::chrono::steady_clock::now();
	if (now - st.last < std::chrono::seconds(5)) {
		return;
	}
	const bool log = SrtStatsLogged();
	if (log) {
		const double s = std::chrono::duration<double>(now - st.last).count();
		std::fprintf(stderr,
		             "[%s] window=%.2fs iters=%llu ms/s: loop=%.0f lock=%.0f empty=%.0f blocked=%.0f process=%.0f commands=%.0f | "
		             "srt-defer deferred=%.0f/s other-queue slices=%.0f/s idle=%.0f ms/s\n",
		             g_gpu_state != nullptr ? "gpu-thread" : "async-thread", s,
		             static_cast<unsigned long long>(st.iters),
		             static_cast<double>(st.loop_us) / 1e6 / s,
		             static_cast<double>(st.lock_us) / 1e6 / s,
		             static_cast<double>(st.empty_us) / 1e6 / s,
		             static_cast<double>(st.blocked_us) / 1e6 / s,
		             static_cast<double>(st.process_us) / 1e6 / s,
		             static_cast<double>(st.command_us) / 1e6 / s,
		             static_cast<double>(st.deferred) / s, static_cast<double>(st.ran_other) / s,
		             static_cast<double>(st.idle_us) / 1e6 / s);
	}
	if ((log || QueueStatsEnabled()) && AsyncThreadEnabled()) {
		const double s     = std::chrono::duration<double>(now - st.last).count();
		const auto   locks = RenderContext::TakeLockStats();
		std::fprintf(stderr,
		             "[%s-lock] renderer mutex: %.0f acquires/s, %.0f contended/s, waited %.0f ms/s, "
		             "off-lock windows %.0f ms/s\n",
		             g_gpu_state != nullptr ? "gpu" : "async", static_cast<double>(locks.acquires) / s,
		             static_cast<double>(locks.contended) / s,
		             static_cast<double>(locks.wait_ns) / 1e6 / s,
		             static_cast<double>(locks.released_ns) / 1e6 / s);
	}
	if (log) {
		const double s = std::chrono::duration<double>(now - st.last).count();
		std::fprintf(stderr,
		             "[gpu-thread2] progress: %.0f calls/s %.0f ms/s (completes %.0f/s) | no-progress: %.0f calls/s %.0f ms/s | all-blocked waits %.0f/s | loop commands %.0f/s (send %.0f, labels %.0f, sync %.0f)\n",
		             static_cast<double>(st.prog_calls) / s, static_cast<double>(st.prog_ns) / 1e6 / s,
		             static_cast<double>(st.completes) / s,
		             static_cast<double>(st.noprog_calls) / s, static_cast<double>(st.noprog_ns) / 1e6 / s,
		             static_cast<double>(st.blocked_waits) / s, static_cast<double>(st.cmd_count) / s,
		             static_cast<double>(g_cmd_from_send.exchange(0)) / s,
		             static_cast<double>(g_cmd_from_try.exchange(0)) / s,
		             static_cast<double>(g_cmd_from_sync.exchange(0)) / s);
		std::vector<std::pair<const uint32_t*, SuspendSite>> sites(g_suspend_sites.begin(),
		                                                           g_suspend_sites.end());
		std::sort(sites.begin(), sites.end(),
		          [](const auto& x, const auto& y) { return x.second.count > y.second.count; });
		for (size_t i = 0; i < sites.size() && i < 8; i++) {
			const auto& t = sites[i].second;
			std::fprintf(stderr,
			             "[suspend] %.0f/s q=%u op=0x%02x packet=%p dw: %08x %08x %08x %08x %08x %08x %08x %08x\n",
			             static_cast<double>(t.count) / s, t.queue, (t.dw[0] >> 8u) & 0xffu,
			             static_cast<const void*>(sites[i].first), t.dw[0], t.dw[1], t.dw[2], t.dw[3],
			             t.dw[4], t.dw[5], t.dw[6], t.dw[7]);
		}
		std::fprintf(stderr, "[suspend] distinct sites=%zu\n", sites.size());
	}
	g_suspend_sites.clear();
	if (g_gpu_state != nullptr) {
		QueueStatsReport(std::chrono::duration<double>(now - st.last).count());
	}
	st = {};
	st.last = now;
}

// The startup precompile (the shader ledger's replay, then the pipeline recipes) finishes before
// the GPU thread runs the game's first command: everything known is built before gameplay instead
// of the first time it is drawn (Mac 11-02; the overlay shows the progress meanwhile).
// KYTY_PRECOMPILE_WAIT=0: the game starts at once, as before.
static void WaitForStartupPrecompile() {
	struct Done { // play starts when this returns, by any path (zero-compile accounting)
		~Done() { GetShaderPrecompileProgress().startup_wait_done.store(true, std::memory_order_release); }
	} done_on_return;
	static const bool wait = [] {
		const char* value = std::getenv("KYTY_PRECOMPILE_WAIT");
		return !(value != nullptr && value[0] == '0');
	}();
	if (!wait) {
		return;
	}
	auto&      progress = GetShaderPrecompileProgress();
	const auto begin    = std::chrono::steady_clock::now();
	const auto elapsed  = [&] { return std::chrono::steady_clock::now() - begin; };
	// Step D4: the pipeline libraries of the shader list too, so a first start only links during
	// play (KYTY_GPL_WARM_WAIT=0: they are made in the background while the game runs).
	static const bool wait_libraries = [] {
		const char* value = std::getenv("KYTY_GPL_WARM_WAIT");
		return !(value != nullptr && value[0] == '0');
	}();
	const auto busy = [&] {
		// The replay's thread may not have started yet. (Checked even when there are no pipeline
		// recipes: their thread then reports finished at once, before the replay ran.)
		const auto finished_at = progress.finished_at.load(std::memory_order_acquire);
		if (progress.active.load(std::memory_order_acquire) || progress.importing.load(std::memory_order_acquire) ||
		    (elapsed() < std::chrono::seconds(1) && finished_at == 0)) {
			return true;
		}
		// The recipes are built after the replay.
		if (progress.pipelines_known.load(std::memory_order_acquire) &&
		    !progress.pipelines_finished.load(std::memory_order_acquire)) {
			return true;
		}
		return wait_libraries && finished_at != 0 && !progress.gpl_warm_finished.load(std::memory_order_acquire);
	};
	if (!busy()) {
		return;
	}
	LOGF("GPU: waiting for the startup shader and pipeline precompile (KYTY_PRECOMPILE_WAIT=0: don't)\n");
	progress.skip_offered.store(true, std::memory_order_release);
	struct SkipOff {
		~SkipOff() {
			auto& p = GetShaderPrecompileProgress();
			p.skip_offered.store(false, std::memory_order_release);
			p.skip_state.store(0, std::memory_order_release);
		}
	} skip_off;
	// A first start with the shipped list compiles for a long time: wait while it advances, and
	// stop waiting only after 10 minutes without progress (it was a fixed 30 minutes).
	auto last_count    = ~uint64_t {0};
	auto last_progress = std::chrono::steady_clock::now();
	while (busy()) {
		if (progress.skip_state.load(std::memory_order_acquire) == 2) {
			// The player skipped (Esc, then Y): the rest is made in the background during play.
			progress.skipped.store(true, std::memory_order_release);
			LOGF("GPU: startup precompile skipped by the player; the rest continues in the background\n");
			std::fprintf(stderr, "[precompile-skip] skipped by the player after %.1f s\n",
			             std::chrono::duration<double>(elapsed()).count());
			break;
		}
		const uint64_t count = uint64_t {progress.done.load()} + progress.import_done.load() +
		                       progress.pipelines_done.load() + progress.gpl_warm_done.load();
		const auto now = std::chrono::steady_clock::now();
		if (count != last_count) {
			last_count    = count;
			last_progress = now;
		} else if (now - last_progress > std::chrono::minutes(10)) {
			LOGF("GPU: startup precompile made no progress for 10 minutes; the game starts\n");
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	const auto seconds = std::chrono::duration<double>(elapsed()).count();
	LOGF("GPU: startup precompile done after %.1f s (shaders %u of %u, %u compiled; pipelines %u of %u; libraries %u of %u)\n",
	     seconds, progress.done.load(), progress.total.load(), progress.compiled.load(),
	     progress.pipelines_done.load(), progress.pipelines_total.load(), progress.gpl_warm_done.load(),
	     progress.gpl_warm_total.load());
	std::fprintf(stderr, "[precompile-wait] %.1f s\n", seconds);
}

uint32_t GuestGpu::RoleOf(uint32_t queue_id) noexcept {
	return AsyncThreadEnabled() && queue_id != 0 ? 1u : 0u;
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	ThreadRunRole(gpu, 0);
}

void GuestGpu::ThreadRunRole(GuestGpu* gpu, uint32_t role) {
	EXIT_IF(gpu == nullptr || role > 1);
	if (role == 0) {
		KYTY_PROFILER_THREAD("Thread_Gpu");
		Common::PreferPerformanceCores("GPU thread");
		Profiler::t_gpu_thread = true;
		g_gpu_state            = gpu; // commands run on role 0 only
		Common::Sampler::RegisterCurrentThread("gpu");
		WaitForStartupPrecompile();
	} else {
		KYTY_PROFILER_THREAD("Thread_GpuAsync");
		Common::PreferPerformanceCores("GPU async-compute thread");
		Common::Sampler::RegisterCurrentThread("gpu-async");
	}
	Common::RaiseGpuThreadPriority();
	g_gpu_thread = true;
	SetBarrierMergeThread();
	const bool     async       = AsyncThreadEnabled();
	// KYTY_SYNC_TOGGLE_MODE=async: which thread runs the compute queues changes with the toggle
	// window, so idle waits are short (the toggle wakes nobody).
	const bool     toggled     = async && !AsyncThreadAlways();
	// The queues this role runs: [first_queue, first_queue + queue_count). Role 1 runs the compute
	// queues while the async thread is active; role 0 runs the graphics queue, and the compute
	// queues too while it is not.
	const uint32_t first_queue = role == 1 ? 1u : 0u;
	const auto     queue_count_now = [async, role] {
		if (role == 1) {
			return AsyncThreadActive() ? QueueCount - 1u : 0u;
		}
		return async && AsyncThreadActive() ? 1u : QueueCount;
	};
	auto&          work        = role == 1 ? gpu->m_async_work : gpu->m_work_available;
	// Under m_queue_mutex: commands (role 0) or submissions of this role's queues are waiting.
	const auto has_work = [gpu, role, async] {
		if (role == 1) {
			return AsyncThreadActive() && gpu->m_role_count[1] != 0;
		}
		return gpu->m_role_count[0] != 0 || !gpu->m_commands.empty() ||
		       (async && !AsyncThreadActive() && gpu->m_role_count[1] != 0);
	};
	const auto wait_work = [&work, gpu, toggled] {
		if (toggled) {
			work.WaitFor(&gpu->m_queue_mutex, 10);
		} else {
			work.Wait(&gpu->m_queue_mutex);
		}
	};
	// Under m_queue_mutex: a completed slice may have written labels the other role waits for.
	const auto unblock_all = [gpu, async, role] {
		for (auto& queue: gpu->m_queues) {
			if (!queue.empty()) {
				queue.front().blocked = false;
			}
		}
		if (async) {
			(role == 0 ? gpu->m_async_work : gpu->m_work_available).Signal();
		}
	};

	auto loop_mark = std::chrono::steady_clock::now();
	for (;;) {
		// v445o (KYTY_TIMERS_CUT; on by default since v445p, =0 off; toggle mode "timers"): this loop read the clock
		// seven times per iteration for statistics, whoever read them. With the cut on a time is
		// taken only when something uses it: the [gpu-thread] lines (srt_timed), [floor]'s thread
		// times, [queue-stats]. Decided once per iteration: a begin and its end agree. Without
		// the [gpu-thread] lines nothing reads g_srt_defer_stats, and its 5 s report (which then
		// prints and resets nothing else) is skipped.
		const bool timers_cut  = SyncToggle::TimersCut();
		const bool srt_timed   = !timers_cut || SrtStatsLogged();
		const bool floor_timed = srt_timed || Common::FloorStats::Enabled();
		const bool srt_report  = srt_timed || QueueStatsEnabled();
		SyncToggle::Bump(srt_timed ? SyncToggle::Hits().loop_timed : SyncToggle::Hits().loop_untimed);
		if (srt_timed) {
			const auto now = std::chrono::steady_clock::now();
			g_srt_defer_stats.loop_us += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(now - loop_mark).count());
			loop_mark = now;
		}
		g_srt_defer_stats.iters++;
		// KYTY_CPU_COST: everything this thread does is under this phase; its self time is the
		// loop's own bookkeeping, the waits for work are "Thread::Wait..." ([cpu-total]).
		const CpuCost::Phase         loop_phase_("Thread::Loop");
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			const auto lock_begin = srt_timed ? std::chrono::steady_clock::now()
			                                  : std::chrono::steady_clock::time_point {};
			std::optional<CpuCost::Phase> lock_phase_(std::in_place, "Thread::QueueLock");
			Common::LockGuard lock(gpu->m_queue_mutex);
			lock_phase_.reset();
			if (srt_timed) {
				g_srt_defer_stats.lock_us += static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now() - lock_begin)
				        .count());
			}
			while (!has_work() && !gpu->m_stopping) {
				gpu->m_processing[role] = false;
				gpu->m_idle.Signal();
				Timeline::Mark("gpu-idle");
				const auto empty_begin = std::chrono::steady_clock::now();
				if (role == 0) {
					LibKernel::GuestWait::GpuEmptyBegin(); // KYTY_GUEST_WAIT_STATS: who ends this wait
				}
				{
					const CpuCost::Phase wait_phase_("Thread::WaitEmpty");
					wait_work();
				}
				if (role == 0) {
					LibKernel::GuestWait::GpuEmptyEnd();
				}
				{
					const auto empty_ns = static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(
					        std::chrono::steady_clock::now() - empty_begin)
					        .count());
					g_srt_defer_stats.empty_us += empty_ns;
					if (Common::FloorStats::Enabled() && g_gpu_state != nullptr) {
						Common::FloorStats::Add(Common::FloorStats::Get().thread_empty_ns, empty_ns);
					}
				}
				Timeline::Mark("gpu-wake");
			}
			if (gpu->m_stopping && !has_work()) {
				gpu->m_processing[role] = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (role == 0 && !gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing[role] = true;
			} else {
				const uint32_t queue_count = queue_count_now();
				if (queue_count == 0) {
					continue; // the window changed since has_work()
				}
				int      selected_queue = -1;
				bool     tick_pending   = false;
				uint64_t pending_tick   = UINT64_MAX;
				auto& scheduler     = gpu->m_renderer.GetCommandScheduler();
				// KYTY_GFX_PRIORITY (GfxPriority): the graphics queue is looked at first, then the
				// compute queues round-robin as before.
				const bool gfx_first = first_queue == 0 && queue_count > 1 && GfxPriority();
				for (uint32_t offset = 0; offset < queue_count; offset++) {
					uint32_t id = 0;
					if (gfx_first) {
						if (offset != 0) {
							// Round-robin over queues 1..queue_count-1 from where it stopped.
							const auto others = queue_count - 1u;
							const auto start  = gpu->m_next_queue[role] == 0 ? 0u : gpu->m_next_queue[role] - 1u;
							id = 1u + (start + offset - 1u) % others;
						}
					} else {
						id = first_queue + (gpu->m_next_queue[role] - first_queue + offset) % queue_count;
					}
					if (gpu->m_queues[id].empty() || gpu->m_queue_busy[id]) {
						continue;
					}
					auto& front = gpu->m_queues[id].front();
					if (front.srt_wait_tick != 0) {
						// A dispatch suspended for SRT bytes to land (KYTY_SRT_DEFER).
						if (!scheduler.IsFree(front.srt_wait_tick)) {
							tick_pending = true;
							pending_tick = std::min(pending_tick, front.srt_wait_tick);
							continue;
						}
						front.blocked = false;
					}
					if (!front.blocked && selected_queue < 0) {
						selected_queue = static_cast<int>(id);
						if (tick_pending || !PipelineCache::SrtDeferEnabled()) {
							break;
						}
					}
				}
				if (selected_queue < 0 && tick_pending) {
					// Nothing else to run: poll for the downloads (and new submissions).
					gpu->m_processing[role] = false;
					const CpuCost::Phase wait_phase_("Thread::WaitSrtDownload");
					const auto wait_begin = std::chrono::steady_clock::now();
					// Wait for the GPU itself (2 ms slices, the queue unlocked so the guest can
					// submit), not for a new submission: nothing signals m_work_available when a
					// download lands, so a 50 ms wait sat out most of it when no other work came
					// (KYTY_SRT_DEFER_IDLE_LEGACY=1: the old wait).
					static const bool legacy_idle = [] {
						const char* value = std::getenv("KYTY_SRT_DEFER_IDLE_LEGACY");
						return value != nullptr && value[0] == '1';
					}();
					if (legacy_idle || pending_tick == UINT64_MAX) {
						work.WaitFor(&gpu->m_queue_mutex, 50);
					} else {
						gpu->m_queue_mutex.Unlock();
						bool idle_flushed = false;
						if (scheduler.Active() && pending_tick >= scheduler.CurrentTick()) {
							RendererScope renderer_scope(gpu->m_renderer);
							(KYTY_SUBMIT_REASON(), scheduler.Flush()); // the download is still in the open command buffer
							idle_flushed = true;
						}
						// v445v (KYTY_SRT_IDLE_STATS): how far the host GPU still was from the tick.
						uint64_t idle_behind = 0;
						if (SrtIdleStats()) {
							scheduler.GetMasterSemaphore().Refresh();
							const auto known = scheduler.GetMasterSemaphore().KnownGpuTick();
							idle_behind      = pending_tick > known ? pending_tick - known : 0;
						}
						const auto idle_slice_begin = std::chrono::steady_clock::now();
						const bool idle_landed      = scheduler.WaitFor(pending_tick, 2'000'000);
						if (SrtIdleStats()) {
							SrtIdleNote(idle_behind, idle_flushed,
							            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
							                                      std::chrono::steady_clock::now() - idle_slice_begin)
							                                      .count()),
							            idle_landed, pending_tick);
						}
						gpu->m_queue_mutex.Lock();
					}
					g_srt_defer_stats.idle_waits++;
					g_srt_defer_stats.idle_us += static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(
					        std::chrono::steady_clock::now() - wait_begin)
					        .count());
					continue;
				}
				if (selected_queue >= 0 && tick_pending) {
					g_srt_defer_stats.ran_other++;
				}
				if (selected_queue < 0) {
					gpu->m_processing[role] = false;
					Timeline::Mark("gpu-blocked");
					const auto blocked_begin = std::chrono::steady_clock::now();
					g_srt_defer_stats.blocked_waits++;
					{
						const CpuCost::Phase wait_phase_("Thread::WaitBlocked");
						work.WaitFor(&gpu->m_queue_mutex, toggled ? 10 : 100);
					}
					{
						const auto blocked_ns = static_cast<uint64_t>(
						    std::chrono::duration_cast<std::chrono::nanoseconds>(
						        std::chrono::steady_clock::now() - blocked_begin)
						        .count());
						g_srt_defer_stats.blocked_us += blocked_ns;
						if (Common::FloorStats::Enabled() && g_gpu_state != nullptr) {
							Common::FloorStats::Add(Common::FloorStats::Get().thread_blocked_ns, blocked_ns);
						}
					}
					Timeline::Mark("gpu-unblocked");
					for (uint32_t id = first_queue; id < first_queue + queue_count; id++) {
						auto& queue = gpu->m_queues[id];
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_role_count[RoleOf(static_cast<uint32_t>(selected_queue))]--;
				gpu->m_queue_busy[static_cast<uint32_t>(selected_queue)] = true;
				// With the graphics queue first, the compute round-robin keeps its own place.
				if (!(gfx_first && selected_queue == 0)) {
					gpu->m_next_queue[role] = static_cast<uint32_t>(selected_queue) + 1u;
				}
				gpu->m_processing[role] = true;
				has_submission          = true;
			}
		}
		if (should_stop) {
			if (role == 0) {
				RendererScope renderer_scope(gpu->m_renderer);
				gpu->m_gfx_cp->BufferWait();
			}
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			const auto command_begin = floor_timed ? std::chrono::steady_clock::now()
			                                       : std::chrono::steady_clock::time_point {};
			{
				const CpuCost::Phase command_phase_("Thread::Command");
				RendererScope renderer_scope(gpu->m_renderer);
				command();
			}
			g_srt_defer_stats.cmd_count++;
			if (floor_timed) {
				const auto command_ns = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now() - command_begin)
				        .count());
				g_srt_defer_stats.command_us += command_ns;
				if (Common::FloorStats::Enabled() && g_gpu_state != nullptr) {
					Common::FloorStats::Add(Common::FloorStats::Get().thread_command_ns, command_ns);
				}
			}
			if (srt_report) {
				ReportSrtDeferStats();
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing[role] = false;
			if (!has_work()) {
				gpu->m_idle.SignalAll();
			}
			// A label write (they run as commands) may release the async thread's waits.
			if (async) {
				unblock_all();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		if (submission.srt_wait_tick != 0) {
			// Its downloads are complete; their publication runs as a priority operation.
			gpu->m_renderer.GetCommandScheduler().WaitPriorityOperations(
			    std::exchange(submission.srt_wait_tick, 0));
		}
		g_srt_wait_tick     = 0;
		const bool process_timed = floor_timed || QueueStatsEnabled();
		const auto process_begin = process_timed ? std::chrono::steady_clock::now()
		                                         : std::chrono::steady_clock::time_point {};
		g_stat_queue = submission.queue_id;
		GpuTiming::SetQueue(submission.queue_id);
		bool complete = false;
		{
			const CpuCost::Phase process_phase_("Thread::Process");
			complete = gpu->Process(submission);
		}
		{
			const auto ns = process_timed
			                    ? static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                                std::chrono::steady_clock::now() - process_begin)
			                                                .count())
			                    : uint64_t {0};
			if (QueueStatsEnabled()) {
				std::lock_guard stats_lock(g_queue_stats_mutex);
				const auto q = submission.queue_id % QueueCount;
				g_queue_stats.process_ns[q] += ns;
				g_queue_stats.calls[q]++;
				if (!complete) g_queue_stats.incomplete[q]++;
				if (role == 1) g_queue_stats.process_async_ns[q] += ns;
			}
			g_srt_defer_stats.process_us += ns;
			if (Common::FloorStats::Enabled() && g_gpu_state != nullptr) {
				Common::FloorStats::Add(Common::FloorStats::Get().thread_process_ns, ns);
			}
			const bool progress = submission.command_execution.MadeProgress() ||
			                      submission.constant_execution.MadeProgress();
			if (progress) {
				g_srt_defer_stats.prog_ns += ns;
				g_srt_defer_stats.prog_calls++;
				if (complete) g_srt_defer_stats.completes++;
			} else {
				g_srt_defer_stats.noprog_ns += ns;
				g_srt_defer_stats.noprog_calls++;
			}
		}
		const auto srt_tick = std::exchange(g_srt_wait_tick, 0);
		if (srt_report) {
			ReportSrtDeferStats();
		}

		const auto lock2_begin = srt_timed ? std::chrono::steady_clock::now()
		                                   : std::chrono::steady_clock::time_point {};
		std::optional<CpuCost::Phase> lock2_phase_(std::in_place, "Thread::QueueLock");
		Common::LockGuard lock(gpu->m_queue_mutex);
		lock2_phase_.reset();
		if (srt_timed) {
			g_srt_defer_stats.lock_us += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - lock2_begin)
			        .count());
		}
		gpu->m_queue_busy[submission.queue_id] = false;
		if (!complete) {
			submission.blocked       = true;
			submission.srt_wait_tick = srt_tick;
			const auto queue_id      = submission.queue_id;
			gpu->m_queues[queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
			gpu->m_role_count[RoleOf(queue_id)]++;
			if (toggled) {
				// The other thread may run this queue now (the window changed).
				unblock_all();
			}
		} else {
			unblock_all();
		}
		gpu->m_processing[role] = false;
		if (!has_work()) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::HasReadySubmission() {
	Common::LockGuard lock(m_queue_mutex);
	for (const auto& queue: m_queues) {
		if (!queue.empty() && !queue.front().blocked) {
			return true;
		}
	}
	return false;
}

// KYTY_COALESCE_SUBMITS=1: a complete guest submission with another one ready behind it is not
// submitted to the host GPU on its own; the next one records into the same command buffer (the
// periodic draw flush still submits every 16 draws). ~108 of the ~160 submissions per frame in
// the jungle were these, 2026-10-18.
static bool CoalesceSubmits() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COALESCE_SUBMITS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_HOT_PAGES: a new generation when a submission starts (the game submits after writing);
// every 60 s every hot page goes back to faulting once (KYTY_FPS_LOG prints [hot-pages]).
static void HotPagesSlice(bool first_slice) {
	if (!HotPagesEnabled()) {
		return;
	}
	if (first_slice) {
		BumpHotGeneration();
	}
	static auto last       = std::chrono::steady_clock::now();
	static auto last_epoch = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	const auto  s    = std::chrono::duration<double>(now - last).count();
	if (s < 5.0) {
		return;
	}
	last = now;
	if (now - last_epoch > std::chrono::seconds(60)) {
		last_epoch = now;
		g_hot_epoch.fetch_add(1, std::memory_order_relaxed);
	}
	if (static const bool log = Common::FpsLogDiagnostics(); log) {
		std::fprintf(stderr,
		             "[hot-pages] faults %.0f/s, promoted %.0f/s, uploads %.0f/s, repeats skipped "
		             "%.0f/s, demoted by GPU writes %.0f/s, generations %.0f/s | label bumps %.0f/s "
		             "polls skipped %.0f/s\n",
		             g_hot_stats.faults.exchange(0) / s, g_hot_stats.promoted.exchange(0) / s,
		             g_hot_stats.uploads.exchange(0) / s, g_hot_stats.repeats.exchange(0) / s,
		             g_hot_stats.demoted_by_gpu.exchange(0) / s,
		             [&] {
			             static uint32_t previous = 0;
			             const auto      current  = g_hot_generation.load();
			             const auto      delta    = current - previous;
			             previous                 = current;
			             return delta / s;
		             }(),
		             g_hot_stats.label_bumps.exchange(0) / s,
		             g_hot_stats.label_skips.exchange(0) / s);
	}
}

bool GuestGpu::Process(Submission& submission) {
	HotPagesSlice(!submission.started);
	const bool first_slice = !submission.started;
	g_run_expected_next    = nullptr; // a run continues only within one slice of one submission
	g_run_last_stop        = BarrierReason::WhySlice;
	auto&      cp          = GetProcessor(submission.queue_id);
	Timeline::Mark("proc-begin", submission.queue_id, static_cast<uint64_t>(submission.type));
	struct EndMark {
		const Submission& submission;
		~EndMark() {
			Timeline::Mark("proc-end", submission.queue_id,
			               submission.command_complete ? 1u : 0u);
		}
	} end_mark {submission};

	{
		const CpuCost::Phase begin_phase_("Process::Begin");
		RendererScope renderer_scope(m_renderer);
		g_bda_submission_cpu_epoch.store(submission.cpu_epoch, std::memory_order_relaxed);
		if (first_slice) {
			g_guest_submission_seq.fetch_add(1, std::memory_order_relaxed);
			// Texture-heap entries the guest wrote before submitting are visible to all of this
			// submission's draws, as on the console.
			m_renderer.GetRenderExecutor().SyncBindlessHeaps();
		}

		if (first_slice) {
			submission.started = true;
			cp.SetSubmitId(++m_submit_id);
			cp.ResetDeCe();
			cp.SetFlip({});
		}

		cp.BufferInit();
	}
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			submission.command_execution.SetPrepSeq(submission.prep_seq, submission.queue_id);
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			const CpuCost::Phase end_phase_("Process::End");
			RendererScope renderer_scope(m_renderer);
			if (progressed) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				if (!(complete && CoalesceSubmits() && HasReadySubmission())) {
					SetSubmitReason("end of graphics submission");
					cp.BufferFlush();
				}
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			submission.command_execution.SetPrepSeq(submission.prep_seq, submission.queue_id);
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
				g_compute_submissions.fetch_add(1, std::memory_order_relaxed);
				g_compute_submission_dw.fetch_add(num_dw, std::memory_order_relaxed);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			const CpuCost::Phase end_phase_("Process::End");
			RendererScope renderer_scope(m_renderer);
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				if (!(complete && CoalesceSubmits() && HasReadySubmission())) {
					SetSubmitReason("end of compute submission");
					cp.BufferFlush();
				}
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation: {
			const CpuCost::Phase flip_phase_("Process::FlipPreparation");
			RendererScope renderer_scope(m_renderer);
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		}
		case SubmissionType::SuspendPoint: {
			const CpuCost::Phase suspend_phase_("Process::SuspendPoint");
			RendererScope renderer_scope(m_renderer);
			cp.EmitGuestBarrier();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			SetSubmitReason("suspend point");
			cp.BufferFlush();
			cp.Reset();
			break;
		}
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (RenderStats::IndirectReplayConfigured()) {
		InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
	}
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	if (t_segment_held || RenderExecutor::QueuedDispatchCount() != 0) {
		AsyncSegmentLeave();
	}
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

bool CommandProcessor::BeginDispatchDefer() {
	if (!PipelineCache::SrtDeferEnabled() || g_current_execution == nullptr ||
	    g_current_execution->m_buffer_stack.empty()) {
		PipelineCache::BeginSrtDefer(false);
		return false;
	}
	auto&       execution = *g_current_execution;
	const auto& cursor    = execution.m_buffer_stack.back();
	const auto* packet    = cursor.commands.data() + cursor.offset_dw;
	if (execution.m_deferred_packet != packet) {
		execution.m_deferred_packet = nullptr;
		execution.m_defer_count     = 0;
		execution.m_deferred_tick   = 0;
	} else if (execution.m_deferred_tick != 0 && !GetScheduler().IsFree(execution.m_deferred_tick)) {
		// Run again before its downloads landed (the constant engine made progress): still waiting.
		g_srt_wait_tick = std::max(g_srt_wait_tick, execution.m_deferred_tick);
		SuspendPm4();
		return true;
	}
	// A packet run again after its downloads landed may find newer GPU writes (another queue
	// ran meanwhile); after a few rounds it waits as before rather than chase them.
	PipelineCache::BeginSrtDefer(execution.m_defer_count < 4);
	return false;
}

void CommandProcessor::EndDispatchDefer() {
	const auto tick = PipelineCache::EndSrtDefer();
	if (tick == 0 || g_current_execution == nullptr || g_current_execution->m_buffer_stack.empty()) {
		return;
	}
	auto&       execution       = *g_current_execution;
	const auto& cursor          = execution.m_buffer_stack.back();
	execution.m_deferred_packet = cursor.commands.data() + cursor.offset_dw;
	execution.m_deferred_tick   = tick;
	execution.m_defer_count++;
	g_srt_wait_tick = std::max(g_srt_wait_tick, tick);
	g_srt_defer_stats.deferred++;
	SuspendPm4();
}

static bool IndirectReplayPacketKeepsBase(const uint32_t* packet, uint32_t remaining_dw) {
	const auto header = packet[0];
	if (header == 0x80000000u) return true;
	if ((header >> 30u) != 3u || (header & 1u) != 0) return false;
	const auto packet_dw = KYTY_PM4_LEN(header);
	if (packet_dw < 2u || packet_dw > remaining_dw) return false;
	const auto opcode = (header >> 8u) & 0xffu;
	if (opcode == Pm4::IT_NOP) {
		const auto subop = KYTY_PM4_R(header);
		return subop == Pm4::R_ZERO || subop == Pm4::R_PUSH_MARKER ||
		       subop == Pm4::R_POP_MARKER;
	}
	if (opcode == Pm4::IT_DRAW_INDIRECT) {
		return GpuIndirectDraws() && header == 0xc0032400u && packet_dw == 5u &&
		       (packet[4] & ~0x20u) == 2u;
	}
	// v326: context register writes (direct or from memory) change only the Context, which replay
	// compares byte for byte with the base's; primitive/index type writes change the user config
	// (compared) or the index type (unused by these non-indexed draws); SET_BASE changes argument
	// bases, and replay checks each draw's converted arguments by address.
	if (opcode == Pm4::IT_SET_CONTEXT_REG || opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT) {
		return true;
	}
	// User-config writes (direct, indexed or from memory) change only the UserConfig (compared) or
	// ignore the value, except VGT_INDEX_TYPE, which sets the command processor's index type
	// (unused by these non-indexed draws; an indexed draw is never replayed).
	if (opcode == Pm4::IT_SET_UCONFIG_REG || opcode == Pm4::IT_SET_UCONFIG_REG_INDEX ||
	    opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT) {
		return true;
	}
	if (opcode == Pm4::IT_SET_BASE) {
		return packet_dw == 4u && (packet[1] & 0xfu) == 1u;
	}
	// v328: SH register loads from memory change only the shader registers (cp.GetShCtx()), which
	// replay compares (user SGPR values through its own checks, the rest byte for byte).
	if (opcode == Pm4::IT_SET_SH_REG_INDIRECT ||
	    (opcode == Pm4::IT_SET_SH_REG && KYTY_PM4_R(header) == Pm4::R_ZERO)) {
		return true;
	}
	if (opcode != Pm4::IT_SET_SH_REG || KYTY_PM4_R(header) != Pm4::R_ZERO) return false;
	const auto reg = packet[1];
	const auto count = packet_dw - 2u;
	if (count == 0) return false;
	for (uint32_t i = 0; i < count; i++) {
		const auto current = reg + i;
		const bool ps_user = current >= Pm4::SPI_SHADER_USER_DATA_PS_0 &&
		                     current <= Pm4::SPI_SHADER_USER_DATA_PS_31;
		const bool gs_user = current >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
		                     current <= Pm4::SPI_SHADER_USER_DATA_GS_31;
		if (!ps_user && !gs_user) return false;
	}
	return true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	// KYTY_DRAW_PIPE: what runs between slices (submission ends, other queues) needs the renderer.
	struct DrawPipeSliceEnd {
		~DrawPipeSliceEnd() {
			DrawPipe::SetDrawPacket(false);
			DrawPipe::Drain(DrawPipe::Reason::Slice);
		}
	} draw_pipe_slice_end;
	const bool draw_pipe = DrawPipe::Enabled();
	const bool draw_prep = execution.m_prep_seq != 0 && DrawPrep::Enabled();
	if (DrawPrep::Enabled() && !draw_prep) {
		DrawPrep::ClearCurrent();
	}
	// KYTY_CPU_COST: the packet loop's own work (headers, skips, replay checks) is this phase's
	// self time; each packet's handler runs under "PM4 <opcode>" (Pm4PhaseName).
	// KYTY_CPU_COST_PACKETS=0: no phase per packet (register writes are most packets; [cpu-total]
	// prints what the records cost); their time is then "Pm4::Loop"'s.
	const CpuCost::Phase loop_phase_("Pm4::Loop");
	static const bool    packet_phases = [] {
        const char* value = std::getenv("KYTY_CPU_COST_PACKETS");
        return value == nullptr || value[0] != '0';
	}();
	const bool cpu_cost = packet_phases && CpuCost::Enabled();
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			if (RenderStats::IndirectReplayConfigured()) {
				InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketBoundary);
			}
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if ((packet_header >> 30u) == 0u) {
			// A type-0 packet writes COUNT+1 registers from a base index; the command processor
			// ignores them, and games leave reserved dwords as zeros, which decode to this
			// (a zero header is one register write of 0 to index 0).
			const auto packet_dw = ((packet_header >> 16u) & 0x3fffu) + 2u;
			static std::atomic<uint32_t> type0_log_count {0};
			if (type0_log_count.fetch_add(1) < 16) {
				LOGF("CP: skipping type-0 packet header=0x%08" PRIx32 " len=%" PRIu32
				     " at offset=0x%05" PRIx32 "\n",
				     packet_header, packet_dw, total_dw - remaining_dw);
			}
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			if (RenderStats::IndirectReplayConfigured()) {
				InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketState);
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		// Draw prep: number the draw and dispatch packets as the scanner does (predicated ones too;
		// a packet suspended and run again keeps its number).
		if (draw_prep && DrawPrep::CountsAsDrawOrDispatch(opcode)) {
			if (execution.m_prep_packet != packet) {
				execution.m_prep_packet = packet;
				execution.m_prep_ordinal++;
			}
			DrawPrep::SetCurrent({.queue   = execution.m_prep_queue,
			                      .seq     = execution.m_prep_seq,
			                      .ordinal = execution.m_prep_ordinal - 1u,
			                      .packet  = packet});
		}

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}
		if (RenderStats::IndirectReplayConfigured() && !RenderStats::IndirectReplayActive()) {
			InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketState);
		} else if (RenderStats::IndirectReplayConfigured() &&
		           !IndirectReplayPacketKeepsBase(packet, remaining_dw)) {
			if (RenderStats::g_indirect_base_maybe.load(std::memory_order_relaxed)) {
				// Which packets end a replay base ([indirect-replay] stops=...): opcode, and the first
				// register of register writes.
				const uint32_t op  = (packet[0] >> 8u) & 0xffu;
				const bool     reg = op == Pm4::IT_SET_SH_REG || op == Pm4::IT_SET_CONTEXT_REG ||
				                 op == Pm4::IT_SET_UCONFIG_REG || op == Pm4::IT_SET_UCONFIG_REG_INDEX;
				RenderStats::NoteIndirectReplayStop(
				    (op << 16u) | (reg && remaining_dw > 1 ? (packet[1] & 0xffffu)
				                   : op == Pm4::IT_NOP ? KYTY_PM4_R(packet[0])
				                                       : 0u));
			}
			InvalidateReplayBase(m_renderer, RenderStats::IndirectReplayNote::PacketState);
		}

		if (draw_pipe) {
			const bool draw_packet = DrawPipeDrawPacket(opcode);
			if (!draw_packet && !DrawPipeSafePacket(opcode, packet)) {
				DrawPipe::Drain(DrawPipe::Reason::Packet, opcode);
			}
			DrawPipe::SetDrawPacket(draw_packet);
		}

		auto handler = g_cp_op_func[opcode];
		// KYTY_PM4_TRACE="seconds:count": after `seconds`, log `count` compute-queue packets.
		{
			static const auto trace = [] {
				std::pair<double, uint64_t> t {0.0, 0};
				if (const char* v = std::getenv("KYTY_PM4_TRACE"); v != nullptr) {
					char* end = nullptr;
					t.first   = std::strtod(v, &end);
					t.second  = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 10) : 2000;
				}
				return t;
			}();
			static const auto trace_start = std::chrono::steady_clock::now();
			static uint64_t   traced      = 0;
			if (trace.second != 0 && traced < trace.second && g_stat_queue != 0 &&
			    std::chrono::duration<double>(std::chrono::steady_clock::now() - trace_start).count() >=
			        trace.first) {
				traced++;
				std::fprintf(stderr,
				             "[pm4] q=%u op=0x%02x r=0x%02x len=%u pkt=%p dw: %08x %08x %08x %08x %08x %08x\n",
				             g_stat_queue, opcode, KYTY_PM4_R(packet_header), KYTY_PM4_LEN(packet_header),
				             static_cast<const void*>(packet), remaining_dw > 1 ? packet[1] : 0,
				             remaining_dw > 2 ? packet[2] : 0, remaining_dw > 3 ? packet[3] : 0,
				             remaining_dw > 4 ? packet[4] : 0, remaining_dw > 5 ? packet[5] : 0,
				             remaining_dw > 6 ? packet[6] : 0);
			}
		}

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		std::optional<Profiler::CommandZone> command_zone;
		if (Profiler::g_counters) {
			if (const auto* location = Pm4ZoneLocation(opcode, KYTY_PM4_R(packet_header));
			    location != nullptr) {
				command_zone.emplace(location);
			}
		}
		std::optional<CpuCost::Phase> packet_phase_;
		if (cpu_cost) {
			packet_phase_.emplace(Pm4PhaseName(opcode, KYTY_PM4_R(packet_header)));
		}
		uint32_t packet_dw = 0;
		// Graphics right of way: the async thread hands the mutex over at this packet boundary
		// (after recording what it queued) instead of holding it for the rest of its segment.
		if (t_segment_held && RenderContext::PoliteAsync() && RenderContext::GraphicsWaiting()) {
			AsyncSegmentLeave();
		}
		if (AsyncBatchEnabled() && GuestGpu::IsAsyncThread() && !RegisterOnlyPacket(opcode)) {
			if ((opcode == Pm4::IT_DISPATCH_DIRECT ||
			     (opcode == Pm4::IT_DISPATCH_INDIRECT && AsyncBatchIndirect())) &&
			    (packet_header & 1u) == 0) {
				// Prepared and queued without the mutex (CommandProcessor::DispatchDirect). Another
				// shader's dispatch records the queue first: its preparation reads guest memory
				// (the SRT walk) as of what has been recorded, and a producer/consumer pair with no
				// packet between them is two shaders (runs of one shader stay batched).
				if (RenderExecutor::QueuedDispatchCount() != 0 &&
				    RenderExecutor::LastQueuedShader() != m_sh_ctx.GetCs().cs_regs.data_addr) {
					AsyncSegmentEnter();
				}
				if (t_segment_held && RenderExecutor::QueuedDispatchCount() == 0) {
					m_renderer.UnlockRenderer();
					t_segment_held = false;
				}
				// The whole packet runs off-lock: a fault on a tracked page takes the mutex itself.
				RenderContext::MarkOffLock(true);
				packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
				RenderContext::MarkOffLock(false);
				if (RenderExecutor::QueuedDispatchCount() >= AsyncBatchMaxValue()) {
					AsyncSegmentEnter();
				}
			} else {
				AsyncSegmentEnter();
				packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			}
		} else if (AsyncThreadEnabled() && !RegisterOnlyPacket(opcode)) {
			// The other command-processor thread may have bound its registers since.
			RendererScope renderer_scope(m_renderer);
			if (g_bound_cp != this) {
				BufferInit();
			}
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		} else {
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (draw_pipe) {
			DrawPipe::SetDrawPacket(false);
		}
		if (execution.m_suspended) {
			NoteSuspendSite(packet, remaining_dw);
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances      = num_instances;
	m_num_instances_args = 0;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	BumpHotGeneration();
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			if (static std::atomic<uint32_t> logged {0}; logged.fetch_add(1) < 4) {
				std::fprintf(stderr, "[occl-real] predication on a query at 0x%016" PRIx64
				             " (wait %u, condition %u)\n",
				             reinterpret_cast<uint64_t>(address), wait_op, condition);
			}
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

// KYTY_LOG_DRAW_PACKET_ES=<hash> (diagnostics): every indexed draw (direct and indirect) whose
// ES, GS or PS has that hash: its arguments, the CP's START_INST_LOC, GS user data 12..20, the
// command dwords around the current packet and, for indirect packets, the argument dwords.
// KYTY_LOG_DRAW_PACKET_COUNT=<n>: only draws of n indices. KYTY_LOG_DRAW_PACKET_EVERY=<n>: only
// every n-th guest flip (default 30), 120 lines on it at most.
struct LoggedDraw {
	const char* kind;
	uint32_t    count;
	uint32_t    instances;
	uint32_t    first_instance;
	int32_t     base_vertex;
	int         source;
	uint64_t    gpu_args;
};
static void LogIndexedDraw(CommandProcessor& cp, const LoggedDraw& args, uint32_t start_inst_reg,
                           uint64_t indirect_base, std::span<const uint32_t> commands,
                           int64_t at) {
	static const std::string watch = [] {
		const char* value = std::getenv("KYTY_LOG_DRAW_PACKET_ES");
		return value != nullptr ? std::string(value) : std::string();
	}();
	if (watch.empty()) {
		return;
	}
	static const uint32_t only_count = [] {
		const char* value = std::getenv("KYTY_LOG_DRAW_PACKET_COUNT");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static const uint64_t every = [] {
		const char* value = std::getenv("KYTY_LOG_DRAW_PACKET_EVERY");
		const auto  n     = value != nullptr ? std::strtoull(value, nullptr, 10) : 30ull;
		return n == 0 ? 1ull : n;
	}();
	if (only_count != 0 && args.count != only_count) {
		return;
	}
	const auto flip = Sync::GuestFlipCount();
	if (flip % every != 0) {
		return;
	}
	const auto& vs   = cp.GetShCtx().GetVs();
	const auto& ps   = cp.GetShCtx().GetPs();
	const auto  hash = [](uint64_t a) { return a != 0 ? ShaderDeclaredHash(a) : 0ull; };
	const auto  listed = [&](uint64_t h) {
		return h != 0 && watch.find(fmt::format("{:016x}", h)) != std::string::npos;
	};
	if (!listed(hash(vs.es_regs.data_addr)) && !listed(hash(vs.gs_regs.data_addr)) &&
	    !listed(hash(ps.ps_regs.data_addr))) {
		return;
	}
	static std::atomic<uint64_t> last_flip {~0ull};
	static std::atomic<uint32_t> lines {0};
	if (last_flip.exchange(flip) != flip) lines.store(0);
	if (lines.fetch_add(1) >= 120) {
		return;
	}
	std::string text = fmt::format(
	    "[idx-draw] flip={} {} count={} inst={} first_inst={} base_vtx={} src={} gpu_args=0x{:x} "
	    "start_inst_reg=0x{:x} ps={:016x} gs12..20:",
	    flip, args.kind, args.count, args.instances, args.first_instance, args.base_vertex,
	    args.source, args.gpu_args, start_inst_reg, hash(ps.ps_regs.data_addr));
	for (uint32_t i = 12; i <= 20; i++) {
		text += fmt::format(" {:08x}", vs.gs_user_sgpr.value[i]);
	}
	if (!commands.empty()) {
		const auto  size = static_cast<int64_t>(commands.size());
		const auto* data = commands.data();
		text += fmt::format(" | at={} pkt:", at);
		for (int64_t i = at; i < std::min(size, at + 10); i++) {
			text += fmt::format(" {:08x}", data[i]);
		}
		text += " | before:";
		for (int64_t i = std::max<int64_t>(0, at - 40); i < at; i++) {
			text += fmt::format(" {:08x}", data[i]);
		}
		if (at < size && indirect_base != 0) {
			const uint32_t header = data[at];
			if ((header == 0xc0032500u || header == 0xc0083800u || header == 0xc0032400u ||
			     header == 0xc0082c00u) &&
			    at + 1 < size) {
				const auto* words =
				    reinterpret_cast<const uint32_t*>(indirect_base + data[at + 1]);
				text += " | args:";
				for (int i = 0; i < 10; i++) {
					text += fmt::format(" {:08x}", words[i]);
				}
			}
		}
	}
	std::fprintf(stderr, "%s\n", text.c_str());
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	const CpuCost::Scope cpu_cost('D');
	Libs::Graphics::SyncToggle::GpuSlowSpin();
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	{
		const LoggedDraw logged {"DrawIndex", args.index_count, args.instance_count,
		                         args.first_instance, args.base_vertex,
		                         static_cast<int>(args.offset_source), 0};
		if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
			const auto& cursor = g_current_execution->m_buffer_stack.back();
			LogIndexedDraw(*this, logged, m_start_instance_register, m_draw_indirect_args_base_addr,
			               cursor.commands, cursor.offset_dw);
		} else {
			LogIndexedDraw(*this, logged, m_start_instance_register, m_draw_indirect_args_base_addr,
			               {}, 0);
		}
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	if (ImgRef::Address() != 0) {
		const auto frame = m_renderer.GetGpu().GetFrameNum();
		const auto& ps   = m_sh_ctx.GetPs();
		ImgRef::Scan("draw-ps", ps.ps_regs.data_addr, ps.ps_user_sgpr, frame, ps.ps_regs.user_data_addr);
		const auto& vs = m_sh_ctx.GetVs();
		ImgRef::Scan("draw-gs", vs.gs_regs.data_addr, vs.gs_user_sgpr, frame);
		ImgRef::Scan("draw-hs", vs.hs_regs.data_addr, vs.hs_user_sgpr, frame);
		static const char* const kRt[8] = {"draw-rt0", "draw-rt1", "draw-rt2", "draw-rt3",
		                                   "draw-rt4", "draw-rt5", "draw-rt6", "draw-rt7"};
		for (uint32_t slot = 0; slot < 8; slot++) {
			ImgRef::Target(kRt[slot], ps.ps_regs.data_addr, m_ctx.GetRenderTarget(slot).base.addr, frame);
		}
		const auto& depth = m_ctx.GetDepthRenderTarget();
		ImgRef::Target("draw-z", ps.ps_regs.data_addr, depth.z_write_base_addr, frame);
		ImgRef::Target("draw-stencil", ps.ps_regs.data_addr, depth.stencil_write_base_addr, frame);
		ImgRef::Target("draw-htile", ps.ps_regs.data_addr, depth.htile_data_base_addr, frame);
	}
	if (DrawPipe::DrawPacket() && PipeDraw(DrawPipe::Kind::DrawIndex, &args, nullptr)) {
		return;
	}
	Occlusion::NoteGuestDraw();
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
	CompleteDraw();
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	const CpuCost::Scope cpu_cost('O');
	Libs::Graphics::SyncToggle::GpuSlowSpin();
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

// The CP writes an indirect draw's start instance into the SH register START_INST_LOC names
// before the draw, as hardware does: fast-launch mesh shaders add that SGPR to their instance
// VGPR themselves (GPU-driven draws pick per-instance records, materials among them, with it).
// KYTY_START_INSTANCE_SGPR=0 restores the old behaviour (the register keeps the guest's value).
void CommandProcessor::WriteIndirectStartInstance(uint32_t start_instance) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_START_INSTANCE_SGPR");
		return value == nullptr || value[0] != '0';
	}();
	const auto reg = m_start_instance_register;
	if (static const bool log = std::getenv("KYTY_LOG_START_INST") != nullptr; log) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 40) {
			std::fprintf(stderr, "[start-inst] cpu reg=0x%x value=%u func=%d\n", reg, start_instance,
			             reg < Pm4::SH_NUM && g_hw_sh_indirect_func[reg] != nullptr ? 1 : 0);
		}
	}
	if (!enabled || reg == 0 || reg >= Pm4::SH_NUM || g_hw_sh_indirect_func[reg] == nullptr) {
		return;
	}
	SetUserDataMarker(HW::UserSgprType::Unknown);
	g_hw_sh_indirect_func[reg](*this, reg, start_instance);
}

// KYTY_MESH_ARGS_RUNS: packet, end: the DRAW_INDIRECT packet and the end of its command buffer.
// Collects the arguments of the non-indexed DRAW_INDIRECTs from this one up to the first packet
// that may write memory, synchronize, change the arguments' base or leave this command buffer.
static void PrepareMeshArgsRun(const CommandProcessor& cp, const uint32_t* packet,
                               const uint32_t* end) {
	auto& run  = MeshArgsRun();
	run.packet = nullptr;
	run.upcoming.clear();
	// KYTY_FPS_LOG: why runs do (not) form, every 5 s.
	static uint64_t calls = 0, no_packet = 0, bad_header = 0, contained = 0, scanned = 0,
	                collected = 0, multi = 0;
	static uint32_t last_header = 0, last_prev = 0, last_stop = 0;
	static auto     last_log    = std::chrono::steady_clock::now();
	if (static const bool log = Common::FpsLogDiagnostics(); log && MeshArgsRunsEnabled()) {
		calls++;
		const auto now = std::chrono::steady_clock::now();
		if (now - last_log >= std::chrono::seconds(5)) {
			last_log = now;
			std::fprintf(stderr,
			             "[mesh-args-scan] calls %llu no-packet %llu bad-header %llu (0x%08x, next "
			             "0x%08x) contained %llu scanned %llu collected %llu runs>1 %llu last-stop "
			             "0x%08x\n",
			             static_cast<unsigned long long>(calls),
			             static_cast<unsigned long long>(no_packet),
			             static_cast<unsigned long long>(bad_header), last_header, last_prev,
			             static_cast<unsigned long long>(contained),
			             static_cast<unsigned long long>(scanned),
			             static_cast<unsigned long long>(collected),
			             static_cast<unsigned long long>(multi), last_stop);
			calls = no_packet = bad_header = contained = scanned = collected = multi = 0;
		}
	}
	if (!MeshArgsRunsEnabled()) {
		return;
	}
	if (packet == nullptr || end == nullptr || packet >= end) {
		no_packet++;
		return;
	}
	if ((packet[0] & ~1u) != 0xc0032400u) {
		bad_header++;
		last_header = packet[0];
		last_prev   = packet + 1 < end ? packet[1] : 0;
		return;
	}
	run.packet = packet;
	if (run.Contains(packet)) {
		contained++;
		return;
	}
	if (RenderStats::IndirectReplayConfigured()) {
		run.upcoming.clear();
		run.converted.clear();
		run.next = 0;
		run.generation++;
		run.last_used_packet = nullptr;
		run.last_used_generation = 0;
	}
	uint64_t base = cp.GetDrawIndirectArgsBaseAddress();
	if (base == 0) {
		return;
	}
	scanned++;
	for (const uint32_t* p = packet; p < end && run.upcoming.size() < 1024;) {
		const uint32_t header = p[0];
		if ((header >> 30u) != 3u || (header & 1u) != 0) {
			break; // not a type-3 packet, or predicated
		}
		const uint32_t op  = (header >> 8u) & 0xffu;
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len < 2 || len > static_cast<uint64_t>(end - p)) {
			break;
		}
		bool go_on = true;
		switch (op) {
			case Pm4::IT_SET_BASE:
				// The arguments' base for the DRAW_INDIRECTs after it (as CpOpSetBase).
				if (len != 4u || (p[1] & 0xfu) != 1u || ((header >> 1u) & 0x3u) > 1u) {
					go_on = false;
				} else if (((header >> 1u) & 0x3u) == 0u) {
					base = (p[2] & ~0x7ull) | (static_cast<uint64_t>(p[3] & 0xffffu) << 32u);
					go_on = base != 0;
				}
				break;
			case Pm4::IT_DRAW_INDIRECT:
				if (header != 0xc0032400u || (p[1] & 3u) != 0) {
					go_on = false;
				} else {
					run.upcoming.push_back({p, base + p[1], {}, 0});
				}
				break;
			// Register state and draws: none writes memory the arguments live in without a guest
			// sync packet before the arguments are read.
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX:
			case Pm4::IT_INDEX_TYPE:
			case Pm4::IT_NUM_INSTANCES:
			case Pm4::IT_INDEX_BASE:
			case Pm4::IT_INDEX_BUFFER_SIZE:
			case Pm4::IT_DRAW_INDEX_AUTO:
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			// Register loads from memory (they write registers only).
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
			// An end-of-pipe label or interrupt: it orders nothing after it on the console (a
			// later read of what it waited for needs WAIT_REG_MEM or a cache sync, which stop the
			// scan), and its label is not a draw's arguments.
			case Pm4::IT_RELEASE_MEM: break;
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				go_on = r == Pm4::R_ZERO || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER ||
				        r == Pm4::R_RELEASE_MEM || r == Pm4::R_CONTEXT_STATE;
				break;
			}
			default: go_on = false; break;
		}
		if (!go_on) {
			last_stop = header;
			if (op == Pm4::IT_EVENT_WRITE && len >= 2) {
				static std::map<uint32_t, uint64_t> events;
				static auto                         events_log = std::chrono::steady_clock::now();
				events[p[1]]++;
				if (const auto now = std::chrono::steady_clock::now();
				    now - events_log >= std::chrono::seconds(5)) {
					events_log = now;
					std::string line = "[mesh-args-scan] EVENT_WRITE stops (payload:count)";
					for (const auto& [payload, count]: events) {
						line += fmt::format(" 0x{:x}:{}", payload, count);
					}
					std::fprintf(stderr, "%s\n", line.c_str());
					events.clear();
				}
			}
			break;
		}
		p += len;
	}
	collected += run.upcoming.size();
	multi += run.upcoming.size() > 1 ? 1 : 0;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	const CpuCost::Scope cpu_cost('R');
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);

	if (!indexed && GpuIndirectDraws()) {
		// The GPU reads the counts (RenderExecutor::DrawAuto and ExecutePreparedDraw): fast-launch
		// mesh draws and vertex-pipeline draws; other mesh draws read them there on the CPU. The
		// arguments are usually written by a dispatch just before; reading them here waited for
		// the whole GPU (6.7 % of the GPU thread in the jungle for the fast-launch draws,
		// 2026-10-03, and 2.9 % for the rest, 2026-10-05).
		m_num_instances_args = reinterpret_cast<uint64_t>(args_addr);
		// START_INST_LOC: the shader loads the start instance into that user SGPR itself.
		uint32_t start_instance_sgpr = 0;
		if (static const bool enabled = [] {
			    const char* value = std::getenv("KYTY_START_INSTANCE_SGPR");
			    return value == nullptr || value[0] != '0';
		    }();
		    enabled && m_start_instance_register >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
		    m_start_instance_register <= Pm4::SPI_SHADER_USER_DATA_GS_31) {
			start_instance_sgpr = m_start_instance_register - Pm4::SPI_SHADER_USER_DATA_GS_0 + 1u;
		}
		if (static const bool log = std::getenv("KYTY_LOG_START_INST") != nullptr; log) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 40) {
				std::fprintf(stderr, "[start-inst] gpu reg=0x%x sgpr=%u args=%u %u %u %u\n",
				             m_start_instance_register, start_instance_sgpr,
				             static_cast<const uint32_t*>(args_addr)[0],
				             static_cast<const uint32_t*>(args_addr)[1],
				             static_cast<const uint32_t*>(args_addr)[2],
				             static_cast<const uint32_t*>(args_addr)[3]);
			}
		}
		{
			const uint32_t* run_packet = nullptr;
			const uint32_t* run_end    = nullptr;
			if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
				const auto& cursor = g_current_execution->m_buffer_stack.back();
				run_packet         = cursor.commands.data() + cursor.offset_dw;
				run_end            = cursor.commands.data() + cursor.commands.size();
			}
			{
				const CpuCost::Phase scan_phase_("R::MeshArgsScan");
				PrepareMeshArgsRun(*this, run_packet, run_end);
			}
			if (RenderStats::IndirectReplayActive() &&
			    m_renderer.GetRenderExecutor().ReplayIndirectDraw(
			        m_submit_id, CurrentBuffer(), reinterpret_cast<uint64_t>(args_addr),
			        start_instance_sgpr, run_packet)) {
				Occlusion::NoteGuestDraw();
				CompleteDraw();
				MeshArgsRun().packet = nullptr;
				return;
			}
			DrawIndexAuto({.offset_source       = DrawOffsetSource::IndirectArgs,
			               .gpu_args_address    = reinterpret_cast<uint64_t>(args_addr),
			               .start_instance_sgpr = start_instance_sgpr,
			               .indirect_packet     = run_packet});
		}
		MeshArgsRun().packet = nullptr;
		return;
	}
	if (!indexed) {
		DrawIndirectArgs args {};
		{
			// The arguments are read here: a fault when the GPU wrote them (a readback).
			const CpuCost::Phase read_phase_("R::ArgsRead");
			std::memcpy(&args, args_addr, sizeof(args));
		}
		if (LogIndirectDraw()) {
			LOGF("Indirect draw: args=0x%010" PRIx64 " vertices=%u instances=%u first=%u\n",
			     reinterpret_cast<uint64_t>(args_addr), args.vertex_count_per_instance,
			     args.instance_count, args.start_vertex_location);
		}
		m_num_instances = args.instance_count;
		m_num_instances_args = 0;
		WriteIndirectStartInstance(args.start_instance_location);
		DrawIndexAuto({.vertex_count   = args.vertex_count_per_instance,
		               .instance_count = args.instance_count,
		               .first_vertex   = args.start_vertex_location,
		               .first_instance = args.start_instance_location,
		               .offset_source  = DrawOffsetSource::IndirectArgs});
		return;
	}

	DrawIndexedIndirectArgs args {};
	{
		const CpuCost::Phase read_phase_("R::ArgsRead");
		std::memcpy(&args, args_addr, sizeof(args));
	}
	if (LogIndirectDraw()) {
		LOGF("Indirect indexed draw: args=0x%010" PRIx64 " indices=%u instances=%u first=%u\n",
		     reinterpret_cast<uint64_t>(args_addr), args.index_count_per_instance,
		     args.instance_count, args.start_index_location);
	}

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size);

	const uint32_t index_count =
	    (m_index_buffer_size != 0 ? std::min(args.index_count_per_instance, m_index_buffer_size)
	                              : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	m_num_instances = args.instance_count;
	m_num_instances_args = 0;
	DrawIndex({.index_count    = index_count,
	           .index_addr     = index_addr,
	           .instance_count = args.instance_count,
	           .base_vertex    = static_cast<int32_t>(args.base_vertex_location),
	           .first_instance = args.start_instance_location,
	           .offset_source  = DrawOffsetSource::IndirectArgs});
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, uint32_t draw_initiator,
                                         bool indexed, uint32_t draw_index_register) {
	const CpuCost::Scope cpu_cost('M');
	Libs::Graphics::SyncToggle::GpuSlowSpin();
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	// KYTY_MULTI_DRAW_SYNC=1: wait for the GPU's writes to the draw count and the argument
	// records before reading them here (as the mesh path's CPU-read arguments do): a count a
	// dispatch of this submission wrote may not have faulted into a readback yet.
	static const bool sync_args = [] {
		const char* value = std::getenv("KYTY_MULTI_DRAW_SYNC");
		return value != nullptr && value[0] == '1';
	}();
	// KYTY_LOG_MULTI_COUNT=<first guest flip>[:<flips>]: every DRAW_(INDEX_)INDIRECT_MULTI of
	// those flips with its count address: the count read here, the GPU's copy of it and of the
	// first argument record, the write state of the count's page and the pixel shader.
	static const auto log_window = [] {
		std::pair<uint64_t, uint64_t> window {UINT64_MAX, 0};
		if (const char* value = std::getenv("KYTY_LOG_MULTI_COUNT"); value != nullptr) {
			unsigned long long first = 0, flips = 1;
			if (std::sscanf(value, "%llu:%llu", &first, &flips) >= 1) {
				window = {first, flips == 0 ? 1 : flips};
			}
		}
		return window;
	}();
	const auto flip    = Sync::GuestFlipCount();
	const bool log_now = flip >= log_window.first && flip - log_window.first < log_window.second;
	auto&      cache   = m_renderer.GetBufferCache();
	const auto args_base = m_draw_indirect_args_base_addr + data_offset;
	const auto args_size_all =
	    static_cast<uint64_t>(stride_in_bytes) * std::max<uint32_t>(max_count_or_count, 1u);
	std::string log_before;
	if (log_now && count_addr != nullptr) {
		log_before = cache.DescribeWriteState(reinterpret_cast<uint64_t>(count_addr));
	}
	if (sync_args) {
		if (count_addr != nullptr) {
			cache.WaitSharedWrites(reinterpret_cast<uint64_t>(count_addr), sizeof(uint32_t));
		}
		cache.WaitSharedWrites(args_base, std::min<uint64_t>(args_size_all, 1u << 20u));
	}
	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		// The count is read here: a fault when the GPU wrote it (a readback).
		const CpuCost::Phase count_phase_("M::CountRead");
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}
	if (log_now) {
		static std::atomic<uint32_t> lines {0};
		if (lines.fetch_add(1) < 4000) {
			const auto& ps   = GetShCtx().GetPs();
			const auto  hash = ps.ps_regs.data_addr != 0 ? ShaderDeclaredHash(ps.ps_regs.data_addr) : 0ull;
			std::string gpu_count = "-";
			std::string gpu_args  = "-";
			std::vector<uint32_t> peek;
			if (count_addr != nullptr &&
			    cache.PeekGpuCopy(reinterpret_cast<uint64_t>(count_addr), 4, peek) && !peek.empty()) {
				gpu_count = fmt::format("{}", peek[0]);
			}
			peek.clear();
			if (cache.PeekGpuCopy(args_base, indexed ? 20u : 16u, peek)) {
				gpu_args.clear();
				for (const auto w: peek) {
					gpu_args += fmt::format("{:08x} ", w);
				}
			}
			std::string cpu_args;
			for (uint32_t i = 0; i < (indexed ? 5u : 4u); i++) {
				cpu_args += fmt::format("{:08x} ", reinterpret_cast<const uint32_t*>(args_base)[i]);
			}
			std::fprintf(stderr,
			             "[multi-count] flip=%llu ps=%016llx indexed=%d count_addr=%llx cpu_count=%u "
			             "max=%u gpu_count=%s stride=%u args=%llx cpu_args=%s gpu_args=%s state=%s\n",
			             static_cast<unsigned long long>(flip), static_cast<unsigned long long>(hash),
			             indexed ? 1 : 0,
			             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(count_addr)),
			             count_addr != nullptr ? *count_addr : max_count_or_count, max_count_or_count,
			             gpu_count.c_str(), stride_in_bytes, static_cast<unsigned long long>(args_base),
			             cpu_args.c_str(), gpu_args.c_str(), log_before.c_str());
		}
	}
	if (LogIndirectDraw()) {
		LOGF("Indirect multi draw: args=0x%010" PRIx64 " indexed=%d count=%u max=%u count_addr=%p "
		     "stride=%u\n",
		     m_draw_indirect_args_base_addr + data_offset, indexed ? 1 : 0, draw_count,
		     max_count_or_count, static_cast<const void*>(const_cast<const uint32_t*>(count_addr)),
		     stride_in_bytes);
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	const bool probe       = RenderStats::MultiDrawProbeEnabled();
	uint64_t   probe_first = 0;
	uint32_t   probe_same  = 0;
	// KYTY_MULTI_DRAW_REPLAY: the draws after the first replayed from the last one drawn in full
	// (RenderExecutor::ReplayMultiDraw); per-draw diagnostics here keep every draw in full.
	const bool replay = !indexed && !probe && draw_count > 1 && RenderStats::MultiDrawReplayActive() &&
	                    ImgRef::Address() == 0 && !LogIndirectDraw();
	auto&      executor = m_renderer.GetRenderExecutor();
	uint32_t   replayed_in_packet = 0;
	// See CompleteDraw. Off with KYTY_MULTI_DRAW_PACKET_FLUSH=0 as well (per sub-draw).
	static const bool defer_flush = [] {
		const char* packet = std::getenv("KYTY_MULTI_DRAW_PACKET_FLUSH");
		const char* defer  = std::getenv("KYTY_MULTI_DRAW_DEFER_FLUSH");
		return (packet == nullptr || packet[0] != '0') && (defer == nullptr || defer[0] != '0');
	}();
	// KYTY_RENDER_STATS: [multi-packets], the packets by class (RenderStats::MultiPacketNote).
	const bool census = RenderStats::Enabled();
	if (census) {
		using Note = RenderStats::MultiPacketNote;
		RenderStats::NoteMultiPacket(Note::Packets);
		RenderStats::NoteMultiPacket(Note::SubDraws, draw_count);
		if (draw_count == 1) {
			RenderStats::NoteMultiPacket(Note::Single);
		}
		if (indexed) {
			RenderStats::NoteMultiPacket(Note::IndexedPackets);
			RenderStats::NoteMultiPacket(Note::IndexedRest, draw_count - 1u);
			executor.CensusMultiDrawIndexed(true);
		}
		if (replay) {
			RenderStats::NoteMultiPacket(Note::ReplayPackets);
		}
	}
	if (replay) {
		executor.BeginMultiDraw();
		m_multi_draw_defer    = defer_flush;
		m_multi_draw_deferred = 0;
	}
	for (uint32_t i = 0; i < draw_count; i++) {
		if (probe && i > 0) {
			// The previous draw's commit, compared with the first draw's.
			const auto h = RenderStats::LastCommitHash();
			if (i == 1) {
				probe_first = h;
			} else if (h == probe_first && h != 0) {
				probe_same++;
			}
			RenderStats::NoteCommitHash(0);
		} else if (probe) {
			RenderStats::NoteCommitHash(0);
		}
		// DRAW_INDIRECT_MULTI writes DrawIndex before each draw, including draws
		// with zero vertices/instances. It starts at zero for each packet.
		if (draw_index_register != Pm4::SH_NOP) {
			EXIT_NOT_IMPLEMENTED(draw_index_register >= Pm4::SH_NUM ||
			                     g_hw_sh_indirect_func[draw_index_register] == nullptr);
			SetUserDataMarker(HW::UserSgprType::Unknown);
			g_hw_sh_indirect_func[draw_index_register](*this, draw_index_register, i);
		}
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args      = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			{
				const CpuCost::Phase read_phase_("M::ArgsRead"); // the record's first read
				m_num_instances = args->instance_count;
			}
			m_num_instances_args = 0;
			WriteIndirectStartInstance(args->start_instance_location);
			if (census && replay && i == 0) {
				executor.CensusMultiDrawFirst(CurrentBuffer());
			}
			if (replay && i > 0 &&
			    executor.ReplayMultiDraw(m_submit_id, CurrentBuffer(), args->vertex_count_per_instance,
			                             args->instance_count, args->start_vertex_location,
			                             args->start_instance_location)) {
				Occlusion::NoteGuestDraw();
				// Fps plan W1b (11-15): a replayed sub-draw does not count toward the draw-flush
				// interval (one in 16 of a 1,900-draw packet was drawn in full again after the
				// submit); the packet counts once, after its loop. KYTY_MULTI_DRAW_PACKET_FLUSH=0: per sub-draw.
				static const bool per_packet = [] {
					const char* value = std::getenv("KYTY_MULTI_DRAW_PACKET_FLUSH");
					return value == nullptr || value[0] != '0';
				}();
				if (Common::FloorStats::Enabled()) {
					Common::FloorStats::Add(Common::FloorStats::Get().replayed, 1);
				}
				if (per_packet) {
					replayed_in_packet++;
				} else {
					CompleteDraw();
				}
				continue;
			}
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);

		std::optional<CpuCost::Phase> read_phase_(std::in_place, "M::ArgsRead"); // the record's first read
		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);
		read_phase_.reset();

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		m_num_instances = args->instance_count;
		m_num_instances_args = 0;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
	if (replay) {
		if (census) {
			executor.CensusMultiDrawEnd(CurrentBuffer());
		}
		executor.EndMultiDraw();
		// The full sub-draws CompleteDraw left for here, counted as it would have counted them,
		// up to the first submit: the rest were recorded into the command buffer that submit sent.
		m_multi_draw_defer = false;
		uint32_t counted   = 0;
		while (counted < m_multi_draw_deferred) {
			const auto tick = GetScheduler().CurrentTick();
			CompleteDraw();
			counted++;
			if (GetScheduler().CurrentTick() != tick) {
				break;
			}
		}
		if (counted < m_multi_draw_deferred && Common::FloorStats::Enabled()) {
			Common::FloorStats::Add(Common::FloorStats::Get().draws, m_multi_draw_deferred - counted);
		}
		m_multi_draw_deferred = 0;
		if (replayed_in_packet != 0) {
			CompleteDraw(); // the packet's replayed sub-draws, once
		}
	}
	if (census && indexed) {
		executor.CensusMultiDrawIndexed(false);
	}
	if (probe) {
		const auto h = RenderStats::LastCommitHash();
		if (draw_count == 1) {
			probe_first = h;
		} else if (h == probe_first && h != 0) {
			probe_same++;
		}
		RenderStats::NoteMultiDrawPacket(draw_count, probe_same, draw_index_register, indexed,
		                                 probe_first);
	}
}

// Profiling (KYTY_FPS_LOG): which queue each compute shader is dispatched from, with its first
// group counts, and the dispatch rate per queue.
static uint64_t g_dispatch_cpu_us[2] {};
// KYTY_PACKET_STATS=1: which dispatch packets run how often (async compute queues), and how many
// compute submissions arrive, every 5 s: a packet that runs hundreds of times a frame is a loop
// in the command stream (a chained indirect buffer that re-runs until a label changes).
static void NotePacket(const uint32_t* packet, bool async, uint64_t shader) {
	static const bool enabled = std::getenv("KYTY_PACKET_STATS") != nullptr;
	if (!enabled || packet == nullptr) {
		return;
	}
	struct Entry {
		uint64_t count  = 0;
		uint64_t shader = 0;
		bool     async  = false;
	};
	static std::unordered_map<const uint32_t*, Entry> counts;
	static std::mutex                                 counts_mutex; // both command-processor threads
	std::lock_guard                                   counts_lock(counts_mutex);
	static auto start = std::chrono::steady_clock::now();
	auto&       e     = counts[packet];
	e.count++;
	e.shader = shader;
	e.async  = async;
	const auto now = std::chrono::steady_clock::now();
	const auto s   = std::chrono::duration<double>(now - start).count();
	if (s < 5.0) {
		return;
	}
	std::vector<std::pair<const uint32_t*, Entry>> list(counts.begin(), counts.end());
	std::sort(list.begin(), list.end(),
	          [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	std::fprintf(stderr,
	             "[packets] %zu distinct dispatch packets; compute submissions %.0f/s (%.0f dw/s)\n",
	             list.size(), static_cast<double>(g_compute_submissions.exchange(0)) / s,
	             static_cast<double>(g_compute_submission_dw.exchange(0)) / s);
	for (size_t i = 0; i < list.size() && i < 10; i++) {
		std::fprintf(stderr, "[packets]   %p %s shader %016llx %.0f/s\n",
		             static_cast<const void*>(list[i].first), list[i].second.async ? "async" : "gfx",
		             static_cast<unsigned long long>(list[i].second.shader),
		             static_cast<double>(list[i].second.count) / s);
	}
	counts.clear();
	start = now;
}

static void NoteDispatchQueue(uint64_t addr, bool async, uint32_t x, uint32_t y, uint32_t z) {
	static const bool log = Common::FpsLogDiagnostics();
	if (!log) {
		return;
	}
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> seen;
	static auto                                   start = std::chrono::steady_clock::now();
	static uint64_t                               counts[2] {};
	std::lock_guard lock(mutex);
	if (x == UINT32_MAX) {
		// CPU time of the previous dispatch's recording, in microseconds (z), on queue `async`.
		g_dispatch_cpu_us[async ? 1 : 0] += z;
		return;
	}
	counts[async ? 1 : 0]++;
	auto& n = seen[addr ^ (async ? 1ull : 0ull)];
	if (n++ < 2) {
		std::fprintf(stderr, "[dq] addr=0x%016" PRIx64 " async=%d groups=%ux%ux%u\n", addr,
		             async ? 1 : 0, x, y, z);
	}
	const auto now = std::chrono::steady_clock::now();
	const auto s   = std::chrono::duration<double>(now - start).count();
	if (s >= 5.0) {
		std::fprintf(stderr, "[dq-rate] gfx=%.0f/s async=%.0f/s cpu_ms_per_s gfx=%.0f async=%.0f\n",
		             counts[0] / s, counts[1] / s, g_dispatch_cpu_us[0] / 1000.0 / s,
		             g_dispatch_cpu_us[1] / 1000.0 / s);
		counts[0] = counts[1] = 0;
		g_dispatch_cpu_us[0] = g_dispatch_cpu_us[1] = 0;
		start                 = now;
	}
}

// KYTY_DISPATCH_RUNS (see dispatchRun.h): the dispatch packet the previous dispatch's run continues
// with, if the stream reaches it with nothing but SET_SH_REG and debug markers in between
// (g_run_expected_next, declared at the top).

// stop, stop_at (KYTY_QUEUE_STATS, [run-breaks]): why and where the scan gave up.
static const uint32_t* NextRunDispatch(const uint32_t* p, const uint32_t* end,
                                       std::array<uint32_t, 3>* local, uint32_t* stop = nullptr,
                                       const uint32_t** stop_at = nullptr) {
	const auto gave_up = [&](uint32_t key, const uint32_t* at) -> const uint32_t* {
		if (stop != nullptr) *stop = key;
		if (stop_at != nullptr) *stop_at = at;
		return nullptr;
	};
	while (p < end) {
		const uint32_t header = p[0];
		if (header == 0x80000000u) {
			p++;
			continue;
		}
		if ((header >> 30u) != 3u || (header & 1u) != 0) {
			// not a type-3 packet, or predicated
			return gave_up((header >> 30u) != 3u ? 2u << 24u : 3u << 24u, nullptr);
		}
		const uint32_t op  = (header >> 8u) & 0xffu;
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len < 2 || len > static_cast<uint64_t>(end - p)) {
			return gave_up(4u << 24u, nullptr);
		}
		if (op == Pm4::IT_DISPATCH_DIRECT || op == Pm4::IT_DISPATCH_INDIRECT) {
			return p;
		}
		if (op == Pm4::IT_SET_SH_REG) {
			if (local != nullptr) {
				const uint32_t reg = p[1] & 0xffffu;
				for (uint32_t i = 0; i + 2 < len; i++) {
					const uint32_t r = reg + i;
					if (r >= 0x207u && r <= 0x209u) {
						(*local)[r - 0x207u] = std::max(p[2 + i], 1u);
					}
				}
			}
			p += len;
			continue;
		}
		if (op == Pm4::IT_NOP) {
			const auto r = KYTY_PM4_R(header);
			if (r == Pm4::R_ZERO || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER) {
				p += len;
				continue;
			}
		}
		if (stop != nullptr) {
			uint32_t sub = 0;
			if (op == Pm4::IT_NOP) {
				sub = KYTY_PM4_R(header);
			} else if (op == Pm4::IT_EVENT_WRITE || op == Pm4::IT_EVENT_WRITE_EOP ||
			           op == Pm4::IT_EVENT_WRITE_EOS) {
				sub = p[1] & 0x3fu;
			} else if (op >= 0x68u && op <= 0x7bu) { // the SET_*_REG family: the first register
				sub = p[1] & 0xffffu;
			}
			return gave_up((op << 16u) | sub, p);
		}
		return nullptr;
	}
	return gave_up(1u << 24u, nullptr);
}

// [run-breaks]: whether a dispatch follows the packet a run stopped at, within 64 packets of the
// same command buffer (the stopping packet itself is skipped).
static bool RunBreakThenDispatch(const uint32_t* p, const uint32_t* end) {
	for (uint32_t n = 0; p != nullptr && p < end && n < 65u; n++) {
		const uint32_t header = p[0];
		if (header == 0x80000000u) {
			p++;
			continue;
		}
		if ((header >> 30u) != 3u) {
			return false;
		}
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len < 2 || len > static_cast<uint64_t>(end - p)) {
			return false;
		}
		const uint32_t op = (header >> 8u) & 0xffu;
		if (n != 0 && (op == Pm4::IT_DISPATCH_DIRECT || op == Pm4::IT_DISPATCH_INDIRECT)) {
			return true;
		}
		p += len;
	}
	return false;
}

// packet, end: the dispatch packet and the end of its command buffer (nullptr: unknown).
static void PrepareDispatchRun(CommandProcessor& cp, bool indirect, const uint32_t* packet,
                               const uint32_t* end) {
	auto& run     = DispatchRun();
	run.packet    = nullptr;
	run.continues = false;
	run.more      = false;
	run.why       = 0;
	static const bool reasons = BarrierReason::Enabled();
	if (!DispatchRunsEnabled() || packet == nullptr) {
		g_run_expected_next = nullptr;
		if (reasons) {
			run.why         = DispatchRunsEnabled() ? BarrierReason::WhyNoPacket : BarrierReason::WhyRunsOff;
			g_run_last_stop = BarrierReason::WhyNoPacket;
		}
		return;
	}
	run.packet             = packet;
	run.continues          = packet == g_run_expected_next;
	if (reasons && !run.continues) {
		run.why = g_run_expected_next != nullptr ? static_cast<uint32_t>(BarrierReason::WhyMismatch)
		                                         : g_run_last_stop;
		if (run.why == 0) run.why = BarrierReason::WhySlice; // the thread's first dispatch
	}
	if (!run.continues && !run.converted.empty()) {
		// Converted counts belong to the run instance that converted them. A command buffer the
		// game submits again (or a run resumed after KYTY_SRT_DEFER) reaches the same packets
		// later: their entries then name ring slots later conversions rewrote, and counts of
		// arguments since rewritten (a refit then ran with the wrong number of groups).
		// KYTY_RUN_CONVERTED_KEEP=1: keep them, as before (the count of such starts is still logged).
		static const bool keep = [] {
			const char* value = std::getenv("KYTY_RUN_CONVERTED_KEEP");
			return value != nullptr && value[0] == '1';
		}();
		if (run.converted.contains(packet)) {
			run.stale_hits++;
		}
		if (!keep) {
			run.converted.clear();
		}
	}
	// KYTY_FPS_LOG: run lengths by the shader of each dispatch (addr -> {dispatches, runs started}).
	// (getenv once: on Windows each call locks and scans the environment, ~1 us a dispatch.)
	static const bool run_log = Common::FpsLogDiagnostics();
	if (run_log) {
		// Both command-processor threads (the async thread prepares dispatches without the mutex).
		static std::mutex run_log_mutex;
		std::lock_guard   run_log_lock(run_log_mutex);
		static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> by_shader;
		static uint64_t                                                   run_len = 0;
		static auto last = std::chrono::steady_clock::now();
		auto&       e    = by_shader[cp.GetShCtx().GetCs().cs_regs.data_addr];
		e.first++;
		if (!run.continues) e.second++;
		const auto now = std::chrono::steady_clock::now();
		if (now - last > std::chrono::seconds(5)) {
			last = now;
			std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> list(by_shader.begin(),
			                                                                      by_shader.end());
			std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
				return a.second.first > b.second.first;
			});
			for (size_t i = 0; i < list.size() && i < 8; i++) {
				std::fprintf(stderr, "[dispatch-runs]   cs 0x%llx dispatches %llu, runs started %llu\n",
				             static_cast<unsigned long long>(list[i].first),
				             static_cast<unsigned long long>(list[i].second.first),
				             static_cast<unsigned long long>(list[i].second.second));
			}
			by_shader.clear();
		}
		(void)run_len;
	}
	run.runs += run.continues ? 0 : 1;
	run.members++;
	static const bool run_breaks = QueueStatsEnabled();
	uint32_t          stop       = 0;
	const uint32_t*   stop_at    = nullptr;
	const uint32_t* next = NextRunDispatch(packet + KYTY_PM4_LEN(packet[0]), end, nullptr,
	                                       run_breaks || reasons ? &stop : nullptr, &stop_at);
	if (reasons) g_run_last_stop = next == nullptr ? stop : 0u;
	if (run_breaks && next == nullptr) {
		QueueStatsNoteRunBreak(stop, RunBreakThenDispatch(stop_at, end));
	}
	run.more             = next != nullptr;
	g_run_expected_next  = next;
	if (!indirect || run.converted.contains(packet)) {
		return;
	}
	// Queued on the async thread (KYTY_ASYNC_BATCH_INDIRECT): the run's conversions happen when its
	// first dispatch is recorded, from the list collected at the run's start, so the later
	// dispatches of the run keep that list.
	if (run.continues && !t_segment_held && AsyncBatchEnabled() && AsyncBatchIndirect() &&
	    GuestGpu::IsAsyncThread()) {
		return;
	}
	run.upcoming.clear();
	std::array<uint32_t, 3> local {0, 0, 0};
	for (const uint32_t* p = packet; p != nullptr && run.upcoming.size() < 1024;) {
		const uint32_t header = p[0] & ~1u;
		if (((header >> 8u) & 0xffu) == Pm4::IT_DISPATCH_INDIRECT) {
			uint64_t args = 0;
			uint32_t mode = 0;
			if (header == 0xc0021600u) {
				args = p[1] | (static_cast<uint64_t>(p[2]) << 32u);
				mode = p[3];
			} else if (header == 0xc0011600u) {
				const auto base = cp.GetDispatchIndirectArgsBaseAddress();
				if (base == 0) break;
				args = base + p[1];
				mode = p[2];
			} else {
				break;
			}
			if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0 && args != 0 &&
			    (args & 3u) == 0) {
				run.upcoming.push_back({p, args, local});
			}
		}
		p = NextRunDispatch(p + KYTY_PM4_LEN(header), end, &local);
	}
}

static void ReportDispatchRuns() {
	static const bool log = Common::FpsLogDiagnostics();
	if (!log || !DispatchRunsEnabled()) return;
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	const double s   = std::chrono::duration<double>(now - last).count();
	last             = now;
	auto& run        = DispatchRun();
	std::fprintf(stderr,
	             "[dispatch-runs] runs %.0f/s dispatches %.0f/s barriers left out %.0f/s indirect "
	             "counts converted in batches %.0f/s, stale conversions dropped %llu\n",
	             static_cast<double>(run.runs) / s, static_cast<double>(run.members) / s,
	             static_cast<double>(run.skipped_barriers) / s, static_cast<double>(run.batched) / s,
	             static_cast<unsigned long long>(run.stale_hits));
	run.runs = run.members = run.skipped_barriers = run.batched = 0;
}

// KYTY_SEQ_HASHES=<hex,...> with KYTY_SEQ_TRIGGER=<file>: while the file exists, each dispatch of a
// listed compute shader and each draw of a listed pixel shader prints one [seq] line, in the order
// the command processors run them (q: 0 graphics, 1 async compute), up to 6000 lines.
void SeqNote(char kind, uint64_t shader_addr, int queue, uint64_t frame, uint64_t presented) {
	static const std::string hashes = [] {
		const char* value = std::getenv("KYTY_SEQ_HASHES");
		return value != nullptr ? std::string(value) : std::string();
	}();
	static const char* trigger = std::getenv("KYTY_SEQ_TRIGGER");
	// KYTY_SEQ_FLIP=<guest flip>: log from that guest flip on, instead of a trigger file.
	static const uint64_t start_flip = [] {
		const char* value = std::getenv("KYTY_SEQ_FLIP");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : UINT64_MAX;
	}();
	if (hashes.empty() || (trigger == nullptr && start_flip == UINT64_MAX) || shader_addr == 0) {
		return;
	}
	static std::atomic<uint64_t> checked {UINT64_MAX};
	static std::atomic_bool      on {false};
	if (start_flip != UINT64_MAX) {
		if (Sync::GuestFlipCount() < start_flip) {
			return;
		}
	} else {
		if (checked.exchange(frame) != frame) {
			std::error_code error;
			on = std::filesystem::exists(trigger, error);
		}
		if (!on) {
			return;
		}
	}
	const uint64_t hash = ShaderDeclaredHash(shader_addr);
	// "allcs" in the list: every compute dispatch too (up to 400000 lines).
	static const bool all_compute = hashes.find("allcs") != std::string::npos;
	if (!(all_compute && (kind == 'X' || kind == 'I')) &&
	    hashes.find(fmt::format("{:016x}", hash)) == std::string::npos) {
		return;
	}
	static std::atomic<uint32_t> lines {0};
	static std::atomic<uint64_t> sequence {0};
	if (lines.fetch_add(1) >= (all_compute ? 60000u : 6000u)) {
		return;
	}
	std::fprintf(stderr, "[seq] n=%llu f=%llu g=%llu %c q=%d %016llx p=%llu\n",
	             static_cast<unsigned long long>(sequence.fetch_add(1)),
	             static_cast<unsigned long long>(frame),
	             static_cast<unsigned long long>(Sync::GuestFlipCount()), kind, queue,
	             static_cast<unsigned long long>(hash), static_cast<unsigned long long>(presented));
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	const CpuCost::Scope cpu_cost('X');
	Libs::Graphics::SyncToggle::GpuSlowSpin();
	if (Common::FloorStats::Enabled()) {
		Common::FloorStats::Add(Common::FloorStats::Get().dispatches, 1);
	}
	if (ImgRef::Address() != 0) {
		ImgRef::Scan("dispatch", m_sh_ctx.GetCs().cs_regs.data_addr, m_sh_ctx.GetCs().cs_user_sgpr,
		             m_renderer.GetGpu().GetFrameNum());
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	NoteDispatchQueue(m_sh_ctx.GetCs().cs_regs.data_addr, IsAsyncComputeQueue(), thread_group_x,
	                  thread_group_y, thread_group_z);
	SeqNote('X', m_sh_ctx.GetCs().cs_regs.data_addr, IsAsyncComputeQueue() ? 1 : 0,
	        static_cast<uint64_t>(m_renderer.GetGpu().GetFrameNum()),
	        m_renderer.GetGraphics().presented_frames.load(std::memory_order_relaxed));
	NotePacket((g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()
	               ? g_current_execution->m_buffer_stack.back().commands.data() +
	                     g_current_execution->m_buffer_stack.back().offset_dw
	               : nullptr), IsAsyncComputeQueue(), m_sh_ctx.GetCs().cs_regs.data_addr);

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		// The CPU time goes only to NoteDispatchQueue, which keeps it for KYTY_FPS_LOG diagnostics.
		const bool time_dispatch = Common::FpsLogDiagnostics() || !SyncToggle::HotPath();
		const auto cpu_begin     = time_dispatch ? std::chrono::steady_clock::now()
		                                         : std::chrono::steady_clock::time_point {};
		if (BeginDispatchDefer()) {
			return;
		}
		{
			const uint32_t* run_packet = nullptr;
			const uint32_t* run_end    = nullptr;
			if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
				const auto& cursor = g_current_execution->m_buffer_stack.back();
				run_packet         = cursor.commands.data() + cursor.offset_dw;
				run_end            = cursor.commands.data() + cursor.commands.size();
			}
			const CpuCost::Phase scan_phase_("Dispatch::RunScan");
			PrepareDispatchRun(*this, false, run_packet, run_end);
		}
		ReportDispatchRuns();
		if (!t_segment_held && AsyncBatchEnabled() && GuestGpu::IsAsyncThread()) {
			m_renderer.GetRenderExecutor().QueueDispatchDirect(m_submit_id, m_sh_ctx.GetCs(),
			                                                   m_ctx.GetShaderRegisters(), thread_group_x,
			                                                   thread_group_y, thread_group_z, mode);
		} else {
			m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
			                                              thread_group_y, thread_group_z, mode);
		}
		EndDispatchDefer();
		if (time_dispatch) {
			NoteDispatchQueue(0, IsAsyncComputeQueue(), UINT32_MAX, 0,
			                  static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - cpu_begin)
			                                            .count()));
		}
	}
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	EXIT_NOT_IMPLEMENTED(IsAsyncComputeQueue());
	m_renderer.GetRenderExecutor().ProcessPssr(CurrentBuffer(), request);
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	const CpuCost::Scope cpu_cost('I');
	Libs::Graphics::SyncToggle::GpuSlowSpin();
	if (Common::FloorStats::Enabled()) {
		Common::FloorStats::Add(Common::FloorStats::Get().dispatches, 1);
	}
	if (ImgRef::Address() != 0) {
		ImgRef::Scan("dispatch", m_sh_ctx.GetCs().cs_regs.data_addr, m_sh_ctx.GetCs().cs_user_sgpr,
		             m_renderer.GetGpu().GetFrameNum());
	}
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	SeqNote('I', m_sh_ctx.GetCs().cs_regs.data_addr, IsAsyncComputeQueue() ? 1 : 0,
	        static_cast<uint64_t>(m_renderer.GetGpu().GetFrameNum()),
	        m_renderer.GetGraphics().presented_frames.load(std::memory_order_relaxed));
	// Thread counts are converted on the GPU. Reading them here drained the GPU for every
	// dispatch, since the previous dispatch writes them; KYTY_INDIRECT_THREADS_ON_CPU=1 restores
	// that path for A/B runs.
	static const bool threads_on_cpu = std::getenv("KYTY_INDIRECT_THREADS_ON_CPU") != nullptr;
	if (threads_on_cpu && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	NoteDispatchQueue(m_sh_ctx.GetCs().cs_regs.data_addr, IsAsyncComputeQueue(), 0, 0, 0);
	// KYTY_INDIRECT_STATS=1: the group counts indirect dispatches had, per shader, from copies of
	// their arguments that never make the GPU thread wait (a frame or more old), every 5 s.
	if (static const bool stats = std::getenv("KYTY_INDIRECT_STATS") != nullptr; stats) {
		struct Stat {
			uint64_t n = 0, zero = 0, groups = 0, seen = 0, max = 0;
		};
		static std::unordered_map<uint64_t, Stat> by_shader;
		static auto start = std::chrono::steady_clock::now();
		auto&       st    = by_shader[m_sh_ctx.GetCs().cs_regs.data_addr];
		st.n++;
		const uint32_t x = Libs::LibKernel::Memory::PeekGpuDword(args_addr);
		const uint32_t y = Libs::LibKernel::Memory::PeekGpuDword(args_addr + 4);
		const uint32_t z = Libs::LibKernel::Memory::PeekGpuDword(args_addr + 8);
		if (x != 0 || y != 0 || z != 0 || st.seen != 0) {
			st.seen++;
			const uint64_t g = uint64_t {x} * std::max(y, 1u) * std::max(z, 1u);
			st.zero += (x == 0 || y == 0 || z == 0) ? 1 : 0;
			st.groups += (x == 0 || y == 0 || z == 0) ? 0 : g;
			st.max = std::max(st.max, (x == 0 || y == 0 || z == 0) ? 0 : g);
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - start > std::chrono::seconds(5)) {
			start = now;
			for (const auto& [addr, e]: by_shader) {
				if (e.n < 100) continue;
				std::fprintf(stderr,
				             "[indirect] cs 0x%llx dispatches=%llu sampled=%llu zero-group=%llu "
				             "avg-groups=%.1f max=%llu (mode 0x%x)\n",
				             static_cast<unsigned long long>(addr),
				             static_cast<unsigned long long>(e.n),
				             static_cast<unsigned long long>(e.seen),
				             static_cast<unsigned long long>(e.zero),
				             e.seen > e.zero ? static_cast<double>(e.groups) / (e.seen - e.zero) : 0.0,
				             static_cast<unsigned long long>(e.max), mode);
			}
			by_shader.clear();
		}
	}
	NotePacket((g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()
	               ? g_current_execution->m_buffer_stack.back().commands.data() +
	                     g_current_execution->m_buffer_stack.back().offset_dw
	               : nullptr), IsAsyncComputeQueue(), m_sh_ctx.GetCs().cs_regs.data_addr);
	// The CPU time goes only to NoteDispatchQueue, which keeps it for KYTY_FPS_LOG diagnostics.
	const bool time_dispatch = Common::FpsLogDiagnostics() || !SyncToggle::HotPath();
	const auto cpu_begin     = time_dispatch ? std::chrono::steady_clock::now()
	                                         : std::chrono::steady_clock::time_point {};
	if (BeginDispatchDefer()) {
		return;
	}
	{
		const uint32_t* run_packet = nullptr;
		const uint32_t* run_end    = nullptr;
		if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
			const auto& cursor = g_current_execution->m_buffer_stack.back();
			run_packet         = cursor.commands.data() + cursor.offset_dw;
			run_end            = cursor.commands.data() + cursor.commands.size();
		}
		const CpuCost::Phase scan_phase_("Dispatch::RunScan");
		PrepareDispatchRun(*this, (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0,
		                   run_packet, run_end);
	}
	ReportDispatchRuns();
	if (!t_segment_held && AsyncBatchEnabled() && AsyncBatchIndirect() && GuestGpu::IsAsyncThread()) {
		m_renderer.GetRenderExecutor().QueueDispatchIndirect(m_submit_id, m_sh_ctx.GetCs(),
		                                                     m_ctx.GetShaderRegisters(), args_addr, mode);
	} else {
		m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode);
	}
	EndDispatchDefer();
	if (time_dispatch) {
		NoteDispatchQueue(0, IsAsyncComputeQueue(), UINT32_MAX, 0,
		                  static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                            std::chrono::steady_clock::now() - cpu_begin)
		                                            .count()));
	}
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_args != 0) {
		DrawPipe::Drain(DrawPipe::Reason::GpuState);
		DrawIndirectArgs args {};
		std::memcpy(&args, reinterpret_cast<const void*>(m_num_instances_args), sizeof(args));
		m_num_instances      = args.instance_count;
		m_num_instances_args = 0;
	}
	return m_num_instances;
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	const CpuCost::Scope cpu_cost('A');
	Libs::Graphics::SyncToggle::GpuSlowSpin();
	{
		const LoggedDraw logged {"DrawIndexAuto", args.vertex_count, args.instance_count,
		                         args.first_instance, static_cast<int32_t>(args.first_vertex),
		                         static_cast<int>(args.offset_source), args.gpu_args_address};
		if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
			const auto& cursor = g_current_execution->m_buffer_stack.back();
			LogIndexedDraw(*this, logged, m_start_instance_register, m_draw_indirect_args_base_addr,
			               cursor.commands, cursor.offset_dw);
		} else {
			LogIndexedDraw(*this, logged, m_start_instance_register, m_draw_indirect_args_base_addr,
			               {}, 0);
		}
	}
	if (args.instance_count == 0 && args.gpu_args_address == 0) {
		if (m_num_instances_args != 0 && GpuIndirectDraws()) {
			// The instance count a DRAW_INDIRECT read on the GPU left: the GPU reads it too
			// (RenderExecutor::DrawAuto). Reading it here drained the GPU (2.35 % of the GPU
			// thread at the jungle's Leap Attack prompt, 2026-10-05). Read at this draw, not
			// latched at the indirect draw: the arguments are not rewritten in between.
			args.gpu_instance_count_address = m_num_instances_args + sizeof(uint32_t);
		} else {
			args.instance_count = NumInstances();
		}
	}
	if (ImgRef::Address() != 0) {
		const auto frame = m_renderer.GetGpu().GetFrameNum();
		const auto& ps   = m_sh_ctx.GetPs();
		ImgRef::Scan("draw-ps", ps.ps_regs.data_addr, ps.ps_user_sgpr, frame, ps.ps_regs.user_data_addr);
		const auto& vs = m_sh_ctx.GetVs();
		ImgRef::Scan("draw-gs", vs.gs_regs.data_addr, vs.gs_user_sgpr, frame);
		ImgRef::Scan("draw-hs", vs.hs_regs.data_addr, vs.hs_user_sgpr, frame);
		static const char* const kRt[8] = {"draw-rt0", "draw-rt1", "draw-rt2", "draw-rt3",
		                                   "draw-rt4", "draw-rt5", "draw-rt6", "draw-rt7"};
		for (uint32_t slot = 0; slot < 8; slot++) {
			ImgRef::Target(kRt[slot], ps.ps_regs.data_addr, m_ctx.GetRenderTarget(slot).base.addr, frame);
		}
		const auto& depth = m_ctx.GetDepthRenderTarget();
		ImgRef::Target("draw-z", ps.ps_regs.data_addr, depth.z_write_base_addr, frame);
		ImgRef::Target("draw-stencil", ps.ps_regs.data_addr, depth.stencil_write_base_addr, frame);
		ImgRef::Target("draw-htile", ps.ps_regs.data_addr, depth.htile_data_base_addr, frame);
	}
	if (DrawPipe::DrawPacket() && args.gpu_args_address == 0 &&
	    PipeDraw(DrawPipe::Kind::DrawAuto, nullptr, &args)) {
		return;
	}
	Occlusion::NoteGuestDraw();
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
	CompleteDraw();
}

// KYTY_DRAW_PIPE: hands a direct draw to the draw thread with a copy of the registers and its
// programs, selected here. False: the draw thread has run dry and the caller draws as before.
bool CommandProcessor::PipeDraw(DrawPipe::Kind kind, const DrawIndexArgs* index_args,
                                const DrawAutoArgs* auto_args) {
	auto& packet = DrawPipe::Acquire(m_renderer);
	packet.kind      = kind;
	packet.submit_id = m_submit_id;
	packet.cp        = this;
	packet.ctx       = m_ctx;
	packet.ucfg      = m_ucfg;
	packet.sh        = m_sh_ctx;
	const bool empty = kind == DrawPipe::Kind::DrawIndex
	                       ? index_args->index_count == 0 || index_args->instance_count == 0
	                       : auto_args->vertex_count == 0 ||
	                             (auto_args->instance_count == 0 &&
	                              auto_args->gpu_instance_count_address == 0);
	if (kind == DrawPipe::Kind::DrawIndex) {
		packet.index_args = *index_args;
	} else {
		packet.auto_args = *auto_args;
	}
	if (empty || !RenderExecutor::PrepareDrawProgramsForPipe(m_renderer, packet.ctx, packet.sh,
	                                                         packet.ucfg, packet.prepared)) {
		DrawPipe::Cancel();
		DrawPipe::Drain(DrawPipe::Reason::Unready);
		DrawPipe::NoteSyncDraw();
		return false;
	}
	// Writes the buffer V#s bound queue with their ranges (stage A drains when it reads one);
	// image or raw-pointer writes have no range and drain here.
	const bool writes = packet.prepared.writes_memory &&
	                    (packet.prepared.writes_unbounded || !DrawPipe::RangesEnabled());
	DrawPipe::Publish();
	if (writes) {
		DrawPipe::Drain(DrawPipe::Reason::Writes);
	}
	return true;
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	// The presenter completes flips under the renderer mutex (KYTY_ASYNC_THREAD holds it here).
	const RenderContext::UnlockedScope unlocked(m_renderer);
	const bool stats = RenderStats::Enabled();
	const auto wait_begin = stats ? std::chrono::steady_clock::now()
	                              : std::chrono::steady_clock::time_point {};
	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
	if (stats) {
		RenderStats::NoteFlipWait(static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(
		        std::chrono::steady_clock::now() - wait_begin)
		        .count()));
	}
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);

	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	const bool at_completion = !with_interrupt && GuestGpu::LabelsAtCompletion();

	QueueStatsNoteLabelWrite(reinterpret_cast<uint64_t>(dst_gpu_addr));
	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		if (at_completion) {
			m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst), data,
			                                    sizeof(data));
		} else {
			const CpuCost::Phase label_phase_("Eop::LabelWrite");
			SettleUploadReads();
			std::memcpy(dst, &data, sizeof(data));
			LibKernel::GuestWait::NoteEvent(LibKernel::GuestWait::Event::Label);
		}

		const CpuCost::Phase record_phase_("Eop::Record");
		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadGpuTimestamp();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					if (at_completion) {
						m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst),
						                                    value, sizeof(value));
					} else {
						const CpuCost::Phase label_phase_("Eop::LabelWrite");
						SettleUploadReads();
						std::memcpy(dst, &value, sizeof(value));
						LibKernel::GuestWait::NoteEvent(LibKernel::GuestWait::Event::Label);
					}
					const CpuCost::Phase record_phase_("Eop::Record");

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier(std::source_location site) {
	Common::LockGuard lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	if (BarrierSinkEnabled()) {
		// Rendering ends there only if the barrier cannot stay pending (barrierSink.h).
		CurrentBuffer().Recorder().guestPipelineBarrier2(dependency, site);
		return;
	}
	GetScheduler().EndRendering();
	CurrentBuffer().Recorder().pipelineBarrier2(dependency, site);
}

void CommandProcessor::EmitGuestBarrier(std::source_location site) {
	if (SyncToggle::NoGuestBarriers()) {
		return;
	}
	EmitGlobalBarrier(site);
}

void CommandProcessor::EmitPartialFlushBarrier(uint32_t event_type, std::source_location site) {
	// Kept with SyncToggle::NoGuestBarriers(): dispatch runs (KYTY_DISPATCH_RUNS) record no barrier
	// between dispatches and rely on the guest's partial flushes to order dependent ones (v331
	// dropped them too: a device fault at boot).
	static const bool precise_env = [] {
		const char* value = std::getenv("KYTY_PRECISE_FLUSHES");
		return value != nullptr && value[0] == '1';
	}();
	if (!precise_env && !SyncToggle::PreciseFlushWindow()) {
		EmitGlobalBarrier(site);
		return;
	}
	// The hardware's partial flush waits for the waves of one stage to finish; the cache
	// writebacks and invalidations come with their own events. The barrier's source is that
	// stage instead of every command, so other stages' work keeps running across it.
	vk::PipelineStageFlags2 source {};
	switch (event_type) {
		case 0x00000007: source = vk::PipelineStageFlagBits2::eComputeShader; break; // CS
		case 0x0000000f: source = vk::PipelineStageFlagBits2::ePreRasterizationShaders; break; // VS/GS
		default: // PS: the pixel shader stage and its attachment writes
			source = vk::PipelineStageFlagBits2::eFragmentShader |
			         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
			         vk::PipelineStageFlagBits2::eLateFragmentTests |
			         vk::PipelineStageFlagBits2::eColorAttachmentOutput;
			break;
	}
	Common::LockGuard lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = source;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	if (BarrierSinkEnabled()) {
		CurrentBuffer().Recorder().guestPipelineBarrier2(dependency, site);
		return;
	}
	GetScheduler().EndRendering();
	CurrentBuffer().Recorder().pipelineBarrier2(dependency, site);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	// KYTY_DRAW_PIPE: a barrier behind the draws in flight runs on the draw thread, in order.
	if (DrawPipe::InFlightHere() && DrawPipe::DeferrableEvent(event_type)) {
		DrawPipe::QueueEvent(*this, event_type, event_index);
		return;
	}
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitPartialFlushBarrier(event_type); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGuestBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGuestBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (Occlusion::Enabled()) {
				Occlusion::Dump(m_renderer, CurrentBuffer(), event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			// Profiling: KYTY_OCCLUSION_ZERO=1 reports every query as occluded (no samples), the
			// upper bound of what real occlusion results could cull. KYTY_FPS_LOG prints the rate.
			static const bool occlusion_zero = [] {
				const char* v = std::getenv("KYTY_OCCLUSION_ZERO");
				return v != nullptr && v[0] == '1';
			}();
			// KYTY_OCCLUSION_STEP=<n>: the synthetic counter advances by n per dump (default 1), so a
			// query reads as n passed samples per DB; games that scale an effect by the visible
			// fraction of a query see it as fully visible with a large n.
			static const uint64_t occlusion_step = [] {
				const char* v = std::getenv("KYTY_OCCLUSION_STEP");
				return v != nullptr ? std::strtoull(v, nullptr, 10) : 1ull;
			}();
			if (!occlusion_zero) {
				m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + occlusion_step) & counter_mask;
			}
			if (static const bool log = Common::FpsLogDiagnostics(); log) {
				static auto     start = std::chrono::steady_clock::now();
				static uint64_t dumps = 0;
				dumps++;
				const auto now = std::chrono::steady_clock::now();
				const auto s   = std::chrono::duration<double>(now - start).count();
				if (s >= 5.0) {
					std::fprintf(stderr, "[occl] %.1f dumps/s\n", static_cast<double>(dumps) / s);
					dumps = 0;
					start = now;
				}
			}
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	SettleUploadReads();
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	LibKernel::GuestWait::NoteEvent(LibKernel::GuestWait::Event::Label);
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	SettleUploadReads();
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	LibKernel::GuestWait::NoteEvent(LibKernel::GuestWait::Event::Label);
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	(KYTY_SUBMIT_REASON(), GetScheduler().Flush());
}

void CommandProcessor::SynchronizeGpu() {
	(KYTY_SUBMIT_REASON(), GetScheduler().Finish());
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

void CommandProcessor::AsyncSegmentEnter() {
	if (!t_segment_held) {
		// Graphics right of way: while the graphics thread waits for the mutex, the async thread
		// does not take it (bounded, so the graphics thread's label waits on compute work cannot
		// starve it).
		if (RenderContext::PoliteAsync() && RenderContext::GraphicsWaiting()) {
			const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
			while (RenderContext::GraphicsWaiting() && std::chrono::steady_clock::now() < until) {
				std::this_thread::yield();
			}
		}
		m_renderer.LockRenderer();
		t_segment_held = true;
	}
	if (g_bound_cp != this) {
		BufferInit();
	}
	if (RenderExecutor::QueuedDispatchCount() != 0) {
		m_renderer.GetRenderExecutor().CommitQueuedDispatches(CurrentBuffer());
	}
}

void CommandProcessor::AsyncSegmentLeave() {
	if (RenderExecutor::QueuedDispatchCount() != 0) {
		AsyncSegmentEnter();
	}
	if (t_segment_held) {
		m_renderer.UnlockRenderer();
		t_segment_held = false;
	}
}

bool GuestGpu::AsyncThreadConfigured() noexcept {
	return AsyncThreadEnabled();
}

bool GuestGpu::AsyncThreadActiveNow() noexcept {
	return AsyncThreadEnabled() && AsyncThreadActive();
}

bool GuestGpu::IsAsyncThread() noexcept {
	return g_gpu_thread && g_gpu_state == nullptr && !g_draw_pipe_thread;
}

void GuestGpu::MarkDrawPipeThread() noexcept {
	g_gpu_thread       = true;
	g_draw_pipe_thread = true;
}

void GuestGpu::RebindCurrentProcessor() {
	if (g_current_processor != nullptr) {
		g_current_processor->BufferInit();
	}
}

} // namespace Libs::Graphics
