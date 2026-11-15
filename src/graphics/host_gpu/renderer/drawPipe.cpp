#include "graphics/host_gpu/renderer/drawPipe.h"

#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/fpsLog.h"
#include "common/profiler.h"
#include "common/sampler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderStats.h"

#include <atomic>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

namespace Libs::Graphics::DrawPipe {

namespace {

// Packets in the ring. Stage B runs a draw in ~10 us; 64 is far more than it falls behind
// between drains, and bounds what a drain waits for.
constexpr uint64_t kSlots = 64;
// Spins (CPU pause) before a thread sleeps on the other: ~20-40 us.
constexpr uint32_t kSpins = 4096;

inline void CpuRelax() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#endif
}

struct Pipe {
	RenderContext*              context = nullptr;
	std::unique_ptr<Packet[]>   slots;
	// Packets published by stage A / finished by stage B (monotonic). Stage B waits on
	// `published`, stage A on `finished`.
	std::atomic<uint64_t>       published {0};
	std::atomic<uint64_t>       finished {0};
	uint64_t                    acquired = 0; // stage A: Acquire handed out slot published
	bool                        pending  = false;
	std::thread                 thread;
	std::once_flag              started;
	// Statistics (stage A writes all but busy_ns).
	uint64_t                    pipelined  = 0;
	uint64_t                    events     = 0;
	uint64_t                    sync_draws = 0;
	uint64_t                    drains[static_cast<size_t>(Reason::Count)] = {};
	uint64_t                    drain_waits = 0; // drains that found packets in flight
	uint64_t                    wait_ns     = 0; // stage A waiting (drains and a full ring)
	uint64_t                    full_waits  = 0;
	std::atomic<uint64_t>       busy_ns {0};      // stage B running packets
	// The writes of queued draws: (packet number, begin, end). Stage A adds before publishing,
	// stage B drops a packet's entries after recording it; readers on any thread look.
	std::mutex                                        pending_mutex;
	std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> pending_writes;
	std::atomic<uint32_t>                             pending_count {0};
	uint64_t                                          ranged = 0; // stage A: writers queued with ranges
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
	uint32_t                    drain_opcodes[256] = {};
};

Pipe& GetPipe() {
	static Pipe pipe;
	return pipe;
}

thread_local bool                    t_stage_a     = false;
thread_local bool                    t_stage_b     = false;
thread_local const PreparedPrograms* t_prepared    = nullptr;
thread_local bool                    t_draw_packet = false;

void RunPacket(Pipe& pipe, Packet& packet) {
	// The whole packet under the renderer mutex: stage A takes it for guest reads of GPU-written
	// bytes while preparing later draws (ReadShaderGuestMemory), and RenderExecutor::DrawIndex
	// pops pending operations before taking it itself.
	Common::LockGuard lock(pipe.context->GetMutex());
	if (packet.kind == Kind::Event) {
		packet.cp->TriggerEvent(packet.event_type, packet.event_index, 0);
		return;
	}
	auto& scheduler = pipe.context->GetCommandScheduler();
	auto& executor  = pipe.context->GetRenderExecutor();
	const auto bound = scheduler.BoundRegisters();
	scheduler.RebindRegisters(packet.ctx, packet.ucfg, packet.sh);
	t_prepared = &packet.prepared;
	Occlusion::NoteGuestDraw();
	if (packet.kind == Kind::DrawIndex) {
		executor.RunPipeDraw(packet.submit_id, scheduler.Current(), &packet.index_args, nullptr);
	} else {
		executor.RunPipeDraw(packet.submit_id, scheduler.Current(), nullptr, &packet.auto_args);
	}
	packet.cp->CompleteDraw();
	t_prepared = nullptr;
	scheduler.RebindRegisters(*bound.registers, *bound.user_config, *bound.shaders);
}

void StageBMain(Pipe* pipe) {
	KYTY_PROFILER_THREAD("Thread_GpuDraw");
	Common::PreferPerformanceCores("GPU draw thread (KYTY_DRAW_PIPE)");
	Common::RaiseGpuThreadPriority();
	Common::Sampler::RegisterCurrentThread("gpu-draw");
	GuestGpu::MarkDrawPipeThread();
	t_stage_b = true;
	uint64_t next = 0;
	for (;;) {
		auto published = pipe->published.load(std::memory_order_acquire);
		// A short spin first: the command processor publishes the next draw within microseconds
		// in a run of draws, and a sleep and wake-up cost more than that.
		for (uint32_t spin = 0; published == next && spin < kSpins; spin++) {
			CpuRelax();
			published = pipe->published.load(std::memory_order_acquire);
		}
		while (published == next) {
			pipe->published.wait(published, std::memory_order_acquire);
			published = pipe->published.load(std::memory_order_acquire);
		}
		const auto begin = std::chrono::steady_clock::now();
		for (; next != published; next++) {
			auto& packet = pipe->slots[next % kSlots];
			RunPacket(*pipe, packet);
			if (packet.kind != Kind::Event && !packet.prepared.write_ranges.empty()) {
				// Recorded: the buffer cache marks those bytes GPU-written now.
				std::lock_guard lock(pipe->pending_mutex);
				std::erase_if(pipe->pending_writes, [&](const auto& entry) { return std::get<0>(entry) <= next; });
				pipe->pending_count.store(static_cast<uint32_t>(pipe->pending_writes.size()), std::memory_order_release);
			}
			pipe->finished.store(next + 1, std::memory_order_release);
			pipe->finished.notify_all();
		}
		pipe->busy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                  std::chrono::steady_clock::now() - begin)
		                                                  .count()),
		                        std::memory_order_relaxed);
	}
}

void WaitFinished(Pipe& pipe, uint64_t target) {
	auto finished = pipe.finished.load(std::memory_order_acquire);
	if (finished >= target) {
		return;
	}
	const auto begin = std::chrono::steady_clock::now();
	for (uint32_t spin = 0; finished < target && spin < kSpins; spin++) {
		CpuRelax();
		finished = pipe.finished.load(std::memory_order_acquire);
	}
	while (finished < target) {
		pipe.finished.wait(finished, std::memory_order_acquire);
		finished = pipe.finished.load(std::memory_order_acquire);
	}
	pipe.wait_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                          std::chrono::steady_clock::now() - begin)
	                                          .count());
}

void MaybeLog(Pipe& pipe) {
	static const bool log = Common::FpsLogDiagnostics();
	if (!log) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - pipe.last_log < std::chrono::seconds(5)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - pipe.last_log).count();
	pipe.last_log        = now;
	const auto   busy    = pipe.busy_ns.exchange(0, std::memory_order_relaxed);
	const auto   per_s   = [&](uint64_t value) { return static_cast<double>(value) / seconds; };
	std::string  top;
	for (int round = 0; round < 5; round++) {
		uint32_t best = 0, best_count = 0;
		for (uint32_t op = 0; op < 256; op++) {
			if (pipe.drain_opcodes[op] > best_count) {
				best       = op;
				best_count = pipe.drain_opcodes[op];
			}
		}
		if (best_count == 0) break;
		char item[32];
		std::snprintf(item, sizeof(item), " 0x%02x:%.0f", best, per_s(best_count));
		top += item;
		pipe.drain_opcodes[best] = 0;
	}
	for (auto& count: pipe.drain_opcodes) count = 0;
	const auto& d = pipe.drains;
	std::fprintf(stderr,
	             "[draw-pipe] 5 s: pipelined %.0f/s, events %.0f/s, on stage A %.0f/s | drains/s packet %.0f commands %.0f "
	             "slice %.0f fault %.0f gpu-state %.0f writes %.0f pending-read %.0f unready %.0f (waited %.0f/s, full %.0f/s) | writers ranged %.0f/s | "
	             "stage A waiting %.0f ms/s, stage B busy %.0f ms/s | drain packets:%s\n",
	             per_s(pipe.pipelined), per_s(pipe.events), per_s(pipe.sync_draws), per_s(d[0]), per_s(d[1]), per_s(d[2]),
	             per_s(d[3]), per_s(d[4]), per_s(d[5]), per_s(d[6]), per_s(d[7]), per_s(pipe.drain_waits),
	             per_s(pipe.full_waits), per_s(pipe.ranged), per_s(pipe.wait_ns) / 1e6, per_s(busy) / 1e6, top.c_str());
	pipe.pipelined = pipe.events = pipe.sync_draws = pipe.ranged = pipe.drain_waits = pipe.full_waits = pipe.wait_ns = 0;
	for (auto& count: pipe.drains) count = 0;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_PIPE");
		if (value == nullptr || value[0] != '1') {
			return false;
		}
		if (GuestGpu::AsyncThreadConfigured() || RenderStats::IndirectReplayConfigured()) {
			std::fprintf(stderr, "[draw-pipe] off: needs the async-compute thread and indirect replay off\n");
			return false;
		}
		std::fprintf(stderr, "[draw-pipe] on: direct draws run on a second thread\n");
		return true;
	}();
	return enabled;
}

bool InFlightHere() {
	if (!t_stage_a) {
		return false;
	}
	auto& pipe = GetPipe();
	return pipe.finished.load(std::memory_order_acquire) != pipe.published.load(std::memory_order_relaxed);
}

bool IsStageB() {
	return t_stage_b;
}

Packet& Acquire(RenderContext& context) {
	auto& pipe = GetPipe();
	std::call_once(pipe.started, [&] {
		pipe.context = &context;
		pipe.slots   = std::make_unique<Packet[]>(kSlots);
		pipe.thread  = std::thread(StageBMain, &pipe);
		pipe.thread.detach();
	});
	t_stage_a = true;
	EXIT_IF(pipe.pending);
	const auto published = pipe.published.load(std::memory_order_relaxed);
	if (published - pipe.finished.load(std::memory_order_acquire) >= kSlots) {
		pipe.full_waits++;
		WaitFinished(pipe, published - kSlots + 1);
	}
	pipe.pending = true;
	return pipe.slots[published % kSlots];
}

void Publish() {
	auto& pipe = GetPipe();
	EXIT_IF(!pipe.pending);
	const auto number = pipe.published.load(std::memory_order_relaxed);
	auto&      packet = pipe.slots[number % kSlots];
	if (packet.kind != Kind::Event && !packet.prepared.write_ranges.empty()) {
		std::lock_guard lock(pipe.pending_mutex);
		for (const auto& [begin, end]: packet.prepared.write_ranges) {
			pipe.pending_writes.emplace_back(number, begin, end);
		}
		pipe.pending_count.store(static_cast<uint32_t>(pipe.pending_writes.size()), std::memory_order_release);
		pipe.ranged++;
	}
	pipe.pending = false;
	pipe.pipelined++;
	pipe.published.fetch_add(1, std::memory_order_release);
	pipe.published.notify_one();
	MaybeLog(pipe);
}

void Cancel() {
	auto& pipe = GetPipe();
	EXIT_IF(!pipe.pending);
	pipe.pending = false;
}

void Drain(Reason reason, uint32_t opcode) {
	if (!t_stage_a) {
		return;
	}
	auto& pipe = GetPipe();
	EXIT_IF(pipe.pending);
	const auto published = pipe.published.load(std::memory_order_relaxed);
	if (pipe.finished.load(std::memory_order_acquire) == published) {
		return;
	}
	pipe.drains[static_cast<size_t>(reason)]++;
	if (reason == Reason::Packet) {
		pipe.drain_opcodes[opcode & 0xffu]++;
	}
	pipe.drain_waits++;
	WaitFinished(pipe, published);
}

bool DeferrableEvent(uint32_t event_type) {
	switch (event_type) {
		// Partial flushes and cache writebacks: barriers (TriggerEvent).
		case 0x07u:
		case 0x0fu:
		case 0x10u:
		case 0x16u:
		case 0x31u:
		case 0x2au:
		case 0x2cu:
		case 0x2eu: return true;
		default: return false;
	}
}

void QueueEvent(CommandProcessor& cp, uint32_t event_type, uint32_t event_index) {
	auto& pipe = GetPipe();
	auto& packet       = Acquire(*pipe.context);
	packet.kind        = Kind::Event;
	packet.cp          = &cp;
	packet.event_type  = event_type;
	packet.event_index = event_index;
	pipe.events++;
	pipe.pending = false;
	pipe.published.fetch_add(1, std::memory_order_release);
	pipe.published.notify_one();
}

bool PendingWriteOverlaps(uint64_t address, uint64_t size) {
	auto& pipe = GetPipe();
	if (pipe.pending_count.load(std::memory_order_acquire) == 0 || size == 0) {
		return false;
	}
	std::lock_guard lock(pipe.pending_mutex);
	const auto end = address + size;
	for (const auto& [number, begin, stop]: pipe.pending_writes) {
		if (begin < end && address < stop) {
			return true;
		}
	}
	return false;
}

void DrainIfPendingWrite(uint64_t address, uint64_t size) {
	if (InFlightHere() && PendingWriteOverlaps(address, size)) {
		Drain(Reason::PendingRead);
	}
}

bool HasPendingWrites() {
	return GetPipe().pending_count.load(std::memory_order_acquire) != 0;
}

bool RangesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_PIPE_RANGES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

const PreparedPrograms* CurrentPrepared() {
	return t_prepared;
}

void SetDrawPacket(bool draw_packet) {
	t_draw_packet = draw_packet;
}

bool DrawPacket() {
	return t_draw_packet;
}

void NoteSyncDraw() {
	auto& pipe = GetPipe();
	pipe.sync_draws++;
	MaybeLog(pipe);
}

} // namespace Libs::Graphics::DrawPipe
