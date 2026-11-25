#include "graphics/host_gpu/renderer/sync.h"

#include "common/assert.h"
#include "common/floorStats.h"
#include "common/frameSkips.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/presentRt.h"
#include "graphics/host_gpu/renderer/rtDrawLog.h"
#include "graphics/host_gpu/renderer/imgTrace.h"
#include "graphics/host_gpu/renderer/gpuPad.h"
#include "graphics/host_gpu/renderer/queueShare.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/renderer/rtSnap.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <cstring>
#include <string>
#include <cstdio>
#include <chrono>
#include <limits>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics::Sync {

constexpr uint64_t GRAPHICS_REFERENCE_CLOCK_FREQUENCY = 100000000;

bool ScaleReferenceClock(uint64_t host_ticks, uint64_t host_frequency, uint64_t& value) {
	if (host_frequency == 0) {
		return false;
	}

	const auto     whole_seconds = host_ticks / host_frequency;
	const auto     remainder     = host_ticks % host_frequency;
	constexpr auto MAX_VALUE     = std::numeric_limits<uint64_t>::max();
	if (whole_seconds > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY ||
	    remainder > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY) {
		return false;
	}

	const auto whole_value      = whole_seconds * GRAPHICS_REFERENCE_CLOCK_FREQUENCY;
	const auto fractional_value = (remainder * GRAPHICS_REFERENCE_CLOCK_FREQUENCY) / host_frequency;
	if (whole_value > MAX_VALUE - fractional_value) {
		return false;
	}
	value = whole_value + fractional_value;
	return true;
}

uint64_t ReadReferenceClock() {
	const auto host_frequency = LibKernel::KernelGetTscFrequency();
	const auto host_ticks     = LibKernel::KernelReadTsc();
	uint64_t   value          = 0;
	if (!ScaleReferenceClock(host_ticks, host_frequency, value)) {
		EXIT("cannot scale host clock, ticks=0x%016" PRIx64 " frequency=%" PRIu64 "\n", host_ticks,
		     host_frequency);
	}
	return value;
}

// KYTY_GPU_TIMESTAMP_STEP=<reference-clock ticks> (experiment, 100 MHz clock): end-of-pipe
// timestamps the guest asks the GPU to write come from the host clock when the command processor
// translates them, so a pass's measured GPU time is the emulator's translation time (tens of
// milliseconds at 15 fps). An engine that scales its work by measured GPU time sees it is far over
// budget every frame. With a step, each flip anchors the timestamps at the host clock and every
// timestamp write in the frame advances them by the step only: the guest's GPU passes look fast.
// Diagnostic: shows whether the game drops work (decals, shadows) by its GPU-time budget.
namespace {
std::atomic<uint64_t> g_timestamp_flips {0};
} // namespace

void NoteTimestampFlip() {
	g_timestamp_flips.fetch_add(1, std::memory_order_relaxed);
}

uint64_t GuestFlipCount() {
	return g_timestamp_flips.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::Sync

namespace Libs::Graphics::ImgTrace {
uint64_t CurrentFlip() {
	return Sync::GuestFlipCount();
}
} // namespace Libs::Graphics::ImgTrace

namespace Libs::Graphics::Sync {

uint64_t ReadGpuTimestamp() {
	static const uint64_t step = [] {
		const char* value = std::getenv("KYTY_GPU_TIMESTAMP_STEP");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0};
	}();
	if (step == 0) {
		return ReadReferenceClock();
	}
	static std::mutex mutex;
	static uint64_t   anchor_flip = ~uint64_t {0};
	static uint64_t   last        = 0;
	std::lock_guard   lock(mutex);
	const auto        flip = g_timestamp_flips.load(std::memory_order_relaxed);
	if (flip != anchor_flip) {
		anchor_flip = flip;
		last        = std::max(last + step, ReadReferenceClock());
		return last;
	}
	last += step;
	return last;
}

enum class EndOfPipeWriteSize : uint32_t { Dword = 4, Qword = 8 };
enum class EndOfPipeWriteAction { Write, WriteBack, Interrupt, InterruptWriteBack };

static CommandBufferDebugOp DebugOperation(EndOfPipeWriteAction action) {
	switch (action) {
		case EndOfPipeWriteAction::Write: return CommandBufferDebugOp::EopWrite;
		case EndOfPipeWriteAction::WriteBack:
		case EndOfPipeWriteAction::InterruptWriteBack: return CommandBufferDebugOp::EopWriteBack;
		case EndOfPipeWriteAction::Interrupt: return CommandBufferDebugOp::EopInterrupt;
	}
	EXIT("unsupported end-of-pipe write action\n");
	return CommandBufferDebugOp::Unknown;
}

static bool TriggersInterrupt(EndOfPipeWriteAction action) {
	return action == EndOfPipeWriteAction::Interrupt ||
	       action == EndOfPipeWriteAction::InterruptWriteBack;
}

static void RecordEndOfPipeWrite(uint64_t submit_id, CommandBuffer& buffer, uint64_t destination,
                                 uint64_t value, EndOfPipeWriteSize size,
                                 EndOfPipeWriteAction action, int interrupt_event_id = 0,
                                 uint32_t context_id = 0) {
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);
	EXIT_IF(destination == 0);
	EXIT_IF(buffer.IsInvalid());

	const auto width      = static_cast<uint32_t>(size);
	const auto value_low  = static_cast<uint32_t>(value);
	const auto value_high = static_cast<uint32_t>(value >> 32u);
	const auto operation  = static_cast<uint32_t>(DebugOperation(action));
	if (TriggersInterrupt(action)) {
		buffer.SetDebugInfo(operation, submit_id, width, context_id, value_low, value_high,
		                    destination);
		TriggerEopEventAtEndOfPipe(buffer, interrupt_event_id, context_id);
	} else {
		buffer.SetDebugInfo(operation, submit_id, width, value_low, value_high, 0, destination);
	}
}

void WriteAtEndOfPipe32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                        uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeGds32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                           uint32_t dw_offset, uint32_t dw_num) {
	EXIT_IF(dst_gpu_addr == nullptr);
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWrite), submit_id,
	                    dw_offset, dw_num, 0, 0, reinterpret_cast<uint64_t>(dst_gpu_addr));
}

void WriteAtEndOfPipe64(uint64_t submit_id, CommandBuffer& buffer, uint64_t* dst_gpu_addr,
                        uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeWithWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithInterruptWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                              uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterruptWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                              uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterrupt64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

void WriteAtEndOfPipeWithInterrupt32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

} // namespace Libs::Graphics::Sync
namespace Libs::Graphics {
uint64_t AllocCountTotal();      // allocStats.cpp
uint64_t AllocCountLocalTotal(); // allocStats.cpp
} // namespace Libs::Graphics
namespace Libs::Graphics::Sync {

// v445o (2): [cut-hits], one line per toggle window that ended (and one for the time before the
// first), when one of v445o's cuts is toggled: how often each cut's path and the path it replaces
// ran in that window. An on window must show its cut's counts, an off window none of them.
static void ReportCutHits() {
	if (!SyncToggle::CutsConfigured()) {
		return;
	}
	static bool     last_on = false;
	static uint64_t before[13] {};
	static uint64_t flips = 0;
	flips++;
	const bool on = SyncToggle::State().load(std::memory_order_relaxed);
	if (on == last_on) {
		return;
	}
	auto&          h      = SyncToggle::Hits();
	const auto&    floor  = Common::FloorStats::Get();
	const uint64_t now[13] = {h.window_scans.load(std::memory_order_relaxed),
	                          h.window_by_page.load(std::memory_order_relaxed),
	                          h.set_reused.load(std::memory_order_relaxed),
	                          h.set_busy.load(std::memory_order_relaxed),
	                          h.set_fresh.load(std::memory_order_relaxed),
	                          h.sync_local.load(std::memory_order_relaxed),
	                          floor.sync_calls.load(std::memory_order_relaxed),
	                          AllocCountLocalTotal(),
	                          AllocCountTotal(),
	                          h.loop_untimed.load(std::memory_order_relaxed),
	                          h.loop_timed.load(std::memory_order_relaxed),
	                          h.program_untimed.load(std::memory_order_relaxed),
	                          h.pair_memo.load(std::memory_order_relaxed)};
	static uint64_t pair_set_before = 0;
	// v445q ("rbsubmit"): copy-queue readbacks the worker submitted and the GPU thread did.
	static uint64_t rb_before[3] {};
	const uint64_t  rb[3] = {h.rb_thread.load(std::memory_order_relaxed), h.rb_inline.load(std::memory_order_relaxed),
	                         h.rb_handoff_ns.load(std::memory_order_relaxed)};
	const uint64_t  pair_set        = h.pair_set.load(std::memory_order_relaxed);
	uint64_t        d[13];
	for (int i = 0; i < 13; i++) {
		d[i]      = now[i] - before[i];
		before[i] = now[i];
	}
	const auto n = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	std::fprintf(stderr,
	             "[cut-hits] window %s, %llu flips | windowscan %llu (by page %llu) | setreuse reused %llu "
	             "busy %llu fresh %llu | syncstats local %llu of %llu | alloccount local %llu of %llu | "
	             "timers loop untimed %llu timed %llu, dispatch untimed %llu, pair memo %llu set %llu, "
	             "phase-off now %d | rbsubmit thread %llu inline %llu, hand-over to submitted avg %.1f us\n",
	             last_on ? "on" : "off", n(flips), n(d[0]), n(d[1]), n(d[2]), n(d[3]), n(d[4]), n(d[5]), n(d[6]),
	             n(d[7]), n(d[8]), n(d[9]), n(d[10]), n(d[11]), n(d[12]), n(pair_set - pair_set_before),
	             CpuCost::Detail::g_phase_off.load(std::memory_order_relaxed) ? 1 : 0, n(rb[0] - rb_before[0]),
	             n(rb[1] - rb_before[1]),
	             rb[0] != rb_before[0]
	                 ? static_cast<double>(rb[2] - rb_before[2]) * 1e-3 / static_cast<double>(rb[0] - rb_before[0])
	                 : 0.0);
	for (int i = 0; i < 3; i++) rb_before[i] = rb[i];
	pair_set_before = pair_set;
	flips           = 0;
	last_on         = on;
}

uint64_t PrepareVideoOutFlip(CommandBuffer& buffer, int handle, int index, int flip_mode,
                             int64_t flip_arg) {
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);
	Common::FrameSkips::Flip();
	NoteTimestampFlip();
	PresentRt::Flip();
	RtDrawLog::Flip();
	SyncToggle::AtFlip(GuestFlipCount());
	// v445o: the cuts that keep a per-flip state (KYTY_TIMERS_CUT, KYTY_SYNC_STATS_LOCAL).
	CpuCost::Detail::g_phase_off.store(SyncToggle::TimersCut() && !CpuCost::Enabled(),
	                                   std::memory_order_relaxed);
	FoldSyncCounters(); // this thread's per-sync [floor] counts
	ReportCutHits();
	QueueShare::FrameEnd();  // KYTY_QUEUE_STATS, [queue-share]: the frame's counts end here
	GpuPad::AtFlip(buffer);  // KYTY_SYNC_TOGGLE_MODE=gpupad, in the on windows: one extra dispatch
	RtSnap::AtFlip(buffer.GetContext());
	if (SyncToggle::IdleAtFlip()) {
		// Everything recorded so far, this frame included, has finished on the host GPU before
		// the flip is submitted; the game's next frame cannot overlap it.
		(KYTY_SUBMIT_REASON(), buffer.GetContext().GetCommandScheduler().Finish());
	}
	for (;;) {
		uint64_t   request_id = 0;
		auto&      video_out  = buffer.GetContext().GetVideoOut();
		const auto result =
		    video_out.SubmitFlipFromGpu(buffer, handle, index, flip_mode, flip_arg, request_id);
		if (result == OK) {
			EXIT_IF(request_id == 0);
			return request_id;
		}
		if (result != VideoOut::VIDEO_OUT_ERROR_FLIP_QUEUE_FULL) {
			EXIT("GPU flip submission failed, result=%d handle=%d index=%d mode=%d arg=%" PRId64
			     "\n",
			     result, handle, index, flip_mode, flip_arg);
		}
		// The presenter frees slots under the renderer mutex (KYTY_ASYNC_THREAD holds it here).
		const RenderContext::UnlockedScope unlocked(buffer.GetContext());
		const auto wait_begin = std::chrono::steady_clock::now();
		const auto before     = Common::FloorStats::Enabled() ? video_out.DebugFlipState() : std::string();
		video_out.WaitForSubmitSlot(handle);
		if (Common::FloorStats::Enabled()) {
			const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			                    std::chrono::steady_clock::now() - wait_begin)
			                    .count();
			if (ms >= 200) {
				std::fprintf(stderr, "[floor-flip] GPU thread waited %lld ms for a flip slot; queue before: %s\n",
				             static_cast<long long>(ms), before.c_str());
			}
		}
	}
}

void WriteAtEndOfPipeWithInterruptWriteBackFlip32(uint64_t submit_id, CommandBuffer& buffer,
                                                  uint32_t* dst_gpu_addr, uint32_t value,
                                                  int handle, int index, int flip_mode,
                                                  int64_t flip_arg, uint64_t request_id,
                                                  int event_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWriteBackFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation([&renderer, event_id, request_id] {
		renderer.GetVideoOut().CompleteFlip(request_id);
		renderer.TriggerInterrupt(event_id, 0);
	});
}

void WriteAtEndOfPipeWithFlip32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                                uint32_t value, int handle, int index, int flip_mode,
                                int64_t flip_arg, uint64_t request_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void WriteAtEndOfPipeOnlyFlip(uint64_t submit_id, CommandBuffer& buffer, int handle, int index,
                              int flip_mode, int64_t flip_arg, uint64_t request_id) {
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopOnlyFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), 0, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void TriggerEopEventAtEndOfPipe(CommandBuffer& buffer, int event_id, uint32_t context_id) {
	DrainProtectionHelper(ProtectDrainReason::GuestSignal);
	EXIT_IF(buffer.IsInvalid());
	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, event_id, context_id] { renderer.TriggerInterrupt(event_id, context_id); });
}

static void InterruptEventResetFunc(LibKernel::EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void InterruptEventTriggerFunc(LibKernel::EventQueue::KernelEqueueEvent* event,
                                      void*                                     trigger_data) {
	EXIT_IF(event == nullptr);

	auto triggered_event = event->event;
	triggered_event.fflags++;
	triggered_event.data = reinterpret_cast<intptr_t>(trigger_data);
	if (event->triggered) {
		event->pending_events.push_back(triggered_event);
	} else {
		event->event     = triggered_event;
		event->triggered = true;
	}
}

int AddEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id,
               void* udata) {
	LibKernel::EventQueue::KernelEqueueEvent event;
	event.triggered           = false;
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS;
	event.event.udata         = udata;
	event.event.fflags        = 0;
	event.event.data          = id;
	event.filter.reset_func   = InterruptEventResetFunc;
	event.filter.trigger_func = InterruptEventTriggerFunc;

	int result = LibKernel::EventQueue::KernelAddEvent(eq, event);

	if (result == 0) {
		renderer.AddInterruptEq(eq, id);
	}

	return result;
}

int DeleteEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id) {
	int result = LibKernel::EventQueue::KernelDeleteEvent(
	    eq, static_cast<uintptr_t>(id), LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS);
	if (result == OK || result == LibKernel::KERNEL_ERROR_ENOENT) {
		renderer.DeleteInterruptEq(eq, id);
	}

	return result;
}

void ReadGds(const Buffer& gds, uint32_t* dst, uint32_t dw_offset, uint32_t dw_size) {
	const auto offset = uint64_t {dw_offset} * sizeof(uint32_t);
	const auto size   = uint64_t {dw_size} * sizeof(uint32_t);
	EXIT_IF(dst == nullptr || offset > gds.Size() || size > gds.Size() - offset ||
	        gds.Mapped().empty());
	std::memcpy(dst, gds.Mapped().data() + offset, static_cast<size_t>(size));
}

} // namespace Libs::Graphics::Sync
