#ifndef EMULATOR_SRC_KERNEL_GUESTWAIT_H_
#define EMULATOR_SRC_KERNEL_GUESTWAIT_H_

#include <cstdint>

// KYTY_GUEST_WAIT_STATS=1 (v445r, counters only): what the game's threads do between the GPU
// thread's work. Two reports every 5 s, both per presented frame:
//
// [guest-wait]   per thread, the time it was blocked inside the emulator, by wait primitive
//                (event queue, condition variable, contended mutex, semaphore, sleep, the page
//                fault handler by outcome, the wait for a GPU readback by step, ...), and the
//                heaviest (thread, primitive, object) entries. Times are self times: a wait
//                inside another (the mutex taken again after a condition wait, a readback inside
//                the fault handler) is counted once, in the inner one.
// [guest-submit] per submitting thread and kind of submission, the time from the event the game
//                most likely waited for (the latest of: a label written, an EOP interrupt
//                delivered, a flip done, this thread's own readback finished) to the submission
//                reaching the GPU thread's queue, the time since this thread last came back from
//                an emulator wait, and the GPU thread's "empty" waits by what ended them.
//
// What it cannot see: a thread polling a label in guest memory without calling the emulator (it
// runs guest code; KYTY_SAMPLE_GUEST shows such a thread as busy) unless the page is protected,
// when each poll is a fault (fault-clean-read).
//
// Off: every hook is one load of Detail::g_enabled and a branch.
namespace Libs::LibKernel::GuestWait {

enum class Prim : uint8_t {
	Equeue,        // sceKernelWaitEqueue
	Cond,          // pthread condition variable wait (the mutex taken again is under Mutex)
	Mutex,         // pthread mutex lock that found the mutex held
	Rwlock,        // pthread rwlock lock that had to wait
	Sema,          // kernel and POSIX semaphores
	EventFlag,     // sceKernelWaitEventFlag
	SyncOnAddress, // sceKernelSyncOnAddressWait, _umtx_op wait
	Sleep,         // usleep, nanosleep, sleep
	Yield,         // pthread_yield
	Vblank,        // sceVideoOutWaitVblank
	SuspendPoint,  // sceAgcSuspendPoint
	MemLock,       // the kernel's memory-operation lock, when another thread held it
	GpuSync,       // a command run on the GPU thread while this thread waits (unmaps, ...)
	FaultStale,    // fault handler: a page left stricter than needed, fixed (lazy release)
	FaultCleanRead, // fault handler: a read of bytes the GPU did not write, emulated
	FaultCleanLift, // fault handler: such a read on a page with no GPU-written bytes left, lifted (v445s)
	FaultWrite,    // fault handler: a write to a watched page (its own time, readback apart)
	FaultRead,     // fault handler: a read of GPU-written bytes (its own time, readback apart)
	FaultOther,    // fault handler: anything else
	ReadbackQueue, // readback: waiting for the GPU thread to take the request
	ReadbackExec,  // readback: the GPU thread running the request (recording the download)
	ReadbackCopy,  // readback: waiting for the copy queue's download
	ReadbackTick,  // readback: waiting for the host GPU to reach the download's tick
	Count
};

enum class Event : uint8_t { Label, Eop, Flip, Vblank, Count };

enum class Submit : uint8_t { Gfx, Compute, FlipPrep, SuspendPoint, Command, Count };

namespace Detail {
extern bool g_enabled; // set once, before main
struct ObjectSlot;
void BeginSlow(Prim prim, const volatile void* object, const char* name, uint64_t& begin, uint64_t& saved_child,
               ObjectSlot*& slot, uint64_t& key) noexcept;
void EndSlow(Prim prim, uint64_t begin, uint64_t saved_child, ObjectSlot* slot, uint64_t key) noexcept;
void NoteEventSlow(Event event) noexcept;
void NoteSubmitSlow(Submit kind, bool wakes_gpu_thread) noexcept;
void GpuEmptyBeginSlow() noexcept;
void GpuEmptyEndSlow() noexcept;
void OnGuestFlipSlow();
} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled;
}

// Steady clock, nanoseconds.
[[nodiscard]] uint64_t NowNs() noexcept;

// The calling thread's time from here to the scope's end, as `prim`. `object` (with its `name`,
// copied at once) itemises it in the top list; nullptr: the primitive's total only.
class Scope {
public:
	Scope() noexcept = default;
	explicit Scope(Prim prim, const volatile void* object = nullptr, const char* name = nullptr) noexcept
	    : m_prim(prim) {
		if (Enabled()) [[unlikely]] {
			Detail::BeginSlow(prim, object, name, m_begin, m_saved_child, m_slot, m_key);
		}
	}
	// For a wait that is only sometimes reached: starts the scope there, once.
	void Begin(Prim prim, const volatile void* object = nullptr, const char* name = nullptr) noexcept {
		if (Enabled() && m_begin == 0) [[unlikely]] {
			m_prim = prim;
			Detail::BeginSlow(prim, object, name, m_begin, m_saved_child, m_slot, m_key);
		}
	}
	// The primitive is known only at the end (the fault handler's outcome).
	void SetPrim(Prim prim) noexcept { m_prim = prim; }
	~Scope() {
		if (m_begin != 0) [[unlikely]] {
			Detail::EndSlow(m_prim, m_begin, m_saved_child, m_slot, m_key);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	uint64_t            m_begin       = 0; // 0: not running
	uint64_t            m_saved_child = 0;
	Detail::ObjectSlot* m_slot        = nullptr;
	uint64_t            m_key         = 0;
	Prim                m_prim        = Prim::Count;
};

// Enabled() only. `ns` of the calling thread's time as `prim`, taken out of the scope open around
// it (a step timed by the caller).
void AddTime(Prim prim, uint64_t ns) noexcept;
// Enabled() only. GuestGpu::SendCommandSync's scope (Prim::GpuSync), unless the caller times the
// round trip itself (SuppressGpuSync).
void BeginGpuSync(Scope& scope) noexcept;
class SuppressGpuSync {
public:
	explicit SuppressGpuSync(bool on) noexcept;
	~SuppressGpuSync();
	SuppressGpuSync(const SuppressGpuSync&)            = delete;
	SuppressGpuSync& operator=(const SuppressGpuSync&) = delete;

private:
	bool m_on;
	bool m_previous = false;
};
// Enabled() only. A readback this thread waited for has finished ([guest-submit]'s event).
void NoteReadbackEnd() noexcept;

// An event a game thread may be waiting for was delivered (any thread).
inline void NoteEvent(Event event) noexcept {
	if (Enabled()) [[unlikely]] {
		Detail::NoteEventSlow(event);
	}
}

// Under the GPU thread's queue mutex: the calling thread queued a submission or a command;
// `wakes_gpu_thread`: it signals the GPU thread (role 0).
inline void NoteSubmit(Submit kind, bool wakes_gpu_thread) noexcept {
	if (Enabled()) [[unlikely]] {
		Detail::NoteSubmitSlow(kind, wakes_gpu_thread);
	}
}

// The GPU thread (role 0), under its queue mutex, around its wait for work.
inline void GpuEmptyBegin() noexcept {
	if (Enabled()) [[unlikely]] {
		Detail::GpuEmptyBeginSlow();
	}
}
inline void GpuEmptyEnd() noexcept {
	if (Enabled()) [[unlikely]] {
		Detail::GpuEmptyEndSlow();
	}
}

// Every presented frame (RenderDocOnGuestFlip): counts it and prints the reports every 5 s.
inline void OnGuestFlip() {
	if (Enabled()) [[unlikely]] {
		Detail::OnGuestFlipSlow();
	}
}

} // namespace Libs::LibKernel::GuestWait

#endif // EMULATOR_SRC_KERNEL_GUESTWAIT_H_
