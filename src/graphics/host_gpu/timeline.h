#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_

#include "common/sampler.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

// Phase's constructor and destructor are inlined at every site (v445o): with KYTY_CPU_COST off
// they are a load and a branch each.
#if defined(_MSC_VER)
#define KYTY_CPU_COST_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define KYTY_CPU_COST_INLINE __attribute__((always_inline)) inline
#else
#define KYTY_CPU_COST_INLINE inline
#endif

// Diagnostics: KYTY_TIMELINE=<first frame>:<frames> logs "TL <microseconds> <thread> <event>"
// lines for guest submissions, GPU-thread processing, host GPU submissions and completions,
// drains and presents, for that window of presented frames. Off by default.
namespace Libs::Graphics::Timeline {

namespace Detail {
inline std::atomic<bool> g_active {false};
void MarkSlow(const char* event, uint64_t a, uint64_t b);
} // namespace Detail

[[nodiscard]] inline bool Active() {
	return Detail::g_active.load(std::memory_order_relaxed);
}

inline void Mark(const char* event, uint64_t a = 0, uint64_t b = 0) {
	if (Active()) {
		Detail::MarkSlow(event, a, b);
	}
}

// Marks `begin` now and `end` when the scope ends (both with `a`).
class Scope {
public:
	Scope(const char* begin, const char* end, uint64_t a): m_end(end), m_a(a), m_active(Active()) {
		if (m_active) {
			Detail::MarkSlow(begin, a, 0);
		}
	}
	~Scope() {
		if (m_active) {
			Detail::MarkSlow(m_end, m_a, 0);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	const char* m_end;
	uint64_t    m_a;
	bool        m_active;
};

// Called once per presented frame.
void Present(uint64_t frame);

} // namespace Libs::Graphics::Timeline

// Diagnostics: KYTY_CPU_COST=1 times every draw and dispatch on the GPU thread, by kind and the
// last program fetched for it (the pixel shader of a draw), and prints the heaviest every 5 s
// ([cpu-cost]). Off by default.
//
// v445m: everything is recorded into fixed tables owned by the recording thread (no mutex, no
// map, no allocation per record) and timed with the CPU's time-stamp counter, which the 5 s
// report converts with the steady clock over the same window. The report prints every phase
// ([cpu-phase], [cpu-self]: all rows) and a [cpu-total] line: the window's wall time, the thread's
// CPU time and the sum of the self times, so what no phase names is visible.
namespace Libs::Graphics::CpuCost {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CPU_COST");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
bool OnGpuThread();

namespace Detail {

[[nodiscard]] inline uint64_t Ticks() noexcept {
#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
	return __builtin_ia32_rdtsc();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
#endif
}

// One (phase name, kind) of one thread. The name is a string literal: the slot is found by its
// address (one literal can have several; the report merges them by text).
struct PhaseSlot {
	const char* name        = nullptr;
	char        kind        = 0;
	uint64_t    count       = 0;
	uint64_t    ticks       = 0; // inclusive
	uint64_t    self_ticks  = 0; // minus the phases nested in it
	uint64_t    frame_count = 0; // KYTY_HITCH_COST_MS: the guest frame in progress
	uint64_t    frame_self  = 0;
};
// A draw's or dispatch's time by kind and program ([cpu-cost]).
struct CostSlot {
	uint64_t key         = 0; // kind << 56 | hash; 0: free
	uint64_t count       = 0;
	uint64_t ticks       = 0;
	uint64_t frame_count = 0;
	uint64_t frame_ticks = 0;
};
constexpr size_t PhaseSlotCount = 4096; // powers of two
constexpr size_t CostSlotCount  = 16384;
struct ThreadTable {
	PhaseSlot phases[PhaseSlotCount];
	CostSlot  costs[CostSlotCount];
	PhaseSlot phase_overflow; // a full table: everything else lands here
	CostSlot  cost_overflow;
	uint32_t  phases_used = 0;
	uint32_t  costs_used  = 0;
	uint64_t  kind_count[128] {};
	uint64_t  kind_ticks[128] {};
	uint64_t  records      = 0; // phase and scope records in the window
	uint64_t  window_ticks = 0; // the window's start on both clocks
	std::chrono::steady_clock::time_point window_time {};
	uint64_t  window_cpu_ns[2] {}; // the thread's user and kernel time at the window's start
	uint64_t  frame_seen = 0;      // the guest frame count this thread last reported (hitch cost)
	bool      gpu_thread = false;
};
// v445o (KYTY_TIMERS_CUT, toggle mode "timers"): true while the cut is on and KYTY_CPU_COST is
// off: Phase's constructor then tests this one byte instead of Enabled()'s guarded static. Set at
// every guest flip (sync.cpp, PrepareVideoOutFlip); false until the first one. A phase is inactive
// either way: Enabled() is false whenever this is true.
inline std::atomic<bool> g_phase_off {false};
inline thread_local ThreadTable* t_table          = nullptr;
inline thread_local uint64_t     t_child_ticks    = 0; // of the phases nested in the open one
ThreadTable*                     CreateTable();      // timeline.cpp
void                             Report(ThreadTable& table);

[[nodiscard]] inline ThreadTable& Table() {
	auto* table = t_table;
	if (table == nullptr) [[unlikely]] {
		table = CreateTable();
	}
	return *table;
}

[[nodiscard]] inline PhaseSlot& FindPhase(ThreadTable& table, const char* name, char kind) noexcept {
	const auto address = reinterpret_cast<uintptr_t>(name);
	auto       index   = static_cast<size_t>(((static_cast<uint64_t>(address) >> 2u) ^ static_cast<uint8_t>(kind)) *
	                                         0x9e3779b97f4a7c15ull >> 40u) & (PhaseSlotCount - 1u);
	for (size_t probe = 0; probe < 64; probe++, index = (index + 1u) & (PhaseSlotCount - 1u)) {
		auto& slot = table.phases[index];
		if (slot.name == name && slot.kind == kind) {
			return slot;
		}
		if (slot.name == nullptr) {
			if (table.phases_used >= PhaseSlotCount / 2u) {
				break;
			}
			table.phases_used++;
			slot.name = name;
			slot.kind = kind;
			return slot;
		}
	}
	return table.phase_overflow;
}

inline void AddPhase(const char* name, char kind, uint64_t ticks, uint64_t self) noexcept {
	auto& table = Table();
	auto& slot  = FindPhase(table, name, kind);
	slot.count++;
	slot.ticks += ticks;
	slot.self_ticks += self;
	slot.frame_count++;
	slot.frame_self += self;
	table.records++;
}

} // namespace Detail

inline thread_local uint64_t t_hash = 0;
inline thread_local char     t_kind = 0;
// The program a draw or dispatch fetched (ProgramCache::Get): its cost and samples go to it.
inline void NoteProgram(uint64_t hash) {
	t_hash = hash;
	if (t_kind != 0) {
		Common::Sampler::g_gpu_tag.store((static_cast<uint64_t>(static_cast<uint8_t>(t_kind)) << 56u) |
		                                     (hash & ((uint64_t {1} << 56u) - 1u)),
		                                 std::memory_order_relaxed);
	}
}
// One draw or dispatch of `kind` that took `ticks` (Detail::Ticks); prints the tables every 5 s.
void Record(char kind, uint64_t hash, uint64_t ticks);

// The time of a draw or dispatch that no phase inside it names ([cpu-self] "<kind> (own)").
inline constexpr const char* ScopeSelfName = "(draw/dispatch own)";

class Scope {
public:
	explicit Scope(char kind): m_kind(kind), m_active(Enabled()) {
		if (m_active) {
			t_hash  = 0;
			t_kind  = kind;
			Common::Sampler::g_gpu_tag.store(static_cast<uint64_t>(static_cast<uint8_t>(kind)) << 56u,
			                                 std::memory_order_relaxed);
			m_saved_child_ticks   = Detail::t_child_ticks;
			Detail::t_child_ticks = 0;
			m_start               = Detail::Ticks();
		}
	}
	~Scope() {
		if (m_active) {
			const auto ticks = Detail::Ticks() - m_start;
			const auto child = Detail::t_child_ticks;
			t_kind = 0;
			Common::Sampler::g_gpu_tag.store(0, std::memory_order_relaxed);
			Detail::AddPhase(ScopeSelfName, m_kind, ticks, ticks > child ? ticks - child : 0);
			Detail::t_child_ticks = m_saved_child_ticks + ticks;
			Record(m_kind, t_hash, ticks);
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	char     m_kind;
	bool     m_active;
	uint64_t m_saved_child_ticks = 0;
	uint64_t m_start             = 0;
};

// KYTY_HITCH_COST_MS=<ms> (with KYTY_CPU_COST=1): at every guest flip, a frame (flip to flip)
// that took at least this long prints where the GPU thread spent that time: the phases' self time
// and the heaviest draws and dispatches of the frame ([hitch-cost], the first 400 such frames).
// Shorter frames print nothing. The 5 s tables stay as they are. (The flip runs on the presenter's
// thread: the GPU thread prints the frame at its next draw or dispatch.)
void FrameEnd(double frame_ms);

// KYTY_CPU_COST=1: GPU-thread time of named phases by the kind of draw or dispatch running ('.':
// outside any): inclusive ([cpu-phase]) and self time, the phase minus the phases nested in it
// ([cpu-self]). The inclusive table double counts nested phases; the self times sum to the time
// under the outermost phases (the GPU thread's loop: all of its time).
class Phase {
public:
	KYTY_CPU_COST_INLINE explicit Phase(const char* name)
	    : m_name(name),
	      m_active(!Detail::g_phase_off.load(std::memory_order_relaxed) && Enabled() && OnGpuThread()) {
		if (m_active) {
			m_saved_child_ticks   = Detail::t_child_ticks;
			Detail::t_child_ticks = 0;
			m_start               = Detail::Ticks();
		}
	}
	KYTY_CPU_COST_INLINE ~Phase() {
		if (m_active) {
			const auto ticks = Detail::Ticks() - m_start;
			const auto child = Detail::t_child_ticks;
			Detail::AddPhase(m_name, t_kind != 0 ? t_kind : '.', ticks, ticks > child ? ticks - child : 0);
			Detail::t_child_ticks = m_saved_child_ticks + ticks;
		}
	}
	Phase(const Phase&)            = delete;
	Phase& operator=(const Phase&) = delete;

private:
	const char* m_name;
	bool        m_active;
	uint64_t    m_saved_child_ticks = 0;
	uint64_t    m_start             = 0;
};

} // namespace Libs::Graphics::CpuCost

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_
