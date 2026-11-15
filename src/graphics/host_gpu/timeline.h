#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_

#include "common/sampler.h"

#include <atomic>
#include <chrono>
#include <cstdint>

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
namespace Libs::Graphics::CpuCost {

bool Enabled();
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
void Record(char kind, uint64_t hash, uint64_t ns);

class Scope {
public:
	explicit Scope(char kind): m_kind(kind), m_active(Enabled()) {
		if (m_active) {
			t_hash  = 0;
			t_kind  = kind;
			Common::Sampler::g_gpu_tag.store(static_cast<uint64_t>(static_cast<uint8_t>(kind)) << 56u,
			                                 std::memory_order_relaxed);
			m_start = std::chrono::steady_clock::now();
		}
	}
	~Scope() {
		if (m_active) {
			t_kind = 0;
			Common::Sampler::g_gpu_tag.store(0, std::memory_order_relaxed);
			Record(m_kind, t_hash,
			       static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                 std::chrono::steady_clock::now() - m_start)
			                                 .count()));
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	char                                  m_kind;
	bool                                  m_active;
	std::chrono::steady_clock::time_point m_start {};
};

// KYTY_HITCH_COST_MS=<ms> (with KYTY_CPU_COST=1): at every guest flip, a frame (flip to flip)
// that took at least this long prints where the GPU thread spent that time: the phases' self time
// and the heaviest draws and dispatches of the frame ([hitch-cost], the first 400 such frames).
// Shorter frames print nothing. The 5 s tables stay as they are.
void FrameEnd(double frame_ms);

// KYTY_CPU_COST=1: inclusive GPU-thread time of named phases ([cpu-phase], with [cpu-cost]),
// by the kind of draw or dispatch running ('.': outside any).
void RecordPhase(char kind, const char* name, uint64_t ns);
// Self time: the phase minus the phases nested in it ([cpu-self]); the inclusive table double
// counts nested phases, this one sums to the thread's time.
void RecordPhaseSelf(char kind, const char* name, uint64_t ns);
bool OnGpuThread();
inline thread_local uint64_t t_phase_child_ns = 0;

class Phase {
public:
	explicit Phase(const char* name): m_name(name), m_active(Enabled() && OnGpuThread()) {
		if (m_active) {
			m_saved_child_ns  = t_phase_child_ns;
			t_phase_child_ns  = 0;
			m_start = std::chrono::steady_clock::now();
		}
	}
	~Phase() {
		if (m_active) {
			const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                          std::chrono::steady_clock::now() - m_start)
			                                          .count());
			const char kind = t_kind != 0 ? t_kind : '.';
			RecordPhase(kind, m_name, ns);
			RecordPhaseSelf(kind, m_name, ns > t_phase_child_ns ? ns - t_phase_child_ns : 0);
			t_phase_child_ns = m_saved_child_ns + ns;
		}
	}
	Phase(const Phase&)            = delete;
	Phase& operator=(const Phase&) = delete;

private:
	const char*                           m_name;
	bool                                  m_active;
	uint64_t                              m_saved_child_ns = 0;
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace Libs::Graphics::CpuCost

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_TIMELINE_H_
