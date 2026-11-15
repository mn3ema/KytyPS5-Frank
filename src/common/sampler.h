#ifndef KYTY_COMMON_SAMPLER_H_
#define KYTY_COMMON_SAMPLER_H_

#include <atomic>
#include <cstdint>
#include <string>

namespace Common {
// " function+0xoffset [file:line]" of one host code address (empty without symbols); assert.cpp.
// Declared here rather than in assert.h: assert.h is part of the shader cache version.
std::string HostAddressName(const void* address);
} // namespace Common

namespace Common::Sampler {

// KYTY_SAMPLE=<start seconds>:<seconds> (Windows): a sampling profiler for the threads that
// register themselves. From <start> seconds after the emulator started, for <seconds>, every
// millisecond each registered thread is suspended, its instruction pointer read, and resumed;
// then the hottest functions of each thread are printed ([sample]). Without KYTY_SAMPLE,
// registering does nothing.
void RegisterCurrentThread(const char* name);
// A guest thread (KYTY_SAMPLE_GUEST=1), with its guest stack [stack_low, stack_high): samples
// taken while it is inside the emulator (an HLE call, a wait) are unwound within that stack, so
// [sample-guest] can say what each busy guest thread waits in.
void RegisterGuestThread(const char* name, uintptr_t stack_low, uintptr_t stack_high);
// Only records the name for [freeze]/[hitch] reports (guest threads: "g:<name>").
void NameCurrentThread(const char* name);

// A tag the GPU thread sets for the work it is doing (CpuCost: kind and shader); samples of the
// thread named "gpu" carry it, and the report breaks the heaviest tags down ([sample-tag]).
inline std::atomic<uint64_t> g_gpu_tag {0};

// Freeze recorder (Windows; KYTY_FREEZE_DUMP=0 turns it off): every guest flip calls
// NoteGuestFrame. When no flip came for 15 s after the game had been flipping, the threads of
// the process that are busy (and those stopped inside the emulator's own code) are suspended one
// at a time and their instruction pointer and return addresses are printed by name ([freeze]
// lines on stderr), three times a second apart; again after 60 s, at most three reports.
void NoteGuestFrame();

// The calling thread is the sampler unwinding a suspended thread's stack: the host exception
// handler leaves faults on it to the unwinder's own handler.
[[nodiscard]] bool InUnwind() noexcept;

} // namespace Common::Sampler

#endif /* KYTY_COMMON_SAMPLER_H_ */
