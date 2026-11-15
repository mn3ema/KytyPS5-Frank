#ifndef EMULATOR_SRC_COMMON_CPUAFFINITY_H_
#define EMULATOR_SRC_COMMON_CPUAFFINITY_H_

namespace Common {

// On a CPU with cores of different speeds (Intel 12th-14th gen and later hybrid designs), keeps
// the calling thread on the fastest cores: the emulator's GPU thread is its bottleneck, and on an
// efficiency core it ran ~1.7x slower per operation (i7-13700F on Linux, 2026-10-06). Windows:
// CPU sets of the highest efficiency class (a scheduler preference). Linux: the cores with the
// highest maximum frequency. A no-op on CPUs whose cores are all alike, on macOS, and with
// KYTY_PCORE_AFFINITY=0. `name` is for the log line.
void PreferPerformanceCores(const char* name);

// KYTY_GPU_THREAD_PRIORITY (default 1): scheduling priority of the calling thread, for the
// threads that feed the host GPU (0 normal, 1 above normal, 2 highest; Windows only). Ported from
// idxtri/wolverine-perf 861aa3d1: ~40 guest threads compete with them at normal priority.
void RaiseGpuThreadPriority();

// Below-normal scheduling priority for the calling thread (background work such as the startup
// shader precompile; Windows only).
void LowerBackgroundThreadPriority();

// Windows 11 power throttling (EcoQoS) lowers the clocks of, and on hybrid CPUs moves to efficiency
// cores, threads the system judges to be background work, and stops honouring timer-resolution
// requests of processes whose window is not visible; on Arrow Lake (Core Ultra 200) under 24H2 this
// was reported to halve throughput. The emulator opts its whole process out at start (execution
// speed and timer resolution), so no guest or host thread is throttled. KYTY_POWER_THROTTLING=1
// leaves the system's default (for an A/B). Windows only; logs one line.
void OptOutOfPowerThrottling();

} // namespace Common

#endif // EMULATOR_SRC_COMMON_CPUAFFINITY_H_
