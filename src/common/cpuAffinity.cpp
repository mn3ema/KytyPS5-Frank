#include "common/cpuAffinity.h"

#include "common/common.h"
#include "common/logging/log.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#elif !defined(__APPLE__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <unistd.h>
#endif

namespace Common {

void RaiseGpuThreadPriority() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_GPU_THREAD_PRIORITY");
		return value != nullptr ? std::atoi(value) : 1;
	}();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const int priority = level >= 2   ? THREAD_PRIORITY_HIGHEST
	                     : level == 1 ? THREAD_PRIORITY_ABOVE_NORMAL
	                                  : THREAD_PRIORITY_NORMAL;
	SetThreadPriority(GetCurrentThread(), priority);
#else
	(void)level;
#endif
}

void LowerBackgroundThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

void OptOutOfPowerThrottling() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (const char* value = std::getenv("KYTY_POWER_THROTTLING"); value != nullptr && value[0] == '1') {
		std::printf("CPU: power throttling left to the system (KYTY_POWER_THROTTLING=1)\n");
		return;
	}
	// Values from the Windows SDK (processthreadsapi.h), defined here for older SDKs.
	constexpr ULONG Version         = 1; // PROCESS_POWER_THROTTLING_CURRENT_VERSION
	constexpr ULONG ExecutionSpeed  = 0x1; // PROCESS_POWER_THROTTLING_EXECUTION_SPEED
	constexpr ULONG IgnoreTimerRes  = 0x4; // PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
	struct {
		ULONG Version;
		ULONG ControlMask;
		ULONG StateMask;
	} state {Version, ExecutionSpeed | IgnoreTimerRes, 0};
	// ControlMask set with StateMask clear: never throttle, always honour timer resolution.
	bool ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
	                                sizeof(state)) != 0;
	if (!ok) {
		// Before Windows 11 the timer-resolution bit is unknown; keep the execution-speed part.
		state.ControlMask = ExecutionSpeed;
		ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
		                           sizeof(state)) != 0;
	}
	std::printf("CPU: power throttling (EcoQoS) %s for the process\n", ok ? "off" : "opt-out failed");
#endif
}

static bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PCORE_AFFINITY");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

void PreferPerformanceCores(const char* name) {
	if (!Enabled()) {
		return;
	}
	ULONG length = 0;
	GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) {
		return;
	}
	std::vector<uint8_t> buffer(length);
	auto* first = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data());
	if (!GetSystemCpuSetInformation(first, length, &length, GetCurrentProcess(), 0)) {
		return;
	}
	struct Set {
		ULONG id;
		BYTE  efficiency;
	};
	std::vector<Set> sets;
	BYTE             lowest = 0xff, highest = 0;
	for (ULONG offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data() + offset);
		if (info->Size == 0) {
			break;
		}
		if (info->Type == CpuSetInformation) {
			sets.push_back({info->CpuSet.Id, info->CpuSet.EfficiencyClass});
			lowest  = std::min(lowest, info->CpuSet.EfficiencyClass);
			highest = std::max(highest, info->CpuSet.EfficiencyClass);
		}
		offset += info->Size;
	}
	if (sets.empty() || lowest == highest) {
		return; // All cores alike.
	}
	std::vector<ULONG> fast;
	for (const auto& set: sets) {
		if (set.efficiency == highest) {
			fast.push_back(set.id);
		}
	}
	const bool ok =
	    SetThreadSelectedCpuSets(GetCurrentThread(), fast.data(), static_cast<ULONG>(fast.size())) != 0;
	LOGF("CPU: %s prefers the %zu fastest of %zu logical processors (efficiency class %u)%s\n", name,
	     fast.size(), sets.size(), static_cast<unsigned>(highest), ok ? "" : " - failed");
}

#elif !defined(__APPLE__)

void PreferPerformanceCores(const char* name) {
	if (!Enabled()) {
		return;
	}
	const long count = sysconf(_SC_NPROCESSORS_CONF);
	if (count <= 0) {
		return;
	}
	std::vector<unsigned long> max_khz(static_cast<size_t>(count), 0);
	unsigned long              lowest = ~0ul, highest = 0;
	for (long cpu = 0; cpu < count; cpu++) {
		std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
		                   "/cpufreq/cpuinfo_max_freq");
		unsigned long khz = 0;
		if (!(file >> khz) || khz == 0) {
			return; // Unknown: leave the scheduler alone.
		}
		max_khz[static_cast<size_t>(cpu)] = khz;
		lowest                            = std::min(lowest, khz);
		highest                           = std::max(highest, khz);
	}
	// Hybrid only: efficiency cores top out well below the performance cores (Raptor Lake 4.1
	// against 5.2 GHz), while one CPU's preferred cores differ by a few percent at most.
	if (lowest * 100ul >= highest * 85ul) {
		return;
	}
	cpu_set_t set;
	CPU_ZERO(&set);
	size_t fast = 0;
	for (long cpu = 0; cpu < count; cpu++) {
		if (max_khz[static_cast<size_t>(cpu)] * 100ul >= highest * 85ul) {
			CPU_SET(static_cast<int>(cpu), &set);
			fast++;
		}
	}
	const bool ok = pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
	LOGF("CPU: %s runs on the %zu fastest of %ld logical processors (up to %lu MHz)%s\n", name, fast,
	     count, highest / 1000ul, ok ? "" : " - failed");
}

#else

void PreferPerformanceCores(const char* /*name*/) {}

#endif

} // namespace Common
