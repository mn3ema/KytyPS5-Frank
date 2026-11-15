// Diagnosis (v437x, Fable 11-14): a breakpoint exception (0x80000003) ended the process during the
// compute warm-up with no message. This logs where it was raised (module and offset) before the
// process goes down; it handles nothing.
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <atomic>
#include <cstdio>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

#ifdef _WIN32
namespace {
LONG CALLBACK BreakpointLogger(EXCEPTION_POINTERS* info) {
	if (info == nullptr || info->ExceptionRecord == nullptr) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	const auto code = info->ExceptionRecord->ExceptionCode;
	if (code != EXCEPTION_BREAKPOINT && code != STATUS_STACK_BUFFER_OVERRUN && code != EXCEPTION_ILLEGAL_INSTRUCTION) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	static std::atomic<int> logged {0};
	if (logged.fetch_add(1) >= 4) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	void*   address = info->ExceptionRecord->ExceptionAddress;
	HMODULE module  = nullptr;
	char    name[MAX_PATH] = "?";
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       static_cast<LPCSTR>(address), &module) &&
	    module != nullptr) {
		GetModuleFileNameA(module, name, sizeof(name));
	}
	std::fprintf(stderr, "[crash] exception 0x%08lx at %p in %s +0x%llx (thread %lu)\n", static_cast<unsigned long>(code),
	             address, name,
	             static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(module)),
	             static_cast<unsigned long>(GetCurrentThreadId()));
	std::fflush(stderr);
	return EXCEPTION_CONTINUE_SEARCH;
}
} // namespace
#endif

void InstallWarmCrashLogger() {
#ifdef _WIN32
	static std::atomic<bool> installed {false};
	if (!installed.exchange(true)) {
		AddVectoredExceptionHandler(0, BreakpointLogger); // last: trace points handle their own breakpoints first
	}
#endif
}

} // namespace Libs::Graphics
