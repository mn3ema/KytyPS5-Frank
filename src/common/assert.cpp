#include "common/assert.h"
#include "common/sampler.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
// windows.h first
#include <dbghelp.h>
#endif

namespace Common {

#ifdef _WIN32
// dbghelp is not thread-safe: two threads symbolising at once crashed inside SymFromAddr.
static std::mutex g_symbol_mutex;

// Call with g_symbol_mutex held. Symbols come from the PDB next to the executable or one
// directory up (the install layout keeps the PDB in the build directory).
static bool SymbolsReady() {
	static const bool initialized = [] {
		char exe_path[MAX_PATH] {};
		GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
		std::string search = exe_path;
		if (const auto slash = search.find_last_of("\\/"); slash != std::string::npos) {
			search.resize(slash);
			std::string parent = search;
			if (const auto up = parent.find_last_of("\\/"); up != std::string::npos) {
				parent.resize(up);
				search += ";" + parent;
			}
		}
		SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
		return SymInitialize(GetCurrentProcess(), search.c_str(), TRUE) != FALSE;
	}();
	return initialized;
}

// Call with g_symbol_mutex held: " name+0xoff [file:line]" for one code address.
static std::string NameAddress(DWORD64 address) {
	std::string out;
	if (!SymbolsReady()) {
		return out;
	}
	const auto process = GetCurrentProcess();
	alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512] {};
	auto*                     symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
	symbol->SizeOfStruct             = sizeof(SYMBOL_INFO);
	symbol->MaxNameLen               = 512;
	DWORD64 displacement             = 0;
	if (SymFromAddr(process, address, &displacement, symbol) != FALSE) {
		out += fmt::format(" {}+0x{:x}", symbol->Name, displacement);
	}
	IMAGEHLP_LINE64 line {};
	line.SizeOfStruct       = sizeof(line);
	DWORD line_displacement = 0;
	if (SymGetLineFromAddr64(process, address, &line_displacement, &line) != FALSE) {
		out += fmt::format(" [{}:{}]", line.FileName, line.LineNumber);
	}
	// The module, for frames outside this executable (a driver or layer DLL names itself only by
	// its nearest export otherwise).
	HMODULE module = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(address), &module) != FALSE &&
	    module != nullptr && module != GetModuleHandleA(nullptr)) {
		char path[MAX_PATH] {};
		if (GetModuleFileNameA(module, path, sizeof(path)) != 0) {
			const char* name = std::strrchr(path, '\\');
			out += fmt::format(" ({}+0x{:x})", name != nullptr ? name + 1 : path,
			                   address - reinterpret_cast<DWORD64>(module));
		}
	}
	return out;
}
#endif

// The host call stack of the failing thread, so a fatal report names its caller instead of
// only the file that noticed.
std::string HostBacktrace() {
#ifdef _WIN32
	static constexpr ULONG MaxFrames = 48;
	void*                  frames[MaxFrames];
	const auto             count = CaptureStackBackTrace(1, MaxFrames, frames, nullptr);
	if (count == 0) {
		return {};
	}
	std::scoped_lock symbol_lock(g_symbol_mutex);
	std::string      out = "--- Host stack ---\n";
	for (ULONG i = 0; i < count; i++) {
		const auto address = reinterpret_cast<DWORD64>(frames[i]);
		out += fmt::format("  #{:<2} 0x{:016x}", i, address) + NameAddress(address) + "\n";
	}
	return out;
#else
	return {};
#endif
}

std::string HostAddressName(const void* address) {
#ifdef _WIN32
	std::scoped_lock symbol_lock(g_symbol_mutex);
	return NameAddress(reinterpret_cast<DWORD64>(address));
#else
	(void)address;
	return {};
#endif
}

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n{}", KYTY_BUILD_LABEL, title, text, file,
	                   line, HostBacktrace());
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
#define KYTY_RECOVERABLE_EXITS 1
#else
#define KYTY_RECOVERABLE_EXITS 0
#endif

static thread_local int g_recoverable_depth = 0;

RecoverableExitScope::RecoverableExitScope() {
	g_recoverable_depth++;
}

RecoverableExitScope::~RecoverableExitScope() {
	g_recoverable_depth--;
}

bool RecoverableExitsSupported() {
	return KYTY_RECOVERABLE_EXITS != 0;
}

static void ThrowIfRecoverable([[maybe_unused]] std::string_view text, [[maybe_unused]] const char* file,
                               [[maybe_unused]] int line) {
#if KYTY_RECOVERABLE_EXITS
	if (g_recoverable_depth > 0) {
		throw RecoverableExit {fmt::format("{} ({}:{})", text, file, line)};
	}
#endif
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	ThrowIfRecoverable(expr, file, line);
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	ThrowIfRecoverable(expr, file, line);
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	ThrowIfRecoverable(text, file, line);
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	ThrowIfRecoverable(text, file, line);
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
