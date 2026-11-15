#include "common/tracePoints.h"
#include "common/sampler.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <memoryapi.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <processthreadsapi.h>
// IWYU pragma: no_include <winnt.h>

namespace Common::TracePoints {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr const char* kRegisterNames[] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                                          "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};

struct Point {
	uint64_t             address   = 0;
	uint64_t             rva       = 0;
	uint8_t              original  = 0;
	int                  reg       = -1;
	int64_t              offset    = 0;
	uint32_t             dwords    = 0;
	uint32_t             max_hits  = 200;
	bool                 deref     = false;
	int64_t              deref_off = 0;
	int                  cond_reg  = -1;    // "?<reg>=<value>" or "?<reg>:<off>=<dword>"
	bool                 cond_mem  = false;
	int64_t              cond_off  = 0;
	uint64_t             cond_val  = 0;
	bool                 arm       = false; // "!<off>": first hit arms the hardware watch on <reg>+<off>
	int64_t              arm_off   = 0;
	std::atomic_uint32_t hits {0};
};

// Fixed after Install(); the handler only reads it.
std::vector<Point*> g_points;
thread_local Point* t_pending = nullptr;

// KYTY_WATCH_WRITE=<address>[:<length 1/2/4/8>[:<max hits>]] (hex): a hardware write watchpoint
// (DR0) on every thread of the process, re-applied every 250 ms so new threads get it too.
std::atomic<uint64_t> g_watch_address {0};
uint32_t             g_watch_length  = 4;
uint32_t             g_watch_max     = 64;
std::atomic_uint32_t g_watch_hits {0};
uint32_t             g_watch_delay_ms = 0; // 4th field: seconds before the watch is armed
// KYTY_WATCH_VAL=<hex dword>: only writes that leave this value at the watched address count.
bool                 g_watch_filter   = false;
uint32_t             g_watch_value    = 0;
// KYTY_WATCH_RW=1: trigger on reads as well as writes (DR7 RW = 11).
uint64_t             g_watch_rw_bits  = 1;
uint64_t             g_module_base = 0;
uint64_t             g_module_size = 0;

uint64_t RegisterValue(const CONTEXT* context, int reg) {
	switch (reg) {
		case 0: return context->Rax;
		case 1: return context->Rbx;
		case 2: return context->Rcx;
		case 3: return context->Rdx;
		case 4: return context->Rsi;
		case 5: return context->Rdi;
		case 6: return context->Rbp;
		case 7: return context->Rsp;
		case 8: return context->R8;
		case 9: return context->R9;
		case 10: return context->R10;
		case 11: return context->R11;
		case 12: return context->R12;
		case 13: return context->R13;
		case 14: return context->R14;
		case 15: return context->R15;
		default: return 0;
	}
}

bool Readable(uint64_t address, size_t size) {
	if (address == 0 || address + size < address) {
		return false;
	}
	for (uint64_t at = address & ~0xfffull; at < address + size; at += 0x1000) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) == 0 ||
		    info.State != MEM_COMMIT || (info.Protect & PAGE_GUARD) != 0 ||
		    info.Protect == PAGE_NOACCESS || info.Protect == PAGE_EXECUTE) {
			return false;
		}
	}
	return true;
}

float LowFloat(const M128A& xmm) {
	float    value = 0.0f;
	uint32_t bits  = static_cast<uint32_t>(xmm.Low);
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

void Append(char* line, size_t capacity, size_t* at, const char* format, ...) {
	if (*at >= capacity) {
		return;
	}
	va_list args;
	va_start(args, format);
	const auto written = std::vsnprintf(line + *at, capacity - *at, format, args);
	va_end(args);
	if (written > 0) {
		*at += static_cast<size_t>(written);
	}
}

void Log(const Point* point, const CONTEXT* context, uint32_t hit) {
	static thread_local char line[16384];
	size_t at = 0;
	Append(line, sizeof(line), &at, "[trace] e+%llx hit=%u tid=%lu t=%llu",
	       static_cast<unsigned long long>(point->rva), hit,
	       static_cast<unsigned long>(GetCurrentThreadId()),
	       static_cast<unsigned long long>(GetTickCount64()));
	for (int reg = 0; reg < 16; reg++) {
		Append(line, sizeof(line), &at, " %s=%llx", kRegisterNames[reg],
		       static_cast<unsigned long long>(RegisterValue(context, reg)));
	}
	Append(line, sizeof(line), &at, " xmm0=%g xmm1=%g xmm2=%g xmm3=%g",
	       static_cast<double>(LowFloat(context->Xmm0)), static_cast<double>(LowFloat(context->Xmm1)),
	       static_cast<double>(LowFloat(context->Xmm2)), static_cast<double>(LowFloat(context->Xmm3)));
	if (point->reg >= 0 && point->dwords > 0) {
		auto address = RegisterValue(context, point->reg) + static_cast<uint64_t>(point->offset);
		if (point->deref) {
			if (Readable(address, 8)) {
				uint64_t pointer = 0;
				std::memcpy(&pointer, reinterpret_cast<const void*>(address), sizeof(pointer));
				address = pointer + static_cast<uint64_t>(point->deref_off);
			} else {
				address = 0;
			}
		}
		Append(line, sizeof(line), &at, " | @%llx:", static_cast<unsigned long long>(address));
		if (Readable(address, static_cast<size_t>(point->dwords) * 4)) {
			for (uint32_t index = 0; index < point->dwords; index++) {
				uint32_t value = 0;
				std::memcpy(&value, reinterpret_cast<const void*>(address + index * 4), sizeof(value));
				Append(line, sizeof(line), &at, " %08x", value);
			}
		} else {
			Append(line, sizeof(line), &at, " (unreadable)");
		}
	}
	Append(line, sizeof(line), &at, "\n");
	std::fwrite(line, 1, at < sizeof(line) ? at : sizeof(line) - 1, stderr);
	std::fflush(stderr);
}

bool ConditionHolds(const Point* point, const CONTEXT* context) {
	if (point->cond_reg < 0) {
		return true;
	}
	const auto value = RegisterValue(context, point->cond_reg);
	if (!point->cond_mem) {
		return value == point->cond_val;
	}
	const auto address = value + static_cast<uint64_t>(point->cond_off);
	if (!Readable(address, 4)) {
		return false;
	}
	uint32_t dword = 0;
	std::memcpy(&dword, reinterpret_cast<const void*>(address), sizeof(dword));
	return dword == static_cast<uint32_t>(point->cond_val);
}

void WriteByte(uint64_t address, uint8_t value) {
	MEMORY_BASIC_INFORMATION info {};
	DWORD                    old_protect = 0;
	auto*                    pointer     = reinterpret_cast<uint8_t*>(address);
	const bool               writable =
	    VirtualQuery(pointer, &info, sizeof(info)) != 0 &&
	    (info.Protect == PAGE_EXECUTE_READWRITE || info.Protect == PAGE_READWRITE ||
	     info.Protect == PAGE_EXECUTE_WRITECOPY || info.Protect == PAGE_WRITECOPY);
	if (!writable) {
		VirtualProtect(pointer, 1, PAGE_EXECUTE_READWRITE, &old_protect);
	}
	*static_cast<volatile uint8_t*>(pointer) = value;
	FlushInstructionCache(GetCurrentProcess(), pointer, 1);
}

LONG WINAPI Filter(PEXCEPTION_POINTERS exception) noexcept {
	const auto code    = exception->ExceptionRecord->ExceptionCode;
	auto*      context = exception->ContextRecord;

	if (code == EXCEPTION_SINGLE_STEP && g_watch_address.load() != 0 && (context->Dr6 & 0xfu) != 0) {
		context->Dr6 = 0;
		// Only writes by the main executable's code count (libc memsets and copies would use the
		// budget up); the caller of a memcpy shows up through the stack dump of the next point.
		if (g_watch_filter) {
			uint32_t now = 0;
			std::memcpy(&now, reinterpret_cast<const void*>(g_watch_address.load()), sizeof(now));
			if (now != g_watch_value) {
				return t_pending == nullptr ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
			}
		}
		const bool in_module = context->Rip >= g_module_base && context->Rip < g_module_base + g_module_size;
		if (!in_module) {
			// e.g. a memcpy in libc: log the stack (64 dwords at rsp) so the caller can be found
			static std::atomic_uint32_t outside {0};
			if (++outside <= 40) {
				static Point stack_point;
				stack_point.rva    = context->Rip;
				stack_point.reg    = 7; // rsp
				stack_point.offset = 0;
				stack_point.dwords = 64;
				std::fprintf(stderr, "[watch] write from outside the main module\n");
				Log(&stack_point, context, outside.load());
				// The emulator's writer by name, and the named return addresses on its stack (its
				// callers), for a write the emulator itself makes.
				std::string named = Common::HostAddressName(reinterpret_cast<const void*>(context->Rip));
				int         callers = 0;
				for (uint64_t at = context->Rsp; at < context->Rsp + 1024 && callers < 8; at += 8) {
					if (!Readable(at, 8)) break;
					uint64_t value = 0;
					std::memcpy(&value, reinterpret_cast<const void*>(at), sizeof(value));
					HMODULE owner = nullptr;
					if (value < 0x10000 ||
					    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
					                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					                       reinterpret_cast<LPCSTR>(value), &owner) == FALSE ||
					    owner != GetModuleHandleA(nullptr)) {
						continue;
					}
					auto name = Common::HostAddressName(reinterpret_cast<const void*>(value - 1));
					if (name.empty() || name == " ?") continue;
					if (const auto cut = name.find(" ["); cut != std::string::npos) name.resize(cut);
					named += " <" + name;
					callers++;
				}
				std::fprintf(stderr, "[watch] writer:%s\n", named.c_str());
			}
			return t_pending == nullptr ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
		}
		const auto hit = ++g_watch_hits;
		if (hit <= g_watch_max) {
			static Point watch_point;
			watch_point.rva     = context->Rip - g_module_base; // the instruction after the access
			watch_point.reg     = 7; // rsp: the return addresses show the caller
			watch_point.dwords  = 16;
			char head[96];
			const uint64_t watched = g_watch_address.load();
			std::snprintf(head, sizeof(head), "[watch] access at %llx (rip after) value now:",
			              static_cast<unsigned long long>(watched));
			uint64_t value = 0;
			std::memcpy(&value, reinterpret_cast<const void*>(watched), 8);
			std::fprintf(stderr, "%s %016llx\n", head, static_cast<unsigned long long>(value));
			Log(&watch_point, context, hit);
		}
		if (t_pending == nullptr) {
			return EXCEPTION_CONTINUE_EXECUTION;
		}
	}

	if (code == EXCEPTION_BREAKPOINT) {
		const auto address = reinterpret_cast<uint64_t>(exception->ExceptionRecord->ExceptionAddress);
		for (auto* point: g_points) {
			if (point->address != address) {
				continue;
			}
			if (ConditionHolds(point, context)) {
				const auto hit = ++point->hits;
				if (hit <= point->max_hits) {
					Log(point, context, hit);
				}
				if (point->arm && point->reg >= 0 && g_watch_address.load() == 0) {
					const uint64_t target =
					    (RegisterValue(context, point->reg) + static_cast<uint64_t>(point->arm_off)) &
					    ~static_cast<uint64_t>(g_watch_length - 1);
					uint64_t expected = 0;
					if (g_watch_address.compare_exchange_strong(expected, target)) {
						std::fprintf(stderr, "[watch] armed by e+%llx on %llx\n",
						             static_cast<unsigned long long>(point->rva),
						             static_cast<unsigned long long>(target));
					}
				}
			}
			WriteByte(address, point->original);
			context->Rip = address;
			context->EFlags |= 0x100u; // trap flag: re-arm after the original instruction
			t_pending = point;
			return EXCEPTION_CONTINUE_EXECUTION;
		}
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (code == EXCEPTION_SINGLE_STEP && t_pending != nullptr) {
		auto* point = t_pending;
		t_pending   = nullptr;
		if (point->hits.load() < point->max_hits) {
			WriteByte(point->address, 0xcc);
		}
		context->EFlags &= ~0x100u;
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	return EXCEPTION_CONTINUE_SEARCH;
}

std::vector<std::string> Split(const std::string& text, char separator) {
	std::vector<std::string> parts;
	size_t                   start = 0;
	for (;;) {
		const auto end = text.find(separator, start);
		parts.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
		if (end == std::string::npos) {
			return parts;
		}
		start = end + 1;
	}
}

int64_t ParseSigned(const std::string& text) {
	if (!text.empty() && text[0] == '-') {
		return -static_cast<int64_t>(std::stoull(text.substr(1), nullptr, 16));
	}
	return static_cast<int64_t>(std::stoull(text, nullptr, 16));
}


DWORD WINAPI WatchThread(LPVOID /*parameter*/) {
	const DWORD self    = GetCurrentThreadId();
	const DWORD process = GetCurrentProcessId();
	uint64_t    len_bits = g_watch_length == 1 ? 0 : g_watch_length == 2 ? 1 : g_watch_length == 8 ? 2 : 3;
	Sleep(g_watch_delay_ms);
	std::fprintf(stderr, "[watch] armed\n");
	for (;;) {
		if (g_watch_hits.load() >= g_watch_max) {
			return 0;
		}
		HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (snapshot != INVALID_HANDLE_VALUE) {
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) {
					continue;
				}
				HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
				                           FALSE, entry.th32ThreadID);
				if (thread == nullptr) {
					continue;
				}
				if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
					CONTEXT context {};
					context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
					const uint64_t watched = g_watch_address.load();
					if (watched != 0 && GetThreadContext(thread, &context) && context.Dr0 != watched) {
						context.Dr0 = watched;
						// L0 enable, RW0 = 01 (write), LEN0
						context.Dr7 = (context.Dr7 & ~0xf0003ull) | 1ull | (g_watch_rw_bits << 16) | (len_bits << 18);
						SetThreadContext(thread, &context);
					}
					ResumeThread(thread);
				}
				CloseHandle(thread);
			}
			CloseHandle(snapshot);
		}
		Sleep(250);
	}
}

std::atomic_bool g_watch_thread {false};

void StartWatchThread() {
	if (!g_watch_thread.exchange(true)) {
		CreateThread(nullptr, 0, WatchThread, nullptr, 0, nullptr);
	}
}

void ReadWatchFilter() {
	if (const char* rw = std::getenv("KYTY_WATCH_RW"); rw != nullptr && rw[0] == '1') {
		g_watch_rw_bits = 3;
		std::fprintf(stderr, "[watch] reads trigger too\n");
	}
	if (const char* value = std::getenv("KYTY_WATCH_VAL"); value != nullptr && *value != '\0') {
		g_watch_filter = true;
		g_watch_value  = static_cast<uint32_t>(std::stoul(value, nullptr, 16));
		std::fprintf(stderr, "[watch] only writes leaving %08x count\n", g_watch_value);
	}
}

void InstallWatch() {
	const char* spec = std::getenv("KYTY_WATCH_WRITE");
	if (spec == nullptr || *spec == '\0') {
		return;
	}
	const auto fields = Split(spec, ':');
	uint64_t   address = std::stoull(fields[0], nullptr, 16);
	if (fields.size() >= 2) {
		g_watch_length = static_cast<uint32_t>(std::stoul(fields[1], nullptr, 16));
	}
	if (fields.size() >= 3) {
		g_watch_max = static_cast<uint32_t>(std::stoul(fields[2], nullptr, 16));
	}
	if (fields.size() >= 4) {
		g_watch_delay_ms = static_cast<uint32_t>(std::stoul(fields[3], nullptr, 10)) * 1000u;
	}
	address &= ~static_cast<uint64_t>(g_watch_length - 1);
	g_watch_address = address;
	ReadWatchFilter();
	StartWatchThread();
	std::fprintf(stderr, "[watch] write watch on %llx len %u, max %u hits\n",
	             static_cast<unsigned long long>(address), g_watch_length, g_watch_max);
	std::fflush(stderr);
}

uint32_t g_trace_delay_ms = 0;

DWORD WINAPI ArmLater(LPVOID /*parameter*/) {
	Sleep(g_trace_delay_ms);
	for (auto* point: g_points) {
		WriteByte(point->address, 0xcc);
	}
	std::fprintf(stderr, "[trace] armed %zu points (delayed)\n", g_points.size());
	std::fflush(stderr);
	return 0;
}

} // namespace

void Install(uint64_t module_base, uint64_t module_size, const char* module_name) {
	static bool handler_installed = false;
	g_module_base = module_base;
	g_module_size = module_size;
	if ((std::getenv("KYTY_WATCH_WRITE") != nullptr || std::getenv("KYTY_WATCH_DRAW") != nullptr) &&
	    !handler_installed) {
		if (AddVectoredExceptionHandler(1, Filter) != nullptr) {
			handler_installed = true;
			InstallWatch();
		}
	}
	const char* spec = std::getenv("KYTY_TRACE_PC");
	if (spec == nullptr || *spec == '\0' || !g_points.empty()) {
		return;
	}

	std::vector<Point*> points;
	std::string         text(spec);
	for (auto& c: text) {
		if (c == '/') {
			c = ','; // alternative separator for launchers that split on commas
		}
	}
	for (const auto& entry: Split(text, ',')) {
		if (entry.empty()) {
			continue;
		}
		auto*      point  = new Point();
		std::string body  = entry;
		if (const auto bang = body.find('!'); bang != std::string::npos) {
			point->arm     = true;
			point->arm_off = ParseSigned(body.substr(bang + 1));
			body           = body.substr(0, bang);
		}
		const auto  q     = body.find('?');
		if (q != std::string::npos) {
			const auto cond     = body.substr(q + 1);
			body                = body.substr(0, q);
			const auto eq       = cond.find_first_of("=~"); // "~" for launchers that split on "="
			const auto lhs      = Split(cond.substr(0, eq), ':');
			for (int reg = 0; reg < 16; reg++) {
				if (lhs[0] == kRegisterNames[reg]) {
					point->cond_reg = reg;
				}
			}
			if (lhs.size() >= 2) {
				point->cond_mem = true;
				point->cond_off = ParseSigned(lhs[1]);
			}
			point->cond_val = std::stoull(cond.substr(eq + 1), nullptr, 16);
		}
		const auto fields = Split(body, ':');
		point->rva        = std::stoull(fields[0], nullptr, 16);
		point->address    = module_base + point->rva;
		if (fields.size() >= 4) {
			for (int reg = 0; reg < 16; reg++) {
				if (fields[1] == kRegisterNames[reg]) {
					point->reg = reg;
				}
			}
			point->offset = ParseSigned(fields[2]);
			point->dwords = static_cast<uint32_t>(std::stoul(fields[3], nullptr, 16));
			if (point->dwords > 1024) {
				point->dwords = 1024;
			}
		}
		if (fields.size() >= 5 && !fields[4].empty()) {
			point->max_hits = static_cast<uint32_t>(std::stoul(fields[4], nullptr, 16));
		}
		if (fields.size() >= 6) {
			point->deref     = true;
			point->deref_off = ParseSigned(fields[5]);
		}
		if (point->rva >= module_size) {
			std::fprintf(stderr, "[trace] e+%llx is outside %s (size %llx), skipped\n",
			             static_cast<unsigned long long>(point->rva), module_name,
			             static_cast<unsigned long long>(module_size));
			delete point;
			continue;
		}
		point->original = *reinterpret_cast<const uint8_t*>(point->address);
		points.push_back(point);
	}
	if (points.empty()) {
		return;
	}

	g_points = points;
	bool any_arm = false;
	for (auto* point: g_points) {
		any_arm = any_arm || point->arm;
	}
	if (!handler_installed && AddVectoredExceptionHandler(1, Filter) == nullptr) {
		std::fprintf(stderr, "[trace] AddVectoredExceptionHandler failed\n");
		g_points.clear();
		return;
	}
	if (any_arm) {
		if (const char* len = std::getenv("KYTY_WATCH_LEN"); len != nullptr && *len != '\0') {
			g_watch_length = static_cast<uint32_t>(std::strtoul(len, nullptr, 16));
		}
		if (const char* max = std::getenv("KYTY_WATCH_MAX"); max != nullptr && *max != '\0') {
			g_watch_max = static_cast<uint32_t>(std::strtoul(max, nullptr, 16));
		}
		ReadWatchFilter();
		StartWatchThread();
	}
	// KYTY_TRACE_DELAY=<seconds>: arm the points later (skips boot and the title screen, where heavy
	// points would slow the game enough to break scripted input).
	if (const char* delay = std::getenv("KYTY_TRACE_DELAY"); delay != nullptr && *delay != '\0') {
		g_trace_delay_ms = static_cast<uint32_t>(std::strtoul(delay, nullptr, 10)) * 1000u;
		CreateThread(nullptr, 0, ArmLater, nullptr, 0, nullptr);
		std::fprintf(stderr, "[trace] %zu points will be armed after %u s\n", g_points.size(), g_trace_delay_ms / 1000u);
		std::fflush(stderr);
		return;
	}
	for (auto* point: g_points) {
		WriteByte(point->address, 0xcc);
		std::fprintf(stderr, "[trace] armed e+%llx (%llx) in %s: byte %02x, max %u hits%s%s\n",
		             static_cast<unsigned long long>(point->rva),
		             static_cast<unsigned long long>(point->address), module_name, point->original,
		             point->max_hits, point->reg >= 0 ? ", with memory dump" : "",
		             point->cond_reg >= 0 ? ", conditional" : "");
	}
	std::fflush(stderr);
}

void SetWatch(uint64_t address, uint32_t length, uint32_t max_hits) {
	if (address == 0 || g_watch_address.load() != 0) {
		return;
	}
	g_watch_length = length;
	g_watch_max    = max_hits;
	g_watch_address = address & ~static_cast<uint64_t>(length - 1);
	ReadWatchFilter();
	std::fprintf(stderr, "[watch] write watch set by the renderer on %llx len %u, max %u hits\n",
	             static_cast<unsigned long long>(address), length, max_hits);
	std::fflush(stderr);
	StartWatchThread();
}

#else

void SetWatch(uint64_t /*address*/, uint32_t /*length*/, uint32_t /*max_hits*/) {}

void Install(uint64_t /*module_base*/, uint64_t /*module_size*/, const char* /*module_name*/) {}

#endif

} // namespace Common::TracePoints
