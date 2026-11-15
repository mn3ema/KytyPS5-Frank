#include "common/sampler.h"

#include "common/assert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <timeapi.h>
#endif

namespace Common::Sampler {

#ifdef _WIN32

namespace {
// Set while this thread unwinds another thread's stack (UnwindStack): a fault there (a frame
// register or stack slot that no longer holds what the unwind tables describe) is not the
// emulator's; the host exception handler passes it on to UnwindStack's __except (InUnwind).
thread_local bool t_in_unwind = false;
std::atomic<uint64_t> g_unwind_faults {0};
} // namespace

bool InUnwind() noexcept {
	return t_in_unwind;
}

namespace {

const auto g_process_start = std::chrono::steady_clock::now();

struct Config {
	bool     enabled = false;
	uint64_t start   = 170;
	uint64_t seconds = 10;
	// KYTY_SAMPLE_TRIGGER=<file>: after `start`, recording waits until this file exists (the
	// launcher creates it at a moment of play that a fixed start time cannot hit).
	std::string trigger;
};

const Config& GetConfig() {
	static const Config config = [] {
		Config c;
		const char* value = std::getenv("KYTY_SAMPLE");
		if (value == nullptr) {
			return c;
		}
		c.enabled = true;
		char* end = nullptr;
		c.start   = std::strtoull(value, &end, 10);
		if (end != nullptr && *end == ':') {
			c.seconds = std::strtoull(end + 1, nullptr, 10);
		}
		if (const char* trigger = std::getenv("KYTY_SAMPLE_TRIGGER"); trigger != nullptr && *trigger != 0) {
			c.trigger = trigger;
		}
		return c;
	}();
	return config;
}

struct Target {
	std::string name;
	HANDLE      handle     = nullptr;
	uintptr_t   stack_base = 0;
	// A guest thread's guest stack (RegisterGuestThread; 0: unknown, not unwound).
	uintptr_t   guest_low  = 0;
	uintptr_t   guest_high = 0;
};

constexpr int MaxCallers = 4;
constexpr int MaxFrames  = 40;

struct Sample {
	uint32_t target;
	DWORD64  rip;
	DWORD64  callers[MaxCallers]; // return addresses into the executable found on the stack
	DWORD64  frames[MaxFrames];   // unwound call stack, innermost first (frames[0] = rip)
	uint32_t depth;
	uint64_t tag; // g_gpu_tag when the thread named "gpu" was sampled
};

// Unwinds a suspended thread's stack with the x64 unwind tables (RtlVirtualUnwind), so the
// report can give inclusive times. Stops at a frame without unwind information whose return
// address does not look like one, or when the stack pointer leaves the thread's stack.
// The loaded modules' address ranges, taken before sampling: unwinding stops at code outside
// them (guest code, JIT), whose lookup could run function-table callbacks on bad addresses.
std::vector<std::pair<uintptr_t, uintptr_t>> g_modules;

bool InModule(DWORD64 address) {
	for (const auto& [begin, end]: g_modules) {
		if (address >= begin && address < end) return true;
	}
	return false;
}

void CollectModules() {
	HMODULE modules[1024];
	DWORD   needed = 0;
	if (EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed) == FALSE) return;
	for (DWORD i = 0; i < needed / sizeof(HMODULE) && i < 1024; i++) {
		MODULEINFO info {};
		if (GetModuleInformation(GetCurrentProcess(), modules[i], &info, sizeof(info)) != FALSE) {
			const auto begin = reinterpret_cast<uintptr_t>(info.lpBaseOfDll);
			g_modules.emplace_back(begin, begin + info.SizeOfImage);
		}
	}
}

// Registers and code bytes of a suspended guest thread (defined with the freeze recorder).
std::string GuestSpinState(HANDLE handle);

bool ReadableRange(uint64_t address, uint64_t size);

// The guest code around a hot address, for offline disassembly: a run of PAUSE (f3 90) through
// it is followed to its ends, then 160 bytes before and 320 after are dumped, 64 a line.
std::string GuestCodeAround(uint64_t rip) {
	MEMORY_BASIC_INFORMATION info {};
	if (rip == 0 || VirtualQuery(reinterpret_cast<const void*>(rip), &info, sizeof(info)) == 0 ||
	    info.State != MEM_COMMIT ||
	    (info.Protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_READONLY | PAGE_READWRITE)) == 0) {
		return {};
	}
	const auto  region_begin = reinterpret_cast<uint64_t>(info.BaseAddress);
	const auto  region_end   = region_begin + info.RegionSize;
	const auto  byte_at      = [](uint64_t a) { return *reinterpret_cast<const uint8_t*>(a); };
	const auto  pause_at     = [&](uint64_t a) {
		return a >= region_begin && a + 2 <= region_end && byte_at(a) == 0xf3 && byte_at(a + 1) == 0x90;
	};
	uint64_t run_begin = rip;
	uint64_t run_end   = rip;
	for (int i = 0; i < 4; i++) { // align onto the run's 2-byte grid
		if (pause_at(rip - i)) {
			run_begin = run_end = rip - i;
			break;
		}
	}
	while (run_begin >= 2 && rip - run_begin < 8192 && pause_at(run_begin - 2)) run_begin -= 2;
	while (run_end - rip < 8192 && pause_at(run_end)) run_end += 2;
	const auto begin = std::max(region_begin, run_begin - 160);
	const auto end   = std::min(region_end, run_end + 320);
	char       line[64];
	std::snprintf(line, sizeof(line), "[sample-code] pause run %llx-%llx\n",
	              static_cast<unsigned long long>(run_begin), static_cast<unsigned long long>(run_end));
	std::string out = line;
	// Strings the code loads as an argument (lea rdi, [rip + disp32]): a wait loop's label.
	for (uint64_t a = begin; a + 7 <= end; a++) {
		if (byte_at(a) != 0x48 || byte_at(a + 1) != 0x8d || byte_at(a + 2) != 0x3d) continue;
		int32_t disp = 0;
		std::memcpy(&disp, reinterpret_cast<const void*>(a + 3), sizeof(disp));
		const auto target = a + 7 + static_cast<int64_t>(disp);
		if (!ReadableRange(target, 1)) continue;
		std::string text;
		for (uint64_t c = target; text.size() < 96 && ReadableRange(c, 1); c++) {
			const auto ch = static_cast<char>(byte_at(c));
			if (ch == 0) break;
			text += (ch >= 32 && ch < 127) ? ch : '.';
		}
		char where[48];
		std::snprintf(where, sizeof(where), "%llx", static_cast<unsigned long long>(a));
		out += std::string("[sample-code] string loaded at ") + where + ": \"" + text + "\"\n";
	}
	for (uint64_t a = begin; a < end; a += 64) {
		std::snprintf(line, sizeof(line), "[sample-code] %llx ", static_cast<unsigned long long>(a));
		out += line;
		for (uint64_t b = a; b < a + 64 && b < end; b++) {
			char hex[4];
			std::snprintf(hex, sizeof(hex), "%02x", byte_at(b));
			out += hex;
		}
		out += "\n";
	}
	return out;
}

// Whether [address, address + size) is committed, readable memory of one region.
bool ReadableRange(uint64_t address, uint64_t size) {
	MEMORY_BASIC_INFORMATION info {};
	if (address == 0 || VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
		return false;
	}
	constexpr DWORD readable = PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_READONLY | PAGE_READWRITE |
	                           PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY;
	return info.State == MEM_COMMIT && (info.Protect & readable) != 0 && (info.Protect & PAGE_GUARD) == 0 &&
	       address + size <= reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
}

std::string HexBytes(uint64_t address, uint64_t size) {
	std::string out;
	for (uint64_t b = address; b < address + size; b++) {
		char hex[4];
		std::snprintf(hex, sizeof(hex), "%02x", *reinterpret_cast<const uint8_t*>(b));
		out += hex;
	}
	return out;
}

// A guest object a hot loop holds in a register: its first 64 bytes, its first 16 vtable entries
// (when its first qword points to readable memory) and 96 code bytes of each, for disassembly.
std::string GuestObjectDump(const char* name, uint64_t object) {
	if (!ReadableRange(object, 64)) {
		return {};
	}
	char        line[96];
	std::string out;
	std::snprintf(line, sizeof(line), "[sample-object] %s=%llx: ", name, static_cast<unsigned long long>(object));
	out += line + HexBytes(object, 64) + "\n";
	const auto vtable = *reinterpret_cast<const uint64_t*>(object);
	if (!ReadableRange(vtable, 16 * 8)) {
		return out;
	}
	for (int i = 0; i < 16; i++) {
		const auto entry = reinterpret_cast<const uint64_t*>(vtable)[i];
		std::snprintf(line, sizeof(line), "[sample-object]   vtable[%d] @%llx = %llx", i,
		              static_cast<unsigned long long>(vtable + 8u * static_cast<uint64_t>(i)),
		              static_cast<unsigned long long>(entry));
		out += line;
		if (ReadableRange(entry, 96)) {
			out += " code " + HexBytes(entry, 96);
		}
		out += "\n";
	}
	return out;
}

uint32_t UnwindStackUnguarded(const CONTEXT& start, uintptr_t stack_base, DWORD64* frames) {
	alignas(16) CONTEXT context = start;
	const auto stack_low = context.Rsp;
	uint32_t   depth     = 0;
	frames[depth++]      = context.Rip;
	while (depth < MaxFrames) {
		if (!InModule(context.Rip)) break;
		DWORD64 image_base = 0;
		auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
		if (function == nullptr) {
			// A leaf function: the return address is at the stack pointer.
			if (context.Rsp < stack_low || context.Rsp + 8 > stack_base) break;
			context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
			context.Rsp += 8;
		} else {
			PVOID   handler_data = nullptr;
			DWORD64 establisher  = 0;
			RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
			                 &handler_data, &establisher, nullptr);
		}
		if (context.Rip == 0 || context.Rsp < stack_low || context.Rsp >= stack_base) break;
		frames[depth++] = context.Rip;
	}
	return depth;
}

// A fault while unwinding (see t_in_unwind) leaves only the sampled instruction pointer.
uint32_t UnwindStack(const CONTEXT& start, uintptr_t stack_base, DWORD64* frames) {
	uint32_t depth = 1;
	t_in_unwind    = true;
	__try {
		depth = UnwindStackUnguarded(start, stack_base, frames);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		frames[0] = start.Rip;
		depth     = 1;
		g_unwind_faults.fetch_add(1, std::memory_order_relaxed);
	}
	t_in_unwind = false;
	return depth;
}

uintptr_t g_image_begin = 0, g_image_end = 0;

// A plausible return address: inside the executable, right after a call instruction.
bool IsReturnAddress(uintptr_t a) {
	if (a < g_image_begin + 16 || a >= g_image_end) return false;
	const auto* code = reinterpret_cast<const uint8_t*>(a);
	if (code[-5] == 0xE8) return true;                                // call rel32
	if (code[-6] == 0xFF && (code[-5] & 0x38) == 0x10) return true;   // call [rip+disp32] / [r+disp32]
	if (code[-2] == 0xFF && (code[-1] & 0x38) == 0x10) return true;   // call r / [r]
	if (code[-3] == 0xFF && (code[-2] & 0x38) == 0x10) return true;   // call [r+disp8]
	if (code[-7] == 0xFF && (code[-6] & 0x38) == 0x10) return true;   // call [r+disp32] with SIB
	return false;
}

// Scans the top of a suspended thread's stack (up to its base) for return addresses.
void ScanStack(DWORD64 rsp, uintptr_t stack_base, DWORD64* callers) {
	const auto* stack = reinterpret_cast<const uintptr_t*>(rsp);
	int         found = 0;
	for (int k = 0; k < 512 && found < MaxCallers; k++) {
		if (rsp + (k + 1) * sizeof(uintptr_t) > stack_base) break;
		if (IsReturnAddress(stack[k])) callers[found++] = stack[k];
	}
}

std::mutex          g_mutex;
std::vector<Target> g_targets;
bool                g_started = false;

std::string FunctionOf(const std::string& symbol) {
	// " name+0xoffset [file:line]" -> "name"
	auto s = symbol;
	if (!s.empty() && s[0] == ' ') s.erase(0, 1);
	if (const auto plus = s.find("+0x"); plus != std::string::npos) s.resize(plus);
	return s.empty() ? "?" : s;
}

std::string FileOf(const std::string& symbol) {
	const auto open  = symbol.find(" [");
	const auto colon = symbol.rfind(':');
	if (open == std::string::npos || colon == std::string::npos || colon < open) return "?";
	auto file = symbol.substr(open + 2, colon - open - 2);
	if (const auto slash = file.find_last_of("\\/"); slash != std::string::npos) file.erase(0, slash + 1);
	return file;
}

void Run() {
	const auto& config = GetConfig();
	std::this_thread::sleep_until(g_process_start + std::chrono::seconds(config.start));
	if (!config.trigger.empty()) {
		std::fprintf(stderr, "[sample] waiting for %s\n", config.trigger.c_str());
		const auto give_up = std::chrono::steady_clock::now() + std::chrono::hours(2);
		while (GetFileAttributesA(config.trigger.c_str()) == INVALID_FILE_ATTRIBUTES) {
			if (std::chrono::steady_clock::now() > give_up) {
				std::fprintf(stderr, "[sample] trigger never appeared, not recording\n");
				return;
			}
			Sleep(200);
		}
		std::fprintf(stderr, "[sample] trigger seen, recording %llu s\n",
		             static_cast<unsigned long long>(config.seconds));
	}
	std::vector<Target> targets;
	{
		std::scoped_lock lock(g_mutex);
		targets = g_targets;
	}
	if (targets.empty()) {
		return;
	}
	{
		MODULEINFO info {};
		if (GetModuleInformation(GetCurrentProcess(), GetModuleHandleW(nullptr), &info, sizeof(info)) != FALSE) {
			g_image_begin = reinterpret_cast<uintptr_t>(info.lpBaseOfDll);
			g_image_end   = g_image_begin + info.SizeOfImage;
		}
	}
	CollectModules();
	timeBeginPeriod(1);
	// No allocation while a target is suspended (it may hold the heap lock).
	std::vector<Sample> samples;
	// Guest threads (KYTY_SAMPLE_GUEST): every 4th sweep, their instruction pointer only.
	size_t guest_targets = 0;
	for (const auto& target: targets) {
		guest_targets += target.name.rfind("g:", 0) == 0 ? 1u : 0u;
	}
	samples.reserve(static_cast<size_t>(config.seconds) *
	                    (1200u * (targets.size() - guest_targets) + 300u * guest_targets) +
	                16u);
	const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(config.seconds);
	uint64_t   sweep = 0;
	while (std::chrono::steady_clock::now() < stop && samples.size() + targets.size() < samples.capacity()) {
		sweep++;
		for (uint32_t i = 0; i < targets.size(); i++) {
			const bool guest = targets[i].name.rfind("g:", 0) == 0;
			if (guest && sweep % 4u != 0u) continue;
			if (SuspendThread(targets[i].handle) == static_cast<DWORD>(-1)) continue;
			alignas(16) CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			const bool ok        = GetThreadContext(targets[i].handle, &context) != FALSE;
			Sample     sample {i, context.Rip, {}, {}, 0, targets[i].name == "gpu" ? g_gpu_tag.load(std::memory_order_relaxed) : 0};
			if (ok && !guest) {
				ScanStack(context.Rsp, targets[i].stack_base, sample.callers);
				sample.depth = UnwindStack(context, targets[i].stack_base, sample.frames);
			} else if (ok && targets[i].guest_high != 0 && InModule(context.Rip) &&
			           context.Rsp >= targets[i].guest_low && context.Rsp < targets[i].guest_high) {
				// Inside the emulator on the guest stack: unwind up to the guest code that called it.
				sample.depth = UnwindStack(context, targets[i].guest_high, sample.frames);
			} else if (ok && targets[i].guest_high != 0 && InModule(context.Rip) &&
			           context.Rsp < targets[i].stack_base &&
			           context.Rsp + (64u << 20u) > targets[i].stack_base) {
				// Inside the emulator on the thread's own (host) stack: everything between the stack
				// pointer and the base is in use, as for host threads.
				sample.depth = UnwindStack(context, targets[i].stack_base, sample.frames);
			}
			ResumeThread(targets[i].handle);
			if (ok) samples.push_back(sample);
		}
		Sleep(1);
	}
	timeEndPeriod(1);
	// Aggregate by function.
	std::unordered_map<DWORD64, std::string> names;
	// One line per thread: how much of the time it was not waiting, and where (guest threads too
	// with KYTY_SAMPLE_GUEST=1). Only the busiest guest threads get the full report below.
	std::vector<uint8_t> detailed(targets.size(), 1u);
	{
		const auto is_wait = [](const std::string& f) {
			return f.find("Wait") != std::string::npos || f.find("Sleep") != std::string::npos ||
			       f.find("Delay") != std::string::npos || f.find("Remove") != std::string::npos ||
			       f.find("ZwYield") != std::string::npos || f.find("NtYield") != std::string::npos;
		};
		std::vector<std::pair<double, uint32_t>> guest_busy;
		std::string                              out;
		for (uint32_t t = 0; t < targets.size(); t++) {
			std::map<std::string, uint64_t> by_function;
			uint64_t                        total = 0, busy = 0;
			for (const auto& s: samples) {
				if (s.target != t) continue;
				std::string f = "(guest code)";
				if (InModule(s.rip)) {
					auto it = names.find(s.rip);
					if (it == names.end()) it = names.emplace(s.rip, Common::HostAddressName(reinterpret_cast<const void*>(s.rip))).first;
					f = FunctionOf(it->second);
				}
				total++;
				if (!is_wait(f)) {
					busy++;
					by_function[f]++;
				}
			}
			if (total == 0) continue;
			std::vector<std::pair<std::string, uint64_t>> f(by_function.begin(), by_function.end());
			std::sort(f.begin(), f.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			char line[512];
			std::snprintf(line, sizeof(line), "[sample-threads] %-24s busy %5.1f%% |",
			              targets[t].name.substr(0, 24).c_str(), 100.0 * static_cast<double>(busy) / static_cast<double>(total));
			out += line;
			for (size_t i = 0; i < f.size() && i < 5; i++) {
				std::snprintf(line, sizeof(line), " %s %.1f%% |", f[i].first.substr(0, 60).c_str(),
				              100.0 * static_cast<double>(f[i].second) / static_cast<double>(total));
				out += line;
			}
			out += "\n";
			if (targets[t].name.rfind("g:", 0) == 0) {
				detailed[t] = 0u;
				guest_busy.emplace_back(static_cast<double>(busy) / static_cast<double>(total), t);
			}
		}
		std::sort(guest_busy.begin(), guest_busy.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		for (size_t i = 0; i < guest_busy.size() && i < 3; i++) {
			detailed[guest_busy[i].second] = 1u;
		}
		// [sample-guest]: the 10 busiest guest threads, every sample by where it was: guest code,
		// or the emulator functions it was in (the innermost three of the executable's own frames,
		// from the unwind), waits included: what a guest thread waits in when it is not running.
		for (size_t i = 0; i < guest_busy.size() && i < 10; i++) {
			const auto                      t = guest_busy[i].second;
			std::map<std::string, uint64_t> where;
			uint64_t                        total = 0;
			for (const auto& s: samples) {
				if (s.target != t) continue;
				total++;
				std::string key;
				if (!InModule(s.rip)) {
					key = "(guest code)";
				} else {
					int kept = 0;
					for (uint32_t d = 0; d < s.depth && kept < 3; d++) {
						const auto address = d == 0 ? s.frames[d] : s.frames[d] - 1;
						if (address < g_image_begin || address >= g_image_end) continue;
						auto it = names.find(address);
						if (it == names.end()) {
							it = names.emplace(address, Common::HostAddressName(reinterpret_cast<const void*>(address))).first;
						}
						if (!key.empty()) key += " < ";
						key += FunctionOf(it->second).substr(0, 80);
						kept++;
					}
					if (key.empty()) {
						auto it = names.find(s.rip);
						if (it == names.end()) {
							it = names.emplace(s.rip, Common::HostAddressName(reinterpret_cast<const void*>(s.rip))).first;
						}
						key = "(not unwound) " + FunctionOf(it->second).substr(0, 60);
					}
				}
				where[key]++;
			}
			std::vector<std::pair<std::string, uint64_t>> w(where.begin(), where.end());
			std::sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			// Guest code by instruction pointer: a few addresses holding most of it is a loop
			// polling memory (a wait written in guest code), not work.
			std::map<DWORD64, uint64_t> rips;
			for (const auto& s: samples) {
				if (s.target == t && !InModule(s.rip)) rips[s.rip]++;
			}
			std::vector<std::pair<DWORD64, uint64_t>> r(rips.begin(), rips.end());
			std::sort(r.begin(), r.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			char line[640];
			std::snprintf(line, sizeof(line), "[sample-guest] %s busy %.1f%% (%llu samples)\n",
			              targets[t].name.substr(0, 40).c_str(), 100.0 * guest_busy[i].first,
			              static_cast<unsigned long long>(total));
			out += line;
			for (size_t j = 0; j < w.size() && j < 8; j++) {
				std::snprintf(line, sizeof(line), "[sample-guest]   %5.1f%%  %s\n",
				              100.0 * static_cast<double>(w[j].second) / static_cast<double>(std::max<uint64_t>(total, 1)),
				              w[j].first.substr(0, 400).c_str());
				out += line;
			}
			if (!r.empty()) {
				std::string text = "[sample-guest]   guest code by address:";
				for (size_t j = 0; j < r.size() && j < 10; j++) {
					std::snprintf(line, sizeof(line), " %llx %.1f%%", static_cast<unsigned long long>(r[j].first),
					              100.0 * static_cast<double>(r[j].second) / static_cast<double>(std::max<uint64_t>(total, 1)));
					text += line;
				}
				std::snprintf(line, sizeof(line), " (%zu addresses)\n", r.size());
				out += text + line;
			}
			// The busiest guest thread caught again near its hottest addresses: registers and code
			// bytes (what a polling loop reads), up to 3 times within ~0.5 s.
			if (i == 0 && !r.empty()) {
				int caught = 0;
				for (int tries = 0; tries < 500 && caught < 3; tries++) {
					const auto state = GuestSpinState(targets[t].handle);
					const auto rip   = state.rfind("rip=", 0) == 0 ? std::strtoull(state.c_str() + 4, nullptr, 16) : 0ull;
					bool       close_to  = false;
					for (size_t j = 0; j < r.size() && j < 3 && !close_to; j++) {
						close_to = rip + 64 >= r[j].first && rip <= r[j].first + 64;
					}
					if (close_to) {
						out += "[sample-guest]   caught: " + state + "\n";
						if (caught == 0) {
							out += GuestCodeAround(rip);
							for (const char* reg: {"rbx", "rdi", "rsi"}) {
								const auto at = state.find(std::string(" ") + reg + "=");
								if (at != std::string::npos) {
									out += GuestObjectDump(reg, std::strtoull(state.c_str() + at + 5, nullptr, 16));
								}
							}
						}
						caught++;
					}
					Sleep(1);
				}
			}
		}
		std::fputs(out.c_str(), stderr);
	}
	for (uint32_t t = 0; t < targets.size(); t++) {
		if (detailed[t] == 0u) continue;
		std::map<std::string, uint64_t> by_function, by_file;
		uint64_t                        total = 0;
		for (const auto& s: samples) {
			if (s.target != t) continue;
			auto it = names.find(s.rip);
			if (it == names.end()) it = names.emplace(s.rip, Common::HostAddressName(reinterpret_cast<const void*>(s.rip))).first;
			by_function[FunctionOf(it->second)]++;
			by_file[FileOf(it->second)]++;
			total++;
		}
		std::vector<std::pair<std::string, uint64_t>> f(by_function.begin(), by_function.end()), g(by_file.begin(), by_file.end());
		std::sort(f.begin(), f.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string out = "[sample] thread " + targets[t].name + ": " + std::to_string(total) + " samples (" +
		                  std::to_string(g_unwind_faults.load(std::memory_order_relaxed)) + " unwinds cut by a fault, all threads)\n";
		char        line[512];
		for (size_t i = 0; i < f.size() && i < 45; i++) {
			std::snprintf(line, sizeof(line), "[sample]   %5.1f%%  %s\n", 100.0 * static_cast<double>(f[i].second) / static_cast<double>(std::max<uint64_t>(total, 1)), f[i].first.c_str());
			out += line;
		}
		// Callers of the hottest leaf functions (heuristic stack scan: may include stale frames).
		for (size_t i = 0; i < f.size() && i < 12; i++) {
			std::map<std::string, uint64_t> chains;
			uint64_t                        n = 0;
			for (const auto& s: samples) {
				if (s.target != t || FunctionOf(names[s.rip]) != f[i].first) continue;
				std::string chain;
				for (const auto c: s.callers) {
					if (c == 0) break;
					auto it = names.find(c);
					if (it == names.end()) it = names.emplace(c, Common::HostAddressName(reinterpret_cast<const void*>(c))).first;
					if (!chain.empty()) chain += " < ";
					chain += FunctionOf(it->second);
				}
				chains[chain.empty() ? "?" : chain]++;
				n++;
			}
			std::vector<std::pair<std::string, uint64_t>> c(chains.begin(), chains.end());
			std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			out += "[sample]  callers of " + f[i].first + ":\n";
			for (size_t j = 0; j < c.size() && j < 5; j++) {
				std::snprintf(line, sizeof(line), "[sample]      %5.1f%%  %s\n", 100.0 * static_cast<double>(c[j].second) / static_cast<double>(std::max<uint64_t>(n, 1)), c[j].first.substr(0, 400).c_str());
				out += line;
			}
		}
		// Inclusive time (each function counted once per sample) and, for the heaviest, what
		// they spend it in (the callee one frame deeper, or "self").
		{
			std::map<std::string, uint64_t>                            inclusive;
			std::map<std::string, std::map<std::string, uint64_t>>     callees;
			for (const auto& s: samples) {
				if (s.target != t) continue;
				std::vector<std::string> stack; // outermost last
				for (uint32_t d = 0; d < s.depth; d++) {
					// A return address points after the call: look up the byte before it.
					const auto address = d == 0 ? s.frames[d] : s.frames[d] - 1;
					auto       it      = names.find(address);
					if (it == names.end()) {
						it = names.emplace(address, Common::HostAddressName(reinterpret_cast<const void*>(address))).first;
					}
					stack.push_back(FunctionOf(it->second));
				}
				std::vector<std::string> seen;
				for (size_t d = 0; d < stack.size(); d++) {
					const auto& name = stack[d];
					if (std::find(seen.begin(), seen.end(), name) != seen.end()) continue;
					seen.push_back(name);
					inclusive[name]++;
					callees[name][d == 0 ? std::string("(self)") : stack[d - 1]]++;
				}
			}
			std::vector<std::pair<std::string, uint64_t>> v(inclusive.begin(), inclusive.end());
			std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			out += "[sample-incl] thread " + targets[t].name + "\n";
			for (size_t i = 0; i < v.size() && i < 80; i++) {
				std::snprintf(line, sizeof(line), "[sample-incl]   %5.1f%%  %s\n",
				              100.0 * static_cast<double>(v[i].second) / static_cast<double>(std::max<uint64_t>(total, 1)),
				              v[i].first.substr(0, 300).c_str());
				out += line;
			}
			for (size_t i = 0; i < v.size() && i < 60; i++) {
				const auto& callee = callees[v[i].first];
				std::vector<std::pair<std::string, uint64_t>> c(callee.begin(), callee.end());
				std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				out += "[sample-tree] " + v[i].first.substr(0, 200) + ":";
				for (size_t j = 0; j < c.size() && j < 8; j++) {
					std::snprintf(line, sizeof(line), " %s %.1f%% |", c[j].first.substr(0, 120).c_str(),
					              100.0 * static_cast<double>(c[j].second) / static_cast<double>(std::max<uint64_t>(total, 1)));
					out += line;
				}
				out += "\n";
			}
		}
		// By tag (the GPU thread's current draw or dispatch): the heaviest tags' inclusive functions.
		{
			std::map<uint64_t, uint64_t> tags;
			for (const auto& s: samples) {
				if (s.target == t && s.tag != 0) tags[s.tag]++;
			}
			std::vector<std::pair<uint64_t, uint64_t>> tv(tags.begin(), tags.end());
			std::sort(tv.begin(), tv.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			for (size_t i = 0; i < tv.size() && i < 6; i++) {
				std::map<std::string, uint64_t> inclusive;
				for (const auto& s: samples) {
					if (s.target != t || s.tag != tv[i].first) continue;
					std::vector<std::string> seen;
					for (uint32_t d = 0; d < s.depth; d++) {
						const auto address = d == 0 ? s.frames[d] : s.frames[d] - 1;
						auto       it      = names.find(address);
						if (it == names.end()) {
							it = names.emplace(address, Common::HostAddressName(reinterpret_cast<const void*>(address))).first;
						}
						auto name = FunctionOf(it->second);
						if (std::find(seen.begin(), seen.end(), name) != seen.end()) continue;
						seen.push_back(name);
						inclusive[name]++;
					}
				}
				std::vector<std::pair<std::string, uint64_t>> v(inclusive.begin(), inclusive.end());
				std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				std::snprintf(line, sizeof(line), "[sample-tag] %c %014llx: %.1f%% of the thread\n",
				              static_cast<char>(tv[i].first >> 56u),
				              static_cast<unsigned long long>(tv[i].first & ((uint64_t {1} << 56u) - 1u)),
				              100.0 * static_cast<double>(tv[i].second) / static_cast<double>(std::max<uint64_t>(total, 1)));
				out += line;
				for (size_t j = 0; j < v.size() && j < 40; j++) {
					std::snprintf(line, sizeof(line), "[sample-tag]   %5.1f%%  %s\n",
					              100.0 * static_cast<double>(v[j].second) / static_cast<double>(std::max<uint64_t>(tv[i].second, 1)),
					              v[j].first.substr(0, 200).c_str());
					out += line;
				}
			}
		}
		out += "[sample]  by file:";
		for (size_t i = 0; i < g.size() && i < 20; i++) {
			std::snprintf(line, sizeof(line), " %s %.1f%%", g[i].first.c_str(), 100.0 * static_cast<double>(g[i].second) / static_cast<double>(std::max<uint64_t>(total, 1)));
			out += line;
		}
		out += "\n";
		std::fputs(out.c_str(), stderr);
	}
	std::fflush(stderr);
}

// --- Freeze recorder ---

std::atomic<int64_t> g_last_frame_ms {0};
std::atomic<bool>    g_freeze_watch_started {false};
// Names of the threads that called RegisterCurrentThread (always kept, for [freeze]/[hitch]).
std::mutex                              g_names_mutex;
std::unordered_map<DWORD, std::string>  g_thread_names;

// A busy thread outside the executable (guest code): its registers and the code bytes around
// its instruction pointer, so a spin loop's polled address can be read off ([hitch] only).
std::string NameOf(DWORD64 address);

std::string GuestSpinState(HANDLE handle) {
	if (SuspendThread(handle) == static_cast<DWORD>(-1)) return {};
	CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	const bool ok        = GetThreadContext(handle, &context) != FALSE;
	ResumeThread(handle);
	if (!ok) return {};
	char text[640];
	std::snprintf(text, sizeof(text),
	              "rip=%llx rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx rbp=%llx rsp=%llx "
	              "r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx",
	              static_cast<unsigned long long>(context.Rip), static_cast<unsigned long long>(context.Rax),
	              static_cast<unsigned long long>(context.Rbx), static_cast<unsigned long long>(context.Rcx),
	              static_cast<unsigned long long>(context.Rdx), static_cast<unsigned long long>(context.Rsi),
	              static_cast<unsigned long long>(context.Rdi), static_cast<unsigned long long>(context.Rbp),
	              static_cast<unsigned long long>(context.Rsp), static_cast<unsigned long long>(context.R8),
	              static_cast<unsigned long long>(context.R9), static_cast<unsigned long long>(context.R10),
	              static_cast<unsigned long long>(context.R11), static_cast<unsigned long long>(context.R12),
	              static_cast<unsigned long long>(context.R13), static_cast<unsigned long long>(context.R14),
	              static_cast<unsigned long long>(context.R15));
	std::string out = text;
	MEMORY_BASIC_INFORMATION info {};
	const auto begin = context.Rip - 48;
	if (VirtualQuery(reinterpret_cast<const void*>(begin), &info, sizeof(info)) != 0 &&
	    info.State == MEM_COMMIT && (info.Protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_READONLY | PAGE_READWRITE)) != 0 &&
	    begin + 80 <= reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize) {
		out += " code[rip-48..+32]=";
		const auto* bytes = reinterpret_cast<const uint8_t*>(begin);
		for (int i = 0; i < 80; i++) {
			char hex[4];
			std::snprintf(hex, sizeof(hex), "%02x", bytes[i]);
			out += hex;
			if (i == 47) out += "|";
		}
	}
	return out;
}

// A named host thread (gpu, present, ...): its call stack unwound with the x64 unwind tables
// while it is suspended (exact, unlike the return-address scan), named after it resumes.
std::string UnwoundStack(HANDLE handle) {
	DWORD64 frames[MaxFrames] {};
	uint32_t depth = 0;
	if (SuspendThread(handle) == static_cast<DWORD>(-1)) return {};
	alignas(16) CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	if (GetThreadContext(handle, &context) != FALSE) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(context.Rsp), &info, sizeof(info)) != 0) {
			depth = UnwindStack(context, reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize, frames);
		}
	}
	ResumeThread(handle);
	std::string out;
	for (uint32_t i = 0; i < depth && i < 28; i++) {
		out += (i == 0 ? "" : " < ") + NameOf(frames[i]);
	}
	return out;
}

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now() - g_process_start)
	    .count();
}

uint64_t ThreadCpuMs(HANDLE handle) {
	FILETIME create {}, exit {}, kernel {}, user {};
	if (GetThreadTimes(handle, &create, &exit, &kernel, &user) == FALSE) return 0;
	const auto k = (static_cast<uint64_t>(kernel.dwHighDateTime) << 32u) | kernel.dwLowDateTime;
	const auto u = (static_cast<uint64_t>(user.dwHighDateTime) << 32u) | user.dwLowDateTime;
	return (k + u) / 10000u;
}

constexpr int FreezeCallers = 14;

struct FreezeSample {
	bool    ok = false;
	DWORD64 rip = 0;
	DWORD64 callers[FreezeCallers] {};
};

// Suspends the thread, reads its instruction pointer and the return addresses on its stack, and
// resumes it; nothing is allocated meanwhile (it may hold the heap lock).
FreezeSample SampleThread(HANDLE handle) {
	FreezeSample sample;
	if (SuspendThread(handle) == static_cast<DWORD>(-1)) return sample;
	CONTEXT context {};
	context.ContextFlags = CONTEXT_CONTROL;
	if (GetThreadContext(handle, &context) != FALSE) {
		sample.ok  = true;
		sample.rip = context.Rip;
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(context.Rsp), &info, sizeof(info)) != 0) {
			const auto top   = reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize;
			const auto* slot = reinterpret_cast<const uintptr_t*>(context.Rsp);
			int         found = 0;
			for (int k = 0; k < 2048 && found < FreezeCallers; k++) {
				if (context.Rsp + (k + 1) * sizeof(uintptr_t) > top) break;
				if (IsReturnAddress(slot[k])) sample.callers[found++] = slot[k];
			}
		}
	}
	ResumeThread(handle);
	return sample;
}

bool InImage(DWORD64 address) {
	return address >= g_image_begin && address < g_image_end;
}

std::string NameOf(DWORD64 address) {
	if (address == 0) return "?";
	auto name = FunctionOf(Common::HostAddressName(reinterpret_cast<const void*>(address)));
	if (name == "?" || name.empty()) {
		char text[48];
		if (InImage(address)) {
			std::snprintf(text, sizeof(text), "exe+0x%llx", static_cast<unsigned long long>(address - g_image_begin));
		} else {
			std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(address));
		}
		name = text;
	}
	return name;
}

// `hitch`: a short stall (KYTY_HITCH_MS), measured over 250 ms instead of 1 s, threads busy for
// a fifth of it listed, lines tagged [hitch].
void FreezeReport(int report, bool hitch = false) {
	const char*  tag        = hitch ? "[hitch]" : "[freeze]";
	const DWORD  measure_ms = hitch ? 250u : 1000u;
	const uint64_t busy_ms  = hitch ? 50u : 100u;
	const DWORD  gap_ms     = hitch ? 60u : 100u;
	struct Thread {
		DWORD    id;
		HANDLE   handle;
		uint64_t cpu0;
	};
	std::vector<Thread> threads;
	const DWORD         pid      = GetCurrentProcessId();
	const DWORD         self     = GetCurrentThreadId();
	HANDLE              snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != FALSE; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) continue;
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
		                           FALSE, entry.th32ThreadID);
		if (handle != nullptr) threads.push_back({entry.th32ThreadID, handle, ThreadCpuMs(handle)});
	}
	CloseHandle(snapshot);
	std::unordered_map<DWORD, std::string> names;
	{
		std::scoped_lock lock(g_names_mutex);
		names = g_thread_names;
	}
	const auto gap_at_start = NowMs() - g_last_frame_ms.load();
	Sleep(measure_ms);
	std::string out;
	char        line[256];
	std::snprintf(line, sizeof(line), "%s report %d: no guest flip for %lld ms, %zu threads\n", tag, report,
	              static_cast<long long>(gap_at_start), threads.size());
	out += line;
	int listed = 0;
	for (auto& thread: threads) {
		const auto cpu  = ThreadCpuMs(thread.handle) - thread.cpu0;
		const bool busy = cpu >= busy_ms;
		FreezeSample samples[3];
		samples[0] = SampleThread(thread.handle);
		bool ours  = samples[0].ok && InImage(samples[0].rip);
		for (const auto c: samples[0].callers) ours = ours || (c != 0 && InImage(c));
		const auto named_it = names.find(thread.id);
		const bool named    = named_it != names.end() && named_it->second.rfind("g:", 0) != 0;
		if (!busy && !(hitch && named) && (!ours || listed >= 48)) continue;
		listed++;
		if (busy || (hitch && named)) {
			Sleep(gap_ms);
			samples[1] = SampleThread(thread.handle);
			Sleep(gap_ms);
			samples[2] = SampleThread(thread.handle);
		}
		char  name[64] = {};
		PWSTR description = nullptr;
		if (named_it != names.end()) {
			std::snprintf(name, sizeof(name), "%s", named_it->second.c_str());
		} else if (SUCCEEDED(GetThreadDescription(thread.handle, &description)) && description != nullptr) {
			WideCharToMultiByte(CP_UTF8, 0, description, -1, name, sizeof(name) - 1, nullptr, nullptr);
			LocalFree(description);
		}
		std::snprintf(line, sizeof(line), "%s thread %lu '%s' cpu %llu ms in %lu ms%s\n", tag,
		              static_cast<unsigned long>(thread.id), name, static_cast<unsigned long long>(cpu),
		              static_cast<unsigned long>(measure_ms), busy ? " BUSY" : "");
		out += line;
		if (hitch && busy && samples[0].ok && !InImage(samples[0].rip)) {
			out += std::string(tag) + "   regs " + GuestSpinState(thread.handle) + "\n";
		}
		if (hitch && named) {
			out += std::string(tag) + "   stack " + UnwoundStack(thread.handle).substr(0, 2500) + "\n";
		}
		for (const auto& sample: samples) {
			if (!sample.ok) continue;
			std::string chain = std::string(tag) + "   at " + NameOf(sample.rip);
			for (const auto c: sample.callers) {
				if (c == 0) break;
				chain += " < " + NameOf(c);
			}
			out += chain.substr(0, 1500) + "\n";
		}
	}
	for (auto& thread: threads) CloseHandle(thread.handle);
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

void FreezeWatch() {
	if (g_modules.empty()) {
		CollectModules();
	}
	{
		MODULEINFO info {};
		if (GetModuleInformation(GetCurrentProcess(), GetModuleHandleW(nullptr), &info, sizeof(info)) != FALSE) {
			g_image_begin = reinterpret_cast<uintptr_t>(info.lpBaseOfDll);
			g_image_end   = g_image_begin + info.SizeOfImage;
		}
	}
	// KYTY_HITCH_MS=<ms>: also report, once per stall, every stall without a guest flip longer
	// than this (the first 40): which threads were busy and where (plan 4.14, the 3 s stalls).
	static const int64_t hitch_ms = [] {
		const char* value = std::getenv("KYTY_HITCH_MS");
		return value != nullptr ? static_cast<int64_t>(std::strtoll(value, nullptr, 10)) : int64_t {0};
	}();
	int     reports       = 0;
	int     hitch_reports = 0;
	int64_t hitch_last    = -1; // the flip time of the stall last reported
	bool    hitch_second  = false;
	int64_t next_report   = 0;
	for (;;) {
		Sleep(hitch_ms > 0 ? 50 : 1000);
		const auto last = g_last_frame_ms.load(std::memory_order_relaxed);
		const auto now  = NowMs();
		if (hitch_ms > 0 && now - last >= hitch_ms && last != hitch_last && hitch_reports < 40 &&
		    now - last < 15000) {
			hitch_last   = last;
			hitch_second = false;
			FreezeReport(++hitch_reports, true);
		} else if (hitch_ms > 0 && last == hitch_last && !hitch_second && now - last >= 2 * hitch_ms &&
		           hitch_reports < 40 && now - last < 15000) {
			// The same stall, a second look: what moved and what did not.
			hitch_second = true;
			FreezeReport(++hitch_reports, true);
		}
		if (now - last < 15000) {
			next_report = 0;
			continue;
		}
		if (reports >= 3 || now < next_report) continue;
		for (int i = 0; i < 3 && reports < 3; i++) {
			FreezeReport(++reports);
		}
		next_report = now + 60000;
	}
}

} // namespace

void NoteGuestFrame() {
	g_last_frame_ms.store(NowMs(), std::memory_order_relaxed);
	if (!g_freeze_watch_started.load(std::memory_order_relaxed) &&
	    !g_freeze_watch_started.exchange(true)) {
		const char* value = std::getenv("KYTY_FREEZE_DUMP");
		if (value == nullptr || value[0] != '0') {
			std::thread(FreezeWatch).detach();
		}
	}
}

void NameCurrentThread(const char* name) {
	std::scoped_lock lock(g_names_mutex);
	g_thread_names[GetCurrentThreadId()] = name;
}

void RegisterGuestThread(const char* name, uintptr_t stack_low, uintptr_t stack_high) {
	RegisterCurrentThread(name);
	if (!GetConfig().enabled) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	const auto id = GetCurrentThreadId();
	for (auto& target: g_targets) {
		if (target.name == name && GetThreadId(target.handle) == id) {
			target.guest_low  = stack_low;
			target.guest_high = stack_high;
		}
	}
}

void RegisterCurrentThread(const char* name) {
	{
		std::scoped_lock lock(g_names_mutex);
		g_thread_names[GetCurrentThreadId()] = name;
	}
	if (!GetConfig().enabled) {
		return;
	}
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
	                    0) == FALSE) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
	g_targets.push_back({name, handle, reinterpret_cast<uintptr_t>(tib->StackBase)});
	if (!g_started) {
		g_started = true;
		std::thread(Run).detach();
	}
}

#else

bool InUnwind() noexcept {
	return false;
}
void RegisterCurrentThread(const char* /*name*/) {}
void RegisterGuestThread(const char* /*name*/, uintptr_t /*stack_low*/, uintptr_t /*stack_high*/) {}
void NameCurrentThread(const char* /*name*/) {}
void NoteGuestFrame() {}

#endif

} // namespace Common::Sampler
