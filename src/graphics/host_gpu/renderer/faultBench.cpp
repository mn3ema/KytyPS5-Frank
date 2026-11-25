#include "graphics/host_gpu/renderer/faultBench.h"

#include "common/common.h"
#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace Libs::Graphics::FaultBench {

namespace {

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

uint64_t EnvNumber(const char* name, uint64_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : fallback;
}

constexpr uint64_t MapBytes = 0x10000;
constexpr uint64_t MapHint  = 0x7f0000000ull;
constexpr char     MapName[] = "KytyFaultBench";
constexpr uint32_t WarmUp   = 16;

// The handler's two times for the fault in progress (the bench thread's own faults only: one run
// at a time).
std::atomic<uint64_t> g_enter_ns {0};
std::atomic<uint64_t> g_leave_ns {0};

std::atomic<bool>     g_running {false};
std::atomic<uint64_t> g_next_ns {0};      // when the next run is due; 0: at the next flip
std::atomic<uint64_t> g_scratch {0};      // the mapping, once made
std::atomic<bool>     g_failed {false};   // the mapping could not be made: no more runs
const auto            g_start = std::chrono::steady_clock::now();

void Run(RenderContext& renderer, uint32_t run) {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	(void)Common::HostException::InitializeThreadSignalStack();
#endif
	auto page = g_scratch.load(std::memory_order_relaxed);
	if (page == 0) {
		void*     address = reinterpret_cast<void*>(MapHint);
		const int prot    = 0x33; // CPU and GPU, read and write: as the game maps GPU memory
		const int result  =
		    LibKernel::Memory::KernelMapNamedFlexibleMemory(&address, MapBytes, prot, 0, MapName);
		page = reinterpret_cast<uint64_t>(address);
		if (result != 0 || page == 0 || !renderer.IsMapped(page, MapBytes)) {
			std::fprintf(stderr,
			             "[fault-bench] no scratch page: the map returned 0x%x at 0x%llx (tracked by the renderer: "
			             "%d); nothing measured\n",
			             static_cast<unsigned>(result), static_cast<unsigned long long>(page),
			             page != 0 && renderer.IsMapped(page, MapBytes) ? 1 : 0);
			g_failed.store(true, std::memory_order_relaxed);
			return;
		}
		g_scratch.store(page, std::memory_order_relaxed);
	}
	// Still the bench's own mapping (a fixed map of the game's would have replaced it).
	{
		LibKernel::Memory::VirtualQueryInfo info {};
		const int result = LibKernel::Memory::KernelVirtualQuery(reinterpret_cast<const void*>(page), 0, &info,
		                                                         sizeof(info));
		if (result != 0 || info.start > page || info.end < page + MapBytes || info.is_flexible == 0 ||
		    info.is_gpu_prt != 0 || std::strncmp(info.name, MapName, sizeof(MapName)) != 0 ||
		    !renderer.IsMapped(page, MapBytes)) {
			std::fprintf(stderr,
			             "[fault-bench] run %u: the scratch page at 0x%llx is no longer the bench's; no more runs\n", run,
			             static_cast<unsigned long long>(page));
			g_failed.store(true, std::memory_order_relaxed);
			return;
		}
	}
	const auto n       = static_cast<uint32_t>(std::clamp<uint64_t>(EnvNumber("KYTY_FAULT_BENCH_N", 2000), 100, 200000));
	auto&      buffers = renderer.GetBufferCache();
	auto*      byte    = reinterpret_cast<volatile uint8_t*>(page);

	std::vector<uint32_t> totals;
	totals.reserve(n);
	uint64_t arm_ns = 0, dispatch_ns = 0, handler_ns = 0, resume_ns = 0, total_ns = 0;
	uint32_t measured = 0, not_faulted = 0;
	Detail::g_page.store(page, std::memory_order_release);
	for (uint32_t i = 0; i < n + WarmUp; i++) {
		const auto t0 = NowNs();
		buffers.FaultBenchArm(page, Detail::PageBytes);
		g_enter_ns.store(0, std::memory_order_relaxed);
		g_leave_ns.store(0, std::memory_order_relaxed);
		const uint8_t value = *byte; // reads are not watched
		const auto    t1    = NowNs();
		*byte               = value; // the write fault
		const auto t2       = NowNs();
		const auto enter = g_enter_ns.load(std::memory_order_relaxed);
		const auto leave = g_leave_ns.load(std::memory_order_relaxed);
		if (i < WarmUp) {
			continue;
		}
		if (enter == 0 || leave == 0) {
			not_faulted++;
			continue;
		}
		measured++;
		arm_ns += t1 - t0;
		dispatch_ns += enter - t1;
		handler_ns += leave - enter;
		resume_ns += t2 - leave;
		total_ns += t2 - t1;
		totals.push_back(static_cast<uint32_t>(std::min<uint64_t>(t2 - t1, UINT32_MAX)));
	}
	Detail::g_page.store(0, std::memory_order_release);

	// One bare host protection change, as the page manager makes it (the page is not watched
	// now: read-only and back leaves it as the tracker has it).
	uint64_t protect_ns = 0;
	bool     protect_ok = true;
	{
		const auto t0 = NowNs();
		for (uint32_t i = 0; i < n; i++) {
			protect_ok &= LibKernel::Memory::ProtectGuestHostMemory(page, Detail::PageBytes,
			                                                         Common::VirtualMemory::Mode::Read);
			protect_ok &= LibKernel::Memory::ProtectGuestHostMemory(page, Detail::PageBytes,
			                                                         Common::VirtualMemory::Mode::ReadWrite);
		}
		protect_ns = NowNs() - t0;
	}
	// The clock itself, read four times per fault above.
	uint64_t clock_ns = 0;
	{
		const auto t0 = NowNs();
		for (uint32_t i = 0; i < 4096; i++) {
			(void)NowNs();
		}
		clock_ns = (NowNs() - t0) / 4096;
	}

	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
	if (measured == 0) {
		std::fprintf(stderr,
		             "[fault-bench] run %u t=%.1f: none of %u armed writes faulted (page 0x%llx): the tracker did "
		             "not write-protect it; nothing measured\n",
		             run, seconds, n, static_cast<unsigned long long>(page));
		return;
	}
	std::sort(totals.begin(), totals.end());
	const auto   us  = [measured](uint64_t ns) { return static_cast<double>(ns) * 1e-3 / static_cast<double>(measured); };
	const auto   at  = [&totals](double q) {
		const auto index = static_cast<size_t>(q * static_cast<double>(totals.size()));
		return static_cast<double>(totals[index < totals.size() ? index : totals.size() - 1]) * 1e-3;
	};
	const double per = us(total_ns);
	std::fprintf(stderr,
	             "[fault-bench] run %u t=%.1f: %u write faults on a write-watched page (0x%llx, a thread of its "
	             "own): %.2f us per fault (median %.2f, p90 %.2f, p99 %.2f, max %.2f) = dispatch to the handler "
	             "%.2f + handler %.2f + resume %.2f | arming (tracker and protection, the GPU thread's side in a "
	             "game) %.2f us | one bare protection change %.2f us%s | clock read %llu ns (x4 inside a fault's "
	             "time) | 1000 faults a frame = %.2f ms | armed writes that did not fault: %u\n",
	             run, seconds, measured, static_cast<unsigned long long>(page), per, at(0.5), at(0.9), at(0.99),
	             static_cast<double>(totals.back()) * 1e-3, us(dispatch_ns), us(handler_ns), us(resume_ns),
	             us(arm_ns), static_cast<double>(protect_ns) * 1e-3 / (2.0 * static_cast<double>(n)),
	             protect_ok ? "" : " (a call failed)", static_cast<unsigned long long>(clock_ns), per, not_faulted);
}

} // namespace

namespace Detail {

bool g_enabled = [] {
	const char* value = std::getenv("KYTY_FAULT_BENCH");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}();
std::atomic<uint64_t> g_page {0};

uint64_t HandlerEnter() noexcept {
	const auto now = NowNs();
	g_enter_ns.store(now, std::memory_order_relaxed);
	return now;
}

void HandlerLeave() noexcept {
	g_leave_ns.store(NowNs(), std::memory_order_relaxed);
}

void OnGuestFlipSlow(RenderContext& renderer) {
	const auto now = NowNs();
	const auto due = g_next_ns.load(std::memory_order_relaxed);
	if (now < due || g_failed.load(std::memory_order_relaxed) || g_running.exchange(true)) {
		return;
	}
	static const uint64_t every_ns = EnvNumber("KYTY_FAULT_BENCH_EVERY_S", 0) * 1000000000ull;
	static uint32_t       runs     = 0; // under g_running
	g_next_ns.store(every_ns != 0 ? now + every_ns : UINT64_MAX, std::memory_order_relaxed);
	const auto run = ++runs;
	std::thread([&renderer, run] {
		Run(renderer, run);
		g_running.store(false, std::memory_order_release);
	}).detach();
}

} // namespace Detail

} // namespace Libs::Graphics::FaultBench
