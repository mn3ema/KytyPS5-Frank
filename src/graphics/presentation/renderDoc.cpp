#include "graphics/presentation/renderDoc.h"
#include "common/floorStats.h"

#include "common/sampler.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/faultBench.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/reuseProbe.h"
#include "graphics/host_gpu/timeline.h"
#include "kernel/guestWait.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <renderdoc_app.h>
#include <string>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <dlfcn.h>
#endif

namespace Libs::Graphics {

enum class RenderDocState : uint32_t {
	Idle,
	Requested,
	Capturing,
};

static RENDERDOC_API_1_6_0*        g_api             = nullptr;
static std::atomic<RenderDocState> g_state           = RenderDocState::Idle;
static uint32_t                    g_captured_flips  = 0;
static std::atomic_bool            g_unavailable_log = false;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static bool BindRenderDocApi(HMODULE module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI"));
#else
static bool BindRenderDocApi(void* module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(module, "RENDERDOC_GetAPI"));
#endif
	if (get_api == nullptr) {
		return false;
	}

	void* api = nullptr;
	if (get_api(eRENDERDOC_API_Version_1_6_0, &api) != 1 || api == nullptr) {
		return false;
	}

	g_api = static_cast<RENDERDOC_API_1_6_0*>(api);
	g_api->SetCaptureKeys(nullptr, 0);
	g_api->UnloadCrashHandler();
	LOGF("RenderDoc: API 1.6.0 bound\n");
	return true;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = GetModuleHandleA("renderdoc.dll");
	if (module == nullptr) {
		HKEY key = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
		                  L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0, KEY_READ,
		                  &key) != ERROR_SUCCESS) {
			return;
		}

		std::array<wchar_t, MAX_PATH> path_buffer {};
		DWORD      path_size = static_cast<DWORD>(path_buffer.size() * sizeof(wchar_t));
		const auto result    = RegQueryValueExW(
		    key, L"", nullptr, nullptr, reinterpret_cast<LPBYTE>(path_buffer.data()), &path_size);
		RegCloseKey(key);
		if (result != ERROR_SUCCESS) {
			return;
		}

		auto path = std::filesystem::path(path_buffer.data()).parent_path() / "renderdoc.dll";
		module    = LoadLibraryW(path.c_str());
		if (module == nullptr) {
			return;
		}
	}

	if (!BindRenderDocApi(module)) {
		LOGF("RenderDoc: API 1.6.0 is unavailable\n");
	}
}

#else

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = ::dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
	if (module == nullptr) {
		module = ::dlopen("librenderdoc.so", RTLD_NOW);
	}
	if (module == nullptr) {
		return;
	}

	if (!BindRenderDocApi(module)) {
		LOGF("RenderDoc: API 1.6.0 is unavailable\n");
		::dlclose(module);
	}
}

#endif

void RenderDocRequestCapture() {
	if (g_api == nullptr) {
		if (!g_unavailable_log.exchange(true)) {
			LOGF("RenderDoc: capture requested, but RenderDoc is unavailable\n");
		}
		return;
	}

	RenderDocState expected = RenderDocState::Idle;
	if (g_state.compare_exchange_strong(expected, RenderDocState::Requested)) {
		LOGF("RenderDoc: capture requested\n");
	}
}

static void StartCapture() {
	if (g_api->IsFrameCapturing() != 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		LOGF("RenderDoc: capture request ignored because a capture is already active\n");
		return;
	}

	const auto capture_id   = std::chrono::duration_cast<std::chrono::microseconds>(
	                              std::chrono::system_clock::now().time_since_epoch())
	                              .count();
	const auto capture_path = "_RenderDoc/kyty_" + std::to_string(capture_id);
	g_api->SetCaptureFilePathTemplate(capture_path.c_str());
	g_api->StartFrameCapture(nullptr, nullptr);
	if (g_api->IsFrameCapturing() == 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		LOGF("RenderDoc: capture failed to start\n");
		return;
	}
	g_captured_flips = 0;
	g_state.store(RenderDocState::Capturing, std::memory_order_release);
	LOGF("RenderDoc: capture started\n");
}

uint64_t AllocCountTotal(); // allocStats.cpp
void     AllocSitesReport(double elapsed); // allocStats.cpp: [alloc-site] (KYTY_ALLOC_SITES)

// KYTY_FLOOR_LOG=1: the `[floor]` line, once a second (floorStats.h).
static void ReportFloor() {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point window_start = Clock::now();
	static Clock::time_point last_flip    = window_start;
	static Clock::time_point run_start    = window_start;
	static uint32_t          flips        = 0;
	static double            worst_ms     = 0.0;
	static constexpr int FloorFields = 32;
	struct Snapshot {
		uint64_t v[FloorFields] {};
	};
	static Snapshot before;
	const auto   now      = Clock::now();
	const double frame_ms = std::chrono::duration<double, std::milli>(now - last_flip).count();
	worst_ms = std::max(worst_ms, frame_ms);
	last_flip = now;
	flips++;
	auto& c = Common::FloorStats::Get();
	// Smoothness plan (11-16). [ft]: every frame time of the second, after its [floor] line, so the
	// percentiles and the count of long frames come from frames and not from seconds
	// (KYTY_FRAME_TIMES=0: off). [long-frame]: what a frame of KYTY_LONG_FRAME_MS (default 100; 0: off)
	// or more held, from counters every run keeps (no KYTY_CPU_COST needed).
	static const bool frame_times_on = [] {
		const char* value = std::getenv("KYTY_FRAME_TIMES");
		return value == nullptr || value[0] != '0';
	}();
	static const double long_frame_ms = [] {
		const char* value = std::getenv("KYTY_LONG_FRAME_MS");
		return value != nullptr ? std::strtod(value, nullptr) : 100.0;
	}();
	static std::vector<float> frame_times;
	if (frame_times_on) {
		frame_times.push_back(static_cast<float>(frame_ms));
	}
	{
		constexpr int                N = 40;
		const std::atomic<uint64_t>* f[N] = {
		    &c.thread_process_ns, &c.thread_empty_ns, &c.thread_blocked_ns, &c.gpu_wait_ns, &c.gpu_waits,
		    &c.ring_wait_ns, &c.pipe_wait_ns, &c.pipe_waits, &c.pipe_built, &c.compiles, &c.compile_ns,
		    &c.tex_created, &c.tex_created_bytes, &c.tex_uploads, &c.tex_upload_bytes, &c.tex_deleted,
		    &c.heap_keys, &c.buf_created, &c.buf_created_bytes, &c.sync_uploads, &c.gpu_unmaps,
		    &c.amm_batches, &c.amm_batch_ns, &c.amm_wait_ns, &c.readback_game_ns, &c.readback_gpu_ns,
		    &c.draws, &c.dispatches, &c.cmd_buffers, &c.pipe_sync_ns, &c.sync_ns,
		    // v438zk: 31-35 the flips, 36-37 textures uploaded again.
		    &c.flip_requests, &c.flip_replaced, &c.flip_refused, &c.flip_cancelled, &c.flip_held,
		    &c.tex_again, &c.tex_same_bytes,
		    // v438zl: 38-39 compute lookups that expanded their functions.
		    &c.expand_ns, &c.expand_slow};
		static uint64_t frame_before[N] {};
		static bool     have_before = false;
		uint64_t        v[N];
		uint64_t        d[N];
		for (int i = 0; i < N; i++) {
			v[i] = f[i]->load(std::memory_order_relaxed);
			d[i] = v[i] - frame_before[i];
			frame_before[i] = v[i];
		}
		if (have_before && long_frame_ms > 0.0 && frame_ms >= long_frame_ms) {
			const auto ms = [&](int i) { return static_cast<double>(d[i]) * 1e-6; };
			const auto mb = [&](int i) { return static_cast<double>(d[i]) / (1024.0 * 1024.0); };
			const auto n  = [&](int i) { return static_cast<unsigned long long>(d[i]); };
			std::fprintf(stderr,
			             "[long-frame] t=%.1f %.0f ms | gpu-thread proc %.0f empty %.0f blocked %.0f ms | gpu "
			             "waits %.0f ms x%llu (ring %.0f) | pipelines wait %.0f ms x%llu (sync %.0f) built %llu, "
			             "translations %llu (%.0f ms) | textures new %llu (%.0f MB) uploads %llu (%.0f MB) "
			             "deleted %llu heap-keys %llu | buffers new %llu (%.0f MB) uploads %llu (%.0f ms) | "
			             "unmaps %llu, amm %llu batches %.0f ms (waits %.0f ms) | readback %.0f/%.0f ms | "
			             "draws %llu dispatches %llu cb %llu | flips asked %llu replaced %llu refused %llu "
			             "cancelled %llu held %llu | textures again %llu (%.0f MB unchanged) | function "
			             "expansions %.0f ms (%llu slow)\n",
			             std::chrono::duration<double>(now - run_start).count(), frame_ms, ms(0), ms(1), ms(2),
			             ms(3), n(4), ms(5), ms(6), n(7), ms(29), n(8), n(9), ms(10), n(11), mb(12), n(13),
			             mb(14), n(15), n(16), n(17), mb(18), n(19), ms(30), n(20), n(21), ms(22), ms(23),
			             ms(24), ms(25), n(26), n(27), n(28), n(31), n(32), n(33), n(34), n(35), n(36), mb(37),
			             ms(38), n(39));
		}
		have_before = true;
	}
	const double elapsed = std::chrono::duration<double>(now - window_start).count();
	if (elapsed < 1.0) {
		return;
	}
	Snapshot s;
	const std::atomic<uint64_t>* fields[FloorFields] = {&c.gpu_busy_ns, &c.thread_process_ns, &c.thread_empty_ns,
	                                           &c.thread_blocked_ns, &c.thread_command_ns, &c.sync_calls,
	                                           &c.sync_bytes, &c.sync_big, &c.sync_ns, &c.sync_uploads,
	                                           &c.readback_game_ns, &c.readback_game_n, &c.readback_gpu_ns,
	                                           &c.compiles, &c.compile_ns, &c.pipe_wait_ns, &c.pipe_waits,
	                                           &c.pipe_timeouts, &c.pipe_sync_ns, &c.pipe_built,
	                                           &c.draws, &c.replayed, &c.dispatches, &c.cmd_buffers,
	                                           // Review 9 (11-16): which wall binds.
	                                           &c.gpu_wait_ns, &c.gpu_waits, &c.ring_wait_ns,
	                                           &c.record_busy_ns, &c.record_drain_ns, &c.record_drains,
	                                           &c.lag_ticks, &c.lag_samples};
	for (int i = 0; i < FloorFields; i++) {
		s.v[i] = fields[i]->load(std::memory_order_relaxed);
	}
	uint64_t d[FloorFields];
	for (int i = 0; i < FloorFields; i++) {
		d[i] = s.v[i] - before.v[i];
	}
	const double ms = 1e-6 / elapsed; // ns -> ms per second
	std::fprintf(stderr,
	             "[floor] t=%.1f fps=%.1f worst=%.0fms gpu=%.0f proc=%.0f empty=%.0f blocked=%.0f cmd=%.0f "
	             "sync=%.0fk/s %.1fGB/s big=%.0f/s syncms=%.0f up=%.0f/s rb=%.0f/%.0f rbgpu=%.0f comp=%llu "
	             "compms=%.0f pwait=%.0f/%llu pto=%llu psync=%.0f pbuilt=%llu draws/f=%.0f replayed/f=%.0f dispatches/f=%.0f cb/f=%.0f "
	             // gwait: GPU thread blocked on the host GPU, ms/s / waits (part of proc); ring: of it,
	             // upload-ring wraps; rec: recording thread busy ms/s; drain: threads blocked on the
	             // recording thread, ms/s / blocking waits (part of proc); lag: command buffers handed
	             // off and not finished by the GPU, mean over the second's samples (compare with cb/f).
	             "gwait=%.0f/%llu ring=%.0f rec=%.0f drain=%.0f/%llu lag=%.1f\n",
	             std::chrono::duration<double>(now - run_start).count(), flips / elapsed, worst_ms,
	             d[0] * ms, d[1] * ms, d[2] * ms, d[3] * ms, d[4] * ms, d[5] / elapsed / 1000.0,
	             static_cast<double>(d[6]) / elapsed / 1e9, d[7] / elapsed, d[8] * ms, d[9] / elapsed,
	             d[10] * ms, d[11] / elapsed, d[12] * ms, static_cast<unsigned long long>(d[13]),
	             d[14] * ms, d[15] * ms, static_cast<unsigned long long>(d[16]),
	             static_cast<unsigned long long>(d[17]), d[18] * ms, static_cast<unsigned long long>(d[19]),
	             flips != 0 ? static_cast<double>(d[20]) / flips : 0.0, flips != 0 ? static_cast<double>(d[21]) / flips : 0.0,
	             flips != 0 ? static_cast<double>(d[22]) / flips : 0.0, flips != 0 ? static_cast<double>(d[23]) / flips : 0.0,
	             d[24] * ms, static_cast<unsigned long long>(d[25]), d[26] * ms, d[27] * ms, d[28] * ms,
	             static_cast<unsigned long long>(d[29]),
	             d[31] != 0 ? static_cast<double>(d[30]) / static_cast<double>(d[31]) : 0.0);
	if (frame_times_on && !frame_times.empty()) {
		std::string line;
		line.reserve(frame_times.size() * 6u + 24u);
		char text[32];
		std::snprintf(text, sizeof(text), "[ft] t=%.1f", std::chrono::duration<double>(now - run_start).count());
		line += text;
		for (const float value: frame_times) {
			std::snprintf(text, sizeof(text), " %.1f", static_cast<double>(value));
			line += text;
		}
		line += '\n';
		std::fputs(line.c_str(), stderr);
		frame_times.clear();
	}
	if (Libs::Graphics::ReuseProbe::Enabled()) {
		Libs::Graphics::ReuseProbe::Report(elapsed, flips);
	}
	before       = s;
	flips        = 0;
	worst_ms     = 0.0;
	window_start = now;
}

void RenderDocOnGuestFlip(RenderContext& renderer) {
	Common::Sampler::NoteGuestFrame(); // freeze recorder
	// KYTY_HITCH_COST_MS (timeline.h): where the GPU thread spent a long frame.
	if (CpuCost::Enabled()) {
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		CpuCost::FrameEnd(std::chrono::duration<double, std::milli>(now - last).count());
		last = now;
	}
	Libs::Graphics::ReuseProbe::g_frame.fetch_add(1, std::memory_order_relaxed);
	if (Common::FloorStats::Enabled()) {
		ReportFloor();
	}
	Libs::LibKernel::GuestWait::OnGuestFlip(); // KYTY_GUEST_WAIT_STATS: [guest-wait], [guest-submit]
	FaultBench::OnGuestFlip(renderer);         // KYTY_FAULT_BENCH: [fault-bench]
	// KYTY_FPS_LOG=1: guest flips per second, every 5 s, to stderr (for scripted runs).
	if (static const bool fps_log = std::getenv("KYTY_FPS_LOG") != nullptr; fps_log) {
		static auto     window_start = std::chrono::steady_clock::now();
		static uint32_t flips        = 0;
		flips++;
		const auto now     = std::chrono::steady_clock::now();
		const auto elapsed = std::chrono::duration<double>(now - window_start).count();
		if (elapsed >= 5.0) {
			std::fprintf(stderr, "[fps] %.2f\n", flips / elapsed);
			// Heap allocations per second (allocStats.cpp).
			static uint64_t allocs_before = 0;
			const uint64_t  allocs        = AllocCountTotal();
			std::fprintf(stderr, "[alloc] %.0f/s\n",
			             static_cast<double>(allocs - allocs_before) / elapsed);
			allocs_before = allocs;
			AllocSitesReport(elapsed);
			flips        = 0;
			window_start = now;
		}
	}
	// KYTY_RENDERDOC_TRIGGER=<file>: the file's appearance requests a capture (it is deleted),
	// for captures timed by a script instead of the F1 key.
	if (static const char* trigger = std::getenv("KYTY_RENDERDOC_TRIGGER");
	    trigger != nullptr && g_api != nullptr &&
	    g_state.load(std::memory_order_acquire) == RenderDocState::Idle) {
		std::error_code error;
		if (std::filesystem::exists(trigger, error)) {
			std::filesystem::remove(trigger, error);
			RenderDocRequestCapture();
		}
	}
	const auto state = g_state.load(std::memory_order_acquire);
	if (g_api == nullptr || state == RenderDocState::Idle) {
		return;
	}
	if (state == RenderDocState::Capturing) {
		LOGF("RenderDoc: captured guest flip %u/2\n", ++g_captured_flips);
		if (g_captured_flips < 2) {
			return;
		}
	}

	// Capture boundaries follow presentation and exclude concurrent queue access.
	Common::LockGuard render_lock(renderer.GetMutex());
	Common::LockGuard queue_lock(renderer.GetGraphics().queue_mutex);
	if (state == RenderDocState::Requested) {
		StartCapture();
	} else {
		const auto ok = g_api->EndFrameCapture(nullptr, nullptr);
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		LOGF(ok != 0 ? "RenderDoc: capture finished\n" : "RenderDoc: capture failed\n");
	}
}

} // namespace Libs::Graphics
