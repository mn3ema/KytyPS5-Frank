#include "graphics/presentation/renderDoc.h"
#include "common/floorStats.h"

#include "common/sampler.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/reuseProbe.h"
#include "graphics/host_gpu/timeline.h"

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

// KYTY_FLOOR_LOG=1: the `[floor]` line, once a second (floorStats.h).
static void ReportFloor() {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point window_start = Clock::now();
	static Clock::time_point last_flip    = window_start;
	static Clock::time_point run_start    = window_start;
	static uint32_t          flips        = 0;
	static double            worst_ms     = 0.0;
	struct Snapshot {
		uint64_t v[24] {};
	};
	static Snapshot before;
	const auto now = Clock::now();
	worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(now - last_flip).count());
	last_flip = now;
	flips++;
	const double elapsed = std::chrono::duration<double>(now - window_start).count();
	if (elapsed < 1.0) {
		return;
	}
	auto&    c = Common::FloorStats::Get();
	Snapshot s;
	const std::atomic<uint64_t>* fields[24] = {&c.gpu_busy_ns, &c.thread_process_ns, &c.thread_empty_ns,
	                                           &c.thread_blocked_ns, &c.thread_command_ns, &c.sync_calls,
	                                           &c.sync_bytes, &c.sync_big, &c.sync_ns, &c.sync_uploads,
	                                           &c.readback_game_ns, &c.readback_game_n, &c.readback_gpu_ns,
	                                           &c.compiles, &c.compile_ns, &c.pipe_wait_ns, &c.pipe_waits,
	                                           &c.pipe_timeouts, &c.pipe_sync_ns, &c.pipe_built,
	                                           &c.draws, &c.replayed, &c.dispatches, &c.cmd_buffers};
	for (int i = 0; i < 24; i++) {
		s.v[i] = fields[i]->load(std::memory_order_relaxed);
	}
	uint64_t d[24];
	for (int i = 0; i < 24; i++) {
		d[i] = s.v[i] - before.v[i];
	}
	const double ms = 1e-6 / elapsed; // ns -> ms per second
	std::fprintf(stderr,
	             "[floor] t=%.1f fps=%.1f worst=%.0fms gpu=%.0f proc=%.0f empty=%.0f blocked=%.0f cmd=%.0f "
	             "sync=%.0fk/s %.1fGB/s big=%.0f/s syncms=%.0f up=%.0f/s rb=%.0f/%.0f rbgpu=%.0f comp=%llu "
	             "compms=%.0f pwait=%.0f/%llu pto=%llu psync=%.0f pbuilt=%llu draws/f=%.0f replayed/f=%.0f dispatches/f=%.0f cb/f=%.0f\n",
	             std::chrono::duration<double>(now - run_start).count(), flips / elapsed, worst_ms,
	             d[0] * ms, d[1] * ms, d[2] * ms, d[3] * ms, d[4] * ms, d[5] / elapsed / 1000.0,
	             static_cast<double>(d[6]) / elapsed / 1e9, d[7] / elapsed, d[8] * ms, d[9] / elapsed,
	             d[10] * ms, d[11] / elapsed, d[12] * ms, static_cast<unsigned long long>(d[13]),
	             d[14] * ms, d[15] * ms, static_cast<unsigned long long>(d[16]),
	             static_cast<unsigned long long>(d[17]), d[18] * ms, static_cast<unsigned long long>(d[19]),
	             flips != 0 ? static_cast<double>(d[20]) / flips : 0.0, flips != 0 ? static_cast<double>(d[21]) / flips : 0.0,
	             flips != 0 ? static_cast<double>(d[22]) / flips : 0.0, flips != 0 ? static_cast<double>(d[23]) / flips : 0.0);
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
