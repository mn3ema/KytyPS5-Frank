#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILEPROGRESS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILEPROGRESS_H_

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>

namespace Libs::Graphics {

// Progress of the startup shader precompile (ProgramCache::StartPrecompile, the shader ledger
// replay, after a one-time import of the disk cache's keys), read by the window title and the
// system overlay ("Preparing shaders", then "Compiling shaders").
struct ShaderPrecompileProgress {
	std::atomic<bool>     active {false};
	std::atomic<uint32_t> done {0};     // recipes finished: found in the disk cache or compiled
	std::atomic<uint32_t> total {0};    // recipes in the ledger
	std::atomic<uint32_t> compiled {0}; // of `done`, how many had to be translated or compiled
	// Before the replay, the first time: the ledger is seeded from the disk cache's keys
	// (ShaderLedger::ImportDiskCache, plan 4.13); files read so far of the files found.
	std::atomic<bool>     importing {false};
	std::atomic<uint32_t> import_done {0};
	std::atomic<uint32_t> import_total {0};
	// Recipes in the shader ledger now (sources + permutations; kept current as the game records
	// new ones), and that number when this session's replay started: the title bar and the panel
	// show "Shaders N (+M new)" for the whole session.
	std::atomic<uint32_t> ledger_total {0};
	std::atomic<uint32_t> ledger_base {0};
	std::atomic<bool>     ledger_known {false};
	// The graphics pipeline recipes built after the replay (PipelineCache::StartPipelinePrecompile).
	std::atomic<bool>     pipelines_known {false};
	std::atomic<bool>     pipelines_active {false};
	std::atomic<bool>     pipelines_finished {false};
	std::atomic<uint32_t> pipelines_total {0};
	std::atomic<uint32_t> pipelines_done {0};
	std::atomic<uint32_t> pipelines_built {0};
	// Pipeline recipes first recorded this session (built during play, PipelineRecipe::Store::Record):
	// the panel and the title bar show "Pipelines N (+M new)" like the shaders.
	std::atomic<uint32_t> pipelines_new {0};
	// Step D4 (GPL): after the replay, the pipeline libraries of the permutations not warmed
	// before ("Preparing pipelines"); finished once checked, also when there was nothing to make.
	std::atomic<bool>     gpl_warm_active {false};
	std::atomic<bool>     gpl_warm_finished {false};
	std::atomic<uint32_t> gpl_warm_total {0};
	std::atomic<uint32_t> gpl_warm_done {0};
	// The GPU thread stopped waiting for the startup precompile (play started): zero-compile
	// accounting counts from here, and the warm-up drops to half its threads.
	std::atomic<bool>     startup_wait_done {false};
	// v438i (Mac 11-15): the player can skip the startup wait. skip_offered: the wait is on (the
	// panel shows "Esc: skip"); skip_state 0 none, 1 confirmation shown, 2 skip confirmed.
	std::atomic<bool>     skip_offered {false};
	std::atomic<int>      skip_state {0};
	std::atomic<bool>     skipped {false}; // the player skipped; the rest runs in the background
	// v438l: the game's GPU thread waiting on a translation or pipeline. Background preparation
	// (replay, warm-up) pauses while it is non-zero, so the game's own compile goes first.
	std::atomic<int>      foreground_waits {0};
	void YieldToForeground() const {
		for (int i = 0; i < 5000 && foreground_waits.load(std::memory_order_acquire) > 0; i++) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	// steady_clock ticks when the replay finished (0 while running or never run).
	std::atomic<int64_t> finished_at {0};

	[[nodiscard]] static int64_t Now() {
		return std::chrono::steady_clock::now().time_since_epoch().count();
	}
	// Seconds since the replay finished, or a negative number when it has not.
	[[nodiscard]] double SecondsSinceFinish() const {
		const auto at = finished_at.load(std::memory_order_acquire);
		if (at == 0) {
			return -1.0;
		}
		return std::chrono::duration<double>(std::chrono::steady_clock::duration(Now() - at)).count();
	}
};

inline ShaderPrecompileProgress& GetShaderPrecompileProgress() {
	static ShaderPrecompileProgress progress;
	return progress;
}

// v438l: marks the GPU thread waiting on a compile (background preparation yields meanwhile).
struct ForegroundWaitScope {
	bool active;
	explicit ForegroundWaitScope(bool on): active(on) {
		if (active) GetShaderPrecompileProgress().foreground_waits.fetch_add(1, std::memory_order_acq_rel);
	}
	~ForegroundWaitScope() {
		if (active) GetShaderPrecompileProgress().foreground_waits.fetch_sub(1, std::memory_order_acq_rel);
	}
	ForegroundWaitScope(const ForegroundWaitScope&)            = delete;
	ForegroundWaitScope& operator=(const ForegroundWaitScope&) = delete;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILEPROGRESS_H_
