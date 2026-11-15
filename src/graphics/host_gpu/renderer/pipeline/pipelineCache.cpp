#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStats.h"
#include "graphics/host_gpu/renderer/paintDraws.h"
#include "common/floorStats.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineRecipe.h"
#include "graphics/host_gpu/timeline.h"
#include "common/fpsLog.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "graphics/host_gpu/aftermath.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPipe.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderLedger.h"
#include "graphics/host_gpu/renderer/pipeline/shaderLedgerImport.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompileProgress.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderStats.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unordered_map>
#include <xxhash.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {
// KYTY_LOG_SKIPS=<n>: log the first n skipped draws/dispatches (resource materialization failed or
// gave up) with the guest flip, to see what a scene drops (stderr).
void LogSkip(const char* why, const char* stage, uint64_t hash, int line) {
	// KYTY_LOG_SKIPS=<n>[:<from guest flip>]
	static uint64_t       from_flip = 0;
	static const uint32_t budget    = [] {
		const char* value = std::getenv("KYTY_LOG_SKIPS");
		if (value == nullptr) {
			return 0u;
		}
		char*      end = nullptr;
		const auto n   = static_cast<uint32_t>(std::strtoul(value, &end, 10));
		if (end != nullptr && *end == ':') {
			from_flip = std::strtoull(end + 1, nullptr, 10);
		}
		return n;
	}();
	static std::atomic<uint32_t> used {0};
	if (budget == 0 || Libs::Graphics::Sync::GuestFlipCount() < from_flip ||
	    used.fetch_add(1, std::memory_order_relaxed) >= budget) {
		return;
	}
	std::fprintf(stderr, "[skip] flip=%llu %s %s %016llx line=%d\n",
	             static_cast<unsigned long long>(Libs::Graphics::Sync::GuestFlipCount()), why,
	             stage, static_cast<unsigned long long>(hash), line);
}
} // namespace

namespace Libs::Graphics {

// Shader translations longer than half a second are named ([slow-translate]): a title-screen stall
// (10-26) had one translator thread busy for minutes while the GPU thread waited.
static ShaderRecompiler::TranslateResult TimedTranslate(std::span<const uint32_t> code,
                                                       const ShaderRecompiler::CompileOptions& options) {
	const ForegroundWaitScope foreground_(Libs::Graphics::GuestGpu::IsGpuThread());
	struct Running {
		uint64_t                              hash;
		uint32_t                              stage;
		size_t                                words;
		std::chrono::steady_clock::time_point start;
	};
	static std::mutex                       running_mutex;
	static std::map<uint64_t, Running>      running; // key: a per-call ticket
	static std::atomic<uint64_t>            tickets {0};
	static std::once_flag                   monitor_once;
	std::call_once(monitor_once, [] {
		// Names translations still running after 10 s, every 10 s (the completion line may never
		// come when the run is killed).
		std::thread([] {
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(10));
				const auto      now = std::chrono::steady_clock::now();
				std::lock_guard lock(running_mutex);
				for (const auto& [ticket, r]: running) {
					const double s = std::chrono::duration<double>(now - r.start).count();
					if (s >= 10.0) {
						std::fprintf(stderr, "[slow-translate] %016llx stage=%u words=%zu still running after %.0f s\n",
						             static_cast<unsigned long long>(r.hash), r.stage, r.words, s);
					}
				}
			}
		}).detach();
	});
	const auto ticket = tickets.fetch_add(1, std::memory_order_relaxed);
	const auto start  = std::chrono::steady_clock::now();
	{
		std::lock_guard lock(running_mutex);
		running[ticket] = {options.shader_hash, static_cast<uint32_t>(options.stage), code.size(), start};
	}
	struct Done {
		uint64_t ticket;
		~Done() {
			std::lock_guard lock(running_mutex);
			running.erase(ticket);
		}
	};
	auto result = [&] {
		const Done done {ticket}; // also when a recoverable exit unwinds (ReplayTranslate)
		return ShaderRecompiler::TranslateProgram(code, options);
	}();
	const auto s      = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	if (s >= 0.5) {
		std::fprintf(stderr, "[slow-translate] %016llx stage=%u words=%zu %.1f s unsupported=%d\n",
		             static_cast<unsigned long long>(options.shader_hash),
		             static_cast<unsigned>(options.stage), code.size(), s, result.unsupported ? 1 : 0);
	}
	return result;
}

// The startup replay's translation of a listed source: a fatal error of the recompiler (a source
// recipe whose code does not decode, e.g. one cut short by the offline tool that listed it) refuses
// the shader instead of ending the first start, which a restart would then repeat (fr4, 11-14).
// `fatal` is set then: no refusal is written (the shader hash may be fine with the game's code).
static ShaderRecompiler::TranslateResult ReplayTranslate(std::span<const uint32_t> code,
                                                        const ShaderRecompiler::CompileOptions& options,
                                                        bool* fatal = nullptr) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
	if (Common::RecoverableExitsSupported()) {
		try {
			Common::RecoverableExitScope recoverable;
			return TimedTranslate(code, options);
		} catch (const Common::RecoverableExit& failure) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 16) {
				std::fprintf(stderr, "[precompile] source of %016" PRIx64 " refused: %s\n", options.shader_hash,
				             failure.text.c_str());
			}
			if (fatal != nullptr) {
				*fatal = true;
			}
			ShaderRecompiler::TranslateResult refused;
			refused.unsupported = true;
			return refused;
		}
	}
#endif
	return TimedTranslate(code, options);
}

// A shader-level refusal in the disk cache: written when a compute shader is skipped for the
// session ("refused with N inputs") or a precompile translation of it comes back unsupported. The
// startup precompile skips such shaders: their ledger sources (e6d76d24f59f8015 has four variants of
// 100k-130k words) take minutes each to translate only to be refused, and a run killed before
// they finish never records the refusal, so every boot stalled at the title behind them (10-26).
static std::vector<uint8_t> ShaderRefusalKey(uint64_t shader_hash) {
	static constexpr char tag[] = "kyty-shader-refused-v1";
	std::vector<uint8_t> key(tag, tag + sizeof(tag) - 1);
	for (int i = 0; i < 8; i++) {
		key.push_back(static_cast<uint8_t>(shader_hash >> (8 * i)));
	}
	return key;
}

bool ShaderFailureNonFatal() {
	static const bool value = true;
	return value;
}

namespace ShaderRecompiler::IR {
const char* MaterializePhase(); // ResourceMaterialization.cpp
} // namespace ShaderRecompiler::IR

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// No emulator revision: the driver keys every pipeline by its whole create info, SPIR-V
	// included, so a new build reuses the pipelines whose shaders did not change. Keying on the
	// revision threw the whole cache away at every rebuild and release, and the first start then
	// compiled everything behind a black screen.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

// The device configuration a pipeline is compiled under (enabled extensions and features that
// change what the driver compiles). The driver cache header names only the driver and GPU, and
// v437q showed a cache written under another configuration (the build's warm-up start without
// GPL) is neither served nor replaced: each configuration gets cache files of its own.
std::string DeviceConfigTag(const GraphicContext& g) {
	const uint32_t bits[] = {g.graphics_pipeline_library_enabled, g.pipeline_robustness_enabled, g.eds3_raster_enabled,
	                         g.mesh_shader_enabled, g.bindless_enabled, g.bindless_storage_enabled,
	                         g.compute_subgroup_size_control_enabled, g.subgroup_size_control_enabled,
	                         g.shader_image_int64_atomics_enabled, g.attachment_feedback_loop_enabled,
	                         g.provoking_vertex_last_enabled, g.sample_rate_shading_enabled,
	                         g.shader_device_clock_enabled, g.depth_color_copy_enabled};
	return fmt::format("{:08x}", static_cast<uint32_t>(XXH3_64bits(bits, sizeof(bits))));
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

// _ShaderCache/<title id>, next to _PipelineCache; empty without a title id.
std::filesystem::path ShaderDiskCacheDirectory() {
	const auto title_id = PipelineCacheTitleId();
	return title_id.empty() ? std::filesystem::path() : std::filesystem::path("_ShaderCache") / title_id;
}

// _ShaderList: the lists of what the game has used (the shader ledger, the pipeline recipes and
// their SPIR-V), apart from the caches, so clearing _ShaderCache or _PipelineCache keeps them and
// the next start builds everything known again before the game runs (Mac 11-02). Small next to
// the caches: recipes, not compiled results.
std::filesystem::path ShaderListDirectory() {
	return std::filesystem::path("_ShaderList");
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// The backing bytes of the guest pages one resource evaluation (MaterializeResources, a function
// expansion) reads, found once per page: every SRT dword read otherwise took the backing store's
// lock, two map lookups and the GPU-dirty range queries (a third of the GPU thread in the
// jungle). Only pages with no GPU-written byte are cached; within one synchronous evaluation on
// the GPU thread nothing marks them GPU-written. A page that is not clean stays on the exact path.
struct GuestPageReadCache {
	static constexpr size_t Capacity = 16;
	struct Entry {
		uint64_t       page    = 0;
		const uint8_t* backing = nullptr;
	};
	std::array<Entry, Capacity> entries {};
	size_t                      count = 0;
	size_t                      next  = 0;

	// The page's backing bytes when it is GPU-clean, else null (SrtRuntime::map_clean_page).
	const uint8_t* Map(uint64_t page) {
		for (size_t i = 0; i < count; i++) {
			if (entries[i].page == page) {
				return entries[i].backing;
			}
		}
		auto& slot = count < Capacity ? entries[count++] : entries[next++ % Capacity];
		slot       = {.page    = page,
		              .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page, TRACKER_PAGE_SIZE)};
		return slot.backing;
	}

	// True when the page is GPU-clean and the bytes were copied from its backing.
	bool Read(uint64_t address, std::span<uint32_t> values) {
		const auto page = Common::AlignDown(address, TRACKER_PAGE_SIZE);
		if (Common::AlignDown(address + values.size_bytes() - 1, TRACKER_PAGE_SIZE) != page) {
			return false;
		}
		const Entry* entry = nullptr;
		for (size_t i = 0; i < count; i++) {
			if (entries[i].page == page) {
				entry = &entries[i];
				break;
			}
		}
		if (entry == nullptr) {
			auto& slot = count < Capacity ? entries[count++] : entries[next++ % Capacity];
			slot       = {.page    = page,
			              .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page,
			                                                                     TRACKER_PAGE_SIZE)};
			entry      = &slot;
		}
		if (entry->backing == nullptr) {
			return false;
		}
		std::memcpy(values.data(), entry->backing + (address - page), values.size_bytes());
		return true;
	}
};

bool UsesShaderClock(const ShaderRecompiler::IR::Program& program) {
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ShaderRecompiler::IR::ValueOpcode::ReadClockRealtime64) {
				return true;
			}
		}
	}
	return false;
}

// Diagnostics: the shader whose resources are being evaluated, and how often its SRT reads
// hit memory the GPU owns (each such read drains the GPU).
thread_local uint64_t t_srt_shader_hash = 0;

// KYTY_SRT_VERIFY (ResourceSnapshot.h, SrtVerify): where the refresh running on this thread
// records the values it used in place of GPU-written bytes (null: read them exactly).
thread_local std::vector<ShaderRecompiler::IR::SrtVerify>* t_srt_verify = nullptr;

// A shader's values are used in place of GPU-written bytes only once the host has seen them equal
// what it then read exactly in SrtVerifyQualifyWalks refreshes in a row that read such bytes (the
// GPU thread reads both while it still waits): a value the GPU produces (a count, an offset it
// writes) never qualifies, so no dispatch is lost to a difference the shader would report. A
// shader that reported one anyway, or failed SrtVerifyGiveUp times, reads exactly from then on.
constexpr uint32_t SrtVerifyQualifyWalks = 32;
constexpr uint32_t SrtVerifyGiveUp       = 16;
struct SrtVerifyState {
	uint32_t clean    = 0; // refreshes in a row whose host values all matched
	uint32_t failures = 0;
	bool     qualified = false;
	bool     blocked   = false;
};
std::shared_mutex                              g_srt_verify_mutex;
std::unordered_map<uint64_t, SrtVerifyState>   g_srt_verify_states;

bool SrtVerifyQualified(uint64_t shader_hash) {
	std::shared_lock lock(g_srt_verify_mutex);
	const auto       found = g_srt_verify_states.find(shader_hash);
	return found != g_srt_verify_states.end() && found->second.qualified && !found->second.blocked;
}

bool SrtVerifyCandidate(uint64_t shader_hash) {
	std::shared_lock lock(g_srt_verify_mutex);
	const auto       found = g_srt_verify_states.find(shader_hash);
	return found == g_srt_verify_states.end() || !found->second.blocked;
}

bool SrtVerifyPlan(const ShaderRecompiler::IR::ResourcePlan& plan, ShaderType stage) {
	return ShaderRecompiler::IR::SrtVerifyEnabled() && stage == ShaderType::Compute &&
	       plan.stage == ShaderType::Compute && !plan.srt_reads.empty();
}

// The refresh of this program uses (and records) host values in place of GPU-written bytes.
bool SrtVerifyProgram(const ShaderRecompiler::IR::ResourcePlan& plan, ShaderType stage,
                      uint64_t shader_hash) {
	return SrtVerifyPlan(plan, stage) && SrtVerifyQualified(shader_hash);
}

// Qualifying refresh on the GPU thread: GPU-written bytes are read exactly, and compared with
// what the host held before.
thread_local bool t_srt_qualify      = false;
thread_local bool t_srt_qualify_read = false; // a GPU-written read happened
thread_local bool t_srt_qualify_diff = false; // and differed

void SrtVerifyNoteWalk(uint64_t shader_hash) {
	if (!t_srt_qualify_read) {
		return;
	}
	std::unique_lock lock(g_srt_verify_mutex);
	auto&            state = g_srt_verify_states[shader_hash];
	if (state.blocked || state.qualified) {
		return;
	}
	if (t_srt_qualify_diff) {
		state.clean = 0;
		if (++state.failures >= SrtVerifyGiveUp) {
			state.blocked = true;
		}
		return;
	}
	if (++state.clean >= SrtVerifyQualifyWalks) {
		state.qualified = true;
		std::fprintf(stderr, "[srt-verify] CS %016llx: host values matched %u refreshes, verified "
		                     "on the GPU from now on\n",
		             static_cast<unsigned long long>(shader_hash), state.clean);
	}
}

// Around an exact read during a qualifying refresh: the host's bytes before it.
struct SrtQualifyRead {
	std::array<uint32_t, 64> stale {};
	bool                     active = false;
	SrtQualifyRead(uint64_t address, std::span<const uint32_t> values) {
		active = t_srt_qualify && !values.empty() && values.size() <= stale.size() &&
		         !ShaderRecompiler::IR::SrtEvaluatingCondition() &&
		         Libs::LibKernel::Memory::IsGpuBufferWritten(address, values.size_bytes()) &&
		         Libs::LibKernel::Memory::TryReadBacking(address, stale.data(), values.size_bytes());
	}
	void Finish(bool ok, std::span<const uint32_t> values) const {
		if (!active || !ok) {
			return;
		}
		t_srt_qualify_read = true;
		if (std::memcmp(stale.data(), values.data(), values.size_bytes()) != 0) {
			t_srt_qualify_diff = true;
		}
	}
};

// A read of GPU-written bytes during a verifying refresh: the bytes the host holds, recorded.
// False when the bytes are not GPU-written (the ordinary read serves them) or cannot be recorded.
bool SrtVerifyRead(uint64_t address, std::span<uint32_t> values) {
	auto* verify = t_srt_verify;
	// A read that may be refused (a condition, a GPU-sized record count: SrtConditionReadRefused)
	// stays refused: what refusing it means (both branches, a conservative count) needs no value.
	if (verify == nullptr || values.empty() || ShaderRecompiler::IR::SrtEvaluatingCondition() ||
	    !Libs::LibKernel::Memory::IsGpuBufferWritten(address, values.size_bytes())) {
		return false;
	}
	if ((address & 3u) != 0u ||
	    verify->size() + values.size() > ShaderRecompiler::IR::SrtVerifyMaxEntries) {
		return false;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	for (size_t i = 0; i < values.size(); i++) {
		const uint64_t at    = address + i * sizeof(uint32_t);
		bool           known = false;
		for (const auto& entry: *verify) {
			if (entry.address == at) {
				known = true;
				break;
			}
		}
		if (!known) {
			verify->push_back({.address = at, .expected = values[i]});
		}
	}
	return true;
}
void CountGpuOwnedSrtRead(uint64_t address, uint32_t bytes) {
	struct Entry {
		uint64_t count   = 0;
		uint64_t address = 0;
		uint32_t bytes   = 0;
	};
	static std::unordered_map<uint64_t, Entry> counts;
	static uint64_t                            total = 0;
	auto& entry   = counts[t_srt_shader_hash];
	entry.count++;
	entry.address = address;
	entry.bytes   = bytes;
	if (++total % 4096 == 0) {
		LOGF("GpuOwnedSrtReads: total=%" PRIu64 " shaders=%zu\n", total, counts.size());
		for (const auto& [hash, e]: counts) {
			LOGF("\t hash=0x%016" PRIx64 " count=%" PRIu64 " last=0x%016" PRIx64 " bytes=%u\n",
			     hash, e.count, e.address, e.bytes);
		}
	}
}

// KYTY_SRT_SPECULATE: SRT reads of bytes the GPU wrote make the GPU thread wait for the GPU
// (a download). In gameplay three compute shaders cause almost all of those waits (Tracy,
// 2026-10-09: 5.5 s of every 20). Their values are likely the same frame to frame, so:
//   1  diagnostic: read the stale backing first, then the exact value, and count differences.
//   2  speculate: serve the stale backing (no wait); every KYTY_SRT_SPECULATE_CHECK-th read of
//      a shader (default 16) is read exactly and compared, and a shader whose stale value ever
//      differs is read exactly from then on.
// A report per shader goes to stderr every 4096 such reads.
static int SrtSpeculateMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_SRT_SPECULATE");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	return mode;
}

static uint32_t SrtSpeculateCheckInterval() {
	static const uint32_t interval = [] {
		const char* value = std::getenv("KYTY_SRT_SPECULATE_CHECK");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 16ul;
		return static_cast<uint32_t>(n == 0 ? 1 : n);
	}();
	return interval;
}

struct SrtSpeculateStats {
	uint64_t reads      = 0;
	uint64_t served     = 0;
	uint64_t checks     = 0;
	uint64_t mismatches = 0;
	bool     exact_only = false;
	uint64_t last_addr  = 0;
};

// A control-flow condition that would read bytes the GPU wrote fails the read instead of waiting
// (SrtConditionsNoWait): the walker treats the condition as not known and walks both successors.
bool SrtConditionReadRefused(uint64_t address, std::span<const uint32_t> values) {
	if (!ShaderRecompiler::IR::SrtEvaluatingCondition() ||
	    !ShaderRecompiler::IR::SrtConditionsNoWait() ||
	    !Libs::Graphics::GuestGpu::IsGpuThread() ||
	    !Libs::LibKernel::Memory::IsGpuBufferWritten(address, values.size_bytes())) {
		return false;
	}
	ShaderRecompiler::IR::SrtNoteConditionRefusal();
	static uint64_t refused = 0;
	if ((++refused & (refused - 1)) == 0) {
		std::fprintf(stderr, "[srt-condition] %llu condition reads of GPU-written bytes left to the GPU "
		                     "(last: shader %016llx addr=0x%llx)\n",
		             static_cast<unsigned long long>(refused),
		             static_cast<unsigned long long>(t_srt_shader_hash),
		             static_cast<unsigned long long>(address));
	}
	return true;
}

// KYTY_SRT_DEFER=1: a dispatch's SRT refresh that would read bytes the GPU wrote does not wait
// for the GPU. Each such read starts its download (submitted, not waited for) and fails; the
// refresh is discarded and the command processor suspends the dispatch packet until the
// download's tick completes, running the other guest queues meanwhile (the console runs them in
// parallel). Before, the GPU thread sat idle in two such waits a frame, ~40 % of its time in
// gameplay (2ea3957af2daf690 on the graphics queue, fb5ee753d43f3f0b on an async queue).
// KYTY_SRT_DEFER_TOP=1: which shaders' SRT refreshes defer (read GPU-written bytes), by phase,
// read size and condition reads, with the last address; the top 10 every 5 s ([srt-defer-top]).
void NoteSrtDeferRead(uint64_t address, uint64_t bytes) {
	static const bool enabled = std::getenv("KYTY_SRT_DEFER_TOP") != nullptr;
	if (!enabled) {
		return;
	}
	struct Stat {
		uint64_t    reads = 0, bytes = 0, conditions = 0;
		uint64_t    last  = 0;
		std::string phases;
	};
	static std::mutex                           mutex;
	static std::unordered_map<uint64_t, Stat>   stats;
	static auto                                 last = std::chrono::steady_clock::now();
	std::lock_guard                             lock(mutex);
	auto&                                       st = stats[t_srt_shader_hash];
	st.reads++;
	st.bytes += bytes;
	st.last = address;
	st.conditions += ShaderRecompiler::IR::SrtEvaluatingCondition() ? 1u : 0u;
	const std::string phase = ShaderRecompiler::IR::MaterializePhase();
	if (st.phases.find(phase) == std::string::npos && st.phases.size() < 64) {
		st.phases += phase + "/" + std::to_string(bytes) + " ";
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - last).count();
	last                 = now;
	std::vector<std::pair<uint64_t, Stat>> top(stats.begin(), stats.end());
	std::sort(top.begin(), top.end(),
	          [](const auto& a, const auto& b) { return a.second.reads > b.second.reads; });
	std::string text = fmt::format("[srt-defer-top] {} shaders\n", top.size());
	for (size_t i = 0; i < top.size() && i < 10; i++) {
		const auto& [hash, e] = top[i];
		text += fmt::format("[srt-defer-top]   {:016x} {:.1f} reads/s avg {} B cond={} last=0x{:x} phases: {}\n",
		                    hash, static_cast<double>(e.reads) / seconds, e.bytes / e.reads,
		                    e.conditions, e.last, e.phases);
	}
	std::fputs(text.c_str(), stderr);
	stats.clear();
}

thread_local bool     t_srt_defer_allowed = false;
thread_local uint64_t t_srt_defer_tick    = 0;
thread_local uint32_t t_srt_defer_reads   = 0;

bool SrtDeferRead(uint64_t address, std::span<const uint32_t> values) {
	if (!t_srt_defer_allowed || !Libs::Graphics::GuestGpu::IsGpuThread() ||
	    !Libs::LibKernel::Memory::IsGpuBufferWritten(address, values.size_bytes())) {
		// Bytes already downloading (not GPU-written any more) are left to the ordinary path,
		// unless this refresh is deferred anyway: then their tick is waited for too.
		if (!t_srt_defer_allowed || t_srt_defer_tick == 0) {
			return false;
		}
	}
	const auto tick =
	    Libs::LibKernel::Memory::StartGpuBufferReadback(address, values.size_bytes());
	if (tick == 0) {
		return false;
	}
	t_srt_defer_tick = std::max(t_srt_defer_tick, tick);
	t_srt_defer_reads++;
	NoteSrtDeferRead(address, values.size_bytes());
	return true;
}

} // namespace
namespace ShaderRecompiler::IR {
const char* MaterializePhase(); // ResourceMaterialization.cpp
} // namespace ShaderRecompiler::IR
namespace {
// KYTY_LOG_GPU_OWNED_SRT=1: the SRT reads that make the GPU thread download GPU-written bytes
// (wait for the GPU), first three per shader and read size.
void LogGpuOwnedSrtRead(const char* tag, uint64_t address, size_t dwords) {
	static const bool enabled = std::getenv("KYTY_LOG_GPU_OWNED_SRT") != nullptr;
	if (!enabled || !Libs::LibKernel::Memory::IsGpuBufferWritten(address, dwords * 4u)) {
		return;
	}
	static std::unordered_map<std::string, uint32_t> seen;
	const char* phase = ShaderRecompiler::IR::MaterializePhase();
	auto& n = seen[fmt::format("{:016x}{}{}", t_srt_shader_hash, dwords, phase)];
	if (n++ < 2) {
		std::fprintf(stderr,
		             "[gpu-owned-srt] %s shader %016llx addr=0x%llx dwords=%zu cond=%d phase=%s\n",
		             tag, static_cast<unsigned long long>(t_srt_shader_hash),
		             static_cast<unsigned long long>(address), dwords,
		             ShaderRecompiler::IR::SrtEvaluatingCondition() ? 1 : 0, phase);
	}
	static uint64_t total = 0;
	static std::unordered_map<std::string, uint64_t> by_phase;
	by_phase[phase]++;
	if (++total % 2000 == 0) {
		std::string text;
		for (const auto& [name, count]: by_phase) text += fmt::format(" {}={}", name, count);
		std::fprintf(stderr, "[gpu-owned-srt] totals:%s\n", text.c_str());
	}
}

// KYTY_SRT_WAIT_STATS=1: time the SRT reads that download GPU-written bytes (the GPU thread waits
// for the GPU), per shader, and print the top shaders by wait time every 10 s of wall time.
bool TimedReadBufferBacking(uint64_t address, uint32_t* data, size_t bytes) {
	static const bool enabled = std::getenv("KYTY_SRT_WAIT_STATS") != nullptr;
	if (!enabled || !Libs::LibKernel::Memory::IsGpuBufferWritten(address, bytes)) {
		return Libs::LibKernel::Memory::TryReadBufferBacking(address, data, bytes);
	}
	const auto start = std::chrono::steady_clock::now();
	const bool ok    = Libs::LibKernel::Memory::TryReadBufferBacking(address, data, bytes);
	const auto us    = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
	struct Stat {
		uint64_t us = 0, count = 0;
		const char* phase = "";
	};
	static std::mutex                          mutex;
	static std::unordered_map<uint64_t, Stat>  stats;
	static auto                                last  = std::chrono::steady_clock::now();
	static uint64_t                            total = 0;
	std::lock_guard lock(mutex);
	auto& s = stats[t_srt_shader_hash];
	s.us += static_cast<uint64_t>(us);
	s.count++;
	s.phase = ShaderRecompiler::IR::MaterializePhase();
	total += static_cast<uint64_t>(us);
	if (std::chrono::steady_clock::now() - last > std::chrono::seconds(10)) {
		last = std::chrono::steady_clock::now();
		std::vector<std::pair<uint64_t, Stat>> sorted(stats.begin(), stats.end());
		std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
		std::fprintf(stderr, "[srt-wait] total=%llu ms shaders=%zu\n",
		             static_cast<unsigned long long>(total / 1000), sorted.size());
		for (size_t i = 0; i < sorted.size() && i < 12; i++) {
			std::fprintf(stderr, "[srt-wait]   %016llx %llu ms n=%llu phase=%s\n",
			             static_cast<unsigned long long>(sorted[i].first),
			             static_cast<unsigned long long>(sorted[i].second.us / 1000),
			             static_cast<unsigned long long>(sorted[i].second.count),
			             sorted[i].second.phase);
		}
		stats.clear();
		total = 0;
	}
	return ok;
}

// GPU thread only. True when `values` was served (stale, or exact after a check); false when the
// caller should take its ordinary path.
bool SrtSpeculativeRead(uint64_t address, std::span<uint32_t> values) {
	const int mode = SrtSpeculateMode();
	if (mode == 0 || t_srt_shader_hash == 0 || !Libs::Graphics::GuestGpu::IsGpuThread() ||
	    !Libs::LibKernel::Memory::IsGpuBufferWritten(address, values.size_bytes())) {
		return false;
	}
	static std::unordered_map<uint64_t, SrtSpeculateStats> stats;
	static uint64_t                                        total = 0;
	auto&                                                  s     = stats[t_srt_shader_hash];
	s.reads++;
	s.last_addr = address;
	if (mode == 1 && s.reads <= 6) {
		// Which bytes: the guest address, the size, and the first dwords before and after the
		// exact read (printed below once the exact value is known).
		std::fprintf(stderr, "[srt-speculate] read %016llx #%llu addr=0x%llx dwords=%zu\n",
		             static_cast<unsigned long long>(t_srt_shader_hash),
		             static_cast<unsigned long long>(s.reads),
		             static_cast<unsigned long long>(address), values.size());
	}
	if (++total % 4096 == 0) {
		std::fprintf(stderr, "[srt-speculate] mode=%d reads=%llu shaders=%zu\n", mode,
		             static_cast<unsigned long long>(total), stats.size());
		for (const auto& [hash, e]: stats) {
			std::fprintf(stderr,
			             "[srt-speculate]   CS/VS %016llx reads=%llu served=%llu checks=%llu "
			             "mismatches=%llu exact_only=%d last=0x%llx\n",
			             static_cast<unsigned long long>(hash),
			             static_cast<unsigned long long>(e.reads),
			             static_cast<unsigned long long>(e.served),
			             static_cast<unsigned long long>(e.checks),
			             static_cast<unsigned long long>(e.mismatches), e.exact_only ? 1 : 0,
			             static_cast<unsigned long long>(e.last_addr));
		}
	}
	std::array<uint32_t, 64> stale_small {};
	std::vector<uint32_t>    stale_large;
	uint32_t*                stale = stale_small.data();
	if (values.size() > stale_small.size()) {
		stale_large.resize(values.size());
		stale = stale_large.data();
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, stale, values.size_bytes())) {
		return false;
	}
	const bool check = mode == 1 || s.exact_only || (s.reads % SrtSpeculateCheckInterval()) == 1u;
	if (!check) {
		std::memcpy(values.data(), stale, values.size_bytes());
		s.served++;
		return true;
	}
	// Exact read: the ordinary path downloads the GPU-written bytes into the backing store.
	if (!Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(),
	                                                     values.size_bytes())) {
		return false;
	}
	if (mode == 1 && s.reads <= 6) {
		std::string text;
		for (size_t i = 0; i < values.size() && i < 8; i++) {
			char word[32];
			std::snprintf(word, sizeof(word), " %08x/%08x", stale[i], values[i]);
			text += word;
		}
		std::fprintf(stderr, "[srt-speculate]   stale/exact:%s\n", text.c_str());
	}
	if (!s.exact_only) {
		s.checks++;
		if (std::memcmp(values.data(), stale, values.size_bytes()) != 0) {
			s.mismatches++;
			if (mode == 2 && s.reads > 1) {
				s.exact_only = true;
				std::fprintf(stderr,
				             "[srt-speculate] shader %016llx: stale value differed at 0x%llx, "
				             "exact reads from now on\n",
				             static_cast<unsigned long long>(t_srt_shader_hash),
				             static_cast<unsigned long long>(address));
			}
		}
	}
	return true;
}

// Ordinary (raw) SRT reads: only memory the guest has committed, read through the guest
// mapping so a GPU-owned page is refreshed first. Without a reader the walker dereferenced
// whatever address a descriptor chain produced, including 0 on a path the shader never takes.
bool ReadShaderGuestMemoryRawImpl(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	if (SrtVerifyRead(address, values)) {
		return true;
	}
	if (SrtConditionReadRefused(address, values)) {
		return false;
	}
	if (SrtDeferRead(address, values)) {
		return false;
	}
	if (SrtSpeculativeRead(address, values)) {
		return true;
	}
	// Bytes the GPU has not written are current in the backing store. Reading them there skips
	// the tracked-page fault, which drains the GPU to refresh whatever else on the page the GPU
	// wrote: constants that share a page with GPU-written arguments cost a drain per dispatch.
	LogGpuOwnedSrtRead("raw", address, values.size());
	if (TimedReadBufferBacking(address, values.data(), values.size_bytes())) {
		return true;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	if (t_srt_shader_hash != 0 && Libs::Graphics::GuestGpu::IsGpuThread()) {
		CountGpuOwnedSrtRead(address, static_cast<uint32_t>(values.size_bytes()));
		if (static const bool log_owned = std::getenv("KYTY_LOG_GPU_OWNED_SRT") != nullptr;
		    log_owned) {
			static std::unordered_map<uint64_t, uint32_t> seen;
			auto& n = seen[t_srt_shader_hash ^ (static_cast<uint64_t>(values.size()) << 56u)];
			if (n++ < 3) {
				std::fprintf(stderr, "[gpu-owned-srt] raw shader %016llx addr=0x%llx dwords=%zu cond=%d\n",
				             static_cast<unsigned long long>(t_srt_shader_hash),
				             static_cast<unsigned long long>(address), values.size(),
				             ShaderRecompiler::IR::SrtEvaluatingCondition() ? 1 : 0);
			}
		}
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// KYTY_SRT_GPU_FILL=0: read every flat SRT slot on the host, waiting for the GPU when it wrote
// the bytes, instead of leaving shader-only slots to the GPU (SrtGpuFill).
bool SrtGpuFillEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_GPU_FILL");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool SrtBytesGpuWritten(uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::IsGpuBufferWritten(address, size);
}

bool ReadShaderGuestMemoryImpl(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	if (SrtVerifyRead(address, values)) {
		return true;
	}
	if (SrtConditionReadRefused(address, values)) {
		return false;
	}
	if (SrtDeferRead(address, values)) {
		return false;
	}
	if (SrtSpeculativeRead(address, values)) {
		return true;
	}
	LogGpuOwnedSrtRead("exact", address, values.size());
	if (TimedReadBufferBacking(address, values.data(), values.size_bytes())) {
		return true;
	}
	// The bytes are mapped but GPU-owned. Reading them through the guest mapping takes the
	// tracked-page fault, which drains the GPU and refreshes the page, so the value read is
	// the one the shader would see. Before, this only ever succeeded because the aggressive
	// garbage collector happened to have downloaded the range first.
	if (static const bool log_owned = std::getenv("KYTY_LOG_GPU_OWNED_SRT") != nullptr;
	    log_owned && Libs::Graphics::GuestGpu::IsGpuThread()) {
		static std::unordered_map<uint64_t, uint32_t> seen;
		auto& n = seen[t_srt_shader_hash ^ (static_cast<uint64_t>(values.size()) << 56u)];
		if (n++ < 3) {
			std::fprintf(stderr, "[gpu-owned-srt] shader %016llx addr=0x%llx dwords=%zu cond=%d\n",
			             static_cast<unsigned long long>(t_srt_shader_hash),
			             static_cast<unsigned long long>(address), values.size(),
			             ShaderRecompiler::IR::SrtEvaluatingCondition() ? 1 : 0);
		}
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

bool ReadShaderGuestMemoryProbe(void* userdata, uint64_t address, std::span<uint32_t> values);

bool ReadShaderGuestMemoryRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
	DrawPipe::DrainIfPendingWrite(address, values.size_bytes());
	// The async-compute thread off the renderer mutex (KYTY_ASYNC_THREAD): bytes no GPU work
	// wrote are served as the exact path would serve them; anything else takes the renderer.
	if (RenderContext::RendererReleased() && ReadShaderGuestMemoryProbe(userdata, address, values)) {
		return true;
	}
	const RenderContext::ReacquireScope reacquire;
	const SrtQualifyRead qualify(address, values);
	const bool           ok = ReadShaderGuestMemoryRawImpl(userdata, address, values);
	qualify.Finish(ok, values);
	return ok;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	DrawPipe::DrainIfPendingWrite(address, values.size_bytes());
	if (RenderContext::RendererReleased() && ReadShaderGuestMemoryProbe(userdata, address, values)) {
		return true;
	}
	const RenderContext::ReacquireScope reacquire;
	const SrtQualifyRead qualify(address, values);
	const bool           ok = ReadShaderGuestMemoryImpl(userdata, address, values);
	qualify.Finish(ok, values);
	return ok;
}

// The reader of a stage walked in the parallel window (StagePrepWorker): bytes no GPU work has
// written, from the backing store, and nothing else: no download, no fault, no wait for the GPU.
// A read it cannot serve fails the walk, which the GPU thread then repeats on the exact path.
bool ReadShaderGuestMemoryProbe(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	return Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(),
	                                                       values.size_bytes());
}

// "VS", "PS", ... for zone texts.
const char* StageLabel(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "VS";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "DS";
		case ShaderType::Pixel: return "PS";
		case ShaderType::Compute: return "CS";
		default: return "??";
	}
}

// --skip-shaders and KYTY_SKIP_SHADER_HASHES="hash,hash,...": skip the draws and dispatches of
// these guest shaders, the same way a shader that fails to compile is skipped. To see what one
// shader contributes, to step past one that loses the device, or to leave out work nothing can
// use (the ray-tracing BVH updates whose only consumer gives up).
bool SkipShaderRequested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const auto parse = [&result](std::string_view list) {
			size_t start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(
					    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		};
		parse(Config::GetSkipShaderHashes());
		if (const char* value = std::getenv("KYTY_SKIP_SHADER_HASHES"); value != nullptr) {
			parse(value);
		}
		for (const auto hash: result) {
			LOGF("ProgramCache: skipping the draws and dispatches of shader 0x%016" PRIx64 "\n",
			     hash);
		}
		return result;
	}();
	if (std::ranges::find(hashes, shader_hash) != hashes.end()) {
		return true;
	}
	// Diagnostics: KYTY_SKIP_SHADER_FILE=<path>: a list of hashes (hex, any separators) read
	// again whenever the file changes (checked every 200 ms), to skip shaders while the game
	// runs: one game session can bisect which shader draws something.
	static const char* skip_file = std::getenv("KYTY_SKIP_SHADER_FILE");
	if (skip_file == nullptr) {
		return false;
	}
	static std::mutex                                   mutex;
	static std::shared_ptr<const std::vector<uint64_t>> dynamic =
	    std::make_shared<const std::vector<uint64_t>>();
	static std::chrono::steady_clock::time_point        next_check {};
	static std::filesystem::file_time_type              seen {};
	std::shared_ptr<const std::vector<uint64_t>>        current;
	{
		std::lock_guard lock(mutex);
		const auto      now = std::chrono::steady_clock::now();
		if (now >= next_check) {
			next_check = now + std::chrono::milliseconds(200);
			std::error_code error;
			const auto      stamp = std::filesystem::last_write_time(skip_file, error);
			if (error) {
				if (!dynamic->empty()) {
					dynamic = std::make_shared<const std::vector<uint64_t>>();
					LOGF("ProgramCache: skip file gone, skipping nothing\n");
				}
			} else if (stamp != seen) {
				seen = stamp;
				std::vector<uint64_t> list;
				if (FILE* file = std::fopen(skip_file, "rb"); file != nullptr) {
					char word[64];
					while (std::fscanf(file, "%63s", word) == 1) {
						const auto value = std::strtoull(word, nullptr, 16);
						if (value != 0) {
							list.push_back(value);
						}
					}
					std::fclose(file);
				}
				std::sort(list.begin(), list.end());
				std::fprintf(stderr, "[skip-file] %zu shaders skipped from now on\n", list.size());
				dynamic = std::make_shared<const std::vector<uint64_t>>(std::move(list));
			}
		}
		current = dynamic;
	}
	return std::binary_search(current->begin(), current->end(), shader_hash);
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	// KYTY_DUMP_SPIRV_HASHES="hash,hash,...": dump these shaders' SPIR-V without the full
	// graphics debug dump.
	static const std::string requested = [] {
		const char* value = std::getenv("KYTY_DUMP_SPIRV_HASHES");
		return value != nullptr ? std::string(value) : std::string();
	}();
	if (!Config::GraphicsDebugDumpEnabled() &&
	    (requested.empty() ||
	     requested.find(fmt::format("{:016x}", shader_hash)) == std::string::npos)) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

bool SlowGraphicsPipelineListed(uint64_t vs_hash, uint64_t ps_hash); // shaders.cpp

// A/B switch for background pipeline compiles (KYTY_ASYNC_PIPELINES=0 compiles every graphics
// pipeline on the GPU thread, as before) and how long a draw waits for a new one
// (KYTY_PIPELINE_WAIT_MS, default 20) before it is skipped.
static const bool g_async_pipelines = [] {
	const char* value = std::getenv("KYTY_ASYNC_PIPELINES");
	return value == nullptr || value[0] != '0';
}();
// Pipelines whose stages are this small (SPIR-V words, all stages; KYTY_PIPELINE_SYNC_WORDS)
// compile on the GPU thread: a few milliseconds, and their draws are never skipped. A skipped
// draw is lost for good when the game draws it once into a cached layer: the title's UI drew its
// WOLVERINE logo once, during the cold-start compile backlog, and the logo stayed missing.
static const uint64_t g_pipeline_sync_words = [] {
	const char* value = std::getenv("KYTY_PIPELINE_SYNC_WORDS");
	return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {12000};
}();
static const bool g_dynamic_raster_state = [] {
	const char* value = std::getenv("KYTY_DYNAMIC_RASTER_STATE");
	return value == nullptr || value[0] != '0';
}();

bool PipelineDynamicRasterStateEnabled() {
	return g_dynamic_raster_state;
}

// Step D2 (EDS3): the draw's values of the pre-rasterization fields a linked pipeline takes as
// dynamic state, from the same registers and rules as the pipeline key (BuildGraphicsKey).
GraphicsDynamicRaster DrawDynamicRaster(const HW::Context& ctx, bool rect_list) {
	const auto&           clip = ctx.GetClipControl();
	const auto&           mc   = ctx.GetModeControl();
	GraphicsDynamicRaster out;
	out.negative_one_to_one = !clip.dx_clip_space;
	out.depth_clip_enable   = clip.IsZClipEnabled();
	out.polygon_mode        = ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	return out;
}

namespace {

// Zeroes key fields that cannot change the pipeline PrepareGraphicsPipeline creates, so draws
// that differ only in them share one pipeline instead of compiling identical copies (idea from
// Jetsku/KytyPS5 ef5e8fd9).
// - Cull mode and front face are dynamic state, recorded per draw from the same registers with
//   the same rect-list rule (SetGraphicsDynamicParams). The polygon mode stays static; it is
//   resolved from the real cull bits before they leave the key.
// - The depth-bounds test enable and bounds are dynamic state as well, so each DB_DEPTH_BOUNDS
//   value no longer creates its own pipeline (before, even with the test off). On macOS the
//   pipeline always disables the test (MoltenVK has no depthBounds), so they never mattered.
// Blend factors and ops of an attachment with blending off, and the alpha ones without separate
// alpha blending, are never written into the key (TryGetGraphicsPipeline), so they need no rule.
// KYTY_DYNAMIC_RASTER_STATE=0 keeps all of these in the key and bakes them into the pipeline.
void NormalizeGraphicsPipelineKey(PipelineStaticParameters& params) {
	if (!g_dynamic_raster_state) {
		return;
	}
	params.cull_front               = false;
	params.cull_back                = false;
	params.face                     = false;
	params.depth_bounds_test_enable = false;
	params.depth_min_bounds         = 0.0f;
	params.depth_max_bounds         = 0.0f;
}

} // namespace

// KYTY_MULTI_DRAW_REPLAY: the user-data words each graphics resource plan reads, by the snapshot
// its stages bind (SourceEntry::resources, which never moves: entries are never erased).
namespace {
struct PlanReads {
	uint64_t mask           = ~0ull;
	uint32_t user_data_base = 0;
};
std::mutex                                                               g_plan_reads_mutex;
std::unordered_map<const ShaderRecompiler::IR::ResourceSnapshot*, PlanReads> g_plan_reads;

// Every user-data register an instruction of the plan reads (its walk, native code, traces and
// specialization all evaluate value_storage); all bits when one cannot be named.
void NotePlanUserDataReads(const ShaderRecompiler::IR::ResourceSnapshot* resources,
                           const ShaderRecompiler::IR::ResourcePlan&     plan) {
	PlanReads reads {0, plan.user_data_base};
	for (const auto& inst: plan.value_storage) {
		if (inst.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) {
			continue;
		}
		if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != ShaderRecompiler::IR::Type::ScalarReg) {
			reads.mask = ~0ull;
			break;
		}
		const auto reg = ShaderRecompiler::IR::RegIndex(inst.Arg(0).ScalarRegister());
		if (reg < plan.user_data_base || reg - plan.user_data_base >= 64u) {
			reads.mask = ~0ull;
			break;
		}
		reads.mask |= 1ull << (reg - plan.user_data_base);
	}
	std::lock_guard lock(g_plan_reads_mutex);
	g_plan_reads[resources] = reads;
}
} // namespace

bool PipelineCache::PlanUserDataReads(const ShaderRecompiler::IR::ResourceSnapshot* resources,
                                      uint64_t& mask, uint32_t& user_data_base) {
	std::lock_guard lock(g_plan_reads_mutex);
	const auto      found = g_plan_reads.find(resources);
	if (found == g_plan_reads.end()) {
		return false;
	}
	mask           = found->second.mask;
	user_data_base = found->second.user_data_base;
	return true;
}

// Draws without depth or stencil whose vertex shaders total at most KYTY_PIPELINE_SYNC_UI_VS_WORDS
// SPIR-V words (default 8000; 0 = off) also compile on the GPU thread, whatever the pixel
// shader's size: those are UI and 2D layer draws, which a game may also draw only once. On cold
// starts the title's frosted panels lost their blur (PS 0x09161b28ae4039bb, 80k words) and
// showed as dark boxes.
static const uint64_t g_pipeline_sync_ui_vs_words = [] {
	const char* value = std::getenv("KYTY_PIPELINE_SYNC_UI_VS_WORDS");
	return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {8000};
}();
static const std::chrono::milliseconds g_pipeline_wait = [] {
	const char* value = std::getenv("KYTY_PIPELINE_WAIT_MS");
	return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 20u);
}();

struct PendingGraphicsPipeline {
	std::unique_ptr<GraphicsPipelineBuild> build;
	// For the profiler: the guest shaders (pixel 0 without one).
	uint64_t                               vs_hash      = 0;
	uint64_t                               ps_hash      = 0;
	vk::PipelineCache                      driver_cache = nullptr;
	vk::Pipeline                           pipeline     = nullptr;
	vk::Result                             result       = vk::Result::eSuccess;
	std::atomic<bool>                      done {false};
	// Recorded in the recipe file once the pipeline is built (PipelineCache::RecipeState).
	PipelineRecipe::Store*                 recipe_store = nullptr;
	PipelineRecipe::Hash128                recipe_hash;
	std::vector<uint8_t>                   recipe;
};

// Pipelines built this run (the periodic driver-cache save writes when it moved).
static std::atomic<uint64_t> g_pipelines_built {0};

// Counts a finished pipeline build and puts it on the timeline (slow ones as a warning).
static void ReportPipelineBuilt(Profiler::Counter counter, const char* kind, uint64_t first_hash,
                                uint64_t second_hash, std::chrono::steady_clock::time_point begin) {
	const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                         std::chrono::steady_clock::now() - begin)
	                                         .count());
	Profiler::Add(counter);
	Profiler::Add(Profiler::Counter::PipelineBuildTime, us);
	g_pipelines_built.fetch_add(1, std::memory_order_relaxed);
	if (Common::FloorStats::Enabled()) {
		auto& floor = Common::FloorStats::Get();
		Common::FloorStats::Add(floor.pipe_built, 1);
		if (Libs::Graphics::GuestGpu::IsGpuThread()) {
			Common::FloorStats::Add(floor.pipe_sync_ns, static_cast<uint64_t>(us) * 1000u);
		}
	}
	// KYTY_FLOOR_LOG: a pipeline that took a second or more, by name (the 3 s stalls of 4.14).
	if (us >= 1000000 && Common::FloorStats::Enabled()) {
		std::fprintf(stderr, "[floor-pipe] %s pipeline built in %.0f ms (%s): 0x%016" PRIx64 " 0x%016" PRIx64 "\n",
		             kind, static_cast<double>(us) / 1000.0,
		             Libs::Graphics::GuestGpu::IsGpuThread() ? "GPU thread" : "worker", first_hash, second_hash);
	}
	Profiler::Message(us >= 100000 ? Profiler::MessageWarning : Profiler::MessageInfo,
	                  "%s pipeline built in %.1f ms: 0x%016" PRIx64 " 0x%016" PRIx64, kind,
	                  static_cast<double>(us) / 1000.0, first_hash, second_hash);
}

static void ReportGraphicsPipelineBuilt(uint64_t vs_hash, uint64_t ps_hash,
                                        std::chrono::steady_clock::time_point begin) {
	ReportPipelineBuilt(Profiler::Counter::GraphicsPipelinesBuilt, "Graphics", vs_hash, ps_hash,
	                    begin);
}

// Compiles graphics pipelines on worker threads. The driver compiled on the GPU thread, which
// stopped the whole emulated GPU for seconds per new pipeline (103 slow pipelines, 170 s of a
// 600 s first pass through the prologue).
class PipelineCompiler {
public:
	explicit PipelineCompiler(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~PipelineCompiler() { Stop(); }
	KYTY_CLASS_NO_COPY(PipelineCompiler);

	// `urgent`: the GPU thread is about to wait for this one (a draw that may not be skipped), so it
	// goes ahead of the background compiles queued for skipped draws; it waited behind all of
	// them before (a new area queues dozens at once: 0.3-0.9 s waits in Mac's 10-26 log).
	void Submit(std::shared_ptr<PendingGraphicsPipeline> job, bool urgent = false) {
		{
			std::lock_guard lock(m_mutex);
			if (urgent) {
				m_queue.push_front(std::move(job));
			} else {
				m_queue.push_back(std::move(job));
			}
		}
		m_work.notify_one();
	}

	// Waits up to the budget; true when the job is done.
	bool Wait(const PendingGraphicsPipeline& job, std::chrono::milliseconds budget) {
		KYTY_PROFILER_BLOCK("PipelineCompiler::Wait");
		std::unique_lock lock(m_mutex);
		return m_done.wait_for(lock, budget,
		                       [&job] { return job.done.load(std::memory_order_acquire); });
	}

	// Finishes the compiles in progress; queued ones are left to their owner.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopped) {
				return;
			}
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
		m_threads.clear();
	}

	[[nodiscard]] bool Stopped() {
		std::lock_guard lock(m_mutex);
		return m_stopped;
	}

private:
	void Run() {
		KYTY_PROFILER_THREAD("PipelineCompiler");
		for (;;) {
			std::shared_ptr<PendingGraphicsPipeline> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			{
				Profiler::Phases zone;
				zone.SetText("VS 0x%016" PRIx64 " PS 0x%016" PRIx64, job->vs_hash, job->ps_hash);
				KYTY_PROFILER_PHASE(zone, "PipelineCompiler::Compile", profiler::colors::DeepOrangeA200);
				const auto begin = std::chrono::steady_clock::now();
				job->result =
				    CreateGraphicsPipeline(*job->build, job->driver_cache, &job->pipeline);
				ReportGraphicsPipelineBuilt(job->vs_hash, job->ps_hash, begin);
				if (job->result == vk::Result::eSuccess && job->recipe_store != nullptr &&
				    !job->recipe.empty()) {
					(void)job->recipe_store->Record(job->recipe_hash, job->recipe);
				}
				Profiler::Add(Profiler::Counter::PipelinesPending, -1);
			}
			{
				std::lock_guard lock(m_mutex);
				job->done.store(true, std::memory_order_release);
			}
			m_done.notify_all();
		}
	}

	std::mutex                                           m_mutex;
	std::condition_variable                              m_work;
	std::condition_variable                              m_done;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_queue;
	std::vector<std::thread>                             m_threads;
	bool                                                 m_stopped = false;
};

// Graphics shader translation on worker threads (plan.md P1; KYTY_ASYNC_TRANSLATION=0 translates
// every shader on the GPU thread, as before). A new shader's translation (decode, CFG, IR,
// resource tracking) took ~85 ms on the GPU thread, and an effect brings dozens: the first
// Square slash in the jungle stopped the emulated GPU for 2-3 s, a new area for minutes. The
// translation depends only on the code and the stage inputs, never on guest memory or the draw's
// user data values (the disk cache's source key holds the same), so a worker can do it while the
// draw is skipped; the materialization and the SPIR-V that need guest memory stay on the GPU
// thread. Only draws with the depth test on are deferred (GetGraphicsPrograms): UI and layer
// draws, which a game may draw once, translate at once, as their pipelines compile at once.
// Compute always translates at once: a skipped dispatch can leave data unwritten.
static const bool g_async_translation = [] {
	const char* value = std::getenv("KYTY_ASYNC_TRANSLATION");
	return value == nullptr || value[0] != '0';
}();

struct TranslationJob {
	std::vector<uint32_t>                code;
	std::vector<uint32_t>                back_code;
	std::vector<uint32_t>                user_data;
	ShaderVertexInputInfo                vertex {};
	ShaderPixelInputInfo                 pixel {};
	ShaderRecompiler::CompileOptions     options;
	ShaderRecompiler::TranslateResult    result;
	std::chrono::steady_clock::time_point queued;
	std::atomic<bool>                    done {false};
};

class TranslationWorkers {
public:
	explicit TranslationWorkers(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~TranslationWorkers() {
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(TranslationWorkers);

	void Submit(std::shared_ptr<TranslationJob> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

private:
	void Run() {
		KYTY_PROFILER_THREAD("ShaderTranslator");
		for (;;) {
			std::shared_ptr<TranslationJob> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			job->result = TimedTranslate(job->code, job->options);
			job->done.store(true, std::memory_order_release);
		}
	}

	std::mutex                                  m_mutex;
	std::condition_variable                     m_work;
	std::deque<std::shared_ptr<TranslationJob>> m_queue;
	std::vector<std::thread>                    m_threads;
	bool                                        m_stopped = false;
};

// Vertex and pixel SRT refresh in parallel (plan.md P10, after Jetsku's draw-prep S3, c7c4da45).
// The SRT refresh is a third of the GPU thread in the jungle, and a draw's two stages walk
// independent plans into independent snapshots. In a short window per draw, this helper walks the
// vertex stage while the GPU thread walks the pixel stage; both only read bytes no GPU work has
// written (ReadShaderGuestMemoryProbe), so neither downloads, faults or waits, and nothing changes
// the GPU-written state the other reads. A walk that fails there (a byte the GPU wrote, memory
// not mapped) is repeated by the GPU thread on the exact path, as before. The helper spins
// KYTY_STAGE_PREP_SPIN_US (default 250) after its last job, then sleeps; a draw never waits for a
// sleeping helper, it wakes it for the next. Off by default: in the jungle a window took ~7 us,
// because the helper's walk runs on cold caches and the GPU thread waits for it, which cost more
// than the walk it saved (Tracy, 2026-10-07). KYTY_STAGE_PREP_PARALLEL=1 turns it on;
// KYTY_STAGE_PREP_VERIFY=1 repeats each parallel walk on the exact path and logs differences.
static const bool g_stage_prep_parallel = [] {
	const char* value = std::getenv("KYTY_STAGE_PREP_PARALLEL");
	return value != nullptr && value[0] == '1';
}();

inline void StagePrepPause() {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

class StagePrepWorker {
public:
	using Job = void (*)(void* context);

	StagePrepWorker() {
		const char* spin = std::getenv("KYTY_STAGE_PREP_SPIN_US");
		m_spin           = std::chrono::microseconds(spin != nullptr ? std::atoi(spin) : 250);
		m_thread         = std::thread([this] { Run(); });
	}
	~StagePrepWorker() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_stop_flag.store(true, std::memory_order_release);
		m_wake.notify_one();
		m_thread.join();
	}
	KYTY_CLASS_NO_COPY(StagePrepWorker);

	// Hands the job to the helper; false when it sleeps (it is woken for the next draw) or is
	// still busy, and the caller does the work itself.
	bool Post(Job job, void* context) {
		if (m_sleeping.load(std::memory_order_acquire)) {
			{
				std::lock_guard lock(m_mutex);
				m_wake_requested = true;
			}
			m_wake.notify_one();
			return false;
		}
		if (m_state.load(std::memory_order_acquire) != Idle) {
			return false;
		}
		m_job     = job;
		m_context = context;
		m_state.store(Posted, std::memory_order_release);
		return true;
	}

	// Waits for the posted job; false when the helper never started it (taken back).
	bool Join() {
		uint32_t expected = Posted;
		if (m_state.compare_exchange_strong(expected, Idle, std::memory_order_acq_rel)) {
			return false;
		}
		while (m_state.load(std::memory_order_acquire) != Done) {
			StagePrepPause();
		}
		m_state.store(Idle, std::memory_order_release);
		return true;
	}

private:
	enum : uint32_t { Idle, Posted, Running, Done };

	void Run() {
		KYTY_PROFILER_THREAD("StagePrep");
		Common::PreferPerformanceCores("stage preparation helper");
		Libs::LibKernel::Memory::SetGpuReadDelegate(true);
		auto last_job = std::chrono::steady_clock::now();
		while (!m_stop_flag.load(std::memory_order_acquire)) {
			uint32_t expected = Posted;
			if (m_state.load(std::memory_order_acquire) == Posted &&
			    m_state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) {
				m_job(m_context);
				m_state.store(Done, std::memory_order_release);
				last_job = std::chrono::steady_clock::now();
				continue;
			}
			if (std::chrono::steady_clock::now() - last_job < m_spin) {
				StagePrepPause();
				continue;
			}
			std::unique_lock lock(m_mutex);
			m_sleeping.store(true, std::memory_order_release);
			m_wake.wait(lock, [this] { return m_wake_requested || m_stop; });
			m_wake_requested = false;
			m_sleeping.store(false, std::memory_order_release);
			last_job = std::chrono::steady_clock::now();
		}
	}

	std::atomic<uint32_t>     m_state {Idle};
	Job                       m_job     = nullptr;
	void*                     m_context = nullptr;
	std::atomic<bool>         m_sleeping {false};
	std::atomic<bool>         m_stop_flag {false};
	std::mutex                m_mutex;
	std::condition_variable   m_wake;
	bool                      m_wake_requested = false;
	bool                      m_stop           = false;
	std::chrono::microseconds m_spin {250};
	std::thread               m_thread;
};

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

// Step D4: the library warm-up's context (the device and the driver cache it fills) and the
// permutations already warmed (<title>.gplwarm in _ShaderList: 8-byte ids, salted with what the
// libraries are made of, so a change of driver, device or library state warms again).
namespace {
GraphicContext*              g_warm_graphics = nullptr;
vk::PipelineCache            g_warm_cache;
vk::PipelineCache            g_warm_main_cache; // compute pipelines: the cache GetComputePipeline uses
std::function<void()>        g_warm_save_hook;  // writes the driver caches (after the warm-up)
std::vector<uint64_t>        g_warm_pending;    // made, not yet in a written cache
bool                         g_warm_caches_complete = true;
// A permutation whose library or pipeline creation was in flight when the process ended (a driver
// crash, a kill) is skipped from then on (<title>.gplwarm.poison), so one bad module cannot make
// every start crash; the in-flight slots (one per worker) are <title>.gplwarm.inflight.
std::FILE*                   g_inflight_file = nullptr;
std::mutex                   g_inflight_mutex;
std::unordered_set<uint64_t> g_poison;
std::unordered_set<uint64_t> g_suspects; // in flight with others at a crash: retried one at a time first
constexpr uint32_t           InflightSlots = 128;
std::mutex                   g_warm_mutex;
bool                         g_warm_loaded = false;
uint64_t                     g_warm_salt   = 0;
std::unordered_set<uint64_t> g_warmed;
std::FILE*                   g_warm_file = nullptr;

// KYTY_COMPUTE_WARM=0: compute pipelines are not warmed (with or without GPL).
bool ComputeWarmEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COMPUTE_WARM");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && g_warm_graphics != nullptr && g_warm_main_cache != nullptr;
}

bool GplWarmEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPL_WARM");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && g_warm_graphics != nullptr && g_warm_graphics->graphics_pipeline_library_enabled;
}

uint64_t GplWarmId(std::span<const uint8_t> key) {
	return XXH3_64bits_withSeed(key.data(), key.size(), g_warm_salt);
}

void GplWarmLoad() {
	if (g_warm_loaded) {
		return;
	}
	g_warm_loaded       = true;
	const auto& props   = g_warm_graphics->GetPhysicalDeviceProperties();
	// v437e: only what shapes the warmed libraries (the v437c record named feedback loops too, and
	// a harness start did not find the first start's record).
	const uint64_t salt_words[] = {0x47504C5741524D03ull, props.driverVersion, props.deviceID, props.vendorID,
	                               GplWarmStateHash(*g_warm_graphics)};
	g_warm_salt = XXH3_64bits(salt_words, sizeof(salt_words));
	const auto title = PipelineCacheTitleId();
	if (title.empty()) {
		return;
	}
	std::error_code error;
	std::filesystem::create_directories(ShaderListDirectory(), error);
	const auto path = ShaderListDirectory() / (title + ".gplwarm");
	if (std::FILE* in = std::fopen(Common::PathToString(path).c_str(), "rb"); in != nullptr) {
		uint64_t id = 0;
		while (std::fread(&id, sizeof(id), 1, in) == 1) {
			g_warmed.insert(id);
		}
		std::fclose(in);
	}
	// KYTY_GPL_WARM_RESET=1: the record is ignored and started over (diagnosis).
	if (const char* reset = std::getenv("KYTY_GPL_WARM_RESET"); reset != nullptr && reset[0] == '1') {
		g_warmed.clear();
		if (std::FILE* trunc = std::fopen(Common::PathToString(path).c_str(), "wb"); trunc != nullptr) {
			std::fclose(trunc);
		}
	}
	// Step D7: the record names libraries and compute pipelines a written cache holds; with a
	// shard file missing or rejected it is dropped (they are made again, mostly cache hits).
	if (!g_warm_caches_complete && !g_warmed.empty()) {
		std::printf("[gpl-warm] record of %zu dropped: a cache shard was missing or rejected\n", g_warmed.size());
		g_warmed.clear();
		if (std::FILE* trunc = std::fopen(Common::PathToString(path).c_str(), "wb"); trunc != nullptr) {
			std::fclose(trunc);
		}
	}
	g_warm_file = std::fopen(Common::PathToString(path).c_str(), "ab");
	{
		auto poison_path   = path;
		poison_path       += ".poison";
		auto inflight_path = path;
		inflight_path     += ".inflight";
		std::vector<uint64_t> crashed;
		if (std::FILE* in = std::fopen(Common::PathToString(inflight_path).c_str(), "rb"); in != nullptr) {
			uint64_t id = 0;
			while (std::fread(&id, sizeof(id), 1, in) == 1) {
				if (id != 0) {
					crashed.push_back(id);
				}
			}
			std::fclose(in);
		}
		if (std::FILE* in = std::fopen(Common::PathToString(poison_path).c_str(), "rb"); in != nullptr) {
			uint64_t id = 0;
			while (std::fread(&id, sizeof(id), 1, in) == 1) {
				g_poison.insert(id);
			}
			std::fclose(in);
		}
		if (crashed.size() == 1) {
			// Alone in flight (the suspects' pass, or one worker): this one ended the process.
			if (std::FILE* out = std::fopen(Common::PathToString(poison_path).c_str(), "ab"); out != nullptr) {
				std::fwrite(crashed.data(), sizeof(uint64_t), crashed.size(), out);
				std::fclose(out);
			}
			g_poison.insert(crashed.begin(), crashed.end());
			std::printf("[gpl-warm] the warm-up item in flight when the last start ended is skipped from now on (%016llx)\n",
			            static_cast<unsigned long long>(crashed.front()));
		} else if (!crashed.empty()) {
			g_suspects.insert(crashed.begin(), crashed.end());
			std::printf("[gpl-warm] %zu warm-up items were in flight when the last start ended: retried one at a time first\n",
			            crashed.size());
		}
		if (!g_poison.empty()) {
			std::printf("[gpl-warm] %zu items skipped (in flight at an earlier crash)\n", g_poison.size());
		}
		g_inflight_file = std::fopen(Common::PathToString(inflight_path).c_str(), "w+b");
		if (g_inflight_file != nullptr) {
			const std::vector<uint64_t> zero(InflightSlots, 0);
			std::fwrite(zero.data(), sizeof(uint64_t), zero.size(), g_inflight_file);
			std::fflush(g_inflight_file);
		}
	}
	std::printf("[gpl-warm] %zu permutations warmed before (%s; salt %016llx, state %016llx)\n", g_warmed.size(),
	            Common::PathToString(path).c_str(), static_cast<unsigned long long>(g_warm_salt),
	            static_cast<unsigned long long>(salt_words[4]));
	std::fflush(stdout);
}

// The precompile replay asks per graphics permutation whether its library is to be warmed.
bool GplWarmWanted(ShaderType stage, std::span<const uint8_t> key) {
	if (stage == ShaderType::Compute) {
		if (!ComputeWarmEnabled()) {
			return false;
		}
	} else if (stage != ShaderType::Vertex && stage != ShaderType::Mesh && stage != ShaderType::Pixel) {
		return false;
	} else if (!GplWarmEnabled()) {
		return false;
	}
	std::lock_guard lock(g_warm_mutex);
	GplWarmLoad();
	const auto id = GplWarmId(key);
	return !g_warmed.contains(id) && !g_poison.contains(id);
}

bool GplWarmLibraryNow(ShaderType stage, std::span<const uint32_t> spirv,
                       const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	if (stage == ShaderType::Compute) {
		return ComputeWarmPipeline(*g_warm_graphics, g_warm_main_cache, spirv, program);
	}
	return GplWarmLibrary(*g_warm_graphics, g_warm_cache, stage, spirv, program);
}

void GplWarmDone(std::span<const uint8_t> key) {
	std::lock_guard lock(g_warm_mutex);
	const auto      id = GplWarmId(key);
	if (g_warmed.insert(id).second) {
		g_warm_pending.push_back(id); // into the record after the next cache write (GplWarmCommit)
	}
}

// The ids made before a cache write started (taken at its start): only they are in the files.
std::vector<uint64_t> GplWarmTakePending() {
	std::lock_guard       lock(g_warm_mutex);
	std::vector<uint64_t> taken;
	taken.swap(g_warm_pending);
	return taken;
}

// The item a warm worker is about to create (empty key: done).
void GplWarmInFlight(uint32_t worker, std::span<const uint8_t> key) {
	if (g_inflight_file == nullptr || worker >= InflightSlots) {
		return;
	}
	const uint64_t  id = key.empty() ? 0 : GplWarmId(key);
	std::lock_guard lock(g_inflight_mutex);
	std::fseek(g_inflight_file, static_cast<long>(worker * sizeof(uint64_t)), SEEK_SET);
	std::fwrite(&id, sizeof(id), 1, g_inflight_file);
	std::fflush(g_inflight_file);
}

void GplWarmCommit(std::vector<uint64_t>& ids, bool written) {
	std::lock_guard lock(g_warm_mutex);
	if (!written) {
		// Back to pending for the next write.
		g_warm_pending.insert(g_warm_pending.end(), ids.begin(), ids.end());
	} else if (g_warm_file != nullptr && !ids.empty()) {
		std::fwrite(ids.data(), sizeof(uint64_t), ids.size(), g_warm_file);
		std::fflush(g_warm_file);
	}
	ids.clear();
}

void GplWarmSetCachesComplete(bool complete) {
	std::lock_guard lock(g_warm_mutex);
	g_warm_caches_complete = complete;
}

uint32_t GplWarmThreads() {
	if (const char* value = std::getenv("KYTY_GPL_WARM_THREADS"); value != nullptr) {
		return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}
	// The game waits for it at startup (KYTY_GPL_WARM_WAIT, graphicsRun.cpp): every core but two;
	// in the background while the game runs: half.
	const uint32_t cores = std::max(2u, std::thread::hardware_concurrency());
	const char*    wait  = std::getenv("KYTY_GPL_WARM_WAIT");
	return wait != nullptr && wait[0] == '0' ? std::max(1u, cores / 2u) : std::max(1u, cores - 2u);
}

// Library prefetch (pipelineCache.h). The index: every listed graphics permutation by its guest
// shader's hash, as the 16-byte name of its compiled record in the shader store (filled by the
// startup replay, which builds each permutation's key anyway). A request takes the shader's
// permutations to the workers: record read, library made and kept (GplWarmLibrary with keep).
// Everything listed was compiled into the driver cache by the warm-up, so these are cache hits,
// the same ones the draw's pipeline would otherwise make while the GPU thread waits for it.
struct PrefetchItem {
	ShaderType stage = ShaderType::Unknown;
	// XXH3-128 of the permutation's key: what the store names the record's file by.
	uint64_t low  = 0;
	uint64_t high = 0;
};
struct Prefetcher {
	std::mutex                                              mutex;
	std::condition_variable                                 wake;
	std::unordered_map<uint64_t, std::vector<PrefetchItem>> index;
	std::unordered_set<uint64_t>                            asked;
	std::deque<uint64_t>                                    queue;
	std::vector<std::thread>                                threads;
	std::filesystem::path                                   directory; // the shader store's
	bool                                                    stop = false;
	std::chrono::steady_clock::time_point                   reported {};
	std::atomic<uint64_t>                                   shaders {0};
	std::atomic<uint64_t>                                   records {0};
	std::atomic<uint64_t>                                   not_read {0};
	std::atomic<uint64_t>                                   not_made {0};
	std::atomic<uint64_t>                                   linked_here {0};
};
Prefetcher& Prefetch() {
	static Prefetcher prefetcher;
	return prefetcher;
}

// The startup replay: a listed permutation of `shader_hash` (not one the warm-up skips).
void GplPrefetchIndexAdd(ShaderType stage, uint64_t shader_hash, std::span<const uint8_t> permutation_key) {
	if ((stage != ShaderType::Vertex && stage != ShaderType::Mesh && stage != ShaderType::Pixel) ||
	    shader_hash == 0 || !GplPrefetchEnabled()) {
		return;
	}
	{
		std::lock_guard lock(g_warm_mutex);
		GplWarmLoad();
		if (g_poison.contains(GplWarmId(permutation_key))) {
			return; // its library was in flight when a start ended: never made again
		}
	}
	const auto      name = XXH3_128bits(permutation_key.data(), permutation_key.size());
	auto&           p    = Prefetch();
	std::lock_guard lock(p.mutex);
	p.index[shader_hash].push_back({stage, name.low64, name.high64});
}

void GplPrefetchSetStore(const std::filesystem::path& directory) {
	auto&           p = Prefetch();
	std::lock_guard lock(p.mutex);
	p.directory = directory;
}

// A listed permutation's compiled record, read by the name of its file. The store's Load takes
// the whole key, and a key holds the shader's code (kilobytes for each of the ~32,000 listed
// graphics permutations); the index keeps the 16 bytes the store names the file by instead
// (shaderDiskCache.cpp: HexKey, the XXH3-128 of the key, high half first). The file, as
// Store::Save writes it: magic, format, key size, payload size, payload hash, key, payload.
// Checked here: the key in the file hashes to the name asked for (so it is the indexed key, as
// Load's comparison would find), the file starts as that key does (every key starts with the
// session key, which starts with the magic and the format), the sizes, and the payload's hash.
// Anything else is a record not read: the draw's pipeline then makes its library as before.
bool GplPrefetchLoadRecord(const std::filesystem::path& directory, const PrefetchItem& item,
                           std::vector<uint8_t>& payload) {
	const auto path = directory / (fmt::format("{:016x}{:016x}", item.high, item.low) + ".bin");
	std::FILE* file = nullptr;
#ifdef _WIN32
	if (_wfopen_s(&file, path.c_str(), L"rb") != 0) {
		file = nullptr;
	}
#else
	file = std::fopen(path.c_str(), "rb");
#endif
	if (file == nullptr) {
		return false;
	}
	std::vector<uint8_t> bytes;
	bool                 ok   = std::fseek(file, 0, SEEK_END) == 0;
	const auto           size = ok ? std::ftell(file) : -1;
	ok = ok && size > 0 && static_cast<uint64_t>(size) <= (uint64_t {256} << 20u) && std::fseek(file, 0, SEEK_SET) == 0;
	if (ok) {
		bytes.resize(static_cast<size_t>(size));
		ok = std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
	}
	std::fclose(file);
	constexpr size_t words  = 2 * sizeof(uint32_t); // magic, format
	constexpr size_t header = words + 3 * sizeof(uint64_t);
	if (!ok || bytes.size() < header) {
		return false;
	}
	uint64_t key_size     = 0;
	uint64_t payload_size = 0;
	uint64_t payload_hash = 0;
	std::memcpy(&key_size, bytes.data() + words, sizeof(uint64_t));
	std::memcpy(&payload_size, bytes.data() + words + sizeof(uint64_t), sizeof(uint64_t));
	std::memcpy(&payload_hash, bytes.data() + words + 2 * sizeof(uint64_t), sizeof(uint64_t));
	const uint64_t rest = bytes.size() - header;
	if (key_size < words || key_size > rest || payload_size != rest - key_size) {
		return false;
	}
	const uint8_t* key  = bytes.data() + header;
	const auto     name = XXH3_128bits(key, static_cast<size_t>(key_size));
	if (name.low64 != item.low || name.high64 != item.high || std::memcmp(bytes.data(), key, words) != 0 ||
	    XXH3_64bits(key + key_size, static_cast<size_t>(payload_size)) != payload_hash) {
		return false;
	}
	payload.assign(key + key_size, key + key_size + payload_size);
	return true;
}

void GplPrefetchReport(bool final_line) {
	auto&    p    = Prefetch();
	uint64_t made = 0;
	uint64_t used = 0;
	uint64_t late = 0;
	GplPrefetchCounters(made, used, late);
	if (p.shaders.load() == 0 && p.linked_here.load() == 0) {
		return;
	}
	std::printf("[gpl-prefetch]%s %llu shaders seen ahead of their draw: %llu records read (%llu not read, %llu "
	            "not made), %llu libraries kept, %llu of them taken by a linked pipeline, %llu late (the draw "
	            "made its own first); %llu pipelines linked on the GPU thread without a worker\n",
	            final_line ? " end:" : "", static_cast<unsigned long long>(p.shaders.load()),
	            static_cast<unsigned long long>(p.records.load()), static_cast<unsigned long long>(p.not_read.load()),
	            static_cast<unsigned long long>(p.not_made.load()), static_cast<unsigned long long>(made),
	            static_cast<unsigned long long>(used), static_cast<unsigned long long>(late),
	            static_cast<unsigned long long>(p.linked_here.load()));
	std::fflush(stdout);
}

void GplPrefetchWork() {
	auto& p = Prefetch();
	for (;;) {
		std::vector<PrefetchItem> items;
		std::filesystem::path     directory;
		{
			std::unique_lock lock(p.mutex);
			p.wake.wait(lock, [&p] { return p.stop || !p.queue.empty(); });
			if (p.stop) {
				return;
			}
			const auto hash = p.queue.front();
			p.queue.pop_front();
			if (const auto found = p.index.find(hash); found != p.index.end()) {
				items = found->second;
			}
			directory = p.directory;
		}
		for (const auto& item: items) {
			std::vector<uint8_t>               payload;
			ShaderDiskCache::PermutationRecord record;
			if (g_warm_graphics == nullptr || directory.empty() || !GplPrefetchLoadRecord(directory, item, payload) ||
			    !ShaderDiskCache::DecodePermutationRecord(payload, record)) {
				p.not_read.fetch_add(1, std::memory_order_relaxed);
			} else if (GplWarmLibrary(*g_warm_graphics, g_warm_cache, item.stage, record.spirv, record.program, true)) {
				p.records.fetch_add(1, std::memory_order_relaxed);
			} else {
				p.not_made.fetch_add(1, std::memory_order_relaxed);
			}
		}
		// One line at most every 10 s while there is work (and one at the end, GplPrefetchStop).
		bool report = false;
		{
			std::lock_guard lock(p.mutex);
			const auto      now = std::chrono::steady_clock::now();
			if (p.queue.empty() && now - p.reported >= std::chrono::seconds(10)) {
				p.reported = now;
				report     = true;
			}
		}
		if (report) {
			GplPrefetchReport(false);
		}
	}
}
} // namespace

bool GplPrefetchEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPL_PREFETCH");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && GplWarmEnabled();
}

void GplPrefetchRequest(uint64_t shader_hash) {
	if (shader_hash == 0 || !GplPrefetchEnabled()) {
		return;
	}
	auto& p = Prefetch();
	{
		std::lock_guard lock(p.mutex);
		// Asked before: nothing to do. Not listed (yet): nothing to make, and not remembered as
		// asked (a start that does not wait for the replay lists it later).
		if (p.stop || p.asked.contains(shader_hash) || !p.index.contains(shader_hash)) {
			return;
		}
		p.asked.insert(shader_hash);
		if (p.threads.empty()) {
			// They run ahead of the GPU thread's draw: normal priority, a few threads (the work is
			// a record read and a driver-cache hit per library). KYTY_GPL_PREFETCH_THREADS=<n>.
			uint32_t count = std::clamp(std::thread::hardware_concurrency() / 6u, 2u, 4u);
			if (const char* value = std::getenv("KYTY_GPL_PREFETCH_THREADS"); value != nullptr) {
				count = std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 16u);
			}
			for (uint32_t i = 0; i < count; i++) {
				p.threads.emplace_back(GplPrefetchWork);
			}
		}
		p.queue.push_back(shader_hash);
	}
	p.shaders.fetch_add(1, std::memory_order_relaxed);
	p.wake.notify_one();
}

void GplPrefetchStop() {
	auto&                    p = Prefetch();
	std::vector<std::thread> threads;
	{
		std::lock_guard lock(p.mutex);
		p.stop = true;
		p.queue.clear();
		threads.swap(p.threads);
	}
	p.wake.notify_all();
	for (auto& thread: threads) {
		thread.join();
	}
	GplPrefetchReport(true);
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;
		// Exact expanded code includes callees and their original return-PC constants.
		// A different target/body must not reuse a program compiled for an earlier call.
		std::vector<uint32_t> function_code;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque: adding a permutation keeps the others in place (the async thread's queued
		// dispatches point at their permutation's program until they are recorded, G4).
		std::deque<Permutation>                     permutations;
		// Held around every walk of resource_plan, whose memo and replay state are mutable: the
		// draw-prep scanner (KYTY_DRAW_PREP) walks programs while the GPU thread does.
		std::mutex walk_mutex;
		// Draw prep: the serial of the prepared result whose refresh `resources` holds now (0 after a
		// walk here), so a later draw of the same packet can use it again.
		uint64_t prepared_serial = 0;
		// xxh3 of the source entry's payload (the shader ledger's plan fingerprint; 0: unknown).
		uint64_t plan_fingerprint = 0;
		// KYTY_MULTI_DRAW_REPLAY: the plan's user-data reads are registered (NotePlanUserDataReads).
		bool plan_reads_noted = false;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			PipelineKeyHash::Mix(hash, key.function_code.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	struct Compiled {
		std::vector<uint32_t>                    spirv;
		ShaderRecompiler::IR::CompiledShaderInfo program;
	};

	Compiled CompileTranslated(const ShaderRecompiler::CompileOptions&             options,
	                           ShaderRecompiler::TranslateResult                   translated,
	                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                           uint32_t push_data_start_dword) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		last_spirv_words = result.spirv.size();
		return {.spirv   = std::move(result.spirv),
		        .program = std::move(result.program).TakeCompiledInfo()};
	}

	// v296: every set-0 binding the SPIR-V declares must be one the program's bindings put in the
	// pipeline layout (and so one CommitBindings writes). A binding the shader declares but the layout
	// lacks is invalid usage the driver does not catch: the shader then reads whatever descriptor
	// state is left in that slot, possibly a header of an image freed long ago (Crash B). Logged once
	// per shader permutation, [layout-check].
	static void CheckSpirvBindings(uint64_t hash, const char* stage_name,
	                               const std::vector<uint32_t>&                    spirv,
	                               const ShaderRecompiler::IR::CompiledShaderInfo& program) {
		std::unordered_map<uint32_t, uint32_t> sets, bindings, constants, array_length, pointee,
		    variable_type, scalar_bytes, array_element;
		std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> composite;
		std::vector<uint32_t>                                       local_variables;
		for (size_t i = 5; i < spirv.size();) {
			const auto words  = spirv[i] >> 16u;
			const auto opcode = spirv[i] & 0xffffu;
			if (words == 0 || i + words > spirv.size()) {
				break;
			}
			if (opcode == 71 && words >= 4) { // OpDecorate target decoration value
				if (spirv[i + 2] == 34) sets[spirv[i + 1]] = spirv[i + 3];
				if (spirv[i + 2] == 33) bindings[spirv[i + 1]] = spirv[i + 3];
			} else if (opcode == 43 && words >= 4) { // OpConstant type result value
				constants[spirv[i + 2]] = spirv[i + 3];
			} else if (opcode == 28 && words >= 4) { // OpTypeArray result element length
				array_length[spirv[i + 1]] = spirv[i + 3];
			} else if (opcode == 29 && words >= 3) { // OpTypeRuntimeArray result element
				array_length[spirv[i + 1]] = 0xffffffffu; // no constant: unknown length
			} else if (opcode == 32 && words >= 4) { // OpTypePointer result storage type
				pointee[spirv[i + 1]] = spirv[i + 3];
			} else if (opcode == 59 && words >= 4) { // OpVariable type result storage
				variable_type[spirv[i + 2]] = spirv[i + 1];
				if (spirv[i + 3] == 6 || spirv[i + 3] == 7) { // Private, Function
					local_variables.push_back(spirv[i + 1]);
				}
			} else if ((opcode == 21 || opcode == 22) && words >= 3) { // OpTypeInt/OpTypeFloat
				scalar_bytes[spirv[i + 1]] = spirv[i + 2] / 8u;
			} else if ((opcode == 23 || opcode == 24) && words >= 4) { // OpTypeVector/OpTypeMatrix
				composite[spirv[i + 1]] = {spirv[i + 2], spirv[i + 3]};
			}
			if (opcode == 28 && words >= 4) {
				array_element[spirv[i + 1]] = spirv[i + 2];
			}
			i += words;
		}
		// v299: the local memory (Function and Private arrays) one invocation needs. NVIDIA's driver
		// sizes a per-device local-memory area for the largest pipeline it has seen and grows it when
		// a larger one is created; NVIDIA bug 6822663 (fixed in a later driver) faults work still in
		// flight when it grows. The largest size so far is logged, [local-mem], to choose
		// KYTY_LOCAL_MEMORY_WARMUP, which sizes the area before any guest work.
		{
			std::function<uint64_t(uint32_t, int)> type_bytes = [&](uint32_t type, int depth) -> uint64_t {
				if (depth > 8) return 0;
				if (const auto s = scalar_bytes.find(type); s != scalar_bytes.end()) return s->second;
				if (const auto c = composite.find(type); c != composite.end()) {
					return type_bytes(c->second.first, depth + 1) * c->second.second;
				}
				if (const auto a = array_element.find(type); a != array_element.end()) {
					const auto length = array_length.find(type);
					const auto value  = length != array_length.end() ? constants.find(length->second)
					                                                 : constants.end();
					return value != constants.end()
					           ? type_bytes(a->second, depth + 1) * value->second
					           : 0;
				}
				return 0;
			};
			uint64_t bytes = 0;
			for (const auto pointer: local_variables) {
				const auto target = pointee.find(pointer);
				if (target != pointee.end() && array_element.count(target->second) != 0) {
					bytes += type_bytes(target->second, 0);
				}
			}
			static std::atomic<uint64_t> largest {0};
			auto                         seen = largest.load();
			while (bytes > seen && !largest.compare_exchange_weak(seen, bytes)) {
			}
			if (bytes > seen) {
				std::fprintf(stderr,
				             "[local-mem] %s %016" PRIx64 ": %" PRIu64
				             " bytes of private arrays per invocation (largest so far)\n",
				             stage_name, hash, bytes);
			}
		}
		std::unordered_map<uint32_t, uint32_t> declared; // binding -> descriptor count
		for (const auto& binding: program.bindings.descriptors) {
			declared[ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind)] =
			    NativeDescriptorCount(binding);
		}
		static std::atomic<uint32_t> reported {0};
		for (const auto& [id, binding]: bindings) {
			const auto set = sets.find(id);
			if (set == sets.end() || set->second != 0) {
				continue;
			}
			// The variable's array length (1 when not an array; 0 when unknown).
			uint32_t length = 1;
			if (const auto type = variable_type.find(id); type != variable_type.end()) {
				if (const auto target = pointee.find(type->second); target != pointee.end()) {
					if (const auto array = array_length.find(target->second); array != array_length.end()) {
						const auto value = constants.find(array->second);
						length = value != constants.end() ? value->second : 0;
					}
				}
			}
			const auto found = declared.find(binding);
			if (found != declared.end() && length == 0) {
				static std::atomic<uint32_t> inconclusive {0};
				if (inconclusive.fetch_add(1) < 16) {
					std::fprintf(stderr,
					             "[layout-check] %s %016" PRIx64 ": set 0 binding %u has an array length the "
					             "check cannot read (runtime or nested array); layout has %u\n",
					             stage_name, hash, binding, found->second);
				}
				continue;
			}
			if (found != declared.end() && length <= found->second) {
				continue;
			}
			if (reported.fetch_add(1) < 64) {
				std::fprintf(stderr,
				             "[layout-check] %s %016" PRIx64 ": SPIR-V set 0 binding %u (id %u, %u descriptors) "
				             "%s (%zu bindings)\n",
				             stage_name, hash, binding, id, length,
				             found == declared.end() ? "is not in the pipeline layout"
				                                     : "has more descriptors than the layout",
				             declared.size());
			}
		}
	}

	// A compiled (or cached) permutation of the source: its module, bound for this stage.
	template <typename InputInfo>
	ShaderProgram AddPermutation(SourceEntry& source, const char* stage_name,
	                             const ShaderRecompiler::CompileOptions&  options,
	                             const std::vector<uint32_t>&             spirv,
	                             ShaderRecompiler::IR::CompiledShaderInfo program,
	                             InputInfo& input_info, uint32_t& push_data_cursor,
	                             const ShaderRecompiler::IR::ResourceSpecialization* module_specialization = nullptr) {
		const CpuCost::Phase cpu_phase_("Program::AddPermutation");
		Profiler::Phases phases;
		phases.SetText("%s 0x%016" PRIx64, stage_name, options.shader_hash);
		KYTY_PROFILER_PHASE(phases, "Shader: SPIR-V validate", profiler::colors::Blue300);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, spirv);

		CheckSpirvBindings(options.shader_hash, stage_name, spirv, program);
		KYTY_PROFILER_PHASE(phases, "Shader: module create", profiler::colors::Blue300);
		const auto module = CompileSPV(spirv, device);
		EXIT_IF(module == nullptr);
		PipelineRecipe::RegisterModule(module, spirv, options.stage != ShaderType::Compute);
		if (options.stage != ShaderType::Compute) {
			RegisterBindlessSet3Twin(device, module, spirv);
		}
		Aftermath::NoteShader(options.shader_hash, spirv.data(), spirv.size());
		phases.End();
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(spirv.size()), options.wave_size);
		}
		source.permutations.push_back({
		    // v425: the specialization the module was made for (a null-buffer match keeps the
		    // matched permutation's, so a later real buffer of that stride cannot pick it wrongly).
		    .specialization = module_specialization != nullptr ? *module_specialization : source.specialization,
		    .program        = std::move(program),
		    .handle         = {.id          = ++next_shader_id,
		                       .module      = module,
		                       .spirv_words = static_cast<uint32_t>(spirv.size()),
		                       .hash        = options.shader_hash},
		});
		const auto& permutation = source.permutations.back();
		// KYTY_SHADER_DATA_STATS=1 (zero-compile plan): each new module's shader-data size, where it
		// lives (push constants or the ShaderData storage buffer) and its descriptor count.
		if (static const bool data_stats = [] {
			    const char* value = std::getenv("KYTY_SHADER_DATA_STATS");
			    return value != nullptr && value[0] == '1';
		    }();
		    data_stats) {
			const auto& layout      = permutation.program.bindings;
			size_t      descriptors = 0;
			for (const auto& binding: layout.descriptors) {
				descriptors += std::max<size_t>(binding.resources.size(), 1u);
			}
			std::fprintf(stderr, "[shader-data] %s %016" PRIx64 " start=%d dwords=%u push=%d buffers=%u descriptors=%zu\n",
			             stage_name, options.shader_hash,
			             layout.UsesPushData() ? static_cast<int>(layout.push_data_start_dword) : -1,
			             layout.ShaderDataDwords(), layout.UsesPushData() ? 1 : 0,
			             layout.memory_offset_count, descriptors);
		}
		input_info.stage = {.program = &permutation.program, .resources = &source.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		if (options.stage != ShaderType::Compute && !source.plan_reads_noted &&
		    RenderStats::MultiDrawReplayConfigured()) {
			source.plan_reads_noted = true;
			NotePlanUserDataReads(&source.resources, source.resource_plan);
		}

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, entry]: programs) {
			counts[static_cast<size_t>(key.stage)] += entry.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// v425 (zero-compile plan): a null buffer descriptor (V# all zero) has no records, so every
	// access to it is out of bounds whatever its stride: loads give zero and stores are dropped.
	// A draw whose buffer is null can therefore use a module specialized for that buffer's real
	// stride (the census: 1,286 of 1,316 stride differences were a null V# against a real one).
	// Everything else, its format and swizzle included (they decide the out-of-bounds value),
	// must match. KYTY_NULL_BUFFER_MATCH=0: exact matches only.
	static bool NullBufferMatchEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_NULL_BUFFER_MATCH");
			return value == nullptr || value[0] != '0';
		}();
		return enabled;
	}
	static bool NullDescriptor(const ShaderRecompiler::IR::ResourceSnapshot& resources, size_t index) {
		if (index >= resources.buffers.size()) {
			return false;
		}
		const auto& value = resources.buffers[index];
		const auto  count = std::min<size_t>(value.dwords.size(), 4u);
		return count == 4u && std::all_of(value.dwords.begin(), value.dwords.begin() + 4,
		                                  [](uint32_t word) { return word == 0u; });
	}
	static bool NullImage(const ShaderRecompiler::IR::ResourceSnapshot& resources, size_t index) {
		if (index >= resources.images.size()) {
			return false;
		}
		const auto& value = resources.images[index];
		return value.dwords[0] == 0u && (value.dwords[1] & 0xffu) == 0u;
	}
	static bool HasNullBuffer(const ShaderRecompiler::IR::ResourceSnapshot& resources,
	                          const ShaderRecompiler::IR::ResourceSpecialization& current) {
		for (size_t i = 0; i < current.buffers.size(); i++) {
			if (NullDescriptor(resources, i)) {
				return true;
			}
		}
		for (size_t i = 0; i < current.images.size(); i++) {
			if (NullImage(resources, i)) {
				return true;
			}
		}
		return false;
	}
	static bool SpecializationFits(const ShaderRecompiler::IR::ResourceSpecialization& candidate,
	                               const ShaderRecompiler::IR::ResourceSpecialization& current,
	                               const ShaderRecompiler::IR::ResourceSnapshot&       resources) {
		if (candidate == current) {
			return true;
		}
		if (!NullBufferMatchEnabled() || candidate.buffers.size() != current.buffers.size() ||
		    candidate.images.size() != current.images.size() || candidate.samplers != current.samplers) {
			return false;
		}
		// v427 (Fable, zero-compile plan step 3): a null T# (base address 0) reads zeros whatever the
		// module's view, and the host binds a null image of the module's own numeric class and
		// dimension (NullTextureDesc, descriptors.cpp), so those two may differ. Cube (a null image
		// is never a cube view), swizzle, conversion format, mips and the rest must match.
		for (size_t i = 0; i < current.images.size(); i++) {
			if (candidate.images[i] == current.images[i]) {
				continue;
			}
			if (!NullImage(resources, i)) {
				return false;
			}
			auto relaxed          = candidate.images[i];
			relaxed.numeric_class = current.images[i].numeric_class;
			relaxed.dimension     = current.images[i].dimension;
			if (!(relaxed == current.images[i])) {
				return false;
			}
		}
		for (size_t i = 0; i < current.buffers.size(); i++) {
			if (candidate.buffers[i] == current.buffers[i]) {
				continue;
			}
			if (!NullDescriptor(resources, i)) {
				return false;
			}
			auto relaxed            = candidate.buffers[i];
			relaxed.packed_stride   = current.buffers[i].packed_stride;
			relaxed.zero_stride_oob = current.buffers[i].zero_stride_oob;
			if (!(relaxed == current.buffers[i])) {
				return false;
			}
		}
		return true;
	}
	// v425: the ledger's permutations by source (kept from the replay), for a draw with a null
	// buffer whose exact permutation is not in the disk cache (SpecializationFits).
	std::mutex                                                                     recorded_mutex;
	std::unordered_map<uint64_t, std::vector<ShaderLedger::PermutationRecipe>>     recorded_by_source;
	bool LoadCompatiblePermutation(uint64_t source_id, uint32_t push_data_cursor, uint64_t fingerprint,
	                               const std::vector<uint8_t>&                         source_key,
	                               const ShaderRecompiler::IR::ResourceSpecialization& current,
	                               const ShaderRecompiler::IR::ResourceSnapshot&       resources,
	                               std::vector<uint8_t>&                               payload,
	                               ShaderRecompiler::IR::ResourceSpecialization&       matched) {
		if (!NullBufferMatchEnabled() || fingerprint == 0 || !HasNullBuffer(resources, current)) {
			return false;
		}
		std::vector<std::vector<uint8_t>> candidates;
		{
			std::lock_guard lock(recorded_mutex);
			const auto      found = recorded_by_source.find(source_id);
			if (found == recorded_by_source.end()) {
				return false;
			}
			for (const auto& recipe: found->second) {
				if (recipe.push_data_cursor == push_data_cursor && recipe.plan_fingerprint == fingerprint) {
					candidates.push_back(recipe.specialization);
				}
			}
		}
		std::vector<uint8_t> key;
		for (const auto& bytes: candidates) {
			ShaderRecompiler::IR::ResourceSpecialization candidate;
			if (!ShaderDiskCache::DecodeSpecialization(bytes, candidate) ||
			    !SpecializationFits(candidate, current, resources)) {
				continue;
			}
			BuildPermutationKey(key, source_key, candidate, push_data_cursor);
			if (disk->Load(key, payload)) {
				matched = std::move(candidate);
				return true;
			}
		}
		return false;
	}

	// The first materialization of a new source; a failure skips this draw (see Get).
	static bool MaterializeFirstUse(SourceEntry&                           source,
	                                const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                ShaderType stage, const char* stage_name, uint64_t hash) {
		bool ok = false;
		{
			std::lock_guard walk_lock(source.walk_mutex);
			source.prepared_serial = 0;
			ok = ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime,
			                                                source.resources, source.specialization);
		}
		if (ok) {
			return true;
		}
		if (!ShaderFailureNonFatal()) {
			EXIT("shader resource materialization failed\n");
		}
		Profiler::Add(Profiler::Counter::SkippedResources);
		LogSkip("first-use", stage_name, hash, ShaderRecompiler::IR::LastIndirectImageFailureLine());
		static std::atomic<uint32_t> reported = 0;
		if (reported.load() < 64) {
			std::fprintf(stderr,
			             "[skip-resources] first use: %s 0x%016" PRIx64 " (materialization line %d)\n",
			             stage_name, hash, ShaderRecompiler::IR::LastIndirectImageFailureLine());
			std::fflush(stderr);
		}
		if (reported.fetch_add(1) < 16) {
			LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
			     ": resource materialization failed on first use (materialization line %d)\n",
			     static_cast<uint32_t>(stage), hash,
			     ShaderRecompiler::IR::LastIndirectImageFailureLine());
			Profiler::Message(Profiler::MessageWarning,
			                  "Resources unreadable on first use, draw skipped: %s 0x%016" PRIx64
			                  " (materialization line %d; first 16 reported)",
			                  stage_name, hash, ShaderRecompiler::IR::LastIndirectImageFailureLine());
		}
		return false;
	}

	static void ReportShaderClock(ShaderType stage, const char* stage_name, uint64_t hash) {
		if (!ShaderFailureNonFatal()) {
			EXIT("S_MEMREALTIME needs shaderDeviceClock\n");
		}
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
			     ": S_MEMREALTIME needs shaderDeviceClock\n",
			     static_cast<uint32_t>(stage), hash);
		}
		Profiler::Add(Profiler::Counter::ShadersGaveUp);
		Profiler::Message(Profiler::MessageFailure,
		                  "Shader gave up: %s 0x%016" PRIx64 ": S_MEMREALTIME needs shaderDeviceClock",
		                  stage_name, hash);
	}

	// The disk cache key of a source: the session key and everything TranslateProgram reads.
	template <typename InputInfo>
	void BuildSourceKey(std::vector<uint8_t>& key, std::span<const uint32_t> code,
	                    const ShaderParams& params, const ShaderRecompiler::CompileOptions& options,
	                    const InputInfo& input_info) {
		if (disk_session_key.empty()) {
			// Built at first use: the device layer has set the emitter's switches by then.
			disk_session_key = ShaderDiskCache::SessionKey(disk_session);
		}
		ShaderDiskCache::ByteWriter writer;
		writer.bytes.reserve(disk_session_key.size() + code.size_bytes() +
		                     params.back_code.size_bytes() + 512u);
		writer.PutBytes(disk_session_key);
		// Resource plans depend on the GPU-sized written buffers switches (ResourceMaterialization).
		static const uint64_t sized_written = [] {
			std::string text;
			for (const char* name: {"KYTY_GPU_SIZED_WRITTEN", "KYTY_GPU_SIZED_WRITTEN_SHADERS"}) {
				const char* value = std::getenv(name);
				text += value != nullptr ? value : "-";
				text += ';';
			}
			return XXH3_64bits(text.data(), text.size());
		}();
		writer.Put(sized_written);
		// The ledger import parses old keys with the same writer (shaderLedgerImport.h).
		ShaderLedger::AppendSourceKeyTail(
		    writer,
		    ShaderLedger::SourceFields {.stage          = static_cast<uint32_t>(options.stage),
		                                .shader_hash    = options.shader_hash,
		                                .wave_size      = options.wave_size,
		                                .user_data_base = options.user_data_base,
		                                .bindless       = options.bindless_images,
		                                .non_fatal      = options.non_fatal,
		                                .user_data      = options.user_data,
		                                .code           = code,
		                                .back_code      = params.back_code},
		    KeyedInput(input_info));
		key = std::move(writer.bytes);
	}

	// v422: the viewport of a clip-disabled draw is per-draw data (BindingLayout::clip_space), so
	// the disk key keeps only that the transform is on. The ledger records the full values (its
	// record format and ids are unchanged), and all of a shader's viewports share one entry.
	template <typename InputInfo>
	static InputInfo KeyedInput(const InputInfo& input_info) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			if (input_info.clip_space.enabled && ShaderRecompiler::IR::ClipSpaceData()) {
				InputInfo keyed = input_info;
				keyed.clip_space = {.enabled = true};
				return keyed;
			}
		}
		// v423: likewise the colour export mapping (BindingLayout::export_mapping).
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			if (ShaderRecompiler::IR::ExportMappingData()) {
				InputInfo keyed = input_info;
				keyed.target_export_mapping = {};
				return keyed;
			}
		}
		return input_info;
	}

	// v391: the build-independent id of a permutation (PipelineRecipe::StableId).
	static PipelineRecipe::Hash128 PermutationStableId(uint64_t source_id, uint32_t push_data_cursor,
	                                                   uint64_t plan_fingerprint,
	                                                   const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		ShaderDiskCache::ByteWriter spec;
		ShaderDiskCache::AppendSpecialization(spec, specialization);
		return PipelineRecipe::StableId(source_id, push_data_cursor, plan_fingerprint, spec.bytes);
	}

	static void BuildPermutationKey(std::vector<uint8_t>& key, std::span<const uint8_t> source_key,
	                                const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                uint32_t push_data_cursor) {
		ShaderDiskCache::ByteWriter writer;
		writer.bytes.reserve(source_key.size() + 256u);
		writer.PutBytes(source_key);
		writer.Put<uint8_t>('P');
		ShaderDiskCache::AppendSpecialization(writer, specialization);
		writer.Put(push_data_cursor);
		key = std::move(writer.bytes);
	}

	// Stores a fresh entry; in verify mode compares it with the cached one first (`cached`, empty
	// when there was none) and stores it only when they differ.
	void StoreEntry(bool source, uint64_t hash, std::span<const uint8_t> key,
	                std::span<const uint8_t> cached, std::span<const uint8_t> payload) {
		const char* kind = source ? "source" : "permutation";
		if (!(source ? ShaderDiskCache::SourceRecordRoundTrips(payload)
		             : ShaderDiskCache::PermutationRecordRoundTrips(payload))) {
			if (disk->counters.round_trip_failures++ < 16) {
				LOGF("Shader disk cache: the %s entry of hash=0x%016" PRIx64
				     " does not decode to itself; not cached\n",
				     kind, hash);
			}
			return;
		}
		if (!cached.empty()) {
			disk->counters.verified++;
			if (std::ranges::equal(cached, payload)) {
				return;
			}
			if (disk->counters.verify_mismatches++ < 32) {
				LOGF("Shader disk cache: verify mismatch: %s entry of hash=0x%016" PRIx64
				     " (%zu bytes cached, %zu fresh)\n",
				     kind, hash, cached.size(), payload.size());
				Profiler::Message(Profiler::MessageFailure,
				                  "Shader disk cache: verify mismatch: %s entry of 0x%016" PRIx64,
				                  kind, hash);
			}
		}
		disk->Save(key, payload);
	}

	void CountDiskLookup() {
		if (++disk_lookups % 4096u == 0) {
			disk->LogTotals("so far");
		}
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		const CpuCost::Phase cpu_phase_("ProgramCache::Get");
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		if (SkipShaderRequested(params.hash)) {
			Profiler::Add(Profiler::Counter::SkippedSkipList);
			LogSkip("skip-list", StageLabel(stage), params.hash, 0);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": skip list", StageLabel(stage),
			                                params.hash);
			return ShaderProgram {};
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		std::optional<CpuCost::Phase> lookup_phase_;
		lookup_phase_.emplace("Program::Lookup");
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		lookup_key.function_code.clear();
		// A compute shader that gave up is dispatched again every frame, and expanding its
		// functions to find it in `unsupported` cost ~3 ms a dispatch (three such dispatches a
		// frame in the jungle). Remember the refusal by shader, code address, static state and
		// user data, and skip before expanding. KYTY_GAVE_UP_FAST=0 turns this off.
		// KYTY_DUMP_SHADER_HASHES="hash,..." with KYTY_DUMP_GAVE_UP=<folder>: the guest code of
		// those shaders, once, at their first lookup, even when the program comes from the disk
		// cache and is never translated in this session (diagnostics, for offline disassembly).
		if (static const char* dump_list = std::getenv("KYTY_DUMP_SHADER_HASHES"); dump_list != nullptr) {
			static const char*                  dump_folder = std::getenv("KYTY_DUMP_GAVE_UP");
			static std::mutex                   dump_mutex;
			static std::unordered_set<uint64_t> dumped;
			if (dump_folder != nullptr && !params.code.empty() &&
			    std::string_view(dump_list).find(fmt::format("{:016x}", params.hash)) !=
			        std::string_view::npos) {
				std::scoped_lock dump_lock(dump_mutex);
				if (dumped.insert(params.hash).second) {
					const auto path = fmt::format("{}/{}_{:016x}.bin", dump_folder, StageLabel(stage),
					                              params.hash);
					if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
						std::fwrite(params.code.data(), sizeof(uint32_t), params.code.size(), file);
						std::fclose(file);
					}
				}
			}
		}
		uint64_t gave_up_key = 0;
		[[maybe_unused]] uint64_t gave_up_shader_key = 0; // the shader as a whole (KYTY_GAVE_UP_SHADER_AFTER)
		[[maybe_unused]] uint32_t gave_up_limit      = 0;
		uint64_t                 fast_key = 0;
		ProgramsMap::value_type* fast_hit = nullptr;
		// When fast_hit came from fast_entries: when that entry was last validated.
		std::chrono::steady_clock::time_point fast_validated {};
		[[maybe_unused]] bool compute_memo_on = false;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// KYTY_COMPUTE_MEMO (toggle "computememo"): a dispatch with exactly the inputs of this
			// cache's previous lookup on this thread (shader, code address and size, wave size, every
			// user data word, static state), which found a program, takes that program without the
			// refusal and fast-program hashes and lookups. The previous lookup is forgotten whenever
			// one finds no program, so a refusal in between (which can refuse the whole shader) is
			// seen; the fast-program window still applies, from the program's last validation.
			auto& compute_memo = ComputeMemoSlot();
			static const bool fast_program_on = [] {
				const char* value = std::getenv("KYTY_FAST_PROGRAM");
				return value == nullptr || value[0] != '0';
			}();
			static const auto memo_window = std::chrono::milliseconds([] {
				const char* value = std::getenv("KYTY_FAST_PROGRAM_MS");
				return value != nullptr ? std::strtoul(value, nullptr, 10) : 1000ul;
			}());
			compute_memo_on = fast_program_on && SyncToggle::ComputeMemo();
			if (compute_memo_on && compute_memo.entry != nullptr && compute_memo.owner == this &&
			    compute_memo.hash == params.hash && compute_memo.base == params.Base() &&
			    compute_memo.code_size == params.code.size() &&
			    compute_memo.wave == input_info.wave_size && compute_memo.count == params.user_data_count &&
			    std::equal(user_data.begin(), user_data.end(), compute_memo.user_data.begin()) &&
			    compute_memo.static_state == lookup_key.static_state &&
			    std::chrono::steady_clock::now() - compute_memo.validated < memo_window) {
				fast_hit       = compute_memo.entry;
				fast_validated = compute_memo.validated;
			}
			compute_memo.entry = nullptr; // set again once this lookup finds its program
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) if (fast_hit == nullptr) {
			static const bool fast = [] {
				const char* value = std::getenv("KYTY_GAVE_UP_FAST");
				return value == nullptr || value[0] != '0';
			}();
			// The shader (code address, static state) as a whole: past KYTY_GAVE_UP_SHADER_AFTER
			// refusals with different inputs (default 16; 0: off) every dispatch of it is taken as
			// refused. The title's call shader fc0537b7278db60c gives up with every call target
			// set, and turning the camera brought new user data each dispatch: each expanded its
			// functions (and built hundreds of expansions) just to give up again.
			static const uint32_t gave_up_after = [] {
				const char* value = std::getenv("KYTY_GAVE_UP_SHADER_AFTER");
				return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 16u;
			}();
			const uint64_t shader_key = XXH3_64bits_withSeed(
			    lookup_key.static_state.data(), lookup_key.static_state.size() * sizeof(uint32_t),
			    params.hash ^ params.Base() ^ (static_cast<uint64_t>(params.code.size()) << 32u));
			gave_up_shader_key = shader_key;
			gave_up_limit      = gave_up_after;
			if (fast && gave_up_after != 0) {
				if (const auto it = gave_up_shaders.find(shader_key);
				    it != gave_up_shaders.end() && it->second >= gave_up_after) {
					Profiler::Add(Profiler::Counter::SkippedGaveUp);
					LogSkip("gave-up", StageLabel(stage), params.hash, 1);
					return ShaderProgram {};
				}
			}
			// By the user data the expansion depends on (ExpansionInputsHash), not all of it.
			const auto make_gave_up_key = [&] {
				return ExpansionInputsHash(params, user_data, input_info.wave_size, shader_key) | 1u;
			};
			// Whether these inputs were refused before (true: the dispatch is skipped).
			const auto refused_before = [&] {
				gave_up_key = make_gave_up_key();
				if (gave_up_fast.contains(gave_up_key)) {
					Profiler::Add(Profiler::Counter::SkippedGaveUp);
					LogSkip("gave-up", StageLabel(stage), params.hash, 0);
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": shader gave up",
					                                StageLabel(stage), params.hash);
					return true;
				}
				return false;
			};
			// KYTY_HOTPATH: a fast-program hit names a program these inputs translated to, so they
			// were never refused (a refusal never gets a program); the refusal lookup and its hash
			// run only on a miss.
			const bool refusal_after_fast = SyncToggle::HotPath();
			if (fast && !refusal_after_fast && refused_before()) {
				return ShaderProgram {};
			}
			// The same dispatch (shader, code address, static state, user data) as within the last
			// 50 ms finds its program without expanding the functions or hashing the expanded code
			// (~12 us a dispatch, 3 s in 20 s of the command thread in the jungle, 2026-10-16).
			// Entries are never erased, so the pointer stays valid. KYTY_FAST_PROGRAM=0 turns it off.
			static const bool fast_program = [] {
				const char* value = std::getenv("KYTY_FAST_PROGRAM");
				return value == nullptr || value[0] != '0';
			}();
			// The program depends on the user data only through the expanded functions
			// (ExpansionInputsHash), and on the static state.
			const auto make_fast_key = [&] {
				const uint64_t inputs = ExpansionInputsHash(
				    params, user_data, input_info.wave_size,
				    params.hash ^ params.Base() ^
				        (static_cast<uint64_t>(input_info.wave_size) << 56u) ^
				        (static_cast<uint64_t>(params.code.size()) << 32u));
				return XXH3_64bits_withSeed(lookup_key.static_state.data(),
				                            lookup_key.static_state.size() * sizeof(uint32_t),
				                            inputs) |
				       1u;
			};
			if (fast_program) {
				fast_key = make_fast_key();
				// A dispatch comes back once a frame (~100 ms at our frame rates), so the window is
				// longer than the expansion reuse's 50 ms. KYTY_FAST_PROGRAM_MS sets it.
				static const auto window = std::chrono::milliseconds([] {
					const char* value = std::getenv("KYTY_FAST_PROGRAM_MS");
					return value != nullptr ? std::strtoul(value, nullptr, 10) : 1000ul;
				}());
				if (const auto it = fast_entries.find(fast_key);
				    it != fast_entries.end() &&
				    std::chrono::steady_clock::now() - it->second.validated < window) {
					fast_hit       = it->second.entry;
					fast_validated = it->second.validated;
				}
			}
			if (fast && refusal_after_fast && fast_hit == nullptr && refused_before()) {
				return ShaderProgram {};
			}
			if (fast_hit == nullptr) {
				KYTY_PROFILER_BLOCK("ProgramCache::ExpandFunctions");
				ExpandShaderFunctions(params, user_data, input_info.wave_size,
				                      lookup_key.function_code);
				if (fast_key != 0) {
					// Kept under the registers this expansion consulted.
					fast_key = make_fast_key();
				}
			}
		}
		if (fast_hit == nullptr && unsupported.contains(lookup_key)) {
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				if (gave_up_key != 0) {
					// Keyed by the registers this expansion consulted.
					gave_up_key = ExpansionInputsHash(params, user_data, input_info.wave_size,
					                                  gave_up_shader_key) |
					              1u;
					if (gave_up_fast.insert(gave_up_key).second && gave_up_limit != 0) {
						auto& refusals = gave_up_shaders[gave_up_shader_key];
						if (++refusals == gave_up_limit) {
							std::fprintf(stderr, "[gave-up] cs %016" PRIx64 " at 0x%016" PRIx64
							             ": refused with %u inputs, skipped from now on\n",
							             params.hash, static_cast<uint64_t>(params.Base()), refusals);
							if (disk != nullptr) {
								const uint8_t marker = 1;
								disk->Save(ShaderRefusalKey(params.hash), std::span(&marker, 1));
							}
						}
					}
				}
			}
			Profiler::Add(Profiler::Counter::SkippedGaveUp);
			LogSkip("gave-up", StageLabel(stage), params.hash, 2);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": shader gave up",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		}
		KYTY_PROFILER_BLOCK("ProgramCache::Get");
		CpuCost::NoteProgram(params.hash);
		ProgramsMap::value_type* entry = nullptr;
		if (fast_hit) {
			entry = fast_hit;
		} else if (const auto found = programs.find(lookup_key); found != programs.end()) {
			entry = &*found;
			if (fast_key != 0) {
				// The title's async chains alone dispatch ~2,800 programs a frame, each with its own
				// user data: at 4,096 entries the map was cleared every frame or two and most
				// dispatches expanded their functions again. KYTY_FAST_PROGRAM_ENTRIES sets the cap.
				static const size_t fast_cap = [] {
					const char* value = std::getenv("KYTY_FAST_PROGRAM_ENTRIES");
					return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10))
					                        : size_t {65536};
				}();
				if (fast_entries.size() > fast_cap) {
					fast_entries.clear();
				}
				fast_validated         = std::chrono::steady_clock::now();
				fast_entries[fast_key] = {entry, fast_validated};
			}
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (compute_memo_on && entry != nullptr &&
			    fast_validated != std::chrono::steady_clock::time_point {}) {
				auto& memo     = ComputeMemoSlot();
				memo.owner     = this;
				memo.hash      = params.hash;
				memo.base      = params.Base();
				memo.code_size = params.code.size();
				memo.wave      = input_info.wave_size;
				memo.count     = params.user_data_count;
				std::copy(user_data.begin(), user_data.end(), memo.user_data.begin());
				memo.static_state = lookup_key.static_state;
				memo.entry        = entry;
				memo.validated    = fast_validated;
			}
		}
		lookup_phase_.reset();
		// Scoped to this call only: see GuestPageReadCache.
		GuestPageReadCache                           clean_read_cache;
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &clean_read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		static_assert(TRACKER_PAGE_SIZE == 4096u);
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &clean_read_cache;
		runtime.page_shift    = 12;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			for (uint32_t axis = 0; axis < 3u; axis++) {
				runtime.workgroup_size[axis] = input_info.threads_num[axis];
			}
		}
		if (entry != nullptr) {
			{
				// One zone per refresh, named by the program (kept with KYTY_PROFILE_ZONES=0).
				const CpuCost::Phase refresh_phase_("Program::SrtRefresh");
				Profiler::Phases refresh_zone;
				refresh_zone.SetText("%s %016" PRIx64, StageLabel(stage), params.hash);
				KYTY_PROFILER_PHASE(refresh_zone, "SRT refresh", profiler::colors::Amber300);
				// Compute: flat slots only the shader uses are left to the GPU when it wrote their
				// bytes; the renderer copies them on the GPU before the dispatch (SrtGpuFill).
				entry->second.resources.gpu_fills.clear();
				if (stage == ShaderType::Compute && SrtGpuFillEnabled()) {
					runtime.gpu_written = SrtBytesGpuWritten;
					runtime.gpu_fills   = &entry->second.resources.gpu_fills;
				}
				// KYTY_SRT_VERIFY: GPU-written bytes the host needs are read as the host holds
				// them and verified by the shader (SrtVerifyRead), not waited for. Cleared only
				// for a walk here: a prepared refresh brings its own.
				const bool verify_walk =
				    SrtVerifyProgram(entry->second.resource_plan, stage, params.hash);
				runtime.peek_gpu_dword = +[](uint64_t address) {
					DrawPipe::DrainIfPendingWrite(address, sizeof(uint32_t));
					const RenderContext::ReacquireScope reacquire;
					return Libs::LibKernel::Memory::PeekGpuDword(address);
				};
				t_srt_shader_hash = params.hash;
				bool materialized = false;
				if (TakePrepared(&entry->second)) {
					entry->second.prepared_serial = 0;
					// Refreshed in this draw's parallel window (PrepareStagesInParallel).
					static const bool verify = std::getenv("KYTY_STAGE_PREP_VERIFY") != nullptr;
					materialized             = true;
					if (verify) {
						VerifyPrepared(entry->second, runtime, stage, params.hash);
					}
				} else if (DrawPrep::Enabled() &&
				           TakeScanned(entry->second, stage, runtime, params.hash)) {
					// Prepared by the draw-prep scanner from the same inputs and bytes.
					materialized = true;
				} else {
					std::lock_guard walk_lock(entry->second.walk_mutex);
					entry->second.prepared_serial = 0;
					const auto walk_begin = std::chrono::steady_clock::now();
					entry->second.resources.srt_verify.clear();
					t_srt_verify = verify_walk ? &entry->second.resources.srt_verify : nullptr;
					t_srt_qualify = !verify_walk &&
					                SrtVerifyPlan(entry->second.resource_plan, stage) &&
					                SrtVerifyCandidate(params.hash);
					t_srt_qualify_read = false;
					t_srt_qualify_diff = false;
					materialized = ShaderRecompiler::IR::MaterializeResources(
					    entry->second.resource_plan, runtime, entry->second.resources,
					    entry->second.specialization);
					t_srt_verify = nullptr;
					if (t_srt_qualify) {
						t_srt_qualify = false;
						if (materialized) {
							SrtVerifyNoteWalk(params.hash);
						}
					}
					// KYTY_LOG_SLOW_SRT=1: refreshes over 1 ms (a guest read that waited for the GPU),
					// by shader: the first 4 of each, for up to 64 shaders.
					if (static const bool log_slow = std::getenv("KYTY_LOG_SLOW_SRT") != nullptr;
					    log_slow) {
						const auto ms = std::chrono::duration<double, std::milli>(
						                    std::chrono::steady_clock::now() - walk_begin)
						                    .count();
						if (ms > 1.0) {
							// Totals by shader; the top 12 every 300 slow refreshes.
							static std::unordered_map<uint64_t, std::pair<uint32_t, double>> slow;
							static uint32_t                                                  events = 0;
							auto& [n, total] = slow[params.hash];
							n++;
							total += ms;
							if (++events % 300 == 0) {
								std::vector<std::pair<uint64_t, std::pair<uint32_t, double>>> top(
								    slow.begin(), slow.end());
								std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
									return a.second.second > b.second.second;
								});
								std::string text = fmt::format("[slow-srt] after {} events:\n", events);
								for (size_t i = 0; i < top.size() && i < 12; i++) {
									text += fmt::format("  {:016x} n={} total={:.0f} ms\n", top[i].first,
									                    top[i].second.first, top[i].second.second);
								}
								std::fputs(text.c_str(), stderr);
								std::fflush(stderr);
							}
						}
					}
				}
				t_srt_shader_hash = 0;
				runtime.gpu_written = nullptr;
				runtime.gpu_fills   = nullptr;
				if (ShaderRecompiler::IR::SrtNativeStats stats;
				    ShaderRecompiler::IR::TakeSrtNativeReport(stats)) {
					LOGF("SRT native: %u plans compiled, %u failed, %" PRIu64 " KB of code, %" PRIu64
					     " instructions, %" PRIu64 " through the interpreter\n",
					     stats.plans, stats.failed, stats.bytes / 1024u, stats.instructions,
					     stats.interpreted);
				}
				if (t_srt_defer_tick != 0) {
					// KYTY_SRT_DEFER: bytes this refresh reads are downloading; the dispatch is
					// suspended and refreshed again once they land (SrtDeferRead).
					Profiler::CommandZone::Annotate("deferred %s %016" PRIx64 ": SRT bytes downloading",
					                                StageLabel(stage), params.hash);
					return ShaderProgram {};
				}
				if (!materialized) {
					// A descriptor source that cannot be read right now (memory the guest has
					// not mapped or filled yet) skips this draw rather than the session; the
					// next one re-evaluates from scratch.
					if (!ShaderFailureNonFatal()) {
						EXIT("shader resource materialization failed\n");
					}
					Profiler::Add(Profiler::Counter::SkippedResources);
					LogSkip("refresh", StageLabel(stage), params.hash,
					        ShaderRecompiler::IR::LastIndirectImageFailureLine());
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": resources unreadable",
					                                StageLabel(stage), params.hash);
					static std::atomic<uint32_t> reported = 0;
					{
						static std::mutex                   seen_mutex;
						static std::unordered_set<uint64_t> seen;
						std::lock_guard                     lock(seen_mutex);
						if (seen.size() < 256 && seen.insert(params.hash).second) {
							std::fprintf(stderr,
							             "[skip-resources] refresh: %s 0x%016" PRIx64
							             " (materialization line %d)\n",
							             StageLabel(stage), params.hash,
							             ShaderRecompiler::IR::LastIndirectImageFailureLine());
							std::fflush(stderr);
						}
					}
					if (reported.fetch_add(1) < 16) {
						LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
						     ": resource materialization failed (materialization line %d)\n",
						     static_cast<uint32_t>(stage), params.hash,
						     ShaderRecompiler::IR::LastIndirectImageFailureLine());
						Profiler::Message(Profiler::MessageWarning,
						                  "Resources unreadable, draw skipped: stage %u 0x%016" PRIx64
						                  " (materialization line %d; first 16 reported)",
						                  static_cast<uint32_t>(stage), params.hash,
						                  ShaderRecompiler::IR::LastIndirectImageFailureLine());
					}
					return ShaderProgram {};
				}
			}
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               SpecializationFits(candidate.specialization, entry->second.specialization,
				                                  entry->second.resources);
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				if (stage != ShaderType::Compute && !entry->second.plan_reads_noted &&
				    RenderStats::MultiDrawReplayConfigured()) {
					entry->second.plan_reads_noted = true;
					NotePlanUserDataReads(&entry->second.resources, entry->second.resource_plan);
				}
				return permutation->handle;
			}
		}

		if (fast_hit != nullptr) {
			// No permutation fits: the compile below needs the expanded code.
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				ExpandShaderFunctions(params, user_data, input_info.wave_size,
				                      lookup_key.function_code);
			}
		}
		// A translation on the workers (TranslationWorkers): the draw waits for it by being skipped.
		if (!translating.empty()) {
			if (const auto job = translating.find(lookup_key);
			    job != translating.end() && !job->second->done.load(std::memory_order_acquire)) {
				translation_skips++;
				Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
				                                StageLabel(stage), params.hash);
				return ShaderProgram {};
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		options.non_fatal = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;
		const auto compile_code = lookup_key.function_code.empty()
		                              ? params.code
		                              : std::span<const uint32_t>(lookup_key.function_code);
		// The shader disk cache (shaderDiskCache.h): a source entry stands in for the translation
		// of a source new to this session, a permutation entry for the compile. In verify mode
		// both are only compared with a fresh translation. A hit is not a compile: it is not
		// counted in ShadersCompiled or ShaderCompileTime.
		std::vector<uint8_t> source_key;
		std::vector<uint8_t> cached_source;
		std::vector<uint8_t> permutation_key;
		std::vector<uint8_t> cached_permutation;
		if (disk != nullptr) {
			Profiler::Phases disk_zone;
			disk_zone.SetText("%s 0x%016" PRIx64, stage_name, params.hash);
			KYTY_PROFILER_PHASE(disk_zone, "Shader disk cache", profiler::colors::CyanA700);
			BuildSourceKey(source_key, compile_code, params, options, input_info);
			if (ledger != nullptr) {
				ledger_source_id = ledger->RecordSource(
				    ShaderLedger::SourceFields {.stage          = static_cast<uint32_t>(options.stage),
				                                .shader_hash    = options.shader_hash,
				                                .wave_size      = options.wave_size,
				                                .user_data_base = options.user_data_base,
				                                .bindless       = options.bindless_images,
				                                .non_fatal      = options.non_fatal,
				                                .user_data      = options.user_data,
				                                .code           = compile_code,
				                                .back_code      = params.back_code},
				    input_info);
			}
			if (entry == nullptr) {
				CountDiskLookup();
				std::vector<uint8_t>          payload;
				ShaderDiskCache::SourceRecord record;
				if (!disk->Load(source_key, payload)) {
					disk->counters.source_misses++;
				} else if (disk_verify) {
					disk->counters.source_hits++;
					cached_source = std::move(payload);
				} else if (!ShaderDiskCache::DecodeSourceRecord(payload, record)) {
					disk->counters.rejected++;
					disk->counters.source_misses++;
				} else {
					disk->counters.source_hits++;
					if (record.unsupported) {
						static std::atomic<uint32_t> reported {0};
						if (reported.fetch_add(1) < 256) {
							std::fprintf(stderr, "[gave-up] %s 0x%016" PRIx64 ": in an earlier run (disk cache)\n",
							             stage_name, params.hash);
							std::fflush(stderr);
						}
						Profiler::Add(Profiler::Counter::ShadersGaveUp);
						Profiler::Message(Profiler::MessageFailure,
						                  "Shader gave up: %s 0x%016" PRIx64 ": in an earlier run (disk cache)",
						                  stage_name, params.hash);
						unsupported.insert(lookup_key);
						// A compute shader refused before: its other sources (other code lengths of
						// the same shader) are refused without translating them (ShaderRefusalKey).
						if (stage == ShaderType::Compute) {
							static std::mutex                   refused_mutex;
							static std::unordered_set<uint64_t> refused_written;
							std::lock_guard                     refused_lock(refused_mutex);
							if (refused_written.insert(params.hash).second) {
								const uint8_t marker = 1;
								disk->Save(ShaderRefusalKey(params.hash), std::span(&marker, 1));
							}
						}
						return ShaderProgram {};
					}
					if (record.uses_clock && !shader_clock) {
						ReportShaderClock(stage, stage_name, params.hash);
						unsupported.insert(lookup_key);
						return ShaderProgram {};
					}
					{
						std::unique_lock programs_lock(programs_mutex);
						entry = &*programs.try_emplace(lookup_key, std::move(record.plan)).first;
					}
					entry->second.plan_fingerprint = XXH3_64bits(payload.data(), payload.size());
					if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
						return ShaderProgram {};
					}
				}
			}
			if (entry != nullptr) {
				BuildPermutationKey(permutation_key, source_key, entry->second.specialization,
				                    push_data_cursor);
				std::vector<uint8_t>               payload;
				ShaderDiskCache::PermutationRecord record;
				bool                               loaded     = false;
				{
					const CpuCost::Phase load_phase_("Program::DiskLoad");
					loaded = disk->Load(permutation_key, payload);
				}
				bool                               compatible = false;
				ShaderRecompiler::IR::ResourceSpecialization matched;
				if (!loaded && !disk_verify) {
					// v425: a null buffer may use a recorded permutation made for its real stride;
					// the null variant is then neither compiled nor recorded.
					compatible = loaded = LoadCompatiblePermutation(
					    ledger_source_id, push_data_cursor, entry->second.plan_fingerprint, source_key,
					    entry->second.specialization, entry->second.resources, payload, matched);
					if (compatible) {
						disk->counters.permutation_compatible++;
					}
				}
				if (ledger != nullptr && !compatible) {
					ledger->RecordPermutation(ledger_source_id, push_data_cursor,
					                          entry->second.specialization,
					                          entry->second.plan_fingerprint);
				}
				if (!loaded) {
					disk->counters.permutation_misses++;
				} else if (disk_verify) {
					disk->counters.permutation_hits++;
					cached_permutation = std::move(payload);
				} else if (!ShaderDiskCache::DecodePermutationRecord(payload, record)) {
					disk->counters.rejected++;
					disk->counters.permutation_misses++;
				} else {
					disk->counters.permutation_hits++;
					Profiler::Message(Profiler::MessageInfo,
					                  "Shader from the disk cache: %s 0x%016" PRIx64 ", %zu SPIR-V words",
					                  stage_name, params.hash, record.spirv.size());
					if (ledger != nullptr && stage != ShaderType::Compute && entry->second.plan_fingerprint != 0 &&
					    !compatible) {
						PipelineRecipe::NoteModule(PermutationStableId(ledger_source_id, push_data_cursor,
						                                               entry->second.plan_fingerprint,
						                                               entry->second.specialization),
						                           record.spirv, false);
					}
					return AddPermutation(entry->second, stage_name, options, record.spirv,
					                      std::move(record.program), input_info, push_data_cursor,
					                      compatible ? &matched : nullptr);
				}
			}
		}

		// A compute shader with a shader-level refusal is not translated in another variant: some
		// take minutes on the GPU thread only to be refused (e6d76d24f59f8015, 10-26).
		if (stage == ShaderType::Compute && disk != nullptr &&
		    disk->Contains(ShaderRefusalKey(params.hash))) {
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1) < 64) {
				std::fprintf(stderr, "[gave-up] %s 0x%016" PRIx64 ": refused shader (disk cache), not translated\n",
				             stage_name, params.hash);
			}
			Profiler::Add(Profiler::Counter::ShadersGaveUp);
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		// The whole compile as one zone on the timeline, its phases below it.
		Profiler::Phases compile_zone;
		compile_zone.SetText("%s 0x%016" PRIx64, stage_name, params.hash);
		KYTY_PROFILER_PHASE(compile_zone, "Shader compile", profiler::colors::DeepOrangeA200);
		const auto compile_begin = std::chrono::steady_clock::now();
		ShaderRecompiler::TranslateResult translated;
		if (const auto job = translating.find(lookup_key); job != translating.end()) {
			// Done (the check above returned while it was not): take the worker's translation.
			const auto waited = std::chrono::duration<double, std::milli>(
			                        std::chrono::steady_clock::now() - job->second->queued)
			                        .count();
			if (translations_done++ < 32) {
				LOGF("ShaderTranslator: %s 0x%016" PRIx64 " ready after %.0f ms, %" PRIu64
				     " draws skipped so far while translating\n",
				     stage_name, params.hash, waited, translation_skips);
			}
			translated = std::move(job->second->result);
			translating.erase(job);
		} else if (defer_translation && stage != ShaderType::Compute && translators != nullptr) {
			auto job = std::make_shared<TranslationJob>();
			job->code.assign(compile_code.begin(), compile_code.end());
			job->back_code.assign(params.back_code.begin(), params.back_code.end());
			job->user_data.assign(user_data.begin(), user_data.end());
			job->options           = options;
			job->options.user_data = job->user_data;
			job->options.back_code = job->back_code;
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				job->vertex                   = input_info;
				job->options.input_info.vertex = &job->vertex;
			} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
				job->pixel                   = input_info;
				job->options.input_info.pixel = &job->pixel;
			}
			job->queued = std::chrono::steady_clock::now();
			translating.emplace(lookup_key, job);
			translators->Submit(std::move(job));
			translation_skips++;
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		} else {
			translated = TimedTranslate(compile_code, options);
		}
		if (translated.unsupported) {
			// Remember the refusal: a skipped shader is dispatched again every frame, and
			// re-deriving the same answer costs as much as a compile each time.
			if (disk != nullptr && entry == nullptr) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(true, false, nullptr, payload)) {
					StoreEntry(true, params.hash, source_key, cached_source, payload);
				}
			}
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		const bool uses_clock = UsesShaderClock(translated.program);
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		uint64_t                                          plan_fingerprint = 0;
		if (entry == nullptr) {
			plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
			if (disk != nullptr) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(false, uses_clock, &*plan, payload)) {
					plan_fingerprint = XXH3_64bits(payload.data(), payload.size());
					StoreEntry(true, params.hash, source_key, cached_source, payload);
				} else if (disk->counters.round_trip_failures++ < 16) {
					LOGF("Shader disk cache: the resource plan of hash=0x%016" PRIx64
					     " refers outside itself; not cached\n",
					     params.hash);
				}
			}
		}
		if (!shader_clock && uses_clock) {
			ReportShaderClock(stage, stage_name, params.hash);
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		if (entry == nullptr) {
			{
				std::unique_lock programs_lock(programs_mutex);
				entry = &*programs.try_emplace(lookup_key, std::move(*plan)).first;
			}
			entry->second.plan_fingerprint = plan_fingerprint;
			if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
				return ShaderProgram {};
			}
			if (disk != nullptr) {
				BuildPermutationKey(permutation_key, source_key, entry->second.specialization,
				                    push_data_cursor);
				if (ledger != nullptr) {
					ledger->RecordPermutation(ledger_source_id, push_data_cursor,
					                          entry->second.specialization,
					                          entry->second.plan_fingerprint);
				}
				if (disk_verify) {
					std::vector<uint8_t> payload;
					if (disk->Load(permutation_key, payload)) {
						cached_permutation = std::move(payload);
					}
				}
			}
		}
		auto compiled = CompileTranslated(options, std::move(translated),
		                                  entry->second.specialization, push_data_cursor);
		std::vector<uint8_t> permutation_payload;
		if (disk != nullptr) {
			ShaderDiskCache::EncodePermutationRecord(compiled.spirv, compiled.program,
			                                         permutation_payload);
		}
		if (ledger != nullptr && disk != nullptr && stage != ShaderType::Compute &&
		    entry->second.plan_fingerprint != 0) {
			PipelineRecipe::NoteModule(PermutationStableId(ledger_source_id, push_data_cursor,
			                                               entry->second.plan_fingerprint,
			                                               entry->second.specialization),
			                           compiled.spirv, false);
		}
		const auto handle = AddPermutation(entry->second, stage_name, options, compiled.spirv,
		                                   std::move(compiled.program), input_info,
		                                   push_data_cursor);
		{
			const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                         std::chrono::steady_clock::now() - compile_begin)
			                                         .count());
			Profiler::Add(Profiler::Counter::ShadersCompiled);
			if (GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed)) {
				ZeroCompileNote(ZeroCompileEvent::Translation, params.hash, 0.0);
			}
			if (Common::FloorStats::Enabled()) {
				Common::FloorStats::Add(Common::FloorStats::Get().compiles, 1);
				Common::FloorStats::Add(Common::FloorStats::Get().compile_ns,
				                        static_cast<uint64_t>(us) * 1000u);
			}
			Profiler::Add(Profiler::Counter::ShaderCompileTime, us);
			Profiler::Message(Profiler::MessageInfo,
			                  "Shader compiled: %s 0x%016" PRIx64 " in %.1f ms, %zu SPIR-V words",
			                  stage_name, params.hash, static_cast<double>(us) / 1000.0,
			                  last_spirv_words);
		}
		compile_zone.End();
		if (disk != nullptr) {
			// After AddPermutation: SPIR-V that fails validation is never stored.
			StoreEntry(false, params.hash, permutation_key, cached_permutation,
			           permutation_payload);
		}
		return handle;
	}

	// `shared` (KYTY_ASYNC_THREAD): the async-compute thread's cache, for compute shaders only,
	// using the main cache's disk store.
	ProgramCache(vk::Device device, bool shader_clock, bool bindless_images,
	             ShaderDiskCache::SessionInfo session, std::filesystem::path disk_directory,
	             const ProgramCache* shared = nullptr)
	    : device(device), shader_clock(shader_clock), bindless_images(bindless_images),
	      disk_session(session), disk_verify(ShaderDiskCache::VerifyEnabled()) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		if (shared != nullptr) {
			disk       = shared->disk;
			ledger     = shared->ledger;
			disk_owner = false;
			// Program ids key the compute pipelines (PipelineCache::GetComputePipeline): this
			// cache's must not collide with the main cache's.
			next_shader_id = uint64_t {1} << 48u;
			return;
		}
		if (Config::GraphicsDebugDumpEnabled()) {
			// The dumps are written while translating: every shader has to be translated.
			PipelineCacheLog("Shader disk cache: off (graphics debug dump)");
		} else {
			disk = ShaderDiskCache::Store::Open(std::move(disk_directory));
			if (disk != nullptr) {
				GplPrefetchSetStore(disk->Directory()); // where the prefetch reads compiled records
				const auto title_id = PipelineCacheTitleId();
				ledger = ShaderLedger::Ledger::Open(
				    title_id.empty() ? disk->Directory() : ShaderListDirectory() / title_id);
			}
		}
		if (g_async_translation) {
			// KYTY_TRANSLATION_THREADS=<n> (default 2).
			const char* value   = std::getenv("KYTY_TRANSLATION_THREADS");
			const auto  threads = std::max<uint32_t>(
			    value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 2u, 1u);
			translators = std::make_unique<TranslationWorkers>(threads);
			PipelineCacheLog("Shader translation: graphics shaders of 3D draws on {} worker threads",
			                 threads);
		}
		if (g_stage_prep_parallel) {
			stage_prep = std::make_unique<StagePrepWorker>();
			PipelineCacheLog("Stage preparation: vertex and pixel SRT refresh in parallel");
		}
	}

	// The startup precompile (plan 4.12): every recipe in the shader ledger is translated again
	// and stored in the disk cache under this session's keys, on background threads, so that the
	// game finds the entries instead of translating in the middle of play. Recipes whose entries
	// exist already only count as done. KYTY_PRECOMPILE=0 turns it off; KYTY_PRECOMPILE_THREADS=n.
	struct PrecompileJob {
		std::vector<ShaderLedger::SourceRecipe>                    sources;
		std::vector<ShaderLedger::PermutationRecipe>               permutations;
		std::unordered_map<uint64_t, std::vector<size_t>>          by_source; // permutation indices
		std::atomic<size_t>                                        next {0};
		std::atomic<uint32_t>                                      running {0};
		std::atomic<uint32_t>                                      stale {0}; // other plans
		// Recipes this build cannot use besides `stale`: refused sources and their permutations,
		// permutations whose specialization does not decode, permutations without their source.
		std::atomic<uint32_t>                                      unusable {0};
		// v420 carry-over: stale permutations replayed against this build's plan, and the ones
		// that did not fit (dropped; the recompiler gave up on them).
		std::atomic<uint32_t>                                      carried {0};
		std::atomic<uint32_t>                                      carry_failed {0};
		std::atomic<uint32_t>                                      carry_duplicates {0};
		std::atomic<uint32_t>                                      rekeyed {0}; // v438f
		std::atomic<uint32_t>                                      aliased {0};
		std::chrono::steady_clock::time_point                      begin;
		// Step D4: graphics permutations whose libraries the driver cache lacks (GplWarm*).
		std::mutex                                                 warm_mutex;
		std::vector<std::pair<ShaderType, std::vector<uint8_t>>>   warm_queue;
	};

	void StartPrecompile() {
		if (ledger == nullptr || disk == nullptr || !disk_owner || !precompile_threads.empty()) {
			return;
		}
		if (const char* value = std::getenv("KYTY_PRECOMPILE"); value != nullptr && value[0] == '0') {
			PipelineCacheLog("Shader precompile: off (KYTY_PRECOMPILE=0)");
			return;
		}
		// On this thread, before any worker reads it (Get builds it lazily otherwise).
		if (disk_session_key.empty()) {
			disk_session_key = ShaderDiskCache::SessionKey(disk_session);
		}
		precompile_threads.emplace_back([this] { RunPrecompile(); });
	}

	// The import of the disk cache's keys into the ledger (plan 4.13): the first time a ledger has
	// none, forced with KYTY_LEDGER_IMPORT=1, never with KYTY_LEDGER_IMPORT=0.
	[[nodiscard]] bool ImportWanted() const {
		const char* value = std::getenv("KYTY_LEDGER_IMPORT");
		if (value != nullptr && value[0] == '0') {
			return false;
		}
		return !ledger->Imported() || (value != nullptr && value[0] == '1');
	}

	// The precompile's first thread: the import if wanted, then the replay on `threads` threads
	// (this one included).
	void RunPrecompile() {
		Common::LowerBackgroundThreadPriority();
		auto&    progress = GetShaderPrecompileProgress();
		uint32_t threads  = std::max(2u, std::thread::hardware_concurrency() / 3u);
		if (const char* value = std::getenv("KYTY_PRECOMPILE_THREADS"); value != nullptr) {
			threads = std::max(1u, static_cast<uint32_t>(std::strtoul(value, nullptr, 10)));
		}
		// KYTY_SHADER_CACHE_PRUNE=1: after the replay, entries of other sessions (older builds,
		// other recompiler switches) are moved to <cache dir>_old, in the background and without
		// a counter (their recipes are in the ledger); nothing is deleted.
		const char* prune_value = std::getenv("KYTY_SHADER_CACHE_PRUNE");
		const bool  prune       = prune_value != nullptr && prune_value[0] == '1';
		if (ImportWanted()) {
			progress.import_done.store(0);
			progress.import_total.store(0);
			progress.importing.store(true, std::memory_order_release);
			progress.active.store(true, std::memory_order_release);
			PipelineCacheLog("Shader ledger import: reading the disk cache's keys into the ledger on {} threads",
			                 threads);
			const auto stats = ShaderLedger::ImportDiskCache(
			    disk->Directory(), *ledger, threads, precompile_stop, progress.import_done,
			    progress.import_total, disk_session_key, std::filesystem::path(), true);
			PipelineCacheLog("Shader ledger import: {}", stats.Summary());
			if (!stats.stopped) {
				const auto summary = stats.Summary();
				ledger->MarkImported({reinterpret_cast<const uint8_t*>(summary.data()), summary.size()});
			}
			ledger->Reload();
			progress.importing.store(false, std::memory_order_release);
		}
		if (PipelineRecipe::ModuleImportWanted()) {
			// v394: once, the modules of earlier sessions that recipes name (pipelineRecipe.h).
			progress.import_done.store(0);
			progress.import_total.store(0);
			progress.importing.store(true, std::memory_order_release);
			progress.active.store(true, std::memory_order_release);
			const auto begin   = std::chrono::steady_clock::now();
			const auto unknown = PipelineRecipe::UnknownModuleCount();
			const auto found   = ShaderLedger::ImportModuleMap(disk->Directory(), threads, precompile_stop,
			                                                   progress.import_done, progress.import_total,
			                                                   disk_session_key);
			const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
			PipelineCacheLog("Pipeline modules: {} of {} earlier-session recipe stages found in the disk cache, {:.1f} s",
			                 found, unknown, seconds);
			std::fprintf(stderr, "[pipeline-modules] %llu of %zu earlier-session recipe stages found in the disk cache (%u files), %.1f s\n",
			             static_cast<unsigned long long>(found), unknown, progress.import_total.load(), seconds);
			progress.importing.store(false, std::memory_order_release);
		}
		progress.ledger_base.store(progress.ledger_total.load());
		Replay(threads);
		if (prune && !precompile_stop.load(std::memory_order_acquire)) {
			auto prune_to = disk->Directory();
			prune_to += "_old";
			std::atomic<uint32_t> done {0};
			std::atomic<uint32_t> total {0};
			const auto stats = ShaderLedger::ImportDiskCache(disk->Directory(), *ledger, 2,
			                                                 precompile_stop, done, total,
			                                                 disk_session_key, prune_to, false);
			PipelineCacheLog("Shader cache clean-up: {} files checked, {} entries of other sessions moved to {} ({:.1f} s{})",
			                 stats.files, stats.moved, prune_to.generic_string(), stats.seconds,
			                 stats.stopped ? ", stopped" : "");
		}
	}

	// The replay of the ledger's recipes on `threads` threads (this one included).
	void Replay(uint32_t threads) {
		auto& progress = GetShaderPrecompileProgress();
		auto job = std::make_shared<PrecompileJob>();
		ledger->TakeRecipes(job->sources, job->permutations);
		{
			std::lock_guard lock(recorded_mutex);
			for (const auto& recipe: job->permutations) {
				recorded_by_source[recipe.source_id].push_back(recipe);
			}
		}
		if (job->sources.empty() || precompile_stop.load(std::memory_order_acquire)) {
			progress.finished_at.store(ShaderPrecompileProgress::Now(), std::memory_order_release);
			progress.active.store(false, std::memory_order_release);
			RunGplWarm(*job); // nothing queued: marks the warm-up checked
			return;
		}
		std::unordered_set<uint64_t> source_ids;
		for (const auto& source: job->sources) {
			source_ids.insert(source.id);
		}
		uint32_t total = static_cast<uint32_t>(job->sources.size());
		for (size_t i = 0; i < job->permutations.size(); i++) {
			const auto id = job->permutations[i].source_id;
			if (source_ids.contains(id)) {
				job->by_source[id].push_back(i);
				total++;
			} else {
				job->unusable.fetch_add(1, std::memory_order_relaxed);
			}
		}
		progress.done.store(0);
		progress.compiled.store(0);
		progress.total.store(total);
		progress.finished_at.store(0);
		progress.active.store(true, std::memory_order_release);
		job->begin = std::chrono::steady_clock::now();
		PipelineCacheLog("Shader precompile: {} sources and {} permutations from the ledger on {} threads",
		                 job->sources.size(), total - job->sources.size(), threads);
		job->running.store(threads);
		std::vector<std::thread> workers;
		for (uint32_t t = 1; t < threads; t++) {
			workers.emplace_back([this, job] { PrecompileWorker(*job); });
		}
		PrecompileWorker(*job);
		for (auto& worker: workers) {
			worker.join();
		}
	}

	void StopPrecompile() {
		precompile_stop.store(true, std::memory_order_release);
		for (auto& thread: precompile_threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}
		precompile_threads.clear();
	}

	void PrecompileWorker(PrecompileJob& job) {
		Common::LowerBackgroundThreadPriority();
		auto& progress = GetShaderPrecompileProgress();
		for (;;) {
			if (precompile_stop.load(std::memory_order_acquire)) {
				break;
			}
			progress.YieldToForeground(); // v438l
			const size_t index = job.next.fetch_add(1);
			if (index >= job.sources.size()) {
				break;
			}
			const auto& recipe = job.sources[index];
			const auto  found  = job.by_source.find(recipe.id);
			static const std::vector<size_t> none;
			const auto& permutations = found != job.by_source.end() ? found->second : none;
			switch (recipe.kind) {
				case ShaderLedger::InputKind::Vertex: {
					auto info = recipe.vertex;
					PrecompileSource(job, recipe, permutations, info);
					break;
				}
				case ShaderLedger::InputKind::Pixel: {
					auto info = recipe.pixel;
					PrecompileSource(job, recipe, permutations, info);
					break;
				}
				case ShaderLedger::InputKind::Compute: {
					auto info = recipe.compute;
					PrecompileSource(job, recipe, permutations, info);
					break;
				}
				default: progress.done.fetch_add(1 + static_cast<uint32_t>(permutations.size())); break;
			}
		}
		if (job.running.fetch_sub(1) == 1) {
			const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
			                                                   job.begin)
			                         .count();
			PipelineCacheLog("Shader precompile: {} of {} done, {} translated or compiled, {} "
			                 "permutations of another resource plan skipped, {} other recipes unusable, in {:.1f} s{}",
			                 progress.done.load(), progress.total.load(), progress.compiled.load(),
			                 job.stale.load(), job.unusable.load(), seconds,
			                 precompile_stop.load() ? " (stopped)" : "");
			std::fprintf(stderr, "[ledger-carry] %u stale permutations carried over to this build's plan, %u did not fit, %u duplicates; %u old module ids now name this build's module\n",
			             job.carried.load(), job.carry_failed.load(), job.carry_duplicates.load(), job.aliased.load());
			std::fprintf(stderr, "[ledger-rekey] %u recorded permutations replayed at this build's push cursors / without zero-stride (KYTY_LEDGER_REKEY=0: off)\n",
			             job.rekeyed.load());
			// "Shaders N" counts what this build can use: the recipes it cannot stay in the
			// ledger for other builds, but leave the count (Mac 11-13: the number included
			// many obsolete shaders). Only after a complete replay, when every one was checked.
			if (!precompile_stop.load(std::memory_order_acquire)) {
				const uint32_t unusable = job.stale.load() + job.unusable.load() + job.carry_failed.load();
				const uint32_t base     = progress.ledger_base.load(std::memory_order_relaxed);
				progress.ledger_base.store(base - std::min(base, unusable), std::memory_order_relaxed);
				ledger->SetUnusable(unusable);
				const auto all = static_cast<uint32_t>(job.sources.size() + job.permutations.size());
				std::fprintf(stderr, "[shader-precompile] %u of %u recipes usable by this build (%u of another resource plan, %u other)\n",
				             all - std::min(all, unusable), all, job.stale.load(), job.unusable.load());
			}
			progress.finished_at.store(ShaderPrecompileProgress::Now(), std::memory_order_release);
			progress.active.store(false, std::memory_order_release);
			RunGplWarm(job);
		}
	}

	// Step D4: the queued permutations' libraries into the driver cache, on this thread and the
	// other precompile threads' count of helpers, after the replay (background; the game runs).
	void RunGplWarm(PrecompileJob& job) {
		auto& progress = GetShaderPrecompileProgress();
		std::vector<std::pair<ShaderType, std::vector<uint8_t>>> queue;
		{
			std::lock_guard lock(job.warm_mutex);
			queue.swap(job.warm_queue);
		}
		struct Finish {
			ShaderPrecompileProgress& progress;
			~Finish() {
				progress.gpl_warm_active.store(false, std::memory_order_release);
				progress.gpl_warm_finished.store(true, std::memory_order_release);
			}
		} finish {progress};
		if (queue.empty()) {
			return;
		}
		InstallWarmCrashLogger();
		progress.gpl_warm_done.store(0, std::memory_order_relaxed);
		progress.gpl_warm_total.store(static_cast<uint32_t>(queue.size()), std::memory_order_relaxed);
		progress.gpl_warm_active.store(true, std::memory_order_release);
		const auto begin = std::chrono::steady_clock::now();
		const size_t queued_total = queue.size(); // the progress lines' total (compute + graphics)
		std::atomic<size_t>   next {0};
		std::atomic<uint32_t> made {0};
		std::atomic<uint32_t> failed {0};
		std::atomic<uint32_t> no_record {0};
		std::atomic<uint32_t> compute_made {0};
		std::atomic<uint32_t> undecodable {0};
		// Past the startup wait (play started, or it gave up) only half the threads go on.
		const uint32_t        keep = std::max(1u, std::thread::hardware_concurrency() / 2u);
		const auto            work = [&](uint32_t worker) {
			Common::LowerBackgroundThreadPriority();
			for (;;) {
				if (precompile_stop.load(std::memory_order_acquire)) {
					break;
				}
				if (worker >= keep && progress.startup_wait_done.load(std::memory_order_acquire)) {
					break;
				}
				progress.YieldToForeground(); // v438l
				const size_t index = next.fetch_add(1);
				if (index >= queue.size()) {
					break;
				}
				const auto& [stage, key] = queue[index];
				// The whole item is in flight (v437t: a crash at ~19,000 also without the library).
				GplWarmInFlight(worker, key);
				static const bool trace = [] {
					const char* value = std::getenv("KYTY_WARM_TRACE");
					return value != nullptr && value[0] == '1';
				}();
				if (trace) {
					std::fprintf(stderr, "[warm-trace] %zu stage %u key %016llx\n", index, static_cast<unsigned>(stage),
					             static_cast<unsigned long long>(XXH3_64bits(key.data(), key.size())));
					std::fflush(stderr);
				}
				std::vector<uint8_t>               payload;
				ShaderDiskCache::PermutationRecord record;
				if (!disk->Load(key, payload)) {
					GplWarmInFlight(worker, {});
					no_record.fetch_add(1); // not compiled yet: warmed at a later start
				} else if (!ShaderDiskCache::DecodePermutationRecord(payload, record)) {
					GplWarmInFlight(worker, {});
					undecodable.fetch_add(1);
					GplWarmDone(key); // the same bytes every start
				} else if ((trace ? (std::fprintf(stderr, "[warm-trace] %zu shader %016llx\n", index,
				                                  static_cast<unsigned long long>(record.program.shader_hash)),
				                     std::fflush(stderr), 0)
				                  : 0),
				           GplWarmLibraryNow(stage, record.spirv, record.program)) {
					GplWarmInFlight(worker, {});
					GplWarmDone(key);
					if (stage == ShaderType::Compute) {
						compute_made.fetch_add(1);
					}
					const auto n = made.fetch_add(1) + 1;
					if (n % 1000 == 0) {
						std::printf("[gpl-warm] %u of %zu items in %.0f s\n", n, queued_total,
						            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
						std::fflush(stdout);
					}
				} else {
					GplWarmInFlight(worker, {});
					failed.fetch_add(1);
					GplWarmDone(key); // not warmable (stage, or the driver refused it): not retried
				}
				progress.gpl_warm_done.fetch_add(1, std::memory_order_relaxed);
			}
		};
		// Suspects (in flight with others at a crash) first, one at a time: a crash among them
		// leaves only that one in flight, and the next start skips it.
		std::vector<std::pair<ShaderType, std::vector<uint8_t>>> suspects;
		{
			std::lock_guard lock(g_warm_mutex);
			if (!g_suspects.empty()) {
				std::vector<std::pair<ShaderType, std::vector<uint8_t>>> rest;
				for (auto& item: queue) {
					(g_suspects.contains(GplWarmId(item.second)) ? suspects : rest).push_back(std::move(item));
				}
				queue.swap(rest);
			}
		}
		if (!suspects.empty()) {
			auto all = std::move(queue);
			queue    = std::move(suspects);
			work(0); // single thread: this queue alone
			std::printf("[gpl-warm] %zu suspects retried one at a time\n", queue.size());
			queue = std::move(all);
			next.store(0);
		}
		// Compute pipelines next, one at a time, with the caches written every 100: the driver's
		// compiler ended the process on predicted compute permutations (fresh run A, v437u), and
		// alone in flight such an item is skipped at the next start at once; the caches keep what
		// was made before it.
		{
			std::vector<std::pair<ShaderType, std::vector<uint8_t>>> compute;
			std::vector<std::pair<ShaderType, std::vector<uint8_t>>> graphics;
			for (auto& item: queue) {
				(item.first == ShaderType::Compute ? compute : graphics).push_back(std::move(item));
			}
			// v438 (fr6, 11-14): the crashing permutations were the predictor's (an image dimension the
			// instruction does not address, fixed in listbuild); one thread made a first start with
			// ~6,000 compute permutations take ~25 min. Now on the warm-up's threads, in chunks of
			// 1,000 with the caches written after each: a crash leaves several in flight, and the next
			// start retries them one at a time first (suspects). KYTY_WARM_COMPUTE_SINGLE=1: one thread.
			if (!compute.empty()) {
				static const bool single = [] {
					const char* value = std::getenv("KYTY_WARM_COMPUTE_SINGLE");
					return value != nullptr && value[0] == '1';
				}();
				const auto     count   = compute.size();
				const size_t   chunk   = single ? 100 : 1000;
				const uint32_t cthreads = single ? 1u : std::max(1u, GplWarmThreads());
				for (size_t first = 0; first < count && !precompile_stop.load(std::memory_order_acquire); first += chunk) {
					queue.assign(std::make_move_iterator(compute.begin() + static_cast<std::ptrdiff_t>(first)),
					             std::make_move_iterator(compute.begin() + static_cast<std::ptrdiff_t>(std::min(count, first + chunk))));
					next.store(0);
					std::vector<std::thread> chelpers;
					for (uint32_t i = 1; i < cthreads; i++) {
						chelpers.emplace_back(work, i);
					}
					work(0);
					for (auto& helper: chelpers) {
						helper.join();
					}
					if (g_warm_save_hook) {
						g_warm_save_hook();
					}
				}
				std::printf("[gpl-warm] %zu compute permutations warmed on %u threads\n", count, cthreads);
				std::fflush(stdout);
			}
			queue = std::move(graphics);
			next.store(0);
		}
		const uint32_t        threads = std::max(1u, GplWarmThreads());
		std::vector<std::thread> helpers;
		for (uint32_t i = 1; i < threads; i++) {
			helpers.emplace_back(work, i);
		}
		work(0);
		for (auto& helper: helpers) {
			helper.join();
		}
		std::printf("[gpl-warm] %u compute pipelines made; %u libraries made, %u not (%u no compiled record, %u undecodable, %u not warmable), of %zu queued, on %u threads, in %.1f s%s\n",
		            compute_made.load(), made.load() - compute_made.load(), no_record.load() + undecodable.load() + failed.load(), no_record.load(),
		            undecodable.load(), failed.load(), queue.size(), threads,
		            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
		            precompile_stop.load() ? " (stopped)" : "");
		std::fflush(stdout);
		// Step D7: the caches written now, so the record holds the warm-up (not only at exit).
		if (made.load() != 0 && g_warm_save_hook) {
			g_warm_save_hook();
		}
	}

	template <typename InputInfo>
	void PrecompileSource(PrecompileJob& job, const ShaderLedger::SourceRecipe& recipe,
	                      const std::vector<size_t>& permutations, InputInfo& input_info) {
		auto& progress = GetShaderPrecompileProgress();
		ShaderParams params;
		params.code            = recipe.code;
		params.back_code       = recipe.back_code;
		params.hash            = recipe.shader_hash;
		params.user_data_count = static_cast<uint32_t>(std::min<size_t>(recipe.user_data.size(),
		                                                                 params.user_data.size()));
		std::copy_n(recipe.user_data.begin(), params.user_data_count, params.user_data.begin());
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		ShaderRecompiler::CompileOptions options;
		options.stage           = static_cast<ShaderType>(recipe.stage);
		options.shader_hash     = recipe.shader_hash;
		options.user_data       = std::span(params.user_data).first(params.user_data_count);
		options.back_code       = params.back_code;
		options.dump_ir         = false;
		options.early_dump      = false;
		options.dump_label      = "Shader precompile";
		options.input_info      = stage_input;
		options.wave_size       = recipe.wave_size;
		options.user_data_base  = recipe.user_data_base;
		// As Get sets them for this session, so the keys are the ones the game will ask for.
		options.non_fatal       = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;

		if (disk->Contains(ShaderRefusalKey(recipe.shader_hash))) {
			progress.done.fetch_add(1 + static_cast<uint32_t>(permutations.size()));
			job.unusable.fetch_add(1 + static_cast<uint32_t>(permutations.size()), std::memory_order_relaxed);
			return;
		}
		std::vector<uint8_t> source_key;
		BuildSourceKey(source_key, recipe.code, params, options, input_info);
		std::vector<uint8_t> payload;
		ShaderRecompiler::TranslateResult translated;
		bool have_translation = false;
		bool usable           = true;
		uint64_t fingerprint = 0; // of this build's source entry: see PermutationRecipe
		// v432: the plan's resource info, to canonicalize recorded specializations
		// (IR::CanonicalizeDynamicFields).
		std::optional<ShaderRecompiler::IR::ShaderInfo> plan_info;
		if (disk->Load(source_key, payload)) {
			ShaderDiskCache::SourceRecord record;
			usable = ShaderDiskCache::DecodeSourceRecord(payload, record) && !record.unsupported;
			fingerprint = XXH3_64bits(payload.data(), payload.size());
			if (usable) {
				plan_info = record.plan.info;
			}
		} else {
			bool fatal = false;
			translated = ReplayTranslate(recipe.code, options, &fatal);
			progress.compiled.fetch_add(1);
			if (fatal) {
				// Recorded for this source key only, so later starts skip it; no shader-wide refusal.
				std::vector<uint8_t> record;
				if (ShaderDiskCache::EncodeSourceRecord(true, false, nullptr, record)) {
					StoreEntry(true, recipe.shader_hash, source_key, {}, record);
				}
				usable = false;
			} else if (translated.unsupported) {
				std::vector<uint8_t> record;
				if (ShaderDiskCache::EncodeSourceRecord(true, false, nullptr, record)) {
					StoreEntry(true, recipe.shader_hash, source_key, {}, record);
				}
				const uint8_t marker = 1;
				disk->Save(ShaderRefusalKey(recipe.shader_hash), std::span(&marker, 1));
				usable = false;
			} else {
				const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
				plan_info       = plan.info;
				std::vector<uint8_t> record;
				if (ShaderDiskCache::EncodeSourceRecord(false, UsesShaderClock(translated.program),
				                                        &plan, record)) {
					fingerprint = XXH3_64bits(record.data(), record.size());
					StoreEntry(true, recipe.shader_hash, source_key, {}, record);
				}
				have_translation = true;
			}
		}
		progress.done.fetch_add(1);
		if (!usable) {
			job.unusable.fetch_add(1 + static_cast<uint32_t>(permutations.size()), std::memory_order_relaxed);
		}
		// v420 carry-over (Mac 11-13: every recompiler change dropped most of the list): a
		// permutation specialized against another plan (an older recompiler) is replayed against
		// this one, by position, when it is not a duplicate of one already recorded for this plan.
		// The current build translates it again, so a specialization the runtime asks for gives
		// exactly the module it would have built; one that does not fit is never asked for. Any
		// fatal error of the recompiler while compiling it only drops it (RecoverableExitScope).
		// KYTY_LEDGER_CARRY=0 turns it off.
		static const bool carry_enabled = [] {
			const char* value = std::getenv("KYTY_LEDGER_CARRY");
			return (value == nullptr || value[0] != '0') && Common::RecoverableExitsSupported();
		}();
		std::unordered_set<uint64_t> seen; // (cursor, specialization bytes) for this source
		std::unordered_set<uint64_t> replayed_keys; // v424: permutation keys replayed for this source
		const auto spec_key = [](const ShaderLedger::PermutationRecipe& p) {
			return XXH3_64bits_withSeed(p.specialization.data(), p.specialization.size(), p.push_data_cursor);
		};
		// v438f re-key (Fable 11-15, Mac's v438c play: 875 translations of permutations his ledger
		// held under the old keys): this build asks for a pixel shader at TopAnchor (v434) and for
		// the stage before it at 0/7 or NoPush, and never specializes a buffer as zero-stride the
		// way builds before v424 recorded null V#s. A recording under another cursor or plan is
		// replayed at the cursors this build uses, with zero_stride_oob cleared, and recorded
		// under them; KYTY_LEDGER_REKEY=0 replays it as recorded.
		static const bool rekey_enabled = [] {
			const char* value = std::getenv("KYTY_LEDGER_REKEY");
			return value == nullptr || value[0] != '0';
		}();
		struct Work {
			size_t   index;
			uint32_t cursor;
			bool     rekey;
			bool     first;
		};
		std::vector<Work> works;
		works.reserve(permutations.size());
		for (const auto index: permutations) {
			const auto& p        = job.permutations[index];
			const auto  recorded = p.push_data_cursor;
			std::array<uint32_t, 2> cursors {recorded, recorded};
			uint32_t                count = 1;
			if (rekey_enabled) {
				using PD = ShaderRecompiler::IR::PushData;
				if (options.stage == ShaderType::Pixel && recorded != PD::TopAnchor) {
					cursors[0] = PD::TopAnchor;
				} else if (options.stage == ShaderType::Vertex && recorded != 0 && recorded != PD::NoPush) {
					cursors = {0u, PD::NoPush};
					count   = 2;
				} else if (options.stage == ShaderType::Mesh && recorded != PD::MeshDrawDwordCount &&
				           recorded != PD::NoPush) {
					cursors = {PD::MeshDrawDwordCount, PD::NoPush};
					count   = 2;
				}
			}
			for (uint32_t c = 0; c < count; c++) {
				const bool moved = cursors[c] != recorded;
				works.push_back({index, cursors[c],
				                 rekey_enabled && (moved || p.plan_fingerprint != fingerprint), c == 0});
			}
		}
		const auto done_one = [&](const Work& work) {
			if (work.first) {
				progress.done.fetch_add(1);
			}
		};
		std::atomic<uint32_t>& rekeyed = job.rekeyed;
		if (carry_enabled) {
			for (const auto index: permutations) {
				if (job.permutations[index].plan_fingerprint == fingerprint) {
					seen.insert(spec_key(job.permutations[index]));
				}
			}
		}
		for (const auto& work: works) {
			progress.YieldToForeground(); // v438l
			const auto index  = work.index;
			const auto cursor = work.cursor;
			if (!usable || precompile_stop.load(std::memory_order_acquire)) {
				done_one(work);
				continue;
			}
			const auto& permutation = job.permutations[index];
			const bool  carry       = permutation.plan_fingerprint != fingerprint || work.rekey;
			if (carry && (!carry_enabled || fingerprint == 0)) {
				// Specialized against another plan (an older recompiler): it may not fit this
				// one, and this build would never ask for it with the same key anyway.
				job.stale.fetch_add(1, std::memory_order_relaxed);
				done_one(work);
				continue;
			}
			if (carry && !seen.insert(XXH3_64bits_withSeed(permutation.specialization.data(), permutation.specialization.size(), cursor)).second) {
				job.carry_duplicates.fetch_add(1, std::memory_order_relaxed);
				job.stale.fetch_add(1, std::memory_order_relaxed);
				done_one(work);
				continue;
			}
			ShaderRecompiler::IR::ResourceSpecialization specialization;
			if (!ShaderDiskCache::DecodeSpecialization(permutation.specialization, specialization)) {
				job.unusable.fetch_add(1, std::memory_order_relaxed);
				done_one(work);
				continue;
			}
			std::vector<uint8_t> rekeyed_bytes;
			if (work.rekey) {
				bool cleared = false;
				for (auto& buffer: specialization.buffers) {
					cleared                = cleared || buffer.zero_stride_oob;
					buffer.zero_stride_oob = false;
				}
				if (cleared) {
					ShaderDiskCache::ByteWriter writer;
					ShaderDiskCache::AppendSpecialization(writer, specialization);
					rekeyed_bytes = std::move(writer.bytes);
				}
				rekeyed.fetch_add(1, std::memory_order_relaxed);
			}
			// v424: a recording made before buffer strides became per-draw data names its strides;
			// this build keys (and records, and names modules) with them canonical.
			const auto spec_bytes = CanonicalSpecialization(
			    specialization, rekeyed_bytes.empty() ? std::span<const uint8_t>(permutation.specialization) : std::span<const uint8_t>(rekeyed_bytes),
			    plan_info ? &*plan_info : nullptr);
			std::vector<uint8_t> permutation_key;
			BuildPermutationKey(permutation_key, source_key, specialization, cursor);
			if (!replayed_keys.insert(XXH3_64bits(permutation_key.data(), permutation_key.size())).second) {
				// Another recording of this source canonicalizes to the same permutation.
				job.stale.fetch_add(1, std::memory_order_relaxed);
				done_one(work);
				continue;
			}
			// v391: graphics modules are recorded under their permutation (PipelineRecipe::NoteModule).
			const bool graphics_stage = options.stage != ShaderType::Compute;
			// Library prefetch: this shader's listed permutations, by the name of their records.
			GplPrefetchIndexAdd(options.stage, recipe.shader_hash, permutation_key);
			// A carried permutation is recorded under this plan (its stable id and ledger record),
			// and its old stable id names the same module, so recipes made from the old module
			// are renamed to it (PipelineRecipe::Remap).
			const auto stable = PipelineRecipe::StableId(recipe.id, cursor, fingerprint, spec_bytes);
			const auto old_stable = PipelineRecipe::StableId(recipe.id, permutation.push_data_cursor,
			                                                 permutation.plan_fingerprint, permutation.specialization);
			const auto note_carried = [&](std::span<const uint32_t> spirv) {
				if (!carry) {
					return;
				}
				if (graphics_stage) {
					PipelineRecipe::NoteModule(old_stable, spirv, true);
				}
				ledger->RecordPermutation(recipe.id, cursor, specialization, fingerprint);
			};
			// Presence only: reading every entry (SPIR-V included) at each boot would be gigabytes.
			// Except once per session key for a graphics permutation the module map lacks (the first
			// start of v391, and a permutation another program compiled first): its SPIR-V is read
			// and recorded, so recipes made from it can be renamed after a recompiler change.
			if (GplWarmWanted(options.stage, permutation_key)) {
				std::lock_guard lock(job.warm_mutex);
				job.warm_queue.emplace_back(options.stage, permutation_key);
			}
			if (disk->Contains(permutation_key)) {
				if (graphics_stage && !PipelineRecipe::HasCurrentModule(stable)) {
					std::vector<uint8_t>               payload;
					ShaderDiskCache::PermutationRecord record;
					if (disk->Load(permutation_key, payload) &&
					    ShaderDiskCache::DecodePermutationRecord(payload, record)) {
						PipelineRecipe::NoteModule(stable, record.spirv, true);
						note_carried(record.spirv);
					}
				} else if (carry) {
					ledger->RecordPermutation(recipe.id, cursor, specialization, fingerprint);
				}
				if (carry) {
					job.carried.fetch_add(1, std::memory_order_relaxed);
				}
				done_one(work);
				continue;
			}
			// v421: a carried permutation that did not fit is remembered (per session key), so the
			// next start does not translate it again.
			std::vector<uint8_t> carry_fail_key;
			if (carry) {
				carry_fail_key = permutation_key;
				carry_fail_key.push_back('X');
				carry_fail_key.push_back('c');
				if (disk->Contains(carry_fail_key)) {
					job.carry_failed.fetch_add(1, std::memory_order_relaxed);
					done_one(work);
					continue;
				}
			}
			if (!have_translation) {
				// IR::Program is move-only and CompileProgram consumes it: translate again.
				translated = ReplayTranslate(recipe.code, options);
				if (translated.unsupported) {
					done_one(work);
					continue;
				}
			}
			have_translation = false;
			ShaderRecompiler::CompileResult compiled;
			if (carry) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
				try {
					Common::RecoverableExitScope recoverable;
					compiled = ShaderRecompiler::CompileProgram(std::move(translated), options,
					                                            specialization, cursor);
				} catch (const Common::RecoverableExit& failure) {
					static std::atomic<uint32_t> logged {0};
					if (logged.fetch_add(1) < 8) {
						std::fprintf(stderr, "[ledger-carry] dropped a permutation of %016" PRIx64 ": %s\n",
						             recipe.shader_hash, failure.text.c_str());
					}
					job.carry_failed.fetch_add(1, std::memory_order_relaxed);
					const uint8_t marker = 1;
					disk->Save(carry_fail_key, std::span(&marker, 1));
					done_one(work);
					continue;
				}
#endif
			} else {
				compiled = ShaderRecompiler::CompileProgram(std::move(translated), options,
				                                            specialization, cursor);
			}
			if (!ValidateShaderSpirv(options.dump_label, recipe.shader_hash, compiled.spirv)) {
				if (carry) {
					job.carry_failed.fetch_add(1, std::memory_order_relaxed);
				}
				done_one(work);
				continue;
			}
			if (graphics_stage) {
				PipelineRecipe::NoteModule(stable, compiled.spirv, true);
			}
			note_carried(compiled.spirv);
			if (carry) {
				job.carried.fetch_add(1, std::memory_order_relaxed);
			}
			const auto info = std::move(compiled.program).TakeCompiledInfo();
			std::vector<uint8_t> record;
			ShaderDiskCache::EncodePermutationRecord(compiled.spirv, info, record);
			StoreEntry(false, recipe.shader_hash, permutation_key, {}, record);
			progress.compiled.fetch_add(1);
			done_one(work);
		}
		// v421: every permutation recorded under another plan whose specialization has a module
		// under this plan (carried, compiled or found above) names that module, duplicates
		// included, so the pipeline recipes made from any of the old modules are renamed.
		// v424f: so does one recorded under this plan with a specialization this build keys
		// differently (strides recorded before they became per-draw data): its old module id
		// names the canonical permutation's module.
		if (carry_enabled && usable && fingerprint != 0 && !precompile_stop.load(std::memory_order_acquire) &&
		    options.stage != ShaderType::Compute) {
			for (const auto index: permutations) {
				const auto& old       = job.permutations[index];
				const auto  canonical = CanonicalSpecializationBytes(old.specialization,
				                                                     plan_info ? &*plan_info : nullptr);
				if (old.plan_fingerprint == fingerprint &&
				    std::equal(canonical.begin(), canonical.end(), old.specialization.begin(),
				               old.specialization.end())) {
					continue;
				}
				if (PipelineRecipe::AliasModule(
				        PipelineRecipe::StableId(recipe.id, old.push_data_cursor, old.plan_fingerprint, old.specialization),
				        PipelineRecipe::StableId(recipe.id, old.push_data_cursor, fingerprint, canonical))) {
					job.aliased.fetch_add(1, std::memory_order_relaxed);
				}
			}
		}
	}

	// v424: zero the specialization fields this build carries as per-draw data (buffer stride and
	// index stride, IR::BufferStrideData), in place; returns its encoding (the input bytes when
	// nothing changed).
	static std::vector<uint8_t> CanonicalSpecialization(ShaderRecompiler::IR::ResourceSpecialization& value,
	                                                    std::span<const uint8_t> recorded,
	                                                    const ShaderRecompiler::IR::ShaderInfo* info) {
		bool changed = false;
		if (info != nullptr) {
			// v432: dynamic buffer formats and image class/layer/cube, as materialization keys them.
			const auto before = value;
			ShaderRecompiler::IR::CanonicalizeDynamicFields(*info, value);
			changed = !(before == value);
		}
		if (ShaderRecompiler::IR::BufferStrideAsData()) {
			for (auto& buffer: value.buffers) {
				const auto canonical = buffer.packed_stride & ((1u << 14u) | (1u << 20u));
				changed              = changed || canonical != buffer.packed_stride;
				buffer.packed_stride = canonical;
			}
		}
		if (!changed) {
			return {recorded.begin(), recorded.end()};
		}
		ShaderDiskCache::ByteWriter writer;
		ShaderDiskCache::AppendSpecialization(writer, value);
		return std::move(writer.bytes);
	}
	static std::vector<uint8_t> CanonicalSpecializationBytes(std::span<const uint8_t> recorded,
	                                                         const ShaderRecompiler::IR::ShaderInfo* info) {
		ShaderRecompiler::IR::ResourceSpecialization value;
		if (!ShaderDiskCache::DecodeSpecialization(recorded, value)) {
			return {recorded.begin(), recorded.end()};
		}
		return CanonicalSpecialization(value, recorded, info);
	}

	void FlushDisk() {
		StopPrecompile();
		if (disk != nullptr && disk_owner) {
			disk->Flush();
			disk->LogTotals("saved");
		}
	}
	~ProgramCache() {
		StopPrecompile();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	// KYTY_EXPAND_BY_CONSULTED=0: the reuse and fast-program keys hash all of the user data.
	static bool KeyByConsulted() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_EXPAND_BY_CONSULTED");
			return value == nullptr || value[0] != '0';
		}();
		return enabled;
	}

	// A hash of the user data as far as the expanded code (and so the program) depends on it:
	// only its size for code without calls, the registers the last complete call resolution
	// consulted (and their values) for code with calls, all of it otherwise. The title's call
	// shader c4df2a00067e0666 is dispatched ~16,000 times a second with different user data but
	// the same call-table registers: hashing all of it missed every reuse and expanded each time
	// (~3 us, 8 % of the GPU thread in the jungle, 2026-10-18).
	uint64_t ExpansionInputsHash(const ShaderParams& params, std::span<const uint32_t> user_data,
	                             uint32_t wave_size, uint64_t seed) {
		if (KeyByConsulted() && params.hash != 0) {
			const uint64_t no_call_key =
			    params.hash ^ (static_cast<uint64_t>(params.code.size()) << 32u) ^
			    (static_cast<uint64_t>(wave_size) << 56u);
			const uint64_t count = user_data.size();
			if (no_call_shaders.contains(no_call_key)) {
				return XXH3_64bits_withSeed(&count, sizeof(count), seed ^ 0x6e6f2d63616c6c73ull);
			}
			if (const auto* regs = function_expander.ConsultedUserData(params.code, params.hash,
			                                                           wave_size)) {
				consulted_scratch.clear();
				consulted_scratch.push_back(static_cast<uint32_t>(count));
				for (const auto reg: *regs) {
					consulted_scratch.push_back(reg);
					consulted_scratch.push_back(reg < user_data.size() ? user_data[reg] : 0u);
				}
				return XXH3_64bits_withSeed(consulted_scratch.data(),
				                            consulted_scratch.size() * sizeof(uint32_t),
				                            seed ^ 0x636f6e73756c7465ull);
			}
		}
		return XXH3_64bits_withSeed(user_data.data(), user_data.size_bytes(), seed);
	}

	// Shader function expansion decodes the shader and every callee, and a shader with table
	// calls can be dispatched hundreds of times a frame; the expander keeps what it can reuse.
	void ExpandShaderFunctions(const ShaderParams& params, std::span<const uint32_t> user_data,
	                           uint32_t wave_size, std::vector<uint32_t>& expanded) {
		// Resolving the call targets and re-reading every callee to validate the kept expansion
		// was ~7.6 us a dispatch, 13 000 dispatches a second (10 % of the GPU thread in the
		// jungle, 2026-10-10). The same shader with the same user data reuses the result for
		// 50 ms before it is validated again. KYTY_EXPAND_REUSE_MS=0 validates every time.
		static const auto reuse = std::chrono::milliseconds([] {
			const char* value = std::getenv("KYTY_EXPAND_REUSE_MS");
			return value != nullptr ? std::strtoul(value, nullptr, 10) : 250ul;
		}());
		// Code without calls expands to nothing whatever the user data: once the expander has
		// found none in this code (named by its content hash and size), skip the user-data hash,
		// the reuse map and the expander (most of the title's ~25,000 dispatches a second).
		const uint64_t no_call_key =
		    params.hash ^ (static_cast<uint64_t>(params.code.size()) << 32u) ^
		    (static_cast<uint64_t>(wave_size) << 56u);
		if (params.hash != 0 && no_call_shaders.contains(no_call_key)) {
			expanded.clear();
			return;
		}
		const auto now = std::chrono::steady_clock::now();
		uint64_t   reuse_key = 0;
		const uint64_t reuse_seed = params.hash ^ params.Base() ^
		                            (static_cast<uint64_t>(wave_size) << 56u) ^
		                            (static_cast<uint64_t>(params.code.size()) << 32u);
		if (reuse.count() != 0) {
			reuse_key = ExpansionInputsHash(params, user_data, wave_size, reuse_seed);
			if (const auto it = expand_reuse.find(reuse_key);
			    it != expand_reuse.end() && now - it->second.validated < reuse) {
				expanded = it->second.code;
				return;
			}
		}
		struct Remember {
			std::unordered_map<uint64_t, ExpandReused>& map;
			uint64_t                               key;
			std::chrono::steady_clock::time_point  now;
			const std::vector<uint32_t>&           result;
			~Remember() {
				if (key == 0) {
					return;
				}
				// ~2,800 programs a frame on the title: 4,096 cleared the map every frame or two.
				if (map.size() > 32768) {
					map.clear();
				}
				auto& entry     = map[key];
				entry.code      = result;
				entry.validated = now;
			}
		} remember {expand_reuse, reuse_key, now, expanded};
		std::string reason;
		const bool  expanded_ok = function_expander.Expand(
		     params.code, params.Base(), user_data,
		     [](uint64_t address, std::span<uint32_t> words) {
			     return ReadShaderGuestMemoryRaw(nullptr, address, words);
		     },
		     expanded, reason, wave_size, params.hash);
		if (reuse_key != 0) {
			// Kept under the registers this resolution consulted.
			remember.key = ExpansionInputsHash(params, user_data, wave_size, reuse_seed);
		}
		if (expanded_ok && expanded.empty() && reason.empty() && params.hash != 0 &&
		    !function_expander.HasCalls(params.code)) {
			no_call_shaders.insert(no_call_key);
		}
		// The draw-prep scanner cannot run the expander (GPU-thread state), and a program key
		// without the expanded functions never finds the entry of a shader with calls (the
		// title's 0xc4df2a00067e0666, ~16 000 dispatches a second, all refreshed on the GPU
		// thread). Keep the last expansion by shader and code address for it; the GPU thread
		// checks that the scanner's entry is the one it finds itself (TakeScanned).
		if (expanded_ok && !expanded.empty()) {
			const uint64_t   scan_key = ScanExpandKey(params, wave_size);
			std::lock_guard  lock(scan_expand_mutex);
			auto&            kept = scan_expansions[scan_key];
			if (kept != expanded) {
				if (scan_expansions.size() > 4096) {
					scan_expansions.clear();
					scan_expansions[scan_key] = expanded;
				} else {
					kept = expanded;
				}
			}
		}
		if (!expanded_ok) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1) < 16) {
				::printf("Shader function expansion hash=%016" PRIx64 ": %s\n", params.hash,
				         reason.c_str());
			}
		}
	}

	using ProgramsMap = std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash>;
	ProgramsMap                                                    programs;
	struct FastEntry {
		ProgramsMap::value_type*              entry = nullptr;
		std::chrono::steady_clock::time_point validated;
	};
	std::unordered_map<uint64_t, FastEntry>                        fast_entries;
	// KYTY_COMPUTE_MEMO: the previous compute lookup on this thread that found a program (Get).
	struct ComputeMemo {
		const void*                           owner = nullptr;
		uint64_t                              hash = 0, base = 0;
		size_t                                code_size = 0;
		uint32_t                              wave = 0, count = 0;
		std::array<uint32_t, 40>              user_data {};
		std::vector<uint32_t>                 static_state;
		ProgramsMap::value_type*              entry = nullptr;
		std::chrono::steady_clock::time_point validated {};
	};
	static ComputeMemo& ComputeMemoSlot() {
		thread_local ComputeMemo memo;
		return memo;
	}
	// Taken exclusively by the GPU thread when it inserts into programs (entries are never
	// erased, so pointers to them stay valid), shared by the draw-prep scanner's lookups.
	std::shared_mutex                                              programs_mutex;
	std::unordered_set<ProgramKey, ProgramKeyHash>                 unsupported;
	std::unordered_set<uint64_t>                                   gave_up_fast;
	std::unordered_map<uint64_t, uint32_t>                         gave_up_shaders;
	struct ExpandReused {
		std::vector<uint32_t>                 code;
		std::chrono::steady_clock::time_point validated;
	};
	std::unordered_map<uint64_t, ExpandReused>                     expand_reuse;
	std::unordered_set<uint64_t>                                   no_call_shaders;
	std::vector<uint32_t>                                          consulted_scratch;
	// The last function expansion by shader and code address, for the draw-prep scanner.
	std::mutex                                                     scan_expand_mutex;
	std::unordered_map<uint64_t, std::vector<uint32_t>>            scan_expansions;
	static uint64_t ScanExpandKey(const ShaderParams& params, uint32_t wave_size) {
		return params.hash ^ params.Base() ^ (static_cast<uint64_t>(wave_size) << 56u) ^
		       (static_cast<uint64_t>(params.code.size()) << 32u);
	}
	ShaderRecompiler::Decoder::ShaderFunctionExpander             function_expander;
	ProgramKey                                                     lookup_key;
	vk::Device                                                  device;
	bool                                                        shader_clock = false;
	bool                                                        bindless_images = false;
	uint64_t                                                    next_shader_id = 0;
	// SPIR-V size of the last permutation compiled (for the profiler).
	size_t                                                      last_spirv_words = 0;
	// The shader disk cache (null when off) and its session key, built at first use.
	ShaderDiskCache::SessionInfo                                disk_session;
	std::vector<uint8_t>                                        disk_session_key;
	bool                                                        disk_verify  = false;
	uint64_t                                                    disk_lookups = 0;
	std::shared_ptr<ShaderDiskCache::Store>                     disk;
	bool                                                        disk_owner = true;
	// The shader ledger (plan 4.12; null when off) and the id of the source Get recorded last.
	std::shared_ptr<ShaderLedger::Ledger>                       ledger;
	uint64_t                                                    ledger_source_id = 0;
	// The startup precompile (StartPrecompile): its workers and their stop flag.
	std::vector<std::thread>                                    precompile_threads;
	std::atomic<bool>                                           precompile_stop {false};
	// Graphics translations in progress on the workers (KYTY_ASYNC_TRANSLATION), by program.
	std::unordered_map<ProgramKey, std::shared_ptr<TranslationJob>, ProgramKeyHash> translating;
	uint64_t                                                    translation_skips = 0;
	uint64_t                                                    translations_done = 0;
	// Set by GetGraphicsPrograms for the draw at hand: may its shaders translate in the background?
	bool                                                        defer_translation = false;
	// The entries whose snapshot the parallel window refreshed for the draw at hand
	// (PrepareStagesInParallel); Get takes them instead of walking again.
	std::array<SourceEntry*, 2>                                 prepared {};
	ProgramKey                                                  prep_key;
	uint64_t                                                    prep_windows    = 0;
	uint64_t                                                    prep_vs_ok      = 0;
	uint64_t                                                    prep_ps_ok      = 0;
	uint64_t                                                    prep_declined   = 0;
	uint64_t                                                    prep_mismatches = 0;
	std::unique_ptr<StagePrepWorker>                            stage_prep;
	// Last: destroyed first, so its threads have stopped before anything else goes.
	std::unique_ptr<TranslationWorkers>                         translators;

	// A stage's program entry as Get would find it, or null (not translated yet, skipped).
	template <typename InputInfo>
	SourceEntry* FindForPrep(const ShaderParams& params, const InputInfo& input_info,
	                         ShaderType stage) {
		if (SkipShaderRequested(params.hash)) {
			return nullptr;
		}
		prep_key.stage           = stage;
		prep_key.hash            = params.hash;
		prep_key.user_data_count = params.user_data_count;
		prep_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, prep_key.static_state);
		prep_key.function_code.clear();
		const auto entry = programs.find(prep_key);
		return entry != programs.end() ? &entry->second : nullptr;
	}

	// The SRT refresh of Get with the probe reader (ReadShaderGuestMemoryProbe).
	static bool WalkProbe(SourceEntry& entry, const ShaderParams& params) {
		GuestPageReadCache               cache;
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryProbe,
		    .userdata                   = &cache,
		    .read_specialization_memory = ReadShaderGuestMemoryProbe,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &cache;
		runtime.page_shift    = 12;
		std::lock_guard walk_lock(entry.walk_mutex);
		return ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
		                                                  entry.resources, entry.specialization);
	}

	// The parallel window of a draw (StagePrepWorker): the helper refreshes the vertex stage while
	// this thread refreshes the pixel stage. Both entries must exist already.
	void PrepareStagesInParallel(const ShaderParams& ps_params, const ShaderPixelInputInfo& ps_info,
	                             const ShaderParams&          vs_params,
	                             const ShaderVertexInputInfo& vs_info) {
		prepared = {};
		if (stage_prep == nullptr) {
			return;
		}
		auto* ps = FindForPrep(ps_params, ps_info, ShaderType::Pixel);
		if (ps == nullptr) {
			return;
		}
		auto* vs = FindForPrep(vs_params, vs_info, vs_info.logical_stage);
		if (vs == nullptr || vs == ps) {
			return;
		}
		struct Context {
			SourceEntry*        entry;
			const ShaderParams* params;
			bool                ok;
		} job {vs, &vs_params, false};
		KYTY_PROFILER_BLOCK("StagePrep::Window");
		if (!stage_prep->Post(
		        [](void* context) {
			        auto& job = *static_cast<Context*>(context);
			        job.ok    = WalkProbe(*job.entry, *job.params);
		        },
		        &job)) {
			prep_declined++;
			return;
		}
		const bool ps_ok = WalkProbe(*ps, ps_params);
		const bool vs_ok = stage_prep->Join() && job.ok;
		prep_windows++;
		prep_ps_ok += ps_ok ? 1u : 0u;
		prep_vs_ok += vs_ok ? 1u : 0u;
		prepared = {ps_ok ? ps : nullptr, vs_ok ? vs : nullptr};
		if ((prep_windows & (prep_windows - 1u)) == 0u && prep_windows >= 1024u) {
			LOGF("Stage preparation: %" PRIu64 " parallel windows, pixel walks served %" PRIu64
			     ", vertex walks served %" PRIu64 ", %" PRIu64 " declined (helper asleep or "
			     "busy), %" PRIu64 " verify mismatches\n",
			     prep_windows, prep_ps_ok, prep_vs_ok, prep_declined, prep_mismatches);
		}
	}

	bool TakePrepared(const SourceEntry* entry) {
		for (auto& slot: prepared) {
			if (slot == entry && entry != nullptr) {
				slot = nullptr;
				return true;
			}
		}
		return false;
	}

	// KYTY_STAGE_PREP_VERIFY=1: the exact walk again after a parallel one, compared.
	void VerifyPrepared(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                    ShaderType stage, uint64_t hash) {
		const auto resources      = entry.resources;
		const auto specialization = entry.specialization;
		bool       ok             = false;
		{
			std::lock_guard walk_lock(entry.walk_mutex);
			ok = ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
			                                                entry.resources, entry.specialization);
		}
		if (!ok) {
			return;
		}
		const auto& fresh = entry.resources;
		if (resources.flattened_srt == fresh.flattened_srt && resources.buffers == fresh.buffers &&
		    resources.images == fresh.images && resources.samplers == fresh.samplers &&
		    specialization == entry.specialization) {
			return;
		}
		if (prep_mismatches++ < 32) {
			LOGF("Stage preparation: verify mismatch %s 0x%016" PRIx64 " (flat %d, buffers %d, "
			     "images %d, samplers %d, specialization %d)\n",
			     StageLabel(stage), hash,
			     static_cast<int>(resources.flattened_srt != fresh.flattened_srt),
			     static_cast<int>(resources.buffers != fresh.buffers),
			     static_cast<int>(resources.images != fresh.images),
			     static_cast<int>(resources.samplers != fresh.samplers),
			     static_cast<int>(!(specialization == entry.specialization)));
		}
	}

	// Draw prep (KYTY_DRAW_PREP, drawPrep.h). Scanner thread: the entry Get would find for these
	// inputs, or null (not translated yet, skipped).
	template <typename InputInfo>
	SourceEntry* FindForScan(const ShaderParams& params, const InputInfo& input_info,
	                         ShaderType stage, ProgramKey& key) {
		if (SkipShaderRequested(params.hash)) {
			return nullptr;
		}
		key.stage           = stage;
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, key.static_state);
		key.function_code.clear();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			std::lock_guard lock(scan_expand_mutex);
			if (const auto it = scan_expansions.find(ScanExpandKey(params, input_info.wave_size));
			    it != scan_expansions.end()) {
				key.function_code = it->second;
			}
		}
		std::shared_lock programs_lock(programs_mutex);
		const auto       entry = programs.find(key);
		// KYTY_FPS_LOG: why a scanned program has no entry (first miss of each shader): the
		// fields of the keys the GPU thread inserted for the same shader that differ.
		if (entry == programs.end()) {
			static const bool log = Common::FpsLogDiagnostics();
			static std::mutex                   diag_mutex;
			static std::unordered_set<uint64_t> diag_seen;
			if (log) {
				std::lock_guard diag_lock(diag_mutex);
				if (diag_seen.size() < 32 && diag_seen.insert(params.hash).second) {
					for (const auto& [other, value]: programs) {
						if (other.hash != key.hash || other.stage != key.stage) continue;
						std::string diff;
						if (other.user_data_count != key.user_data_count) diff += fmt::format(" ud {}!={}", other.user_data_count, key.user_data_count);
						if (other.code_size != key.code_size) diff += fmt::format(" size {}!={}", other.code_size, key.code_size);
						if (!other.function_code.empty()) diff += fmt::format(" function_code {} dwords", other.function_code.size());
						for (size_t i = 0; i < std::max(other.static_state.size(), key.static_state.size()); i++) {
							const uint32_t a = i < other.static_state.size() ? other.static_state[i] : 0xdeadu;
							const uint32_t b = i < key.static_state.size() ? key.static_state[i] : 0xdeadu;
							if (a != b) diff += fmt::format(" static[{}] {:x}!={:x}", i, a, b);
						}
						std::fprintf(stderr, "[draw-prep-key] %016llx entry differs:%s\n",
						             static_cast<unsigned long long>(params.hash), diff.c_str());
					}
				}
			}
		}
		return entry != programs.end() ? &entry->second : nullptr;
	}

	// KYTY_FPS_LOG: the shaders the scanner could not prepare most often (no entry / walk failed),
	// every 10 s.
	static void NoteScanMiss(uint64_t hash, bool walk) {
		static const bool log = Common::FpsLogDiagnostics();
		if (!log) return;
		static std::mutex                                                  mutex;
		static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> misses;
		static auto                                                        last = std::chrono::steady_clock::now();
		std::lock_guard                                                    lock(mutex);
		auto&                                                              m = misses[hash];
		(walk ? m.second : m.first)++;
		const auto now = std::chrono::steady_clock::now();
		if (now - last < std::chrono::seconds(10)) return;
		last = now;
		std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> list(misses.begin(), misses.end());
		std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
			return a.second.first + a.second.second > b.second.first + b.second.second;
		});
		for (size_t i = 0; i < list.size() && i < 10; i++) {
			std::fprintf(stderr, "[draw-prep-miss] %016llx noentry %llu walkfail %llu\n",
			             static_cast<unsigned long long>(list[i].first),
			             static_cast<unsigned long long>(list[i].second.first),
			             static_cast<unsigned long long>(list[i].second.second));
		}
		misses.clear();
	}

	// The probe reader (bytes no GPU work has written, nothing else), recording what it read.
	struct RecordingReader {
		GuestPageReadCache  cache;
		DrawPrep::ReadSet*  reads  = nullptr;
		bool                failed = false;
		uint64_t            hash   = 0;
	};
	// A read the probe refuses fails the preparation even where the walk tolerates it (a guarded
	// read through a null base reads zero): the GPU thread's walk would read those bytes, and the
	// certificate cannot cover a read it did not record.
	static bool ReadRecorded(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto& reader = *static_cast<RecordingReader*>(userdata);
		if (!ReadShaderGuestMemoryProbe(&reader.cache, address, values) &&
		    !SrtVerifyRead(address, values)) {
			reader.failed = true;
			// KYTY_FPS_LOG: the first refused reads of each shader, and why.
			static const bool log = Common::FpsLogDiagnostics();
			if (log) {
				static std::mutex                                 diag_mutex;
				static std::unordered_map<uint64_t, uint32_t>     diag_seen;
				std::lock_guard                                   diag_lock(diag_mutex);
				if (auto& n = diag_seen[reader.hash]; n < 4 && diag_seen.size() < 64) {
					n++;
					std::fprintf(stderr,
					             "[draw-prep-read] %016llx refused 0x%llx dwords=%zu gpu_written=%d cond=%d\n",
					             static_cast<unsigned long long>(reader.hash),
					             static_cast<unsigned long long>(address), values.size(),
					             SrtBytesGpuWritten(address, values.size_bytes()) ? 1 : 0,
					             ShaderRecompiler::IR::SrtEvaluatingCondition() ? 1 : 0);
				}
			}
			return false;
		}
		reader.reads->Add(address, values.data(), static_cast<uint32_t>(values.size_bytes()));
		return true;
	}

	// Scanner thread: the SRT refresh Get would run for these inputs, into `result`.
	template <typename InputInfo>
	bool ScanStage(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	               DrawPrep::StageResult& result) {
		result.ok = false;
		thread_local ProgramKey key;
		auto*                   entry = FindForScan(params, input_info, stage, key);
		if (entry == nullptr) {
			DrawPrep::Add(DrawPrep::Counter::SkipNoEntry);
			NoteScanMiss(params.hash, false);
			return false;
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		result.stage       = static_cast<uint32_t>(stage);
		result.entry       = entry;
		result.shader_base = params.Base();
		result.user_data.assign(user_data.begin(), user_data.end());
		result.reads.Clear();
		RecordingReader                  reader {.reads = &result.reads, .hash = params.hash};
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadRecorded,
		    .userdata                   = &reader,
		    .read_specialization_memory = ReadRecorded,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		// No map_clean_page: every read goes through the recorder.
		result.workgroup_counts = {};
		result.workgroup_size   = {};
		result.gpu_fills        = false;
		result.resources.gpu_fills.clear();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// As Get: the dispatch size, and GPU-written slots left to the GPU.
			for (uint32_t axis = 0; axis < 3u; axis++) {
				result.workgroup_counts[axis] = input_info.workgroup_counts[axis];
				result.workgroup_size[axis]   = input_info.threads_num[axis];
			}
			runtime.workgroup_counts = result.workgroup_counts;
			runtime.workgroup_size   = result.workgroup_size;
			if (SrtGpuFillEnabled()) {
				result.gpu_fills    = true;
				runtime.gpu_written = SrtBytesGpuWritten;
				runtime.gpu_fills   = &result.resources.gpu_fills;
			}
		}
		bool ok = false;
		{
			std::lock_guard walk_lock(entry->walk_mutex);
			result.resources.srt_verify.clear();
			t_srt_verify = SrtVerifyProgram(entry->resource_plan, stage, params.hash)
			                   ? &result.resources.srt_verify
			                   : nullptr;
			ok = ShaderRecompiler::IR::MaterializeResources(entry->resource_plan, runtime,
			                                                result.resources, result.specialization);
			t_srt_verify = nullptr;
		}
		if (!ok || reader.failed) {
			if (static const bool log = Common::FpsLogDiagnostics(); log && !reader.failed) {
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1) < 16) {
					std::fprintf(stderr, "[draw-prep-walk] %016llx walk failed without a refused read\n",
					             static_cast<unsigned long long>(params.hash));
				}
			}
			DrawPrep::Add(DrawPrep::Counter::FailWalk);
			NoteScanMiss(params.hash, true);
			return false;
		}
		static std::atomic<uint64_t> serials {0};
		result.ok       = true;
		result.consumed = false;
		result.serial   = ++serials;
		DrawPrep::Add(DrawPrep::Counter::Prepared);
		return true;
	}

	// The first field in which two refreshes differ, or null.
	static const char* DiffRefresh(const ShaderRecompiler::IR::ResourceSnapshot&       a,
	                               const ShaderRecompiler::IR::ResourceSpecialization& a_spec,
	                               const ShaderRecompiler::IR::ResourceSnapshot&       b,
	                               const ShaderRecompiler::IR::ResourceSpecialization& b_spec) {
		auto sorted = [](std::vector<std::pair<uint64_t, uint64_t>> reads) {
			std::ranges::sort(reads);
			reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
			return reads;
		};
		if (a.flattened_srt != b.flattened_srt) return "flattened_srt";
		if (a.buffers != b.buffers) return "buffers";
		if (a.images != b.images) return "images";
		if (a.samplers != b.samplers) return "samplers";
		if (a.user_data != b.user_data) return "user_data";
		if (a.buffer_write_extents != b.buffer_write_extents) return "buffer_write_extents";
		if (a.bindless_heaps != b.bindless_heaps) return "bindless_heaps";
		if (a.bindless_sampler_heaps != b.bindless_sampler_heaps) return "bindless_sampler_heaps";
		if (!(a.uniform_fill == b.uniform_fill)) return "uniform_fill";
		if (a.gpu_fills != b.gpu_fills) return "gpu_fills";
		if (a.srt_verify != b.srt_verify) return "srt_verify";
		if (sorted(a.specialization_reads) != sorted(b.specialization_reads)) {
			return "specialization_reads";
		}
		if (!(a_spec == b_spec)) return "specialization";
		return nullptr;
	}

	// GPU thread: the scanner's refresh of this stage for the current packet, when it was made
	// from the same program, user data and shader base and its bytes are unchanged and clean.
	bool TakeScanned(SourceEntry& entry, ShaderType stage,
	                 const ShaderRecompiler::IR::SrtRuntime& runtime, uint64_t hash) {
		auto* result = DrawPrep::Take(static_cast<uint32_t>(stage));
		if (result == nullptr) {
			return false;
		}
		if (result->entry != &entry) {
			DrawPrep::Add(DrawPrep::Counter::MissEntry);
			return false;
		}
		// Verified values only while the shader verifies (it may have reported since the scan).
		if (!result->resources.srt_verify.empty() &&
		    !SrtVerifyProgram(entry.resource_plan, stage, hash)) {
			DrawPrep::Add(DrawPrep::Counter::MissInputs);
			return false;
		}
		// Used already by an earlier draw of this packet: the entry must still hold its refresh.
		const bool again = result->consumed;
		if (again && entry.prepared_serial != result->serial) {
			DrawPrep::Add(DrawPrep::Counter::MissNone);
			return false;
		}
		if (result->shader_base != runtime.shader_base ||
		    !std::ranges::equal(result->user_data, runtime.user_data) ||
		    result->gpu_fills != (runtime.gpu_written != nullptr)) {
			DrawPrep::Add(DrawPrep::Counter::MissInputs);
			return false;
		}
		if (stage == ShaderType::Compute &&
		    (!std::ranges::equal(result->workgroup_counts, runtime.workgroup_counts) ||
		     result->workgroup_size != runtime.workgroup_size)) {
			DrawPrep::Add(DrawPrep::Counter::MissInputs);
			return false;
		}
		if (!result->reads.Validate()) {
			DrawPrep::Add(DrawPrep::Counter::MissReads);
			return false;
		}
		// A slot left to the GPU must still be GPU-written, or the walk here would read it.
		for (const auto& fill: (again ? entry.resources : result->resources).gpu_fills) {
			if (!SrtBytesGpuWritten(fill.address, sizeof(uint32_t))) {
				DrawPrep::Add(DrawPrep::Counter::MissReads);
				return false;
			}
		}
		result->consumed = true;
		// The refresh as it will be used: the result's before the swap, the entry's when again.
		const auto& used_resources      = again ? entry.resources : result->resources;
		const auto& used_specialization = again ? entry.specialization : result->specialization;
		if (DrawPrep::VerifyMode() != 0) {
			ShaderRecompiler::IR::ResourceSnapshot       fresh;
			ShaderRecompiler::IR::ResourceSpecialization fresh_specialization;
			bool                                         ok = false;
			auto                                         fresh_runtime = runtime;
			if (fresh_runtime.gpu_fills != nullptr) {
				fresh_runtime.gpu_fills = &fresh.gpu_fills;
			}
			{
				std::lock_guard walk_lock(entry.walk_mutex);
				ok = ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, fresh_runtime,
				                                                fresh, fresh_specialization);
			}
			DrawPrep::Add(DrawPrep::Counter::VerifyChecks);
			const char* field = ok ? DiffRefresh(used_resources, used_specialization, fresh,
			                                     fresh_specialization)
			                       : "walk failed";
			if (field != nullptr) {
				DrawPrep::Add(DrawPrep::Counter::VerifyMismatches);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::string detail;
					if (std::strcmp(field, "specialization_reads") == 0) {
						const auto& a = used_resources.specialization_reads;
						const auto& b = fresh.specialization_reads;
						detail = fmt::format(" (prepared {} reads, walk {}:", a.size(), b.size());
						for (const auto& read: a) {
							if (std::ranges::find(b, read) == b.end()) {
								detail += fmt::format(" +{:x}/{}", read.first, read.second);
							}
						}
						for (const auto& read: b) {
							if (std::ranges::find(a, read) == a.end()) {
								detail += fmt::format(" -{:x}/{}", read.first, read.second);
							}
						}
						detail += ")";
					}
					LOGF("DrawPrep verify: %s 0x%016" PRIx64 " differs from the walk: %s%s\n",
					     StageLabel(stage), hash, field, detail.c_str());
				}
				if (DrawPrep::VerifyMode() == 2) {
					EXIT("DrawPrep verify: %s 0x%016" PRIx64 " differs from the walk: %s\n",
					     StageLabel(stage), hash, field);
				}
			}
		}
		if (again) {
			DrawPrep::Add(DrawPrep::Counter::Reused);
			return true;
		}
		// Swapped, not moved: the entry's old buffers go back to the scanner with the result.
		std::swap(entry.resources, result->resources);
		std::swap(entry.specialization, result->specialization);
		entry.prepared_serial = result->serial;
		DrawPrep::Add(DrawPrep::Counter::Used);
		return true;
	}

	void SetDeferTranslation(bool value) { defer_translation = value; }
};

// Graphics pipelines built at startup from the recipes of every pipeline built before
// (pipelineRecipe.h; KYTY_PIPELINE_RECIPES=0: off). The thread waits for the shader ledger's
// replay (it creates the modules the recipes name), then builds the recipes on
// KYTY_PIPELINE_PRECOMPILE_THREADS workers (default 6) into `prebuilt`; a draw whose prepared
// pipeline has a prebuilt recipe takes it (TryGetGraphicsPipeline).
struct PipelineCache::RecipeState {
	PipelineRecipe::Store                                                              store;
	std::mutex                                                                         mutex;
	std::unordered_map<PipelineRecipe::Hash128, vk::Pipeline, PipelineRecipe::Hash128Hasher> prebuilt;
	std::thread                                                                        thread;
	std::atomic<bool>                                                                  stop {false};
	std::atomic<uint64_t>                                                              hits {0};

	vk::Pipeline Take(const PipelineRecipe::Hash128& hash) {
		std::lock_guard lock(mutex);
		const auto      found = prebuilt.find(hash);
		if (found == prebuilt.end()) {
			return nullptr;
		}
		const auto pipeline = found->second;
		prebuilt.erase(found);
		return pipeline;
	}
};

// The driver cache was written only by Save, at a clean exit: a run that crashed or was closed
// any other way lost every pipeline it compiled, and the next run compiled them again (the
// pipeline stalls of plan 4.14: the file kept its 14:20 size through a whole chain of runs).
// KYTY_PIPELINE_CACHE_SAVE_S (default 180; 0 = only at exit): every that many seconds, when
// pipelines were built since the last write, a low-priority thread writes it.
struct PipelineCache::DriverCacheSaver {
	std::thread             thread;
	std::mutex              mutex;
	std::condition_variable wake;
	bool                    stop = false;

	void Stop() {
		{
			std::lock_guard lock(mutex);
			stop = true;
		}
		wake.notify_all();
		if (thread.joinable()) {
			thread.join();
		}
	}
	~DriverCacheSaver() { Stop(); }
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_program_cache(std::make_unique<ProgramCache>(
          graphics.device, graphics.shader_device_clock_enabled,
          graphics.bindless_enabled && Config::BindlessImagesEnabled(),
          ShaderDiskCache::SessionInfo {
              .vendor_id           = graphics.GetPhysicalDeviceProperties().vendorID,
              .device_id           = graphics.GetPhysicalDeviceProperties().deviceID,
              .bindless_images     = graphics.bindless_enabled && Config::BindlessImagesEnabled(),
              .float_image_atomics = Config::FloatImageAtomicsEnabled()},
          ShaderDiskCacheDirectory())) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	if (GuestGpu::AsyncThreadConfigured()) {
		m_async_program_cache = std::make_unique<ProgramCache>(
		    graphics.device, graphics.shader_device_clock_enabled,
		    graphics.bindless_enabled && Config::BindlessImagesEnabled(),
		    ShaderDiskCache::SessionInfo {
		        .vendor_id           = graphics.GetPhysicalDeviceProperties().vendorID,
		        .device_id           = graphics.GetPhysicalDeviceProperties().deviceID,
		        .bindless_images     = graphics.bindless_enabled && Config::BindlessImagesEnabled(),
		        .float_image_atomics = Config::FloatImageAtomicsEnabled()},
		    ShaderDiskCacheDirectory(), m_program_cache.get());
		PipelineCacheLog("Shader programs: the async-compute thread has its own program cache");
	}
	InitializeDriverCache();
	if (g_async_pipelines) {
		// KYTY_PIPELINE_THREADS: the driver compiles pipelines in parallel; a new area brings dozens
		// at once (up to 171 a second at a save load), so a third of the cores, 2 to 8.
		const auto threads = [] {
			if (const char* value = std::getenv("KYTY_PIPELINE_THREADS"); value != nullptr) {
				return std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 16u);
			}
			return std::clamp(std::thread::hardware_concurrency() / 3u, 2u, 8u);
		}();
		m_compiler         = std::make_unique<PipelineCompiler>(threads);
		PipelineCacheLog("Vulkan pipelines: compiled on {} worker threads (wait {} ms)", threads,
		                 g_pipeline_wait.count());
	}
	// The device layer has set the emitter's switches by now (vulkanWindow.cpp, before the render
	// context is created), so the session key the replay builds is the one Get will build.
	// First: it sets the SPIR-V store up before any module is registered.
	StartPipelinePrecompile();
	g_warm_graphics = &m_graphics;
	g_warm_cache      = m_driver_cache; // libraries and compute go to their shards (AuxCacheFor)
	g_warm_main_cache = m_driver_cache;
	g_warm_save_hook  = [this] { (void)WriteDriverCache(); };
	m_program_cache->StartPrecompile();
}

void PipelineCache::StartPipelinePrecompile() {
	// With GPL a pipeline the game asks for is linked from warmed libraries (0.01 ms), so only
	// the pipelines that stay monolithic get recipes (v438e: rect lists, built with generated
	// tessellation stages per VS/PS pair; fr7's opening had 10 monolithic misses of 5-430 ms).
	// They are kept in their own file (<title>.mrecipes), so the recipes of sessions before GPL
	// (v436z zr2 lost the device with those and GPL on) are not built. KYTY_PIPELINE_RECIPES=0: off.
	const char* value   = std::getenv("KYTY_PIPELINE_RECIPES");
	const bool  enabled = value == nullptr || value[0] != '0';
	if (!enabled || m_driver_cache_path.empty()) {
		return;
	}
	m_recipes = std::make_unique<RecipeState>();
	// In _ShaderList with the ledger (ShaderListDirectory), not with the driver cache.
	std::error_code list_error;
	std::filesystem::create_directories(ShaderListDirectory(), list_error);
	// Named by the title only, like the ledger (v438e; v437r had put the device-configuration tag
	// of the driver cache in these names): recipes name modules by stable id and are renamed to
	// this session's modules at startup, so they carry over to another device configuration.
	auto title = m_driver_cache_path.stem().string();
	if (const auto underscore = title.find('_'); underscore != std::string::npos) {
		title.resize(underscore);
	}
	auto path = ShaderListDirectory() / (title + (m_graphics.graphics_pipeline_library_enabled ? ".mrecipes" : ".recipes"));
	m_recipes->store.Open(path);
	auto spirv_dir = ShaderListDirectory() / (title + "_spirv");
	PipelineRecipe::SetSpirvStore(spirv_dir);
	{
		// v391: the session the replay and Get build their keys with (ProgramCache::disk_session_key).
		const auto session_key = ShaderDiskCache::SessionKey(ShaderDiskCache::SessionInfo {
		    .vendor_id           = m_graphics.GetPhysicalDeviceProperties().vendorID,
		    .device_id           = m_graphics.GetPhysicalDeviceProperties().deviceID,
		    .bindless_images     = m_graphics.bindless_enabled && Config::BindlessImagesEnabled(),
		    .float_image_atomics = Config::FloatImageAtomicsEnabled()});
		auto modules_path = path;
		modules_path.replace_extension(".modules");
		PipelineRecipe::OpenModuleMap(modules_path, XXH3_64bits(session_key.data(), session_key.size()));
	}
	auto loaded = std::make_shared<std::vector<PipelineRecipe::Store::Entry>>(m_recipes->store.TakeLoaded());
	PipelineRecipe::CollectUnknownModules(*loaded); // v394: before the precompile thread's import
	auto& progress = GetShaderPrecompileProgress();
	progress.pipelines_total.store(static_cast<uint32_t>(loaded->size()), std::memory_order_relaxed);
	progress.pipelines_known.store(true, std::memory_order_release);
	if (loaded->empty()) {
		progress.pipelines_finished.store(true, std::memory_order_release);
		return;
	}
	auto* state         = m_recipes.get();
	state->thread       = std::thread([this, state, loaded] {
        Common::LowerBackgroundThreadPriority();
        auto& progress = GetShaderPrecompileProgress();
        // The modules the recipes name exist once the shader replay finished.
        while (!state->stop.load(std::memory_order_acquire) &&
               progress.finished_at.load(std::memory_order_acquire) == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        // v391: the replay's SPIR-V for renamed recipes is written by the store's thread.
        PipelineRecipe::WaitSpirvStore();
        // v418 (Mac 11-13): only recipes this build can use are built and counted. A recipe whose
        // renamed copy is a recipe of its own is built as that copy; a recipe with a stage of
        // another recompiler session (no module of this session) would build from the SPIR-V
        // store's old copy a pipeline no draw asks for. Both stay in the file (another build or
        // a later carry-over may use them). KYTY_PIPELINE_RECIPE_FILTER=0: build every recipe.
        static const bool filter = [] {
            const char* value = std::getenv("KYTY_PIPELINE_RECIPE_FILTER");
            return value == nullptr || value[0] != '0';
        }();
        uint32_t left_duplicate = 0, left_obsolete = 0;
        if (filter && !state->stop.load(std::memory_order_acquire)) {
            std::vector<PipelineRecipe::Store::Entry> usable;
            usable.reserve(loaded->size());
            for (auto& entry: *loaded) {
                std::vector<uint8_t> renamed = entry.bytes;
                const auto           remap   = PipelineRecipe::Remap(renamed);
                if (remap.changed && state->store.Known(PipelineRecipe::HashBytes(renamed))) {
                    left_duplicate++;
                    continue;
                }
                if (!PipelineRecipe::StagesCurrent(remap.changed ? renamed : entry.bytes)) {
                    left_obsolete++;
                    continue;
                }
                usable.push_back(std::move(entry));
            }
            *loaded = std::move(usable);
            progress.pipelines_total.store(static_cast<uint32_t>(loaded->size()), std::memory_order_relaxed);
            std::fprintf(stderr, "[pipeline-recipes] %zu usable by this build; left out: %u duplicate (renamed copy recorded), %u of another session\n",
                         loaded->size(), left_duplicate, left_obsolete);
        }
        progress.pipelines_active.store(true, std::memory_order_release);
        const auto begin   = std::chrono::steady_clock::now();
        const auto threads = [] {
            if (const char* value = std::getenv("KYTY_PIPELINE_PRECOMPILE_THREADS"); value != nullptr) {
                return std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 16u);
            }
            return 6u;
        }();
        std::atomic<size_t>   next {0};
        std::atomic<uint32_t> built {0}, skipped {0};
        std::atomic<uint32_t> renamed_built {0}, duplicate {0}, unmapped {0}, differs {0};
        std::vector<std::thread> workers;
        for (uint32_t t = 0; t < threads; t++) {
            workers.emplace_back([&] {
                Common::LowerBackgroundThreadPriority();
                for (;;) {
                    const auto index = next.fetch_add(1);
                    if (index >= loaded->size() || state->stop.load(std::memory_order_acquire)) {
                        return;
                    }
                    const auto& entry    = (*loaded)[index];
                    // v391: a recipe of another recompiler session is built with this session's
                    // modules of the same permutations, and kept under its new hash.
                    std::vector<uint8_t> renamed = entry.bytes;
                    const auto remap = PipelineRecipe::Remap(renamed);
                    auto       hash  = entry.hash;
                    if (remap.unknown != 0) {
                        unmapped++;
                    }
                    if (remap.interface_differs) {
                        differs++;
                    }
                    if (remap.changed) {
                        hash = PipelineRecipe::HashBytes(renamed);
                        if (state->store.Known(hash)) {
                            // Its renamed copy is a recipe of its own (built by itself).
                            duplicate++;
                            GetShaderPrecompileProgress().pipelines_done.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                    }
                    vk::Pipeline pipeline = nullptr;
                    const auto   result   = PipelineRecipe::Build(m_graphics, remap.changed ? renamed : entry.bytes,
                                                                  m_driver_cache, &pipeline);
                    if (result == vk::Result::eSuccess && pipeline != nullptr) {
                        if (remap.changed) {
                            (void)state->store.Record(hash, renamed, false);
                            renamed_built++;
                        }
                        std::lock_guard lock(state->mutex);
                        if (!state->prebuilt.emplace(hash, pipeline).second) {
                            m_graphics.device.destroyPipeline(pipeline, nullptr);
                        }
                        built++;
                    } else {
                        skipped++;
                    }
                    GetShaderPrecompileProgress().pipelines_done.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& worker: workers) {
            worker.join();
        }
        progress.pipelines_built.store(built.load(), std::memory_order_relaxed);
        progress.pipelines_active.store(false, std::memory_order_release);
        progress.pipelines_finished.store(true, std::memory_order_release);
        PipelineCacheLog("Pipeline recipes: {} built, {} skipped (module not in this session), in {:.1f} s",
                         built.load(), skipped.load(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
        std::fprintf(stderr, "[pipeline-recipes] %u of %zu built at startup, %u skipped, %.1f s\n",
                     built.load(), loaded->size(), skipped.load(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
        // v391: recipes renamed to this session's modules (built / already a recipe), recipes with a
        // stage recorded before the module map, recipes whose module interface changed.
        std::fprintf(stderr,
                     "[pipeline-recipes] renamed to this session's shaders: %u built, %u already known; "
                     "%u with a stage from before v391, %u with a changed shader interface\n",
                     renamed_built.load(), duplicate.load(), unmapped.load(), differs.load());
    });
}

PipelineCache::~PipelineCache() {
	if (m_recipes != nullptr) {
		m_recipes->stop.store(true, std::memory_order_release);
		if (m_recipes->thread.joinable()) {
			m_recipes->thread.join();
		}
	}
	Save();
	if (m_recipes != nullptr) {
		std::lock_guard lock(m_recipes->mutex);
		for (const auto& [hash, pipeline]: m_recipes->prebuilt) {
			(void)hash;
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
		m_recipes->prebuilt.clear();
	}
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->pending && pipeline->pending->done.load(std::memory_order_acquire)) {
				m_graphics.device.destroyPipeline(pipeline->pending->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			if (pipeline->optimized != nullptr) {
				m_graphics.device.destroyPipeline(vk::Pipeline(pipeline->optimized->exchange(VK_NULL_HANDLE)), nullptr);
			}
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
			if (pipeline->ps_set_layout != nullptr) {
				m_graphics.device.destroyDescriptorSetLayout(pipeline->ps_set_layout, nullptr);
			}
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}

	const auto config       = DeviceConfigTag(m_graphics);
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + "_" + config + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		// Pipelines of older builds stay in the file; past 3 GiB it starts over (1 GiB before: Mac's
		// cache grew ~50 MB a session, so it would have been thrown away within weeks).
		constexpr uint64_t max_file_size = uint64_t {3} << 30u;
		if (file_size >= signature.size() + sizeof(uint64_t) && file_size <= max_file_size) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	const size_t initial_data_size = initial_data.size();
	initial_data.clear();
	initial_data.shrink_to_fit();
	m_driver_cache_saved  = initial_data_size;
	m_driver_cache_loaded = initial_data_size;
	// Step D7: the libraries (KYTY_GPL) and compute pipelines in driver caches of their own, sharded
	// by module (<title>_lib_NN.bin, <title>_cs_NN.bin): the warm-up's tens of thousands stay far
	// below the per-file limit, and a save writes only the shards that changed.
	// KYTY_AUX_CACHES=0: everything in the main cache.
	static const bool aux_enabled = [] {
		const char* value = std::getenv("KYTY_AUX_CACHES");
		return value == nullptr || value[0] != '0';
	}();
	bool aux_complete = true;
	for (uint32_t which = 0; which < 2 && aux_enabled; which++) {
		if (which == 0 && !m_graphics.graphics_pipeline_library_enabled) {
			continue;
		}
		const uint32_t count = which == 0 ? 16u : 8u;
		auto&          shards = m_aux[which];
		size_t         loaded = 0;
		uint32_t       missing = 0;
		for (uint32_t i = 0; i < count; i++) {
			char name[32];
			std::snprintf(name, sizeof(name), "_%s_%02u.bin", which == 0 ? "lib" : "cs", i);
			auto path = std::filesystem::path("_PipelineCache") / (title_id + "_" + config + name);
			auto data = ReadDriverCacheFile(path);
			vk::PipelineCacheCreateInfo info {};
			info.initialDataSize = data.size();
			info.pInitialData    = data.empty() ? nullptr : data.data();
			vk::PipelineCache cache;
			if (m_graphics.device.createPipelineCache(&info, nullptr, &cache) != vk::Result::eSuccess) {
				data.clear();
				info = vk::PipelineCacheCreateInfo {};
				if (m_graphics.device.createPipelineCache(&info, nullptr, &cache) != vk::Result::eSuccess) {
					cache = nullptr;
				}
			}
			if (data.empty()) {
				missing++;
			}
			if (cache == nullptr) {
				// A shard that cannot exist: no shards of this kind at all.
				for (auto made: shards.caches) {
					m_graphics.device.destroyPipelineCache(made, nullptr);
				}
				shards = {};
				break;
			}
			loaded += data.size();
			shards.caches.push_back(cache);
			shards.paths.push_back(std::move(path));
			shards.saved.push_back(data.size());
			shards.loaded.push_back(data.size());
		}
		aux_complete = aux_complete && missing == 0;
		SetAuxCaches(static_cast<AuxCache>(which), shards.caches);
		PipelineCacheLog("Vulkan pipeline cache: {} in {} shards ({} bytes loaded, {} empty or rejected)",
		                 which == 0 ? "libraries" : "compute pipelines", shards.caches.size(), loaded, missing);
	}
	GplWarmSetCachesComplete(aux_complete);
	static const unsigned long save_s = [] {
		const char* value = std::getenv("KYTY_PIPELINE_CACHE_SAVE_S");
		return value != nullptr ? std::strtoul(value, nullptr, 10) : 180ul;
	}();
	if (save_s != 0) {
		m_driver_cache_saver = std::make_unique<DriverCacheSaver>();
		auto* saver          = m_driver_cache_saver.get();
		saver->thread        = std::thread([this, saver] {
            Common::LowerBackgroundThreadPriority();
            uint64_t saved_count = g_pipelines_built.load(std::memory_order_relaxed);
            for (;;) {
                {
                    std::unique_lock lock(saver->mutex);
                    if (saver->wake.wait_for(lock, std::chrono::seconds(save_s),
                                             [saver] { return saver->stop; })) {
                        return;
                    }
                }
                // Step D7: every period; only the caches whose data changed are written.
                (void)saved_count;
                (void)WriteDriverCache();
            }
        });
		PipelineCacheLog("Vulkan pipeline cache: written every {} s when pipelines were built", save_s);
	}
}

void PipelineCache::Save() {
	if (m_driver_cache_saver != nullptr) {
		m_driver_cache_saver->Stop();
	}
	m_program_cache->FlushDisk();
	// No worker may use the driver cache once it is destroyed below; later pipelines are
	// compiled on the GPU thread (FinishPending).
	if (m_compiler != nullptr) {
		m_compiler->Stop();
	}
	GplPrefetchStop();  // its libraries use the driver cache too
	GplOptimizerStop(); // step D6: its links use the driver cache
	if (m_driver_cache == nullptr) {
		return;
	}
	g_warm_save_hook = nullptr;
	ZeroCompileReport("exit");
	{
		// What the caches gained since startup: after a complete warm-up (a restart),
		// new entries are compiles the warm-ups did not cover (the gate with the counters).
		std::lock_guard lock(m_write_mutex);
		const auto grown = [&](vk::PipelineCache cache, size_t saved) -> long long {
			size_t size = 0;
			return cache != nullptr && m_graphics.device.getPipelineCacheData(cache, &size, nullptr) == vk::Result::eSuccess
			           ? static_cast<long long>(size) - static_cast<long long>(saved)
			           : 0;
		};
		long long lib = 0, cs = 0;
		for (size_t i = 0; i < m_aux[0].caches.size(); i++) lib += grown(m_aux[0].caches[i], m_aux[0].loaded[i]);
		for (size_t i = 0; i < m_aux[1].caches.size(); i++) cs += grown(m_aux[1].caches[i], m_aux[1].loaded[i]);
		std::fprintf(stderr, "[zero-compile] cache growth since startup: libraries %+lld, compute %+lld, main %+lld bytes\n",
		             lib, cs, m_driver_cache != nullptr ? grown(m_driver_cache, m_driver_cache_loaded) : 0ll);
	}
	if (WriteDriverCache()) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
		m_driver_cache = nullptr;
		for (uint32_t which = 0; which < 2; which++) {
			SetAuxCaches(static_cast<AuxCache>(which), {});
			for (auto cache: m_aux[which].caches) {
				m_graphics.device.destroyPipelineCache(cache, nullptr);
			}
			m_aux[which] = {};
		}
	}
}

std::vector<uint8_t> PipelineCache::ReadDriverCacheFile(const std::filesystem::path& cache_path) const {
	std::vector<uint8_t> data;
	if (!Common::File::IsFileExisting(cache_path)) {
		return data;
	}
	Common::File file(cache_path, Common::File::Mode::Read);
	const auto   file_size = file.IsInvalid() ? 0 : file.Size();
	const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	constexpr uint64_t max_file_size = uint64_t {3} << 30u;
	if (file_size < signature.size() + sizeof(uint64_t) || file_size > max_file_size) {
		file.Close();
		return data;
	}
	std::string cached_signature(signature.size(), '\0');
	uint64_t    payload_hash = 0;
	data.resize(file_size - signature.size() - sizeof(payload_hash));
	uint32_t signature_read = 0;
	uint32_t hash_read      = 0;
	uint32_t payload_read   = 0;
	file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()), &signature_read);
	file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
	file.Read(data.data(), static_cast<uint32_t>(data.size()), &payload_read);
	file.Close();
	if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
	    payload_read != data.size() || cached_signature != signature ||
	    XXH3_64bits(data.data(), data.size()) != payload_hash) {
		data.clear();
	}
	return data;
}

bool PipelineCache::WriteDriverCache() {
	std::lock_guard lock(m_write_mutex);
	if (m_driver_cache == nullptr) {
		return false;
	}
	auto       pending = GplWarmTakePending(); // made before this write: in the data it fetches
	const bool main = WriteDriverCacheIfChanged(m_driver_cache, m_driver_cache_path, m_driver_cache_saved);
	bool       aux  = true;
	for (auto& shards: m_aux) {
		for (size_t i = 0; i < shards.caches.size(); i++) {
			aux = WriteDriverCacheIfChanged(shards.caches[i], shards.paths[i], shards.saved[i]) && aux;
		}
	}
	// Step D7: the warm record names only what a written cache holds.
	GplWarmCommit(pending, aux);
	return main;
}

bool PipelineCache::WriteDriverCacheIfChanged(vk::PipelineCache cache, const std::filesystem::path& cache_path,
                                              size_t& saved) {
	size_t size = 0;
	if (m_graphics.device.getPipelineCacheData(cache, &size, nullptr) == vk::Result::eSuccess && size == saved &&
	    Common::File::IsFileExisting(cache_path)) {
		return true;
	}
	size_t written = 0;
	if (!WriteDriverCacheFile(cache, cache_path, &written)) {
		return false;
	}
	saved = written;
	return true;
}

bool PipelineCache::WriteDriverCacheFile(vk::PipelineCache cache, const std::filesystem::path& cache_path,
                                         size_t* written) {
	const auto& m_driver_cache      = cache;      // NOLINT: the body below names the main cache's members
	const auto& m_driver_cache_path = cache_path; // NOLINT
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	if (written != nullptr) {
		*written = payload.size();
	}
	return true;
}

// Glyphs, icons and other cached layers are drawn once into small offscreen targets, and a draw
// skipped while its shaders translate or its pipeline compiles is lost for good: the Leap Attack
// prompt lost letters ("L p Att k") in every new game whose session translated shaders
// (2026-10-08, ISSUES #27), whatever the depth state of those draws. They translate and compile
// at once instead; a one-time cost per program, none once the disk caches hold it.
bool SmallColorTargetDraw(const HW::Context& context) {
	static const uint32_t limit = [] {
		const char* value = std::getenv("KYTY_SMALL_TARGET_SYNC");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1024u;
	}();
	if (limit == 0) {
		return false;
	}
	const auto mask  = context.GetRenderTargetMask();
	bool       found = false;
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) == 0 || target.base.addr == 0) {
			continue;
		}
		if (target.attrib2.width + 1u > limit || target.attrib2.height + 1u > limit) {
			return false;
		}
		found = true;
	}
	return found;
}

// A pixel shader that may kill pixels still has its depth/stencil test run early on the
// guest GPU when DB_SHADER_CONTROL asks for EARLY_Z_THEN_LATE_Z: a kill only matters for the
// depth/stencil writes, which the late pass does. Vulkan runs the tests after any shader with
// side effects (storage writes) unless the shader declares EarlyFragmentTests, so fragments
// the guest culls still run and write their storage. With no depth or stencil writes the early
// test is exact, so those draws get EarlyFragmentTests. KYTY_EARLY_Z_KILL=0 turns this off;
// KYTY_LOG_EARLY_Z_KILL=1 lists the pixel shaders it applies to ([early-z-kill], once each).
static void ApplyEarlyTestsWithKill(const HW::ShaderRegisters& sh, const HW::Context& context,
                                    uint64_t hash, ShaderPixelInputInfo& pixel_info) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EARLY_Z_KILL");
		return value == nullptr || value[0] != '0';
	}();
	const auto& db = sh.db_shader_control;
	if (!enabled || pixel_info.ps_early_z || db.shader_z_behavior != 1 || !db.shader_kill_enable ||
	    db.shader_z_export_enable || db.shader_mask_export_enable) {
		return;
	}
	const auto& depth = context.GetDepthControl();
	if (!depth.z_enable && !depth.stencil_enable) {
		return;
	}
	if (depth.z_enable && depth.z_write_enable) {
		return;
	}
	if (depth.stencil_enable) {
		const auto& mask = context.GetStencilMask();
		if (mask.stencil_writemask != 0 || (depth.backface_enable && mask.stencil_writemask_bf != 0)) {
			return;
		}
	}
	// The kill flag only gates EarlyFragmentTests in the SPIR-V emitter; the discards come from
	// the shader code. Clearing it here keeps the emitter (and the shader cache version) as is.
	pixel_info.ps_early_z           = true;
	pixel_info.ps_pixel_kill_enable = false;
	static const bool log = [] {
		const char* value = std::getenv("KYTY_LOG_EARLY_Z_KILL");
		return value != nullptr && value[0] == '1';
	}();
	if (log) {
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged;
		std::lock_guard                     lock(logged_mutex);
		if (logged.insert(hash).second) {
			std::fprintf(stderr, "[early-z-kill] ps=%016" PRIx64 " zfunc=%u stencil=%d\n", hash,
			             depth.zfunc, depth.stencil_enable ? 1 : 0);
			std::fflush(stderr);
		}
	}
}

// DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER (bit 12, AMD gfx9+ register spec; kept in other_bits): the
// depth/stencil test and its writes happen before the pixel shader runs, so pixels the shader then
// kills keep the depth they wrote (GL/Vulkan early fragment tests; Mesa sets it for
// early_fragment_tests together with KILL_ENABLE when the shader discards). Vulkan does the same
// with EarlyFragmentTests, which the emitter adds when ps_early_z is set and the kill flag clear.
// KYTY_DEPTH_BEFORE_SHADER=1 applies it (11-13, hex holes test); KYTY_LOG_DEPTH_BEFORE_SHADER=1
// lists the pixel shaders whose draws set the bit ([depth-before-shader], once each).
static void ApplyDepthBeforeShader(const HW::ShaderRegisters& sh, uint64_t hash,
                                   ShaderPixelInputInfo& pixel_info) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DEPTH_BEFORE_SHADER");
		return value != nullptr && value[0] == '1';
	}();
	static const bool log = [] {
		const char* value = std::getenv("KYTY_LOG_DEPTH_BEFORE_SHADER");
		return value != nullptr && value[0] == '1';
	}();
	const auto& db = sh.db_shader_control;
	if ((db.other_bits & 0x00001000u) == 0) {
		return;
	}
	const bool apply = enabled && !db.shader_z_export_enable && !db.shader_mask_export_enable;
	if (log) {
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint64_t> logged;
		std::lock_guard                     lock(logged_mutex);
		if (logged.insert(hash).second) {
			std::fprintf(stderr,
			             "[depth-before-shader] ps=%016" PRIx64 " z_order=%u kill=%d z_export=%d "
			             "mask_export=%d other_bits=0x%08x %s\n",
			             hash, db.shader_z_behavior, db.shader_kill_enable ? 1 : 0,
			             db.shader_z_export_enable ? 1 : 0, db.shader_mask_export_enable ? 1 : 0,
			             db.other_bits, apply ? "applied" : "not applied");
			std::fflush(stderr);
		}
	}
	if (apply) {
		pixel_info.ps_early_z           = true;
		pixel_info.ps_pixel_kill_enable = false;
	}
}

bool PipelineCache::DescribeGraphicsStages(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    std::array<ShaderParams, 3>& vertex_params, ShaderParams& pixel_params) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	if (tess_active) {
		if (!PrepareTessellationPrograms(vertex_regs, context, vertex_info, vertex_params)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("unsupported tessellation programs\n");
			}
			return false;
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		// The pipeline requires the wave size where it can (PrepareGraphicsPipeline).
		const auto required =
		    m_graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT, mesh.wave_size);
		mesh.host_subgroup_size = required != 0 ? required : m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		// A subgroup with more threads than a mesh workgroup may have (NVIDIA: 128) runs its
		// waves in passes; see EmitMeshEntryPoint.
		mesh.passes = mesh.PassesFor(
		    std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
		if (mesh.passes == 0 || mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			// Skipped like the draws of a shader that gives up, and reported once per shader.
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.insert(vertex_params[0].hash).second) {
				std::fprintf(stderr,
				             "[mesh-limits] mesh shader 0x%016" PRIx64
				             " exceeds host limits, draws skipped: wave%u threads=%u passes=%u "
				             "vertices=%u/%u primitives=%u/%u LDS=%u/%u bytes fast=%d\n",
				             vertex_params[0].hash, mesh.wave_size, mesh.HostThreads(), mesh.passes,
				             mesh.max_vertices, limits.maxMeshOutputVertices, mesh.max_primitives,
				             limits.maxMeshOutputPrimitives,
				             static_cast<uint32_t>(mesh.lds_size_dwords * sizeof(uint32_t)),
				             limits.maxMeshSharedMemorySize, mesh.fast_launch ? 1 : 0);
				std::fflush(stderr);
			}
			return false;
		}
	}
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		ApplyDepthBeforeShader(sh, pixel_params.hash, pixel_info);
		ApplyEarlyTestsWithKill(sh, context, pixel_params.hash, pixel_info);
		// SPI_SHADER_COL_FORMAT describes export packing, not the attachment numeric type.
		// In particular, 32-bit exports can carry raw integer material data.
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr == 0 ||
			    render_target_mask_slot(context.GetRenderTargetMask(), slot) == 0) {
				continue;
			}
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << slot;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << slot;
			}
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           ((pixel_info.target_uint_mask | pixel_info.target_sint_mask) & 1u) == 0 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
		if (pixel_info.dual_source_blending) {
			// The second source takes the numeric type of target 0.
			pixel_info.target_uint_mask =
			    (pixel_info.target_uint_mask & ~2u) | ((pixel_info.target_uint_mask & 1u) << 1u);
			pixel_info.target_sint_mask =
			    (pixel_info.target_sint_mask & ~2u) | ((pixel_info.target_sint_mask & 1u) << 1u);
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	return true;
}

// The vertex-side identity of a draw for DrawPrep::NoteGoodPair / IsGoodPair.
static uint64_t DrawPrepVertexIdentity(const HW::VertexShaderInfo& vertex_regs) {
	return vertex_regs.es_regs.data_addr ^ (vertex_regs.gs_regs.data_addr << 1u);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const CpuCost::Phase cpu_phase_("GetGraphicsPrograms");
	std::array<ShaderParams, 3> vertex_params;
	ShaderParams                pixel_params;
	if (!DescribeGraphicsStages(vertex_regs, pixel_regs, sh, context, user_config,
	                            target_export_mapping, pixel_active, vertex_info, pixel_info,
	                            vertex_params, pixel_params)) {
		return {};
	}
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	// New shaders of 3D draws (depth test on) translate on the workers while the draw is skipped;
	// UI and layer draws translate at once (TranslationWorkers).
	struct DeferScope {
		ProgramCache& cache;
		~DeferScope() { cache.SetDeferTranslation(false); }
	} defer_scope {*m_program_cache};
	m_program_cache->SetDeferTranslation(context.GetDepthControl().z_enable &&
	                                     !SmallColorTargetDraw(context));
	struct PreparedScope {
		ProgramCache& cache;
		~PreparedScope() { cache.prepared = {}; }
	} prepared_scope {*m_program_cache};
	if (pixel_active && !tess_active) {
		m_program_cache->PrepareStagesInParallel(pixel_params, pixel_info, vertex_params[0],
		                                         vertex_info[0]);
	}
	// v434 (KYTY_PUSH_ANCHOR=0: the old chain, the stage after the pixel shader starting behind
	// its data): the pixel shader's data is anchored at the top of the push range, the stage before
	// it starts at 0/7, so a module does not depend on its partner (the shipped list needs no
	// pairing). Tessellation pipelines keep the chain.
	static const bool anchor_enabled = [] {
		const char* value = std::getenv("KYTY_PUSH_ANCHOR");
		return value == nullptr || value[0] != '0';
	}();
	const bool anchored = anchor_enabled && !tess_active;
	if (pixel_active) {
		uint32_t pixel_cursor = anchored ? ShaderRecompiler::IR::PushData::TopAnchor : push_data_cursor;
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, pixel_cursor);
		if (!anchored) {
			push_data_cursor = pixel_cursor;
		}
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		uint32_t cursor = push_data_cursor;
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], cursor);
		if (!result.vertex[i]) {
			return {};
		}
		if (anchored && result.pixel && pixel_info.stage.program != nullptr &&
		    vertex_info[i].stage.program != nullptr) {
			const auto& vs = vertex_info[i].stage.program->bindings;
			const auto& ps = pixel_info.stage.program->bindings;
			if (vs.UsesPushData() && ps.UsesPushData() &&
			    vs.push_data_start_dword + vs.ShaderDataDwords() > ps.push_data_start_dword) {
				// The two stages' data overlap: the stage before the pixel shader reads its data
				// from the ShaderData buffer.
				cursor           = ShaderRecompiler::IR::PushData::NoPush;
				result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], cursor);
				if (!result.vertex[i]) {
					return {};
				}
			}
		}
		if (!anchored) {
			push_data_cursor = cursor;
		}
	}
	if (DrawPrep::Enabled() && !tess_active && (!pixel_active || result.pixel)) {
		DrawPrep::NoteGoodPair(DrawPrepVertexIdentity(vertex_regs),
		                       pixel_active ? pixel_regs.ps_regs.data_addr : 0u);
	}
	return result;
}

bool PipelineCache::DrawPrepGraphics(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    DrawPrep::PacketResult& out) {
	out.stage_count = 0;
	if (user_config.GetPrimType() == Prospero::PrimitiveType::kPatch) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return false;
	}
	if (!DrawPrep::IsGoodPair(DrawPrepVertexIdentity(vertex_regs),
	                          pixel_active ? pixel_regs.ps_regs.data_addr : 0u)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		// A pair the GPU thread has not drawn yet, seen here before it gets to the draw: the stage
		// libraries of its shaders are made on the prefetch workers meanwhile (once per shader),
		// so the draw's pipeline finds them and only links.
		if (GplPrefetchEnabled()) {
			if (pixel_active && pixel_regs.ps_regs.data_addr != 0) {
				GplPrefetchRequest(ShaderDeclaredHash(pixel_regs.ps_regs.data_addr));
			}
			if (vertex_regs.es_regs.data_addr != 0) {
				const uint64_t front = ShaderDeclaredHash(vertex_regs.es_regs.data_addr);
				GplPrefetchRequest(front);
				// A geometry front half with a separately compiled back half is one program, named by
				// both hashes (shader.cpp, PrepareProgram); a request for a name the list lacks is a lookup.
				if (front != 0 && vertex_regs.gs_regs.data_addr != 0) {
					if (const uint64_t back = ShaderDeclaredHash(vertex_regs.gs_regs.data_addr); back != 0) {
						const uint64_t hashes[] = {front, back};
						GplPrefetchRequest(XXH3_64bits(hashes, sizeof(hashes)));
					}
				}
			}
		}
		return false;
	}
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 pixel_info;
	std::array<ShaderParams, 3>          vertex_params;
	ShaderParams                         pixel_params;
	if (!DescribeGraphicsStages(vertex_regs, pixel_regs, sh, context, user_config,
	                            target_export_mapping, pixel_active, vertex_info, pixel_info,
	                            vertex_params, pixel_params)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return false;
	}
	if (pixel_active &&
	    m_program_cache->ScanStage(pixel_params, pixel_info, ShaderType::Pixel, out.stages[0])) {
		out.stage_count++;
	}
	if (m_program_cache->ScanStage(vertex_params[0], vertex_info[0], vertex_info[0].logical_stage,
	                               out.stages[out.stage_count])) {
		out.stage_count++;
	}
	return out.stage_count != 0;
}

bool PipelineCache::SrtDeferEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_DEFER");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void PipelineCache::BeginSrtDefer(bool allowed) {
	t_srt_defer_allowed = allowed;
	t_srt_defer_tick    = 0;
	t_srt_defer_reads   = 0;
}

uint64_t PipelineCache::EndSrtDefer() {
	t_srt_defer_allowed = false;
	return std::exchange(t_srt_defer_tick, 0);
}

bool PipelineCache::SrtDeferPending() {
	return t_srt_defer_tick != 0;
}

bool PipelineCache::IsSkipListed(uint64_t shader_addr) {
	return SkipShaderRequested(ShaderDeclaredHash(shader_addr));
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	const CpuCost::Phase cpu_phase_("GetComputeProgram");
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	auto&             cache            = m_async_program_cache != nullptr && GuestGpu::IsAsyncThread()
	                                         ? *m_async_program_cache
	                                         : *m_program_cache;
	auto              program          = cache.Get(params, input_info, push_data_cursor);
	if (program && input_info.stage.program != nullptr) {
		const auto* compiled = input_info.stage.program;
		GpuTiming::NoteComputeLanes(compiled->shader_hash,
		                            compiled->wave_size == 64u && input_info.host_subgroup_size == 32u ? 2u : 1u);
	}
	if (program && DrawPrep::Enabled()) {
		// Compute shaders join the known-good list with a vertex identity no draw has.
		DrawPrep::NoteGoodPair(regs.cs_regs.data_addr, UINT64_MAX);
	}
	return program;
}

bool PipelineCache::DrawPrepCompute(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                                    ShaderComputeInputInfo& input_info, DrawPrep::PacketResult& out,
                                    uint32_t queue) {
	out.stage_count = 0;
	if (!DrawPrep::IsGoodPair(regs.cs_regs.data_addr, UINT64_MAX)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		return false;
	}
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage        = input_info.lds_size_dwords * 4u >
	                         m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	// The cache the queue's dispatches will look up in: the async thread's while it runs the
	// compute queues (with KYTY_SYNC_TOGGLE_MODE=async only in the on windows).
	auto& cache = m_async_program_cache != nullptr && queue != 0 && GuestGpu::AsyncThreadActiveNow()
	                  ? *m_async_program_cache
	                  : *m_program_cache;
	if (cache.ScanStage(params, input_info, ShaderType::Compute, out.stages[0])) {
		out.stage_count = 1;
	}
	return out.stage_count != 0;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	auto* pipeline = TryGetGraphicsPipeline(colors, depth, vertex_info, command, ps_input_info,
	                                        topology, primitive_restart_enable, programs, false);
	EXIT_IF(pipeline == nullptr);
	return *pipeline;
}

bool PipelineCache::FinishPending(Pipeline& pipeline) {
	auto& job = *pipeline.pending;
	if (!job.done.load(std::memory_order_acquire)) {
		if (m_compiler != nullptr && !m_compiler->Stopped()) {
			return false;
		}
		// The workers stopped before this job started: compile it here.
		job.result = CreateGraphicsPipeline(*job.build, m_driver_cache, &job.pipeline);
		Profiler::Add(Profiler::Counter::PipelinesPending, -1);
		job.done.store(true, std::memory_order_release);
	}
	EXIT_NOT_IMPLEMENTED(job.result != vk::Result::eSuccess || job.pipeline == nullptr);
	pipeline.pipeline = job.pipeline;
	pipeline.pending.reset();
	return true;
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool may_defer) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);
	const CpuCost::Phase cpu_phase_("GetGraphicsPipeline");

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	// KYTY_PAINT_PS (paintDraws.h): add a constant colour wherever the listed shaders' fragments
	// land. Part of the key, so painted draws get their own pipelines.
	if (ps_active && PaintDraws::Index(pixel_program.hash) >= 0) {
		for (uint32_t i = 0; i < color_count; i++) {
			const auto slot                          = colors[i].target_slot;
			static_params.color_mask[slot]           = PaintDraws::Opaque() ? 0x07u : 0x0fu;
			static_params.blend_enable[slot]         = !PaintDraws::Opaque();
			static_params.color_srcblend[slot]       = static_cast<uint8_t>(Prospero::BlendFactor::kConstantColor);
			static_params.color_destblend[slot]      = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
			static_params.color_comb_fcn[slot]       = static_cast<uint8_t>(Prospero::BlendOp::kAdd);
			static_params.separate_alpha_blend[slot] = false;
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);
	NormalizeGraphicsPipelineKey(static_params);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	const auto defer = [&] {
		Profiler::Add(Profiler::Counter::SkippedPipelinePending);
		ZeroCompileNote(ZeroCompileEvent::SkippedDraw, 0, 0.0);
		LogSkip("pipeline-pending", "vs/ps", (vs_id << 32u) ^ ps_id, 0);
		Profiler::CommandZone::Annotate("skipped: pipeline compiling");
		// Diagnostics: each pipeline whose draws are skipped, once (the first 128).
		static std::unordered_set<uint64_t> logged;
		const auto pipeline_key = (vs_id << 32u) ^ ps_id;
		if (logged.size() < 128 && logged.insert(pipeline_key).second) {
			uint32_t vs_words = 0;
			for (const auto& program: programs.vertex) {
				vs_words += program.spirv_words;
			}
			LOGF("PipelineCache: draw skipped, pipeline compiling: VS 0x%016" PRIx64
			     " (%u words) PS 0x%016" PRIx64
			     " (%u words) colors=%u depth=%d/%d stencil=%d\n",
			     vertex_program.hash, vs_words, ps_active ? pixel_program.hash : 0,
			     ps_active ? pixel_program.spirv_words : 0, color_count,
			     depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
			     depth.stencil_test_enable ? 1 : 0);
		}
		if (++m_deferred_draws % 256 == 1) {
			LOGF("PipelineCache: %" PRIu64 " draws skipped while their pipeline compiled in the "
			     "background\n",
			     m_deferred_draws);
		}
		return nullptr;
	};
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (found.pending && !FinishPending(found)) {
			return defer();
		}
		return &found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	// A pipeline the driver never finished compiling in an earlier run (shaders.cpp): its draws
	// are skipped. KYTY_SLOW_GRAPHICS_TRY=1: build it anyway, without driver optimization
	// (PrepareGraphicsPipeline), on the worker; its draws wait for it like any other pipeline's.
	static const bool slow_try = [] {
		const char* value = std::getenv("KYTY_SLOW_GRAPHICS_TRY");
		return value != nullptr && value[0] == '1';
	}();
	if (!slow_try &&
	    SlowGraphicsPipelineListed(vs_input_info.stage.program->shader_hash,
	                               ps_active ? ps_input_info->stage.program->shader_hash : 0)) {
		return defer();
	}
	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	std::unique_ptr<GraphicsPipelineBuild> build;
	{
		const CpuCost::Phase prepare_phase_("Pipeline::Prepare");
		build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                                ps_input_info, programs, static_params);
	}
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	// A pipeline built at startup from its recipe (StartPipelinePrecompile): taken as is.
	std::vector<uint8_t>    recipe;
	PipelineRecipe::Hash128 recipe_hash;
	// Linked (GPL) pipelines get no recipe: their libraries are warmed instead.
	if (m_recipes != nullptr && !build->gpl && PipelineRecipe::Serialize(*build, recipe)) {
		recipe_hash = PipelineRecipe::HashBytes(recipe);
		if (const auto prebuilt = m_recipes->Take(recipe_hash); prebuilt != nullptr) {
			cached->pipeline = prebuilt;
			PipelineStats::Report(m_graphics.device, prebuilt, 'G', build->vs_hash, build->ps_hash);
			const auto hits  = m_recipes->hits.fetch_add(1, std::memory_order_relaxed) + 1;
			if (Common::FloorStats::Enabled() && (hits & (hits - 1)) == 0) {
				std::fprintf(stderr, "[pipeline-recipes] %llu draws found their pipeline prebuilt\n",
				             static_cast<unsigned long long>(hits));
			}
			auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
			EXIT_IF(!inserted);
			return iter->second.get();
		}
	} else {
		recipe.clear();
	}
	uint64_t vertex_words = 0;
	for (const auto& program: programs.vertex) {
		vertex_words += program.spirv_words;
	}
	const uint64_t spirv_words = vertex_words + pixel_program.spirv_words;
	// Stencil does not count: UI libraries clip their panels with stencil masks.
	const bool layer_draw = !depth.depth_test_enable && !depth.depth_write_enable &&
	                        vertex_words <= g_pipeline_sync_ui_vs_words;
	// A draw whose shaders store to memory (UAV buffers or images) produces data later work
	// reads -- Wolverine's GPU-driven draws take their counts and instances from it. Skipping it
	// while its pipeline compiles left that data unwritten: the next draws read garbage counts and
	// faulted (device lost as the title scene first appeared, ~1 run in 4). These compile at once.
	// KYTY_DEFER_WRITING_DRAWS=1 restores deferring them.
	static const bool defer_writing = std::getenv("KYTY_DEFER_WRITING_DRAWS") != nullptr;
	const auto stage_writes = [](const auto* program) {
		if (program == nullptr) {
			return false;
		}
		for (const auto& buffer: program->info.buffers) {
			if (buffer.written || buffer.atomic) {
				return true;
			}
		}
		for (const auto& image: program->info.images) {
			if (image.written || image.atomic) {
				return true;
			}
		}
		return false;
	};
	bool writes_memory = ps_active && stage_writes(ps_input_info->stage.program);
	for (const auto& info: vertex_info) {
		writes_memory = writes_memory || stage_writes(info.stage.program);
	}
	// The pipelines above that compile "at once", and those of draws that may not be skipped
	// (they write memory), compile on a worker too, waited for up to KYTY_PIPELINE_SYNC_WAIT_MS
	// (default 3000) and KYTY_PIPELINE_WRITE_WAIT_MS (default 10000; 0 for either: on the GPU
	// thread, as before): NVIDIA's compiler never finished one pipeline of the Wolverine save of
	// 2026-10-20 (VS 0xd1ea70c7117d4d3a, PS 0x85892b3b3d0a61ba), which froze the game for good on
	// the GPU thread. Past the wait the draw is skipped like a background compile's.
	static const std::chrono::milliseconds sync_wait = [] {
		const char* value = std::getenv("KYTY_PIPELINE_SYNC_WAIT_MS");
		return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 3000u);
	}();
	static const std::chrono::milliseconds write_wait = [] {
		const char* value = std::getenv("KYTY_PIPELINE_WRITE_WAIT_MS");
		return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 10000u);
	}();
	const bool background = may_defer && (defer_writing || !writes_memory) &&
	                        spirv_words > g_pipeline_sync_words && !layer_draw;
	const auto budget = background ? g_pipeline_wait : may_defer ? sync_wait : write_wait;
	// A linked pipeline whose two stage libraries are already made (for an earlier pipeline, or by
	// the prefetch) needs only the other two libraries (state, no shader: 0.01 ms each) and its
	// fast link (0.02 ms, 0.13 ms at most in the 11-15 runs): made here instead of on a worker the
	// GPU thread then waits for (0.4-1.2 ms a wait in those runs, ~100 waits in the first second of
	// an area). KYTY_GPL_INLINE_LINK=0: as before.
	static const bool inline_link = [] {
		const char* value = std::getenv("KYTY_GPL_INLINE_LINK");
		return value == nullptr || value[0] != '0';
	}();
	const bool link_now = inline_link && build->gpl && GplLibrariesReady(*build);
	if (!link_now && (background || budget.count() > 0) && m_compiler != nullptr && !m_compiler->Stopped()) {
		// A pipeline the driver has cached finishes within the wait; a new one is compiled in
		// the background, and its draws are skipped until it is ready.
		auto job          = std::make_shared<PendingGraphicsPipeline>();
		job->build        = std::move(build);
		job->driver_cache = m_driver_cache;
		job->vs_hash      = vs_input_info.stage.program->shader_hash;
		job->ps_hash      = ps_active ? ps_input_info->stage.program->shader_hash : 0;
		if (!recipe.empty()) {
			job->recipe_store = &m_recipes->store;
			job->recipe_hash  = recipe_hash;
			job->recipe       = std::move(recipe);
		}
		cached->pending   = job;
		Profiler::Add(Profiler::Counter::PipelinesPending);
		static const bool urgent_first = [] {
			const char* value = std::getenv("KYTY_PIPELINE_URGENT_FIRST");
			return value == nullptr || value[0] != '0';
		}();
		m_compiler->Submit(job, urgent_first && !background);
		{
			const auto wait_begin = std::chrono::steady_clock::now();
			const CpuCost::Phase wait_phase_("Pipeline::WaitWorker");
			const ForegroundWaitScope foreground_(true);
			const bool finished   = m_compiler->Wait(*job, budget);
			if (Common::FloorStats::Enabled()) {
				auto&      floor   = Common::FloorStats::Get();
				const auto wait_ns = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_begin)
				        .count());
				Common::FloorStats::Add(floor.pipe_wait_ns, wait_ns);
				Common::FloorStats::Add(floor.pipe_waits, 1);
				Common::FloorStats::Add(floor.pipe_timeouts, finished ? 0u : 1u);
				if (wait_ns >= 100'000'000u && !background) {
					std::fprintf(stderr,
					             "[floor-pipe] waited %llu ms (%s%s) for VS 0x%016" PRIx64 " PS 0x%016" PRIx64
					             ", %" PRIu64 " SPIR-V words\n",
					             static_cast<unsigned long long>(wait_ns / 1000000u),
					             may_defer ? "sync wait" : "write wait", finished ? "" : ", skipped",
					             job->vs_hash, job->ps_hash, spirv_words);
				} else if (!finished && budget.count() >= 1000) {
					std::fprintf(stderr,
					             "[floor-pipe] waited %lld ms for VS 0x%016" PRIx64 " PS 0x%016" PRIx64
					             " (%s), skipped; %" PRIu64 " SPIR-V words\n",
					             static_cast<long long>(wait_ns / 1000000u), job->vs_hash, job->ps_hash,
					             may_defer ? "sync wait" : "write wait", spirv_words);
				}
			}
		}
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
		EXIT_IF(!inserted);
		auto& pipeline = *iter->second;
		if (!FinishPending(pipeline)) {
			return defer();
		}
		LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
		return &pipeline;
	}
	const auto build_begin = std::chrono::steady_clock::now();
	const CpuCost::Phase create_phase_(link_now ? "Pipeline::LinkHere" : "Pipeline::CreateHere");
	if (link_now) {
		Prefetch().linked_here.fetch_add(1, std::memory_order_relaxed);
	}
	vk::Result result;
	{
		const ForegroundWaitScope foreground_(true);
		result = CreateGraphicsPipeline(*build, m_driver_cache, &cached->pipeline);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (!recipe.empty()) {
		(void)m_recipes->store.Record(recipe_hash, recipe);
	}
	ReportGraphicsPipelineBuilt(vs_input_info.stage.program->shader_hash,
	                            ps_active ? ps_input_info->stage.program->shader_hash : 0,
	                            build_begin);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}
	const CpuCost::Phase cpu_phase_("ComputePipeline::Create");

	auto cached = std::make_unique<Pipeline>();
	{
		const auto hash = input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0;
		Profiler::Phases zone;
		zone.SetText("CS 0x%016" PRIx64, hash);
		KYTY_PROFILER_PHASE(zone, "Compute pipeline build", profiler::colors::RedA100);
		const auto begin = std::chrono::steady_clock::now();
		const vk::PipelineCache shard = AuxCacheFor(AuxCache::Compute, ModuleShardHash(compute_program.module));
		{
			const ForegroundWaitScope foreground_(Libs::Graphics::GuestGpu::IsGpuThread());
			CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, shard ? shard : m_driver_cache);
		}
		ReportPipelineBuilt(Profiler::Counter::ComputePipelinesBuilt, "Compute", hash, 0, begin);
		// NVIDIA (621.14) reports no application cache hit for compute pipelines its cache does
		// hold (v437l: same keys as the warm-up, the shards never grow, under 5 ms): a compute
		// creation counts as a miss only when it is also slow; cache growth is reported at exit.
		const bool nohit = LastPipelineCacheHit() == 0;
		if (const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		    ms > 5.0 && GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed)) {
			ZeroCompileNote(nohit ? ZeroCompileEvent::ComputeMiss : ZeroCompileEvent::SlowCompute, hash, ms);
		}
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
void SrtVerifyReportMiss(uint64_t shader_hash) {
	bool inserted = false;
	{
		std::unique_lock lock(g_srt_verify_mutex);
		auto&            state = g_srt_verify_states[shader_hash];
		inserted               = !state.blocked;
		state.blocked          = true;
	}
	if (inserted) {
		std::fprintf(stderr,
		             "[srt-verify] CS %016llx: a value the host read differed from the GPU's; "
		             "that dispatch did nothing, exact reads for this shader from now on\n",
		             static_cast<unsigned long long>(shader_hash));
		LOGF("SRT verify: shader 0x%016" PRIx64 " differed, exact reads from now on\n", shader_hash);
	}
}

// Zero-compile accounting: after the startup wait, everything here is something the shipped list
// (and the warm-ups) should have covered. Each event is logged (the first 300) with the time since
// play started; totals at exit.
namespace {
std::array<std::atomic<uint64_t>, static_cast<size_t>(ZeroCompileEvent::Count)> g_zero_counts {};
std::atomic<uint32_t>                                                             g_zero_logged {0};
std::atomic<int64_t>                                                              g_zero_start {0};
constexpr const char* ZeroName(ZeroCompileEvent event) {
	switch (event) {
		case ZeroCompileEvent::Translation: return "translation";
		case ZeroCompileEvent::LedgerRecord: return "ledger-record";
		case ZeroCompileEvent::SlowLibrary: return "library>5ms";
		case ZeroCompileEvent::SlowCompute: return "compute>5ms";
		case ZeroCompileEvent::SlowPipeline: return "pipeline>5ms";
		case ZeroCompileEvent::SkippedDraw: return "skipped-draw";
		case ZeroCompileEvent::LibraryMiss: return "library-miss";
		case ZeroCompileEvent::ComputeMiss: return "compute-miss";
		case ZeroCompileEvent::PipelineMiss: return "pipeline-miss";
		default: return "?";
	}
}
} // namespace

void ZeroCompileNote(ZeroCompileEvent event, uint64_t hash, double ms) {
	if (!GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed)) {
		return;
	}
	const auto now = ShaderPrecompileProgress::Now();
	int64_t    expected = 0;
	(void)g_zero_start.compare_exchange_strong(expected, now);
	g_zero_counts[static_cast<size_t>(event)].fetch_add(1, std::memory_order_relaxed);
	if (g_zero_logged.fetch_add(1, std::memory_order_relaxed) < 300) {
		std::fprintf(stderr, "[zero-compile] %s %016llx %.1f ms\n", ZeroName(event),
		             static_cast<unsigned long long>(hash), ms);
	}
}

void ZeroCompileReport(const char* when) {
	std::fprintf(stderr, "[zero-compile] totals (%s):", when);
	for (size_t i = 0; i < g_zero_counts.size(); i++) {
		std::fprintf(stderr, " %s %llu", ZeroName(static_cast<ZeroCompileEvent>(i)),
		             static_cast<unsigned long long>(g_zero_counts[i].load()));
	}
	std::fprintf(stderr, "\n");
	std::fflush(stderr);
}

} // namespace Libs::Graphics
