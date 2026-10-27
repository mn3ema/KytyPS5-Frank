#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cpuAffinity.h"
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
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
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
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
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
#include <xxhash.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

bool ShaderFailureNonFatal() {
	static const bool value = true;
	return value;
}

namespace {

// Switches only the SPIR-V emitter reads. A source entry's key records them as unset, a permutation
// entry's key (built on the full session key) with their values, so a shader translated under one
// value is not translated again under another: the translation is the expensive half (~85 ms a
// compute shader, on the GPU thread). KYTY_DENORM_FLUSH=0 had made every source a miss after
// sessions with the default, and every first draw of a shader stalled again (2026-10-27).
constexpr std::string_view EmitterOnlySwitches[] = {"KYTY_DENORM_FLUSH"};

void AppendSwitch(std::vector<uint8_t>& bytes, std::string_view name, std::string_view value) {
	const uint64_t name_size  = name.size();
	const uint64_t value_size = value.size();
	const auto*    name_bytes = reinterpret_cast<const uint8_t*>(&name_size);
	bytes.insert(bytes.end(), name_bytes, name_bytes + sizeof(name_size));
	bytes.insert(bytes.end(), name.begin(), name.end());
	bytes.push_back(1u);
	const auto* value_bytes = reinterpret_cast<const uint8_t*>(&value_size);
	bytes.insert(bytes.end(), value_bytes, value_bytes + sizeof(value_size));
	bytes.insert(bytes.end(), value.begin(), value.end());
}

// ShaderDiskCache::SessionKey's bytes with the emitter-only switches as SessionKey writes an unset
// one: [name size][name][0] instead of [name size][name][1][value size][value].
std::vector<uint8_t> SourceSessionKey(std::vector<uint8_t> key) {
	for (const auto name: EmitterOnlySwitches) {
		const char* value = std::getenv(std::string(name).c_str());
		if (value == nullptr) {
			continue;
		}
		std::vector<uint8_t> set;
		AppendSwitch(set, name, value);
		const auto found = std::search(key.begin(), key.end(), set.begin(), set.end());
		if (found == key.end()) {
			continue;
		}
		const auto flag = (found - key.begin()) + static_cast<ptrdiff_t>(sizeof(uint64_t) + name.size());
		key.erase(key.begin() + flag, key.begin() + (found - key.begin()) + static_cast<ptrdiff_t>(set.size()));
		key.insert(key.begin() + flag, uint8_t {0});
	}
	return key;
}

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

bool ReadShaderGuestMemoryRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
	const SrtQualifyRead qualify(address, values);
	const bool           ok = ReadShaderGuestMemoryRawImpl(userdata, address, values);
	qualify.Finish(ok, values);
	return ok;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
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
	return std::ranges::find(hashes, shader_hash) != hashes.end();
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
};

// Counts a finished pipeline build and puts it on the timeline (slow ones as a warning).
static void ReportPipelineBuilt(Profiler::Counter counter, const char* kind, uint64_t first_hash,
                                uint64_t second_hash, std::chrono::steady_clock::time_point begin) {
	const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                         std::chrono::steady_clock::now() - begin)
	                                         .count());
	Profiler::Add(counter);
	Profiler::Add(Profiler::Counter::PipelineBuildTime, us);
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

	void Submit(std::shared_ptr<PendingGraphicsPipeline> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
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
			job->result = ShaderRecompiler::TranslateProgram(job->code, job->options);
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
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		// Held around every walk of resource_plan, whose memo and replay state are mutable: the
		// draw-prep scanner (KYTY_DRAW_PREP) walks programs while the GPU thread does.
		std::mutex walk_mutex;
		// Draw prep: the serial of the prepared result whose refresh `resources` holds now (0 after a
		// walk here), so a later draw of the same packet can use it again.
		uint64_t prepared_serial = 0;
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

	// A compiled (or cached) permutation of the source: its module, bound for this stage.
	template <typename InputInfo>
	ShaderProgram AddPermutation(SourceEntry& source, const char* stage_name,
	                             const ShaderRecompiler::CompileOptions&  options,
	                             const std::vector<uint32_t>&             spirv,
	                             ShaderRecompiler::IR::CompiledShaderInfo program,
	                             InputInfo& input_info, uint32_t& push_data_cursor) {
		Profiler::Phases phases;
		phases.SetText("%s 0x%016" PRIx64, stage_name, options.shader_hash);
		KYTY_PROFILER_PHASE(phases, "Shader: SPIR-V validate", profiler::colors::Blue300);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, spirv);

		KYTY_PROFILER_PHASE(phases, "Shader: module create", profiler::colors::Blue300);
		const auto module = CompileSPV(spirv, device);
		EXIT_IF(module == nullptr);
		phases.End();
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(spirv.size()), options.wave_size);
		}
		source.permutations.push_back({
		    .specialization = source.specialization,
		    .program        = std::move(program),
		    .handle         = {.id          = ++next_shader_id,
		                       .module      = module,
		                       .spirv_words = static_cast<uint32_t>(spirv.size()),
		                       .hash        = options.shader_hash},
		});
		const auto& permutation = source.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &source.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

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
			disk_full_session_key = ShaderDiskCache::SessionKey(disk_session);
			disk_session_key      = SourceSessionKey(disk_full_session_key);
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
		writer.Put<uint8_t>('S');
		writer.Put(static_cast<uint32_t>(options.stage));
		writer.Put(options.shader_hash);
		writer.Put(options.wave_size);
		writer.Put(options.user_data_base);
		writer.Put(static_cast<uint32_t>(options.user_data.size()));
		writer.Put(options.bindless_images);
		writer.Put(options.non_fatal);
		writer.PutWords(code);
		writer.PutWords(params.back_code);
		ShaderDiskCache::AppendInputKey(writer, input_info);
		key = std::move(writer.bytes);
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

	// A permutation key is built on the full session key (SourceSessionKey: the source key leaves
	// the emitter-only switches unset).
	void BuildDiskPermutationKey(std::vector<uint8_t>& key, std::span<const uint8_t> source_key,
	                             const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                             uint32_t push_data_cursor) const {
		if (disk_full_session_key == disk_session_key) {
			BuildPermutationKey(key, source_key, specialization, push_data_cursor);
			return;
		}
		std::vector<uint8_t> full;
		full.reserve(disk_full_session_key.size() + source_key.size() - disk_session_key.size());
		full.insert(full.end(), disk_full_session_key.begin(), disk_full_session_key.end());
		full.insert(full.end(), source_key.begin() + static_cast<ptrdiff_t>(disk_session_key.size()),
		            source_key.end());
		BuildPermutationKey(key, full, specialization, push_data_cursor);
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
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": skip list", StageLabel(stage),
			                                params.hash);
			return ShaderProgram {};
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
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
		uint64_t gave_up_key = 0;
		[[maybe_unused]] uint64_t gave_up_shader_key = 0; // the shader as a whole (KYTY_GAVE_UP_SHADER_AFTER)
		[[maybe_unused]] uint32_t gave_up_limit      = 0;
		uint64_t                 fast_key = 0;
		ProgramsMap::value_type* fast_hit = nullptr;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
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
					return ShaderProgram {};
				}
			}
			// By the user data the expansion depends on (ExpansionInputsHash), not all of it.
			const auto make_gave_up_key = [&] {
				return ExpansionInputsHash(params, user_data, input_info.wave_size, shader_key) | 1u;
			};
			if (fast) {
				gave_up_key = make_gave_up_key();
				if (gave_up_fast.contains(gave_up_key)) {
					Profiler::Add(Profiler::Counter::SkippedGaveUp);
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": shader gave up",
					                                StageLabel(stage), params.hash);
					return ShaderProgram {};
				}
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
					fast_hit = it->second.entry;
				}
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
						}
					}
				}
			}
			Profiler::Add(Profiler::Counter::SkippedGaveUp);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": shader gave up",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		}
		KYTY_PROFILER_BLOCK("ProgramCache::Get");
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
				fast_entries[fast_key] = {entry, std::chrono::steady_clock::now()};
			}
		}
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
				               candidate.specialization == entry->second.specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
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
					if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
						return ShaderProgram {};
					}
				}
			}
			if (entry != nullptr) {
				BuildDiskPermutationKey(permutation_key, source_key, entry->second.specialization,
				                        push_data_cursor);
				std::vector<uint8_t>               payload;
				ShaderDiskCache::PermutationRecord record;
				if (!disk->Load(permutation_key, payload)) {
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
					return AddPermutation(entry->second, stage_name, options, record.spirv,
					                      std::move(record.program), input_info, push_data_cursor);
				}
			}
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
			translated = ShaderRecompiler::TranslateProgram(compile_code, options);
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
		if (entry == nullptr) {
			plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
			if (disk != nullptr) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(false, uses_clock, &*plan, payload)) {
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
			if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
				return ShaderProgram {};
			}
			if (disk != nullptr) {
				BuildDiskPermutationKey(permutation_key, source_key, entry->second.specialization,
				                        push_data_cursor);
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
		const auto handle = AddPermutation(entry->second, stage_name, options, compiled.spirv,
		                                   std::move(compiled.program), input_info,
		                                   push_data_cursor);
		{
			const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                         std::chrono::steady_clock::now() - compile_begin)
			                                         .count());
			Profiler::Add(Profiler::Counter::ShadersCompiled);
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

	ProgramCache(vk::Device device, bool shader_clock, bool bindless_images,
	             ShaderDiskCache::SessionInfo session, std::filesystem::path disk_directory)
	    : device(device), shader_clock(shader_clock), bindless_images(bindless_images),
	      disk_session(session), disk_verify(ShaderDiskCache::VerifyEnabled()) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		if (Config::GraphicsDebugDumpEnabled()) {
			// The dumps are written while translating: every shader has to be translated.
			PipelineCacheLog("Shader disk cache: off (graphics debug dump)");
		} else {
			disk = ShaderDiskCache::Store::Open(std::move(disk_directory));
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

	void FlushDisk() {
		if (disk != nullptr) {
			disk->Flush();
			disk->LogTotals("saved");
		}
	}
	~ProgramCache() {
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
	std::vector<uint8_t>                                        disk_full_session_key;
	bool                                                        disk_verify  = false;
	uint64_t                                                    disk_lookups = 0;
	std::unique_ptr<ShaderDiskCache::Store>                     disk;
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
			static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
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
		static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
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
			static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr;
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
			if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log && !reader.failed) {
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
	InitializeDriverCache();
	if (g_async_pipelines) {
		const auto threads = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 4u);
		m_compiler         = std::make_unique<PipelineCompiler>(threads);
		PipelineCacheLog("Vulkan pipelines: compiled on {} worker threads (wait {} ms)", threads,
		                 g_pipeline_wait.count());
	}
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->pending && pipeline->pending->done.load(std::memory_order_acquire)) {
				m_graphics.device.destroyPipeline(pipeline->pending->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
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

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
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
		// Pipelines of older builds stay in the file; past 1 GiB it starts over.
		constexpr uint64_t max_file_size = uint64_t {1} << 30u;
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
}

void PipelineCache::Save() {
	m_program_cache->FlushDisk();
	// No worker may use the driver cache once it is destroyed below; later pipelines are
	// compiled on the GPU thread (FinishPending).
	if (m_compiler != nullptr) {
		m_compiler->Stop();
	}
	if (m_driver_cache == nullptr) {
		return;
	}

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
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
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
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
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
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		if (!result.vertex[i]) {
			return {};
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
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	auto              program          = m_program_cache->Get(params, input_info, push_data_cursor);
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
                                    ShaderComputeInputInfo& input_info, DrawPrep::PacketResult& out) {
	out.stage_count = 0;
	if (!DrawPrep::IsGoodPair(regs.cs_regs.data_addr, UINT64_MAX)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		return false;
	}
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage        = input_info.lds_size_dwords * 4u >
	                         m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	if (m_program_cache->ScanStage(params, input_info, ShaderType::Compute, out.stages[0])) {
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
	auto build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input,
	                                     vertex_info, ps_input_info, programs, static_params);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
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
	if ((background || budget.count() > 0) && m_compiler != nullptr && !m_compiler->Stopped()) {
		// A pipeline the driver has cached finishes within the wait; a new one is compiled in
		// the background, and its draws are skipped until it is ready.
		auto job          = std::make_shared<PendingGraphicsPipeline>();
		job->build        = std::move(build);
		job->driver_cache = m_driver_cache;
		job->vs_hash      = vs_input_info.stage.program->shader_hash;
		job->ps_hash      = ps_active ? ps_input_info->stage.program->shader_hash : 0;
		cached->pending   = job;
		Profiler::Add(Profiler::Counter::PipelinesPending);
		m_compiler->Submit(job);
		(void)m_compiler->Wait(*job, budget);
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
	const auto result = CreateGraphicsPipeline(*build, m_driver_cache, &cached->pipeline);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
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

	auto cached = std::make_unique<Pipeline>();
	{
		const auto hash = input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0;
		Profiler::Phases zone;
		zone.SetText("CS 0x%016" PRIx64, hash);
		KYTY_PROFILER_PHASE(zone, "Compute pipeline build", profiler::colors::RedA100);
		const auto begin = std::chrono::steady_clock::now();
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
		ReportPipelineBuilt(Profiler::Counter::ComputePipelinesBuilt, "Compute", hash, 0, begin);
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

} // namespace Libs::Graphics
