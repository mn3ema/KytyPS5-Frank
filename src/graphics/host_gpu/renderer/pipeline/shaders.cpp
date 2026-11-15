#include "common/cpuAffinity.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompileProgress.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineRecipe.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStats.h"
#include "shaderCacheVersion.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cinttypes>
#include <cstdlib>
#include <limits>
#include <cstdio>
#include <thread>
#include <mutex>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

#include <xxhash.h>

namespace Libs::Graphics {

// KYTY_DISABLE_OPTIMIZATION_HASHES="hash,hash,...": build the pipelines of these guest shaders
// with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT, to tell driver code-generation problems apart
// from emulation problems.
// Slow pipelines (KYTY_SLOW_PIPELINES=0 turns this off): a pipeline the driver has been compiling
// for 30 s is written to _ShaderDump/slow_pipelines.txt ([slow-pipeline] on stderr) while the
// compile still runs, since NVIDIA's compiler can loop on a shader for good (VS 0xd1ea70c7117d4d3a
// with PS 0x85892b3b3d0a61ba, with or without optimization). From the next start, the draws of a
// listed graphics pipeline are skipped (PipelineCache) and a listed compute pipeline is built
// without driver optimization.
namespace {
bool SlowPipelinesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SLOW_PIPELINES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}
constexpr const char* SlowPipelineFile = "_ShaderDump/slow_pipelines.txt";
struct SlowKey {
	char     kind; // 'G' graphics (a: VS, b: PS), 'C' compute (a: CS)
	uint64_t a, b;
	bool     operator==(const SlowKey&) const = default;
};
const std::vector<SlowKey>& ListedSlowPipelines() {
	static const std::vector<SlowKey> listed = [] {
		std::vector<SlowKey> result;
		if (!SlowPipelinesEnabled()) {
			return result;
		}
		// Each line ends with the shader recompiler's version (its sources' hash): a pipeline is
		// listed only for the code that produced it, so a recompiler change gets a new try (the
		// sewer scent trail's pipeline compiles since its formatted loads share one function).
		if (auto* file = std::fopen(SlowPipelineFile, "r"); file != nullptr) {
			char line[256];
			while (std::fgets(line, sizeof(line), file) != nullptr) {
				char               kind = 0;
				unsigned long long a = 0, b = 0, version = 0;
				const int fields = std::sscanf(line, " %c %llx %llx %llx", &kind, &a, &b, &version);
				if (fields < 3) {
					continue;
				}
				if (fields < 4 || version != ShaderDiskCacheVersion::SourceVersion) {
					std::fprintf(stderr, "[slow-pipeline] %c %016llx %016llx: listed by another recompiler, tried again\n", kind, a, b);
					continue;
				}
				result.push_back({kind, a, b});
				std::fprintf(stderr, "[slow-pipeline] %c %016llx %016llx: listed (graphics: draws skipped; compute: no optimization)\n", kind, a, b);
			}
			std::fclose(file);
		}
		return result;
	}();
	return listed;
}
bool SlowPipelineListed(char kind, uint64_t a, uint64_t b) {
	const auto& listed = ListedSlowPipelines();
	return std::ranges::find(listed, SlowKey {kind, a, b}) != listed.end();
}
struct InFlight {
	SlowKey                               key;
	std::chrono::steady_clock::time_point start;
	bool                                  written = false;
};
std::mutex                     g_inflight_mutex;
std::vector<InFlight*>         g_inflight;
std::atomic<bool>              g_slow_watch_started {false};
void SlowWatch() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::lock_guard lock(g_inflight_mutex);
		const auto      now = std::chrono::steady_clock::now();
		for (auto* entry: g_inflight) {
			if (entry->written || now - entry->start < std::chrono::seconds(30)) {
				continue;
			}
			entry->written = true;
			std::fprintf(stderr, "[slow-pipeline] %c %016llx %016llx: compiling for 30 s; listed in %s\n",
			             entry->key.kind, static_cast<unsigned long long>(entry->key.a),
			             static_cast<unsigned long long>(entry->key.b), SlowPipelineFile);
			std::fflush(stderr);
			if (auto* file = std::fopen(SlowPipelineFile, "a"); file != nullptr) {
				std::fprintf(file, "%c %016llx %016llx %016llx\n", entry->key.kind,
				             static_cast<unsigned long long>(entry->key.a),
				             static_cast<unsigned long long>(entry->key.b),
				             static_cast<unsigned long long>(ShaderDiskCacheVersion::SourceVersion));
				std::fclose(file);
			}
		}
	}
}
} // namespace

// PipelineCache: the draws of this graphics pipeline are skipped (it never finished compiling).
bool SlowGraphicsPipelineListed(uint64_t vs_hash, uint64_t ps_hash) {
	return SlowPipelineListed('G', vs_hash, ps_hash);
}

namespace {
// Around one driver pipeline compile.
class CompileWatch {
public:
	CompileWatch(char kind, uint64_t a, uint64_t b): m_entry {{kind, a, b}, std::chrono::steady_clock::now()} {
		if (!SlowPipelinesEnabled()) {
			return;
		}
		if (!g_slow_watch_started.exchange(true)) {
			std::thread(SlowWatch).detach();
		}
		std::lock_guard lock(g_inflight_mutex);
		g_inflight.push_back(&m_entry);
		m_registered = true;
	}
	~CompileWatch() {
		// A compile of 5 s or more is named when it ends (one that was listed above, too).
		const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
		                                                   m_entry.start)
		                         .count();
		if (seconds >= 5.0) {
			std::fprintf(stderr, "[slow-pipeline] %c %016llx %016llx: compiled in %.1f s\n",
			             m_entry.key.kind, static_cast<unsigned long long>(m_entry.key.a),
			             static_cast<unsigned long long>(m_entry.key.b), seconds);
		}
		if (!m_registered) {
			return;
		}
		std::lock_guard lock(g_inflight_mutex);
		std::erase(g_inflight, &m_entry);
	}
	KYTY_CLASS_NO_COPY(CompileWatch);

private:
	InFlight m_entry;
	bool     m_registered = false;
};
} // namespace

static bool DisableOptimizationFor(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		if (const char* value = std::getenv("KYTY_DISABLE_OPTIMIZATION_HASHES"); value != nullptr) {
			std::string list(value);
			size_t      start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(std::strtoull(list.substr(start, end - start).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		}
		return result;
	}();
	return std::ranges::find(hashes, shader_hash) != hashes.end();
}

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

bool VertexAttributeFetched(const ShaderVertexInputInfo& info, int attribute) {
	// KYTY_VERTEX_CONST_ATTRIB_OMIT=0 binds every attribute as before.
	static const bool omit = [] {
		const char* value = std::getenv("KYTY_VERTEX_CONST_ATTRIB_OMIT");
		return value == nullptr || value[0] != '0';
	}();
	if (!omit || !info.fetch_embedded || info.stage.program == nullptr || attribute < 0 ||
	    static_cast<size_t>(attribute) >= info.stage.program->info.vertex_fetch_components.size()) {
		return true;
	}
	return info.stage.program->info.vertex_fetch_components[attribute] != 0;
}

// The dynamic states of a graphics pipeline (and so of its libraries: LibraryDynamicStates picks
// each kind's), in one order for every caller, the library warm-up included.
static std::vector<vk::DynamicState> GraphicsDynamicStates(const GraphicContext& graphics, uint32_t color_count,
                                                           bool gpl) {
	std::vector<vk::DynamicState> states = {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	if (color_count != 0) {
		states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}
	if (PipelineDynamicRasterStateEnabled()) {
		// Core Vulkan 1.3 (cull mode, front face, depth-bounds test enable) and 1.0 (depth bounds;
		// the depthBounds feature is required on these hosts). The key holds zeroes for them; the
		// draw records the values from the same registers (SetGraphicsDynamicParams).
		states.push_back(vk::DynamicState::eCullMode);
		states.push_back(vk::DynamicState::eFrontFace);
#if !defined(__APPLE__)
		states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
		states.push_back(vk::DynamicState::eDepthBounds);
#endif
	}
	if (gpl && graphics.eds3_raster_enabled) {
		// Step D2: pre-rasterization libraries per module only (DrawDynamicRaster sets these).
		states.push_back(vk::DynamicState::eDepthClipEnableEXT);
		states.push_back(vk::DynamicState::eDepthClipNegativeOneToOneEXT);
		states.push_back(vk::DynamicState::ePolygonModeEXT);
	}
	return states;
}

static uint32_t GplKnob(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr ? 0u : static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static bool logged_113 = false;
		if (!logged_113) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
			logged_113 = true;
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static bool logged_121 = false;
		if (!logged_121) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
			logged_121 = true;
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u descriptor=%08x,%08x,%08x,%08x\n", raw_format,
		     res.fields[0], res.fields[1], res.fields[2], res.fields[3]);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (const auto& binding: program.bindings.descriptors) {
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind),
		     NativeDescriptorType(binding.kind), NativeDescriptorCount(binding), stage, nullptr});
	}
}

static bool UsesBindless(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return std::ranges::any_of(program.info.images, &ShaderRecompiler::IR::ImageResource::bindless) ||
	       std::ranges::any_of(program.info.samplers,
	                           &ShaderRecompiler::IR::SamplerResource::bindless);
}

// The content of one stage set of a linked pipeline (set 0 or set 2) with what set 1 is: part of
// the stage libraries' keys (step C). The library prefetch builds the same key from one stage's
// bindings (GplWarmLibrary).
static uint64_t StageSetKey(std::span<const vk::DescriptorSetLayoutBinding> bindings, bool push,
                            bool uses_bindless, bool stage_bindless) {
	std::vector<uint32_t> words {push ? 1u : 0u, uses_bindless ? 1u : 0u, stage_bindless ? 1u : 0u};
	for (const auto& binding: bindings) {
		words.push_back(binding.binding);
		words.push_back(static_cast<uint32_t>(binding.descriptorType));
		words.push_back(binding.descriptorCount);
		words.push_back(static_cast<uint32_t>(VkShaderStageFlags(binding.stageFlags)));
		EXIT_IF(binding.pImmutableSamplers != nullptr);
	}
	return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

// Set 0 is the pipeline's own descriptors; set 1, when a stage samples bindless images, is the
// shared bindless table.
static void FillSetLayouts(GraphicContext& graphics, const PipelineCache::Pipeline& pipeline,
                           std::array<vk::DescriptorSetLayout, 2>& layouts, uint32_t& count) {
	layouts[0] = pipeline.descriptor_set_layout;
	count      = 1;
	if (pipeline.uses_bindless) {
		EXIT_IF(graphics.bindless_layout == nullptr);
		layouts[1] = graphics.bindless_layout;
		count      = 2;
	}
}

static vk::DescriptorSetLayout CreateSetLayout(GraphicContext& graphics,
                                               std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                               bool push) {
	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                           : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	vk::DescriptorSetLayout layout;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &layout) != vk::Result::eSuccess);
	return layout;
}

// KYTY_STAGE_SETS: set 1 when no stage samples bindless images (layouts must not have holes).
vk::DescriptorSetLayout EmptySetLayout(GraphicContext& graphics) {
	static std::mutex              mutex;
	static vk::DescriptorSetLayout layout = nullptr;
	std::lock_guard                lock(mutex);
	if (layout == nullptr) {
		layout = CreateSetLayout(graphics, {}, false);
	}
	return layout;
}

vk::DescriptorSet EmptyDescriptorSet(GraphicContext& graphics) {
	static std::mutex        mutex;
	static vk::DescriptorSet set = nullptr;
	std::lock_guard          lock(mutex);
	if (set == nullptr) {
		const vk::DescriptorPoolSize size {vk::DescriptorType::eSampler, 1};
		vk::DescriptorPoolCreateInfo pool_info {};
		pool_info.maxSets       = 1;
		pool_info.poolSizeCount = 1;
		pool_info.pPoolSizes    = &size;
		vk::DescriptorPool pool;
		EXIT_IF(graphics.device.createDescriptorPool(&pool_info, nullptr, &pool) != vk::Result::eSuccess);
		const auto layout = EmptySetLayout(graphics);
		vk::DescriptorSetAllocateInfo alloc {};
		alloc.descriptorPool     = pool;
		alloc.descriptorSetCount = 1;
		alloc.pSetLayouts        = &layout;
		EXIT_IF(graphics.device.allocateDescriptorSets(&alloc, &set) != vk::Result::eSuccess);
	}
	return set;
}

namespace {
constexpr uint32_t                                                     AuxShardMax = 32;
std::array<std::array<std::atomic<VkPipelineCache>, AuxShardMax>, 2> g_aux_caches {};
std::array<std::atomic<uint32_t>, 2>                                   g_aux_count {};
} // namespace

void SetAuxCaches(AuxCache which, std::span<const vk::PipelineCache> caches) {
	const auto w = static_cast<size_t>(which);
	g_aux_count[w].store(0, std::memory_order_release);
	const auto n = std::min<size_t>(caches.size(), AuxShardMax);
	for (size_t i = 0; i < n; i++) {
		g_aux_caches[w][i].store(static_cast<VkPipelineCache>(caches[i]), std::memory_order_relaxed);
	}
	g_aux_count[w].store(static_cast<uint32_t>(n), std::memory_order_release);
}

vk::PipelineCache AuxCacheFor(AuxCache which, uint64_t module_hash) {
	const auto w = static_cast<size_t>(which);
	const auto n = g_aux_count[w].load(std::memory_order_acquire);
	if (n == 0) {
		return {};
	}
	return vk::PipelineCache(g_aux_caches[w][module_hash % n].load(std::memory_order_relaxed));
}

namespace {
thread_local int      g_last_cache_hit = -1;
thread_local uint64_t g_warm_module_hash = 0; // the compute warm-up's (unregistered) module
void NoteFeedback(const vk::PipelineCreationFeedback& feedback) {
	g_last_cache_hit = !(feedback.flags & vk::PipelineCreationFeedbackFlagBits::eValid) ? -1
	                   : (feedback.flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit) ? 1
	                                                                                                          : 0;
}
} // namespace

int LastPipelineCacheHit() {
	return g_last_cache_hit;
}

uint64_t ModuleShardHash(vk::ShaderModule module) {
	PipelineRecipe::Hash128 hash;
	return PipelineRecipe::FindModuleHash(module, hash) ? hash.low : 0;
}

bool GplBindlessSet3() {
	static const bool enabled = GplKnob("KYTY_GPL_BINDLESS3") != 0;
	return enabled;
}

static std::mutex                                            g_twin_mutex;
static std::unordered_map<VkShaderModule, vk::ShaderModule> g_twins;

void RegisterBindlessSet3Twin(vk::Device device, vk::ShaderModule module, std::span<const uint32_t> spirv) {
	if (!GplBindlessSet3() || spirv.size() < 5) {
		return;
	}
	std::vector<uint32_t> words(spirv.begin(), spirv.end());
	bool                  patched = false;
	for (size_t i = 5; i < words.size();) {
		const uint32_t count  = words[i] >> 16u;
		const uint32_t opcode = words[i] & 0xffffu;
		if (count == 0 || i + count > words.size()) {
			return;
		}
		// OpDecorate %id DescriptorSet 1
		if (opcode == 71u && count == 4 && words[i + 2] == 34u && words[i + 3] == 1u) {
			words[i + 3] = 3u;
			patched      = true;
		}
		i += count;
	}
	if (!patched) {
		return;
	}
	vk::ShaderModuleCreateInfo info {};
	info.codeSize = words.size() * sizeof(uint32_t);
	info.pCode    = words.data();
	vk::ShaderModule twin;
	if (device.createShaderModule(&info, nullptr, &twin) != vk::Result::eSuccess) {
		return;
	}
	std::lock_guard lock(g_twin_mutex);
	g_twins[static_cast<VkShaderModule>(module)] = twin;
}

vk::ShaderModule BindlessSet3Twin(vk::ShaderModule module) {
	std::lock_guard lock(g_twin_mutex);
	const auto      found = g_twins.find(static_cast<VkShaderModule>(module));
	return found == g_twins.end() ? vk::ShaderModule {} : found->second;
}

// KYTY_STAGE_SETS: which of the two stage sets is the push set (one per pipeline layout): the
// pixel shader's when it fits, else the pre-rasterization stage's when that fits.
void SplitStageSets(uint32_t max_push, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                    std::vector<vk::DescriptorSetLayoutBinding>& pre,
                    std::vector<vk::DescriptorSetLayoutBinding>& ps, bool& pre_push, bool& ps_push) {
	uint32_t pre_count = 0;
	uint32_t ps_count  = 0;
	for (const auto& binding: bindings) {
		if (binding.stageFlags == vk::ShaderStageFlagBits::eFragment) {
			ps.push_back(binding);
			ps_count += binding.descriptorCount;
		} else {
			pre.push_back(binding);
			pre_count += binding.descriptorCount;
		}
	}
	ps_push  = !ps.empty() && ps_count <= max_push;
	pre_push = !ps_push && !pre.empty() && pre_count <= max_push;
}

static void CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	uint32_t descriptor_count = 0;
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
	}
	pipeline.uses_push_descriptors = descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &create, nullptr, &pipeline.descriptor_set_layout) != vk::Result::eSuccess);
}

GraphicsPipelineBuild::~GraphicsPipelineBuild() {
	if (tess_control != nullptr) {
		device.destroyShaderModule(tess_control, nullptr);
	}
	if (tess_eval != nullptr) {
		device.destroyShaderModule(tess_eval, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::unique_ptr<GraphicsPipelineBuild>
PrepareGraphicsPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                        const PipelineRenderingState&          rendering,
                        const PipelineVertexInputState&        vertex_input,
                        std::span<const ShaderVertexInputInfo> vertex_info,
                        const ShaderPixelInputInfo*            ps_input_info,
                        const PipelineCache::GraphicsPrograms& programs,
                        const PipelineStaticParameters&        static_params) {
	auto  owner     = std::make_unique<GraphicsPipelineBuild>();
	auto& build     = *owner;
	build.device    = graphics.device;
	build.rendering = rendering;

	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		build.rect_tcs_spirv = shaders.control;
		build.rect_tes_spirv = shaders.evaluation;
		build.tess_control = CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(build.tess_control));
		}

		build.tess_eval = CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(build.tess_eval));
		}
	}

	EXIT_NOT_IMPLEMENTED(rect_list &&
	                     (build.tess_control == nullptr || build.tess_eval == nullptr));

	uint32_t shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		build.stages[shader_stage_count++] = {.stage  = NativeShaderStage(vertex_info[i].logical_stage),
		                                      .module = programs.vertex[i].module,
		                                      .pName  = "main"};
	}
	if (rect_list) {
		build.stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eTessellationControl,
		                                      .module = build.tess_control,
		                                      .pName  = "main"};
		build.stages[shader_stage_count++] = {.stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                      .module = build.tess_eval,
		                                      .pName  = "main"};
	}
	if (ps_active) {
		build.stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                      .module = pixel_program.module,
		                                      .pName  = "main"};
	}
	// One guest wave per host subgroup where the device lets the stage require it (mesh and
	// pixel shaders on AMD); GraphicsSubgroupSize and the mesh's host_subgroup_size agree.
	for (uint32_t index = 0; index < shader_stage_count; index++) {
		auto&      stage   = build.stages[index];
		const bool is_mesh = stage.stage == vk::ShaderStageFlagBits::eMeshEXT;
		if (!is_mesh && stage.stage != vk::ShaderStageFlagBits::eFragment) {
			continue;
		}
		const auto wave_size = is_mesh ? vs_input_info.mesh.wave_size
		                               : ps_input_info->stage.program->wave_size;
		const auto required  = graphics.GraphicsSubgroupSize(stage.stage, wave_size);
		if (required != 0) {
			build.subgroup_sizes[index].requiredSubgroupSize = required;
			stage.pNext = &build.subgroup_sizes[index];
		} else if (wave_size < graphics.subgroup_size) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true)) {
				LOGF("Vulkan subgroup: wave%u %s shader on a %u-wide host subgroup the device "
				     "cannot narrow; its lane operations mix two waves\n",
				     wave_size, is_mesh ? "mesh" : "pixel", graphics.subgroup_size);
			}
		}
	}

	auto& input_attr = build.input_attr;
	auto& input_desc = build.input_desc;
	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	uint32_t input_attribute_count = 0;
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		// An embedded fetch whose components are all constants reads no memory and has no
		// shader input; its descriptor is not consumed and can hold anything (format 0 ended
		// the emulator in GetInputFormat). The remaining attributes keep their locations.
		if (!VertexAttributeFetched(vs_input_info, static_cast<int>(index))) {
			continue;
		}
		auto& native_attribute    = input_attr[input_attribute_count++];
		native_attribute.binding  = vertex_input.attributes[index].binding;
		native_attribute.location = index;
		native_attribute.offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], native_attribute.format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, native_attribute.binding, native_attribute.offset,
				     input_desc[native_attribute.binding].stride,
				     static_cast<int>(native_attribute.format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static bool logged = false;
			if (!logged) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
				logged = true;
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	build.vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	build.vertex_input_info.pVertexBindingDescriptions      = input_desc.data();
	build.vertex_input_info.vertexAttributeDescriptionCount = input_attribute_count;
	build.vertex_input_info.pVertexAttributeDescriptions    = input_attr.data();

	build.input_assembly.topology = static_params.topology;
	build.input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	build.depth_clip_control.negativeOneToOne =
	    (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);
	build.viewport_state.pNext = &build.depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	build.clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	auto& rasterizer = build.rasterizer;
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &build.clip_ext;
#endif
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		build.provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		build.provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &build.provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	build.multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	build.multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	build.multisampling.minSampleShading     = 1.0f;

	auto& color_blend_attachment = build.color_blend;
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	for (uint32_t i = 0; i < rendering.color_count; i++) {
		build.color_write_enable[i] = VK_TRUE;
	}

	build.color_write.attachmentCount    = rendering.color_count;
	build.color_write.pColorWriteEnables = build.color_write_enable.data();

	auto& color_blending = build.color_blending;
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &build.color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment.data();

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		graphics_stages |= native_stage;
		pipeline.uses_bindless |= UsesBindless(*stage.stage.program);
		build.pre_bindless |= UsesBindless(*stage.stage.program);
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_bindless |= UsesBindless(*ps_input_info->stage.program);
		build.ps_bindless = UsesBindless(*ps_input_info->stage.program);
	}
	vk::PushConstantRange push_constants {graphics_stages, 0,
	                                      ShaderRecompiler::IR::NativePushConstantSize};
	build.descriptor_bindings = descriptor_bindings;
	build.uses_bindless       = pipeline.uses_bindless;
	build.push_stages         = graphics_stages;
	build.push_size           = ShaderRecompiler::IR::NativePushConstantSize;

	std::array<vk::DescriptorSetLayout, 4> set_layouts {};
	uint32_t                               set_layout_count = 0;
	if (ShaderRecompiler::IR::StageDescriptorSets() && ps_active) {
		std::vector<vk::DescriptorSetLayoutBinding> pre_bindings;
		std::vector<vk::DescriptorSetLayoutBinding> ps_bindings;
		bool                                        pre_push = false;
		bool                                        ps_push  = false;
		// Linked pipelines with a push-descriptor set broke on NVIDIA 621.14: mesh ones faulted the
		// GPU (InstructionPointer + ReadInvalid at 0x5800000 within a minute), vertex ones stalled
		// the title at the same spot (v436i bisect; with or without the driver cache; the
		// validation layer crashed in vkCmdPushDescriptorSetKHR). They take both stage sets from
		// the heap. KYTY_GPL_NOPUSH: 0 push allowed, 1 (default) no push in linked pipelines,
		// 2 only mesh ones without push.
		static const uint32_t gpl_no_push = [] {
			const char* value = std::getenv("KYTY_GPL_NOPUSH");
			return value == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
		}();
		const bool no_push = graphics.graphics_pipeline_library_enabled && !rect_list &&
		                     (gpl_no_push == 1 || (gpl_no_push == 2 && mesh));
		SplitStageSets(no_push ? 0u : graphics.max_push_descriptors, descriptor_bindings, pre_bindings,
		               ps_bindings, pre_push, ps_push);
		pipeline.descriptor_set_layout = CreateSetLayout(graphics, pre_bindings, pre_push);
		pipeline.uses_push_descriptors = pre_push;
		pipeline.ps_set_layout         = CreateSetLayout(graphics, ps_bindings, ps_push);
		pipeline.ps_uses_push          = ps_push;
		set_layouts[0] = pipeline.descriptor_set_layout;
		set_layouts[1] = pipeline.uses_bindless ? graphics.bindless_layout : EmptySetLayout(graphics);
		set_layouts[2] = pipeline.ps_set_layout;
		EXIT_IF(set_layouts[1] == nullptr);
		set_layout_count = 3;
		static const uint32_t gpl_skip = GplKnob("KYTY_GPL_SKIP");
		// KYTY_GPL_HASH_MASK / KYTY_GPL_HASH_VALUE (hex): only pairs whose VS^PS hash matches are
		// linked (bisect to one pipeline, v436o).
		static const uint64_t hash_mask = [] {
			const char* v = std::getenv("KYTY_GPL_HASH_MASK");
			return v == nullptr ? 0ull : std::strtoull(v, nullptr, 16);
		}();
		static const uint64_t hash_value = [] {
			const char* v = std::getenv("KYTY_GPL_HASH_VALUE");
			return v == nullptr ? 0ull : std::strtoull(v, nullptr, 16);
		}();
		const uint64_t pair_hash = vs_input_info.stage.program->shader_hash ^
		                           ps_input_info->stage.program->shader_hash;
		const bool     hash_ok   = (pair_hash & hash_mask) == (hash_value & hash_mask);
		// KYTY_GPL_BARY=0: pixel shaders with fragment barycentrics stay monolithic. The faults
		// first blamed on them (v436o/p bisects) were the bindless table's eAll stage flags in an
		// independent-sets layout (bindlessTable.cpp); with those fixed they link cleanly (v436w).
		static const bool link_bary = [] {
			const char* value = std::getenv("KYTY_GPL_BARY");
			return value == nullptr || value[0] != '0';
		}();
		// Any fragment-barycentric use: per-vertex inputs or the BaryCoord built-ins (v436p bisect:
		// VS 0eb1a5925ffaa072 + PS 8aaab2777faaaa72, BaryCoordKHR without per-vertex inputs,
		// faults alone).
		const bool        ps_per_vertex = std::ranges::any_of(
            ps_input_info->stage.program->info.inputs,
            [](const ShaderRecompiler::IR::StageInput& input) {
	            using Kind = ShaderRecompiler::IR::StageInputKind;
	            return input.per_vertex || input.kind == Kind::BaryCoordSmooth ||
	                   input.kind == Kind::BaryCoordSmoothCentroid ||
	                   input.kind == Kind::BaryCoordNoPerspective;
            });
		// Bisect bits (v436h): 4 PS push set, 8 pre-raster push set, 16 bindless, 32 no depth
		// attachment, 64 no pixel shader, 128 everything with a pixel shader, 256 non-mesh,
		// 512 provoking vertex last, 1024 provoking vertex first (v436s).
		if (graphics.graphics_pipeline_library_enabled) {
			static std::atomic<uint32_t> last {0};
			static std::atomic<uint32_t> total {0};
			if (static_params.provoking_vtx_last) last.fetch_add(1);
			if (const auto n = total.fetch_add(1) + 1; (n & 63u) == 0) {
				std::printf("[gpl] provoking vertex last in %u of %u stage-set pipelines\n", last.load(), n);
				std::fflush(stdout);
			}
		}
		if (ps_per_vertex && !link_bary && graphics.graphics_pipeline_library_enabled) {
			static std::atomic<uint32_t> kept {0};
			if (const auto n = kept.fetch_add(1) + 1; (n & 31u) == 0) {
				std::printf("[gpl] %u pipelines with a barycentric pixel shader kept monolithic\n", n);
				std::fflush(stdout);
			}
		}
		build.gpl = graphics.graphics_pipeline_library_enabled && !rect_list && hash_ok &&
		            (link_bary || !ps_per_vertex) &&
		            !(mesh && (gpl_skip & 1u) != 0) && !(tessellation && (gpl_skip & 2u) != 0) &&
		            !(ps_push && (gpl_skip & 4u) != 0) && !(pre_push && (gpl_skip & 8u) != 0) &&
		            !(pipeline.uses_bindless && (gpl_skip & 16u) != 0) &&
		            !(!with_depth && (gpl_skip & 32u) != 0) && !(!ps_active && (gpl_skip & 64u) != 0) &&
		            !(ps_active && (gpl_skip & 128u) != 0) && !(!mesh && (gpl_skip & 256u) != 0) &&
		            !(static_params.provoking_vtx_last && (gpl_skip & 512u) != 0) &&
		            !(!static_params.provoking_vtx_last && (gpl_skip & 1024u) != 0);
		// Step D3: a linked pipeline always has the bindless table as set 1 (bound every draw),
		// so a library's layout depends on its own stage only, not on whether the other stage
		// samples bindless (KYTY_GPL_BINDLESS_ALWAYS=0: only when a stage uses it).
		static const bool bindless_always = [] {
			const char* value = std::getenv("KYTY_GPL_BINDLESS_ALWAYS");
			return value == nullptr || value[0] != '0';
		}();
		if (build.gpl && bindless_always && graphics.bindless_layout != nullptr) {
			pipeline.uses_bindless = true;
			set_layouts[1]         = graphics.bindless_layout;
		}
		// KYTY_GPL_BINDLESS3: in independent-sets layouts the bindless table moves to set 3 (set 1
		// stays an empty set) and the stages use twins of their modules patched to match.
		if (build.gpl && pipeline.uses_bindless && GplBindlessSet3()) {
			set_layouts[1]        = EmptySetLayout(graphics);
			set_layouts[3]        = graphics.bindless_layout;
			set_layout_count      = 4;
			pipeline.bindless_set = 3;
			for (uint32_t i = 0; i < shader_stage_count; i++) {
				if (const auto twin = BindlessSet3Twin(build.stages[i].module); twin) {
					build.stages[i].module = twin;
				}
			}
		}
		build.set_layouts = set_layouts;
		build.set_count   = set_layout_count;
		pipeline.independent_sets = build.gpl;
		// Step D5: a linked pipeline's push range names every graphics stage the device has, so
		// a fragment library does not depend on whether the pipeline is vertex, mesh or
		// tessellated (the commit pushes with Pipeline::push_stages).
		if (build.gpl) {
			// Tessellation is a required device feature (vulkanWindow.cpp).
			vk::ShaderStageFlags all = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
			                           vk::ShaderStageFlagBits::eTessellationControl |
			                           vk::ShaderStageFlagBits::eTessellationEvaluation;
			if (graphics.mesh_shader_enabled) {
				all |= vk::ShaderStageFlagBits::eMeshEXT;
			}
			push_constants.stageFlags = all;
			build.push_stages         = all;
			pipeline.push_stages      = all;
		}
		build.robustness          = graphics.pipeline_robustness_enabled;
		// The libraries' keys: each stage set's content, with set 1 (bindless or empty).
		build.gpl_pre_set_key = StageSetKey(pre_bindings, pre_push, pipeline.uses_bindless, build.pre_bindless);
		build.gpl_ps_set_key  = StageSetKey(ps_bindings, ps_push, pipeline.uses_bindless, build.ps_bindless);
	} else {
		CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
		std::array<vk::DescriptorSetLayout, 2> two {};
		FillSetLayouts(graphics, pipeline, two, set_layout_count);
		set_layouts[0] = two[0];
		set_layouts[1] = two[1];
	}
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_layout_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	if (build.gpl) {
		pipeline_layout_info.flags = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
	}
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout begin VS=%" PRIu64 " PS=%" PRIu64
		     " set_layouts=1 push_constants=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0, 1u);
	}
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                   &pipeline.pipeline_layout);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout done result=%s layout=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	auto& depth_stencil_info = build.depth_stencil;
	depth_stencil_info.depthBoundsTestEnable =
#if defined(__APPLE__)
	    VK_FALSE; // MoltenVK lacks the depthBounds feature; depth-bounds testing is disabled
#else
	    (static_params.depth_bounds_test_enable ? VK_TRUE : VK_FALSE);
#endif
	depth_stencil_info.minDepthBounds    = static_params.depth_min_bounds;
	depth_stencil_info.maxDepthBounds    = static_params.depth_max_bounds;

	build.dynamic_states = GraphicsDynamicStates(graphics, rendering.color_count, build.gpl);
	pipeline.eds3_raster = build.gpl && graphics.eds3_raster_enabled;

	build.dynamic_state.dynamicStateCount = static_cast<uint32_t>(build.dynamic_states.size());
	build.dynamic_state.pDynamicStates    = build.dynamic_states.data();

	auto& pipeline_info                          = build.pipeline_info;
	build.rendering_info.colorAttachmentCount    = build.rendering.color_count;
	build.rendering_info.pColorAttachmentFormats = build.rendering.color_formats.data();
	build.rendering_info.depthAttachmentFormat   = build.rendering.depth_format;
	build.rendering_info.stencilAttachmentFormat = build.rendering.stencil_format;
	pipeline_info.pNext                          = &build.rendering_info;
	pipeline_info.stageCount                     = shader_stage_count;
	pipeline_info.pStages                        = build.stages.data();
	pipeline_info.pVertexInputState              = mesh ? nullptr : &build.vertex_input_info;
	pipeline_info.pInputAssemblyState            = mesh ? nullptr : &build.input_assembly;
	build.tessellation.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	pipeline_info.pTessellationState  = (rect_list || tessellation) ? &build.tessellation : nullptr;
	pipeline_info.pViewportState      = &build.viewport_state;
	pipeline_info.pRasterizationState = &build.rasterizer;
	pipeline_info.pMultisampleState   = &build.multisampling;
	pipeline_info.pDepthStencilState  = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState    = &build.color_blending;
	pipeline_info.pDynamicState       = &build.dynamic_state;
	pipeline_info.layout              = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex   = -1;
	build.mesh                        = mesh;

	const uint64_t ps_hash = ps_active && ps_input_info->stage
	                             ? ps_input_info->stage.program->shader_hash
	                             : 0;
	build.vs_hash = vs_input_info.stage.program->shader_hash;
	build.ps_hash = ps_hash;
	bool disable_optimization = (ps_hash != 0 && DisableOptimizationFor(ps_hash)) ||
	                            SlowPipelineListed('G', build.vs_hash, ps_hash);
	for (const auto& info: vertex_info) {
		disable_optimization = disable_optimization ||
		                       (info.stage && DisableOptimizationFor(info.stage.program->shader_hash));
	}
	if (disable_optimization) {
		pipeline_info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
		LOGF("PipelineTrace: graphics pipeline VS=0x%016" PRIx64 " PS=0x%016" PRIx64
		     " built without optimization\n",
		     vs_input_info.stage.program->shader_hash, ps_hash);
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     build.dynamic_state.dynamicStateCount);
	}
	if (build.gpl && GplOptimizeEnabled()) {
		pipeline.optimized   = std::make_shared<std::atomic<VkPipeline>>(VK_NULL_HANDLE);
		build.optimized_slot = pipeline.optimized;
	}
	return owner;
}

// KYTY_GPL bisect knobs (v436c): KYTY_GPL_SKIP bit 1 mesh, bit 2 tessellation (monolithic);
// KYTY_GPL_FULLSETS=1 every library gets all three set layouts; KYTY_GPL_LTO=1 the link is
// optimized (libraries retain the link-time information); KYTY_GPL_NOCACHE=1 no driver cache,
// 2 a private in-memory one. KYTY_GPL_SHARE=0: libraries per pipeline (step B1).

// Step C: the libraries of one kind are shared by every pipeline whose state for that kind is the
// same (a content key), so a new vertex/pixel pairing costs only a link. Libraries live for the
// session (the driver crashed binding a pipeline whose libraries were destroyed, v436).
namespace {

// Step D1: the dynamic states a library of `kind` consumes (the driver ignores the rest). Both
// its key and its create info use only these, so a library is the same object whatever else
// the pipeline it was first made for had dynamic (and hits the driver cache the same way).
std::vector<vk::DynamicState> LibraryDynamicStates(const vk::PipelineDynamicStateCreateInfo* dynamic,
                                                   vk::GraphicsPipelineLibraryFlagsEXT     kind) {
	using D = vk::DynamicState;
	using K = vk::GraphicsPipelineLibraryFlagBitsEXT;
	const auto relevant = [&](D state) {
		switch (state) {
			case D::eViewportWithCount:
			case D::eScissorWithCount:
			case D::eLineWidth:
			case D::eDepthBiasEnable:
			case D::eDepthBias:
			case D::eCullMode:
			case D::eFrontFace:
			case D::eDepthClipEnableEXT:
			case D::eDepthClipNegativeOneToOneEXT:
			case D::ePolygonModeEXT: return static_cast<bool>(kind & K::ePreRasterizationShaders);
			case D::eDepthTestEnable:
			case D::eDepthWriteEnable:
			case D::eDepthCompareOp:
			case D::eStencilTestEnable:
			case D::eStencilOp:
			case D::eStencilCompareMask:
			case D::eStencilReference:
			case D::eStencilWriteMask:
			case D::eDepthBoundsTestEnable:
			case D::eDepthBounds: return static_cast<bool>(kind & K::eFragmentShader);
			case D::eBlendConstants:
			case D::eColorWriteEnableEXT: return static_cast<bool>(kind & K::eFragmentOutputInterface);
			default: return true; // unknown: keep it in every library
		}
	};
	std::vector<vk::DynamicState> states;
	for (uint32_t i = 0; dynamic != nullptr && i < dynamic->dynamicStateCount; i++) {
		if (relevant(dynamic->pDynamicStates[i])) {
			states.push_back(dynamic->pDynamicStates[i]);
		}
	}
	return states;
}

struct LibraryKey {
	std::vector<uint8_t> bytes;
	template <typename T>
	void Add(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto* p = reinterpret_cast<const uint8_t*>(&value);
		bytes.insert(bytes.end(), p, p + sizeof(T));
	}
	// Step D1: only the dynamic states the library's kind consumes (the rest is ignored by the
	// driver), so e.g. colour-write-enable (output state) does not split pre-raster libraries.
	void AddDynamic(const vk::PipelineDynamicStateCreateInfo* dynamic,
	                vk::GraphicsPipelineLibraryFlagsEXT kind) {
		const auto states = LibraryDynamicStates(dynamic, kind);
		Add(static_cast<uint32_t>(states.size()));
		for (const auto state: states) {
			Add(state);
		}
	}
	// A module by its SPIR-V's hash, not its handle: handles of destroyed modules (the recipe
	// warm-up's temporaries) come back for new ones, and a library keyed by handle then linked
	// another shader's code (v436x rc1: InstructionPointerFault with recipes on). A module the
	// program cache did not register makes the key unusable (`unique`).
	bool unique = true;
	void AddStage(const vk::PipelineShaderStageCreateInfo& stage, const PipelineRecipe::Hash128* given = nullptr) {
		Add(stage.stage);
		PipelineRecipe::Hash128 hash;
		if (given != nullptr) {
			// The prefetch: its module is a temporary the program cache never registered, named
			// by the hash of the SPIR-V it was made from (what RegisterModule would record).
			Add(given->low);
			Add(given->high);
		} else if (PipelineRecipe::FindModuleHash(stage.module, hash)) {
			Add(hash.low);
			Add(hash.high);
		} else {
			unique = false;
			Add(reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(stage.module)));
		}
		const auto* subgroup =
		    static_cast<const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo*>(stage.pNext);
		Add(subgroup == nullptr ? 0u : subgroup->requiredSubgroupSize);
	}
	void AddMultisample(const vk::PipelineMultisampleStateCreateInfo* ms) {
		Add(ms->rasterizationSamples);
		Add(ms->sampleShadingEnable);
		Add(ms->minSampleShading);
		Add(ms->alphaToCoverageEnable);
		Add(ms->alphaToOneEnable);
		Add(ms->pSampleMask == nullptr ? ~0u : ms->pSampleMask[0]);
	}
	uint64_t Hash(uint32_t kind) const { return XXH3_64bits_withSeed(bytes.data(), bytes.size(), kind); }
	uint64_t Hash(uint32_t kind, bool robust) const { return Hash(kind | (robust ? 0x80000000u : 0u)); }
};

std::mutex                                g_library_mutex;
std::unordered_map<uint64_t, vk::Pipeline> g_libraries;
std::atomic<uint64_t>                     g_library_hits {0};
std::atomic<uint64_t>                     g_library_misses {0};
// Libraries made per kind (VI, pre-raster, fragment, output) and the distinct stage modules the
// pre-raster and fragment ones were made for: made/modules is the state multiplier step D pays.
std::atomic<uint64_t>                     g_library_made_kind[4] {};
std::atomic<uint64_t>                     g_library_us_kind[4] {};
std::mutex                                g_library_module_mutex;
std::unordered_set<uint64_t>              g_library_modules[2];
// Library prefetch (GplWarmLibrary with keep): stage libraries made on the prefetch workers and put
// in g_libraries before the draw that asks for them. Under g_library_mutex: g_prefetched holds
// the ones it put there, g_prefetch_taken those a linked pipeline has taken, g_library_made_at
// when a draw's pipeline made one. The counters ([gpl], [gpl-prefetch]): made (kept), used
// (taken, each once), late (the prefetch found the library made by a draw's pipeline within the
// second before: the GPU thread got to the draw first).
std::unordered_set<uint64_t>              g_prefetched;
std::unordered_set<uint64_t>              g_prefetch_taken;
std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> g_library_made_at;
std::atomic<uint64_t>                     g_prefetch_made {0};
std::atomic<uint64_t>                     g_prefetch_used {0};
std::atomic<uint64_t>                     g_prefetch_late {0};
std::atomic<uint64_t>                     g_prefetch_us {0};

} // namespace

// KYTY_GPL: the pipeline as four libraries (vertex input, pre-rasterization, fragment shader,
// fragment output) fast-linked without link-time optimization.
// Step D6: the optimized links, on background threads (lowered priority). Each job names the
// libraries (alive until exit), the pipeline's layout (alive until exit) and the slot the draws
// read (PipelineCache::Pipeline::Bound).
namespace {
struct GplOptimizeJob {
	vk::Device                               device;
	vk::PipelineCache                        cache;
	vk::PipelineLayout                       layout;
	std::vector<vk::Pipeline>                libraries;
	std::shared_ptr<std::atomic<VkPipeline>> slot;
	bool                                     robust = false;
};
struct GplOptimizer {
	std::mutex                 mutex;
	std::condition_variable    wake;
	std::deque<GplOptimizeJob> queue;
	std::vector<std::thread>   threads;
	bool                       stop    = false;
	std::atomic<uint64_t>      made {0};
	std::atomic<uint64_t>      failed {0};
	std::atomic<uint64_t>      made_us {0};
	std::atomic<uint64_t>      queued {0};
	// An exit without PipelineCache::Save: no std::terminate from joinable threads.
	~GplOptimizer() {
		{
			std::lock_guard lock(mutex);
			stop = true;
			queue.clear();
		}
		wake.notify_all();
		for (auto& thread: threads) {
			if (thread.joinable()) {
				thread.detach();
			}
		}
	}
};
GplOptimizer& Optimizer() {
	static GplOptimizer optimizer;
	return optimizer;
}
void GplOptimizeWork() {
	Common::LowerBackgroundThreadPriority();
	auto& o = Optimizer();
	for (;;) {
		GplOptimizeJob job;
		{
			std::unique_lock lock(o.mutex);
			o.wake.wait(lock, [&] { return o.stop || !o.queue.empty(); });
			if (o.stop) {
				return;
			}
			job = std::move(o.queue.front());
			o.queue.pop_front();
		}
		vk::PipelineRobustnessCreateInfoEXT robustness {};
		robustness.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2;
		robustness.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2;
		robustness.vertexInputs   = vk::PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2;
		robustness.images         = vk::PipelineRobustnessImageBehaviorEXT::eDeviceDefault;
		vk::PipelineLibraryCreateInfoKHR link {};
		link.libraryCount = static_cast<uint32_t>(job.libraries.size());
		link.pLibraries   = job.libraries.data();
		if (job.robust) {
			link.pNext = &robustness;
		}
		vk::GraphicsPipelineCreateInfo info {};
		info.pNext             = &link;
		info.layout            = job.layout;
		info.flags             = vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT;
		info.basePipelineIndex = -1;
		const auto   begin = std::chrono::steady_clock::now();
		vk::Pipeline made;
		if (job.device.createGraphicsPipelines(job.cache, 1, &info, nullptr, &made) == vk::Result::eSuccess) {
			job.slot->store(static_cast<VkPipeline>(made), std::memory_order_release);
			o.made.fetch_add(1, std::memory_order_relaxed);
			o.made_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                              std::chrono::steady_clock::now() - begin)
			                                              .count()),
			                    std::memory_order_relaxed);
		} else {
			o.failed.fetch_add(1, std::memory_order_relaxed);
		}
	}
}
void GplOptimizePush(GplOptimizeJob job) {
	auto&           o = Optimizer();
	std::lock_guard lock(o.mutex);
	if (o.stop) {
		return;
	}
	if (o.threads.empty()) {
		uint32_t count = 2;
		if (const char* value = std::getenv("KYTY_GPL_OPTIMIZE_THREADS"); value != nullptr) {
			count = std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 16u);
		}
		for (uint32_t i = 0; i < count; i++) {
			o.threads.emplace_back(GplOptimizeWork);
		}
	}
	o.queue.push_back(std::move(job));
	o.queued.fetch_add(1, std::memory_order_relaxed);
	o.wake.notify_one();
}
} // namespace

// Libraries keep link-time information (needed for optimized links); part of the library and
// of the warm record, so it is a knob of its own (KYTY_GPL_RETAIN=0) apart from the optimizer.
bool GplRetainEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPL_RETAIN");
		return value == nullptr || value[0] != '0';
	}();
	return enabled || GplKnob("KYTY_GPL_LTO") != 0;
}

bool GplOptimizeEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPL_OPTIMIZE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && GplRetainEnabled() && GplKnob("KYTY_GPL_LTO") == 0;
}

void GplOptimizerStop() {
	auto&                    o = Optimizer();
	std::vector<std::thread> threads;
	{
		std::lock_guard lock(o.mutex);
		o.stop = true;
		o.queue.clear();
		threads.swap(o.threads);
	}
	o.wake.notify_all();
	for (auto& thread: threads) {
		thread.join();
	}
	if (o.queued.load() != 0) {
		std::printf("[gpl-optimize] %llu optimized links made (avg %.1f ms), %llu failed, of %llu queued\n",
		            static_cast<unsigned long long>(o.made.load()),
		            o.made_us.load() / 1000.0 / std::max<uint64_t>(1, o.made.load()),
		            static_cast<unsigned long long>(o.failed.load()),
		            static_cast<unsigned long long>(o.queued.load()));
		std::fflush(stdout);
	}
}

namespace {
bool GplFullSets() {
	static const bool enabled = GplKnob("KYTY_GPL_FULLSETS") != 0;
	return enabled;
}
bool GplRobustKnob() {
	static const bool enabled = GplKnob("KYTY_GPL_ROBUST") != 0;
	return enabled;
}
bool GplShareEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPL_SHARE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}
bool HasDynamicState(const vk::PipelineDynamicStateCreateInfo* dynamic, vk::DynamicState wanted) {
	return dynamic != nullptr &&
	       std::ranges::any_of(std::span(dynamic->pDynamicStates, dynamic->dynamicStateCount),
	                           [wanted](vk::DynamicState state) { return state == wanted; });
}

// The keys of a build's pre-rasterization and fragment-shader libraries (step C): the stages and
// the state of that kind that is not dynamic. One function each for the draw's pipeline
// (CreateLinkedGraphicsPipeline), for GplLibrariesReady and for the prefetch (GplWarmLibrary
// with keep, which passes the hash of its SPIR-V as `given`), so all three name the same library.
LibraryKey PreLibraryKey(const GraphicsPipelineBuild&                          build,
                         std::span<const vk::PipelineShaderStageCreateInfo> pre_stages,
                         const PipelineRecipe::Hash128*                       given = nullptr) {
	const auto& base      = build.pipeline_info;
	const auto* rendering = static_cast<const vk::PipelineRenderingCreateInfo*>(base.pNext);
	const auto* rs        = base.pRasterizationState;
	const bool  eds3      = HasDynamicState(base.pDynamicState, vk::DynamicState::ePolygonModeEXT);
	LibraryKey  key;
	key.Add(static_cast<uint32_t>(pre_stages.size()));
	for (const auto& stage: pre_stages) {
		key.AddStage(stage, given);
	}
	key.Add(rs->depthClampEnable);
	key.Add(rs->rasterizerDiscardEnable);
	key.Add(eds3 ? vk::PolygonMode::eFill : rs->polygonMode);
	key.Add(rs->cullMode);
	key.Add(rs->frontFace);
	key.Add(eds3 ? VkBool32 {1} : build.clip_ext.depthClipEnable);
	key.Add(build.provoking_vertex.provokingVertexMode);
	key.Add(eds3 ? VkBool32 {0} : build.depth_clip_control.negativeOneToOne);
	key.Add(base.pTessellationState == nullptr ? 0u : base.pTessellationState->patchControlPoints);
	key.Add(rendering->viewMask);
	key.Add(build.gpl_pre_set_key);
	key.Add(build.set_count);
	key.Add(GplFullSets() ? build.gpl_ps_set_key : uint64_t {0});
	key.Add(static_cast<VkShaderStageFlags>(build.push_stages));
	key.Add(build.push_size);
	key.AddDynamic(base.pDynamicState, vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders);
	return key;
}

LibraryKey FragmentLibraryKey(const GraphicsPipelineBuild&                          build,
                              std::span<const vk::PipelineShaderStageCreateInfo> fragment_stages,
                              const PipelineRecipe::Hash128*                       given = nullptr) {
	const auto& base      = build.pipeline_info;
	const auto* rendering = static_cast<const vk::PipelineRenderingCreateInfo*>(base.pNext);
	const auto* ds        = base.pDepthStencilState;
	LibraryKey  key;
	key.Add(static_cast<uint32_t>(fragment_stages.size()));
	for (const auto& stage: fragment_stages) {
		key.AddStage(stage, given);
	}
	// With the bounds dynamic too, every depth/stencil field is dynamic: no key (a pass
	// without depth gets the all-off state, the draw sets the rest).
	if (!HasDynamicState(base.pDynamicState, vk::DynamicState::eDepthBounds)) {
		key.Add(ds != nullptr);
		if (ds != nullptr) {
			key.Add(ds->depthBoundsTestEnable);
			key.Add(ds->minDepthBounds);
			key.Add(ds->maxDepthBounds);
		}
	}
	key.AddMultisample(base.pMultisampleState);
	key.Add(rendering->viewMask);
	key.Add(build.gpl_ps_set_key);
	key.Add(build.set_count);
	key.Add(GplFullSets() ? build.gpl_pre_set_key : uint64_t {0});
	key.Add(static_cast<VkShaderStageFlags>(build.push_stages));
	key.Add(build.push_size);
	key.AddDynamic(base.pDynamicState, vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader);
	return key;
}

uint32_t LibraryKind(vk::GraphicsPipelineLibraryFlagBitsEXT kind) {
	return static_cast<uint32_t>(vk::GraphicsPipelineLibraryFlagsEXT(kind));
}
} // namespace

// Whether both stage libraries of a linked build are already in the shared map (made for an
// earlier pipeline, or by the prefetch): the pipeline then costs only its fast link, which the
// caller can do at once instead of handing it to a worker and waiting.
bool GplLibrariesReady(const GraphicsPipelineBuild& build) {
	if (!build.gpl || !GplShareEnabled() || build.tess_control != nullptr || build.tess_eval != nullptr) {
		return false;
	}
	const auto& base = build.pipeline_info;
	std::vector<vk::PipelineShaderStageCreateInfo> pre_stages;
	std::vector<vk::PipelineShaderStageCreateInfo> fragment_stages;
	for (const auto& stage: std::span(base.pStages, base.stageCount)) {
		(stage.stage == vk::ShaderStageFlagBits::eFragment ? fragment_stages : pre_stages).push_back(stage);
	}
	const bool robust = GplRobustKnob() && build.robustness;
	const auto pre    = PreLibraryKey(build, pre_stages);
	const auto frag   = FragmentLibraryKey(build, fragment_stages);
	if (!pre.unique || !frag.unique) {
		return false;
	}
	std::lock_guard lock(g_library_mutex);
	return g_libraries.contains(
	           pre.Hash(LibraryKind(vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders), robust)) &&
	       g_libraries.contains(
	           frag.Hash(LibraryKind(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader), robust));
}

static vk::Result CreateLinkedGraphicsPipeline(const GraphicsPipelineBuild& build,
                                               vk::PipelineCache driver_cache, vk::Pipeline* pipeline) {
	const auto& base   = build.pipeline_info;
	const auto  device = build.device;
	static const bool full_sets      = GplKnob("KYTY_GPL_FULLSETS") != 0;
	static const bool bindless_split = GplKnob("KYTY_GPL_BINDLESS_SPLIT") != 0;
	// KYTY_GPL_ROBUST=1: the libraries and the link name robustBufferAccess2 for buffers
	// explicitly (VK_EXT_pipeline_robustness): null descriptors (address < page, nullDescriptor)
	// were the fixed-address reads' candidate (v436n).
	static const bool robust_knob = GplKnob("KYTY_GPL_ROBUST") != 0;
	vk::PipelineRobustnessCreateInfoEXT robustness {};
	robustness.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2;
	robustness.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2;
	robustness.vertexInputs   = vk::PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2;
	robustness.images         = vk::PipelineRobustnessImageBehaviorEXT::eDeviceDefault;
	const bool robust = robust_knob && build.robustness;
	static const bool lto       = GplKnob("KYTY_GPL_LTO") != 0;
	const bool        optimize  = GplOptimizeEnabled();
	const bool        retain    = GplRetainEnabled();
	static const bool share     = [] {
		const char* value = std::getenv("KYTY_GPL_SHARE");
		return value == nullptr || value[0] != '0';
	}();
	static const uint32_t no_cache = GplKnob("KYTY_GPL_NOCACHE");
	if (no_cache == 1) {
		driver_cache = nullptr;
	} else if (no_cache == 2) {
		static const vk::PipelineCache private_cache = [&] {
			vk::PipelineCacheCreateInfo info {};
			vk::PipelineCache           made;
			return device.createPipelineCache(&info, nullptr, &made) == vk::Result::eSuccess
			           ? made
			           : vk::PipelineCache {};
		}();
		driver_cache = private_cache;
	}
	const auto* rendering = static_cast<const vk::PipelineRenderingCreateInfo*>(base.pNext);
	const vk::PushConstantRange push {build.push_stages, 0, build.push_size};
	// Library layouts: pre-rasterization leaves out the pixel shader's set, the fragment library
	// set 0. Made per library and destroyed after it (a library keeps what it needs).
	const auto make_layout = [&](bool fragment, vk::PipelineLayout* layout) {
		std::array<vk::DescriptorSetLayout, 4> sets = build.set_layouts;
		if (!full_sets) {
			sets[fragment ? 0 : 2] = nullptr;
		}
		// KYTY_GPL_BINDLESS_SPLIT=1: the bindless set only in the libraries whose stages use it
		// (bisect: linked pipelines with bindless faulted, v436l/m).
		if (bindless_split && !(fragment ? build.ps_bindless : build.pre_bindless)) {
			sets[1] = nullptr;
		}
		vk::PipelineLayoutCreateInfo info {};
		info.flags                  = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
		info.setLayoutCount         = build.set_count;
		info.pSetLayouts            = sets.data();
		info.pushConstantRangeCount = 1;
		info.pPushConstantRanges    = &push;
		return device.createPipelineLayout(&info, nullptr, layout);
	};
	const auto stages = std::span(base.pStages, base.stageCount);
	std::vector<vk::PipelineShaderStageCreateInfo> pre_stages;
	std::vector<vk::PipelineShaderStageCreateInfo> fragment_stages;
	for (const auto& stage: stages) {
		(stage.stage == vk::ShaderStageFlagBits::eFragment ? fragment_stages : pre_stages).push_back(stage);
	}
	// Per-build tessellation modules (rect lists) are destroyed with the build: never shared.
	const bool shareable_build = share && build.tess_control == nullptr && build.tess_eval == nullptr;

	std::vector<vk::Pipeline> libraries;
	auto       result          = vk::Result::eSuccess;
	const auto libraries_begin = std::chrono::steady_clock::now();
	const auto make = [&](vk::GraphicsPipelineLibraryFlagsEXT kind, const LibraryKey& key,
	                      int layout_kind, auto&& fill) {
		if (result != vk::Result::eSuccess) return;
		const auto hash      = key.Hash(static_cast<uint32_t>(kind), robust);
		const bool shareable = shareable_build && key.unique;
		if (shareable) {
			std::lock_guard lock(g_library_mutex);
			if (auto it = g_libraries.find(hash); it != g_libraries.end()) {
				libraries.push_back(it->second);
				g_library_hits.fetch_add(1, std::memory_order_relaxed);
				if (!g_prefetched.empty() && g_prefetched.contains(hash) && g_prefetch_taken.insert(hash).second) {
					g_prefetch_used.fetch_add(1, std::memory_order_relaxed);
				}
				return;
			}
		}
		vk::PipelineLayout layout;
		if (layout_kind >= 0 && make_layout(layout_kind == 1, &layout) != vk::Result::eSuccess) {
			result = vk::Result::eErrorUnknown;
			return;
		}
		vk::GraphicsPipelineLibraryCreateInfoEXT library {};
		library.flags = kind;
		// Only the output library takes the formats; the others need the view mask alone (and
		// formats in their create info would make the driver cache tell them apart).
		vk::PipelineRenderingCreateInfo view_mask_only {};
		view_mask_only.viewMask = rendering->viewMask;
		library.pNext = kind == vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface
		                    ? const_cast<void*>(base.pNext)
		                    : static_cast<void*>(&view_mask_only);
		const auto states = LibraryDynamicStates(base.pDynamicState, kind);
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(states.size());
		dynamic.pDynamicStates    = states.data();
		vk::PipelineRobustnessCreateInfoEXT library_robustness = robustness;
		if (robust) {
			library_robustness.pNext = library.pNext;
			library.pNext            = &library_robustness;
		}
		vk::GraphicsPipelineCreateInfo info {};
		info.pNext             = &library;
		info.flags             = vk::PipelineCreateFlagBits::eLibraryKHR;
		if (lto || retain) {
			info.flags |= vk::PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT;
		}
		info.pDynamicState     = &dynamic;
		info.basePipelineIndex = -1;
		info.layout            = layout_kind >= 0 ? layout : vk::PipelineLayout {};
		fill(info);
		vk::Pipeline made;
		const auto   made_begin = std::chrono::steady_clock::now();
		// Step D7: the shard of the stage library's module (the warm-up picks the same one).
		const auto shard_stages = kind == vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders
		                              ? std::span<const vk::PipelineShaderStageCreateInfo>(pre_stages)
		                          : kind == vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader
		                              ? std::span<const vk::PipelineShaderStageCreateInfo>(fragment_stages)
		                              : std::span<const vk::PipelineShaderStageCreateInfo> {};
		const vk::PipelineCache library_cache =
		    AuxCacheFor(AuxCache::Library, shard_stages.empty() ? 0 : ModuleShardHash(shard_stages.front().module));
		vk::PipelineCreationFeedback           feedback {};
		vk::PipelineCreationFeedbackCreateInfo feedback_info {};
		feedback_info.pPipelineCreationFeedback = &feedback;
		feedback_info.pNext                     = info.pNext;
		info.pNext                              = &feedback_info;
		result = device.createGraphicsPipelines(library_cache ? library_cache : driver_cache, 1, &info,
		                                        nullptr, &made);
		NoteFeedback(feedback);
		const bool library_miss = !shard_stages.empty() && g_last_cache_hit == 0;
		const auto   made_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                     std::chrono::steady_clock::now() - made_begin)
                                                     .count());
		if ((library_miss || made_us > 5000) &&
		    GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed)) {
			ZeroCompileNote(library_miss ? ZeroCompileEvent::LibraryMiss : ZeroCompileEvent::SlowLibrary,
			                shard_stages.empty() ? static_cast<uint64_t>(static_cast<VkFlags>(kind))
			                                     : ModuleShardHash(shard_stages.front().module),
			                static_cast<double>(made_us) / 1000.0);
		}
		// The library's layout stays alive with it (the validation layer crashed in a later
		// vkCmdPushDescriptorSetKHR with them destroyed, v436f).
		if (result != vk::Result::eSuccess) return;
		g_library_misses.fetch_add(1, std::memory_order_relaxed);
		{
			const int k = kind & vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface ? 0
			            : kind & vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders ? 1
			            : kind & vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader ? 2
			                                                                             : 3;
			g_library_made_kind[k].fetch_add(1, std::memory_order_relaxed);
			g_library_us_kind[k].fetch_add(made_us, std::memory_order_relaxed);
			if (k == 1 || k == 2) {
				uint64_t modules = 0;
				for (const auto& stage: k == 1 ? pre_stages : fragment_stages) {
					modules = modules * 31u + reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(stage.module));
				}
				std::lock_guard lock(g_library_module_mutex);
				g_library_modules[k - 1].insert(modules);
			}
		}
		if (shareable) {
			std::lock_guard lock(g_library_mutex);
			auto [it, inserted] = g_libraries.emplace(hash, made);
			if (!inserted) { // another thread made the same one meanwhile
				device.destroyPipeline(made, nullptr);
				made = it->second;
			} else {
				g_library_made_at.emplace(hash, std::chrono::steady_clock::now());
			}
		}
		libraries.push_back(made);
	};

	if (!build.mesh) {
		LibraryKey key;
		const auto* vi = base.pVertexInputState;
		key.Add(vi->vertexBindingDescriptionCount);
		for (uint32_t i = 0; i < vi->vertexBindingDescriptionCount; i++) {
			key.Add(vi->pVertexBindingDescriptions[i]);
		}
		key.Add(vi->vertexAttributeDescriptionCount);
		for (uint32_t i = 0; i < vi->vertexAttributeDescriptionCount; i++) {
			key.Add(vi->pVertexAttributeDescriptions[i]);
		}
		key.Add(base.pInputAssemblyState->topology);
		key.Add(base.pInputAssemblyState->primitiveRestartEnable);
		key.AddDynamic(base.pDynamicState, vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface);
		make(vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface, key, -1, [&](auto& info) {
			info.pVertexInputState   = base.pVertexInputState;
			info.pInputAssemblyState = base.pInputAssemblyState;
		});
	}
	{
		const auto key  = PreLibraryKey(build, pre_stages);
		const bool eds3 = HasDynamicState(base.pDynamicState, vk::DynamicState::ePolygonModeEXT);
		// The dynamic fields get fixed values in the create info as in the key (the driver
		// ignores them; the driver cache then sees one library per module and state).
		const bool cull_dynamic = std::ranges::any_of(
		    std::span(base.pDynamicState->pDynamicStates, base.pDynamicState->dynamicStateCount),
		    [](vk::DynamicState state) { return state == vk::DynamicState::eCullMode; });
		auto clip_control = build.depth_clip_control;
		auto viewport     = *base.pViewportState;
		auto rasterizer   = *base.pRasterizationState;
		auto clip_enable  = build.clip_ext;
		auto provoking    = build.provoking_vertex;
		if (eds3) {
			clip_control.negativeOneToOne = VK_FALSE;
			clip_enable.depthClipEnable   = VK_TRUE;
			rasterizer.polygonMode        = vk::PolygonMode::eFill;
		}
		if (cull_dynamic) {
			rasterizer.cullMode  = vk::CullModeFlagBits::eNone;
			rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
		}
		viewport.pNext = viewport.pNext != nullptr ? &clip_control : nullptr;
		// The rasterizer's chain as built (shaders.cpp above): provoking vertex, then depth clip.
		if (rasterizer.pNext == &build.provoking_vertex) {
			provoking.pNext  = provoking.pNext != nullptr ? &clip_enable : nullptr;
			rasterizer.pNext = &provoking;
		} else if (rasterizer.pNext == &build.clip_ext) {
			rasterizer.pNext = &clip_enable;
		}
		make(vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders, key, 0, [&](auto& info) {
			info.stageCount          = static_cast<uint32_t>(pre_stages.size());
			info.pStages             = pre_stages.data();
			info.pViewportState      = &viewport;
			info.pRasterizationState = &rasterizer;
			info.pTessellationState  = base.pTessellationState;
		});
	}
	{
		const auto  key = FragmentLibraryKey(build, fragment_stages);
		const auto* ds  = base.pDepthStencilState;
		// With the bounds dynamic too, every depth/stencil field is dynamic: no key (a pass
		// without depth gets the all-off state below, the draw sets the rest).
		const bool ds_dynamic = HasDynamicState(base.pDynamicState, vk::DynamicState::eDepthBounds);
		// A fragment-shader library needs depth/stencil state even for a pass without a depth
		// attachment (the formats are not its state): all tests off then (VUID-06590; a null one
		// was the v436 device faults' candidate).
		static const vk::PipelineDepthStencilStateCreateInfo no_depth {};
		make(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader, key, 1, [&](auto& info) {
			info.stageCount         = static_cast<uint32_t>(fragment_stages.size());
			info.pStages            = fragment_stages.data();
			info.pDepthStencilState = ds != nullptr && !ds_dynamic ? ds : &no_depth;
			info.pMultisampleState  = base.pMultisampleState;
		});
	}
	{
		LibraryKey key;
		const auto* cb = base.pColorBlendState;
		key.Add(cb->logicOpEnable);
		key.Add(cb->logicOp);
		key.Add(cb->attachmentCount);
		for (uint32_t i = 0; i < cb->attachmentCount; i++) {
			key.Add(cb->pAttachments[i]);
		}
		key.Add(cb->pNext != nullptr);
		key.AddMultisample(base.pMultisampleState);
		key.Add(rendering->viewMask);
		key.Add(rendering->colorAttachmentCount);
		for (uint32_t i = 0; i < rendering->colorAttachmentCount; i++) {
			key.Add(rendering->pColorAttachmentFormats[i]);
		}
		key.Add(rendering->depthAttachmentFormat);
		key.Add(rendering->stencilAttachmentFormat);
		key.AddDynamic(base.pDynamicState, vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface);
		make(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface, key, -1, [&](auto& info) {
			info.pColorBlendState  = base.pColorBlendState;
			info.pMultisampleState = base.pMultisampleState;
		});
	}
	const auto link_begin = std::chrono::steady_clock::now();
	if (result == vk::Result::eSuccess) {
		vk::PipelineLibraryCreateInfoKHR link {};
		link.libraryCount = static_cast<uint32_t>(libraries.size());
		link.pLibraries   = libraries.data();
		if (robust) {
			link.pNext = &robustness;
		}
		vk::GraphicsPipelineCreateInfo info {};
		info.pNext             = &link;
		info.layout            = base.layout;
		info.flags             = PipelineStats::CaptureFlags();
		if (lto) {
			info.flags |= vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT;
		}
		info.basePipelineIndex = -1;
		// A fast link is cheaper than a cache lookup: no cache (an optimized link uses it).
		result = device.createGraphicsPipelines(lto ? driver_cache : vk::PipelineCache {}, 1, &info, nullptr,
		                                        pipeline);
		// Step D6: the optimized link of the same libraries, in the background (into the driver
		// cache: a later start makes it from the cache).
		if (result == vk::Result::eSuccess && optimize && build.optimized_slot != nullptr) {
			const vk::PipelineCache optimized_cache =
			    AuxCacheFor(AuxCache::Library, pre_stages.empty() ? 0 : ModuleShardHash(pre_stages.front().module));
			GplOptimizePush({device, optimized_cache ? optimized_cache : driver_cache, base.layout, libraries,
			                 build.optimized_slot, robust});
		}
	}
	if (result == vk::Result::eSuccess) {
		// KYTY_GPL timing: libraries and link, summed, printed every 256 linked pipelines.
		static std::atomic<uint64_t> count {0};
		static std::atomic<uint64_t> library_us {0};
		static std::atomic<uint64_t> link_us {0};
		static std::atomic<uint64_t> link_max_us {0};
		const auto end = std::chrono::steady_clock::now();
		const auto lib = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(link_begin - libraries_begin).count());
		const auto lnk = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(end - link_begin).count());
		library_us.fetch_add(lib, std::memory_order_relaxed);
		link_us.fetch_add(lnk, std::memory_order_relaxed);
		auto max = link_max_us.load(std::memory_order_relaxed);
		while (lnk > max && !link_max_us.compare_exchange_weak(max, lnk)) {
		}
		if (const auto n = count.fetch_add(1) + 1; (n & 127u) == 0) {
			size_t pre_modules = 0;
			size_t frag_modules = 0;
			{
				std::lock_guard lock(g_library_module_mutex);
				pre_modules  = g_library_modules[0].size();
				frag_modules = g_library_modules[1].size();
			}
			std::printf("[gpl] %llu linked: libraries avg %.2f ms (%llu made, %llu shared), link avg "
			            "%.2f ms, link max %.2f ms | made VI %llu, pre %llu for %zu modules, frag %llu "
			            "for %zu modules, out %llu | ms per library VI %.2f pre %.2f frag %.2f out %.2f | optimized %llu of %llu (avg %.1f ms)"
			            " | prefetched %llu (avg %.2f ms), used %llu, late %llu\n",
			            static_cast<unsigned long long>(n), library_us.load() / 1000.0 / n,
			            static_cast<unsigned long long>(g_library_misses.load()),
			            static_cast<unsigned long long>(g_library_hits.load()),
			            link_us.load() / 1000.0 / n, link_max_us.load() / 1000.0,
			            static_cast<unsigned long long>(g_library_made_kind[0].load()),
			            static_cast<unsigned long long>(g_library_made_kind[1].load()), pre_modules,
			            static_cast<unsigned long long>(g_library_made_kind[2].load()), frag_modules,
			            static_cast<unsigned long long>(g_library_made_kind[3].load()),
			            g_library_us_kind[0].load() / 1000.0 / std::max<uint64_t>(1, g_library_made_kind[0].load()),
			            g_library_us_kind[1].load() / 1000.0 / std::max<uint64_t>(1, g_library_made_kind[1].load()),
			            g_library_us_kind[2].load() / 1000.0 / std::max<uint64_t>(1, g_library_made_kind[2].load()),
			            g_library_us_kind[3].load() / 1000.0 / std::max<uint64_t>(1, g_library_made_kind[3].load()),
			            static_cast<unsigned long long>(Optimizer().made.load()),
			            static_cast<unsigned long long>(Optimizer().queued.load()),
			            Optimizer().made_us.load() / 1000.0 / std::max<uint64_t>(1, Optimizer().made.load()),
			            static_cast<unsigned long long>(g_prefetch_made.load()),
			            g_prefetch_us.load() / 1000.0 / std::max<uint64_t>(1, g_prefetch_made.load()),
			            static_cast<unsigned long long>(g_prefetch_used.load()),
			            static_cast<unsigned long long>(g_prefetch_late.load()));
			std::fflush(stdout);
		}
	}
	// Libraries not in the shared map (not shareable, or a failed link) stay alive too: see above.
	return result;
}

// Step D4: one stage's pre-rasterization or fragment library made the way a linked pipeline
// makes it in the common state (no MSAA, no tessellation, every dynamic field as above), so the
// driver cache holds it before any draw asks: a draw's pipeline then costs a cache hit and a
// link. Returns false when the stage is not warmed (other stages, knobs that change the library).
//
// `keep` (the library prefetch): the library is not destroyed but put in the shared map under the
// key a linked pipeline of this module in the common state builds, so the draw that asks for it
// finds it made and only links. The temporary module is destroyed either way (a library keeps
// what it needs); a kept library's layouts stay alive with it, as a draw's own library's do.
bool GplWarmLibrary(GraphicContext& graphics, vk::PipelineCache driver_cache, ShaderType stage,
                    std::span<const uint32_t> spirv, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                    bool keep) {
	if (!graphics.graphics_pipeline_library_enabled || graphics.bindless_layout == nullptr ||
	    GplKnob("KYTY_GPL_LTO") != 0 || GplKnob("KYTY_GPL_ROBUST") != 0 || GplKnob("KYTY_GPL_FULLSETS") != 0 ||
	    GplKnob("KYTY_GPL_BINDLESS_SPLIT") != 0 || GplBindlessSet3()) {
		return false;
	}
	const bool fragment = stage == ShaderType::Pixel;
	const auto native   = stage == ShaderType::Vertex ? vk::ShaderStageFlagBits::eVertex
	                      : stage == ShaderType::Mesh ? vk::ShaderStageFlagBits::eMeshEXT
	                      : fragment                  ? vk::ShaderStageFlagBits::eFragment
	                                                  : vk::ShaderStageFlagBits {};
	if (native == vk::ShaderStageFlagBits {} || (stage == ShaderType::Mesh && !graphics.mesh_shader_enabled)) {
		return false;
	}
	const auto device = graphics.device;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = spirv.size() * sizeof(uint32_t);
	module_info.pCode    = spirv.data();
	vk::ShaderModule module;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) {
		return false;
	}
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	AddLayoutBindings(bindings, program, native);
	const auto stage_set = CreateSetLayout(graphics, bindings, false);
	std::array<vk::DescriptorSetLayout, 3> sets {};
	sets[1]              = graphics.bindless_layout;
	sets[fragment ? 2 : 0] = stage_set;
	vk::ShaderStageFlags push_stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
	                                   vk::ShaderStageFlagBits::eTessellationControl |
	                                   vk::ShaderStageFlagBits::eTessellationEvaluation;
	if (graphics.mesh_shader_enabled) {
		push_stages |= vk::ShaderStageFlagBits::eMeshEXT;
	}
	const vk::PushConstantRange push {push_stages, 0, ShaderRecompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.flags                  = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
	layout_info.setLayoutCount         = 3;
	layout_info.pSetLayouts            = sets.data();
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	vk::PipelineLayout layout;
	auto result = device.createPipelineLayout(&layout_info, nullptr, &layout);

	const auto kind = fragment ? vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader
	                           : vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders;
	const auto all_states = GraphicsDynamicStates(graphics, 1, true);
	vk::PipelineDynamicStateCreateInfo all_dynamic {};
	all_dynamic.dynamicStateCount = static_cast<uint32_t>(all_states.size());
	all_dynamic.pDynamicStates    = all_states.data();
	const auto states = LibraryDynamicStates(&all_dynamic, kind);
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(states.size());
	dynamic.pDynamicStates    = states.data();

	vk::PipelineRenderingCreateInfo view_mask_only {};
	vk::GraphicsPipelineLibraryCreateInfoEXT library {};
	library.flags = kind;
	library.pNext = &view_mask_only;
	vk::PipelineShaderStageCreateInfo stage_info {};
	stage_info.stage  = native;
	stage_info.module = module;
	stage_info.pName  = "main";

	// As CreatePipelineInternal builds them, with the dynamic fields at the library values.
	vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control {};
	vk::PipelineViewportStateCreateInfo               viewport {};
	viewport.pNext = &clip_control;
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT       clip_enable {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking {};
	vk::PipelineRasterizationStateCreateInfo                   rasterizer {};
#if !defined(__APPLE__)
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext            = &clip_enable;
#endif
	if (graphics.provoking_vertex_last_enabled) {
		provoking.provokingVertexMode = vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking.pNext               = rasterizer.pNext;
		rasterizer.pNext              = &provoking;
	}
	rasterizer.cullMode    = vk::CullModeFlagBits::eNone;
	rasterizer.frontFace   = vk::FrontFace::eCounterClockwise;
	rasterizer.polygonMode = vk::PolygonMode::eFill;
	rasterizer.lineWidth   = 1.0f;
	const bool eds3        = std::ranges::find(states, vk::DynamicState::ePolygonModeEXT) != states.end();
	clip_enable.depthClipEnable = VK_TRUE; // the common state clips (and with EDS3 it is dynamic)
	(void)eds3;
	static const vk::PipelineDepthStencilStateCreateInfo no_depth {};
	vk::PipelineMultisampleStateCreateInfo               multisample {};
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	multisample.minSampleShading     = 1.0f;

	vk::GraphicsPipelineCreateInfo info {};
	info.pNext             = &library;
	info.flags             = vk::PipelineCreateFlagBits::eLibraryKHR;
	if (GplRetainEnabled()) {
		info.flags |= vk::PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT;
	}
	info.pDynamicState     = &dynamic;
	info.basePipelineIndex = -1;
	info.layout            = layout;
	info.stageCount        = 1;
	info.pStages           = &stage_info;
	if (fragment) {
		info.pDepthStencilState = &no_depth;
		info.pMultisampleState  = &multisample;
	} else {
		info.pViewportState      = &viewport;
		info.pRasterizationState = &rasterizer;
	}
	vk::Pipeline made;
	const auto              module_hash = PipelineRecipe::HashWords(spirv);
	const vk::PipelineCache shard       = AuxCacheFor(AuxCache::Library, module_hash.low);
	// Diagnosis (v437t): KYTY_WARM_NOPIPE=1 makes everything but the library itself (the process
	// ended after ~20,000 libraries in one start).
	static const bool no_pipe = GplKnob("KYTY_WARM_NOPIPE") != 0;
	// Prefetch: the key CreateLinkedGraphicsPipeline builds for this stage's library, from the same
	// functions and the fields above. Not kept where a draw's key holds more than the common
	// state names: a required subgroup size (the stage create info here has none), or depth
	// bounds that are not dynamic (the key then holds the pass's depth state).
	uint64_t keep_hash = 0;
	keep = keep && result == vk::Result::eSuccess && !no_pipe && GplShareEnabled() &&
	       (native == vk::ShaderStageFlagBits::eVertex ||
	        graphics.GraphicsSubgroupSize(native, program.wave_size) == 0) &&
	       (!fragment || HasDynamicState(&all_dynamic, vk::DynamicState::eDepthBounds));
	bool already = false;
	bool late    = false; // a draw's pipeline made this library just before the prefetch got to it
	if (keep) {
		GraphicsPipelineBuild common; // only what the key functions read
		common.pipeline_info.pNext               = &view_mask_only;
		common.pipeline_info.pRasterizationState = &rasterizer;
		common.pipeline_info.pMultisampleState   = &multisample;
		common.pipeline_info.pDynamicState       = &all_dynamic;
		common.clip_ext                          = clip_enable;
		common.provoking_vertex                  = provoking;
		common.depth_clip_control                = clip_control;
		common.set_count                         = 3;
		common.push_stages                       = push_stages;
		common.push_size                         = ShaderRecompiler::IR::NativePushConstantSize;
		// As PrepareGraphicsPipeline: no push set in linked pipelines, set 1 the bindless table.
		const auto set_key = StageSetKey(bindings, false, true, UsesBindless(program));
		(fragment ? common.gpl_ps_set_key : common.gpl_pre_set_key) = set_key;
		const std::span<const vk::PipelineShaderStageCreateInfo> one(&stage_info, 1);
		const auto key = fragment ? FragmentLibraryKey(common, one, &module_hash)
		                          : PreLibraryKey(common, one, &module_hash);
		keep_hash = key.Hash(LibraryKind(kind), false);
		std::lock_guard lock(g_library_mutex);
		already = g_libraries.contains(keep_hash); // a draw (or another prefetch) made it first
		if (const auto drawn = g_library_made_at.find(keep_hash); already && drawn != g_library_made_at.end()) {
			late = std::chrono::steady_clock::now() - drawn->second < std::chrono::seconds(1);
		}
	}
	bool kept = false;
	if (already) {
		made = nullptr;
	} else if (result == vk::Result::eSuccess && no_pipe) {
		made = nullptr;
	} else if (result == vk::Result::eSuccess) {
		const auto begin = std::chrono::steady_clock::now();
		result = device.createGraphicsPipelines(shard ? shard : driver_cache, 1, &info, nullptr, &made);
		if (result == vk::Result::eSuccess && keep) {
			std::lock_guard lock(g_library_mutex);
			if (g_libraries.emplace(keep_hash, made).second) {
				g_prefetched.insert(keep_hash);
				kept = true;
			} else {
				late = g_library_made_at.contains(keep_hash); // made by a draw meanwhile
			}
		}
		if (kept) {
			g_prefetch_made.fetch_add(1, std::memory_order_relaxed);
			g_prefetch_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                                  std::chrono::steady_clock::now() - begin)
			                                                  .count()),
			                        std::memory_order_relaxed);
		} else if (result == vk::Result::eSuccess) {
			device.destroyPipeline(made, nullptr);
		}
	}
	if (late) {
		g_prefetch_late.fetch_add(1, std::memory_order_relaxed);
	}
	if (!kept) {
		if (layout) {
			device.destroyPipelineLayout(layout, nullptr);
		}
		device.destroyDescriptorSetLayout(stage_set, nullptr);
	}
	// KYTY_GPL_PREFETCH_MODULES=1: a kept library's module stays alive as well (a bisect step: it
	// is the one object a kept library outlives; a draw's own library is made from the program
	// cache's module, which lives for the session).
	static const bool keep_modules = GplKnob("KYTY_GPL_PREFETCH_MODULES") != 0;
	if (!(kept && keep_modules)) {
		device.destroyShaderModule(module, nullptr);
	}
	return result == vk::Result::eSuccess;
}

void GplPrefetchCounters(uint64_t& made, uint64_t& used, uint64_t& late) {
	made = g_prefetch_made.load(std::memory_order_relaxed);
	used = g_prefetch_used.load(std::memory_order_relaxed);
	late = g_prefetch_late.load(std::memory_order_relaxed);
}

uint64_t GplWarmStateHash(GraphicContext& graphics) {
	std::vector<uint32_t> words;
	const auto all_states = GraphicsDynamicStates(graphics, 1, true);
	vk::PipelineDynamicStateCreateInfo all_dynamic {};
	all_dynamic.dynamicStateCount = static_cast<uint32_t>(all_states.size());
	all_dynamic.pDynamicStates    = all_states.data();
	for (const auto kind: {vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders,
	                       vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader}) {
		words.push_back(0xFFFFFFFFu);
		for (const auto state: LibraryDynamicStates(&all_dynamic, kind)) {
			words.push_back(static_cast<uint32_t>(state));
		}
	}
	words.push_back(graphics.mesh_shader_enabled ? 1u : 0u);
	words.push_back(graphics.provoking_vertex_last_enabled ? 1u : 0u);
	words.push_back(graphics.bindless_layout != nullptr ? 1u : 0u);
	words.push_back(static_cast<uint32_t>(GplKnob("KYTY_GPL_LTO") | GplKnob("KYTY_GPL_ROBUST") |
	                                      GplKnob("KYTY_GPL_FULLSETS") | GplKnob("KYTY_GPL_BINDLESS_SPLIT")));
	words.push_back(GplBindlessSet3() ? 1u : 0u);
	words.push_back(GplRetainEnabled() ? 1u : 0u); // the libraries retain link-time info
	words.push_back(g_aux_count[0].load()); // the library cache's shard count
	return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

vk::Result CreateGraphicsPipeline(const GraphicsPipelineBuild& build, vk::PipelineCache driver_cache,
                                  vk::Pipeline* pipeline) {
	const auto create_begin = std::chrono::steady_clock::now();
	const CompileWatch watch('G', build.vs_hash, build.ps_hash);
	// KYTY_PIPE_STATS: the capture flag is added here only, never to the recipe.
	auto create_info = build.pipeline_info;
	create_info.flags |= PipelineStats::CaptureFlags();
	vk::PipelineCreationFeedback           feedback {};
	vk::PipelineCreationFeedbackCreateInfo feedback_info {};
	feedback_info.pPipelineCreationFeedback = &feedback;
	feedback_info.pNext                     = create_info.pNext;
	create_info.pNext                       = &feedback_info;
	g_last_cache_hit                        = -1;
	// KYTY_GPL_MONO=1: the independent-sets layout but an ordinary (monolithic) pipeline: the
	// layout and set binding without libraries (bisect, v436t).
	static const bool gpl_mono = GplKnob("KYTY_GPL_MONO") != 0;
	auto result = build.gpl && !gpl_mono
	                  ? CreateLinkedGraphicsPipeline(build, driver_cache, pipeline)
	                  : build.device.createGraphicsPipelines(driver_cache, 1, &create_info, nullptr, pipeline);
	if (build.gpl && result != vk::Result::eSuccess) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			std::printf("PipelineTrace: linked pipeline VS=0x%016" PRIx64 " PS=0x%016" PRIx64
			     " failed (%s)\n",
			     build.vs_hash, build.ps_hash, vk::to_string(result).c_str());
		}
		// No monolithic fallback: the layout has independent sets, which only libraries take.
	}
	if (build.gpl && result == vk::Result::eSuccess && GplKnob("KYTY_GPL_LOG_PAIRS") != 0) {
		std::printf("[gpl-pair] VS=%016" PRIx64 " PS=%016" PRIx64 "\n", build.vs_hash, build.ps_hash);
		std::fflush(stdout);
	}
	if (build.gpl && result == vk::Result::eSuccess) {
		static std::atomic<bool> first {true};
		if (first.exchange(false)) {
			std::printf("Graphics pipeline libraries: first linked pipeline built\n");
		}
	}
	if (result == vk::Result::eSuccess) {
		PipelineStats::Report(build.device, *pipeline, 'G', build.vs_hash, build.ps_hash);
	}
	const bool mono_miss = !(build.gpl && !gpl_mono) && result == vk::Result::eSuccess &&
	                       (feedback.flags & vk::PipelineCreationFeedbackFlagBits::eValid) &&
	                       !(feedback.flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit);
	if (const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	    (mono_miss || total_ms > 5.0) && GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed)) {
		ZeroCompileNote(mono_miss ? ZeroCompileEvent::PipelineMiss : ZeroCompileEvent::SlowPipeline,
		                build.ps_hash != 0 ? build.ps_hash : build.vs_hash, total_ms);
	}
	// On the GPU thread, a slow pipeline stops every queue.
	if (const auto create_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - create_begin)
	                               .count();
	    create_ms >= 500) {
		LOGF("PipelineTrace: slow graphics pipeline VS=0x%016" PRIx64 " PS=0x%016" PRIx64
		     " ms=%lld\n",
		     build.vs_hash, build.ps_hash, static_cast<long long>(create_ms));
	}
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(*pipeline));
	}
	return result;
}

void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache) {
	const auto build = PrepareGraphicsPipeline(graphics, pipeline, rendering, vertex_input,
	                                           vertex_info, ps_input_info, programs, static_params);
	EXIT_IF(pipeline.pipeline != nullptr);
	const auto result = CreateGraphicsPipeline(*build, driver_cache, &pipeline.pipeline);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_bindless = UsesBindless(*input_info.stage.program);
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	std::array<vk::DescriptorSetLayout, 2> set_layouts {};
	uint32_t                               set_layout_count = 0;
	FillSetLayouts(graphics, pipeline, set_layouts, set_layout_count);
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_layout_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;
	if (DisableOptimizationFor(input_info.stage.program->shader_hash) ||
	    SlowPipelineListed('C', input_info.stage.program->shader_hash, 0)) {
		info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
		LOGF("PipelineTrace: compute shader 0x%016" PRIx64 " built without optimization\n",
		     input_info.stage.program->shader_hash);
	}

	EXIT_IF(pipeline.pipeline != nullptr);

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	const auto create_begin = std::chrono::steady_clock::now();
	const CompileWatch watch('C', input_info.stage.program->shader_hash, 0);
	info.flags |= PipelineStats::CaptureFlags();
	vk::PipelineCreationFeedback           feedback {};
	vk::PipelineCreationFeedbackCreateInfo feedback_info {};
	feedback_info.pPipelineCreationFeedback = &feedback;
	feedback_info.pNext                     = info.pNext;
	info.pNext                              = &feedback_info;
	size_t size_before = 0;
	static const bool cs_diag = [] {
		const char* value = std::getenv("KYTY_WARM_LOG");
		return value != nullptr && value[0] == '1';
	}();
	const bool play_now = cs_diag && GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed);
	if (play_now && driver_cache) {
		(void)graphics.device.getPipelineCacheData(driver_cache, &size_before, nullptr);
	}
	static std::atomic<int> compute_in_flight {0};
	const int               others = compute_in_flight.fetch_add(1);
	const auto create_only_begin = std::chrono::steady_clock::now();
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	compute_in_flight.fetch_sub(1);
	const double create_only_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_only_begin).count();
	NoteFeedback(feedback);
	// Diagnosis (v437n): a slow compute creation in play, with what the cache gained by it.
	if (play_now && (create_only_ms > 5.0 || g_last_cache_hit == 0)) {
		size_t size_after = 0;
		if (driver_cache) {
			(void)graphics.device.getPipelineCacheData(driver_cache, &size_after, nullptr);
		}
		// The same creation again at once: does the cache serve it now?
		int    retry_hit = -2;
		double retry_ms  = 0.0;
		if (result == vk::Result::eSuccess) {
			vk::PipelineCreationFeedback retry_fb {};
			feedback_info.pPipelineCreationFeedback = &retry_fb;
			vk::Pipeline retry;
			const auto   t0 = std::chrono::steady_clock::now();
			if (graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr, &retry) == vk::Result::eSuccess) {
				retry_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
				retry_hit = !(retry_fb.flags & vk::PipelineCreationFeedbackFlagBits::eValid) ? -1
				            : (retry_fb.flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit) ? 1 : 0;
				graphics.device.destroyPipeline(retry, nullptr);
			}
			feedback_info.pPipelineCreationFeedback = &feedback;
		}
		std::fprintf(stderr, "[cs-create] shader %016llx module %016llx %.1f ms hit %d cache %p +%lld bytes others %d retry hit %d %.2f ms\n",
		             static_cast<unsigned long long>(input_info.stage.program->shader_hash),
		             static_cast<unsigned long long>(ModuleShardHash(compute_module)), create_only_ms, g_last_cache_hit,
		             static_cast<void*>(static_cast<VkPipelineCache>(driver_cache)),
		             static_cast<long long>(size_after) - static_cast<long long>(size_before), others, retry_hit, retry_ms);
	}
	// Diagnosis (v437l): what keys a compute pipeline in the driver cache, for the warm-up's
	// creations (KYTY_WARM_LOG=1) and every runtime cache miss after the startup wait.
	{
		static const bool warm_log = [] {
			const char* value = std::getenv("KYTY_WARM_LOG");
			return value != nullptr && value[0] == '1';
		}();
		const bool play = GetShaderPrecompileProgress().startup_wait_done.load(std::memory_order_relaxed);
		if (warm_log) {
			std::vector<uint32_t> words;
			for (const auto& b: descriptor_bindings) {
				words.push_back(b.binding);
				words.push_back(static_cast<uint32_t>(b.descriptorType));
				words.push_back(b.descriptorCount);
				words.push_back(static_cast<uint32_t>(static_cast<VkShaderStageFlags>(b.stageFlags)));
				words.push_back(b.pImmutableSamplers != nullptr ? 1u : 0u);
			}
			std::string list;
			for (const auto& b: descriptor_bindings) {
				char item[48];
				std::snprintf(item, sizeof(item), "%u:%u:%u ", b.binding, static_cast<uint32_t>(b.descriptorType), b.descriptorCount);
				list += item;
			}
			std::fprintf(stderr, "[cs-key] %s shader %016llx module %016llx bhash %016llx [%s] wave %u subgroup %u bindless %d bindings %zu push-set %d flags %x cache %p\n",
			             play ? (g_last_cache_hit == 0 ? "miss" : "run") : "warm",
			             static_cast<unsigned long long>(input_info.stage.program->shader_hash),
			             static_cast<unsigned long long>(play ? ModuleShardHash(compute_module) : g_warm_module_hash),
			             static_cast<unsigned long long>(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t))), list.c_str(), wave_size,
			             comp_shader_stage_info.pNext != nullptr ? comp_subgroup_size.requiredSubgroupSize : 0u,
			             pipeline.uses_bindless ? 1 : 0, descriptor_bindings.size(), pipeline.uses_push_descriptors ? 1 : 0,
			             static_cast<unsigned>(static_cast<VkPipelineCreateFlags>(info.flags)),
			             static_cast<void*>(static_cast<VkPipelineCache>(driver_cache)));
		}
	}
	if (result == vk::Result::eSuccess) {
		PipelineStats::Report(graphics.device, pipeline.pipeline, 'C', input_info.stage.program->shader_hash,
		                      0);
	}
	if (const auto create_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - create_begin)
	                               .count();
	    create_ms >= 500) {
		LOGF("PipelineTrace: slow compute pipeline CS=0x%016" PRIx64 " ms=%lld\n",
		     input_info.stage.program->shader_hash, static_cast<long long>(create_ms));
	}
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

// Compute warm-up (with step D4): a compute permutation's pipeline made as GetComputePipeline makes
// it, into the driver cache the draws' dispatches use, then destroyed: a first dispatch then costs
// a cache hit.
bool ComputeWarmPipeline(GraphicContext& graphics, vk::PipelineCache driver_cache, std::span<const uint32_t> spirv,
                         const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	if (driver_cache == nullptr) {
		return false;
	}
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = spirv.size() * sizeof(uint32_t);
	module_info.pCode    = spirv.data();
	vk::ShaderModule module;
	if (graphics.device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) {
		return false;
	}
	// The pipeline needs the program alone; the stage check also asks for resources.
	static const ShaderRecompiler::IR::ResourceSnapshot no_resources {};
	ShaderComputeInputInfo input_info {};
	input_info.stage.program   = &program;
	input_info.stage.resources = &no_resources;
	PipelineCache::Pipeline pipeline;
	const uint64_t          spirv_hash = PipelineRecipe::HashWords(spirv).low;
	const vk::PipelineCache shard      = AuxCacheFor(AuxCache::Compute, spirv_hash);
	g_warm_module_hash                 = spirv_hash;
	CreatePipelineInternal(graphics, pipeline, input_info, module, shard ? shard : driver_cache);
	g_warm_module_hash = 0;
	// Diagnosis (v437n, KYTY_WARM_LOG=1): the same pipeline again at once: does the cache serve it?
	static const bool warm_log = [] {
		const char* value = std::getenv("KYTY_WARM_LOG");
		return value != nullptr && value[0] == '1';
	}();
	if (warm_log && pipeline.pipeline != nullptr) {
		// v437p: which object makes the difference: a new module object (same SPIR-V) with the
		// warm-up's layout, then the warm-up's module with a new layout of the same definition.
		const auto try_create = [&](vk::ShaderModule use_module, vk::PipelineLayout use_layout, const char* what) {
			vk::PipelineShaderStageCreateInfo stage {};
			stage.stage  = vk::ShaderStageFlagBits::eCompute;
			stage.module = use_module;
			stage.pName  = "main";
			vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {};
			if (graphics.compute_subgroup_size_control_enabled && program.wave_size >= graphics.min_subgroup_size &&
			    program.wave_size <= graphics.max_subgroup_size) {
				subgroup.requiredSubgroupSize = program.wave_size;
				stage.pNext                   = &subgroup;
			}
			vk::PipelineCreationFeedback           fb {};
			vk::PipelineCreationFeedbackCreateInfo fb_info {};
			fb_info.pPipelineCreationFeedback = &fb;
			vk::ComputePipelineCreateInfo again {};
			again.pNext             = &fb_info;
			again.stage             = stage;
			again.layout            = use_layout;
			again.basePipelineIndex = -1;
			const auto   t0 = std::chrono::steady_clock::now();
			vk::Pipeline second;
			if (graphics.device.createComputePipelines(shard ? shard : driver_cache, 1, &again, nullptr, &second) ==
			    vk::Result::eSuccess) {
				const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
				NoteFeedback(fb);
				std::fprintf(stderr, "[cs-again] %s shader %016llx %.2f ms hit %d\n", what,
				             static_cast<unsigned long long>(program.shader_hash), ms, g_last_cache_hit);
				graphics.device.destroyPipeline(second, nullptr);
			}
		};
		vk::ShaderModule other_module;
		if (graphics.device.createShaderModule(&module_info, nullptr, &other_module) == vk::Result::eSuccess) {
			try_create(other_module, pipeline.pipeline_layout, "new-module");
			graphics.device.destroyShaderModule(other_module, nullptr);
		}
		PipelineCache::Pipeline other;
		CreatePipelineInternal(graphics, other, input_info, module, shard ? shard : driver_cache);
		graphics.device.destroyPipeline(other.pipeline, nullptr);
		try_create(module, other.pipeline_layout, "new-layout");
		graphics.device.destroyPipelineLayout(other.pipeline_layout, nullptr);
		graphics.device.destroyDescriptorSetLayout(other.descriptor_set_layout, nullptr);
	}
	const bool made = pipeline.pipeline != nullptr;
	graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	graphics.device.destroyShaderModule(module, nullptr);
	return made;
}

} // namespace Libs::Graphics
