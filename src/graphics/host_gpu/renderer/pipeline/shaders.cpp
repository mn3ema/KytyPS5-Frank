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
		if (auto* file = std::fopen(SlowPipelineFile, "r"); file != nullptr) {
			char               kind = 0;
			unsigned long long a = 0, b = 0;
			while (std::fscanf(file, " %c %llx %llx", &kind, &a, &b) == 3) {
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
				std::fprintf(file, "%c %016llx %016llx\n", entry->key.kind,
				             static_cast<unsigned long long>(entry->key.a),
				             static_cast<unsigned long long>(entry->key.b));
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

// KYTY_VRS_HASHES="hash,...": pipelines whose pixel shader is listed shade one invocation per
// KYTY_VRS_RATE block of pixels (2x2 by default, or 1x2, 2x1, 2x4, 4x2, 4x4). Meant for a full-screen
// pass whose translation costs far more than the original, because the predicated code runs every
// branch for every pixel (Wolverine's swamp water, 0x283e883115bc9cd7: 19 ms of a 47 ms frame).
static bool CoarseShadingFor(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		if (const char* value = std::getenv("KYTY_VRS_HASHES"); value != nullptr) {
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

static vk::Extent2D CoarseShadingRate() {
	static const vk::Extent2D rate = [] {
		unsigned width  = 2;
		unsigned height = 2;
		if (const char* value = std::getenv("KYTY_VRS_RATE"); value != nullptr) {
			if (std::sscanf(value, "%ux%u", &width, &height) != 2) {
				width  = 2;
				height = 2;
			}
		}
		return vk::Extent2D {width, height};
	}();
	return rate;
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
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_bindless |= UsesBindless(*ps_input_info->stage.program);
	}
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {graphics_stages, 0,
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

	build.dynamic_states = {
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
	if (rendering.color_count != 0) {
		build.dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		build.dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}
	if (PipelineDynamicRasterStateEnabled()) {
		// Core Vulkan 1.3 (cull mode, front face, depth-bounds test enable) and 1.0 (depth bounds;
		// the depthBounds feature is required on these hosts). The key holds zeroes for them; the
		// draw records the values from the same registers (SetGraphicsDynamicParams).
		build.dynamic_states.push_back(vk::DynamicState::eCullMode);
		build.dynamic_states.push_back(vk::DynamicState::eFrontFace);
#if !defined(__APPLE__)
		build.dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
		build.dynamic_states.push_back(vk::DynamicState::eDepthBounds);
#endif
	}

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
	if (graphics.pipeline_shading_rate_enabled && ps_hash != 0 && CoarseShadingFor(ps_hash)) {
		build.shading_rate.fragmentSize = CoarseShadingRate();
		build.shading_rate.pNext        = const_cast<void*>(pipeline_info.pNext);
		pipeline_info.pNext             = &build.shading_rate;
		LOGF("PipelineTrace: graphics pipeline VS=0x%016" PRIx64 " PS=0x%016" PRIx64
		     " shades at %ux%u\n",
		     build.vs_hash, ps_hash, build.shading_rate.fragmentSize.width,
		     build.shading_rate.fragmentSize.height);
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
	return owner;
}

vk::Result CreateGraphicsPipeline(const GraphicsPipelineBuild& build, vk::PipelineCache driver_cache,
                                  vk::Pipeline* pipeline) {
	const auto create_begin = std::chrono::steady_clock::now();
	const CompileWatch watch('G', build.vs_hash, build.ps_hash);
	const auto result =
	    build.device.createGraphicsPipelines(driver_cache, 1, &build.pipeline_info, nullptr, pipeline);
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
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
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

} // namespace Libs::Graphics
