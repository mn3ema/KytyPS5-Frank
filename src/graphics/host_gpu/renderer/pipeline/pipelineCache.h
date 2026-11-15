#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderParams;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

namespace DrawPrep {
struct PacketResult;
} // namespace DrawPrep

// KYTY_SRT_VERIFY (ResourceSnapshot.h): a verifying shader reported a value the host used that
// differed from the GPU's. Its refreshes read GPU-written bytes exactly from now on.
void SrtVerifyReportMiss(uint64_t shader_hash);

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       depth_bounds_test_enable = false;
	float                      depth_min_bounds         = 0.0f;
	float                      depth_max_bounds         = 0.0f;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 125);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;
	uint32_t         spirv_words = 0;
	uint64_t         hash        = 0;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// Everything vkCreateGraphicsPipelines reads for one graphics pipeline, owned in one place so the
// driver compile can run on a worker thread (the structures point into each other: it stays
// where it was allocated).
struct GraphicsPipelineBuild {
	GraphicsPipelineBuild() = default;
	~GraphicsPipelineBuild();
	KYTY_CLASS_NO_COPY(GraphicsPipelineBuild);

	vk::Device                                                            device;
	vk::ShaderModule                                                      tess_control = nullptr;
	vk::ShaderModule                                                      tess_eval    = nullptr;
	std::array<vk::PipelineShaderStageCreateInfo, 4>                      stages {};
	std::array<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo, 4>  subgroup_sizes {};
	std::array<vk::VertexInputAttributeDescription, ShaderVertexInputInfo::RES_MAX> input_attr {};
	std::array<vk::VertexInputBindingDescription, ShaderVertexInputInfo::RES_MAX>   input_desc {};
	vk::PipelineVertexInputStateCreateInfo                                vertex_input_info {};
	vk::PipelineInputAssemblyStateCreateInfo                              input_assembly {};
	vk::PipelineViewportDepthClipControlCreateInfoEXT                     depth_clip_control {};
	vk::PipelineViewportStateCreateInfo                                   viewport_state {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT                  clip_ext {};
	vk::PipelineRasterizationStateCreateInfo                              rasterizer {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT            provoking_vertex {};
	vk::PipelineMultisampleStateCreateInfo                                multisampling {};
	std::array<vk::PipelineColorBlendAttachmentState, RENDER_COLOR_ATTACHMENTS_MAX> color_blend {};
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX>                  color_write_enable {};
	vk::PipelineColorWriteCreateInfoEXT                                   color_write {};
	vk::PipelineColorBlendStateCreateInfo                                 color_blending {};
	vk::PipelineDepthStencilStateCreateInfo                               depth_stencil {};
	std::vector<vk::DynamicState>                                         dynamic_states;
	vk::PipelineDynamicStateCreateInfo                                    dynamic_state {};
	vk::PipelineTessellationStateCreateInfo                               tessellation {};
	PipelineRenderingState                                                rendering;
	vk::PipelineRenderingCreateInfo                                       rendering_info {};
	vk::GraphicsPipelineCreateInfo                                        pipeline_info {};
	uint64_t                                                              vs_hash = 0;
	uint64_t                                                              ps_hash = 0;
	// For the pipeline recipe (pipelineRecipe.h): what the layouts were made of, and the
	// rect-list tessellation stages' SPIR-V (generated per pipeline, not in the program cache).
	std::vector<vk::DescriptorSetLayoutBinding>                           descriptor_bindings;
	bool                                                                  uses_bindless = false;
	vk::ShaderStageFlags                                                  push_stages;
	uint32_t                                                              push_size = 0;
	std::vector<uint32_t>                                                 rect_tcs_spirv;
	std::vector<uint32_t>                                                 rect_tes_spirv;
	// KYTY_GPL: linked from graphics pipeline libraries (set layouts 0, 1, 2 of the independent-set
	// layout; the libraries' layouts leave out the other stage's set).
	bool                                                                  gpl = false;
	bool                                                                  mesh = false;
	std::array<vk::DescriptorSetLayout, 4>                                set_layouts {};
	uint32_t                                                              set_count = 3;
	// Step C: the content of set 0 / set 2 (bindings and push), for the shared libraries' keys.
	uint64_t                                                              gpl_pre_set_key = 0;
	uint64_t                                                              gpl_ps_set_key  = 0;
	bool                                                                  pre_bindless    = false;
	bool                                                                  ps_bindless     = false;
	bool                                                                  robustness      = false; // VK_EXT_pipeline_robustness
	// Step D6: where the background optimized (link-time optimized) link of this linked pipeline
	// goes; null when not optimized.
	std::shared_ptr<std::atomic<VkPipeline>>                              optimized_slot;
};

// A graphics pipeline a worker thread compiles (see PipelineCache::TryGetGraphicsPipeline).
struct PendingGraphicsPipeline;
class PipelineCompiler;

// The owning renderer serializes access, including saves while the GPU is running.
// Whether every color target the draw writes is at most KYTY_SMALL_TARGET_SYNC pixels wide and
// high (default 1024; 0 = never): such draws are never skipped while their shaders translate or
// their pipeline compiles (pipelineCache.cpp).
bool SmallColorTargetDraw(const HW::Context& context);

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// KYTY_STAGE_SETS: the pixel shader's own set (2) and whether it is the push set (then set 0
		// is not); null without a pixel shader or with one set.
		vk::DescriptorSetLayout ps_set_layout         = nullptr;
		bool                    ps_uses_push          = false;
		// KYTY_GPL: linked from libraries with an independent-sets layout; every set gets bound
		// (an empty one where the stage has nothing).
		bool                    independent_sets      = false;
		// Step D2: depth clip, clip-space depth range and polygon mode are dynamic (EDS3).
		bool                    eds3_raster           = false;
		// Step D5: the stages every push names (empty: the stages of the pipeline's programs).
		vk::ShaderStageFlags    push_stages;
		// KYTY_GPL_BINDLESS3: the bindless table is set 3 of this layout (set 1 an empty set).
		uint32_t                bindless_set          = 1;
		// Samples bindless images: descriptor set 1 is the bindless table.
		bool                    uses_bindless         = false;
		// Set while a worker compiles `pipeline`; draws skip it until the compile finishes.
		std::shared_ptr<PendingGraphicsPipeline> pending;
		// Step D6: a linked pipeline's optimized link, made in the background; bound instead of
		// `pipeline` once there (same layout and state).
		std::shared_ptr<std::atomic<VkPipeline>> optimized;

		[[nodiscard]] vk::Pipeline Bound() const {
			if (optimized != nullptr) {
				if (const VkPipeline handle = optimized->load(std::memory_order_acquire); handle != VK_NULL_HANDLE) {
					return handle;
				}
			}
			return pipeline;
		}
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
	// Draw prep (KYTY_DRAW_PREP, drawPrep.h), scanner thread: the SRT refreshes GetGraphicsPrograms
	// would run for these registers, into `out`. False when nothing was prepared.
	bool DrawPrepGraphics(const HW::VertexShaderInfo& vertex_regs,
	                      const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                      const HW::Context& context, const HW::UserConfig& user_config,
	                      std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                      bool pixel_active, DrawPrep::PacketResult& out);
	// The same for a dispatch: the SRT refresh GetComputeProgram would run.
	// `queue`: the guest queue (the async-compute thread's queues scan with its program cache).
	bool DrawPrepCompute(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
	                     ShaderComputeInputInfo& input_info, DrawPrep::PacketResult& out,
	                     uint32_t queue = 0);
	// KYTY_ASYNC_THREAD: the async-compute thread has its own program cache, so its program
	// lookups and SRT refreshes run without the renderer mutex.
	[[nodiscard]] bool HasAsyncProgramCache() const noexcept { return m_async_program_cache != nullptr; }
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);
	// Whether the shader at this address is on the skip list (--skip-shaders): its dispatches can
	// be dropped before anything is prepared for them.
	[[nodiscard]] static bool IsSkipListed(uint64_t shader_addr);
	// KYTY_MULTI_DRAW_REPLAY (RenderExecutor::ReplayMultiDraw): the user-data words the resource
	// plan behind a stage's snapshot reads (bit i: the snapshot's user_data[i]) and the plan's
	// user-data base register. False when the plan is not known (the replay then never applies).
	[[nodiscard]] static bool PlanUserDataReads(const ShaderRecompiler::IR::ResourceSnapshot* resources,
	                                            uint64_t& mask, uint32_t& user_data_base);
	// KYTY_SRT_DEFER=1 (GPU thread, around one dispatch): SRT reads of GPU-written bytes start
	// their download and fail instead of waiting; EndSrtDefer returns the tick to wait for before
	// the dispatch runs again (0: it ran). SrtDeferPending: the current refresh was deferred.
	[[nodiscard]] static bool     SrtDeferEnabled();
	static void                   BeginSrtDefer(bool allowed);
	[[nodiscard]] static uint64_t EndSrtDefer();
	[[nodiscard]] static bool     SrtDeferPending();

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	// As GetGraphicsPipeline; with may_defer, a pipeline the driver does not compile within a
	// short wait is compiled on a worker thread, and nullptr says to skip this draw.
	Pipeline* TryGetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                                 const RenderDepthInfo&                 depth,
	                                 std::span<const ShaderVertexInputInfo> vertex_info,
	                                 CommandBuffer&                         command,
	                                 const ShaderPixelInputInfo*            ps_input_info,
	                                 vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                 const GraphicsPrograms& programs, bool may_defer);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	struct ProgramCache;

	// GetGraphicsPrograms' stage inputs from the registers (shared with DrawPrepGraphics).
	bool DescribeGraphicsStages(const HW::VertexShaderInfo& vertex_regs,
	                            const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                            const HW::Context& context, const HW::UserConfig& user_config,
	                            std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                            bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                            ShaderPixelInputInfo& pixel_info,
	                            std::array<ShaderParams, 3>& vertex_params, ShaderParams& pixel_params);

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	std::unique_ptr<ProgramCache> m_async_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	size_t                        m_driver_cache_saved = 0; // payload size last read or written
	// Step D7: the libraries' (with their optimized links) and the compute pipelines' driver
	// caches, sharded by module (<title>_lib_NN.bin, <title>_cs_NN.bin); index = AuxCache.
	struct AuxShards {
		std::vector<vk::PipelineCache>     caches;
		std::vector<std::filesystem::path> paths;
		std::vector<size_t>                saved;
		std::vector<size_t>                loaded; // at startup
	};
	size_t                        m_driver_cache_loaded = 0;
	std::array<AuxShards, 2>      m_aux;
	std::mutex                    m_write_mutex;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	std::unique_ptr<PipelineCompiler>                       m_compiler;
	uint64_t                                                m_deferred_draws = 0;
	struct DriverCacheSaver;
	std::unique_ptr<DriverCacheSaver> m_driver_cache_saver;
	struct RecipeState;
	std::unique_ptr<RecipeState> m_recipes;

	// Builds the recorded graphics pipeline recipes at startup (pipelineRecipe.h).
	void StartPipelinePrecompile();

	void InitializeDriverCache();
	// Writes the driver cache to m_driver_cache_path (temp file, then rename); safe while
	// pipelines compile (vkGetPipelineCacheData needs no external synchronization).
	bool WriteDriverCache();
	bool WriteDriverCacheFile(vk::PipelineCache cache, const std::filesystem::path& cache_path,
	                          size_t* written = nullptr);
	// Writes `cache` only when its data size differs from `saved` (updated on success).
	bool WriteDriverCacheIfChanged(vk::PipelineCache cache, const std::filesystem::path& cache_path, size_t& saved);
	[[nodiscard]] std::vector<uint8_t> ReadDriverCacheFile(const std::filesystem::path& cache_path) const;
	// Takes a finished background compile into the pipeline; false while it still runs.
	bool FinishPending(Pipeline& pipeline);
};

// KYTY_DYNAMIC_RASTER_STATE (default on): cull mode and front face (and, except on macOS, the
// depth-bounds test enable and bounds) are dynamic state of every renderer graphics pipeline,
// recorded per draw by SetGraphicsDynamicParams, and leave the pipeline key; key fields that
// cannot change the pipeline are zeroed. =0 bakes them into each pipeline as before.
[[nodiscard]] bool PipelineDynamicRasterStateEnabled();
struct GraphicsDynamicRaster {
	bool            negative_one_to_one = false;
	bool            depth_clip_enable   = true;
	vk::PolygonMode polygon_mode        = vk::PolygonMode::eFill;
};
[[nodiscard]] GraphicsDynamicRaster DrawDynamicRaster(const HW::Context& ctx, bool rect_list);

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// KYTY_STAGE_SETS (shaders.cpp): an empty set layout (set 1 without bindless), and the split of a
// graphics pipeline's bindings into the pre-rasterization set (0) and the pixel shader's (2), with
// which one is the push set.
vk::DescriptorSetLayout EmptySetLayout(GraphicContext& graphics);
// A descriptor set of EmptySetLayout (allocated once), for the unbound sets of linked pipelines.
vk::DescriptorSet       EmptyDescriptorSet(GraphicContext& graphics);
// KYTY_GPL_BINDLESS3 (bisect, v436u): graphics modules whose bindless table is set 1 also get a
// twin with the table at set 3 (SPIR-V patched at module creation); null when none.
bool             GplBindlessSet3();
// Step D7: sharded driver caches for the libraries (and their optimized links) and for compute
// pipelines; a shard is picked by the module's SPIR-V hash (ModuleShardHash, or the low half of
// PipelineRecipe::HashWords for SPIR-V in hand). Null when there are none.
enum class AuxCache : uint32_t { Library = 0, Compute = 1 };
void              SetAuxCaches(AuxCache which, std::span<const vk::PipelineCache> caches);
vk::PipelineCache AuxCacheFor(AuxCache which, uint64_t module_hash);
uint64_t          ModuleShardHash(vk::ShaderModule module);
// Zero-compile accounting (after the startup wait): events the shipped list should have covered.
// Slow*: over 5 ms with the driver reporting a cache hit (or no feedback); *Miss: the driver
// reported no pipeline cache hit (VK_EXT_pipeline_creation_feedback, core in 1.3): a compile.
enum class ZeroCompileEvent : uint32_t { Translation, LedgerRecord, SlowLibrary, SlowCompute, SlowPipeline, SkippedDraw,
	                                     LibraryMiss, ComputeMiss, PipelineMiss, Count };
// The last pipeline creation on this thread: 1 cache hit, 0 miss, -1 no feedback.
int LastPipelineCacheHit();
void ZeroCompileNote(ZeroCompileEvent event, uint64_t hash, double ms);
void ZeroCompileReport(const char* when);
// Step D4: one stage's library made as a linked pipeline would make it, into `driver_cache`.
// `keep` (the library prefetch): it stays alive in the linked pipelines' shared map instead.
bool GplWarmLibrary(GraphicContext& graphics, vk::PipelineCache driver_cache, ShaderType stage,
                    std::span<const uint32_t> spirv, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                    bool keep = false);
// Library prefetch: the stage libraries of a shader pair the draw-prep scanner saw before the GPU
// thread reached its draw are made on worker threads from the shader list's compiled records
// (cache hits after the warm-up) and kept, so the draw's pipeline only links. KYTY_GPL_PREFETCH=0
// turns it off. Request: any thread, by the guest shader's hash; a hash asked before or not in the
// list costs a lookup.
bool GplPrefetchEnabled();
void GplPrefetchRequest(uint64_t shader_hash);
void GplPrefetchStop();
void GplPrefetchCounters(uint64_t& made, uint64_t& used, uint64_t& late);
// Whether both stage libraries of a linked build are already made (shaders.cpp).
bool GplLibrariesReady(const GraphicsPipelineBuild& build);
// What shapes those libraries besides the shader (dynamic states per kind, push stages,
// rasterizer chain, knobs that stop the warm-up): part of the warm record's salt.
uint64_t GplWarmStateHash(GraphicContext& graphics);
// A compute permutation's pipeline made as GetComputePipeline makes it, into `driver_cache`.
bool ComputeWarmPipeline(GraphicContext& graphics, vk::PipelineCache driver_cache, std::span<const uint32_t> spirv,
                         const ShaderRecompiler::IR::CompiledShaderInfo& program);
// Step D6 (KYTY_GPL_OPTIMIZE, default on): linked pipelines get an optimized link on background
// threads, bound once made. Stopped before the driver cache is written at exit.
bool GplOptimizeEnabled();
bool GplRetainEnabled();
// Logs where a breakpoint / fast-fail exception was raised (warm-up crash diagnosis); handles nothing.
void InstallWarmCrashLogger();
void GplOptimizerStop();
void             RegisterBindlessSet3Twin(vk::Device device, vk::ShaderModule module,
                                          std::span<const uint32_t> spirv);
vk::ShaderModule BindlessSet3Twin(vk::ShaderModule module);
void SplitStageSets(uint32_t max_push, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                    std::vector<vk::DescriptorSetLayoutBinding>& pre,
                    std::vector<vk::DescriptorSetLayoutBinding>& ps, bool& pre_push, bool& ps_push);
// Creates the pipeline's layouts and fills the create info; CreateGraphicsPipeline compiles it.
std::unique_ptr<GraphicsPipelineBuild>
PrepareGraphicsPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                        const PipelineRenderingState&          rendering,
                        const PipelineVertexInputState&        vertex_input,
                        std::span<const ShaderVertexInputInfo> vertex_info,
                        const ShaderPixelInputInfo*            ps_input_info,
                        const PipelineCache::GraphicsPrograms& programs,
                        const PipelineStaticParameters&        static_params);
// False for an embedded-fetch attribute whose components the shader only replaces by constants:
// it gets no native vertex attribute, and its buffer is not bound for it.
[[nodiscard]] bool VertexAttributeFetched(const ShaderVertexInputInfo& info, int attribute);
// Any thread.
vk::Result CreateGraphicsPipeline(const GraphicsPipelineBuild& build, vk::PipelineCache driver_cache,
                                  vk::Pipeline* pipeline);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
