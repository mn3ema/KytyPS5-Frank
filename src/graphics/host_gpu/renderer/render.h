#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/indirectDispatch.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/renderer/fsr3.h"
#include "graphics/host_gpu/renderer/upscaler/upscaler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <source_location>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
struct ComputeShaderInfo;
struct ShaderRegisters;
} // namespace HW

struct ShaderProgram;

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
namespace DrawPipe {
struct PreparedPrograms;
} // namespace DrawPipe
class CommandScheduler;
class CommandRecorder;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	DispatchIndirect,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// DRAW_INDIRECT read on the GPU: the guest address of its DrawIndirectArgs (vertex count,
	// instance count, start vertex, start instance). The counts above are then unknown on the
	// CPU; only fast-launch mesh draws come this way.
	uint64_t         gpu_args_address           = 0;
	// Nonzero (vertex count known): the instance count is the dword the GPU reads there (the
	// NUM_INSTANCES a DRAW_INDIRECT read on the GPU left).
	uint64_t         gpu_instance_count_address = 0;
	// With gpu_args_address: the user SGPR (index + 1, 0 = none) START_INST_LOC names, which
	// the shader loads from the arguments' start instance (the CP writes it on hardware).
	uint32_t         start_instance_sgpr        = 0;
	const uint32_t*  indirect_packet             = nullptr;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering(std::source_location site = std::source_location::current()) const;
	// Whether a dynamic-rendering instance is open, and with which attachments.
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }
	[[nodiscard]] const RenderState& CurrentRenderState() const noexcept { return m_render_state; }

	// The raw command buffer. With the recording thread (CommandScheduler::EnableRecordingThread)
	// this first waits until the thread has run everything queued; hot paths use Recorder().
	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Records directly, or through the recording thread.
	[[nodiscard]] CommandRecorder Recorder() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	CommandScheduler&   m_scheduler;
	// Begun and not submitted yet; m_buffer is its handle when not recorded on a thread.
	bool                m_open            = false;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

// v445b (renderCompute.cpp): KYTY_DISPATCH_DELTA=0 turns the dispatch delta path off;
// KYTY_DISPATCH_DELTA_VERIFY=1 also runs the full path and compares.
[[nodiscard]] bool DispatchDeltaEnabled();
[[nodiscard]] bool DispatchDeltaVerify();
namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	// KYTY_DRAW_PIPE (drawPipe.h), the command processor: the programs a direct draw's
	// RefreshShaders would select for these registers, with copies of their snapshots.
	static bool PrepareDrawProgramsForPipe(RenderContext& context, const HW::Context& ctx,
	                                       const HW::Shader& sh_ctx, const HW::UserConfig& ucfg,
	                                       DrawPipe::PreparedPrograms& out);
	// KYTY_DRAW_PIPE, the draw thread: a direct draw as the command processor would have run it.
	void RunPipeDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs* index_args,
	                 const DrawAutoArgs* auto_args);
	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode);
	// G4 stage 1 (docs/g4_split_plan_20261102.md), the async-compute thread only: a direct
	// dispatch prepared without the renderer mutex (registers copied, program looked up in the
	// thread's own program cache, resource snapshot copied) and queued on this thread; the queue
	// is recorded in order by CommitQueuedDispatches, under the mutex, before the thread's next
	// packet that needs the renderer.
	void QueueDispatchDirect(uint64_t submit_id, const HW::ComputeShaderInfo& cs_regs,
	                         const HW::ShaderRegisters& sh_regs, uint32_t thread_group_x,
	                         uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	// The same for DISPATCH_INDIRECT (KYTY_ASYNC_BATCH_INDIRECT=1).
	void QueueDispatchIndirect(uint64_t submit_id, const HW::ComputeShaderInfo& cs_regs,
	                           const HW::ShaderRegisters& sh_regs, uint64_t args_addr, uint32_t mode);
	void CommitQueuedDispatches(CommandBuffer& buffer);
	[[nodiscard]] static size_t QueuedDispatchCount() noexcept;
	// The CS address of the last queued dispatch (0: none queued).
	[[nodiscard]] static uint64_t LastQueuedShader() noexcept;
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	// Bindless heaps: register the draw's heaps, patch their regions, and once per frame resolve
	// the keys shaders flagged as pending.
	void PrepareBindlessHeaps(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void ResolveBindlessRequests();
	// Engine method (engine-method.md, E2), at the start of a guest submission: the heap entries
	// the guest wrote since the last one are compared and their keys settled again.
	void SyncBindlessHeaps();
	void SyncWrittenBindlessEntries();
	// E3: settle entries no draw has asked for yet, within the per-submission budget.
	void SettleBindlessHeaps();
	bool ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// The key's T# changed: release what it was settled to and settle it again.
	bool ResettleBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// Release only: the key is pending again and its image unpinned when nothing else uses it.
	void ReleaseBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// Settles again, within the budget, the keys of a window of each heap whose T# changed since
	// they were settled; returns how many it resolved.
	uint32_t RevalidateBindlessKeys(uint32_t budget);
	void                           FindBuffers(PreparedBindings& bindings);
	void                           RebindBuffers(PreparedBindings& bindings);
	// KYTY_REBIND_AFTER_JOIN=1 (after chenxiao07/KytyPS5 ed6f063e1 via Jetsku d93c804e): the
	// bindings resolved again while BufferCache::UnregisterEpoch moved since `epoch` (a later
	// lookup of the same draw or dispatch joined a buffer an earlier one resolved), at most 4
	// times; returns whether it resolved them again (off: only reported). Logs the first 16 occurrences and powers of two.
	bool                           RebindAfterJoin(std::span<PreparedBindings* const> stages,
	                                               uint64_t& epoch, const char* kind);
	uint64_t                       m_rebind_epoch = 0; // UnregisterEpoch when the draw's bindings resolved
	void                           RebindImages(PreparedBindings& bindings);
	void                           CaptureBuffers(const PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

	// KYTY_MULTI_DRAW_REPLAY (RenderStats::MultiDrawReplayActive): around the draws of one
	// DRAW_INDIRECT_MULTI packet. A draw recorded in full inside it becomes the replay base when
	// it qualifies; ReplayMultiDraw records a later draw of the packet again from the base, with
	// only its arguments and the user data the CP wrote changed, when nothing its resources are
	// resolved from changed and the base's command buffer, rendering and descriptors still hold.
	// False: draw it in full (nothing was recorded).
	void BeginMultiDraw();
	void EndMultiDraw();
	// A base stands inside a multi-draw packet (the sub-draw just drawn in full captured one).
	[[nodiscard]] bool MultiDrawBaseValid() const noexcept { return m_replay.active && m_replay.valid; }
	// KYTY_RENDER_STATS census ([multi-packets], RenderStats::MultiPacketNote); nothing is drawn
	// or changed by these. First: before a replayed packet's first sub-draw. End: before
	// EndMultiDraw. Indexed: around an indexed packet's loop.
	void CensusMultiDrawFirst(CommandBuffer& buffer);
	void CensusMultiDrawEnd(CommandBuffer& buffer);
	void CensusMultiDrawIndexed(bool inside) noexcept { m_replay.census_indexed = inside; }
	[[nodiscard]] bool ReplayMultiDraw(uint64_t submit_id, CommandBuffer& buffer,
	                                   uint32_t vertex_count, uint32_t instance_count,
	                                   uint32_t first_vertex, uint32_t first_instance);
	[[nodiscard]] bool ReplayIndirectDraw(uint64_t submit_id, CommandBuffer& buffer,
	                                      uint64_t gpu_args_address, uint32_t start_instance_sgpr,
	                                      const uint32_t* packet);
	void InvalidateIndirectReplay(uint32_t reason);

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// ResolveTexture, plus the second (null) binding of a dynamic-class image (v429).
	[[nodiscard]] TextureBinding ResolveStageTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                                 const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TextureBinding FindResolvedTexture(
	    const ShaderRecompiler::IR::ImageResource& resource, const ShaderTextureResource& descriptor,
	    TextureCache::ImageDesc desc, bool shader_conversion, vk::Format pixel_format,
	    vk::Format view_format, uint32_t size);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	// DispatchIndirect after its program lookup (renderer mutex).
	void DispatchIndirectRecord(uint64_t submit_id, CommandBuffer& buffer,
	                            const HW::ComputeShaderInfo& cs_regs, ShaderComputeInputInfo& input_info,
	                            const ShaderProgram& compute_program, uint64_t args_addr, uint32_t mode);
	// DispatchDirect after its program lookup: bindings, commit and recording (renderer mutex).
	void DispatchDirectRecord(uint64_t submit_id, CommandBuffer& buffer,
	                          const HW::ComputeShaderInfo& cs_regs, ShaderComputeInputInfo& input_info,
	                          const ShaderProgram& compute_program, uint32_t thread_group_x,
	                          uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool        Upscale(vk::CommandBuffer command, const Pssr::Request& request,
	                                  Dlss::Inputs& inputs);

	// A null feature is one the upscaler refused or failed.
	struct UpscalerContext {
		uint64_t                           context;
		Dlss::Creation                     creation;
		std::unique_ptr<Upscaler::Feature> feature;
	};
	[[nodiscard]] bool UpscaleFsr(vk::CommandBuffer command, const Pssr::Request& request,
	                              const Fsr3::Inputs& inputs);

	// FSR 3.1 history of a PSSR context; a null fsr is one FSR refused or failed.
	struct FsrContext {
		uint64_t        context;
		Fsr3::Creation  creation;
		Fsr3::Context*  fsr;
	};

	// KYTY_MULTI_DRAW_REPLAY: called at the end of a draw recorded in full inside a multi-draw.
	void CaptureReplayBase(CommandBuffer& buffer, const DrawCallInfo& draw, const DrawRenderState& state,
	                       const DrawEmitInfo& emit, vk::PipelineLayout layout, bool writes,
	                       vk::Buffer mesh_indirect_buffer = nullptr,
	                       vk::DeviceSize mesh_indirect_offset = 0,
	                       const void* gpu_args_buffer = nullptr);
	// The vertex stage's (commit stage 0) shader data replaced with `data`: uploaded, and the
	// last commit's descriptors (or push data) issued again. False when the upload moved the
	// command buffer on (nothing was issued).
	[[nodiscard]] bool RecommitShaderData(CommandBuffer& buffer, std::span<const uint32_t> data);

	// What the last CommitBindings issued, for RecommitShaderData.
	struct LastCommit {
		uint64_t                       generation = 0;
		vk::PipelineBindPoint          bind_point = vk::PipelineBindPoint::eGraphics;
		vk::PipelineLayout             layout;
		vk::DescriptorSetLayout        set_layout;
		bool                           uses_push_descriptors = false;
		// KYTY_STAGE_SETS: the pixel shader's set (2); writes [0, pre_writes) go to set 0, the rest
		// to set 2.
		vk::DescriptorSetLayout        ps_set_layout;
		bool                           ps_uses_push = false;
		size_t                         pre_writes   = 0;
		bool                           has_push_data         = false;
		vk::ShaderStageFlags           push_stages;
		std::array<uint32_t, ShaderRecompiler::IR::PushData::DwordCount> push_dwords {};
		uint32_t                       push_bytes = 0;
		// Commit stage 0's shader data: its slot in m_descriptor_buffers (-1: none) and, when it
		// travels as push data, its first dword there (-1: not push data).
		int32_t                        stage0_data_slot = -1;
		int32_t                        stage0_push_start = -1;
	};
	LastCommit m_last_commit;

	// v445b, KYTY_DISPATCH_DELTA (renderCompute.cpp): what the last guest dispatch recorded here
	// resolved and left on the compute bind point. The next dispatch of the same shader on the same
	// queue, in the same command buffer (tick), takes from it the read-only buffer bindings whose
	// source is unchanged and CPU-clean, the pipeline, and pushes only the descriptors that differ.
	struct DispatchDelta {
		bool        valid      = false;
		const void* owner      = nullptr; // the queue's compute registers (one per command processor)
		const void* program    = nullptr; // ShaderStageRuntime::program
		uint64_t    program_id = 0;
		void*       pipeline   = nullptr; // PipelineCache::GetComputePipeline's
		bool        thread_dimensions   = false;
		bool        dimensions_indirect = false;
		// As they were when the base's dispatch bound its pipeline (v445d: not at its end, which may
		// come after a flush): the tick, the command buffer, the compute bind point's serial and the
		// unregister epoch its bindings were resolved under; and the pipeline handle it bound.
		uint64_t     tick = 0, unregister_epoch = 0, compute_serial = 0, command_serial = 0;
		vk::Pipeline bound_pipeline;
		std::vector<PreparedBindings::BufferSource> sources;
		std::vector<vk::DescriptorBufferInfo>       buffers;
		std::vector<uint32_t>                       shader_data;
		// The writes last pushed to the compute bind point's set 0 under `layout`, flattened
		// (PushDescriptorsDelta); writes_valid: they are what the bind point holds while
		// compute_serial and tick stand.
		bool                  writes_valid = false;
		vk::PipelineLayout    layout;
		size_t                write_count = 0;
		std::vector<uint64_t> writes;
	};
	DispatchDelta        m_dispatch_delta;
	const DispatchDelta* m_delta_reuse = nullptr; // set around RebindBuffers: slots may come from it
	bool                 m_delta_push  = false;   // set around CommitBindings: push only what differs
	uint32_t             m_delta_reused = 0, m_delta_pushed = 0, m_delta_writes = 0; // the last rebind's and commit's
	std::vector<uint64_t>               m_delta_flat;    // PushDescriptorsDelta scratch
	// KYTY_RENDER_STATS census ([draw-delta], descriptors.cpp): what the previous draw's stage at
	// each position resolved. Nothing is bound from it.
	std::array<DispatchDelta, 4> m_draw_census;
	void NoteDrawDeltaBefore(uint32_t stage, const PreparedBindings& prepared);
	void NoteDrawDeltaAfter(uint32_t stage, const PreparedBindings& prepared);
	std::vector<vk::WriteDescriptorSet> m_delta_changed; // PushDescriptorsDelta scratch
	std::vector<vk::DescriptorBufferInfo> m_delta_verify_buffers; // KYTY_DISPATCH_DELTA_VERIFY
	std::vector<uint32_t>                 m_delta_verify_data;
	[[nodiscard]] const DispatchDelta* DispatchDeltaBase(const void* owner, const ShaderComputeInputInfo& input,
	                                                     const ShaderProgram& program) const;
	[[nodiscard]] bool DispatchDeltaStateKept(const DispatchDelta* delta, const void* pipeline,
	                                          vk::Pipeline handle) const;
	// The scheduler's state at a dispatch's pipeline bind, for CaptureDispatchDelta.
	struct DispatchDeltaMark {
		uint64_t tick = 0, command_serial = 0, compute_serial = 0, unregister_epoch = 0;
		bool     set = false;
	};
	[[nodiscard]] DispatchDeltaMark MarkDispatchDelta(uint64_t unregister_epoch) const;
	// RebindBuffers with the base's slots; with KYTY_DISPATCH_DELTA_VERIFY the full path runs after
	// it, the two are compared, and `delta` is cleared (the dispatch commits and binds in full).
	void RebindBuffersDelta(PreparedBindings& bindings, const DispatchDelta*& delta, uint64_t shader_hash);
	void CaptureDispatchDelta(const void* owner, const ShaderComputeInputInfo& input, const ShaderProgram& program,
	                          void* pipeline, vk::Pipeline handle, const PreparedBindings& bindings,
	                          const DispatchDeltaMark& mark, bool usable);
	[[nodiscard]] bool DeltaSlotReusable(uint32_t slot, const PreparedBindings::BufferSource& source, bool written,
	                                     bool table_read, bool formatted, uint32_t offset_dword) const;
	void PushDescriptorsDelta(const CommandRecorder& recorder, vk::PipelineBindPoint bind_point,
	                          vk::PipelineLayout layout);

	struct MultiDrawReplayBase {
		bool                     active = false; // inside a DRAW_INDIRECT_MULTI
		bool                     valid  = false;
		uint64_t                 tick   = 0;
		const CommandBuffer*     buffer = nullptr;
		RenderState              rendering;
		uint64_t                 commit_generation = 0;
		vk::PipelineLayout       layout;
		ShaderMeshInputInfo      mesh;
		uint64_t                 vs_params_hash = 0;
		uint64_t                 vs_shader_hash = 0;
		uint64_t                 ps_shader_hash = 0;
		const ShaderRecompiler::IR::CompiledShaderInfo* vs_program = nullptr;
		std::array<uint32_t, 40> vs_user {};
		uint32_t                 vs_user_count = 0;
		uint64_t                 vs_plan_reads = ~0ull;
		std::array<uint32_t, 32> ps_user {};
		uint32_t                 ps_user_count = 0;
		bool                     ps_active     = false;
		std::vector<uint32_t>    vs_shader_data;
		std::vector<uint32_t>    bound_data; // the stage's shader data the commit holds now
		std::vector<uint32_t>    scratch;
		bool                     indirect = false;
		uint64_t                 args_run_generation = 0;
		uint64_t                 buffer_unregister_epoch = 0;
		const void*              gpu_args_buffer = nullptr;
		vk::Buffer               mesh_indirect_buffer;
		vk::DeviceSize           mesh_indirect_offset = 0;
		uint32_t                 start_instance_sgpr = 0;
		uint32_t                 indirect_invalid_reason = 1;
		// Context and user-config registers at capture (bytes): register packets between indirect
		// draws are allowed only because replay checks the state they leave is the same.
		std::vector<uint8_t>     ctx_snapshot;
		std::vector<uint8_t>     ucfg_snapshot;
		std::vector<uint8_t>     sh_snapshot; // HW::Shader bytes, GS/PS user SGPR values zeroed
		// KYTY_RENDER_STATS census ([multi-packets]): the state the last replayed packet ended
		// with, while its base still stood (the other base fields are that base's).
		bool                     census_indexed = false; // inside an indexed multi-draw packet
		bool                     census_valid   = false;
		uint64_t                 census_tick    = 0;
		uint64_t                 census_commit_generation = 0;
		std::vector<uint8_t>     census_ctx;
		std::vector<uint8_t>     census_ucfg;
		std::vector<uint8_t>     census_sh;
		std::vector<uint8_t>     census_scratch;
	};
	MultiDrawReplayBase m_replay;

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	// Bindless heaps already surveyed (base ^ table offset << 48).
	std::unordered_set<uint64_t>          m_bindless_surveyed;

	void PrepareBindlessSamplers(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
	                    uint32_t index_offset, int32_t base_vertex, uint32_t first_vertex,
	                    int32_t host_vertex_offset, uint32_t host_first_instance, bool indirect);
	uint64_t                              m_bindless_frame = UINT64_MAX;
	std::vector<uint32_t>                 m_bindless_requests;
	// v439b: (value heap, key) pairs ProcessValueRequests created this round.
	std::vector<std::pair<BindlessTable::Heap*, uint32_t>> m_bindless_value_created;
	std::vector<std::array<uint32_t, 8>>  m_bindless_window;
	std::vector<std::pair<uint64_t, uint64_t>> m_bindless_written;
	std::vector<ImageId>                       m_bindless_changed;
	// Images resolved into the bindless storage array (by ImageId index | generation << 32), with
	// the storage binding a draw that may write them commits again; they stay in the general layout.
	std::unordered_map<uint64_t, TextureCache::ImageDesc> m_bindless_storage;
	uint64_t                                   m_bindless_eager_frame = UINT64_MAX;
	uint32_t                                   m_bindless_eager_left  = 0;
	bool                                       m_bindless_eager_device_full = false;
	uint64_t                                   m_bindless_eager_bytes = 0;
	// Bindless usage probe (UpdateBindlessUsageProbe): the frame it was armed in (0 = idle), the
	// first frame the next may start, and the frame the pending feedback snapshot was recorded.
	uint64_t                              m_bindless_probe_frame      = 0;
	uint64_t                              m_bindless_next_probe       = 0;
	uint64_t                              m_bindless_snapshot_frame   = 0;
	std::vector<ImageId>                  m_bindless_unused;
	void UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame);
	std::vector<uint32_t>                 m_bindless_srt;
	std::vector<uint32_t>                 m_verify_srt; // KYTY_SRT_VERIFY: flat SRT with pairs
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<uint32_t>                 m_draw_bindless_regions; // KYTY_FAULT_MAP draw record
	// KYTY_FAULT_MAP=1: the last CommitBindings' buffer bindings (and, for dispatches, image
	// views and bindless regions) into the fault record of the work just noted.
	void NoteFaultMapBindings(bool with_views) const;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	// Created at the first thread-dimension indirect dispatch.
	std::unique_ptr<IndirectDispatchGroups> m_indirect_groups;
	// v440p: set by DispatchIndirectRecord around a callable split's GPU-converted launch.
	const IndirectDispatchGroups::RtLaunch* m_rt_indirect_launch = nullptr;
	// Created at the first indirect fast-launch mesh draw with counts on the GPU.
	std::unique_ptr<MeshIndirectArgs>       m_mesh_args;
	// Ring of VkDrawMeshTasksIndirectCommandEXT for indirect fast-launch mesh draws (16 bytes an
	// entry); an entry is rewritten only after the ring has gone round, behind its last read.
	std::unique_ptr<Buffer>                 m_mesh_indirect;
	uint32_t                                m_mesh_indirect_next = 0;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;
	std::vector<UpscalerContext>          m_upscaler_contexts;
	std::vector<FsrContext>               m_fsr_contexts;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
