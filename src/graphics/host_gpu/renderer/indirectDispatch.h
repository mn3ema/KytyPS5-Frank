#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDISPATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDISPATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

class CommandRecorder;

struct GraphicContext;
class CommandScheduler;

// DISPATCH_INDIRECT with the thread-dimension initiator carries thread counts, not workgroup
// counts. The command processor used to read them on the CPU, but the previous dispatch writes
// them on the GPU, so every read drained the GPU; the title issues about 200 of these a frame.
// This records a one-invocation pass that converts the counts on the GPU instead.
class IndirectDispatchGroups {
public:
	IndirectDispatchGroups(GraphicContext& graphics, CommandScheduler& scheduler);
	~IndirectDispatchGroups();
	KYTY_CLASS_NO_COPY(IndirectDispatchGroups);

	struct Result {
		vk::Buffer        groups_buffer;
		vk::DeviceSize    groups_offset = 0;
		// A copy of the thread counts that stays valid until the dispatch has run, unlike the
		// cache buffer the guest's arguments live in, which a later binding may merge away.
		vk::DeviceAddress threads = 0;
	};
	// Records the conversion of the three thread counts at `threads` and returns the workgroup
	// counts' location, ready for dispatchIndirect. Binds a compute pipeline and push state, so
	// call it before committing the guest dispatch's bindings.
	// pre_barrier false: the caller has already ordered the counts' writers (and the entry's
	// previous indirect read) before this conversion. post_barrier false: the caller orders the
	// groups before their indirect read (a batch of conversions, KYTY_DISPATCH_RUNS).
	// Counts beyond the device's limits or KYTY_INDIRECT_MAX_GROUPS in all (default 4M groups; 0
	// turns the check off) dispatch nothing; the first such dispatch per report is printed
	// ([indirect-clamp], with shader_hash when the caller knows it). Group-count arguments
	// (no thread-dimension initiator) pass local_size {1, 1, 1}.
	[[nodiscard]] Result Convert(const CommandRecorder& command, vk::DeviceAddress threads,
	                             const std::array<uint32_t, 3>& local_size, bool pre_barrier = true,
	                             bool post_barrier = true, uint64_t shader_hash = 0);

	// v440p, a callable split (ray-tracing pipeline) launch: the thread extents per axis the
	// ray-generation shader reads by address, and the {width, 1, 1} vkCmdTraceRaysIndirectKHR
	// command. ConvertRt converts the guest's indirect group counts at `group_counts` on the GPU
	// (binds compute state and push constants: call it before committing the dispatch's
	// bindings; it orders its writes before the shader and indirect reads). WriteRtExtents writes
	// a direct dispatch's extents with vkCmdUpdateBuffer (the caller orders the transfer write
	// before the trace).
	struct RtLaunch {
		vk::DeviceAddress extents = 0;
		vk::DeviceAddress command = 0;
	};
	[[nodiscard]] RtLaunch ConvertRt(const CommandRecorder& command, vk::DeviceAddress group_counts,
	                                 const std::array<uint32_t, 3>& local_size, uint64_t shader_hash);
	[[nodiscard]] RtLaunch WriteRtExtents(const CommandRecorder& command, const std::array<uint32_t, 3>& extents);

	// Diagnostics: the ring of converted counts (entry n at n * 32 bytes: groups, then threads).
	[[nodiscard]] vk::Buffer           RingHandle() const { return m_groups.Handle(); }
	[[nodiscard]] static constexpr uint64_t RingBytes() { return uint64_t {Entries} * EntryDwords * 4u; }

private:
	// Ring of converted counts. An entry is rewritten only after 4096 later conversions, each
	// ordered behind the indirect reads before it.
	static constexpr uint32_t Entries      = 4096;
	static constexpr uint32_t EntryDwords  = 8; // groups at 0, thread counts at 4

	void Report();
	[[nodiscard]] uint32_t NextEntry();

	GraphicContext&         m_graphics;
	Buffer                  m_groups;
	Buffer                  m_report; // host-visible: see dispatch_indirect_groups.comp
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	uint32_t                m_next            = 0;
	uint32_t                m_max_total       = 0;
	uint64_t                m_converted       = 0;
	uint64_t                m_emptied         = 0;
	int64_t                 m_last_report_ms  = 0;
};

// DRAW_INDIRECT of a fast-launch mesh draw: the guest's vertex and instance counts become the
// mesh-tasks command on the GPU (MeshIndirectArgs::Convert), validated against the device's
// limits there, instead of the command processor reading them, which drained the GPU per draw.
class MeshIndirectArgs {
public:
	MeshIndirectArgs(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshIndirectArgs();
	KYTY_CLASS_NO_COPY(MeshIndirectArgs);

	struct Result {
		vk::Buffer     buffer;
		vk::DeviceSize offset = 0;
	};
	// Records the conversion of the counts at `args` (vertex count, instance count) and returns
	// the command's location, ready for drawMeshTasksIndirectEXT. Binds a compute pipeline and
	// push state, so call it outside rendering and before committing the draw's bindings.
	// pre_barrier / post_barrier = false: the caller orders the counts' writers before and the
	// indirect reads after a batch of conversions itself (KYTY_MESH_ARGS_RUNS).
	[[nodiscard]] Result Convert(const CommandRecorder& command, vk::DeviceAddress args,
	                             bool pre_barrier = true, bool post_barrier = true);

private:
	static constexpr uint32_t Entries     = 4096;
	static constexpr uint32_t EntryDwords = 4;

	GraphicContext&         m_graphics;
	Buffer                  m_commands;
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	uint32_t                m_next            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDISPATCH_H_
