#include "graphics/host_gpu/renderer/indirectDispatch.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/dispatch_indirect_groups_spv.h"
#include "gpu_tiler_shaders/mesh_indirect_args_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

struct PushConstants {
	uint32_t threads_low;
	uint32_t threads_high;
	uint32_t local_size[3];
	uint32_t output_dword;
	uint32_t max_x;
	uint32_t max_y;
	uint32_t max_z;
	uint32_t max_total;
	uint32_t report_low;
	uint32_t report_high;
	uint32_t hash_low;
	uint32_t hash_high;
	uint32_t rt_mode;
};

struct ClampReport {
	uint32_t count;
	uint32_t hash_low;
	uint32_t hash_high;
	uint32_t threads[3];
	uint32_t local_size[3];
};

uint32_t MaxIndirectGroups() {
	const char* value = std::getenv("KYTY_INDIRECT_MAX_GROUPS");
	return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0)) : (1u << 22u);
}

} // namespace

IndirectDispatchGroups::IndirectDispatchGroups(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_groups(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
                   vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst,
               Entries * EntryDwords * sizeof(uint32_t)),
      m_report(graphics, scheduler, MemoryUsage::Download, 0,
               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
               256),
      m_max_total(MaxIndirectGroups()) {
	SetVulkanObjectNameF(graphics.device, m_groups.Handle(), "Indirect Dispatch Groups");
	SetVulkanObjectNameF(graphics.device, m_report.Handle(), "Indirect Dispatch Clamp Report");
	std::memset(m_report.Mapped().data(), 0, sizeof(ClampReport));
	m_report.Flush(0, m_report.Size());

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                               &m_set_layout),
	                     "create indirect-dispatch descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create indirect-dispatch pipeline layout");

	const auto                        module = CompileSPV(DISPATCH_INDIRECT_GROUPS_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create indirect-dispatch pipeline");
}

IndirectDispatchGroups::~IndirectDispatchGroups() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

IndirectDispatchGroups::Result IndirectDispatchGroups::Convert(
    const CommandRecorder& command, vk::DeviceAddress threads,
    const std::array<uint32_t, 3>& local_size, bool pre_barrier, bool post_barrier,
    uint64_t shader_hash) {
	if ((++m_converted & 511u) == 0u) {
		Report();
	}
	const auto entry  = NextEntry();
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);

	// The counts come from earlier shader or transfer writes; the entry's previous indirect read
	// must finish before it is rewritten.
	if (pre_barrier) {
		vk::MemoryBarrier counts_ready {};
		counts_ready.srcAccessMask =
		    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		counts_ready.dstAccessMask =
		    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &counts_ready, 0,
		                        nullptr, 0, nullptr);
	}

	const vk::DescriptorBufferInfo info {m_groups.Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	const auto& limits = m_graphics.physical_device_properties.limits;
	const auto  report = m_report.HasDeviceAddress() ? m_report.BufferDeviceAddress() : 0;
	const PushConstants push {
	    static_cast<uint32_t>(threads), static_cast<uint32_t>(threads >> 32u),
	    {local_size[0], local_size[1], local_size[2]}, entry * EntryDwords,
	    limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1],
	    limits.maxComputeWorkGroupCount[2], m_max_total,
	    static_cast<uint32_t>(report), static_cast<uint32_t>(report >> 32u),
	    static_cast<uint32_t>(shader_hash), static_cast<uint32_t>(shader_hash >> 32u), 0u};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);
	if (!post_barrier) {
		return {m_groups.Handle(), offset,
		        m_groups.BufferDeviceAddress() + offset + 4u * sizeof(uint32_t)};
	}

	vk::MemoryBarrier groups_ready {};
	groups_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	groups_ready.dstAccessMask =
	    vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect |
	                            vk::PipelineStageFlagBits::eComputeShader,
	                        {}, 1, &groups_ready, 0, nullptr, 0, nullptr);
	return {m_groups.Handle(), offset,
	        m_groups.BufferDeviceAddress() + offset + 4u * sizeof(uint32_t)};
}

uint32_t IndirectDispatchGroups::NextEntry() {
	const auto entry = m_next;
	m_next           = (m_next + 1u) % Entries;
	return entry;
}

IndirectDispatchGroups::RtLaunch IndirectDispatchGroups::ConvertRt(const CommandRecorder&         command,
                                                                   vk::DeviceAddress              group_counts,
                                                                   const std::array<uint32_t, 3>& local_size,
                                                                   uint64_t                       shader_hash) {
	const auto entry  = NextEntry();
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);
	// The counts come from earlier shader or transfer writes; the entry's previous reads finish
	// before it is rewritten.
	vk::MemoryBarrier counts_ready {};
	counts_ready.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	counts_ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, 1,
	                        &counts_ready, 0, nullptr, 0, nullptr);
	const vk::DescriptorBufferInfo info {m_groups.Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	const auto& limits = m_graphics.physical_device_properties.limits;
	const PushConstants push {static_cast<uint32_t>(group_counts),
	                          static_cast<uint32_t>(group_counts >> 32u),
	                          {local_size[0], local_size[1], local_size[2]},
	                          entry * EntryDwords,
	                          limits.maxComputeWorkGroupCount[0],
	                          limits.maxComputeWorkGroupCount[1],
	                          limits.maxComputeWorkGroupCount[2],
	                          m_max_total,
	                          0u,
	                          0u,
	                          static_cast<uint32_t>(shader_hash),
	                          static_cast<uint32_t>(shader_hash >> 32u),
	                          1u};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(1, 1, 1);
	vk::MemoryBarrier launch_ready {};
	launch_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	launch_ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eRayTracingShaderKHR, {},
	                        1, &launch_ready, 0, nullptr, 0, nullptr);
	const auto base = m_groups.BufferDeviceAddress() + offset;
	return {base, base + 4u * sizeof(uint32_t)};
}

IndirectDispatchGroups::RtLaunch IndirectDispatchGroups::WriteRtExtents(const CommandRecorder&         command,
                                                                        const std::array<uint32_t, 3>& extents) {
	const auto entry  = NextEntry();
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);
	const uint32_t words[8] = {extents[0], extents[1], extents[2], 0u, extents[0] * extents[1] * extents[2], 1u, 1u, 0u};
	command.updateBuffer(m_groups.Handle(), offset, sizeof(words), words);
	const auto base = m_groups.BufferDeviceAddress() + offset;
	return {base, base + 4u * sizeof(uint32_t)};
}

// Every 512 conversions, at most every 5 s: the dispatches emptied since the last report (the
// GPU's count is read without waiting, so a report can trail by a frame).
void IndirectDispatchGroups::Report() {
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	if (now - m_last_report_ms < 5000) {
		return;
	}
	m_last_report_ms = now;
	m_report.Invalidate(0, sizeof(ClampReport));
	ClampReport report {};
	std::memcpy(&report, m_report.Mapped().data(), sizeof(report));
	if (report.count == 0) {
		return;
	}
	m_emptied += report.count;
	std::fprintf(stderr,
	             "[indirect-clamp] %u dispatches emptied (total %" PRIu64 ", conversions %" PRIu64
	             "); first: hash=%08x%08x args=%u,%u,%u local=%u,%u,%u\n",
	             report.count, m_emptied, m_converted, report.hash_high, report.hash_low,
	             report.threads[0], report.threads[1], report.threads[2], report.local_size[0],
	             report.local_size[1], report.local_size[2]);
	std::fflush(stderr);
	// Racy with a dispatch emptied right now (its count can be lost); diagnostics only.
	std::memset(m_report.Mapped().data(), 0, sizeof(ClampReport));
	m_report.Flush(0, sizeof(ClampReport));
}

namespace {

struct MeshPushConstants {
	uint32_t args_low;
	uint32_t args_high;
	uint32_t output_dword;
	uint32_t max_x;
	uint32_t max_y;
	uint32_t max_total;
};

} // namespace

MeshIndirectArgs::MeshIndirectArgs(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_commands(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                 vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
                 Entries * EntryDwords * sizeof(uint32_t)) {
	SetVulkanObjectNameF(graphics.device, m_commands.Handle(), "Mesh Indirect Commands");

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                               &m_set_layout),
	                     "create mesh-indirect descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(MeshPushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create mesh-indirect pipeline layout");

	const auto module = CompileSPV(MESH_INDIRECT_ARGS_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh-indirect pipeline");
}

MeshIndirectArgs::~MeshIndirectArgs() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

MeshIndirectArgs::Result MeshIndirectArgs::Convert(const CommandRecorder& command,
                                                   vk::DeviceAddress args, bool pre_barrier,
                                                   bool post_barrier) {
	const auto entry  = m_next;
	m_next            = (m_next + 1u) % Entries;
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);

	// The counts' writer (a dispatch or a copy) and the entry's last indirect read come first.
	if (pre_barrier) {
		vk::MemoryBarrier counts_ready {};
		counts_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                             vk::AccessFlagBits::eTransferWrite |
		                             vk::AccessFlagBits::eIndirectCommandRead;
		counts_ready.dstAccessMask =
		    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &counts_ready, 0,
		                        nullptr, 0, nullptr);
	}

	const vk::DescriptorBufferInfo info {m_commands.Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	const auto& limits = m_graphics.mesh_shader_properties;
	const MeshPushConstants push {static_cast<uint32_t>(args), static_cast<uint32_t>(args >> 32u),
	                              entry * EntryDwords, limits.maxMeshWorkGroupCount[0],
	                              limits.maxMeshWorkGroupCount[1],
	                              limits.maxMeshWorkGroupTotalCount};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);

	if (post_barrier) {
		vk::MemoryBarrier command_ready {};
		command_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
		command_ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
		                        vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &command_ready, 0,
		                        nullptr, 0, nullptr);
	}
	return {m_commands.Handle(), offset};
}

} // namespace Libs::Graphics
