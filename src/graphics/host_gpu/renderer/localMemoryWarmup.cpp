#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "gpu_tiler_shaders/local_memory_warmup_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuPad.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace Libs::Graphics {

// KYTY_LOCAL_MEMORY_WARMUP=<bytes per thread>: before any guest work, run one dispatch that needs
// that much local memory per thread, so the driver sizes its local-memory area up front instead of
// growing it while guest command buffers are in flight. A diagnostic for device losses that fault
// on writes above every allocation Kyty made.
void WarmUpLocalMemory(GraphicContext& graphics, CommandScheduler& /*scheduler*/) {
	const char* value = std::getenv("KYTY_LOCAL_MEMORY_WARMUP");
	if (value == nullptr) {
		return;
	}
	const auto bytes = std::strtoul(value, nullptr, 0);
	if (bytes < 4) {
		return;
	}
	const uint32_t array_size = static_cast<uint32_t>(bytes / 4u);

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	vk::DescriptorSetLayout set_layout {};
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &set_layout),
	                     "create warm-up descriptor layout");

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &set_layout;
	vk::PipelineLayout pipeline_layout {};
	RequireVulkanSuccess(
	    graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout),
	    "create warm-up pipeline layout");

	const vk::SpecializationMapEntry entry {0, 0, sizeof(uint32_t)};
	vk::SpecializationInfo           specialization {};
	specialization.mapEntryCount = 1;
	specialization.pMapEntries   = &entry;
	specialization.dataSize      = sizeof(array_size);
	specialization.pData         = &array_size;

	const auto                        module = CompileSPV(LOCAL_MEMORY_WARMUP_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage               = vk::ShaderStageFlagBits::eCompute;
	stage.module              = module;
	stage.pName               = "main";
	stage.pSpecializationInfo = &specialization;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = pipeline_layout;
	vk::Pipeline pipeline {};
	const auto   result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create warm-up pipeline");

	// This runs before the guest GPU thread exists, so it records and submits on its own instead of
	// going through the command scheduler.
	constexpr vk::DeviceSize OutputSize = 64 * sizeof(uint32_t);
	vk::BufferCreateInfo     buffer_info {};
	buffer_info.size  = OutputSize;
	buffer_info.usage = vk::BufferUsageFlagBits::eStorageBuffer;
	vk::Buffer output {};
	RequireVulkanSuccess(graphics.device.createBuffer(&buffer_info, nullptr, &output),
	                     "create warm-up buffer");
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(output, &requirements);
	uint32_t memory_type = UINT32_MAX;
	for (uint32_t i = 0; i < graphics.physical_device_memory_properties.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (graphics.physical_device_memory_properties.memoryTypes[i].propertyFlags &
		     vk::MemoryPropertyFlagBits::eDeviceLocal)) {
			memory_type = i;
			break;
		}
	}
	EXIT_IF(memory_type == UINT32_MAX);
	vk::MemoryAllocateInfo allocate_info {};
	allocate_info.allocationSize  = requirements.size;
	allocate_info.memoryTypeIndex = memory_type;
	vk::DeviceMemory memory {};
	RequireVulkanSuccess(graphics.device.allocateMemory(&allocate_info, nullptr, &memory),
	                     "allocate warm-up memory");
	graphics.device.bindBufferMemory(output, memory, 0);

	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = graphics.queue_family;
	vk::CommandPool pool {};
	RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &pool),
	                     "create warm-up command pool");
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool        = pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command {};
	RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&command_info, &command),
	                     "allocate warm-up command buffer");

	const vk::DescriptorBufferInfo info {output, 0, OutputSize};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;

	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin warm-up command buffer");
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &write);
	command.dispatch(4096, 1, 1);
	command.end();

	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence {};
	RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create warm-up fence");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.submit(1, &submit, fence), "submit warm-up");
	}
	RequireVulkanSuccess(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for warm-up");

	graphics.device.destroyFence(fence, nullptr);
	graphics.device.destroyCommandPool(pool, nullptr);
	graphics.device.destroyBuffer(output, nullptr);
	graphics.device.freeMemory(memory, nullptr);
	graphics.device.destroyPipeline(pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(set_layout, nullptr);
	LOGF("Local memory warm-up: %u bytes per thread\n", array_size * 4u);
}

// GpuPad (gpuPad.h): KYTY_SYNC_TOGGLE_MODE=gpupad's extra dispatch. Everything here runs on the
// GPU thread: the flip records it, and the timestamps are read by an operation deferred to the
// command buffer's tick (CommandScheduler::DeferOperation, run by the GPU thread).
namespace GpuPad {

namespace {

constexpr uint32_t ArraySize     = 256; // loop steps and local dwords a thread (the shader's ARRAY_SIZE)
constexpr uint32_t GroupsPerRow  = 64;  // dispatch(GroupsPerRow, rows, 1), 64 threads a group
constexpr uint32_t TimingSlots   = 32;
constexpr uint32_t CalibrateMax  = 240; // measurements before the size is kept whatever they say

struct State {
	bool                    created = false;
	bool                    failed  = false;
	std::unique_ptr<Buffer> scratch;
	vk::DescriptorSetLayout set_layout      = nullptr;
	vk::PipelineLayout      pipeline_layout = nullptr;
	vk::Pipeline            pipeline        = nullptr;
	vk::QueryPool           pool            = nullptr;
	vk::Device              device;
	double                  period_ns = 0.0;
	uint32_t                max_rows  = 65535;
	uint32_t                rows      = 64;
	bool                    fixed     = false; // KYTY_GPU_PAD_ROWS, or the adjustment is over
	double                  target_us = 3000.0;
	std::array<bool, TimingSlots> in_flight {};
	uint32_t                next_slot = 0;
	uint32_t                measured  = 0; // measurements used for the adjustment
	uint32_t                settled   = 0; // of them, in a row, within 5 % of the target
	// The report window.
	uint64_t                dispatches = 0, samples = 0;
	double                  sum_us = 0.0, min_us = 0.0, max_us = 0.0;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

// Never destroyed: its Vulkan objects would outlive the device's allocator at exit otherwise
// (a measurement mode; the driver frees them with the device).
State& Get() {
	static State* const state = new State;
	return *state;
}

bool Create(State& state, RenderContext& context) {
	auto& graphics = context.GetGraphics();
	state.device   = graphics.device;
	if (const char* value = std::getenv("KYTY_GPU_PAD_US"); value != nullptr) {
		state.target_us = std::max(1.0, std::strtod(value, nullptr));
	}
	state.max_rows = std::min<uint32_t>(65535u, graphics.physical_device_properties.limits.maxComputeWorkGroupCount[1]);
	if (const char* value = std::getenv("KYTY_GPU_PAD_ROWS"); value != nullptr) {
		state.rows  = std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, state.max_rows);
		state.fixed = true;
	}
	state.scratch = std::make_unique<Buffer>(graphics, context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
	                                         vk::BufferUsageFlagBits::eStorageBuffer, 64 * sizeof(uint32_t));
	SetVulkanObjectNameF(graphics.device, state.scratch->Handle(), "GPU Pad Scratch");

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	if (graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &state.set_layout) != vk::Result::eSuccess) {
		return false;
	}
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &state.set_layout;
	if (graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &state.pipeline_layout) !=
	    vk::Result::eSuccess) {
		return false;
	}
	const uint32_t                   array_size = ArraySize;
	const vk::SpecializationMapEntry entry {0, 0, sizeof(uint32_t)};
	vk::SpecializationInfo           specialization {};
	specialization.mapEntryCount = 1;
	specialization.pMapEntries   = &entry;
	specialization.dataSize      = sizeof(array_size);
	specialization.pData         = &array_size;
	const auto                        module = CompileSPV(LOCAL_MEMORY_WARMUP_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage               = vk::ShaderStageFlagBits::eCompute;
	stage.module              = module;
	stage.pName               = "main";
	stage.pSpecializationInfo = &specialization;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = state.pipeline_layout;
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &state.pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	if (result != vk::Result::eSuccess) {
		return false;
	}
	// Timestamps are optional: without them the size stays as it starts (or as KYTY_GPU_PAD_ROWS says).
	state.period_ns = graphics.physical_device_properties.limits.timestampPeriod;
	if (state.period_ns > 0.0) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = TimingSlots * 2u;
		if (graphics.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
			state.pool = nullptr;
		}
	}
	std::fprintf(stderr,
	             "[gpu-pad] ready: target %.0f us, %u rows to start (%s), a row = %u groups x 64 threads x %u steps, "
	             "timestamps %s\n",
	             state.target_us, state.rows, state.fixed ? "fixed by KYTY_GPU_PAD_ROWS" : "adjusted over the first on window",
	             GroupsPerRow, ArraySize, state.pool ? "on" : "OFF (no adjustment)");
	return true;
}

// One measurement of a dispatch of `rows` rows.
void Measured(State& state, uint32_t rows, double us) {
	state.samples++;
	state.sum_us += us;
	state.min_us = state.samples == 1 ? us : std::min(state.min_us, us);
	state.max_us = std::max(state.max_us, us);
	if (state.fixed || rows != state.rows || us <= 0.0) {
		return; // a measurement of an older size says nothing about this one
	}
	const double ratio = state.target_us / us;
	state.measured++;
	state.settled = ratio > 0.95 && ratio < 1.05 ? state.settled + 1u : 0u;
	if (state.settled >= 30u || state.measured >= CalibrateMax) {
		state.fixed = true;
		std::fprintf(stderr, "[gpu-pad] size kept from here on: %u rows (%.1f M loop steps), last measured %.0f us (%s)\n",
		             state.rows, static_cast<double>(state.rows) * GroupsPerRow * 64.0 * ArraySize / 1e6, us,
		             state.settled >= 30u ? "30 measurements in a row within 5 % of the target"
		                                  : "NOT settled: the GPU time of the same work varies");
		return;
	}
	if (ratio > 0.97 && ratio < 1.03) {
		return;
	}
	// Half of the way to the size the measurement asks for, at most a factor of two a step.
	const double step = 1.0 + (std::clamp(ratio, 0.5, 2.0) - 1.0) * 0.5;
	const auto   next = static_cast<uint32_t>(static_cast<double>(state.rows) * step + 0.5);
	state.rows        = std::clamp<uint32_t>(next == state.rows ? (ratio > 1.0 ? next + 1u : next - 1u) : next, 1u,
	                                         state.max_rows);
}

} // namespace

void RecordSlow(CommandBuffer& buffer) {
	auto& state = Get();
	if (state.failed) {
		return;
	}
	auto& context = buffer.GetContext();
	if (!state.created) {
		state.created = true;
		if (!Create(state, context)) {
			state.failed = true;
			std::fprintf(stderr, "[gpu-pad] could not be created: no pad is recorded\n");
			return;
		}
	}
	auto& scheduler = context.GetCommandScheduler();
	buffer.EndRendering();
	const auto command = buffer.Recorder();
	const auto rows    = state.rows;
	// By itself on the GPU: everything recorded before it finishes first, and it finishes before
	// anything recorded after it. Its only memory is the scratch buffer.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, 1,
	                        &before, 0, nullptr, 0, nullptr);
	int32_t slot = -1;
	if (state.pool && !state.in_flight[state.next_slot % TimingSlots]) {
		slot = static_cast<int32_t>(state.next_slot++ % TimingSlots);
		state.in_flight[static_cast<size_t>(slot)] = true;
		command.resetQueryPool(state.pool, static_cast<uint32_t>(slot) * 2u, 2u);
		command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool, static_cast<uint32_t>(slot) * 2u);
	}
	const vk::DescriptorBufferInfo info {state.scratch->Handle(), 0, 64 * sizeof(uint32_t)};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, state.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, state.pipeline_layout, 0, 1, &write);
	command.dispatch(GroupsPerRow, rows, 1);
	if (slot >= 0) {
		command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
		                       static_cast<uint32_t>(slot) * 2u + 1u);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands, {}, 1,
	                        &after, 0, nullptr, 0, nullptr);
	state.dispatches++;
	if (slot >= 0) {
		// Runs on this thread once the command buffer's tick has finished on the GPU.
		scheduler.DeferOperation([slot, rows] {
			auto&    state = Get();
			uint64_t values[2] {};
			const auto result = state.device.getQueryPoolResults(state.pool, static_cast<uint32_t>(slot) * 2u, 2u,
			                                                     sizeof(values), values, sizeof(uint64_t),
			                                                     vk::QueryResultFlagBits::e64);
			state.in_flight[static_cast<size_t>(slot)] = false;
			if (result == vk::Result::eSuccess && values[1] >= values[0]) {
				Measured(state, rows, static_cast<double>(values[1] - values[0]) * state.period_ns / 1e3);
			}
		});
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - state.last >= std::chrono::seconds(5)) {
		const double seconds = std::chrono::duration<double>(now - state.last).count();
		std::fprintf(stderr,
		             "[gpu-pad] %.0f s on: %llu dispatches (%.1f/s), %u rows (%.1f M loop steps, %s), GPU time measured: "
		             "avg %.0f us, min %.0f, max %.0f in %llu samples (target %.0f us)\n",
		             seconds, static_cast<unsigned long long>(state.dispatches),
		             static_cast<double>(state.dispatches) / seconds, state.rows,
		             static_cast<double>(state.rows) * GroupsPerRow * 64.0 * ArraySize / 1e6,
		             state.fixed ? "kept" : "still being adjusted",
		             state.samples != 0 ? state.sum_us / static_cast<double>(state.samples) : 0.0, state.min_us,
		             state.max_us, static_cast<unsigned long long>(state.samples), state.target_us);
		state.last       = now;
		state.dispatches = state.samples = 0;
		state.sum_us = state.min_us = state.max_us = 0.0;
	}
}

} // namespace GpuPad

} // namespace Libs::Graphics
