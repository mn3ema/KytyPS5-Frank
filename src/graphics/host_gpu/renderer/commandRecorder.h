#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/barrierSink.h"
#include "graphics/host_gpu/renderer/barrierStats.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

// image.cpp: false (and a [stale-view] report) when the view is not alive.
bool CheckLiveImageView(vk::ImageView view, const char* where);


// The Vulkan commands of the hot recording paths, with vk::CommandBuffer's names and arguments
// (ported from IDXTRI's wolverine-perf branch, d9958e28 + ad1e1396).
// Bound to a raw command buffer it records directly; bound to a threaded scheduler
// (KYTY_RECORD_THREAD) it copies the arguments, and every array they point to, into the
// scheduler's command chunk and queues the call for the recording thread. Nothing the caller
// passes by pointer is used after the method returns.
class CommandRecorder {
public:
	// NOLINTNEXTLINE(google-explicit-constructor): raw command buffers stay usable as recorders.
	CommandRecorder(vk::CommandBuffer direct): m_direct(direct) {}
	CommandRecorder(CommandScheduler& scheduler, vk::CommandBuffer direct)
	    : m_scheduler(&scheduler), m_direct(direct) {}

	[[nodiscard]] bool Threaded() const noexcept { return m_direct == nullptr; }
	[[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
		return m_direct == nullptr && m_scheduler == nullptr;
	}

	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		Run([=](vk::CommandBuffer command) { command.bindPipeline(bind_point, pipeline); });
	}
	void draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex,
	          uint32_t first_instance) const {
		GpuTiming::Note(GpuTiming::GapDraw);
		const bool safe = NextDrawSafe();
		Run([=](vk::CommandBuffer command) {
			SetCommandDrawSafe(safe);
			command.draw(vertices, instances, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index, int32_t offset,
	                 uint32_t first_instance) const {
		GpuTiming::Note(GpuTiming::GapDraw);
		const bool safe = NextDrawSafe();
		Run([=](vk::CommandBuffer command) {
			SetCommandDrawSafe(safe);
			command.drawIndexed(indices, instances, first_index, offset, first_instance);
		});
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                  uint32_t stride) const {
		GpuTiming::Note(GpuTiming::GapDraw);
		const bool safe = NextDrawSafe();
		Run([=](vk::CommandBuffer command) {
			SetCommandDrawSafe(safe);
			command.drawIndirect(buffer, offset, count, stride);
		});
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                         uint32_t stride) const {
		GpuTiming::Note(GpuTiming::GapDraw);
		const bool safe = NextDrawSafe();
		Run([=](vk::CommandBuffer command) {
			SetCommandDrawSafe(safe);
			command.drawIndexedIndirect(buffer, offset, count, stride);
		});
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		GpuTiming::Note(GpuTiming::GapDraw);
		const bool safe = NextDrawSafe();
		Run([=](vk::CommandBuffer command) {
			SetCommandDrawSafe(safe);
			command.drawMeshTasksEXT(x, y, z);
		});
	}
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                              uint32_t stride) const {
		GpuTiming::Note(GpuTiming::GapDraw);
		const bool safe = NextDrawSafe();
		Run([=](vk::CommandBuffer command) {
			SetCommandDrawSafe(safe);
			command.drawMeshTasksIndirectEXT(buffer, offset, count, stride);
		});
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		GpuTiming::Note(GpuTiming::GapDispatch);
		Run([=](vk::CommandBuffer command) { command.dispatch(x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		GpuTiming::Note(GpuTiming::GapDispatch);
		Run([=](vk::CommandBuffer command) { command.dispatchIndirect(buffer, offset); });
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		Run([=](vk::CommandBuffer command) { command.bindIndexBuffer(buffer, offset, type); });
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		if (!Threaded()) {
			m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
			return;
		}
		const auto* b = m_scheduler->Stash(buffers, count);
		const auto* o = m_scheduler->Stash(offsets, count);
		const auto* s = m_scheduler->Stash(sizes, sizes != nullptr ? count : 0);
		const auto* d = m_scheduler->Stash(strides, strides != nullptr ? count : 0);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.bindVertexBuffers2(first, count, b, o, s, d);
		});
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		if (!Threaded()) {
			m_direct.pushConstants(layout, stages, offset, size, values);
			return;
		}
		const auto* data = m_scheduler->Stash(static_cast<const std::byte*>(values), size);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.pushConstants(layout, stages, offset, size, data);
		});
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets) const {
		if (!Threaded()) {
			m_direct.bindDescriptorSets(bind_point, layout, first, count, sets, dynamic_count,
			                            dynamic_offsets);
			return;
		}
		const auto* s = m_scheduler->Stash(sets, count);
		const auto* d = m_scheduler->Stash(dynamic_offsets, dynamic_count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.bindDescriptorSets(bind_point, layout, first, count, s, dynamic_count, d);
		});
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) const {
		if (!Threaded()) {
			m_direct.pushDescriptorSetKHR(bind_point, layout, set, count, writes);
			return;
		}
		const auto* copy = StashWrites(writes, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			CheckPushedViews(copy, count);
			command.pushDescriptorSetKHR(bind_point, layout, set, count, copy);
		});
	}
	// v125 (KYTY_FAULT_MAP=1): the views a push descriptor names, checked on the recording thread
	// at the moment the driver copies their headers (MakeImageInfo checked them earlier, on the
	// GPU thread; an image destroyed in between would hand the GPU a header of freed memory).
	static void CheckPushedViews(const vk::WriteDescriptorSet* writes, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			if (writes[i].pImageInfo == nullptr) {
				continue;
			}
			for (uint32_t j = 0; j < writes[i].descriptorCount; j++) {
				(void)CheckLiveImageView(writes[i].pImageInfo[j].imageView, "a push descriptor at record time");
			}
		}
	}
	// vkUpdateDescriptorSets of a set allocated for this command buffer, then its bind: both move
	// to the recording thread (the update is driver work too). The set must not be in use yet.
	void updateAndBindDescriptorSet(vk::Device device, vk::PipelineBindPoint bind_point,
	                                vk::PipelineLayout layout, vk::DescriptorSet set,
	                                uint32_t count, const vk::WriteDescriptorSet* writes,
	                                uint32_t first_set = 0) const {
		if (!Threaded()) {
			device.updateDescriptorSets(count, writes, 0, nullptr);
			m_direct.bindDescriptorSets(bind_point, layout, first_set, 1, &set, 0, nullptr);
			return;
		}
		const auto* copy = StashWrites(writes, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			device.updateDescriptorSets(count, copy, 0, nullptr);
			command.bindDescriptorSets(bind_point, layout, first_set, 1, &set, 0, nullptr);
		});
	}
	void pipelineBarrier(vk::PipelineStageFlags source, vk::PipelineStageFlags destination,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images,
	                     std::source_location site = std::source_location::current()) const {
		GpuTiming::Note(GpuTiming::GapBarrier);
		if (!Threaded()) {
			SetBarrierSite(&site);
			m_direct.pipelineBarrier(source, destination, flags, memory_count, memory, buffer_count,
			                         buffers, image_count, images);
			SetBarrierSite(nullptr);
			return;
		}
		RequireNoNext(memory, memory_count);
		RequireNoNext(buffers, buffer_count);
		RequireNoNext(images, image_count);
		const auto* m = m_scheduler->Stash(memory, memory_count);
		const auto* b = m_scheduler->Stash(buffers, buffer_count);
		const auto* i = m_scheduler->Stash(images, image_count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			SetBarrierSite(&site);
			command.pipelineBarrier(source, destination, flags, memory_count, m, buffer_count, b,
			                        image_count, i);
			SetBarrierSite(nullptr);
		});
	}
	void pipelineBarrier2(const vk::DependencyInfo& dependency,
	                      std::source_location site = std::source_location::current()) const {
		GpuTiming::Note(GpuTiming::GapBarrier);
		if (!Threaded()) {
			SetBarrierSite(&site);
			m_direct.pipelineBarrier2(dependency);
			SetBarrierSite(nullptr);
			return;
		}
		EXIT_IF(dependency.pNext != nullptr);
		RequireNoNext(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		RequireNoNext(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount);
		RequireNoNext(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		auto copy = dependency;
		copy.pMemoryBarriers =
		    m_scheduler->Stash(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		copy.pBufferMemoryBarriers = m_scheduler->Stash(dependency.pBufferMemoryBarriers,
		                                                dependency.bufferMemoryBarrierCount);
		copy.pImageMemoryBarriers =
		    m_scheduler->Stash(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		m_scheduler->Record([copy, site](vk::CommandBuffer command) {
			SetBarrierSite(&site);
			command.pipelineBarrier2(copy);
			SetBarrierSite(nullptr);
		});
	}
	// The guest's own barrier (CommandProcessor::EmitGlobalBarrier): with KYTY_BARRIER_SINK it may be
	// requested inside rendering and stay pending (barrierSink.h).
	void guestPipelineBarrier2(const vk::DependencyInfo& dependency,
	                           std::source_location site = std::source_location::current()) const {
		GpuTiming::Note(GpuTiming::GapGuestBarrier);
		EXIT_IF(dependency.pNext != nullptr || dependency.memoryBarrierCount != 1 ||
		        dependency.bufferMemoryBarrierCount != 0 || dependency.imageMemoryBarrierCount != 0);
		const vk::MemoryBarrier2 barrier = dependency.pMemoryBarriers[0];
		Run([=](vk::CommandBuffer command) {
			vk::DependencyInfo copy {};
			copy.memoryBarrierCount = 1;
			copy.pMemoryBarriers    = &barrier;
			SetBarrierSite(&site);
			SetCommandGuestBarrier(true);
			command.pipelineBarrier2(copy);
			SetCommandGuestBarrier(false);
			SetBarrierSite(nullptr);
		});
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const {
		GpuTiming::Note(GpuTiming::GapCopyBuffer);
		if (!Threaded()) {
			m_direct.copyBuffer(source, destination, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.copyBuffer(source, destination, count, r); });
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, const vk::BufferCopy& region) const {
		GpuTiming::Note(GpuTiming::GapCopyBuffer);
		copyBuffer(source, destination, 1, &region);
	}
	void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                  const void* data) const {
		GpuTiming::Note(GpuTiming::GapFill);
		if (!Threaded()) {
			m_direct.updateBuffer(buffer, offset, size, data);
			return;
		}
		const auto* copy = m_scheduler->Stash(static_cast<const std::byte*>(data), size);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.updateBuffer(buffer, offset, size, copy);
		});
	}
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const {
		Run([=](vk::CommandBuffer command) { command.resetQueryPool(pool, first, count); });
	}
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		Run([=](vk::CommandBuffer command) { command.beginQuery(pool, query, flags); });
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer command) { command.endQuery(pool, query); });
	}
	void writeTimestamp(vk::PipelineStageFlagBits stage, vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer command) { command.writeTimestamp(stage, pool, query); });
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t value) const {
		GpuTiming::Note(GpuTiming::GapFill);
		Run([=](vk::CommandBuffer command) { command.fillBuffer(buffer, offset, size, value); });
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		if (!Threaded()) {
			m_direct.copyBufferToImage(source, destination, layout, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyBufferToImage(source, destination, layout, count, r);
		});
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       const vk::BufferImageCopy& region) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		copyBufferToImage(source, destination, layout, 1, &region);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		if (!Threaded()) {
			m_direct.copyImageToBuffer(source, layout, destination, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyImageToBuffer(source, layout, destination, count, r);
		});
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       const vk::BufferImageCopy& region) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		copyImageToBuffer(source, layout, destination, 1, &region);
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		if (!Threaded()) {
			m_direct.copyImage(source, source_layout, destination, destination_layout, count,
			                   regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyImage(source, source_layout, destination, destination_layout, count, r);
		});
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, const vk::ImageCopy& region) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		copyImage(source, source_layout, destination, destination_layout, 1, &region);
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, const vk::ImageResolve& region) const {
		GpuTiming::Note(GpuTiming::GapCopyImage);
		Run([=](vk::CommandBuffer command) {
			command.resolveImage(source, source_layout, destination, destination_layout, 1,
			                     &region);
		});
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges) const {
		GpuTiming::Note(GpuTiming::GapClear);
		if (!Threaded()) {
			m_direct.clearColorImage(image, layout, color, count, ranges);
			return;
		}
		const auto  value = *color;
		const auto* r     = m_scheduler->Stash(ranges, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.clearColorImage(image, layout, &value, count, r);
		});
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges) const {
		GpuTiming::Note(GpuTiming::GapClear);
		if (!Threaded()) {
			m_direct.clearDepthStencilImage(image, layout, value, count, ranges);
			return;
		}
		const auto  clear = *value;
		const auto* r     = m_scheduler->Stash(ranges, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.clearDepthStencilImage(image, layout, &clear, count, r);
		});
	}
	template <typename Writes>
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, const Writes& writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, static_cast<uint32_t>(std::size(writes)),
		                     std::data(writes));
	}
	void beginRendering(const vk::RenderingInfo* info) const { beginRendering(*info); }
	void beginRendering(const vk::RenderingInfo& info) const {
		GpuTiming::Note(GpuTiming::GapRendering);
		if (!Threaded()) {
			m_direct.beginRendering(info);
			return;
		}
		EXIT_IF(info.pNext != nullptr);
		RequireNoNext(info.pColorAttachments, info.colorAttachmentCount);
		RequireNoNext(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1u : 0u);
		RequireNoNext(info.pStencilAttachment, info.pStencilAttachment != nullptr ? 1u : 0u);
		auto copy = info;
		copy.pColorAttachments =
		    m_scheduler->Stash(info.pColorAttachments, info.colorAttachmentCount);
		copy.pDepthAttachment =
		    m_scheduler->Stash(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1u : 0u);
		copy.pStencilAttachment = m_scheduler->Stash(info.pStencilAttachment,
		                                             info.pStencilAttachment != nullptr ? 1u : 0u);
		m_scheduler->Record([copy](vk::CommandBuffer command) {
			// v126 (KYTY_FAULT_MAP=1): the attachments' views at the moment the driver reads them.
			for (uint32_t i = 0; i < copy.colorAttachmentCount; i++) {
				(void)CheckLiveImageView(copy.pColorAttachments[i].imageView,
				                         "a colour attachment at record time");
			}
			if (copy.pDepthAttachment != nullptr) {
				(void)CheckLiveImageView(copy.pDepthAttachment->imageView, "a depth attachment at record time");
			}
			command.beginRendering(copy);
		});
	}
	void endRendering(std::source_location site = std::source_location::current()) const {
		GpuTiming::Note(GpuTiming::GapRendering);
		Run([site](vk::CommandBuffer command) {
			SetBarrierSite(&site);
			command.endRendering();
			SetBarrierSite(nullptr);
		});
	}
	void setCheckpointNV(const void* marker) const {
		Run([=](vk::CommandBuffer command) { command.setCheckpointNV(marker); });
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		if (!Threaded()) {
			m_direct.setViewportWithCount(count, viewports);
			return;
		}
		const auto* v = m_scheduler->Stash(viewports, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setViewportWithCount(count, v); });
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		if (!Threaded()) {
			m_direct.setScissorWithCount(count, scissors);
			return;
		}
		const auto* s = m_scheduler->Stash(scissors, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setScissorWithCount(count, s); });
	}
	void setLineWidth(float width) const {
		Run([=](vk::CommandBuffer command) { command.setLineWidth(width); });
	}
	void setBlendConstants(const float constants[4]) const {
		const std::array<float, 4> values {constants[0], constants[1], constants[2], constants[3]};
		Run([=](vk::CommandBuffer command) { command.setBlendConstants(values.data()); });
	}
	void setDepthClipEnableEXT(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setDepthClipEnableEXT(enable); });
	}
	void setDepthClipNegativeOneToOneEXT(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setDepthClipNegativeOneToOneEXT(enable); });
	}
	void setPolygonModeEXT(vk::PolygonMode mode) const {
		Run([=](vk::CommandBuffer command) { command.setPolygonModeEXT(mode); });
	}
	void setCullMode(vk::CullModeFlags mode) const {
		Run([=](vk::CommandBuffer command) { command.setCullMode(mode); });
	}
	void setFrontFace(vk::FrontFace face) const {
		Run([=](vk::CommandBuffer command) { command.setFrontFace(face); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		Run([=](vk::CommandBuffer command) { command.setDepthBounds(min_bounds, max_bounds); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		Run([=](vk::CommandBuffer command) { command.setDepthCompareOp(op); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setDepthBiasEnable(enable); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		Run([=](vk::CommandBuffer command) { command.setDepthBias(constant, clamp, slope); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		Run([=](vk::CommandBuffer command) { command.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags face, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		Run([=](vk::CommandBuffer command) {
			command.setStencilOp(face, fail, pass, depth_fail, compare);
		});
	}
	void setStencilCompareMask(vk::StencilFaceFlags face, uint32_t mask) const {
		Run([=](vk::CommandBuffer command) { command.setStencilCompareMask(face, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags face, uint32_t mask) const {
		Run([=](vk::CommandBuffer command) { command.setStencilWriteMask(face, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags face, uint32_t reference) const {
		Run([=](vk::CommandBuffer command) { command.setStencilReference(face, reference); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		if (!Threaded()) {
			m_direct.setColorWriteEnableEXT(count, enables);
			return;
		}
		const auto* e = m_scheduler->Stash(enables, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setColorWriteEnableEXT(count, e); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		Run([=](vk::CommandBuffer command) {
			command.setAttachmentFeedbackLoopEnableEXT(aspects);
		});
	}

private:
	template <typename F>
	void Run(F&& fn) const {
		if (!Threaded()) {
			fn(m_direct);
		} else {
			m_scheduler->Record(std::forward<F>(fn));
		}
	}
	template <typename T>
	static void RequireNoNext(const T* items, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			EXIT_IF(items[i].pNext != nullptr);
		}
	}
	// A deep copy of descriptor writes: the writes and the info arrays they point to.
	const vk::WriteDescriptorSet* StashWrites(const vk::WriteDescriptorSet* writes,
	                                          uint32_t                      count) const {
		RequireNoNext(writes, count);
		auto* copy = const_cast<vk::WriteDescriptorSet*>(m_scheduler->Stash(writes, count));
		for (uint32_t i = 0; i < count; i++) {
			auto& write      = copy[i];
			write.pImageInfo = m_scheduler->Stash(
			    write.pImageInfo, write.pImageInfo != nullptr ? write.descriptorCount : 0);
			write.pBufferInfo = m_scheduler->Stash(
			    write.pBufferInfo, write.pBufferInfo != nullptr ? write.descriptorCount : 0);
			write.pTexelBufferView =
			    m_scheduler->Stash(write.pTexelBufferView,
			                       write.pTexelBufferView != nullptr ? write.descriptorCount : 0);
		}
		return copy;
	}

	CommandScheduler* m_scheduler = nullptr;
	vk::CommandBuffer m_direct    = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
