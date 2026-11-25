#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include <string>
#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <atomic>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;
struct SparseImageState;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct DiagnosticCheckpoint {
	uint32_t op        = 0;
	uint64_t submit_id = 0;
	uint32_t arg0      = 0;
	uint32_t arg1      = 0;
	uint32_t arg2      = 0;
	uint32_t arg3      = 0;
	uint64_t arg4      = 0;
	uint64_t sequence  = 0;
};

struct GraphicContext;

// Returns a stable marker for vkCmdSetCheckpointNV.
[[nodiscard]] const DiagnosticCheckpoint*
RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint);
// After a device loss: VK_EXT_device_fault info, the NVIDIA crash dump written to
// _device_fault.nv-gpudmp, and the queue's last diagnostic checkpoints.
void DumpDeviceLossDiagnostics(GraphicContext& graphics);
void DumpGpuHangCheckpoints(GraphicContext& graphics, uint64_t tick, double seconds);

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	// KYTY_ADDRESS_BINDING_REPORT=1: logs every GPU address range bound or unbound, so a
	// device-fault address can be matched to the object that owned it.
	vk::DebugUtilsMessengerEXT         address_binding_messenger             = nullptr;
	bool                               address_binding_report_enabled        = false;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	bool                               external_memory_host_enabled          = false;
	bool                               diagnostic_checkpoints_enabled        = false;
	// v438w: what a diagnostic run adds to the device (a normal play start has none): bit 0 the NV
	// checkpoint extension, bit 1 the NV diagnostics config, bit 2 pipeline statistics queries.
	// The driver caches and the warmed record of such a device are files of their own
	// (DeviceConfigTag, GplWarmLoad).
	uint32_t                           diagnostic_device_bits                = 0;
	// With --profile, for the host GPU timeline (renderer/gpuProfiler).
	bool                               host_query_reset_enabled              = false;
	bool                               calibrated_timestamps_enabled         = false;
	bool                               device_fault_enabled                  = false;
	bool                               shader_device_clock_enabled           = false;
	bool                               compute_subgroup_size_control_enabled = false;
	// subgroupSizeControl is enabled: the device has more than one subgroup size.
	bool                               subgroup_size_control_enabled         = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	// bool fp64_denorm_preserve = false; // Temporarily disabled.
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	// KYTY_GPL=1 and VK_EXT_graphics_pipeline_library: graphics pipelines are linked from libraries.
	bool                               graphics_pipeline_library_enabled     = false;
	bool                               pipeline_robustness_enabled           = false;
	// KYTY_GPL: VK_EXT_extended_dynamic_state3 depth clip enable, depth clip -1..1, polygon mode.
	bool                               eds3_raster_enabled                   = false;
	bool                               depth_color_copy_enabled               = false;
	bool                               supports_block_texel_view              = false;
	bool                                      mesh_shader_enabled                   = false;
	// v440 (KYTY_RT_CALLABLES): VK_KHR_ray_tracing_pipeline, for callable splits of the guest's
	// ray-tracing shaders (no acceleration structures are built). The shader group handle layout.
	bool                               ray_tracing_pipeline_enabled          = false;
	uint32_t                           rt_group_handle_size                  = 0;
	uint32_t                           rt_group_handle_alignment             = 0;
	uint32_t                           rt_group_base_alignment               = 0;
	uint32_t                           rt_max_group_stride                   = 0; // maxShaderGroupStride
	bool                               rt_pipeline_library_enabled           = false; // VK_KHR_pipeline_library
	bool                               rt_trace_indirect_enabled             = false; // rayTracingPipelineTraceRaysIndirect
	// v442: descriptorBindingPartiallyBound is enabled (a callable split's fixed set layout).
	bool                               partially_bound_enabled               = false;
	// Descriptor indexing for bindless images: runtime arrays indexed non-uniformly, partially
	// bound and updated after bind.
	bool                               bindless_enabled                      = false;
	uint32_t                           bindless_max_sampled_images           = 0;
	uint32_t                           bindless_max_samplers                 = 0;
	// Storage images in the bindless set too: indexed non-uniformly, updated after bind.
	bool                               bindless_storage_enabled              = false;
	uint32_t                           bindless_max_storage_images           = 0;
	bool                               occlusion_query_precise               = false;
	vk::DescriptorSetLayout            bindless_layout                       = nullptr;
	vk::DescriptorSet                  bindless_set                          = nullptr;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;
	// Queue 1 of queue_family when the device has one (BufferCache copy-queue readbacks). Only
	// the GPU thread submits to it.
	uint32_t                           queue_count    = 1;
	vk::Queue                          readback_queue = nullptr;
	// KYTY_DMA_READBACK (v309, default on since v311): a queue of a transfer-only family (the copy engines, which run
	// alongside the graphics engine) for copy-queue readbacks; buffers are then shared by both
	// families (concurrent). transfer_family is -1 without it.
	uint32_t                           transfer_family = static_cast<uint32_t>(-1);
	vk::Queue                          transfer_queue  = nullptr;
	// Frames presented so far: the clock the caches age their entries by.
	std::atomic<uint64_t>              presented_frames {0};

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] const vk::PhysicalDeviceMemoryProperties&
	GetPhysicalDeviceMemoryProperties() const {
		return physical_device_memory_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
	}

	// The subgroup size a graphics stage must require so that one host subgroup is one guest
	// wave, or 0 when the default already is or the device cannot require it for the stage. On
	// AMD the default is 64; a wave32 pixel or mesh shader would otherwise share a subgroup with
	// another wave, and its ballots, lane reads and reductions would mix the two.
	[[nodiscard]] uint32_t GraphicsSubgroupSize(vk::ShaderStageFlagBits stage,
	                                            uint32_t                wave_size) const noexcept {
		if (!subgroup_size_control_enabled || wave_size == subgroup_size ||
		    !(required_subgroup_size_stages & stage) || wave_size < min_subgroup_size ||
		    wave_size > max_subgroup_size) {
			return 0;
		}
		return wave_size;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	// v386 [mem-census]: one line of device memory (VMA heaps: usage, budget, allocated, blocks;
	// the free-delay quarantine) for the census.
	[[nodiscard]] std::string DescribeDeviceMemory() const;
	// v438zr: sparse residency images (one layer, 2D). CreateSparseImage makes the image with no
	// memory; BindSparseLevels gives memory to the levels from `first` on that have none yet (and
	// to the mip tail the first time) and returns when the device has bound it.
	[[nodiscard]] bool CreateSparseImage(const vk::ImageCreateInfo& image_info, VulkanImage& image,
	                                     SparseImageState& state);
	[[nodiscard]] bool BindSparseLevels(VulkanImage& image, SparseImageState& state, uint32_t first);
	void               DeleteSparseImage(VulkanImage& image, SparseImageState& state);
	// v438zv: a sparse bind no longer waits for the device on the thread that asked for it (the
	// GPU thread: 4.9 % of it in Mac's play of 11-18 was this wait). Its fence is kept here, and
	// every queue submission first waits for all of them (WaitSparseBinds, on the thread that
	// submits): a command buffer recorded after a bind is submitted after it, so it still runs
	// with the memory bound. KYTY_TEX_SPARSE_WAIT=1: wait at the bind, as v438zr did.
	void               WaitSparseBinds();
	std::mutex             sparse_fence_mutex;
	std::vector<vk::Fence> sparse_fences;
	// The sparse features are enabled on the device (set at device creation).
	bool sparse_images = false;
	// v438zo [vma-frag]: per memory type, the blocks, what is allocated in them and the free space
	// left inside them (how many free ranges, the largest). Walks the allocator: for the census only.
	[[nodiscard]] std::string DescribeFragmentation() const;
	// The bytes the allocator gave this image (0 when it has none).
	[[nodiscard]] uint64_t ImageAllocationBytes(const VulkanImage& image) const;
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept { return memory_budget_ext_enabled; }
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;
	// Memory pressure in bytes, read at most every 250 ms; the caches' collectors lower their
	// thresholds by it, and DeviceMemoryAtLeast reports true while it is not 0. The larger of:
	// - the system's free commit below a reserve (KYTY_COMMIT_RESERVE_MB; default 0 = off, see
	//   MemoryShortfall for why). On Windows every GB of video memory in use is also charged to
	//   the commit limit (WDDM backs pageable allocations), so a PC with a small page file runs
	//   out of commit with video memory to spare (ISSUES #21);
	// - everything the device holds past 90 % of the budget, which KYTY_VRAM_TARGET_MB caps
	//   (default 14336: the emulator plans for a 16 GB card whatever card it runs on).
	[[nodiscard]] uint64_t MemoryShortfall() const;
	// v387: everything the device holds past percent of the budget (no commit part), read at most
	// every 250 ms. The texture collector trims at KYTY_VRAM_TRIM_PERCENT (default 85) so the total
	// stays clear of the budget, past which the driver moves allocations to system memory (fps ~6
	// after ~20 min of play, 2026-11-11).
	[[nodiscard]] uint64_t DeviceExcessOver(uint32_t percent) const;
	// Whether everything the device holds (images, buffers, the driver's own) has reached percent
	// of the device-local budget, or memory is short (MemoryShortfall). False when the driver
	// cannot report usage and nothing is short.
	[[nodiscard]] bool DeviceMemoryAtLeast(uint32_t percent) const {
		if (MemoryShortfall() != 0) {
			return true;
		}
		if (!CanReportMemoryUsage()) {
			return false;
		}
		const auto budget = GetTotalMemoryBudget();
		return budget != 0 && GetDeviceMemoryUsage() / 100 >= budget / 100 * percent / 100;
	}
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);
	// vmaDestroyBuffer, or, with KYTY_FREE_DELAY_MS=<ms>, that long later (diagnostics: GPU reads
	// of freed memory stop faulting while the memory is held).
	void                   DestroyBufferMemory(VkBuffer buffer, VmaAllocation allocation);

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
	// Set by a guest draw's or dispatch's bindings (KYTY_GUEST_STORAGE_REPEAT, Image::GetBarriers).
	bool guest = false;
};

// v438zr (KYTY_TEX_SPARSE): a sparse image's memory, level by level. Levels from `bound_first` on
// have memory; the levels of the mip tail (from `tail_first` on) always do, together.
struct SparseImageState {
	bool                       sparse      = false;
	uint32_t                   levels      = 0;
	uint32_t                   tail_first  = 0;
	uint32_t                   bound_first = UINT32_MAX; // UINT32_MAX: nothing bound yet
	bool                       tail_bound  = false;
	vk::Extent3D               granularity = {1, 1, 1};
	uint64_t                   block_bytes = 0;
	uint64_t                   tail_offset = 0;
	uint64_t                   tail_size   = 0;
	uint32_t                   memory_type_bits = 0;
	uint64_t                   bound_bytes = 0;
	uint32_t                   binds       = 0;
	std::vector<VmaAllocation> allocations;
};

// KYTY_TEX_SPARSE: 0 off, 1 sparse images with every level bound, 2 only the levels a view can
// reach. Unset: KYTY_TEX_SPARSE_NVIDIA on NVIDIA GPUs (Frank default 2, v438zy), off elsewhere.
// KYTY_TEX_SPARSE_VERIFY=1: an exit where a view reaches a level without memory.
[[nodiscard]] uint32_t TexSparseMode();
void                   SetTexSparseDefault(uint32_t mode);
[[nodiscard]] bool     TexSparseVerify();

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	VmaAllocation                allocation = nullptr;
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
