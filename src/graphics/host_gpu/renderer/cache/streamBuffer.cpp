#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/staleHooks.h"
#include "common/fpsLog.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <cinttypes>
#include <cstdio>
#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <numeric>
#include <vk_mem_alloc.h>

#include "graphics/host_gpu/renderer/image/imageHistory.h"
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}


// Every buffer with a device address, live and recently destroyed, so a GPU fault address
// (VK_EXT_device_fault) can be named: which buffer, what guest range, alive or freed (and when).
struct BdaEntry {
	uint64_t device = 0, size = 0, cpu = 0;
	uint64_t created_tick = 0, released_tick = 0;
	std::chrono::steady_clock::time_point created, released;
	MemoryUsage usage  = MemoryUsage::DeviceLocal;
	bool        shared = false;
	bool        no_bda = false; // a buffer without a device address, placed by a probe (KYTY_FAULT_MAP)
};
struct BdaRegistry {
	std::unordered_map<const void*, uint64_t> probed; // buffers without a device address -> va
	std::unordered_map<const void*, uint64_t> handle_va; // VkBuffer -> va (bindings a draw records)
	std::mutex                   mutex;
	std::map<uint64_t, BdaEntry> live;
	std::vector<BdaEntry>        released; // ring
	size_t                       released_next = 0;
	uint64_t                     created_count = 0, released_count = 0;
	// KYTY_FAULT_MAP=1: images too, at the device addresses of their memory (found with a probe
	// buffer bound to the image's memory object).
	struct Image {
		uint64_t va = 0, size = 0;
		uint32_t width = 0, height = 0, depth = 0, layers = 0, levels = 0, format = 0, usage = 0;
		const void* image = nullptr; // VkImage, matched against img= in [gpu-hang] draw views
		std::chrono::steady_clock::time_point created, released;
	};
	std::unordered_map<const void*, Image> live_images;
	std::unordered_map<const void*, std::pair<uint64_t, uint64_t>> image_va; // VkImage -> memory range
	std::vector<Image>                     released_images; // ring
	size_t                                 released_images_next = 0;
	uint64_t                               images_mapped = 0, images_unmapped = 0;
};
BdaRegistry& Bda() {
	static auto* registry = new BdaRegistry; // never destroyed: buffers die at exit too
	return *registry;
}
void BdaNoteCreated(uint64_t device, uint64_t size, uint64_t cpu, MemoryUsage usage, bool shared,
                    uint64_t tick) {
	if (device == 0) return;
	auto& r = Bda();
	std::lock_guard lock(r.mutex);
	BdaEntry e;
	e.device = device; e.size = size; e.cpu = cpu; e.usage = usage; e.shared = shared;
	e.created_tick = tick;
	e.created = std::chrono::steady_clock::now();
	r.live[device] = e;
	r.created_count++;
}
void BdaNoteReleased(uint64_t device, uint64_t tick) {
	if (device == 0) return;
	auto& r = Bda();
	std::lock_guard lock(r.mutex);
	const auto found = r.live.find(device);
	if (found == r.live.end()) return;
	auto e          = found->second;
	e.released      = std::chrono::steady_clock::now();
	e.released_tick = tick;
	r.live.erase(found);
	constexpr size_t Ring = 8192;
	if (r.released.size() < Ring) {
		r.released.push_back(e);
	} else {
		r.released[r.released_next] = e;
		r.released_next = (r.released_next + 1) % Ring;
	}
	r.released_count++;
}
const char* BdaUsageName(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal: return "device-local";
		case MemoryUsage::Upload: return "upload";
		case MemoryUsage::Download: return "download";
		case MemoryUsage::Stream: return "stream";
	}
	return "?";
}
void BdaPrint(const char* what, const BdaEntry& e, uint64_t address, uint64_t now_tick) {
	const auto now = std::chrono::steady_clock::now();
	const auto ms  = [&](std::chrono::steady_clock::time_point t) {
		return std::chrono::duration<double, std::milli>(now - t).count();
	};
	char released[96] = "";
	if (e.released_tick != 0 || e.released.time_since_epoch().count() != 0) {
		std::snprintf(released, sizeof(released), " released %.1f ms ago at tick %" PRIu64,
		              ms(e.released), e.released_tick);
	}
	std::fprintf(stderr,
	             "[device-fault]   %s: dev 0x%016" PRIx64 "+0x%" PRIx64 " (fault at +0x%" PRIx64
	             "%s) guest 0x%016" PRIx64 " %s%s, created %.1f ms ago at tick %" PRIu64 "%s"
	             " (now tick %" PRIu64 ")\n",
	             what, e.device, e.size, address - e.device,
	             address >= e.device + e.size ? ", past its end" : "", e.cpu,
	             BdaUsageName(e.usage), e.shared ? " shared" : (e.no_bda ? " no-BDA" : ""), ms(e.created), e.created_tick,
	             released, now_tick);
}

bool FaultMapEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FAULT_MAP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void PrintImage(const char* what, const BdaRegistry::Image& e, uint64_t address) {
	const auto now = std::chrono::steady_clock::now();
	const auto ms  = [&](std::chrono::steady_clock::time_point t) {
		return std::chrono::duration<double, std::milli>(now - t).count();
	};
	char released[64] = "";
	if (e.released.time_since_epoch().count() != 0) {
		std::snprintf(released, sizeof(released), ", destroyed %.1f ms ago", ms(e.released));
	}
	std::fprintf(stderr,
	             "[device-fault]   %s: image va 0x%016" PRIx64 "+0x%" PRIx64 " (fault at +0x%" PRIx64
	             ") %ux%ux%u layers=%u levels=%u format=%u usage=0x%x img=%p, created %.1f ms ago%s\n",
	             what, e.va, e.size, address - e.va, e.width, e.height, e.depth, e.layers, e.levels,
	             e.format, e.usage, e.image, ms(e.created), released);
}

// The device address of a memory object's start: a small buffer bound there (0: not mappable).
uint64_t ProbeMemoryBase(GraphicContext& graphics, VkDeviceMemory memory, uint32_t memory_type) {
	const auto&        d      = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto         device = static_cast<VkDevice>(graphics.device);
	VkBufferCreateInfo probe_info {};
	probe_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	probe_info.size        = 256;
	probe_info.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer probe         = VK_NULL_HANDLE;
	uint64_t base          = 0;
	if (d.vkCreateBuffer(device, &probe_info, nullptr, &probe) == VK_SUCCESS) {
		VkMemoryRequirements req {};
		d.vkGetBufferMemoryRequirements(device, probe, &req);
		if ((req.memoryTypeBits & (1u << memory_type)) != 0 &&
		    d.vkBindBufferMemory(device, probe, memory, 0) == VK_SUCCESS) {
			vk::BufferDeviceAddressInfo address_info {};
			address_info.buffer = vk::Buffer(probe);
			base                = graphics.device.getBufferAddress(address_info);
		}
		d.vkDestroyBuffer(device, probe, nullptr);
	}
	return base;
}

} // namespace

// KYTY_FAULT_MAP=1: a buffer without a device address, registered at the address of its memory.
void FaultMapNoteBuffer(GraphicContext& graphics, const void* owner, VmaAllocation allocation,
                        uint64_t size, uint64_t cpu, MemoryUsage usage, uint64_t tick, bool created) {
	if (!FaultMapEnabled()) {
		return;
	}
	auto& r = Bda();
	if (!created) {
		uint64_t va = 0;
		{
			std::lock_guard lock(r.mutex);
			const auto found = r.probed.find(owner);
			if (found == r.probed.end()) return;
			va = found->second;
			r.probed.erase(found);
		}
		BdaNoteReleased(va, tick);
		return;
	}
	if (allocation == nullptr) return;
	VmaAllocationInfo ai {};
	vmaGetAllocationInfo(graphics.allocator, allocation, &ai);
	const auto base = ProbeMemoryBase(graphics, ai.deviceMemory, ai.memoryType);
	if (base == 0) return;
	const auto va = base + ai.offset;
	BdaNoteCreated(va, size, cpu, usage, false, tick);
	std::lock_guard lock(r.mutex);
	if (auto found = r.live.find(va); found != r.live.end()) found->second.no_bda = true;
	r.probed[owner] = va;
}

// KYTY_FAULT_MAP=1: VkBuffer -> device address of its memory (va 0 forgets it), so the buffer
// bindings a draw or dispatch records can be compared with a fault address.
void FaultMapNoteHandle(const void* handle, const void* owner, uint64_t device_address) {
	if (!FaultMapEnabled() || handle == nullptr) {
		return;
	}
	auto&           r = Bda();
	std::lock_guard lock(r.mutex);
	uint64_t        va = device_address;
	if (va == 0 && owner != nullptr) {
		if (const auto found = r.probed.find(owner); found != r.probed.end()) va = found->second;
	}
	if (va == 0) {
		r.handle_va.erase(handle);
	} else {
		r.handle_va[handle] = va;
	}
}
void FaultMapForgetHandle(const void* handle) {
	if (!FaultMapEnabled() || handle == nullptr) {
		return;
	}
	auto&           r = Bda();
	std::lock_guard lock(r.mutex);
	r.handle_va.erase(handle);
}
// KYTY_BDA_AUDIT: what a device address points into: "live", "destroyed N ms ago" or "nothing known".
std::string DeviceAddressState(uint64_t address) {
	auto&           r = Bda();
	std::lock_guard lock(r.mutex);
	if (auto it = r.live.upper_bound(address); it != r.live.begin()) {
		--it;
		if (address - it->second.device < it->second.size) {
			return fmt::format("live buffer guest 0x{:x}", it->second.cpu);
		}
	}
	const auto now = std::chrono::steady_clock::now();
	double     best = -1;
	uint64_t   cpu  = 0;
	for (const auto& e: r.released) {
		if (e.size != 0 && address >= e.device && address - e.device < e.size) {
			const auto ms = std::chrono::duration<double, std::milli>(now - e.released).count();
			if (best < 0 || ms < best) {
				best = ms;
				cpu  = e.cpu;
			}
		}
	}
	if (best >= 0) {
		return fmt::format("DESTROYED buffer guest 0x{:x}, freed {:.0f} ms ago", cpu, best);
	}
	return "no buffer known there";
}

uint64_t FaultMapBufferVa(const void* handle) {
	auto&           r = Bda();
	std::lock_guard lock(r.mutex);
	const auto      found = r.handle_va.find(handle);
	return found == r.handle_va.end() ? 0 : found->second;
}

void FaultMapNoteImage(GraphicContext& graphics, VmaAllocation allocation,
                       const vk::ImageCreateInfo& info, bool created, VkImage image) {
	if (!FaultMapEnabled() || allocation == nullptr) {
		return;
	}
	auto& r = Bda();
	if (!created) {
		std::lock_guard lock(r.mutex);
		const auto found = r.live_images.find(allocation);
		if (found == r.live_images.end()) return;
		auto e     = found->second;
		e.released = std::chrono::steady_clock::now();
		r.live_images.erase(found);
		r.image_va.erase(e.image);
		constexpr size_t Ring = 8192;
		if (r.released_images.size() < Ring) {
			r.released_images.push_back(e);
		} else {
			r.released_images[r.released_images_next] = e;
			r.released_images_next = (r.released_images_next + 1) % Ring;
		}
		return;
	}
	VmaAllocationInfo ai {};
	vmaGetAllocationInfo(graphics.allocator, allocation, &ai);
	// The device address of the memory object: a small buffer bound at its start.
	const auto&        d      = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto         device = static_cast<VkDevice>(graphics.device);
	VkBufferCreateInfo probe_info {};
	probe_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	probe_info.size        = 256;
	probe_info.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	probe_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer probe         = VK_NULL_HANDLE;
	uint64_t base          = 0;
	if (d.vkCreateBuffer(device, &probe_info, nullptr, &probe) == VK_SUCCESS) {
		VkMemoryRequirements req {};
		d.vkGetBufferMemoryRequirements(device, probe, &req);
		if ((req.memoryTypeBits & (1u << ai.memoryType)) != 0 &&
		    d.vkBindBufferMemory(device, probe, ai.deviceMemory, 0) == VK_SUCCESS) {
			vk::BufferDeviceAddressInfo address_info {};
			address_info.buffer = vk::Buffer(probe);
			base                = graphics.device.getBufferAddress(address_info);
		}
		d.vkDestroyBuffer(device, probe, nullptr);
	}
	std::lock_guard lock(r.mutex);
	if (base == 0) {
		r.images_unmapped++;
		return;
	}
	BdaRegistry::Image e;
	e.va      = base + ai.offset;
	e.size    = ai.size;
	e.width   = info.extent.width;
	e.height  = info.extent.height;
	e.depth   = info.extent.depth;
	e.layers  = info.arrayLayers;
	e.levels  = info.mipLevels;
	e.format  = static_cast<uint32_t>(info.format);
	e.usage   = static_cast<uint32_t>(info.usage);
	e.image   = image;
	e.created = std::chrono::steady_clock::now();
	r.live_images[allocation] = e;
	r.image_va[image] = {e.va, e.size};
	r.images_mapped++;
}

// diagnosticCheckpoints.cpp: the memory range of a live image (false: unknown or freed).
bool FaultMapImageVa(const void* image, uint64_t& va, uint64_t& size) {
	auto&           r = Bda();
	std::lock_guard lock(r.mutex);
	const auto      found = r.image_va.find(image);
	if (found == r.image_va.end()) {
		return false;
	}
	va   = found->second.first;
	size = found->second.second;
	return true;
}

// image.cpp (v125): the device address range of an image's memory, for the hang listing.
std::string FaultMapImageRange(const void* image) {
	if (!FaultMapEnabled() || image == nullptr) {
		return {};
	}
	auto&           r = Bda();
	std::lock_guard lock(r.mutex);
	char            text[96];
	for (const auto& [key, e]: r.live_images) {
		if (e.image == image) {
			std::snprintf(text, sizeof(text), " va=0x%016llx+0x%llx", static_cast<unsigned long long>(e.va),
			              static_cast<unsigned long long>(e.size));
			return text;
		}
	}
	for (auto it = r.released_images.rbegin(); it != r.released_images.rend(); ++it) {
		if (it->image == image) {
			std::snprintf(text, sizeof(text), " va=0x%016llx+0x%llx (memory freed)",
			              static_cast<unsigned long long>(it->va), static_cast<unsigned long long>(it->size));
			return text;
		}
	}
	return " (no memory record)";
}

// bufferCache.cpp (KYTY_FAULT_MAP=1): BDA page-table entries pointing into [lo, hi).
void BdaShadowDescribe(uint64_t lo, uint64_t hi);

// Declared where it is used (diagnosticCheckpoints.cpp).
void DescribeDeviceAddress(uint64_t address, uint64_t precision, uint64_t now_tick);
void DescribeDeviceAddress(uint64_t address, uint64_t precision, uint64_t now_tick) {
	auto& r = Bda();
	std::lock_guard lock(r.mutex);
	const uint64_t lo = address & ~(precision > 0 ? precision - 1 : 0);
	const uint64_t hi = lo + (precision > 0 ? precision : 1);
	std::fprintf(stderr,
	             "[device-fault] address 0x%016" PRIx64 ": %zu live buffers with device addresses "
	             "(%" PRIu64 " created, %" PRIu64 " destroyed)\n",
	             address, r.live.size(), r.created_count, r.released_count);
	bool any = false;
	auto it  = r.live.upper_bound(hi);
	if (it != r.live.begin()) {
		auto below = std::prev(it);
		if (below->second.device + below->second.size > lo) {
			BdaPrint("in live buffer", below->second, address, now_tick);
			any = true;
		} else {
			BdaPrint("nearest live below", below->second, address, now_tick);
		}
	}
	if (it != r.live.end()) {
		BdaPrint("nearest live above", it->second, address, now_tick);
	}
	// Destroyed buffers that covered it, newest first.
	std::vector<const BdaEntry*> hits;
	for (const auto& e: r.released) {
		if (e.device < hi && e.device + e.size > lo) hits.push_back(&e);
	}
	std::sort(hits.begin(), hits.end(),
	          [](const BdaEntry* a, const BdaEntry* b) { return a->released > b->released; });
	for (size_t i = 0; i < hits.size() && i < 6; i++) {
		BdaPrint("in destroyed buffer", *hits[i], address, now_tick);
		any = true;
	}
	if (FaultMapEnabled()) {
		std::fprintf(stderr, "[device-fault]   images: %zu live mapped, %" PRIu64 " mapped, %" PRIu64
		                     " not mappable\n",
		             r.live_images.size(), r.images_mapped, r.images_unmapped);
		const BdaRegistry::Image* below_image = nullptr;
		const BdaRegistry::Image* above_image = nullptr;
		for (const auto& [key, e]: r.live_images) {
			if (e.va < hi && e.va + e.size > lo) {
				PrintImage("in live image", e, address);
				any = true;
			} else if (e.va + e.size <= lo) {
				if (below_image == nullptr || e.va > below_image->va) below_image = &e;
			} else if (e.va >= hi) {
				if (above_image == nullptr || e.va < above_image->va) above_image = &e;
			}
		}
		// v121: the live images either side, to compare with what Aftermath says lives there.
		if (below_image != nullptr) PrintImage("nearest live image below", *below_image, address);
		if (above_image != nullptr) PrintImage("nearest live image above", *above_image, address);
		std::vector<const BdaRegistry::Image*> image_hits;
		for (const auto& e: r.released_images) {
			if (e.va < hi && e.va + e.size > lo) image_hits.push_back(&e);
		}
		std::sort(image_hits.begin(), image_hits.end(),
		          [](const auto* a, const auto* b) { return a->released > b->released; });
		for (size_t i = 0; i < image_hits.size() && i < 6; i++) {
			PrintImage("in destroyed image", *image_hits[i], address);
			any = true;
		}
		// v122: what the host did with these images (bindless slots, set-0 binds, attachments).
		for (const auto& [key, e]: r.live_images) {
			if (e.va < hi && e.va + e.size > lo) {
				ImageHistory::Describe(static_cast<VkImage>(const_cast<void*>(e.image)), stderr);
			}
		}
		for (size_t i = 0; i < image_hits.size() && i < 3; i++) {
			ImageHistory::Describe(static_cast<VkImage>(const_cast<void*>(image_hits[i]->image)), stderr);
		}
		if (below_image != nullptr) {
			ImageHistory::Describe(static_cast<VkImage>(const_cast<void*>(below_image->image)), stderr);
		}
		if (above_image != nullptr) {
			ImageHistory::Describe(static_cast<VkImage>(const_cast<void*>(above_image->image)), stderr);
		}
	}
	if (!any) {
		std::fprintf(stderr, "[device-fault]   no buffer (live or among the last %zu destroyed) covers it\n",
		             r.released.size());
	}
	BdaShadowDescribe(lo, hi);
	std::fflush(stderr);
}

VmaPool BufferPoolFor(VmaAllocator allocator, const VkBufferCreateInfo& info,
                      const VmaAllocationCreateInfo& alloc_info);
void GpuPoolNoteAllocation(VmaAllocator allocator, VmaAllocation allocation); // vma.cpp

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;
	// KYTY_DMA_READBACK: readbacks copy from these on a transfer-only queue family.
	const std::array<uint32_t, 2> sharing_families {graphics.queue_family, graphics.transfer_family};
	if (graphics.transfer_family != static_cast<uint32_t>(-1)) {
		buffer_info.sharingMode           = vk::SharingMode::eConcurrent;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = sharing_families.data();
	}

	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	const VmaAllocationCreateFlags bda_flag =
	    with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	const auto        create        = [&](const VmaAllocationCreateInfo& create_info) {
        return static_cast<vk::Result>(vmaCreateBuffer(
            graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &create_info,
            &native_buffer, &m_allocation, &allocation_result));
	};
	vk::Result result = vk::Result::eErrorOutOfDeviceMemory;
	if (usage == MemoryUsage::DeviceLocal) {
		// KYTY_BUFFER_POOL_MB (vma.cpp): placed in a never-released pool instead of its own memory.
		if (const auto pool = BufferPoolFor(graphics.allocator,
		                                    *static_cast<const VkBufferCreateInfo*>(buffer_info),
		                                    allocation_info);
		    pool != VK_NULL_HANDLE) {
			auto pooled  = allocation_info;
			pooled.flags &= ~static_cast<VmaAllocationCreateFlags>(
			    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
			pooled.pool = pool;
			result      = create(pooled);
			if (result != vk::Result::eSuccess) {
				native_buffer = VK_NULL_HANDLE;
				m_allocation  = nullptr;
			} else {
				GpuPoolNoteAllocation(graphics.allocator, m_allocation);
			}
		}
	}
	if (result != vk::Result::eSuccess) {
		result = create(allocation_info);
	}
	if (result != vk::Result::eSuccess) {
		// The heap it prefers is out of budget: video memory, on a 4-8 GB card, for the game's
		// large buffers. As images do (GraphicContext::CreateImage), take system memory and pay in
		// bandwidth rather than stop; then the preferred heap past its budget, which the driver
		// may still serve by paging. Only when both fail is the session out of memory.
		auto spill           = allocation_info;
		spill.usage          = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		spill.preferredFlags = 0;
		const auto first     = result;
		result               = create(spill);
		const char* where    = "system memory";
		if (result != vk::Result::eSuccess) {
			auto over  = allocation_info;
			over.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
			result     = create(over);
			where      = "the preferred heap past its budget";
		}
		static std::atomic<uint32_t> spills {0};
		if (spills.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("Buffer spilled (%s): %" PRIu64 " MB, usage %d, first attempt %s -> %s\n", where,
			     size >> 20u, static_cast<int>(usage), vk::to_string(first).c_str(),
			     vk::to_string(result).c_str());
		}
	}
	if (result != vk::Result::eSuccess) {
		graphics.LogMemoryBudget();
		EXIT("Out of memory: a %" PRIu64 " MB GPU buffer fits neither in video memory nor in system "
		     "memory (%s). The VMA heap lines above give usage and budget. Close other programs; on "
		     "a card with little video memory lower the resolution; if Windows' commit limit is "
		     "reached, enlarge the page file.\n",
		     size >> 20u, vk::to_string(result).c_str());
	}

	m_buffer = native_buffer;
	StaleHooksNoteBuffer(native_buffer, size, with_bda ? "VMA, BDA" : "VMA", true);
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}
	BdaNoteCreated(m_device_address, size, cpu_address, usage, false, scheduler.CurrentTick());
	if (m_device_address == 0) {
		FaultMapNoteBuffer(graphics, this, m_allocation, size, cpu_address, usage,
		                   scheduler.CurrentTick(), true);
	}
	FaultMapNoteHandle(static_cast<VkBuffer>(m_buffer), this, m_device_address);

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t cpu_address,
               void* host_pointer, vk::BufferUsageFlags flags, uint64_t size, ImportHostMemory)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(MemoryUsage::DeviceLocal),
      m_cpu_address(cpu_address), m_size(size) {
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (!graphics.external_memory_host_enabled || d.vkGetMemoryHostPointerPropertiesEXT == nullptr ||
	    host_pointer == nullptr || size == 0) {
		return;
	}
	const auto device = static_cast<VkDevice>(graphics.device);
	VkMemoryHostPointerPropertiesEXT host_props {};
	host_props.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
	if (d.vkGetMemoryHostPointerPropertiesEXT(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	                                          host_pointer, &host_props) != VK_SUCCESS ||
	    host_props.memoryTypeBits == 0) {
		return;
	}
	VkExternalMemoryBufferCreateInfo external {};
	external.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.pNext       = &external;
	info.size        = size;
	info.usage       = static_cast<VkBufferUsageFlags>(flags);
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer buffer  = VK_NULL_HANDLE;
	if (d.vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) {
		return;
	}
	VkMemoryRequirements requirements {};
	d.vkGetBufferMemoryRequirements(device, buffer, &requirements);
	const auto& memory = graphics.physical_device_memory_properties;
	const auto  types  = requirements.memoryTypeBits & host_props.memoryTypeBits;
	uint32_t    type   = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
		if ((types & (1u << i)) != 0) {
			type = i;
			break;
		}
	}
	if (type == UINT32_MAX || requirements.size > size) {
		d.vkDestroyBuffer(device, buffer, nullptr);
		return;
	}
	VkImportMemoryHostPointerInfoEXT import {};
	import.sType        = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
	import.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	import.pHostPointer = host_pointer;
	VkMemoryAllocateFlagsInfo allocate_flags {};
	allocate_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	allocate_flags.pNext = &import;
	allocate_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.pNext           = &allocate_flags;
	allocate.allocationSize  = size;
	allocate.memoryTypeIndex = type;
	VkDeviceMemory device_memory = VK_NULL_HANDLE;
	if (d.vkAllocateMemory(device, &allocate, nullptr, &device_memory) != VK_SUCCESS) {
		d.vkDestroyBuffer(device, buffer, nullptr);
		return;
	}
	if (d.vkBindBufferMemory(device, buffer, device_memory, 0) != VK_SUCCESS) {
		d.vkFreeMemory(device, device_memory, nullptr);
		d.vkDestroyBuffer(device, buffer, nullptr);
		return;
	}
	m_buffer   = vk::Buffer(buffer);
	m_imported = device_memory;
	m_coherent = (static_cast<VkMemoryPropertyFlags>(memory.memoryTypes[type].propertyFlags) &
	              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (flags & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		BdaNoteCreated(m_device_address, size, cpu_address, MemoryUsage::DeviceLocal, true,
		               scheduler.CurrentTick());
		FaultMapNoteHandle(static_cast<VkBuffer>(m_buffer), nullptr, m_device_address);
	}
}

void Buffer::ReleaseImport() noexcept {
	if (m_imported == VK_NULL_HANDLE) {
		return;
	}
	const auto& d      = VULKAN_HPP_DEFAULT_DISPATCHER;
	BdaNoteReleased(m_device_address, m_scheduler->CurrentTick());
	FaultMapForgetHandle(static_cast<VkBuffer>(m_buffer));
	const auto  device = static_cast<VkDevice>(m_graphics->device);
	if (m_buffer != nullptr) {
		d.vkDestroyBuffer(device, static_cast<VkBuffer>(m_buffer), nullptr);
	}
	d.vkFreeMemory(device, m_imported, nullptr);
	m_buffer         = nullptr;
	m_imported       = VK_NULL_HANDLE;
	m_device_address = 0;
}

Buffer::~Buffer() {
	if (m_buffer != nullptr) {
		FaultMapForgetHandle(static_cast<VkBuffer>(m_buffer));
	}
	if (m_device_address == 0 && m_graphics != nullptr && m_scheduler != nullptr) {
		FaultMapNoteBuffer(*m_graphics, this, nullptr, 0, 0, MemoryUsage::DeviceLocal,
		                   m_scheduler->CurrentTick(), false);
	}
	if (m_device_address != 0 && m_scheduler != nullptr) {
		BdaNoteReleased(m_device_address, m_scheduler->CurrentTick());
	}
	if (m_imported != VK_NULL_HANDLE) {
		const auto& d      = VULKAN_HPP_DEFAULT_DISPATCHER;
		const auto  device = static_cast<VkDevice>(m_graphics->device);
		if (m_buffer != nullptr) {
			d.vkDestroyBuffer(device, static_cast<VkBuffer>(m_buffer), nullptr);
		}
		d.vkFreeMemory(device, m_imported, nullptr);
		return;
	}
	if (m_buffer != nullptr) {
		StaleHooksNoteBuffer(static_cast<VkBuffer>(m_buffer), m_size, nullptr, false);
		m_graphics->DestroyBufferMemory(static_cast<VkBuffer>(m_buffer), m_allocation);
	}
}

vk::DeviceAddress Buffer::BufferDeviceAddress() const noexcept {
	EXIT_IF(m_device_address == 0);
	return m_device_address;
}

bool Buffer::IsInBounds(uint64_t address, uint64_t size) const noexcept {
	return address >= m_cpu_address && size <= Size() && address - m_cpu_address <= Size() - size;
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	// Any host-mapped buffer (Download, Stream, Upload) may be invalidated: vmaInvalidateAllocation is
	// valid for every host-visible allocation. The device-loss audit reads Stream buffers this way.
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size, vk::AccessFlags source_before,
                      vk::AccessFlags destination_before, vk::AccessFlags source_after,
                      vk::AccessFlags destination_after) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, source_before, vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, destination_before, vk::AccessFlagBits::eTransferWrite),
	};
	const auto host_access  = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
	auto       before_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_before | destination_before) & host_access)) {
		before_stage |= vk::PipelineStageFlagBits::eHost;
	}
	const auto native = command.Recorder();
	native.pipelineBarrier(before_stage, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, source_after),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, destination_after),
	};
	auto after_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_after | destination_after) & host_access)) {
		after_stage |= vk::PipelineStageFlagBits::eHost;
	}
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, after_stage,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before =
	    Barrier(offset, size, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	            vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Recorder();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite,
	                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size)
    : Buffer(graphics, scheduler, usage, 0, AllFlags, size),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	return {Mapped().data() + m_offset, m_offset};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		if (!Scheduler().IsFree(watch.tick) && !allow_wait) {
			return false;
		}
		// Profiling-only zone: a stream-buffer ring wrap forcing the CPU to wait for the GPU to
		// finish with the memory it's about to reclaim. Scheduler().IsFree() above is the fast,
		// non-blocking path -- only reaching Wait() itself means the GPU genuinely hasn't caught
		// up yet.
		{
			KYTY_PROFILER_BLOCK("StreamBuffer::WaitPendingOperations (ring wrap wait)");
			// KYTY_FPS_LOG: [ring-wait] per 5 s, the waits and their time by ring size.
			static const bool log = Common::FpsLogDiagnostics();
			const auto        wait_start = std::chrono::steady_clock::now();
			Scheduler().Wait(watch.tick);
			if (log) {
				static std::mutex                                         mutex;
				static std::map<uint64_t, std::pair<uint64_t, double>>    waits;
				static auto last = std::chrono::steady_clock::now();
				const auto  now  = std::chrono::steady_clock::now();
				std::lock_guard lock(mutex);
				auto&           entry = waits[Size() >> 20u];
				entry.first++;
				entry.second += std::chrono::duration<double, std::milli>(now - wait_start).count();
				if (now - last > std::chrono::seconds(5)) {
					std::string text;
					for (const auto& [mib, stat]: waits) {
						char part[64];
						std::snprintf(part, sizeof(part), " %lluMB:%llux/%.0fms",
						              static_cast<unsigned long long>(mib),
						              static_cast<unsigned long long>(stat.first), stat.second);
						text += part;
					}
					std::fprintf(stderr, "[ring-wait] 5 s:%s\n", text.c_str());
					waits.clear();
					last = now;
				}
			}
			if (Usage() == MemoryUsage::Download) {
				Scheduler().WaitPriorityOperations(watch.tick);
			}
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
