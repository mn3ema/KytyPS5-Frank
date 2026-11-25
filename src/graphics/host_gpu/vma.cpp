#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/staleHooks.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <deque>
#include <optional>
#include <chrono>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	// Testing aid: KYTY_VRAM_LIMIT_MB=<n> caps every device-local heap at n MB, so a large card
	// behaves like a 4, 6 or 8 GB one: VMA reports the capped budget, allocations past it fail
	// or spill as they would there, and the caches' collection thresholds follow the budget.
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_limits {};
	if (const char* value = std::getenv("KYTY_VRAM_LIMIT_MB"); value != nullptr) {
		const auto  limit      = static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
		const auto& properties = GetPhysicalDeviceMemoryProperties();
		if (limit != 0) {
			for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
				const bool device_local = static_cast<bool>(properties.memoryHeaps[heap].flags &
				                                            vk::MemoryHeapFlagBits::eDeviceLocal);
				heap_limits[heap] = device_local ? std::min<VkDeviceSize>(
				                                       limit, properties.memoryHeaps[heap].size)
				                                 : VK_WHOLE_SIZE;
				if (device_local) {
					LOGF("KYTY_VRAM_LIMIT_MB: device heap %u capped at %" PRIu64 " MB (of %" PRIu64
					     " MB)\n",
					     heap, static_cast<uint64_t>(heap_limits[heap] >> 20u),
					     static_cast<uint64_t>(properties.memoryHeaps[heap].size >> 20u));
				}
			}
			info.pHeapSizeLimit = heap_limits.data();
		}
	}

	// KYTY_VMA_BLOCK_MB=<n> (v438zo; unset: the allocator's own 256 MB): the size of the blocks the
	// default pools are made of. An allocation over half a block gets memory of its own, and a
	// block goes back to the driver only when everything in it is freed. In Mac's play of 11-17 the
	// free space left inside 256 MB blocks grew from 0.6 GB at minute 10 to 1.9 GB at minute 22
	// (16.2 GB allocated in 18.1 GB of blocks), while the device sat at its ceiling: textures come
	// and go as he moves, and one live texture keeps its whole block. Smaller blocks empty sooner.
	// It changes where memory comes from, not what is in it. Drivers limit the number of memory
	// objects (maxMemoryAllocationCount; 4,096 is the usual value and was not read from this
	// device), so 16 is the floor: 18 GB of 16 MB blocks is ~1,150 of them, next to one per guest
	// buffer. [vma-frag] shows the result.
	if (const char* value = std::getenv("KYTY_VMA_BLOCK_MB"); value != nullptr) {
		const auto mb = std::strtoull(value, nullptr, 10);
		if (mb >= 16 && mb <= 1024) {
			info.preferredLargeHeapBlockSize = static_cast<VkDeviceSize>(mb) << 20u;
			std::fprintf(stderr, "[vma-frag] KYTY_VMA_BLOCK_MB: default pool blocks of %llu MB\n",
			             static_cast<unsigned long long>(mb));
		}
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}

	// A system whose commit limit is nearly used up stops the session later with "Out of memory"
	// (ISSUES #21); say so at start. KYTY_COMMIT_WARN_MB sets the threshold (default 16 GB).
	uint64_t commit_limit = 0;
	uint64_t commit_free  = 0;
	if (Common::VirtualMemory::QueryCommit(commit_limit, commit_free)) {
		const char*    value     = std::getenv("KYTY_COMMIT_WARN_MB");
		const uint64_t warn_mb   = value != nullptr ? std::strtoull(value, nullptr, 10) : 16384u;
		LOGF("Commit limit (start): %" PRIu64 " MB free of %" PRIu64 " MB\n", commit_free >> 20u,
		     commit_limit >> 20u);
		if ((commit_free >> 20u) < warn_mb) {
			std::fprintf(stderr,
			             "Warning: only %" PRIu64 " MB of the system's commit limit (%" PRIu64
			             " MB, RAM plus page file) are free; a game can need 16-24 GB. If the "
			             "emulator stops with \"Out of memory\", close other programs or enlarge "
			             "the page file.\n",
			             commit_free >> 20u, commit_limit >> 20u);
			LOGF("Warning: only %" PRIu64 " MB of the commit limit are free (threshold %" PRIu64
			     " MB)\n",
			     commit_free >> 20u, warn_mb);
		}
	}
	return true;
}

// KYTY_IMAGE_POOL_MB=<n> (v299, default off): render targets and storage images (the images the
// guest recreates every frame) are placed in a VMA pool of 256 MB blocks, n MB of them made at the
// first such image and never released, instead of one dedicated memory object each. A freed image
// then leaves its memory mapped, as freed memory always is on the console, so a GPU read through a
// reference the host has retired (Crash B: always a just-freed transient render target) returns old
// data instead of faulting the device. Also a test: if Crash B stops with it, the reads are of
// freed render-target memory. Images larger than a block, or with the pool full beyond its blocks,
// take the normal path.
namespace {
std::mutex                                        g_image_pool_mutex;
std::array<VmaPool, VK_MAX_MEMORY_TYPES>          g_image_pools {};
std::array<bool, VK_MAX_MEMORY_TYPES>             g_image_pool_failed {};
std::atomic<uint64_t>                             g_image_pool_hits {0}, g_image_pool_misses {0};

// KYTY_GPU_POOL=1 (v371, Crash B plan C1, default off): every image (sampled textures too) and
// every device-local buffer up to a block goes to the never-released pools (4096 MB of image
// blocks and 2048 MB of buffer blocks reserved unless KYTY_IMAGE_POOL_MB / KYTY_BUFFER_POOL_MB say
// otherwise), a block that would become empty keeps its last allocation parked so VMA never frees
// it, and whatever is not pooled (larger than a block, or host-visible) is never freed while the
// mode is on. So no memory the GPU ever saw is unmapped: if Crash B stops, every faulting read was
// of memory Kyty had freed; if it persists, the address was never mapped. [gpu-pool] every 10 s.
// v371 runs (11-09): mode 1 still faulted, inside a pool block that was alive, at images destroyed
// 11 s before: NVIDIA maps an image's (or buffer's) pages into the GPU address space when it is
// bound and unmaps them when the object is destroyed, whatever happens to the memory. And the
// never-freed large objects (2.7 GB buffers, re-created) reached 35 GB.
// KYTY_GPU_POOL=2 (v372): the same pools, but a pooled image or buffer is never destroyed: its
// suballocation goes back to the pool (the memory is reused, as on the console) while the object,
// and so its mapping, stays. Objects outside the pools are destroyed as usual.
uint32_t GpuPoolMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_GPU_POOL");
		return value != nullptr && (value[0] == '1' || value[0] == '2') ? static_cast<uint32_t>(value[0] - '0') : 0u;
	}();
	return mode;
}
bool GpuPoolEnabled() {
	return GpuPoolMode() != 0;
}

uint32_t ImagePoolMb() {
	static const uint32_t mb = [] {
		const char* value = std::getenv("KYTY_IMAGE_POOL_MB");
		const auto  set   = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
		return set != 0 ? set : (GpuPoolEnabled() ? 4096u : 0u);
	}();
	return mb;
}

constexpr VkDeviceSize ImagePoolBlockSize = VkDeviceSize {256} << 20u;

VmaPool ImagePoolFor(VmaAllocator allocator, const VkImageCreateInfo& info,
                     const VmaAllocationCreateInfo& alloc_info) {
	const auto mb = ImagePoolMb();
	if (mb == 0 || (!GpuPoolEnabled() && (info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
	                                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
	                                                    VK_IMAGE_USAGE_STORAGE_BIT)) == 0)) {
		return VK_NULL_HANDLE;
	}
	uint32_t type = 0;
	if (vmaFindMemoryTypeIndexForImageInfo(allocator, &info, &alloc_info, &type) != VK_SUCCESS ||
	    type >= VK_MAX_MEMORY_TYPES) {
		return VK_NULL_HANDLE;
	}
	std::scoped_lock lock(g_image_pool_mutex);
	if (g_image_pools[type] == VK_NULL_HANDLE && !g_image_pool_failed[type]) {
		VmaPoolCreateInfo pool_info {};
		pool_info.memoryTypeIndex = type;
		pool_info.blockSize       = ImagePoolBlockSize;
		pool_info.minBlockCount   = std::max<size_t>(1, (VkDeviceSize {mb} << 20u) / ImagePoolBlockSize);
		pool_info.maxBlockCount   = 0; // grows past the reserve if it must
		const auto result         = vmaCreatePool(allocator, &pool_info, &g_image_pools[type]);
		g_image_pool_failed[type] = result != VK_SUCCESS;
		std::fprintf(stderr, "[image-pool] memory type %u: %zu blocks of 256 MB reserved for render targets%s\n",
		             type, pool_info.minBlockCount, result == VK_SUCCESS ? "" : " - FAILED");
	}
	return g_image_pools[type];
}
std::array<VmaPool, VK_MAX_MEMORY_TYPES> g_buffer_pools {};
std::array<bool, VK_MAX_MEMORY_TYPES>    g_buffer_pool_failed {};
} // namespace

// KYTY_BUFFER_POOL_MB=<n> (v300, default off): device-local guest buffers up to 64 MiB go to a pool
// of 256 MB blocks, n MB of them made at the first such buffer and never released, instead of a
// dedicated memory object each (one vkAllocateMemory, ~100 us on Windows, per buffer; and a hole in
// the GPU's address space at every free). With KYTY_IMAGE_POOL_MB nothing the game frees leaves
// unmapped memory behind, as on the console. Returns VK_NULL_HANDLE when off or not eligible.
VmaPool BufferPoolFor(VmaAllocator allocator, const VkBufferCreateInfo& info,
                      const VmaAllocationCreateInfo& alloc_info) {
	static const uint32_t mb = [] {
		const char* value = std::getenv("KYTY_BUFFER_POOL_MB");
		const auto  set   = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
		return set != 0 ? set : (GpuPoolEnabled() ? 2048u : 0u);
	}();
	const VkDeviceSize limit = GpuPoolEnabled() ? ImagePoolBlockSize : (VkDeviceSize {64} << 20u);
	if (mb == 0 || info.size > limit) {
		return VK_NULL_HANDLE;
	}
	uint32_t type = 0;
	if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &info, &alloc_info, &type) != VK_SUCCESS ||
	    type >= VK_MAX_MEMORY_TYPES) {
		return VK_NULL_HANDLE;
	}
	std::scoped_lock lock(g_image_pool_mutex);
	if (g_buffer_pools[type] == VK_NULL_HANDLE && !g_buffer_pool_failed[type]) {
		VmaPoolCreateInfo pool_info {};
		pool_info.memoryTypeIndex = type;
		pool_info.blockSize       = ImagePoolBlockSize;
		pool_info.minBlockCount   = std::max<size_t>(1, (VkDeviceSize {mb} << 20u) / ImagePoolBlockSize);
		pool_info.maxBlockCount   = 0;
		const auto result          = vmaCreatePool(allocator, &pool_info, &g_buffer_pools[type]);
		g_buffer_pool_failed[type] = result != VK_SUCCESS;
		std::fprintf(stderr, "[buffer-pool] memory type %u: %zu blocks of 256 MB reserved for guest buffers%s\n",
		             type, pool_info.minBlockCount, result == VK_SUCCESS ? "" : " - FAILED");
	}
	return g_buffer_pools[type];
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	for (auto& pool: g_buffer_pools) {
		if (pool != VK_NULL_HANDLE) {
			vmaDestroyPool(allocator, pool);
			pool = VK_NULL_HANDLE;
		}
	}
	for (auto& pool: g_image_pools) {
		if (pool != VK_NULL_HANDLE) {
			vmaDestroyPool(allocator, pool);
			pool = VK_NULL_HANDLE;
		}
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

// The system's commit limit (Windows: RAM + page file): an allocation fails when it is used up,
// whatever the VMA budget says (ISSUES #21: one Wolverine session commits ~23 GB).
static void LogCommit(const char* when) {
	uint64_t limit     = 0;
	uint64_t available = 0;
	if (Common::VirtualMemory::QueryCommit(limit, available)) {
		LOGF("Commit limit (%s): %" PRIu64 " MB free of %" PRIu64 " MB\n", when, available >> 20u,
		     limit >> 20u);
	}
}

// The video memory the emulator plans for, whatever the card has (KYTY_VRAM_TARGET_MB; 0, the
// default, = the card's own budget; Senaxx's 14336 = what a 16 GB card leaves an application). On
// a 24 GB card the 14 GB target had the caches collecting from 90 % of it on: in the Trask
// compound (~19 GB in use) the buffer collector ran ~1,900 times per 5 s, waited for the GPU
// 1.3-1.6 s of every 5 and the readback ring ~1.9 s: 2-4 fps; with the card's budget 12-22 fps
// (2026-10-20). On cards with 16 GB or less the card's budget is the smaller one anyway.
static uint64_t VramTargetBytes() {
	static const uint64_t target = [] {
		const char* value = std::getenv("KYTY_VRAM_TARGET_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0}) << 20u;
	}();
	return target;
}

uint64_t GraphicContext::DeviceExcessOver(uint32_t percent) const {
	if (!CanReportMemoryUsage() || percent == 0 || percent >= 100) {
		return 0;
	}
	static std::atomic<int64_t>  next_ms {0};
	static std::atomic<uint64_t> excess {0};
	static std::atomic<uint32_t> cached_percent {0};
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto next = next_ms.load(std::memory_order_relaxed);
	if ((now >= next || cached_percent.load(std::memory_order_relaxed) != percent) &&
	    next_ms.compare_exchange_strong(next, now + 250, std::memory_order_relaxed)) {
		const auto budget = GetTotalMemoryBudget();
		auto       usage  = GetDeviceMemoryUsage();
		// v393: the free space inside the allocator's blocks is reused before any new block is
		// made, so it is not pressure. After a level unload (11-11, mac_v390_1: textures 6.7 ->
		// 2.1 GB, device use still 18.6 GB with ~6 GB free in blocks) counting it kept the trim
		// freeing live textures for nothing (60,000 passes in 20 s).
		if (allocator != nullptr) {
			VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
			vmaGetHeapBudgets(allocator, budgets);
			const bool discrete = physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
			uint64_t   slack    = 0;
			for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
				const bool device_local = static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
				                                            vk::MemoryHeapFlagBits::eDeviceLocal);
				const auto& statistics  = budgets[heap].statistics;
				if ((!discrete || device_local) && statistics.blockBytes > statistics.allocationBytes) {
					slack += statistics.blockBytes - statistics.allocationBytes;
				}
			}
			usage -= std::min(usage, slack);
		}
		const auto mark = budget / 100 * percent;
		excess.store(budget != 0 && usage > mark ? usage - mark : 0, std::memory_order_relaxed);
		cached_percent.store(percent, std::memory_order_relaxed);
	}
	return excess.load(std::memory_order_relaxed);
}

uint64_t GraphicContext::MemoryShortfall() const {
	// KYTY_COMMIT_RESERVE_MB (default 0 = off): collecting video memory for a commit shortfall
	// was tried on 2026-10-09 and withdrawn. On a PC whose commit limit is short for other reasons
	// (other programs, guest memory, host heaps) it never catches up: the collectors wrote back
	// and freed ~100 images per interval from 7 GB of device use on, the frame rate fell to 7 fps
	// and glyph atlases lost letters. The limit is also dynamic on a system-managed page file. The
	// start-up warning and the log lines remain; the video memory target is the pressure that counts.
	static const uint64_t reserve = [] {
		const char* value = std::getenv("KYTY_COMMIT_RESERVE_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0}) << 20u;
	}();
	static std::atomic<int64_t>  next_ms {0};
	static std::atomic<uint64_t> shortfall {0};
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto next = next_ms.load(std::memory_order_relaxed);
	if (now >= next && next_ms.compare_exchange_strong(next, now + 250, std::memory_order_relaxed)) {
		// The system's commit limit below the reserve.
		uint64_t limit     = 0;
		uint64_t available = 0;
		const uint64_t commit =
		    reserve != 0 && Common::VirtualMemory::QueryCommit(limit, available) &&
		            available < reserve
		        ? reserve - available
		        : 0;
		// Everything the device holds (images, buffers, the driver's own) past 90 % of the
		// budget, which the video memory target caps: the caches' own thresholds leave out the
		// bindless heap textures and render targets, so they alone cannot keep the total in it.
		uint64_t device = 0;
		if (CanReportMemoryUsage()) {
			const auto budget = GetTotalMemoryBudget();
			const auto usage  = GetDeviceMemoryUsage();
			const auto mark   = budget / 10 * 9;
			device            = budget != 0 && usage > mark ? usage - mark : 0;
		}
		const uint64_t value    = std::max(commit, device);
		const auto     previous = shortfall.exchange(value, std::memory_order_relaxed);
		static std::atomic<uint32_t> reported {0};
		if ((previous == 0) != (value == 0) && reported.fetch_add(1) < 64) {
			LOGF("Memory: %s (commit %" PRIu64 " MB free of %" PRIu64 " MB, reserve %" PRIu64
			     " MB; device %" PRIu64 " MB of %" PRIu64 " MB budget)\n",
			     value != 0 ? "short, the caches collect" : "room again", available >> 20u,
			     limit >> 20u, reserve >> 20u, GetDeviceMemoryUsage() >> 20u,
			     GetTotalMemoryBudget() >> 20u);
		}
	}
	return shortfall.load(std::memory_order_relaxed);
}

void GraphicContext::LogMemoryBudget() const {
	LogCommit("now");
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		const auto own = budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
		const auto target = VramTargetBytes();
		return target != 0 ? std::min(own, target) : own;
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

// streamBuffer.cpp (KYTY_FAULT_MAP=1): images by device address, to name GPU fault addresses.
void FaultMapNoteImage(GraphicContext& graphics, VmaAllocation allocation,
                       const vk::ImageCreateInfo& info, bool created, VkImage image);

void FaultMapNoteSparseImage(VkImage image, bool created); // streamBuffer.cpp

void GpuPoolNoteAllocation(VmaAllocator allocator, VmaAllocation allocation); // below

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	vk::Result       result       = vk::Result::eErrorOutOfDeviceMemory;
	if (const auto pool = ImagePoolFor(
	        allocator, *static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), alloc_info);
	    pool != VK_NULL_HANDLE) {
		auto pooled = alloc_info;
		pooled.pool = pool;
		result      = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &pooled,
            &native_image, &image.allocation, nullptr));
		if (result == vk::Result::eSuccess) {
			GpuPoolNoteAllocation(allocator, image.allocation);
		}
		const auto n = result == vk::Result::eSuccess ? g_image_pool_hits.fetch_add(1) + 1
		                                              : g_image_pool_misses.fetch_add(1) + 1;
		if (result != vk::Result::eSuccess && (n & (n - 1)) == 0) {
			std::fprintf(stderr, "[image-pool] %ux%ux%u format %d not pooled (%s), %llu so far\n",
			             image_info.extent.width, image_info.extent.height, image_info.arrayLayers,
			             static_cast<int>(image_info.format), vk::to_string(result).c_str(),
			             static_cast<unsigned long long>(n));
		}
	}
	if (result != vk::Result::eSuccess) {
		native_image = VK_NULL_HANDLE;
		result       = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
	}
	if (result != vk::Result::eSuccess) {
		// The device heap is full. The guest has a unified 16 GB pool, so the image is
		// real; place it in host memory and pay in bandwidth rather than fail.
		alloc_info.requiredFlags  = 0;
		alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		result                    = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
            &alloc_info, &native_image, &image.allocation, nullptr));
		static std::atomic<uint32_t> spill_count {0};
		if (const auto seen = spill_count.fetch_add(1, std::memory_order_relaxed); seen < 32) {
			LOGF("Image spilled to host memory (%u): %ux%ux%u layers=%u levels=%u format=%d -> %s\n",
			     seen + 1, image_info.extent.width, image_info.extent.height,
			     image_info.extent.depth, image_info.arrayLayers, image_info.mipLevels,
			     static_cast<int>(image_info.format), vk::to_string(result).c_str());
		}
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();
	FaultMapNoteImage(*this, image.allocation, image_info, true, image.image);
	StaleHooksNoteImage(image.image, &image_info, true);

	return true;
}

namespace {
// KYTY_FREE_DELAY_MS=<ms>: images and buffers keep their memory that long after they are
// destroyed. A diagnostic: if GPU faults in freed memory stop, something on the GPU still reads
// memory the emulator freed. KYTY_FREE_DELAY_IMAGES_MS=<ms>: the same for images only (buffers
// churn gigabytes a minute; images are a few GB per 2 minutes), to tell a read of a destroyed
// image's memory from a read of anything else: with it, a fault "in destroyed image" can only
// name an image destroyed at least that long ago.
uint32_t FreeDelayMs(bool image) {
	static const uint32_t ms = [] {
		const char* value = std::getenv("KYTY_FREE_DELAY_MS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static const uint32_t images_ms = [] {
		const char* value = std::getenv("KYTY_FREE_DELAY_IMAGES_MS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return image ? std::max(ms, images_ms) : ms;
}
// KYTY_FREE_KEEP_MB=<MB> (v377, the Crash B stopgap's cap, default 3072 when a delay is set; C13 peaked at 2746 MB with 3 s): the
// quarantine never holds more than this; past it the oldest entries are freed before their time.
uint64_t FreeKeepBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_FREE_KEEP_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 3072ull) << 20u;
	}();
	return bytes;
}
struct Quarantined {
	VkImage                               image      = VK_NULL_HANDLE;
	VkBuffer                              buffer     = VK_NULL_HANDLE;
	VmaAllocation                         allocation = nullptr;
	std::chrono::steady_clock::time_point due;
	uint64_t                              canary     = 0; // KYTY_FREE_CANARY batch that clears it
};

// KYTY_FREE_CANARY=1 (v377, Crash B; needs KYTY_FREE_DELAY_IMAGES_MS): an image entering the
// quarantine is cleared to the canary colour (0.0625, 0.9375, 0.3125, 1; depth 0.0625 / stencil 0x55) on the graphics queue, so a stale
// read of a destroyed image samples the canary instead of old contents and shows on screen. The
// GPU is done with the image when it is destroyed (DeferRelease waited for its last tick), so the
// clear needs no ordering with guest work; its memory is not freed before the clear has executed.
bool CanaryEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FREE_CANARY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

class FreeCanary {
public:
	// Records the clear of a dying image; returns the batch serial (0: not cleared).
	uint64_t Add(GraphicContext& graphics, const VulkanImage& image) {
		const auto name = vk::to_string(image.format);
		if (!(image.usage & vk::ImageUsageFlagBits::eTransferDst) || image.format == vk::Format::eUndefined ||
		    name.find("Block") != std::string::npos || name.find("Plane") != std::string::npos ||
		    name.find("Pvrtc") != std::string::npos) {
			m_skipped++;
			return 0;
		}
		const bool has_depth   = name.rfind("D16", 0) == 0 || name.rfind("D24", 0) == 0 ||
		                         name.rfind("D32", 0) == 0 || name.rfind("X8D24", 0) == 0;
		const bool has_stencil = name.find("S8Uint") != std::string::npos;
		std::scoped_lock lock(m_mutex);
		if (m_pool == nullptr) {
			vk::CommandPoolCreateInfo pool_info {};
			pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
			pool_info.queueFamilyIndex = graphics.queue_family;
			RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &m_pool),
			                     "create canary command pool");
			vk::CommandBufferAllocateInfo command_info {};
			command_info.commandPool        = m_pool;
			command_info.level              = vk::CommandBufferLevel::ePrimary;
			command_info.commandBufferCount = 1;
			for (auto& slot: m_slots) {
				RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&command_info, &slot.command),
				                     "allocate canary command buffer");
				vk::FenceCreateInfo fence_info {};
				fence_info.flags = vk::FenceCreateFlagBits::eSignaled;
				RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &slot.fence),
				                     "create canary fence");
			}
			std::fprintf(stderr, "[free-canary] on: destroyed images are cleared to magenta before their memory is held\n");
		}
		if (m_open == nullptr) {
			auto& slot = m_slots[m_next_slot];
			m_next_slot = (m_next_slot + 1) % m_slots.size();
			if (slot.serial != 0) {
				(void)graphics.device.waitForFences(1, &slot.fence, VK_TRUE, 2'000'000'000ull);
			}
			(void)graphics.device.resetFences(1, &slot.fence);
			slot.serial = ++m_serial;
			(void)slot.command.reset();
			vk::CommandBufferBeginInfo begin {};
			begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			RequireVulkanSuccess(slot.command.begin(&begin), "begin canary command buffer");
			m_open       = &slot;
			m_open_since = std::chrono::steady_clock::now();
			m_open_count = 0;
		}
		vk::ImageSubresourceRange range {};
		range.aspectMask = has_depth || has_stencil
		                       ? ((has_depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlags {}) |
		                          (has_stencil ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlags {}))
		                       : vk::ImageAspectFlagBits::eColor;
		range.levelCount = VK_REMAINING_MIP_LEVELS;
		range.layerCount = VK_REMAINING_ARRAY_LAYERS;
		vk::ImageMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.oldLayout           = vk::ImageLayout::eUndefined;
		barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image.image;
		barrier.subresourceRange    = range;
		auto& command = m_open->command;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
		                        nullptr, 0, nullptr, 1, &barrier);
		if (has_depth || has_stencil) {
			const vk::ClearDepthStencilValue value {0.0625f, 0x55u}; // v380: the detector's 1-channel signature
			command.clearDepthStencilImage(image.image, vk::ImageLayout::eTransferDstOptimal, &value, 1, &range);
		} else {
			vk::ClearColorValue value {};
			// v381: integer images take the float canary's bit patterns, which a float view of them
			// reads as the canary and KYTY_CANARY_DETECT matches in integer samples too.
			if (name.find("Uint") != std::string::npos) {
				value.uint32 = std::array<uint32_t, 4> {0x3d800000u, 0x3f700000u, 0x3ea00000u, 0x3f800000u};
			} else if (name.find("Sint") != std::string::npos) {
				value.int32 = std::array<int32_t, 4> {0x3d800000, 0x3f700000, 0x3ea00000, 0x3f800000};
			} else {
				// v379: an unusual value (exact in fp16 and fp11/fp10, within 0.002 in unorm8) that
				// KYTY_CANARY_DETECT can tell from real content; pure magenta is common in mask textures.
				// v380: alpha 1 (2-bit alpha formats cannot hold 0.5).
				value.float32 = std::array<float, 4> {0.0625f, 0.9375f, 0.3125f, 1.0f};
			}
			command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &value, 1, &range);
		}
		m_cleared++;
		const auto serial = m_open->serial;
		if (++m_open_count >= 32 || std::chrono::steady_clock::now() - m_open_since > std::chrono::milliseconds(30)) {
			SubmitLocked(graphics);
		}
		return serial;
	}

	// Before an image's memory is freed: its clear has been submitted and has executed.
	void EnsureDone(GraphicContext& graphics, uint64_t serial) {
		if (serial == 0) {
			return;
		}
		std::scoped_lock lock(m_mutex);
		if (m_open != nullptr && m_open->serial == serial) {
			SubmitLocked(graphics);
		}
		for (auto& slot: m_slots) {
			if (slot.serial == serial) {
				if (graphics.device.waitForFences(1, &slot.fence, VK_TRUE, 2'000'000'000ull) != vk::Result::eSuccess &&
				    m_timeouts++ < 4) {
					std::fprintf(stderr, "[free-canary] batch %" PRIu64 " not done after 2 s\n", serial);
				}
				return;
			}
		}
	}

	void Report() const {
		std::fprintf(stderr, "[free-canary] %" PRIu64 " images cleared, %" PRIu64 " skipped (no transfer-dst or compressed)\n",
		             m_cleared, m_skipped.load());
	}

private:
	struct Slot {
		vk::CommandBuffer command {};
		vk::Fence         fence {};
		uint64_t          serial = 0;
	};
	void SubmitLocked(GraphicContext& graphics) {
		auto* slot = m_open;
		m_open     = nullptr;
		(void)slot->command.end();
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &slot->command;
		vk::Result result {};
		{
			Common::LockGuard lock(graphics.queue_mutex);
			result = graphics.queue.submit(1, &submit, slot->fence);
		}
		// A device loss is reported by the scheduler's own submit and wait paths (with the
		// fault dump); here it only stops the canary.
		if (result != vk::Result::eSuccess && m_timeouts++ < 4) {
			std::fprintf(stderr, "[free-canary] submit failed: %s\n", vk::to_string(result).c_str());
		}
	}

	std::mutex                            m_mutex;
	vk::CommandPool                       m_pool = nullptr;
	std::array<Slot, 64>                  m_slots {};
	size_t                                m_next_slot  = 0;
	Slot*                                 m_open       = nullptr;
	std::chrono::steady_clock::time_point m_open_since {};
	uint32_t                              m_open_count = 0;
	uint64_t                              m_serial     = 0;
	uint64_t                              m_cleared    = 0;
	std::atomic<uint64_t>                 m_skipped    = 0;
	uint32_t                              m_timeouts   = 0;
};
FreeCanary g_free_canary;
GraphicContext* g_canary_graphics = nullptr;
std::mutex               g_quarantine_mutex;
std::deque<Quarantined>  g_quarantine;
uint64_t                 g_quarantine_bytes = 0;

void DestroyNow(VmaAllocator allocator, const Quarantined& entry) {
	if (entry.image != VK_NULL_HANDLE) {
		vmaDestroyImage(allocator, entry.image, entry.allocation);
	} else {
		vmaDestroyBuffer(allocator, entry.buffer, entry.allocation);
	}
}

// KYTY_GPU_POOL bookkeeping: per pool block, the live allocations and the parked one.
struct PoolBlock {
	uint32_t                   live = 0;
	std::optional<Quarantined> parked;
};
std::mutex                                     g_gpu_pool_mutex;
std::unordered_map<VkDeviceMemory, PoolBlock>  g_gpu_pool_blocks;
std::unordered_map<VmaAllocation, VkDeviceMemory> g_gpu_pooled;
uint64_t g_gpu_pool_kept_count = 0, g_gpu_pool_kept_bytes = 0, g_gpu_pool_parked = 0;
uint64_t g_gpu_pool_kept_objects = 0;

// A pooled object's end: mode 1 destroys it with its suballocation; mode 2 frees only the
// suballocation and keeps the object (its GPU mapping stays).
void ReleasePooled(VmaAllocator allocator, const Quarantined& entry) {
	if (GpuPoolMode() == 2) {
		vmaFreeMemory(allocator, entry.allocation);
		g_gpu_pool_kept_objects++;
		return;
	}
	DestroyNow(allocator, entry);
}

void ReportGpuPool() {
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(10)) {
		return;
	}
	last = now;
	std::fprintf(stderr,
	             "[gpu-pool] %zu blocks (%zu MB), %zu pooled allocations live, %" PRIu64
	             " parked; never freed: %" PRIu64 " objects, %" PRIu64 " MB; objects kept past their memory: %" PRIu64 "\n",
	             g_gpu_pool_blocks.size(), g_gpu_pool_blocks.size() * (ImagePoolBlockSize >> 20u),
	             g_gpu_pooled.size(), g_gpu_pool_parked, g_gpu_pool_kept_count,
	             g_gpu_pool_kept_bytes >> 20u, g_gpu_pool_kept_objects);
}

} // namespace

// Called after every successful pooled vmaCreateImage/vmaCreateBuffer (KYTY_GPU_POOL).
void GpuPoolNoteAllocation(VmaAllocator allocator, VmaAllocation allocation) {
	if (!GpuPoolEnabled() || allocation == nullptr) {
		return;
	}
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	std::optional<Quarantined> release;
	{
		std::scoped_lock lock(g_gpu_pool_mutex);
		g_gpu_pooled[allocation] = info.deviceMemory;
		auto& block              = g_gpu_pool_blocks[info.deviceMemory];
		block.live++;
		if (block.parked) {
			// The block holds a live allocation again: the parked one can go.
			release = std::move(block.parked);
			block.parked.reset();
			g_gpu_pool_parked--;
		}
	}
	if (release) {
		std::scoped_lock lock(g_gpu_pool_mutex);
		ReleasePooled(allocator, *release);
	}
}

namespace {

void FreeNow(VmaAllocator allocator, const Quarantined& entry) {
	if (!GpuPoolEnabled()) {
		DestroyNow(allocator, entry);
		return;
	}
	std::scoped_lock lock(g_gpu_pool_mutex);
	ReportGpuPool();
	const auto pooled = g_gpu_pooled.find(entry.allocation);
	if (pooled == g_gpu_pooled.end() && GpuPoolMode() == 2) {
		DestroyNow(allocator, entry);
		return;
	}
	if (pooled == g_gpu_pooled.end()) {
		// Not in a pool: kept (memory and object) for the rest of the session.
		VmaAllocationInfo info {};
		vmaGetAllocationInfo(allocator, entry.allocation, &info);
		g_gpu_pool_kept_count++;
		g_gpu_pool_kept_bytes += info.size;
		return;
	}
	auto& block = g_gpu_pool_blocks[pooled->second];
	g_gpu_pooled.erase(pooled);
	if (block.live > 0) {
		block.live--;
	}
	if (block.live == 0 && !block.parked) {
		// Its last allocation: parked, so VMA does not free the empty block.
		block.parked = entry;
		g_gpu_pool_parked++;
		return;
	}
	ReleasePooled(allocator, entry);
}

void Quarantine(VmaAllocator allocator, VkImage image, VkBuffer buffer, VmaAllocation allocation,
                uint64_t canary = 0) {
	const auto delay = FreeDelayMs(image != VK_NULL_HANDLE);
	if (delay == 0) {
		FreeNow(allocator, {image, buffer, allocation, {}});
		return;
	}
	const auto       now = std::chrono::steady_clock::now();
	std::scoped_lock lock(g_quarantine_mutex);
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	g_quarantine_bytes += info.size;
	g_quarantine.push_back({image, buffer, allocation, now + std::chrono::milliseconds(delay), canary});
	static uint64_t early = 0;
	while (!g_quarantine.empty() && (g_quarantine.front().due <= now || g_quarantine_bytes > FreeKeepBytes())) {
		early += g_quarantine.front().due > now ? 1u : 0u;
		VmaAllocationInfo front {};
		vmaGetAllocationInfo(allocator, g_quarantine.front().allocation, &front);
		g_quarantine_bytes -= std::min<uint64_t>(g_quarantine_bytes, front.size);
		if (g_quarantine.front().canary != 0 && g_canary_graphics != nullptr) {
			g_free_canary.EnsureDone(*g_canary_graphics, g_quarantine.front().canary);
		}
		FreeNow(allocator, g_quarantine.front());
		g_quarantine.pop_front();
	}
	static auto last_report = now;
	if (now - last_report > std::chrono::seconds(10)) {
		last_report = now;
		std::fprintf(stderr, "[free-delay] %zu allocations, %llu MB held (cap %llu MB, %llu freed early by the cap)\n",
		             g_quarantine.size(), static_cast<unsigned long long>(g_quarantine_bytes >> 20u),
		             static_cast<unsigned long long>(FreeKeepBytes() >> 20u), static_cast<unsigned long long>(early));
		if (CanaryEnabled()) {
			g_free_canary.Report();
		}
	}
}
} // namespace

void GraphicContext::DestroyBufferMemory(VkBuffer buffer, VmaAllocation allocation) {
	Quarantine(allocator, VK_NULL_HANDLE, buffer, allocation);
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	FaultMapNoteImage(*this, image.allocation, {}, false, image.image);
	StaleHooksNoteImage(image.image, nullptr, false);
	uint64_t canary = 0;
	if (CanaryEnabled() && FreeDelayMs(true) != 0) {
		g_canary_graphics = this;
		canary            = g_free_canary.Add(*this, image);
	}
	Quarantine(allocator, image.image, VK_NULL_HANDLE, image.allocation, canary);
	image.image      = nullptr;
	image.allocation = nullptr;
}

// v438zy: the mode when KYTY_TEX_SPARSE is not set (vulkanWindow.cpp sets it once, before the
// device is made: KYTY_TEX_SPARSE_NVIDIA on NVIDIA GPUs, else off).
static std::atomic<uint32_t> g_tex_sparse_default {0};

void SetTexSparseDefault(uint32_t mode) {
	g_tex_sparse_default.store(mode <= 2 ? mode : 0u, std::memory_order_relaxed);
}

uint32_t TexSparseMode() {
	static const int env = [] {
		const char* value = std::getenv("KYTY_TEX_SPARSE");
		if (value == nullptr || value[0] == '\0') {
			return -1;
		}
		return value[0] == '1' || value[0] == '2' ? value[0] - '0' : 0;
	}();
	return env >= 0 ? static_cast<uint32_t>(env) : g_tex_sparse_default.load(std::memory_order_relaxed);
}

bool TexSparseVerify() {
	static const bool verify = [] {
		const char* value = std::getenv("KYTY_TEX_SPARSE_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return verify;
}

// v438zr (KYTY_TEX_SPARSE). The image keeps its size and all its levels; memory is bound level
// by level. Outside the allocator's image path: no pool, no quarantine, no fault map.
bool GraphicContext::CreateSparseImage(const vk::ImageCreateInfo& image_info, VulkanImage& image,
                                       SparseImageState& state) {
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);
	if (!sparse_images || image_info.arrayLayers != 1 || image_info.imageType != vk::ImageType::e2D ||
	    image_info.samples != vk::SampleCountFlagBits::e1 || image_info.mipLevels < 2) {
		return false;
	}
	auto info   = image_info;
	info.flags |= vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency;
	vk::Image native = nullptr;
	if (device.createImage(&info, nullptr, &native) != vk::Result::eSuccess) {
		return false;
	}
	vk::MemoryRequirements memory {};
	device.getImageMemoryRequirements(native, &memory);
	uint32_t count = 0;
	device.getImageSparseMemoryRequirements(native, &count, nullptr);
	std::vector<vk::SparseImageMemoryRequirements> sparse(count);
	if (count != 0) {
		device.getImageSparseMemoryRequirements(native, &count, sparse.data());
	}
	const vk::SparseImageMemoryRequirements* color = nullptr;
	for (uint32_t i = 0; i < count; i++) {
		if (sparse[i].formatProperties.aspectMask & vk::ImageAspectFlagBits::eColor) {
			color = &sparse[i];
			break;
		}
	}
	// One colour aspect, blocks of whole texels, a tail the driver sizes: anything else stays whole.
	if (color == nullptr || memory.alignment == 0 || color->formatProperties.imageGranularity.width == 0 ||
	    color->formatProperties.imageGranularity.height == 0 || color->imageMipTailSize % memory.alignment != 0 ||
	    static_cast<bool>(color->formatProperties.flags & vk::SparseImageFormatFlagBits::eNonstandardBlockSize)) {
		device.destroyImage(native, nullptr);
		return false;
	}
	state                  = {};
	state.sparse           = true;
	state.levels           = image_info.mipLevels;
	state.tail_first       = std::min(color->imageMipTailFirstLod, image_info.mipLevels);
	state.granularity      = color->formatProperties.imageGranularity;
	state.block_bytes      = memory.alignment;
	state.tail_offset      = color->imageMipTailOffset;
	state.tail_size        = color->imageMipTailSize;
	state.memory_type_bits = memory.memoryTypeBits;

	image.image      = native;
	image.allocation = nullptr;
	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();
	FaultMapNoteSparseImage(static_cast<VkImage>(image.image), true);
	return true;
}

bool GraphicContext::BindSparseLevels(VulkanImage& image, SparseImageState& state, uint32_t first) {
	EXIT_IF(!state.sparse || image.image == nullptr);
	first                 = std::min(first, state.tail_first);
	const uint32_t before = std::min(state.bound_first, state.tail_first);
	if (first >= before && state.tail_bound) {
		return true;
	}
	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	std::vector<vk::SparseImageMemoryBind> binds;
	std::vector<VmaAllocation>             made;
	uint64_t                               bytes = 0;
	const auto allocate = [&](uint64_t size, vk::DeviceMemory& out_memory, vk::DeviceSize& out_offset) {
		VkMemoryRequirements requirements {};
		requirements.size           = size;
		requirements.alignment      = state.block_bytes;
		requirements.memoryTypeBits = state.memory_type_bits;
		VmaAllocation     allocation = nullptr;
		VmaAllocationInfo placed {};
		if (vmaAllocateMemory(allocator, &requirements, &alloc_info, &allocation, &placed) != VK_SUCCESS) {
			return false;
		}
		made.push_back(allocation);
		out_memory = placed.deviceMemory;
		out_offset = placed.offset;
		bytes     += size;
		return true;
	};
	bool ok = true;
	for (uint32_t level = first; level < before && ok; level++) {
		const uint32_t width  = std::max(image.extent.width >> level, 1u);
		const uint32_t height = std::max(image.extent.height >> level, 1u);
		const uint64_t blocks = static_cast<uint64_t>((width + state.granularity.width - 1) / state.granularity.width) *
		                        ((height + state.granularity.height - 1) / state.granularity.height);
		vk::SparseImageMemoryBind bind {};
		bind.subresource = vk::ImageSubresource {vk::ImageAspectFlagBits::eColor, level, 0};
		bind.offset      = vk::Offset3D {0, 0, 0};
		bind.extent      = vk::Extent3D {width, height, 1};
		ok               = allocate(blocks * state.block_bytes, bind.memory, bind.memoryOffset);
		binds.push_back(bind);
	}
	vk::SparseMemoryBind tail {};
	const bool           with_tail = !state.tail_bound && state.tail_size != 0;
	if (ok && with_tail) {
		tail.resourceOffset = state.tail_offset;
		tail.size           = state.tail_size;
		ok                  = allocate(state.tail_size, tail.memory, tail.memoryOffset);
	}
	vk::Fence fence = nullptr;
	if (ok) {
		vk::SparseImageMemoryBindInfo image_bind {};
		image_bind.image     = image.image;
		image_bind.bindCount = static_cast<uint32_t>(binds.size());
		image_bind.pBinds    = binds.data();
		vk::SparseImageOpaqueMemoryBindInfo opaque_bind {};
		opaque_bind.image     = image.image;
		opaque_bind.bindCount = 1;
		opaque_bind.pBinds    = &tail;
		vk::BindSparseInfo bind_info {};
		bind_info.imageBindCount       = binds.empty() ? 0u : 1u;
		bind_info.pImageBinds          = &image_bind;
		bind_info.imageOpaqueBindCount = with_tail ? 1u : 0u;
		bind_info.pImageOpaqueBinds    = &opaque_bind;
		const vk::FenceCreateInfo fence_info {};
		ok = device.createFence(&fence_info, nullptr, &fence) == vk::Result::eSuccess;
		if (ok) {
			// The same queue as the submits, under its mutex; the caller goes on only when the
			// device has bound the memory, so every command recorded after this sees it bound.
			vk::Result result;
			{
				Common::LockGuard lock(queue_mutex);
				result = queue.bindSparse(1, &bind_info, fence);
			}
			static const bool wait_here = [] {
				const char* value = std::getenv("KYTY_TEX_SPARSE_WAIT");
				return value != nullptr && value[0] == '1';
			}();
			ok = result == vk::Result::eSuccess;
			if (ok && !wait_here) {
				std::scoped_lock lock(sparse_fence_mutex);
				sparse_fences.push_back(fence);
				fence = nullptr; // WaitSparseBinds waits for it and destroys it
			} else if (ok) {
				ok = device.waitForFences(1, &fence, VK_TRUE, UINT64_C(10'000'000'000)) == vk::Result::eSuccess;
			}
		}
	}
	if (fence) {
		device.destroyFence(fence, nullptr);
	}
	if (!ok) {
		for (const auto allocation: made) {
			vmaFreeMemory(allocator, allocation);
		}
		return false;
	}
	state.allocations.insert(state.allocations.end(), made.begin(), made.end());
	state.bound_first  = std::min(state.bound_first, first);
	state.tail_bound   = state.tail_bound || with_tail || state.tail_size == 0;
	state.bound_bytes += bytes;
	state.binds++;
	return true;
}

void GraphicContext::WaitSparseBinds() {
	std::vector<vk::Fence> fences;
	{
		std::scoped_lock lock(sparse_fence_mutex);
		if (sparse_fences.empty()) {
			return;
		}
		fences.swap(sparse_fences);
	}
	// A bind waits behind the queue's work, so a GPU hang ends here as often as in a tick's wait
	// (MasterSemaphore::Wait). v443e: as there, the hang is reported after 5 s ([gpu-hang]), and the
	// device loss it ends in is reported ([device-fault], what the GPU was in) before this exits;
	// the fatal used to come first, at 10 s, and the run ended without either (g443c).
	const auto begin  = std::chrono::steady_clock::now();
	const auto waited = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(); };
	auto result = device.waitForFences(static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE, UINT64_C(5'000'000'000));
	for (uint32_t reports = 0; result == vk::Result::eTimeout && reports < 3; reports++) {
		DumpGpuHangCheckpoints(*this, 0, waited());
		result = device.waitForFences(static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE, UINT64_C(15'000'000'000));
	}
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(*this);
		}
		EXIT("sparse texture memory was not bound: %s after %.1f s (%zu binds)\n", vk::to_string(result).c_str(), waited(),
		     fences.size());
	}
	for (const auto fence: fences) {
		device.destroyFence(fence, nullptr);
	}
}

void GraphicContext::DeleteSparseImage(VulkanImage& image, SparseImageState& state) {
	EXIT_IF(allocator == nullptr || !state.sparse || image.image == nullptr);
	WaitSparseBinds(); // no bind of this image may still be on its way
	FaultMapNoteSparseImage(static_cast<VkImage>(image.image), false);
	device.destroyImage(image.image, nullptr);
	for (const auto allocation: state.allocations) {
		vmaFreeMemory(allocator, allocation);
	}
	image.image = nullptr;
	state       = {};
}

std::string GraphicContext::DescribeFragmentation() const {
	std::string text;
	if (allocator == nullptr || physical_device == nullptr) {
		return text;
	}
	const auto&        properties = GetPhysicalDeviceMemoryProperties();
	VmaTotalStatistics total {};
	vmaCalculateStatistics(allocator, &total);
	char line[256];
	for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
		const auto& type = total.memoryType[i];
		if (type.statistics.blockCount == 0) {
			continue;
		}
		const auto free_bytes = type.statistics.blockBytes - type.statistics.allocationBytes;
		std::snprintf(line, sizeof(line),
		              " | type %u (heap %u): %u blocks %" PRIu64 " MB, %u allocations %" PRIu64 " MB (largest %" PRIu64
		              " MB), free inside blocks %" PRIu64 " MB in %u ranges (largest %" PRIu64 " MB)",
		              i, properties.memoryTypes[i].heapIndex, type.statistics.blockCount,
		              static_cast<uint64_t>(type.statistics.blockBytes) >> 20u, type.statistics.allocationCount,
		              static_cast<uint64_t>(type.statistics.allocationBytes) >> 20u,
		              type.statistics.allocationCount != 0 ? static_cast<uint64_t>(type.allocationSizeMax) >> 20u : 0,
		              static_cast<uint64_t>(free_bytes) >> 20u, type.unusedRangeCount,
		              type.unusedRangeCount != 0 ? static_cast<uint64_t>(type.unusedRangeSizeMax) >> 20u : 0);
		text += line;
	}
	return text;
}

uint64_t GraphicContext::ImageAllocationBytes(const VulkanImage& image) const {
	if (allocator == nullptr || image.allocation == nullptr) {
		return 0;
	}
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, image.allocation, &info);
	return info.size;
}

std::string GraphicContext::DescribeDeviceMemory() const {
	std::string text;
	if (allocator == nullptr || physical_device == nullptr) {
		return text;
	}
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	char line[192];
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		std::snprintf(line, sizeof(line), " heap%u%s use %" PRIu64 "/%" PRIu64 " MB (vma %" PRIu64 " in %" PRIu64 " MB blocks, %u allocs)", i,
		              static_cast<bool>(properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? "(local)" : "",
		              static_cast<uint64_t>(budgets[i].usage) >> 20u, static_cast<uint64_t>(budgets[i].budget) >> 20u,
		              static_cast<uint64_t>(budgets[i].statistics.allocationBytes) >> 20u,
		              static_cast<uint64_t>(budgets[i].statistics.blockBytes) >> 20u, budgets[i].statistics.allocationCount);
		text += line;
	}
	{
		std::scoped_lock lock(g_quarantine_mutex);
		std::snprintf(line, sizeof(line), " | quarantine %zu (%" PRIu64 " MB)", g_quarantine.size(), g_quarantine_bytes >> 20u);
		text += line;
	}
	return text;
}

} // namespace Libs::Graphics
