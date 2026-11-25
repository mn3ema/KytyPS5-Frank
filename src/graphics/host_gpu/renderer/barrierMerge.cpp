#include "graphics/host_gpu/renderer/barrierMerge.h"

#include "graphics/host_gpu/renderer/barrierSink.h"

#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics {

namespace {

thread_local bool t_merge_thread = false;

bool MergeConfigured() {
	static const bool configured = [] {
		// Default on since v438c (fps plan step 2, 11-15: GPU -1.2 ms a frame in the fight with replay
		// on; a full barrier right behind another adds nothing). KYTY_BARRIER_MERGE=0: off.
		const char* value = std::getenv("KYTY_BARRIER_MERGE");
		return (value == nullptr || value[0] != '0') ||
		       (SyncToggle::PeriodSeconds() != 0 &&
		        (SyncToggle::ModeMask() & SyncToggle::ModeBarrierMerge) != 0);
	}();
	return configured;
}

bool MergeActive() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_BARRIER_MERGE");
		const bool  toggled = SyncToggle::PeriodSeconds() != 0 && (SyncToggle::ModeMask() & SyncToggle::ModeBarrierMerge) != 0;
		return !toggled && (value == nullptr || value[0] != '0');
	}();
	return always || SyncToggle::State().load(std::memory_order_relaxed);
}

// Per command buffer: the handle, with bit 0 set while the last recorded command was a full
// barrier. Direct-mapped; a slot another command buffer took reads as "not behind a full barrier"
// (nothing is left out), and a command buffer is recorded by one thread at a time.
constexpr size_t                       SlotCount = 1024;
std::array<std::atomic<uint64_t>, SlotCount> g_slots {};

inline std::atomic<uint64_t>& SlotOf(VkCommandBuffer command) {
	const auto value = reinterpret_cast<uint64_t>(command);
	return g_slots[((value >> 4u) ^ (value >> 14u)) & (SlotCount - 1u)];
}
inline void NoteCommand(VkCommandBuffer command) {
	SlotOf(command).store(reinterpret_cast<uint64_t>(command), std::memory_order_relaxed);
}
inline bool BehindFullBarrier(VkCommandBuffer command) {
	return SlotOf(command).load(std::memory_order_relaxed) == (reinterpret_cast<uint64_t>(command) | 1u);
}
inline void NoteBarrier(VkCommandBuffer command, bool full) {
	SlotOf(command).store(reinterpret_cast<uint64_t>(command) | (full ? 1u : 0u),
	                      std::memory_order_relaxed);
}

std::atomic<uint64_t> g_seen {0}, g_full {0}, g_dropped {0}, g_coalesced {0};

void Report() {
	// The clock is read once every 4096 barriers.
	static std::atomic<uint32_t> calls {0};
	if ((calls.fetch_add(1, std::memory_order_relaxed) & 4095u) != 0) return;
	static std::atomic<int64_t> last {0};
	const auto now  = std::chrono::steady_clock::now().time_since_epoch().count();
	auto       prev = last.load(std::memory_order_relaxed);
	if (now - prev < std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(5)).count() ||
	    !last.compare_exchange_strong(prev, now)) {
		return;
	}
	const auto seen = g_seen.exchange(0), full = g_full.exchange(0), dropped = g_dropped.exchange(0);
	const auto coalesced = g_coalesced.exchange(0);
	if (prev != 0) {
		const double s = static_cast<double>(now - prev) / static_cast<double>(std::chrono::steady_clock::period::den);
		// coalesced: barrier calls saved by KYTY_BARRIER_COALESCE (the barriers of a group beyond its first).
		std::fprintf(stderr,
		             "[barrier-merge] %.1f s: %.0f/s barriers, %.0f/s full, %.0f/s left out, %.0f/s coalesced\n", s,
		             static_cast<double>(seen) / s, static_cast<double>(full) / s,
		             static_cast<double>(dropped) / s, static_cast<double>(coalesced) / s);
	}
}

constexpr VkPipelineStageFlags2 AllCommands  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
constexpr VkAccessFlags2        MemoryWrite  = VK_ACCESS_2_MEMORY_WRITE_BIT;
constexpr VkAccessFlags2        MemoryRW     = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

bool FullMemory(VkPipelineStageFlags2 src, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst,
                VkAccessFlags2 dst_access) {
	return (src & AllCommands) != 0 && (dst & AllCommands) != 0 && (src_access & MemoryWrite) != 0 &&
	       (dst_access & MemoryRW) == MemoryRW;
}

// Host stages and accesses are outside ALL_COMMANDS: a barrier that makes device writes visible to
// the host (readback) or orders host accesses is never left out.
constexpr uint64_t HostStage  = VK_PIPELINE_STAGE_2_HOST_BIT;
constexpr uint64_t HostAccess = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;

bool NoHost(VkPipelineStageFlags src, VkPipelineStageFlags dst, uint32_t memory_count,
            const VkMemoryBarrier* memory, uint32_t buffer_count, const VkBufferMemoryBarrier* buffers) {
	if (((src | dst) & HostStage) != 0) return false;
	for (uint32_t i = 0; i < memory_count; i++) {
		if (((memory[i].srcAccessMask | memory[i].dstAccessMask) & HostAccess) != 0) return false;
	}
	for (uint32_t i = 0; i < buffer_count; i++) {
		if (((buffers[i].srcAccessMask | buffers[i].dstAccessMask) & HostAccess) != 0) return false;
	}
	return true;
}

bool NoHost2(const VkDependencyInfo* info) {
	for (uint32_t i = 0; i < info->memoryBarrierCount; i++) {
		const auto& m = info->pMemoryBarriers[i];
		if (((m.srcStageMask | m.dstStageMask) & HostStage) != 0 ||
		    ((m.srcAccessMask | m.dstAccessMask) & HostAccess) != 0) {
			return false;
		}
	}
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
		const auto& b = info->pBufferMemoryBarriers[i];
		if (((b.srcStageMask | b.dstStageMask) & HostStage) != 0 ||
		    ((b.srcAccessMask | b.dstAccessMask) & HostAccess) != 0) {
			return false;
		}
	}
	return true;
}

template <typename BufferBarrier>
bool NoQueueTransfer(uint32_t count, const BufferBarrier* barriers) {
	for (uint32_t i = 0; i < count; i++) {
		if (barriers[i].srcQueueFamilyIndex != barriers[i].dstQueueFamilyIndex &&
		    barriers[i].srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED &&
		    barriers[i].dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) {
			return false;
		}
	}
	return true;
}

PFN_vkCmdPipelineBarrier  g_barrier      = nullptr;
PFN_vkCmdPipelineBarrier2 g_barrier2     = nullptr;
PFN_vkCmdPipelineBarrier2 g_barrier2_khr = nullptr;

// ---- Coalescing (barrierMerge.h) ----

bool CoalesceConfigured() {
	static const bool configured = [] {
		const char* value = std::getenv("KYTY_BARRIER_COALESCE");
		const bool  wanted = (value != nullptr && value[0] == '1') ||
		                     (SyncToggle::PeriodSeconds() != 0 &&
		                      (SyncToggle::ModeMask() & SyncToggle::ModeBarrierCoalesce) != 0);
		return wanted && !BarrierSinkEnabled();
	}();
	return configured;
}

bool CoalesceActive() {
	static const bool toggled = SyncToggle::PeriodSeconds() != 0 &&
	                            (SyncToggle::ModeMask() & SyncToggle::ModeBarrierCoalesce) != 0;
	if (!CoalesceConfigured()) return false;
	return !toggled || SyncToggle::State().load(std::memory_order_relaxed);
}

// The group of barriers held back for one command buffer. A command buffer is recorded by one
// thread at a time (the recording thread, or the GPU thread after it drained the recording), so
// its group is only ever touched in recording order; the mutex covers two command buffers that
// share a slot from different threads.
struct PendingGroup {
	static constexpr uint32_t MaxImages = 16;
	VkCommandBuffer       command    = VK_NULL_HANDLE;
	bool                  active     = false;
	bool                  has_memory = false;
	VkPipelineStageFlags2 src_stage  = 0;
	VkPipelineStageFlags2 dst_stage  = 0;
	VkAccessFlags2        src_access = 0;
	VkAccessFlags2        dst_access = 0;
	uint32_t              image_count = 0;
	uint32_t              calls       = 0;
	std::array<VkImageMemoryBarrier2, MaxImages> images {};
};
constexpr size_t                           PendingSlotCount = 64;
std::mutex                                 g_pending_mutex;
std::array<PendingGroup, PendingSlotCount> g_pending {};
std::atomic<uint32_t>                      g_pending_active {0}; // groups held back right now

inline PendingGroup& PendingOf(VkCommandBuffer command) {
	const auto value = reinterpret_cast<uint64_t>(command);
	return g_pending[((value >> 4u) ^ (value >> 10u) ^ (value >> 16u)) & (PendingSlotCount - 1u)];
}

// g_pending_mutex held. Records the group as one barrier.
void RecordGroup(PendingGroup& group) {
	if (!group.active) return;
	VkMemoryBarrier2 memory {};
	memory.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	memory.srcStageMask  = group.src_stage;
	memory.srcAccessMask = group.src_access;
	memory.dstStageMask  = group.dst_stage;
	memory.dstAccessMask = group.dst_access;
	VkDependencyInfo info {};
	info.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	info.memoryBarrierCount      = group.has_memory ? 1u : 0u;
	info.pMemoryBarriers         = group.has_memory ? &memory : nullptr;
	info.imageMemoryBarrierCount = group.image_count;
	info.pImageMemoryBarriers    = group.image_count != 0 ? group.images.data() : nullptr;
	(g_barrier2 != nullptr ? g_barrier2 : g_barrier2_khr)(group.command, &info);
	if (group.calls > 1) {
		g_coalesced.fetch_add(group.calls - 1u, std::memory_order_relaxed);
	}
	group.active      = false;
	group.has_memory  = false;
	group.src_stage   = 0;
	group.dst_stage   = 0;
	group.src_access  = 0;
	group.dst_access  = 0;
	group.image_count = 0;
	group.calls       = 0;
	g_pending_active.fetch_sub(1, std::memory_order_release);
}

// g_pending_mutex held.
inline void Activate(PendingGroup& group) {
	if (!group.active) {
		group.active = true;
		g_pending_active.fetch_add(1, std::memory_order_release);
	}
}
inline void AddMemory(PendingGroup& group, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                      VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
	Activate(group);
	group.has_memory = true;
	group.src_stage |= src_stage;
	group.src_access |= src_access;
	group.dst_stage |= dst_stage;
	group.dst_access |= dst_access;
}
inline void AddImage(PendingGroup& group, const VkImageMemoryBarrier2& image) {
	if (image.oldLayout == image.newLayout) {
		// No transition: a memory dependency on that image, which the global barrier covers.
		AddMemory(group, image.srcStageMask, image.srcAccessMask, image.dstStageMask, image.dstAccessMask);
		return;
	}
	bool repeated = group.image_count == PendingGroup::MaxImages;
	for (uint32_t i = 0; i < group.image_count && !repeated; i++) {
		repeated = group.images[i].image == image.image;
	}
	if (repeated) {
		// A second transition of the same image (or a full list): the group so far goes first.
		RecordGroup(group);
	}
	Activate(group);
	group.images[group.image_count++] = image;
}

// Before a command that is not a barrier (or a barrier that is recorded at once): the command
// buffer's group is recorded first. discard: the command buffer starts over (begin, reset).
inline void RecordPendingBefore(VkCommandBuffer command, bool discard) {
	if (g_pending_active.load(std::memory_order_acquire) == 0) return;
	std::lock_guard lock(g_pending_mutex);
	auto&           group = PendingOf(command);
	if (group.command != command || !group.active) return;
	if (discard) {
		group.active      = false;
		group.has_memory  = false;
		group.src_stage   = 0;
		group.dst_stage   = 0;
		group.src_access  = 0;
		group.dst_access  = 0;
		group.image_count = 0;
		group.calls       = 0;
		g_pending_active.fetch_sub(1, std::memory_order_release);
		return;
	}
	RecordGroup(group);
}

constexpr uint64_t HostStageBit   = VK_PIPELINE_STAGE_2_HOST_BIT;
constexpr uint64_t HostAccessBits = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;

// The slot of this command buffer for a new barrier, or nullptr when another command buffer's
// group holds it (the barrier is then recorded at once; this command buffer has no group).
// g_pending_mutex held.
inline PendingGroup* ClaimGroup(VkCommandBuffer command) {
	auto& group = PendingOf(command);
	if (group.command != command) {
		if (group.active) return nullptr;
		group.command = command;
	}
	return &group;
}

// True: the barrier joined its command buffer's group and must not be recorded now. False: the
// group (if any) was recorded and the caller records this barrier itself.
bool Coalesce1(VkCommandBuffer command, VkPipelineStageFlags src, VkPipelineStageFlags dst,
               VkDependencyFlags flags, uint32_t memory_count, const VkMemoryBarrier* memory,
               uint32_t buffer_count, const VkBufferMemoryBarrier* buffers, uint32_t image_count,
               const VkImageMemoryBarrier* images) {
	bool eligible = t_merge_thread && flags == 0 && CoalesceActive() &&
	                NoQueueTransfer(buffer_count, buffers) && NoQueueTransfer(image_count, images) &&
	                NoHost(src, dst, memory_count, memory, buffer_count, buffers);
	for (uint32_t i = 0; i < memory_count && eligible; i++) {
		eligible = memory[i].pNext == nullptr;
	}
	for (uint32_t i = 0; i < buffer_count && eligible; i++) {
		eligible = buffers[i].pNext == nullptr;
	}
	for (uint32_t i = 0; i < image_count && eligible; i++) {
		eligible = images[i].pNext == nullptr &&
		           ((images[i].srcAccessMask | images[i].dstAccessMask) & HostAccessBits) == 0;
	}
	std::lock_guard lock(g_pending_mutex);
	auto*           group = ClaimGroup(command);
	if (group == nullptr) return false;
	if (!eligible) {
		RecordGroup(*group);
		return false;
	}
	if (memory_count == 0 && buffer_count == 0 && image_count == 0) {
		AddMemory(*group, src, 0, dst, 0); // an execution dependency only
	}
	for (uint32_t i = 0; i < memory_count; i++) {
		AddMemory(*group, src, memory[i].srcAccessMask, dst, memory[i].dstAccessMask);
	}
	for (uint32_t i = 0; i < buffer_count; i++) {
		AddMemory(*group, src, buffers[i].srcAccessMask, dst, buffers[i].dstAccessMask);
	}
	for (uint32_t i = 0; i < image_count; i++) {
		VkImageMemoryBarrier2 image {};
		image.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		image.srcStageMask        = src;
		image.srcAccessMask       = images[i].srcAccessMask;
		image.dstStageMask        = dst;
		image.dstAccessMask       = images[i].dstAccessMask;
		image.oldLayout           = images[i].oldLayout;
		image.newLayout           = images[i].newLayout;
		image.srcQueueFamilyIndex = images[i].srcQueueFamilyIndex;
		image.dstQueueFamilyIndex = images[i].dstQueueFamilyIndex;
		image.image               = images[i].image;
		image.subresourceRange    = images[i].subresourceRange;
		AddImage(*group, image);
	}
	group->calls++;
	return true;
}

bool Coalesce2(VkCommandBuffer command, const VkDependencyInfo* info) {
	bool eligible = t_merge_thread && info->pNext == nullptr && info->dependencyFlags == 0 &&
	                CoalesceActive() &&
	                NoQueueTransfer(info->bufferMemoryBarrierCount, info->pBufferMemoryBarriers) &&
	                NoQueueTransfer(info->imageMemoryBarrierCount, info->pImageMemoryBarriers) &&
	                NoHost2(info);
	for (uint32_t i = 0; i < info->memoryBarrierCount && eligible; i++) {
		eligible = info->pMemoryBarriers[i].pNext == nullptr;
	}
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount && eligible; i++) {
		eligible = info->pBufferMemoryBarriers[i].pNext == nullptr;
	}
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount && eligible; i++) {
		const auto& image = info->pImageMemoryBarriers[i];
		eligible = image.pNext == nullptr &&
		           ((image.srcStageMask | image.dstStageMask) & HostStageBit) == 0 &&
		           ((image.srcAccessMask | image.dstAccessMask) & HostAccessBits) == 0;
	}
	std::lock_guard lock(g_pending_mutex);
	auto*           group = ClaimGroup(command);
	if (group == nullptr) return false;
	if (!eligible) {
		RecordGroup(*group);
		return false;
	}
	if (info->memoryBarrierCount == 0 && info->bufferMemoryBarrierCount == 0 &&
	    info->imageMemoryBarrierCount == 0) {
		return true; // an empty dependency orders nothing
	}
	for (uint32_t i = 0; i < info->memoryBarrierCount; i++) {
		const auto& m = info->pMemoryBarriers[i];
		AddMemory(*group, m.srcStageMask, m.srcAccessMask, m.dstStageMask, m.dstAccessMask);
	}
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
		const auto& b = info->pBufferMemoryBarriers[i];
		AddMemory(*group, b.srcStageMask, b.srcAccessMask, b.dstStageMask, b.dstAccessMask);
	}
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
		AddImage(*group, info->pImageMemoryBarriers[i]);
	}
	group->calls++;
	return true;
}

VKAPI_ATTR void VKAPI_CALL HookBarrier(VkCommandBuffer command, VkPipelineStageFlags src,
                                       VkPipelineStageFlags dst, VkDependencyFlags flags,
                                       uint32_t memory_count, const VkMemoryBarrier* memory,
                                       uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                       uint32_t image_count, const VkImageMemoryBarrier* images) {
	g_seen.fetch_add(1, std::memory_order_relaxed);
	if (t_merge_thread && image_count == 0 && BehindFullBarrier(command) &&
	    NoQueueTransfer(buffer_count, buffers) &&
	    NoHost(src, dst, memory_count, memory, buffer_count, buffers) && MergeActive()) {
		g_dropped.fetch_add(1, std::memory_order_relaxed);
		Report();
		return; // the state stays "behind a full barrier"
	}
	bool full = false;
	if (image_count == 0 && (flags & VK_DEPENDENCY_BY_REGION_BIT) == 0) {
		for (uint32_t i = 0; i < memory_count && !full; i++) {
			full = FullMemory(src, memory[i].srcAccessMask, dst, memory[i].dstAccessMask);
		}
	}
	g_full.fetch_add(full ? 1 : 0, std::memory_order_relaxed);
	if (!CoalesceConfigured() || !Coalesce1(command, src, dst, flags, memory_count, memory, buffer_count,
	                                        buffers, image_count, images)) {
		g_barrier(command, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count,
		          images);
	}
	NoteBarrier(command, full);
	Report();
}

template <PFN_vkCmdPipelineBarrier2* Original>
VKAPI_ATTR void VKAPI_CALL HookBarrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
	g_seen.fetch_add(1, std::memory_order_relaxed);
	if (t_merge_thread && info->imageMemoryBarrierCount == 0 && BehindFullBarrier(command) &&
	    NoQueueTransfer(info->bufferMemoryBarrierCount, info->pBufferMemoryBarriers) &&
	    NoHost2(info) && MergeActive()) {
		g_dropped.fetch_add(1, std::memory_order_relaxed);
		Report();
		return;
	}
	bool full = false;
	if (info->imageMemoryBarrierCount == 0 && (info->dependencyFlags & VK_DEPENDENCY_BY_REGION_BIT) == 0) {
		for (uint32_t i = 0; i < info->memoryBarrierCount && !full; i++) {
			const auto& m = info->pMemoryBarriers[i];
			full = FullMemory(m.srcStageMask, m.srcAccessMask, m.dstStageMask, m.dstAccessMask);
		}
	}
	g_full.fetch_add(full ? 1 : 0, std::memory_order_relaxed);
	if (!CoalesceConfigured() || !Coalesce2(command, info)) {
		(*Original)(command, info);
	}
	NoteBarrier(command, full);
	Report();
}

// Every other hooked command clears the state of its command buffer.
template <int Id, typename F>
struct Hook;
template <int Id, typename R, typename... A>
struct Hook<Id, R(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	using Fn                  = R(VKAPI_PTR*)(VkCommandBuffer, A...);
	static inline Fn original = nullptr;
	static R VKAPI_PTR Call(VkCommandBuffer command, A... args) {
		// KYTY_BARRIER_COALESCE: the barriers held back for this command buffer come first
		// (hooks 0 and 2, begin and reset: the command buffer starts over, nothing to record).
		RecordPendingBefore(command, Id == 0 || Id == 2);
		NoteCommand(command);
		return original(command, args...);
	}
	static void Install(Fn& slot) {
		if (slot != nullptr && slot != &Call) {
			original = slot;
			slot     = &Call;
		}
	}
};

} // namespace

void SetBarrierMergeThread() noexcept {
	t_merge_thread = true;
}

void NoteUnhookedCommands(vk::CommandBuffer command) {
	const auto handle = static_cast<VkCommandBuffer>(command);
	RecordPendingBefore(handle, false);
	NoteCommand(handle);
}

void InstallBarrierMerge() {
	if (!MergeConfigured() && !CoalesceConfigured()) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkCmdPipelineBarrier == nullptr || d.vkCmdPipelineBarrier2 == nullptr) {
		std::fprintf(stderr, "[barrier-merge] missing entry points: off\n");
		return;
	}
	g_barrier              = d.vkCmdPipelineBarrier;
	d.vkCmdPipelineBarrier = HookBarrier;
	g_barrier2              = d.vkCmdPipelineBarrier2;
	d.vkCmdPipelineBarrier2 = HookBarrier2<&g_barrier2>;
	if (d.vkCmdPipelineBarrier2KHR != nullptr) {
		g_barrier2_khr             = d.vkCmdPipelineBarrier2KHR;
		d.vkCmdPipelineBarrier2KHR = HookBarrier2<&g_barrier2_khr>;
	}
#define KYTY_MERGE_HOOK(n, name) Hook<n, decltype(d.name)>::Install(d.name)
	KYTY_MERGE_HOOK(0, vkBeginCommandBuffer);
	KYTY_MERGE_HOOK(1, vkEndCommandBuffer);
	KYTY_MERGE_HOOK(2, vkResetCommandBuffer);
	KYTY_MERGE_HOOK(3, vkCmdDraw);
	KYTY_MERGE_HOOK(4, vkCmdDrawIndexed);
	KYTY_MERGE_HOOK(5, vkCmdDrawIndirect);
	KYTY_MERGE_HOOK(6, vkCmdDrawIndexedIndirect);
	KYTY_MERGE_HOOK(7, vkCmdDrawIndirectCount);
	KYTY_MERGE_HOOK(8, vkCmdDrawIndexedIndirectCount);
	KYTY_MERGE_HOOK(9, vkCmdDrawIndirectCountKHR);
	KYTY_MERGE_HOOK(10, vkCmdDrawIndexedIndirectCountKHR);
	KYTY_MERGE_HOOK(11, vkCmdDrawMeshTasksEXT);
	KYTY_MERGE_HOOK(12, vkCmdDrawMeshTasksIndirectEXT);
	KYTY_MERGE_HOOK(13, vkCmdDrawMeshTasksIndirectCountEXT);
	KYTY_MERGE_HOOK(14, vkCmdDispatch);
	KYTY_MERGE_HOOK(15, vkCmdDispatchBase);
	KYTY_MERGE_HOOK(16, vkCmdDispatchIndirect);
	KYTY_MERGE_HOOK(17, vkCmdCopyBuffer);
	KYTY_MERGE_HOOK(18, vkCmdCopyBuffer2);
	KYTY_MERGE_HOOK(19, vkCmdCopyBuffer2KHR);
	KYTY_MERGE_HOOK(20, vkCmdCopyImage);
	KYTY_MERGE_HOOK(21, vkCmdCopyImage2);
	KYTY_MERGE_HOOK(22, vkCmdCopyImage2KHR);
	KYTY_MERGE_HOOK(23, vkCmdCopyBufferToImage);
	KYTY_MERGE_HOOK(24, vkCmdCopyBufferToImage2);
	KYTY_MERGE_HOOK(25, vkCmdCopyBufferToImage2KHR);
	KYTY_MERGE_HOOK(26, vkCmdCopyImageToBuffer);
	KYTY_MERGE_HOOK(27, vkCmdCopyImageToBuffer2);
	KYTY_MERGE_HOOK(28, vkCmdCopyImageToBuffer2KHR);
	KYTY_MERGE_HOOK(29, vkCmdBlitImage);
	KYTY_MERGE_HOOK(30, vkCmdBlitImage2);
	KYTY_MERGE_HOOK(31, vkCmdBlitImage2KHR);
	KYTY_MERGE_HOOK(32, vkCmdResolveImage);
	KYTY_MERGE_HOOK(33, vkCmdResolveImage2);
	KYTY_MERGE_HOOK(34, vkCmdResolveImage2KHR);
	KYTY_MERGE_HOOK(35, vkCmdFillBuffer);
	KYTY_MERGE_HOOK(36, vkCmdUpdateBuffer);
	KYTY_MERGE_HOOK(37, vkCmdClearColorImage);
	KYTY_MERGE_HOOK(38, vkCmdClearDepthStencilImage);
	KYTY_MERGE_HOOK(39, vkCmdClearAttachments);
	KYTY_MERGE_HOOK(40, vkCmdBeginRendering);
	KYTY_MERGE_HOOK(41, vkCmdBeginRenderingKHR);
	KYTY_MERGE_HOOK(42, vkCmdEndRendering);
	KYTY_MERGE_HOOK(43, vkCmdEndRenderingKHR);
	KYTY_MERGE_HOOK(44, vkCmdBeginRenderPass);
	KYTY_MERGE_HOOK(45, vkCmdBeginRenderPass2);
	KYTY_MERGE_HOOK(46, vkCmdNextSubpass);
	KYTY_MERGE_HOOK(47, vkCmdNextSubpass2);
	KYTY_MERGE_HOOK(48, vkCmdEndRenderPass);
	KYTY_MERGE_HOOK(49, vkCmdEndRenderPass2);
	KYTY_MERGE_HOOK(50, vkCmdWriteTimestamp);
	KYTY_MERGE_HOOK(51, vkCmdWriteTimestamp2);
	KYTY_MERGE_HOOK(52, vkCmdWriteTimestamp2KHR);
	KYTY_MERGE_HOOK(53, vkCmdResetQueryPool);
	KYTY_MERGE_HOOK(54, vkCmdBeginQuery);
	KYTY_MERGE_HOOK(55, vkCmdEndQuery);
	KYTY_MERGE_HOOK(56, vkCmdCopyQueryPoolResults);
	KYTY_MERGE_HOOK(57, vkCmdSetEvent);
	KYTY_MERGE_HOOK(58, vkCmdSetEvent2);
	KYTY_MERGE_HOOK(59, vkCmdResetEvent);
	KYTY_MERGE_HOOK(60, vkCmdResetEvent2);
	KYTY_MERGE_HOOK(61, vkCmdWaitEvents);
	KYTY_MERGE_HOOK(62, vkCmdWaitEvents2);
	KYTY_MERGE_HOOK(63, vkCmdExecuteCommands);
	KYTY_MERGE_HOOK(64, vkCmdWriteBufferMarkerAMD);
	KYTY_MERGE_HOOK(65, vkCmdWriteBufferMarker2AMD);
	KYTY_MERGE_HOOK(66, vkCmdSetCheckpointNV);
	KYTY_MERGE_HOOK(67, vkCmdBeginConditionalRenderingEXT);
	KYTY_MERGE_HOOK(68, vkCmdEndConditionalRenderingEXT);
	KYTY_MERGE_HOOK(69, vkCmdSetEvent2KHR);
	KYTY_MERGE_HOOK(70, vkCmdResetEvent2KHR);
	KYTY_MERGE_HOOK(71, vkCmdWaitEvents2KHR);
#undef KYTY_MERGE_HOOK
	std::fprintf(stderr, "[barrier-merge] barriers behind a full barrier are left out%s\n",
	             SyncToggle::PeriodSeconds() != 0 ? " (in the toggle's on windows)" : "");
	if (CoalesceConfigured()) {
		std::fprintf(stderr, "[barrier-merge] back-to-back barriers are recorded as one (KYTY_BARRIER_COALESCE)\n");
	}
}

} // namespace Libs::Graphics
