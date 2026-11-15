#include "graphics/host_gpu/renderer/barrierMerge.h"

#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

std::atomic<uint64_t> g_seen {0}, g_full {0}, g_dropped {0};

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
	if (prev != 0) {
		const double s = static_cast<double>(now - prev) / static_cast<double>(std::chrono::steady_clock::period::den);
		std::fprintf(stderr, "[barrier-merge] %.1f s: %.0f/s barriers, %.0f/s full, %.0f/s left out\n", s,
		             static_cast<double>(seen) / s, static_cast<double>(full) / s,
		             static_cast<double>(dropped) / s);
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
	g_barrier(command, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count, images);
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
	(*Original)(command, info);
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

void InstallBarrierMerge() {
	if (!MergeConfigured()) {
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
}

} // namespace Libs::Graphics
