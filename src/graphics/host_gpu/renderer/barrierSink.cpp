#include "graphics/host_gpu/renderer/barrierSink.h"
#include "common/fpsLog.h"

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

namespace {

constexpr uint32_t MaxColors = 8;

struct CommandState {
	bool rendering = false;
	bool sinkable  = false; // the active instance's begin info was copied (can be restarted)
	bool pending   = false;
	bool epoch     = false; // see barrierSink.h (a)
	uint32_t instances = 0; // rendering instances begun since the epoch's full barrier
	bool     khr_begin = false;
	VkMemoryBarrier2 barrier {};
	VkRenderingInfo  info {};
	std::array<VkRenderingAttachmentInfo, MaxColors> colors {};
	VkRenderingAttachmentInfo depth {};
	VkRenderingAttachmentInfo stencil {};
};

thread_local bool            t_next_draw_safe    = false; // renderer thread
thread_local bool            t_command_draw_safe = false; // recording thread
thread_local bool            t_command_guest     = false;
thread_local VkCommandBuffer t_cached_command    = VK_NULL_HANDLE;
thread_local CommandState*   t_cached_state      = nullptr;

std::mutex                                                         g_states_mutex;
std::unordered_map<VkCommandBuffer, std::unique_ptr<CommandState>> g_states;

std::atomic<uint64_t> g_sunk {0}, g_restarts {0}, g_flushed_at_end {0}, g_guest {0};
std::atomic<uint64_t> g_flush_unsafe {0}, g_flush_epoch {0}, g_flush_other {0}, g_guest_outside {0},
    g_guest_no_epoch {0}, g_draws_safe {0}, g_draws_unsafe {0};

CommandState& State(VkCommandBuffer command) {
	if (command == t_cached_command && t_cached_state != nullptr) {
		return *t_cached_state;
	}
	std::scoped_lock lock(g_states_mutex);
	auto&            state = g_states[command];
	if (state == nullptr) {
		state = std::make_unique<CommandState>();
	}
	t_cached_command = command;
	t_cached_state   = state.get();
	return *state;
}

PFN_vkBeginCommandBuffer   g_begin_command   = nullptr;
PFN_vkEndCommandBuffer     g_end_command     = nullptr;
PFN_vkCmdBeginRendering    g_begin_rendering = nullptr;
PFN_vkCmdBeginRendering    g_begin_rendering_khr = nullptr;
PFN_vkCmdEndRendering      g_end_rendering   = nullptr;
PFN_vkCmdEndRendering      g_end_rendering_khr = nullptr;
PFN_vkCmdPipelineBarrier   g_barrier         = nullptr;
PFN_vkCmdPipelineBarrier2  g_barrier2        = nullptr;
PFN_vkCmdPipelineBarrier2  g_barrier2_khr    = nullptr;

constexpr VkAccessFlags2 ReadWrite = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

bool IsFull(const VkMemoryBarrier2& b) {
	return (b.srcStageMask & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) != 0 &&
	       (b.srcAccessMask & VK_ACCESS_2_MEMORY_WRITE_BIT) != 0 &&
	       (b.dstStageMask & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) != 0 &&
	       (b.dstAccessMask & ReadWrite) == ReadWrite;
}

bool IsFull(const VkDependencyInfo& d) {
	for (uint32_t i = 0; i < d.memoryBarrierCount; i++) {
		if (IsFull(d.pMemoryBarriers[i])) {
			return true;
		}
	}
	return false;
}

bool IsFull1(VkPipelineStageFlags src, VkPipelineStageFlags dst, uint32_t count,
             const VkMemoryBarrier* barriers) {
	if ((src & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) == 0 ||
	    (dst & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) == 0) {
		return false;
	}
	for (uint32_t i = 0; i < count; i++) {
		if ((barriers[i].srcAccessMask & VK_ACCESS_MEMORY_WRITE_BIT) != 0 &&
		    (barriers[i].dstAccessMask & (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)) ==
		        (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)) {
			return true;
		}
	}
	return false;
}

void Barrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
	(g_barrier2 != nullptr ? g_barrier2 : g_barrier2_khr)(command, info);
}

void NoteFull(CommandState& s) {
	s.epoch     = true;
	s.instances = 0;
}

// Records the pending barrier (outside rendering).
void RecordPending(VkCommandBuffer command, CommandState& s) {
	VkDependencyInfo dependency {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &s.barrier;
	Barrier2(command, &dependency);
	s.pending = false;
	s.barrier = {};
	NoteFull(s);
}

// Inside a restartable instance with a barrier pending: end it, record the barrier, begin it
// again with every attachment loaded (what earlier draws wrote is kept).
void Restart(VkCommandBuffer command, CommandState& s) {
	(s.khr_begin ? g_end_rendering_khr : g_end_rendering)(command);
	RecordPending(command, s);
	VkRenderingInfo info = s.info;
	std::array<VkRenderingAttachmentInfo, MaxColors> colors = s.colors;
	VkRenderingAttachmentInfo depth = s.depth, stencil = s.stencil;
	for (uint32_t i = 0; i < info.colorAttachmentCount; i++) {
		if (colors[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) colors[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	}
	if (depth.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	if (stencil.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) stencil.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	info.pColorAttachments  = info.colorAttachmentCount != 0 ? colors.data() : nullptr;
	info.pDepthAttachment   = s.info.pDepthAttachment != nullptr ? &depth : nullptr;
	info.pStencilAttachment = s.info.pStencilAttachment != nullptr ? &stencil : nullptr;
	(s.khr_begin ? g_begin_rendering_khr : g_begin_rendering)(command, &info);
	s.instances = 1;
	g_restarts.fetch_add(1, std::memory_order_relaxed);
}

// Before a command that must not pass the pending barrier.
void Flush(VkCommandBuffer command, CommandState& s) {
	if (!s.pending) {
		return;
	}
	if (s.rendering) {
		Restart(command, s);
	} else {
		RecordPending(command, s);
	}
}

void MergePending(CommandState& s, const VkMemoryBarrier2& b) {
	if (!s.pending) {
		s.barrier       = b;
		s.barrier.pNext = nullptr;
		s.pending       = true;
		return;
	}
	s.barrier.srcStageMask |= b.srcStageMask;
	s.barrier.srcAccessMask |= b.srcAccessMask;
	s.barrier.dstStageMask |= b.dstStageMask;
	s.barrier.dstAccessMask |= b.dstAccessMask;
}

void Report() {
	static const bool log = Common::FpsLogDiagnostics();
	if (!log) return;
	static std::atomic<int64_t> last {0};
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto prev = last.load(std::memory_order_relaxed);
	if (prev == 0) {
		last.compare_exchange_strong(prev, now);
		return;
	}
	if (now - prev < 5000 || !last.compare_exchange_strong(prev, now)) return;
	const double s = static_cast<double>(now - prev) / 1000.0;
	std::fprintf(stderr,
	             "[barrier-sink] guest barriers %.0f/s (outside rendering %.0f, inside without epoch "
	             "%.0f), kept pending past a draw %.0f/s, rendering restarted %.0f/s (unsafe draw "
	             "%.0f, epoch %.0f, other %.0f), recorded at the end of rendering %.0f/s; draws safe "
	             "%.0f/s unsafe %.0f/s\n",
	             static_cast<double>(g_guest.exchange(0)) / s,
	             static_cast<double>(g_guest_outside.exchange(0)) / s,
	             static_cast<double>(g_guest_no_epoch.exchange(0)) / s,
	             static_cast<double>(g_sunk.exchange(0)) / s,
	             static_cast<double>(g_restarts.exchange(0)) / s,
	             static_cast<double>(g_flush_unsafe.exchange(0)) / s,
	             static_cast<double>(g_flush_epoch.exchange(0)) / s,
	             static_cast<double>(g_flush_other.exchange(0)) / s,
	             static_cast<double>(g_flushed_at_end.exchange(0)) / s,
	             static_cast<double>(g_draws_safe.exchange(0)) / s,
	             static_cast<double>(g_draws_unsafe.exchange(0)) / s);
}

// --- hooks ---

VKAPI_ATTR VkResult VKAPI_CALL HookBeginCommand(VkCommandBuffer command,
                                                const VkCommandBufferBeginInfo* info) {
	auto& s = State(command);
	s       = {};
	return g_begin_command(command, info);
}

VKAPI_ATTR VkResult VKAPI_CALL HookEndCommand(VkCommandBuffer command) {
	auto& s = State(command);
	if (s.pending) {
		if (s.rendering) {
			(s.khr_begin ? g_end_rendering_khr : g_end_rendering)(command);
			s.rendering = false;
		}
		RecordPending(command, s);
	}
	return g_end_command(command);
}

void BeginRendering(VkCommandBuffer command, const VkRenderingInfo* info, bool khr) {
	auto& s = State(command);
	if (s.pending) {
		RecordPending(command, s);
	}
	bool copy = info->pNext == nullptr && info->colorAttachmentCount <= MaxColors;
	for (uint32_t i = 0; copy && i < info->colorAttachmentCount; i++) {
		copy = info->pColorAttachments[i].pNext == nullptr;
	}
	copy = copy && (info->pDepthAttachment == nullptr || info->pDepthAttachment->pNext == nullptr);
	copy = copy &&
	       (info->pStencilAttachment == nullptr || info->pStencilAttachment->pNext == nullptr);
	copy = copy && (info->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)) == 0;
	s.sinkable = copy;
	if (copy) {
		s.info = *info;
		for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
			s.colors[i] = info->pColorAttachments[i];
		}
		s.depth   = info->pDepthAttachment != nullptr ? *info->pDepthAttachment : VkRenderingAttachmentInfo {};
		s.stencil = info->pStencilAttachment != nullptr ? *info->pStencilAttachment : VkRenderingAttachmentInfo {};
	}
	if (s.instances > 0) {
		s.epoch = false;
	}
	s.instances++;
	s.rendering = true;
	s.khr_begin = khr;
	(khr ? g_begin_rendering_khr : g_begin_rendering)(command, info);
}

VKAPI_ATTR void VKAPI_CALL HookBeginRendering(VkCommandBuffer command, const VkRenderingInfo* info) {
	BeginRendering(command, info, false);
}
VKAPI_ATTR void VKAPI_CALL HookBeginRenderingKhr(VkCommandBuffer command,
                                                 const VkRenderingInfo* info) {
	BeginRendering(command, info, true);
}

void EndRendering(VkCommandBuffer command, bool khr) {
	auto& s = State(command);
	(khr ? g_end_rendering_khr : g_end_rendering)(command);
	s.rendering = false;
	s.sinkable  = false;
	s.epoch     = false;
	if (s.pending) {
		RecordPending(command, s);
		g_flushed_at_end.fetch_add(1, std::memory_order_relaxed);
	}
	Report();
}

VKAPI_ATTR void VKAPI_CALL HookEndRendering(VkCommandBuffer command) {
	EndRendering(command, false);
}
VKAPI_ATTR void VKAPI_CALL HookEndRenderingKhr(VkCommandBuffer command) {
	EndRendering(command, true);
}

void PipelineBarrier2(VkCommandBuffer command, const VkDependencyInfo* info, bool khr) {
	auto&      s     = State(command);
	const bool guest = t_command_guest;
	t_command_guest  = false;
	const bool memory_only = info->memoryBarrierCount == 1 && info->bufferMemoryBarrierCount == 0 &&
	                         info->imageMemoryBarrierCount == 0 && info->pNext == nullptr &&
	                         info->pMemoryBarriers[0].pNext == nullptr;
	if (guest && memory_only) {
		g_guest.fetch_add(1, std::memory_order_relaxed);
		if (!s.rendering) {
			g_guest_outside.fetch_add(1, std::memory_order_relaxed);
		} else if (!(s.epoch && s.sinkable)) {
			g_guest_no_epoch.fetch_add(1, std::memory_order_relaxed);
		}
		if (s.rendering) {
			MergePending(s, info->pMemoryBarriers[0]);
			if (!(s.epoch && s.sinkable)) {
				// Not provably independent of what follows: as before, rendering ends here.
				if (!s.sinkable) {
					// The renderer's rendering infos have no extension structures; anything else
					// cannot be restarted and must not reach here with a guest barrier.
					std::fprintf(stderr, "[barrier-sink] guest barrier in a rendering instance "
					                     "that cannot be restarted\n");
					std::abort();
				}
				Restart(command, s);
			}
			return;
		}
	}
	if (s.rendering) {
		// Not expected (barriers are recorded outside rendering); keep the recorded order.
		Flush(command, s);
		(khr ? g_barrier2_khr : g_barrier2)(command, info);
		return;
	}
	if (s.pending) {
		RecordPending(command, s);
	}
	(khr ? g_barrier2_khr : g_barrier2)(command, info);
	if (IsFull(*info)) {
		NoteFull(s);
	}
}

VKAPI_ATTR void VKAPI_CALL HookBarrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
	PipelineBarrier2(command, info, false);
}
VKAPI_ATTR void VKAPI_CALL HookBarrier2Khr(VkCommandBuffer command, const VkDependencyInfo* info) {
	PipelineBarrier2(command, info, true);
}

VKAPI_ATTR void VKAPI_CALL HookBarrier(VkCommandBuffer command, VkPipelineStageFlags src,
                                       VkPipelineStageFlags dst, VkDependencyFlags flags,
                                       uint32_t memory_count, const VkMemoryBarrier* memory,
                                       uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                       uint32_t image_count, const VkImageMemoryBarrier* images) {
	auto& s         = State(command);
	t_command_guest = false;
	Flush(command, s);
	g_barrier(command, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count,
	          images);
	if (!s.rendering && IsFull1(src, dst, memory_count, memory)) {
		NoteFull(s);
	}
}

enum class Kind { Draw, InsideWork, OutsideWork };

template <int Id, Kind K, typename F>
struct Hook;
template <int Id, Kind K, typename R, typename... A>
struct Hook<Id, K, R(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	using Fn                  = R(VKAPI_PTR*)(VkCommandBuffer, A...);
	static inline Fn original = nullptr;
	static R VKAPI_PTR Call(VkCommandBuffer command, A... args) {
		auto&      s    = State(command);
		const bool safe = K == Kind::Draw && t_command_draw_safe;
		t_command_draw_safe = false;
		if (K == Kind::Draw) {
			(safe ? g_draws_safe : g_draws_unsafe).fetch_add(1, std::memory_order_relaxed);
		}
		if (s.pending) {
			if (safe && s.rendering && s.epoch && s.sinkable) {
				g_sunk.fetch_add(1, std::memory_order_relaxed);
			} else {
				(K != Kind::Draw ? g_flush_other : !safe ? g_flush_unsafe : g_flush_epoch)
				    .fetch_add(1, std::memory_order_relaxed);
				Flush(command, s);
			}
		}
		if (!safe) {
			s.epoch = false;
		}
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

bool BarrierSinkEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BARRIER_SINK");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void SetNextDrawSafe(bool safe) noexcept {
	t_next_draw_safe = safe;
}
bool NextDrawSafe() noexcept {
	return t_next_draw_safe;
}
void SetCommandDrawSafe(bool safe) noexcept {
	t_command_draw_safe = safe;
}
void SetCommandGuestBarrier(bool guest) noexcept {
	t_command_guest = guest;
}

void InstallBarrierSink() {
	if (!BarrierSinkEnabled()) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkBeginCommandBuffer == nullptr || d.vkEndCommandBuffer == nullptr ||
	    (d.vkCmdBeginRendering == nullptr && d.vkCmdBeginRenderingKHR == nullptr) ||
	    (d.vkCmdPipelineBarrier2 == nullptr && d.vkCmdPipelineBarrier2KHR == nullptr) ||
	    d.vkCmdPipelineBarrier == nullptr) {
		std::fprintf(stderr, "[barrier-sink] missing entry points: off\n");
		return;
	}
	g_begin_command       = d.vkBeginCommandBuffer;
	d.vkBeginCommandBuffer = HookBeginCommand;
	g_end_command         = d.vkEndCommandBuffer;
	d.vkEndCommandBuffer  = HookEndCommand;
	g_begin_rendering     = d.vkCmdBeginRendering;
	g_begin_rendering_khr = d.vkCmdBeginRenderingKHR != nullptr ? d.vkCmdBeginRenderingKHR : d.vkCmdBeginRendering;
	g_end_rendering       = d.vkCmdEndRendering;
	g_end_rendering_khr   = d.vkCmdEndRenderingKHR != nullptr ? d.vkCmdEndRenderingKHR : d.vkCmdEndRendering;
	if (g_begin_rendering == nullptr) g_begin_rendering = g_begin_rendering_khr;
	if (g_end_rendering == nullptr) g_end_rendering = g_end_rendering_khr;
	if (d.vkCmdBeginRendering != nullptr) d.vkCmdBeginRendering = HookBeginRendering;
	if (d.vkCmdBeginRenderingKHR != nullptr) d.vkCmdBeginRenderingKHR = HookBeginRenderingKhr;
	if (d.vkCmdEndRendering != nullptr) d.vkCmdEndRendering = HookEndRendering;
	if (d.vkCmdEndRenderingKHR != nullptr) d.vkCmdEndRenderingKHR = HookEndRenderingKhr;
	g_barrier     = d.vkCmdPipelineBarrier;
	d.vkCmdPipelineBarrier = HookBarrier;
	g_barrier2     = d.vkCmdPipelineBarrier2;
	g_barrier2_khr = d.vkCmdPipelineBarrier2KHR != nullptr ? d.vkCmdPipelineBarrier2KHR : d.vkCmdPipelineBarrier2;
	if (g_barrier2 == nullptr) g_barrier2 = g_barrier2_khr;
	if (d.vkCmdPipelineBarrier2 != nullptr) d.vkCmdPipelineBarrier2 = HookBarrier2;
	if (d.vkCmdPipelineBarrier2KHR != nullptr) d.vkCmdPipelineBarrier2KHR = HookBarrier2Khr;

	// Draws: may pass a pending barrier when marked safe.
	Hook<0, Kind::Draw, decltype(d.vkCmdDraw)>::Install(d.vkCmdDraw);
	Hook<1, Kind::Draw, decltype(d.vkCmdDrawIndexed)>::Install(d.vkCmdDrawIndexed);
	Hook<2, Kind::Draw, decltype(d.vkCmdDrawIndirect)>::Install(d.vkCmdDrawIndirect);
	Hook<3, Kind::Draw, decltype(d.vkCmdDrawIndexedIndirect)>::Install(d.vkCmdDrawIndexedIndirect);
	Hook<4, Kind::Draw, decltype(d.vkCmdDrawIndirectCount)>::Install(d.vkCmdDrawIndirectCount);
	Hook<5, Kind::Draw, decltype(d.vkCmdDrawIndexedIndirectCount)>::Install(
	    d.vkCmdDrawIndexedIndirectCount);
	Hook<6, Kind::Draw, decltype(d.vkCmdDrawMeshTasksEXT)>::Install(d.vkCmdDrawMeshTasksEXT);
	Hook<7, Kind::Draw, decltype(d.vkCmdDrawMeshTasksIndirectEXT)>::Install(
	    d.vkCmdDrawMeshTasksIndirectEXT);
	Hook<8, Kind::Draw, decltype(d.vkCmdDrawMeshTasksIndirectCountEXT)>::Install(
	    d.vkCmdDrawMeshTasksIndirectCountEXT);
	Hook<9, Kind::Draw, decltype(d.vkCmdDrawIndirectCountKHR)>::Install(d.vkCmdDrawIndirectCountKHR);
	Hook<10, Kind::Draw, decltype(d.vkCmdDrawIndexedIndirectCountKHR)>::Install(
	    d.vkCmdDrawIndexedIndirectCountKHR);
	// Inside rendering, never past a pending barrier.
	Hook<20, Kind::InsideWork, decltype(d.vkCmdClearAttachments)>::Install(d.vkCmdClearAttachments);
	Hook<21, Kind::InsideWork, decltype(d.vkCmdBeginQuery)>::Install(d.vkCmdBeginQuery);
	Hook<22, Kind::InsideWork, decltype(d.vkCmdEndQuery)>::Install(d.vkCmdEndQuery);
	Hook<23, Kind::InsideWork, decltype(d.vkCmdWriteTimestamp)>::Install(d.vkCmdWriteTimestamp);
	Hook<24, Kind::InsideWork, decltype(d.vkCmdWriteTimestamp2)>::Install(d.vkCmdWriteTimestamp2);
	Hook<25, Kind::InsideWork, decltype(d.vkCmdWriteTimestamp2KHR)>::Install(d.vkCmdWriteTimestamp2KHR);
	Hook<26, Kind::InsideWork, decltype(d.vkCmdExecuteCommands)>::Install(d.vkCmdExecuteCommands);
	// Outside rendering (a pending barrier was recorded at its end already; they end the epoch).
	Hook<40, Kind::OutsideWork, decltype(d.vkCmdDispatch)>::Install(d.vkCmdDispatch);
	Hook<41, Kind::OutsideWork, decltype(d.vkCmdDispatchIndirect)>::Install(d.vkCmdDispatchIndirect);
	Hook<42, Kind::OutsideWork, decltype(d.vkCmdDispatchBase)>::Install(d.vkCmdDispatchBase);
	Hook<43, Kind::OutsideWork, decltype(d.vkCmdCopyBuffer)>::Install(d.vkCmdCopyBuffer);
	Hook<44, Kind::OutsideWork, decltype(d.vkCmdCopyImage)>::Install(d.vkCmdCopyImage);
	Hook<45, Kind::OutsideWork, decltype(d.vkCmdCopyBufferToImage)>::Install(d.vkCmdCopyBufferToImage);
	Hook<46, Kind::OutsideWork, decltype(d.vkCmdCopyImageToBuffer)>::Install(d.vkCmdCopyImageToBuffer);
	Hook<47, Kind::OutsideWork, decltype(d.vkCmdFillBuffer)>::Install(d.vkCmdFillBuffer);
	Hook<48, Kind::OutsideWork, decltype(d.vkCmdUpdateBuffer)>::Install(d.vkCmdUpdateBuffer);
	Hook<49, Kind::OutsideWork, decltype(d.vkCmdClearColorImage)>::Install(d.vkCmdClearColorImage);
	Hook<50, Kind::OutsideWork, decltype(d.vkCmdClearDepthStencilImage)>::Install(
	    d.vkCmdClearDepthStencilImage);
	Hook<51, Kind::OutsideWork, decltype(d.vkCmdBlitImage)>::Install(d.vkCmdBlitImage);
	Hook<52, Kind::OutsideWork, decltype(d.vkCmdResolveImage)>::Install(d.vkCmdResolveImage);
	Hook<53, Kind::OutsideWork, decltype(d.vkCmdResetQueryPool)>::Install(d.vkCmdResetQueryPool);
	Hook<54, Kind::OutsideWork, decltype(d.vkCmdCopyQueryPoolResults)>::Install(
	    d.vkCmdCopyQueryPoolResults);
	Hook<55, Kind::OutsideWork, decltype(d.vkCmdCopyBuffer2)>::Install(d.vkCmdCopyBuffer2);
	Hook<56, Kind::OutsideWork, decltype(d.vkCmdCopyImage2)>::Install(d.vkCmdCopyImage2);
	Hook<57, Kind::OutsideWork, decltype(d.vkCmdCopyBufferToImage2)>::Install(d.vkCmdCopyBufferToImage2);
	Hook<58, Kind::OutsideWork, decltype(d.vkCmdCopyImageToBuffer2)>::Install(d.vkCmdCopyImageToBuffer2);
	Hook<59, Kind::OutsideWork, decltype(d.vkCmdBlitImage2)>::Install(d.vkCmdBlitImage2);
	Hook<60, Kind::OutsideWork, decltype(d.vkCmdResolveImage2)>::Install(d.vkCmdResolveImage2);
	Hook<61, Kind::OutsideWork, decltype(d.vkCmdSetEvent)>::Install(d.vkCmdSetEvent);
	Hook<62, Kind::OutsideWork, decltype(d.vkCmdWaitEvents)>::Install(d.vkCmdWaitEvents);
	Hook<63, Kind::OutsideWork, decltype(d.vkCmdSetEvent2)>::Install(d.vkCmdSetEvent2);
	Hook<64, Kind::OutsideWork, decltype(d.vkCmdWaitEvents2)>::Install(d.vkCmdWaitEvents2);
	std::fprintf(stderr, "[barrier-sink] guest barriers may stay pending across safe draws\n");
}

} // namespace Libs::Graphics
