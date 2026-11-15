#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <utility>
#include <span>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	// The renderer mutex with a per-thread depth, for the command-processor threads
	// (KYTY_ASYNC_THREAD): an off-lock window releases every level its thread holds.
	void                LockRenderer();
	void                UnlockRenderer();
	// Graphics right of way (KYTY_ASYNC_POLITE, on by default with the async thread): the graphics
	// command-processor thread is waiting for the renderer mutex. The async thread steps aside
	// at its next packet and does not take the mutex while this holds.
	[[nodiscard]] static bool GraphicsWaiting() noexcept;
	[[nodiscard]] static bool PoliteAsync() noexcept;
	// Releases all of this thread's tracked levels and marks it off-lock (guest-memory reads go
	// through the concurrent queries, SetGpuReadDelegate); returns the depth to restore.
	uint32_t            ReleaseRenderer();
	void                ReacquireRenderer(uint32_t depth);
	// This thread is inside an off-lock window (ReleaseRenderer).
	[[nodiscard]] static bool RendererReleased() noexcept;
	// This thread holds the renderer through LockRenderer (ReacquireScope among them).
	[[nodiscard]] static bool HoldsRenderer() noexcept;
	// A thread holding no level marks itself off-lock (true) or not (false) as ReleaseRenderer
	// does, for work it does without the mutex (the async thread's queued dispatches, G4).
	static void MarkOffLock(bool off);
	// This thread's renderer-mutex statistics since the last call: time waiting for it, how
	// often it was held by another thread, and time spent in off-lock windows.
	struct LockStats {
		uint64_t wait_ns     = 0;
		uint64_t contended   = 0;
		uint64_t acquires    = 0;
		uint64_t released_ns = 0;
	};
	static LockStats TakeLockStats() noexcept;
	// Releases this thread's tracked renderer levels for a blocking wait whose end needs another
	// thread that takes the renderer mutex (the presenter, for flips), and restores them after,
	// with the thread's registers bound again.
	class UnlockedScope {
	public:
		explicit UnlockedScope(RenderContext& context);
		~UnlockedScope();
		KYTY_CLASS_NO_COPY(UnlockedScope);

	private:
		RenderContext& m_context;
		uint32_t       m_depth = 0;
	};
	// Inside an off-lock window, holds the renderer mutex (and leaves the delegate mode) for the
	// scope: the exact guest-memory paths (downloads, the GPU-thread caches) need it.
	class ReacquireScope {
	public:
		explicit ReacquireScope(RenderContext& context);
		// The renderer instance (there is one).
		ReacquireScope();
		~ReacquireScope();
		KYTY_CLASS_NO_COPY(ReacquireScope);

	private:
		RenderContext* m_context = nullptr;
	};
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	BindlessTable&      GetBindlessTable() { return m_bindless_table; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// PageManager::ResolveStale, before the fault is emulated or handled.
	[[nodiscard]] bool ResolveStaleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
		return m_page_manager.ResolveStale(fault_vaddr, access);
	}
	// Any thread: a read of these bytes faulted on a page protected because the GPU wrote to it,
	// but none of them is GPU-written, so guest memory already holds their value.
	[[nodiscard]] bool CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
	                                     uint64_t size) const noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	// UnmapMemory for several ranges with one GPU-thread round trip and one drain (an AMM batch
	// remaps dozens of ranges; one round trip each made the batch take tens of milliseconds).
	void               UnmapMemoryRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges);
	// GPU thread, inside an unmap: drains the GPU only when the ranges hold a shared buffer (F4.1).
	void               DrainForUnmap(std::span<const std::pair<uint64_t, uint64_t>> ranges);
	// The host protection of the range was replaced behind the page tracker
	// (PageManager::ResyncHostProtection): the watched pages get the tracker's protection back.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size) {
		m_page_manager.ResyncHostProtection(vaddr, size);
	}
	void ForgetProtectionFaults(uint64_t vaddr, uint64_t size) noexcept {
		m_page_manager.ForgetProtectionFaults(vaddr, size);
	}
	void               PrepareBda();
	void               RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	BindlessTable             m_bindless_table;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_bda_logged = false;
	uint64_t                  m_bda_synced_epoch      = 0;
	uint64_t                  m_bda_synced_submission = 0;
	uint64_t                  m_bda_synced_waits      = 0; // g_bda_resync_seq at the last walk (G6)

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
