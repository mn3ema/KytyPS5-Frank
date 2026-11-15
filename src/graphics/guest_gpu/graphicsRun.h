#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <vector>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <unordered_map>

namespace Libs::Graphics {

class RenderContext;
class DrawPrepScanner;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	// Like SendCommand, but false instead of exiting once the GPU no longer accepts commands.
	bool               TrySendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// End-of-pipe labels (RELEASE_MEM without an interrupt) become visible to the CPU once the
	// host GPU has executed the work recorded before them, as on hardware, instead of when the
	// packet is recorded. Until then the GPU thread's own label checks (WAIT_REG_MEM,
	// COND_INDIRECT_BUFFER) see the recorded value through ReadLabel.
	// KYTY_LABELS_AT_COMPLETION=1 enables it.
	[[nodiscard]] static bool LabelsAtCompletion();
	// Whether any label write has been deferred (so ReadLabel must consult the pending ones).
	[[nodiscard]] static bool LabelsDeferred() noexcept;
	// GPU thread.
	void DeferLabelWrite(uint64_t address, uint64_t value, uint32_t size);
	template <typename T>
	[[nodiscard]] T ReadLabel(const volatile T* address) const;

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> draw_commands,
	            std::span<const uint32_t> constant_commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	// The async-compute command-processor thread (KYTY_ASYNC_THREAD).
	[[nodiscard]] static bool IsAsyncThread() noexcept;
	// KYTY_DRAW_PIPE: marks the calling thread (the draw thread, stage B) as a GPU thread.
	static void MarkDrawPipeThread() noexcept;
	// KYTY_ASYNC_THREAD=1, or KYTY_SYNC_TOGGLE_MODE=async: the async-compute thread exists (and
	// both threads take the renderer mutex around the renderer).
	[[nodiscard]] static bool AsyncThreadConfigured() noexcept;
	// The async-compute thread runs the compute queues now (always with KYTY_ASYNC_THREAD=1).
	[[nodiscard]] static bool AsyncThreadActiveNow() noexcept;
	// After an off-lock window: binds this thread's command processor's registers to the
	// scheduler's command buffer again (the other thread may have bound its own).
	static void RebindCurrentProcessor();

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		uint64_t                  flip_request_id   = 0;
		uint64_t                  prep_seq          = 0; // draw prep: graphics submission number
		uint64_t                  srt_wait_tick     = 0; // KYTY_SRT_DEFER: run again once free
		// g_cpu_dirty_epoch when the guest submitted: the CPU writes its commands may read.
		uint64_t                  cpu_epoch         = 0;
		// KYTY_COPY_SUBMISSIONS=1: the command words as they were at submission (the spans point
		// here), in case the game rewrites the buffer before the GPU thread reaches it.
		std::shared_ptr<std::vector<uint32_t>> owned_words;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	// A submission waits in a queue and is not blocked (KYTY_COALESCE_SUBMITS).
	bool              HasReadySubmission();
	static void       ThreadRun(void* data);
	// KYTY_ASYNC_THREAD=1: role 0 runs the graphics queue and the commands, role 1 (its own
	// thread) the compute queues; without it role 0 runs everything.
	static void       ThreadRunRole(GuestGpu* gpu, uint32_t role);
	[[nodiscard]] static uint32_t RoleOf(uint32_t queue_id) noexcept;
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue[2]     = {0, 1};
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing[2]     = {false, false};
	uint32_t                                       m_role_count[2]     = {0, 0};
	// A submission of the queue is being processed (outside m_queue_mutex): with the toggle
	// (KYTY_SYNC_TOGGLE_MODE=async) either thread can run a compute queue, one at a time.
	std::array<bool, QueueCount>                   m_queue_busy {};
	Common::CondVar                                m_async_work;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	struct PendingLabel {
		uint64_t value    = 0;
		uint32_t size     = 0;
		uint64_t sequence = 0;
	};
	// Label writes recorded but not yet visible to the CPU. GPU thread only.
	std::unordered_map<uint64_t, PendingLabel> m_pending_labels;
	uint64_t                                   m_label_sequence = 0;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;
	// KYTY_DRAW_PREP: the scanner, and the number of the last graphics submission queued
	// (under m_queue_mutex).
	std::unique_ptr<DrawPrepScanner> m_draw_prep;
	uint64_t                         m_prep_seq = 0;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;
	std::jthread    m_async_thread;

	friend class CommandProcessor;
};

// Diagnostics (KYTY_TRACE_GS_SLOT with KYTY_TRACE_GS_SLOT_SEQ): the last write to the traced GS
// user-data slot on this thread. source: 0 none, 1 SET_SH_REG, 2 SET_SH_REG_INDIRECT table entry,
// 3 other register-list write. where: the dword in the command buffer or table.
struct GsSlotWriter {
	uint32_t        source = 0;
	const uint32_t* where  = nullptr;
	uint32_t        value  = 0;
	uint64_t        seq    = 0;
	uint32_t        start  = 0; // SET_SH_REG: first slot and register count
	uint32_t        count  = 0;
};
GsSlotWriter GetGsSlotLastWriter();
// Lines left for the [gs-seq] trace (0 when it is off); counts one down.
bool TakeGsSlotSeqLine();
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
