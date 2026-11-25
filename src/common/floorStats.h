#ifndef EMULATOR_SRC_COMMON_FLOORSTATS_H_
#define EMULATOR_SRC_COMMON_FLOORSTATS_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>

// KYTY_FLOOR_LOG=1 (plan 4.14, step 1): one `[floor]` line a second with what each part of the
// emulator spent in it, so every second under 30 fps can be labelled by its cause (GPU-bound,
// GPU-thread-bound, starved, buffer-sync spikes, compiles). The counters are plain relaxed atomics,
// added to where the existing (5-10 s) statistics are kept; the line is printed from the guest
// flip (RenderDocOnGuestFlip). Off: one predictable branch per site.
namespace Common::FloorStats {

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FLOOR_LOG");
		return value != nullptr && value[0] != '0';
	}();
	return enabled;
}

struct Counters {
	std::atomic<uint64_t> gpu_busy_ns {0};      // GPU busy (KYTY_GPU_TIME's command-buffer spans)
	std::atomic<uint64_t> thread_process_ns {0}; // GPU thread: processing guest submissions
	std::atomic<uint64_t> thread_empty_ns {0};   // GPU thread: no guest work queued
	std::atomic<uint64_t> thread_blocked_ns {0}; // GPU thread: every queue blocked
	std::atomic<uint64_t> thread_command_ns {0}; // GPU thread: external commands (unmaps, flips)
	std::atomic<uint64_t> sync_calls {0};        // BufferCache::SynchronizeBuffer calls
	std::atomic<uint64_t> sync_bytes {0};        // bytes those calls covered
	std::atomic<uint64_t> sync_big {0};          // calls over 16 MB
	std::atomic<uint64_t> sync_ns {0};           // time in them (with KYTY_SYNC_STATS)
	std::atomic<uint64_t> sync_uploads {0};      // calls that uploaded
	std::atomic<uint64_t> readback_game_ns {0};  // game threads waiting for GPU-written data
	std::atomic<uint64_t> readback_game_n {0};
	std::atomic<uint64_t> readback_gpu_ns {0};   // the GPU thread reading GPU-written data
	std::atomic<uint64_t> compiles {0};          // shader compiles (not disk-cache hits)
	std::atomic<uint64_t> compile_ns {0};        // time in those compiles (any thread)
	std::atomic<uint64_t> pipe_wait_ns {0};      // GPU thread waiting for a worker's pipeline compile
	std::atomic<uint64_t> pipe_waits {0};        //   waits
	std::atomic<uint64_t> pipe_timeouts {0};     //   waits that ran out (the draw was skipped)
	std::atomic<uint64_t> pipe_sync_ns {0};      // pipelines compiled on the GPU thread itself
	std::atomic<uint64_t> pipe_built {0};        // pipelines built (any thread)
	// Fps plan W1b.4 (11-15): item counts in every run, without KYTY_CPU_COST.
	std::atomic<uint64_t> draws {0};             // draws completed in full (a replayed packet once)
	std::atomic<uint64_t> replayed {0};          // multi-draw sub-draws replayed
	std::atomic<uint64_t> dispatches {0};        // DispatchDirect + DispatchIndirect packets
	std::atomic<uint64_t> cmd_buffers {0};       // command buffers submitted
	// Smoothness plan (11-16): what a long frame held ([long-frame], renderDoc.cpp).
	std::atomic<uint64_t> gpu_wait_ns {0};       // GPU thread waiting for the host GPU (scheduler Wait/Finish)
	std::atomic<uint64_t> gpu_waits {0};         //   waits
	std::atomic<uint64_t> ring_wait_ns {0};      //   of it: an upload ring wrapped onto bytes the GPU still needs
	std::atomic<uint64_t> tex_created {0};       // images made
	std::atomic<uint64_t> tex_created_bytes {0};
	std::atomic<uint64_t> tex_uploads {0};       // image uploads (guest bytes read, detiled, copied)
	std::atomic<uint64_t> tex_upload_bytes {0};
	std::atomic<uint64_t> tex_deleted {0};       // images deleted
	std::atomic<uint64_t> heap_keys {0};         // heap textures settled in a feedback round (a draw asked)
	std::atomic<uint64_t> gpu_unmaps {0};        // ranges unmapped on the GPU thread
	std::atomic<uint64_t> amm_batches {0};       // the game's map/unmap batches (any thread)
	std::atomic<uint64_t> amm_batch_ns {0};      //   time in them
	std::atomic<uint64_t> amm_wait_ns {0};       //   of it: waiting on an address or a counter
	std::atomic<uint64_t> buf_created {0};       // guest buffers made (a new range, or a merge of older ones)
	std::atomic<uint64_t> buf_created_bytes {0};
	// Which wall binds (review 9, 11-16): the stages behind the GPU thread, per second in [floor].
	std::atomic<uint64_t> record_busy_ns {0};    // recording thread: executing chunks (Vulkan calls, vkQueueSubmit)
	std::atomic<uint64_t> record_drain_ns {0};   // a thread waiting for the recording thread (DrainRecording)
	std::atomic<uint64_t> record_drains {0};     //   waits that had to block
	std::atomic<uint64_t> lag_ticks {0};         // command buffers handed off and not finished by the GPU, summed
	std::atomic<uint64_t> lag_samples {0};       //   over these samples (every 8th submit)
	// v438zk (next-phase plan, S5 and S2): what became of the flips the game asked for, and texture
	// bytes uploaded again unchanged, per frame in [long-frame].
	std::atomic<uint64_t> flip_requests {0};     // flips the game asked for (FlipQueue::Reserve)
	std::atomic<uint64_t> flip_replaced {0};     //   replaced by a newer request before they were shown
	std::atomic<uint64_t> flip_refused {0};      //   refused: the port's flip queue was full
	std::atomic<uint64_t> flip_cancelled {0};    //   cancelled (the port closed or changed, a failed prepare)
	std::atomic<uint64_t> flip_held {0};         // a ready flip the present thread held back (flip rate, vblank)
	std::atomic<uint64_t> tex_again {0};         // uploads of bytes uploaded before (KYTY_TEX_UPLOAD_STATS)
	std::atomic<uint64_t> tex_same_bytes {0};    //   of their bytes: unchanged since that upload
	std::atomic<uint64_t> expand_ns {0};         // v438zl: compute lookups that expanded their functions: time
	std::atomic<uint64_t> expand_slow {0};       //   of them, the ones over KYTY_EXPAND_SLOW_MS (5 ms)
};

inline Counters& Get() {
	static Counters counters;
	return counters;
}

inline void Add(std::atomic<uint64_t>& counter, uint64_t value) {
	counter.fetch_add(value, std::memory_order_relaxed);
}

} // namespace Common::FloorStats

#endif // EMULATOR_SRC_COMMON_FLOORSTATS_H_
