#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCachePages.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <span>
#include <string>
#include <unordered_map>
#include <chrono>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// KYTY_UPLOAD_WORKER: the number of upload reads queued so far (0 when off), and a wait until the
// reads before that number are done (bufferCache.cpp).
[[nodiscard]] uint64_t UploadWorkerPushed();
void                   WaitUploadWorker(uint64_t upto);

// v445o (KYTY_SYNC_STATS_LOCAL): adds the calling thread's per-sync [floor] counts to the shared
// counters (bufferCache.cpp RecordSync). Called at every guest flip.
void FoldSyncCounters() noexcept;

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

struct ReadbackCauseSample; // bufferCache.cpp (KYTY_READBACK_CAUSE)
struct ExactVerifySample;   // bufferCache.cpp (KYTY_EXACT_WRITE_RANGES_VERIFY)

class BufferCache {
public:
	// Buffers unregistered so far (a join retires the buffers it absorbs; unmaps and the collector
	// delete them): a draw or dispatch whose bindings were resolved before the count moved may
	// hold a retired buffer (KYTY_REBIND_AFTER_JOIN, RenderExecutor).
	[[nodiscard]] uint64_t UnregisterEpoch() const noexcept { return m_unregister_epoch; }
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static_assert(CACHING_PAGEBITS == BufferCachePages::kPageBits &&
	              CACHING_NUMPAGES == BufferCachePages::kNumPages &&
	              LOWER_ADDRESS_SIZE == BufferCachePages::kLowerAddressSize &&
	              LibKernel::Memory::kExtendedMemoryBase == BufferCachePages::kExtendedMemoryBase,
	              "bufferCachePages.h is out of date");
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// KYTY_FAULT_BENCH (faultBench.cpp) only: what a read-only sync that finds nothing to upload
	// does to the tracker for the range, and no more: its CPU-dirty pages become CPU-clean and
	// write-watched, so the next CPU write to them faults. Any thread.
	void                   FaultBenchArm(uint64_t vaddr, uint64_t size);
	// RenderContext: what a CPU write fault on [vaddr, vaddr + size) does (buffer cache, texture
	// cache, bindless table), for host code about to write guest memory itself (CopyBuffer).
	std::function<void(uint64_t, uint64_t)> on_cpu_write_range;
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	// v386 [mem-census]: what the cache holds.
	[[nodiscard]] uint64_t TotalUsedMemory() const noexcept { return m_total_used_memory; }
	[[nodiscard]] size_t   BufferCount() const noexcept { return m_slot_buffers.size(); }
	// v0.5b: true when [vaddr, vaddr+size) overlaps a live host zone without fitting inside it
	// (KYTY_ZONE_CROSS_SNAPSHOT=0: always false).
	[[nodiscard]] bool CrossesHostZone(uint64_t vaddr, uint64_t size);
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// The existing buffer holding [vaddr, vaddr + size), or NULL_BUFFER_ID; never creates one
	// (diagnostics: KYTY_BUFFER_PROBE).
	[[nodiscard]] BufferId PeekBuffer(uint64_t vaddr, uint64_t size);
	// needs_device_address: the caller reads the data through a buffer device address, which
	// the stream buffer used for small CPU-written reads does not have.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer      = false,
	                                                        BufferId id                   = {},
	                                                        bool     needs_device_address = false);
	// A written binding of which the shader's stores reach only [written_vaddr, written_vaddr +
	// written_size): the whole range is synchronized, only that part becomes GPU-written.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferWritten(uint64_t vaddr, uint64_t size,
	                                                               uint64_t written_vaddr,
	                                                               uint64_t written_size,
	                                                               BufferId id = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer*       GetGdsBuffer() noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() {
		ClearBdaPageTableOnce();
		return &m_bda_pagetable_buffer;
	}
	// v443l: the guest address table starts as zeroes (no page is there). The device does not hand
	// out zeroed memory, and the table was never cleared: the entry of a page no buffer had ever
	// been registered for was whatever the memory held, which a shader then took for the page's
	// device address (g443m: guest 0x10a080dad8 -> dev 0x500001b73). Before the table's first write
	// or binding, on the GPU thread (as the fault buffer's, FaultManager::GetFaultBuffer).
	// KYTY_BDA_TABLE_CLEAR=0: not cleared, as before.
	void ClearBdaPageTableOnce();
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// GPU write bitmap (KYTY_WRITE_BITMAP=1, rework.md Phase 2 first step): one bit per 4 KiB
	// guest page below 1 TiB, set by shaders through bindings whose writes the host cannot bound.
	// Its device address, or 0 when it is off or unavailable.
	[[nodiscard]] vk::DeviceAddress WriteBitmapAddress();
	// KYTY_SRT_VERIFY: the report buffer verifying shaders write when a value the host used
	// differs from the GPU's (count, hash low, hash high), host-coherent; 0 when unavailable.
	[[nodiscard]] vk::DeviceAddress SrtVerifyReportAddress();
	// The shader the report names when its count changed since the last call, or 0; `detail`
	// gets the first differing pair's index, the host's value and the GPU's.
	[[nodiscard]] uint64_t TakeSrtVerifyReport(std::array<uint32_t, 3>* detail = nullptr);
	// The guard-trip report (IR::BindingLayout::loop_guard, LoopGuard.h): its device address for
	// the shader data (0: unavailable), and the new trips since the last call, tallied per shader
	// and printed ([loop-guard]); at_loss prints the whole tally.
	[[nodiscard]] vk::DeviceAddress LoopGuardReportAddress();
	void                            TakeLoopGuardReport(bool at_loss = false);
	// ObtainBuffer for such a written binding: its pages become bitmap-tracked, so a guest read
	// of a page the GPU did not write needs no download (ReadMemoryStep).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferTracked(uint64_t vaddr, uint64_t size,
	                                                               BufferId id);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool              IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// The same, under the dirty-range lock: for a GPU-read delegate (Memory::SetGpuReadDelegate).
	[[nodiscard]] bool              HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool       DirtyFilterConfigured();
	// Whether an asynchronous download of bytes in the range is still in flight: those bytes are
	// no longer GPU-dirty, but guest memory holds them only once the download is published.
	[[nodiscard]] bool              HasPendingDownload(uint64_t vaddr, uint64_t size);
	// GPU thread: waits until no download of bytes in the range is in flight (published).
	void                            WaitDownloadsLanded(uint64_t vaddr, uint64_t size);
	// GPU thread, before it reads guest bytes on the host (indirect arguments, COND_EXEC): shared
	// buffers are not page-protected, so a read there does not fault into a sync. Waits for the
	// GPU writes to them recorded so far.
	void WaitSharedWrites(uint64_t vaddr, uint64_t size);
	// GPU thread (KYTY_SRT_DEFER): records and submits the download of the GPU-written bytes of
	// [vaddr, vaddr + size) without waiting. Returns the tick to wait for (its completion and
	// priority operations publish the bytes), or 0 when the bytes can be read now.
	[[nodiscard]] uint64_t StartSrtReadback(uint64_t vaddr, uint64_t size);
	// GPU thread (KYTY_GPU_SIZED_DYNAMIC): see Memory::PeekGpuDword.
	[[nodiscard]] uint32_t PeekDword(uint64_t vaddr);
	// Diagnostics: the write-tracking state of the page holding vaddr.
	[[nodiscard]] std::string DescribeWriteState(uint64_t vaddr);
	// Any thread: none of the bytes is GPU-written, or on its way back from the GPU, so guest
	// memory holds their current value even when their page is protected.
	[[nodiscard]] bool              IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool              IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool              IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// KYTY_VERTEX_VERIFY=<n> (GPU thread, off unless set): every n-th draw, the bytes the draw
	// binds for its vertex, index and read-only storage ranges are copied back from the host
	// buffer and compared with guest memory ([vertex-verify] lines). 0: off.
	[[nodiscard]] static uint32_t   VertexVerifyInterval();
	// At the start of a draw: decides whether this draw's ranges are checked.
	void                            VertexVerifyArm();
	void                            VertexVerifyDisarm() noexcept { m_vertex_verify_armed = false; }
	[[nodiscard]] bool              VertexVerifyArmed() const noexcept { return m_vertex_verify_armed; }
	// A range the armed draw binds: guest [address, address + size) is at host_offset of host.
	// Records a copy of (a window of) it into the current command buffer and compares the copy
	// with guest memory when the GPU has run it. Changes nothing the draw reads.
	[[nodiscard]] int               VertexVerifyGpuOwner(uint64_t address, uint64_t size);
	void                            VertexVerifyRange(const char* kind, uint32_t slot, uint64_t shader,
	                                                  uint64_t address, uint64_t size, vk::Buffer host,
	                                                  uint64_t host_offset);
	// The next sync of the range uploads it from guest memory (KYTY_SYNC_TOGGLE_MODE=upload).
	void                            MarkRegionCpuModified(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	void                            ProcessFaultBuffer();
	// GPU thread: writes bytes on a page protected because the GPU wrote to it, without
	// downloading the page: the host copy through the backing store, the GPU copy in the command
	// stream, after the GPU's earlier writes. Bytes the GPU had written are then current on both
	// sides and no longer need a download. False when there is nothing to save (the page is not
	// protected as GPU-written) or it would be wrong (a download of these bytes is in flight, or
	// no cached buffer covers them); the caller then writes normally.
	[[nodiscard]] bool              WriteClean(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] ShaderFaultReport CollectFaults() { return m_fault_manager.CollectFaults(); }
	[[nodiscard]] uint64_t          UnattributedFaults() const noexcept {
		return m_fault_manager.UnattributedFaults();
	}
	uint64_t                        SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Same, but visits only the tracker regions that may hold CPU-dirty pages.
	void                            SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size);
	// The BDA walk over every mapped range, limited to regions with new CPU writes.
	void                            SynchronizeTouchedMappedBuffers(const RangeSet& mapped);
	static bool                     BdaTouchedEnabled();
	static uint64_t                 BufferWindowBytes();
	static constexpr uint64_t       ResidentBlockBits = 16; // 64 KB
	// Window buffers: makes the blocks of [vaddr, vaddr + size) resident (page-table entries
	// written in stream order). Returns how many blocks became resident.
	uint64_t                        EnsureResident(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void                            MakeRangeResident(uint64_t vaddr, uint64_t size);
	// G6 (KYTY_G6): a buffer a shader reads only through the GPU address table. Keeps it present
	// (created if missing), resident over the range and in use by this recording (the GC sees
	// it); no sync: the CPU's writes reach it through RenderContext::PrepareBda. Only window
	// buffers (KYTY_BUFFER_WINDOW_MB) qualify, as they hold the guest's bytes from creation;
	// false otherwise, and the caller binds the buffer as before.
	bool                            PrepareTableRead(uint64_t vaddr, uint64_t size, BufferId id);
	void                            RunGarbageCollector();
	// Diagnostics, GPU thread: makes guest memory current for [vaddr, vaddr + size) by
	// downloading what the GPU wrote there (drains the GPU).
	void                            DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size);
	// Diagnostics (GPU thread): the GPU copy of [vaddr, vaddr + size) as the cached buffer holding
	// it has it now, waiting for the GPU; false when no single buffer holds the range.
	bool                            PeekGpuCopy(uint64_t vaddr, uint64_t size, std::vector<uint32_t>& out);
	// Diagnostics (GPU thread): the GPU's BDA page table entry of vaddr's page, and the device
	// address the cached buffer holding that page expects there (0 when none holds it); waits
	// for the GPU. False when the table has no such page.
	bool PeekBdaEntry(uint64_t vaddr, uint64_t& entry, uint64_t& expected);
	// Diagnostics (GPU thread): records copies of the GPU's bytes of each guest range, and of
	// [raw_offset, raw_offset + raw_size) of the host buffer raw, behind all work recorded so far,
	// and calls done with them once that work has run (no wait). A range no single cached buffer
	// holds comes back empty.
	using SnapshotDone = std::function<void(std::vector<std::vector<uint32_t>>&, std::vector<uint32_t>&)>;
	void SnapshotGpu(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, vk::Buffer raw,
	                 uint64_t raw_offset, uint64_t raw_size, SnapshotDone done);

	// Diagnostics: the guest shader whose bindings are being prepared on this thread, if any.
	inline static thread_local uint64_t s_diag_shader_hash = 0;
	// v445s ([readback-cause]): its stage, a ShaderType as a number (0: none or unknown).
	inline static thread_local uint8_t s_diag_shader_stage = 0;
	// v445s, KYTY_FAST_FINISH_CLEANREAD / toggle mode "ffclean" (only with fastfinish on). Fault
	// handler of a game thread, for a read that faulted on a page watched because the GPU wrote to
	// it: 1 when the page had no GPU-written bytes left and its protection is lifted (the access
	// runs again), 2 under KYTY_FAST_FINISH_VERIFY when it would have been (the caller sends the
	// ordinary readback step, which is compared), 0 otherwise (the caller emulates the load).
	[[nodiscard]] static bool FastFinishCleanReadEnabled();
	[[nodiscard]] int         TryFastFinishCleanRead(uint64_t fault_vaddr);
	// The lifted page is still read-watched (another watcher): counted, the load is emulated.
	static void               NoteCleanReadStillWatched();

	// A ReadMemoryStep result with this bit set names a copy-queue readback (see bufferCache.cpp)
	// for the caller to complete, instead of a tick to wait for.
	static constexpr uint64_t CopyReadbackMarker = uint64_t {1} << 63u;
	// Any thread: waits for a copy-queue readback and publishes its bytes to guest memory (once;
	// true means a normal, non-snapshot copy's bytes are published, including an already-done marker).
	bool CompleteCopyReadback(uint64_t marker);
	// GPU thread, at the end of a submission: records the downloads of the windows the game
	// threads keep reading back that the submission wrote to (KYTY_EAGER_READBACK).
	void EagerDownloads();
	// GPU thread, before guest memory is unmapped: drops the shared buffers over it (their
	// imports point into that memory).
	void UnmapShared(uint64_t vaddr, uint64_t size);
	// GPU thread: whether an unmap of the range drops a shared buffer (an import of the guest
	// memory itself, freed at once), the one case an unmap must drain the GPU for (F4.1).
	[[nodiscard]] bool HasSharedBuffersIn(uint64_t vaddr, uint64_t size) const {
		return IsShared(vaddr, size);
	}

private:
	// Shared buffers (KYTY_SHARED_BUFFERS=1): a small buffer the game threads keep reading back is
	// replaced by one over the guest's own memory (VK_EXT_external_memory_host), so GPU writes land
	// in guest memory and the reads no longer wait for downloads. GPU thread only.
	[[nodiscard]] bool IsShared(uint64_t vaddr, uint64_t size) const {
		return !m_shared_ranges.Empty() && m_shared_ranges.Intersects(vaddr, size);
	}
	// A game-thread readback of this buffer: votes for sharing it, and shares it at enough votes.
	// True when the buffer is shared now and the read may proceed.
	bool VoteShared(BufferId id);
	uint64_t m_unregister_epoch = 0; // UnregisterEpoch()
	// Replaces the buffer with one over guest memory (the core of VoteShared).
	bool ShareBuffer(BufferId id, const char* reason);
	// KYTY_SHARE_WRITTEN=1: buffers uploaded from CPU-dirty pages in consecutive frames.
	struct UploadStreak {
		uint64_t last_frame = 0; // presented frame of the latest upload
		uint32_t frames     = 0;
	};
	uint64_t m_streak_frame = 0;
	uint32_t m_share_written_count = 0;
	std::unordered_map<uint64_t, UploadStreak> m_upload_streaks; // by buffer start
	std::mutex                                 m_upload_streaks_mutex; // SynchronizeBuffer runs on the scanner too
	void NoteCpuUpload(const Buffer& buffer, bool uploaded, bool gpu_written);
	std::unordered_set<uint64_t> m_gpu_written_buffers; // by start: bound as written at least once
	void ShareRewrittenBuffers(); // GPU thread, once per frame (RunGarbageCollector)
	RangeSet                                   m_shared_ranges;
	std::unordered_map<uint64_t, uint32_t>     m_shared_votes;
	std::unordered_map<uint64_t, uint32_t>     m_shared_attempts;
	// The tick of the latest GPU write recorded into each shared buffer, until it completes.
	std::unordered_map<uint64_t, uint64_t>     m_shared_write_ticks;
	// Host zones (KYTY_HOST_ZONES=1, after Revivedx 31cf25b7): guest ranges the game's threads
	// keep writing (per-object constants) get one shared buffer over the guest's own memory, so
	// their pages stay writable (no write faults, no uploads, no re-protection) and the GPU reads
	// the bytes when it runs, as on the console. A zone falls back to normal buffers for good when
	// a buffer would cross it or it holds GPU-written bytes.
	struct HostZone {
		uint64_t begin  = 0;
		uint64_t size   = 0;
		BufferId id     = NULL_BUFFER_ID;
		bool     has_id = false;
		bool     failed = false;
		uint32_t waits  = 0; // not mapped (or not in one mapping) yet
	};
	std::vector<HostZone> m_host_zones;
	HostZone*             FindHostZone(uint64_t vaddr, uint64_t size);
	static bool           HostZonesInWindows();
	bool                  CreateHostZoneBuffer(HostZone& zone);
	void                  DropHostZone(HostZone& zone, const char* reason);
	// v0.5b: a read crossing a zone's edge is served from a snapshot; the zone stays.
	[[nodiscard]] std::pair<Buffer*, uint64_t> SnapshotZoneCrossing(uint64_t vaddr, uint64_t size,
	                                                                bool needs_device_address);
	friend struct BufferCacheTestAccess;

	void ReleaseBuffer(BufferId id);
	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// Records bytes a binding makes GPU-written; caller holds m_gpu_write_state_mutex across the
	// preceding tracker mark and this range publication.
	void MarkGpuWritten(uint64_t vaddr, uint64_t size, bool tracked = false);
	// ReadMemoryStep: drops the pages of the readback window the bitmap shows the GPU did not
	// write. True with wait_tick set when an asynchronous caller must first wait for that tick.
	// v445t: clean_page, the faulting pages hold no GPU-written bytes and no download (the caller
	// lifts them whatever the bitmap says): KYTY_BITMAP_CLEAN_LIFT then skips the wait.
	bool RefineWithWriteBitmap(const Buffer& buffer, uint64_t vaddr, uint64_t size, bool async,
	                           uint64_t& wait_tick, bool clean_page = false);
	// When a submission has completed: tracked pages its shaders did not write are released
	// (no longer GPU-written, no longer protected). GPU thread.
	void SettleWriteBitmap(uint64_t completed_tick);
	// Whole pages whose GPU-written bytes were synced (downloaded or overwritten) leave
	// m_bitmap_written. Caller holds m_dirty_ranges_mutex.
	void ClearWriteBits(uint64_t vaddr, uint64_t size);
	// Once per frame (GPU thread): the next epoch's bitmap moves its bits into m_bitmap_written
	// and is cleared, then takes the frame's writes.
	void AdvanceWriteBitmapEra();
	// The written bits of [word_begin, word_end) ORed over all epochs (or of only_epoch), read
	// with streaming loads.
	void LoadWrittenWords(uint64_t word_begin, uint64_t word_end, std::vector<uint32_t>& out,
	                      int only_epoch = -1) const;
	uint64_t m_bitmap_settle_frame = UINT64_MAX;
	// Eras: one per frame. At each era's end the GPU copies the era's bitmap words to host-cached
	// staging and zeroes them, in command order, so each copy holds exactly the pages that era
	// wrote (the host never reads the bitmap itself: device memory, ~1 us a word).
	// m_bitmap_written: pages known written and not synced since.
	static constexpr uint32_t                          WriteBitmapEpochs = 1;
	uint64_t                                           m_bitmap_era      = 0;
	std::array<uint64_t, 4>                            m_era_start_tick {};
	// region -> last era it was marked in; erased once that era's copy is processed
	std::unordered_map<uint64_t, uint64_t>             m_region_era;
	RangeSet                                           m_bitmap_written;
	struct BitmapCopy {
		uint64_t                                   era  = 0;
		uint64_t                                   tick = 0;
		std::unique_ptr<Buffer>                    staging;
		std::vector<std::pair<uint64_t, uint64_t>> spans;   // bitmap word ranges
		std::vector<uint64_t>                      offsets; // their first word in staging
	};
	std::deque<BitmapCopy>                 m_bitmap_copies;
	std::vector<std::unique_ptr<Buffer>>   m_bitmap_staging_pool;
	// Write prediction (KYTY_WRITE_PREDICT, on with the bitmap): unbounded written bindings
	// protect only the pages the GPU wrote in recent eras; a page written outside them is
	// protected when its era's copy is processed.
	RangeSet                               m_predicted;
	uint64_t                               m_predicted_generation = 0;
	// v445c (KYTY_TRACKED_RANGE_MEMO): where m_predicted changed, oldest first, so a tracked binding
	// can tell a change inside its range from one elsewhere (ObtainBufferTracked). Kept only with
	// the memo, its verify or KYTY_QUEUE_STATS. Changes up to m_predicted_log_floor are not in it.
	// Under m_dirty_ranges_mutex, as m_predicted.
	struct PredictedChange {
		uint64_t generation = 0; // m_predicted_generation after the change
		uint64_t begin      = 0;
		uint64_t end        = 0;
	};
	std::vector<PredictedChange> m_predicted_log;
	uint64_t                     m_predicted_log_floor = 0;
	// Every change of m_predicted goes through here, right after it (it moves the generation).
	void               NotePredictedChange(uint64_t begin, uint64_t bytes);
	[[nodiscard]] bool PredictedChangedIn(uint64_t since_generation, uint64_t vaddr, uint64_t size) const;
	std::unordered_map<uint64_t, uint64_t> m_predict_seen; // page -> last era written
	std::unordered_map<uint64_t, uint64_t> m_predict_seen_blocks; // 32-page block -> last era
	struct BindingPrediction {
		uint64_t era        = 0; // era + 1 it was last marked in
		uint64_t generation = 0; // of m_predicted then
		uint32_t eras_seen  = 0; // the first eras mark it whole, to learn its writes
	};
	std::unordered_map<uint64_t, BindingPrediction> m_predict_marked;
	// Tracked bindings obtained so far, and per page the count when WaitSharedWrites last
	// protected it: a page is protected again only after another tracked binding.
	uint64_t                                        m_tracked_serial = 0;
	std::unordered_map<uint64_t, uint64_t>          m_host_read_serial;
	void               ProcessBitmapCopies();
	void               ProcessBitmapCopy(BitmapCopy& copy);
	[[nodiscard]] bool PredictWrites() const;
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void                   Unregister(BufferId id);
	template <bool insert>
	void                     ChangeRegister(BufferId id);
	void                     DeleteBuffer(BufferId id);
	[[nodiscard]] bool       SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                           bool is_written, bool is_texel_buffer);
	void SynchronizeAndMarkWritten(Buffer& buffer,
	                               std::span<const std::pair<uint64_t, uint64_t>> ranges);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// One readback attempt on the GPU thread: 0 when done, else the tick to wait for.
	[[nodiscard]] uint64_t ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
	                                      bool from_gpu_thread, bool async);
	// Game-thread only, after a normal copy-queue readback published its bytes.
	// v445s: lift = false only evaluates the predicate (same locks), for KYTY_FAST_FINISH_VERIFY.
	[[nodiscard]] bool FastFinishReadback(uint64_t vaddr, uint64_t size, bool lift = true);
	// v445s, KYTY_READBACK_CAUSE ([readback-cause]; GPU thread). What a game thread's readback is
	// about to wait for on the main queue, and after the wait whether the write bitmap shows the
	// faulting page written.
	void NoteReadbackCauseWait(ReadbackCauseSample& sample, int reason, uint64_t vaddr, uint64_t size,
	                           uint64_t window_begin, uint64_t window_end);
	void ResolveReadbackCause(const ReadbackCauseSample& sample);
	// 1 / 0: the page's bit in the write bitmap as the GPU has left it so far; -1: no bitmap there.
	[[nodiscard]] int WriteBitmapLiveBit(uint64_t page) const;
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	[[nodiscard]] bool DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	void DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
	                          uint64_t total_size);

	// Copy-queue readbacks (KYTY_COPY_QUEUE_READBACK, default on). GPU thread: when every GPU
	// write to the window has executed, copies its GPU-written bytes on the second queue and
	// returns the readback's marker (0: not taken, the caller downloads as before).
	// v445t: exact_finished, the caller has found no unfinished write reaching the window by the
	// writes' own ranges (ExactReadbackWindow); the 1 MiB block ticks are not asked.
	[[nodiscard]] uint64_t TryCopyQueueReadback(Buffer& buffer, uint64_t window_begin,
	                                            uint64_t window_end, bool exact_finished = false);
	// GPU thread: the marker of an unfinished copy-queue readback overlapping the range, or 0.
	[[nodiscard]] uint64_t FindCopyReadback(uint64_t vaddr, uint64_t size) const;
	// Any thread: waits until no copy-queue readback is reading a cache buffer (before one is
	// destroyed). Does not publish.
	void                   WaitCopyReadbacks();
	// GPU thread: the tick the bytes MarkGpuWritten records are written in, by 1 MiB block.
	void                   NoteGpuWriteTick(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t LatestGpuWriteTick(uint64_t begin, uint64_t end) const;
	// v445t (KYTY_EXACT_WRITE_RANGES / toggle mode "exactrange"; GPU thread). The writes
	// NoteGpuWriteTick is told of, each with its own range and tick, while their ticks may still
	// run (tick order). The block ticks above say "a writer still runs" for a whole 1 MiB block
	// (and the readback asks them over its 4 MiB window); this log says where.
	struct PendingWrite {
		uint64_t begin;
		uint64_t end;
		uint64_t tick;
	};
	std::deque<PendingWrite> m_pending_writes;
	// Entries dropped at the log's cap: until this tick has finished every range counts as reached.
	uint64_t m_pending_writes_overflow = 0;
	void     NotePendingWrite(uint64_t begin, uint64_t end, uint64_t tick);
	// Narrows [window_begin, window_end) to its largest part around the pages [page_begin,
	// page_end) that no unfinished write reaches, on page bounds. False when one reaches the pages
	// themselves (the readback has to wait for it) or the log cannot tell.
	[[nodiscard]] bool ExactReadbackWindow(uint64_t page_begin, uint64_t page_end, uint64_t& window_begin,
	                                       uint64_t& window_end);
	// KYTY_EXACT_WRITE_RANGES_VERIFY: the GPU's bytes of the window now, copied on the copy queue
	// (nothing else changes); the main-queue download that follows compares its bytes with them
	// when it publishes.
	[[nodiscard]] std::shared_ptr<ExactVerifySample> TakeExactVerifySample(Buffer& buffer, uint64_t page,
	                                                                      uint64_t window_begin,
	                                                                      uint64_t window_end);
	std::shared_ptr<ExactVerifySample> m_exact_verify_pending;
	struct CopyReadbackSlot {
		std::mutex                  mutex;
		std::atomic<uint32_t>       busy {0};
		uint32_t                    generation = 0;
		vk::CommandPool             pool       = nullptr;
		vk::CommandBuffer           command    = nullptr;
		vk::CommandPool             dma_pool    = nullptr; // KYTY_DMA_READBACK
		vk::CommandBuffer           dma_command = nullptr;
		vk::Fence                   fence      = nullptr;
		std::unique_ptr<Buffer>     staging;
		uint64_t                    buffer_address = 0;
		uint64_t                    window_begin   = 0;
		uint64_t                    window_end     = 0;
		uint64_t                    total_size     = 0;
		std::vector<vk::BufferCopy> copies;
		bool                        snapshot       = false; // KYTY_READ_SNAPSHOT
		// KYTY_READBACK_SUBMIT_THREAD (v445q): 0 while the worker thread has not yet submitted
		// the slot's copy; the fence is not waited for before.
		std::atomic<uint32_t>       submitted {1};
	};
	// KYTY_READ_SNAPSHOT: the GPU's current bytes of a window whose writers are still running,
	// copied without waiting for them; and the windows whose snapshot is in guest memory, waiting
	// for the GPU thread to let reads through.
	[[nodiscard]] uint64_t TrySnapshotReadback(Buffer& buffer, uint64_t window_begin,
	                                           uint64_t window_end);
	std::mutex                                        m_snapshot_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_snapshots_ready;
	static constexpr uint32_t CopyReadbackSlots = 8;
	std::array<CopyReadbackSlot, CopyReadbackSlots>   m_copy_slots;
	std::atomic<uint32_t>                             m_copy_in_flight {0};
	std::unordered_map<uint64_t, uint64_t>            m_write_ticks;
	// Writes too large to note block by block: begin, end, tick.
	std::vector<std::array<uint64_t, 3>>              m_large_writes;
	void                                              NoteHotReadback(uint64_t vaddr);
	void                                              NoteEagerWrite(uint64_t vaddr, uint64_t size);
	std::unordered_map<uint64_t, uint64_t>            m_hot_windows;
	std::unordered_set<uint64_t>                      m_eager_pending;
	uint64_t                                          m_eager_count  = 0;
	uint64_t                                          m_copy_taken   = 0;
	uint64_t                                          m_copy_busy    = 0;
	uint64_t                                          m_copy_running = 0;
	uint64_t m_dma_inflight = 0; // v314
	uint64_t m_narrow_copies = 0; // v315
	uint64_t m_narrow_refused = 0;

	GraphicContext&                                    m_graphics;
	CommandScheduler&                                  m_scheduler;
	FaultManager                                       m_fault_manager;
	Buffer                                             m_gds_buffer;
	Buffer                                             m_bda_pagetable_buffer;
	bool                                               m_bda_pagetable_cleared = false;
	Common::SlotVector<Buffer>                         m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	struct GcStats {
		uint64_t runs = 0, aggressive = 0, deleted = 0, waits = 0, created = 0;
		double   wait_ms = 0.0;
		std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	} m_gc_stats;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	// Serializes tracker marks and byte-range publication against the post-copy lift.
	std::mutex                                         m_gpu_write_state_mutex;
	// Bytes whose download is recorded but not yet in guest memory.
	RangeSet                                           m_downloading_ranges;
	// Every in-flight download's ranges, one entry each: m_downloading_ranges is their union. A
	// publication removes only its own entries (ISSUES #26: the first of two downloads of one
	// window cleared the range while the second was in flight; a game thread then took the
	// page as current, wrote it, and the second publication put older GPU bytes over that).
	std::vector<std::pair<uint64_t, uint64_t>>         m_downloading_list;
	// Whether m_downloading_ranges holds anything, set under m_dirty_ranges_mutex after each change:
	// HasPendingDownload (every clean guest read of the SRT walks and draw-prep checks) answers
	// without the lock while nothing downloads.
	std::atomic<bool>                                  m_downloading_any {false};
	void NoteDownloading() {
		m_downloading_any.store(!m_downloading_ranges.Empty(), std::memory_order_release);
	}
	// Write bitmap (WriteBitmapAddress): the buffer, the whole pages whose GPU writes all came
	// through tracked bindings, and per 4 MiB region the last tick a tracked binding wrote it.
	// Guarded by m_dirty_ranges_mutex, as m_gpu_modified_ranges.
	std::unique_ptr<Buffer>                            m_write_bitmap;
	std::unique_ptr<Buffer>                            m_srt_verify_report;
	bool                                               m_srt_verify_report_failed = false;
	uint32_t                                           m_srt_verify_reports_seen  = 0;
	std::unique_ptr<Buffer>                            m_loop_guard_report;
	bool                                               m_loop_guard_report_failed = false;
	uint32_t                                           m_loop_guard_seen          = 0;
	uint64_t                                           m_loop_guard_frame         = 0;
	std::unordered_map<uint64_t, uint32_t>             m_loop_guard_trips;
	bool                                               m_write_bitmap_failed = false;
	RangeSet                                           m_bitmap_tracked;
	std::unordered_map<uint64_t, uint64_t>             m_bitmap_ticks;
	// Tracked written bindings whose ticks may still run, in tick order (KYTY_BITMAP_RANGE_TICKS):
	// a readback waits for the writers of its window, not for any writer of its 4 MiB region.
	struct PendingBitmapWrite {
		uint64_t begin;
		uint64_t end;
		uint64_t tick;
	};
	std::deque<PendingBitmapWrite>                     m_pending_bitmap_writes;
	bool m_narrow_window = false; // RefineWithWriteBitmap: download only the faulting pages
	bool m_srt_readback  = false; // StartSrtReadback is asking (KYTY_SRT_NARROW_READBACK)
	uint64_t                                           m_bitmap_dropped_pages = 0;
	uint64_t                                           m_bitmap_written_pages = 0;
	// The tick of the latest asynchronous readback (KYTY_ASYNC_READBACK); GPU thread only.
	uint64_t                                           m_last_async_download_tick = 0;
	std::atomic<uint64_t>                              m_download_publish_tick {0}; // latest download's publication tick
	// PeekDword: a small host-visible buffer of dword copies, one slot per guest address.
	struct PeekSlot {
		uint32_t index = 0;
		uint64_t tick  = 0;
		uint32_t value = 0;
	};
	std::unique_ptr<Buffer>                            m_peek_buffer;
	std::unordered_map<uint64_t, PeekSlot>             m_peek_slots;
	// Guards changes to both range sets (GPU thread and download completions) against
	// IsCleanForConcurrentRead; the GPU thread reads them without it.
	mutable std::shared_mutex                          m_dirty_ranges_mutex;
	MemoryTracker                                      m_memory_tracker;
	StreamBuffer                                       m_staging_buffer;
	// Guest reads UploadCopies leaves to the recording thread: staging destination, guest
	// address, bytes.
	struct UploadRead {
		uint8_t* destination;
		uint64_t address;
		uint64_t size;
	};
	std::vector<UploadRead> m_upload_reads;
	StreamBuffer                                       m_stream_buffer;
	StreamBuffer                                       m_download_buffer;
	bool                                               m_vertex_verify_armed  = false;
	uint32_t                                           m_vertex_verify_ranges = 0;
	uint64_t                                           m_vertex_verify_draw   = 0;
	StreamBuffer                                       m_device_buffer;
	TextureCache&                                      m_texture_cache;
	uint64_t                                           m_total_used_memory = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	// As configured; the ones above come down while memory is short (as in the texture cache).
	uint64_t m_base_trigger_gc_memory     = 0;
	uint64_t m_base_critical_gc_memory    = 0;
	uint64_t m_applied_trigger_gc_memory  = UINT64_MAX; // as in the texture cache
	uint64_t m_applied_critical_gc_memory = UINT64_MAX;
	uint64_t m_gc_tick            = 0;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
