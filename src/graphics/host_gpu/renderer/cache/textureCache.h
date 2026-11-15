#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include <chrono>
#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

class TextureCache {
public:
	// Called before a bindless-pinned image is unregistered.
	std::function<void(ImageId)> on_bindless_unregister;
	// Called right before an image is destroyed; true keeps it one more GPU tick (a bindless slot
	// still held one of its views and was repointed).
	std::function<bool(const Image&)> on_image_release;
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	// Texture dump (KYTY_DUMP_TEXTURES_TRIGGER): textures bound in the frame after `frame`, plus
	// the bindless images the usage probe saw sampled (AddTextureDumpImages), are written out.
	void BeginTextureDump(uint64_t frame);
	// KYTY_WATCH_IMAGE_ADDR (descriptors.cpp): copies every layer of mip 0 of one image out, with
	// the guest bytes at the same moment, and writes both to _ShaderDump once the copy lands.
	void WatchImageDump(ImageId id, uint64_t frame, const char* tag = "");
	// Diagnostics (KYTY_GBUF_PATCHES): an 8x5 grid of 16x16 patches of the image, read back
	// in command order; PollWatchImageDumps prints one hash a patch ([gpatch]).
	void WatchImagePatches(ImageId id, uint64_t frame, std::string label);
	void PollWatchImageDumps();
	void AddTextureDumpImages(std::span<const ImageId> ids);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	// Null when the slot is free (diagnostics outside the renderer, KYTY_PRESENT_RT).
	[[nodiscard]] Image*        TryGetImage(ImageId id) { return m_slot_images.try_get(id); }
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	// A shader bound a written storage buffer starting at address: images that start there and
	// hold guest-memory contents (not GPU-rendered ones, whose data lives only in the image) are
	// refreshed from the buffer before their next use.
	void               NoteShaderBufferWrite(uint64_t address);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	void RunGarbageCollector();
	// The guest no longer refers to this bindless image from any heap entry (it replaced or
	// cleared them): free it once no draw has used it for a few frames (engine method).
	void ReleaseDroppedByGuest(ImageId id);
	// Bindless images no draw sampled during the last usage probe (BindlessTable::
	// CollectUnusedImages). They stay pinned, but under memory pressure the collector may free
	// them like any other image; a key that samples one again asks for it anew. Replaces the
	// previous set.
	void SetBindlessEvictable(std::span<const ImageId> ids);
	// Whether the cache's images are above the collector's pressure threshold.
	// All images, pinned ones included, against the bound: eager settling pauses on it. Settling
	// ahead while the collector freed other images under the bound lost the device at ~57 s into
	// a new game, every time (2026-10-07, ISSUES #17); the collector's own thresholds use
	// CollectableMemory.
	[[nodiscard]] bool UnderPressure() const noexcept {
		return m_total_used_memory >= m_pressure_gc_memory;
	}
	// v386 [mem-census]: what the cache holds.
	[[nodiscard]] uint64_t TotalUsedMemory() const noexcept { return m_total_used_memory; }
	[[nodiscard]] uint64_t PinnedMemory() const noexcept { return m_pinned_bytes; }
	[[nodiscard]] size_t   ImageCount() const noexcept { return m_slot_images.size(); }
	[[nodiscard]] uint64_t PressureThreshold() const noexcept { return m_pressure_gc_memory; }
	[[nodiscard]] size_t   EvictableCount() const noexcept { return m_bindless_evictable.size(); }
	// v387: the collector's totals since start ([mem-census]): images freed, MB freed, of them
	// bindless ones (unsampled in the last usage probe), GPU-written ones written back first.
	struct GcTotals {
		uint64_t freed = 0, freed_mb = 0, bindless = 0, written_back = 0, trim_runs = 0, trim_backoffs = 0,
		         trim_gpu_skipped = 0;
	};
	[[nodiscard]] GcTotals GetGcTotals() const;
	// Image bytes the collector may free: all but the bindless-pinned ones (refreshed once a frame).
	// v392: under memory pressure (the trim mark or the shared shortfall) the pinned images the
	// last usage probe found unsampled count too: the collector may free them then, and with only
	// the unpinned bytes counted it stopped short of them (an early return under 512 MB) while
	// the pinned ones were what grew.
	[[nodiscard]] uint64_t CollectableMemory() const noexcept {
		const uint64_t fixed = m_pinned_bytes - std::min(m_pinned_bytes, m_pressure_counts_evictable ? m_evictable_bytes : 0);
		return m_total_used_memory - std::min(m_total_used_memory, fixed);
	}

private:
	enum class TransferDirection { Upload, Download };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      ReleaseImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	// evicting: the collector writes the image back before freeing it; the write is dropped if
	// the guest bytes change before it lands (KYTY_EVICT_WRITEBACK_GUARD).
	// `landed`, when given, is set once the write-back has reached guest memory (or was dropped
	// because the guest wrote the bytes first): an evicted image is freed only then.
	[[nodiscard]] bool DownloadImageMemory(ImageId id, bool evicting = false,
	                                       std::shared_ptr<std::atomic<bool>>* landed = nullptr);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	// KYTY_DUMP_TEXTURES_TRIGGER=<file>: the textures sampled in the frame after the file appears
	// are copied out (mip 0, layer 0) and written to _ShaderDump/texdump with their guest layout.
	struct TextureDump {
		ImageId                 id;
		std::unique_ptr<Buffer> buffer;
		uint64_t                bytes = 0;
	};
	void                     DumpTextureNote(ImageId id); // caller holds m_lock
	struct WatchDump {
		ImageId                 id;
		std::unique_ptr<Buffer> buffer;
		uint64_t                bytes = 0;
		uint64_t                tick  = 0;
		uint64_t                frame = 0;
		std::string             tag;
		std::vector<uint8_t>    guest;
	};
	std::vector<WatchDump>   m_watch_dumps;
	struct WatchPatches {
		std::unique_ptr<Buffer> buffer;
		uint64_t                bytes = 0;
		uint64_t                tick  = 0;
		uint64_t                frame = 0;
		uint32_t                patch_bytes = 0;
		uint32_t                count       = 0;
		std::string             label;
	};
	std::vector<WatchPatches> m_watch_patches;
	uint64_t                 m_dump_checked_frame = UINT64_MAX;
	uint64_t                 m_dump_frame         = UINT64_MAX;
	uint64_t                 m_dump_tick          = 0;
	bool                     m_dump_done          = false;
	bool                     m_dump_wait_bindless = false;
	std::vector<ImageId>     m_dump_ids;
	std::vector<TextureDump> m_dumps;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::map<std::pair<vk::Format, Prospero::ImageType>, ImageId> m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::unordered_set<ImageId>                       m_bindless_evictable;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	// The thresholds as configured; the ones above come down from them while memory is short
	// (GraphicContext::MemoryShortfall: device use past 90 % of the video memory target).
	uint64_t                                          m_base_trigger_gc_memory  = 0;
	uint64_t                                          m_base_pressure_gc_memory = 0;
	uint64_t                                          m_base_critical_gc_memory = 0;
	// What the collector last wrote into the thresholds: values that differ were set anew
	// (the constructor's caps, a test's configuration) and become the base.
	uint64_t                                          m_applied_trigger_gc_memory  = UINT64_MAX;
	uint64_t                                          m_applied_pressure_gc_memory = UINT64_MAX;
	uint64_t                                          m_applied_critical_gc_memory = UINT64_MAX;
	// Engine method: images the guest dropped from its texture heaps (no key refers to them any
	// more), freed once no draw has used them for a few frames, whatever the memory pressure.
	uint64_t                                          m_pinned_bytes = 0;
	uint64_t                                          m_evictable_bytes = 0; // v392: pinned and in m_bindless_evictable
	bool                                              m_pressure_counts_evictable = false; // v392
	uint64_t                                          m_pinned_frame = UINT64_MAX;
	std::vector<ImageId>                              m_guest_dropped;
	uint64_t                                          m_guest_dropped_freed       = 0;
	uint64_t                                          m_guest_dropped_freed_bytes = 0;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	// Evictions are budgeted per frame, not per collector run: see RunGarbageCollector.
	uint64_t         m_gc_budget_frame        = 0;
	uint64_t         m_gc_freed_bytes_frame   = 0;
	// v388: the trim (device memory past KYTY_VRAM_TRIM_PERCENT, below the shared 90 % shortfall)
	// frees only images it can drop without a write-back, and backs off for 2 s after a pass that
	// freed nothing (the 70 % test spun tens of thousands of passes writing GPU images back).
	bool                                  m_trim_only = false;
	uint64_t                              m_trim_freed_bytes_run = 0;
	std::chrono::steady_clock::time_point m_trim_backoff_until {};
	size_t           m_gc_freed_images_frame  = 0;
	uint64_t         m_gc_written_back_bytes_frame = 0;
	// Two-phase eviction of GPU-written images: the write-back is deferred until the GPU is done,
	// and the image stays registered until it has landed. Freeing it at once let a draw in between
	// recreate it from the old guest bytes, without what the GPU had drawn into it (the glyph
	// atlases: "half of the text is missing", 2026-10-09). One used again in the meantime stays.
	struct PendingEviction {
		ImageId                            id;
		uint64_t                           address = 0; // the slot may be reused: checked
		uint64_t                           frame   = 0; // frame_accessed_last at eviction
		std::shared_ptr<std::atomic<bool>> landed;
	};
	std::vector<PendingEviction> m_pending_evictions;
	std::unordered_set<ImageId>  m_evicting;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents (a loading screen) still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
