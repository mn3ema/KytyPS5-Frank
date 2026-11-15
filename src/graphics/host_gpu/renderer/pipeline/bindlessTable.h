#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <utility>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class PageManager;
class SamplerCache;

// The engine method for the guest's texture heaps (engine-method.md): heap entries are noticed
// when the guest writes them, through write-watched pages, instead of by a rolling scan.
// On by default; KYTY_BINDLESS_ENGINE=0 restores the rolling scan and the per-draw image walk.
[[nodiscard]] bool BindlessEngineEnabled();

// Descriptor set 1 of every pipeline that samples bindless images: typed arrays of sampled
// images a shader indexes with a slot it looks up per pixel, the key -> slot translation the
// host fills per guest descriptor heap, and a bitset where shaders flag keys whose texture is
// not resident yet. Created only when the device supports descriptor indexing.
class BindlessTable {
public:
	enum Binding : uint32_t {
		Images2D      = 0,
		Images2DArray = 1,
		ImagesCube    = 2,
		Images3D      = 3,
		Storage2D     = 4,
		Translation   = 5,
		Feedback      = 6,
		Samplers      = 7,
	};
	static_assert(Images2D == ShaderRecompiler::IR::BindlessImages2D &&
	              Images2DArray == ShaderRecompiler::IR::BindlessImages2DArray &&
	              ImagesCube == ShaderRecompiler::IR::BindlessImagesCube &&
	              Images3D == ShaderRecompiler::IR::BindlessImages3D &&
	              Storage2D == ShaderRecompiler::IR::BindlessStorage2D &&
	              Translation == ShaderRecompiler::IR::BindlessTranslation &&
	              Feedback == ShaderRecompiler::IR::BindlessFeedback &&
	              Samplers == ShaderRecompiler::IR::BindlessSamplers);
	static constexpr uint32_t ImageArrays        = 5; // the sampled arrays and Storage2D
	static constexpr uint32_t SampledImageArrays = 4;
	static constexpr uint32_t PlaceholderColors  = 3;
	static constexpr uint32_t Placeholders       = ImageArrays * PlaceholderColors;
	static constexpr uint32_t MaxImagesPerArray  = 65536; // E3 settles every entry of a heap
	static_assert(MaxImagesPerArray == ShaderRecompiler::IR::BindlessMaxSlots);
	static constexpr uint32_t TranslationEntries = 1u << 20u;
	static constexpr uint32_t MaxSamplers        = 4096;

	BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager);
	~BindlessTable();
	KYTY_CLASS_NO_COPY(BindlessTable);

	[[nodiscard]] bool                    Enabled() const noexcept { return m_set != nullptr; }
	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] vk::DescriptorSet       Set() const noexcept { return m_set; }
	[[nodiscard]] uint32_t                ImagesPerArray() const noexcept { return m_images_per_array; }
	// Slots of one array: the storage array may be smaller (device limits), or only its
	// placeholders when the device cannot index storage images bindlessly.
	[[nodiscard]] uint32_t SlotsPerArray(uint32_t binding) const noexcept {
		return binding == Storage2D ? m_storage_per_array : m_images_per_array;
	}
	[[nodiscard]] static vk::DescriptorType DescriptorTypeFor(uint32_t binding) noexcept {
		return binding == Storage2D ? vk::DescriptorType::eStorageImage
		                            : vk::DescriptorType::eSampledImage;
	}

	// A guest descriptor heap as one view type samples it: a region of the translation (and
	// feedback) buffer, one entry per key, and the textures resolved so far.
	struct Heap {
		uint64_t                            base         = 0;
		uint32_t                            table_offset = 0;
		uint32_t                            binding      = 0;
		uint32_t                            region       = 0;
		// Keys the region holds. The guest appends textures to a heap as it streams them in and
		// its descriptor grows; each draw patches its own descriptor's entry count.
		uint32_t                            entries      = 0;
		// The most keys a draw's heap descriptor has covered (entries adds headroom to it).
		uint32_t                            guest_entries = 0;
		// Engine method: the entries WatchHeap watched, and the watch generation then. A draw calls
		// WatchHeap only when either differs (the heap grew, or an unmap dropped its watch).
		uint32_t                            watched_entries  = 0;
		uint64_t                            watch_generation = 0;
		// Engine method, E3: the next key the translation pass looks at, and whether a full pass
		// found nothing left to settle (cleared when the heap grows or the guest writes it).
		uint32_t                            eager_cursor   = 0;
		bool                                eager_complete = false;
		// Per key: how often E3 settled it since the guest last wrote it. A key whose image the
		// texture cache keeps replacing (format aliases of one texture, or one texture under two
		// view types) would be settled again every frame; past MaxEagerSettles it is left to
		// the request path until the guest rewrites it.
		std::vector<uint8_t>                eager_count;
		// Keys whose T# only changed view fields (streaming lowers min LOD) while no draw sampled
		// them: they keep their slot, the old view of the same texture, as an engine keeps a
		// texture's descriptor until finer mips are needed, and are settled again once a draw
		// samples them.
		std::vector<uint8_t>                stale;
		std::vector<uint32_t>               stale_keys;
		ShaderRecompiler::IR::ImageResource resource;
		// Per key, 0 = not resolved; else the translation entry: the slot in this heap's binding,
		// or (BindlessSlotArrayFlag) in the 2D-array binding. SlotBinding/SlotIndex decode it.
		std::vector<uint32_t>               slots;
		std::vector<uint8_t>                settled;  // per key; resolved, or known placeholder
		std::vector<ImageId>                resolved; // images to keep readable for draws
		std::vector<ImageId>                unchecked; // settled since the last bindless draw (E5)
		// Per settled key, the T# it was settled from and the image it resolved to (none for a
		// placeholder). The guest rewrites entries as it streams textures out and others in;
		// RenderExecutor::RevalidateBindlessKeys settles a key whose T# changed again.
		std::vector<std::array<uint32_t, 8>> descriptors;
		std::vector<ImageId>                 images;
		uint32_t                             revalidate_cursor = 0; // next key to compare
		// What RenderExecutor::CommitBindings last checked: g_bindless_state_generation then, and
		// how many resolved images. Unchanged since, only the images resolved after need a look.
		uint64_t checked_generation = 0;
		size_t   checked_count      = 0;
	};

	// The heap for (base, table offset, view binding), created with every entry pending and
	// room to grow, or moved to a larger region when a descriptor covers more keys; null when
	// the translation buffer is full.
	[[nodiscard]] Heap* FindOrCreateHeap(uint64_t base, uint32_t table_offset, uint32_t binding,
	                                     uint32_t entries,
	                                     const ShaderRecompiler::IR::ImageResource& resource);
	[[nodiscard]] std::deque<Heap>& Heaps() noexcept { return m_heaps; }
	// v375 (device loss): every heap key whose T# in guest memory now points into [lo, hi), and
	// every key Kyty settled from such a T#, with its slot, translation entry and resolved image.
	void DescribeKeysForGuestRange(uint64_t lo, uint64_t hi, FILE* out) const;
	// v376 (KYTY_SLOT_CAPTURE=1, device loss): the translation value the GPU last read for each
	// heap key against the host's, and the slots the GPU read that the host has since rewritten.
	void CompareSlotCapture(FILE* out) const;
	// KYTY_CANARY_DETECT=1 (v378): new canary hits the shaders left in the feedback buffer
	// (at most every 100 ms unless forced).
	void ReportCanaryHits(bool force) const;
	// 0 when the array is full. A released slot is reused once the GPU has finished the work
	// recorded before its release.
	// The binding and array slot a Heap::slots entry names.
	[[nodiscard]] static uint32_t SlotBinding(const Heap& heap, uint32_t entry) {
		return (entry & ShaderRecompiler::IR::BindlessSlotFlagMask) ==
		               ShaderRecompiler::IR::BindlessSlotArrayFlag
		           ? static_cast<uint32_t>(Images2DArray)
		           : heap.binding;
	}
	[[nodiscard]] static uint32_t SlotIndex(uint32_t entry) {
		return (entry & ShaderRecompiler::IR::BindlessSlotFlagMask) ==
		               ShaderRecompiler::IR::BindlessSlotArrayFlag
		           ? entry & ~ShaderRecompiler::IR::BindlessSlotFlagMask
		           : entry;
	}
	// Placeholder 0 (transparent, or grey with KYTY_BINDLESS_DEBUG_COLORS) of a binding's view type.
	[[nodiscard]] vk::ImageView PlaceholderView(uint32_t binding) const {
		return m_placeholder_views[binding];
	}
	[[nodiscard]] uint32_t AllocateSlot(uint32_t binding);
	// image/heap_base/key: what the slot now holds, for the fault report's image history (v122);
	// null image = a placeholder.
	// width/height/layers: the view's level-0 size, what textureSize reports for it (1x1x1 for a
	// placeholder), for the sweep's verification.
	void WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view, vk::ImageLayout layout,
	               VkImage image = VK_NULL_HANDLE, uint64_t heap_base = 0, uint32_t key = 0,
	               uint32_t width = 1, uint32_t height = 1, uint32_t layers = 1);
	// Resolve a key to a slot (0 = placeholder), or back to pending.
	void SetTranslation(const Heap& heap, uint32_t key, uint32_t slot);
	// The host reads the feedback flags from a copy in host-cached memory. The feedback buffer
	// itself is device memory, and reading every key of three heaps from it uncached took a
	// fifth of the GPU thread. Records that copy at the current point of the command stream.
	void RecordFeedbackSnapshot(CommandScheduler& scheduler);
	// KYTY_BINDLESS_SWEEP=1 (v122): one dispatch per image array that fetches a texel from every
	// allocated slot, so a descriptor of a destroyed image faults in that dispatch within a frame
	// (the fault report then names the slot through the image history) instead of in whichever
	// draw happens to index it seconds later. Call once a frame.
	void RecordSlotSweep(CommandScheduler& scheduler, uint64_t frame);
	// The slots the last sweep had not finished when the GPU stopped (frame = the sweep's).
	void ReportSweep(uint64_t frame) const;
	[[nodiscard]] bool SnapshotRecorded() const noexcept { return m_snapshot_tick != 0; }
	// Whether the GPU has executed the recorded copy; never waits.
	[[nodiscard]] bool SnapshotReady(CommandScheduler& scheduler);
	void               ConsumeSnapshot() noexcept { m_snapshot_tick = 0; }
	// Pending keys flagged in the snapshot; their flags are cleared.
	void TakeRequests(const Heap& heap, std::vector<uint32_t>& keys);
	// Diagnostics: feedback word 0 (the last out-of-range key, top bit set) in the snapshot,
	// cleared.
	[[nodiscard]] uint32_t TakeWordZero();
	// v127: the last translation value that named no slot (0: none) in the snapshot, cleared.
	[[nodiscard]] uint32_t TakeOutOfRangeSlot();
	// v301: a feedback word's snapshot value, clearing the word when non-zero ([indirect-oob]).
	[[nodiscard]] uint32_t TakeFeedbackWord(uint32_t word);
	// Diagnostics (KYTY_KEY_CAPTURE): the non-zero snapshot words of [first, first + count), as
	// (index - first, word); cleared.
	void TakeCapture(uint32_t first, uint32_t count, std::vector<std::pair<uint32_t, uint32_t>>& marked);
	// The last feedback snapshot's words [first, first + count), unchanged.
	void ReadSnapshot(uint32_t first, uint32_t count, std::vector<uint32_t>& words);
	// Usage probe. Every draw that samples a key stores its feedback word (0 for a resident
	// texture, 1 for a pending one), so marking every resident key's word with ProbeMark and
	// reading a snapshot some frames later shows which keys no draw sampled in between.
	// ArmUsageProbe marks them; CollectUnusedImages lists the images all of whose keys still
	// carry the mark in the current snapshot.
	static constexpr uint32_t ProbeMark = 2;
	uint32_t                  ArmUsageProbe(); // v388: returns the keys marked
	// Engine method, E3: a key settled before any draw asked for it carries the probe mark until a
	// draw samples it (the shader stores 0). WasSampled reads the last snapshot: true unless the
	// mark is still there. FlushFeedback publishes marks set with MarkUnsampled.
	void               MarkUnsampled(const Heap& heap, uint32_t key);
	void               FlushFeedback(const Heap& heap);
	[[nodiscard]] bool WasSampled(const Heap& heap, uint32_t key) const;
	void                      CollectUnusedImages(std::vector<ImageId>& unused);
	[[nodiscard]] size_t      ReferencedImageCount() const noexcept { return m_image_refs.size(); }
	// v389: tells the shaders a usage probe runs (BindlessProbeActiveWord), so sparse feedback
	// stores the zeros that clear the probe's marks.
	void SetProbeActive(bool active);
	// The opposite, for the texture dump: images with a key some draw sampled since the marks.
	void                      CollectSampledImages(std::vector<ImageId>& used);
	void AddImageReference(ImageId id, Heap& heap, uint32_t key);
	// The key no longer samples what it was settled to: it is pending again, and its image loses
	// the key's reference. Its slot keeps the old view, which frames in flight may still sample;
	// a new resolution takes a new slot. True when the image has no reference left, so it need
	// not stay pinned in the texture cache.
	[[nodiscard]] bool ReleaseKey(Heap& heap, uint32_t key);

	// A guest sampler heap as one kind of use samples it: a region of the sampler array that
	// mirrors its S# records, key for key. Uses differ in whether the depth-compare function is
	// kept, whether point filtering is forced and whether border colours are integer, so each
	// kind gets its own region.
	static constexpr uint32_t SamplerDepthCompare   = 1u;
	static constexpr uint32_t SamplerPointFiltering = 2u;
	static constexpr uint32_t SamplerIntegerBorder  = 4u;
	struct SamplerHeap {
		uint64_t                             base          = 0;
		uint32_t                             table_offset  = 0;
		uint32_t                             flags         = 0;
		uint32_t                             region        = 0; // first slot; 0 = none yet
		uint32_t                             capacity      = 0;
		std::vector<std::array<uint32_t, 4>> records;           // mirrored S# records
		uint64_t                             checked_frame = UINT64_MAX;
	};
	[[nodiscard]] SamplerHeap* FindOrCreateSamplerHeap(uint64_t base, uint32_t table_offset,
	                                                   uint32_t flags);
	// Mirrors the heap's records into its region. New records are appended in place (their slots
	// were never used); a changed record, or more records than the region holds, moves the heap
	// to a new region, because slots the GPU may be reading are never rewritten. False when the
	// sampler array is full; the heap then keeps its previous records.
	bool MirrorSamplerHeap(SamplerHeap& heap, std::span<const std::array<uint32_t, 4>> records,
	                       SamplerCache& cache);
	// Slot 0 of the sampler array, used for keys outside their heap; written once.
	void WriteDefaultSampler(vk::Sampler sampler);
	[[nodiscard]] bool SamplersEnabled() const noexcept { return m_samplers_per_array != 0; }
	// Engine method, E1. The guest pages holding a heap's entries are write-watched; a CPU write
	// to one (a page fault on a game thread, or an explicit invalidation such as a file read into
	// the heap) stops watching that page and records it as written. TakeWrittenRanges, on the GPU
	// thread at the start of a guest submission, watches the written pages again before it
	// returns them, so a write after that is caught by the next submission; the caller then
	// reads the entries and compares them with the mirror. Heaps of different view bindings over
	// the same guest table share one watch.
	void WatchHeap(Heap& heap);
	[[nodiscard]] bool NeedsWatch(const Heap& heap) const noexcept {
		return heap.watched_entries != heap.guest_entries ||
		       heap.watch_generation != m_watch_generation.load(std::memory_order_acquire);
	}
	void OnCpuWrite(uint64_t vaddr, uint64_t size); // any thread
	void UnwatchRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasWrittenRanges() const noexcept {
		return m_heap_writes.load(std::memory_order_acquire);
	}
	void TakeWrittenRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges);

	// The texture cache dropped a resolved image: point its slots back at the placeholder and
	// make its keys pending again, so a draw that still needs it asks for it anew.
	void OnImageUnregistered(ImageId id);
	// Keys made pending because the texture cache dropped their image (statistics; reset on read).
	[[nodiscard]] uint64_t TakeUnregisteredKeys() noexcept { return std::exchange(m_unregistered_keys, 0); }
	// Before an image is destroyed: a slot that still holds one of its views is repointed to the
	// placeholder and the keys that use it are pending again. True when one did, so the caller
	// keeps the image until the GPU has finished the work recorded with that slot.
	// KYTY_FAULT_MAP=1: names the slots that still hold one of these views.
	void ReportHeldViews(std::span<const vk::ImageView> views, const char* what);
	[[nodiscard]] bool ReleaseViews(std::span<const vk::ImageView> views);
	// KYTY_FAULT_MAP=1, at device loss: every translation entry (live heap regions and the regions
	// heaps moved away from) against what the CPU thinks its slot holds: destroyed views, free
	// slots, slots past the allocated range ([bindless-audit]).
	void AuditForFault() const;
	// C4b: queued translation entries become visible to the GPU (after each vkQueueSubmit).
	void PublishTranslations();
	// v384 (KYTY_BINDLESS_PUBLISH_SAFE=1): a translation entry naming a slot becomes visible to the
	// GPU only once every submission made before its descriptor write has completed (GPU thread).
	void PublishSafe();

private:
	[[nodiscard]] bool AllocateRegion(Heap& heap, uint32_t entries);
public:

private:
	// Slot 0 of every image array: a 1x1 grey texture of that view type, sampled for keys
	// outside a heap and while a texture is pending. Created, cleared and made read-only once.
	void CreatePlaceholders(CommandScheduler& scheduler);

	[[nodiscard]] static uint64_t ImageKey(ImageId id) {
		return static_cast<uint64_t>(id.index) | (static_cast<uint64_t>(id.generation) << 32u);
	}

	// A key gave up its slot: free it for reuse after the GPU passes the current tick.
	void ReleaseSlot(uint32_t binding, uint32_t slot);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	PageManager&            m_page_manager;
	// Write-watched guest ranges of the heaps, page by page (TRACKER_PAGE_SIZE).
	struct Watch {
		uint64_t             begin = 0;
		uint64_t             end   = 0;
		std::vector<uint8_t> watched;
		std::vector<uint8_t> written;
	};
	std::mutex            m_watch_mutex;
	std::vector<Watch>    m_watches;
	std::atomic_bool      m_heap_writes {false};
	// Bounds of every watch, so a write far from the heaps returns without the lock.
	std::atomic<uint64_t> m_watch_low {UINT64_MAX};
	std::atomic<uint64_t> m_watch_high {0};
	std::atomic<uint64_t> m_watch_generation {1};
	uint64_t              m_unregistered_keys = 0;
	std::deque<Heap>        m_heaps;
	std::array<std::deque<std::pair<uint32_t, uint64_t>>, ImageArrays> m_free_slots; // slot, tick
	std::unordered_map<uint64_t, std::vector<std::pair<Heap*, uint32_t>>> m_image_refs;
	uint32_t                m_next_region = 1; // translation[0] is the out-of-range entry
	std::array<uint32_t, ImageArrays> m_next_slot {PlaceholderColors, PlaceholderColors,
	                                               PlaceholderColors, PlaceholderColors,
	                                               PlaceholderColors};
	std::array<VulkanImage, Placeholders>   m_placeholders;
	std::array<vk::ImageView, Placeholders> m_placeholder_views {};
	vk::DescriptorPool      m_pool   = nullptr;
	vk::DescriptorSetLayout m_layout = nullptr;
	vk::DescriptorSet       m_set    = nullptr;
	uint32_t                m_images_per_array = 0;
	uint32_t                m_storage_per_array = 0;
	uint32_t                m_samplers_per_array = 0;
	uint32_t                m_next_sampler_slot  = 1; // slot 0 is the default sampler
	bool                    m_default_sampler_written = false;
	std::deque<SamplerHeap> m_sampler_heaps;
	std::unique_ptr<Buffer> m_translation;
	std::unique_ptr<Buffer> m_feedback;
	std::unique_ptr<Buffer> m_feedback_snapshot;
	uint64_t                m_snapshot_tick = 0; // 0 = none recorded
	// The view each image slot holds, and the slots each view is in (placeholders excluded).
	std::array<std::vector<vk::ImageView>, ImageArrays>                        m_slot_views;
	std::array<std::vector<VkImage>, ImageArrays>                              m_slot_images; // v122
	// v124: per slot, the size the host expects textureSize to report, and the tick it was written.
	struct SlotExpect {
		uint32_t width = 1, height = 1, layers = 1, key = 0;
		uint64_t tick = 0, heap = 0;
	};
	std::array<std::vector<SlotExpect>, ImageArrays> m_slot_expect;
	// v383 (KYTY_CANARY_DETECT): the last writes of each slot, printed with a canary hit on it.
	struct SlotWrite {
		uint64_t    tick = 0;
		double      ms   = 0;
		VkImageView view = VK_NULL_HANDLE;
		VkImage     image = VK_NULL_HANDLE;
		uint32_t    width = 0, height = 0, layers = 0, key = 0;
		uint64_t    heap = 0;
	};
	struct SlotLog {
		std::array<SlotWrite, 8> writes {};
		uint32_t                 next = 0;
	};
	std::array<std::vector<SlotLog>, ImageArrays> m_slot_log;
	std::unordered_map<VkImageView, std::vector<std::pair<uint32_t, uint32_t>>> m_view_slots;
	// C4b: a slot whose image goes away is not rewritten while in-flight draws may index it: its
	// keys turn pending first, and the placeholder is written (and the slot freed) once the GPU has
	// finished the current tick (RetireSlot). Slots waiting for that, as binding << 32 | slot.
	void                                RetireSlot(uint32_t binding, uint32_t slot);
	std::unordered_map<uint64_t, bool>  m_retiring;
	// v122 sweep (RecordSlotSweep).
	vk::DescriptorSetLayout m_sweep_set_layout      = nullptr;
	vk::PipelineLayout      m_sweep_pipeline_layout = nullptr;
	vk::Pipeline            m_sweep_pipeline        = nullptr;
	std::unique_ptr<Buffer> m_sweep_sink;
	uint64_t                m_sweep_frame = 0;
	uint64_t                m_sweep_tick  = 0;
	uint64_t                m_verify_mismatches = 0, m_verify_checked = 0, m_verify_frames = 0;
	void                    VerifySweep();
	std::array<uint32_t, ImageArrays> m_sweep_counts {};
	// C4b: translation writes queued until the next vkQueueSubmit (KYTY_BINDLESS_PUBLISH_AT_SUBMIT).
	mutable std::mutex                          m_publish_mutex;
	std::vector<std::pair<uint32_t, uint32_t>>  m_publish; // entry, value
	struct SafeEntry {
		uint32_t entry = 0, value = 0;
		uint64_t tick  = 0; // the tick being recorded when the slot's descriptor was written
		uint64_t generation = 0;
	};
	std::deque<SafeEntry> m_publish_safe; // under m_publish_mutex
	// v385: the latest queued generation per entry (a later value supersedes queued ones in O(1)).
	std::unordered_map<uint32_t, uint64_t> m_publish_safe_latest;
	uint64_t m_publish_safe_generation = 0;
	uint64_t m_publish_safe_published = 0, m_publish_safe_superseded = 0, m_publish_safe_max = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
