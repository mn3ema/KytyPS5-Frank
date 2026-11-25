#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include "graphics/host_gpu/renderer/image/imageHistory.h"
#include "kernel/memory.h"
#include "gpu_tiler_shaders/bindless_sweep_spv.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <atomic>

#include <cstdio>
#include <cinttypes>
#include <mutex>
#include <set>
#include <string>
#include <chrono>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <array>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

bool        FaultMapImageVa(const void* image, uint64_t& va, uint64_t& size); // streamBuffer.cpp
std::string FaultMapImageRange(const void* image);                            // streamBuffer.cpp

namespace {
// C4b parts, opt-in since 10-24 (B0: no effect on the freed-memory crash in 12 romp runs).
bool PublishAtSubmit() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PUBLISH_AT_SUBMIT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// v384: KYTY_BINDLESS_PUBLISH_SAFE=1. Crash B (C22): draws already submitted read translation
// entries live, so an entry written while they ran sent them to a slot whose descriptor was written
// after their submission; the driver then resolved it to a recycled texture header of a destroyed
// render target, and the read faulted once that memory was freed. Vulkan allows an update after
// bind to reach only submissions made after it: the entry now waits until every submission made
// before the descriptor write has completed. Pending entries (a slot going away) apply at once.
bool PublishSafeEnabled() {
	// v386: on by default (C23, Mac's 26 + 22 min sessions: no Crash B); =0 restores live writes.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PUBLISH_SAFE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}
// v383: KYTY_CANARY_DETECT (the recompiler's switch) also keeps a write log per slot.
bool CanaryDetectEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CANARY_DETECT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool DeferredRetire() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_DEFERRED_RETIRE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// KYTY_BINDLESS_BLIND=1 (v374, Crash B bisection, diagnostics only): keys resolve, textures load
// and slots are written exactly as usual, but every translation entry the GPU sees names the grey
// placeholder (slot 0) instead of the key's slot, so no draw ever reads a texture through a
// bindless slot. With v373's KYTY_BINDLESS_NONE (no textures at all) this separates "the GPU reads
// through the table" from "the texture load changes memory and timing".
uint32_t VisibleSlot(uint32_t slot) {
	static const bool blind = [] {
		const char* value = std::getenv("KYTY_BINDLESS_BLIND");
		return value != nullptr && value[0] == '1';
	}();
	return blind && slot != ShaderRecompiler::IR::BindlessPending ? 0u : slot;
}
inline void StoreFence() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_sfence();
#else
	std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}
} // namespace


// image.cpp (KYTY_FAULT_MAP=1): false, and a report, for a view of a destroyed image.
bool CheckLiveImageView(vk::ImageView view, const char* where);
bool IsImageViewLive(VkImageView view);       // image.cpp
std::string DescribeViewForFault(uint64_t view); // image.cpp

bool BindlessEngineEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void BindlessTable::WatchHeap(Heap& heap) {
	heap.watched_entries  = heap.guest_entries;
	heap.watch_generation = m_watch_generation.load(std::memory_order_acquire);
	const auto table = heap.base + heap.table_offset;
	const auto begin = Common::AlignDown(table, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(table + uint64_t {heap.guest_entries} * 32u, TRACKER_PAGE_SIZE);
	if (end <= begin) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	auto watch = std::ranges::find_if(m_watches, [&](const Watch& w) { return w.begin == begin; });
	if (watch == m_watches.end()) {
		watch        = m_watches.insert(m_watches.end(), Watch {});
		watch->begin = begin;
		watch->end   = begin;
	}
	heap.eager_complete = false; // new or more entries to settle
	if (end <= watch->end) {
		return;
	}
	// The heap grew (or is new): watch the pages it now covers.
	const auto first = static_cast<size_t>((watch->end - begin) / TRACKER_PAGE_SIZE);
	const auto pages = static_cast<size_t>((end - begin) / TRACKER_PAGE_SIZE);
	watch->watched.resize(pages, 0u);
	watch->written.resize(pages, 0u);
	watch->end = end;
	for (size_t page = first; page < pages; page++) {
		watch->watched[page] = 1;
	}
	m_page_manager.UpdatePageWatchers<true>(begin + first * TRACKER_PAGE_SIZE,
	                                        (pages - first) * TRACKER_PAGE_SIZE);
	if (begin < m_watch_low.load(std::memory_order_relaxed)) {
		m_watch_low.store(begin, std::memory_order_release);
	}
	if (end > m_watch_high.load(std::memory_order_relaxed)) {
		m_watch_high.store(end, std::memory_order_release);
	}
}

void BindlessTable::OnCpuWrite(uint64_t vaddr, uint64_t size) {
	// Called for every CPU write fault: most are nowhere near a heap and must not take the lock.
	if (vaddr + size <= m_watch_low.load(std::memory_order_acquire) ||
	    vaddr >= m_watch_high.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	for (auto& watch: m_watches) {
		const auto begin = std::max(watch.begin, Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
		const auto end   = std::min(watch.end, vaddr + size);
		for (auto address = begin; address < end; address += TRACKER_PAGE_SIZE) {
			const auto page = static_cast<size_t>((address - watch.begin) / TRACKER_PAGE_SIZE);
			if (watch.watched[page] == 0) {
				continue;
			}
			// The faulting write must go through: stop watching the page until the next sync.
			watch.watched[page] = 0;
			watch.written[page] = 1;
			m_page_manager.UpdatePageWatchers<false>(address, TRACKER_PAGE_SIZE);
			m_heap_writes.store(true, std::memory_order_release);
		}
	}
}

void BindlessTable::UnwatchRange(uint64_t vaddr, uint64_t size) {
	std::scoped_lock lock {m_watch_mutex};
	for (auto& watch: m_watches) {
		if (vaddr + size <= watch.begin || vaddr >= watch.end) {
			continue;
		}
		// Part of the table is unmapped: drop the whole watch. WatchHeap builds it again the
		// next time a draw uses the heap, over what is mapped then.
		for (size_t page = 0; page < watch.watched.size(); page++) {
			if (watch.watched[page] != 0) {
				m_page_manager.UpdatePageWatchers<false>(watch.begin + page * TRACKER_PAGE_SIZE,
				                                         TRACKER_PAGE_SIZE);
			}
		}
		watch.watched.clear();
		watch.written.clear();
		watch.end = watch.begin;
		m_watch_generation.fetch_add(1, std::memory_order_acq_rel);
	}
}

void BindlessTable::TakeWrittenRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	ranges.clear();
	std::scoped_lock lock {m_watch_mutex};
	m_heap_writes.store(false, std::memory_order_release);
	for (auto& watch: m_watches) {
		for (size_t page = 0; page < watch.written.size(); page++) {
			if (watch.written[page] == 0) {
				continue;
			}
			const auto address  = watch.begin + page * TRACKER_PAGE_SIZE;
			watch.written[page] = 0;
			// Watched again before the caller reads the entries: a later write is not lost.
			if (watch.watched[page] == 0) {
				watch.watched[page] = 1;
				m_page_manager.UpdatePageWatchers<true>(address, TRACKER_PAGE_SIZE);
			}
			if (!ranges.empty() && ranges.back().second == address) {
				ranges.back().second += TRACKER_PAGE_SIZE;
			} else {
				ranges.emplace_back(address, address + TRACKER_PAGE_SIZE);
			}
		}
	}
}

BindlessTable::BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler,
                             PageManager& page_manager)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager) {
	if (!graphics.bindless_enabled) {
		return;
	}
	m_images_per_array =
	    std::min(MaxImagesPerArray, graphics.bindless_max_sampled_images / SampledImageArrays);
	// KYTY_BINDLESS_IMAGES=n: at most n images per array (bisect, v436w).
	if (const char* cap = std::getenv("KYTY_BINDLESS_IMAGES"); cap != nullptr) {
		m_images_per_array = std::min(m_images_per_array, static_cast<uint32_t>(std::strtoul(cap, nullptr, 0)));
	}
	m_storage_per_array = graphics.bindless_storage_enabled
	                          ? std::max(PlaceholderColors,
	                                     std::min({m_images_per_array, 4096u,
	                                               graphics.bindless_max_storage_images}))
	                          : PlaceholderColors;
	if (m_images_per_array == 0) {
		LOGF("Bindless table: the device allows no update-after-bind sampled images\n");
		return;
	}

	// Bisect knob (v436v): KYTY_BINDLESS_NO_UAB=1 drops update-after-bind (updates while
	// pending become invalid: a fault test only).
	static const bool no_uab = [] {
		const char* v = std::getenv("KYTY_BINDLESS_NO_UAB");
		return v != nullptr && v[0] == '1';
	}();
	// The table names the stages that read it, not eAll: with eAll in an independent-sets
	// pipeline layout (graphics pipeline libraries) every bindless draw could fault the GPU on
	// NVIDIA 621.14 (ReadInvalid at 0x5800000; v436v: eAll faults, these stages are clean, same
	// run otherwise). Default with KYTY_GPL (the layouts of other runs stay as they were, and
	// with them the driver cache); KYTY_BINDLESS_STAGES=1/0 forces it either way.
	const bool few_stages = [&] {
		const char* v = std::getenv("KYTY_BINDLESS_STAGES");
		return v != nullptr ? v[0] == '1' : graphics.graphics_pipeline_library_enabled;
	}();
	const vk::ShaderStageFlags stages =
	    few_stages ? (vk::ShaderStageFlagBits::eAllGraphics | vk::ShaderStageFlagBits::eCompute |
	                  (graphics.mesh_shader_enabled ? vk::ShaderStageFlagBits::eMeshEXT
	                                                : vk::ShaderStageFlags {}) |
	                  // v440 M3: callable splits (KYTY_RT_CALLABLES) sample through it too.
	                  (graphics.ray_tracing_pipeline_enabled
	                       ? vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eCallableKHR
	                       : vk::ShaderStageFlags {}))
	               : vk::ShaderStageFlagBits::eAll;
	const auto image_flags = no_uab ? vk::DescriptorBindingFlags {vk::DescriptorBindingFlagBits::ePartiallyBound}
	                                : vk::DescriptorBindingFlagBits::ePartiallyBound |
	                                      vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	                                      vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	// The buffers are written once, here, so they need no update-after-bind.
	const auto buffer_flags = vk::DescriptorBindingFlags {};
	// A device without update-after-bind samplers still gets a one-entry array, holding the
	// default sampler, so every bindless pipeline sees the same layout.
	m_samplers_per_array = std::min(MaxSamplers, graphics.bindless_max_samplers);
	const std::array<vk::DescriptorSetLayoutBinding, 8> bindings {{
	    {Images2D, vk::DescriptorType::eSampledImage, m_images_per_array, stages,
	     nullptr},
	    {Images2DArray, vk::DescriptorType::eSampledImage, m_images_per_array,
	     stages, nullptr},
	    {ImagesCube, vk::DescriptorType::eSampledImage, m_images_per_array,
	     stages, nullptr},
	    {Images3D, vk::DescriptorType::eSampledImage, m_images_per_array, stages,
	     nullptr},
	    {Storage2D, vk::DescriptorType::eStorageImage, m_storage_per_array, stages,
	     nullptr},
	    {Translation, vk::DescriptorType::eStorageBuffer, 1, stages, nullptr},
	    {Feedback, vk::DescriptorType::eStorageBuffer, 1, stages, nullptr},
	    {Samplers, vk::DescriptorType::eSampler, std::max(m_samplers_per_array, 1u),
	     stages, nullptr},
	}};
	// Without update-after-bind storage images the storage array holds its placeholders only,
	// written here, before any draw binds the set.
	const auto storage_flags = graphics.bindless_storage_enabled
	                               ? image_flags
	                               : vk::DescriptorBindingFlags {vk::DescriptorBindingFlagBits::ePartiallyBound};
	const std::array<vk::DescriptorBindingFlags, 8> binding_flags {
	    image_flags, image_flags, image_flags, image_flags, storage_flags, buffer_flags, buffer_flags,
	    image_flags};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = static_cast<uint32_t>(binding_flags.size());
	flags_info.pBindingFlags = binding_flags.data();
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = no_uab ? vk::DescriptorSetLayoutCreateFlags {}
	                                  : vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout),
	                     "create bindless descriptor layout");

	const std::array<vk::DescriptorPoolSize, 4> sizes {{
	    {vk::DescriptorType::eSampledImage, m_images_per_array * SampledImageArrays},
	    {vk::DescriptorType::eStorageImage, m_storage_per_array},
	    {vk::DescriptorType::eStorageBuffer, 2},
	    {vk::DescriptorType::eSampler, std::max(m_samplers_per_array, 1u)},
	}};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = no_uab ? vk::DescriptorPoolCreateFlags {}
	                                 : vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool_info.pPoolSizes    = sizes.data();
	RequireVulkanSuccess(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool),
	                     "create bindless descriptor pool");

	vk::DescriptorSetAllocateInfo allocate_info {};
	allocate_info.descriptorPool     = m_pool;
	allocate_info.descriptorSetCount = 1;
	allocate_info.pSetLayouts        = &m_layout;
	RequireVulkanSuccess(graphics.device.allocateDescriptorSets(&allocate_info, &m_set),
	                     "allocate bindless descriptor set");

	// Both live in host-visible device memory: the host writes translations and clears feedback
	// flags directly, while shaders access them at device speed. The host reads the flags from
	// a host-cached snapshot (RecordFeedbackSnapshot).
	m_translation = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, TranslationEntries * sizeof(uint32_t));
	m_feedback    = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, TranslationEntries * sizeof(uint32_t));
	m_feedback_snapshot = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                               AllFlags, TranslationEntries * sizeof(uint32_t));
	EXIT_IF(m_translation->Mapped().empty() || m_feedback->Mapped().empty() ||
	        m_feedback_snapshot->Mapped().empty());
	// A region allocated after a snapshot was recorded reads as "no requests" until the next.
	std::fill(m_feedback_snapshot->Mapped().begin(), m_feedback_snapshot->Mapped().end(),
	          uint8_t {0});
	m_feedback_snapshot->Flush(0, m_feedback_snapshot->Size());
	// v439: the address directory (translation words) and its request words (feedback) start
	// empty; heap regions come after them.
	{
		using namespace ShaderRecompiler::IR;
		auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
		auto* feedback    = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
		std::fill_n(translation + BindlessAddressDirectoryBase, BindlessAddressDirectoryWords, 0u);
		std::fill_n(feedback + BindlessAddressDirectoryBase, BindlessAddressDirectoryWords, 0u);
		m_translation->Flush(BindlessAddressDirectoryBase * sizeof(uint32_t),
		                     BindlessAddressDirectoryWords * sizeof(uint32_t));
		m_feedback->Flush(BindlessAddressDirectoryBase * sizeof(uint32_t),
		                  BindlessAddressDirectoryWords * sizeof(uint32_t));
		m_next_region = std::max(m_next_region, BindlessAddressDirectoryBase + BindlessAddressDirectoryWords);
		// v439b: the value directories (empty) and the value heaps' fixed regions (pending).
		std::fill_n(translation + BindlessValueDirectoryBase, BindlessValueBindings * BindlessValueWords, 0u);
		std::fill_n(feedback + BindlessValueDirectoryBase, BindlessValueBindings * BindlessValueWords, 0u);
		for (uint32_t binding = 0; binding < BindlessValueBindings; binding++) {
			auto& heap        = m_value_heaps[binding];
			heap.value        = true;
			heap.binding      = binding;
			heap.region       = BindlessValueRegionBase + binding * BindlessValueEntries;
			heap.entries      = BindlessValueEntries;
			heap.guest_entries = BindlessValueEntries;
			heap.slots.assign(BindlessValueEntries, 0u);
			heap.settled.assign(BindlessValueEntries, 0u);
			heap.eager_count.assign(BindlessValueEntries, 0u);
			heap.stale.assign(BindlessValueEntries, 0u);
			heap.descriptors.assign(BindlessValueEntries, {});
			heap.images.assign(BindlessValueEntries, {});
			heap.resource.resource_class =
			    binding == BindlessStorage2D ? ImageResourceClass::Storage : ImageResourceClass::Sampled;
			heap.resource.numeric_class  = Prospero::TextureNumericClass::Float;
			heap.resource.dimension      = binding == BindlessImages3D        ? ShaderRecompiler::Decoder::ImageDimension::Dim3D
			                               : binding == BindlessImages2DArray ? ShaderRecompiler::Decoder::ImageDimension::Dim2DArray
			                               : binding == BindlessImagesCube    ? ShaderRecompiler::Decoder::ImageDimension::Dim2DArray
			                                                                  : ShaderRecompiler::Decoder::ImageDimension::Dim2D;
			heap.resource.cube = binding == BindlessImagesCube;
			for (uint32_t key = 0; key < BindlessValueEntries; key++) {
				translation[heap.region + key] = VisibleSlot(BindlessPending);
				feedback[heap.region + key]    = 0;
			}
		}
		m_translation->Flush(BindlessValueDirectoryBase * sizeof(uint32_t),
		                     (BindlessReservedEntries - BindlessValueDirectoryBase) * sizeof(uint32_t));
		m_feedback->Flush(BindlessValueDirectoryBase * sizeof(uint32_t),
		                  (BindlessReservedEntries - BindlessValueDirectoryBase) * sizeof(uint32_t));
		m_next_region = std::max(m_next_region, BindlessReservedEntries);
	}
	SetVulkanObjectNameF(graphics.device, m_translation->Handle(), "Bindless Translation");
	SetVulkanObjectNameF(graphics.device, m_feedback->Handle(), "Bindless Feedback");

	const std::array<vk::DescriptorBufferInfo, 2> buffer_infos {{
	    {m_translation->Handle(), 0, VK_WHOLE_SIZE},
	    {m_feedback->Handle(), 0, VK_WHOLE_SIZE},
	}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Translation + i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &buffer_infos[i];
	}
	graphics.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
	                                     nullptr);
	CreatePlaceholders(scheduler);
	graphics.bindless_layout = m_layout;
	graphics.bindless_set    = m_set;
	LOGF("Bindless table: %u images per array, %u storage images, %u translation entries, %u samplers\n",
	     m_images_per_array, m_storage_per_array, TranslationEntries, m_samplers_per_array);
	ShaderRecompiler::IR::g_bindless_storage_supported.store(m_storage_per_array > PlaceholderColors,
	                                                        std::memory_order_relaxed);
	if (PublishAtSubmit()) {
		m_scheduler.SetAfterSubmit([this] { PublishTranslations(); });
	}
}

BindlessTable::SamplerHeap* BindlessTable::FindOrCreateSamplerHeap(uint64_t base,
                                                                   uint32_t table_offset,
                                                                   uint32_t flags) {
	for (auto& heap: m_sampler_heaps) {
		if (heap.base == base && heap.table_offset == table_offset && heap.flags == flags) {
			return &heap;
		}
	}
	auto& heap        = m_sampler_heaps.emplace_back();
	heap.base         = base;
	heap.table_offset = table_offset;
	heap.flags        = flags;
	return &heap;
}

void BindlessTable::WriteDefaultSampler(vk::Sampler sampler) {
	if (m_default_sampler_written || m_set == nullptr) {
		return;
	}
	vk::DescriptorImageInfo info {sampler, nullptr, vk::ImageLayout::eUndefined};
	vk::WriteDescriptorSet  write {};
	write.dstSet          = m_set;
	write.dstBinding      = Samplers;
	write.dstArrayElement = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampler;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	m_default_sampler_written = true;
}

// KYTY_LOG_BINDLESS_SAMPLERS=1: sampler heap mirroring on stderr ([bl-samp]): each heap update,
// the array-full case, and every distinct sampler record decoded once.
static bool LogBindlessSamplers() {
	static const bool enabled = std::getenv("KYTY_LOG_BINDLESS_SAMPLERS") != nullptr;
	return enabled;
}

static void LogSamplerRecord(uint64_t base, size_t key, const ShaderSamplerResource& r) {
	static std::mutex                          mutex;
	static std::set<std::array<uint32_t, 4>>   seen;
	{
		std::lock_guard lock(mutex);
		if (seen.size() > 4096 || !seen.insert({r.fields[0], r.fields[1], r.fields[2], r.fields[3]}).second) {
			return;
		}
	}
	std::fprintf(stderr,
	             "[bl-samp] rec heap=0x%012" PRIx64 " key=%zu %08x %08x %08x %08x clamp=%u%u%u aniso=%u"
	             " cmp=%u unorm=%d mag=%u min=%u zf=%u mip=%u minlod=%u maxlod=%u bias=%d biassec=%u"
	             " perfmip=%u filtermode=%u border=%u/%u trunc=%d\n",
	             base, key, r.fields[0], r.fields[1], r.fields[2], r.fields[3], r.ClampX(),
	             r.ClampY(), r.ClampZ(), r.MaxAnisoRatio(), r.DepthCompareFunc(),
	             r.ForceUnormCoords() ? 1 : 0, r.XyMagFilter(), r.XyMinFilter(), r.ZFilter(),
	             r.MipFilter(), r.MinLod(), r.MaxLod(),
	             static_cast<int>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)),
	             r.LodBiasSec(), r.PerfMip(), r.FilterMode(), r.BorderColorType(),
	             r.BorderColorPtr(), r.TruncCoord() ? 1 : 0);
}

bool BindlessTable::MirrorSamplerHeap(SamplerHeap&                              heap,
                                      std::span<const std::array<uint32_t, 4>> records,
                                      SamplerCache&                             cache) {
	const auto same_prefix =
	    records.size() >= heap.records.size() &&
	    std::equal(heap.records.begin(), heap.records.end(), records.begin());
	if (same_prefix && records.size() == heap.records.size()) {
		return true;
	}
	uint32_t first = static_cast<uint32_t>(heap.records.size());
	if (!same_prefix || records.size() > heap.capacity || heap.region == 0) {
		const auto capacity = std::max<uint32_t>(64u, static_cast<uint32_t>(records.size()) * 2u);
		if (m_samplers_per_array <= 1u || capacity > m_samplers_per_array - m_next_sampler_slot) {
			static std::atomic<uint32_t> reported = 0;
			if (LogBindlessSamplers() && reported.load() < 64) {
				std::fprintf(stderr, "[bl-samp] array full (%u of %u slots) heap 0x%012" PRIx64
				             " keeps %zu records, wanted %zu\n",
				             m_next_sampler_slot, m_samplers_per_array, heap.base,
				             heap.records.size(), records.size());
			}
			if (reported.fetch_add(1) < 8) {
				LOGF("Bindless samplers: array full (%u of %u slots), heap 0x%016" PRIx64
				     " keeps %zu records\n",
				     m_next_sampler_slot, m_samplers_per_array, heap.base, heap.records.size());
			}
			return false;
		}
		heap.region   = m_next_sampler_slot;
		heap.capacity = capacity;
		m_next_sampler_slot += capacity;
		first = 0;
	}
	std::vector<vk::DescriptorImageInfo> infos;
	infos.reserve(records.size() - first);
	for (size_t key = first; key < records.size(); key++) {
		ShaderSamplerResource descriptor;
		std::copy(records[key].begin(), records[key].end(), descriptor.fields);
		if (LogBindlessSamplers()) {
			LogSamplerRecord(heap.base, key, descriptor);
		}
		// Diagnostics: KYTY_BL_SAMP_FORCE=<letters> edits the mirrored game samplers to bisect a
		// wrong surface: w = wrap on every axis, b = no LOD bias, a = no anisotropy, l = no min/max
		// LOD clamp, k = first key's sampler (key 0) for every key.
		static const std::string force = [] {
			const char* value = std::getenv("KYTY_BL_SAMP_FORCE");
			return value != nullptr ? std::string(value) : std::string();
		}();
		// KYTY_BL_SAMP_FORCE_KEYS=<lo>-<hi>: only those keys (inclusive).
		static const std::pair<size_t, size_t> force_keys = [] {
			std::pair<size_t, size_t> range {0, SIZE_MAX};
			if (const char* value = std::getenv("KYTY_BL_SAMP_FORCE_KEYS"); value != nullptr) {
				char* end    = nullptr;
				range.first  = std::strtoull(value, &end, 10);
				range.second = (end != nullptr && *end == '-') ? std::strtoull(end + 1, nullptr, 10)
				                                               : range.first;
			}
			return range;
		}();
		if (!force.empty() && key >= force_keys.first && key <= force_keys.second) {
			if (force.find('k') != std::string::npos) {
				std::copy(records[0].begin(), records[0].end(), descriptor.fields);
			}
			if (force.find('w') != std::string::npos) {
				descriptor.fields[0] &= ~0x1ffu;
			}
			if (force.find('b') != std::string::npos) {
				descriptor.fields[2] &= ~0x3fffu;
			}
			if (force.find('a') != std::string::npos) {
				descriptor.fields[0] &= ~(0x7u << 9u);
				if (descriptor.XyMagFilter() >= 2u) {
					descriptor.fields[2] = (descriptor.fields[2] & ~(0x3u << 20u)) | (1u << 20u);
				}
				if (descriptor.XyMinFilter() >= 2u) {
					descriptor.fields[2] = (descriptor.fields[2] & ~(0x3u << 22u)) | (1u << 22u);
				}
			}
			if (force.find('l') != std::string::npos) {
				descriptor.fields[1] = (descriptor.fields[1] & ~0xffffffu) | (0xfffu << 12u);
			}
		}
		if ((heap.flags & SamplerDepthCompare) == 0u) {
			descriptor.fields[0] &= ~(0x7u << 12u);
		}
		if ((heap.flags & SamplerPointFiltering) != 0u) {
			descriptor.SetPointFiltering();
		}
		infos.push_back({cache.GetSampler(descriptor, (heap.flags & SamplerIntegerBorder) != 0u),
		                 nullptr, vk::ImageLayout::eUndefined});
	}
	if (!infos.empty()) {
		vk::WriteDescriptorSet write {};
		write.dstSet          = m_set;
		write.dstBinding      = Samplers;
		write.dstArrayElement = heap.region + first;
		write.descriptorCount = static_cast<uint32_t>(infos.size());
		write.descriptorType  = vk::DescriptorType::eSampler;
		write.pImageInfo      = infos.data();
		m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	}
	if (LogBindlessSamplers()) {
		static std::atomic<uint32_t> printed = 0;
		if (printed.fetch_add(1) < 256) {
			std::fprintf(stderr,
			             "[bl-samp] heap 0x%012" PRIx64 "+0x%x flags=%u region=%u records=%zu (%u new)"
			             " same_prefix=%d next_slot=%u\n",
			             heap.base, heap.table_offset, heap.flags, heap.region, records.size(),
			             static_cast<uint32_t>(infos.size()), same_prefix ? 1 : 0,
			             m_next_sampler_slot);
		}
	}
	static std::atomic<uint32_t> logged = 0;
	if (logged.fetch_add(1) < 32) {
		LOGF("Bindless samplers: heap 0x%016" PRIx64 "+0x%x flags=%u region=%u records=%zu"
		     " (%u new)\n",
		     heap.base, heap.table_offset, heap.flags, heap.region, records.size(),
		     static_cast<uint32_t>(infos.size()));
	}
	heap.records.assign(records.begin(), records.end());
	return true;
}

void BindlessTable::CreatePlaceholders(CommandScheduler& /*scheduler*/) {
	struct Kind {
		vk::ImageType        type;
		vk::ImageViewType    view;
		uint32_t             layers;
		vk::ImageCreateFlags flags;
	};
	const std::array<Kind, ImageArrays> kinds {{
	    {vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::e2DArray, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::eCube, 6, vk::ImageCreateFlagBits::eCubeCompatible},
	    {vk::ImageType::e3D, vk::ImageViewType::e3D, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}}, // Storage2D
	}};
	const auto layout_of = [](uint32_t i) {
		return i % ImageArrays == Storage2D ? vk::ImageLayout::eGeneral
		                                    : vk::ImageLayout::eShaderReadOnlyOptimal;
	};
	for (uint32_t i = 0; i < Placeholders; i++) {
		vk::ImageCreateInfo info {};
		info.flags         = kinds[i % ImageArrays].flags;
		info.imageType     = kinds[i % ImageArrays].type;
		info.format        = vk::Format::eR8G8B8A8Unorm;
		info.extent        = vk::Extent3D {1, 1, 1};
		info.mipLevels     = 1;
		info.arrayLayers   = kinds[i % ImageArrays].layers;
		info.samples       = vk::SampleCountFlagBits::e1;
		info.tiling        = vk::ImageTiling::eOptimal;
		info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		if (i % ImageArrays == Storage2D) {
			info.usage |= vk::ImageUsageFlagBits::eStorage;
		}
		info.initialLayout = vk::ImageLayout::eUndefined;
		EXIT_IF(!m_graphics.CreateImage(info, m_placeholders[i]));
		vk::ImageViewCreateInfo view {};
		view.image            = m_placeholders[i].image;
		view.viewType         = kinds[i % ImageArrays].view;
		view.format           = info.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, kinds[i % ImageArrays].layers};
		RequireVulkanSuccess(
		    m_graphics.device.createImageView(&view, nullptr, &m_placeholder_views[i]),
		    "create bindless placeholder view");
	}

	// One fenced submission, before any rendering: clears and a buffer fill are not allowed
	// inside the render passes draws record into.
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient;
	pool_info.queueFamilyIndex = m_graphics.queue_family;
	vk::CommandPool pool       = nullptr;
	RequireVulkanSuccess(m_graphics.device.createCommandPool(&pool_info, nullptr, &pool),
	                     "create bindless init pool");
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer command   = nullptr;
	RequireVulkanSuccess(m_graphics.device.allocateCommandBuffers(&allocate, &command),
	                     "allocate bindless init command buffer");
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin bindless init command buffer");

	std::array<vk::ImageMemoryBarrier, Placeholders> to_transfer {};
	std::array<vk::ImageMemoryBarrier, Placeholders> to_read {};
	for (uint32_t i = 0; i < Placeholders; i++) {
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0,
		                                       kinds[i % ImageArrays].layers};
		to_transfer[i].srcAccessMask       = {};
		to_transfer[i].dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		to_transfer[i].oldLayout           = vk::ImageLayout::eUndefined;
		to_transfer[i].newLayout           = vk::ImageLayout::eTransferDstOptimal;
		to_transfer[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].image               = m_placeholders[i].image;
		to_transfer[i].subresourceRange    = range;
		to_read[i]                         = to_transfer[i];
		to_read[i].srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
		to_read[i].dstAccessMask           = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		to_read[i].oldLayout               = vk::ImageLayout::eTransferDstOptimal;
		to_read[i].newLayout               = layout_of(i);
	}
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr,
	                        Placeholders, to_transfer.data());
	// Transparent black by default. KYTY_BINDLESS_DEBUG_COLORS=1: slot 0 grey (not resident), slot
	// 1 red (key outside its heap), slot 2 blue (pending), to see which case a surface is in.
	const bool debug_colors = std::getenv("KYTY_BINDLESS_DEBUG_COLORS") != nullptr;
	const auto color        = [debug_colors](float r, float g, float b) {
        return debug_colors ? vk::ClearColorValue {std::array<float, 4> {r, g, b, 1.0f}}
		                           : vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f}};
	};
	const std::array<vk::ClearColorValue, PlaceholderColors> colors {
	    color(0.5f, 0.5f, 0.5f), color(0.8f, 0.1f, 0.1f), color(0.1f, 0.2f, 0.8f)};
	for (uint32_t i = 0; i < Placeholders; i++) {
		command.clearColorImage(m_placeholders[i].image, vk::ImageLayout::eTransferDstOptimal,
		                        &colors[i / ImageArrays], 1, &to_transfer[i].subresourceRange);
	}
	// translation[0], the entry of keys outside their heap, is the red slot.
	command.fillBuffer(m_translation->Handle(), 4, VK_WHOLE_SIZE, 0u);
	command.fillBuffer(m_translation->Handle(), 0, 4, 1u);
	command.fillBuffer(m_feedback->Handle(), 0, VK_WHOLE_SIZE, 0u);
	vk::BufferMemoryBarrier translation_ready {};
	translation_ready.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	translation_ready.dstAccessMask       = vk::AccessFlagBits::eShaderRead;
	translation_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	translation_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	translation_ready.buffer              = m_translation->Handle();
	translation_ready.offset              = 0;
	translation_ready.size                = VK_WHOLE_SIZE;
	auto feedback_ready                   = translation_ready;
	feedback_ready.dstAccessMask          = vk::AccessFlagBits::eShaderRead |
	                                        vk::AccessFlagBits::eShaderWrite |
	                                        vk::AccessFlagBits::eHostRead;
	feedback_ready.buffer                 = m_feedback->Handle();
	const std::array<vk::BufferMemoryBarrier, 2> buffers_ready {translation_ready, feedback_ready};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eAllCommands |
	                            vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, static_cast<uint32_t>(buffers_ready.size()),
	                        buffers_ready.data(), Placeholders, to_read.data());
	RequireVulkanSuccess(command.end(), "end bindless init command buffer");

	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	RequireVulkanSuccess(m_graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create bindless init fence");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		RequireVulkanSuccess(m_graphics.queue.submit(1, &submit, fence), "submit bindless init");
	}
	RequireVulkanSuccess(m_graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for bindless init");
	m_graphics.device.destroyFence(fence, nullptr);
	m_graphics.device.destroyCommandPool(pool, nullptr);

	std::array<vk::DescriptorImageInfo, Placeholders> infos {};
	std::array<vk::WriteDescriptorSet, Placeholders> writes {};
	for (uint32_t i = 0; i < Placeholders; i++) {
		infos[i]                  = {nullptr, m_placeholder_views[i], layout_of(i)};
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Images2D + i % ImageArrays;
		writes[i].dstArrayElement = i / ImageArrays;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = DescriptorTypeFor(i % ImageArrays);
		writes[i].pImageInfo      = &infos[i];
	}
	m_graphics.device.updateDescriptorSets(Placeholders, writes.data(), 0, nullptr);

	// C4 (KYTY_BINDLESS_PREFILL=0 disables): every other slot of every image array holds its
	// binding's placeholder from the start. A draw already executing when a slot is first written
	// (UPDATE_AFTER_BIND: the write is only guaranteed visible to later submissions) may fetch the
	// slot's descriptor from a GPU cache line loaded earlier for a neighbouring slot, or before the
	// write reached memory. With never-written slots (KYTY_BINDLESS_SLOT_REUSE=0) that was
	// uninitialized pool memory: a garbage texture header, read through a garbage address. Now it
	// is a placeholder, which is what a pending key samples anyway.
	static const bool prefill = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PREFILL");
		return value == nullptr || value[0] != '0';
	}();
	if (prefill && m_images_per_array > PlaceholderColors) {
		std::vector<vk::DescriptorImageInfo> fill;
		for (uint32_t binding = 0; binding < ImageArrays; binding++) {
			if (SlotsPerArray(binding) <= PlaceholderColors) {
				continue;
			}
			const uint32_t count = SlotsPerArray(binding) - PlaceholderColors;
			fill.assign(count, vk::DescriptorImageInfo {nullptr, m_placeholder_views[binding],
			                                            layout_of(binding)});
			vk::WriteDescriptorSet write {};
			write.dstSet          = m_set;
			write.dstBinding      = Images2D + binding;
			write.dstArrayElement = PlaceholderColors;
			write.descriptorCount = count;
			write.descriptorType  = DescriptorTypeFor(binding);
			write.pImageInfo      = fill.data();
			m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
		}
		LOGF("Bindless table: %u slots per array pre-filled with placeholders\n",
		     m_images_per_array - PlaceholderColors);
	}
}

BindlessTable::~BindlessTable() {
	m_scheduler.SetAfterSubmit({});
	m_sweep_sink.reset();
	if (m_sweep_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_sweep_pipeline, nullptr);
	}
	if (m_sweep_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_sweep_pipeline_layout, nullptr);
	}
	if (m_sweep_set_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_sweep_set_layout, nullptr);
	}
	for (uint32_t i = 0; i < Placeholders; i++) {
		if (m_placeholder_views[i] != nullptr) {
			m_graphics.device.destroyImageView(m_placeholder_views[i], nullptr);
		}
		if (m_placeholders[i].image != nullptr) {
			m_graphics.DeleteImage(m_placeholders[i]);
		}
	}
	m_translation.reset();
	m_feedback.reset();
	m_feedback_snapshot.reset();
	if (m_pool != nullptr) {
		m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
	}
	if (m_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
	}
}

uint32_t BindlessTable::RegisterAddressResource(const ShaderRecompiler::IR::ImageResource& resource) {
	std::lock_guard lock(m_address_mutex);
	for (uint32_t i = 0; i < m_address_resources.size(); i++) {
		if (m_address_resources[i] == resource) {
			return i;
		}
	}
	if (m_address_resources.size() >= 0xffffu) {
		return 0; // the first resource stands in (its heaps resolve by its class and dimension)
	}
	m_address_resources.push_back(resource);
	return static_cast<uint32_t>(m_address_resources.size() - 1u);
}

bool BindlessTable::InsertAddressDirectory(uint64_t block, uint32_t phase, uint32_t binding,
                                           uint32_t region) {
	using namespace ShaderRecompiler::IR;
	const auto low  = static_cast<uint32_t>(block);
	const auto high = static_cast<uint32_t>(block >> 32u);
	const auto tag  = BindlessAddressTag(binding, phase);
	const auto hash = BindlessAddressHash(low, high, tag);
	auto*      translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	for (uint32_t probe = 0; probe < BindlessAddressProbes; probe++) {
		const auto slot = (hash + probe) & (BindlessAddressDirectoryEntries - 1u);
		const auto word = BindlessAddressDirectoryBase + slot * 4u;
		if (translation[word + 2u] != 0u) {
			continue;
		}
		translation[word]      = low;
		translation[word + 1u] = high;
		translation[word + 3u] = region;
		// The tag last: a shader that matches it reads the block's words and region too (the
		// fence orders the write-combined stores, as SetTranslation's does).
		StoreFence();
		translation[word + 2u] = tag;
		if (!m_translation->IsCoherent()) {
			m_translation->Flush(word * sizeof(uint32_t), 4u * sizeof(uint32_t));
		}
		return true;
	}
	return false;
}

void BindlessTable::ProcessAddressRequests() {
	using namespace ShaderRecompiler::IR;
	if (!Enabled()) {
		return;
	}
	std::vector<std::pair<uint32_t, uint32_t>> marked;
	TakeCapture(BindlessAddressDirectoryBase, BindlessAddressDirectoryWords, marked);
	if (marked.empty()) {
		return;
	}
	std::unordered_map<uint32_t, std::array<uint32_t, 4>> requests;
	for (const auto& [index, word]: marked) {
		requests[index / 4u][index % 4u] = word;
	}
	uint32_t created = 0;
	for (const auto& [record, words]: requests) {
		const auto tag     = words[2];
		const auto low     = words[0];
		const auto binding = (tag >> 24u) & 0x7fu;
		const auto phase   = tag & 31u;
		const auto res     = (tag >> 8u) & 0xffffu;
		if ((tag & BindlessAddressTagValid) == 0u || binding >= SampledImageArrays ||
		    (phase & 3u) != 0u || (low & (BindlessAddressBlockBytes - 1u)) != 0u) {
			continue; // a request two lanes wrote at once, or no request
		}
		const uint64_t block = (static_cast<uint64_t>(words[1]) << 32u) | low;
		const uint64_t key   = block | (phase >> 2u) | (static_cast<uint64_t>(binding) << 3u);
		if (m_address_heap_index.contains(key)) {
			continue; // created after the snapshot was recorded
		}
		ImageResource resource;
		{
			std::lock_guard lock(m_address_mutex);
			if (res >= m_address_resources.size()) {
				continue;
			}
			resource = m_address_resources[res];
		}
		auto* heap = FindOrCreateHeap(block, phase, binding, BindlessAddressKeysPerBlock, resource);
		if (heap == nullptr) {
			continue; // the translation buffer is full: such blocks keep sampling slot 0
		}
		heap->address       = true;
		heap->guest_entries = std::max(heap->guest_entries, BindlessAddressKeysPerBlock);
		if (!InsertAddressDirectory(block, phase, binding, heap->region)) {
			if (m_address_directory_full++ < 16u) {
				std::fprintf(stderr, "[bindless-address] directory full around block 0x%012llx\n",
				             static_cast<unsigned long long>(block));
			}
			continue;
		}
		m_address_heap_index.emplace(key, heap);
		m_address_heaps[binding].push_back(heap);
		created++;
	}
	static std::atomic<uint32_t> logged {0};
	if (created != 0 && logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
		size_t total = 0;
		for (const auto& list: m_address_heaps) total += list.size();
		std::fprintf(stderr, "[bindless-address] %u new blocks (%zu in all)\n", created, total);
	}
}

void BindlessTable::ProcessValueRequests(std::vector<std::pair<Heap*, uint32_t>>& created) {
	using namespace ShaderRecompiler::IR;
	created.clear();
	if (!Enabled()) {
		return;
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	for (uint32_t binding = 0; binding < BindlessValueBindings; binding++) {
		const auto directory = BindlessValueDirectoryBase + binding * BindlessValueWords;
		std::vector<std::pair<uint32_t, uint32_t>> marked;
		TakeCapture(directory, BindlessValueWords, marked);
		if (marked.empty()) {
			continue;
		}
		std::unordered_map<uint32_t, std::array<uint32_t, 8>> requests;
		for (const auto& [index, word]: marked) {
			requests[index / 8u][index % 8u] = word; // words left out were 0
		}
		auto& heap = m_value_heaps[binding];
		for (auto& [record, words]: requests) {
			// Two lanes writing different T#s at once leave a mix: the hash tells it apart.
			if (BindlessValueHash(words.data()) != record ||
			    std::ranges::all_of(words, [](uint32_t w) { return w == 0u; })) {
				continue;
			}
			uint32_t free_entry = UINT32_MAX;
			bool     present    = false;
			for (uint32_t probe = 0; probe < BindlessAddressProbes && !present; probe++) {
				const auto entry = (record + probe) & (BindlessValueEntries - 1u);
				const auto* e    = translation + directory + entry * 8u;
				if (std::equal(words.begin(), words.end(), e)) {
					present = true;
				} else if (free_entry == UINT32_MAX && std::all_of(e, e + 8, [](uint32_t w) { return w == 0u; })) {
					free_entry = entry;
				}
			}
			if (present) {
				continue;
			}
			if (free_entry == UINT32_MAX) {
				if (m_value_directory_full++ < 16u) {
					std::fprintf(stderr, "[bindless-value] directory %u full around entry %u\n", binding, record);
				}
				continue;
			}
			heap.descriptors[free_entry] = words;
			heap.settled[free_entry]     = 0;
			auto* e = translation + directory + free_entry * 8u;
			for (uint32_t i = 1; i < 8u; i++) {
				e[i] = words[i];
			}
			// Word 0 last (never 0 for a T#: the base address >> 8): a lookup matches whole.
			StoreFence();
			e[0] = words[0];
			if (!m_translation->IsCoherent()) {
				m_translation->Flush((directory + free_entry * 8u) * sizeof(uint32_t), 8u * sizeof(uint32_t));
			}
			created.emplace_back(&heap, free_entry);
		}
	}
	static std::atomic<uint32_t> logged {0};
	if (!created.empty() && logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
		std::fprintf(stderr, "[bindless-value] %zu new T#s\n", created.size());
	}
}

bool BindlessTable::AllocateRegion(Heap& heap, uint32_t entries) {
	// Headroom so a growing heap rarely moves; a move re-writes the keys resolved so far.
	const auto capacity = std::max(entries, std::min<uint32_t>(entries * 2u, 1u << 16u));
	if (capacity > TranslationEntries - m_next_region) {
		return false;
	}
	const auto old_region  = heap.region;
	const auto old_entries = heap.entries;
	heap.region            = m_next_region;
	heap.entries           = capacity;
	m_next_region += capacity;
	heap.slots.resize(capacity, 0u);
	heap.settled.resize(capacity, 0u);
	heap.eager_count.resize(capacity, 0u);
	heap.stale.resize(capacity, 0u);
	heap.descriptors.resize(capacity);
	heap.images.resize(capacity);
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	auto* feedback    = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	for (uint32_t key = 0; key < capacity; key++) {
		uint32_t value = ShaderRecompiler::IR::BindlessPending;
		if (key < old_entries && heap.settled[key] != 0) {
			value = heap.slots[key];
		}
		translation[heap.region + key] = VisibleSlot(value);
		feedback[heap.region + key]    = 0;
	}
	if (!m_translation->IsCoherent()) {
		m_translation->Flush(heap.region * sizeof(uint32_t), capacity * sizeof(uint32_t));
	}
	LOGF("Bindless heap region: base=0x%016" PRIx64 " binding=%u region=%u entries=%u (was %u at "
	     "%u)\n",
	     heap.base, heap.binding, heap.region, capacity, old_entries, old_region);
	return true;
}

BindlessTable::Heap* BindlessTable::FindOrCreateHeap(
    uint64_t base, uint32_t table_offset, uint32_t binding, uint32_t entries,
    const ShaderRecompiler::IR::ImageResource& resource) {
	if (entries == 0) {
		return nullptr;
	}
	for (auto& heap: m_heaps) {
		if (heap.base == base && heap.table_offset == table_offset && heap.binding == binding) {
			if (entries > heap.entries && !AllocateRegion(heap, entries)) {
				return nullptr;
			}
			heap.guest_entries = std::max(heap.guest_entries, entries);
			return &heap;
		}
	}
	auto& heap         = m_heaps.emplace_back();
	heap.base          = base;
	heap.table_offset  = table_offset;
	heap.binding       = binding;
	heap.resource      = resource;
	heap.guest_entries = entries;
	if (!AllocateRegion(heap, std::max(entries, 1u << 14u))) {
		m_heaps.pop_back();
		return nullptr;
	}
	return &heap;
}

uint32_t BindlessTable::AllocateSlot(uint32_t binding) {
	if (binding >= ImageArrays) {
		return 0;
	}
	// Keys are settled again whenever the guest rewrites a heap entry or a texture is evicted,
	// and each settle took a new slot: the array would fill up over a long session.
	if (auto& free = m_free_slots[binding];
	    !free.empty() && m_scheduler.IsFree(free.front().second)) {
		const auto slot = free.front().first;
		free.pop_front();
		return slot;
	}
	if (m_next_slot[binding] >= SlotsPerArray(binding)) {
		return 0;
	}
	return m_next_slot[binding]++;
}

void BindlessTable::ReleaseSlot(uint32_t binding, uint32_t slot) {
	static const bool reuse = [] {
		const char* value = std::getenv("KYTY_BINDLESS_SLOT_REUSE");
		return value == nullptr || value[0] != '0';
	}();
	if (reuse && binding < ImageArrays && slot >= PlaceholderColors) {
		m_free_slots[binding].emplace_back(slot, m_scheduler.CurrentTick());
	}
}

void BindlessTable::WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view,
                              vk::ImageLayout layout, VkImage image, uint64_t heap_base,
                              uint32_t key, uint32_t width, uint32_t height, uint32_t layers) {
	if (std::ranges::find(m_placeholder_views, view) == m_placeholder_views.end()) {
		(void)CheckLiveImageView(view, "a bindless slot");
	}
	// A storage image is written in the general layout, whatever the caller passed for a
	// placeholder.
	const vk::DescriptorImageInfo info {nullptr, view,
	                                    binding == Storage2D ? vk::ImageLayout::eGeneral : layout};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = binding;
	write.dstArrayElement = slot;
	write.descriptorCount = 1;
	write.descriptorType  = DescriptorTypeFor(binding);
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	if (binding >= ImageArrays) {
		return;
	}
	auto& views = m_slot_views[binding];
	if (views.size() <= slot) {
		views.resize(slot + 1u);
	}
	if (ImageHistory::Enabled()) {
		auto& images = m_slot_images[binding];
		if (images.size() <= slot) {
			images.resize(slot + 1u, VK_NULL_HANDLE);
		}
		ImageHistory::NoteSlotWrite(binding, slot, image, images[slot], m_scheduler.CurrentTick(),
		                            heap_base, key);
		images[slot] = image;
	}
	{
		auto& expect = m_slot_expect[binding];
		if (expect.size() <= slot) {
			expect.resize(slot + 1u);
		}
		expect[slot] = {width, height, layers, key, m_scheduler.CurrentTick(), heap_base};
	}
	if (CanaryDetectEnabled()) {
		auto& logs = m_slot_log[binding];
		if (logs.size() <= slot) {
			logs.resize(slot + 1u);
		}
		auto& log = logs[slot];
		const double now_ms = std::chrono::duration<double, std::milli>(
		                          std::chrono::steady_clock::now().time_since_epoch()).count();
		const bool placeholder_write =
		    std::ranges::find(m_placeholder_views, view) != m_placeholder_views.end();
		log.writes[log.next % log.writes.size()] = {m_scheduler.CurrentTick(), now_ms,
		                                            static_cast<VkImageView>(view),
		                                            placeholder_write ? VK_NULL_HANDLE : image,
		                                            width, height, layers, key, heap_base};
		log.next++;
	}
	if (const auto old = views[slot]; old != nullptr) {
		if (const auto found = m_view_slots.find(static_cast<VkImageView>(old));
		    found != m_view_slots.end()) {
			std::erase(found->second, std::pair {binding, slot});
			if (found->second.empty()) {
				m_view_slots.erase(found);
			}
		}
	}
	const bool placeholder =
	    std::ranges::find(m_placeholder_views, view) != m_placeholder_views.end();
	views[slot] = placeholder ? vk::ImageView {} : view;
	if (!placeholder && view != nullptr) {
		m_view_slots[static_cast<VkImageView>(view)].emplace_back(binding, slot);
	}
}

void BindlessTable::ReportHeldViews(std::span<const vk::ImageView> views, const char* what) {
	for (const auto view: views) {
		const auto found = m_view_slots.find(static_cast<VkImageView>(view));
		if (found == m_view_slots.end()) {
			continue;
		}
		for (const auto& [binding, slot]: found->second) {
			std::fprintf(stderr, "[stale-view] bindless binding %u slot %u still holds a view of %s\n",
			             binding, slot, what);
		}
	}
}

bool BindlessTable::ReleaseViews(std::span<const vk::ImageView> views) {
	bool held = false;
	for (const auto view: views) {
		const auto found = m_view_slots.find(static_cast<VkImageView>(view));
		if (found == m_view_slots.end()) {
			continue;
		}
		const auto slots = found->second;
		for (const auto& [binding, slot]: slots) {
			held = true;
			for (auto& heap: m_heaps) {
				for (uint32_t key = 0; key < heap.slots.size(); key++) {
					if (heap.slots[key] == 0 || SlotBinding(heap, heap.slots[key]) != binding ||
					    SlotIndex(heap.slots[key]) != slot) {
						continue;
					}
					if (const auto id = heap.images[key]; id) {
						if (const auto refs = m_image_refs.find(ImageKey(id)); refs != m_image_refs.end()) {
							std::erase(refs->second, std::pair<Heap*, uint32_t> {&heap, key});
							if (refs->second.empty()) {
								m_image_refs.erase(refs);
							}
						}
						std::erase(heap.resolved, id);
						heap.checked_generation = 0;
					}
					heap.slots[key]   = 0;
					heap.settled[key] = 0;
					heap.images[key]  = {};
					SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
				}
			}
			// C4b: placeholder and free list once the GPU is past the current tick; the caller
			// keeps the image alive until then (held).
			RetireSlot(binding, slot);
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 16) {
				LOGF("Bindless: binding %u slot %u still held an image being destroyed; "
				     "repointed to the placeholder\n",
				     binding, slot);
			}
		}
	}
	return held;
}

void BindlessTable::SetTranslation(const Heap& heap, uint32_t key, uint32_t slot) {
	if (PublishSafeEnabled()) {
		const uint32_t entry = heap.region + key;
		std::lock_guard lock(m_publish_mutex);
		// A later value for the entry supersedes the queued ones (v385: by generation, O(1)).
		if (slot != ShaderRecompiler::IR::BindlessPending) {
			const auto generation          = ++m_publish_safe_generation;
			m_publish_safe_latest[entry]   = generation;
			m_publish_safe.push_back({entry, VisibleSlot(slot), m_scheduler.CurrentTick(), generation});
			m_publish_safe_max = std::max<uint64_t>(m_publish_safe_max, m_publish_safe.size());
			return;
		}
		m_publish_safe_latest.erase(entry);
		auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
		StoreFence();
		translation[entry] = VisibleSlot(slot);
		if (!m_translation->IsCoherent()) {
			m_translation->Flush(entry * sizeof(uint32_t), sizeof(uint32_t));
		}
		return;
	}
	if (PublishAtSubmit()) {
		// C4b: visible to the GPU after the next vkQueueSubmit, so no draw already executing
		// can find a slot through an entry written while it ran (PublishTranslations).
		std::lock_guard lock(m_publish_mutex);
		m_publish.emplace_back(heap.region + key, VisibleSlot(slot));
		return;
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	// C4: the descriptor write that made the slot valid (vkUpdateDescriptorSets, a store into
	// write-combined descriptor memory) must reach memory before a draw can find the slot through
	// this entry; x86 does not order write-combined stores against later ones without a fence.
	StoreFence();
	translation[heap.region + key] = VisibleSlot(slot);
	if (!m_translation->IsCoherent()) {
		m_translation->Flush((heap.region + key) * sizeof(uint32_t), sizeof(uint32_t));
	}
}

void BindlessTable::PublishSafe() {
	if (!PublishSafeEnabled()) {
		return;
	}
	std::lock_guard lock(m_publish_mutex);
	// v385: the GPU progress the scheduler already knows (no driver call per bindless draw; the
	// scheduler refreshes it often enough).
	const auto& master = m_scheduler.GetMasterSemaphore();
	static auto last_report = std::chrono::steady_clock::now();
	if (const auto now = std::chrono::steady_clock::now(); now - last_report > std::chrono::seconds(10)) {
		last_report = now;
		std::fprintf(stderr,
		             "[publish-safe] queued %zu (max %" PRIu64 "), published %" PRIu64 ", superseded %" PRIu64
		             ", oldest waits for tick %" PRIu64 " (GPU at %" PRIu64 ", recording %" PRIu64 ")\n",
		             m_publish_safe.size(), m_publish_safe_max, m_publish_safe_published, m_publish_safe_superseded,
		             m_publish_safe.empty() ? 0 : m_publish_safe.front().tick - 1u, master.KnownGpuTick(),
		             m_scheduler.CurrentTick());
		m_publish_safe_max = m_publish_safe.size();
	}
	if (m_publish_safe.empty() || !master.IsFree(m_publish_safe.front().tick - 1u)) {
		return;
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	// The descriptor writes these entries name were made before they were queued.
	StoreFence();
	uint32_t low = UINT32_MAX, high = 0;
	while (!m_publish_safe.empty() && master.IsFree(m_publish_safe.front().tick - 1u)) {
		const auto e = m_publish_safe.front();
		m_publish_safe.pop_front();
		const auto latest = m_publish_safe_latest.find(e.entry);
		if (latest == m_publish_safe_latest.end() || latest->second != e.generation) {
			m_publish_safe_superseded++;
			continue;
		}
		m_publish_safe_latest.erase(latest);
		translation[e.entry] = e.value;
		low                  = std::min(low, e.entry);
		high                 = std::max(high, e.entry);
		m_publish_safe_published++;
	}
	StoreFence();
	if (!m_translation->IsCoherent() && low <= high) {
		m_translation->Flush(low * sizeof(uint32_t), (high - low + 1u) * sizeof(uint32_t));
	}
}

void BindlessTable::PublishTranslations() {
	std::vector<std::pair<uint32_t, uint32_t>> entries;
	{
		std::lock_guard lock(m_publish_mutex);
		if (m_publish.empty()) {
			return;
		}
		entries.swap(m_publish);
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	// The descriptor writes these entries name were made before they were queued.
	StoreFence();
	uint32_t low = UINT32_MAX, high = 0;
	for (const auto& [entry, value]: entries) {
		translation[entry] = value;
		low                = std::min(low, entry);
		high               = std::max(high, entry);
	}
	StoreFence();
	if (!m_translation->IsCoherent()) {
		m_translation->Flush(low * sizeof(uint32_t), (high - low + 1u) * sizeof(uint32_t));
	}
}

void BindlessTable::RetireSlot(uint32_t binding, uint32_t slot) {
	if (!DeferredRetire()) {
		WriteSlot(binding, slot, PlaceholderView(binding), vk::ImageLayout::eShaderReadOnlyOptimal);
		ReleaseSlot(binding, slot);
		return;
	}
	const uint64_t id = (uint64_t {binding} << 32u) | slot;
	if (!m_retiring.emplace(id, true).second) {
		return; // already waiting
	}
	// The keys' pending entries were queued before this; the placeholder goes into the slot (and
	// the slot back to the free list) only after the GPU has finished every draw recorded so far,
	// which is also after those entries were published (the submit of this tick publishes them).
	m_scheduler.DeferRelease([this, binding, slot, id] {
		WriteSlot(binding, slot, PlaceholderView(binding), vk::ImageLayout::eShaderReadOnlyOptimal);
		ReleaseSlot(binding, slot);
		m_retiring.erase(id);
	});
}

void BindlessTable::RecordFeedbackSnapshot(CommandScheduler& scheduler) {
	// Regions are allocated contiguously after word 0, so one copy covers every heap.
	static const bool capture = std::getenv("KYTY_KEY_CAPTURE") != nullptr ||
	                            std::getenv("KYTY_PROBE_RECT") != nullptr ||
	                            std::getenv("KYTY_BINDLESS_OOB_INFO") != nullptr ||
	                            std::getenv("KYTY_NAN_MAGENTA") != nullptr;
	const vk::DeviceSize size = vk::DeviceSize {capture ? TranslationEntries : m_next_region} *
	                            sizeof(uint32_t);
	scheduler.EndRendering();
	auto                    command = scheduler.Current().Recorder();
	vk::BufferMemoryBarrier flags_written {};
	flags_written.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
	flags_written.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	flags_written.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.buffer              = m_feedback->Handle();
	flags_written.offset              = 0;
	flags_written.size                = VK_WHOLE_SIZE;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1,
	                        &flags_written, 0, nullptr);
	// v127: plus the out-of-range slot word (above every region).
	constexpr vk::DeviceSize oob = vk::DeviceSize {ShaderRecompiler::IR::BindlessOutOfRangeSlotWord} *
	                               sizeof(uint32_t);
	const std::array<vk::BufferCopy, 2> copies {{{0, 0, size}, {oob, oob, sizeof(uint32_t)}}};
	command.copyBuffer(m_feedback->Handle(), m_feedback_snapshot->Handle(),
	                   size > oob ? 1u : 2u, copies.data());
	auto snapshot_written          = flags_written;
	snapshot_written.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	snapshot_written.dstAccessMask = vk::AccessFlagBits::eHostRead;
	snapshot_written.buffer        = m_feedback_snapshot->Handle();
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
	                        &snapshot_written, 0, nullptr);
	m_snapshot_tick = scheduler.CurrentTick();
}

static bool SlotSweepEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_SWEEP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void BindlessTable::RecordSlotSweep(CommandScheduler& scheduler, uint64_t frame) {
	if (!SlotSweepEnabled() || m_set == nullptr || frame == m_sweep_frame) {
		return;
	}
	if (m_sweep_pipeline == nullptr) {
		const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
		                                              vk::ShaderStageFlagBits::eCompute, nullptr};
		vk::DescriptorSetLayoutCreateInfo    layout_info {};
		layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		layout_info.bindingCount = 1;
		layout_info.pBindings    = &binding;
		RequireVulkanSuccess(
		    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_sweep_set_layout),
		    "create bindless sweep descriptor layout");
		const std::array<vk::DescriptorSetLayout, 2> sets {m_sweep_set_layout, m_layout};
		const vk::PushConstantRange                  range {vk::ShaderStageFlagBits::eCompute, 0, 12};
		vk::PipelineLayoutCreateInfo                 pipeline_layout_info {};
		pipeline_layout_info.setLayoutCount         = static_cast<uint32_t>(sets.size());
		pipeline_layout_info.pSetLayouts            = sets.data();
		pipeline_layout_info.pushConstantRangeCount = 1;
		pipeline_layout_info.pPushConstantRanges    = &range;
		RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
		                                                            &m_sweep_pipeline_layout),
		                     "create bindless sweep pipeline layout");
		const auto                        module = CompileSPV(BINDLESS_SWEEP_SPV, m_graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_sweep_pipeline_layout;
		const auto result    = m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info,
		                                                                nullptr, &m_sweep_pipeline);
		m_graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create bindless sweep pipeline");
		SetVulkanObjectNameF(m_graphics.device, m_sweep_pipeline, "Kyty.BindlessSweep");
		// Download usage: host-cached, so the report can read what the GPU wrote (Invalidate).
		m_sweep_sink = std::make_unique<Buffer>(m_graphics, scheduler, MemoryUsage::Download, 0,
		                                        AllFlags,
		                                        vk::DeviceSize {ImageArrays} * 65536u * 2u * sizeof(uint32_t));
		LOGF("Bindless sweep: on (one texel from every allocated slot, once a frame)\n");
	}
	// The previous sweep's answers, once the GPU has run it.
	if (m_sweep_tick != 0 && scheduler.IsFree(m_sweep_tick)) {
		VerifySweep();
	}
	m_sweep_frame = frame;
	m_sweep_tick  = scheduler.CurrentTick();
	scheduler.EndRendering();
	auto command = scheduler.Current().Recorder();
	// Everything drawn so far is visible to the sweep's reads.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr,
	                        0, nullptr);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_sweep_pipeline);
	const vk::DescriptorBufferInfo sink {m_sweep_sink->Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &sink;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_sweep_pipeline_layout, 0, 1, &write);
	command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_sweep_pipeline_layout, 1, 1, &m_set,
	                           0, nullptr);
	for (uint32_t binding = 0; binding < ImageArrays; binding++) {
		// The cube fetch samples through sampler slot 0, written with the first bindless sampler.
		const uint32_t count =
		    (binding == ImagesCube && !m_default_sampler_written) || binding == Storage2D
		        ? 0u
		        : std::min(m_next_slot[binding], 65536u);
		m_sweep_counts[binding] = count;
		if (count == 0) {
			continue;
		}
		const std::array<uint32_t, 3> constants {binding, count, static_cast<uint32_t>(frame)};
		command.pushConstants(m_sweep_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                      sizeof(constants), constants.data());
		command.dispatch((count + 255u) / 256u, 1, 1);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eMemoryRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	                        {}, 1, &after, 0, nullptr, 0, nullptr);
}

// v124: what the GPU's descriptors say each slot's image is, against what the host wrote. A slot
// the host repointed to the placeholder that still reports the old texture's size is a descriptor
// write that did not take effect on the GPU side ([bindless-verify] STALE); the sweep's texel
// fetches of such a slot fault once that memory is unmapped.
void BindlessTable::VerifySweep() {
	m_sweep_sink->Invalidate(0, m_sweep_sink->Size());
	const auto*    sink  = reinterpret_cast<const uint32_t*>(m_sweep_sink->Mapped().data());
	const uint32_t frame = static_cast<uint32_t>(m_sweep_frame) & 0xffffu;
	uint32_t       printed = 0;
	m_verify_frames++;
	for (uint32_t binding = 0; binding < ImageArrays; binding++) {
		const auto& expect = m_slot_expect[binding];
		for (uint32_t slot = PlaceholderColors; slot < m_sweep_counts[binding] && slot < expect.size();
		     slot++) {
			const auto& e = expect[slot];
			if (e.tick >= m_sweep_tick) {
				continue; // written after the sweep was recorded; the next sweep sees it
			}
			const uint32_t w0 = sink[(binding * 65536u + slot) * 2u];
			const uint32_t w1 = sink[(binding * 65536u + slot) * 2u + 1u];
			if ((w1 >> 16u) != frame) {
				continue; // the sweep did not reach it
			}
			const uint32_t width  = w0 & 0xffffu;
			const uint32_t height = w0 >> 16u;
			const uint32_t layers = w1 & 0xffffu;
			m_verify_checked++;
			const bool layers_known = binding != ImagesCube;
			if (width == (e.width & 0xffffu) && height == (e.height & 0xffffu) &&
			    (!layers_known || layers == (e.layers & 0xffffu))) {
				continue;
			}
			m_verify_mismatches++;
			static uint32_t total_printed = 0;
			if (printed < 8 && total_printed < 400) {
				printed++;
				total_printed++;
				const VkImage image = binding < m_slot_images.size() && slot < m_slot_images[binding].size()
				                          ? m_slot_images[binding][slot]
				                          : VK_NULL_HANDLE;
				std::fprintf(stderr,
				             "[bindless-verify] frame %llu binding %u slot %u: GPU %ux%ux%u, host %ux%ux%u%s "
				             "(image %p, written tick %llu, heap 0x%llx key %u, sweep tick %llu)\n",
				             static_cast<unsigned long long>(m_sweep_frame), binding, slot, width, height,
				             layers, e.width, e.height, e.layers,
				             e.width == 1 && e.height == 1 ? " STALE (host: placeholder)" : "",
				             static_cast<void*>(image), static_cast<unsigned long long>(e.tick),
				             static_cast<unsigned long long>(e.heap), e.key,
				             static_cast<unsigned long long>(m_sweep_tick));
				if (image != VK_NULL_HANDLE) {
					ImageHistory::Describe(image, stderr);
				}
			}
		}
	}
	if (m_verify_frames % 300 == 0) {
		std::fprintf(stderr, "[bindless-verify] %llu frames: %llu slots checked, %llu mismatches\n",
		             static_cast<unsigned long long>(m_verify_frames),
		             static_cast<unsigned long long>(m_verify_checked),
		             static_cast<unsigned long long>(m_verify_mismatches));
	}
}

void BindlessTable::ReportSweep(uint64_t /*frame*/) const {
	if (!SlotSweepEnabled() || m_sweep_sink == nullptr || m_sweep_sink->Mapped().empty()) {
		return;
	}
	m_sweep_sink->Invalidate(0, m_sweep_sink->Size());
	const auto*    sink  = reinterpret_cast<const uint32_t*>(m_sweep_sink->Mapped().data());
	const uint32_t frame = static_cast<uint32_t>(m_sweep_frame);
	std::fprintf(stderr, "[bindless-sweep] last sweep recorded for frame %u (tick %llu); %llu frames verified, %llu slots checked, %llu mismatches\n",
	             frame, static_cast<unsigned long long>(m_sweep_tick),
	             static_cast<unsigned long long>(m_verify_frames),
	             static_cast<unsigned long long>(m_verify_checked),
	             static_cast<unsigned long long>(m_verify_mismatches));
	for (uint32_t binding = 0; binding < ImageArrays; binding++) {
		uint32_t unfinished = 0, first = UINT32_MAX, last = 0, previous_frame = 0;
		for (uint32_t slot = 0; slot < m_sweep_counts[binding]; slot++) {
			const auto value = sink[(binding * 65536u + slot) * 2u + 1u] >> 16u;
			if (value != (frame & 0xffffu)) {
				unfinished++;
				first = std::min(first, slot);
				last  = std::max(last, slot);
				previous_frame = value;
			}
		}
		std::fprintf(stderr,
		             "[bindless-sweep] binding %u: %u slots swept, %u not finished with this frame's "
		             "sweep (slots %u..%u; they hold the value %u)\n",
		             binding, m_sweep_counts[binding], unfinished, first == UINT32_MAX ? 0 : first,
		             last, previous_frame);
	}
}

bool BindlessTable::SnapshotReady(CommandScheduler& scheduler) {
	return m_snapshot_tick != 0 && scheduler.IsFree(m_snapshot_tick);
}

void BindlessTable::TakeRequests(const Heap& heap, std::vector<uint32_t>& keys) {
	ReportCanaryHits(false);
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*       feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(heap.region * sizeof(uint32_t),
	                                heap.entries * sizeof(uint32_t));
	bool cleared = false;
	for (uint32_t key = 0; key < heap.entries; key++) {
		if (snapshot[heap.region + key] == 1u) {
			// A flag the GPU sets again after the snapshot may be lost here; a key that is
			// still pending is flagged again by the next draw that samples it. Other values
			// are usage-probe marks (ArmUsageProbe), not requests.
			feedback[heap.region + key] = 0;
			keys.push_back(key);
			cleared = true;
		}
	}
	if (cleared) {
		m_feedback->Flush(heap.region * sizeof(uint32_t), heap.entries * sizeof(uint32_t));
	}
}

void BindlessTable::TakeCapture(uint32_t first, uint32_t count,
                                std::vector<std::pair<uint32_t, uint32_t>>& marked) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*       feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(first * sizeof(uint32_t), count * sizeof(uint32_t));
	bool cleared = false;
	for (uint32_t i = 0; i < count; i++) {
		if (snapshot[first + i] != 0u) {
			marked.emplace_back(i, snapshot[first + i]);
			feedback[first + i] = 0;
			cleared             = true;
		}
	}
	if (cleared) {
		m_feedback->Flush(first * sizeof(uint32_t), count * sizeof(uint32_t));
	}
}

void BindlessTable::ReadSnapshot(uint32_t first, uint32_t count, std::vector<uint32_t>& words) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	m_feedback_snapshot->Invalidate(first * sizeof(uint32_t), count * sizeof(uint32_t));
	words.assign(snapshot + first, snapshot + first + count);
}

uint32_t BindlessTable::TakeWordZero() {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*       feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(0, sizeof(uint32_t));
	const auto value = snapshot[0];
	if (value != 0) {
		feedback[0] = 0;
		m_feedback->Flush(0, sizeof(uint32_t));
	}
	return value;
}

std::atomic<uint32_t> g_bindless_oob_seen {0}; // descriptors.cpp counts [bindless-oob]

uint32_t BindlessTable::TakeFeedbackWord(uint32_t word) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*       feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(word * sizeof(uint32_t), sizeof(uint32_t));
	const auto value = snapshot[word];
	if (value != 0) {
		feedback[word] = 0;
		m_feedback->Flush(word * sizeof(uint32_t), sizeof(uint32_t));
	}
	return value;
}

uint32_t BindlessTable::TakeOutOfRangeSlot() {
	constexpr uint32_t word     = ShaderRecompiler::IR::BindlessOutOfRangeSlotWord;
	const auto*        snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*              feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(word * sizeof(uint32_t), sizeof(uint32_t));
	const auto value = snapshot[word];
	if (value != 0) {
		feedback[word] = 0;
		m_feedback->Flush(word * sizeof(uint32_t), sizeof(uint32_t));
	}
	return value;
}

void BindlessTable::SetProbeActive(bool active) {
	constexpr uint32_t word = ShaderRecompiler::IR::BindlessProbeActiveWord;
	if (m_feedback == nullptr || m_feedback->Mapped().size() < (word + 1u) * sizeof(uint32_t)) {
		return;
	}
	StoreFence();
	reinterpret_cast<uint32_t*>(m_feedback->Mapped().data())[word] = active ? 1u : 0u;
	StoreFence();
	m_feedback->Flush(word * sizeof(uint32_t), sizeof(uint32_t));
}

uint32_t BindlessTable::ArmUsageProbe() {
	auto*    feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	uint32_t total    = 0;
	for (const auto& heap: m_heaps) {
		bool marked = false;
		for (uint32_t key = 0; key < heap.entries && key < heap.images.size(); key++) {
			if (heap.images[key]) {
				feedback[heap.region + key] = ProbeMark;
				marked                      = true;
				total++;
			}
		}
		if (marked) {
			m_feedback->Flush(heap.region * sizeof(uint32_t), heap.entries * sizeof(uint32_t));
		}
	}
	return total;
}

void BindlessTable::MarkUnsampled(const Heap& heap, uint32_t key) {
	reinterpret_cast<uint32_t*>(m_feedback->Mapped().data())[heap.region + key] = ProbeMark;
}

void BindlessTable::FlushFeedback(const Heap& heap) {
	m_feedback->Flush(heap.region * sizeof(uint32_t), heap.entries * sizeof(uint32_t));
}

bool BindlessTable::WasSampled(const Heap& heap, uint32_t key) const {
	// The snapshot region was invalidated when TakeRequests last read it (every frame); a key
	// marked after that snapshot reads as sampled, which only settles it again at once.
	return reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data())[heap.region + key] !=
	       ProbeMark;
}

void BindlessTable::CollectUnusedImages(std::vector<ImageId>& unused) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	m_feedback_snapshot->Invalidate(0, vk::DeviceSize {m_next_region} * sizeof(uint32_t));
	// A key resolved, or a heap moved, after the probe was armed reads 0: in use.
	for (const auto& [image, refs]: m_image_refs) {
		if (refs.empty()) {
			continue;
		}
		const bool sampled = std::any_of(refs.begin(), refs.end(), [&](const auto& ref) {
			return snapshot[ref.first->region + ref.second] != ProbeMark;
		});
		if (!sampled) {
			unused.push_back(refs.front().first->images[refs.front().second]);
		}
	}
}

void BindlessTable::CollectSampledImages(std::vector<ImageId>& used) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	m_feedback_snapshot->Invalidate(0, vk::DeviceSize {m_next_region} * sizeof(uint32_t));
	for (const auto& [image, refs]: m_image_refs) {
		if (refs.empty()) {
			continue;
		}
		const bool sampled = std::any_of(refs.begin(), refs.end(), [&](const auto& ref) {
			return snapshot[ref.first->region + ref.second] != ProbeMark;
		});
		if (sampled) {
			used.push_back(refs.front().first->images[refs.front().second]);
		}
	}
}

void BindlessTable::AddImageReference(ImageId id, Heap& heap, uint32_t key) {
	m_image_refs[ImageKey(id)].emplace_back(&heap, key);
}

bool BindlessTable::ReleaseKey(Heap& heap, uint32_t key) {
	const auto id      = heap.images[key];
	bool       no_refs = false;
	if (id) {
		const auto found = m_image_refs.find(ImageKey(id));
		if (found != m_image_refs.end()) {
			std::erase(found->second, std::pair<Heap*, uint32_t> {&heap, key});
			const bool in_heap = std::any_of(found->second.begin(), found->second.end(),
			                                 [&](const auto& ref) { return ref.first == &heap; });
			if (!in_heap) {
				std::erase(heap.resolved, id);
				// Erasing shifts the images RenderExecutor::CommitBindings counted as checked.
				heap.checked_generation = 0;
			}
			if (found->second.empty()) {
				m_image_refs.erase(found);
				no_refs = true;
			}
		}
	}
	const auto entry  = heap.slots[key];
	heap.slots[key]   = 0;
	heap.settled[key] = 0;
	heap.images[key]  = {};
	SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
	if (entry != 0) {
		// v122: the placeholder goes into the slot before it is freed (RetireSlot). Freeing it with
		// the old view still in it let ReleaseViews find the slot again when that image was
		// destroyed and free it a second time: with slot reuse on, two keys then shared one slot
		// (the second key's texture showed on the first key's objects).
		RetireSlot(SlotBinding(heap, entry), SlotIndex(entry));
	}
	return no_refs;
}

void BindlessTable::OnImageUnregistered(ImageId id) {
	const auto found = m_image_refs.find(ImageKey(id));
	if (found == m_image_refs.end()) {
		return;
	}
	for (const auto& [heap, key]: found->second) {
		const auto entry = heap->slots[key];
		heap->slots[key]   = 0;
		heap->settled[key] = 0;
		heap->images[key]  = {};
		// C4b: the key turns pending first; the slot keeps the old view until the GPU is past
		// every draw that may have read the old entry (RetireSlot), then gets placeholder 0 of
		// its own view type (colour 0 of binding b is placeholder entry b).
		SetTranslation(*heap, key, ShaderRecompiler::IR::BindlessPending);
		if (entry != 0) {
			RetireSlot(SlotBinding(*heap, entry), SlotIndex(entry));
		}
		std::erase(heap->resolved, id);
		m_unregistered_keys++;
	}
	m_image_refs.erase(found);
}


void BindlessTable::DescribeKeysForGuestRange(uint64_t lo, uint64_t hi, FILE* out) const {
	const auto TsharpBase = [](const uint32_t* d) {
		return ((static_cast<uint64_t>(d[0]) | (static_cast<uint64_t>(d[1]) << 32u)) & 0xFFFFFFFFFFull) << 8u;
	};
	const auto* translation = m_translation != nullptr && !m_translation->Mapped().empty()
	                              ? reinterpret_cast<const uint32_t*>(m_translation->Mapped().data())
	                              : nullptr;
	uint32_t printed = 0, now_matches = 0, settled_matches = 0, heaps = 0;
	std::vector<uint32_t> words;
	for (const auto& heap: m_heaps) {
		heaps++;
		const uint32_t count = std::max(heap.entries, heap.guest_entries);
		words.assign(static_cast<size_t>(count) * 8u, 0u);
		const bool readable = count != 0 &&
		                      Libs::LibKernel::Memory::TryReadBacking(heap.base + heap.table_offset, words.data(),
		                                                              static_cast<uint64_t>(count) * 32u);
		for (uint32_t key = 0; key < count; key++) {
			const uint64_t now     = readable ? TsharpBase(&words[static_cast<size_t>(key) * 8u]) : 0;
			const uint64_t settled = key < heap.descriptors.size() ? TsharpBase(heap.descriptors[key].data()) : 0;
			const bool     now_in  = now >= lo && now < hi;
			const bool     set_in  = settled >= lo && settled < hi;
			if (!now_in && !set_in) {
				continue;
			}
			now_matches += now_in ? 1u : 0u;
			settled_matches += set_in ? 1u : 0u;
			if (printed++ >= 48) {
				continue;
			}
			const uint32_t slot  = key < heap.slots.size() ? heap.slots[key] : 0u;
			const uint32_t entry = translation != nullptr && heap.region + key < TranslationEntries
			                           ? translation[heap.region + key]
			                           : 0u;
			const auto     image = key < heap.images.size() ? heap.images[key] : ImageId {};
			std::fprintf(out,
			             "[bindless-guest] heap 0x%012" PRIx64 "+0x%x binding %u region %u key %u: T# now 0x%" PRIx64
			             "%s, settled from 0x%" PRIx64 "%s; settled=%u stale=%u slot=0x%x translation=0x%x image=%u.%u\n",
			             heap.base, heap.table_offset, heap.binding, heap.region, key, now, now_in ? " (in range)" : "",
			             settled, set_in ? " (in range)" : "",
			             key < heap.settled.size() ? heap.settled[key] : 0u, key < heap.stale.size() ? heap.stale[key] : 0u,
			             slot, entry, static_cast<uint32_t>(image.index), static_cast<uint32_t>(image.generation));
		}
	}
	std::fprintf(out,
	             "[bindless-guest] guest 0x%" PRIx64 "-0x%" PRIx64 ": %u keys' T# point into it now, %u keys were "
	             "settled from such a T# (%u heaps)\n",
	             lo, hi, now_matches, settled_matches, heaps);
	std::fflush(out);
}

void BindlessTable::CompareSlotCapture(FILE* out) const {
	if (m_translation == nullptr || m_translation->Mapped().empty() || m_feedback == nullptr ||
	    m_feedback->Mapped().size() < (ShaderRecompiler::IR::BindlessSlotCaptureBase +
	                                   ShaderRecompiler::IR::BindlessSlotCaptureEntries) * sizeof(uint32_t)) {
		std::fprintf(out, "[slot-capture] no capture area\n");
		return;
	}
	m_feedback->Invalidate(ShaderRecompiler::IR::BindlessSlotCaptureBase * sizeof(uint32_t),
	                       ShaderRecompiler::IR::BindlessSlotCaptureEntries * sizeof(uint32_t));
	const auto* feedback    = reinterpret_cast<const uint32_t*>(m_feedback->Mapped().data());
	const auto* translation = reinterpret_cast<const uint32_t*>(m_translation->Mapped().data());
	const auto  now_tick    = m_scheduler.CurrentTick();
	uint64_t    sampled = 0, same = 0, pending_both = 0, differ = 0, printed = 0;
	for (const auto& heap: m_heaps) {
		for (uint32_t key = 0; key < heap.entries; key++) {
			const uint32_t index = heap.region + key;
			if (index >= ShaderRecompiler::IR::BindlessSlotCaptureEntries) {
				continue;
			}
			const uint32_t word = feedback[ShaderRecompiler::IR::BindlessSlotCaptureBase + index];
			if (word == 0) {
				continue;
			}
			sampled++;
			const bool     gpu_pending = word == 0xfffffffeu;
			const uint32_t gpu_raw     = gpu_pending ? ShaderRecompiler::IR::BindlessPending : (word & 0x7fffffffu);
			const uint32_t host_raw    = translation[index];
			if (gpu_raw == host_raw) {
				(gpu_pending ? pending_both : same)++;
				continue;
			}
			differ++;
			if (printed++ >= 64) {
				continue;
			}
			// What the slot the GPU read serves now, per the host.
			char gpu_slot[160] = "pending (placeholder 2)";
			if (!gpu_pending) {
				const uint32_t binding = SlotBinding(heap, gpu_raw);
				const uint32_t slot    = SlotIndex(gpu_raw);
				if (binding < ImageArrays && slot < m_slot_expect[binding].size()) {
					const auto& e = m_slot_expect[binding][slot];
					std::snprintf(gpu_slot, sizeof(gpu_slot),
					              "binding %u slot %u, now %ux%ux%u for heap 0x%" PRIx64 " key %u written %" PRIu64
					              " ticks ago",
					              binding, slot, e.width, e.height, e.layers, e.heap, e.key, now_tick - e.tick);
				} else {
					std::snprintf(gpu_slot, sizeof(gpu_slot), "binding %u slot %u (never written by the host)",
					              binding, slot);
				}
			}
			std::fprintf(out,
			             "[slot-capture] heap 0x%012" PRIx64 " binding %u region %u key %u: GPU read 0x%x (%s), host "
			             "now 0x%x, host slot 0x%x settled=%u\n",
			             heap.base, heap.binding, heap.region, key, gpu_raw, gpu_slot, host_raw,
			             key < heap.slots.size() ? heap.slots[key] : 0u, key < heap.settled.size() ? heap.settled[key] : 0u);
		}
	}
	std::fprintf(out,
	             "[slot-capture] %" PRIu64 " keys sampled since start: %" PRIu64 " read what the host has now, %" PRIu64
	             " pending on both sides, %" PRIu64 " differ\n",
	             sampled, same, pending_both, differ);
	std::fflush(out);
}

void BindlessTable::ReportCanaryHits(bool force) const {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CANARY_DETECT");
		return value != nullptr && value[0] == '1';
	}();
	namespace IR = ShaderRecompiler::IR;
	if (!enabled || m_feedback == nullptr ||
	    m_feedback->Mapped().size() < (IR::BindlessCanarySize + IR::BindlessCanarySlotMask + 1u) * sizeof(uint32_t)) {
		return;
	}
	static auto     last_check = std::chrono::steady_clock::now();
	static uint32_t last_count = 0;
	static std::vector<uint32_t> reported(IR::BindlessCanarySlotMask + 1u, 0u);
	const auto      now        = std::chrono::steady_clock::now();
	if (!force && now - last_check < std::chrono::milliseconds(100)) {
		return;
	}
	last_check = now;
	m_feedback->Invalidate(IR::BindlessCanaryWord * sizeof(uint32_t), 8u * sizeof(uint32_t));
	const auto* feedback = reinterpret_cast<const uint32_t*>(m_feedback->Mapped().data());
	const uint32_t count = feedback[IR::BindlessCanaryWord];
	if (count == last_count) {
		if (force) {
			std::fprintf(stderr, "[canary-hit] total %u (none new)\n", count);
		}
		return;
	}
	std::fprintf(stderr,
	             "[canary-hit] total %u (+%u; 4ch %u, 2ch %u, 1ch %u) at tick %" PRIu64
	             "; last: shader %08x%08x translation 0x%08x\n",
	             count, count - last_count, feedback[IR::BindlessCanaryWord + 4u], feedback[IR::BindlessCanaryWord + 5u],
	             feedback[IR::BindlessCanaryWord + 6u], m_scheduler.CurrentTick(), feedback[IR::BindlessCanaryWord + 2u],
	             feedback[IR::BindlessCanaryWord + 1u], feedback[IR::BindlessCanaryWord + 3u]);
	last_count = count;
	m_feedback->Invalidate(IR::BindlessCanarySlots * sizeof(uint32_t),
	                       3u * (IR::BindlessCanarySlotMask + 1u) * sizeof(uint32_t));
	uint32_t printed = 0;
	for (uint32_t slot = 0; slot <= IR::BindlessCanarySlotMask && printed < 32; slot++) {
		const uint32_t hash = feedback[IR::BindlessCanarySlots + slot];
		const uint32_t raw  = feedback[IR::BindlessCanaryRaw + slot];
		const uint32_t size = feedback[IR::BindlessCanarySize + slot];
		if (hash == 0 || reported[slot] == (hash ^ raw ^ 0x5a5a5a5au)) {
			continue;
		}
		reported[slot] = hash ^ raw ^ 0x5a5a5a5au;
		printed++;
		std::string views;
		for (uint32_t binding = 0; binding < ImageArrays; binding++) {
			const auto& list = m_slot_views[binding];
			if (slot >= list.size() || !list[slot]) {
				continue;
			}
			const auto view = static_cast<VkImageView>(list[slot]);
			views += " b" + std::to_string(binding) + (IsImageViewLive(view) ? " live " : " DESTROYED ") +
			         DescribeViewForFault(reinterpret_cast<uint64_t>(view)) + ";";
		}
		std::fprintf(stderr, "[canary-hit]   slot %u: shader ..%08x translation 0x%08x signature %u, descriptor %ux%u, slot now:%s\n",
		             slot, hash, raw & ~0x03000000u, (raw >> 24u) & 3u, size & 0xffffu, size >> 16u,
		             views.empty() ? " (no view)" : views.c_str());
		// v383: the slot's recent writes (any binding), oldest first, with each view's state now
		// and, for a hit whose descriptor size matches none of them, the images' histories.
		const double now_ms = std::chrono::duration<double, std::milli>(
		                          std::chrono::steady_clock::now().time_since_epoch()).count();
		bool any_size_match = false;
		std::vector<VkImage> images;
		for (uint32_t binding = 0; binding < ImageArrays; binding++) {
			const auto& logs = m_slot_log[binding];
			if (slot >= logs.size() || logs[slot].next == 0) {
				continue;
			}
			const auto& log   = logs[slot];
			const auto  count = std::min<uint32_t>(log.next, static_cast<uint32_t>(log.writes.size()));
			for (uint32_t i = 0; i < count; i++) {
				const auto& w = log.writes[(log.next - count + i) % log.writes.size()];
				if (w.image != VK_NULL_HANDLE && (w.width | (w.height << 16u)) == size) {
					any_size_match = true;
				}
				if (w.image != VK_NULL_HANDLE && std::ranges::find(images, w.image) == images.end()) {
					images.push_back(w.image);
				}
				std::fprintf(stderr,
				             "[canary-hit]     b%u write tick %" PRIu64 " (%.0f ms ago): %s view %p image %p %ux%ux%u key %u heap 0x%" PRIx64
				             ", view %s now\n",
				             binding, w.tick, now_ms - w.ms, w.image == VK_NULL_HANDLE ? "placeholder" : "texture",
				             static_cast<void*>(w.view), static_cast<void*>(w.image), w.width, w.height, w.layers, w.key,
				             w.heap, IsImageViewLive(w.view) ? "live" : "DESTROYED");
			}
		}
		if (!any_size_match && size != 0xffffffffu && ImageHistory::Enabled()) {
			std::fprintf(stderr, "[canary-hit]     descriptor size %ux%u matches no write of this slot\n", size & 0xffffu,
			             size >> 16u);
			for (const auto image: images) {
				ImageHistory::Describe(image, stderr);
			}
		}
	}
	std::fflush(stderr);
}

void BindlessTable::AuditForFault() const {
	if (m_translation == nullptr || m_translation->Mapped().empty()) {
		std::fprintf(stderr, "[bindless-audit] no translation buffer\n");
		return;
	}
	ReportCanaryHits(true);
	if (static const bool slot_capture = [] {
		    const char* value = std::getenv("KYTY_SLOT_CAPTURE");
		    return value != nullptr && value[0] == '1';
	    }();
	    slot_capture) {
		CompareSlotCapture(stderr);
	}
	// v127: a translation value that named no slot since the last snapshot (shader clamp report),
	// read live from the feedback buffer, so a value from the faulting frame shows.
	if (m_feedback != nullptr && !m_feedback->Mapped().empty()) {
		constexpr uint32_t word = ShaderRecompiler::IR::BindlessOutOfRangeSlotWord;
		m_feedback->Invalidate(word * sizeof(uint32_t), sizeof(uint32_t));
		const auto value = reinterpret_cast<const uint32_t*>(m_feedback->Mapped().data())[word];
		std::fprintf(stderr, "[bindless-oob] at device loss: %s (0x%08x); seen during the run: %u\n",
		             value != 0 ? "a translation value named no slot" : "none pending", value,
		             g_bindless_oob_seen.load());
	}
	// Entries still queued for publication (KYTY_BINDLESS_PUBLISH_AT_SUBMIT) count as written.
	std::vector<uint32_t> translation_copy(
	    reinterpret_cast<const uint32_t*>(m_translation->Mapped().data()),
	    reinterpret_cast<const uint32_t*>(m_translation->Mapped().data()) + TranslationEntries);
	{
		std::lock_guard lock(m_publish_mutex);
		for (const auto& [entry, value]: m_publish) {
			if (entry < translation_copy.size()) {
				translation_copy[entry] = value;
			}
		}
	}
	const uint32_t* translation = translation_copy.data();
	// Which translation entries belong to a live heap region (and that heap's binding).
	std::vector<int32_t> owner(m_next_region, -1);
	for (size_t h = 0; h < m_heaps.size(); h++) {
		const auto& heap = m_heaps[h];
		for (uint32_t k = 0; k < heap.entries && heap.region + k < m_next_region; k++) {
			owner[heap.region + k] = static_cast<int32_t>(h);
		}
	}
	uint64_t live_entries = 0, stale_entries = 0, pending = 0, placeholder = 0, dead = 0,
	         free_slot = 0, past_end = 0, live_view = 0, mismatched = 0;
	uint32_t printed = 0;
	const auto is_free = [&](uint32_t binding, uint32_t slot) {
		for (const auto& [free, tick]: m_free_slots[binding]) {
			if (free == slot) return true;
		}
		return false;
	};
	for (uint32_t i = 1; i < m_next_region; i++) {
		const auto value = translation[i];
		if (value == ShaderRecompiler::IR::BindlessPending) {
			pending++;
			continue;
		}
		const bool stale = owner[i] < 0;
		(stale ? stale_entries : live_entries)++;
		uint32_t binding = 0;
		if ((value & ShaderRecompiler::IR::BindlessSlotFlagMask) == ShaderRecompiler::IR::BindlessSlotArrayFlag) {
			binding = ShaderRecompiler::IR::BindlessImages2DArray;
		} else if (!stale) {
			binding = m_heaps[static_cast<size_t>(owner[i])].binding;
		} else {
			binding = ImageArrays; // unknown: a moved-away region keeps no binding
		}
		const auto slot = SlotIndex(value);
		const char* problem = nullptr;
		std::string detail;
		if (binding < ImageArrays) {
			if (slot >= m_next_slot[binding]) {
				past_end++;
				problem = "slot past the allocated range";
			} else if (slot < PlaceholderColors) {
				placeholder++;
			} else if (is_free(binding, slot)) {
				free_slot++;
				problem = "slot is on the free list";
			} else {
				const auto& views = m_slot_views[binding];
				const auto  view  = slot < views.size() ? views[slot] : vk::ImageView {};
				if (!view) {
					placeholder++;
				} else if (!IsImageViewLive(static_cast<VkImageView>(view))) {
					dead++;
					problem = "slot holds a destroyed view";
					detail  = DescribeViewForFault(reinterpret_cast<uint64_t>(static_cast<VkImageView>(view)));
				} else {
					live_view++;
					if (!stale) {
						const auto& heap = m_heaps[static_cast<size_t>(owner[i])];
						const auto  key  = i - heap.region;
						if (key < heap.slots.size() && heap.slots[key] != value) {
							mismatched++;
							problem = "GPU translation differs from the heap's CPU slot";
						}
					}
				}
			}
		}
		if (problem != nullptr && printed < 32) {
			printed++;
			std::fprintf(stderr, "[bindless-audit]   entry %u (%s region%s) -> binding %u slot %u: %s %s\n", i,
			             stale ? "moved-away" : "live",
			             stale ? "" : (" of heap " + std::to_string(owner[i])).c_str(), binding, slot, problem,
			             detail.c_str());
		}
	}
	std::fprintf(stderr,
	             "[bindless-audit] translation entries 1..%u: pending %" PRIu64 ", in live regions %" PRIu64
	             ", in moved-away regions %" PRIu64 "; slots: placeholder %" PRIu64 ", live view %" PRIu64
	             ", DESTROYED view %" PRIu64 ", free %" PRIu64 ", past end %" PRIu64
	             ", translation!=heap %" PRIu64 "; heaps %zu, next slots %u/%u/%u/%u\n",
	             m_next_region - 1, pending, live_entries, stale_entries, placeholder, live_view, dead,
	             free_slot, past_end, mismatched, m_heaps.size(), m_next_slot[0], m_next_slot[1],
	             m_next_slot[2], m_next_slot[3]);
	// The views held by slots no translation entry names any more can still be sampled by a draw
	// that cached an old translation: count destroyed views anywhere in the arrays too.
	uint64_t dead_anywhere = 0;
	for (uint32_t b = 0; b < ImageArrays; b++) {
		for (const auto view: m_slot_views[b]) {
			if (view && !IsImageViewLive(static_cast<VkImageView>(view))) dead_anywhere++;
		}
	}
	std::fprintf(stderr, "[bindless-audit] destroyed views held by any slot: %" PRIu64 "\n", dead_anywhere);
	// v295: a view handle the driver reused passes the liveness check above. The image a slot was
	// written with (m_slot_images) tells: its view now belongs to another image, or the image itself
	// is gone (its memory freed), while the descriptor still holds the old view's header.
	uint64_t replaced = 0, image_gone = 0, printed_slots = 0, no_address = 0;
	for (uint32_t b = 0; b < ImageArrays; b++) {
		const auto& views  = m_slot_views[b];
		const auto& images = m_slot_images[b];
		for (uint32_t s = PlaceholderColors; s < views.size() && s < images.size(); s++) {
			const auto view  = static_cast<VkImageView>(views[s]);
			const auto image = images[s];
			if (view == VK_NULL_HANDLE || image == VK_NULL_HANDLE ||
			    std::ranges::find(m_placeholder_views, views[s]) != m_placeholder_views.end()) {
				continue;
			}
			const auto owner = ImageHistory::ImageOfView(view);
			uint64_t   va = 0, size = 0;
			const bool live_image = FaultMapImageVa(image, va, size);
			const bool view_moved = owner != VK_NULL_HANDLE && owner != image;
			// v443k: a live image the map has no address for (a sparse texture; streamBuffer.cpp).
			no_address += live_image && size == 0 ? 1 : 0;
			if (!view_moved && live_image) {
				continue;
			}
			replaced += view_moved ? 1 : 0;
			image_gone += live_image ? 0 : 1;
			if (printed_slots++ < 24) {
				std::fprintf(stderr,
				             "[bindless-audit]   binding %u slot %u: view %p written with image %p%s%s;%s\n", b, s,
				             static_cast<void*>(view), static_cast<void*>(image),
				             view_moved ? " (the view handle now belongs to another image)" : "",
				             live_image ? "" : " (that image is no longer live)",
				             FaultMapImageRange(image).c_str());
			}
		}
	}
	std::fprintf(stderr,
	             "[bindless-audit] slots whose view handle was reused: %" PRIu64
	             ", slots whose image is gone: %" PRIu64 "; slots of live images without an address in the map "
	             "(sparse textures): %" PRIu64 "\n",
	             replaced, image_gone, no_address);
}

} // namespace Libs::Graphics
