#include "graphics/host_gpu/vulkanCommon.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/addressBindings.h"
#include "graphics/host_gpu/aftermath.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageHistory.h"


#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t CHECKPOINT_RING_SIZE = 1u << 16u;

std::array<DiagnosticCheckpoint, CHECKPOINT_RING_SIZE> g_checkpoints {};
std::atomic<uint64_t>                                  g_checkpoint_sequence {0};

const char* OpName(uint32_t op) {
	switch (op) {
		case 0: return "DispatchDirect";
		case 1: return "DrawIndex";
		case 2: return "DrawIndexAuto";
		case 3: return "EopWrite";
		case 4: return "EopInterrupt";
		case 5: return "EopWriteBack";
		case 6: return "EopFlip";
		case 7: return "EopWriteBackFlip";
		case 8: return "EopOnlyFlip";
		case 9: return "DispatchIndirect";
		case 11: return "TexUpload";
		case 12: return "TexDownload";
		case 13: return "TexClear";
		case 14: return "TexCopy";
		default: return "Unknown";
	}
}

void Print(const char* stage, const DiagnosticCheckpoint& checkpoint) {
	std::printf("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64
	            "\n",
	            stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id,
	            checkpoint.arg0, checkpoint.arg1, checkpoint.arg2, checkpoint.arg3,
	            checkpoint.arg4);
	LOGF("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id, checkpoint.arg0,
	     checkpoint.arg1, checkpoint.arg2, checkpoint.arg3, checkpoint.arg4);
}

} // namespace

const DiagnosticCheckpoint* RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint) {
	const auto sequence = g_checkpoint_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      slot     = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
	slot                = checkpoint;
	slot.sequence       = sequence;
	return &slot;
}

// streamBuffer.cpp: names a fault address by the buffer (live or recently destroyed) covering it.
void DescribeDeviceAddress(uint64_t address, uint64_t precision, uint64_t now_tick);

extern std::atomic<int64_t>  g_gpu_progress_ns;   // masterSemaphore.cpp
extern std::atomic<uint64_t> g_gpu_progress_tick;

// Fault addresses from VK_EXT_device_fault (data reads and writes), for the coverage scan.
std::vector<uint64_t> g_fault_data_addresses;
// streamBuffer.cpp: the memory range of a live image.
bool FaultMapImageVa(const void* image, uint64_t& va, uint64_t& size);
void (*g_device_lost_extra)() = nullptr; // set by RenderContext (bindless audit)
// v375, set by RenderContext: the bindless heap keys whose T# points into a guest range.
void (*g_device_fault_guest_scan)(uint64_t lo, uint64_t hi) = nullptr;

static void DumpDeviceFault(GraphicContext& graphics) {
	if (!graphics.device_fault_enabled) {
		return;
	}
	// Several threads see the device loss; one report, finished before any of them exits.
	static std::mutex dump_mutex;
	static bool       dumped = false;
	std::lock_guard   dump_lock(dump_mutex);
	if (dumped) {
		return;
	}
	dumped = true;
	// Under dump_mutex: other threads that see the loss wait here instead of exiting under it.
	if (g_device_lost_extra != nullptr) {
		g_device_lost_extra();
		std::fflush(stderr);
	}
	if (const auto progress = g_gpu_progress_ns.load(); progress != 0) {
		const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
		std::fprintf(stderr, "[device-fault] GPU tick last advanced %.1f ms ago (to %llu)\n",
		             static_cast<double>(now - progress) / 1e6,
		             static_cast<unsigned long long>(g_gpu_progress_tick.load()));
	}
	vk::DeviceFaultCountsEXT counts {};
	auto result = graphics.device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		std::printf("--- Device fault: query failed: %s ---\n", vk::to_string(result).c_str());
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	std::vector<uint8_t>                       binary(static_cast<size_t>(counts.vendorBinarySize));
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
	result = graphics.device.getFaultInfoEXT(&counts, &info);
	std::printf("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	            vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	            counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	LOGF("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	     vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	     counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& address = addresses[i];
		std::printf("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		            vk::to_string(address.addressType).c_str(),
		            static_cast<uint64_t>(address.reportedAddress),
		            static_cast<uint64_t>(address.addressPrecision));
		std::fprintf(stderr, "[device-fault] address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		             vk::to_string(address.addressType).c_str(),
		             static_cast<uint64_t>(address.reportedAddress),
		             static_cast<uint64_t>(address.addressPrecision));
		LOGF("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		     vk::to_string(address.addressType).c_str(),
		     static_cast<uint64_t>(address.reportedAddress),
		     static_cast<uint64_t>(address.addressPrecision));
		if (address.addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerUnknown &&
		    address.addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerInvalid &&
		    address.addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerFault) {
			DescribeDeviceAddress(address.reportedAddress, address.addressPrecision, 0);
			AddressBindings::Describe(address.reportedAddress, address.addressPrecision, stderr);
			// v375: which bindless heap keys point at the images that were freed there (Crash B:
			// the stale read goes through a bindless slot, C8).
			if (g_device_fault_guest_scan != nullptr && ImageHistory::Enabled()) {
				std::vector<std::pair<uint64_t, uint64_t>> ranges;
				for (const auto handle: AddressBindings::RecentlyUnboundImages(address.reportedAddress, 30.0)) {
					ImageHistory::RecentGuestRanges(reinterpret_cast<VkImage>(handle), 30000.0, &ranges);
				}
				std::sort(ranges.begin(), ranges.end());
				ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
				for (const auto& [lo, hi]: ranges) {
					g_device_fault_guest_scan(lo, hi);
				}
			}
			g_fault_data_addresses.push_back(static_cast<uint64_t>(address.reportedAddress));
		}
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& vendor = vendors[i];
		std::printf("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		            vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		            static_cast<uint64_t>(vendor.vendorFaultData));
		LOGF("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		     vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		     static_cast<uint64_t>(vendor.vendorFaultData));
	}
	std::fflush(stdout);
	if (!binary.empty()) {
		if (auto* file = std::fopen("_device_fault.nv-gpudmp", "wb"); file != nullptr) {
			std::fwrite(binary.data(), 1, binary.size(), file);
			std::fclose(file);
			std::printf("  vendor binary written to _device_fault.nv-gpudmp\n");
		}
	}
	std::fflush(stdout);
}

// The draws and dispatches recorded per tick (CP side, always on): when a wait never finishes, the
// work of the first tick the GPU has not completed is named ([gpu-hang] work lines), without
// KYTY_NV_CHECKPOINTS (which serializes the GPU and hides the hang).
namespace {
struct TickWork {
	std::atomic<uint64_t> tick {0};
	uint64_t              a = 0, b = 0;
	char                  kind = 0;
	// KYTY_FAULT_MAP=1: the image views the draw bound (descriptors, then attachments) and the
	// bindless translation regions its shaders index.
	uint8_t                  num_views   = 0;
	uint8_t                  num_regions = 0;
	std::array<uint64_t, 24> views {};
	std::array<uint32_t, 6>  regions {};
	// KYTY_FAULT_MAP=1: the buffer bindings, as device-address ranges [lo, lo + size) (0: unknown).
	uint8_t                  num_buffers = 0;
	struct Range {
		uint64_t lo = 0, size = 0;
	};
	std::array<Range, 16> buffers {};
	// The memory ranges of the images behind `views` (lo 0: not a known live image).
	std::array<Range, 24> image_ranges {};
};
constexpr size_t      TickWorkSize = 1u << 16u;
TickWork              g_tick_work[TickWorkSize];
std::atomic<uint64_t> g_tick_work_next {0};
thread_local TickWork* t_last_work = nullptr;

bool TickWorkDetail() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FAULT_MAP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
} // namespace

// image.cpp: "live|destroyed <extent fmt guest img>" of a view the stale-view registry knows.
std::string DescribeViewForFault(uint64_t view);
// bindlessTable.cpp (via renderContext): audit of the bindless table at device loss.


void NoteTickWork(uint64_t tick, char kind, uint64_t a, uint64_t b) {
	auto& slot = g_tick_work[g_tick_work_next.fetch_add(1, std::memory_order_relaxed) % TickWorkSize];
	slot.tick.store(0, std::memory_order_relaxed);
	slot.a           = a;
	slot.b           = b;
	slot.kind        = kind;
	slot.num_views   = 0;
	slot.num_regions = 0;
	slot.num_buffers = 0;
	slot.tick.store(tick, std::memory_order_release);
	t_last_work = &slot;
}

// The work NoteTickWork just recorded on this thread: what it bound.
void NoteTickWorkViews(const uint64_t* views, size_t count, const uint32_t* regions,
                       size_t num_regions) {
	if (!TickWorkDetail() || t_last_work == nullptr) {
		return;
	}
	auto& slot      = *t_last_work;
	slot.num_views  = static_cast<uint8_t>(std::min(count, slot.views.size()));
	std::copy_n(views, slot.num_views, slot.views.begin());
	for (uint32_t i = 0; i < slot.num_views; i++) {
		slot.image_ranges[i] = {};
		const auto image = ImageHistory::ImageOfView(reinterpret_cast<VkImageView>(views[i]));
		uint64_t   va = 0, size = 0;
		if (image != VK_NULL_HANDLE && FaultMapImageVa(image, va, size)) {
			slot.image_ranges[i] = {va, size};
		}
	}
	slot.num_regions = static_cast<uint8_t>(std::min(num_regions, slot.regions.size()));
	std::copy_n(regions, slot.num_regions, slot.regions.begin());
}

// streamBuffer.cpp: device address of a VkBuffer's memory (0: unknown).
uint64_t FaultMapBufferVa(const void* handle);

// The work NoteTickWork just recorded on this thread: its buffer bindings (handle, offset, range
// triples; range VK_WHOLE_SIZE counts as unknown size).
void NoteTickWorkBuffers(const uint64_t* triples, size_t count) {
	if (!TickWorkDetail() || t_last_work == nullptr) {
		return;
	}
	auto& slot       = *t_last_work;
	slot.num_buffers = static_cast<uint8_t>(std::min(count, slot.buffers.size()));
	for (size_t i = 0; i < slot.num_buffers; i++) {
		if (triples[i * 3] == 0) {
			slot.buffers[i] = {0, ~uint64_t {0}}; // null descriptor (robustness2 nullDescriptor)
			continue;
		}
		const auto va = FaultMapBufferVa(reinterpret_cast<const void*>(triples[i * 3]));
		slot.buffers[i].lo   = va == 0 ? 0 : va + triples[i * 3 + 1];
		slot.buffers[i].size = triples[i * 3 + 2] == VK_WHOLE_SIZE ? 0 : triples[i * 3 + 2];
	}
}


// Every recorded draw or dispatch (the last TickWorkSize) whose buffer bindings cover a fault
// address: if one does, the GPU reached freed memory through a bound range (a stale or
// oversized binding); if none does, the read came from no buffer binding.
static void DumpFaultCoverage() {
	if (!TickWorkDetail() || g_fault_data_addresses.empty()) {
		return;
	}
	uint64_t scanned = 0, with_buffers = 0, covering = 0, unknown = 0, image_hits = 0, views_known = 0,
	         views_unknown = 0;
	for (const auto& slot: g_tick_work) {
		const auto tick = slot.tick.load(std::memory_order_acquire);
		if (tick == 0) continue;
		scanned++;
		if (slot.num_buffers != 0) with_buffers++;
		for (uint32_t i = 0; i < slot.num_views; i++) {
			if (slot.views[i] == 0) continue; // sampler entries and null views
			const auto& range = slot.image_ranges[i];
			if (range.lo == 0) {
				views_unknown++;
				continue;
			}
			views_known++;
			for (const auto address: g_fault_data_addresses) {
				const auto page = address & ~uint64_t {0xfff};
				if (page + 0x1000 > range.lo && page < range.lo + range.size && ++image_hits <= 40) {
					std::fprintf(stderr,
					             "[fault-cover] tick %" PRIu64 " work %c %016" PRIx64 " %016" PRIx64
					             " view %u (%p): image memory 0x%016" PRIx64 "+0x%" PRIx64 " covers fault 0x%016" PRIx64 "\n",
					             tick, slot.kind, slot.a, slot.b, i, reinterpret_cast<void*>(slot.views[i]),
					             range.lo, range.size, address);
				}
			}
		}
		for (uint32_t i = 0; i < slot.num_buffers; i++) {
			const auto& range = slot.buffers[i];
			if (range.lo == 0 && range.size == ~uint64_t {0}) {
				continue; // null descriptor
			}
			if (range.lo == 0) {
				unknown++;
				continue;
			}
			for (const auto address: g_fault_data_addresses) {
				const auto page = address & ~uint64_t {0xfff};
				const bool hit  = range.size == 0 ? (address >= range.lo && address - range.lo < (64ull << 20u))
				                                  : (page + 0x1000 > range.lo && page < range.lo + range.size);
				if (hit && ++covering <= 40) {
					std::fprintf(stderr,
					             "[fault-cover] tick %" PRIu64 " work %c %016" PRIx64 " %016" PRIx64
					             " binding %u: dev 0x%016" PRIx64 "+0x%" PRIx64 "%s covers fault 0x%016" PRIx64 "\n",
					             tick, slot.kind, slot.a, slot.b, i, range.lo, range.size,
					             range.size == 0 ? " (whole size, within 64 MB)" : "", address);
				}
			}
		}
	}
	std::fprintf(stderr,
	             "[fault-cover] scanned %" PRIu64 " recorded draws/dispatches (%" PRIu64
	             " with buffer bindings, %" PRIu64 " bindings of unknown address): %" PRIu64
	             " bindings cover a fault address\n",
	             scanned, with_buffers, unknown, covering);
	std::fprintf(stderr,
	             "[fault-cover] image views: %" PRIu64 " with known memory, %" PRIu64
	             " unknown (destroyed since, or not a texture-cache image): %" PRIu64 " cover a fault address\n",
	             views_known, views_unknown, image_hits);
}

void DumpHangWork(uint64_t first_incomplete, uint64_t waited) {
	struct Entry {
		char     kind;
		uint64_t a, b;
		uint32_t count;
	};
	std::vector<Entry> entries;
	for (const auto& slot: g_tick_work) {
		const auto tick = slot.tick.load(std::memory_order_acquire);
		if (tick != first_incomplete) continue;
		auto it = std::find_if(entries.begin(), entries.end(), [&](const Entry& e) {
			return e.kind == slot.kind && e.a == slot.a && e.b == slot.b;
		});
		if (it == entries.end()) {
			entries.push_back({slot.kind, slot.a, slot.b, 1});
		} else {
			it->count++;
		}
	}
	std::fprintf(stderr, "[gpu-hang] first unfinished tick %" PRIu64 " (waited %" PRIu64 "): %zu distinct draws/dispatches\n",
	             first_incomplete, waited, entries.size());
	for (size_t i = 0; i < entries.size() && i < 60; i++) {
		std::fprintf(stderr, "[gpu-hang]   work %c %016" PRIx64 " %016" PRIx64 " x%u\n", entries[i].kind,
		             entries[i].a, entries[i].b, entries[i].count);
	}
	if (TickWorkDetail()) {
		// Every draw of the unfinished tick with what it bound: the fault report's destroyed
		// image (img=) is either among these views or the draw reached it some other way.
		uint32_t printed = 0;
		for (const auto& slot: g_tick_work) {
			if (slot.tick.load(std::memory_order_acquire) != first_incomplete || printed >= 400) {
				continue;
			}
			printed++;
			std::string regions;
			for (uint32_t r = 0; r < slot.num_regions; r++) {
				regions += " " + std::to_string(slot.regions[r]);
			}
			std::fprintf(stderr, "[gpu-hang]   draw %c %016" PRIx64 " %016" PRIx64 " regions:%s\n",
			             slot.kind, slot.a, slot.b, regions.empty() ? " none" : regions.c_str());
			for (uint32_t i = 0; i < slot.num_buffers; i++) {
				std::fprintf(stderr, "[gpu-hang]     buffer %u: dev 0x%016" PRIx64 "+0x%" PRIx64 "\n", i,
				             slot.buffers[i].lo, slot.buffers[i].size);
			}
			for (uint32_t v = 0; v < slot.num_views; v++) {
				std::fprintf(stderr, "[gpu-hang]     view %p %s\n",
				             reinterpret_cast<void*>(slot.views[v]),
				             DescribeViewForFault(slot.views[v]).c_str());
			}
		}
	}
	std::fflush(stderr);
}

// A wait that has not finished after seconds: the GPU's last checkpoints (the draw or dispatch it is
// stuck in), without losing the device. Requires KYTY_NV_CHECKPOINTS=1.
void DumpGpuHangCheckpoints(GraphicContext& graphics, uint64_t tick, double seconds) {
	if (!graphics.diagnostic_checkpoints_enabled || graphics.queue == nullptr) {
		std::fprintf(stderr, "[gpu-hang] tick %" PRIu64 " not done after %.0f s (KYTY_NV_CHECKPOINTS=1 names the work)\n",
		             tick, seconds);
		return;
	}
	std::vector<vk::CheckpointDataNV> data;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		data = graphics.queue.getCheckpointDataNV();
	}
	std::fprintf(stderr, "[gpu-hang] tick %" PRIu64 " not done after %.0f s; checkpoints %zu, last recorded seq=%" PRIu64 "\n",
	             tick, seconds, data.size(), g_checkpoint_sequence.load(std::memory_order_relaxed));
	for (const auto& entry: data) {
		const auto* checkpoint = static_cast<const DiagnosticCheckpoint*>(entry.pCheckpointMarker);
		const bool  in_ring    = checkpoint >= g_checkpoints.data() &&
		                      checkpoint < g_checkpoints.data() + g_checkpoints.size();
		if (checkpoint == nullptr || !in_ring) {
			continue;
		}
		std::fprintf(stderr,
		             "[gpu-hang]   [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
		             vk::to_string(entry.stage).c_str(), checkpoint->sequence, OpName(checkpoint->op),
		             checkpoint->submit_id, checkpoint->arg0, checkpoint->arg1, checkpoint->arg2,
		             checkpoint->arg3, checkpoint->arg4);
	}
	std::fflush(stderr);
}

void DumpDeviceLossDiagnostics(GraphicContext& graphics) {
	Aftermath::WaitForCrashDump();
	DumpDeviceFault(graphics);
	// v127: the draws of the first tick the GPU did not finish, with their views, here as well:
	// the thread that sees the device loss first (an SRT readback, say) exits right after this,
	// before the GPU thread's wait can print them (10:56 capture). Once per process.
	{
		static std::mutex hang_mutex;
		static bool       hang_dumped = false;
		std::lock_guard   hang_lock(hang_mutex);
		if (!hang_dumped) {
			hang_dumped         = true;
			const auto progress = g_gpu_progress_tick.load();
			uint64_t   first    = UINT64_MAX;
			for (const auto& slot: g_tick_work) {
				const auto tick = slot.tick.load(std::memory_order_acquire);
				if (tick > progress && tick < first) first = tick;
			}
			if (first != UINT64_MAX) {
				std::fprintf(stderr, "[device-fault] GPU completed tick %" PRIu64
				                     "; draws of the first unfinished recorded tick:\n",
				             static_cast<uint64_t>(progress));
				DumpHangWork(first, progress);
				DumpFaultCoverage();
			} else {
				std::fprintf(stderr, "[device-fault] no recorded work after tick %" PRIu64 "\n",
				             static_cast<uint64_t>(progress));
				// v398 (11-11, the upgrade menu's null write): the progress tick reads as UINT64_MAX
				// after the loss, so name the most recent recorded work instead: the last 4 ticks'
				// distinct draws/dispatches, and (KYTY_FAULT_MAP=1) every one of them with a buffer
				// binding of no address (a null descriptor, or a buffer the fault map does not know).
				uint64_t last = 0;
				for (const auto& slot: g_tick_work) {
					last = std::max(last, slot.tick.load(std::memory_order_acquire));
				}
				for (uint64_t tick = last >= 3 ? last - 3 : 0; last != 0 && tick <= last; tick++) {
					DumpHangWork(tick, progress);
				}
				if (TickWorkDetail()) {
					uint32_t printed = 0;
					for (const auto& slot: g_tick_work) {
						const auto tick = slot.tick.load(std::memory_order_acquire);
						if (last == 0 || tick + 3 < last || tick > last || printed >= 200) {
							continue;
						}
						bool null_binding = false;
						for (uint32_t i = 0; i < slot.num_buffers; i++) {
							null_binding |= slot.buffers[i].lo == 0;
						}
						if (!null_binding) {
							continue;
						}
						printed++;
						std::string bindings;
						for (uint32_t i = 0; i < slot.num_buffers; i++) {
							char text[64];
							std::snprintf(text, sizeof(text), " %u:%s", i,
							              slot.buffers[i].lo != 0 ? "ok"
							              : slot.buffers[i].size == ~uint64_t {0} ? "null"
							                                                      : "unknown");
							bindings += text;
						}
						std::fprintf(stderr, "[device-fault] tick %" PRIu64 " work %c %016" PRIx64 " %016" PRIx64
						                     " buffer bindings:%s\n",
						             tick, slot.kind, slot.a, slot.b, bindings.c_str());
					}
				}
			}
		}
	}

	if (!graphics.diagnostic_checkpoints_enabled || graphics.queue == nullptr) {
		return;
	}
	std::vector<vk::CheckpointDataNV> data;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		data = graphics.queue.getCheckpointDataNV();
	}
	std::printf("--- Diagnostic checkpoints (%zu, last recorded seq=%" PRIu64 ") ---\n",
	            data.size(), g_checkpoint_sequence.load(std::memory_order_relaxed));
	LOGF("--- Diagnostic checkpoints (%zu, last recorded seq=%" PRIu64 ") ---\n", data.size(),
	     g_checkpoint_sequence.load(std::memory_order_relaxed));
	for (const auto& entry: data) {
		const auto* checkpoint = static_cast<const DiagnosticCheckpoint*>(entry.pCheckpointMarker);
		const auto  stage      = vk::to_string(entry.stage);
		const bool  in_ring    = checkpoint >= g_checkpoints.data() &&
		                      checkpoint < g_checkpoints.data() + g_checkpoints.size();
		std::printf("  [%s] marker=%p ring=[%p, %p) %s\n", stage.c_str(),
		            static_cast<const void*>(checkpoint),
		            static_cast<const void*>(g_checkpoints.data()),
		            static_cast<const void*>(g_checkpoints.data() + g_checkpoints.size()),
		            in_ring ? "inside" : "OUTSIDE");
		if (checkpoint == nullptr || !in_ring) {
			continue;
		}
		Print(stage.c_str(), *checkpoint);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics
