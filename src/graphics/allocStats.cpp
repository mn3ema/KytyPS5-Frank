// KYTY_FPS_LOG: heap allocations per second, next to the frame rate ([alloc] lines, renderDoc.cpp).
// Every C++ allocation of the process goes through these replacements of the global operator new
// and delete; they count into per-thread slots (a relaxed add on a line no other thread writes)
// and use malloc and free as the default ones do.

//
// v438zk (next-phase plan, C3): KYTY_ALLOC_SITES=<n> (off by default; 4096 is a good n). Every
// n-th allocation of a thread records the code addresses that asked for it (8 return addresses,
// no allocation, no lock) in a fixed table by call site; [alloc-site] lists the heaviest sites
// every 5 s with the [alloc] line (KYTY_FPS_LOG). A sample stands for n allocations of its
// thread. Names need the .pdb next to the exe (Common::HostAddressName).

#include "common/sampler.h" // Common::HostAddressName
#include "graphics/host_gpu/renderer/syncToggle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#if defined(_WIN32)
#include <malloc.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <execinfo.h>
#endif

namespace Libs::Graphics {

namespace {
struct alignas(64) AllocSlot {
	std::atomic<uint64_t> count {0};
};
constexpr uint32_t        AllocSlots = 64;
AllocSlot                 g_alloc_slots[AllocSlots];
std::atomic<uint32_t>     g_alloc_next_slot {0};

// KYTY_ALLOC_SITES: the sampling interval; 0: off. Read at the first allocation, which can come
// before main: std::getenv allocates nothing.
uint32_t AllocSiteInterval() {
	static const uint32_t interval = [] {
		const char* value = std::getenv("KYTY_ALLOC_SITES");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		return static_cast<uint32_t>(n > 0x40000000ul ? 0x40000000ul : n);
	}();
	return interval;
}

constexpr uint32_t SiteFrames = 10;
constexpr uint32_t SiteSlots  = 8192; // open addressing; a full table drops new sites (counted)
struct AllocSite {
	std::atomic<uint64_t> key {0}; // 0: free
	std::atomic<uint32_t> samples {0};
	std::atomic<uint64_t> bytes {0};
	void*                 frames[SiteFrames] {};
};
AllocSite             g_alloc_sites[SiteSlots];
std::atomic<uint32_t> g_alloc_site_dropped {0};
thread_local bool     t_alloc_site_busy = false; // the sample or the report itself allocating

void SampleAllocSite(std::size_t size) {
	if (t_alloc_site_busy) {
		return;
	}
	t_alloc_site_busy = true;
	void* frames[SiteFrames] {};
#if defined(_WIN32)
	// v438zl: from this function's caller on. Skipping three frames (this function, Allocate,
	// operator new) dropped the allocating function and its caller wherever those were inlined
	// (A6's rows began two calls up); the report leaves the allocator's own frames out by name.
	const auto count = RtlCaptureStackBackTrace(1, SiteFrames, frames, nullptr);
#else
	void*      raw[SiteFrames + 3] {};
	const auto got   = backtrace(raw, static_cast<int>(SiteFrames + 3));
	const auto count = static_cast<uint32_t>(got > 3 ? got - 3 : 0);
	for (uint32_t i = 0; i < count; i++) {
		frames[i] = raw[i + 3];
	}
#endif
	uint64_t key = 0xcbf29ce484222325ull;
	for (uint32_t i = 0; i < count; i++) {
		key = (key ^ reinterpret_cast<uintptr_t>(frames[i])) * 0x100000001b3ull;
	}
	key |= 1u; // never 0
	for (uint32_t probe = 0; probe < 64; probe++) {
		auto&    site     = g_alloc_sites[(key + probe) % SiteSlots];
		uint64_t expected = site.key.load(std::memory_order_acquire);
		if (expected == 0 && site.key.compare_exchange_strong(expected, key, std::memory_order_acq_rel)) {
			for (uint32_t i = 0; i < SiteFrames; i++) {
				site.frames[i] = frames[i]; // read only by the report, seconds later
			}
			expected = key;
		}
		if (expected == key) {
			site.samples.fetch_add(1, std::memory_order_relaxed);
			site.bytes.fetch_add(size, std::memory_order_relaxed);
			t_alloc_site_busy = false;
			return;
		}
	}
	g_alloc_site_dropped.fetch_add(1, std::memory_order_relaxed);
	t_alloc_site_busy = false;
}

// v445o (KYTY_ALLOC_COUNT_LOCAL; on by default since v445p, =0 turns it off; toggle mode "alloccount"): the count above is a
// locked add into one of 64 slots that more than 64 threads share, behind a guarded thread-local,
// on every operator new (0.31 ms a frame of the GPU thread in g450g). With the cut on a thread
// counts into a block only it writes (a load and a store, no lock prefix); the blocks are in a
// list AllocCountTotal walks. A block comes from malloc at its thread's first counted allocation
// and is never freed or reused: a thread that ended leaves its count in the total (64 bytes per
// thread ever started). The sampling countdown of KYTY_ALLOC_SITES is in the block too.
// The mode is decided at the first allocation, which can come before main: the toggle's mask and
// period are read from the environment without allocating (syncToggle.h), its state is one load.
struct alignas(64) AllocBlock {
	std::atomic<uint64_t> count {0}; // written by its thread only
	uint32_t              until_sample = 0;
	AllocBlock*           next         = nullptr;
};
std::atomic<AllocBlock*>          g_alloc_blocks {nullptr};
constinit thread_local AllocBlock* t_alloc_block = nullptr;
enum : int { AllocModeUnknown = 0, AllocModeShared = 1, AllocModeLocal = 2, AllocModeToggled = 3 };
std::atomic<int> g_alloc_mode {AllocModeUnknown};

int InitAllocMode() {
	// v445p: on by default (KYTY_ALLOC_COUNT_LOCAL=0: the shared slots); a toggled mode overrides
	// the switch. The default is decided here: this runs at the first allocation, before main
	// applies its table of defaults to the environment.
	const char* value = std::getenv("KYTY_ALLOC_COUNT_LOCAL");
	int         mode  = value != nullptr && value[0] == '0' ? AllocModeShared : AllocModeLocal;
	if (SyncToggle::AllocCountConfigured()) {
		mode = AllocModeToggled;
	}
	g_alloc_mode.store(mode, std::memory_order_relaxed); // every thread computes the same value
	return mode;
}

AllocBlock* CreateAllocBlock() {
	void* memory = nullptr;
#if defined(_WIN32)
	memory = _aligned_malloc(sizeof(AllocBlock), alignof(AllocBlock));
#else
	memory = std::aligned_alloc(alignof(AllocBlock), sizeof(AllocBlock));
#endif
	if (memory == nullptr) {
		return nullptr;
	}
	auto* block         = new (memory) AllocBlock(); // placement: no allocation
	block->until_sample = AllocSiteInterval();
	block->next         = g_alloc_blocks.load(std::memory_order_relaxed);
	while (!g_alloc_blocks.compare_exchange_weak(block->next, block, std::memory_order_release,
	                                             std::memory_order_relaxed)) {
	}
	t_alloc_block = block;
	return block;
}

inline void CountAlloc(std::size_t size) {
	int mode = g_alloc_mode.load(std::memory_order_relaxed);
	if (mode == AllocModeUnknown) [[unlikely]] {
		mode = InitAllocMode();
	}
	if (mode == AllocModeLocal ||
	    (mode == AllocModeToggled && SyncToggle::State().load(std::memory_order_relaxed))) {
		auto* block = t_alloc_block;
		if (block == nullptr) [[unlikely]] {
			block = CreateAllocBlock();
		}
		if (block != nullptr) [[likely]] {
			block->count.store(block->count.load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
			if (block->until_sample != 0 && --block->until_sample == 0) {
				block->until_sample = AllocSiteInterval();
				SampleAllocSite(size);
			}
			return;
		}
	}
	thread_local AllocSlot* slot =
	    &g_alloc_slots[g_alloc_next_slot.fetch_add(1, std::memory_order_relaxed) % AllocSlots];
	slot->count.fetch_add(1, std::memory_order_relaxed);
	thread_local uint32_t until_sample = AllocSiteInterval(); // 0: off
	if (until_sample != 0 && --until_sample == 0) {
		until_sample = AllocSiteInterval();
		SampleAllocSite(size);
	}
}

void* Allocate(std::size_t size) {
	CountAlloc(size);
	for (;;) {
		if (void* p = std::malloc(size != 0 ? size : 1); p != nullptr) {
			return p;
		}
		if (auto handler = std::get_new_handler(); handler != nullptr) {
			handler();
		} else {
			std::abort(); // out of memory
		}
	}
}

void* AllocateAligned(std::size_t size, std::align_val_t alignment) {
	CountAlloc(size);
	const auto align = static_cast<std::size_t>(alignment);
	for (;;) {
#if defined(_WIN32)
		void* p = _aligned_malloc(size != 0 ? size : 1, align);
#else
		void* p = std::aligned_alloc(align, ((size != 0 ? size : 1) + align - 1) / align * align);
#endif
		if (p != nullptr) {
			return p;
		}
		if (auto handler = std::get_new_handler(); handler != nullptr) {
			handler();
		} else {
			std::abort(); // out of memory
		}
	}
}

void FreeAligned(void* p) {
#if defined(_WIN32)
	_aligned_free(p);
#else
	std::free(p);
#endif
}
} // namespace

uint64_t AllocCountTotal() {
	uint64_t total = 0;
	for (const auto& slot: g_alloc_slots) {
		total += slot.count.load(std::memory_order_relaxed);
	}
	for (const auto* block = g_alloc_blocks.load(std::memory_order_acquire); block != nullptr;
	     block             = block->next) {
		total += block->count.load(std::memory_order_relaxed);
	}
	return total;
}

// v445o (2): of the total, what was counted in thread-owned blocks ([cut-hits], sync.cpp).
uint64_t AllocCountLocalTotal() {
	uint64_t total = 0;
	for (const auto* block = g_alloc_blocks.load(std::memory_order_acquire); block != nullptr;
	     block             = block->next) {
		total += block->count.load(std::memory_order_relaxed);
	}
	return total;
}

// [alloc-site]: the sites sampled since the last call, heaviest first (KYTY_ALLOC_SITES). Called
// with the [alloc] line; counts start over, the sites stay.
void AllocSitesReport(double elapsed) {
	const auto interval = AllocSiteInterval();
	if (interval == 0 || elapsed <= 0.0) {
		return;
	}
	t_alloc_site_busy = true; // the report's own allocations are not sampled
	struct Row {
		uint32_t samples;
		uint64_t bytes;
		uint32_t index;
	};
	std::array<Row, 16> top {};
	uint64_t            total = 0;
	uint32_t            sites = 0;
	for (uint32_t i = 0; i < SiteSlots; i++) {
		auto& site = g_alloc_sites[i];
		if (site.key.load(std::memory_order_acquire) == 0) {
			continue;
		}
		const Row row {site.samples.exchange(0, std::memory_order_relaxed), site.bytes.exchange(0, std::memory_order_relaxed), i};
		if (row.samples == 0) {
			continue;
		}
		total += row.samples;
		sites++;
		auto least = std::min_element(top.begin(), top.end(), [](const Row& a, const Row& b) { return a.samples < b.samples; });
		if (row.samples > least->samples) {
			*least = row;
		}
	}
	std::sort(top.begin(), top.end(), [](const Row& a, const Row& b) { return a.samples > b.samples; });
	std::fprintf(stderr, "[alloc-site] %.1f s: %llu samples (one per %u allocations of a thread, ~%.0f allocations/s) at %u sites, %u dropped (table full)\n",
	             elapsed, static_cast<unsigned long long>(total), interval,
	             static_cast<double>(total) * interval / elapsed, sites, g_alloc_site_dropped.exchange(0, std::memory_order_relaxed));
	for (const auto& row: top) {
		if (row.samples == 0) {
			break;
		}
		std::string names;
		for (const auto* frame: g_alloc_sites[row.index].frames) {
			if (frame == nullptr) {
				break;
			}
			auto name = Common::HostAddressName(frame);
			if (names.empty() && (name.find("operator new") != std::string::npos ||
			                      name.find("Allocate") != std::string::npos ||
			                      name.find("SampleAllocSite") != std::string::npos ||
			                      name.find("CountAlloc") != std::string::npos)) {
				continue; // the allocator itself
			}
			if (name.empty()) {
				char text[32];
				std::snprintf(text, sizeof(text), " %p", frame);
				name = text;
			}
			names += names.empty() ? "" : " <-";
			names += name;
		}
		std::fprintf(stderr, "[alloc-site]   %5.1f %%  ~%.0f/s  avg %llu B |%s\n",
		             total != 0 ? 100.0 * row.samples / static_cast<double>(total) : 0.0,
		             static_cast<double>(row.samples) * interval / elapsed,
		             static_cast<unsigned long long>(row.bytes / row.samples), names.c_str());
	}
	t_alloc_site_busy = false;
}

} // namespace Libs::Graphics

using Libs::Graphics::Allocate;
using Libs::Graphics::AllocateAligned;
using Libs::Graphics::FreeAligned;

void* operator new(std::size_t size) { return Allocate(size); }
void* operator new[](std::size_t size) { return Allocate(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return Allocate(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return Allocate(size); }
void* operator new(std::size_t size, std::align_val_t alignment) { return AllocateAligned(size, alignment); }
void* operator new[](std::size_t size, std::align_val_t alignment) { return AllocateAligned(size, alignment); }
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
	return AllocateAligned(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
	return AllocateAligned(size, alignment);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { FreeAligned(p); }
