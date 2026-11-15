#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_IMGREF_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_IMGREF_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

// KYTY_IMG_REF=<hex guest address> (diagnostics, changes nothing): every draw and dispatch whose
// user data reaches an image descriptor (T#) with that base address, inline in the user SGPRs, in
// a table a user SGPR pair points to, or one pointer further, logs the shader address and where
// the descriptor was found ([img-ref]). Runs before translation, so it also names shaders that are
// refused or skipped and never touch the texture cache (a writer KYTY_IMG_TRACE cannot see).
namespace Libs::Graphics::ImgRef {

inline uint64_t Address() {
	static const uint64_t address = [] {
		const char* value = std::getenv("KYTY_IMG_REF");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : uint64_t {0};
	}();
	return address;
}

constexpr uint32_t kOuterDwords = 0x1000; // 16 KiB from a user SGPR pointer
constexpr uint32_t kInnerDwords = 0x400;  // 4 KiB from a pointer inside that table

// KYTY_IMG_REF_SCAN=1 adds the descriptor-table scan for draws and dispatches (slow: a few fps),
// =2 for dispatches only. Without it only draws whose color or depth targets sit at the address
// are logged, which is cheap.
inline int ScanMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_IMG_REF_SCAN");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	return mode;
}

inline void Target(const char* kind, uint64_t shader, uint64_t target, uint32_t frame) {
	const uint64_t watched = Address();
	if (watched == 0 || target == 0 || (target != watched && (target << 8u) != watched)) return;
	static std::atomic<uint32_t> lines {0};
	if (lines.fetch_add(1, std::memory_order_relaxed) >= 3000) return;
	std::fprintf(stderr, "[img-ref] frame=%u %s ps=0x%" PRIx64 "\n", frame, kind, shader);
}

inline bool IsDescriptor(const uint32_t* dw, uint64_t watched) {
	const uint64_t base = (static_cast<uint64_t>(dw[0]) | (static_cast<uint64_t>(dw[1] & 0xffu) << 32u))
	                      << 8u;
	return base == watched;
}

// Readability per 64 KiB page, cached (VirtualQuery per candidate pointer is far too slow).
// Only addresses at or above 4 GiB: lower ones are host allocations (a stale "readable" there
// crashed the scan), and the guest's direct and flexible memory sit higher.
inline bool Readable(uint64_t address, uint64_t size) {
	if (address < (1ull << 32u) || !GuestRange {address, size}.Valid()) return false;
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, bool> pages;
	std::lock_guard                           lock(mutex);
	for (uint64_t page = address >> 16u; page <= (address + size - 1) >> 16u; page++) {
		auto it = pages.find(page);
		if (it == pages.end()) {
			it = pages.emplace(page, HostMemoryRangeIsReadable(page << 16u, 0x10000)).first;
		}
		if (!it->second) return false;
	}
	return true;
}

inline uint64_t PointerAt(const uint32_t* dw) {
	const uint64_t p = static_cast<uint64_t>(dw[0]) | (static_cast<uint64_t>(dw[1] & 0xffffu) << 32u);
	return p != 0 && (p & 3u) == 0 && Readable(p, kInnerDwords * 4u) ? p : 0;
}

// Returns the dword offset of a matching descriptor in the table at address, or -1.
inline int ScanTable(uint64_t address, uint32_t dwords, uint64_t watched) {
	const auto* table = reinterpret_cast<const uint32_t*>(address);
	for (uint32_t i = 0; i + 1 < dwords; i++) {
		if (IsDescriptor(table + i, watched)) return static_cast<int>(i);
	}
	return -1;
}

inline void Scan(const char* kind, uint64_t shader, const HW::UserSgprInfo& user, uint32_t frame,
                 uint64_t extra_table = 0) {
	const uint64_t watched = Address();
	const bool dispatch = std::strcmp(kind, "dispatch") == 0;
	if (watched == 0 || ScanMode() == 0 || (ScanMode() == 2 && !dispatch)) return;
	static std::atomic<uint32_t> lines {0};
	char where[160] {};
	bool found = false;
	for (uint32_t i = 0; !found && i + 1 < user.count; i++) {
		if (IsDescriptor(user.value + i, watched)) {
			std::snprintf(where, sizeof(where), "inline s%u", i);
			found = true;
		}
	}
	// Per frame, each table is scanned once; its result (a suffix or "") is reused.
	static std::mutex                                  cache_mutex;
	static std::unordered_map<uint64_t, std::string>   cache;
	static std::unordered_set<uint64_t>                inner_misses; // inner tables without it
	static uint32_t                                    cache_frame = UINT32_MAX;
	const auto scan_pointer = [&](uint64_t table, const char* label) {
		if (found || table == 0) return;
		const uint32_t dwords = Readable(table, kOuterDwords * 4u) ? kOuterDwords
		                        : Readable(table, kInnerDwords * 4u) ? kInnerDwords
		                                                              : 0;
		if (dwords == 0) return;
		std::string result;
		{
			std::lock_guard lock(cache_mutex);
			if (cache_frame != frame) {
				cache.clear();
				inner_misses.clear();
				cache_frame = frame;
			}
			if (const auto it = cache.find(table); it != cache.end()) {
				if (it->second.empty()) return;
				std::snprintf(where, sizeof(where), "%s 0x%" PRIx64 "%s", label, table, it->second.c_str());
				found = true;
				return;
			}
		}
		char suffix[128] {};
		if (const int at = ScanTable(table, dwords, watched); at >= 0) {
			std::snprintf(suffix, sizeof(suffix), " +0x%x", at * 4);
		} else {
			const auto* dw = reinterpret_cast<const uint32_t*>(table);
			for (uint32_t j = 0; j + 1 < dwords; j++) {
				const uint64_t inner = PointerAt(dw + j);
				if (inner == 0) continue;
				{
					std::lock_guard lock(cache_mutex);
					if (inner_misses.contains(inner)) continue;
				}
				if (const int in_at = ScanTable(inner, kInnerDwords, watched); in_at >= 0) {
					std::snprintf(suffix, sizeof(suffix), " +0x%x -> 0x%" PRIx64 " +0x%x", j * 4, inner,
					              in_at * 4);
					break;
				}
				std::lock_guard lock(cache_mutex);
				inner_misses.insert(inner);
			}
		}
		result = suffix;
		{
			std::lock_guard lock(cache_mutex);
			cache[table] = result;
		}
		if (!result.empty()) {
			std::snprintf(where, sizeof(where), "%s 0x%" PRIx64 "%s", label, table, result.c_str());
			found = true;
		}
	};
	for (uint32_t i = 0; !found && i + 1 < user.count; i++) {
		char label[16];
		std::snprintf(label, sizeof(label), "s%u", i);
		scan_pointer(PointerAt(user.value + i), label);
	}
	scan_pointer(extra_table, "extra");
	if (!found) return;
	{
		// Each shader at most 4 times, so a frame-wide table every draw sees doesn't fill the log.
		static std::mutex                             seen_mutex;
		static std::unordered_map<uint64_t, uint32_t> seen;
		std::lock_guard                               lock(seen_mutex);
		if (++seen[shader] > 4) return;
	}
	if (lines.fetch_add(1, std::memory_order_relaxed) >= 3000) return;
	std::fprintf(stderr, "[img-ref] frame=%u %s shader=0x%" PRIx64 " %s\n", frame, kind, shader, where);
}

} // namespace Libs::Graphics::ImgRef

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_IMGREF_H_
