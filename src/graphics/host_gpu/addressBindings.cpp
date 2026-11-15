#include "graphics/host_gpu/addressBindings.h"

#include "graphics/host_gpu/renderer/image/imageHistory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::AddressBindings {

namespace {

struct Range {
	int32_t  type     = 0;
	uint64_t handle   = 0;
	uint64_t address  = 0;
	uint64_t size     = 0;
	double   bound    = 0; // seconds since start
	double   unbound  = -1;
	bool     internal = false;
};

struct Map {
	std::mutex                          mutex;
	// Live ranges, keyed by (handle, address): an object can be bound more than once (sparse,
	// re-binds), and handles are reused by the driver after an unbind.
	std::unordered_map<uint64_t, std::vector<Range>> live;
	std::deque<Range>                   gone;
	uint64_t                            binds = 0, unbinds = 0, unmatched = 0;
};

Map& Get() {
	static Map map;
	return map;
}

double Now() {
	static const auto start = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

constexpr size_t GoneCapacity = 262144;

const char* TypeName(int32_t type) {
	switch (type) {
		case 8: return "DeviceMemory";  // VK_OBJECT_TYPE_DEVICE_MEMORY
		case 9: return "Buffer";        // VK_OBJECT_TYPE_BUFFER
		case 10: return "Image";        // VK_OBJECT_TYPE_IMAGE
		case 1000150000: return "AccelerationStructure";
		default: return "Object";
	}
}

void Print(FILE* out, const char* what, const Range& r, uint64_t address, double now) {
	std::fprintf(out,
	             "[addr-map] %s %s handle=0x%016" PRIx64 " range 0x%016" PRIx64 "+0x%" PRIx64
	             " (fault at %+" PRId64 ") bound %.1f s ago",
	             what, TypeName(r.type), r.handle, r.address, r.size,
	             static_cast<int64_t>(address - r.address), now - r.bound);
	if (r.unbound >= 0) {
		std::fprintf(out, ", unbound %.3f s ago", now - r.unbound);
	}
	std::fprintf(out, "%s\n", r.internal ? " (driver-internal)" : "");
	if (r.type == 10 && ImageHistory::Enabled()) {
		ImageHistory::Describe(reinterpret_cast<VkImage>(r.handle), out);
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ADDRESS_BINDING_REPORT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void Note(bool bind, int32_t object_type, uint64_t handle, uint64_t address, uint64_t size, bool internal) {
	auto&           map = Get();
	const double    now = Now();
	std::lock_guard lock(map.mutex);
	auto&           ranges = map.live[handle];
	if (bind) {
		map.binds++;
		ranges.push_back({object_type, handle, address, size, now, -1, internal});
		return;
	}
	map.unbinds++;
	const auto it = std::find_if(ranges.begin(), ranges.end(), [&](const Range& r) {
		return r.address == address && r.size == size;
	});
	Range gone;
	if (it != ranges.end()) {
		gone = *it;
		ranges.erase(it);
	} else {
		map.unmatched++;
		gone = {object_type, handle, address, size, -1, -1, internal};
	}
	gone.unbound = now;
	if (ranges.empty()) {
		map.live.erase(handle);
	}
	map.gone.push_back(gone);
	if (map.gone.size() > GoneCapacity) {
		map.gone.pop_front();
	}
}

std::vector<uint64_t> RecentlyUnboundImages(uint64_t address, double within_s) {
	std::vector<uint64_t> handles;
	if (!Enabled()) {
		return handles;
	}
	auto&           map = Get();
	const double    now = Now();
	std::lock_guard lock(map.mutex);
	for (auto it = map.gone.rbegin(); it != map.gone.rend() && handles.size() < 32; ++it) {
		if (now - it->unbound > within_s) {
			break;
		}
		if (it->type == 10 && it->address <= address && address < it->address + it->size &&
		    std::find(handles.begin(), handles.end(), it->handle) == handles.end()) {
			handles.push_back(it->handle);
		}
	}
	return handles;
}

void Describe(uint64_t address, uint64_t precision, FILE* out) {
	if (!Enabled()) {
		return;
	}
	auto&           map = Get();
	const double    now = Now();
	std::lock_guard lock(map.mutex);
	const uint64_t  lo  = address & ~(precision > 1 ? precision - 1 : 0);
	const uint64_t  hi  = lo + std::max<uint64_t>(precision, 1);
	std::fprintf(out,
	             "[addr-map] fault 0x%016" PRIx64 ": driver map has %zu live objects, %zu unbound kept "
	             "(%" PRIu64 " binds, %" PRIu64 " unbinds, %" PRIu64 " unbinds without a bind)\n",
	             address, map.live.size(), map.gone.size(), map.binds, map.unbinds, map.unmatched);
	int           containing = 0;
	const Range*  below      = nullptr;
	const Range*  above      = nullptr;
	for (const auto& [handle, ranges]: map.live) {
		for (const auto& r: ranges) {
			if (r.address < hi && lo < r.address + r.size) {
				Print(out, "LIVE", r, address, now);
				containing++;
			} else if (r.address + r.size <= lo && lo - (r.address + r.size) < (32ull << 20u)) {
				if (below == nullptr || r.address + r.size > below->address + below->size) below = &r;
			} else if (r.address >= hi && r.address - hi < (32ull << 20u)) {
				if (above == nullptr || r.address < above->address) above = &r;
			}
		}
	}
	int gone_printed = 0;
	for (auto it = map.gone.rbegin(); it != map.gone.rend() && gone_printed < 24; ++it) {
		if (it->address < hi && lo < it->address + it->size) {
			Print(out, "UNBOUND", *it, address, now);
			gone_printed++;
		}
	}
	if (containing == 0) {
		std::fprintf(out, "[addr-map] no live object contains the fault address\n");
		if (below != nullptr) Print(out, "nearest live below", *below, address, now);
		if (above != nullptr) Print(out, "nearest live above", *above, address, now);
	}
	if (containing == 0 && gone_printed == 0) {
		std::fprintf(out, "[addr-map] the address was never bound in this run (or its unbind left the ring)\n");
	}
	std::fflush(out);
}

} // namespace Libs::Graphics::AddressBindings
