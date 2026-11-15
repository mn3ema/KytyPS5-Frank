#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HTILEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HTILEWATCH_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

// KYTY_HTILE_READERS=1 (diagnostics, changes nothing): the HTile ranges of the depth targets
// drawn so far, and every shader that binds a buffer overlapping one ([htile-reader], once per
// shader and range). The host depth attachment keeps its own compression, so guest HTile memory
// never holds what the console's depth block writes there; a shader reading it (tile depth
// bounds for tiled lighting or decals) reads whatever else is in that memory.
namespace Libs::Graphics::HtileWatch {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HTILE_READERS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_META_CLEAR_LOG=1 (diagnostics, changes nothing): every compute dispatch that writes a buffer
// overlapping a depth target's HTile ([meta-clear]: consumed as a host clear or executed on guest
// memory), and every depth bind of an HTile target ([depth-bind]: whether it load-clears).
inline bool MetaClearLog() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_META_CLEAR_LOG");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// At most 150 lines a second (about six frames' worth at 25 fps), so a long run stays covered.
inline bool LogBudget() {
	static std::atomic<uint64_t> window {0};
	static std::atomic<uint32_t> lines {0};
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count());
	if (window.exchange(now, std::memory_order_relaxed) != now) {
		lines.store(0, std::memory_order_relaxed);
	}
	return lines.fetch_add(1, std::memory_order_relaxed) < 150;
}

struct State {
	std::mutex                                       mutex;
	std::set<std::pair<uint64_t, uint64_t>>          ranges; // htile begin, size
	std::set<std::pair<uint64_t, uint64_t>>          logged; // shader hash, htile begin
};
inline State& Get() {
	static auto* state = new State;
	return *state;
}

inline void NoteDepthTarget(uint64_t htile_address, uint64_t htile_size, uint64_t depth_address,
                            uint32_t width, uint32_t height) {
	if ((!Enabled() && !MetaClearLog()) || htile_address == 0 || htile_size == 0) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (s.ranges.insert({htile_address, htile_size}).second) {
		std::fprintf(stderr, "[htile] depth 0x%llx %ux%u htile 0x%llx+0x%llx\n",
		             static_cast<unsigned long long>(depth_address), width, height,
		             static_cast<unsigned long long>(htile_address),
		             static_cast<unsigned long long>(htile_size));
	}
}

inline void NoteBuffer(uint64_t shader_hash, uint32_t stage, uint64_t address, uint64_t size,
                       bool written) {
	if (!Enabled() || size == 0) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	for (const auto& [begin, bytes]: s.ranges) {
		if (address < begin + bytes && begin < address + size) {
			if (s.logged.insert({shader_hash, begin}).second) {
				std::fprintf(stderr,
				             "[htile-reader] sh=%016llx stage=%u buf 0x%llx+0x%llx %s htile 0x%llx+0x%llx\n",
				             static_cast<unsigned long long>(shader_hash), stage,
				             static_cast<unsigned long long>(address),
				             static_cast<unsigned long long>(size), written ? "written" : "read",
				             static_cast<unsigned long long>(begin),
				             static_cast<unsigned long long>(bytes));
			}
		}
	}
}

// The begin of the HTile range [address, address + size) overlaps, or 0.
inline uint64_t OverlappingHtile(uint64_t address, uint64_t size, uint64_t* htile_size = nullptr) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	for (const auto& [begin, bytes]: s.ranges) {
		if (address < begin + bytes && begin < address + size) {
			if (htile_size != nullptr) *htile_size = bytes;
			return begin;
		}
	}
	return 0;
}

inline void NoteComputeWrite(uint64_t shader_hash, uint64_t address, uint64_t size, bool read,
                             bool consumed, bool xor_op) {
	if (!MetaClearLog() || size == 0) {
		return;
	}
	uint64_t htile_size = 0;
	const auto htile    = OverlappingHtile(address, size, &htile_size);
	if (htile == 0 || !LogBudget()) {
		return;
	}
	std::fprintf(stderr,
	             "[meta-clear] sh=%016llx buf 0x%llx+0x%llx htile 0x%llx+0x%llx exact=%d read=%d "
	             "xor=%d %s\n",
	             static_cast<unsigned long long>(shader_hash), static_cast<unsigned long long>(address),
	             static_cast<unsigned long long>(size), static_cast<unsigned long long>(htile),
	             static_cast<unsigned long long>(htile_size), address == htile ? 1 : 0, read ? 1 : 0,
	             xor_op ? 1 : 0, consumed ? "CONSUMED" : "EXECUTED");
}

inline void NoteDepthBind(uint64_t depth_address, uint64_t htile_address, uint32_t layer,
                          bool explicit_clear, bool meta_clear, float clear_value,
                          bool depth_write) {
	if (!MetaClearLog() || !LogBudget()) {
		return;
	}
	std::fprintf(stderr,
	             "[depth-bind] depth 0x%llx htile 0x%llx layer %u clear=%d meta=%d value=%g write=%d\n",
	             static_cast<unsigned long long>(depth_address),
	             static_cast<unsigned long long>(htile_address), layer, explicit_clear ? 1 : 0,
	             meta_clear ? 1 : 0, static_cast<double>(clear_value), depth_write ? 1 : 0);
}

} // namespace Libs::Graphics::HtileWatch

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HTILEWATCH_H_
