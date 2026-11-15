#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CHAINWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CHAINWATCH_H_

// KYTY_CHAIN_WATCH=<hash>/<hash>... (diagnostic, off by default; plan 4.8.1 Probe E).
// For the listed compute shaders, every dispatch is remembered (last 32 per shader: presented
// frame, the first 16 user-data dwords, every bound guest buffer range). When one of them trips
// the loop guard (the guard-trip report), the rings of all listed shaders are printed, and each
// buffer of the tripped shader (up to 4 MB) is read from guest memory and its linked chains are
// checked: next = (word >> 3) & 0x7ffffff, index 0 ends a chain (the walkers' encoding). Output
// to stderr with the prefix [chain].

#include "kernel/memory.h"

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ChainWatch {

struct Entry {
	uint64_t                                  frame = 0;
	std::array<uint32_t, 16>                  user {};
	uint32_t                                  user_count = 0;
	std::vector<std::pair<uint64_t, uint64_t>> buffers; // guest address, size
};

struct State {
	std::mutex                                     mutex;
	std::unordered_map<uint64_t, std::deque<Entry>> rings;
	std::unordered_map<uint64_t, uint64_t>          last_dump_frame;
	uint32_t                                       dumps = 0;
};

inline State& Get() {
	static State state;
	return state;
}

inline const std::string& List() {
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_CHAIN_WATCH");
		return value != nullptr ? std::string(value) : std::string();
	}();
	return list;
}

inline bool Listed(uint64_t hash) {
	const auto& list = List();
	if (list.empty()) {
		return false;
	}
	char text[17];
	std::snprintf(text, sizeof(text), "%016" PRIx64, hash);
	return list.find(text) != std::string::npos;
}

template <typename Sources>
void Note(uint64_t hash, uint64_t frame, const uint32_t* user, uint32_t user_count,
          const Sources& sources) {
	if (!Listed(hash)) {
		return;
	}
	Entry entry;
	entry.frame      = frame;
	entry.user_count = user_count < 16u ? user_count : 16u;
	for (uint32_t i = 0; i < entry.user_count; i++) {
		entry.user[i] = user[i];
	}
	for (const auto& source: sources) {
		entry.buffers.emplace_back(source.address, source.size);
	}
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	auto&           ring = state.rings[hash];
	ring.push_back(std::move(entry));
	if (ring.size() > 32u) {
		ring.pop_front();
	}
}

// Chain check over one buffer's words. Each start index i >= 1 ends (reaches 0), leaves the
// buffer (next >= words), or loops; the first loop found is printed (up to 16 of its nodes).
inline void Analyze(uint64_t address, const std::vector<uint32_t>& words) {
	const size_t n = words.size();
	// 0 unknown, 1 on the current path, 2 ends, 3 loops, 4 leaves the buffer
	std::vector<uint8_t>  state(n, 0);
	std::vector<uint32_t> path;
	size_t                ends = 0, loops = 0, leaves = 0, longest = 0;
	bool                  printed = false;
	for (size_t start = 1; start < n; start++) {
		if (state[start] != 0) {
			const auto s = state[start];
			ends += s == 2 ? 1 : 0;
			loops += s == 3 ? 1 : 0;
			leaves += s == 4 ? 1 : 0;
			continue;
		}
		path.clear();
		size_t  cur    = start;
		uint8_t result = 2;
		for (;;) {
			if (cur == 0) {
				result = 2;
				break;
			}
			if (cur >= n) {
				result = 4;
				break;
			}
			if (state[cur] == 1) {
				result = 3;
				if (!printed) {
					printed = true;
					std::string nodes;
					size_t      node = cur;
					for (int k = 0; k < 16; k++) {
						char item[48];
						std::snprintf(item, sizeof(item), " %zu:%08x", node, words[node]);
						nodes += item;
						node = (words[node] >> 3u) & 0x7ffffffu;
						if (node == cur || node >= n) break;
					}
					std::fprintf(stderr, "[chain]   0x%012" PRIx64 " loop from start %zu at index %zu:%s\n",
					             address, start, cur, nodes.c_str());
				}
				break;
			}
			if (state[cur] >= 2) {
				result = state[cur];
				break;
			}
			state[cur] = 1;
			path.push_back(static_cast<uint32_t>(cur));
			cur = (words[cur] >> 3u) & 0x7ffffffu;
		}
		if (path.size() > longest) longest = path.size();
		for (const auto p: path) state[p] = result;
		state[start] = result;
		ends += result == 2 ? 1 : 0;
		loops += result == 3 ? 1 : 0;
		leaves += result == 4 ? 1 : 0;
	}
	std::fprintf(stderr,
	             "[chain]   0x%012" PRIx64 " words=%zu starts: end=%zu loop=%zu leave=%zu longest_new_path=%zu\n",
	             address, n, ends, loops, leaves, longest);
}

// Called from the guard-trip report (BufferCache), once per frame per shader at most, at the
// 1st and every 64th trip, 16 dumps in all.
template <typename Cache>
void OnTrip(uint64_t hash, uint32_t trips, uint64_t frame, Cache& cache) {
	if (!Listed(hash) || (trips != 1u && (trips % 64u) != 0u)) {
		return;
	}
	auto&                  state = Get();
	std::vector<std::pair<uint64_t, uint64_t>> mine;
	{
		std::lock_guard lock(state.mutex);
		auto& last = state.last_dump_frame[hash];
		if ((last == frame && trips != 1u) || state.dumps >= 16u) {
			return;
		}
		last = frame;
		state.dumps++;
		std::fprintf(stderr, "[chain] dump %u: %016" PRIx64 " tripped (%u) at frame %" PRIu64 "\n",
		             state.dumps, hash, trips, frame);
		for (const auto& [h, ring]: state.rings) {
			for (const auto& e: ring) {
				std::string text;
				char        item[64];
				for (uint32_t i = 0; i < e.user_count; i++) {
					std::snprintf(item, sizeof(item), " %08x", e.user[i]);
					text += item;
				}
				text += " |";
				for (size_t i = 0; i < e.buffers.size(); i++) {
					std::snprintf(item, sizeof(item), " b%zu=0x%" PRIx64 "+0x%" PRIx64, i,
					              e.buffers[i].first, e.buffers[i].second);
					text += item;
				}
				std::fprintf(stderr, "[chain] %016" PRIx64 " frame=%" PRIu64 " parity=%u user:%s\n", h,
				             e.frame, e.user_count > 0 ? (e.user[0] & 1u) : 0u, text.c_str());
			}
		}
		const auto found = state.rings.find(hash);
		if (found != state.rings.end() && !found->second.empty()) {
			mine = found->second.back().buffers;
		}
	}
	for (size_t i = 0; i < mine.size(); i++) {
		const auto [address, size] = mine[i];
		if (address == 0 || size == 0 || size > (4ull << 20u)) {
			std::fprintf(stderr, "[chain]   b%zu 0x%012" PRIx64 "+0x%" PRIx64 ": skipped\n", i, address, size);
			continue;
		}
		std::vector<uint32_t> words(size / 4u);
		if (!Libs::LibKernel::Memory::TryReadBufferBacking(address, words.data(), words.size() * 4u)) {
			std::fprintf(stderr, "[chain]   b%zu 0x%012" PRIx64 "+0x%" PRIx64 ": not readable\n", i, address, size);
			continue;
		}
		std::fprintf(stderr, "[chain]   b%zu 0x%012" PRIx64 "+0x%" PRIx64 " gpu_dirty=%d registered=%d\n", i,
		             address, size, cache.HasGpuDirtyBytes(address, size) ? 1 : 0,
		             cache.IsRegionRegistered(address, size) ? 1 : 0);
		Analyze(address, words);
	}
	std::fflush(stderr);
}

} // namespace Libs::Graphics::ChainWatch

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CHAINWATCH_H_
