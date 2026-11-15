#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTDRAWLOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTDRAWLOG_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

// KYTY_RT_DRAWLOG=<hex guest address> (diagnostics, changes nothing): the draws into the color
// target at that address, grouped per frame by pixel shader and depth/stencil state. Every flip
// [rt-draws] prints the groups whose count changed from the previous frame (a draw group that
// comes and goes is a draw dropped on some frames); every 100 flips the whole table with states.
// "skip" counts draws dropped while their pipeline compiles.
namespace Libs::Graphics::RtDrawLog {

inline uint64_t Address() {
	static const uint64_t address = [] {
		const char* value = std::getenv("KYTY_RT_DRAWLOG");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : uint64_t {0};
	}();
	return address;
}

struct Key {
	uint64_t ps_hash;
	uint64_t state;
	bool     operator<(const Key& o) const { return std::tie(ps_hash, state) < std::tie(o.ps_hash, o.state); }
};
struct Group {
	uint32_t    draws   = 0;
	uint32_t    skipped = 0;
	std::string text;
};

struct State {
	std::mutex           mutex;
	std::map<Key, Group> frame;
	std::map<Key, Group> previous;
	uint64_t             flips = 0;
};
inline State& Get() {
	static auto* state = new State;
	return *state;
}

inline void NoteDraw(uint64_t ps_hash, uint64_t state_hash, bool skipped, const char* text) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	auto&           group = s.frame[{ps_hash, state_hash}];
	(skipped ? group.skipped : group.draws)++;
	if (group.text.empty()) {
		group.text = text;
	}
}

inline void Flip() {
	if (Address() == 0) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	s.flips++;
	std::string diff;
	auto        note = [&](const Key& key, uint32_t before, uint32_t now, uint32_t skipped) {
        char text[96];
        std::snprintf(text, sizeof(text), " %016llx/%08llx:%u->%u%s",
                      static_cast<unsigned long long>(key.ps_hash),
                      static_cast<unsigned long long>(key.state & 0xffffffffu), before, now,
                      skipped != 0 ? "(skip)" : "");
        diff += text;
	};
	for (const auto& [key, group]: s.frame) {
		const auto old = s.previous.find(key);
		const auto was = old != s.previous.end() ? old->second.draws : 0u;
		if (was != group.draws || group.skipped != 0) note(key, was, group.draws, group.skipped);
	}
	for (const auto& [key, group]: s.previous) {
		if (!s.frame.contains(key)) note(key, group.draws, 0, 0);
	}
	if (!diff.empty() && !s.frame.empty() && !s.previous.empty()) {
		std::fprintf(stderr, "[rt-draws] flip %llu:%s\n", static_cast<unsigned long long>(s.flips),
		             diff.c_str());
	}
	if (s.flips % 100 == 1 && !s.frame.empty()) {
		for (const auto& [key, group]: s.frame) {
			std::fprintf(stderr, "[rt-draws-table] flip %llu ps %016llx/%08llx x%u %s\n",
			             static_cast<unsigned long long>(s.flips),
			             static_cast<unsigned long long>(key.ps_hash),
			             static_cast<unsigned long long>(key.state & 0xffffffffu), group.draws,
			             group.text.c_str());
		}
	}
	s.previous = std::move(s.frame);
	s.frame.clear();
}

} // namespace Libs::Graphics::RtDrawLog

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTDRAWLOG_H_
