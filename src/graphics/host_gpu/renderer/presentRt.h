#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PRESENTRT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PRESENTRT_H_

#include "graphics/host_gpu/renderer/image/image.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// KYTY_PRESENT_RT=<w>x<h>[:<k>] (diagnostics; changes what is shown): every frame the color
// targets of that extent are tracked in the order they were last drawn to; at the flip, the k-th
// from the last one (default 0, the last) is shown in place of the guest's display buffer, scaled
// to it. Shows an engine's intermediate image (a scene before its upscaler, a G-buffer) on screen
// and to KYTY_FRAME_PROBE. Every 100 flips [present-rt] lists that frame's targets.
namespace Libs::Graphics::PresentRt {

struct Config {
	bool     enabled = false;
	uint32_t width   = 0;
	uint32_t height  = 0;
	uint32_t pick    = 0;
	bool     cycle   = false; // ":c[<n>]": show each target in turn, n flips each (default 400)
	uint32_t period  = 400;
};

inline const Config& Get() {
	static const Config config = [] {
		Config      result;
		const char* value = std::getenv("KYTY_PRESENT_RT");
		if (value == nullptr) {
			return result;
		}
		unsigned w = 0, h = 0, k = 0;
		const int n = std::sscanf(value, "%ux%u:%u", &w, &h, &k);
		if (n >= 2 && w != 0 && h != 0) {
			result = {true, w, h, n >= 3 ? k : 0u, false};
			const char* colon = std::strchr(value, ':');
			result.cycle      = colon != nullptr && colon[1] == 'c';
			if (result.cycle && colon[2] >= '0' && colon[2] <= '9') {
				result.period = std::max(1u, static_cast<uint32_t>(std::strtoul(colon + 2, nullptr, 10)));
			}
		}
		return result;
	}();
	return config;
}

struct Entry {
	uint64_t address = 0;
	ImageId  id {};
	uint32_t format  = 0;
	uint32_t slot    = 0;
	uint32_t uses    = 0;
};

struct State {
	std::mutex         mutex;
	std::vector<Entry> frame;
	ImageId            chosen {};
	uint64_t           chosen_address = 0;
	uint64_t           flips          = 0;
	uint64_t           cycle_start    = 0; // flip the cycle started at (first frame with targets)
	size_t             shown          = SIZE_MAX;
};

inline State& Get_state() {
	static auto* state = new State;
	return *state;
}

inline void NoteTarget(ImageId id, uint64_t address, uint32_t width, uint32_t height,
                       uint32_t format, uint32_t slot) {
	const auto& config = Get();
	if (!config.enabled || width != config.width || height != config.height) {
		return;
	}
	auto&           s = Get_state();
	std::lock_guard lock(s.mutex);
	uint32_t        uses = 0;
	for (auto it = s.frame.begin(); it != s.frame.end(); ++it) {
		if (it->address == address) {
			uses = it->uses;
			s.frame.erase(it);
			break;
		}
	}
	s.frame.push_back({address, id, format, slot, uses + 1});
}

inline void Flip() {
	const auto& config = Get();
	if (!config.enabled) {
		return;
	}
	auto&           s = Get_state();
	std::lock_guard lock(s.mutex);
	s.flips++;
	if (!s.frame.empty()) {
		size_t pick = config.pick;
		if (config.cycle) {
			if (s.cycle_start == 0) {
				s.cycle_start = s.flips;
			}
			pick = static_cast<size_t>((s.flips - s.cycle_start) / config.period) % s.frame.size();
		}
		const size_t index = pick < s.frame.size() ? s.frame.size() - 1 - pick : 0;
		if (pick != s.shown) {
			s.shown = pick;
			std::fprintf(stderr, "[present-rt] flip %llu now showing [%zu] 0x%llx fmt%u slot%u\n",
			             static_cast<unsigned long long>(s.flips), pick,
			             static_cast<unsigned long long>(s.frame[index].address),
			             s.frame[index].format, s.frame[index].slot);
		}
		s.chosen           = s.frame[index].id;
		s.chosen_address   = s.frame[index].address;
		if (s.flips % 100 == 1) {
			std::string list;
			for (size_t i = s.frame.size(); i-- > 0;) {
				char text[96];
				std::snprintf(text, sizeof(text), " [%zu]0x%llx/fmt%u/slot%u/x%u%s",
				              s.frame.size() - 1 - i,
				              static_cast<unsigned long long>(s.frame[i].address), s.frame[i].format,
				              s.frame[i].slot, s.frame[i].uses, i == index ? "*" : "");
				list += text;
			}
			std::fprintf(stderr, "[present-rt] flip %llu, last drawn first:%s\n",
			             static_cast<unsigned long long>(s.flips), list.c_str());
		}
	}
	s.frame.clear();
}

inline ImageId Chosen() {
	auto&           s = Get_state();
	std::lock_guard lock(s.mutex);
	return s.chosen;
}

} // namespace Libs::Graphics::PresentRt

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PRESENTRT_H_
