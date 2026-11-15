#ifndef EMULATOR_SRC_COMMON_FRAMESKIPS_H_
#define EMULATOR_SRC_COMMON_FRAMESKIPS_H_

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// KYTY_FRAME_SKIPS=1 (diagnostic, flicker hunt 10-26): one [skips] line per guest flip on the GPU
// thread, counting the draws and dispatches the renderer left out since the previous flip, by
// reason, and the draws of the pixel shaders in KYTY_FRAME_SKIPS_PS (comma-separated hashes; the
// sewer's decals fcfb268bca438b2d, 1574cecd82c28df4 by default). A decal layer that vanishes for
// one frame (the sewer wall: dirt and lamp-cage shadow gone ~1 frame in 5) shows here if its draw
// was skipped that frame. Lines are only printed when something was skipped or a watched shader
// drew a different number of times than in the previous frame.
namespace Common::FrameSkips {

enum Reason : uint32_t { GaveUp, SkipList, Resources, PipelinePending, NoPipeline, PrepFailed, Count };

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FRAME_SKIPS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline std::atomic<uint32_t> g_skips[Count] {};
inline std::atomic<uint32_t> g_draws {0};
inline std::atomic<uint32_t> g_watched {0};
inline std::atomic<uint32_t> g_watched_nopipe {0};
// Guest flips seen by the GPU thread (the [decal-in] lines are numbered by it).
inline std::atomic<uint64_t> g_flip {0};
// Every guest flip, whether or not KYTY_FRAME_SKIPS is on (KYTY_REPEAT_STATS frames).
inline std::atomic<uint64_t> g_guest_flips {0};

inline void Skip(Reason reason) {
	if (Enabled()) {
		g_skips[reason].fetch_add(1, std::memory_order_relaxed);
	}
}

inline bool Watched(uint64_t ps_hash) {
	static const auto list = [] {
		struct List {
			uint64_t hashes[16] {};
			int      count = 0;
		} result;
		const char* value = std::getenv("KYTY_FRAME_SKIPS_PS");
		if (value == nullptr) {
			value = "fcfb268bca438b2d,1574cecd82c28df4,845545cc4ff30ff8";
		}
		while (*value != '\0' && result.count < 16) {
			char* end = nullptr;
			const auto hash = std::strtoull(value, &end, 16);
			if (end == value) {
				break;
			}
			result.hashes[result.count++] = hash;
			value = *end == ',' ? end + 1 : end;
		}
		return result;
	}();
	for (int i = 0; i < list.count; i++) {
		if (list.hashes[i] == ps_hash) {
			return true;
		}
	}
	return false;
}

// GPU thread, at each guest flip.
inline void Flip() {
	g_guest_flips.fetch_add(1, std::memory_order_relaxed);
	if (!Enabled()) {
		return;
	}
	static uint32_t last_watched = UINT32_MAX;
	const uint64_t  flip         = g_flip.fetch_add(1, std::memory_order_relaxed) + 1;
	uint32_t skips[Count];
	uint32_t any = 0;
	for (uint32_t i = 0; i < Count; i++) {
		skips[i] = g_skips[i].exchange(0, std::memory_order_relaxed);
		any += skips[i];
	}
	const auto draws   = g_draws.exchange(0, std::memory_order_relaxed);
	const auto watched = g_watched.exchange(0, std::memory_order_relaxed);
	const auto nopipe  = g_watched_nopipe.exchange(0, std::memory_order_relaxed);
	if (any == 0 && watched == last_watched && nopipe == 0) {
		return;
	}
	last_watched = watched;
	std::fprintf(stderr,
	             "[skips] flip=%llu draws=%u watched=%u watched_nopipe=%u | gaveup=%u skiplist=%u "
	             "resources=%u pending=%u nopipe=%u prepfail=%u\n",
	             static_cast<unsigned long long>(flip), draws, watched, nopipe, skips[GaveUp],
	             skips[SkipList], skips[Resources], skips[PipelinePending], skips[NoPipeline],
	             skips[PrepFailed]);
}

} // namespace Common::FrameSkips

#endif // EMULATOR_SRC_COMMON_FRAMESKIPS_H_
