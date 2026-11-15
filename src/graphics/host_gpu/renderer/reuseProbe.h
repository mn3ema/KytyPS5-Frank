#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REUSEPROBE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REUSEPROBE_H_

#include "graphics/guest_gpu/hardwareContext.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <xxhash.h>

// KYTY_REUSE_PROBE=1 (measurement only, changes nothing): how many draws and dispatches of a
// frame repeat one of the last three frames' register states exactly. The draw-reuse plan
// (remembering each draw's preparation while nothing under it changes, Mac 10-26: "10% isn't
// enough") only pays if most of a standing frame's draws come back with the same registers and
// user data; a game that rebuilds its descriptor tables in a moving ring every frame would
// defeat it. The `[reuse]` line follows each `[floor]` line.
namespace Libs::Graphics::ReuseProbe {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_REUSE_PROBE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Guest flips (RenderDocOnGuestFlip).
inline std::atomic<uint64_t> g_frame {0};

struct Stats {
	std::atomic<uint64_t> draws {0};       // draws seen
	std::atomic<uint64_t> draw_hits {0};   // full state seen in one of the last 3 frames
	std::atomic<uint64_t> draw_sh_hits {0}; // shader registers + user data only
	std::atomic<uint64_t> draw_dups {0};   // full state already seen this frame
	std::atomic<uint64_t> disps {0};
	std::atomic<uint64_t> disp_hits {0};
	std::atomic<uint64_t> disp_dups {0};
	std::atomic<uint64_t> buffers_created {0}; // BufferCache::CreateBuffer (FindBuffer misses)
};
inline Stats g_stats;

// One key kind's last frames (GPU thread only).
class Window {
public:
	// Whether key was seen in an earlier frame of the window; dup: seen this frame already.
	bool Note(uint64_t key, bool* dup) {
		const auto frame = g_frame.load(std::memory_order_relaxed);
		if (frame != m_frame) {
			m_current = (m_current + 1) % m_sets.size();
			m_sets[m_current].clear();
			m_frame = frame;
		}
		const bool repeated = !m_sets[m_current].insert(key).second;
		if (dup != nullptr) {
			*dup = repeated;
		}
		for (size_t i = 0; i < m_sets.size(); i++) {
			if (i != m_current && m_sets[i].count(key) != 0) {
				return true;
			}
		}
		return false;
	}

private:
	std::array<std::unordered_set<uint64_t>, 4> m_sets;
	size_t                                      m_current = 0;
	uint64_t                                    m_frame   = ~uint64_t {0};
};

inline void NoteDraw(const HW::Shader& shaders, const HW::Context& context,
                     const HW::UserConfig& config, uint32_t slice_offset, uint32_t kind) {
	static thread_local Window full;
	static thread_local Window shader_only;
	const auto& vs     = shaders.GetVs();
	const auto& ps     = shaders.GetPs();
	auto        sh_key = XXH3_64bits(&vs, sizeof(vs));
	sh_key             = XXH3_64bits_withSeed(&ps, sizeof(ps), sh_key);
	auto key           = XXH3_64bits_withSeed(&context, sizeof(context), sh_key);
	key                = XXH3_64bits_withSeed(&config, sizeof(config), key);
	key ^= (static_cast<uint64_t>(slice_offset) << 8u | kind) * 0x9e3779b97f4a7c15ull;
	bool dup = false;
	g_stats.draws.fetch_add(1, std::memory_order_relaxed);
	if (full.Note(key, &dup)) {
		g_stats.draw_hits.fetch_add(1, std::memory_order_relaxed);
	}
	if (dup) {
		g_stats.draw_dups.fetch_add(1, std::memory_order_relaxed);
	}
	if (shader_only.Note(sh_key, nullptr)) {
		g_stats.draw_sh_hits.fetch_add(1, std::memory_order_relaxed);
	}
}

inline void NoteDispatch(const HW::Shader& shaders, uint32_t x, uint32_t y, uint32_t z,
                         uint64_t indirect_args) {
	static thread_local Window window;
	const auto&    cs      = shaders.GetCs();
	const uint64_t dims[3] = {(static_cast<uint64_t>(x) << 32u) | y, z, indirect_args};
	const auto     key     = XXH3_64bits_withSeed(dims, sizeof(dims), XXH3_64bits(&cs, sizeof(cs)));
	bool           dup     = false;
	g_stats.disps.fetch_add(1, std::memory_order_relaxed);
	if (window.Note(key, &dup)) {
		g_stats.disp_hits.fetch_add(1, std::memory_order_relaxed);
	}
	if (dup) {
		g_stats.disp_dups.fetch_add(1, std::memory_order_relaxed);
	}
}

// Once a second, after the [floor] line: per-frame counts and repeat rates.
inline void Report(double elapsed, double frames) {
	auto take = [](std::atomic<uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
	const double draws   = static_cast<double>(take(g_stats.draws));
	const double hits    = static_cast<double>(take(g_stats.draw_hits));
	const double sh_hits = static_cast<double>(take(g_stats.draw_sh_hits));
	const double dups    = static_cast<double>(take(g_stats.draw_dups));
	const double disps   = static_cast<double>(take(g_stats.disps));
	const double d_hits  = static_cast<double>(take(g_stats.disp_hits));
	const double d_dups  = static_cast<double>(take(g_stats.disp_dups));
	const double created = static_cast<double>(take(g_stats.buffers_created));
	const double f       = frames > 0 ? frames : 1.0;
	std::fprintf(stderr,
	             "[reuse] draws/frame=%.0f repeat=%.0f%% (shader+userdata %.0f%%, same frame %.0f%%) "
	             "dispatches/frame=%.0f repeat=%.0f%% (same frame %.0f%%) buffers created/frame=%.1f\n",
	             draws / f, draws > 0 ? 100.0 * hits / draws : 0.0,
	             draws > 0 ? 100.0 * sh_hits / draws : 0.0, draws > 0 ? 100.0 * dups / draws : 0.0,
	             disps / f, disps > 0 ? 100.0 * d_hits / disps : 0.0,
	             disps > 0 ? 100.0 * d_dups / disps : 0.0, created / f);
	(void)elapsed;
}

} // namespace Libs::Graphics::ReuseProbe

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REUSEPROBE_H_
