#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERREASON_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERREASON_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <source_location>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// v445q, KYTY_BARRIER_REASON_STATS=1 (statistics only, no behaviour change): for every two
// consecutive dispatches of the SAME compute shader that a thread records into one command
// buffer, what was recorded between them and why. KYTY_GPU_TIMING (g450e) found 3.3 ms a frame of
// GPU time in the gaps of such pairs; this tells whose decision each gap is:
//   none    nothing was recorded between the two (a dispatch run left the barriers out);
//   excl    the guest stream runs them back to back, and the exclusion list
//           (KYTY_DISPATCH_RUN_EXCLUDE, renderCompute.cpp) kept their barriers;
//   guest   the guest stream has a packet between them that ends a dispatch run (which one:
//           "op=0x<opcode>:0x<sub>", as in [run-breaks]): the game orders them;
//   end ... the run ended for a reason that is the emulator's own: the command buffer ended (end),
//           a new slice of the submission began (slice), the scan expected another dispatch
//           (mismatch), the packet was unknown (nopacket), or was not a plain type-3 packet
//           (type, pred, len);
//   noruns  dispatch runs are off (KYTY_DISPATCH_RUNS, KYTY_DISPATCH_SYNC);
//   other   the run continued and the shader is not excluded, yet something was recorded between
//           them: the "extra" call sites say what (a buffer upload, an image transition, the
//           thread-count conversion of an indirect dispatch, ...).
// The renderer has no per-range conflict test between dispatches: outside a run every dispatch
// that writes storage gets the write-hazard barrier before it ("pre") and every dispatch the
// access barrier after it ("post"), so those two follow from the run's end, whatever ended it.
// One "[barrier-reason]" block every 5 s on stderr: a total line, then the
// KYTY_BARRIER_REASON_TOP (default 16) shaders with the most pairs that had something between
// them. Counts are per presented frame.
namespace Libs::Graphics::BarrierReason {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BARRIER_REASON_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Why a dispatch does not continue the dispatch run of the one before it (DispatchRunState::why):
// 0: it continues it; kind << 24 | opcode << 16 | sub as [run-breaks] has it (kind 0: a type-3
// packet that is not a run member; 1: the command buffer ended; 2: not a type-3 packet; 3: a
// predicated packet; 4: a bad length), and the kinds below.
enum Why : uint32_t {
	WhySlice    = 5u << 24u, // the first dispatch of a slice of a submission
	WhyMismatch = 6u << 24u, // the scan found another dispatch packet as the run's next
	WhyNoPacket = 7u << 24u, // the dispatch's packet is not known
	WhyRunsOff  = 8u << 24u, // dispatch runs are off
};

// Whose barrier a call site records (Tag): the dispatch's own ones are named, the rest are listed
// by file and line.
enum Own : uint32_t {
	OwnNone = 0,
	OwnPre,     // the write-hazard barrier before a dispatch that writes storage
	OwnExcl,    // the access barrier before a dispatch of the exclusion list inside a run
	OwnPost,    // the access barrier after a dispatch
	OwnRunPost, // the access barrier after a run whose last dispatch left its own out
	OwnRt,      // around a callable split's ray-tracing launch
	OwnConvert, // around an indirect dispatch's thread-count conversion
	OwnCount,
};

enum SiteKind : uint32_t { SiteBarrier = 0, SiteGuestBarrier = 1, SiteWork = 2 };

struct Site {
	const char* file = nullptr;
	uint32_t    line = 0;
	uint32_t    kind = 0;
};

struct Between {
	uint32_t                 mask     = 0; // GpuTiming::GapNote bits
	uint32_t                 barriers = 0; // emulator barriers, the dispatch's own included
	uint32_t                 guest    = 0; // the guest's own barriers (EmitGlobalBarrier)
	uint32_t                 work     = 0; // internal copies, fills and dispatches
	std::array<uint32_t, OwnCount> own {}; // barriers by Own
	uint32_t                 site_count = 0; // sites that are not the dispatch's own
	std::array<Site, 6>      sites {};
};

struct ThreadState {
	Between  between;  // recorded since the last timed draw or dispatch
	Between  snapshot; // what was between that one and the command being timed now
	uint32_t tag       = OwnNone;
	bool     have_prev = false;
	bool     prev_dispatch = false;
	uint64_t prev_key  = 0;
	uint64_t prev_tick = 0;
	// The dispatch being recorded (renderCompute.cpp, before GpuTiming::After).
	uint64_t tick     = 0;
	uint32_t why      = 0;
	bool     excluded = false;
};

inline ThreadState& Thread() {
	thread_local ThreadState state;
	return state;
}

// Names the barriers recorded in its scope as the dispatch's own.
class Tag {
public:
	explicit Tag(uint32_t own): m_on(Enabled()) {
		if (m_on) {
			auto& t = Thread();
			m_saved = t.tag;
			t.tag   = own;
		}
	}
	~Tag() {
		if (m_on) Thread().tag = m_saved;
	}
	Tag(const Tag&)            = delete;
	Tag& operator=(const Tag&) = delete;

private:
	bool     m_on;
	uint32_t m_saved = OwnNone;
};

// GpuTiming::Note (every command the recorder puts between timed commands).
inline void NoteBit(uint32_t bit) {
	if (Enabled()) Thread().between.mask |= bit;
}

// CommandRecorder: a barrier, or an internal copy, fill or dispatch, and its call site.
inline void NoteSite(const std::source_location& site, uint32_t kind) {
	if (!Enabled()) return;
	auto& t = Thread();
	auto& b = t.between;
	if (kind == SiteBarrier) {
		b.barriers++;
		b.own[t.tag < OwnCount ? t.tag : OwnNone]++;
		if (t.tag != OwnNone) return;
	} else if (kind == SiteGuestBarrier) {
		b.guest++;
	} else {
		b.work++;
	}
	if (b.site_count < b.sites.size()) {
		b.sites[b.site_count] = {site.file_name(), site.line(), kind};
	}
	b.site_count++;
}

struct PairStat {
	uint64_t pairs = 0; // in one command buffer
	uint64_t cross = 0; // a submission between them: not a gap inside a command buffer
	uint64_t none = 0, excl = 0, guest = 0, emu = 0, noruns = 0, other = 0;
	uint64_t barriers = 0, guest_barriers = 0, work = 0, extra = 0;
	std::array<uint64_t, OwnCount> own {};
	std::map<uint32_t, uint64_t>                        stops; // Why keys of guest and emu
	std::map<std::pair<std::string, uint32_t>, uint64_t> sites; // extra call sites ("file:line", kind)
};

struct Shared {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, PairStat> stats;
	std::chrono::steady_clock::time_point  last = std::chrono::steady_clock::now();
	std::atomic<uint64_t>                  frames {0};
};

inline Shared& GetShared() {
	static Shared shared;
	return shared;
}

// GpuTiming::Present: once per presented frame.
inline void NoteFrame() {
	if (Enabled()) GetShared().frames.fetch_add(1, std::memory_order_relaxed);
}

inline std::string WhyName(uint32_t why) {
	static constexpr const char* Kinds[] = {"op", "end", "type", "pred", "len", "slice", "mismatch", "nopacket", "noruns"};
	const uint32_t kind = std::min<uint32_t>(why >> 24u, 8u);
	if (kind != 0) return Kinds[kind];
	const uint32_t op  = (why >> 16u) & 0xffu;
	const uint32_t sub = why & 0xffffu;
	const char*    name = nullptr;
	switch (op) {
		case 0x10: name = "NOP"; break;
		case 0x37: name = "WRITE_DATA"; break;
		case 0x3c: name = "WAIT_REG_MEM"; break;
		case 0x3f: name = "INDIRECT_BUFFER"; break;
		case 0x40: name = "COPY_DATA"; break;
		case 0x46: name = "EVENT_WRITE"; break;
		case 0x47: name = "EVENT_WRITE_EOP"; break;
		case 0x48: name = "EVENT_WRITE_EOS"; break;
		case 0x49: name = "RELEASE_MEM"; break;
		case 0x50: name = "DMA_DATA"; break;
		case 0x58: name = "ACQUIRE_MEM"; break;
		case 0x68: name = "SET_CONFIG_REG"; break;
		case 0x69: name = "SET_CONTEXT_REG"; break;
		case 0x78: name = "SET_QUEUE_REG"; break;
		case 0x79: name = "SET_UCONFIG_REG"; break;
		case 0x93: name = "WAIT_REG_MEM_64"; break;
		default: break;
	}
	char text[64];
	if (name != nullptr) {
		std::snprintf(text, sizeof(text), "op=0x%02x:0x%x(%s)", op, sub, name);
	} else {
		std::snprintf(text, sizeof(text), "op=0x%02x:0x%x", op, sub);
	}
	return text;
}

inline void Report(Shared& shared, double seconds) {
	// Under shared.mutex.
	const uint64_t frames = shared.frames.exchange(0, std::memory_order_relaxed);
	const double   f      = frames != 0 ? static_cast<double>(frames) : 1.0;
	static const size_t top = [] {
		const char* value = std::getenv("KYTY_BARRIER_REASON_TOP");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		return n != 0 ? static_cast<size_t>(n) : size_t {16};
	}();
	std::vector<std::pair<uint64_t, const PairStat*>> list;
	PairStat                                          all;
	for (const auto& [key, stat]: shared.stats) {
		list.emplace_back(key, &stat);
		all.pairs += stat.pairs;
		all.cross += stat.cross;
		all.none += stat.none;
		all.excl += stat.excl;
		all.guest += stat.guest;
		all.emu += stat.emu;
		all.noruns += stat.noruns;
		all.other += stat.other;
	}
	std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) {
		const auto x = a.second->pairs - a.second->none;
		const auto y = b.second->pairs - b.second->none;
		return x != y ? x > y : a.first < b.first;
	});
	std::fprintf(stderr,
	             "[barrier-reason] %.1f s, %llu frames, %zu shaders: same-shader dispatch pairs in a command buffer "
	             "%.1f/f (across buffers %.1f/f) | none %.1f excl %.1f guest %.1f emu %.1f noruns %.1f other %.1f\n",
	             seconds, static_cast<unsigned long long>(frames), list.size(), all.pairs / f, all.cross / f,
	             all.none / f, all.excl / f, all.guest / f, all.emu / f, all.noruns / f, all.other / f);
	char text[320];
	for (size_t i = 0; i < list.size() && i < top; i++) {
		const auto& s = *list[i].second;
		if (s.pairs == s.none) break;
		const double p = s.pairs != 0 ? static_cast<double>(s.pairs) : 1.0;
		std::string  out;
		std::snprintf(text, sizeof(text),
		              "[barrier-reason]   %016llx pairs %.1f/f (across %.1f) | none %.1f excl %.1f guest %.1f emu %.1f "
		              "noruns %.1f other %.1f",
		              static_cast<unsigned long long>(list[i].first), s.pairs / f, s.cross / f, s.none / f, s.excl / f,
		              s.guest / f, s.emu / f, s.noruns / f, s.other / f);
		out += text;
		std::snprintf(text, sizeof(text),
		              " | per pair: barriers %.2f (pre %.2f excl %.2f post %.2f runpost %.2f rt %.2f convert %.2f "
		              "extra %.2f)",
		              s.barriers / p, s.own[OwnPre] / p, s.own[OwnExcl] / p, s.own[OwnPost] / p, s.own[OwnRunPost] / p,
		              s.own[OwnRt] / p, s.own[OwnConvert] / p, s.own[OwnNone] / p);
		out += text;
		std::snprintf(text, sizeof(text), " guest-barriers %.2f work %.2f", s.guest_barriers / p, s.work / p);
		out += text;
		if (!s.stops.empty()) {
			std::vector<std::pair<uint32_t, uint64_t>> stops(s.stops.begin(), s.stops.end());
			std::sort(stops.begin(), stops.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			out += " | run ended by:";
			for (size_t k = 0; k < stops.size() && k < 5; k++) {
				std::snprintf(text, sizeof(text), " %s %.1f", WhyName(stops[k].first).c_str(), stops[k].second / f);
				out += text;
			}
		}
		if (!s.sites.empty()) {
			std::vector<std::pair<std::pair<std::string, uint32_t>, uint64_t>> sites(s.sites.begin(), s.sites.end());
			std::sort(sites.begin(), sites.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			out += " | extra sites:";
			static constexpr const char* Kinds[] = {"barrier", "guest-barrier", "work"};
			for (size_t k = 0; k < sites.size() && k < 5; k++) {
				std::snprintf(text, sizeof(text), " %s %s %.1f", sites[k].first.first.c_str(),
				              Kinds[std::min<uint32_t>(sites[k].first.second, 2u)], sites[k].second / f);
				out += text;
			}
		}
		std::fprintf(stderr, "%s\n", out.c_str());
	}
	shared.stats.clear();
}

// GpuTiming::Before: a timed guest draw or dispatch is about to be recorded.
inline void Before() {
	if (!Enabled()) return;
	auto& t    = Thread();
	t.snapshot = t.between;
	t.between  = {};
}

// renderCompute.cpp, before GpuTiming::After of a dispatch: its command buffer (the scheduler's
// tick), why it does not continue the run of the dispatch before it, and whether its shader is
// on the exclusion list.
inline void SetDispatch(uint64_t tick, uint32_t why, bool excluded) {
	if (!Enabled()) return;
	auto& t    = Thread();
	t.tick     = tick;
	t.why      = why;
	t.excluded = excluded;
}

// GpuTiming::After: the timed command was recorded (dispatch: kind 2, GpuTiming::Dispatch).
inline void After(uint64_t key, uint32_t kind) {
	if (!Enabled()) return;
	auto&      t        = Thread();
	const bool dispatch = kind == 2u;
	if (dispatch && t.have_prev && t.prev_dispatch && t.prev_key == key) {
		const auto&     b      = t.snapshot;
		auto&           shared = GetShared();
		std::lock_guard lock(shared.mutex);
		auto&           s = shared.stats[key];
		if (t.prev_tick != t.tick) {
			s.cross++;
		} else {
			s.pairs++;
			const bool clear = b.barriers == 0 && b.guest == 0 && b.work == 0;
			const uint32_t kind_of = t.why >> 24u;
			if (t.why == WhyRunsOff) {
				s.noruns++;
			} else if (t.why != 0) {
				(kind_of == 0 ? s.guest : s.emu)++;
				s.stops[t.why]++;
			} else if (t.excluded) {
				s.excl++;
			} else if (clear) {
				s.none++;
			} else {
				s.other++;
			}
			s.barriers += b.barriers;
			s.guest_barriers += b.guest;
			s.work += b.work;
			s.extra += b.site_count;
			for (uint32_t i = 0; i < OwnCount; i++) s.own[i] += b.own[i];
			for (uint32_t i = 0; i < b.site_count && i < b.sites.size(); i++) {
				const char* file = b.sites[i].file != nullptr ? b.sites[i].file : "?";
				if (const char* slash = std::strrchr(file, '/'); slash != nullptr) file = slash + 1;
				if (const char* slash = std::strrchr(file, '\\'); slash != nullptr) file = slash + 1;
				s.sites[{std::string(file) + ":" + std::to_string(b.sites[i].line), b.sites[i].kind}]++;
			}
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - shared.last >= std::chrono::seconds(5)) {
			Report(shared, std::chrono::duration<double>(now - shared.last).count());
			shared.last = now;
		}
	}
	t.have_prev     = true;
	t.prev_dispatch = dispatch;
	t.prev_key      = key;
	t.prev_tick     = dispatch ? t.tick : 0;
	t.between       = {}; // the command's own recording is not a gap (as GpuTiming has it)
	t.why           = 0;
	t.excluded      = false;
}

} // namespace Libs::Graphics::BarrierReason

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERREASON_H_
