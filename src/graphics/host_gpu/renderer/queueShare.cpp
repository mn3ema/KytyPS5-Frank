#include "graphics/host_gpu/renderer/queueShare.h"

#include "graphics/host_gpu/timeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

uint32_t QueueStatsQueue(); // graphicsRun.cpp: the guest queue the calling thread is processing

namespace QueueShare::Detail {

namespace {

// Per range and class: the sequence number of the class's last read and last write this frame
// (0: none).
struct Access {
	uint32_t read[2]  = {0, 0};
	uint32_t write[2] = {0, 0};
};
struct Interval {
	uint64_t end = 0;
	Access   access;
};

enum Kind : uint32_t { ReadAfterWrite, WriteAfterRead, WriteAfterWrite, KindCount };
constexpr const char* KindNames[KindCount] = {"read-after-write", "write-after-read", "write-after-write"};

struct FrameCounts {
	uint64_t accesses[2] {};         // by class: 0 graphics, 1 compute
	uint64_t image_accesses[2] {};
	uint64_t repeats = 0;            // the same range again in one run of a class: nothing new
	uint64_t conflicts[KindCount][2] {}; // by kind and resource: 0 buffer, 1 image
	uint64_t conflict_accesses = 0;  // accesses with at least one conflict
	uint64_t joins[2] {};            // by the class that has to wait: forced joins
	uint64_t joins_extra[2] {};      // of them, not already ordered by a guest wait
	uint64_t guest_waits[2] {};      // waits that passed on a label the other class wrote
	uint64_t guest_waits_other[2] {}; // other waits that passed
	uint64_t guest_labels[2] {};
	uint64_t ranges = 0;             // distinct pieces tracked at the frame's end
};

struct TopRange {
	uint64_t begin = 0, size = 0;
	uint64_t count[KindCount] {};
	bool     image = false;
};

struct RepeatSlot {
	uint64_t begin = 0, end = 0, run = 0;
	bool     write = false;
};

struct State {
	std::map<uint64_t, Interval> intervals;
	uint32_t                     seq[2]       = {0, 0};
	uint32_t                     cut[2]       = {0, 0}; // by class: everything up to here is submitted (forced joins)
	uint32_t                     cut_guest[2] = {0, 0}; // the same with the guest's own joins applied
	uint64_t                     run          = 1;      // moves when the class of the accesses changes, or a cut does
	int                          last_class   = -1;
	std::array<RepeatSlot, 4096> repeats {};
	bool                         sampling = false;
	uint64_t                     frame    = 0;
	FrameCounts                  now;
	// The report window.
	FrameCounts                              sum;
	uint64_t                                 frames = 0, all_frames = 0;
	uint64_t                                 max_joins = 0, max_joins_extra = 0, max_conflicts = 0;
	std::unordered_map<uint64_t, TopRange>   top;
	std::chrono::steady_clock::time_point    last = std::chrono::steady_clock::now();
};

State& Get() {
	static State state;
	return state;
}

uint64_t Every() {
	static const uint64_t every = [] {
		const char* value = std::getenv("KYTY_QUEUE_SHARE_EVERY");
		const auto  n     = value != nullptr ? std::strtoull(value, nullptr, 10) : 4ull;
		return n == 0 ? 1ull : n;
	}();
	return every;
}

void Add(FrameCounts& to, const FrameCounts& from) {
	for (int c = 0; c < 2; c++) {
		to.accesses[c] += from.accesses[c];
		to.image_accesses[c] += from.image_accesses[c];
		to.joins[c] += from.joins[c];
		to.joins_extra[c] += from.joins_extra[c];
		to.guest_waits[c] += from.guest_waits[c];
		to.guest_waits_other[c] += from.guest_waits_other[c];
		to.guest_labels[c] += from.guest_labels[c];
		for (uint32_t k = 0; k < KindCount; k++) {
			to.conflicts[k][c] += from.conflicts[k][c];
		}
	}
	to.repeats += from.repeats;
	to.conflict_accesses += from.conflict_accesses;
	to.ranges += from.ranges;
}

void Report(State& state, double seconds) {
	const double per = state.frames != 0 ? 1.0 / static_cast<double>(state.frames) : 0.0;
	const auto&  s   = state.sum;
	const auto   f   = [per](uint64_t value) { return static_cast<double>(value) * per; };
	char         line[640];
	std::string  text;
	std::snprintf(line, sizeof(line),
	              "[queue-share] %.0f s, %llu of %llu frames tracked (1 in %llu); per tracked frame: accesses graphics "
	              "%.0f (images %.0f) compute %.0f (images %.0f), repeats left out %.0f, ranges %.0f | cross-queue "
	              "conflicts in %.1f accesses: read-after-write %.1f (images %.1f), write-after-read %.1f (images %.1f), "
	              "write-after-write %.1f (images %.1f)\n",
	              seconds, static_cast<unsigned long long>(state.frames), static_cast<unsigned long long>(state.all_frames),
	              static_cast<unsigned long long>(Every()), f(s.accesses[0]), f(s.image_accesses[0]), f(s.accesses[1]),
	              f(s.image_accesses[1]), f(s.repeats), f(s.ranges), f(s.conflict_accesses),
	              f(s.conflicts[ReadAfterWrite][0] + s.conflicts[ReadAfterWrite][1]), f(s.conflicts[ReadAfterWrite][1]),
	              f(s.conflicts[WriteAfterRead][0] + s.conflicts[WriteAfterRead][1]), f(s.conflicts[WriteAfterRead][1]),
	              f(s.conflicts[WriteAfterWrite][0] + s.conflicts[WriteAfterWrite][1]),
	              f(s.conflicts[WriteAfterWrite][1]));
	text += line;
	std::snprintf(line, sizeof(line),
	              "[queue-share]   forced joins per frame: %.1f (most in a frame %llu): graphics behind compute %.1f, "
	              "compute behind graphics %.1f | beyond the guest's own joins: %.1f (most %llu): %.1f, %.1f | guest per "
	              "frame: waits passed on the other class's label graphics %.1f compute %.1f, other waits passed %.1f "
	              "%.1f, label writes %.1f %.1f | most conflicting accesses in a frame %llu\n",
	              f(s.joins[0] + s.joins[1]), static_cast<unsigned long long>(state.max_joins), f(s.joins[0]),
	              f(s.joins[1]), f(s.joins_extra[0] + s.joins_extra[1]),
	              static_cast<unsigned long long>(state.max_joins_extra), f(s.joins_extra[0]), f(s.joins_extra[1]),
	              f(s.guest_waits[0]), f(s.guest_waits[1]), f(s.guest_waits_other[0]), f(s.guest_waits_other[1]),
	              f(s.guest_labels[0]), f(s.guest_labels[1]), static_cast<unsigned long long>(state.max_conflicts));
	text += line;
	std::vector<TopRange> top;
	top.reserve(state.top.size());
	for (const auto& entry: state.top) {
		top.push_back(entry.second);
	}
	const auto total = [](const TopRange& r) { return r.count[0] + r.count[1] + r.count[2]; };
	std::sort(top.begin(), top.end(), [&](const TopRange& a, const TopRange& b) { return total(a) > total(b); });
	for (size_t i = 0; i < top.size() && i < 8; i++) {
		const auto& r = top[i];
		if (r.image) {
			std::snprintf(line, sizeof(line), "[queue-share]   #%zu image slot %llu:", i + 1,
			              static_cast<unsigned long long>((r.begin & ~(uint64_t {1} << 62u)) >> 12u));
		} else {
			std::snprintf(line, sizeof(line), "[queue-share]   #%zu buffer 0x%012llx+0x%llx:", i + 1,
			              static_cast<unsigned long long>(r.begin), static_cast<unsigned long long>(r.size));
		}
		text += line;
		for (uint32_t k = 0; k < KindCount; k++) {
			std::snprintf(line, sizeof(line), " %s %.1f", KindNames[k], f(r.count[k]));
			text += line;
		}
		text += " a frame\n";
	}
	std::fputs(text.c_str(), stderr);
	state.sum    = {};
	state.frames = state.all_frames = 0;
	state.max_joins = state.max_joins_extra = state.max_conflicts = 0;
	state.top.clear();
}

// The intervals covering [begin, end) exist and have their borders there; returns the first.
std::map<uint64_t, Interval>::iterator Cover(std::map<uint64_t, Interval>& map, uint64_t begin, uint64_t end) {
	auto it = map.upper_bound(begin);
	if (it != map.begin()) {
		auto before = std::prev(it);
		if (before->second.end > begin) {
			if (before->first < begin) {
				// Split it at begin.
				const Interval tail = before->second;
				before->second.end  = begin;
				it                  = map.emplace_hint(it, begin, tail);
			} else {
				it = before;
			}
		}
	}
	const auto first_key = begin;
	uint64_t   at        = begin;
	while (at < end) {
		if (it == map.end() || it->first > at) {
			// A gap up to the next interval or the end.
			const uint64_t gap_end = it == map.end() ? end : std::min(end, it->first);
			it                     = map.emplace_hint(it, at, Interval {gap_end, {}});
		} else if (it->second.end > end) {
			// Split it at end.
			const Interval tail = it->second;
			it->second.end      = end;
			map.emplace_hint(std::next(it), end, tail);
		}
		at = it->second.end;
		++it;
	}
	return map.find(first_key);
}

} // namespace

void NoteSlow(uint64_t begin, uint64_t end, bool write, bool image) {
	if (!CpuCost::OnGpuThread() || end <= begin) {
		return; // the async-compute thread's accesses are not this thread's order
	}
	auto& state = Get();
	if (!state.sampling) {
		return;
	}
	const int cls = QueueStatsQueue() != 0 ? 1 : 0;
	if (cls != state.last_class) {
		state.last_class = cls;
		state.run++;
	}
	state.now.accesses[cls]++;
	state.now.image_accesses[cls] += image ? 1u : 0u;
	auto& repeat = state.repeats[static_cast<size_t>(((begin ^ (end << 17u)) * 0x9e3779b97f4a7c15ull) >> 52u)];
	if (repeat.begin == begin && repeat.end == end && repeat.run == state.run && (repeat.write || !write)) {
		state.now.repeats++;
		return;
	}
	repeat             = {begin, end, state.run, write};
	const uint32_t seq = ++state.seq[cls];
	const int      other = 1 - cls;
	bool           found[KindCount] {};
	uint32_t       latest = 0; // the other class's latest conflicting access
	for (auto it = Cover(state.intervals, begin, end); it != state.intervals.end() && it->first < end; ++it) {
		auto& a = it->second.access;
		if (write) {
			if (a.read[other] != 0) {
				found[WriteAfterRead] = true;
				latest                = std::max(latest, a.read[other]);
			}
			if (a.write[other] != 0) {
				found[WriteAfterWrite] = true;
				latest                 = std::max(latest, a.write[other]);
			}
			a.write[cls] = seq;
			a.read[cls]  = 0; // the write stands for both from here on
		} else {
			if (a.write[other] != 0) {
				found[ReadAfterWrite] = true;
				latest                = std::max(latest, a.write[other]);
			}
			a.read[cls] = seq;
		}
	}
	if (latest == 0) {
		return;
	}
	state.now.conflict_accesses++;
	auto& top = state.top[(begin * 0x9e3779b97f4a7c15ull) ^ end];
	top.begin = begin;
	top.size  = end - begin;
	top.image = image;
	for (uint32_t k = 0; k < KindCount; k++) {
		if (found[k]) {
			state.now.conflicts[k][image ? 1 : 0]++;
			top.count[k]++;
		}
	}
	// The other stream is cut where it stands: everything it recorded so far goes first.
	if (latest > state.cut[other]) {
		state.cut[other] = state.seq[other];
		state.now.joins[cls]++;
	}
	if (latest > state.cut_guest[other]) {
		state.cut_guest[other] = state.seq[other];
		state.now.joins_extra[cls]++;
	}
}

void GuestLabelSlow() {
	if (!CpuCost::OnGpuThread()) {
		return;
	}
	auto& state = Get();
	if (state.sampling) {
		state.now.guest_labels[QueueStatsQueue() != 0 ? 1 : 0]++;
	}
}

void GuestWaitSlow(bool cross_queue, bool passed) {
	if (!CpuCost::OnGpuThread() || !passed) {
		return;
	}
	auto& state = Get();
	if (!state.sampling) {
		return;
	}
	const int cls = QueueStatsQueue() != 0 ? 1 : 0;
	if (!cross_queue) {
		state.now.guest_waits_other[cls]++;
		return;
	}
	// The guest's own join: what the other class recorded before its label is submitted first.
	state.now.guest_waits[cls]++;
	state.cut_guest[1 - cls] = state.seq[1 - cls];
	state.run++;
}

void FrameEndSlow() {
	if (!CpuCost::OnGpuThread()) {
		return;
	}
	auto& state = Get();
	state.all_frames++;
	if (state.sampling) {
		state.now.ranges = state.intervals.size();
		Add(state.sum, state.now);
		state.frames++;
		state.max_joins       = std::max(state.max_joins, state.now.joins[0] + state.now.joins[1]);
		state.max_joins_extra = std::max(state.max_joins_extra, state.now.joins_extra[0] + state.now.joins_extra[1]);
		state.max_conflicts   = std::max(state.max_conflicts, state.now.conflict_accesses);
	}
	state.now = {};
	state.intervals.clear();
	state.seq[0] = state.seq[1] = 0;
	state.cut[0] = state.cut[1] = 0;
	state.cut_guest[0] = state.cut_guest[1] = 0;
	state.last_class = -1;
	state.run++;
	state.frame++;
	state.sampling = state.frame % Every() == 0;
	const auto now = std::chrono::steady_clock::now();
	if (now - state.last >= std::chrono::seconds(5)) {
		const double seconds = std::chrono::duration<double>(now - state.last).count();
		state.last           = now;
		Report(state, seconds);
	}
}

} // namespace QueueShare::Detail

} // namespace Libs::Graphics
