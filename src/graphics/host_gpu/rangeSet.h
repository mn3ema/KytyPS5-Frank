#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>

namespace Libs::Graphics {

class RangeSet final {
public:
	RangeSet() = default;
	RangeSet(const RangeSet& other): m_ranges(other.m_ranges) {
		if (other.m_filter) {
			m_filter = std::make_unique<Filter>(*other.m_filter);
		}
	}
	RangeSet& operator=(const RangeSet& other) {
		if (this != &other) {
			m_ranges = other.m_ranges;
			m_filter = other.m_filter ? std::make_unique<Filter>(*other.m_filter) : nullptr;
		}
		return *this;
	}
	RangeSet(RangeSet&&) noexcept            = default;
	RangeSet& operator=(RangeSet&&) noexcept = default;
	~RangeSet()                              = default;

	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		uint64_t begin = address;
		uint64_t last  = end;
		while (it != m_ranges.end() && it->first <= last) {
			begin = std::min(begin, it->first);
			last  = std::max(last, it->second);
			it    = m_ranges.erase(it);
		}
		m_ranges.emplace(begin, last);
		if (m_filter) {
			m_filter->Mark(address, end);
		}
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
		// Subtract leaves the filter's bits set (they only over-report); rebuild it now and then.
		if (m_filter && ++m_filter->subtracts >= Filter::RebuildEvery) {
			RebuildFilter();
		}
	}

	void Clear() {
		m_ranges.clear();
		if (m_filter) {
			m_filter->ClearAll();
		}
	}

	// A block filter for IntersectsFiltered (v322): one bit per 64 KiB block, hashed into 2^20
	// bits. Every byte in the set has its block's bit set (Add marks the added range, Subtract keeps
	// bits, a rebuild recomputes them), so a query whose blocks are all clear has no byte in the set:
	// the answer is exact, only faster. Collisions and stale bits just fall back to the map.
	void EnableFilter() {
		if (!m_filter) {
			m_filter = std::make_unique<Filter>();
			RebuildFilter();
		}
	}

	// Intersects, answering "no" from the filter when it can. Counters: fast "no", map lookups.
	[[nodiscard]] bool IntersectsFiltered(uint64_t address, uint64_t size, uint64_t* fast = nullptr,
	                                      uint64_t* slow = nullptr) const {
		if (m_filter && size != 0 && size <= UINT64_MAX - address) {
			const uint64_t first = address >> Filter::BlockBits;
			const uint64_t last  = (address + size - 1) >> Filter::BlockBits;
			if (last - first < 64) {
				bool any = false;
				for (uint64_t block = first; block <= last && !any; block++) {
					any = m_filter->Test(block);
				}
				if (!any) {
					if (fast != nullptr) {
						(*fast)++;
					}
					return false;
				}
			}
		}
		if (slow != nullptr) {
			(*slow)++;
		}
		return Intersects(address, size);
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	// Like ForEachInRange, but passes each overlapping range whole instead of clipped.
	template <typename Func>
	void ForEachOverlapping(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			if (it->second > address) {
				func(it->first, it->second);
			}
		}
	}

	[[nodiscard]] bool   Empty() const { return m_ranges.empty(); }
	// The lowest start and the highest end (the set must not be empty).
	[[nodiscard]] uint64_t Lowest() const { return m_ranges.begin()->first; }
	[[nodiscard]] uint64_t Highest() const { return m_ranges.rbegin()->second; }
	[[nodiscard]] size_t Count() const { return m_ranges.size(); }

private:
	// Two bit arrays: Mark sets bits in the active one; a rebuild fills the other and then makes it
	// active, so a reader that does not hold the writer's lock (the GPU thread's unlocked
	// HasGpuDirtyBytes, as with the map) never sees a half-cleared array. Words are relaxed atomics.
	struct Filter {
		static constexpr unsigned BlockBits    = 16;
		static constexpr uint64_t IndexBits    = 20;
		static constexpr uint64_t IndexMask    = (uint64_t {1} << IndexBits) - 1u;
		static constexpr size_t   WordCount    = (size_t {1} << IndexBits) / 64;
		static constexpr uint32_t RebuildEvery = 2048;
		std::unique_ptr<std::atomic<uint64_t>[]> bits[2] {
		    std::make_unique<std::atomic<uint64_t>[]>(WordCount),
		    std::make_unique<std::atomic<uint64_t>[]>(WordCount)};
		std::atomic<int> active {0};
		uint32_t         subtracts = 0;

		Filter() = default;
		Filter(const Filter& other) {
			const int a = other.active.load(std::memory_order_acquire);
			for (size_t i = 0; i < WordCount; i++) {
				bits[0][i].store(other.bits[a][i].load(std::memory_order_relaxed),
				                 std::memory_order_relaxed);
			}
			subtracts = other.subtracts;
		}
		Filter& operator=(const Filter&) = delete;

		[[nodiscard]] bool Test(uint64_t block) const {
			const auto index = block & IndexMask;
			const auto word  = bits[active.load(std::memory_order_acquire)][index >> 6u].load(
                std::memory_order_relaxed);
			return ((word >> (index & 63u)) & 1u) != 0;
		}
		static void MarkInto(std::atomic<uint64_t>* words, uint64_t begin, uint64_t end) {
			const uint64_t first = begin >> BlockBits;
			const uint64_t last  = (end - 1) >> BlockBits;
			if (last - first >= IndexMask) {
				for (size_t i = 0; i < WordCount; i++) {
					words[i].store(~uint64_t {0}, std::memory_order_relaxed);
				}
				return;
			}
			for (uint64_t block = first; block <= last; block++) {
				const auto index = block & IndexMask;
				words[index >> 6u].fetch_or(uint64_t {1} << (index & 63u), std::memory_order_relaxed);
			}
		}
		void Mark(uint64_t begin, uint64_t end) {
			MarkInto(bits[active.load(std::memory_order_relaxed)].get(), begin, end);
		}
		void ClearAll() {
			for (auto& array: bits) {
				for (size_t i = 0; i < WordCount; i++) {
					array[i].store(0, std::memory_order_relaxed);
				}
			}
			subtracts = 0;
		}
	};

	void RebuildFilter() {
		const int next  = 1 - m_filter->active.load(std::memory_order_relaxed);
		auto*     words = m_filter->bits[next].get();
		for (size_t i = 0; i < Filter::WordCount; i++) {
			words[i].store(0, std::memory_order_relaxed);
		}
		for (const auto& [begin, end]: m_ranges) {
			Filter::MarkInto(words, begin, end);
		}
		m_filter->active.store(next, std::memory_order_release);
		m_filter->subtracts = 0;
	}

	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
	std::unique_ptr<Filter>      m_filter;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
