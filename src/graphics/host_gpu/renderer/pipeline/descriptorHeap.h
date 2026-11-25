#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	static constexpr uint32_t DescriptorSetBatch = 32;

	// `max_batch`: how many sets of the layout are allocated ahead at most (a layout with many
	// descriptors and few uses per frame, as a callable split's fixed one, takes a small number).
	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout, uint32_t max_batch = DescriptorSetBatch);

private:
	struct UsedSet {
		vk::DescriptorSet set;
		uint64_t          tick = 0;
	};
	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
		// v445o (KYTY_DESC_SET_REUSE): the sets of the current pool handed out while the cut was
		// on, oldest first, each with the tick of the command buffer that bound it: a ring in a
		// vector (head, count; the capacity is a power of two).
		std::vector<UsedSet> used;
		uint32_t             used_head  = 0;
		uint32_t             used_count = 0;
	};
	void NoteUsed(Batch& batch, vk::DescriptorSet set, uint64_t tick);

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool();

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
