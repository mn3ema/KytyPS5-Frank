#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/syncToggle.h"

#include <algorithm>

namespace Libs::Graphics {
namespace {

// Sets that do not fit the push-descriptor limit (32 on NVIDIA) come from these pools, ~40
// storage buffers each: a pool of 8,192 held ~200 sets, so the GPU thread switched pools (failed
// allocations, then a reset or a new pool) every few hundred such draws. Eight times larger pools
// switch eight times less often (~4 MB of descriptors each).
constexpr uint32_t   DescriptorHeapCount = 4096;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 65536},
    // The shader-data buffer as a uniform buffer (KYTY_SHADER_DATA_UBO): one per stage.
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 16384},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 65536},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 8192},
};

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout, uint32_t max_batch) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);
	max_batch = std::clamp(max_batch, 1u, DescriptorSetBatch);

	// v445o (KYTY_DESC_SET_REUSE; on by default since v445p, =0 off; toggle mode "setreuse"): a set of this layout whose
	// command buffer the host GPU has finished is handed out again instead of a newly allocated
	// one (allocateDescriptorSets every frame was 0.34-0.8 ms a frame in g450g). Every caller
	// writes its whole write list into the set and binds it at once, in the command buffer being
	// recorded (CurrentTick): the update runs with the bind, on the recording thread when there is
	// one, and a finished tick's update and bind have both run. The sets noted belong to the
	// current pool: a pool switch below drops them with m_sets, so a set of a retired (later
	// reset or destroyed) pool is never handed out. With the cut off nothing is noted or reused;
	// what an earlier on window noted stays valid until the pool switch.
	const bool reuse = SyncToggle::SetReuse();
	auto&      batch = m_sets[layout];
	if (reuse && batch.used_count != 0) {
		const auto head = batch.used[batch.used_head];
		if (m_master_semaphore.IsFree(head.tick)) {
			batch.used_head = (batch.used_head + 1u) & (static_cast<uint32_t>(batch.used.size()) - 1u);
			batch.used_count--;
			NoteUsed(batch, head.set, m_master_semaphore.CurrentTick());
			SyncToggle::Bump(SyncToggle::Hits().set_reused);
			return head.set;
		}
		SyncToggle::Bump(SyncToggle::Hits().set_busy);
	}
	SyncToggle::Bump(SyncToggle::Hits().set_fresh);
	if (batch.size != 0) {
		const auto set = batch.sets[--batch.size];
		if (reuse) {
			NoteUsed(batch, set, m_master_semaphore.CurrentTick());
		}
		return set;
	}
	batch.allocation = std::min(batch.allocation, max_batch);
	if (Allocate(layout, batch)) {
		const auto set = batch.sets[--batch.size];
		if (reuse) {
			NoteUsed(batch, set, m_master_semaphore.CurrentTick());
		}
		return set;
	}

	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear(); // with every layout's noted sets: they were the retired pool's
	auto& fresh_batch = m_sets[layout];
	fresh_batch.allocation = std::min(fresh_batch.allocation, max_batch);
	EXIT_IF(!Allocate(layout, fresh_batch));
	const auto set = fresh_batch.sets[--fresh_batch.size];
	if (reuse) {
		NoteUsed(fresh_batch, set, m_master_semaphore.CurrentTick());
	}
	return set;
}

// Appends to the layout's ring (ticks never decrease: the head is the oldest).
void DescriptorHeap::NoteUsed(Batch& batch, vk::DescriptorSet set, uint64_t tick) {
	auto capacity = static_cast<uint32_t>(batch.used.size());
	if (batch.used_count == capacity) {
		const auto           grown = capacity != 0 ? capacity * 2u : 64u;
		std::vector<UsedSet> ring(grown);
		for (uint32_t i = 0; i < batch.used_count; i++) {
			ring[i] = batch.used[(batch.used_head + i) & (capacity - 1u)];
		}
		batch.used      = std::move(ring);
		batch.used_head = 0;
		capacity        = grown;
	}
	batch.used[(batch.used_head + batch.used_count) & (capacity - 1u)] = {set, tick};
	batch.used_count++;
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool() {
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
	create.pPoolSizes    = DescriptorPoolSizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
