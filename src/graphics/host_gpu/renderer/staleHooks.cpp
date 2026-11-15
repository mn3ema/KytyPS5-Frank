#include "graphics/host_gpu/renderer/staleHooks.h"

#include "common/sampler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace {

double NowMs() {
	using namespace std::chrono;
	static const auto start = steady_clock::now();
	return duration<double, std::milli>(steady_clock::now() - start).count();
}

// Every object made gets a new generation: a handle the driver hands out again (NVIDIA reuses
// them at once) names a new object, while a descriptor written with the old one still holds the
// old object's data.
uint64_t g_next_gen = 1;
struct ImageRec {
	std::string desc;
	bool        live         = true;
	double      destroyed_ms = 0;
	uint64_t    gen          = 0;
};
struct ViewRec {
	VkImage  image        = VK_NULL_HANDLE;
	bool     live         = true;
	double   destroyed_ms = 0;
	uint64_t gen          = 0;
	uint64_t image_gen    = 0;
};
struct ViewEntry {
	VkImageView view = VK_NULL_HANDLE;
	uint64_t    gen  = 0;
};
struct BufferEntry {
	VkBuffer buffer = VK_NULL_HANDLE;
	uint64_t gen    = 0;
};
struct SlotKey {
	VkDescriptorSet set  = VK_NULL_HANDLE;
	uint64_t        slot = 0; // binding << 32 | element
	bool operator==(const SlotKey&) const = default;
};
struct SlotKeyHash {
	size_t operator()(const SlotKey& k) const noexcept {
		return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.set) * 0x9E3779B97F4A7C15ull ^
		                             k.slot);
	}
};

std::mutex                                                      g_mutex;
std::unordered_map<VkImage, ImageRec>                           g_images;
std::unordered_map<VkImageView, ViewRec>                        g_views;
std::unordered_map<VkImage, std::vector<VkImageView>>           g_live_views_of;
std::unordered_map<VkDescriptorSet, std::unordered_map<uint64_t, ViewEntry>> g_set_contents;
std::unordered_map<VkImageView, std::unordered_set<SlotKey, SlotKeyHash>>      g_holders;
std::unordered_map<VkDescriptorSet, uint32_t>                   g_dead_in_set;
std::unordered_map<VkDescriptorSet, VkDescriptorPool>           g_set_pool;
std::unordered_map<VkDescriptorPool, std::unordered_set<VkDescriptorSet>> g_pool_sets;
size_t g_dead_images = 0, g_dead_views = 0;
uint32_t g_mask = 15;

std::atomic<uint64_t> g_reports_total {0};

// In-flight tracking: the command buffer (and its recording, by epoch) that last used each view,
// image and descriptor set, and the timeline value each recording's submission signals. A view
// or image destroyed while such a submission is still pending on the GPU is destroyed too early.
struct Use {
	VkCommandBuffer cb    = VK_NULL_HANDLE;
	uint64_t        epoch = 0;
	double          ms    = 0;
	const char*     what  = "";
};
struct Submission {
	uint64_t    epoch = 0;
	VkSemaphore sem   = VK_NULL_HANDLE;
	uint64_t    value = 0;
	double      ms    = 0;
};
thread_local VkCommandBuffer t_cb = VK_NULL_HANDLE;
VkDevice                     g_device = VK_NULL_HANDLE;
std::unordered_map<VkCommandBuffer, uint64_t>   g_cb_epoch;
std::unordered_map<VkCommandBuffer, Submission> g_cb_submit;
std::unordered_map<VkImageView, Use>            g_view_use;
std::unordered_map<VkImage, Use>                g_image_use;
std::unordered_map<VkDescriptorSet, Use>        g_set_use;
std::unordered_set<VkSemaphore>                 g_timelines;

struct BufferRec {
	std::string desc;
	bool        live         = true;
	double      destroyed_ms = 0;
	uint64_t    gen          = 0;
};
std::unordered_map<VkBuffer, BufferRec> g_buffers;
std::unordered_map<VkBuffer, Use>       g_buffer_use;
std::unordered_map<VkDescriptorSet, std::unordered_map<uint64_t, BufferEntry>> g_set_buffers;
std::unordered_map<VkBuffer, std::unordered_set<SlotKey, SlotKeyHash>>      g_buffer_holders;
size_t g_dead_buffers = 0;
std::atomic<uint64_t>                           g_premature {0};

PFN_vkBeginCommandBuffer  g_begin_cb       = nullptr;
PFN_vkQueueSubmit         g_submit         = nullptr;
PFN_vkQueueSubmit2        g_submit2        = nullptr;
PFN_vkQueueSubmit2        g_submit2_khr    = nullptr;
PFN_vkCreateSemaphore     g_create_sem     = nullptr;
PFN_vkGetSemaphoreCounterValue g_counter   = nullptr;
PFN_vkCreateBuffer        g_create_buffer  = nullptr;
PFN_vkDestroyBuffer       g_destroy_buffer = nullptr;
PFN_vkCmdBindVertexBuffers  g_bind_vertex  = nullptr;
PFN_vkCmdBindVertexBuffers2 g_bind_vertex2 = nullptr;
PFN_vkCmdBindIndexBuffer    g_bind_index   = nullptr;
PFN_vkCmdCopyBuffer         g_copy_buffer  = nullptr;
PFN_vkCmdCopyBuffer2        g_copy_buffer2 = nullptr;
PFN_vkCmdFillBuffer         g_fill_buffer  = nullptr;
PFN_vkCmdUpdateBuffer       g_update_buffer = nullptr;
PFN_vkCmdDrawIndirect       g_draw_indirect = nullptr;
PFN_vkCmdDrawIndexedIndirect g_draw_indexed_indirect = nullptr;
PFN_vkCmdDrawIndirectCount  g_draw_indirect_count = nullptr;
PFN_vkCmdDrawIndexedIndirectCount g_draw_indexed_indirect_count = nullptr;
PFN_vkCmdDispatchIndirect   g_dispatch_indirect = nullptr;
PFN_vkCmdDrawMeshTasksIndirectEXT g_mesh_indirect = nullptr;
PFN_vkCmdDrawMeshTasksIndirectCountEXT g_mesh_indirect_count = nullptr;

PFN_vkCreateImageView           g_create_view        = nullptr;
PFN_vkDestroyImageView          g_destroy_view       = nullptr;
PFN_vkUpdateDescriptorSets      g_update_sets        = nullptr;
PFN_vkAllocateDescriptorSets    g_allocate_sets      = nullptr;
PFN_vkFreeDescriptorSets        g_free_sets          = nullptr;
PFN_vkResetDescriptorPool       g_reset_pool         = nullptr;
PFN_vkDestroyDescriptorPool     g_destroy_pool       = nullptr;
PFN_vkCmdBindDescriptorSets     g_bind_sets          = nullptr;
PFN_vkCmdPushDescriptorSetKHR   g_push_khr           = nullptr;
PFN_vkCmdBeginRendering         g_begin_rendering    = nullptr;
PFN_vkCmdBeginRendering         g_begin_rendering_khr = nullptr;
PFN_vkCmdCopyImage              g_copy_image         = nullptr;
PFN_vkCmdCopyImage2             g_copy_image2        = nullptr;
PFN_vkCmdCopyImage2             g_copy_image2_khr    = nullptr;
PFN_vkCmdBlitImage              g_blit_image         = nullptr;
PFN_vkCmdBlitImage2             g_blit_image2        = nullptr;
PFN_vkCmdBlitImage2             g_blit_image2_khr    = nullptr;
PFN_vkCmdCopyBufferToImage      g_buffer_to_image    = nullptr;
PFN_vkCmdCopyBufferToImage2     g_buffer_to_image2   = nullptr;
PFN_vkCmdCopyBufferToImage2     g_buffer_to_image2_khr = nullptr;
PFN_vkCmdCopyImageToBuffer      g_image_to_buffer    = nullptr;
PFN_vkCmdCopyImageToBuffer2     g_image_to_buffer2   = nullptr;
PFN_vkCmdCopyImageToBuffer2     g_image_to_buffer2_khr = nullptr;
PFN_vkCmdClearColorImage        g_clear_color        = nullptr;
PFN_vkCmdClearDepthStencilImage g_clear_depth        = nullptr;
PFN_vkCmdResolveImage           g_resolve            = nullptr;
PFN_vkCmdPipelineBarrier        g_barrier            = nullptr;
PFN_vkCmdPipelineBarrier2       g_barrier2           = nullptr;
PFN_vkCmdPipelineBarrier2       g_barrier2_khr       = nullptr;

// Host stack of the reporting call, by name: who handed the stale handle to Vulkan.
std::string HostStack() {
	std::string text;
#ifdef _WIN32
	void*      frames[10] {};
	const auto count = CaptureStackBackTrace(2, 10, frames, nullptr);
	for (USHORT i = 0; i < count; i++) {
		auto name = Common::HostAddressName(frames[i]);
		if (name.empty()) {
			char buffer[32];
			std::snprintf(buffer, sizeof(buffer), " %p", frames[i]);
			name = buffer;
		}
		text += "\n[stale-use]     at" + name;
	}
#endif
	return text;
}

// Called with g_mutex held. Per kind: up to 24 detailed reports (with host stacks), then a count
// at powers of two. "held" (a set slot still naming a view as it is destroyed) is harmless until
// the set is bound again, so it gets 4 lines and no stacks.
void Report(const char* what, const std::string& detail) {
	static std::unordered_map<std::string, uint64_t> counts;
	const bool held  = std::string_view(what) == "held";
	const auto n     = ++counts[what];
	const auto limit = held ? 4u : 24u;
	g_reports_total.fetch_add(1);
	if (n <= limit) {
		std::fprintf(stderr, "[stale-use] %s: %s%s\n", what, detail.c_str(),
		             held ? "" : HostStack().c_str());
		std::fflush(stderr);
	} else if ((n & (n - 1)) == 0) {
		std::fprintf(stderr, "[stale-use] %s: %llu so far (latest: %s)\n", what,
		             static_cast<unsigned long long>(n), detail.c_str());
		std::fflush(stderr);
	}
}

std::string ImageText(VkImage image) {
	char       buffer[96];
	const auto found = g_images.find(image);
	if (found == g_images.end()) {
		std::snprintf(buffer, sizeof(buffer), "image %p (not made by VMA)", static_cast<void*>(image));
		return buffer;
	}
	std::string text = found->second.desc;
	if (!found->second.live) {
		std::snprintf(buffer, sizeof(buffer), " [image destroyed %.1f ms ago]",
		              NowMs() - found->second.destroyed_ms);
		text += buffer;
	}
	return text;
}

bool ImageDead(VkImage image) {
	const auto found = g_images.find(image);
	return found != g_images.end() && !found->second.live;
}

Use MakeUse(const char* what) {
	const auto epoch = g_cb_epoch.find(t_cb);
	return Use {t_cb, epoch != g_cb_epoch.end() ? epoch->second : 0, NowMs(), what};
}

// With g_mutex held. Empty when the use is no longer pending on the GPU.
std::string PendingText(const Use& use) {
	if (use.cb == VK_NULL_HANDLE) {
		return {};
	}
	const auto epoch = g_cb_epoch.find(use.cb);
	if (epoch == g_cb_epoch.end() || epoch->second != use.epoch) {
		return {}; // re-recorded since: that submission was waited for
	}
	char       buffer[192];
	const auto submit = g_cb_submit.find(use.cb);
	if (submit == g_cb_submit.end() || submit->second.epoch != use.epoch) {
		std::snprintf(buffer, sizeof(buffer),
		              "used %.1f ms ago (%s) by command buffer %p, not submitted yet", NowMs() - use.ms,
		              use.what, static_cast<void*>(use.cb));
		return buffer;
	}
	uint64_t counter = 0;
	if (submit->second.sem == VK_NULL_HANDLE || g_counter == nullptr ||
	    g_counter(g_device, submit->second.sem, &counter) != VK_SUCCESS ||
	    counter >= submit->second.value) {
		return {};
	}
	std::snprintf(buffer, sizeof(buffer),
	              "used %.1f ms ago (%s) by command buffer %p, submitted %.1f ms ago signalling %llu; GPU at %llu",
	              NowMs() - use.ms, use.what, static_cast<void*>(use.cb), NowMs() - submit->second.ms,
	              static_cast<unsigned long long>(submit->second.value),
	              static_cast<unsigned long long>(counter));
	return buffer;
}

// Live view of a live image? With g_mutex held. Unknown views count as dead: every view is made
// after the hooks went in.
bool ViewUsable(VkImageView view) {
	if (view == VK_NULL_HANDLE) {
		return true;
	}
	const auto found = g_views.find(view);
	if (found == g_views.end() || !found->second.live) {
		return false;
	}
	const auto image = g_images.find(found->second.image);
	return image == g_images.end() ||
	       (image->second.live && image->second.gen == found->second.image_gen);
}

// A set slot's view: the very object written there, still alive, over its image still alive.
bool EntryUsable(const ViewEntry& entry) {
	if (entry.view == VK_NULL_HANDLE) {
		return true;
	}
	const auto found = g_views.find(entry.view);
	return found != g_views.end() && found->second.gen == entry.gen && ViewUsable(entry.view);
}

uint64_t ViewGen(VkImageView view) {
	const auto found = g_views.find(view);
	return found != g_views.end() ? found->second.gen : 0;
}

std::string ViewText(VkImageView view) {
	char       buffer[128];
	const auto found = g_views.find(view);
	if (found == g_views.end()) {
		std::snprintf(buffer, sizeof(buffer), "view %p (unknown)", static_cast<void*>(view));
		return buffer;
	}
	if (found->second.live) {
		std::snprintf(buffer, sizeof(buffer), "view %p (live) of ", static_cast<void*>(view));
	} else {
		std::snprintf(buffer, sizeof(buffer), "view %p (destroyed %.1f ms ago) of ",
		              static_cast<void*>(view), NowMs() - found->second.destroyed_ms);
	}
	return buffer + ImageText(found->second.image);
}

std::string EntryText(const ViewEntry& entry) {
	const auto found = g_views.find(entry.view);
	if (found != g_views.end() && found->second.gen != entry.gen) {
		char buffer[128];
		std::snprintf(buffer, sizeof(buffer),
		              "view %p written in generation %llu, a destroyed view whose handle now names "
		              "another view (",
		              static_cast<void*>(entry.view), static_cast<unsigned long long>(entry.gen));
		return buffer + ViewText(entry.view) + ")";
	}
	return ViewText(entry.view);
}

void CheckView(VkImageView view, const char* what, const char* where) {
	if (!ViewUsable(view)) {
		Report(what, std::string(where) + " " + ViewText(view));
	} else if (view != VK_NULL_HANDLE && t_cb != VK_NULL_HANDLE && (g_mask & 2u) != 0) {
		g_view_use[view] = MakeUse(what);
	}
}

void CheckImage(VkImage image, const char* what) {
	if (image != VK_NULL_HANDLE && ImageDead(image)) {
		Report(what, ImageText(image));
	} else if (image != VK_NULL_HANDLE && t_cb != VK_NULL_HANDLE && (g_mask & 2u) != 0) {
		g_image_use[image] = MakeUse(what);
	}
}

bool BufferDead(VkBuffer buffer) {
	const auto found = g_buffers.find(buffer);
	return found != g_buffers.end() && !found->second.live;
}

bool BufferEntryUsable(const BufferEntry& entry) {
	const auto found = g_buffers.find(entry.buffer);
	return found == g_buffers.end() || (found->second.live && found->second.gen == entry.gen);
}

uint64_t BufferGen(VkBuffer buffer) {
	const auto found = g_buffers.find(buffer);
	return found != g_buffers.end() ? found->second.gen : 0;
}

std::string BufferText(VkBuffer buffer) {
	char       buffer_text[96];
	const auto found = g_buffers.find(buffer);
	if (found == g_buffers.end()) {
		std::snprintf(buffer_text, sizeof(buffer_text), "buffer %p (untracked)", static_cast<void*>(buffer));
		return buffer_text;
	}
	std::string text = found->second.desc;
	if (!found->second.live) {
		std::snprintf(buffer_text, sizeof(buffer_text), " [destroyed %.1f ms ago]",
		              NowMs() - found->second.destroyed_ms);
		text += buffer_text;
	}
	return text;
}

void CheckBuffer(VkBuffer buffer, const char* what) {
	if (buffer == VK_NULL_HANDLE) {
		return;
	}
	if (BufferDead(buffer)) {
		Report(what, BufferText(buffer));
	} else if (t_cb != VK_NULL_HANDLE && (g_mask & 2u) != 0) {
		g_buffer_use[buffer] = MakeUse(what);
	}
}

bool IsBufferDescriptor(VkDescriptorType type) {
	return type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
	       type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
	       type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
}

bool IsImageDescriptor(VkDescriptorType type) {
	return type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
	       type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
	       type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}

void DropSlot(VkDescriptorSet set, uint64_t slot, const ViewEntry& entry) {
	const auto old = entry.view;
	if (old == VK_NULL_HANDLE) {
		return;
	}
	if (auto holders = g_holders.find(old); holders != g_holders.end()) {
		holders->second.erase(SlotKey {set, slot});
		if (holders->second.empty()) {
			g_holders.erase(holders);
		}
	}
	if (!EntryUsable(entry)) {
		if (auto dead = g_dead_in_set.find(set); dead != g_dead_in_set.end() && dead->second > 0) {
			if (--dead->second == 0) {
				g_dead_in_set.erase(dead);
			}
		}
	}
}

void ForgetSet(VkDescriptorSet set) {
	if (auto contents = g_set_contents.find(set); contents != g_set_contents.end()) {
		for (const auto& [slot, entry]: contents->second) {
			if (auto holders = g_holders.find(entry.view); holders != g_holders.end()) {
				holders->second.erase(SlotKey {set, slot});
				if (holders->second.empty()) {
					g_holders.erase(holders);
				}
			}
		}
		g_set_contents.erase(contents);
	}
	if (auto contents = g_set_buffers.find(set); contents != g_set_buffers.end()) {
		for (const auto& [slot, entry]: contents->second) {
			if (auto holders = g_buffer_holders.find(entry.buffer); holders != g_buffer_holders.end()) {
				holders->second.erase(SlotKey {set, slot});
				if (holders->second.empty()) {
					g_buffer_holders.erase(holders);
				}
			}
		}
		g_set_buffers.erase(contents);
	}
	g_dead_in_set.erase(set);
	g_set_pool.erase(set);
	g_set_use.erase(set);
}

// A view (or the image under it) stopped being usable: every set slot still holding it.
void NoteUnusable(VkImageView view, const char* why) {
	const auto holders = g_holders.find(view);
	if (holders == g_holders.end()) {
		return;
	}
	const auto current = ViewGen(view);
	for (const auto& key: holders->second) {
		const auto contents = g_set_contents.find(key.set);
		if (contents == g_set_contents.end()) {
			continue;
		}
		const auto entry = contents->second.find(key.slot);
		if (entry == contents->second.end() || entry->second.gen != current) {
			continue; // an older object under this handle: counted when it went
		}
		g_dead_in_set[key.set]++;
		char buffer[160];
		std::snprintf(buffer, sizeof(buffer), "set %p binding %u element %u still holds %s; ",
		              static_cast<void*>(key.set), static_cast<uint32_t>(key.slot >> 32u),
		              static_cast<uint32_t>(key.slot), why);
		Report("held", buffer + ViewText(view));
	}
}

VKAPI_ATTR VkResult VKAPI_CALL HookCreateView(VkDevice device, const VkImageViewCreateInfo* info,
                                              const VkAllocationCallbacks* allocator,
                                              VkImageView* view) {
	const auto result = g_create_view(device, info, allocator, view);
	if (result == VK_SUCCESS) {
		std::scoped_lock lock(g_mutex);
		auto&            rec = g_views[*view];
		if (!rec.live) {
			g_dead_views--;
		}
		const auto image = g_images.find(info->image);
		rec = ViewRec {info->image, true, 0, g_next_gen++,
		               image != g_images.end() ? image->second.gen : 0};
		g_live_views_of[info->image].push_back(*view);
		CheckImage(info->image, "view made on a destroyed image");
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL HookDestroyView(VkDevice device, VkImageView view,
                                           const VkAllocationCallbacks* allocator) {
	if (view != VK_NULL_HANDLE) {
		std::scoped_lock lock(g_mutex);
		if (auto use = g_view_use.find(view); use != g_view_use.end()) {
			if (const auto pending = PendingText(use->second); !pending.empty()) {
				g_premature.fetch_add(1);
				Report("view destroyed while in use", ViewText(view) + "; " + pending);
			}
			g_view_use.erase(use);
		}
		if (const auto holders = g_holders.find(view); holders != g_holders.end()) {
			for (const auto& key: holders->second) {
				const auto use = g_set_use.find(key.set);
				if (use == g_set_use.end()) {
					continue;
				}
				if (const auto pending = PendingText(use->second); !pending.empty()) {
					g_premature.fetch_add(1);
					char buffer[96];
					std::snprintf(buffer, sizeof(buffer), "set %p binding %u element %u holds ",
					              static_cast<void*>(key.set), static_cast<uint32_t>(key.slot >> 32u),
					              static_cast<uint32_t>(key.slot));
					Report("view destroyed while a set holding it is in use",
					       buffer + ViewText(view) + "; set " + pending);
				}
			}
		}
		if (auto found = g_views.find(view); found != g_views.end() && found->second.live) {
			const bool was_usable = ViewUsable(view);
			found->second.live         = false;
			found->second.destroyed_ms = NowMs();
			g_dead_views++;
			if (auto list = g_live_views_of.find(found->second.image); list != g_live_views_of.end()) {
				std::erase(list->second, view);
				if (list->second.empty()) {
					g_live_views_of.erase(list);
				}
			}
			if (was_usable) {
				NoteUnusable(view, "a view being destroyed");
			}
		}
		if (g_dead_views > 200000) {
			std::erase_if(g_views, [](const auto& entry) {
				return !entry.second.live && !g_holders.contains(entry.first);
			});
			g_dead_views = 0;
			for (const auto& [v, rec]: g_views) {
				g_dead_views += rec.live ? 0 : 1;
			}
		}
	}
	g_destroy_view(device, view, allocator);
}

void NoteBufferWrite(const VkWriteDescriptorSet& write, bool push, const char* where) {
	for (uint32_t i = 0; i < write.descriptorCount; i++) {
		const auto buffer = write.pBufferInfo[i].buffer;
		if (push) {
			if (BufferDead(buffer)) {
				Report("push descriptor", std::string(where) + " " + BufferText(buffer));
			} else if (buffer != VK_NULL_HANDLE && t_cb != VK_NULL_HANDLE && (g_mask & 2u) != 0) {
				g_buffer_use[buffer] = MakeUse("push descriptor");
			}
			continue;
		}
		const uint64_t slot =
		    static_cast<uint64_t>(write.dstBinding) << 32u | (write.dstArrayElement + i);
		auto& contents = g_set_buffers[write.dstSet];
		if (const auto old = contents.find(slot); old != contents.end()) {
			if (auto holders = g_buffer_holders.find(old->second.buffer); holders != g_buffer_holders.end()) {
				holders->second.erase(SlotKey {write.dstSet, slot});
				if (holders->second.empty()) {
					g_buffer_holders.erase(holders);
				}
			}
			if (!BufferEntryUsable(old->second)) {
				if (auto dead = g_dead_in_set.find(write.dstSet);
				    dead != g_dead_in_set.end() && dead->second > 0 && --dead->second == 0) {
					g_dead_in_set.erase(dead);
				}
			}
			contents.erase(old);
		}
		if (buffer == VK_NULL_HANDLE) {
			continue;
		}
		contents[slot] = BufferEntry {buffer, BufferGen(buffer)};
		g_buffer_holders[buffer].insert(SlotKey {write.dstSet, slot});
		if (BufferDead(buffer)) {
			g_dead_in_set[write.dstSet]++;
			char text[96];
			std::snprintf(text, sizeof(text), "set %p binding %u element %u <- ",
			              static_cast<void*>(write.dstSet), write.dstBinding, write.dstArrayElement + i);
			Report("descriptor write", text + BufferText(buffer));
		}
	}
}

void NoteWrite(const VkWriteDescriptorSet& write, bool push, const char* where) {
	if (IsBufferDescriptor(write.descriptorType) && write.pBufferInfo != nullptr) {
		NoteBufferWrite(write, push, where);
		return;
	}
	if (!IsImageDescriptor(write.descriptorType) || write.pImageInfo == nullptr) {
		return;
	}
	for (uint32_t i = 0; i < write.descriptorCount; i++) {
		const auto view = write.pImageInfo[i].imageView;
		if (push) {
			CheckView(view, "push descriptor", where);
			continue;
		}
		const uint64_t slot =
		    static_cast<uint64_t>(write.dstBinding) << 32u | (write.dstArrayElement + i);
		auto&      contents = g_set_contents[write.dstSet];
		auto&      entry    = contents[slot];
		DropSlot(write.dstSet, slot, entry);
		entry = ViewEntry {view, ViewGen(view)};
		if (view == VK_NULL_HANDLE) {
			contents.erase(slot);
			continue;
		}
		g_holders[view].insert(SlotKey {write.dstSet, slot});
		if (!ViewUsable(view)) {
			g_dead_in_set[write.dstSet]++;
			char buffer[96];
			std::snprintf(buffer, sizeof(buffer), "set %p binding %u element %u <- ",
			              static_cast<void*>(write.dstSet), write.dstBinding,
			              write.dstArrayElement + i);
			Report("descriptor write", buffer + ViewText(view));
		}
	}
}

VKAPI_ATTR void VKAPI_CALL HookUpdateSets(VkDevice device, uint32_t write_count,
                                          const VkWriteDescriptorSet* writes, uint32_t copy_count,
                                          const VkCopyDescriptorSet* copies) {
	{
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < write_count; i++) {
			NoteWrite(writes[i], false, "");
		}
		if (copy_count != 0) {
			static std::atomic<bool> once {false};
			if (!once.exchange(true)) {
				std::fprintf(stderr, "[stale-use] note: descriptor copies are not tracked\n");
			}
		}
	}
	g_update_sets(device, write_count, writes, copy_count, copies);
}

VKAPI_ATTR VkResult VKAPI_CALL HookAllocateSets(VkDevice device,
                                                const VkDescriptorSetAllocateInfo* info,
                                                VkDescriptorSet* sets) {
	const auto result = g_allocate_sets(device, info, sets);
	if (result == VK_SUCCESS) {
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
			ForgetSet(sets[i]);
			g_set_pool[sets[i]] = info->descriptorPool;
			g_pool_sets[info->descriptorPool].insert(sets[i]);
		}
	}
	return result;
}

void ForgetPool(VkDescriptorPool pool) {
	if (auto found = g_pool_sets.find(pool); found != g_pool_sets.end()) {
		for (const auto set: found->second) {
			ForgetSet(set);
		}
		g_pool_sets.erase(found);
	}
}

VKAPI_ATTR VkResult VKAPI_CALL HookFreeSets(VkDevice device, VkDescriptorPool pool,
                                            uint32_t count, const VkDescriptorSet* sets) {
	{
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < count; i++) {
			ForgetSet(sets[i]);
			if (auto found = g_pool_sets.find(pool); found != g_pool_sets.end()) {
				found->second.erase(sets[i]);
			}
		}
	}
	return g_free_sets(device, pool, count, sets);
}

VKAPI_ATTR VkResult VKAPI_CALL HookResetPool(VkDevice device, VkDescriptorPool pool,
                                             VkDescriptorPoolResetFlags flags) {
	{
		std::scoped_lock lock(g_mutex);
		ForgetPool(pool);
	}
	return g_reset_pool(device, pool, flags);
}

VKAPI_ATTR void VKAPI_CALL HookDestroyPool(VkDevice device, VkDescriptorPool pool,
                                           const VkAllocationCallbacks* allocator) {
	{
		std::scoped_lock lock(g_mutex);
		ForgetPool(pool);
	}
	g_destroy_pool(device, pool, allocator);
}

VKAPI_ATTR void VKAPI_CALL HookBindSets(VkCommandBuffer command, VkPipelineBindPoint bind_point,
                                        VkPipelineLayout layout, uint32_t first, uint32_t count,
                                        const VkDescriptorSet* sets, uint32_t dynamic_count,
                                        const uint32_t* dynamic_offsets) {
	{
		std::scoped_lock lock(g_mutex);
		t_cb = command;
		for (uint32_t i = 0; (g_mask & 2u) != 0 && i < count; i++) {
			g_set_use[sets[i]] = MakeUse("bound set");
		}
		for (uint32_t i = 0; i < count; i++) {
			const auto dead = g_dead_in_set.find(sets[i]);
			if (dead == g_dead_in_set.end() || dead->second == 0) {
				continue;
			}
			std::string first_dead;
			if (const auto contents = g_set_contents.find(sets[i]);
			    contents != g_set_contents.end()) {
				for (const auto& [slot, entry]: contents->second) {
					if (!EntryUsable(entry)) {
						char buffer[64];
						std::snprintf(buffer, sizeof(buffer), "binding %u element %u: ",
						              static_cast<uint32_t>(slot >> 32u),
						              static_cast<uint32_t>(slot));
						first_dead = buffer + EntryText(entry);
						break;
					}
				}
			}
			if (first_dead.empty()) {
				if (const auto contents = g_set_buffers.find(sets[i]); contents != g_set_buffers.end()) {
					for (const auto& [slot, held]: contents->second) {
						if (!BufferEntryUsable(held)) {
							char text[64];
							std::snprintf(text, sizeof(text), "binding %u element %u: ",
							              static_cast<uint32_t>(slot >> 32u), static_cast<uint32_t>(slot));
							first_dead = text + BufferText(held.buffer) +
							             (BufferGen(held.buffer) != held.gen ? " (handle reused since)" : "");
							break;
						}
					}
				}
			}
			char buffer[128];
			std::snprintf(buffer, sizeof(buffer), "set %p (index %u, bind point %d) holds %u destroyed views/buffers; first ",
			              static_cast<void*>(sets[i]), first + i, static_cast<int>(bind_point),
			              dead->second);
			Report("bind", buffer + first_dead);
		}
	}
	g_bind_sets(command, bind_point, layout, first, count, sets, dynamic_count, dynamic_offsets);
}

void CheckPush(uint32_t count, const VkWriteDescriptorSet* writes) {
	std::scoped_lock lock(g_mutex);
	for (uint32_t i = 0; i < count; i++) {
		char where[32];
		std::snprintf(where, sizeof(where), "binding %u", writes[i].dstBinding);
		NoteWrite(writes[i], true, where);
	}
}

VKAPI_ATTR void VKAPI_CALL HookPushKhr(VkCommandBuffer command, VkPipelineBindPoint bind_point,
                                       VkPipelineLayout layout, uint32_t set, uint32_t count,
                                       const VkWriteDescriptorSet* writes) {
	t_cb = command;
	CheckPush(count, writes);
	g_push_khr(command, bind_point, layout, set, count, writes);
}

void CheckRendering(const VkRenderingInfo* info) {
	std::scoped_lock lock(g_mutex);
	char             where[32];
	for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
		std::snprintf(where, sizeof(where), "color %u", i);
		CheckView(info->pColorAttachments[i].imageView, "rendering attachment", where);
		CheckView(info->pColorAttachments[i].resolveImageView, "rendering attachment",
		          "color resolve");
	}
	if (info->pDepthAttachment != nullptr) {
		CheckView(info->pDepthAttachment->imageView, "rendering attachment", "depth");
	}
	if (info->pStencilAttachment != nullptr) {
		CheckView(info->pStencilAttachment->imageView, "rendering attachment", "stencil");
	}
}

VKAPI_ATTR void VKAPI_CALL HookBeginRendering(VkCommandBuffer command, const VkRenderingInfo* info) {
	t_cb = command;
	CheckRendering(info);
	g_begin_rendering(command, info);
}
VKAPI_ATTR void VKAPI_CALL HookBeginRenderingKhr(VkCommandBuffer command,
                                                 const VkRenderingInfo* info) {
	t_cb = command;
	CheckRendering(info);
	g_begin_rendering_khr(command, info);
}

void CheckImages(const char* what, VkImage a, VkImage b = VK_NULL_HANDLE) {
	std::scoped_lock lock(g_mutex);
	CheckImage(a, what);
	CheckImage(b, what);
}

VKAPI_ATTR void VKAPI_CALL HookCopyImage(VkCommandBuffer c, VkImage src, VkImageLayout sl,
                                         VkImage dst, VkImageLayout dl, uint32_t n,
                                         const VkImageCopy* r) {
	t_cb = c;
	CheckImages("copy image", src, dst);
	g_copy_image(c, src, sl, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookCopyImage2(VkCommandBuffer c, const VkCopyImageInfo2* i) {
	t_cb = c;
	CheckImages("copy image", i->srcImage, i->dstImage);
	g_copy_image2(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookCopyImage2Khr(VkCommandBuffer c, const VkCopyImageInfo2* i) {
	t_cb = c;
	CheckImages("copy image", i->srcImage, i->dstImage);
	g_copy_image2_khr(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookBlit(VkCommandBuffer c, VkImage src, VkImageLayout sl, VkImage dst,
                                    VkImageLayout dl, uint32_t n, const VkImageBlit* r,
                                    VkFilter f) {
	t_cb = c;
	CheckImages("blit", src, dst);
	g_blit_image(c, src, sl, dst, dl, n, r, f);
}
VKAPI_ATTR void VKAPI_CALL HookBlit2(VkCommandBuffer c, const VkBlitImageInfo2* i) {
	t_cb = c;
	CheckImages("blit", i->srcImage, i->dstImage);
	g_blit_image2(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookBlit2Khr(VkCommandBuffer c, const VkBlitImageInfo2* i) {
	t_cb = c;
	CheckImages("blit", i->srcImage, i->dstImage);
	g_blit_image2_khr(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookBufferToImage(VkCommandBuffer c, VkBuffer b, VkImage dst,
                                             VkImageLayout dl, uint32_t n,
                                             const VkBufferImageCopy* r) {
	t_cb = c;
	CheckImages("buffer to image", dst);
	{
		std::scoped_lock lock(g_mutex);
		CheckBuffer(b, "buffer to image");
	}
	g_buffer_to_image(c, b, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookBufferToImage2(VkCommandBuffer c,
                                              const VkCopyBufferToImageInfo2* i) {
	t_cb = c;
	CheckImages("buffer to image", i->dstImage);
	{
		std::scoped_lock lock(g_mutex);
		CheckBuffer(i->srcBuffer, "buffer/image copy");
	}
	g_buffer_to_image2(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookBufferToImage2Khr(VkCommandBuffer c,
                                                 const VkCopyBufferToImageInfo2* i) {
	t_cb = c;
	CheckImages("buffer to image", i->dstImage);
	{
		std::scoped_lock lock(g_mutex);
		CheckBuffer(i->srcBuffer, "buffer/image copy");
	}
	g_buffer_to_image2_khr(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookImageToBuffer(VkCommandBuffer c, VkImage src, VkImageLayout sl,
                                             VkBuffer b, uint32_t n,
                                             const VkBufferImageCopy* r) {
	t_cb = c;
	CheckImages("image to buffer", src);
	{
		std::scoped_lock lock(g_mutex);
		CheckBuffer(b, "image to buffer");
	}
	g_image_to_buffer(c, src, sl, b, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookImageToBuffer2(VkCommandBuffer c,
                                              const VkCopyImageToBufferInfo2* i) {
	t_cb = c;
	CheckImages("image to buffer", i->srcImage);
	{
		std::scoped_lock lock(g_mutex);
		CheckBuffer(i->dstBuffer, "buffer/image copy");
	}
	g_image_to_buffer2(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookImageToBuffer2Khr(VkCommandBuffer c,
                                                 const VkCopyImageToBufferInfo2* i) {
	t_cb = c;
	CheckImages("image to buffer", i->srcImage);
	{
		std::scoped_lock lock(g_mutex);
		CheckBuffer(i->dstBuffer, "buffer/image copy");
	}
	g_image_to_buffer2_khr(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookClearColor(VkCommandBuffer c, VkImage image, VkImageLayout l,
                                          const VkClearColorValue* v, uint32_t n,
                                          const VkImageSubresourceRange* r) {
	t_cb = c;
	CheckImages("clear color image", image);
	g_clear_color(c, image, l, v, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookClearDepth(VkCommandBuffer c, VkImage image, VkImageLayout l,
                                          const VkClearDepthStencilValue* v, uint32_t n,
                                          const VkImageSubresourceRange* r) {
	t_cb = c;
	CheckImages("clear depth image", image);
	g_clear_depth(c, image, l, v, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookResolve(VkCommandBuffer c, VkImage src, VkImageLayout sl,
                                       VkImage dst, VkImageLayout dl, uint32_t n,
                                       const VkImageResolve* r) {
	t_cb = c;
	CheckImages("resolve", src, dst);
	g_resolve(c, src, sl, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookBarrier(VkCommandBuffer command, VkPipelineStageFlags src,
                                       VkPipelineStageFlags dst, VkDependencyFlags flags,
                                       uint32_t memory_count, const VkMemoryBarrier* memory,
                                       uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                       uint32_t image_count, const VkImageMemoryBarrier* images) {
	t_cb = command;
	if (image_count != 0) {
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < image_count; i++) {
			CheckImage(images[i].image, "image barrier");
		}
	}
	if (buffer_count != 0) {
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < buffer_count; i++) {
			CheckBuffer(buffers[i].buffer, "buffer barrier");
		}
	}
	g_barrier(command, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count,
	          images);
}
void CheckDependency(const VkDependencyInfo* info) {
	if (info->imageMemoryBarrierCount == 0 && info->bufferMemoryBarrierCount == 0) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
		CheckBuffer(info->pBufferMemoryBarriers[i].buffer, "buffer barrier");
	}
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
		CheckImage(info->pImageMemoryBarriers[i].image, "image barrier");
	}
}
VKAPI_ATTR void VKAPI_CALL HookBarrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
	t_cb = command;
	CheckDependency(info);
	g_barrier2(command, info);
}
VKAPI_ATTR void VKAPI_CALL HookBarrier2Khr(VkCommandBuffer command, const VkDependencyInfo* info) {
	t_cb = command;
	CheckDependency(info);
	g_barrier2_khr(command, info);
}

VKAPI_ATTR VkResult VKAPI_CALL HookBeginCb(VkCommandBuffer command,
                                           const VkCommandBufferBeginInfo* info) {
	{
		std::scoped_lock lock(g_mutex);
		g_cb_epoch[command]++;
	}
	return g_begin_cb(command, info);
}

VKAPI_ATTR VkResult VKAPI_CALL HookCreateSemaphore(VkDevice device,
                                                   const VkSemaphoreCreateInfo* info,
                                                   const VkAllocationCallbacks* allocator,
                                                   VkSemaphore* semaphore) {
	const auto result = g_create_sem(device, info, allocator, semaphore);
	if (result == VK_SUCCESS) {
		for (auto* next = static_cast<const VkBaseInStructure*>(info->pNext); next != nullptr;
		     next = next->pNext) {
			if (next->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO &&
			    reinterpret_cast<const VkSemaphoreTypeCreateInfo*>(next)->semaphoreType ==
			        VK_SEMAPHORE_TYPE_TIMELINE) {
				std::scoped_lock lock(g_mutex);
				g_timelines.insert(*semaphore);
			}
		}
	}
	return result;
}

void NoteSubmitted(VkCommandBuffer cb, VkSemaphore sem, uint64_t value) {
	const auto epoch = g_cb_epoch.find(cb);
	g_cb_submit[cb]  = Submission {epoch != g_cb_epoch.end() ? epoch->second : 0, sem, value, NowMs()};
}

VKAPI_ATTR VkResult VKAPI_CALL HookSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                                          VkFence fence) {
	{
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < count; i++) {
			const auto& submit = submits[i];
			const VkTimelineSemaphoreSubmitInfo* timeline = nullptr;
			for (auto* next = static_cast<const VkBaseInStructure*>(submit.pNext); next != nullptr;
			     next = next->pNext) {
				if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
					timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
				}
			}
			VkSemaphore sem   = VK_NULL_HANDLE;
			uint64_t    value = 0;
			for (uint32_t j = 0; timeline != nullptr && j < submit.signalSemaphoreCount &&
			                     j < timeline->signalSemaphoreValueCount;
			     j++) {
				if (g_timelines.contains(submit.pSignalSemaphores[j]) &&
				    timeline->pSignalSemaphoreValues[j] > value) {
					sem   = submit.pSignalSemaphores[j];
					value = timeline->pSignalSemaphoreValues[j];
				}
			}
			for (uint32_t j = 0; j < submit.commandBufferCount; j++) {
				NoteSubmitted(submit.pCommandBuffers[j], sem, value);
			}
		}
	}
	return g_submit(queue, count, submits, fence);
}

void NoteSubmit2(uint32_t count, const VkSubmitInfo2* submits) {
	std::scoped_lock lock(g_mutex);
	for (uint32_t i = 0; i < count; i++) {
		VkSemaphore sem   = VK_NULL_HANDLE;
		uint64_t    value = 0;
		for (uint32_t j = 0; j < submits[i].signalSemaphoreInfoCount; j++) {
			const auto& info = submits[i].pSignalSemaphoreInfos[j];
			if (g_timelines.contains(info.semaphore) && info.value > value) {
				sem   = info.semaphore;
				value = info.value;
			}
		}
		for (uint32_t j = 0; j < submits[i].commandBufferInfoCount; j++) {
			NoteSubmitted(submits[i].pCommandBufferInfos[j].commandBuffer, sem, value);
		}
	}
}
VKAPI_ATTR VkResult VKAPI_CALL HookSubmit2(VkQueue queue, uint32_t count,
                                           const VkSubmitInfo2* submits, VkFence fence) {
	NoteSubmit2(count, submits);
	return g_submit2(queue, count, submits, fence);
}
VKAPI_ATTR VkResult VKAPI_CALL HookSubmit2Khr(VkQueue queue, uint32_t count,
                                              const VkSubmitInfo2* submits, VkFence fence) {
	NoteSubmit2(count, submits);
	return g_submit2_khr(queue, count, submits, fence);
}

void CheckBuffers(VkCommandBuffer c, const char* what, VkBuffer a, VkBuffer b = VK_NULL_HANDLE) {
	t_cb = c;
	std::scoped_lock lock(g_mutex);
	CheckBuffer(a, what);
	CheckBuffer(b, what);
}
VKAPI_ATTR void VKAPI_CALL HookBindVertex(VkCommandBuffer c, uint32_t first, uint32_t n,
                                          const VkBuffer* buffers, const VkDeviceSize* offsets) {
	t_cb = c;
	{
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < n; i++) CheckBuffer(buffers[i], "vertex buffer");
	}
	g_bind_vertex(c, first, n, buffers, offsets);
}
VKAPI_ATTR void VKAPI_CALL HookBindVertex2(VkCommandBuffer c, uint32_t first, uint32_t n,
                                           const VkBuffer* buffers, const VkDeviceSize* offsets,
                                           const VkDeviceSize* sizes, const VkDeviceSize* strides) {
	t_cb = c;
	{
		std::scoped_lock lock(g_mutex);
		for (uint32_t i = 0; i < n; i++) CheckBuffer(buffers[i], "vertex buffer");
	}
	g_bind_vertex2(c, first, n, buffers, offsets, sizes, strides);
}
VKAPI_ATTR void VKAPI_CALL HookBindIndex(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                         VkIndexType t) {
	CheckBuffers(c, "index buffer", b);
	g_bind_index(c, b, o, t);
}
VKAPI_ATTR void VKAPI_CALL HookCopyBuffer(VkCommandBuffer c, VkBuffer s, VkBuffer d, uint32_t n,
                                          const VkBufferCopy* r) {
	CheckBuffers(c, "copy buffer", s, d);
	g_copy_buffer(c, s, d, n, r);
}
VKAPI_ATTR void VKAPI_CALL HookCopyBuffer2(VkCommandBuffer c, const VkCopyBufferInfo2* i) {
	CheckBuffers(c, "copy buffer", i->srcBuffer, i->dstBuffer);
	g_copy_buffer2(c, i);
}
VKAPI_ATTR void VKAPI_CALL HookFillBuffer(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                          VkDeviceSize n, uint32_t v) {
	CheckBuffers(c, "fill buffer", b);
	g_fill_buffer(c, b, o, n, v);
}
VKAPI_ATTR void VKAPI_CALL HookUpdateBuffer(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                            VkDeviceSize n, const void* data) {
	CheckBuffers(c, "update buffer", b);
	g_update_buffer(c, b, o, n, data);
}
VKAPI_ATTR void VKAPI_CALL HookDrawIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                            uint32_t n, uint32_t stride) {
	CheckBuffers(c, "indirect buffer", b);
	g_draw_indirect(c, b, o, n, stride);
}
VKAPI_ATTR void VKAPI_CALL HookDrawIndexedIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                                   uint32_t n, uint32_t stride) {
	CheckBuffers(c, "indirect buffer", b);
	g_draw_indexed_indirect(c, b, o, n, stride);
}
VKAPI_ATTR void VKAPI_CALL HookDrawIndirectCount(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                                 VkBuffer cb, VkDeviceSize co, uint32_t n,
                                                 uint32_t stride) {
	CheckBuffers(c, "indirect buffer", b, cb);
	g_draw_indirect_count(c, b, o, cb, co, n, stride);
}
VKAPI_ATTR void VKAPI_CALL HookDrawIndexedIndirectCount(VkCommandBuffer c, VkBuffer b,
                                                        VkDeviceSize o, VkBuffer cb,
                                                        VkDeviceSize co, uint32_t n,
                                                        uint32_t stride) {
	CheckBuffers(c, "indirect buffer", b, cb);
	g_draw_indexed_indirect_count(c, b, o, cb, co, n, stride);
}
VKAPI_ATTR void VKAPI_CALL HookDispatchIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o) {
	CheckBuffers(c, "indirect buffer", b);
	g_dispatch_indirect(c, b, o);
}
VKAPI_ATTR void VKAPI_CALL HookMeshIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                            uint32_t n, uint32_t stride) {
	CheckBuffers(c, "mesh indirect buffer", b);
	g_mesh_indirect(c, b, o, n, stride);
}
VKAPI_ATTR void VKAPI_CALL HookMeshIndirectCount(VkCommandBuffer c, VkBuffer b, VkDeviceSize o,
                                                 VkBuffer cb, VkDeviceSize co, uint32_t n,
                                                 uint32_t stride) {
	CheckBuffers(c, "mesh indirect buffer", b, cb);
	g_mesh_indirect_count(c, b, o, cb, co, n, stride);
}

void NoteBufferDestroyed(VkBuffer buffer) {
	auto found = g_buffers.find(buffer);
	if (found == g_buffers.end() || !found->second.live) {
		return;
	}
	if (auto use = g_buffer_use.find(buffer); use != g_buffer_use.end()) {
		if (const auto pending = PendingText(use->second); !pending.empty()) {
			g_premature.fetch_add(1);
			Report("buffer freed while in use", BufferText(buffer) + "; " + pending);
		}
		g_buffer_use.erase(use);
	}
	if (const auto holders = g_buffer_holders.find(buffer); holders != g_buffer_holders.end()) {
		for (const auto& key: holders->second) {
			const auto contents = g_set_buffers.find(key.set);
			if (contents == g_set_buffers.end()) {
				continue;
			}
			const auto entry = contents->second.find(key.slot);
			if (entry == contents->second.end() || entry->second.gen != found->second.gen) {
				continue;
			}
			g_dead_in_set[key.set]++;
			const auto use = g_set_use.find(key.set);
			if (use == g_set_use.end()) {
				continue;
			}
			if (const auto pending = PendingText(use->second); !pending.empty()) {
				g_premature.fetch_add(1);
				char text[96];
				std::snprintf(text, sizeof(text), "set %p binding %u element %u holds ",
				              static_cast<void*>(key.set), static_cast<uint32_t>(key.slot >> 32u),
				              static_cast<uint32_t>(key.slot));
				Report("buffer freed while a set holding it is in use",
				       text + BufferText(buffer) + "; set " + pending);
			}
		}
	}
	found->second.live         = false;
	found->second.destroyed_ms = NowMs();
	if (++g_dead_buffers > 200000) {
		std::erase_if(g_buffers, [](const auto& entry) {
			return !entry.second.live && !g_buffer_holders.contains(entry.first);
		});
		g_dead_buffers = 0;
	}
}

void NoteBufferCreated(VkBuffer buffer, uint64_t size, const char* kind) {
	char text[96];
	std::snprintf(text, sizeof(text), "buffer %p size 0x%llx (%s)", static_cast<void*>(buffer),
	              static_cast<unsigned long long>(size), kind);
	auto& rec = g_buffers[buffer];
	if (!rec.live) {
		g_dead_buffers--;
	}
	rec = BufferRec {text, true, 0, g_next_gen++};
}

VKAPI_ATTR VkResult VKAPI_CALL HookCreateBuffer(VkDevice device, const VkBufferCreateInfo* info,
                                                const VkAllocationCallbacks* allocator,
                                                VkBuffer* buffer) {
	const auto result = g_create_buffer(device, info, allocator, buffer);
	if (result == VK_SUCCESS) {
		std::scoped_lock lock(g_mutex);
		NoteBufferCreated(*buffer, info->size, "vkCreateBuffer");
	}
	return result;
}
VKAPI_ATTR void VKAPI_CALL HookDestroyBuffer(VkDevice device, VkBuffer buffer,
                                             const VkAllocationCallbacks* allocator) {
	if (buffer != VK_NULL_HANDLE) {
		std::scoped_lock lock(g_mutex);
		NoteBufferDestroyed(buffer);
	}
	g_destroy_buffer(device, buffer, allocator);
}

template <typename Fn>
void Install(Fn& slot, Fn& original, Fn hook) {
	if (slot != nullptr && slot != hook) {
		original = slot;
		slot     = hook;
	}
}

} // namespace

bool StaleHooksEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STALE_HOOKS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void StaleHooksNoteImage(VkImage image, const vk::ImageCreateInfo* info, bool created) {
	if (!StaleHooksEnabled() || image == VK_NULL_HANDLE || (g_mask & 1u) == 0) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	if (created) {
		char buffer[128];
		std::snprintf(buffer, sizeof(buffer), "image %p %ux%ux%u layers=%u levels=%u fmt=%d",
		              static_cast<void*>(image), info->extent.width, info->extent.height,
		              info->extent.depth, info->arrayLayers, info->mipLevels,
		              static_cast<int>(info->format));
		auto& rec = g_images[image];
		if (!rec.live) {
			g_dead_images--;
		}
		rec = ImageRec {buffer, true, 0, g_next_gen++};
		return;
	}
	auto found = g_images.find(image);
	if (found == g_images.end() || !found->second.live) {
		return;
	}
	if (auto use = g_image_use.find(image); use != g_image_use.end()) {
		if (const auto pending = PendingText(use->second); !pending.empty()) {
			g_premature.fetch_add(1);
			Report("image freed while in use", ImageText(image) + "; " + pending);
		}
		g_image_use.erase(use);
	}
	// Views of it that stay alive past this point name freed memory.
	if (auto list = g_live_views_of.find(image); list != g_live_views_of.end()) {
		for (const auto view: list->second) {
			Report("image freed under a live view", ViewText(view));
			NoteUnusable(view, "a view of an image being freed");
		}
	}
	found->second.live         = false;
	found->second.destroyed_ms = NowMs();
	if (++g_dead_images > 200000) {
		std::erase_if(g_images, [](const auto& entry) { return !entry.second.live; });
		g_dead_images = 0;
	}
}

void StaleHooksNoteBuffer(VkBuffer buffer, uint64_t size, const char* kind, bool created) {
	if (!StaleHooksEnabled() || buffer == VK_NULL_HANDLE || (g_mask & 4u) == 0) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	if (created) {
		NoteBufferCreated(buffer, size, kind);
	} else {
		NoteBufferDestroyed(buffer);
	}
}

void InstallStaleHooks(vk::Device device) {
	if (!StaleHooksEnabled()) {
		return;
	}
	// KYTY_STALE_HOOKS_MASK: 1 images, views, descriptors, rendering, image commands; 2 command
	// buffer and submission tracking (in-flight checks); 4 buffer lifetimes (descriptor buffers
	// checked by 1); 8 buffer commands (vertex, index, indirect, copies). Default 15.
	if (const char* mask = std::getenv("KYTY_STALE_HOOKS_MASK"); mask != nullptr) {
		g_mask = static_cast<uint32_t>(std::strtoul(mask, nullptr, 0));
	}
	auto& d  = VULKAN_HPP_DEFAULT_DISPATCHER;
	g_device  = static_cast<VkDevice>(device);
	g_counter = d.vkGetSemaphoreCounterValue != nullptr ? d.vkGetSemaphoreCounterValue
	                                                    : d.vkGetSemaphoreCounterValueKHR;
	if ((g_mask & 2u) != 0) {
		Install(d.vkBeginCommandBuffer, g_begin_cb, &HookBeginCb);
		Install(d.vkCreateSemaphore, g_create_sem, &HookCreateSemaphore);
		Install(d.vkQueueSubmit, g_submit, &HookSubmit);
		Install(d.vkQueueSubmit2, g_submit2, &HookSubmit2);
		Install(d.vkQueueSubmit2KHR, g_submit2_khr, &HookSubmit2Khr);
	}
	if ((g_mask & 4u) != 0) {
		Install(d.vkCreateBuffer, g_create_buffer, &HookCreateBuffer);
		Install(d.vkDestroyBuffer, g_destroy_buffer, &HookDestroyBuffer);
	}
	if ((g_mask & 8u) != 0) {
		Install(d.vkCmdBindVertexBuffers, g_bind_vertex, &HookBindVertex);
		Install(d.vkCmdBindVertexBuffers2, g_bind_vertex2, &HookBindVertex2);
		Install(d.vkCmdBindIndexBuffer, g_bind_index, &HookBindIndex);
		Install(d.vkCmdCopyBuffer, g_copy_buffer, &HookCopyBuffer);
		Install(d.vkCmdCopyBuffer2, g_copy_buffer2, &HookCopyBuffer2);
		Install(d.vkCmdFillBuffer, g_fill_buffer, &HookFillBuffer);
		Install(d.vkCmdUpdateBuffer, g_update_buffer, &HookUpdateBuffer);
		Install(d.vkCmdDrawIndirect, g_draw_indirect, &HookDrawIndirect);
		Install(d.vkCmdDrawIndexedIndirect, g_draw_indexed_indirect, &HookDrawIndexedIndirect);
		Install(d.vkCmdDrawIndirectCount, g_draw_indirect_count, &HookDrawIndirectCount);
		Install(d.vkCmdDrawIndexedIndirectCount, g_draw_indexed_indirect_count,
	        &HookDrawIndexedIndirectCount);
		Install(d.vkCmdDispatchIndirect, g_dispatch_indirect, &HookDispatchIndirect);
		Install(d.vkCmdDrawMeshTasksIndirectEXT, g_mesh_indirect, &HookMeshIndirect);
		Install(d.vkCmdDrawMeshTasksIndirectCountEXT, g_mesh_indirect_count, &HookMeshIndirectCount);
	}
	if ((g_mask & 1u) != 0) {
		Install(d.vkCreateImageView, g_create_view, &HookCreateView);
		Install(d.vkDestroyImageView, g_destroy_view, &HookDestroyView);
		Install(d.vkUpdateDescriptorSets, g_update_sets, &HookUpdateSets);
		Install(d.vkAllocateDescriptorSets, g_allocate_sets, &HookAllocateSets);
		Install(d.vkFreeDescriptorSets, g_free_sets, &HookFreeSets);
		Install(d.vkResetDescriptorPool, g_reset_pool, &HookResetPool);
		Install(d.vkDestroyDescriptorPool, g_destroy_pool, &HookDestroyPool);
		Install(d.vkCmdBindDescriptorSets, g_bind_sets, &HookBindSets);
		Install(d.vkCmdPushDescriptorSetKHR, g_push_khr, &HookPushKhr);
		Install(d.vkCmdBeginRendering, g_begin_rendering, &HookBeginRendering);
		Install(d.vkCmdBeginRenderingKHR, g_begin_rendering_khr, &HookBeginRenderingKhr);
		Install(d.vkCmdCopyImage, g_copy_image, &HookCopyImage);
		Install(d.vkCmdCopyImage2, g_copy_image2, &HookCopyImage2);
		Install(d.vkCmdCopyImage2KHR, g_copy_image2_khr, &HookCopyImage2Khr);
		Install(d.vkCmdBlitImage, g_blit_image, &HookBlit);
		Install(d.vkCmdBlitImage2, g_blit_image2, &HookBlit2);
		Install(d.vkCmdBlitImage2KHR, g_blit_image2_khr, &HookBlit2Khr);
		Install(d.vkCmdCopyBufferToImage, g_buffer_to_image, &HookBufferToImage);
		Install(d.vkCmdCopyBufferToImage2, g_buffer_to_image2, &HookBufferToImage2);
		Install(d.vkCmdCopyBufferToImage2KHR, g_buffer_to_image2_khr, &HookBufferToImage2Khr);
		Install(d.vkCmdCopyImageToBuffer, g_image_to_buffer, &HookImageToBuffer);
		Install(d.vkCmdCopyImageToBuffer2, g_image_to_buffer2, &HookImageToBuffer2);
		Install(d.vkCmdCopyImageToBuffer2KHR, g_image_to_buffer2_khr, &HookImageToBuffer2Khr);
		Install(d.vkCmdClearColorImage, g_clear_color, &HookClearColor);
		Install(d.vkCmdClearDepthStencilImage, g_clear_depth, &HookClearDepth);
		Install(d.vkCmdResolveImage, g_resolve, &HookResolve);
		Install(d.vkCmdPipelineBarrier, g_barrier, &HookBarrier);
		Install(d.vkCmdPipelineBarrier2, g_barrier2, &HookBarrier2);
		Install(d.vkCmdPipelineBarrier2KHR, g_barrier2_khr, &HookBarrier2Khr);
	}
	std::fprintf(stderr, "[stale-use] hooks installed (mask %u)\n", g_mask);
}

} // namespace Libs::Graphics
