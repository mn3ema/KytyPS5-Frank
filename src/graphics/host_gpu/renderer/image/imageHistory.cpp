#include "graphics/host_gpu/renderer/image/imageHistory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ImageHistory {

namespace {

double NowMs() {
	using namespace std::chrono;
	static const auto start = steady_clock::now();
	return duration<double, std::milli>(steady_clock::now() - start).count();
}

struct SlotEvent {
	uint64_t tick     = 0;
	double   ms       = 0;
	uint32_t binding  = 0;
	uint32_t slot     = 0;
	uint32_t key      = 0;
	uint64_t heap     = 0;
	// 'W' a view of this image written into the slot; 'P' the slot repointed to a placeholder;
	// 'R' the slot rewritten with another image's view (`other`).
	char     what     = 'W';
	VkImage  other    = VK_NULL_HANDLE;
};

struct Generation {
	uint64_t guest       = 0;
	uint64_t guest_size  = 0;
	uint32_t width       = 0;
	uint32_t height      = 0;
	uint32_t layers      = 0;
	uint32_t levels      = 0;
	uint32_t format      = 0;
	uint64_t created_tick = 0;
	double   created_ms   = 0;
	uint64_t destroyed_tick = 0;
	double   destroyed_ms   = 0;
	bool     destroyed      = false;
	// Set-0 descriptor binds, attachments.
	uint64_t last_descriptor_tick = 0, last_descriptor_shader = 0, descriptor_binds = 0;
	uint64_t last_attachment_tick = 0, attachment_binds = 0;
	char     last_attachment_kind = 0;
	uint64_t last_storage_tick = 0, last_storage_shader = 0;
	std::vector<SlotEvent> slots; // capped
	uint32_t               slots_dropped = 0;
};

constexpr size_t MaxGenerations = 4;
constexpr size_t MaxSlotEvents  = 48;
constexpr size_t MaxDeadImages  = 20000;

std::mutex                                             g_mutex;
std::unordered_map<VkImage, std::deque<Generation>>    g_images;
std::unordered_map<VkImageView, VkImage>               g_views;
std::deque<VkImage>                                    g_dead; // destroyed handles, oldest first
size_t                                                 g_dead_count = 0;

Generation* Current(VkImage image) {
	const auto found = g_images.find(image);
	if (found == g_images.end() || found->second.empty()) {
		return nullptr;
	}
	return &found->second.back();
}

void AddSlotEvent(VkImage image, SlotEvent event) {
	auto* gen = Current(image);
	if (gen == nullptr) {
		return;
	}
	if (gen->slots.size() >= MaxSlotEvents) {
		gen->slots.erase(gen->slots.begin());
		gen->slots_dropped++;
	}
	gen->slots.push_back(event);
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FAULT_MAP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void NoteCreate(VkImage image, uint64_t guest, uint64_t guest_size, uint32_t width, uint32_t height,
                uint32_t layers, uint32_t levels, uint32_t format, uint64_t tick) {
	if (!Enabled() || image == VK_NULL_HANDLE) {
		return;
	}
	std::lock_guard lock(g_mutex);
	auto& gens = g_images[image];
	if (gens.size() >= MaxGenerations) {
		gens.pop_front();
	}
	Generation gen;
	gen.guest        = guest;
	gen.guest_size   = guest_size;
	gen.width        = width;
	gen.height       = height;
	gen.layers       = layers;
	gen.levels       = levels;
	gen.format       = format;
	gen.created_tick = tick;
	gen.created_ms   = NowMs();
	gens.push_back(gen);
}

void NoteDestroy(VkImage image, uint64_t tick) {
	if (!Enabled() || image == VK_NULL_HANDLE) {
		return;
	}
	std::lock_guard lock(g_mutex);
	if (auto* gen = Current(image); gen != nullptr && !gen->destroyed) {
		gen->destroyed      = true;
		gen->destroyed_tick = tick;
		gen->destroyed_ms   = NowMs();
	}
	g_dead.push_back(image);
	// Bound the table: a handle whose every generation is old and destroyed goes.
	while (g_dead.size() > MaxDeadImages) {
		const auto old = g_dead.front();
		g_dead.pop_front();
		const auto found = g_images.find(old);
		if (found != g_images.end() && !found->second.empty() && found->second.back().destroyed &&
		    NowMs() - found->second.back().destroyed_ms > 60000.0) {
			g_images.erase(found);
		}
	}
	// Views of a destroyed image: forget them (the handle values get reused).
	std::erase_if(g_views, [image](const auto& entry) { return entry.second == image; });
}

VkImage ImageOfView(VkImageView view) {
	if (!Enabled() || view == VK_NULL_HANDLE) {
		return VK_NULL_HANDLE;
	}
	std::lock_guard lock(g_mutex);
	const auto      found = g_views.find(view);
	return found == g_views.end() ? VK_NULL_HANDLE : found->second;
}

void NoteView(VkImageView view, VkImage image) {
	if (!Enabled() || view == VK_NULL_HANDLE) {
		return;
	}
	std::lock_guard lock(g_mutex);
	g_views[view] = image;
}

void NoteBind(VkImage image, uint64_t tick, uint64_t shader_hash, char kind) {
	if (!Enabled() || image == VK_NULL_HANDLE) {
		return;
	}
	std::lock_guard lock(g_mutex);
	auto* gen = Current(image);
	if (gen == nullptr) {
		return;
	}
	switch (kind) {
		case 'D':
			gen->last_descriptor_tick   = tick;
			gen->last_descriptor_shader = shader_hash;
			gen->descriptor_binds++;
			break;
		case 'S':
			gen->last_storage_tick   = tick;
			gen->last_storage_shader = shader_hash;
			break;
		default:
			gen->last_attachment_tick = tick;
			gen->last_attachment_kind = kind;
			gen->attachment_binds++;
			break;
	}
}

void NoteBindView(VkImageView view, uint64_t tick, uint64_t shader_hash, char kind) {
	if (!Enabled() || view == VK_NULL_HANDLE) {
		return;
	}
	VkImage image = VK_NULL_HANDLE;
	{
		std::lock_guard lock(g_mutex);
		const auto found = g_views.find(view);
		if (found == g_views.end()) {
			return;
		}
		image = found->second;
	}
	NoteBind(image, tick, shader_hash, kind);
}

void NoteSlotWrite(uint32_t binding, uint32_t slot, VkImage image, VkImage previous, uint64_t tick,
                   uint64_t heap_base, uint32_t key) {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	const double now = NowMs();
	if (previous != VK_NULL_HANDLE && previous != image) {
		AddSlotEvent(previous, {tick, now, binding, slot, key, heap_base,
		                        image == VK_NULL_HANDLE ? 'P' : 'R', image});
	}
	if (image != VK_NULL_HANDLE) {
		AddSlotEvent(image, {tick, now, binding, slot, key, heap_base, 'W', previous});
	}
}

void RecentGuestRanges(VkImage image, double within_ms, std::vector<std::pair<uint64_t, uint64_t>>* out) {
	if (!Enabled() || image == VK_NULL_HANDLE || out == nullptr) {
		return;
	}
	std::lock_guard lock(g_mutex);
	const auto      found = g_images.find(image);
	if (found == g_images.end()) {
		return;
	}
	const double now = NowMs();
	for (const auto& g: found->second) {
		if (g.destroyed && now - g.destroyed_ms <= within_ms && g.guest != 0) {
			out->emplace_back(g.guest, g.guest + std::max<uint64_t>(g.guest_size, 1));
		}
	}
}

void Describe(VkImage image, FILE* out) {
	if (!Enabled() || image == VK_NULL_HANDLE) {
		return;
	}
	std::lock_guard lock(g_mutex);
	const auto      found = g_images.find(image);
	if (found == g_images.end() || found->second.empty()) {
		std::fprintf(out, "[image-history] %p: no record\n", static_cast<void*>(image));
		return;
	}
	const double now = NowMs();
	int          n   = 0;
	for (auto it = found->second.rbegin(); it != found->second.rend(); ++it, ++n) {
		const auto& g = *it;
		std::fprintf(out,
		             "[image-history] %p gen-%d: guest=0x%016" PRIx64 "+0x%" PRIx64 " %ux%u layers=%u "
		             "levels=%u format=%u created tick %" PRIu64 " (%.1f ms ago)%s",
		             static_cast<void*>(image), n, g.guest, g.guest_size, g.width, g.height, g.layers,
		             g.levels, g.format, g.created_tick, now - g.created_ms,
		             g.destroyed ? "" : ", LIVE\n");
		if (g.destroyed) {
			std::fprintf(out, ", destroyed tick %" PRIu64 " (%.1f ms ago)\n", g.destroyed_tick,
			             now - g.destroyed_ms);
		}
		std::fprintf(out,
		             "[image-history]   set-0 binds %" PRIu64 " (last tick %" PRIu64 " shader %016" PRIx64
		             "), attachment binds %" PRIu64 " (last tick %" PRIu64 " kind %c), storage last tick %" PRIu64
		             " shader %016" PRIx64 "\n",
		             g.descriptor_binds, g.last_descriptor_tick, g.last_descriptor_shader,
		             g.attachment_binds, g.last_attachment_tick,
		             g.last_attachment_kind != 0 ? g.last_attachment_kind : '-', g.last_storage_tick,
		             g.last_storage_shader);
		if (g.slots.empty()) {
			std::fprintf(out, "[image-history]   never in a bindless slot\n");
		}
		if (g.slots_dropped != 0) {
			std::fprintf(out, "[image-history]   (%u older slot events dropped)\n", g.slots_dropped);
		}
		for (const auto& e: g.slots) {
			char suffix[48] = "";
			if (e.what == 'R') {
				std::snprintf(suffix, sizeof(suffix), " by %p", static_cast<void*>(e.other));
			} else if (e.what == 'W' && e.other != VK_NULL_HANDLE) {
				std::snprintf(suffix, sizeof(suffix), " (was %p)", static_cast<void*>(e.other));
			}
			const char* what = e.what == 'W'   ? "view written into"
			                   : e.what == 'P' ? "placeholder replaced it in"
			                                   : "another image replaced it in";
			std::fprintf(out,
			             "[image-history]   tick %" PRIu64 " (%.1f ms ago): %s binding %u slot %u (heap 0x%" PRIx64
			             " key %u)%s\n",
			             e.tick, now - e.ms, what, e.binding, e.slot, e.heap, e.key, suffix);
		}
	}
}

} // namespace Libs::Graphics::ImageHistory
