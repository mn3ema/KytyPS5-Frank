#include "graphics/host_gpu/renderer/rtSnap.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics::RtSnap {

static void ProbeAtFlip(RenderContext& context);

namespace {

struct Config {
	uint64_t address = 0;
	uint32_t x = 0, y = 0, w = 0, h = 0;
};

const Config& GetConfig() {
	static const Config config = [] {
		Config      c;
		const char* value = std::getenv("KYTY_RT_SNAP");
		if (value == nullptr) {
			return c;
		}
		unsigned long long address = 0;
		unsigned           x = 0, y = 0, w = 0, h = 0;
		// Any one character between the numbers (',' splits the benchmark scripts' configs: use '+').
		if (std::sscanf(value, "%llx:%u%*c%u%*c%u%*c%u", &address, &x, &y, &w, &h) == 5 && w != 0 &&
		    h != 0) {
			c = {address, x, y, w, h};
		}
		return c;
	}();
	return config;
}

struct Slot {
	VkBuffer      buffer     = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void*         mapped     = nullptr;
	bool          busy       = false;
	uint64_t      tick       = 0;
	uint64_t      flip       = 0;
	uint32_t      stage      = 0;
	uint64_t      shader     = 0;
	char          kind       = '?';
	uint64_t      bytes      = 0;
};

struct State {
	std::vector<Slot> slots;
	uint64_t          flip          = 0;
	uint32_t          stage         = 0;
	bool              texture_taken = false;
	uint64_t          dropped       = 0;
};

State& GetState() {
	static State state;
	return state;
}

constexpr uint32_t SlotCount = 512;

} // namespace

bool Enabled() {
	return GetConfig().address != 0;
}

static void Snapshot(RenderContext& context, Image& image, ImageId id, uint64_t shader, char kind) {
	auto&       state  = GetState();
	const auto& config = GetConfig();
	auto&       graphics = context.GetGraphics();
	const auto  texel  = vk::blockSize(image.backing.format);
	const auto  bytes  = static_cast<uint64_t>(config.w) * config.h * texel;
	if (state.slots.empty()) {
		state.slots.resize(SlotCount);
		for (auto& slot: state.slots) {
			vk::BufferCreateInfo buffer {};
			buffer.size  = bytes;
			buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
			VmaAllocationCreateInfo allocate {};
			allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			allocate.flags =
			    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VmaAllocationInfo info {};
			const auto        raw = static_cast<VkBufferCreateInfo>(buffer);
			RequireVulkanSuccess(static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, &raw,
			                                                             &allocate, &slot.buffer,
			                                                             &slot.allocation, &info)),
			                     "allocate rt-snap buffer");
			slot.mapped = info.pMappedData;
			slot.bytes  = bytes;
		}
		std::fprintf(stderr, "[rt-snap] 0x%llx %u,%u %ux%u, %llu bytes a snapshot\n",
		             static_cast<unsigned long long>(config.address), config.x, config.y, config.w,
		             config.h, static_cast<unsigned long long>(bytes));
	}
	Slot* free_slot = nullptr;
	for (auto& slot: state.slots) {
		if (!slot.busy && slot.bytes >= bytes) {
			free_slot = &slot;
			break;
		}
	}
	if (free_slot == nullptr) {
		state.dropped++;
		return;
	}
	auto& scheduler = context.GetCommandScheduler();
	vk::BufferImageCopy region {};
	region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.imageOffset      = vk::Offset3D {static_cast<int32_t>(config.x), static_cast<int32_t>(config.y), 0};
	region.imageExtent      = vk::Extent3D {config.w, config.h, 1};
	image.Download(std::span(&region, 1), vk::Buffer(free_slot->buffer), 0, bytes);
	vk::MemoryBarrier to_host {};
	to_host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                               vk::PipelineStageFlagBits::eHost, {}, 1,
	                                               &to_host, 0, nullptr, 0, nullptr);
	free_slot->busy   = true;
	free_slot->tick   = scheduler.CurrentTick();
	free_slot->flip   = Sync::GuestFlipCount();
	free_slot->stage  = state.stage++;
	free_slot->shader = shader;
	free_slot->kind   = kind;
	(void)id;
}

void AfterCommand(RenderContext& context, const char* kind,
                  std::span<PreparedBindings* const> stages) {
	if (!Enabled()) {
		return;
	}
	auto&       state  = GetState();
	const auto& config = GetConfig();
	const auto  flip   = Sync::GuestFlipCount();
	if (flip != state.flip) {
		state.flip          = flip;
		state.stage         = 0;
		state.texture_taken = false;
	}
	auto& cache = context.GetTextureCache();
	for (const auto* prepared: stages) {
		if (prepared == nullptr || prepared->runtime == nullptr || prepared->runtime->program == nullptr) {
			continue;
		}
		for (const auto& binding: prepared->images) {
			auto* image = cache.TryGetImage(binding.image_id);
			if (image == nullptr || image->info.data.address != config.address ||
			    image->backing.samples != 1 ||
			    config.x + config.w > image->backing.extent.width ||
			    config.y + config.h > image->backing.extent.height || image->info.IsDepth()) {
				continue;
			}
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (!storage && state.texture_taken) {
				continue;
			}
			if (!storage) {
				state.texture_taken = true;
			}
			Snapshot(context, *image, binding.image_id, prepared->runtime->program->shader_hash,
			         storage ? kind[0] : 't');
			return; // one snapshot per command
		}
	}
}

static void BufferProbeAtFlip(RenderContext& context);

void AtFlip(RenderContext& context) {
	if (ProbeEnabled()) {
		ProbeAtFlip(context);
	}
	BufferProbeAtFlip(context);
	if (!Enabled()) {
		return;
	}
	auto& state     = GetState();
	auto& scheduler = context.GetCommandScheduler();
	auto& graphics  = context.GetGraphics();
	for (auto& slot: state.slots) {
		if (!slot.busy || !scheduler.IsFree(slot.tick)) {
			continue;
		}
		(void)vmaInvalidateAllocation(graphics.allocator, slot.allocation, 0, VK_WHOLE_SIZE);
		const auto hash = XXH3_64bits(slot.mapped, slot.bytes);
		std::fprintf(stderr, "[rt-snap] flip=%llu stage=%u kind=%c sh=%016llx h=%016llx\n",
		             static_cast<unsigned long long>(slot.flip), slot.stage, slot.kind,
		             static_cast<unsigned long long>(slot.shader),
		             static_cast<unsigned long long>(hash));
		slot.busy = false;
	}
	if (state.dropped != 0) {
		std::fprintf(stderr, "[rt-snap] %llu snapshots dropped (ring full)\n",
		             static_cast<unsigned long long>(state.dropped));
		state.dropped = 0;
	}
}

namespace {

struct ProbeConfig {
	uint32_t target_w = 0, target_h = 0;
	uint32_t x = 0, y = 0, w = 0, h = 0;
	const char* trigger = nullptr;
};

const ProbeConfig& GetProbeConfig() {
	static const ProbeConfig config = [] {
		ProbeConfig c;
		const char* value   = std::getenv("KYTY_PIXEL_PROBE");
		const char* trigger = std::getenv("KYTY_PIXEL_PROBE_TRIGGER");
		unsigned    tw = 0, th = 0, x = 0, y = 0, w = 0, h = 0;
		if (value != nullptr && trigger != nullptr &&
		    std::sscanf(value, "%u%*c%u%*c%u%*c%u%*c%u%*c%u", &tw, &th, &x, &y, &w, &h) == 6 && w != 0 &&
		    h != 0 && w <= 64 && h <= 64 && x + w <= tw && y + h <= th) {
			c = {tw, th, x, y, w, h, trigger};
		}
		return c;
	}();
	return config;
}

struct ProbeEntry {
	uint64_t offset = 0, bytes = 0, tick = 0, vs = 0, ps = 0;
	uint32_t draw = 0, texel = 0;
	int      attachment = 0; // 0..7 colour, -1 depth
	int      format     = 0;
	char     kind[16]   = {};
};

struct ProbeState {
	VkBuffer                buffer     = VK_NULL_HANDLE;
	VmaAllocation           allocation = nullptr;
	uint8_t*                mapped     = nullptr;
	uint64_t                capacity   = 0;
	uint64_t                used       = 0;
	uint64_t                armed_flip = UINT64_MAX; // the frame to probe
	bool                    done       = false;
	uint64_t                checked    = UINT64_MAX;
	uint32_t                draw       = 0;
	uint64_t                dropped    = 0;
	std::vector<ProbeEntry> entries;
};

ProbeState& GetProbeState() {
	static ProbeState state;
	return state;
}

} // namespace

bool ProbeEnabled() {
	return GetProbeConfig().trigger != nullptr;
}

static void ProbeImage(RenderContext& context, ProbeState& state, Image& image, int attachment,
                       uint64_t vs, uint64_t ps, const char* kind) {
	const auto& config = GetProbeConfig();
	const bool  depth  = attachment < 0;
	uint32_t    texel  = 0;
	if (depth) {
		switch (image.backing.format) {
			case vk::Format::eD16Unorm:
			case vk::Format::eD16UnormS8Uint: texel = 2; break;
			default: texel = 4; break;
		}
	} else {
		if (vk::blockExtent(image.backing.format)[0] != 1) {
			return;
		}
		texel = vk::blockSize(image.backing.format);
	}
	const uint64_t bytes = uint64_t {config.w} * config.h * texel;
	if (state.used + bytes > state.capacity) {
		state.dropped++;
		return;
	}
	vk::BufferImageCopy region {};
	region.bufferOffset     = state.used;
	region.imageSubresource = {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.imageOffset      = vk::Offset3D {static_cast<int32_t>(config.x), static_cast<int32_t>(config.y), 0};
	region.imageExtent      = vk::Extent3D {config.w, config.h, 1};
	image.Download(std::span(&region, 1), vk::Buffer(state.buffer), state.used, bytes);
	auto&             scheduler = context.GetCommandScheduler();
	vk::MemoryBarrier to_host {};
	to_host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                               vk::PipelineStageFlagBits::eHost, {}, 1, &to_host,
	                                               0, nullptr, 0, nullptr);
	ProbeEntry entry;
	entry.offset     = state.used;
	entry.bytes      = bytes;
	entry.tick       = scheduler.CurrentTick();
	entry.vs         = vs;
	entry.ps         = ps;
	entry.draw       = state.draw;
	entry.texel      = texel;
	entry.attachment = attachment;
	entry.format     = static_cast<int>(image.backing.format);
	std::snprintf(entry.kind, sizeof(entry.kind), "%s", kind);
	state.entries.push_back(entry);
	state.used += (bytes + 15u) & ~uint64_t {15};
}

void ProbeAfterDraw(RenderContext& context, std::span<const ImageId> colors, ImageId depth,
                    uint64_t vs, uint64_t ps, const char* kind) {
	if (!ProbeEnabled()) {
		return;
	}
	auto&       state  = GetProbeState();
	const auto& config = GetProbeConfig();
	if (state.done) {
		return;
	}
	const auto flip = Sync::GuestFlipCount();
	if (state.armed_flip == UINT64_MAX) {
		if (state.checked == flip) {
			return;
		}
		state.checked = flip;
		std::error_code error;
		if (!std::filesystem::exists(config.trigger, error)) {
			return;
		}
		state.armed_flip = flip + 1;
		std::fprintf(stderr, "[pixel-probe] armed for guest flip %llu: %ux%u targets, region %u,%u %ux%u\n",
		             static_cast<unsigned long long>(state.armed_flip), config.target_w,
		             config.target_h, config.x, config.y, config.w, config.h);
		return;
	}
	if (flip < state.armed_flip) {
		return;
	}
	if (flip > state.armed_flip) {
		state.done = true;
		std::fprintf(stderr, "[pixel-probe] frame done: %u draws, %zu copies, %llu dropped\n", state.draw,
		             state.entries.size(), static_cast<unsigned long long>(state.dropped));
		return;
	}
	if (state.buffer == VK_NULL_HANDLE) {
		auto&                graphics = context.GetGraphics();
		vk::BufferCreateInfo buffer {};
		state.capacity = uint64_t {64} << 20u;
		buffer.size    = state.capacity;
		buffer.usage   = vk::BufferUsageFlagBits::eTransferDst;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo info {};
		const auto        raw = static_cast<VkBufferCreateInfo>(buffer);
		RequireVulkanSuccess(static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, &raw, &allocate,
		                                                             &state.buffer, &state.allocation, &info)),
		                     "allocate pixel-probe buffer");
		state.mapped = static_cast<uint8_t*>(info.pMappedData);
	}
	auto& cache = context.GetTextureCache();
	auto  fits  = [&](const Image& image) {
        return image.backing.samples == 1 && image.backing.extent.width == config.target_w &&
               image.backing.extent.height == config.target_h;
	};
	bool any = false;
	for (size_t i = 0; i < colors.size(); i++) {
		auto* image = cache.TryGetImage(colors[i]);
		if (image != nullptr && !image->info.IsDepth() && fits(*image)) {
			ProbeImage(context, state, *image, static_cast<int>(i), vs, ps, kind);
			any = true;
		}
	}
	if (auto* image = depth ? cache.TryGetImage(depth) : nullptr; image != nullptr && fits(*image)) {
		ProbeImage(context, state, *image, -1, vs, ps, kind);
		any = true;
	}
	if (any) {
		state.draw++;
	}
}

static void ProbeAtFlip(RenderContext& context) {
	auto& state = GetProbeState();
	if (!state.done || state.entries.empty()) {
		return;
	}
	auto& scheduler = context.GetCommandScheduler();
	for (const auto& entry: state.entries) {
		if (!scheduler.IsFree(entry.tick)) {
			return;
		}
	}
	(void)vmaInvalidateAllocation(context.GetGraphics().allocator, state.allocation, 0, VK_WHOLE_SIZE);
	const auto&                       config = GetProbeConfig();
	std::unordered_map<int, uint64_t> last;
	const uint64_t centre = (uint64_t {config.h / 2} * config.w + config.w / 2);
	for (const auto& entry: state.entries) {
		const auto* data = state.mapped + entry.offset;
		const auto  hash = XXH3_64bits(data, entry.bytes);
		const auto  it   = last.find(entry.attachment);
		const bool  first   = it == last.end();
		const bool  changed = !first && it->second != hash;
		last[entry.attachment] = hash;
		if (!first && !changed) {
			continue;
		}
		std::string bytes;
		for (uint32_t k = 0; k < entry.texel; k++) {
			bytes += fmt::format("{:02x}", data[centre * entry.texel + k]);
		}
		std::string value;
		if (entry.attachment < 0 && entry.texel == 4) {
			float f = 0;
			std::memcpy(&f, data + centre * 4, 4);
			value = fmt::format(" depth={:.7f}", f);
		}
		std::fprintf(stderr, "[pixel-probe] d=%u %s vs=%016llx ps=%016llx att=%s%d fmt=%d centre=%s%s h=%016llx%s\n",
		             entry.draw, entry.kind, static_cast<unsigned long long>(entry.vs),
		             static_cast<unsigned long long>(entry.ps), entry.attachment < 0 ? "Z" : "C",
		             entry.attachment < 0 ? 0 : entry.attachment, entry.format, bytes.c_str(),
		             value.c_str(), static_cast<unsigned long long>(hash), first ? " (first)" : "");
	}
	std::fflush(stderr);
	state.entries.clear();
}


namespace {

struct BufferProbeTarget {
	uint64_t                                   hash = 0;
	bool                                       all  = false;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
};

struct BufferProbeConfig {
	std::vector<BufferProbeTarget> targets;
	const char*                    trigger    = nullptr;
	std::string                    dir;
	uint32_t                       dispatches = 4;
	uint64_t                       each       = uint64_t {64} << 20u;
	uint64_t                       skip       = 0; // "all" leaves out buffers larger than this
	uint64_t                       capacity   = uint64_t {768} << 20u;
};

const BufferProbeConfig& GetBufferProbeConfig() {
	static const BufferProbeConfig config = [] {
		BufferProbeConfig c;
		const char*       value   = std::getenv("KYTY_BUFFER_PROBE");
		const char*       trigger = std::getenv("KYTY_BUFFER_PROBE_TRIGGER");
		const char*       dir     = std::getenv("KYTY_BUFFER_PROBE_DIR");
		if (value == nullptr || trigger == nullptr || dir == nullptr) {
			return c;
		}
		// <target>[/<target>...], each <hash>:<range>[,<range>...].
		const char* cursor = value;
		while (cursor != nullptr && *cursor != '\0') {
			BufferProbeTarget target;
			char*             end = nullptr;
			target.hash           = std::strtoull(cursor, &end, 16);
			while (end != nullptr && (*end == ':' || *end == ',')) {
				const char* item = end + 1;
				if (std::strncmp(item, "all", 3) == 0) {
					target.all = true;
					end        = const_cast<char*>(item + 3);
					continue;
				}
				const auto address = std::strtoull(item, &end, 16);
				uint64_t   bytes   = 4;
				if (end != nullptr && *end == '+') {
					bytes = std::strtoull(end + 1, &end, 16);
				}
				if (address != 0 && bytes != 0) {
					target.ranges.emplace_back(address, bytes);
				}
			}
			c.targets.push_back(std::move(target));
			cursor = end != nullptr && *end == '/' ? end + 1 : nullptr;
		}
		if (const char* n = std::getenv("KYTY_BUFFER_PROBE_DISPATCHES"); n != nullptr) {
			c.dispatches = static_cast<uint32_t>(std::strtoul(n, nullptr, 10));
		}
		if (const char* mb = std::getenv("KYTY_BUFFER_PROBE_EACH_MB"); mb != nullptr) {
			c.each = std::strtoull(mb, nullptr, 10) << 20u;
		}
		if (const char* mb = std::getenv("KYTY_BUFFER_PROBE_SKIP_MB"); mb != nullptr) {
			c.skip = std::strtoull(mb, nullptr, 10) << 20u;
		}
		c.trigger = trigger;
		c.dir     = dir;
		return c;
	}();
	return config;
}

struct BufferProbeEntry {
	uint64_t offset = 0, bytes = 0, address = 0, tick = 0, hash = 0;
	uint32_t dispatch = 0;
};

struct BufferProbeState {
	VkBuffer                      buffer     = VK_NULL_HANDLE;
	VmaAllocation                 allocation = nullptr;
	uint8_t*                      mapped     = nullptr;
	uint64_t                      used       = 0;
	uint64_t                      armed_flip = UINT64_MAX;
	uint64_t                      checked    = UINT64_MAX;
	uint32_t                      dispatches = 0;
	bool                          done       = false;
	bool                          written    = false;
	bool                          second     = false; // KYTY_BUFFER_PROBE_RANGES_FILE taken
	std::vector<BufferProbeEntry> entries;
};

BufferProbeState& GetBufferProbeState() {
	static BufferProbeState state;
	return state;
}

} // namespace

static void BufferProbeCapture(RenderContext& context, BufferProbeState& state, uint32_t dispatch,
                               uint64_t hash, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	const auto& config = GetBufferProbeConfig();
	auto&             cache     = context.GetBufferCache();
	auto&             scheduler = context.GetCommandScheduler();
	auto              recorder  = scheduler.Current().Recorder();
	vk::MemoryBarrier to_copy {};
	to_copy.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	to_copy.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                         {}, 1, &to_copy, 0, nullptr, 0, nullptr);
	for (const auto& [address, wanted]: ranges) {
		uint64_t bytes = wanted & ~uint64_t {3};
		if (bytes == 0 || state.used + bytes > config.capacity) {
			std::fprintf(stderr, "[buffer-probe] dispatch %u 0x%llx+0x%llx dropped (capacity)\n", dispatch,
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(wanted));
			continue;
		}
		const auto id = cache.PeekBuffer(address, bytes);
		if (!id) {
			std::fprintf(stderr, "[buffer-probe] dispatch %u 0x%llx+0x%llx not in one cached buffer\n", dispatch,
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
			continue;
		}
		auto& source = cache.GetBuffer(id);
		if (!source.Handle()) {
			std::fprintf(stderr, "[buffer-probe] dispatch %u 0x%llx: buffer has no device handle\n", dispatch,
			             static_cast<unsigned long long>(address));
			continue;
		}
		vk::BufferCopy copy {source.Offset(address), state.used, bytes};
		recorder.copyBuffer(source.Handle(), vk::Buffer(state.buffer), 1, &copy);
		state.entries.push_back({state.used, bytes, address, scheduler.CurrentTick(), hash, dispatch});
		state.used += (bytes + 255u) & ~uint64_t {255};
	}
	vk::MemoryBarrier to_host {};
	to_host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eShaderRead |
	                        vk::AccessFlagBits::eShaderWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                         vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eAllCommands, {}, 1,
	                         &to_host, 0, nullptr, 0, nullptr);
}

void BufferProbeAfterDispatch(RenderContext& context, uint64_t hash, const PreparedBindings& bindings,
                              const uint32_t* user_data, uint32_t user_count) {
	const auto& config = GetBufferProbeConfig();
	if (config.trigger == nullptr) {
		return;
	}
	const BufferProbeTarget* target = nullptr;
	for (const auto& candidate: config.targets) {
		if (candidate.hash == hash) target = &candidate;
	}
	if (target == nullptr) {
		return;
	}
	auto& state = GetBufferProbeState();
	// KYTY_BUFFER_PROBE_RANGES_FILE=<file>: after the first capture was written, the next dispatch
	// once the file exists reads back the ranges it lists ("<hex address>+<hex bytes>" per line),
	// e.g. ones computed from the first capture; files bp_99_*.
	if (state.done) {
		static const char* ranges_file = std::getenv("KYTY_BUFFER_PROBE_RANGES_FILE");
		std::error_code    error;
		if (ranges_file == nullptr || !state.written || state.second ||
		    !std::filesystem::exists(ranges_file, error)) {
			return;
		}
		std::vector<std::pair<uint64_t, uint64_t>> listed;
		if (std::FILE* file = std::fopen(ranges_file, "r"); file != nullptr) {
			unsigned long long address = 0, bytes = 0;
			while (std::fscanf(file, " %llx+%llx", &address, &bytes) == 2) {
				listed.emplace_back(address, bytes);
			}
			std::fclose(file);
		}
		if (listed.empty()) {
			return; // still being written
		}
		state.second  = true;
		state.written = false;
		state.entries.clear();
		state.used = 0;
		BufferProbeCapture(context, state, 99, hash, listed);
		std::fprintf(stderr, "[buffer-probe] second capture: %zu ranges\n", listed.size());
		return;
	}
	const auto flip = Sync::GuestFlipCount();
	if (state.armed_flip == UINT64_MAX) {
		if (state.checked == flip) {
			return;
		}
		state.checked = flip;
		std::error_code error;
		if (!std::filesystem::exists(config.trigger, error)) {
			return;
		}
		// The next dispatches, whatever flip they belong to: the guest flip count runs ahead of the
		// GPU thread by a variable amount, so a once-or-twice-a-frame shader can miss a flip window.
		state.armed_flip = flip;
		std::fprintf(stderr, "[buffer-probe] armed at guest flip %llu\n", static_cast<unsigned long long>(flip));
	}
	if (state.dispatches >= config.dispatches) {
		state.done = true;
		return;
	}
	auto& graphics = context.GetGraphics();
	if (state.buffer == VK_NULL_HANDLE) {
		vk::BufferCreateInfo buffer {};
		buffer.size  = config.capacity;
		buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo info {};
		const auto        raw = static_cast<VkBufferCreateInfo>(buffer);
		RequireVulkanSuccess(static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, &raw, &allocate,
		                                                             &state.buffer, &state.allocation, &info)),
		                     "allocate buffer-probe buffer");
		state.mapped = static_cast<uint8_t*>(info.pMappedData);
	}
	const auto dispatch = state.dispatches++;
	if (state.dispatches >= config.dispatches) {
		state.done = true;
	}
	{
		std::string line = fmt::format("[buffer-probe] dispatch {} {:016x} user:", dispatch, hash);
		for (uint32_t i = 0; i < user_count && i < 16u; i++) line += fmt::format(" {:08x}", user_data[i]);
		for (size_t i = 0; i < bindings.buffer_sources.size(); i++) {
			line += fmt::format(" | b{} {:x}+{:x}", i, bindings.buffer_sources[i].address,
			                    bindings.buffer_sources[i].size);
		}
		std::fprintf(stderr, "%s\n", line.c_str());
	}
	std::vector<std::pair<uint64_t, uint64_t>> ranges = target->ranges;
	if (target->all) {
		for (const auto& source: bindings.buffer_sources) {
			if (source.address != 0 && source.size != 0 && (config.skip == 0 || source.size <= config.skip)) {
				ranges.emplace_back(source.address, std::min(source.size, config.each));
			}
		}
	}
	BufferProbeCapture(context, state, dispatch, hash, ranges);
}

static void BufferProbeAtFlip(RenderContext& context) {
	auto& state = GetBufferProbeState();
	if (!state.done || state.written || state.entries.empty()) {
		return;
	}
	auto& scheduler = context.GetCommandScheduler();
	for (const auto& entry: state.entries) {
		if (!scheduler.IsFree(entry.tick)) {
			return;
		}
	}
	(void)vmaInvalidateAllocation(context.GetGraphics().allocator, state.allocation, 0, VK_WHOLE_SIZE);
	const auto& config = GetBufferProbeConfig();
	std::error_code error;
	std::filesystem::create_directories(config.dir, error);
	for (const auto& entry: state.entries) {
		const auto path = fmt::format("{}/bp_{:02}_{:016x}_{:x}.bin", config.dir, entry.dispatch, entry.hash,
		                              entry.address);
		if (std::FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
			std::fwrite(state.mapped + entry.offset, 1, entry.bytes, file);
			std::fclose(file);
		}
		std::fprintf(stderr, "[buffer-probe] wrote dispatch %u 0x%llx+0x%llx xxh=%016llx\n", entry.dispatch,
		             static_cast<unsigned long long>(entry.address), static_cast<unsigned long long>(entry.bytes),
		             static_cast<unsigned long long>(XXH3_64bits(state.mapped + entry.offset, entry.bytes)));
	}
	std::fflush(stderr);
	state.written = true;
}

} // namespace Libs::Graphics::RtSnap
