#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "common/frameSkips.h"
#include "graphics/host_gpu/renderer/htileWatch.h"
#include "graphics/host_gpu/renderer/imgTrace.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/renderer/renderStats.h"

#include "graphics/host_gpu/renderer/image/imageHistory.h"
#include "graphics/host_gpu/timeline.h"
#include "common/floorStats.h"
#include "common/fpsLog.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <chrono>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <span>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

// image.cpp (KYTY_FAULT_MAP=1): false, and a report, for a view of a destroyed image.
bool CheckLiveImageView(vk::ImageView view, const char* where);

// Diagnostics (KYTY_KEY_CAPTURE): the buffers the captured shader binds, as (base, bytes).
static std::mutex                                 g_capture_buffers_mutex;
static std::vector<std::pair<uint64_t, uint64_t>> g_capture_buffers;

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

struct WatchedReread {
	uint64_t frame   = 0;
	uint64_t address = 0;
	uint64_t size    = 0;
};

std::mutex                 g_watched_reread_mutex;
std::vector<WatchedReread> g_watched_rereads;

void LogWords(const uint32_t* words, size_t count) {
	for (size_t row = 0; row < count; row += 8) {
		std::string line;
		for (size_t k = row; k < std::min(count, row + 8); k++) {
			line += fmt::format(" {:08x}", words[k]);
		}
		LOGF("    +0x%03zx:%s\n", row * 4, line.c_str());
	}
}

} // namespace

static uint64_t WatchedShaderHash() {
	static const uint64_t hash = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	return hash;
}

static std::atomic<uint64_t> g_watched_first_frame {UINT64_MAX};

uint64_t WatchedShaderFirstFrame() {
	return g_watched_first_frame.load();
}

bool InWatchedShaderFrame(uint64_t hash, uint64_t presented_frame, std::atomic<uint32_t>& logged) {
	static const uint64_t delta = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER_FRAME");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : UINT64_MAX;
	}();
	if (hash == 0 || hash != WatchedShaderHash()) {
		return false;
	}
	uint64_t unset = UINT64_MAX;
	g_watched_first_frame.compare_exchange_strong(unset, presented_frame);
	return delta != UINT64_MAX && presented_frame == g_watched_first_frame.load() + delta &&
	       logged.fetch_add(1) < 64u;
}

void LogWatchedRereads(BufferCache& cache, uint64_t presented_frames) {
	std::vector<WatchedReread> due;
	{
		std::lock_guard lock(g_watched_reread_mutex);
		for (auto it = g_watched_rereads.begin(); it != g_watched_rereads.end();) {
			if (it->frame <= presented_frames) {
				due.push_back(*it);
				it = g_watched_rereads.erase(it);
			} else {
				++it;
			}
		}
	}
	for (const auto& reread: due) {
		std::vector<uint32_t> words(reread.size / 4u);
		const bool gpu_written = cache.HasGpuDirtyBytes(reread.address, reread.size);
		const bool read =
		    Libs::LibKernel::Memory::TryReadBacking(reread.address, words.data(), reread.size);
		LOGF("WatchShader reread at frame %" PRIu64 ": 0x%012" PRIx64 "+0x%" PRIx64
		     " gpu_written=%d%s\n",
		     presented_frames, reread.address, reread.size, gpu_written ? 1 : 0,
		     read ? "" : " (unreadable)");
		if (read) {
			LogWords(words.data(), words.size());
		}
	}
}

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::ShaderData:
			return ShaderRecompiler::IR::ShaderDataUniform() ? vk::DescriptorType::eUniformBuffer
			                                                  : vk::DescriptorType::eStorageBuffer;
		case BindingKind::FlattenedSrt:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (texture.dynamic_class) {
		// The float group's occurrence comes first (binding kinds in order), then the uint group's.
		EXIT_IF(element > 1u);
		const bool real = (element == 1u) == texture.real_uint;
		const auto view = real ? texture.image_view : texture.alt_view;
		const auto layout = real ? texture.layout : texture.alt_layout;
		EXIT_IF(!texture.image_id || !texture.alt_image_id || view == nullptr ||
		        layout == vk::ImageLayout::eUndefined);
		(void)CheckLiveImageView(view, "a draw/dispatch descriptor");
		return {nullptr, view, layout};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	(void)CheckLiveImageView(view, "a draw/dispatch descriptor");
	return {nullptr, view, texture.layout};
}

static const char* ShaderStageResourceName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "Vertex";
		case ShaderType::Mesh: return "Mesh";
		case ShaderType::Local: return "Local";
		case ShaderType::TessellationControl: return "Hull";
		case ShaderType::TessellationEvaluation: return "Domain";
		case ShaderType::Pixel: return "Pixel";
		case ShaderType::Compute: return "Compute";
		default: return "Unknown";
	}
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// A/B switch for bounded buffer writes (KYTY_BOUNDED_BUFFER_WRITES=0 marks every written buffer's
// whole range GPU-written, as before).
static const bool g_bounded_buffer_writes = [] {
	const char* value = std::getenv("KYTY_BOUNDED_BUFFER_WRITES");
	return value == nullptr || value[0] != '0';
}();

// Diagnostics: names the shader whose bindings are prepared, for KYTY_WATCH_GPU_WRITE.
struct DiagShaderScope {
	explicit DiagShaderScope(uint64_t hash, ShaderType stage = ShaderType::Unknown) {
		BufferCache::s_diag_shader_hash  = hash;
		BufferCache::s_diag_shader_stage = static_cast<uint8_t>(stage);
	}
	~DiagShaderScope() {
		BufferCache::s_diag_shader_hash  = 0;
		BufferCache::s_diag_shader_stage = 0;
	}
	KYTY_CLASS_NO_COPY(DiagShaderScope);
};

// KYTY_WRITE_BITMAP_MIN_KB (default 1024), as BufferCache's.
static uint64_t WriteBitmapMinBytesForBinding() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP_MIN_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {1024}) * 1024u;
	}();
	return bytes;
}

// KYTY_MEMO_STATS=1 (diagnostics, no behavior change): buffer obtains that repeat an earlier one of
// the same range and kind in the same recording (the hit ceiling of a per-submission obtain memo),
// and how long obtains take, first vs repeat. [obtain-memo] every 5 s.
static void NoteObtain(uint64_t tick, uint64_t address, uint64_t size, uint32_t kind, uint64_t ns) {
	static const bool enabled = std::getenv("KYTY_MEMO_STATS") != nullptr;
	if (!enabled) {
		return;
	}
	struct State {
		uint64_t                     tick = UINT64_MAX;
		std::unordered_set<uint64_t> seen;
		uint64_t calls[3] {}, repeats[3] {}, ns_first[3] {}, ns_repeat[3] {}, ticks = 0;
		std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	};
	static State st;
	if (tick != st.tick) {
		st.tick = tick;
		st.seen.clear();
		st.ticks++;
	}
	const uint64_t words[3] = {address, size, kind};
	const auto     key      = XXH3_64bits(words, sizeof(words));
	st.calls[kind]++;
	if (!st.seen.insert(key).second) {
		st.repeats[kind]++;
		st.ns_repeat[kind] += ns;
	} else {
		st.ns_first[kind] += ns;
	}
	if (const auto now = std::chrono::steady_clock::now(); now - st.last > std::chrono::seconds(5)) {
		static const char* names[3] = {"read", "written", "tracked"};
		std::string text = fmt::format("[obtain-memo] 5 s, {} recordings:", st.ticks);
		for (uint32_t k = 0; k < 3; k++) {
			const auto first = st.calls[k] - st.repeats[k];
			text += fmt::format(" | {} {}/s repeat {:.1f} % (first {:.2f} us, repeat {:.2f} us; {:.1f} ms/s in repeats)",
			                    names[k], st.calls[k] / 5, st.calls[k] ? 100.0 * double(st.repeats[k]) / double(st.calls[k]) : 0.0,
			                    first ? double(st.ns_first[k]) / double(first) / 1000.0 : 0.0,
			                    st.repeats[k] ? double(st.ns_repeat[k]) / double(st.repeats[k]) / 1000.0 : 0.0,
			                    double(st.ns_repeat[k]) / 5e6);
			st.calls[k] = st.repeats[k] = st.ns_first[k] = st.ns_repeat[k] = 0;
		}
		std::fprintf(stderr, "%s\n", text.c_str());
		st.ticks = 0;
		st.last  = now;
	}
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource&    resource,
                    const ShaderRecompiler::IR::BufferWriteExtent* extent, uint32_t& buffer_offset,
                    bool track_writes, uint64_t& tracked_base) {
	buffer_offset = 0;
	tracked_base  = 0;

	const auto& [address, size, id] = source;
	if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
		// With device bounds checks the shader no longer guards these: a null descriptor
		// (nullDescriptor) reads zero and drops writes, so the shared fallback stays untouched.
		if (ShaderRecompiler::Spirv::GetRobustBufferAccess()) {
			return {nullptr, 0, VK_WHOLE_SIZE};
		}
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	if (SyncToggle::ForceUpload() && !resource.written && size <= (16u << 20u)) {
		auto& cache = context.GetBufferCache();
		if (!cache.IsRegionGpuModified(address, size)) {
			cache.MarkRegionCpuModified(address, size);
		}
	}
	// The stores' addresses bound the bytes this draw or dispatch writes: only those become
	// GPU-written, so the guest's accesses elsewhere in the range do not drain the GPU.
	const bool bounded_write = resource.written && extent != nullptr && extent->valid &&
	                           g_bounded_buffer_writes;
	const auto written_begin = bounded_write ? std::min(extent->begin, size) : 0u;
	const auto written_end   = bounded_write ? std::min(extent->end, size) : size;
	// Unbounded writes through a large range: the shader marks the pages it writes in the GPU
	// write bitmap, and only those are downloaded (BufferCache::ObtainBufferTracked).
	const bool tracked = track_writes && resource.written && !bounded_write &&
	                     size >= WriteBitmapMinBytesForBinding() &&
	                     address + size <= (uint64_t {1} << 40u) &&
	                     context.GetBufferCache().WriteBitmapAddress() != 0;
	const CpuCost::Phase obtain_phase_(bounded_write ? "Rebind::ObtainWritten"
	                                   : tracked   ? "Rebind::ObtainTracked"
	                                               : "Rebind::Obtain");
	// The obtain is timed only for KYTY_MEMO_STATS (two clock reads per binding, ~600 k a second).
	static const bool memo_stats = std::getenv("KYTY_MEMO_STATS") != nullptr;
	const bool        time_obtain = memo_stats || !SyncToggle::HotPath();
	const auto        obtain_start =
	    time_obtain ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	auto [buffer, offset] =
	    bounded_write
	        ? context.GetBufferCache().ObtainBufferWritten(address, size, address + written_begin,
	                                                       written_end - std::min(written_begin,
	                                                                              written_end),
	                                                       id)
	    : tracked ? context.GetBufferCache().ObtainBufferTracked(address, size, id)
	              : context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                      resource.formatted, id);
	if (time_obtain) {
		NoteObtain(context.GetCommandScheduler().CurrentTick(), address, size,
		           bounded_write || resource.written ? (tracked ? 2u : 1u) : 0u,
		           static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                     std::chrono::steady_clock::now() - obtain_start)
		                                     .count()));
	}
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	if (tracked) {
		tracked_base = address - adjustment; // the guest address of the bound range's first byte
	}
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written && written_end > written_begin) {
		const CpuCost::Phase cpu_phase_("Rebind::InvalidateTextures");
		context.GetTextureCache().InvalidateMemoryFromGPU(address + written_begin,
		                                                  written_end - written_begin);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	const bool encoding_ok = IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	// A storage read returns the view's channels and 0, 0, 1 for the ones its format lacks, which
	// is also what a T# selecting only the format's own channels in order asks for (an R16 float
	// image read as R, 0, 0, 1).
	const auto channels = vk::componentCount(VulkanFormat(descriptor.Format()));
	const bool natural_swizzle =
	    channels >= 1u && channels <= 4u &&
	    swizzle == DstSel(4, channels >= 2u ? 5 : 0, channels >= 3u ? 6 : 0, channels >= 4u ? 7 : 1);
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || natural_swizzle || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

static bool IsSupportedStorageTextureFormat(const ShaderRecompiler::IR::ImageResource& resource,
                                            const ShaderTextureResource& descriptor) {
	const auto format           = descriptor.Format();
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const bool raw_atomic_storage = resource.atomic && uint_resource &&
	                                (format == Prospero::BufferFormat::k32UInt ||
	                                 format == Prospero::BufferFormat::k32SInt ||
	                                 format == Prospero::BufferFormat::k32Float);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	return raw_sint_storage || raw_float_atomic || raw_atomic_storage ||
	       (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	        numeric_class != Prospero::TextureNumericClass::Sint &&
	        uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	        (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                          : Prospero::BufferFormat::k32UInt)));
}

// What ValidateStorageTexture accepts, without exiting: a bindless storage key whose T# the
// storage path cannot bind keeps the placeholder.
static bool IsUsableStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                   const ShaderTextureResource&               descriptor) {
	return IsSupportedStorageImageResource(resource) &&
	       IsSupportedStorageTextureDescriptor(resource, descriptor) &&
	       IsSupportedStorageTextureEncoding(resource, descriptor) &&
	       IsSupportedStorageTextureFormat(resource, descriptor);
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const bool raw_atomic_storage = resource.atomic && uint_resource &&
	                                (format == Prospero::BufferFormat::k32UInt ||
	                                 format == Prospero::BufferFormat::k32SInt ||
	                                 format == Prospero::BufferFormat::k32Float);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic || raw_atomic_storage ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	// The view must match the dimension the shader declares (Vulkan requires the view type to
	// match the OpTypeImage), so 1D and 3D slots get 1D and 3D null images.
	auto image_type = Prospero::ImageType::kColor2D;
	auto view_type  = vk::ImageViewType::e2D;
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			image_type = Prospero::ImageType::kColor1D;
			view_type  = vk::ImageViewType::e1D;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			image_type = Prospero::ImageType::kColor1D;
			view_type  = vk::ImageViewType::e1DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view_type = vk::ImageViewType::e2DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			image_type = Prospero::ImageType::kColor3D;
			view_type  = vk::ImageViewType::e3D;
			break;
		default: break;
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = image_type;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = view_type;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	// KYTY_IGNORE_MIN_LOD=1: diagnostic, sample every mip regardless of the T#'s MIN_LOD clamp.
	static const bool ignore_min_lod = std::getenv("KYTY_IGNORE_MIN_LOD") != nullptr;
	if (!ignore_min_lod && descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			// v418: a 2D T# read by a 2D-array shader slot (ResourceMaterialization, 2D as array)
			// is exactly its own layer, also when it aliases one slice of an array image.
			view.layer_count = descriptor.Type() == Prospero::ImageType::kColor2D
			                       ? 1u
			                       : image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// ResolveTexture's description of a texture (tiling, sizes, mip layout, view) depends only on the
// T# and on how the shader uses the image: it was computed again for every bound image of every
// draw, 2.4 % of the GPU thread at the jungle prompt (DEBUGGING.md, 2026-10-05). The texture
// cache lookup that follows still runs every time. KYTY_TEXTURE_DESC_CACHE=0 turns this off.
struct TextureDescKey {
	std::array<uint32_t, 8> dwords {};
	uint32_t                resource_class = 0;
	uint32_t                numeric_class  = 0;
	uint32_t                dimension      = 0;
	uint32_t                mip_mode       = 0;
	uint32_t                mip_count      = 0;
	uint32_t                conversion     = 0;
	uint32_t                swizzle        = 0;
	uint32_t                flags          = 0;

	bool operator==(const TextureDescKey&) const = default;
};
static_assert(sizeof(TextureDescKey) == 64);

struct TextureDescKeyHash {
	size_t operator()(const TextureDescKey& key) const noexcept {
		return static_cast<size_t>(XXH3_64bits(&key, sizeof(key)));
	}
};

struct TextureDescEntry {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint32_t                size              = 0;
	bool                    shader_conversion = false;
};

TextureDescKey MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
                                  const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureDescKey key;
	std::copy_n(value.dwords.begin(), 8, key.dwords.begin());
	key.resource_class = static_cast<uint32_t>(resource.resource_class);
	key.numeric_class  = static_cast<uint32_t>(resource.numeric_class);
	key.dimension      = static_cast<uint32_t>(resource.dimension);
	key.mip_mode       = static_cast<uint32_t>(resource.mip_mode);
	key.mip_count      = resource.mip_count;
	key.conversion     = static_cast<uint32_t>(resource.conversion_format);
	key.swizzle        = resource.shader_swizzle;
	key.flags = (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) |
	            (resource.atomic ? 4u : 0u) | (resource.depth_compare ? 8u : 0u) |
	            (resource.cube ? 16u : 0u) | (resource.r128 ? 32u : 0u) |
	            (resource.bindless ? 64u : 0u) | (value.dword_count << 8u);
	return key;
}

bool TextureDescCacheEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TEXTURE_DESC_CACHE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

} // namespace

TextureBinding RenderExecutor::ResolveStageTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                                   const ShaderRecompiler::IR::DescriptorValue& value) {
	if (resource.dynamic_3d) {
		// v433: the T#'s view in the group of its type (2D or 3D), a null image in the other.
		const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
		const bool real_2d    = !descriptor.IsNull() && descriptor.Type() == Prospero::ImageType::kColor2D;
		auto       real       = resource;
		real.dimension = real_2d ? ShaderRecompiler::Decoder::ImageDimension::Dim2D
		                         : ShaderRecompiler::Decoder::ImageDimension::Dim3D;
		auto binding    = ResolveTexture(real, value);
		auto other      = resource;
		other.dimension = real_2d ? ShaderRecompiler::Decoder::ImageDimension::Dim3D
		                          : ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		ShaderRecompiler::IR::DescriptorValue null_value;
		null_value.dword_count = value.dword_count;
		auto null_binding      = ResolveTexture(other, null_value);
		binding.dynamic_class  = true;
		binding.real_uint      = !real_2d;
		binding.alt_image_id   = null_binding.image_id;
		binding.alt_desc       = std::move(null_binding.desc);
		return binding;
	}
	if (!resource.dynamic_class) {
		return ResolveTexture(resource, value);
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool real_uint  = !descriptor.IsNull() &&
	                       Prospero::SampledTextureNumericClass(descriptor.Format()) ==
	                           Prospero::TextureNumericClass::Uint;
	auto real          = resource;
	real.numeric_class = real_uint ? Prospero::TextureNumericClass::Uint : Prospero::TextureNumericClass::Float;
	auto binding       = ResolveTexture(real, value);
	auto other          = resource;
	other.numeric_class = real_uint ? Prospero::TextureNumericClass::Float : Prospero::TextureNumericClass::Uint;
	ShaderRecompiler::IR::DescriptorValue null_value; // all zero: a null T# of the same size
	null_value.dword_count = value.dword_count;
	auto null_binding     = ResolveTexture(other, null_value);
	binding.dynamic_class = true;
	binding.real_uint     = real_uint;
	binding.alt_image_id  = null_binding.image_id;
	binding.alt_desc      = std::move(null_binding.desc);
	return binding;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
	const CpuCost::Phase cpu_phase_("ResolveTexture");
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}

	thread_local std::unordered_map<TextureDescKey, TextureDescEntry, TextureDescKeyHash> cache;
	const bool use_cache = TextureDescCacheEnabled();
	const auto key       = use_cache ? MakeTextureDescKey(resource, value) : TextureDescKey {};
	if (use_cache) {
		if (const auto it = cache.find(key); it != cache.end()) {
			return FindResolvedTexture(resource, descriptor, it->second.desc,
			                           it->second.shader_conversion, it->second.pixel_format,
			                           it->second.view_format, it->second.size);
		}
	}

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		if (!ResolveTextureMipView(physical, !resource.r128 && descriptor.MetaCompress(),
		                           view_levels, levels, view_base)) {
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			EXIT("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers,
		                        physical_levels, tile, volume, size);
	}
	EXIT_NOT_IMPLEMENTED(size.size == 0 || size.align == 0 ||
	                     (address & (static_cast<uint64_t>(size.align) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	// Raw sint stores and all image atomics run as uint operations on the texel bits
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && (resource.atomic ||
	                                               format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth &&
	    !desc.info.IsDepth()) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;

	if (use_cache) {
		if (cache.size() >= 65536) {
			cache.clear();
		}
		cache.emplace(key, TextureDescEntry {desc, pixel_format, view_format, size.size,
		                                     shader_conversion});
	}
	return FindResolvedTexture(resource, descriptor, std::move(desc), shader_conversion,
	                           pixel_format, view_format, size.size);
}

// ResolveTexture's texture cache lookup for a description (FindImage may adjust it).
TextureBinding RenderExecutor::FindResolvedTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                                   const ShaderTextureResource&               descriptor,
                                                   TextureCache::ImageDesc desc,
                                                   bool shader_conversion, vk::Format pixel_format,
                                                   vk::Format view_format, uint32_t size) {
	auto&      texture_cache       = m_context.GetTextureCache();
	const bool storage             = resource.written;
	// v438zr (KYTY_TEX_SPARSE): a texture read without filtering, or written, keeps every level.
	desc.whole_chain               = resource.unfiltered || resource.written;
	auto       id                  = texture_cache.FindImage(desc, shader_conversion);
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format, size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	if (ImgTrace::Overlaps(desc.info.data.address, desc.info.data.size)) {
		ImgTrace::Log("%s sh=%016llx want 0x%llx+0x%llx %ux%u fmt%u -> id=%u img 0x%llx+0x%llx %ux%u fmt%u vk=%p gpumod=%d",
		              storage ? "storage" : "texture",
		              static_cast<unsigned long long>(BufferCache::s_diag_shader_hash),
		              static_cast<unsigned long long>(desc.info.data.address),
		              static_cast<unsigned long long>(desc.info.data.size), desc.info.extent.width,
		              desc.info.extent.height, static_cast<uint32_t>(desc.info.pixel_format),
		              static_cast<uint32_t>(id.index), static_cast<unsigned long long>(image->info.data.address),
		              static_cast<unsigned long long>(image->info.data.size), image->info.extent.width,
		              image->info.extent.height, static_cast<uint32_t>(image->backing.format),
		              static_cast<const void*>(static_cast<VkImage>(image->backing.image)),
		              image->IsGpuModified() ? 1 : 0);
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

// KYTY_STAGE_SETS: one set's writes, pushed or from the heap, bound at `set_index`.
template <typename Recorder>
static void CommitSetWrites(RenderContext& context, const Recorder& recorder,
                            vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                            uint32_t set_index, vk::DescriptorSetLayout set_layout, bool push,
                            std::span<vk::WriteDescriptorSet> writes) {
	if (writes.empty()) {
		return;
	}
	if (push) {
		recorder.pushDescriptorSetKHR(bind_point, layout, set_index, static_cast<uint32_t>(writes.size()),
		                              writes.data());
		return;
	}
	const auto set = context.GetDescriptorHeap().Commit(set_layout);
	for (auto& write: writes) {
		write.dstSet = set;
	}
	recorder.updateAndBindDescriptorSet(context.GetGraphics().device, bind_point, layout, set,
	                                    static_cast<uint32_t>(writes.size()), writes.data(), set_index);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data);

// The shader-data buffer: a uniform buffer (KYTY_SHADER_DATA_UBO) is read as uvec4s, so whole
// vectors are uploaded.
static vk::DescriptorBufferInfo NativeUploadShaderData(RenderContext&            context,
                                                       std::span<const uint32_t> data) {
	if (!ShaderRecompiler::IR::ShaderDataUniform() || data.size() % 4u == 0u) {
		return NativeUpload(context, data);
	}
	std::array<uint32_t, 256> padded {};
	EXIT_IF(data.size() + 3u > padded.size());
	std::ranges::copy(data, padded.begin());
	return NativeUpload(context, std::span<const uint32_t>(padded.data(), (data.size() + 3u) & ~size_t {3}));
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

// The flat slots the refresh left to the GPU (SrtGpuFill): their guest bytes are copied, where
// the buffer cache holds them, into the uploaded flat buffer before the command, behind the
// writes that produced them. Reading them on the host instead waited for the GPU (2.8 ms a frame
// for one dispatch in the jungle).
static void FillFlatSlotsOnGpu(RenderContext& context, const vk::DescriptorBufferInfo& flat,
                               std::span<const ShaderRecompiler::IR::SrtGpuFill> fills,
                               std::span<const ShaderRecompiler::IR::SrtVerify>  verify = {},
                               uint32_t                                          verify_base = 0) {
	if (fills.empty() && verify.empty()) {
		return;
	}
	auto& cache   = context.GetBufferCache();
	auto& command = context.GetCommandScheduler().Current();
	command.EndRendering();
	const auto      recorder = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                         nullptr);
	// The dwords to copy (guest address, flat offset), by address. Near ones share one obtained
	// range (one buffer lookup and sync for a descriptor's dwords, not one each), and each source
	// buffer gets one copy command with a region per dword. Every range is obtained before any
	// copy is recorded: obtaining one can merge buffers obtained before it.
	struct Item {
		uint64_t address;
		uint64_t flat_offset;
	};
	thread_local std::vector<Item> items;
	items.clear();
	for (const auto& fill: fills) {
		items.push_back({fill.address, uint64_t {fill.flat_offset}});
	}
	for (size_t i = 0; i < verify.size(); i++) {
		items.push_back({verify[i].address, uint64_t {verify_base} + 2u * i + 1u});
	}
	std::sort(items.begin(), items.end(),
	          [](const Item& a, const Item& b) { return a.address < b.address; });
	thread_local std::vector<std::pair<uint64_t, uint64_t>> ranges; // begin, end
	ranges.clear();
	for (const auto& item: items) {
		if (!ranges.empty() && item.address <= ranges.back().second + 256u &&
		    (item.address >> 12u) == (ranges.back().first >> 12u)) {
			ranges.back().second = std::max(ranges.back().second, item.address + sizeof(uint32_t));
		} else {
			ranges.emplace_back(item.address, item.address + sizeof(uint32_t));
		}
	}
	if (ranges.size() > 1u) {
		for (const auto& [begin, end_address]: ranges) {
			(void)cache.ObtainBuffer(begin, end_address - begin, false);
		}
	}
	thread_local std::vector<std::pair<vk::Buffer, vk::BufferCopy>> copies;
	copies.clear();
	size_t next = 0;
	for (const auto& [begin, end_address]: ranges) {
		auto [buffer, offset] = cache.ObtainBuffer(begin, end_address - begin, false);
		(void)offset;
		for (; next < items.size() && items[next].address < end_address; next++) {
			copies.emplace_back(buffer->Handle(),
			                    vk::BufferCopy {buffer->Offset(items[next].address),
			                                    flat.offset + items[next].flat_offset * sizeof(uint32_t),
			                                    sizeof(uint32_t)});
		}
	}
	std::stable_sort(copies.begin(), copies.end(), [](const auto& a, const auto& b) {
		return static_cast<VkBuffer>(a.first) < static_cast<VkBuffer>(b.first);
	});
	thread_local std::vector<vk::BufferCopy> regions;
	for (size_t i = 0; i < copies.size();) {
		size_t end_index = i;
		regions.clear();
		while (end_index < copies.size() && copies[end_index].first == copies[i].first) {
			regions.push_back(copies[end_index].second);
			end_index++;
		}
		recorder.copyBuffer(copies[i].first, flat.buffer, static_cast<uint32_t>(regions.size()),
		                    regions.data());
		i = end_index;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                         vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                         nullptr);
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (!image.binding.is_target) {
		NoteBindlessStateChange(image);
	}
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			if (image->binding.is_target) {
				NoteBindlessStateChange(*image);
			}
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

// Diagnostics: the first time a draw uses a bindless heap,
// read its texture descriptors and log how many are valid, of which types, and roughly how
// much memory they would take as host textures. Nothing is created.
static void SurveyBindlessHeap(const ShaderRecompiler::IR::BindlessHeapUse& use) {
	constexpr uint64_t MaxEntries = 1u << 20u;
	const uint64_t     span       = use.size > use.table_offset ? use.size - use.table_offset : 0;
	const uint64_t     entries    = std::min<uint64_t>(span / 32u, MaxEntries);
	uint64_t valid = 0, bytes = 0, unreadable = 0;
	std::array<uint64_t, 16> types {};
	std::vector<uint32_t> chunk(8u * 1024u);
	for (uint64_t first = 0; first < entries; first += chunk.size() / 8u) {
		const auto count   = std::min<uint64_t>(chunk.size() / 8u, entries - first);
		const auto address = use.base + use.table_offset + first * 32u;
		if (!Libs::LibKernel::Memory::TryReadBacking(address, chunk.data(), count * 32u)) {
			unreadable += count;
			continue;
		}
		for (uint64_t i = 0; i < count; i++) {
			ShaderRecompiler::IR::DescriptorValue value {};
			value.dword_count = 8u;
			std::copy_n(chunk.begin() + static_cast<ptrdiff_t>(i * 8u), 8, value.dwords.begin());
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
			if (descriptor.IsNull()) {
				continue;
			}
			valid++;
			types[static_cast<uint32_t>(descriptor.Type()) & 15u]++;
			const uint64_t width  = static_cast<uint64_t>(descriptor.Width5()) + 1u;
			const uint64_t height = static_cast<uint64_t>(descriptor.Height5()) + 1u;
			const uint64_t bpe    = std::max<uint32_t>(Prospero::NumBytesPerElement(descriptor.Format()), 1u);
			bytes += width * height * bpe * 4u / 3u;
		}
	}
	std::string histogram;
	for (uint32_t t = 0; t < types.size(); t++) {
		if (types[t] != 0) {
			histogram += fmt::format(" t{}={}", t, types[t]);
		}
	}
	LOGF("Bindless heap: base=0x%016" PRIx64 " size=0x%" PRIx64 " offset=%u entries=%" PRIu64
	     " valid=%" PRIu64 " unreadable=%" PRIu64 " est=%" PRIu64 "MB types:%s\n",
	     use.base, use.size, use.table_offset, entries, valid, unreadable, bytes >> 20u,
	     histogram.c_str());
}

static uint32_t BindlessBindingFor(const ShaderRecompiler::IR::ImageResource& resource) {
	using ShaderRecompiler::Decoder::ImageDimension;
	if (resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return ShaderRecompiler::IR::BindlessStorage2D;
	}
	if (resource.cube) {
		return ShaderRecompiler::IR::BindlessImagesCube;
	}
	if (resource.dimension == ImageDimension::Dim3D) {
		return ShaderRecompiler::IR::BindlessImages3D;
	}
	if (resource.dimension == ImageDimension::Dim2DArray ||
	    resource.dimension == ImageDimension::Dim2DMsaaArray) {
		return ShaderRecompiler::IR::BindlessImages2DArray;
	}
	return ShaderRecompiler::IR::BindlessImages2D;
}

// A heap entry the bindless array of this binding can hold: a float-sampled, single-sample
// texture of a matching shape. Anything else stays the placeholder rather than reaching
// ResolveTexture, which exits on descriptors a draw would never bind that way.
static bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	if (binding == ShaderRecompiler::IR::BindlessStorage2D) {
		// Its format and layout are checked against the resource (IsUsableStorageTexture).
		return !descriptor.IsNull() && descriptor.Type() == Prospero::ImageType::kColor2D;
	}
	if (descriptor.IsNull() ||
	    Prospero::SampledTextureNumericClass(descriptor.Format()) !=
	        Prospero::TextureNumericClass::Float) {
		return false;
	}
	switch (descriptor.Type()) {
		// A 3D image instruction reads a 2D or 2D-array T# as a 2D array (BindlessSlotArrayFlag).
		case Prospero::ImageType::kColor2D:
		case Prospero::ImageType::kColor2DArray:
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray ||
			       binding == ShaderRecompiler::IR::BindlessImages3D;
		// A cube's faces are a 2D array to the hardware: shaders that address faces themselves
		// (a 2D-array sample of a cube T#) were left the transparent placeholder, ~10 % of the
		// jungle's heap keys. KYTY_BINDLESS_CUBE_AS_ARRAY=0 restores that.
		case Prospero::ImageType::kCube:
			return binding == ShaderRecompiler::IR::BindlessImagesCube ||
			       (binding == ShaderRecompiler::IR::BindlessImages2DArray && [] {
				       static const bool enabled = [] {
					       const char* value = std::getenv("KYTY_BINDLESS_CUBE_AS_ARRAY");
					       return value == nullptr || value[0] != '0';
				       }();
				       return enabled;
			       }());
		case Prospero::ImageType::kColor3D: return binding == ShaderRecompiler::IR::BindlessImages3D;
		// A 1D texture (Marvel's Wolverine's particle colour gradients) is held as a 2D texture
		// of height 1 (ResolveBindlessKey); a 1D instruction samples its row 0, as does a 2D one on
		// the console, which addresses by the T# type.
		// KYTY_BINDLESS_1D=0 keeps 1D textures the placeholder (as before 2026-10-30).
		case Prospero::ImageType::kColor1D: {
			static const bool enabled = [] {
				const char* value = std::getenv("KYTY_BINDLESS_1D");
				return value == nullptr || value[0] != '0';
			}();
			return enabled && binding == ShaderRecompiler::IR::BindlessImages2D &&
			       descriptor.Height5() == 0 && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0;
		}
		default: return false;
	}
}

// KYTY_BINDLESS_STATS=1: per-reason counts of bindless key resolutions, to stderr every 64 frames.
enum BindlessStat : uint32_t {
	StatOk,
	StatNoBacking,
	StatNullDesc,
	StatNotFloat,
	StatBadType,
	StatEmptyImage,
	StatUnregistered,
	StatNoSlot,
	StatBudgetHit,
	StatRequested,
	StatBaseLevel,
	StatMinLod,
	StatRevalidated,
	StatCount
};
static std::atomic<uint64_t> g_bindless_stats[StatCount] {};
static std::atomic<uint64_t> g_bindless_width_stats[16] {};
static std::atomic<uint64_t> g_bindless_levels_stats[16] {};
static std::atomic<uint64_t> g_bindless_minlod_stats[16] {};
static std::atomic<uint64_t> g_bindless_type_stats[64] {};
static bool BindlessStatsEnabled() {
	static const bool enabled = std::getenv("KYTY_BINDLESS_STATS") != nullptr;
	return enabled;
}
static void BindlessStat(BindlessStat stat, uint64_t n = 1) {
	if (BindlessStatsEnabled()) {
		g_bindless_stats[stat].fetch_add(n, std::memory_order_relaxed);
	}
}
static void BindlessStatsReport(uint64_t frame) {
	if (!BindlessStatsEnabled() || (frame & 63u) != 0) {
		return;
	}
	std::fprintf(stderr,
	             "[bindless-stats] frame=%llu ok=%llu nobacking=%llu null=%llu notfloat=%llu "
	             "badtype=%llu empty=%llu unregistered=%llu noslot=%llu budget=%llu requested=%llu",
	             static_cast<unsigned long long>(frame),
	             static_cast<unsigned long long>(g_bindless_stats[StatOk].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatNoBacking].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatNullDesc].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatNotFloat].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatBadType].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatEmptyImage].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatUnregistered].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatNoSlot].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatBudgetHit].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatRequested].load()));
	std::fprintf(stderr, " baselevel=%llu minlod=%llu revalidated=%llu widthlog2:",
	             static_cast<unsigned long long>(g_bindless_stats[StatBaseLevel].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatMinLod].load()),
	             static_cast<unsigned long long>(g_bindless_stats[StatRevalidated].load()));
	for (uint32_t i = 0; i < 16; i++) {
		if (const auto n = g_bindless_width_stats[i].load(); n != 0) {
			std::fprintf(stderr, " %u=%llu", i, static_cast<unsigned long long>(n));
		}
	}
	std::fprintf(stderr, " minlodlevels:");
	for (uint32_t i = 0; i < 16; i++) {
		if (const auto n = g_bindless_minlod_stats[i].load(); n != 0) {
			std::fprintf(stderr, " %u=%llu", i, static_cast<unsigned long long>(n));
		}
	}
	std::fprintf(stderr, " lastlevel:");
	for (uint32_t i = 0; i < 16; i++) {
		if (const auto n = g_bindless_levels_stats[i].load(); n != 0) {
			std::fprintf(stderr, " %u=%llu", i, static_cast<unsigned long long>(n));
		}
	}
	std::fprintf(stderr, " types:");
	for (uint32_t i = 0; i < 64; i++) {
		if (const auto n = g_bindless_type_stats[i].load(); n != 0) {
			std::fprintf(stderr, " %u=%llu", i, static_cast<unsigned long long>(n));
		}
	}
	std::fprintf(stderr, "\n");
	std::fflush(stderr);
}

// KYTY_BINDLESS_LOG_KEYS=<binding>[:<key>]: each key resolution of that binding (0 2D, 1 2D array,
// 2 cube, 3 3D), or only that key, with its outcome and descriptor; up to 256 lines.
struct BindlessKeyLogSetting {
	uint32_t binding = UINT32_MAX;
	uint32_t key     = UINT32_MAX;
};
static const BindlessKeyLogSetting& BindlessKeyLog() {
	static const BindlessKeyLogSetting setting = [] {
		BindlessKeyLogSetting result;
		if (const char* value = std::getenv("KYTY_BINDLESS_LOG_KEYS"); value != nullptr) {
			char* end      = nullptr;
			result.binding = static_cast<uint32_t>(std::strtoul(value, &end, 10));
			if (end != nullptr && *end == ':') {
				result.key = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
			}
		}
		return result;
	}();
	return setting;
}

static void LogBindlessKey(const BindlessTable::Heap& heap, uint32_t key, const char* outcome,
                           const ShaderTextureResource* descriptor) {
	const auto& setting = BindlessKeyLog();
	if (setting.binding != heap.binding || (setting.key != UINT32_MAX && setting.key != key)) {
		return;
	}
	static std::atomic<uint32_t> lines {0};
	if (lines.fetch_add(1, std::memory_order_relaxed) >= 256u) {
		return;
	}
	if (descriptor == nullptr) {
		std::fprintf(stderr, "[bindless-key] binding=%u key=%u %s\n", heap.binding, key, outcome);
	} else {
		std::fprintf(stderr,
		             "[bindless-key] binding=%u key=%u %s type=%u format=%u tile=%u %ux%ux%u levels=%u-%u"
		             " base=0x%llx dst=0x%03x\n",
		             heap.binding, key, outcome, static_cast<uint32_t>(descriptor->Type()),
		             static_cast<uint32_t>(descriptor->Format()),
		             static_cast<uint32_t>(descriptor->TileMode()),
		             static_cast<uint32_t>(descriptor->Width5()) + 1u,
		             static_cast<uint32_t>(descriptor->Height5()) + 1u,
		             static_cast<uint32_t>(descriptor->Depth()) + 1u,
		             static_cast<uint32_t>(descriptor->BaseLevel()),
		             static_cast<uint32_t>(descriptor->LastLevel()),
		             static_cast<unsigned long long>(descriptor->Base40()),
		             static_cast<uint32_t>(descriptor->DstSelXYZW()));
	}
	std::fflush(stderr);
}

bool RenderExecutor::ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&                                 table = m_context.GetBindlessTable();
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count  = 8u;
	const auto address = heap.base + heap.table_offset + static_cast<uint64_t>(key) * 32u;
	heap.settled[key]  = 1;
	heap.stale[key]    = 0;
	if (heap.value) {
		// v439b: the T# the shader asked for (the key's value directory entry), not guest memory.
		std::copy_n(heap.descriptors[key].begin(), 8, value.dwords.begin());
	}
	if (!heap.value) {
		heap.descriptors[key] = {};
	}
	if (!heap.value && !Libs::LibKernel::Memory::TryReadBacking(address, value.dwords.data(), 32u)) {
		BindlessStat(StatNoBacking);
		LogBindlessKey(heap, key, "no-backing", nullptr);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	std::copy_n(value.dwords.data(), 8, heap.descriptors[key].begin());
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	if (!BindlessCompatible(descriptor, heap.binding)) {
		LogBindlessKey(heap, key, "incompatible", &descriptor);
		if (BindlessStatsEnabled()) {
			if (descriptor.IsNull()) {
				BindlessStat(StatNullDesc);
			} else if (Prospero::SampledTextureNumericClass(descriptor.Format()) !=
			           Prospero::TextureNumericClass::Float) {
				BindlessStat(StatNotFloat);
				std::fprintf(stderr,
				             "[bindless-notfloat] key=%u binding=%u format=%u type=%u %ux%u "
				             "resource_numeric=%u base=0x%llx\n",
				             key, heap.binding, static_cast<uint32_t>(descriptor.Format()),
				             static_cast<uint32_t>(descriptor.Type()),
				             static_cast<uint32_t>(descriptor.Width5()) + 1u,
				             static_cast<uint32_t>(descriptor.Height5()) + 1u,
				             static_cast<uint32_t>(heap.resource.numeric_class),
				             static_cast<unsigned long long>(descriptor.Base40()));
			} else {
				BindlessStat(StatBadType);
				g_bindless_type_stats[(static_cast<uint32_t>(descriptor.Type()) & 15u) |
				                      ((heap.binding & 3u) << 4u)]
				    .fetch_add(1, std::memory_order_relaxed);
			}
		}
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	// The GPU addresses by the T# type: a 3D instruction's 2D or 2D-array texture is a 2D array
	// at layer round(z). It lives in the 2D-array binding; the flag tells the shader.
	const bool as_array = heap.binding == ShaderRecompiler::IR::BindlessImages3D &&
	                      (descriptor.Type() == Prospero::ImageType::kColor2D ||
	                       descriptor.Type() == Prospero::ImageType::kColor2DArray);
	const uint32_t slot_binding =
	    as_array ? ShaderRecompiler::IR::BindlessImages2DArray : heap.binding;
	auto resource = heap.resource;
	if (as_array) {
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2DArray;
	}
	// The 2D array holds 2D views only: a 1D T# becomes a 2D one of height 1 (same address, size
	// and layout; GFX10 lays a 1D surface out as a 2D one of height 1), and a 1D resource's
	// heap resolves 2D views (the shader declares bindless 1D images 2D, BindlessAs2D).
	if (slot_binding == ShaderRecompiler::IR::BindlessImages2D) {
		if (descriptor.Type() == Prospero::ImageType::kColor1D) {
			value.dwords[3] = (value.dwords[3] & 0x0fffffffu) |
			                  (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u);
			descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
		}
		if (resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D) {
			resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		}
	}
	const bool storage = slot_binding == ShaderRecompiler::IR::BindlessStorage2D;
	if (storage && !IsUsableStorageTexture(resource, descriptor)) {
		BindlessStat(StatBadType);
		LogBindlessKey(heap, key, "storage-unsupported", &descriptor);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	auto binding = ResolveTexture(resource, value);
	auto& texture_cache = m_context.GetTextureCache();
	auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		BindlessStat(StatEmptyImage);
		LogBindlessKey(heap, key, image == nullptr ? "no-image" : "empty-image", &descriptor);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	if (storage && binding.desc.type != TextureCache::BindingType::Storage) {
		LogBindlessKey(heap, key, "storage-not-bound-as-storage", &descriptor);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindImage(binding.image_id, storage);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image != nullptr && !image->registered) {
		// Dropped by the cache (its destruction waits for the GPU): a slot holding its view would
		// outlive it, since its unregistration is over. The key is asked for again next frame.
		heap.settled[key] = 0;
		BindlessStat(StatUnregistered);
		LogBindlessKey(heap, key, "unregistered", &descriptor);
		table.SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
		return false;
	}
	const auto slot = table.AllocateSlot(slot_binding);
	if (image == nullptr || view == nullptr || slot == 0) {
		BindlessStat(StatNoSlot);
		LogBindlessKey(heap, key, "no-slot", &descriptor);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindlessStat(StatOk);
	LogBindlessKey(heap, key, "ok", &descriptor);
	if (BindlessStatsEnabled()) {
		// Width class, base level and min-LOD clamp of resolved textures: is streaming stuck low?
		const uint32_t width = static_cast<uint32_t>(descriptor.Width5()) + 1u;
		uint32_t       log2  = 0;
		while ((1u << (log2 + 1u)) <= width && log2 < 15u) log2++;
		g_bindless_width_stats[log2].fetch_add(1, std::memory_order_relaxed);
		if (descriptor.BaseLevel() != 0) BindlessStat(StatBaseLevel);
		if (descriptor.MinLod() != 0) {
			BindlessStat(StatMinLod);
			// Clamp in whole levels below the top, bucketed: 0..15.
			const uint32_t clamp_levels = std::min<uint32_t>(descriptor.MinLod() / 256u, 15u);
			g_bindless_minlod_stats[clamp_levels].fetch_add(1, std::memory_order_relaxed);
		}
		g_bindless_levels_stats[std::min<uint32_t>(descriptor.LastLevel(), 15u)].fetch_add(
		    1, std::memory_order_relaxed);
	}
	const auto layout = storage                  ? vk::ImageLayout::eGeneral
	                    : image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                            : vk::ImageLayout::eShaderReadOnlyOptimal;
	if (storage) {
		image->usage.storage = true;
		m_bindless_storage[static_cast<uint64_t>(binding.image_id.index) |
		                   (static_cast<uint64_t>(binding.image_id.generation) << 32u)] = binding.desc;
	}
	{
		const auto&    vi      = binding.desc.view_info;
		const uint32_t width   = std::max(image->info.extent.width >> vi.base_level, 1u);
		const uint32_t height  = std::max(image->info.extent.height >> vi.base_level, 1u);
		const uint32_t layers  = slot_binding == ShaderRecompiler::IR::BindlessImages3D
		                             ? std::max(image->info.extent.depth >> vi.base_level, 1u)
		                             : vi.layer_count;
		table.WriteSlot(slot_binding, slot, view, layout, static_cast<VkImage>(image->backing.image),
		                heap.base, key, width, height, layers);
	}
	// v448: a cube T# in the 2D-array binding (KYTY_BINDLESS_CUBE_AS_ARRAY): the shader's
	// coordinates are cube face coordinates (BindlessSlotCubeFlag); without the flag cube 1 and
	// up read the wrong layers (Marvel's Wolverine's light cookies: white blotches on metal).
	// KYTY_BINDLESS_CUBE_FACES=0 leaves the flag off (raw coordinates, as before v448).
	static const bool cube_faces = [] {
		const char* value = std::getenv("KYTY_BINDLESS_CUBE_FACES");
		return value == nullptr || value[0] != '0';
	}();
	const bool     cube_in_array = cube_faces && !as_array &&
	                               slot_binding == ShaderRecompiler::IR::BindlessImages2DArray &&
	                               descriptor.Type() == Prospero::ImageType::kCube;
	const uint32_t entry    = as_array        ? (slot | ShaderRecompiler::IR::BindlessSlotArrayFlag)
	                          : cube_in_array ? (slot | ShaderRecompiler::IR::BindlessSlotCubeFlag)
	                                          : slot;
	heap.slots[key]         = entry;
	heap.images[key]        = binding.image_id;
	image->bindless_pinned  = true;
	image->usage.texture    = true;
	heap.resolved.push_back(binding.image_id);
	heap.unchecked.push_back(binding.image_id);
	table.AddImageReference(binding.image_id, heap, key);
	table.SetTranslation(heap, key, entry);
	return true;
}

// Engine-method counters (SyncBindlessHeaps), logged and reset with the heap statistics.
struct BindlessEngineStats {
	std::atomic<uint64_t> syncs {0};
	std::atomic<uint64_t> written_bytes {0};
	std::atomic<uint64_t> view_changes {0};
	std::atomic<uint64_t> resource_changes {0};
	std::atomic<uint64_t> resettled {0};
	std::atomic<uint64_t> eager_settled {0};
	std::atomic<uint64_t> eager_failed {0};
	std::atomic<uint64_t> eager_paused {0};
	std::atomic<uint64_t> requested {0}; // keys draws asked for through feedback
	std::atomic<uint64_t> released {0};  // rewritten keys no draw had sampled: left to E3
	std::atomic<uint64_t> eager_capped {0}; // keys E3 stopped settling (MaxEagerSettles)
	std::atomic<uint64_t> kept_stale {0};   // view changes of unsampled keys left as they were
	std::atomic<uint64_t> stale_settled {0}; // such keys settled again once a draw sampled them
};
constexpr uint8_t MaxEagerSettles = 2;
static BindlessEngineStats g_bindless_engine_stats;

// With the engine method, every entry the guest has written is settled before draws ask for it
// (E3), at most KYTY_BINDLESS_ENGINE_BUDGET (default 256) per presented frame. On by default;
// KYTY_BINDLESS_ENGINE_EAGER=0 settles keys only when draws ask for them.
// Eager settling (E3) is off by default since 2026-10-08: in a new game it paused under the
// texture cache's bound for nearly the whole session, yet kept the ~3 GB it settled early (heap
// textures 9.3 GB against 6.0 GB without it, peak video memory 20.2 against 17.1 GB), with as
// many late requests (544 against 620 in the first 5,500 frames) and the same or better frame
// rate (prompt 21.6 against 22.1 fps). Textures load when a draw samples them; rewrites of keys
// already settled are still followed (E2). KYTY_BINDLESS_ENGINE_EAGER=1 turns it on.
static bool BindlessEngineEager() {
	static const bool eager = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_EAGER");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return eager;
}

// KYTY_BINDLESS_ENGINE_BUDGET_MB: the image memory eager settling may touch per presented frame
// (default 32). A count alone let a new area's load settle ~9,800 textures in 300 frames, large
// ones uploaded and detiled together, and the frames stalled; engines stream with a bandwidth.
static uint64_t BindlessEngineBudgetBytes() {
	static const uint64_t bytes = [] {
		const char* value  = std::getenv("KYTY_BINDLESS_ENGINE_BUDGET_MB");
		const auto  parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		return (parsed != 0 ? parsed : 32ull) << 20u;
	}();
	return bytes;
}

static uint32_t BindlessEngineBudget() {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_BUDGET");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		return parsed != 0 ? static_cast<uint32_t>(parsed) : 256u;
	}();
	return budget;
}

// KYTY_BINDLESS_ENGINE_VERIFY=1 (with KYTY_BINDLESS_ENGINE=1): the rolling scan keeps running and
// logs every rewritten entry the write watch did not report.
static bool BindlessEngineVerify() {
	static const bool verify = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_VERIFY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return verify;
}

void RenderExecutor::ReleaseBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&      table         = m_context.GetBindlessTable();
	auto&      texture_cache = m_context.GetTextureCache();
	const auto old_image     = heap.images[key];
	if (table.ReleaseKey(heap, key)) {
		if (auto* image = texture_cache.m_slot_images.try_get(old_image);
		    image != nullptr && image->bindless_pinned) {
			NoteBindlessStateChange(*image);
			image->bindless_pinned = false;
			if (BindlessEngineEnabled()) {
				texture_cache.ReleaseDroppedByGuest(old_image);
			}
		}
	}
}

bool RenderExecutor::ResettleBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	ReleaseBindlessKey(heap, key);
	return ResolveBindlessKey(heap, key);
}

void RenderExecutor::SyncBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	if (!BindlessEngineEnabled() || !table.Enabled()) {
		return;
	}
	if (table.HasWrittenRanges()) {
		SyncWrittenBindlessEntries();
	}
	if (BindlessEngineEager()) {
		SettleBindlessHeaps();
	}
}

void RenderExecutor::SettleBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	auto& stats = g_bindless_engine_stats;
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame != m_bindless_eager_frame) {
		m_bindless_eager_frame = frame;
		m_bindless_eager_left  = BindlessEngineBudget();
		m_bindless_eager_bytes = BindlessEngineBudgetBytes();
		// Settling ahead stops while the device is at 80 % of its budget, below the 90 % at which
		// the texture cache starts evicting, so the two do not chase each other. A new game's
		// jungle load filled the 32 GB card when only the images' own threshold was checked.
		m_bindless_eager_device_full = m_context.GetGraphics().DeviceMemoryAtLeast(80);
		// Once a frame: stale keys a draw has sampled since are settled again (finer mips).
		uint32_t stale_settled = 0;
		for (auto& heap: table.Heaps()) {
			std::erase_if(heap.stale_keys, [&](uint32_t key) {
				if (heap.stale[key] == 0) {
					return true; // settled again meanwhile
				}
				if (m_bindless_eager_left == 0 || !table.WasSampled(heap, key)) {
					return false;
				}
				m_bindless_eager_left--;
				heap.stale[key] = 0;
				stale_settled += ResettleBindlessKey(heap, key) ? 1u : 0u;
				return true;
			});
		}
		stats.stale_settled.fetch_add(stale_settled, std::memory_order_relaxed);
	}
	// Engine method, E3: entries no draw has asked for yet are settled too, as an engine creates a
	// texture's descriptor when it loads the texture. A pass stops at the budget and goes on from
	// its cursor at the next submission; the request path covers what is not settled yet. Paused
	// while the texture cache is short of memory, so eviction and settling do not chase each other.
	if (std::ranges::all_of(table.Heaps(), [](const auto& heap) { return heap.eager_complete; })) {
		return;
	}
	if (m_context.GetTextureCache().UnderPressure() || m_bindless_eager_device_full) {
		stats.eager_paused.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (m_bindless_eager_left == 0) {
		return;
	}
	KYTY_PROFILER_FUNCTION();
	constexpr uint32_t Window   = 2048;
	uint32_t&          budget   = m_bindless_eager_left;
	uint32_t           settled  = 0;
	uint32_t           failed   = 0;
	uint32_t           capped   = 0;
	for (auto& heap: table.Heaps()) {
		// A storage key is resolved only when a draw stores to it: settled ahead, every 2D
		// texture of the heap would be bound writable and committed as GPU-written each draw.
		if (heap.binding == ShaderRecompiler::IR::BindlessStorage2D) {
			heap.eager_complete = true;
			continue;
		}
		if (heap.eager_complete || budget == 0) {
			continue;
		}
		const auto keys = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto base = heap.base + heap.table_offset;
		uint32_t   key  = heap.eager_cursor < keys ? heap.eager_cursor : 0u;
		uint32_t   seen = 0;
		bool       marked = false;
		while (seen < keys && budget != 0) {
			const auto count = std::min(Window, keys - key);
			m_bindless_window.resize(count);
			uint32_t   looked   = count;
			const bool readable = Libs::LibKernel::Memory::TryReadBacking(
			    base + uint64_t {key} * 32u, m_bindless_window.data(), uint64_t {count} * 32u);
			for (uint32_t i = 0; readable && i < count; i++) {
				if (budget == 0) {
					looked = i;
					break;
				}
				if (heap.settled[key + i] != 0) {
					continue;
				}
				if (heap.eager_count[key + i] >= MaxEagerSettles) {
					continue; // its image keeps being replaced: the request path has it
				}
				ShaderTextureResource descriptor {};
				std::copy_n(m_bindless_window[i].data(), 8, descriptor.fields);
				if (descriptor.IsNull() || !BindlessCompatible(descriptor, heap.binding)) {
					continue;
				}
				budget--;
				if (++heap.eager_count[key + i] == MaxEagerSettles) {
					capped++;
				}
				if (ResolveBindlessKey(heap, key + i)) {
					if (const auto* image =
					        m_context.GetTextureCache().m_slot_images.try_get(heap.images[key + i]);
					    image != nullptr) {
						const auto size = image->AccountedSize();
						m_bindless_eager_bytes -= std::min(m_bindless_eager_bytes, size);
						if (m_bindless_eager_bytes == 0) {
							budget = 0; // this frame's bandwidth is spent
						}
					}
					// Not asked for by a draw: marked until one samples it (WasSampled), which also
					// lets the usage probe count it as unused under memory pressure.
					table.MarkUnsampled(heap, key + i);
					marked = true;
					settled++;
				} else {
					failed++;
				}
			}
			seen += looked;
			key = key + looked >= keys ? 0u : key + looked;
		}
		heap.eager_cursor = key;
		heap.eager_complete = seen >= keys && budget != 0;
		if (marked) {
			table.FlushFeedback(heap);
		}
	}
	stats.eager_settled.fetch_add(settled, std::memory_order_relaxed);
	stats.eager_failed.fetch_add(failed, std::memory_order_relaxed);
	stats.eager_capped.fetch_add(capped, std::memory_order_relaxed);
}

void RenderExecutor::SyncWrittenBindlessEntries() {
	auto& table = m_context.GetBindlessTable();
	KYTY_PROFILER_FUNCTION();
	// Engine method, E2: the pages the guest wrote since the last submission, already watched
	// again. Every entry in them is compared with the T# its key was settled from.
	table.TakeWrittenRanges(m_bindless_written);
	// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6); see LogBindlessHeapStats.
	constexpr std::array<uint32_t, 8> ViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
	                                            0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};
	uint32_t view_changes = 0, resource_changes = 0, resettled = 0, released = 0, kept = 0;
	for (auto& heap: table.Heaps()) {
		const auto heap_begin = heap.base + heap.table_offset;
		const auto keys       = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto heap_end   = heap_begin + uint64_t {keys} * 32u;
		for (const auto& [begin, end]: m_bindless_written) {
			const auto from = std::max(begin, heap_begin);
			const auto to   = std::min(end, heap_end);
			if (from >= to) {
				continue;
			}
			heap.eager_complete = false; // entries may have become valid
			const auto first = static_cast<uint32_t>((from - heap_begin) / 32u);
			const auto last  = static_cast<uint32_t>((to - heap_begin + 31u) / 32u);
			m_bindless_window.resize(last - first);
			if (!Libs::LibKernel::Memory::TryReadBacking(heap_begin + uint64_t {first} * 32u,
			                                              m_bindless_window.data(),
			                                              uint64_t {last - first} * 32u)) {
				continue;
			}
			for (uint32_t key = first; key < last; key++) {
				const auto& words = m_bindless_window[key - first];
				if (heap.descriptors[key] != words) {
					heap.eager_count[key] = 0; // the guest wrote it: E3 may settle it again
				}
				if (heap.settled[key] == 0 || heap.descriptors[key] == words) {
					continue;
				}
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((heap.descriptors[key][i] ^ words[i]) & ~ViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
				if (BindlessEngineEager() && !table.WasSampled(heap, key)) {
					// No draw has sampled it since it was settled.
					if (!resource) {
						// Streaming: the same texture with finer mips. The old view stays valid;
						// it is settled again when a draw samples the key.
						if (heap.stale[key] == 0) {
							heap.stale[key] = 1;
							heap.stale_keys.push_back(key);
						}
						kept++;
						continue;
					}
					// Another texture: the settle pass takes it within its budget, instead of
					// every rewrite of a load at once.
					ReleaseBindlessKey(heap, key);
					released++;
					continue;
				}
				resettled += ResettleBindlessKey(heap, key) ? 1u : 0u;
			}
		}
	}
	uint64_t bytes = 0;
	for (const auto& [begin, end]: m_bindless_written) bytes += end - begin;
	auto& stats = g_bindless_engine_stats;
	stats.syncs.fetch_add(1, std::memory_order_relaxed);
	stats.written_bytes.fetch_add(bytes, std::memory_order_relaxed);
	stats.view_changes.fetch_add(view_changes, std::memory_order_relaxed);
	stats.resource_changes.fetch_add(resource_changes, std::memory_order_relaxed);
	stats.resettled.fetch_add(resettled, std::memory_order_relaxed);
	stats.released.fetch_add(released, std::memory_order_relaxed);
	stats.kept_stale.fetch_add(kept, std::memory_order_relaxed);
}

uint32_t RenderExecutor::RevalidateBindlessKeys(uint32_t budget) {
	KYTY_PROFILER_FUNCTION();
	if (BindlessEngineEnabled() && !BindlessEngineVerify()) {
		return 0; // the write watch reports rewritten entries (SyncBindlessHeaps)
	}
	// The guest rewrites heap entries as it streams textures out and others in. A key settled
	// from the old T# kept sampling the old texture, and kept it pinned in the texture cache for
	// the rest of the run (in the jungle, 14-21 keys and 44-65 MB at any time). Each frame a
	// window of every heap is read back in one go and compared; a changed key is settled again
	// into a new slot, and an image no key refers to any more is unpinned, so the cache can
	// collect it. A heap of 64 Ki keys is covered in 32 frames.
	constexpr uint32_t Window = 2048;
	auto&              table         = m_context.GetBindlessTable();
	uint32_t           resolved      = 0;
	for (auto& heap: table.Heaps()) {
		const auto keys = static_cast<uint32_t>(heap.settled.size());
		if (keys == 0) {
			continue;
		}
		const uint32_t begin = heap.revalidate_cursor < keys ? heap.revalidate_cursor : 0u;
		const uint32_t count = std::min(Window, keys - begin);
		m_bindless_window.resize(count);
		const auto address = heap.base + heap.table_offset + static_cast<uint64_t>(begin) * 32u;
		const bool readable = Libs::LibKernel::Memory::TryReadBacking(
		    address, m_bindless_window.data(), static_cast<uint64_t>(count) * 32u);
		uint32_t next = begin + count;
		for (uint32_t i = 0; readable && i < count; i++) {
			const auto key = begin + i;
			if (heap.settled[key] == 0 || heap.descriptors[key] == m_bindless_window[i]) {
				continue;
			}
			if (resolved >= budget) {
				next = key; // look at it again next frame
				break;
			}
			if (BindlessEngineEnabled()) {
				static std::atomic<uint32_t> missed = 0;
				if (missed.fetch_add(1) < 64) {
					LOGF("Bindless engine: write not reported, heap=0x%" PRIx64 "+0x%x binding=%u "
					     "key=%u\n",
					     heap.base, heap.table_offset, heap.binding, key);
				}
			}
			resolved += ResettleBindlessKey(heap, key) ? 1u : 0u;
		}
		heap.revalidate_cursor = next >= keys ? 0u : next;
	}
	return resolved;
}

// Bytes of a T#'s mip chain from first_level down, ignoring tiling padding: an estimate.
static uint64_t MipChainBytes(const ShaderTextureResource& descriptor, uint32_t first_level) {
	const auto     format = descriptor.Format();
	const auto     block  = Prospero::BlockCompressedBytesPerBlock(format);
	const auto     texel  = block != 0 ? 0u : Prospero::NumBytesPerElement(format);
	const uint64_t width  = descriptor.Width5() + 1u;
	const uint64_t height = descriptor.Height5() + 1u;
	uint64_t       layers = descriptor.Type() == Prospero::ImageType::kColor2D
	                            ? 1u
	                            : static_cast<uint64_t>(descriptor.Depth()) + 1u;
	if (descriptor.Type() == Prospero::ImageType::kCube) layers = 6u;
	uint64_t bytes = 0;
	for (uint32_t level = first_level; level <= descriptor.MaxMip(); level++) {
		const auto w = std::max<uint64_t>(width >> level, 1u);
		const auto h = std::max<uint64_t>(height >> level, 1u);
		bytes += layers * (block != 0 ? ((w + 3u) / 4u) * ((h + 3u) / 4u) * block : w * h * texel);
	}
	return bytes;
}

// Diagnostics for the engine method (engine-method.md, E0): KYTY_BINDLESS_HEAP_STATS=1 logs every
// 300 presented frames, per heap:
// - valid: entries holding a texture this heap's view type can sample;
// - textures: distinct texture addresses among them (format aliases of one texture count once);
// - all_mb: their full mip chains; resident_mb: only the mips at or above each T#'s min LOD (what
//   the game has streamed in); sampled_mb: resident_mb of the textures draws asked for so far,
//   what translation on demand holds;
// - rewrites since the last line: view fields only (min LOD: streaming) or resource fields;
// - mip_stats: entries enabling the PS5's mip statistics;
// then the write watch's totals since the last line.
static void LogBindlessHeapStats(BindlessTable& table, uint64_t frame) {
	static const bool enabled = std::getenv("KYTY_BINDLESS_HEAP_STATS") != nullptr;
	static uint64_t   next    = 0;
	if (!enabled || frame < next) {
		return;
	}
	next = frame + 300;
	// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6): min LOD; dst_sel, base
	// and last level, BC swizzle; base array; min LOD warning, perf mod, mip-stats enable;
	// mip-stats slot and cache policy. Everything else describes the texture itself.
	constexpr std::array<uint32_t, 8> ViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
	                                            0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};
	static std::unordered_map<const BindlessTable::Heap*, std::vector<std::array<uint32_t, 8>>>
	                                     previous;
	std::vector<std::array<uint32_t, 8>> words;
	for (const auto& heap: table.Heaps()) {
		const auto entries =
		    std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		if (entries == 0) {
			continue;
		}
		words.resize(entries);
		if (!Libs::LibKernel::Memory::TryReadBacking(heap.base + heap.table_offset, words.data(),
		                                              static_cast<uint64_t>(entries) * 32u)) {
			continue;
		}
		uint32_t valid = 0, mip_stats = 0, settled = 0, view_changes = 0, resource_changes = 0;
		// Per texture address: the largest full chain and resident part seen, and whether a
		// sampled key refers to it.
		struct Texture {
			uint64_t all      = 0;
			uint64_t resident = 0;
			bool     sampled  = false;
		};
		std::unordered_map<uint64_t, Texture> textures;
		auto& last = previous[&heap];
		for (uint32_t key = 0; key < entries; key++) {
			settled += heap.settled[key] != 0 ? 1u : 0u;
			if (key < last.size() && last[key] != words[key]) {
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((last[key][i] ^ words[key][i]) & ~ViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
			}
			ShaderTextureResource descriptor {};
			std::copy_n(words[key].data(), 8, descriptor.fields);
			if (descriptor.IsNull() || !BindlessCompatible(descriptor, heap.binding)) {
				continue;
			}
			valid++;
			mip_stats += descriptor.MipStatsCntEn() ? 1u : 0u;
			const auto first =
			    std::max<uint32_t>(descriptor.BaseLevel(), descriptor.MinLod() >> 8u);
			auto& texture    = textures[descriptor.Base40()];
			texture.all      = std::max(texture.all, MipChainBytes(descriptor, 0));
			texture.resident = std::max(texture.resident, MipChainBytes(descriptor, first));
			texture.sampled |= heap.settled[key] != 0;
		}
		uint64_t all = 0, resident = 0, sampled = 0, sampled_textures = 0;
		for (const auto& [address, texture]: textures) {
			all += texture.all;
			resident += texture.resident;
			if (texture.sampled) {
				sampled += texture.resident;
				sampled_textures++;
			}
		}
		last = words;
		LOGF("Bindless heap stats: frame=%" PRIu64 " base=0x%" PRIx64 "+0x%x binding=%u entries=%u "
		     "valid=%u textures=%zu sampled_textures=%" PRIu64 " all_mb=%" PRIu64
		     " resident_mb=%" PRIu64 " sampled_mb=%" PRIu64 " settled=%u rewrites_view=%u "
		     "rewrites_resource=%u mip_stats=%u\n",
		     frame, heap.base, heap.table_offset, heap.binding, entries, valid, textures.size(),
		     sampled_textures, all >> 20u, resident >> 20u, sampled >> 20u, settled, view_changes,
		     resource_changes, mip_stats);
	}
	auto& stats = g_bindless_engine_stats;
	LOGF("Bindless engine stats: frame=%" PRIu64 " syncs=%" PRIu64 " written=%" PRIu64
	     " KiB view=%" PRIu64 " resource=%" PRIu64 " resettled=%" PRIu64 " eager_settled=%" PRIu64
	     " eager_failed=%" PRIu64 " eager_paused=%" PRIu64 " requested=%" PRIu64
	     " released=%" PRIu64 " eager_capped=%" PRIu64 " unregistered_keys=%" PRIu64
	     " kept_stale=%" PRIu64 " stale_settled=%" PRIu64 "\n",
	     frame, stats.syncs.exchange(0), stats.written_bytes.exchange(0) >> 10u,
	     stats.view_changes.exchange(0), stats.resource_changes.exchange(0),
	     stats.resettled.exchange(0), stats.eager_settled.exchange(0),
	     stats.eager_failed.exchange(0), stats.eager_paused.exchange(0),
	     stats.requested.exchange(0), stats.released.exchange(0), stats.eager_capped.exchange(0),
	     table.TakeUnregisteredKeys(), stats.kept_stale.exchange(0),
	     stats.stale_settled.exchange(0));
}

void RenderExecutor::ResolveBindlessRequests() {
	KYTY_PROFILER_FUNCTION();
	const CpuCost::Phase cpu_phase_("Bindless::ResolveRequests");
	// v384: queued translation entries whose earlier submissions have completed.
	m_context.GetBindlessTable().PublishSafe();
	// KYTY_BINDLESS_NONE=1 (v373, Crash B bisection, diagnostics only): no bindless key is ever
	// resolved, so every bindless sample reads its binding's placeholder (blue) and no texture is
	// reachable through the table. If Crash B persists, the stale read is not a bindless sample.
	if (static const bool none = [] {
		    const char* value = std::getenv("KYTY_BINDLESS_NONE");
		    return value != nullptr && value[0] == '1';
	    }();
	    none) {
		return;
	}
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	// One feedback round a presented frame left newly seen textures transparent for several
	// frames after a camera cut (a second or more at this title's frame rates): alpha-tested
	// foliage, debris and characters vanished until then. While requests keep coming, further
	// rounds run within the frame as soon as the previous snapshot has executed.
	// KYTY_BINDLESS_ROUNDS=<n> caps the rounds a frame (default 4; 1 restores one a frame).
	static const uint32_t max_rounds = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ROUNDS");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 4ul;
		return static_cast<uint32_t>(n < 1 ? 1 : n);
	}();
	static uint32_t rounds_this_frame = 0;
	static uint32_t last_requested    = 0;
	if (frame == m_bindless_frame &&
	    (last_requested == 0 || rounds_this_frame >= max_rounds)) {
		return;
	}
	auto& table     = m_context.GetBindlessTable();
	auto& scheduler = m_context.GetCommandScheduler();
	// The flags come from a snapshot recorded in an earlier frame. Until the GPU has executed
	// it, check again at the next bindless draw instead of waiting.
	if (table.SnapshotRecorded() && !table.SnapshotReady(scheduler)) {
		return;
	}
	if (frame != m_bindless_frame) {
		rounds_this_frame = 0;
	}
	rounds_this_frame++;
	m_bindless_frame = frame;
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot(scheduler);
		m_bindless_snapshot_frame = frame;
		return;
	}
	table.ConsumeSnapshot();
	const auto snapshot_frame = m_bindless_snapshot_frame;
	BindlessStatsReport(frame);
	// Each texture may upload and detile; spread first sight of a scene over a few frames.
	constexpr uint32_t Budget   = 128;
	uint32_t           resolved = RevalidateBindlessKeys(Budget);
	uint32_t           requested = 0;
	static std::atomic<uint32_t> frames_logged = 0;
	const bool log_frame = frames_logged.fetch_add(1) < 12;
	if (const auto word0 = m_context.GetBindlessTable().TakeWordZero(); word0 != 0) {
		static std::atomic<uint32_t> keys_logged = 0;
		if (keys_logged.fetch_add(1) < 64) {
			LOGF("Bindless out-of-range key: frame=%" PRIu64 " key=%u (0x%08x)\n", frame,
			     word0 & 0x7fffffffu, word0);
			// KYTY_BINDLESS_OOB_INFO=1: the shader and the count it compared the key with.
			if (static const bool info = std::getenv("KYTY_BINDLESS_OOB_INFO") != nullptr; info) {
				std::vector<uint32_t> words;
				m_context.GetBindlessTable().ReadSnapshot(ShaderRecompiler::IR::BindlessOobInfoWord, 3,
				                                          words);
				std::fprintf(stderr,
				             "[bindless-oob-key] frame %" PRIu64 ": key %u (word0 0x%08x); last out-of-range "
				             "lookup: shader ..%08x key %u count %u\n",
				             frame, word0 & 0x7fffffffu, word0, words[0], words[2], words[1]);
				const auto heaps = m_context.GetBindlessTable().Heaps().size();
				for (const auto& heap: m_context.GetBindlessTable().Heaps()) {
					if (heap.guest_entries == words[1] || keys_logged.load() <= 2) {
						std::fprintf(stderr,
						             "[bindless-oob-key]   heap base=0x%llx offset=%u binding=%u region=%u "
						             "entries=%u guest_entries=%u (of %zu heaps)\n",
						             static_cast<unsigned long long>(heap.base), heap.table_offset,
						             heap.binding, heap.region, heap.entries, heap.guest_entries, heaps);
					}
				}
			}
		}
	}
	// KYTY_NAN_MAGENTA=1: the last pixel shader that exported a NaN or Inf colour.
	if (static const bool nan_magenta = std::getenv("KYTY_NAN_MAGENTA") != nullptr; nan_magenta) {
		std::vector<uint32_t> words;
		m_context.GetBindlessTable().ReadSnapshot(ShaderRecompiler::IR::BindlessNanExportWord, 1, words);
		static uint32_t last = 0;
		static std::atomic<uint32_t> reported {0};
		if (!words.empty() && words[0] != 0 && words[0] != last && reported.fetch_add(1) < 200) {
			last = words[0];
			std::fprintf(stderr, "[nan-export] frame %" PRIu64 ": pixel shader ..%08x exported NaN/Inf\n",
			             frame, words[0]);
		}
	}
	// v127: a translation value that named no slot of its array (the shader sampled slot 0).
	if (const auto oob = m_context.GetBindlessTable().TakeOutOfRangeSlot(); oob != 0) {
		extern std::atomic<uint32_t> g_bindless_oob_seen;
		if (g_bindless_oob_seen.fetch_add(1) < 64) {
			std::fprintf(stderr,
			             "[bindless-oob] frame %" PRIu64 ": translation value 0x%08x names no slot "
			             "(snapshot frame %" PRIu64 "); sampled slot 0 instead\n",
			             frame, oob, snapshot_frame);
		}
	}
	// v301: an indirect image selection past its candidate list (the shader sampled the root).
	if (const auto value = m_context.GetBindlessTable().TakeFeedbackWord(
	        ShaderRecompiler::IR::IndirectOutOfRangeWord + 1u);
	    value != 0) {
		const auto     shader = m_context.GetBindlessTable().TakeFeedbackWord(
		    ShaderRecompiler::IR::IndirectOutOfRangeWord);
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1) < 64) {
			std::fprintf(stderr,
			             "[indirect-oob] frame %" PRIu64 ": shader ..%08x selected image %u of %u "
			             "candidates (snapshot frame %" PRIu64 "); sampled the root instead\n",
			             frame, shader, value & 0xffffffu, value >> 24u, snapshot_frame);
		}
	}
	// Diagnostics: KYTY_PROBE_RECT_FILE=<path> (with KYTY_PROBE_RECT): the probed pixels' values
	// (BindlessCaptureProbe), rewritten every 30th snapshot.
	if (static const char* probe_file = std::getenv("KYTY_PROBE_RECT_FILE"); probe_file != nullptr) {
		static uint32_t probe_rounds = 0;
		if (++probe_rounds % 30u == 0u) {
			std::vector<uint32_t> words;
			m_context.GetBindlessTable().ReadSnapshot(ShaderRecompiler::IR::BindlessCaptureProbe,
			                                          ShaderRecompiler::IR::BindlessCaptureProbeWords,
			                                          words);
			if (FILE* file = std::fopen(probe_file, "wb"); file != nullptr) {
				std::fwrite(words.data(), sizeof(uint32_t), words.size(), file);
				std::fclose(file);
			}
		}
	}
	// Diagnostics: KYTY_KEY_CAPTURE=<hash>: the image and sampler keys that shader sampled since the
	// last print, every 60th snapshot, up to 12 prints.
	if (static const bool capture = std::getenv("KYTY_KEY_CAPTURE") != nullptr; capture) {
		static uint32_t rounds  = 0;
		static uint32_t printed = 0;
		if (++rounds % 60u == 0u && printed < 12u) {
			std::vector<std::pair<uint32_t, uint32_t>> images;
			std::vector<std::pair<uint32_t, uint32_t>> samplers;
			auto& table = m_context.GetBindlessTable();
			table.TakeCapture(ShaderRecompiler::IR::BindlessCaptureImages, 0x10000u, images);
			table.TakeCapture(ShaderRecompiler::IR::BindlessCaptureSamplers, 0x10000u, samplers);
			printed += (images.empty() && samplers.empty()) ? 0u : 1u;
			std::string line;
			char        item[48];
			if (!images.empty() || !samplers.empty()) {
			for (const auto& [key, word]: images) {
				std::snprintf(item, sizeof(item), " %u(img%u,s%u)", key, (word >> 16u) & 0xffu,
				              word & 0xffffu);
				line += item;
				if (line.size() > 3000) break;
			}
			std::fprintf(stderr, "[key-capture] frame=%" PRIu64 " image keys=%zu:%s\n", frame,
			             images.size(), line.c_str());
			line.clear();
			for (const auto& [key, word]: samplers) {
				std::snprintf(item, sizeof(item), " tex%u:%08x", key, word);
				line += item;
				if (line.size() > 3000) break;
			}
			std::fprintf(stderr, "[key-capture] frame=%" PRIu64 " raw sampler dwords by texture key=%zu:%s\n", frame,
			             samplers.size(), line.c_str());
			std::fflush(stderr);
			// Once: search the bound buffers' CPU copies for the first captured keys' records.
			static bool scanned = false;
			std::vector<std::pair<uint64_t, uint64_t>> buffers;
			{
				std::lock_guard lock(g_capture_buffers_mutex);
				buffers = g_capture_buffers;
			}
			// KYTY_KEY_SCAN_EXTRA=<hex address>:<hex size>: one more range to search (a source the
			// shader does not bind).
			if (const char* extra = std::getenv("KYTY_KEY_SCAN_EXTRA"); extra != nullptr) {
				char*          end  = nullptr;
				const uint64_t base = std::strtoull(extra, &end, 16);
				const uint64_t size = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 16) : 0u;
				buffers.emplace_back(base, size);
			}
			if (!scanned && !images.empty() && !buffers.empty()) {
				scanned = true;
				for (size_t b = 0; b < buffers.size(); b++) {
					const auto [base, size] = buffers[b];
					const uint64_t bytes    = std::min<uint64_t>(size, 32u << 20u) & ~uint64_t {3};
					const bool     dirty    = base != 0 && bytes != 0 &&
					                   m_context.GetBufferCache().HasGpuDirtyBytes(base, bytes);
					std::fprintf(stderr, "[key-scan] buffer[%zu] base=0x%012" PRIx64 " bytes=0x%" PRIx64 " gpu_dirty=%d\n",
					             b, base, size, dirty ? 1 : 0);
					{
						std::string aliases;
						for (const auto& [alias, alias_size]: Libs::LibKernel::Memory::FindDirectMemoryAliases(base, 0x1000u)) {
							aliases += fmt::format(" 0x{:x}+0x{:x}", alias, alias_size);
						}
						std::fprintf(stderr, "[key-scan]   mappings of its first page:%s\n", aliases.c_str());
					}
					if (base == 0 || bytes < 16u) continue;
					std::vector<uint32_t> words(bytes / 4u);
					// Chunk by chunk: the readable prefix (a growing store may end before its bound).
					uint64_t readable = 0;
					while (readable < bytes) {
						const uint64_t chunk = std::min<uint64_t>(bytes - readable, 0x10000u);
						if (!Libs::LibKernel::Memory::TryReadBacking(base + readable, words.data() + readable / 4u, chunk)) {
							break;
						}
						readable += chunk;
					}
					words.resize(readable / 4u);
					if (readable != bytes) {
						std::fprintf(stderr, "[key-scan]   readable 0x%" PRIx64 " of 0x%" PRIx64 "\n", readable, bytes);
					}
					for (size_t k = 0; k < std::min<size_t>(images.size(), 3u); k++) {
						const uint32_t key  = images[k].first;
						uint32_t       hits = 0;
						for (size_t w = 0; w < words.size() && hits < 3u; w++) {
							if (!(((w & 3u) == 1u && (words[w] >> 16u) == key) ||
							      ((w & 3u) == 2u && (words[w] & 0xffffu) == key))) continue;
							hits++;
							std::string around;
							char        item[16];
							const size_t from = w >= 2 ? w - 2 : 0;
							for (size_t j = from; j < std::min(words.size(), from + 24); j++) {
								std::snprintf(item, sizeof(item), " %08x", words[j]);
								around += item;
							}
							const uint32_t plus64 = w + 16 < words.size() ? words[w + 16] : 0xdeadbeefu;
							std::fprintf(stderr, "[key-scan]   key %u at +0x%zx (%08x), +64 -> %08x; from +0x%zx:%s\n",
							             key, w * 4u, words[w], plus64, from * 4u, around.c_str());
							// KYTY_KEY_PEEK=<hex source>:<hex destination>: for a hit in the copy source,
							// the GPU copies of its record (rows 0..4) in the source and in the destination.
							if (const char* peek = std::getenv("KYTY_KEY_PEEK"); peek != nullptr && base != 0) {
								char*          end = nullptr;
								const uint64_t src = std::strtoull(peek, &end, 16);
								const uint64_t dst = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 16) : 0u;
								if (base == src && dst != 0) {
									const uint64_t row = base + (w * 4u & ~uint64_t {15});
									for (const uint64_t at: {row, dst + (row - src)}) {
										std::vector<uint32_t> gpu;
										std::string           text;
										if (m_context.GetBufferCache().PeekGpuCopy(at, 0x50u, gpu)) {
											for (const auto value: gpu) {
												text += fmt::format(" {:08x}", value);
											}
										} else {
											text = " (no single buffer)";
										}
										std::fprintf(stderr, "[key-scan]   GPU copy at 0x%012" PRIx64 ":%s\n", at, text.c_str());
									}
								}
							}
						}
					}
				}
			}
			}
		}
	}
	// v439: blocks address-keyed images asked for become heaps first; their keys are requested
	// from the next snapshot on.
	m_context.GetBindlessTable().ProcessAddressRequests();
	// v439b: T#s value-keyed images asked for get directory entries, resolved right away; then
	// the value heaps' own pending keys (a texture that was not resolvable yet).
	{
		auto& table_ref = m_context.GetBindlessTable();
		table_ref.ProcessValueRequests(m_bindless_value_created);
		for (const auto& [heap, key]: m_bindless_value_created) {
			if (resolved < Budget) {
				resolved += ResolveBindlessKey(*heap, key) ? 1u : 0u;
			}
		}
		for (uint32_t binding = 0; binding < ShaderRecompiler::IR::BindlessValueBindings; binding++) {
			auto* heap = table_ref.ValueHeap(binding);
			if (heap == nullptr) {
				continue;
			}
			m_bindless_requests.clear();
			table_ref.TakeRequests(*heap, m_bindless_requests);
			requested += static_cast<uint32_t>(m_bindless_requests.size());
			for (const auto key: m_bindless_requests) {
				if (heap->settled[key] != 0 || heap->descriptors[key][0] == 0u || resolved >= Budget) {
					continue;
				}
				resolved += ResolveBindlessKey(*heap, key) ? 1u : 0u;
			}
		}
	}
	for (auto& heap: m_context.GetBindlessTable().Heaps()) {
		m_bindless_requests.clear();
		m_context.GetBindlessTable().TakeRequests(heap, m_bindless_requests);
		requested += static_cast<uint32_t>(m_bindless_requests.size());
		if (log_frame) {
			LOGF("Bindless feedback: frame=%" PRIu64 " binding=%u region=%u flagged=%zu\n",
			     frame, heap.binding, heap.region, m_bindless_requests.size());
		}
		for (const auto key: m_bindless_requests) {
			if (heap.settled[key] != 0) {
				continue;
			}
			if (resolved >= Budget) {
				BindlessStat(StatBudgetHit);
				break; // still pending: the next frame flags it again
			}
			resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
		}
	}
	Profiler::Add(Profiler::Counter::BindlessResolved, resolved);
	if (Common::FloorStats::Enabled()) {
		Common::FloorStats::Add(Common::FloorStats::Get().heap_keys, resolved);
	}
	g_bindless_engine_stats.requested.fetch_add(requested, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged = 0;
	if (requested != 0 && logged.fetch_add(1) < 64) {
		LOGF("Bindless requests: frame=%" PRIu64 " requested=%u resolved=%u\n", frame,
		     requested, resolved);
	}
	UpdateBindlessUsageProbe(frame, snapshot_frame);
	LogBindlessHeapStats(table, frame);
	table.RecordFeedbackSnapshot(scheduler);
	table.RecordSlotSweep(scheduler, frame);
	m_bindless_snapshot_frame = frame;
}

// Bindless textures stay pinned in the texture cache while a heap key refers to them, and the
// guest's heaps refer to every texture it has streamed in: on 4-8 GB cards they grew to 2-5 GB
// and never shrank, until allocations failed (ISSUES #15). Under memory pressure a probe finds
// the ones no draw samples any more: it marks every resident key's feedback word, draws that
// sample a key overwrite its mark, and a snapshot taken ProbeFrames later lists the images whose
// keys all kept it. The texture cache may then free those like unpinned images; their keys go
// back to pending, and a draw that samples one again asks for it anew. KYTY_BINDLESS_EVICT=0
// disables the probe.
void RenderExecutor::UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame) {
	// KYTY_DUMP_TEXTURES_TRIGGER (TextureCache::DumpTextureNote): the probe's marks also tell
	// which bindless textures the frames after the trigger sample.
	if (static const char* dump_trigger = std::getenv("KYTY_DUMP_TEXTURES_TRIGGER");
	    dump_trigger != nullptr) {
		static uint64_t dump_armed = 0;
		auto&           dump_table = m_context.GetBindlessTable();
		auto&           dump_cache = m_context.GetTextureCache();
		if (dump_armed == 0) {
			std::error_code error;
			if (std::filesystem::exists(dump_trigger, error)) {
				std::filesystem::remove(dump_trigger, error);
				dump_table.ArmUsageProbe();
				dump_armed = frame;
				dump_cache.BeginTextureDump(frame);
			}
		} else if (dump_armed != UINT64_MAX && snapshot_frame >= dump_armed + 2) {
			std::vector<ImageId> used;
			dump_table.CollectSampledImages(used);
			std::fprintf(stderr, "[texdump] bindless: %zu images sampled since frame %llu\n",
			             used.size(), static_cast<unsigned long long>(dump_armed));
			dump_cache.AddTextureDumpImages(used);
			dump_armed = 0; // the next trigger file starts another dump
		}
		return; // no eviction probe meanwhile
	}
	// v389: with sparse feedback (Frank's default KYTY_BINDLESS_EVICT=0) the probe still runs, the
	// shaders storing every lookup's word only while it does (BindlessProbeActiveWord).
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_EVICT");
		return value == nullptr || value[0] != '0' || ShaderRecompiler::IR::BindlessRuntimeProbeEnabled();
	}();
	if (!enabled) {
		return;
	}
	constexpr uint64_t ProbeFrames   = 16; // frames a texture must go unsampled
	// Frames between the end of one probe and the next. v389: 300 (~10 s) when the probe runs on
	// sparse feedback, since every lookup stores its word while it runs.
	const uint64_t     ProbeInterval = [] {
		const char* value = std::getenv("KYTY_BINDLESS_EVICT");
		return value != nullptr && value[0] == '0' ? uint64_t {300} : uint64_t {32};
	}();
	auto&              table         = m_context.GetBindlessTable();
	auto&              texture_cache = m_context.GetTextureCache();
	if (m_bindless_probe_frame != 0) {
		// The snapshot read here must have been recorded ProbeFrames after the marks.
		if (snapshot_frame < m_bindless_probe_frame + ProbeFrames) {
			return;
		}
		m_bindless_unused.clear();
		table.CollectUnusedImages(m_bindless_unused);
		table.SetProbeActive(false);
		texture_cache.SetBindlessEvictable(m_bindless_unused);
		// v388: every probe result on stderr (LOGF is off under KYTY_FPS_ONLY): C26 showed an empty
		// evictable set in every census.
		{
			uint64_t unused_bytes = 0;
			for (const auto id: m_bindless_unused) {
				if (const auto* image = texture_cache.m_slot_images.try_get(id); image != nullptr) {
					unused_bytes += image->AccountedSize();
				}
			}
			std::fprintf(stderr,
			             "[bindless-probe] armed at frame %" PRIu64 ", read at %" PRIu64 " (snapshot of %" PRIu64
			             "): %zu of %zu referenced images unsampled (%" PRIu64 " MB)\n",
			             m_bindless_probe_frame, frame, snapshot_frame, m_bindless_unused.size(),
			             table.ReferencedImageCount(), unused_bytes >> 20u);
		}
		m_bindless_probe_frame = 0;
		m_bindless_next_probe  = frame + ProbeInterval;
		static std::atomic<uint32_t> logged = 0;
		if (logged.fetch_add(1) < 32) {
			uint64_t bytes = 0;
			for (const auto id: m_bindless_unused) {
				if (const auto* image = texture_cache.m_slot_images.try_get(id); image != nullptr) {
					bytes += image->AccountedSize();
				}
			}
			LOGF("Bindless probe: frame=%" PRIu64 " unused=%zu (%" PRIu64 " MB) may be evicted\n",
			     frame, m_bindless_unused.size(), bytes >> 20u);
		}
		return;
	}
	// The heap's textures follow the guest (settled when it writes an entry, freed when it drops
	// them), so the probe evicts them only when the device itself is short of memory (4-8 GB
	// cards), not at the texture cache's bound: under a 6 GiB bound it evicted 27,915 of them in
	// 14 minutes, each settled again when drawn.
	// v387: from 80 % of the budget on (KYTY_BINDLESS_PROBE_PERCENT), so the collector has a list of
	// unsampled heap textures before it trims at KYTY_VRAM_TRIM_PERCENT (85).
	static const uint32_t probe_percent = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PROBE_PERCENT");
		const auto  set   = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 80u;
		return set == 0 || set > 100 ? 90u : set;
	}();
	if (frame >= m_bindless_next_probe && m_context.GetGraphics().DeviceMemoryAtLeast(probe_percent)) {
		table.SetProbeActive(true);
		const auto marked = table.ArmUsageProbe();
		m_bindless_probe_frame = frame;
		static std::atomic<uint32_t> armed_logged = 0;
		if (armed_logged.fetch_add(1) < 64) {
			std::fprintf(stderr, "[bindless-probe] armed at frame %" PRIu64 ": %u keys marked\n", frame, marked);
		}
	}
}

// Bindless samplers: the S# records of each sampler heap a draw indexes are mirrored into the
// sampler array of set 1, once per frame per heap, and the shader gets the region base and record
// count. Keys outside it, and heaps that cannot be read, use the default sampler in slot 0.
void RenderExecutor::PrepareBindlessSamplers(const ShaderStageRuntime& runtime,
                                             PreparedBindings&         prepared) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (snapshot.bindless_sampler_heaps.empty()) {
		return;
	}
	auto& table = m_context.GetBindlessTable();
	auto& cache = m_context.GetSamplerCache();
	{
		ShaderSamplerResource default_sampler;
		std::ranges::copy(ShaderRecompiler::IR::BindlessDefaultSampler, default_sampler.fields);
		table.WriteDefaultSampler(cache.GetSampler(default_sampler, false));
	}
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	constexpr uint64_t MaxRecords = 1024;
	for (const auto& use: snapshot.bindless_sampler_heaps) {
		uint32_t region  = 0;
		uint32_t entries = 0;
		if (table.SamplersEnabled() && use.sampler < program.info.samplers.size()) {
			const auto&    sampler = program.info.samplers[use.sampler];
			const uint32_t flags =
			    (sampler.depth_compare ? BindlessTable::SamplerDepthCompare : 0u) |
			    (sampler.force_point_filtering ? BindlessTable::SamplerPointFiltering : 0u) |
			    (sampler.integer_border ? BindlessTable::SamplerIntegerBorder : 0u);
			auto* heap = table.FindOrCreateSamplerHeap(use.base, use.table_offset, flags);
			if (heap->checked_frame != frame) {
				heap->checked_frame = frame;
				const uint64_t available =
				    use.size > use.table_offset ? (use.size - use.table_offset) / 16u : 0u;
				std::vector<std::array<uint32_t, 4>> records(std::min(available, MaxRecords));
				const uint64_t address = use.base + use.table_offset;
				const uint64_t bytes   = records.size() * sizeof(records[0]);
				if (!records.empty() &&
				    !m_context.GetBufferCache().HasGpuDirtyBytes(address, bytes) &&
				    Libs::LibKernel::Memory::TryReadBacking(address, records.data(), bytes)) {
					(void)table.MirrorSamplerHeap(*heap, records, cache);
				}
			}
			region  = heap->region;
			entries = region != 0 ? static_cast<uint32_t>(heap->records.size()) : 0u;
			// Diagnostics: KYTY_KEY_CAPTURE=<hash>: the sampler heaps that shader indexes and the
			// S# its key 0 selects, a few times.
			static const uint64_t capture_hash = [] {
				const char* value = std::getenv("KYTY_KEY_CAPTURE");
				return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
			}();
			static std::atomic<uint32_t> capture_printed {0};
			if (capture_hash != 0 && program.shader_hash == capture_hash && capture_printed.fetch_add(1) < 24u) {
				const uint64_t address = use.base + use.table_offset;
				const bool     dirty   = m_context.GetBufferCache().HasGpuDirtyBytes(address, 64u);
				std::array<uint32_t, 4> rec {};
				const bool read = Libs::LibKernel::Memory::TryReadBacking(address, rec.data(), sizeof(rec));
				ShaderSamplerResource r;
				std::ranges::copy(rec, r.fields);
				std::fprintf(stderr,
				             "[key-capture] sampler use=%u heap=0x%012" PRIx64 "+0x%x size=0x%" PRIx64
				             " gpu_dirty=%d read=%d region=%u entries=%u rec0=%08x %08x %08x %08x clamp=%u%u%u"
				             " mag=%u min=%u mip=%u aniso=%u\n",
				             use.sampler, use.base, use.table_offset, use.size, dirty ? 1 : 0, read ? 1 : 0,
				             region, entries, rec[0], rec[1], rec[2], rec[3], r.ClampX(), r.ClampY(),
				             r.ClampZ(), r.XyMagFilter(), r.XyMinFilter(), r.MipFilter(), r.MaxAnisoRatio());
			}
		}
		prepared.bindless_patches.push_back({use.mapping_offset, region, entries});
	}
}

void RenderExecutor::PrepareBindlessHeaps(const ShaderStageRuntime& runtime,
                                          PreparedBindings& prepared) {
	prepared.bindless_patches.clear();
	prepared.bindless_heaps.clear();
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       table    = m_context.GetBindlessTable();
	if ((snapshot.bindless_heaps.empty() && snapshot.bindless_sampler_heaps.empty()) ||
	    !table.Enabled()) {
		return;
	}
	PrepareBindlessSamplers(runtime, prepared);
	if (snapshot.bindless_heaps.empty()) {
		return;
	}
	ResolveBindlessRequests();
	uint32_t address_bindings = 0; // v439: the bindings whose address heaps this draw took
	uint32_t value_bindings   = 0; // v439b: the bindings whose value heap this draw took
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) {
			continue;
		}
		const auto& resource = program.info.images[use.image];
		if (use.value) {
			// v439b: a value-keyed image: the value heap of its binding, whose textures the
			// shader may reach.
			const auto binding = BindlessBindingFor(resource);
			prepared.bindless_patches.push_back({use.mapping_offset, 0u, 0u});
			if (binding < 32u && (value_bindings & (1u << binding)) == 0u) {
				value_bindings |= 1u << binding;
				if (auto* heap = table.ValueHeap(binding); heap != nullptr) {
					prepared.bindless_heaps.push_back(heap);
				}
			}
			continue;
		}
		if (use.address) {
			// v439: an address-keyed image: its resource index for requests, and every address
			// heap of its binding, whose textures the shader may reach.
			const auto binding = BindlessBindingFor(resource);
			prepared.bindless_patches.push_back(
			    {use.mapping_offset, table.RegisterAddressResource(resource), 0u});
			if (binding < 32u && (address_bindings & (1u << binding)) == 0u) {
				address_bindings |= 1u << binding;
				for (auto* heap: table.AddressHeaps(binding)) {
					if (BindlessEngineEnabled() && table.NeedsWatch(*heap)) {
						table.WatchHeap(*heap);
					}
					prepared.bindless_heaps.push_back(heap);
				}
			}
			continue;
		}
		const uint64_t span  = use.size > use.table_offset ? use.size - use.table_offset : 0;
		const auto entries   = static_cast<uint32_t>(std::min<uint64_t>(span / 32u, 1u << 18u));
		auto* heap = table.FindOrCreateHeap(use.base, use.table_offset, BindlessBindingFor(resource),
		                                    entries, resource);
		if (heap == nullptr) {
			continue; // region 0, count 0: every key samples the placeholder
		}
		if (BindlessEngineEnabled() && table.NeedsWatch(*heap)) {
			table.WatchHeap(*heap);
		}
		prepared.bindless_patches.push_back(
		    {use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_heaps.push_back(heap);
	}
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	const CpuCost::Phase cpu_phase_("PrepareBindings");
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	const DiagShaderScope diag_shader_scope(program.shader_hash, program.stage);
	// Diagnostics: KYTY_WATCH_SHADER=<hash> logs the buffer and texture descriptors a shader is
	// bound with (first few uses; KYTY_WATCH_SHADER_EVERY=<n> adds every n-th use, up to 8 more).
	const uint64_t        watch_shader = WatchedShaderHash();
	static const uint32_t watch_every = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER_EVERY");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static std::atomic<uint32_t> watched_uses {0};
	const auto                   watched_use = watch_shader != 0 && program.shader_hash == watch_shader
	                                               ? watched_uses.fetch_add(1)
	                                               : UINT32_MAX;
	static std::atomic<uint32_t> frame_uses {0};
	const bool in_watched_frame =
	    watched_use != UINT32_MAX &&
	    InWatchedShaderFrame(program.shader_hash,
	                         m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                         frame_uses);
	// KYTY_WATCH_SHADER_TRIGGER=<file>: while that file exists, every use of the watched shader
	// is listed (up to 240 uses), on stderr, with the presented frame, to compare frames.
	static const char*           watch_trigger = std::getenv("KYTY_WATCH_SHADER_TRIGGER");
	static std::atomic<uint32_t> triggered_uses {0};
	static std::atomic<uint64_t> trigger_checked_frame {UINT64_MAX};
	static std::atomic_bool      trigger_on {false};
	if (watch_trigger != nullptr && watched_use != UINT32_MAX) {
		const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		if (trigger_checked_frame.exchange(frame) != frame) {
			std::error_code error;
			trigger_on = std::filesystem::exists(watch_trigger, error);
		}
	}
	// KYTY_WATCH_COMPACT=1: one short line per use instead (bases, small-buffer hashes, image
	// bases), up to 4000 uses, so the capture barely slows the game.
	static const bool watch_compact = std::getenv("KYTY_WATCH_COMPACT") != nullptr;
	if (watch_compact && watch_trigger != nullptr && watched_use != UINT32_MAX && trigger_on &&
	    triggered_uses.fetch_add(1) < 4000u) {
		const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		std::string line = fmt::format("[wc] f={} use={}", frame, watched_use);
		for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			const uint64_t bytes =
			    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
			uint64_t hash = 0;
			if (r.Base48() != 0 && bytes != 0 && bytes <= 4096) {
				std::vector<uint8_t> data(bytes);
				if (Libs::LibKernel::Memory::TryReadBacking(r.Base48(), data.data(), bytes)) {
					hash = XXH3_64bits(data.data(), data.size());
				}
			}
			const bool gpu_written = r.Base48() != 0 && bytes != 0 &&
			                         m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), bytes);
			line += fmt::format(" b{}={:x}/{:x}/{}{}", i, r.Base48(), r.NumRecords(), gpu_written ? "g" : "c",
			                    hash != 0 ? fmt::format("#{:08x}", static_cast<uint32_t>(hash)) : "");
		}
		for (size_t i = 0; i < program.info.images.size() && i < snapshot.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			if (r.Base40() != 0) line += fmt::format(" i{}={:x}", i, r.Base40());
		}
		line += fmt::format(" ud#{:08x}", static_cast<uint32_t>(XXH3_64bits(
		                                     snapshot.user_data.data(), snapshot.user_data.size() * 4u)));
		std::fprintf(stderr, "%s\n", line.c_str());
	}
	const bool trigger_use = !watch_compact && watch_trigger != nullptr && watched_use != UINT32_MAX &&
	                         trigger_on && triggered_uses.fetch_add(1) < 240u;
	if (trigger_use) {
		const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		const auto out   = [&](const std::string& text) {
			std::fprintf(stderr, "[watch] f=%" PRIu64 " %s\n", frame, text.c_str());
		};
		out(fmt::format("use={} buffers={} images={}", watched_use, program.info.buffers.size(),
		                program.info.images.size()));
		for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			const uint64_t bytes =
			    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
			const bool gpu_written = r.Base48() != 0 && bytes != 0 &&
			                         m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), bytes);
			out(fmt::format("  buffer[{}] written={} base={:012x} stride={} records={} gpu_written={} "
			                "words={:08x} {:08x} {:08x} {:08x}",
			                i, program.info.buffers[i].written ? 1 : 0, r.Base48(), r.Stride(),
			                r.NumRecords(), gpu_written ? 1 : 0, r.fields[0], r.fields[1],
			                r.fields[2], r.fields[3]));
			if (r.Base48() != 0 && bytes != 0 && bytes <= 4096) {
				std::vector<uint32_t> words((bytes + 3) / 4);
				if (Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), bytes)) {
					std::string line;
					for (const auto word: words) line += fmt::format(" {:08x}", word);
					out("    data:" + line);
				}
			}
		}
		for (size_t i = 0; i < program.info.images.size() && i < snapshot.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			const auto& res = program.info.images[i];
			out(fmt::format("  image[{}] written={} bindless={} dim={} base={:010x} format={} type={} "
			                "{}x{} depth={} base_array={} levels={}-{} min_lod={} words={:08x} {:08x} "
			                "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
			                i, res.written ? 1 : 0, res.bindless ? 1 : 0,
			                static_cast<uint32_t>(res.dimension), r.Base40(),
			                static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Type()),
			                static_cast<uint32_t>(r.Width5()) + 1u,
			                static_cast<uint32_t>(r.Height5()) + 1u,
			                static_cast<uint32_t>(r.Depth()) + 1u, static_cast<uint32_t>(r.BaseArray5()),
			                static_cast<uint32_t>(r.BaseLevel()), static_cast<uint32_t>(r.LastLevel()),
			                static_cast<uint32_t>(r.MinLod()), r.fields[0], r.fields[1], r.fields[2],
			                r.fields[3], r.fields[4], r.fields[5], r.fields[6], r.fields[7]));
		}
		{
			std::string line;
			for (const auto word: snapshot.user_data) line += fmt::format(" {:08x}", word);
			out("  user_data:" + line);
		}
		const auto& user = snapshot.user_data;
		for (size_t reg = 0; reg + 4 <= user.size(); reg += 4) {
			ShaderBufferResource r;
			r.fields[0] = user[reg];
			r.fields[1] = user[reg + 1];
			r.fields[2] = user[reg + 2];
			r.fields[3] = user[reg + 3];
			const uint64_t size = r.Stride() != 0 ? static_cast<uint64_t>(r.Stride()) * r.NumRecords()
			                                      : r.NumRecords();
			if (r.Base48() == 0 || size == 0 || size > (1u << 20u)) {
				continue;
			}
			std::vector<uint32_t> words(std::min<uint64_t>(size, 0x200u) / 4u);
			if (words.empty() ||
			    !Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), words.size() * 4u)) {
				continue;
			}
			std::string line;
			for (const auto word: words) line += fmt::format(" {:08x}", word);
			out(fmt::format("  user V# s{} base={:012x} gpu_written={}:", program.user_data_base + reg,
			                r.Base48(),
			                m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), words.size() * 4u)
			                    ? 1
			                    : 0) +
			    line);
		}
	}
	if (watched_use < 4u ||
	    (watched_use != UINT32_MAX && watch_every != 0 && watched_use % watch_every == 0 &&
	     watched_use / watch_every <= 8u) ||
	    in_watched_frame) {
		LOGF("WatchShader 0x%016" PRIx64 " use=%u frame=%" PRIu64 " stage=%u buffers=%zu images=%zu\n",
		     program.shader_hash, watched_use,
		     m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
		     static_cast<uint32_t>(program.stage), program.info.buffers.size(),
		     program.info.images.size());
		for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			// Whether the GPU has written bytes of the buffer that the CPU copy lacks.
			const uint64_t bytes =
			    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
			const bool gpu_written = r.Base48() != 0 && bytes != 0 &&
			                         m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), bytes);
			LOGF("  buffer[%zu]: pc=0x%x formatted=%d written=%d base=0x%012" PRIx64
			     " stride=%u records=%u format=%u dst_sel=%u%u%u%u gpu_written=%d\n",
			     i, program.info.buffers[i].first_use_pc, program.info.buffers[i].formatted ? 1 : 0,
			     program.info.buffers[i].written ? 1 : 0, r.Base48(), r.Stride(), r.NumRecords(),
			     r.RawFormat(), r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW(),
			     gpu_written ? 1 : 0);
			// Small buffers (constants, tables) are listed in full, as guest memory holds them.
			if (r.Base48() != 0 && bytes != 0 && bytes <= 4096 && !gpu_written) {
				std::vector<uint32_t> words((bytes + 3) / 4);
				if (Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), bytes)) {
					for (size_t row = 0; row < words.size(); row += 8) {
						std::string line;
						for (size_t k = row; k < std::min(words.size(), row + 8); k++) {
							line += fmt::format(" {:08x}", words[k]);
						}
						LOGF("    +0x%03zx:%s\n", row * 4, line.c_str());
					}
				}
			}
		}
		for (size_t i = 0; i < program.info.images.size() && i < snapshot.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			LOGF("  image[%zu]: bindless=%d format=%u dst_sel=%u%u%u%u type=%u extent=%ux%u\n", i,
			     program.info.images[i].bindless ? 1 : 0, static_cast<uint32_t>(r.Format()),
			     r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW(),
			     static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Width5()) + 1u,
			     static_cast<uint32_t>(r.Height5()) + 1u);
		}
		// User data, and the first bytes of every buffer a user-data V# points at: the shader
		// reads constants and heap indices from these with scalar loads.
		const auto& user = snapshot.user_data;
		for (size_t row = 0; row < user.size(); row += 8) {
			std::string line;
			for (size_t k = row; k < std::min(user.size(), row + 8); k++) {
				line += fmt::format(" {:08x}", user[k]);
			}
			LOGF("  user_data[s%zu]:%s\n", program.user_data_base + row, line.c_str());
		}
		for (size_t reg = 0; reg + 4 <= user.size(); reg += 4) {
			ShaderBufferResource r;
			r.fields[0]         = user[reg];
			r.fields[1]         = user[reg + 1];
			r.fields[2]         = user[reg + 2];
			r.fields[3]         = user[reg + 3];
			const uint64_t size = r.Stride() != 0 ? static_cast<uint64_t>(r.Stride()) * r.NumRecords()
			                                      : r.NumRecords();
			if (r.Base48() == 0 || size == 0) {
				continue;
			}
			std::vector<uint32_t> words(std::min<uint64_t>(size, 0x100u) / 4u);
			if (words.empty() || m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), words.size() * 4u) ||
			    !Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), words.size() * 4u)) {
				continue;
			}
			LOGF("  user V# s[%zu:%zu]: base=0x%012" PRIx64 " stride=%u records=%u\n",
			     program.user_data_base + reg, program.user_data_base + reg + 3, r.Base48(), r.Stride(),
			     r.NumRecords());
			LogWords(words.data(), words.size());
			std::lock_guard lock(g_watched_reread_mutex);
			g_watched_rereads.push_back(
			    {m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed) + 2u,
			     r.Base48(), words.size() * 4u});
		}
	}
	// Diagnostics: KYTY_TRACE_WRITERS=<hex address>[:<hex size>]: every shader bound with a
	// written buffer overlapping that range, once per (shader, buffer) ([writers] on stderr).
	{
		static const std::pair<uint64_t, uint64_t> target = [] {
			std::pair<uint64_t, uint64_t> range {0, 0};
			if (const char* value = std::getenv("KYTY_TRACE_WRITERS"); value != nullptr) {
				char* end    = nullptr;
				range.first  = std::strtoull(value, &end, 16);
				range.second = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 4u;
			}
			return range;
		}();
		// KYTY_TRACE_ORDER_FRAME=<n>: in presented frames n and n+1 every write is printed with a
		// sequence number, and so is every use of KYTY_TRACE_ORDER_HASH=<hash> (the consumer), to see
		// the order the GPU runs them in.
		static const uint64_t order_frame = [] {
			const char* value = std::getenv("KYTY_TRACE_ORDER_FRAME");
			return value != nullptr ? std::strtoull(value, nullptr, 10) : UINT64_MAX;
		}();
		static const uint64_t order_hash = [] {
			const char* value = std::getenv("KYTY_TRACE_ORDER_HASH");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		}();
		static std::atomic<uint64_t> order_seq {0};
		const uint64_t frame_now = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		static const uint64_t order_frames = [] {
			const char* value = std::getenv("KYTY_TRACE_ORDER_FRAMES");
			return value != nullptr ? std::strtoull(value, nullptr, 10) : 2ull;
		}();
		const bool in_order_window = order_frame != UINT64_MAX && frame_now >= order_frame &&
		                             frame_now < order_frame + order_frames;
		// Diagnostics: KYTY_TRACE_COPY=<hash>: every use of that shader prints each buffer's range,
		// whether its bytes are GPU-written, and dwords 0..2 and 16..18 of its CPU copy ([copy] lines,
		// up to KYTY_TRACE_COPY_MAX, default 400).
		{
			static const uint64_t copy_hash = [] {
				const char* value = std::getenv("KYTY_TRACE_COPY");
				return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
			}();
			static const uint32_t copy_max = [] {
				const char* value = std::getenv("KYTY_TRACE_COPY_MAX");
				return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 400u;
			}();
			static const uint64_t copy_from = [] {
				const char* value = std::getenv("KYTY_TRACE_COPY_FROM");
				return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
			}();
			static std::atomic<uint32_t> copy_logs {0};
			if (copy_hash != 0 && program.shader_hash == copy_hash && frame_now >= copy_from &&
			    copy_logs.fetch_add(1) < copy_max) {
				std::string line;
				for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
					const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
					const uint64_t bytes = std::max<uint64_t>(r.Stride(), 1u) * r.NumRecords();
					uint32_t       w[20] {};
					const bool     read  = r.Base48() != 0 && bytes >= 4u &&
					                  Libs::LibKernel::Memory::TryReadBacking(r.Base48(), w, std::min<uint64_t>(bytes, 80u));
					line += fmt::format(" | b{}{} 0x{:x}+0x{:x} dirty={} cpu[0..2]={:08x} {:08x} {:08x} cpu[16..18]={:08x} {:08x} {:08x}{}",
					                    i, program.info.buffers[i].written ? "w" : "r", r.Base48(), bytes,
					                    r.Base48() != 0 && bytes != 0 && m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), bytes) ? 1 : 0,
					                    w[0], w[1], w[2], w[16], w[17], w[18], read ? "" : " (unread)");
				}
				std::string user;
				for (size_t k = 0; k < snapshot.user_data.size() && k < 16; k++) {
					user += fmt::format(" {:08x}", snapshot.user_data[k]);
				}
				std::fprintf(stderr, "[copy] frame=%llu %016" PRIx64 " user:%s%s\n",
				             static_cast<unsigned long long>(frame_now), program.shader_hash, user.c_str(),
				             line.c_str());
			}
		}
		// Diagnostics: KYTY_KEY_CAPTURE=<hash>: the buffers that shader binds (from frame 2100), which
		// the [key-capture] print searches for the material records of the keys it captured.
		{
			static const uint64_t capture_hash = [] {
				const char* value = std::getenv("KYTY_KEY_CAPTURE");
				return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
			}();
			if (capture_hash != 0 && program.shader_hash == capture_hash && frame_now >= 2100u) {
				std::lock_guard lock(g_capture_buffers_mutex);
				if (g_capture_buffers.empty()) {
					// KYTY_KEY_CAPTURE_SRT=<user reg>:<off1>:<off2>: follow user[reg..reg+1] -> V# at
					// +off1 -> V# at its base +off2, and print both (the terrain's material block is
					// 28:80:1376).
					if (const char* chain = std::getenv("KYTY_KEY_CAPTURE_SRT"); chain != nullptr) {
						char*          end  = nullptr;
						const auto     reg  = std::strtoul(chain, &end, 10);
						const uint32_t off1 = end != nullptr && *end == ':' ? std::strtoul(end + 1, &end, 10) : 0u;
						const uint32_t off2 = end != nullptr && *end == ':' ? std::strtoul(end + 1, &end, 10) : 0u;
						if (reg + 1u < snapshot.user_data.size()) {
							const uint64_t root = (static_cast<uint64_t>(snapshot.user_data[reg + 1u] & 0xffffu) << 32u) |
							                      snapshot.user_data[reg];
							ShaderBufferResource v1 {};
							ShaderBufferResource v2 {};
							const bool r1 = Libs::LibKernel::Memory::TryReadBacking(root + off1, v1.fields, 16u);
							const bool r2 = r1 && Libs::LibKernel::Memory::TryReadBacking(v1.Base48() + off2, v2.fields, 16u);
							auto& cache = m_context.GetBufferCache();
							std::fprintf(stderr,
							             "[key-scan] srt root=0x%012" PRIx64 " (dirty=%d) V#1@+%u read=%d base=0x%012" PRIx64
							             " stride=%u records=%u (dirty=%d) V#2@+%u read=%d base=0x%012" PRIx64
							             " stride=%u records=%u fmt=%u\n",
							             root, cache.HasGpuDirtyBytes(root + off1, 16u) ? 1 : 0, off1, r1 ? 1 : 0,
							             v1.Base48(), v1.Stride(), v1.NumRecords(),
							             r1 && cache.HasGpuDirtyBytes(v1.Base48() + off2, 16u) ? 1 : 0, off2,
							             r2 ? 1 : 0, v2.Base48(), v2.Stride(), v2.NumRecords(), v2.RawFormat());
						}
					}
					for (size_t i = 0; i < snapshot.buffers.size(); i++) {
						const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
						const uint64_t bytes = std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
						g_capture_buffers.emplace_back(r.Base48(), bytes);
					}
				}
			}
		}
		if (in_order_window && order_hash != 0 && program.shader_hash == order_hash) {
			std::fprintf(stderr, "[order] #%llu frame=%llu consumer %016" PRIx64 "\n",
			             static_cast<unsigned long long>(order_seq.fetch_add(1)),
			             static_cast<unsigned long long>(frame_now), program.shader_hash);
		}
		if (target.first != 0) {
			static std::mutex                   writers_mutex;
			static std::unordered_set<uint64_t> writers_printed;
			if (in_order_window) {
				for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
					if (!program.info.buffers[i].written) {
						continue;
					}
					const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
					const uint64_t bytes = std::max<uint64_t>(r.Stride(), 1u) *
					                       static_cast<uint64_t>(r.NumRecords());
					if (r.Base48() != 0 && r.Base48() < target.first + target.second &&
					    target.first < r.Base48() + bytes) {
						std::fprintf(stderr,
						             "[order] #%llu frame=%llu writer %016" PRIx64 " 0x%012" PRIx64 "+0x%" PRIx64 "\n",
						             static_cast<unsigned long long>(order_seq.fetch_add(1)),
						             static_cast<unsigned long long>(frame_now), program.shader_hash,
						             r.Base48(), bytes);
					}
				}
			}
			for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
				if (!program.info.buffers[i].written) {
					continue;
				}
				const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
				const uint64_t bytes = std::max<uint64_t>(r.Stride(), 1u) *
				                       static_cast<uint64_t>(r.NumRecords());
				if (r.Base48() == 0 || r.Base48() >= target.first + target.second ||
				    target.first >= r.Base48() + bytes) {
					continue;
				}
				std::lock_guard lock(writers_mutex);
				if (writers_printed.insert(program.shader_hash ^ (static_cast<uint64_t>(i) << 56u)).second) {
					std::fprintf(stderr,
					             "[writers] %016" PRIx64 " stage=%u buffer[%zu] base=0x%012" PRIx64 "+0x%" PRIx64
					             " stride=%u formatted=%d fmt=%u\n",
					             program.shader_hash, static_cast<uint32_t>(program.stage), i, r.Base48(),
					             bytes, r.Stride(), program.info.buffers[i].formatted ? 1 : 0, r.RawFormat());
					std::fflush(stderr);
				}
			}
		}
	}
	// Diagnostics: KYTY_TRACE_READERS=<hash>[,<hash>...]: the buffer ranges those shaders write
	// are recorded, and every other shader bound with a buffer overlapping one is printed once
	// ([readers] lines on stderr). Finds the consumers of a shader's output.
	{
		static const std::vector<uint64_t> sources = [] {
			std::vector<uint64_t> result;
			if (const char* value = std::getenv("KYTY_TRACE_READERS"); value != nullptr) {
				std::string_view list(value);
				size_t           start = 0;
				while (start < list.size()) {
					const auto end = std::min(list.find(',', start), list.size());
					if (end > start) {
						result.push_back(std::strtoull(
						    std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
					}
					start = end + 1;
				}
			}
			return result;
		}();
		if (!sources.empty()) {
			struct Range {
				uint64_t base, size, source;
				int      level;
			};
			static std::mutex                             trace_mutex;
			static std::vector<Range>                     ranges;
			static std::unordered_set<uint64_t>           printed;
			// The consumers that write buffers become sources too (KYTY_TRACE_READERS_DEPTH
			// levels, default 2), so a chain (settings -> simulation -> rendering) shows up.
			static std::unordered_map<uint64_t, int>      levels;
			static const int max_level = [] {
				const char* value = std::getenv("KYTY_TRACE_READERS_DEPTH");
				return value != nullptr ? std::atoi(value) : 2;
			}();
			std::lock_guard lock(trace_mutex);
			if (levels.empty()) {
				for (const auto hash: sources) {
					levels[hash] = 0;
				}
			}
			const auto level_it  = levels.find(program.shader_hash);
			const bool is_source = level_it != levels.end();
			const int  level     = is_source ? level_it->second : -1;
			// With KYTY_TRACE_WRITERS set, only the dispatches that write into that range are logged.
			bool writes_target = true;
			if (const char* value = std::getenv("KYTY_TRACE_WRITERS"); value != nullptr && level == 0) {
				char*          end   = nullptr;
				const uint64_t first = std::strtoull(value, &end, 16);
				const uint64_t size  = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 16) : 4u;
				writes_target        = false;
				for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
					const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
					const uint64_t bytes = std::max<uint64_t>(r.Stride(), 1u) * r.NumRecords();
					writes_target |= program.info.buffers[i].written && r.Base48() < first + size &&
					                 first < r.Base48() + bytes;
				}
			}
			if (level == 0 && writes_target) {
				static std::atomic<uint32_t> source_logs {0};
				if (source_logs.fetch_add(1) < 6u) {
					std::string user;
					for (size_t k = 0; k < snapshot.user_data.size() && k < 16; k++) {
						user += fmt::format(" {:08x}", snapshot.user_data[k]);
					}
					std::fprintf(stderr, "[readers] source %016" PRIx64 " buffers=%zu/%zu dma=%d user:%s\n",
					             program.shader_hash, program.info.buffers.size(),
					             snapshot.buffers.size(), program.info.uses_dma ? 1 : 0, user.c_str());
					for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
						const auto r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
						std::fprintf(stderr,
						             "[readers]   buffer[%zu] base=0x%012" PRIx64 " stride=%u records=%u written=%d"
						             " formatted=%d fmt=%u dst_sel=%u%u%u%u words=%08x %08x %08x %08x gpu_written=%d\n",
						             i, r.Base48(), r.Stride(), r.NumRecords(),
						             program.info.buffers[i].written ? 1 : 0,
						             program.info.buffers[i].formatted ? 1 : 0, r.RawFormat(), r.DstSelX(),
						             r.DstSelY(), r.DstSelZ(), r.DstSelW(), r.fields[0], r.fields[1],
						             r.fields[2], r.fields[3],
						             r.Base48() != 0 && m_context.GetBufferCache().HasGpuDirtyBytes(
						                                    r.Base48(), std::max<uint64_t>(r.Stride(), 1u) *
						                                                    std::max<uint32_t>(r.NumRecords(), 1u))
						                 ? 1
						                 : 0);
						// KYTY_TRACE_READERS_DATA=<dwords>: the start of each read-only buffer's CPU copy.
						static const uint32_t data_dwords = [] {
							const char* value = std::getenv("KYTY_TRACE_READERS_DATA");
							return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
						}();
						if (data_dwords != 0u && !program.info.buffers[i].written && r.Base48() != 0) {
							std::vector<uint32_t> data(std::min<uint64_t>(
							    data_dwords, std::max<uint64_t>(r.Stride(), 1u) * r.NumRecords() / 4u));
							if (!data.empty() &&
							    Libs::LibKernel::Memory::TryReadBacking(r.Base48(), data.data(), data.size() * 4u)) {
								for (size_t row = 0; row < data.size(); row += 8) {
									std::string line;
									for (size_t k = row; k < std::min(data.size(), row + 8); k++) {
										line += fmt::format(" {:08x}", data[k]);
									}
									std::fprintf(stderr, "[readers]     +0x%03zx:%s\n", row * 4u, line.c_str());
								}
							}
						}
					}
				}
			}
			for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
				const auto     r     = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
				const uint64_t bytes = std::max<uint64_t>(r.Stride(), 1u) *
				                       static_cast<uint64_t>(r.NumRecords());
				if (r.Base48() == 0 || bytes == 0) {
					continue;
				}
				if (is_source) {
					// Huge arenas: only their first 64 KB (a writer of a fixed record at the base).
					const uint64_t traced = std::min<uint64_t>(bytes, 64u << 10u);
					if (program.info.buffers[i].written &&
					    std::ranges::none_of(ranges, [&](const Range& x) {
						    return x.base == r.Base48() && x.size == traced;
					    })) {
						ranges.push_back({r.Base48(), traced, program.shader_hash, level});
						std::fprintf(stderr,
						             "[readers] source L%d %016" PRIx64 " writes 0x%012" PRIx64
						             "+0x%" PRIx64 " stage=%u\n",
						             level, program.shader_hash, r.Base48(), bytes,
						             static_cast<uint32_t>(program.stage));
					}
				}
				for (size_t k = 0; k < ranges.size(); k++) {
					const auto x = ranges[k];
					if (x.source == program.shader_hash) {
						continue;
					}
					if (r.Base48() < x.base + x.size && x.base < r.Base48() + bytes &&
					    printed.insert(program.shader_hash ^ (static_cast<uint64_t>(i) << 56u)).second) {
						std::fprintf(stderr,
						             "[readers] L%d %016" PRIx64 " stage=%u buffer[%zu] base=0x%012" PRIx64
						             "+0x%" PRIx64 " written=%d overlaps %016" PRIx64
						             " 0x%012" PRIx64 "+0x%" PRIx64 " images=%zu\n",
						             x.level + 1, program.shader_hash,
						             static_cast<uint32_t>(program.stage), i, r.Base48(), bytes,
						             program.info.buffers[i].written ? 1 : 0,
						             x.source, x.base, x.size, program.info.images.size());
						std::fflush(stderr);
						if (!is_source && x.level + 1 <= max_level) {
							levels.emplace(program.shader_hash, x.level + 1);
						}
					}
				}
			}
		}
	}
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto binding = ResolveStageTexture(program.info.images[i], snapshot.images[i]);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		if (binding.dynamic_class) {
			BindImage(binding.alt_image_id, false);
		}
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.address) {
			continue; // v439: no table to survey
		}
		if (m_bindless_surveyed.insert(use.base ^ (static_cast<uint64_t>(use.table_offset) << 48u))
		        .second) {
			SurveyBindlessHeap(use);
		}
	}
	PrepareBindlessHeaps(runtime, prepared);
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	// Diagnostics: KYTY_LOG_SHADER_DATA=<hash>: that shader's user-data registers and the values
	// its shader data carries for them, for the first 8 preparations.
	if (static const std::string log_hashes = [] {
		    const char* value = std::getenv("KYTY_LOG_SHADER_DATA");
		    return value != nullptr ? std::string(value) : std::string();
	    }();
	    !log_hashes.empty() &&
	    log_hashes.find(fmt::format("{:016x}", program.shader_hash)) != std::string::npos) {
		static std::mutex                      layouts_mutex;
		static std::unordered_set<std::string> layouts;
		std::string layout_key = fmt::format("{:x}/{}/{}:", reinterpret_cast<uintptr_t>(&program),
		                                     program.bindings.ShaderDataDwords(),
		                                     program.bindings.push_data_start_dword);
		for (const auto reg: program.bindings.user_data_registers) {
			layout_key += fmt::format(" {}", reg);
		}
		bool fresh = false;
		{
			std::lock_guard lock(layouts_mutex);
			fresh = layouts.size() < 64 && layouts.insert(layout_key).second;
		}
		static std::atomic<uint32_t> logged {0};
		if (fresh || logged.fetch_add(1) < 4u) {
			std::string line = " program=" + layout_key;
			for (size_t i = 0; i < program.bindings.user_data_registers.size(); i++) {
				line += fmt::format(" s{}={:08x}", program.bindings.user_data_registers[i],
				                    prepared.shader_data[i]);
			}
			std::string raw;
			for (size_t k = 0; k < snapshot.user_data.size(); k++) {
				raw += fmt::format(" {:08x}", snapshot.user_data[k]);
			}
			std::fprintf(stderr,
			             "[shader-data] %016llx stage=%u vertex_offset_sgpr=%d instance_offset_sgpr=%d "
			             "push_start=%u dwords=%u base=%u count=%u:%s | user_data(%zu):%s\n",
			             static_cast<unsigned long long>(program.shader_hash),
			             program.info.vertex_offset_sgpr, program.info.instance_offset_sgpr,
			             static_cast<uint32_t>(program.stage), program.bindings.push_data_start_dword,
			             program.bindings.ShaderDataDwords(), program.user_data_base,
			             program.user_data_count, line.c_str(), snapshot.user_data.size(), raw.c_str());
			std::fflush(stderr);
		}
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	const CpuCost::Phase cpu_phase_("FindBuffers");
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       cache    = m_context.GetBufferCache();

	prepared.buffer_sources.clear();
	const auto& layout = program.bindings;
	if (layout.memory_offset_count == 0) {
		return;
	}
	const auto& resources = layout.descriptors.front().resources;
	prepared.buffer_sources.reserve(resources.size());
	for (const auto resource: resources) {
		auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
		const auto address = descriptor.Base48();
		const auto requested_size = descriptor.GetSize();
		if (address != 0 && address < BufferCache::CACHING_PAGESIZE) {
			// A base in guest page 0 cannot be read on the hardware either; the shader must not
			// touch it, so it binds as null. Logged so the descriptor's origin can be traced.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("RenderExecutor: buffer descriptor %u in guest page 0: stage=%s words=[0x%08x "
				     "0x%08x 0x%08x 0x%08x] size=0x%016" PRIx64 "\n",
				     resource, ShaderStageResourceName(program.stage), descriptor.fields[0],
				     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
				     requested_size);
			}
		}
		if (address < BufferCache::CACHING_PAGESIZE || requested_size == 0) {
			prepared.buffer_sources.push_back({});
			continue;
		}
		uint64_t size = 0;
		{
			KYTY_PROFILER_BLOCK("FindBuffers::ClampRangeSize");
			size = Libs::LibKernel::Memory::TryClampRangeSize(address, requested_size);
		}
		if (size == 0) {
			// Nothing is mapped at the base yet. Streaming titles bind regions they back later (a
			// dispatch can precede the mapping by thousands of calls), and an unbacked region reads
			// zero and drops writes on the hardware; binding null gives the same.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("RenderExecutor: buffer descriptor %u outside mapped memory: stage=%s "
				     "base=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
				     resource, ShaderStageResourceName(program.stage), address, requested_size);
			}
			prepared.buffer_sources.push_back({});
			continue;
		}
		// v0.5b: a range that crosses a host zone's edge is not looked up here (the lookup would
		// end the zone): a read gets a snapshot in ObtainBuffer, the zone stays.
		if (cache.CrossesHostZone(address, size)) {
			const auto& info = program.info.buffers[resource];
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				std::fprintf(stderr,
				             "[host-zone] crossing binding 0x%llx+0x%llx: shader %016llx stage=%s resource %u "
				             "written=%d formatted=%d table_read=%d\n",
				             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
				             static_cast<unsigned long long>(program.shader_hash),
				             ShaderStageResourceName(program.stage), resource, info.written ? 1 : 0,
				             info.formatted ? 1 : 0, info.table_read ? 1 : 0);
			}
			if (!info.written && !info.formatted && !info.table_read) {
				prepared.buffer_sources.push_back({address, size, BufferId {}});
				continue;
			}
		}
		prepared.buffer_sources.push_back({address, size, cache.FindBuffer(address, size)});
	}
}

// KYTY_REPEAT_STATS=1 (diagnostics, no behavior change): how many draws and dispatches bind exactly
// what one did in the previous guest frame (same program, buffer ranges and flattened SRT words) -
// the hit rate a reuse of last frame's prepared bindings could reach. [repeat] every 5 s.
static void NoteRepeat(uint64_t shader_hash, uint32_t stage,
                       const std::vector<PreparedBindings::BufferSource>& sources,
                       std::span<const uint32_t> flat) {
	static const bool enabled = std::getenv("KYTY_REPEAT_STATS") != nullptr;
	if (!enabled) {
		return;
	}
	uint64_t h = XXH3_64bits_withSeed(flat.data(), flat.size() * sizeof(uint32_t), shader_hash);
	// v438zk (next-phase plan, C2): the same without the flattened SRT words (the constants).
	uint64_t resources = shader_hash;
	for (const auto& src: sources) {
		const uint64_t words[2] = {src.address, src.size};
		h         = XXH3_64bits_withSeed(words, sizeof(words), h);
		resources = XXH3_64bits_withSeed(words, sizeof(words), resources);
	}
	// [repeat-stage]: per stage type, how many calls have the program and buffer ranges of the
	// previous call of that stage type (the previous draw's vertex stage, the previous draw's
	// pixel stage, the previous dispatch), and how many of those also the same constants. A stage
	// that repeats could keep its descriptor set bound and push only its constants (plan C2).
	// Buffers only: images go through the bindless table.
	struct StageRepeat {
		uint64_t calls = 0, same_resources = 0, same_all = 0, previous_resources = 0, previous_all = 0;
	};
	static StageRepeat stage_repeat[16];
	{
		auto& r = stage_repeat[stage & 15u];
		r.calls++;
		if (r.calls > 1 && r.previous_resources == resources) {
			r.same_resources++;
			if (r.previous_all == h) {
				r.same_all++;
			}
		}
		r.previous_resources = resources;
		r.previous_all       = h;
	}
	struct State {
		uint64_t                     frame = 0;
		std::unordered_set<uint64_t> current, previous;
		uint64_t calls[2] {}, prev_hits[2] {}, dup[2] {}, frames = 0;
		std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	};
	static State st;
	const auto frame = Common::FrameSkips::g_guest_flips.load(std::memory_order_relaxed);
	if (frame != st.frame) {
		st.frame = frame;
		st.previous.swap(st.current);
		st.current.clear();
		st.frames++;
	}
	const uint32_t k = stage == static_cast<uint32_t>(ShaderType::Compute) ? 1u : 0u;
	st.calls[k]++;
	if (st.previous.contains(h)) {
		st.prev_hits[k]++;
	}
	if (!st.current.insert(h).second) {
		st.dup[k]++;
	}
	if (const auto now = std::chrono::steady_clock::now(); now - st.last > std::chrono::seconds(5)) {
		const auto pct = [](uint64_t a, uint64_t b) { return b == 0 ? 0.0 : 100.0 * double(a) / double(b); };
		std::fprintf(stderr,
		             "[repeat] 5 s, %llu frames: graphics %llu (%.0f/frame) same as last frame %.1f %%, "
		             "repeated within frame %.1f %% | compute %llu (%.0f/frame) same as last frame %.1f "
		             "%%, repeated within frame %.1f %%\n",
		             static_cast<unsigned long long>(st.frames), static_cast<unsigned long long>(st.calls[0]),
		             st.frames ? double(st.calls[0]) / double(st.frames) : 0.0, pct(st.prev_hits[0], st.calls[0]),
		             pct(st.dup[0], st.calls[0]), static_cast<unsigned long long>(st.calls[1]),
		             st.frames ? double(st.calls[1]) / double(st.frames) : 0.0, pct(st.prev_hits[1], st.calls[1]),
		             pct(st.dup[1], st.calls[1]));
		std::string line = "[repeat-stage] 5 s, same program and buffers as the previous call of the stage type (and also the same constants):";
		for (uint32_t type = 0; type < 16; type++) {
			auto& r = stage_repeat[type];
			if (r.calls == 0) {
				continue;
			}
			line += fmt::format(" | {} {} calls: {:.1f} % ({:.1f} %)", ShaderStageResourceName(static_cast<ShaderType>(type)),
			                    r.calls, pct(r.same_resources, r.calls), pct(r.same_all, r.calls));
			r.calls = r.same_resources = r.same_all = 0; // the previous hashes stay
		}
		std::fprintf(stderr, "%s\n", line.c_str());
		st.calls[0] = st.calls[1] = st.prev_hits[0] = st.prev_hits[1] = st.dup[0] = st.dup[1] = 0;
		st.frames = 0;
		st.last   = now;
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	const CpuCost::Phase cpu_phase_("RebindBuffers");
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.memory_offset_count);
	const DiagShaderScope diag_shader_scope(program.shader_hash, program.stage);

	prepared.buffers.clear();
	prepared.buffers.reserve(layout.memory_offset_count);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	const bool write_tracking = layout.write_tracking;
	if (write_tracking) {
		const auto bitmap = m_context.GetBufferCache().WriteBitmapAddress();
		prepared.shader_data[layout.WriteTrackDword()]      = static_cast<uint32_t>(bitmap);
		prepared.shader_data[layout.WriteTrackDword() + 1u] = static_cast<uint32_t>(bitmap >> 32u);
	}
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t i = 0; i < layout.memory_offset_count; i++) {
		const auto resource = layout.descriptors.front().resources[i];
		uint32_t buffer_offset = 0;
		uint64_t tracked_base  = 0;
		// G6 (KYTY_G6): read only through the GPU address table. The buffer is kept present and
		// resident; the descriptor is a placeholder no load uses, and the length is 0.
		// KYTY_G6_VERIFY: the buffer stays resident for the table reads and is bound as well (the
		// shader compares the two reads).
		if (const auto& source = prepared.buffer_sources[i];
		    ShaderRecompiler::IR::G6Verify() && program.info.buffers[resource].table_read &&
		    source.size != 0 && source.address >= BufferCache::CACHING_PAGESIZE) {
			(void)m_context.GetBufferCache().PrepareTableRead(source.address, source.size, source.id);
		} else if (program.info.buffers[resource].table_read && source.size != 0 &&
		           source.address >= BufferCache::CACHING_PAGESIZE &&
		           m_context.GetBufferCache().PrepareTableRead(source.address, source.size, source.id)) {
			const CpuCost::Phase table_phase_("Rebind::TableRead");
			prepared.buffers.push_back(
			    ShaderRecompiler::Spirv::GetRobustBufferAccess()
			        ? vk::DescriptorBufferInfo {nullptr, 0, VK_WHOLE_SIZE}
			        : vk::DescriptorBufferInfo {m_context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16});
			if (write_tracking) {
				const auto dword                 = layout.WriteTrackDword() + 2u + 2u * i;
				prepared.shader_data[dword]      = 0;
				prepared.shader_data[dword + 1u] = 0;
			}
			prepared.shader_data[layout.BufferLengthDword() + i] = 0;
			continue;
		}
		if (m_delta_reuse != nullptr &&
		    DeltaSlotReusable(i, prepared.buffer_sources[i], program.info.buffers[resource].written,
		                      program.info.buffers[resource].table_read, program.info.buffers[resource].formatted,
		                      layout.memory_offset_dword)) {
			// v445b (KYTY_DISPATCH_DELTA): the previous dispatch's binding of the same read-only,
			// CPU-clean range; tracked_base stays 0 (only written bindings are write-tracked).
			prepared.buffers.push_back(m_delta_reuse->buffers[i]);
			buffer_offset =
			    (m_delta_reuse->shader_data[layout.memory_offset_dword + i / 4u] >> ((i % 4u) * 8u)) & 0xffu;
			m_delta_reused++;
		} else {
		prepared.buffers.push_back(NativeStorageBuffer(
		    m_context, prepared.buffer_sources[i], program.info.buffers[resource],
		    resource < snapshot.buffer_write_extents.size() ? &snapshot.buffer_write_extents[resource]
		                                                    : nullptr,
		    buffer_offset, write_tracking && program.info.buffers[resource].write_tracked,
		    tracked_base));
		}
		// KYTY_VERTEX_VERIFY: a read-only range of a checked draw is compared with guest memory.
		if (auto& verify = m_context.GetBufferCache();
		    verify.VertexVerifyArmed() && !program.info.buffers[resource].written) {
			const auto& bound = prepared.buffers.back();
			verify.VertexVerifyRange("storage buffer", i, program.shader_hash,
			                         prepared.buffer_sources[i].address,
			                         prepared.buffer_sources[i].size, bound.buffer,
			                         bound.offset + buffer_offset);
		}
		if (write_tracking) {
			const auto dword                     = layout.WriteTrackDword() + 2u + 2u * i;
			prepared.shader_data[dword]          = static_cast<uint32_t>(tracked_base);
			prepared.shader_data[dword + 1u]     = static_cast<uint32_t>(tracked_base >> 32u);
		}
		pack_memory_offset(i, buffer_offset);
		// The fallback allocation makes the Vulkan descriptor valid, but an empty guest
		// buffer still has no accessible elements. In particular, never let a null store
		// modify the shared fallback and affect subsequent null reads.
		// v443m: a callable split's modules check the length themselves (spirvEmitterMemoryHelpers.cpp):
		// a null descriptor (an address below the first page) has none, so they never read through it.
		prepared.shader_data[layout.BufferLengthDword() + i] =
		    prepared.buffer_sources[i].size == 0 || (layout.rt_fixed && !prepared.buffers.back().buffer)
		        ? 0u
		        : static_cast<uint32_t>(prepared.buffers.back().range / sizeof(uint32_t));
	}
	// KYTY_WATCH_BINDINGS=1 with KYTY_FRAME_SKIPS=1 (11-25, the opening-cinematic hang): for a watched
	// shader, the guest address each binding slot really points the shader at (host buffer's guest
	// base + descriptor offset + the offset byte the shader adds), against the guest range of its V#.
	// "MISMATCH" when they differ. A slot bound to a ring (stream/upload) prints "ring".
	if (static const bool watch_bindings = [] {
		    const char* value = std::getenv("KYTY_WATCH_BINDINGS");
		    return value != nullptr && value[0] == '1';
	    }();
	    watch_bindings && Common::FrameSkips::Enabled() && Common::FrameSkips::Watched(program.shader_hash)) {
		static std::atomic<uint32_t> lines {0};
		auto&                        cache = m_context.GetBufferCache();
		std::string                  text;
		bool                         mismatch = false;
		for (uint32_t i = 0; i < layout.memory_offset_count && i < prepared.buffers.size(); i++) {
			const auto& src   = prepared.buffer_sources[i];
			const auto& bound = prepared.buffers[i];
			const auto  adj   = (prepared.shader_data[layout.memory_offset_dword + i / 4u] >> ((i % 4u) * 8u)) & 0xffu;
			text += fmt::format(" [{} res={} {:x}:{:x}", i, layout.descriptors.front().resources[i], src.address, src.size);
			if (src.size == 0) {
				text += " null]";
				continue;
			}
			const auto* host = src.id ? &cache.GetBuffer(src.id) : nullptr;
			if (host != nullptr && host->Handle() == bound.buffer) {
				const auto seen = host->CpuAddress() + bound.offset + adj;
				text += fmt::format(" seen={:x} off={:x} adj={} range={:x}", seen, bound.offset, adj, bound.range);
				if (seen != src.address) {
					text += " MISMATCH";
					mismatch = true;
				}
			} else {
				text += fmt::format(" ring off={:x} adj={} range={:x}", bound.offset, adj, bound.range);
			}
			text += "]";
		}
		if (mismatch || lines.load(std::memory_order_relaxed) < 200) {
			if (lines.fetch_add(1) < 20000) {
				std::fprintf(stderr, "[watch-bind] flip=%llu sh=%016llx stage=%s%s\n",
				             static_cast<unsigned long long>(Common::FrameSkips::g_flip.load(std::memory_order_relaxed)),
				             static_cast<unsigned long long>(program.shader_hash),
				             ShaderStageResourceName(program.stage), text.c_str());
			}
		}
	}
	NoteRepeat(program.shader_hash, static_cast<uint32_t>(program.stage), prepared.buffer_sources,
	           snapshot.flattened_srt);
	// KYTY_SRT_VERIFY: the values the host used in place of GPU-written bytes, each followed by
	// a slot the GPU fills with its own copy (FillFlatSlotsOnGpu); the shader compares them.
	std::span<const ShaderRecompiler::IR::SrtVerify> verify;
	if (layout.srt_verify) {
		auto& cache = m_context.GetBufferCache();
		// The last values each shader used, to name the one that differed in a report.
		static std::unordered_map<uint64_t, std::vector<ShaderRecompiler::IR::SrtVerify>> last_used;
		std::array<uint32_t, 3> detail {};
		if (const auto missed = cache.TakeSrtVerifyReport(&detail); missed != 0) {
			const auto found = last_used.find(missed);
			const auto* used = found != last_used.end() && detail[0] < found->second.size()
			                       ? &found->second[detail[0]]
			                       : nullptr;
			std::fprintf(stderr,
			             "[srt-verify] CS %016llx pair %u: host 0x%08x gpu 0x%08x at 0x%llx\n",
			             static_cast<unsigned long long>(missed), detail[0], detail[1], detail[2],
			             static_cast<unsigned long long>(used != nullptr ? used->address : 0));
			SrtVerifyReportMiss(missed);
		}
		static const bool keep_used = Common::FpsLogDiagnostics();
		if (keep_used && !snapshot.srt_verify.empty()) {
			auto& kept = last_used[program.shader_hash];
			kept.assign(snapshot.srt_verify.begin(), snapshot.srt_verify.end());
		}
		const auto report = cache.SrtVerifyReportAddress();
		verify            = snapshot.srt_verify;
		if (report == 0) {
			EXIT_IF(!verify.empty());
		}
		prepared.shader_data[layout.VerifyDword() + 1u] = static_cast<uint32_t>(verify.size());
		prepared.shader_data[layout.VerifyDword() + 2u] = static_cast<uint32_t>(report);
		prepared.shader_data[layout.VerifyDword() + 3u] = static_cast<uint32_t>(report >> 32u);
	} else if (!snapshot.srt_verify.empty()) {
		EXIT("SRT verification values for shader %016" PRIx64 " without a verifying program\n",
		     program.shader_hash);
	}
	if (layout.clip_space) {
		for (uint32_t i = 0; i < 6; i++) {
			prepared.shader_data[layout.ClipSpaceDword() + i] = std::bit_cast<uint32_t>(prepared.clip_space[i]);
		}
	}
	if (layout.export_mapping) {
		prepared.shader_data[layout.ExportMappingDword()]      = prepared.export_mapping[0];
		prepared.shader_data[layout.ExportMappingDword() + 1u] = prepared.export_mapping[1];
	}
	if (layout.buffer_stride_count != 0 && prepared.runtime != nullptr && prepared.runtime->resources != nullptr) {
		const auto& buffers = prepared.runtime->resources->buffers;
		for (uint32_t i = 0; i < layout.buffer_stride_count; i++) {
			// v432: a dynamic-format buffer's word also carries its format and selects.
			const bool dynamic_format = i < program.info.buffers.size() && program.info.buffers[i].dynamic_format;
			prepared.shader_data[layout.BufferStrideDword() + i] =
			    i >= buffers.size() ? 0u
			    : dynamic_format    ? ShaderRecompiler::IR::BufferFormatData(buffers[i])
			                        : ShaderRecompiler::IR::BufferStrideData(buffers[i]);
		}
	}
	if ((layout.image_flags & ShaderRecompiler::IR::BindingLayout::ImageClassFlags) != 0) {
		// v429: bit i set when image i's T# is an unsigned-integer format (dynamic_class images).
		uint32_t bits = 0;
		for (uint32_t i = 0; i < program.info.images.size() && i < 32u; i++) {
			if (program.info.images[i].dynamic_class && i < snapshot.images.size()) {
				const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
				if (!descriptor.IsNull() && Prospero::SampledTextureNumericClass(descriptor.Format()) ==
				                                Prospero::TextureNumericClass::Uint) {
					bits |= 1u << i;
				}
			}
		}
		prepared.shader_data[layout.ImageClassDword()] = bits;
	}
	if ((layout.image_flags & ShaderRecompiler::IR::BindingLayout::ImageLayerFlags) != 0) {
		// v430: bit i set when image i's T# is a plain 2D texture (dynamic_layer images).
		uint32_t bits = 0;
		for (uint32_t i = 0; i < program.info.images.size() && i < 32u; i++) {
			if (program.info.images[i].dynamic_layer && i < snapshot.images.size()) {
				const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
				if (!descriptor.IsNull() && descriptor.Type() == Prospero::ImageType::kColor2D) {
					bits |= 1u << i;
				}
			}
		}
		prepared.shader_data[layout.ImageLayerDword()] = bits;
	}
	if ((layout.image_flags & ShaderRecompiler::IR::BindingLayout::ImageCubeFlags) != 0) {
		// v431: bit i set when image i's T# is a cube (dynamic_cube images).
		uint32_t bits = 0;
		for (uint32_t i = 0; i < program.info.images.size() && i < 32u; i++) {
			if (program.info.images[i].dynamic_cube && i < snapshot.images.size()) {
				const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
				if (!descriptor.IsNull() && descriptor.Type() == Prospero::ImageType::kCube) {
					bits |= 1u << i;
				}
			}
		}
		prepared.shader_data[layout.ImageCubeDword()] = bits;
	}
	if ((layout.image_flags & ShaderRecompiler::IR::BindingLayout::ImageDim3DFlags) != 0) {
		// v433: bit i set when image i's T# is a 2D texture (dynamic_3d images).
		uint32_t bits = 0;
		for (uint32_t i = 0; i < program.info.images.size() && i < 32u; i++) {
			if (program.info.images[i].dynamic_3d && i < snapshot.images.size()) {
				const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
				if (!descriptor.IsNull() && descriptor.Type() == Prospero::ImageType::kColor2D) {
					bits |= 1u << i;
				}
			}
		}
		prepared.shader_data[layout.ImageDim3DDword()] = bits;
	}
	// The guard-trip report's address (IR::BindingLayout::loop_guard; 0 when unavailable: the
	// shader then skips the report and only caps its loops).
	if (layout.loop_guard) {
		const auto report = m_context.GetBufferCache().LoopGuardReportAddress();
		prepared.shader_data[layout.GuardDword()]      = static_cast<uint32_t>(report);
		prepared.shader_data[layout.GuardDword() + 1u] = static_cast<uint32_t>(report >> 32u);
	}
	const auto with_verify = [&](std::span<const uint32_t> flat) -> std::span<const uint32_t> {
		if (layout.srt_verify) {
			prepared.shader_data[layout.VerifyDword()] = static_cast<uint32_t>(flat.size());
		}
		if (verify.empty()) {
			return flat;
		}
		m_verify_srt.assign(flat.begin(), flat.end());
		for (const auto& entry: verify) {
			m_verify_srt.push_back(entry.expected);
			m_verify_srt.push_back(~entry.expected); // replaced on the GPU
		}
		return m_verify_srt;
	};
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		const CpuCost::Phase srt_phase_("Rebind::FlatSrt");
		if (prepared.bindless_patches.empty()) {
			const auto flat = with_verify(snapshot.flattened_srt);
			if (RenderStats::MultiDrawProbeEnabled()) {
				uint64_t flat_hash = flat.size();
				for (const auto word: flat) {
					flat_hash ^= word + 0x9e3779b97f4a7c15ull + (flat_hash << 6u) + (flat_hash >> 2u);
				}
				RenderStats::MixProbe(flat_hash);
			}
			prepared.flattened_srt = NativeUpload(m_context, flat);
			FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills, verify,
			                   static_cast<uint32_t>(snapshot.flattened_srt.size()));
		} else {
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& [offset, region, entries]: prepared.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[offset]      = region;
					m_bindless_srt[offset + 1u] = entries;
				}
				// The first 256 distinct patches are logged; after that no draw takes the lock.
				static std::atomic<bool>            logging_done = false;
				static std::mutex                   logged_mutex;
				static std::unordered_set<uint64_t> logged;
				if (logging_done.load(std::memory_order_relaxed)) {
					continue;
				}
				std::scoped_lock lock(logged_mutex);
				if (logged.size() >= 256) {
					logging_done.store(true, std::memory_order_relaxed);
				} else if (logged.insert(program.shader_hash ^ offset).second) {
					LOGF("Bindless patch: stage=%u hash=0x%016" PRIx64
					     " offset=%u region=%u entries=%u srt_words=%zu\n",
					     static_cast<uint32_t>(program.stage), program.shader_hash, offset, region,
					     entries, m_bindless_srt.size());
				}
			}
			const auto base = static_cast<uint32_t>(m_bindless_srt.size());
			const auto flat = with_verify(m_bindless_srt);
			prepared.flattened_srt = NativeUpload(m_context, flat);
			FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills, verify, base);
		}
	}
	if (HtileWatch::Enabled()) {
		for (const auto& src: prepared.buffer_sources) {
			HtileWatch::NoteBuffer(program.shader_hash, static_cast<uint32_t>(program.stage), src.address,
			                       src.size, false);
		}
	}
	// KYTY_FRAME_SKIPS: the watched shaders' inputs as this draw resolved them, per flip (the
	// flicker hunt compares frames where the sewer wall loses its decals with frames where not).
	if (Common::FrameSkips::Enabled() && Common::FrameSkips::Watched(program.shader_hash)) {
		const auto hash_values = [](const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
			uint64_t h = 0x9e3779b97f4a7c15ull;
			for (const auto& v: values) {
				h = XXH3_64bits_withSeed(v.dwords.data(), v.dword_count * sizeof(uint32_t), h);
			}
			return h;
		};
		const auto hash_words = [](const std::vector<uint32_t>& words) {
			return words.empty() ? 0ull : XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
		};
		std::string bufs;
		for (const auto& src: prepared.buffer_sources) {
			bufs += fmt::format(" {:x}:{:x}", src.address, src.size);
		}
		std::fprintf(stderr,
		             "[decal-in] flip=%llu sh=%016llx bufs_h=%016llx imgs_h=%016llx srt_h=%016llx ud_h=%016llx sd_h=%016llx bufs=%s\n",
		             static_cast<unsigned long long>(Common::FrameSkips::g_flip.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(program.shader_hash),
		             static_cast<unsigned long long>(hash_values(snapshot.buffers)),
		             static_cast<unsigned long long>(hash_values(snapshot.images)),
		             static_cast<unsigned long long>(hash_words(snapshot.flattened_srt)),
		             static_cast<unsigned long long>(hash_words(snapshot.user_data)),
		             static_cast<unsigned long long>(hash_words(prepared.shader_data)), bufs.c_str());
		// KYTY_WATCH_COUNTS=1 (11-25, the opening-cinematic hang): the loop bounds a watched shader
		// reads, as guest memory holds them when the draw takes its copy (or, for a host zone, shortly
		// before the GPU reads them): per bound buffer, dword 127 (byte 508) and dword 151 (byte 604)
		// when in range, and for a buffer of 128-byte records the largest (dword 27 >> 16) (f9a169937ee2eacf's
		// per-light inner count at record byte 108) and how many records exceed 4096. Printed when it
		// differs from the previous line for the shader, at most KYTY_WATCH_COUNTS_MAX (3000) lines.
		static const bool watch_counts = [] {
			const char* value = std::getenv("KYTY_WATCH_COUNTS");
			return value != nullptr && value[0] == '1';
		}();
		if (watch_counts) {
			static const uint64_t max_lines = [] {
				const char* value = std::getenv("KYTY_WATCH_COUNTS_MAX");
				return value != nullptr ? std::strtoull(value, nullptr, 10) : 3000ull;
			}();
			static std::mutex                             counts_mutex;
			static std::unordered_map<uint64_t, std::string> last_line;
			static uint64_t                               lines = 0;
			std::string                                   text;
			for (const auto& src: prepared.buffer_sources) {
				if (src.size == 0) {
					continue;
				}
				text += fmt::format(" {:x}:{:x}", src.address, src.size);
				uint32_t word = 0;
				// dword 0 (a count at the head of a list, b9f49649da3973b7's volume loop) and whether
				// the GPU wrote the range (then guest memory may not hold the GPU's bytes yet).
				if (Libs::LibKernel::Memory::TryReadBacking(src.address, &word, 4)) {
					text += fmt::format(" @0={}", word);
				}
				text += m_context.GetBufferCache().IsRegionGpuModified(src.address, src.size) ? " gpu" : "";
				if (src.size >= 512 && Libs::LibKernel::Memory::TryReadBacking(src.address + 508, &word, 4)) {
					text += fmt::format(" @508={}", word);
				}
				if (src.size >= 608 && Libs::LibKernel::Memory::TryReadBacking(src.address + 604, &word, 4)) {
					text += fmt::format(" @604={}", word);
				}
				if (src.size >= 128 && src.size % 128 == 0 && src.size <= (1u << 20u)) {
					static thread_local std::vector<uint32_t> records;
					records.resize(src.size / 4);
					if (Libs::LibKernel::Memory::TryReadBacking(src.address, records.data(), src.size)) {
						uint32_t largest = 0, above = 0;
						for (size_t i = 27; i < records.size(); i += 32) {
							const auto count = records[i] >> 16u;
							largest          = std::max(largest, count);
							above += count > 4096 ? 1u : 0u;
						}
						text += fmt::format(" rec_max={} rec_over4k={}", largest, above);
					}
				}
			}
			std::lock_guard lock(counts_mutex);
			auto&           last = last_line[program.shader_hash];
			if (text != last && lines < max_lines) {
				lines++;
				last = text;
				std::fprintf(stderr, "[watch-counts] flip=%llu sh=%016llx%s\n",
				             static_cast<unsigned long long>(Common::FrameSkips::g_flip.load(std::memory_order_relaxed)),
				             static_cast<unsigned long long>(program.shader_hash), text.c_str());
			}
		}
		// KYTY_DECAL_CHK=1: the guest bytes of each bound buffer of at most KYTY_DECAL_CHK_MAX_KB
		// (default 16384) are hashed now, when the draw takes its copy of them, and again at a later
		// watched draw once the GPU has finished this one's command buffer (on this thread: a
		// completion callback reading guest memory deadlocks against an unmap holding the backing
		// store). A buffer whose bytes changed in between was still being written by the game (or
		// downloaded into) after the emulator read it: on the console the GPU reads those bytes at
		// execution, later than the emulator does here.
		static const bool     check     = [] {
			const char* value = std::getenv("KYTY_DECAL_CHK");
			return value != nullptr && value[0] == '1';
		}();
		static const uint64_t check_max = [] {
			const char* value = std::getenv("KYTY_DECAL_CHK_MAX_KB");
			return (value != nullptr ? std::strtoull(value, nullptr, 10) : 16384ull) * 1024ull;
		}();
		if (check) {
			struct Item {
				uint64_t address;
				uint64_t size;
				uint64_t hash;
			};
			struct Pending {
				std::vector<Item> items;
				uint64_t          flip;
				uint64_t          shader;
				uint64_t          tick;
			};
			static std::mutex          pending_mutex;
			static std::deque<Pending> pending;
			static uint64_t            checked = 0, changed_draws = 0;
			static thread_local std::vector<uint8_t> scratch;
			auto&                      scheduler = m_context.GetCommandScheduler();
			std::lock_guard            lock(pending_mutex);
			while (!pending.empty() && scheduler.IsFree(pending.front().tick)) {
				const auto& entry = pending.front();
				std::string changed;
				for (const auto& item: entry.items) {
					scratch.resize(item.size);
					if (!Libs::LibKernel::Memory::TryReadBacking(item.address, scratch.data(), item.size)) {
						changed += fmt::format(" {:x}:{:x}(unreadable)", item.address, item.size);
					} else if (XXH3_64bits(scratch.data(), item.size) != item.hash) {
						changed += fmt::format(" {:x}:{:x}", item.address, item.size);
					}
				}
				checked++;
				if (!changed.empty()) {
					changed_draws++;
					std::fprintf(stderr, "[decal-chk] flip=%llu sh=%016llx changed:%s\n",
					             static_cast<unsigned long long>(entry.flip),
					             static_cast<unsigned long long>(entry.shader), changed.c_str());
				}
				if (checked % 1000 == 0) {
					std::fprintf(stderr, "[decal-chk] %llu draws checked, %llu with bytes changed after the copy\n",
					             static_cast<unsigned long long>(checked),
					             static_cast<unsigned long long>(changed_draws));
				}
				pending.pop_front();
			}
			Pending entry {{}, Common::FrameSkips::g_flip.load(std::memory_order_relaxed), program.shader_hash,
			               scheduler.CurrentTick()};
			for (const auto& src: prepared.buffer_sources) {
				if (src.size == 0 || src.size > check_max) {
					continue;
				}
				scratch.resize(src.size);
				if (Libs::LibKernel::Memory::TryReadBacking(src.address, scratch.data(), src.size)) {
					entry.items.push_back({src.address, src.size, XXH3_64bits(scratch.data(), src.size)});
				}
			}
			if (pending.size() < 4096) {
				pending.push_back(std::move(entry));
			}
		}
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		const CpuCost::Phase data_phase_("Rebind::ShaderData");
		prepared.shader_data_buffer = NativeUploadShaderData(m_context, prepared.shader_data);
		// KYTY_LOG_SHADER_DATA: what is uploaded, for each new first dword.
		if (static const std::string log_hashes = [] {
			    const char* value = std::getenv("KYTY_LOG_SHADER_DATA");
			    return value != nullptr ? std::string(value) : std::string();
		    }();
		    !log_hashes.empty() &&
		    log_hashes.find(fmt::format("{:016x}", program.shader_hash)) != std::string::npos) {
			static std::mutex                   first_mutex;
			static std::unordered_set<uint32_t> firsts;
			bool fresh = false;
			{
				std::lock_guard lock(first_mutex);
				fresh = firsts.size() < 32 && firsts.insert(prepared.shader_data[0]).second;
			}
			if (fresh) {
				std::string line;
				for (size_t i = 0; i < prepared.shader_data.size() && i < 8; i++) {
					line += fmt::format(" {:08x}", prepared.shader_data[i]);
				}
				std::fprintf(stderr, "[shader-data-upload] %016llx offset=0x%llx size=%llu:%s\n",
				             static_cast<unsigned long long>(program.shader_hash),
				             static_cast<unsigned long long>(prepared.shader_data_buffer.offset),
				             static_cast<unsigned long long>(prepared.shader_data_buffer.range),
				             line.c_str());
				std::fflush(stderr);
			}
		}
	}
}

// KYTY_WATCH_IMAGE_ADDR=<hex guest address>: while the KYTY_WATCH_SHADER_TRIGGER file exists,
// each frame's first bind of the image at that address prints which host image it resolved to and
// that image's state, plus a per-layer count of nonzero guest bytes; every 8th frame (4 at most per
// trigger) the host image's layers and the guest bytes are written to _ShaderDump/wimg_*.bin.
static void WatchImageBind(TextureCache& cache, ImageId id, const TextureCache::ImageDesc& desc,
                           uint32_t slot, uint64_t shader_hash, uint64_t frame) {
	// Several addresses: KYTY_WATCH_IMAGE_ADDR=<hex>,<hex>,... (one line per frame, address and
	// host image each).
	static const std::vector<uint64_t> addresses = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_WATCH_IMAGE_ADDR");
		// "all": every image the watched shader (KYTY_WATCH_SHADER) binds; each is dumped once.
		if (value != nullptr && std::string_view(value) == "all") {
			result.push_back(0);
			return result;
		}
		while (value != nullptr && *value != '\0') {
			char* end = nullptr;
			result.push_back(std::strtoull(value, &end, 16));
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static const char* trigger = std::getenv("KYTY_WATCH_SHADER_TRIGGER");
	if (addresses.empty() || trigger == nullptr) {
		return;
	}
	cache.PollWatchImageDumps();
	// With KYTY_WATCH_SHADER set, only that shader's binds count (not for KYTY_GBUF_PATCHES).
	static const uint64_t only_shader = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	if (only_shader != 0 && shader_hash != only_shader) {
		return;
	}
	auto&    image   = cache.GetImage(id);
	uint64_t address = 0;
	const bool watch_all = addresses.size() == 1 && addresses[0] == 0;
	if (watch_all && only_shader != 0) {
		address = image.info.data.address;
	}
	for (const auto candidate: addresses) {
		if (watch_all) {
			break;
		}
		if (image.info.data.address == candidate ||
		    (image.info.HasStencil() && image.info.stencil.address == candidate)) {
			address = candidate;
			break;
		}
	}
	if (address == 0) {
		return;
	}
	const bool stencil_of_depth = image.info.HasStencil() && image.info.stencil.address == address;
	static uint64_t checked_frame = UINT64_MAX;
	static bool     on            = false;
	static uint64_t first_frame   = 0;
	static uint32_t lines         = 0;
	static uint32_t dumps         = 0;
	static uint64_t last_dump_frame = UINT64_MAX;
	static std::set<std::pair<uint64_t, uint32_t>> seen_this_frame;
	static uint64_t                                seen_frame = UINT64_MAX;
	if (seen_frame != frame) {
		seen_frame = frame;
		seen_this_frame.clear();
	}
	if (!seen_this_frame.emplace(address, id.index).second) {
		return; // one line per frame, address and host image
	}
	const bool first_of_frame = checked_frame != frame;
	checked_frame = frame;
	std::error_code error;
	const bool      now_on = std::filesystem::exists(trigger, error);
	if (now_on && !on) {
		first_frame = frame;
		dumps       = 0;
	}
	on = now_on;
	if (!on || lines >= 4000) {
		return;
	}
	lines++;
	const auto& info = image.info;
	std::string layer_stats;
	const auto  layers      = std::max<uint32_t>(info.resources.layers, 1u);
	const auto  layer_bytes = info.data.size / layers;
	if (!info.IsDepth() && layer_bytes != 0 && info.data.size <= (64ull << 20u)) {
		std::vector<uint8_t> guest(info.data.size);
		if (Libs::LibKernel::Memory::TryReadBacking(info.data.address, guest.data(), guest.size())) {
			for (uint32_t l = 0; l < layers && l < 64; l++) {
				uint64_t nonzero = 0;
				for (uint64_t b = 0; b < layer_bytes; b++) nonzero += guest[l * layer_bytes + b] != 0 ? 1 : 0;
				layer_stats += fmt::format(" {}", nonzero * 100 / layer_bytes);
			}
		} else {
			layer_stats = " unreadable";
		}
	}
	std::fprintf(stderr,
	             "[wimg] f=%llu shader=%016llx slot=%u id=%u.%u guest=%llx size=%llx %ux%ux%u layers=%u "
	             "levels=%u tile=%u fmt=%d bpb=%u registered=%d cpu_dirty=%d maybe_dirty=%d gpu_mod=%d "
	             "view_layers=%u+%u depth=%d stencil_of_depth=%d depth_id=%u stencil=%llx view_aspect=%u "
	             "view_fmt=%d nonzero%%/layer:%s\n",
	             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(shader_hash),
	             slot, id.index, id.generation, static_cast<unsigned long long>(info.data.address),
	             static_cast<unsigned long long>(info.data.size), info.extent.width,
	             info.extent.height, info.extent.depth, info.resources.layers, info.resources.levels,
	             static_cast<uint32_t>(info.tile_mode), static_cast<int>(info.pixel_format),
	             info.bytes_per_block, image.registered ? 1 : 0, image.IsDefinitelyCpuDirty() ? 1 : 0,
	             image.IsMaybeCpuDirty() ? 1 : 0, image.IsGpuModified() ? 1 : 0,
	             desc.view_info.base_layer, desc.view_info.layer_count, info.IsDepth() ? 1 : 0,
	             stencil_of_depth ? 1 : 0, image.depth_id.index,
	             static_cast<unsigned long long>(info.stencil.address),
	             static_cast<uint32_t>(desc.view_info.aspect),
	             static_cast<int>(desc.view_info.format), layer_stats.c_str());
	// KYTY_WATCH_IMAGE_DUMPS=<n>: n consecutive frames from the trigger instead of every 8th (4).
	static const uint32_t consecutive = [] {
		const char* value = std::getenv("KYTY_WATCH_IMAGE_DUMPS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	const bool dump_now = consecutive != 0 ? dumps < consecutive
	                                       : ((frame - first_frame) % 8 == 0 && dumps < 4);
	if (watch_all) {
		// Each image once, the first frames after the trigger.
		static std::set<uint64_t> dumped;
		if ((!info.IsDepth() || info.HasStencil()) && dumped.size() < 48 &&
		    dumped.insert(info.data.address).second) {
			cache.WatchImageDump(id, frame, fmt::format("_s{}", slot).c_str());
		}
		return;
	}
	if ((!info.IsDepth() || info.HasStencil()) && dump_now && first_of_frame &&
	    last_dump_frame != frame) {
		last_dump_frame = frame;
		dumps++;
		cache.WatchImageDump(id, frame, "_bind");
	}
}


// KYTY_GBUF_PATCHES=<pixel/compute shader hash> with KYTY_WATCH_SHADER_TRIGGER=<file>: the host
// images that shader binds are tracked; while the file exists, the first bind each frame of a
// tracked image by each shader reads back patches of it as it is at that point of the frame
// (TextureCache::WatchImagePatches, [gpatch] lines).
static void GbufPatchBind(TextureCache& cache, ImageId id, uint32_t slot, uint64_t shader_hash,
                          uint64_t frame) {
	static const uint64_t source = [] {
		const char* value = std::getenv("KYTY_GBUF_PATCHES");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	static const char* trigger = std::getenv("KYTY_WATCH_SHADER_TRIGGER");
	if (source == 0 || trigger == nullptr) {
		return;
	}
	cache.PollWatchImageDumps();
	static std::set<uint32_t>                      tracked;
	static uint64_t                                patch_frame = UINT64_MAX;
	static bool                                    patch_on    = false;
	static std::set<std::pair<uint32_t, uint64_t>> seen;
	static uint32_t                                count = 0;
	if (patch_frame != frame) {
		patch_frame = frame;
		seen.clear();
		std::error_code error;
		patch_on = std::filesystem::exists(trigger, error);
	}
	if (shader_hash == source) {
		tracked.insert(id.index);
	}
	static uint64_t calls = 0, source_calls = 0;
	calls++;
	source_calls += shader_hash == source ? 1 : 0;
	if ((calls % 20000) == 0) {
		std::fprintf(stderr, "[gpatch-dbg] calls=%llu source=%llu tracked=%zu on=%d count=%u frame=%llu\n",
		             static_cast<unsigned long long>(calls),
		             static_cast<unsigned long long>(source_calls), tracked.size(), patch_on ? 1 : 0,
		             count, static_cast<unsigned long long>(frame));
	}
	if (!patch_on || count >= 60000 || tracked.count(id.index) == 0 ||
	    !seen.emplace(id.index, shader_hash).second) {
		return;
	}
	count++;
	const auto& image = cache.GetImage(id);
	cache.WatchImagePatches(id, frame,
	                        fmt::format("img={} a={:x} s={:016x} slot={}", id.index,
	                                    image.info.data.address, shader_hash, slot));
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	const CpuCost::Phase cpu_phase_("RebindImages");
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			images[i] = ResolveStageTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
			if (images[i].dynamic_class) {
				BindImage(images[i].alt_image_id, false);
			}
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		if (binding.dynamic_class) {
			binding.alt_view = texture_cache.FindTexture(binding.alt_image_id, binding.alt_desc);
		}
		WatchImageBind(texture_cache, binding.image_id, binding.desc, i, program.shader_hash,
		               m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed));
		GbufPatchBind(texture_cache, binding.image_id, i, program.shader_hash,
		              m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed));
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
	}
}


// KYTY_BUF_PATCHES=<shader hash> with KYTY_WATCH_SHADER_TRIGGER=<file>: while the file exists, at
// that shader's first draw each frame every buffer it binds (its first 8 MB) is copied back in
// command order; per buffer, the 4 KB chunks whose contents changed since the previous capture
// are printed ([gbuf] f=<presented> b=<binding> a=<address> n=<chunks changed> idx:hash ...).
namespace {
struct BufCapture {
	std::unique_ptr<Buffer> buffer;
	uint64_t                bytes   = 0;
	uint64_t                tick    = 0;
	uint64_t                frame   = 0;
	uint64_t                address = 0;
	uint32_t                binding = 0;
};
std::vector<BufCapture>                         g_buf_captures;
std::map<std::pair<uint32_t, uint64_t>, std::vector<uint32_t>> g_buf_last;
} // namespace

static void PollBufCaptures(CommandScheduler& scheduler) {
	for (auto it = g_buf_captures.begin(); it != g_buf_captures.end();) {
		if (!scheduler.IsFree(it->tick)) {
			++it;
			continue;
		}
		it->buffer->Invalidate(0, it->bytes);
		const auto* data   = it->buffer->Mapped().data();
		const auto  chunks = static_cast<uint32_t>((it->bytes + 4095) / 4096);
		std::vector<uint32_t> hashes(chunks);
		for (uint32_t c = 0; c < chunks; c++) {
			uint32_t   hash = 2166136261u;
			const auto end  = std::min<uint64_t>(it->bytes, uint64_t {c + 1} * 4096);
			for (uint64_t b = uint64_t {c} * 4096; b < end; b++) {
				hash = (hash ^ data[b]) * 16777619u;
			}
			hashes[c] = hash;
		}
		auto&       last = g_buf_last[{it->binding, it->address}];
		std::string list;
		uint32_t    changed = 0;
		for (uint32_t c = 0; c < chunks; c++) {
			if (c >= last.size() || last[c] != hashes[c]) {
				changed++;
				if (changed <= 64) {
					list += fmt::format(" {}:{:08x}", c, hashes[c]);
				}
			}
		}
		std::fprintf(stderr, "[gbuf] f=%llu b=%u a=%llx bytes=%llx n=%u%s\n",
		             static_cast<unsigned long long>(it->frame), it->binding,
		             static_cast<unsigned long long>(it->address),
		             static_cast<unsigned long long>(it->bytes), changed, list.c_str());
		last = std::move(hashes);
		it   = g_buf_captures.erase(it);
	}
}

void RenderExecutor::CaptureBuffers(const PreparedBindings& prepared) {
	const CpuCost::Phase cpu_phase_("CaptureBuffers");
	static const uint64_t source = [] {
		const char* value = std::getenv("KYTY_BUF_PATCHES");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	static const char* trigger = std::getenv("KYTY_WATCH_SHADER_TRIGGER");
	if (source == 0 || trigger == nullptr) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	PollBufCaptures(scheduler);
	if (prepared.runtime == nullptr || prepared.runtime->program->shader_hash != source) {
		return;
	}
	const uint64_t  frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	static uint64_t done_frame = UINT64_MAX;
	if (done_frame == frame || g_buf_captures.size() > 256) {
		return;
	}
	done_frame = frame;
	std::error_code error;
	if (!std::filesystem::exists(trigger, error)) {
		return;
	}
	auto& cache = m_context.GetBufferCache();
	for (uint32_t i = 0; i < prepared.buffer_sources.size(); i++) {
		const auto& src = prepared.buffer_sources[i];
		if (src.address == 0 || src.size == 0) {
			continue;
		}
		auto& host = cache.GetBuffer(src.id);
		if (!host.Handle() || !host.IsInBounds(src.address, 1)) {
			continue;
		}
		const uint64_t bytes = std::min<uint64_t>({src.size, uint64_t {8} << 20u,
		                                           host.Size() - host.Offset(src.address)});
		if (bytes == 0) {
			continue;
		}
		BufCapture capture {std::make_unique<Buffer>(m_context.GetGraphics(), scheduler,
		                                             MemoryUsage::Download, 0,
		                                             vk::BufferUsageFlagBits::eTransferDst, bytes),
		                    bytes, 0, frame, src.address, i};
		scheduler.EndRendering();
		auto            command = scheduler.Current().Recorder();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
		                        nullptr);
		const vk::BufferCopy copy {host.Offset(src.address), 0, bytes};
		command.copyBuffer(host.Handle(), capture.buffer->Handle(), 1, &copy);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eMemoryRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
		                        {}, 1, &after, 0, nullptr, 0, nullptr);
		capture.tick = scheduler.CurrentTick();
		g_buf_captures.push_back(std::move(capture));
	}
}

// KYTY_RENDER_STATS, [draw-delta] (11-20): would the dispatch delta path's buffer take-over pay
// for draws? For a draw stage that runs the program the stage at the same position of the
// previous draw ran, in the same command buffer, the buffer slots by class and by whether their
// source (address, size, cache buffer) is the previous draw's:
// - read-only: bound from the cache buffer;
// - written: the program writes it (never taken over);
// - ring copy: the previous draw bound a record-time copy in the stream or upload buffer (a
//   small CPU-written range; never taken over);
// - table: read through the GPU address table (no binding).
// "take-over" is RenderExecutor::DeltaSlotReusable itself, asked before the stage's
// RebindBuffers with the previous draw's stage as the base: the slots a draw delta path would
// not resolve again. "unchanged read-only refused" are read-only slots with the same source it
// still refuses (CPU-dirty pages in the range, a formatted buffer, another host buffer).
// Counted only with the stats on; nothing is bound from the census.
namespace {
struct DrawDeltaStats {
	std::atomic<uint64_t> draws {0}, stages {0}, slots {0}, same {0}, same_slots {0};
	std::atomic<uint64_t> ro_same {0}, ro_changed {0}, written_same {0}, written_changed {0};
	std::atomic<uint64_t> ring_same {0}, ring_changed {0}, table {0}, takeover {0}, ro_refused {0};
	std::array<std::atomic<uint64_t>, 6> histogram {}; // slots left to resolve: 0, 1, 2, 3, 4, 5+
	std::atomic<int64_t> last_report {0};
};
DrawDeltaStats g_draw_delta;

void ReportDrawDelta() {
	auto&      g    = g_draw_delta;
	const auto now  = std::chrono::steady_clock::now().time_since_epoch();
	const auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
	auto       last = g.last_report.load(std::memory_order_relaxed);
	if (last == 0) {
		g.last_report.compare_exchange_strong(last, ms);
		return;
	}
	if (ms - last < 5000 || !g.last_report.compare_exchange_strong(last, ms)) {
		return;
	}
	const double seconds = static_cast<double>(ms - last) / 1000.0;
	const auto   stages  = g.stages.exchange(0);
	const auto   same    = g.same.exchange(0);
	const auto   slots   = g.slots.exchange(0);
	const auto   same_slots = g.same_slots.exchange(0);
	const double per     = same != 0 ? 1.0 / static_cast<double>(same) : 0.0;
	std::array<double, 6> h {};
	for (size_t k = 0; k < h.size(); k++) {
		h[k] = 100.0 * static_cast<double>(g.histogram[k].exchange(0)) * per;
	}
	const auto take = [&](std::atomic<uint64_t>& v) { return static_cast<double>(v.exchange(0)) * per; };
	const double ro_same = take(g.ro_same), ro_changed = take(g.ro_changed);
	const double wr_same = take(g.written_same), wr_changed = take(g.written_changed);
	const double ring_same = take(g.ring_same), ring_changed = take(g.ring_changed);
	const double table = take(g.table), ro_refused = take(g.ro_refused);
	const auto   takeover = g.takeover.exchange(0);
	std::fprintf(stderr,
	             "[draw-delta] %.0f draws/s, %.0f stages/s (%.1f buffer slots each), %.0f/s same program as "
	             "the previous draw's stage: per such stage %.1f slots = read-only %.2f same + %.2f changed, "
	             "written %.2f same + %.2f changed, ring copy %.2f same + %.2f changed, table %.2f | take-over "
	             "%.2f a stage (%.0f%% of their slots, %.0f%% of all slots), unchanged read-only refused %.2f | "
	             "slots left to resolve 0 %.0f%% 1 %.0f%% 2 %.0f%% 3 %.0f%% 4 %.0f%% 5+ %.0f%%\n",
	             static_cast<double>(g.draws.exchange(0)) / seconds, static_cast<double>(stages) / seconds,
	             stages != 0 ? static_cast<double>(slots) / static_cast<double>(stages) : 0.0,
	             static_cast<double>(same) / seconds, static_cast<double>(same_slots) * per, ro_same, ro_changed,
	             wr_same, wr_changed, ring_same, ring_changed, table, static_cast<double>(takeover) * per,
	             same_slots != 0 ? 100.0 * static_cast<double>(takeover) / static_cast<double>(same_slots) : 0.0,
	             slots != 0 ? 100.0 * static_cast<double>(takeover) / static_cast<double>(slots) : 0.0, ro_refused,
	             h[0], h[1], h[2], h[3], h[4], h[5]);
}
} // namespace

void RenderExecutor::NoteDrawDeltaBefore(uint32_t stage, const PreparedBindings& prepared) {
	if (stage >= m_draw_census.size() || prepared.runtime == nullptr || !*prepared.runtime) {
		return;
	}
	auto&       g       = g_draw_delta;
	const auto& program = *prepared.runtime->program;
	const auto& layout  = program.bindings;
	auto&       base    = m_draw_census[stage];
	auto&       cache   = m_context.GetBufferCache();
	const auto  count   = layout.memory_offset_count;
	g.stages.fetch_add(1, std::memory_order_relaxed);
	g.slots.fetch_add(count, std::memory_order_relaxed);
	const bool same = base.valid && base.program == static_cast<const void*>(&program) &&
	                  base.tick == m_context.GetCommandScheduler().CurrentTick() &&
	                  base.sources.size() == count && base.buffers.size() == count &&
	                  prepared.buffer_sources.size() == count;
	base.valid = false;
	if (!same) {
		return;
	}
	g.same.fetch_add(1, std::memory_order_relaxed);
	g.same_slots.fetch_add(count, std::memory_order_relaxed);
	const auto stream   = cache.GetUtilityBuffer(MemoryUsage::Stream).Handle();
	const auto upload   = cache.GetUtilityBuffer(MemoryUsage::Upload).Handle();
	const auto* reuse   = std::exchange(m_delta_reuse, &base);
	uint32_t   resolved = 0;
	for (uint32_t i = 0; i < count; i++) {
		const auto  resource  = layout.descriptors.front().resources[i];
		const auto& info      = program.info.buffers[resource];
		const auto& source    = prepared.buffer_sources[i];
		const auto& old       = base.sources[i];
		const bool  unchanged = old.address == source.address && old.size == source.size && old.id == source.id;
		const bool  ring      = base.buffers[i].buffer == stream || base.buffers[i].buffer == upload;
		const bool  read_only = !info.table_read && !info.written && !ring;
		if (info.table_read) {
			g.table.fetch_add(1, std::memory_order_relaxed);
		} else if (info.written) {
			(unchanged ? g.written_same : g.written_changed).fetch_add(1, std::memory_order_relaxed);
		} else if (ring) {
			(unchanged ? g.ring_same : g.ring_changed).fetch_add(1, std::memory_order_relaxed);
		} else {
			(unchanged ? g.ro_same : g.ro_changed).fetch_add(1, std::memory_order_relaxed);
		}
		if (DeltaSlotReusable(i, source, info.written, info.table_read, info.formatted, layout.memory_offset_dword)) {
			g.takeover.fetch_add(1, std::memory_order_relaxed);
		} else {
			resolved++;
			if (read_only && unchanged) {
				g.ro_refused.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
	m_delta_reuse = reuse;
	g.histogram[std::min(resolved, 5u)].fetch_add(1, std::memory_order_relaxed);
}

void RenderExecutor::NoteDrawDeltaAfter(uint32_t stage, const PreparedBindings& prepared) {
	if (stage >= m_draw_census.size() || prepared.runtime == nullptr || !*prepared.runtime) {
		return;
	}
	auto& base            = m_draw_census[stage];
	base.program          = static_cast<const void*>(&*prepared.runtime->program);
	base.tick             = m_context.GetCommandScheduler().CurrentTick();
	base.unregister_epoch = m_context.GetBufferCache().UnregisterEpoch();
	base.sources          = prepared.buffer_sources;
	base.buffers          = prepared.buffers;
	base.shader_data      = prepared.shader_data;
	base.valid            = true;
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma = false;
	for (auto* stage: stages) {
		FindBuffers(*stage);
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		KYTY_PROFILER_BLOCK("Draw::PrepareBda");
		m_context.PrepareBda();
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	for (auto* stage: stages) {
		CaptureBuffers(*stage);
	}
	KYTY_PROFILER_BLOCK("Draw::ColorTargets");
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
	if (RenderStats::Enabled()) {
		// [draw-delta]: the census asks before each stage's rebind (a rebind uploads and cleans
		// the range) and keeps what the rebind resolved for the next draw.
		g_draw_delta.draws.fetch_add(1, std::memory_order_relaxed);
		uint32_t position = 0;
		for (auto* stage: stages) {
			NoteDrawDeltaBefore(position, *stage);
			RebindBuffers(*stage);
			NoteDrawDeltaAfter(position, *stage);
			position++;
		}
		ReportDrawDelta();
		return;
	}
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

static bool RebindAfterJoinEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_REBIND_AFTER_JOIN");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool RenderExecutor::RebindAfterJoin(std::span<PreparedBindings* const> stages, uint64_t& epoch,
                                     const char* kind) {
	auto& cache = m_context.GetBufferCache();
	if (cache.UnregisterEpoch() == epoch) {
		return false;
	}
	static std::atomic<uint32_t> seen {0};
	const auto n = seen.fetch_add(1, std::memory_order_relaxed);
	if (n < 16 || (n & (n - 1)) == 0) {
		std::fprintf(stderr, "[rebind-after-join] a %s's buffers lost one to a join (%u so far)%s\n", kind,
		             n + 1, RebindAfterJoinEnabled() ? "; resolved again" : "");
	}
	if (!RebindAfterJoinEnabled()) {
		epoch = cache.UnregisterEpoch();
		return false; // reported only
	}
	for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != epoch; pass++) {
		epoch = cache.UnregisterEpoch();
		for (auto* stage: stages) {
			RebindBuffers(*stage);
		}
	}
	return true;
}

// KYTY_DESC_STATS=1 (diagnostics, off by default): every 5 s, [desc-stats] counts the descriptor
// commits, their writes and descriptors, push-descriptor against fresh-set commits, and the
// commits whose buffers and images are exactly those the previous commit with the same pipeline
// layout wrote (an unchanged commit: what a binding template or set reuse could skip).
namespace {
struct DescStats {
	std::mutex                                 mutex;
	int64_t                                    window_start = 0;
	uint64_t                                   commits      = 0;
	uint64_t                                   writes       = 0;
	uint64_t                                   descriptors  = 0;
	uint64_t                                   pushed       = 0;
	uint64_t                                   fresh_sets   = 0;
	uint64_t                                   unchanged    = 0;
	std::unordered_map<const void*, uint64_t> last_by_layout;
};

bool DescStatsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DESC_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

uint64_t DescHashMix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
	return h;
}

void NoteDescCommit(const void* layout, std::span<const vk::WriteDescriptorSet> writes,
                    std::span<const vk::DescriptorBufferInfo> buffers,
                    std::span<const vk::DescriptorImageInfo> images, bool pushed) {
	uint64_t hash        = 0;
	uint64_t descriptors = 0;
	for (const auto& write: writes) {
		hash = DescHashMix(hash, write.dstBinding);
		hash = DescHashMix(hash, write.descriptorCount);
		hash = DescHashMix(hash, static_cast<uint64_t>(write.descriptorType));
		descriptors += write.descriptorCount;
	}
	for (const auto& info: buffers) {
		hash = DescHashMix(hash, reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer)));
		hash = DescHashMix(hash, info.offset);
		hash = DescHashMix(hash, info.range);
	}
	for (const auto& info: images) {
		hash = DescHashMix(hash, reinterpret_cast<uint64_t>(static_cast<VkSampler>(info.sampler)));
		hash = DescHashMix(hash, reinterpret_cast<uint64_t>(static_cast<VkImageView>(info.imageView)));
		hash = DescHashMix(hash, static_cast<uint64_t>(info.imageLayout));
	}
	static DescStats stats;
	std::lock_guard  lock(stats.mutex);
	stats.commits++;
	stats.writes += writes.size();
	stats.descriptors += descriptors;
	(pushed ? stats.pushed : stats.fresh_sets)++;
	auto [it, inserted] = stats.last_by_layout.try_emplace(layout, hash);
	if (!inserted) {
		if (it->second == hash) {
			stats.unchanged++;
		}
		it->second = hash;
	}
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	if (stats.window_start == 0) {
		stats.window_start = now;
	} else if (now - stats.window_start >= 5000) {
		const double seconds = static_cast<double>(now - stats.window_start) / 1000.0;
		std::fprintf(stderr,
		             "[desc-stats] commits=%.0f/s writes=%.0f/s descriptors=%.0f/s push=%.0f/s "
		             "fresh-sets=%.0f/s unchanged=%.0f/s (%.0f%%) layouts=%zu\n",
		             static_cast<double>(stats.commits) / seconds,
		             static_cast<double>(stats.writes) / seconds,
		             static_cast<double>(stats.descriptors) / seconds,
		             static_cast<double>(stats.pushed) / seconds,
		             static_cast<double>(stats.fresh_sets) / seconds,
		             static_cast<double>(stats.unchanged) / seconds,
		             100.0 * static_cast<double>(stats.unchanged) /
		                 static_cast<double>((std::max)(stats.commits, uint64_t {1})),
		             stats.last_by_layout.size());
		std::fflush(stderr);
		stats.commits = stats.writes = stats.descriptors = stats.pushed = stats.fresh_sets =
		    stats.unchanged                                                   = 0;
		stats.window_start                                                    = now;
	}
}
} // namespace

void NoteTickWorkViews(const uint64_t* views, size_t count, const uint32_t* regions,
                       size_t num_regions);                    // diagnosticCheckpoints.cpp
void NoteTickWorkBuffers(const uint64_t* triples, size_t count); // diagnosticCheckpoints.cpp

void RenderExecutor::NoteFaultMapBindings(bool with_views) const {
	static const bool fault_map = [] { const char* v = std::getenv("KYTY_FAULT_MAP"); return v != nullptr && v[0] == '1'; }();
	if (!fault_map) {
		return;
	}
	std::array<uint64_t, 48> triples {};
	size_t                   count = 0;
	for (const auto& info: m_descriptor_buffers) {
		if (count == 16) break;
		triples[count * 3]     = reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer));
		triples[count * 3 + 1] = info.offset;
		triples[count * 3 + 2] = info.range;
		count++;
	}
	if (with_views) {
		std::array<uint64_t, 24> views {};
		size_t                   num_views = 0;
		for (const auto& info: m_descriptor_images) {
			if (num_views == views.size()) break;
			views[num_views++] = reinterpret_cast<uint64_t>(static_cast<VkImageView>(info.imageView));
		}
		std::array<uint32_t, 6> regions {};
		size_t                  num_regions = 0;
		for (const auto region: m_draw_bindless_regions) {
			if (num_regions == regions.size()) break;
			regions[num_regions++] = region;
		}
		NoteTickWorkViews(views.data(), num_views, regions.data(), num_regions);
	}
	NoteTickWorkBuffers(triples.data(), count);
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	const CpuCost::Phase cpu_phase_("CommitBindings");
	KYTY_PROFILER_FUNCTION();
	m_last_commit.generation++;
	m_last_commit.stage0_data_slot  = -1;
	m_last_commit.stage0_push_start = -1;
	m_draw_bindless_regions.clear();
	for (const auto* prepared: prepared_bindings) {
		if (prepared == nullptr) continue;
		for (const auto& patch: prepared->bindless_patches) {
			m_draw_bindless_regions.push_back(patch[1]);
		}
	}
	const auto vk_buffer    = buffer.Recorder();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	// v434: the stages' push ranges must not overlap (anchored layout); a clean stop beats a GPU
	// reading another stage's data.
	uint64_t push_used = 0;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += program.bindings.descriptors.size();
		for (const auto& binding: program.bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// Scalar reads the host specialized on must not overlap what this draw or dispatch writes.
	// The written ranges are gathered once, and a reader's reads are compared one by one only
	// against a range that meets their span: a per-read scan of every bound image and written
	// buffer cost the jungle 1.6 fps (11.1 against 12.7, warm).
	struct WrittenRange {
		uint64_t                                address = 0;
		uint64_t                                size    = 0;
		const ShaderRecompiler::IR::CompiledShaderInfo* writer = nullptr; // null: an image or target
		uint32_t                                resource = 0;
		uint64_t                                descriptor_address = 0;
		uint64_t                                descriptor_size    = 0;
		bool                                    bounded  = false;
	};
	std::vector<WrittenRange> written_ranges;
	bool                      written_gathered = false;
	for (const auto* reader: prepared_bindings) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		if (!written_gathered) {
			written_gathered = true;
			for (const auto* writer: prepared_bindings) {
				if (writer->runtime->program->has_address_writes) {
					EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
				}
			}
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0) {
						written_ranges.push_back({.address = written.address, .size = written.size});
					}
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				const auto& extents = writer->runtime->resources->buffer_write_extents;
				for (uint32_t i = 0; i < writer->buffer_sources.size(); ++i) {
					const auto resource = program.bindings.descriptors.front().resources[i];
					if (!program.info.buffers[resource].written) continue;
					const auto& written = writer->buffer_sources[i];
					// The bytes the dispatch can store to, when its stores are bounded
					// (BufferWriteExtent): a title's heap-wide V# otherwise covers the whole
					// heap and every descriptor read in it.
					auto       written_address = written.address;
					auto       written_size    = written.size;
					const bool bounded = resource < extents.size() && extents[resource].valid;
					if (bounded) {
						written_address += extents[resource].begin;
						written_size = extents[resource].end > extents[resource].begin
						                   ? extents[resource].end - extents[resource].begin
						                   : 0;
					}
					if (written_size != 0) {
						written_ranges.push_back({.address            = written_address,
						                          .size               = written_size,
						                          .writer             = &program,
						                          .resource           = resource,
						                          .descriptor_address = written.address,
						                          .descriptor_size    = written.size,
						                          .bounded            = bounded});
					}
				}
			}
		}
		if (written_ranges.empty()) break;
		uint64_t span_begin = UINT64_MAX;
		uint64_t span_end   = 0;
		for (const auto [address, size]: reads) {
			span_begin = std::min(span_begin, address);
			span_end   = std::max(span_end, address + size);
		}
		for (const auto& written: written_ranges) {
			if (span_end <= span_begin ||
			    !ImageRangeOverlaps(span_begin, span_end - span_begin, written.address, written.size)) {
				continue;
			}
			for (const auto [address, size]: reads) {
				if (!ImageRangeOverlaps(address, size, written.address, written.size)) continue;
				if (written.writer == nullptr) {
					EXIT("scalar resource reads overlap an image or attachment write\n");
				}
				// Upstream (6409be28) exits here. Wolverine's CS 0x85a58319a7a75e48 reads a
				// branch input from a 256-byte buffer the same dispatch updates; the host
				// evaluates it from the state before the dispatch, as the walk before this
				// check did (and the game ran correctly). Reported once per shader pair.
				static std::mutex                   reported_mutex;
				static std::unordered_set<uint64_t> reported;
				const auto reader_hash = reader->runtime->program->shader_hash;
				std::scoped_lock lock(reported_mutex);
				if (reported.insert(reader_hash ^ (written.writer->shader_hash << 1u)).second) {
					LOGF("Warning: scalar resource reads overlap a shader buffer write: "
					     "reader 0x%016" PRIx64 " reads 0x%" PRIx64 "+0x%" PRIx64
					     ", writer 0x%016" PRIx64 " buffer %u writes 0x%" PRIx64 "+0x%" PRIx64
					     " (descriptor 0x%" PRIx64 "+0x%" PRIx64 ", extent %s)\n",
					     reader_hash, address, size, written.writer->shader_hash, written.resource,
					     written.address, written.size, written.descriptor_address,
					     written.descriptor_size, written.bounded ? "bounded" : "none");
				}
				break;
			}
		}
	}
	m_descriptor_buffers.clear();
	// KYTY_MULTI_DRAW_PROBE: the per-draw stream uploads (shader data, flattened SRT) are left
	// out of the binding hash; the flattened SRT's contents are hashed where it is uploaded.
	thread_local std::vector<size_t> probe_stream_slots;
	probe_stream_slots.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);

	for (auto* prepared: prepared_bindings) {
		const bool  stage0        = prepared == prepared_bindings.front();
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		// Bindless textures are sampled through set 1 with a read-only layout. A heap holds
		// thousands of resolved images; one whose images' states did not change since it was
		// last checked (g_bindless_state_generation) needs a look only at the images resolved
		// since. KYTY_BINDLESS_VERIFY=1 checks every image and stops if a skipped one needed it.
		static const bool verify = std::getenv("KYTY_BINDLESS_VERIFY") != nullptr;
		if (BindlessEngineEnabled() && !verify && !descriptors.bindless_heaps.empty()) {
			// Engine method, E5: only images settled since the last bindless draw, and pinned
			// images whose state changed (NoteBindlessStateChange), are made readable again; a
			// heap of thousands of images is not walked whenever any one of them changed.
			auto&      cache         = m_context.GetTextureCache();
			const auto make_readable = [&](ImageId id) {
				auto* image = cache.m_slot_images.try_get(id);
				if (image == nullptr || !image->registered || !image->bindless_pinned ||
				    image->binding.is_target || image->info.data.Empty() ||
				    m_bindless_storage.contains(static_cast<uint64_t>(id.index) |
				                                (static_cast<uint64_t>(id.generation) << 32u))) {
					return;
				}
				const auto layout = image->info.IsDepth()
				                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                        : vk::ImageLayout::eShaderReadOnlyOptimal;
				if (image->backing.state.layout != layout) {
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			};
			for (auto* heap: descriptors.bindless_heaps) {
				for (const auto id: heap->unchecked) {
					make_readable(id);
				}
				heap->unchecked.clear();
			}
			{
				std::scoped_lock lock {g_bindless_state_changes.mutex};
				m_bindless_changed.swap(g_bindless_state_changes.ids);
			}
			for (const auto id: m_bindless_changed) {
				make_readable(id);
			}
			m_bindless_changed.clear();
		}
		using HeapSpan = std::span<BindlessTable::Heap* const>;
		for (auto* heap: BindlessEngineEnabled() && !verify ? HeapSpan {}
		                                                    : HeapSpan {descriptors.bindless_heaps}) {
			const auto& resolved = heap->resolved;
			size_t      begin =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			         ? heap->checked_count
			         : 0;
			if (verify) {
				begin = 0;
			}
			const auto skipped =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			        ? heap->checked_count
			        : 0;
			for (size_t i = begin; i < resolved.size(); i++) {
				auto* image = m_context.GetTextureCache().m_slot_images.try_get(resolved[i]);
				if (image == nullptr || !image->registered || image->binding.is_target ||
				    image->info.data.Empty() ||
				    m_bindless_storage.contains(static_cast<uint64_t>(resolved[i].index) |
				                                (static_cast<uint64_t>(resolved[i].generation) << 32u))) {
					continue;
				}
				const auto layout = image->info.IsDepth()
				                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                        : vk::ImageLayout::eShaderReadOnlyOptimal;
				if (image->backing.state.layout != layout) {
					if (verify && i < skipped) {
						EXIT("bindless: image %u of heap 0x%016" PRIx64
						     " changed layout without a state generation change\n",
						     static_cast<uint32_t>(i), heap->base);
					}
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			}
			// After the transitions above, which bump the generation themselves.
			heap->checked_generation = g_bindless_state_generation.load(std::memory_order_relaxed);
			heap->checked_count      = resolved.size();
		}
		// Bindless storage images: a draw whose shader may store to them commits the write again,
		// as a bound storage image does every draw (the texture cache then treats the memory as
		// GPU-written), and keeps them in the general layout, read-write.
		for (auto* heap: descriptors.bindless_heaps) {
			if (heap->binding != ShaderRecompiler::IR::BindlessStorage2D) {
				continue;
			}
			auto& cache = m_context.GetTextureCache();
			for (const auto id: heap->resolved) {
				auto* image = cache.m_slot_images.try_get(id);
				if (image == nullptr || !image->registered || image->depth_id ||
				    image->binding.needs_rebind || image->binding.is_target ||
				    image->info.data.Empty()) {
					continue;
				}
				const auto found = m_bindless_storage.find(
				    static_cast<uint64_t>(id.index) | (static_cast<uint64_t>(id.generation) << 32u));
				if (found == m_bindless_storage.end()) {
					continue;
				}
				(void)cache.FindTexture(id, found->second);
				image->Transit(vk::ImageLayout::eGeneral,
				               vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {},
				               vk_buffer);
			}
		}
		const Image::GuestTransitScope guest_transits;
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			if (ImageHistory::Enabled()) {
				ImageHistory::NoteBind(static_cast<VkImage>(image.backing.image),
				                       m_context.GetCommandScheduler().CurrentTick(), program.shader_hash,
				                       binding.desc.type == TextureCache::BindingType::Storage ? 'S' : 'D');
			}
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
			if (binding.dynamic_class) {
				// The null image of the other class (v429): never a target, empty data.
				auto&      null_image = m_context.GetTextureCache().GetImage(binding.alt_image_id);
				const auto& null_view = binding.alt_desc.view_info;
				null_image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
				                   ImageSubresourceRange {null_view.base_level, null_view.level_count,
				                                          null_view.base_layer, null_view.layer_count},
				                   vk_buffer);
				binding.alt_layout = null_image.backing.state.layout;
			}
		}

		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != binding.resources.size());
						for (const auto& view: descriptors.buffers) {
							// A null descriptor only with device bounds checks (NativeStorageBuffer).
							EXIT_IF(view.buffer == nullptr &&
							        !ShaderRecompiler::Spirv::GetRobustBufferAccess());
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (binding.kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (binding.kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						if (stage0 && binding.kind == BindingKind::ShaderData) {
							m_last_commit.stage0_data_slot =
							    static_cast<int32_t>(m_descriptor_buffers.size() - 1u);
						}
						if (binding.kind == BindingKind::FlattenedSrt ||
						    binding.kind == BindingKind::ShaderData) {
							probe_stream_slots.push_back(m_descriptor_buffers.size() - 1u);
						}
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].dynamic_class ? 2u
			    : descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		// v443i: the slots of a callable split's fixed set layout that the program does not fill
		// (its arrays have capacities, and it has every non-array binding) are written too, as null
		// descriptors: a read through one gives zero. They were left as the set's memory had them,
		// which in a pool used before is whatever another set had there, views of images destroyed
		// since included. Nothing should read them; three device losses (g443e, g443g, g443h) were a
		// read inside a destroyed image none of the dispatch's own views named.
		// KYTY_RT_NULL_SLOTS=0: left unwritten, as before.
		static const bool null_slots = [] {
			const char* value = std::getenv("KYTY_RT_NULL_SLOTS");
			return value == nullptr || value[0] != '0';
		}();
		if (null_slots && pipeline.ray_tracing && program.bindings.rt_fixed &&
		    ShaderRecompiler::Spirv::GetRobustBufferAccess()) {
			thread_local std::vector<vk::DescriptorImageInfo>  null_images, spare_samplers;
			thread_local std::vector<vk::DescriptorBufferInfo> null_buffers;
			thread_local std::array<vk::DescriptorBufferInfo, 2> address_buffers;
			const auto fixed = ShaderRecompiler::IR::RtFixedSetLayout(program.bindings);
			uint32_t   most  = 1;
			for (const auto& slot: fixed) {
				most = std::max(most, slot.count);
			}
			const vk::Sampler spare = descriptors.samplers.empty() ? vk::Sampler {} : descriptors.samplers.front();
			null_images.assign(most, vk::DescriptorImageInfo {nullptr, nullptr, vk::ImageLayout::eGeneral});
			spare_samplers.assign(most, vk::DescriptorImageInfo {spare, nullptr, vk::ImageLayout::eUndefined});
			null_buffers.assign(most, vk::DescriptorBufferInfo {nullptr, 0, VK_WHOLE_SIZE});
			for (const auto& slot: fixed) {
				const auto*    binding = ShaderRecompiler::IR::FindBinding(program.bindings, slot.kind);
				const uint32_t written = binding != nullptr ? NativeDescriptorCount(*binding) : 0u;
				if (written >= slot.count) {
					continue;
				}
				vk::WriteDescriptorSet write {};
				write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, slot.kind);
				write.dstArrayElement = written;
				write.descriptorType  = NativeDescriptorType(slot.kind);
				write.descriptorCount = slot.count - written;
				if (ShaderRecompiler::IR::ImageBindingResourceClass(slot.kind) !=
				    ShaderRecompiler::IR::ImageResourceClass::None) {
					write.pImageInfo = null_images.data();
				} else if (slot.kind == BindingKind::Samplers) {
					if (!spare) {
						continue; // a sampler cannot be null, and the program has none to stand in
					}
					write.pImageInfo = spare_samplers.data();
				} else if (slot.kind == BindingKind::BdaPagetable || slot.kind == BindingKind::FaultBuffer) {
					// Every module of the split can translate an address: the real tables.
					auto&       cache  = m_context.GetBufferCache();
					const auto* buffer = slot.kind == BindingKind::BdaPagetable ? cache.GetBdaPageTableBuffer()
					                                                            : cache.GetFaultBuffer();
					auto& info = address_buffers[slot.kind == BindingKind::BdaPagetable ? 0u : 1u];
					info       = vk::DescriptorBufferInfo {buffer->Handle(), 0, buffer->Size()};
					write.pBufferInfo = &info;
				} else {
					write.pBufferInfo = null_buffers.data();
				}
				m_descriptor_writes.push_back(write);
			}
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			const auto start = program.bindings.push_data_start_dword;
			EXIT_IF(start + shader_data_dwords > ShaderRecompiler::IR::PushData::DwordCount);
			const uint64_t range = (shader_data_dwords >= 64u ? ~uint64_t {0} : ((uint64_t {1} << shader_data_dwords) - 1u)) << start;
			if ((push_used & range) != 0u) {
				EXIT("push data of shader %016" PRIx64 " (dwords %u-%u) overlaps another stage's\n",
				     program.shader_hash, start, start + shader_data_dwords - 1u);
			}
			push_used |= range;
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
			if (stage0) {
				m_last_commit.stage0_push_start =
				    static_cast<int32_t>(program.bindings.push_data_start_dword);
			}
		}
	}

	if ((pipeline_bind_point == vk::PipelineBindPoint::eGraphics ||
	     pipeline_bind_point == vk::PipelineBindPoint::eRayTracingKHR) &&
	    pipeline.push_stages) {
		// v440 M3: a callable split's ray-tracing pipeline pushes to its ray-generation and
		// callable stages.
		push_stages = pipeline.push_stages;
	}
	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}
	static_assert(sizeof(m_last_commit.push_dwords) == sizeof(push_data.dwords));
	m_last_commit.bind_point            = pipeline_bind_point;
	m_last_commit.layout                = pipeline.pipeline_layout;
	m_last_commit.set_layout            = pipeline.descriptor_set_layout;
	m_last_commit.uses_push_descriptors = pipeline.uses_push_descriptors;
	m_last_commit.ps_set_layout         = pipeline.ps_set_layout;
	m_last_commit.ps_uses_push          = pipeline.ps_uses_push;
	m_last_commit.pre_writes            = m_descriptor_writes.size();
	m_last_commit.has_push_data         = has_push_data;
	m_last_commit.push_stages           = push_stages;
	if (has_push_data && RenderStats::MultiDrawReplayConfigured()) {
		m_last_commit.push_dwords = push_data.dwords;
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (RenderStats::MultiDrawProbeEnabled()) {
			uint64_t hash = DescHashMix(0, reinterpret_cast<uint64_t>(
			                                   static_cast<VkPipeline>(pipeline.pipeline)));
			for (size_t slot = 0; slot < m_descriptor_buffers.size(); slot++) {
				if (std::ranges::find(probe_stream_slots, slot) != probe_stream_slots.end()) {
					continue;
				}
				const auto& info = m_descriptor_buffers[slot];
				hash = DescHashMix(hash, reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer)));
				hash = DescHashMix(hash, info.offset);
				hash = DescHashMix(hash, info.range);
			}
			for (const auto& info: m_descriptor_images) {
				hash = DescHashMix(hash, reinterpret_cast<uint64_t>(static_cast<VkImageView>(info.imageView)));
				hash = DescHashMix(hash, reinterpret_cast<uint64_t>(static_cast<VkSampler>(info.sampler)));
			}
			RenderStats::NoteCommitHash(hash);
		}
		if (DescStatsEnabled()) {
			NoteDescCommit(static_cast<const void*>(static_cast<VkPipelineLayout>(pipeline.pipeline_layout)),
			               m_descriptor_writes, m_descriptor_buffers, m_descriptor_images,
			               pipeline.uses_push_descriptors);
		}
		if (pipeline.ps_set_layout != nullptr) {
			// KYTY_STAGE_SETS: the pixel shader's writes (its NativeBinding group) to set 2.
			const auto group = static_cast<uint32_t>(BindingKind::Count);
			const auto split = std::stable_partition(
			    m_descriptor_writes.begin(), m_descriptor_writes.end(),
			    [&](const vk::WriteDescriptorSet& write) {
				    return write.dstBinding < group || write.dstBinding >= 2u * group;
			    });
			m_last_commit.pre_writes = static_cast<size_t>(split - m_descriptor_writes.begin());
			CommitSetWrites(m_context, vk_buffer, pipeline_bind_point, pipeline.pipeline_layout, 0,
			                pipeline.descriptor_set_layout, pipeline.uses_push_descriptors,
			                std::span(m_descriptor_writes.data(), m_last_commit.pre_writes));
			CommitSetWrites(m_context, vk_buffer, pipeline_bind_point, pipeline.pipeline_layout,
			                ShaderRecompiler::IR::PixelDescriptorSet, pipeline.ps_set_layout,
			                pipeline.ps_uses_push,
			                std::span(m_descriptor_writes.data() + m_last_commit.pre_writes,
			                          m_descriptor_writes.size() - m_last_commit.pre_writes));
		} else if (pipeline.uses_push_descriptors) {
			PushDescriptorsDelta(vk_buffer, pipeline_bind_point, pipeline.pipeline_layout);
		} else {
			if (pipeline_bind_point != vk::PipelineBindPoint::eGraphics) {
				m_dispatch_delta.writes_valid = false; // a bound set, not pushed descriptors
			}
			// v442: a callable split's fixed set layout holds hundreds of descriptors (capacities,
			// IR/RtCallable.h) and is used a few times a frame: few sets are allocated ahead.
			const auto set = m_context.GetDescriptorHeap().Commit(
			    pipeline.descriptor_set_layout, pipeline.ray_tracing ? 4u : DescriptorHeap::DescriptorSetBatch);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			// The update moves to the recording thread with the bind: the set is fresh.
			vk_buffer.updateAndBindDescriptorSet(
			    m_context.GetGraphics().device, pipeline_bind_point, pipeline.pipeline_layout, set,
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data());
		}
	}
	if (pipeline.uses_bindless) {
		const auto set = m_context.GetGraphics().bindless_set;
		EXIT_IF(set == nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout,
		                             pipeline.bindless_set, 1, &set, 0, nullptr);
	}
	if (pipeline.independent_sets) {
		// KYTY_GPL: a set the stages leave empty is still bound (an empty set; its layout is
		// EmptySetLayout's twin), so no set of a linked pipeline is ever undefined.
		const bool pre_bound = !m_descriptor_writes.empty() && m_last_commit.pre_writes != 0;
		const bool ps_bound  = !m_descriptor_writes.empty() &&
		                      m_last_commit.pre_writes != m_descriptor_writes.size();
		const auto empty     = EmptyDescriptorSet(m_context.GetGraphics());
		if (!pre_bound) {
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &empty,
			                             0, nullptr);
		}
		if (!pipeline.uses_bindless || pipeline.bindless_set != 1) {
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 1, 1, &empty,
			                             0, nullptr);
		}
		if (!ps_bound) {
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout,
			                             ShaderRecompiler::IR::PixelDescriptorSet, 1, &empty, 0,
			                             nullptr);
		}
	}
}

// v445b (KYTY_DISPATCH_DELTA). A slot of the dispatch being rebound may take the base's binding when
// - the shader only reads it, not through the address table, and not as a formatted (texel) buffer,
//   which an obtain may refresh from an image the GPU wrote over the same memory;
// - its source (guest range and cache buffer) is the base's, and no buffer was unregistered or
//   joined since (the epoch RebindAfterJoin and multi-draw replay use);
// - the base bound the cache's own buffer, not a copy of the guest's bytes made at record time
//   (the stream and upload buffers: such ranges are not tracked, their bytes may have changed),
//   and the cache buffer's host buffer is still that one;
// - no page of it is CPU-modified now: a guest write since the base faulted and marked it.
// GPU writes to the range land in the same cache buffer, so they need no new binding.
bool RenderExecutor::DeltaSlotReusable(uint32_t slot, const PreparedBindings::BufferSource& source, bool written,
                                       bool table_read, bool formatted, uint32_t offset_dword) const {
	const auto& base  = *m_delta_reuse;
	auto&       cache = m_context.GetBufferCache();
	if (written || table_read || formatted || slot >= base.sources.size() || slot >= base.buffers.size() ||
	    static_cast<size_t>(offset_dword) + slot / 4u >= base.shader_data.size() ||
	    source.address < BufferCache::CACHING_PAGESIZE || source.size == 0 ||
	    cache.UnregisterEpoch() != base.unregister_epoch || SyncToggle::ForceUpload()) {
		return false;
	}
	const auto& old   = base.sources[slot];
	const auto  bound = base.buffers[slot].buffer;
	if (old.address != source.address || old.size != source.size || !source.id || !(old.id == source.id) ||
	    !bound || bound == cache.GetUtilityBuffer(MemoryUsage::Stream).Handle() ||
	    bound == cache.GetUtilityBuffer(MemoryUsage::Upload).Handle() ||
	    // The cache buffer still has the host buffer the base bound (sharing replaces it in place).
	    cache.GetBuffer(source.id).Handle() != bound) {
		return false;
	}
	return !cache.IsRegionCpuModified(source.address, source.size);
}

// v445b (KYTY_DISPATCH_DELTA). Pushes m_descriptor_writes to set 0. At the compute bind point the
// writes are also kept, flattened to values; when the dispatch says the bind point still holds
// what the previous push left (m_delta_push: same pipeline, same command buffer, nothing bound or
// recorded raw since), only the writes whose values differ are pushed: a push descriptor stays as
// it is until it is pushed again or the bind point is disturbed with another layout.
void RenderExecutor::PushDescriptorsDelta(const CommandRecorder& recorder, vk::PipelineBindPoint bind_point,
                                          vk::PipelineLayout layout) {
	const bool delta = std::exchange(m_delta_push, false);
	const auto count = static_cast<uint32_t>(m_descriptor_writes.size());
	m_delta_writes   = count;
	m_delta_pushed   = count;
	if (bind_point != vk::PipelineBindPoint::eCompute || !DispatchDeltaEnabled()) {
		if (bind_point != vk::PipelineBindPoint::eGraphics) {
			m_dispatch_delta.writes_valid = false;
		}
		recorder.pushDescriptorSetKHR(bind_point, layout, 0, count, m_descriptor_writes.data());
		return;
	}
	// binding << 32 | element, count << 32 | type, then three words per descriptor.
	auto& flat = m_delta_flat;
	flat.clear();
	bool comparable = true;
	for (const auto& write: m_descriptor_writes) {
		flat.push_back((uint64_t {write.dstBinding} << 32u) | write.dstArrayElement);
		flat.push_back((uint64_t {write.descriptorCount} << 32u) | static_cast<uint32_t>(write.descriptorType));
		comparable = comparable && write.pNext == nullptr;
		for (uint32_t i = 0; i < write.descriptorCount; i++) {
			switch (write.descriptorType) {
				case vk::DescriptorType::eSampler:
				case vk::DescriptorType::eCombinedImageSampler:
				case vk::DescriptorType::eSampledImage:
				case vk::DescriptorType::eStorageImage:
				case vk::DescriptorType::eInputAttachment: {
					const auto& info = write.pImageInfo[i];
					flat.push_back(reinterpret_cast<uint64_t>(static_cast<VkSampler>(info.sampler)));
					flat.push_back(reinterpret_cast<uint64_t>(static_cast<VkImageView>(info.imageView)));
					flat.push_back(static_cast<uint64_t>(info.imageLayout));
					break;
				}
				case vk::DescriptorType::eUniformBuffer:
				case vk::DescriptorType::eStorageBuffer:
				case vk::DescriptorType::eUniformBufferDynamic:
				case vk::DescriptorType::eStorageBufferDynamic: {
					const auto& info = write.pBufferInfo[i];
					flat.push_back(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer)));
					flat.push_back(info.offset);
					flat.push_back(info.range);
					break;
				}
				case vk::DescriptorType::eUniformTexelBuffer:
				case vk::DescriptorType::eStorageTexelBuffer:
					flat.push_back(reinterpret_cast<uint64_t>(static_cast<VkBufferView>(write.pTexelBufferView[i])));
					flat.push_back(0);
					flat.push_back(0);
					break;
				default: comparable = false; break; // values in an extension structure: always pushed
			}
		}
	}
	auto& base   = m_dispatch_delta;
	bool  pushed = false;
	if (delta && comparable && base.writes_valid && base.layout == layout && base.write_count == count &&
	    base.writes.size() == flat.size()) {
		m_delta_changed.clear();
		size_t at     = 0;
		bool   shaped = true; // the same bindings, elements, counts and types in the same order
		for (uint32_t i = 0; i < count && shaped; i++) {
			const size_t words = 2u + 3u * static_cast<size_t>(m_descriptor_writes[i].descriptorCount);
			shaped = at + words <= flat.size() && flat[at] == base.writes[at] && flat[at + 1u] == base.writes[at + 1u];
			if (shaped && !std::equal(flat.begin() + static_cast<std::ptrdiff_t>(at + 2u),
			                          flat.begin() + static_cast<std::ptrdiff_t>(at + words),
			                          base.writes.begin() + static_cast<std::ptrdiff_t>(at + 2u))) {
				m_delta_changed.push_back(m_descriptor_writes[i]);
			}
			at += words;
		}
		if (shaped) {
			if (!m_delta_changed.empty()) {
				recorder.pushDescriptorSetKHR(bind_point, layout, 0, static_cast<uint32_t>(m_delta_changed.size()),
				                              m_delta_changed.data());
			}
			m_delta_pushed = static_cast<uint32_t>(m_delta_changed.size());
			pushed         = true;
		}
	}
	if (!pushed) {
		recorder.pushDescriptorSetKHR(bind_point, layout, 0, count, m_descriptor_writes.data());
	}
	base.writes.swap(flat);
	base.layout       = layout;
	base.write_count  = count;
	base.writes_valid = comparable;
}

bool RenderExecutor::RecommitShaderData(CommandBuffer& buffer, std::span<const uint32_t> data) {
	const auto& commit = m_last_commit;
	if (commit.stage0_data_slot < 0 && commit.stage0_push_start < 0) {
		return true; // the stage has no shader data (nothing to change)
	}
	auto&      scheduler = m_context.GetCommandScheduler();
	const auto tick      = scheduler.CurrentTick();
	vk::DescriptorBufferInfo uploaded;
	if (commit.stage0_data_slot >= 0) {
		uploaded = NativeUploadShaderData(m_context, data);
		if (scheduler.CurrentTick() != tick || !buffer.IsRendering()) {
			return false;
		}
	}
	const auto vk_buffer = buffer.Recorder();
	if (commit.stage0_push_start >= 0) {
		const auto start = static_cast<size_t>(commit.stage0_push_start);
		EXIT_IF(start + data.size() > m_last_commit.push_dwords.size());
		std::ranges::copy(data, m_last_commit.push_dwords.begin() + static_cast<std::ptrdiff_t>(start));
		vk_buffer.pushConstants(commit.layout, commit.push_stages, 0,
		                        static_cast<uint32_t>(sizeof(m_last_commit.push_dwords)),
		                        m_last_commit.push_dwords.data());
	}
	if (commit.stage0_data_slot >= 0) {
		EXIT_IF(static_cast<size_t>(commit.stage0_data_slot) >= m_descriptor_buffers.size() ||
		        m_descriptor_writes.empty());
		// The writes point into m_descriptor_buffers; the recorder copies them when it records.
		m_descriptor_buffers[static_cast<size_t>(commit.stage0_data_slot)] = uploaded;
		if (commit.ps_set_layout != nullptr) {
			// v438z: only the set that holds the changed slot. The first stage's shader data is in
			// set 0; the pixel shader's set (set 2) was committed with this layout and nothing of it
			// changed, so it stays bound: binding set 0 again with the same pipeline layout does not
			// disturb a higher set. Before, every replayed sub-draw whose shader data lives in a
			// buffer allocated, wrote and bound the pixel shader's set again as well.
			// KYTY_RECOMMIT_PS_SET=1 restores that.
			static const bool both_sets = [] {
				const char* value = std::getenv("KYTY_RECOMMIT_PS_SET");
				return value != nullptr && value[0] == '1';
			}();
			const auto* changed = &m_descriptor_buffers[static_cast<size_t>(commit.stage0_data_slot)];
			bool        in_pre  = false;
			for (size_t i = 0; i < commit.pre_writes && i < m_descriptor_writes.size(); i++) {
				const auto& write = m_descriptor_writes[i];
				if (write.pBufferInfo != nullptr && write.pBufferInfo <= changed &&
				    changed < write.pBufferInfo + write.descriptorCount) {
					in_pre = true;
					break;
				}
			}
			CommitSetWrites(m_context, vk_buffer, commit.bind_point, commit.layout, 0, commit.set_layout,
			                commit.uses_push_descriptors,
			                std::span(m_descriptor_writes.data(), commit.pre_writes));
			if (both_sets || !in_pre) {
				CommitSetWrites(m_context, vk_buffer, commit.bind_point, commit.layout,
				                ShaderRecompiler::IR::PixelDescriptorSet, commit.ps_set_layout,
				                commit.ps_uses_push,
				                std::span(m_descriptor_writes.data() + commit.pre_writes,
				                          m_descriptor_writes.size() - commit.pre_writes));
			}
		} else if (commit.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(commit.bind_point, commit.layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(commit.set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			vk_buffer.updateAndBindDescriptorSet(m_context.GetGraphics().device, commit.bind_point,
			                                     commit.layout, set,
			                                     static_cast<uint32_t>(m_descriptor_writes.size()),
			                                     m_descriptor_writes.data());
		}
	}
	return true;
}

} // namespace Libs::Graphics
