#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectDescriptorProbes = 65536u;

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto base_level = (descriptor.dwords[3] >> 12u) & 0xfu;
		const auto fragments  = (descriptor.dwords[3] >> 16u) & 0xfu;
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && fragments >= 1 && fragments <= 3 &&
		       (r128 || max_mip == fragments);
	}
	return true;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

enum class SamplerClass : uint8_t { Float, Integer, PointInteger };

SamplerClass ClassifySampler(const ImageResource& image) {
	if (image.numeric_class == Prospero::TextureNumericClass::Sint ||
	    (image.numeric_class == Prospero::TextureNumericClass::Uint &&
	     image.conversion_format != Prospero::BufferFormat::kInvalid)) {
		return SamplerClass::PointInteger;
	}
	return image.numeric_class == Prospero::TextureNumericClass::Uint ? SamplerClass::Integer
	                                                               : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return base <= last ? last - base + 1u : 0u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

} // namespace

uint32_t BufferStrideData(const DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor) || descriptor.Type() != 0) {
		return 0;
	}
	auto       packed_stride = descriptor.PackedStride();
	const auto stride        = packed_stride & 0x3fffu;
	const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
	if (stride == 0u) {
		packed_stride &= ~((1u << 14u) | (3u << 16u));
	} else if (!swizzle) {
		packed_stride &= ~(3u << 16u);
	}
	return packed_stride;
}

uint32_t BufferFormatData(const DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor) || descriptor.Type() != 0) {
		return 0;
	}
	const auto stride = descriptor.PackedStride() & 0x3fffu;
	const auto format = static_cast<uint32_t>(descriptor.RawFormat()) & 0x7fu;
	const auto info   = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
	const auto layout = static_cast<uint32_t>(Format::kBufferLayouts.id[format]);
	const auto type   = layout != 0u ? static_cast<uint32_t>(info.type) : 0u;
	const auto count  = layout != 0u ? info.component_count : 0u;
	const auto selects = descriptor.DstSelXYZW();
	uint32_t   code    = 0;
	uint32_t   scale   = 1;
	for (uint32_t component = 0; component < 4; component++) {
		const auto select = (selects >> (component * 3u)) & 7u;
		const auto state  = select == 1u                       ? 1u
		                    : select >= 4u && count != 0u      ? 2u + (select - 4u) % count
		                                                       : 0u;
		code += state * scale;
		scale *= 6u;
	}
	return stride | (layout << 14u) | (type << 18u) | (code << 21u);
}

bool DynamicBufferFormat(const ShaderInfo& info, const ResourceSpecialization& specialization,
                         uint32_t index) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFFER_DYNAMIC_FORMAT");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled || !BufferStrideAsData() || index >= info.buffers.size() || index >= 0xffu ||
	    index >= specialization.buffers.size()) {
		return false;
	}
	const auto& base = info.buffers[index];
	const auto& spec = specialization.buffers[index];
	if (!base.formatted || base.written || base.atomic || base.table_read ||
	    base.image_alias != BufferResource::NoImageAlias ||
	    spec.indirect_root != BufferResource::NoIndirectBuffer || spec.zero_stride_oob ||
	    (spec.packed_stride & (1u << 14u)) != 0u) {
		return false;
	}
	return std::none_of(specialization.buffers.begin(), specialization.buffers.end(),
	                    [&](const auto& other) { return other.indirect_root == index; });
}

static bool Image2DAsArray();

void CanonicalizeDynamicFields(const ShaderInfo& info, ResourceSpecialization& specialization) {
	for (uint32_t i = 0; i < specialization.buffers.size(); i++) {
		if (DynamicBufferFormat(info, specialization, i)) {
			specialization.buffers[i].descriptor_format  = Prospero::BufferFormat::kInvalid;
			specialization.buffers[i].descriptor_swizzle = DstSel(4, 5, 6, 7);
		}
	}
	for (uint32_t i = 0; i < specialization.images.size() && i < info.images.size(); i++) {
		auto& image = specialization.images[i];
		const auto& base = info.images[i];
		// v418's collapse (a filtered array instruction reads a 2D T# as a one-layer array), for
		// recordings made before it.
		if (image.dimension == Decoder::ImageDimension::Dim2D &&
		    base.dimension == Decoder::ImageDimension::Dim2DArray &&
		    base.resource_class == ImageResourceClass::Sampled && !base.unfiltered && !base.written &&
		    !base.atomic && !base.cube && !image.bindless && Image2DAsArray()) {
			image.dimension = Decoder::ImageDimension::Dim2DArray;
		}
		if (DynamicImageClass(info, i, image)) {
			image.numeric_class = Prospero::TextureNumericClass::Float;
		}
		if (DynamicImageLayer(info, i, image)) {
			image.dimension = Decoder::ImageDimension::Dim2DArray;
		}
		if (DynamicImageCube(info, i, image)) {
			image.cube = false;
		}
		if (DynamicImage3D(info, i, image)) {
			image.dimension = Decoder::ImageDimension::Dim3D;
		}
	}
}

namespace {

struct ReadCapture {
	SrtRuntime                                  source;
	std::vector<std::pair<uint64_t, uint64_t>>& ranges;
};

bool CaptureStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_specialization_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (capture.source.read_memory != nullptr) {
		if (!capture.source.read_memory(capture.source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value>       pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto value = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < slots.size()) {
				slots[slot.U32()] = 1u;
				pending.push_back(program.srt_reads[slot.U32()].value);
			}
			continue;
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t dynamic_offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto offset = dynamic_offset & ~uint64_t {3};
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	base &= AddressMask & ~uint64_t {3};
	if (offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

thread_local int g_indirect_failure_line = 0;
} // namespace
// Which part of MaterializeResources is reading (diagnostics: KYTY_LOG_GPU_OWNED_SRT).
thread_local const char* g_materialize_phase = "";
// The image descriptors as evaluated, before invalid ones are zeroed (KYTY_DEBUG_RESOURCES).
thread_local std::vector<DescriptorValue> g_debug_raw_images;
const char* MaterializePhase() { return g_materialize_phase; }
namespace {

bool FailIndirect(int line) {
	if (g_indirect_failure_line == 0) {
		g_indirect_failure_line = line;
	}
	return false;
}

// The shader looks the key up in the bindless translation table: nothing to enumerate. Record
// the heap for the host and reserve the two words it patches (region base, entry count); zeros
// resolve every key to the placeholder in slot 0.
bool MaterializeBindlessImage(const DescriptorSource::IndirectDescriptor& indirect,
                               const DescriptorValue& table_value, uint32_t image_index,
                               ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	ShaderBufferResource heap;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, heap)) {
		return FailIndirect(__LINE__);
	}
	const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
	snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
	snapshot.images[image_index] = {.dword_count = 8u};
	snapshot.bindless_heaps.push_back({.base           = heap.Base48(),
	                                   .size           = heap.GetSize(),
	                                   .table_offset   = indirect.table_offset,
	                                   .image          = image_index,
	                                   .mapping_offset = mapping_offset});
	auto& root                      = specialization.images[image_index];
	root.indirect_root              = image_index;
	root.indirect_mapping_offset    = mapping_offset;
	root.indirect_search_iterations = 0;
	root.bindless                   = true;
	return true;
}

// A material table too large for the per-draw probe (Wolverine's terrain and scatter pick their
// images by a key from a 20 MB, 1.3M-record table): read it in bulk, keep its unique keys, and
// reuse them for KYTY_MATERIAL_KEYS_MS (default 2000; the table is asset data). Off with
// KYTY_MATERIAL_PROBES=0.
bool LargeMaterialKeys(const ShaderBufferResource& material, uint64_t first, uint64_t step,
                       uint64_t count, const DescriptorSource::IndirectDescriptor& indirect,
                       const SrtRuntime& runtime, std::vector<uint32_t>& keys, uint64_t hash) {
	static const uint64_t max_probes = [] {
		const char* env = std::getenv("KYTY_MATERIAL_PROBES");
		// Off by default: the only tables seen this large pick store-only images (a per-material
		// accumulation the null-image fallback drops), and their 536 candidates are incompatible.
		return env != nullptr ? std::strtoull(env, nullptr, 10) : uint64_t {0};
	}();
	static const int64_t reuse_ms = [] {
		const char* env = std::getenv("KYTY_MATERIAL_KEYS_MS");
		return env != nullptr ? std::strtoll(env, nullptr, 10) : int64_t {2000};
	}();
	if (count == 0u || count > max_probes || (step & 3u) != 0u) return false;
	struct Entry {
		std::vector<uint32_t>                 keys;
		std::chrono::steady_clock::time_point time;
	};
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, Entry>    cache;
	uint64_t id = material.Base48() * 0x9e3779b97f4a7c15ull;
	for (const uint64_t v: {material.GetSize(), first, step, uint64_t {indirect.key_shift},
	                        uint64_t {indirect.key_mask}}) {
		id = (id ^ v) * 0x100000001b3ull + (id >> 29u);
	}
	const auto now = std::chrono::steady_clock::now();
	{
		std::lock_guard lock(mutex);
		if (auto it = cache.find(id); it != cache.end() &&
		    now - it->second.time < std::chrono::milliseconds(reuse_ms)) {
			keys = it->second.keys;
			return true;
		}
	}
	std::vector<uint32_t> found;
	std::vector<uint32_t> chunk;
	constexpr uint64_t    ChunkProbes = 65536u;
	const uint64_t        stride_words = step / 4u;
	for (uint64_t probe = 0; probe < count; probe += ChunkProbes) {
		const auto n     = std::min(ChunkProbes, count - probe);
		const auto words = (n - 1u) * stride_words + 1u;
		chunk.resize(words);
		if (!ReadScalarTable(material.Base48(), material.GetSize(), first + probe * step, runtime,
		                     chunk)) {
			return false;
		}
		for (uint64_t i = 0; i < n; ++i) {
			found.push_back((chunk[i * stride_words] >> indirect.key_shift) & indirect.key_mask);
		}
		if ((probe / ChunkProbes) % 4u == 3u) {
			std::ranges::sort(found);
			found.erase(std::unique(found.begin(), found.end()), found.end());
		}
	}
	std::ranges::sort(found);
	found.erase(std::unique(found.begin(), found.end()), found.end());
	static std::atomic<uint32_t> printed {0};
	if (printed.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr, "[material-keys] %016llx probes=%llu unique=%zu\n",
		             static_cast<unsigned long long>(hash), static_cast<unsigned long long>(count),
		             found.size());
	}
	keys = found;
	std::lock_guard lock(mutex);
	if (cache.size() > 256u) cache.clear();
	cache[id] = {std::move(found), now};
	return true;
}

bool NormalizeIndirectStoreBuffer(DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    descriptor.Base48() == 0u || descriptor.GetSize() == 0u) {
		value.dwords.fill(0u);
	} else if (descriptor.Stride() != 1u || descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	           descriptor.OutOfBounds() != 3u) {
		return SpecializationFail("indirect buffer stores require linear byte bounds");
	}
	return true;
}

template <typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   const DescriptorSource::IndirectDescriptor& indirect,
                                   uint32_t resource_index, uint32_t dword_count,
                                   const SrtRuntime& runtime, SrtWalker& clean,
                                   ResourceSnapshot&             snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   uint32_t maximum_resources, Normalize&& normalize) {
	const auto  descriptor_bytes = dword_count * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	auto& keys = program.material_keys;
	const auto read_keys = [&](const ShaderBufferResource& material, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (step == 4u) {
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, keys))
				return false;
		} else {
			for (auto& key: keys) {
				if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime,
				                     {&key, 1}))
					return false;
				first += step;
			}
		}
		// A key packed into a bit-field of the material word.
		for (auto& key: keys) key = (key >> indirect.key_shift) & indirect.key_mask;
		return true;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	if (sources.empty()) {
		keys.clear();
		DescriptorValue material_value;
		DescriptorValue table_value;
		if ((indirect.material_source != UINT32_MAX &&
		     !clean.EvaluateDescriptor(indirect.material_source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return FailIndirect(__LINE__);
		}
		ShaderBufferResource table;
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return FailIndirect(__LINE__);
		}
		if (indirect.material_source == UINT32_MAX) {
			uint32_t key_count = 0;
			if (indirect.workgroup_axis != UINT32_MAX) {
				if (indirect.workgroup_axis >= runtime.workgroup_counts.size()) return FailIndirect(__LINE__);
				key_count = runtime.workgroup_counts[indirect.workgroup_axis];
				if (key_count == 0u) return FailIndirect(__LINE__);
			} else {
				if (table_value.dword_count != 2u || !clean.Evaluate(indirect.key_count, key_count))
					return FailIndirect(__LINE__);
				if (std::bit_cast<int32_t>(key_count) <= 0) key_count = 0;
			}
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (key_count != 0u && uint64_t {indirect.table_offset} +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return FailIndirect(__LINE__);
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), 0u);
		} else if (!indirect.selector_first.IsEmpty()) {
			ShaderBufferResource material;
			uint32_t             first = 0, count = 0;
			if (!DecodeBufferDescriptor(material_value, material) || material.Type() != 0u ||
			    (indirect.selector_shift != 0u && (material.Base48() & 3u) != 0u) ||
			    material.SwizzleEnabled() ||
			    material.AddTid() || material.OutOfBounds() != 0u ||
			    uint64_t {indirect.selector_offset} + 4u > material.Stride() ||
			    !clean.Evaluate(indirect.selector_first, first) ||
			    !clean.Evaluate(indirect.key_count, count) || count > MaxIndirectDescriptorProbes ||
			    uint64_t {first} + count > material.NumRecords())
				return false;
			if (count != 0u && (uint64_t {first} + count - 1u) * material.Stride() +
			                           indirect.selector_offset + 4u >
			                       uint64_t {UINT32_MAX} + 1u)
				return false;
			if (!read_keys(material, uint64_t {first} * material.Stride() + indirect.selector_offset,
			               material.Stride(), count)) return false;
		} else if (!indirect.selector_mask.IsEmpty()) {
			uint32_t mask = 0;
			uint32_t count = 0;
			if (material_value.dword_count != 2u || table_value.dword_count != 2u ||
			    !clean.Evaluate(indirect.selector_mask, mask) ||
			    !clean.Evaluate(indirect.key_count, count) || count == 0u || count > 32u) {
				return FailIndirect(__LINE__);
			}
			if (count < 32u) mask &= (1u << count) - 1u;
			const auto material_base =
			    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
			keys.reserve(std::popcount(mask));
			while (mask != 0u) {
				const auto index = std::countr_zero(mask);
				const auto offset = static_cast<uint64_t>(indirect.selector_offset) +
				                    static_cast<uint64_t>(index) * indirect.selector_stride;
				if (offset > UINT32_MAX) return FailIndirect(__LINE__);
				uint32_t key = 0;
				if (!ReadScalarTable(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
				                     runtime, {&key, 1})) return FailIndirect(__LINE__);
				keys.push_back((key >> indirect.key_shift) & indirect.key_mask);
				mask &= mask - 1u;
			}
		} else {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material) || table_value.dword_count != 4u) {
				return FailIndirect(__LINE__);
			}
			// The first aligned offset includes the immediate added after shader U32 arithmetic.
			const auto step = std::max<uint64_t>(4u,
			    std::gcd<uint64_t>(indirect.selector_stride, uint64_t {1} << 32u));
			const uint64_t first = indirect.selector_offset;
			const auto size = material.GetSize();
			const auto limit = std::min(first + (uint64_t {1} << 32u) - step, size >= 4u ? size - 4u : 0u);
			const auto probe_count = size >= 4u && first <= limit ? (limit - first) / step + 1u : 0u;
			bool large_keys = false;
			if (probe_count > MaxIndirectDescriptorProbes && dword_count == 8u &&
			    LargeMaterialKeys(material, first, step, probe_count, indirect, runtime, keys,
			                      program.shader_hash)) {
				// Enumerated from the large-table cache (keys are unique and sorted).
				large_keys = true;
			} else if (probe_count > MaxIndirectDescriptorProbes) {
				static std::atomic<uint32_t> printed {0};
				if (printed.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::fprintf(stderr,
					             "[probe-cap] %016llx dwords=%u size=%llu first=%llu step=%llu "
					             "probes=%llu shift=%u mask=%08x sel_shift=%u tstride=%u toff=%u\n",
					             static_cast<unsigned long long>(program.shader_hash), dword_count,
					             (unsigned long long)size, (unsigned long long)first,
					             (unsigned long long)step, (unsigned long long)probe_count,
					             indirect.key_shift, indirect.key_mask, indirect.selector_shift,
					             indirect.table_stride, indirect.table_offset);
				}
				return FailIndirect(__LINE__);
			}
			if (!large_keys && !read_keys(material, first, step, probe_count))
				return FailIndirect(__LINE__);
			if (indirect.selector_shift == 0u) keys.push_back(0u);
		}
		if (indirect.material_source != UINT32_MAX) {
			if (indirect.selector_shift != 0u) {
				for (auto& key: keys) key >>= indirect.selector_shift;
				keys.push_back(0u); // An out-of-range material load returns zero.
			}
			std::ranges::sort(keys);
			keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		}
	}

	const auto children_begin = descriptors.size();
	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto root_resource  = specializations[resource_index];
	const auto key_count = sources.empty() ? keys.size() : sources.size();
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_count * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? keys[entry] : entry;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			const auto table_offset =
			    static_cast<uint32_t>(key * indirect.table_stride) + indirect.table_offset;
			if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(dword_count))) {
				return FailIndirect(__LINE__);
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return FailIndirect(__LINE__);
		}
		// KYTY_DEBUG_RESOURCES=<hash>: each candidate of that shader's indirect descriptors.
		if (static const std::string debug = [] {
			    const char* env = std::getenv("KYTY_DEBUG_RESOURCES");
			    return env != nullptr ? std::string(env) : std::string();
		    }();
		    !debug.empty() && debug.find(fmt::format("{:016x}", program.shader_hash)) != std::string::npos) {
			static std::atomic<uint32_t> printed {0};
			if (printed.fetch_add(1, std::memory_order_relaxed) < 64) {
				std::string words;
				for (uint32_t i = 0; i < dword_count; i++) words += fmt::format(" {:08x}", candidate.dwords[i]);
				std::fprintf(stderr, "[indirect-cand] %016llx res=%u entry=%u/%u key=%u table=0x%llx+%llu null=%d valid=%d:%s\n",
				             static_cast<unsigned long long>(program.shader_hash), resource_index, entry,
				             static_cast<uint32_t>(key_count), key,
				             static_cast<unsigned long long>(table_base),
				             static_cast<unsigned long long>(table_size), NullImageDescriptor(candidate) ? 1 : 0,
				             dword_count == 8u && ValidImageDescriptor(candidate) ? 1 : 0, words.c_str());
			}
		}
		if (!normalize(candidate)) return false;
		uint32_t ordinal = 0;
		if (entry == 0) {
			descriptors[resource_index] = candidate;
		} else if (descriptors[resource_index] != candidate) {
			const auto found =
			    std::find(descriptors.begin() + children_begin, descriptors.end(), candidate);
			ordinal = static_cast<uint32_t>(found - descriptors.begin() - children_begin + 1u);
			if (found == descriptors.end()) {
				if (descriptors.size() >= maximum_resources) {
					return FailIndirect(__LINE__);
				}
				descriptors.push_back(candidate);
				auto child          = root_resource;
				child.indirect_root = resource_index;
				specializations.push_back(child);
			}
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (key_count > 4096u) {
		static std::atomic<uint32_t> printed {0};
		if (printed.fetch_add(1, std::memory_order_relaxed) < 16) {
			uint32_t nulls = 0;
			for (auto d = children_begin; d < descriptors.size(); ++d)
				nulls += std::ranges::all_of(descriptors[d].dwords, [](uint32_t w) { return w == 0u; });
			std::fprintf(stderr, "[material-descs] %016llx keys=%zu distinct=%zu null=%u root_null=%d\n",
			             static_cast<unsigned long long>(program.shader_hash), size_t(key_count),
			             descriptors.size() - children_begin, nulls,
			             std::ranges::all_of(descriptors[resource_index].dwords,
			                                 [](uint32_t w) { return w == 0u; }) ? 1 : 0);
		}
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_count);
	}
	return true;
}
} // namespace

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::array<std::array<uint32_t, 3>, ShaderInfo::MaxSamplers> mapping;
	std::array<Binding, ShaderInfo::MaxSamplers>                bindings;
	uint32_t                                                  sampler_count = 0;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	// what names the use, for the fatal message.
	uint32_t operator()(uint32_t index, const char* what) const {
		if (index >= source_count) {
			EXIT("image remap: %s refers to image %u, but the shader has %u\n", what, index,
			     source_count);
		}
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

static bool Image2DAsArray() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_2D_AS_ARRAY");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool DynamicImageClassEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_DYNAMIC_CLASS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// A direct, sampled slot the shader only loads from or queries (no filtered sampling: a float
// sampler on an integer view is undefined, and implicit derivatives need one image per sample).
static bool LoadOnlyImageSlot(const ShaderInfo& info, uint32_t index, const ResourceSpecialization::Image& image) {
	if (index >= info.images.size() || index >= 32u) {
		return false;
	}
	const auto& base = info.images[index];
	if (base.resource_class != ImageResourceClass::Sampled || !base.unfiltered || base.written ||
	    base.atomic || base.depth_compare || base.cube || base.r128 ||
	    base.mip_mode == ImageMipMode::Dynamic || image.bindless || image.cube || image.fmask ||
	    image.indirect_root != ImageResource::NoIndirectImage) {
		return false;
	}
	return std::none_of(info.sampled_pairs.begin(), info.sampled_pairs.end(),
	                    [&](const auto& pair) { return pair.image == index; });
}

bool DynamicImageLayer(const ShaderInfo& info, uint32_t index, const ResourceSpecialization::Image& image) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_DYNAMIC_LAYER");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && index < info.images.size() &&
	       info.images[index].dimension == Decoder::ImageDimension::Dim2DArray &&
	       (image.dimension == Decoder::ImageDimension::Dim2D ||
	        image.dimension == Decoder::ImageDimension::Dim2DArray) &&
	       LoadOnlyImageSlot(info, index, image);
}

bool DynamicImageCube(const ShaderInfo& info, uint32_t index, const ResourceSpecialization::Image& image) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_DYNAMIC_CUBE");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled || index >= info.images.size() || index >= 32u) {
		return false;
	}
	const auto& base = info.images[index];
	return base.resource_class == ImageResourceClass::Sampled && !base.written && !base.atomic &&
	       !base.cube && base.dimension == Decoder::ImageDimension::Dim2DArray && !image.bindless &&
	       !image.fmask && image.indirect_root == ImageResource::NoIndirectImage &&
	       image.dimension == Decoder::ImageDimension::Dim2DArray;
}

bool DynamicImage3D(const ShaderInfo& info, uint32_t index, const ResourceSpecialization::Image& image) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_DYNAMIC_3D");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled || index >= info.images.size() || index >= 32u) {
		return false;
	}
	const auto& base = info.images[index];
	return base.resource_class == ImageResourceClass::Sampled && !base.written && !base.atomic &&
	       !base.cube && !base.r128 && base.mip_mode != ImageMipMode::Dynamic &&
	       base.dimension == Decoder::ImageDimension::Dim3D && !image.bindless && !image.fmask &&
	       !image.cube && image.indirect_root == ImageResource::NoIndirectImage &&
	       (image.dimension == Decoder::ImageDimension::Dim3D ||
	        image.dimension == Decoder::ImageDimension::Dim2D) &&
	       !DynamicImageClass(info, index, image);
}

bool DynamicImageClass(const ShaderInfo& info, uint32_t index, const ResourceSpecialization::Image& image) {
	if (!DynamicImageClassEnabled() || !LoadOnlyImageSlot(info, index, image) ||
	    image.conversion_format != Prospero::BufferFormat::kInvalid ||
	    (image.numeric_class != Prospero::TextureNumericClass::Float &&
	     image.numeric_class != Prospero::TextureNumericClass::Uint)) {
		return false;
	}
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D:
		case Decoder::ImageDimension::Dim1DArray:
		case Decoder::ImageDimension::Dim2D:
		case Decoder::ImageDimension::Dim2DArray:
		case Decoder::ImageDimension::Dim3D: break;
		default: return false;
	}
	return true;
}

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization,
                                        bool                    float_image_atomics) {
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(
			    fmt::format("image descriptor {} has an invalid mip range", i));
		}
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			// v421 (Fable, census leftovers): a null T# reads zeros whatever its shape, so it takes
			// the dimension the instruction samples (the host binds a null image of that view type,
			// NullImageDescription) instead of making a 2D variant next to the array/3D/1D one.
			if (!image.bindless && !base.cube && Image2DAsArray() &&
			    (base.dimension == Decoder::ImageDimension::Dim1D ||
			     base.dimension == Decoder::ImageDimension::Dim1DArray ||
			     base.dimension == Decoder::ImageDimension::Dim2DArray ||
			     base.dimension == Decoder::ImageDimension::Dim3D)) {
				image.dimension = base.dimension;
			}
			if (image.bindless) {
				// No one descriptor describes a bindless image: keep the dimension the
				// instruction samples, which picks the set-1 array.
				image.dimension = base.dimension == Decoder::ImageDimension::Unknown
				                      ? Decoder::ImageDimension::Dim2D
				                      : base.dimension;
				image.cube      = base.cube;
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		// v418 (zero-compile plan, census: 2D vs 2D-array was the largest variant field): an array
		// instruction that only samples a 2D T# reads it as a one-layer 2D array (the host binds a
		// 1-layer 2D_ARRAY view, descriptors.cpp). Hardware addresses a 2D T# without the slice and
		// filtered sampling clamps the layer to 0, so the result is the same and the shader needs
		// one variant for both. Texel loads and size queries keep the 2D type (an out-of-range
		// layer is undefined there). KYTY_IMAGE_2D_AS_ARRAY=0 turns it off.
		if (descriptor_dimension == Decoder::ImageDimension::Dim2D &&
		    base.dimension == Decoder::ImageDimension::Dim2DArray &&
		    base.resource_class == ImageResourceClass::Sampled && !base.unfiltered && !base.written &&
		    !base.atomic && !base.cube && Image2DAsArray() &&
		    static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
		        Prospero::ImageType::kColor2D) {
			image.dimension = Decoder::ImageDimension::Dim2DArray;
		}
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		// Image atomics operate on the raw texel bits, and only the integer atomics decode, so an
		// atomic on an R32 sint or float surface is a uint atomic on the same bits.
		// --no-float-image-atomics still turns the float case off.
		const bool float_atomic =
		    float_image_atomics && base.atomic && format == Prospero::BufferFormat::k32Float;
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt && !float_atomic)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const bool raw_sint_storage = storage && format == Prospero::BufferFormat::k32SInt &&
		                              base.written && !base.read && !base.atomic;
		image.numeric_class         = Prospero::SampledTextureNumericClass(format);
		if (storage) {
			if ((!raw_sint_storage && !base.atomic &&
			     image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			// Integer image atomics operate on the raw 32-bit texel, independently of
			// the descriptor numeric format. R32 float/sint loads and stores preserve
			// the same register bits through this uint view.
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		// v429: a load-only slot reads either class through its two bindings (DynamicImageClass):
		// the specialization keeps Float, the draw tells the shader which binding holds the T#.
		if (base_index == i && DynamicImageClass(program.info, i, image)) {
			image.numeric_class = Prospero::TextureNumericClass::Float;
		}
		// v430: an array instruction's load-only slot is a 2D array whatever its T# (2D or array):
		// the draw tells the shader to read layer 0 of a 2D T# (DynamicImageLayer).
		if (base_index == i && DynamicImageLayer(program.info, i, image)) {
			image.dimension = Decoder::ImageDimension::Dim2DArray;
		}
		// v431: a cube T# under an array instruction shares the 2D-array module (DynamicImageCube).
		if (base_index == i && DynamicImageCube(program.info, i, image)) {
			image.cube = false;
		}
		// v433: a 3D instruction's slot shares the 3D module for a 3D or 2D T# (DynamicImage3D).
		if (base_index == i && DynamicImage3D(program.info, i, image)) {
			image.dimension = Decoder::ImageDimension::Dim3D;
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (root.indirect_search_iterations == 0u || key_count < 2u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * 2u >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			if (image.numeric_class != image_class.numeric_class ||
			    (!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension))) ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	if (base.samplers.size() > plan.mapping.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			// A bindless sampler is one slot of the host's table for all of its uses: its other
			// classes share the first one's binding rather than adding copies.
			if (!first && base.samplers[index].bindless) {
				mapping[type] = index;
				continue;
			}
			const auto target = first ? index : plan.sampler_count++;
			if (target >= ShaderInfo::MaxSamplers) return false;
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

// A walk visits a block's dominators before it and its post-dominators after it whatever the
// outcomes, so a condition whose skippable blocks (reachable from it, not post-dominating it) only
// hold sources and SRT reads those always-visited blocks already hold cannot change the active
// sources or the refreshed reads. Evaluating it only reads memory: in Wolverine's
// 06544f382c6b0850 a per-record flag the GPU has just written, a readback drain per frame.
// Port of Revivedx/KytyPS5 5f533fcb (MarkIrrelevantConditions).
static uint32_t MarkIrrelevantConditions(std::vector<ResourceBlock>& blocks, std::vector<uint8_t>& irrelevant) {
	irrelevant.assign(blocks.size(), 0u);
	const size_t n = blocks.size();
	if (n == 0u || n > 4096u) return 0;
	const size_t words = (n + 63u) / 64u;
	using Bits         = std::vector<uint64_t>;
	const auto test    = [](const Bits& bits, size_t i) { return ((bits[i / 64u] >> (i % 64u)) & 1u) != 0u; };
	const auto set     = [](Bits& bits, size_t i) { bits[i / 64u] |= uint64_t {1} << (i % 64u); };
	std::vector<std::vector<uint32_t>> preds(n);
	for (uint32_t i = 0; i < n; i++) {
		for (const auto s: blocks[i].successors) {
			if (s >= n) return 0;
			preds[s].push_back(i);
		}
	}
	std::vector<uint8_t>  live(n, 0u);
	std::vector<uint32_t> stack {0u};
	while (!stack.empty()) {
		const auto i = stack.back();
		stack.pop_back();
		if (live[i] != 0u) continue;
		live[i] = 1u;
		for (const auto s: blocks[i].successors) stack.push_back(s);
	}
	// Post-dominance needs every live block to reach an exit.
	std::vector<uint8_t> exits(n, 0u);
	for (uint32_t i = 0; i < n; i++) {
		if (live[i] != 0u && blocks[i].successors.empty()) {
			exits[i] = 1u;
			stack.push_back(i);
		}
	}
	while (!stack.empty()) {
		const auto i = stack.back();
		stack.pop_back();
		for (const auto p: preds[i]) {
			if (live[p] != 0u && exits[p] == 0u) {
				exits[p] = 2u;
				stack.push_back(p);
			}
		}
	}
	for (uint32_t i = 0; i < n; i++) {
		if (live[i] != 0u && exits[i] == 0u) return 0;
	}
	const Bits         ones(words, ~uint64_t {0});
	std::vector<Bits>  dom(n, ones);
	std::vector<Bits>  pdom(n, ones);
	dom[0].assign(words, 0u);
	set(dom[0], 0);
	for (uint32_t i = 0; i < n; i++) {
		if (exits[i] == 1u) {
			pdom[i].assign(words, 0u);
			set(pdom[i], i);
		}
	}
	for (bool changed = true; changed;) {
		changed = false;
		for (uint32_t i = 1; i < n; i++) {
			if (live[i] == 0u) continue;
			Bits next = ones;
			for (const auto p: preds[i]) {
				if (live[p] == 0u) continue;
				for (size_t w = 0; w < words; w++) next[w] &= dom[p][w];
			}
			set(next, i);
			if (next != dom[i]) {
				dom[i]  = std::move(next);
				changed = true;
			}
		}
	}
	for (bool changed = true; changed;) {
		changed = false;
		for (uint32_t i = static_cast<uint32_t>(n); i-- > 0;) {
			if (live[i] == 0u || exits[i] == 1u) continue;
			Bits next = ones;
			for (const auto s: blocks[i].successors) {
				for (size_t w = 0; w < words; w++) next[w] &= pdom[s][w];
			}
			set(next, i);
			if (next != pdom[i]) {
				pdom[i] = std::move(next);
				changed = true;
			}
		}
	}
	std::vector<std::vector<uint32_t>> reads(n);
	for (uint32_t i = 0; i < n; i++) {
		reads[i] = blocks[i].srt_reads;
		std::ranges::sort(reads[i]);
		reads[i].erase(std::unique(reads[i].begin(), reads[i].end()), reads[i].end());
	}
	uint32_t marked = 0;
	for (uint32_t b = 0; b < n; b++) {
		auto& block = blocks[b];
		if (block.condition.IsEmpty()) continue;
		if (live[b] == 0u) {
			irrelevant[b] = 1u;
			marked++;
			continue;
		}
		std::vector<uint32_t> always_sources;
		std::vector<uint32_t> always_reads;
		for (uint32_t x = 0; x < n; x++) {
			if (live[x] != 0u && (test(dom[b], x) || test(pdom[b], x))) {
				always_sources.insert(always_sources.end(), blocks[x].sources.begin(), blocks[x].sources.end());
				always_reads.insert(always_reads.end(), reads[x].begin(), reads[x].end());
			}
		}
		std::ranges::sort(always_sources);
		std::ranges::sort(always_reads);
		std::vector<uint8_t> seen(n, 0u);
		stack.assign(block.successors.begin(), block.successors.end());
		bool covered = true;
		while (!stack.empty() && covered) {
			const auto x = stack.back();
			stack.pop_back();
			if (seen[x] != 0u) continue;
			seen[x] = 1u;
			if (!test(pdom[b], x)) {
				covered = std::ranges::includes(always_sources, blocks[x].sources) &&
				          std::ranges::includes(always_reads, reads[x]);
			}
			for (const auto s: blocks[x].successors) stack.push_back(s);
		}
		if (covered) {
			irrelevant[b] = 1u;
			marked++;
		}
	}
	return marked;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	// The renderer checks captured scalar reads against final buffer and image write ranges.
	if (program.blocks.size() != program.block_info.size() || program.has_address_writes) {
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> indices;
	for (uint32_t i = 0; i < program.block_info.size(); i++) {
		if (!indices.emplace(program.block_info[i].id, i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	std::vector<std::vector<uint32_t>> predecessors(blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&           info       = program.block_info[i];
		const auto&           terminator = info.terminator;
		std::vector<uint32_t> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors.push_back(terminator.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors = {terminator.true_block, terminator.false_block};
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch:
				successors = terminator.indirect_targets;
				break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
			predecessors[found->second].push_back(i);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	// Host branches only matter when they guard a later resource access. Leave
	// data-only branches (for example a mutable flag checked before a trap) on the GPU.
	std::vector<uint8_t> resource_reachable(blocks.size());
	std::vector<uint32_t> pending;
	for (uint32_t i = 0; i < blocks.size(); ++i) {
		if (!blocks[i].sources.empty() || !blocks[i].srt_reads.empty()) {
			resource_reachable[i] = 1;
			pending.push_back(i);
		}
	}
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		for (const auto predecessor: predecessors[index]) {
			if (resource_reachable[predecessor]) continue;
			resource_reachable[predecessor] = 1;
			pending.push_back(predecessor);
		}
	}
	for (auto& block: blocks) {
		block.condition = std::ranges::any_of(block.successors, [&](uint32_t successor) {
			return resource_reachable[successor] != 0;
		}) ? predicate(block.condition) : Value {};
	}
	// KYTY_PRUNE_CONDITIONS (default 1; 0 keeps them): conditions that cannot change the walk are
	// dropped at plan time. Revivedx measured ~1 CP readback drain fewer per frame, +3.8 %.
	static const bool prune = [] {
		const char* value = std::getenv("KYTY_PRUNE_CONDITIONS");
		return value == nullptr || value[0] != '0';
	}();
	if (prune && std::ranges::any_of(
	                 blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		std::vector<uint8_t> irrelevant;
		if (MarkIrrelevantConditions(blocks, irrelevant) != 0u) {
			for (size_t i = 0; i < blocks.size(); i++) {
				if (irrelevant[i] != 0u) blocks[i].condition = Value {};
			}
		}
	}
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard,
                                                      uint32_t depth = 0) {
	value = ResolveActiveU32(value, guard);
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, guard, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, guard, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    program.blocks.size() != program.block_info.size() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<uint32_t> visited;
	uint32_t                     index = 0;
	const Inst*                  store = nullptr;
	for (;;) {
		if (!visited.insert(index).second) return {};
		for (const auto& inst: *program.blocks[index]) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = program.block_info[index].terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		const auto next = std::ranges::find(program.block_info, term.true_block, &BlockInfo::id);
		if (next == program.block_info.end()) return {};
		index = static_cast<uint32_t>(next - program.block_info.begin());
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    !IsFullDispatchPredicate(store->Arg(3))) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

namespace {

// Integer operations whose result interval follows from their operands' intervals (see
// WriteInterval).
bool IsIntervalOperation(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::IMul32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::SelectU32: return true;
		default: return false;
	}
}

bool IsBoundedInvocationId(const ResourcePlan& program, const Inst& inst) {
	if (inst.GetOpcode() != ValueOpcode::GetBuiltin || program.stage != ShaderType::Compute ||
	    !inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate() || inst.Arg(1).U32() >= 3u) {
		return false;
	}
	switch (static_cast<StageInputKind>(inst.Arg(0).U32())) {
		case StageInputKind::WorkgroupId:
		case StageInputKind::LocalInvocationId:
		case StageInputKind::LocalInvocationIndex:
		case StageInputKind::GlobalInvocationId: return true;
		default: return false;
	}
}

// A store address operand the host can bound per dispatch: values it evaluates, compute invocation
// IDs, and interval operations on them. The select on the store's own predicate only matters for
// the stored lanes.
bool BoundableWriteOperand(const Program& program, Value value, Value predicate, uint32_t depth) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32u) {
		return false;
	}
	if (IsBoundedInvocationId(program, *inst)) {
		return true;
	}
	if (IsIntervalOperation(inst->GetOpcode())) {
		if (inst->GetOpcode() == ValueOpcode::SelectU32 && inst->Arg(0).Resolve() == predicate) {
			return BoundableWriteOperand(program, inst->Arg(1), predicate, depth + 1u);
		}
		const auto first = inst->GetOpcode() == ValueOpcode::SelectU32 ? 1u : 0u;
		for (size_t i = first; i < inst->NumArgs(); ++i) {
			if (!BoundableWriteOperand(program, inst->Arg(i), predicate, depth + 1u)) {
				return false;
			}
		}
		return true;
	}
	return value.GetType() == Type::U32 &&
	       ValidateRuntimeValue(program, value, RuntimeValueType::Integer);
}

} // namespace

namespace {
bool GpuSizedWrittenEnabled(uint64_t shader_hash);
bool GpuSizedDynamicEnabled();

// KYTY_GPU_SIZED_ATOMIC_SHADERS=<hash>[,<hash>...] (hex): atomically updated buffers of the
// listed shaders also bind a conservative record count when the GPU wrote theirs (with
// KYTY_GPU_SIZED_WRITTEN on and the shader in its list): only for shaders whose atomics index
// within the real count (the shader's own data), never ones that rely on out-of-range drops.
bool GpuSizedAtomicEnabled(uint64_t shader_hash) {
	// KYTY_GPU_SIZED_DYNAMIC: the shader bounds its atomics by the exact count too, so an atomic
	// past it is dropped as on the console (CS 48ba2226632e8209 waited for its atomic buffer's
	// count every frame on the title screen, ~20 % of the GPU thread). KYTY_GPU_SIZED_ATOMIC_SHADERS
	// still limits it to the listed shaders.
	static const bool listed = std::getenv("KYTY_GPU_SIZED_ATOMIC_SHADERS") != nullptr;
	if (GpuSizedDynamicEnabled() && !listed) {
		return GpuSizedWrittenEnabled(shader_hash);
	}
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_GPU_SIZED_ATOMIC_SHADERS");
		if (value == nullptr) return result;
		const std::string list = value;
		size_t            at   = 0;
		while (at < list.size()) {
			const auto end = std::min(list.find_first_of(",; ", at), list.size());
			if (end > at) result.push_back(std::strtoull(list.substr(at, end - at).c_str(), nullptr, 16));
			at = end + 1;
		}
		return result;
	}();
	return !shaders.empty() && GpuSizedWrittenEnabled(shader_hash) &&
	       std::find(shaders.begin(), shaders.end(), shader_hash) != shaders.end();
}

// KYTY_SRT_EXTENTS_NO_WAIT_SHADERS=<hash>[,<hash>...] (hex): KYTY_SRT_EXTENTS_NO_WAIT for the
// listed shaders only. Their write extents are evaluated without waiting for the GPU (the whole
// buffer when they cannot be), so the slots the extents read can be left to the GPU.
bool ExtentsNoWaitShader(uint64_t shader_hash) {
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_SRT_EXTENTS_NO_WAIT_SHADERS");
		if (value == nullptr) return result;
		const std::string list = value;
		size_t            at   = 0;
		while (at < list.size()) {
			const auto end  = std::min(list.find_first_of(",; ", at), list.size());
			if (end > at) result.push_back(std::strtoull(list.substr(at, end - at).c_str(), nullptr, 16));
			at = end + 1;
		}
		return result;
	}();
	return !shaders.empty() && std::find(shaders.begin(), shaders.end(), shader_hash) != shaders.end();
}
} // namespace

// Flat slots the host never needs, so their bytes can be left to the GPU when it wrote them
// (SrtGpuFill): the slot's value is a raw scalar read whose address comes from user data and
// immediates alone, and no value the host evaluates refers to the slot (descriptor dwords and
// keys, branch conditions, uniform fills, buffer write bounds, other reads' addresses). The host
// then only uploads the flat buffer; the shader reads the slot.
static std::vector<uint8_t> GpuFillSlots(const ResourcePlan& plan) {
	const auto           count = plan.srt_reads.size();
	std::vector<uint8_t> host(count, 0u);
	std::vector<Value>   pending;
	std::unordered_set<const Inst*> visited;
	// Why a slot stays on the host (KYTY_SRT_FILL_REASON="hash,..." prints it): D descriptor,
	// C condition, U uniform fill, W write extent, A another read's address, S clean slot,
	// K not a raw scalar read, M address depends on memory.
	uint8_t    tag  = 'D';
	// Sources only read-only, non-indirect buffers use: their record count (dword 2) is not a host
	// need (GpuSizedBuffers binds a conservative count when the GPU wrote it).
	std::vector<uint8_t> sized_sources(plan.descriptor_sources.size(), 0u);
	// For KYTY_SRT_FILL_REASON: why a source's record count stays on the host (w written,
	// a atomic, i image, s sampler).
	std::vector<std::string> source_why(plan.descriptor_sources.size());
	{
		std::vector<uint8_t> other(plan.descriptor_sources.size(), 0u);
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.source >= sized_sources.size()) continue;
			if ((!buffer.written || GpuSizedWrittenEnabled(plan.shader_hash)) &&
			    (!buffer.atomic || GpuSizedAtomicEnabled(plan.shader_hash))) {
				sized_sources[buffer.source] = 1u;
			} else {
				other[buffer.source] = 1u;
				source_why[buffer.source] += buffer.atomic ? 'a' : 'w';
			}
		}
		for (const auto& image: plan.info.images) {
			if (image.source < other.size()) {
				other[image.source] = 1u;
				source_why[image.source] += 'i';
			}
		}
		for (const auto& sampler: plan.info.samplers) {
			if (sampler.source < other.size()) {
				other[sampler.source] = 1u;
				source_why[sampler.source] += 's';
			}
		}
		for (size_t i = 0; i < sized_sources.size(); ++i) {
			if (other[i] != 0u || plan.descriptor_sources[i].dword_count != 4u ||
			    plan.descriptor_sources[i].indirect_descriptor.has_value()) {
				sized_sources[i] = 0u;
			}
		}
	}
	std::vector<int64_t> host_desc(count, -1);
	int64_t              current_desc = -1;
	const auto mark = [&](Value root) {
		pending.push_back(root);
		while (!pending.empty()) {
			const auto  value = pending.back().Resolve();
			pending.pop_back();
			const auto* inst  = value.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) {
				continue;
			}
			if (inst->GetOpcode() == ValueOpcode::ReadConst) {
				const auto slot = inst->Arg(1).Resolve();
				if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < count) {
					if (host[slot.U32()] == 0u) {
						host[slot.U32()]       = tag;
						host_desc[slot.U32()]  = tag == 'D' ? current_desc : -1;
					}
					pending.push_back(plan.srt_reads[slot.U32()].value);
				}
				continue;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				pending.push_back(inst->Arg(arg));
			}
		}
	};
	for (size_t source_index = 0; source_index < plan.descriptor_sources.size(); ++source_index) {
		const auto& source = plan.descriptor_sources[source_index];
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			if (dword == 2u && sized_sources[source_index] != 0u) {
				continue;
			}
			current_desc = static_cast<int64_t>(source_index * 16u + dword);
			mark(source.dwords[dword]);
		}
		current_desc = static_cast<int64_t>(source_index * 16u + 15u);
		if (source.indirect_descriptor.has_value()) {
			mark(source.indirect_descriptor->key_count);
			mark(source.indirect_descriptor->selector_first);
			mark(source.indirect_descriptor->selector_mask);
		}
	}
	// Branch conditions only choose which descriptor sources are active. With SrtConditionsNoWait
	// a condition the host cannot read without waiting for the GPU is "not known" (both successors
	// walked), so the slots it reads need not be on the host: the GPU fills them for the shader.
	tag = 'C';
	if (!SrtConditionsNoWait()) {
		for (const auto& block: plan.control_flow) {
			mark(block.condition);
		}
	}
	tag = 'U';
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		mark(plan.uniform_fill.values[i]);
	}
	tag = 'W';
	if (std::getenv("KYTY_SRT_EXTENTS_NO_WAIT") == nullptr && !ExtentsNoWaitShader(plan.shader_hash)) {
		for (const auto& write: plan.buffer_writes) {
			// A GPU-sized buffer's extents are evaluated without waiting (t_gpu_sized_snapshot);
			// they need no host slots.
			if (write.buffer < plan.info.buffers.size() &&
			    plan.info.buffers[write.buffer].source < sized_sources.size() &&
			    sized_sources[plan.info.buffers[write.buffer].source] != 0u) {
				continue;
			}
			mark(write.index);
			mark(write.offset);
			mark(write.soffset);
			mark(write.predicate);
		}
	}
	// Another read's address operands (the read itself is not a host use of its own slot).
	tag = 'A';
	for (const auto& read: plan.srt_reads) {
		if (const auto* inst = read.value.Resolve().TryInstruction(); inst != nullptr) {
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				mark(inst->Arg(arg));
			}
		}
	}
	// The read's own address must not depend on memory: one host read per slot, the slot's own.
	const auto memory_free = [&](const Inst& read) {
		std::vector<Value>              stack;
		std::unordered_set<const Inst*> seen;
		for (size_t arg = 0; arg < read.NumArgs(); arg++) {
			stack.push_back(read.Arg(arg));
		}
		while (!stack.empty()) {
			const auto* inst = stack.back().Resolve().TryInstruction();
			stack.pop_back();
			if (inst == nullptr || !seen.insert(inst).second) {
				continue;
			}
			switch (inst->GetOpcode()) {
				case ValueOpcode::ReadConst:
				case ValueOpcode::LoadAddressU32:
				case ValueOpcode::ReadConstBuffer:
				case ValueOpcode::LoadBufferU32:
				case ValueOpcode::Phi: return false;
				default: break;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				stack.push_back(inst->Arg(arg));
			}
		}
		return true;
	};
	std::vector<uint8_t> fill(count, 0u);
	for (size_t slot = 0; slot < count; slot++) {
		const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction();
		if (inst == nullptr) {
			host[slot] = host[slot] != 0u ? host[slot] : 'K';
			continue;
		}
		if (host[slot] != 0u) {
			continue;
		}
		if (slot < plan.clean_flat_slots.size() && plan.clean_flat_slots[slot] != 0u) {
			host[slot] = 'S';
			continue;
		}
		const auto op    = inst->GetOpcode();
		const auto index = inst->Flags<MemoryFlags>().index;
		if ((op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) ||
		    index >= plan.memory_info.size()) {
			host[slot] = 'K';
			continue;
		}
		const auto kind = plan.memory_info[index].kind;
		if (!((op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
		      (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer))) {
			host[slot] = 'K';
			continue;
		}
		// 1: the slot's read is its only read. 2: its address comes from other reads; those are done
		// on the host and only the slot's own read is left to the GPU (SrtWalker::m_fill_inst).
		fill[slot] = memory_free(*inst) ? 1u : 2u;
	}
	if (const char* list = std::getenv("KYTY_SRT_FILL_REASON");
	    list != nullptr &&
	    std::string_view(list).find(fmt::format("{:016x}", plan.shader_hash)) != std::string_view::npos) {
		std::string text;
		for (size_t slot = 0; slot < count; slot++) {
			const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction();
			text += fmt::format(" {}:{}({})", slot, fill[slot] != 0u ? 'G' : static_cast<char>(host[slot]),
			                    inst != nullptr ? ValueOpcodeName(inst->GetOpcode()) : "-");
			if (fill[slot] == 0u && host[slot] == 'D' && host_desc[slot] >= 0) {
				const auto  src    = static_cast<size_t>(host_desc[slot] / 16);
				const auto  dw     = host_desc[slot] % 16;
				const auto& source = plan.descriptor_sources[src];
				text += fmt::format("[src{} {} n={} ind={} why={}]", src,
				                    dw == 15 ? std::string("key") : fmt::format("d{}", dw),
				                    source.dword_count, source.indirect_descriptor.has_value() ? 1 : 0,
				                    source_why[src]);
			}
		}
		std::fprintf(stderr, "[srt-fill] %016llx slots=%zu%s\n",
		             static_cast<unsigned long long>(plan.shader_hash), count, text.c_str());
	}
	return fill;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;
	plan.bindless_images            = program.bindless_images;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		target.bindless_sampler = source.bindless_sampler;
		if (target.indirect_descriptor.has_value()) {
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// A proven uniform factor can decide a branch even when its other lanes are unknown.
	// Keep only that Boolean structure, never the varying shader dependency graph.
	Value unknown;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) return Clone(value);
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef || op == ValueOpcode::LogicalNot) {
			const auto operand = ClonePredicate(inst->Arg(0));
			if (operand.IsEmpty() || op == ValueOpcode::ConditionRef) return operand;
			auto& node = plan.value_storage.emplace_back(op);
			node.SetArg(0, operand);
			return Value(&node);
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr) return {};
		auto left = ClonePredicate(inst->Arg(0));
		auto right = ClonePredicate(inst->Arg(1));
		if (left.IsEmpty() && right.IsEmpty()) return {};
		if (left.IsEmpty() || right.IsEmpty()) {
			if (unknown.IsEmpty()) unknown = Value(&plan.value_storage.emplace_back(ValueOpcode::UndefU1));
			if (left.IsEmpty()) left = unknown;
			if (right.IsEmpty()) right = unknown;
		}
		auto& node = plan.value_storage.emplace_back(op);
		node.SetArg(0, left);
		node.SetArg(1, right);
		return Value(&node);
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	bool capture_indirect_reads = false;
	for (const auto& source: plan.descriptor_sources) {
		if (!source.indirect_descriptor.has_value()) continue;
		const auto& indirect                = *source.indirect_descriptor;
		plan.requires_specialization_memory = true;
		capture_indirect_reads |= indirect.material_source != UINT32_MAX || !indirect.sources.empty();
		MarkCleanFlatSlots(plan, Source(plan, indirect.material_source), plan.clean_flat_slots,
		                   indirect.selector_mask);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.selector_first);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.key_count);
		if (indirect.sources.empty()) {
			MarkCleanFlatSlots(plan, Source(plan, indirect.table_source), plan.clean_flat_slots);
		} else {
			for (const auto candidate: indirect.sources) {
				MarkCleanFlatSlots(plan, Source(plan, candidate), plan.clean_flat_slots);
			}
		}
	}
	for (const auto& sampler: plan.info.samplers) {
		const auto* source = Source(plan, sampler.source);
		if (source == nullptr || !source->bindless_sampler.has_value()) {
			continue;
		}
		plan.requires_specialization_memory = true;
		MarkCleanFlatSlots(plan, source, plan.clean_flat_slots);
	}
	plan.capture_specialization_reads |= capture_indirect_reads || !plan.control_flow.empty();
	if (plan.capture_specialization_reads) {
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (!buffer.written) continue;
			const auto* source = Source(plan, buffer.source);
			// A GPU-sized written buffer (KYTY_GPU_SIZED_WRITTEN_SHADERS) binds a conservative record
			// count, so its dword 2 is no host need here either: the GPU fills the slots under it.
			if (source != nullptr && source->dword_count == 4u &&
			    !source->indirect_descriptor.has_value() && GpuSizedWrittenEnabled(plan.shader_hash) &&
			    (!buffer.atomic || GpuSizedAtomicEnabled(plan.shader_hash))) {
				for (const uint32_t dword: {0u, 1u, 3u}) {
					MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, source->dwords[dword]);
				}
				continue;
			}
			MarkCleanFlatSlots(plan, source, plan.clean_flat_slots);
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanFlatSlots(plan, Source(plan, image.source), plan.clean_flat_slots);
		}
	}
	if (capture_indirect_reads) plan.resource_tracking_complete &= !program.has_address_writes;
	// After the decisions above: the clones below must not count as the shader's own reads.
	// Store addresses bound the bytes a dispatch writes: a title's heap-wide V# otherwise marks
	// the whole heap GPU-written, and every CPU access to it drains the GPU.
	const auto buffer_count = program.info.buffers.size();
	plan.buffer_writes_bounded.assign(buffer_count, 1u);
	std::vector<uint32_t> listed_writes(buffer_count, 0u);
	for (uint32_t i = 0; i < buffer_count; ++i) {
		if (program.info.buffers[i].image_alias != BufferResource::NoImageAlias) {
			plan.buffer_writes_bounded[i] = 0u;
		}
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) {
				continue;
			}
			const auto flags = inst.Flags<MemoryFlags>();
			if (flags.index >= program.memory_info.size()) {
				plan.buffer_writes_bounded.assign(buffer_count, 0u);
				continue;
			}
			const auto& memory = program.memory_info[flags.index];
			if (memory.kind != ResourceKind::Buffer || memory.resource >= buffer_count) {
				continue;
			}
			const auto buffer = memory.resource;
			if (plan.buffer_writes_bounded[buffer] == 0u) {
				continue;
			}
			const auto predicate = inst.Arg(inst.NumArgs() - 1u).Resolve();
			if (!BoundableWriteOperand(program, inst.Arg(1), predicate, 0u) ||
			    !BoundableWriteOperand(program, inst.Arg(2), predicate, 0u) ||
			    !BoundableWriteOperand(program, inst.Arg(3), predicate, 0u)) {
				plan.buffer_writes_bounded[buffer] = 0u;
				continue;
			}
			listed_writes[buffer]++;
			plan.buffer_writes.push_back({.buffer    = buffer,
			                              .immediate = memory.offset,
			                              .index     = Clone(inst.Arg(1)),
			                              .offset    = Clone(inst.Arg(2)),
			                              .soffset   = Clone(inst.Arg(3)),
			                              .predicate = Clone(predicate)});
		}
	}
	for (uint32_t i = 0; i < buffer_count; ++i) {
		if (listed_writes[i] == 0u) {
			plan.buffer_writes_bounded[i] = 0u;
		}
	}
	plan.gpu_fill_slots = GpuFillSlots(plan);
	return plan;
}

namespace {

struct WriteInterval {
	uint64_t lo = 0;
	uint64_t hi = 0;
};

// The unsigned 32-bit range a store address operand takes over the dispatch: invocation IDs range
// over the workgroup count and size, everything else the host evaluates exactly. Fails on an
// unknown dispatch size or a range that could wrap.
bool EvaluateWriteInterval(const ResourcePlan& program, const SrtRuntime& runtime, SrtWalker& walker,
                           Value value, Value predicate, WriteInterval& out, uint32_t& budget) {
	constexpr uint64_t Max32 = UINT32_MAX;
	if (budget == 0u) {
		return false;
	}
	--budget;
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) {
			return false;
		}
		out = {value.U32(), value.U32()};
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (IsBoundedInvocationId(program, *inst)) {
		const auto  axis  = inst->Arg(1).U32();
		if (runtime.workgroup_counts.size() < 3u) {
			return false;
		}
		const auto& count = runtime.workgroup_counts;
		const auto& size  = runtime.workgroup_size;
		uint64_t    total = 0;
		switch (static_cast<StageInputKind>(inst->Arg(0).U32())) {
			case StageInputKind::WorkgroupId: total = count[axis]; break;
			case StageInputKind::LocalInvocationId: total = size[axis]; break;
			case StageInputKind::LocalInvocationIndex:
				total = uint64_t {size[0]} * size[1] * size[2];
				break;
			case StageInputKind::GlobalInvocationId:
				total = uint64_t {count[axis]} * size[axis];
				break;
			default: return false;
		}
		if (total == 0u || total - 1u > Max32) {
			return false;
		}
		out = {0u, total - 1u};
		return true;
	}
	const auto op = inst->GetOpcode();
	if (!IsIntervalOperation(op)) {
		uint32_t exact = 0;
		if (!walker.Evaluate(value, exact)) {
			return false;
		}
		out = {exact, exact};
		return true;
	}
	if (op == ValueOpcode::SelectU32 && inst->Arg(0).Resolve() == predicate) {
		return EvaluateWriteInterval(program, runtime, walker, inst->Arg(1), predicate, out, budget);
	}
	std::array<WriteInterval, 3> args {};
	const size_t                 first = op == ValueOpcode::SelectU32 ? 1u : 0u;
	if (inst->NumArgs() > args.size()) {
		return false;
	}
	for (size_t i = first; i < inst->NumArgs(); ++i) {
		if (!EvaluateWriteInterval(program, runtime, walker, inst->Arg(i), predicate, args[i],
		                           budget)) {
			return false;
		}
	}
	const auto& a = args[0];
	const auto& b = args[1];
	switch (op) {
		case ValueOpcode::IAdd32: out = {a.lo + b.lo, a.hi + b.hi}; break;
		case ValueOpcode::IMul32: out = {a.lo * b.lo, a.hi * b.hi}; break;
		case ValueOpcode::ShiftLeftLogical32:
			if (b.hi > 31u) {
				return false;
			}
			out = {a.lo << b.lo, a.hi << b.hi};
			break;
		case ValueOpcode::ShiftRightLogical32:
			if (b.hi > 31u) {
				return false;
			}
			out = {a.lo >> b.hi, a.hi >> b.lo};
			break;
		case ValueOpcode::BitwiseAnd32: out = {0u, std::min(a.hi, b.hi)}; break;
		case ValueOpcode::BitwiseOr32: {
			const auto high = std::max(a.hi, b.hi);
			out = {std::max(a.lo, b.lo), (uint64_t {1} << std::bit_width(high)) - 1u};
			break;
		}
		case ValueOpcode::UMin32: out = {std::min(a.lo, b.lo), std::min(a.hi, b.hi)}; break;
		case ValueOpcode::UMax32: out = {std::max(a.lo, b.lo), std::max(a.hi, b.hi)}; break;
		case ValueOpcode::BitFieldUExtract: {
			// (base >> offset) masked to count bits: at most both.
			if (b.hi > 31u) {
				return false;
			}
			const auto& count = args[2];
			const auto  mask  = count.hi >= 32u ? Max32 : (uint64_t {1} << count.hi) - 1u;
			out               = {0u, std::min(mask, a.hi >> b.lo)};
			break;
		}
		case ValueOpcode::SelectU32:
			out = {std::min(args[1].lo, args[2].lo), std::max(args[1].hi, args[2].hi)};
			break;
		default: return false;
	}
	return out.hi <= Max32;
}

// Set while a snapshot has a GPU-sized buffer (GpuSizedRecords): its write extents are evaluated
// without waiting, and one that needs GPU-written bytes covers the (bounded) binding.
thread_local bool t_gpu_sized_snapshot = false;
thread_local uint64_t t_extents_shader_hash = 0; // for OptionalScope below

// See ResourceSnapshot::buffer_write_extents.
void ComputeBufferWriteExtents(const ResourcePlan& program, const SrtRuntime& runtime,
                               SrtWalker& walker, ResourceSnapshot& snapshot) {
	snapshot.buffer_write_extents.assign(program.info.buffers.size(), {});
	t_extents_shader_hash = program.shader_hash;
	if (program.buffer_writes.empty()) {
		return;
	}
	std::vector<uint8_t> failed(program.info.buffers.size(), 0u);
	// Extents only narrow the bytes marked GPU-written; one the host cannot evaluate without
	// waiting for the GPU falls back to the whole buffer (SrtConditionsNoWait).
	// Measured 2026-10-09: the whole-buffer fallback marks more memory GPU-written, and later SRT
	// reads of it wait instead (CS 4e7f2c6bb9b158a1: 1.0 -> 4.4 s of waits per 20 s). Opt-in only.
	static const bool extents_no_wait = std::getenv("KYTY_SRT_EXTENTS_NO_WAIT") != nullptr;
	struct OptionalScope {
		bool active = t_gpu_sized_snapshot || extents_no_wait ||
		              ExtentsNoWaitShader(t_extents_shader_hash);
		OptionalScope() {
			if (active) SrtSetEvaluatingCondition(true);
		}
		~OptionalScope() {
			if (active) SrtSetEvaluatingCondition(false);
		}
	} scope;
	for (const auto& write: program.buffer_writes) {
		if (write.buffer >= failed.size() || failed[write.buffer] != 0u) {
			continue;
		}
		auto&                extent = snapshot.buffer_write_extents[write.buffer];
		ShaderBufferResource descriptor;
		WriteInterval        index;
		WriteInterval        offset;
		WriteInterval        soffset;
		uint32_t             budget    = 1024u;
		const auto           predicate = write.predicate.Resolve();
		if (program.buffer_writes_bounded[write.buffer] == 0u ||
		    !DecodeBufferDescriptor(snapshot.buffers[write.buffer], descriptor) ||
		    descriptor.SwizzleEnabled() ||
		    !EvaluateWriteInterval(program, runtime, walker, write.index, predicate, index, budget) ||
		    !EvaluateWriteInterval(program, runtime, walker, write.offset, predicate, offset, budget) ||
		    !EvaluateWriteInterval(program, runtime, walker, write.soffset, predicate, soffset,
		                           budget)) {
			failed[write.buffer] = 1u;
			extent               = {};
			continue;
		}
		if (descriptor.AddTid()) {
			index.hi += 63u;
		}
		// PS5 ISA, buffer addressing: base + soffset + offset + index * stride, at most 16 bytes a
		// store (dwordx4, format xyzw), and a store outside the descriptor's range is dropped.
		const uint64_t stride = descriptor.Stride();
		const uint64_t size   = descriptor.GetSize();
		const uint64_t begin =
		    std::min(index.lo * stride + offset.lo + soffset.lo + write.immediate, size);
		const uint64_t end =
		    std::min(index.hi * stride + offset.hi + soffset.hi + write.immediate + 16u, size);
		if (!extent.valid) {
			extent = {.begin = begin, .end = end, .valid = true};
		} else {
			extent.begin = std::min(extent.begin, begin);
			extent.end   = std::max(extent.end, end);
		}
	}
}

} // namespace

int LastIndirectImageFailureLine() {
	return g_indirect_failure_line;
}

namespace {

// KYTY_GPU_SIZED_BUFFERS=0 waits for GPU-written record counts of read-only buffers, as before.
bool GpuSizedBuffersEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_BUFFERS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// The record count bound for a read-only buffer whose count the GPU wrote: KYTY_GPU_SIZED_BUFFER_KB
// (default 256) bytes of records at the V#'s stride (dword 1, bits 16-29), or that many bytes for
// a raw buffer. The binding is clamped to mapped guest memory like any other.
uint32_t GpuSizedRecords(uint32_t dword1, bool written = false) {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_BUFFER_KB");
		const auto  kb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull;
		return (kb == 0 ? 256ull : kb) * 1024ull;
	}();
	static const uint64_t written_bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN_KB");
		const auto  kb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 4096ull;
		return (kb == 0 ? 4096ull : kb) * 1024ull;
	}();
	const auto     limit  = written ? written_bytes : bytes;
	const uint32_t stride = (dword1 >> 16u) & 0x3fffu;
	return static_cast<uint32_t>(stride != 0u ? limit / stride : limit);
}

// KYTY_GPU_SIZED_WRITTEN=0 waits for GPU-written record counts of written buffers, as before.
// On: a written, non-atomic buffer whose count the GPU wrote is bound with a conservative count
// (KYTY_GPU_SIZED_WRITTEN_KB, default 4096 KB of records) instead of waiting for the GPU. Its
// shader tests its records against the count it reads itself (CS fb5ee753d43f3f0b, a sort of
// GPU-counted keys: ~20 ms of GPU-thread wait a frame at the title menu, 2026-10-09).
// Only for the shaders listed (KYTY_GPU_SIZED_WRITTEN_SHADERS, hex hashes, default the sort
// above): applied to every written buffer, CS 380b6ffba703cef1's binding lost the device.
// KYTY_GPU_SIZED_DYNAMIC=1: a buffer bound with a conservative record count (its count, V#
// dword 2, was written by the GPU) keeps its exact bounds on the GPU: the shader reads the count
// from the flattened SRT (the GPU fills that slot before the dispatch) and bounds its accesses by
// it (BufferResource::gpu_records_flat). Written buffers then size without waiting too: the
// conservative count alone let stores past the real count land (KYTY_GPU_SIZED_WRITTEN lost the
// device), with the exact bound they are dropped as on the console. Default shaders: the two whose
// counts made the GPU thread wait for the GPU every frame in gameplay (fb5ee753d43f3f0b, an async
// sort, and 380b6ffba703cef1); every shader unless KYTY_GPU_SIZED_WRITTEN_SHADERS lists some.
bool GpuSizedDynamicEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_DYNAMIC");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool GpuSizedWrittenEnabled(uint64_t shader_hash) {
	// Opt-in (KYTY_GPU_SIZED_WRITTEN=1): on by default, the title menu lost the device at ~57 s in
	// three runs of three (a DISPATCH_INDIRECT hung, 2026-10-09); off, it did not.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN");
		return (value != nullptr && value[0] == '1') || GpuSizedDynamicEnabled();
	}();
	// Dynamic bounds make the conservative count safe, so every shader by default: the GPU-thread
	// wait otherwise moves to the next shader reading the same GPU-written counts (CS
	// 380b6ffba703cef1 once fb5ee753d43f3f0b no longer waited, 2026-10-17).
	static const bool all = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN_SHADERS");
		return value != nullptr ? std::string_view(value) == "all" : GpuSizedDynamicEnabled();
	}();
	if (all) {
		return enabled;
	}
	static const std::vector<uint64_t> shaders = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_WRITTEN_SHADERS");
		std::string list  = value != nullptr ? value
		                    : GpuSizedDynamicEnabled() ? "fb5ee753d43f3f0b,2ea3957af2daf690"
		                                               : "fb5ee753d43f3f0b";
		std::vector<uint64_t> result;
		size_t                at = 0;
		while (at < list.size()) {
			const auto end = list.find_first_of(",; ", at);
			const auto item =
			    list.substr(at, end == std::string::npos ? std::string::npos : end - at);
			if (!item.empty()) {
				result.push_back(std::strtoull(item.c_str(), nullptr, 16));
			}
			if (end == std::string::npos) break;
			at = end + 1;
		}
		return result;
	}();
	return enabled && std::find(shaders.begin(), shaders.end(), shader_hash) != shaders.end();
}

// Whether `value` reads a flat slot the GPU fills in this refresh (SrtGpuFill): the host's copy
// of such a slot is a placeholder (0), not the value.
bool DependsOnGpuFill(const ResourcePlan& program, Value value,
                      const std::vector<SrtGpuFill>& fills) {
	if (fills.empty()) {
		return false;
	}
	std::vector<Value>              stack {value};
	std::unordered_set<const Inst*> seen;
	while (!stack.empty() && seen.size() < 256u) {
		const auto* inst = stack.back().Resolve().TryInstruction();
		stack.pop_back();
		if (inst == nullptr || !seen.insert(inst).second) {
			continue;
		}
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 &&
			    slot.U32() < program.srt_reads.size()) {
				const auto offset = program.srt_reads[slot.U32()].flat_offset;
				for (const auto& fill: fills) {
					if (fill.flat_offset == offset) {
						return true;
					}
				}
			}
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			stack.push_back(inst->Arg(arg));
		}
	}
	// Too large to follow: not known.
	return !stack.empty();
}

// KYTY_GPU_SIZED_DYNAMIC: the bound record count of a buffer the shader bounds by its exact count
// (gpu_records_flat). A fixed bound is either too small (CS fb5ee753d43f3f0b's count passed 4 MB of
// records: its keys past the bound were dropped and a later draw, 618eca57f179de7d, cost 25-80 ms
// of GPU time a frame) or binds, and marks GPU-written, far more memory than the shader touches (64
// MB: other shaders then waited for those bytes, and the game crashed once). The bound is twice
// the largest count seen so far, from copies of the count that never make the host wait
// (SrtRuntime::peek_gpu_dword), at least KYTY_GPU_SIZED_DYNAMIC_MIN_KB (default 256) and at most
// KYTY_GPU_SIZED_DYNAMIC_MAX_KB (default 262144) of records; the fixed bound until a copy lands.
uint32_t LearnedRecords(uint64_t shader_hash, uint32_t buffer, uint32_t flat_offset,
                        uint32_t add_flat, uint32_t add, uint32_t dword1, uint32_t fallback,
                        const SrtRuntime& runtime, const ResourceSnapshot& snapshot) {
	static const uint64_t min_bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_DYNAMIC_MIN_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull) * 1024ull;
	}();
	static const uint64_t max_bytes = [] {
		const char* value = std::getenv("KYTY_GPU_SIZED_DYNAMIC_MAX_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 262144ull) * 1024ull;
	}();
	if (runtime.peek_gpu_dword == nullptr) {
		return fallback;
	}
	// A slot the GPU fills: its last seen value; one the host read: its value.
	bool       any       = false;
	const auto slot_value = [&](uint32_t offset) -> uint64_t {
		for (const auto& fill: snapshot.gpu_fills) {
			if (fill.flat_offset == offset) {
				any = true;
				return runtime.peek_gpu_dword(fill.address);
			}
		}
		return offset < snapshot.flattened_srt.size() ? snapshot.flattened_srt[offset] : 0u;
	};
	uint64_t total = slot_value(flat_offset) + add;
	if (add_flat != BufferResource::NoGpuRecords) {
		total += slot_value(add_flat);
	}
	if (!any) {
		return fallback;
	}
	const auto seen = static_cast<uint32_t>(std::min<uint64_t>(total, UINT32_MAX));
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> largest;
	uint32_t                                      most = 0;
	{
		std::lock_guard lock(mutex);
		auto&           entry = largest[shader_hash ^ (uint64_t {buffer} << 48u)];
		const bool      grew  = seen > entry;
		entry                 = std::max(entry, seen);
		most                  = entry;
		if (grew) {
			if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
				std::fprintf(stderr, "[gpu-sized-dyn] %016llx buffer %u: largest count %u\n",
				             static_cast<unsigned long long>(shader_hash), buffer, most);
			}
		}
	}
	if (most == 0) {
		return fallback;
	}
	const uint64_t stride = (dword1 >> 16u) & 0x3fffu;
	const uint64_t unit   = stride == 0u ? 1u : stride;
	const uint64_t bytes  = std::clamp<uint64_t>(uint64_t {most} * unit * 2u, min_bytes, max_bytes);
	return static_cast<uint32_t>(std::min<uint64_t>(bytes / unit, UINT32_MAX));
}

void NoteGpuSizedBuffer(uint64_t shader_hash) {
	static std::atomic<uint64_t> count {0};
	const auto                   n = count.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((n & (n - 1)) == 0) {
		std::fprintf(stderr, "[gpu-sized-buffer] %llu bindings sized without waiting (last: %016llx)\n",
		             static_cast<unsigned long long>(n), static_cast<unsigned long long>(shader_hash));
	}
}

bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                              ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                              bool& served) {
	served                  = false;
	g_indirect_failure_line = 0;
	t_gpu_sized_snapshot    = false;
	snapshot.bindless_heaps.clear();
	snapshot.bindless_sampler_heaps.clear();
	if (!program.resource_tracking_complete ||
	    (program.requires_specialization_memory && runtime.read_specialization_memory == nullptr)) {
		return FailIndirect(__LINE__);
	}
	const bool capture_reads = program.capture_specialization_reads;
	auto& reads = snapshot.specialization_reads;
	reads.clear();
	ReadCapture capture {runtime, reads};
	SrtRuntime observed = runtime;
	if (capture_reads) {
		observed.userdata = &capture;
		observed.read_specialization_memory = runtime.read_specialization_memory != nullptr
		                                         ? CaptureStrictRead : nullptr;
		observed.read_memory = CaptureOrdinaryRead;
		// Reads through the fast page path are captured there.
		observed.capture_ranges = &reads;
	}
	SrtWalker clean(program, CleanRuntime(observed));
	SrtWalker walker(program, observed, program.clean_flat_slots,
	                 capture_reads || program.requires_specialization_memory ? &clean : nullptr);
	SrtTraceSession trace(program, clean, walker, observed);
	g_materialize_phase = "flat";
	if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
		if (static const bool debug = std::getenv("KYTY_DEBUG_RESOURCES") != nullptr; debug) {
			static std::atomic<uint32_t> printed {0};
			if (printed.fetch_add(1) < 12) {
				std::fprintf(stderr,
				             "[flat-fail] %016llx plan_complete=%d reads=%zu failure=%s addr=0x%llx "
				             "clean_failure=%s clean_addr=0x%llx user_data=%zu\n",
				             static_cast<unsigned long long>(program.shader_hash),
				             program.srt_plan_complete ? 1 : 0, program.srt_reads.size(),
				             walker.ReadFailure() ? walker.ReadFailure() : "-",
				             static_cast<unsigned long long>(walker.ReadFailureAddress()),
				             clean.ReadFailure() ? clean.ReadFailure() : "-",
				             static_cast<unsigned long long>(clean.ReadFailureAddress()),
				             runtime.user_data.size());
				std::string text = "  user_data:";
				for (size_t u = 0; u < runtime.user_data.size(); u++) {
					text += fmt::format("{}{:08x}", u % 8 == 0 ? fmt::format("\n   s{:02}: ", u) : " ",
					                    runtime.user_data[u]);
				}
				text += "\n";
				std::fputs(text.c_str(), stderr);
			}
		}
		return FailIndirect(__LINE__);
	}
	const auto active = std::span<const uint8_t>(program.active_sources);
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	const auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = clean.Evaluate(fill.values[i], stored[i]) && stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return (capture_reads && written ? clean : walker).EvaluateDescriptor(source, value);
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	g_materialize_phase = "buffers";
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		// KYTY_GPU_SIZED_DYNAMIC: the flat SRT dword holding this buffer's exact record count.
		uint32_t gpu_records_flat = BufferResource::NoGpuRecords;
		uint32_t gpu_records_add_flat = BufferResource::NoGpuRecords;
		uint32_t gpu_records_add      = 0;
		bool     gpu_records_ssa      = false;
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return false;
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return false;
			if (static const char* diag = std::getenv("KYTY_SRT_FILL_REASON");
			    diag != nullptr &&
			    std::string_view(diag).find(fmt::format("{:016x}", program.shader_hash)) !=
			        std::string_view::npos) {
				static std::atomic<uint32_t> printed {0};
				if (printed.fetch_add(1, std::memory_order_relaxed) < 64) {
					std::fprintf(stderr,
					             "[gpu-sized-buffer] %016llx buffer %u source %u written=%d atomic=%d "
					             "scalar=%d indirect=%d active=%d dwords=%u\n",
					             static_cast<unsigned long long>(program.shader_hash), i, base.source,
					             base.written ? 1 : 0, base.atomic ? 1 : 0, base.scalar ? 1 : 0,
					             source->indirect_descriptor.has_value() ? 1 : 0,
					             active.empty() || active[base.source] ? 1 : 0, source->dword_count);
				}
			}
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (active.empty() || active[base.source]) {
					if (!MaterializeIndirectDescriptor(
					        program, *source->indirect_descriptor, i, 4u, observed, clean, snapshot,
					        snapshot.buffers, specialization.buffers, ShaderInfo::MaxBuffers,
					        NormalizeIndirectStoreBuffer))
						return false;
				}
			} else if ((!base.written || GpuSizedWrittenEnabled(program.shader_hash)) &&
			           (!base.atomic || GpuSizedAtomicEnabled(program.shader_hash)) &&
			           GpuSizedBuffersEnabled() && (active.empty() || active[base.source])) {
				// A read-only buffer whose record count (V# dword 2) the GPU wrote: binding it
				// needs the base, stride and format, not the exact count. Read the descriptor
				// refusing GPU-written bytes; when that fails, read dwords 0, 1 and 3 normally and
				// give dword 2 a conservative count (GpuSizedRecords), so the GPU thread does not
				// wait for the GPU. The shader keeps its own bounds tests (the count in its SGPRs
				// comes from the flat buffer, which the GPU fills).
				auto&       walker_ref = capture_reads && base.written ? clean : walker;
				const auto& src        = program.descriptor_sources[base.source];
				// A refused read can still let an evaluation succeed, with a value that is not
				// known (a condition's): dwords 0, 1 and 3 are read normally, and dword 2 counts as
				// refused when any read under it was (SrtConditionRefusals). Evaluating the whole
				// descriptor refusing reads bound such a value as the count (the title menu lost
				// the device in three runs of three, 2026-10-09).
				bool ok = false;
				if (src.dword_count != 4u) {
					ok = evaluate(base.source, snapshot.buffers[i], base.written);
				} else {
					auto& value       = snapshot.buffers[i];
					value             = {};
					value.dword_count = 4u;
					ok                = true;
					bool sized        = false;
					for (uint32_t d = 0; d < 4u && ok; ++d) {
						if (d == 2u) {
							const auto refusals = SrtConditionRefusals();
							SrtSetEvaluatingCondition(true);
							sized = !walker_ref.Evaluate(src.dwords[2], value.dwords[2]);
							SrtSetEvaluatingCondition(false);
							sized = sized || SrtConditionRefusals() != refusals ||
							        DependsOnGpuFill(program, src.dwords[2], snapshot.gpu_fills);
							continue;
						}
						ok = walker_ref.Evaluate(src.dwords[d], value.dwords[d]);
					}
					if (ok && sized && GpuSizedDynamicEnabled()) {
						// The count must be one flat slot read as it is, so the shader can read
						// it there; bounds of swizzled or lane-indexed buffers are not a plain
						// length. Otherwise a written buffer waits for its exact count.
						// Also flat + flat and flat + immediate (CS 380b6ffba703cef1, f8eeaa7f21b1450a).
						const auto flat_of = [&](Value v) -> uint32_t {
							const auto* inst = v.Resolve().TryInstruction();
							if (inst == nullptr || inst->GetOpcode() != ValueOpcode::ReadConst) {
								return BufferResource::NoGpuRecords;
							}
							const auto slot = inst->Arg(1).Resolve();
							return slot.IsImmediate() && slot.GetType() == Type::U32 &&
							               slot.U32() < program.srt_reads.size()
							           ? program.srt_reads[slot.U32()].flat_offset
							           : BufferResource::NoGpuRecords;
						};
						const auto imm_of = [](Value v, uint32_t& out) {
							v = v.Resolve();
							if (!v.IsImmediate() || v.GetType() != Type::U32) return false;
							out = v.U32();
							return true;
						};
						const auto* read = src.dwords[2].Resolve().TryInstruction();
						if (read != nullptr && ((value.dwords[1] >> 31u) & 1u) == 0u &&
						    ((value.dwords[3] >> 23u) & 1u) == 0u) {
							if (read->GetOpcode() == ValueOpcode::ReadConst) {
								gpu_records_flat = flat_of(src.dwords[2]);
							} else if (read->GetOpcode() == ValueOpcode::IAdd32) {
								const auto a = flat_of(read->Arg(0));
								const auto b = flat_of(read->Arg(1));
								uint32_t   imm = 0;
								if (a != BufferResource::NoGpuRecords && b != BufferResource::NoGpuRecords) {
									gpu_records_flat     = a;
									gpu_records_add_flat = b;
								} else if (a != BufferResource::NoGpuRecords && imm_of(read->Arg(1), imm)) {
									gpu_records_flat = a;
									gpu_records_add  = imm;
								} else if (b != BufferResource::NoGpuRecords && imm_of(read->Arg(0), imm)) {
									gpu_records_flat = b;
									gpu_records_add  = imm;
								}
							}
						}
						if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log) {
							static std::mutex                   mutex;
							static std::unordered_set<uint64_t> seen;
							std::lock_guard                     lock(mutex);
							if (seen.insert(program.shader_hash ^ (uint64_t {i} << 56u)).second) {
								// The count's expression, two levels deep, when it is not a flat slot.
								std::string shape;
								const std::function<void(Value, int)> describe = [&](Value v, int depth) {
									v = v.Resolve();
									if (v.IsImmediate()) {
										shape += fmt::format("#{:x}", v.GetType() == Type::U32 ? v.U32() : 0u);
										return;
									}
									const auto* inst = v.TryInstruction();
									if (inst == nullptr) {
										shape += "?";
										return;
									}
									shape += ValueOpcodeName(inst->GetOpcode());
									if (depth == 0 || inst->NumArgs() == 0) return;
									shape += "(";
									for (size_t a = 0; a < inst->NumArgs(); a++) {
										if (a != 0) shape += ",";
										describe(inst->Arg(a), depth - 1);
									}
									shape += ")";
								};
								if (gpu_records_flat == BufferResource::NoGpuRecords) {
									describe(src.dwords[2], 3);
								}
								std::fprintf(stderr,
								             "[gpu-sized-dyn] %016llx buffer %u written=%d flat=%d %s\n",
								             static_cast<unsigned long long>(program.shader_hash),
								             i, base.written ? 1 : 0,
								             gpu_records_flat == BufferResource::NoGpuRecords
								                 ? -1
								                 : static_cast<int>(gpu_records_flat),
								             shape.c_str());
							}
						}
						// No flat slot: the shader bounds its accesses by the count its own
						// descriptor computes (KYTY_GPU_SIZED_SSA=0 waits for the count instead).
						static const bool ssa = [] {
							const char* env = std::getenv("KYTY_GPU_SIZED_SSA");
							return env == nullptr || env[0] != '0';
						}();
						if (gpu_records_flat == BufferResource::NoGpuRecords && ssa &&
						    ((value.dwords[1] >> 31u) & 1u) == 0u &&
						    ((value.dwords[3] >> 23u) & 1u) == 0u) {
							gpu_records_ssa = true;
						}
						if (gpu_records_flat == BufferResource::NoGpuRecords && !gpu_records_ssa &&
						    base.written) {
							ok    = walker_ref.Evaluate(src.dwords[2], value.dwords[2]);
							sized = false;
						}
					}
					if (ok && sized) {
						value.dwords[2] = GpuSizedRecords(value.dwords[1], base.written);
						if (gpu_records_flat != BufferResource::NoGpuRecords) {
							value.dwords[2] = LearnedRecords(program.shader_hash, i, gpu_records_flat,
							                                 gpu_records_add_flat, gpu_records_add,
							                                 value.dwords[1], value.dwords[2],
							                                 runtime, snapshot);
						}
						t_gpu_sized_snapshot = true;
						NoteGpuSizedBuffer(program.shader_hash);
					}
				}
				if (!ok) {
					return false;
				}
			} else if (!evaluate(base.source, snapshot.buffers[i], base.written)) {
				return false;
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		const bool null_descriptor =
		    std::all_of(descriptor_value.dwords.begin(), descriptor_value.dwords.begin() + 4,
		                [](uint32_t word) { return word == 0u; });
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		if (BufferStrideAsData()) {
			// v424: stride and index stride are per-draw data (BufferStrideData below); only the
			// swizzle enable and ADD_TID stay in the specialization. A null V# (all zero) reads
			// zeros through the zero bound length whatever its stride, so it is not specialized
			// as zero-stride either (one variant with the real buffer's).
			buffer.packed_stride &= (1u << 14u) | (1u << 20u);
			if (null_descriptor) {
				buffer.zero_stride_oob = false;
			}
		}
		buffer.gpu_records_flat   = gpu_records_flat;
		buffer.gpu_records_add_flat = gpu_records_add_flat;
		buffer.gpu_records_add      = gpu_records_add;
		buffer.gpu_records_ssa      = gpu_records_ssa;
		buffer.write_tracked      = WriteBitmapEnabled() && i < program.info.buffers.size() &&
		                            base.written && base.image_alias == BufferResource::NoImageAlias &&
		                            i < program.buffer_writes_bounded.size() &&
		                            program.buffer_writes_bounded[i] == 0u;
	}
	{
		// Write extents only size the GPU-written range; their reads do not specialize the
		// shader, so they stay out of the specialization-read proof.
		const auto captured = reads.size();
		g_materialize_phase = "extents";
		ComputeBufferWriteExtents(program, observed, walker, snapshot);
		if (capture_reads) {
			reads.resize(captured);
		}
	}
	g_materialize_phase = "images";
	g_debug_raw_images.assign(program.info.images.size(), DescriptorValue {});
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube = image.cube,
		    .bindless = image.bindless,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return FailIndirect(__LINE__);
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			if (indirect.bindless) {
				DescriptorValue table_value;
				if (!clean.EvaluateDescriptor(indirect.table_source, table_value) ||
				    !MaterializeBindlessImage(indirect, table_value, i, snapshot, specialization)) {
					return FailIndirect(__LINE__);
				}
				continue;
			}
			const auto srt_size        = snapshot.flattened_srt.size();
			const auto image_count     = snapshot.images.size();
			const auto spec_count      = specialization.images.size();
			const auto failure_line    = g_indirect_failure_line;
			if (!MaterializeIndirectDescriptor(
			        program, indirect, i, 8u, observed, clean, snapshot, snapshot.images,
			        specialization.images, UINT32_MAX, [&](DescriptorValue& value) {
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				// A store-only image picked per material from a table too large to enumerate
				// (Wolverine's terrain and scatter write a per-material image from a 20 MB
				// material table): bind a null image, dropping those stores, rather than
				// skipping the draw itself.
				static const bool null_writes = [] {
					const char* env = std::getenv("KYTY_NULL_UNENUMERABLE_WRITES");
					return env == nullptr || env[0] != '0';
				}();
				if (!null_writes || image.read || image.atomic || !image.written) {
					return FailIndirect(__LINE__);
				}
				if (static const std::string debug = [] {
					    const char* env = std::getenv("KYTY_DEBUG_RESOURCES");
					    return env != nullptr ? std::string(env) : std::string();
				    }();
				    !debug.empty() &&
				    debug.find(fmt::format("{:016x}", program.shader_hash)) != std::string::npos) {
					static std::atomic<uint32_t> printed_debug {0};
					if (printed_debug.fetch_add(1, std::memory_order_relaxed) < 16) {
						DescriptorValue material_value, table_value;
						const bool m = indirect.material_source != UINT32_MAX &&
						               clean.EvaluateDescriptor(indirect.material_source, material_value);
						const bool t = clean.EvaluateDescriptor(indirect.table_source, table_value);
						std::fprintf(stderr,
						             "[null-write-why] %016llx image %u line=%d material(%d,n=%u)=%08x %08x %08x %08x "
						             "table(%d,n=%u)=%08x %08x %08x %08x\n",
						             static_cast<unsigned long long>(program.shader_hash), i, g_indirect_failure_line,
						             m ? 1 : 0, material_value.dword_count, material_value.dwords[0],
						             material_value.dwords[1], material_value.dwords[2], material_value.dwords[3],
						             t ? 1 : 0, table_value.dword_count, table_value.dwords[0], table_value.dwords[1],
						             table_value.dwords[2], table_value.dwords[3]);
					}
				}
				snapshot.flattened_srt.resize(srt_size);
				snapshot.images.resize(image_count);
				specialization.images.resize(spec_count);
				snapshot.images[i] = {.dword_count = 8u};
				g_indirect_failure_line = failure_line;
				static std::atomic<uint32_t> printed {0};
				if (printed.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::fprintf(stderr, "[null-write-image] %016llx image %u\n",
					             static_cast<unsigned long long>(program.shader_hash), i);
				}
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return FailIndirect(__LINE__);
			}
			if (g_debug_raw_images.size() < snapshot.images.size()) {
				g_debug_raw_images.resize(snapshot.images.size());
			}
			g_debug_raw_images[i] = snapshot.images[i];
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	g_materialize_phase = "samplers";
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
		const auto  source_index = program.info.samplers[i].source;
		const auto* source       = Source(program, source_index);
		if (source != nullptr && source->bindless_sampler.has_value()) {
			// The shader indexes the host's mirror of the heap with two patched words (region
			// base, entry count); zeros select the default sampler in slot 0. Set 0 still binds
			// the default sampler for this resource.
			const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			snapshot.samplers[i].dword_count = 4u;
			std::ranges::copy(BindlessDefaultSampler, snapshot.samplers[i].dwords.begin());
			specialization.samplers[i] = {.bindless = true, .bindless_mapping_offset = mapping_offset};
			DescriptorValue      table;
			ShaderBufferResource heap;
			if ((active.empty() || active[source_index]) &&
			    clean.EvaluateDescriptor(source_index, table) &&
			    DecodeBufferDescriptor(table, heap) && heap.Base48() != 0) {
				snapshot.bindless_sampler_heaps.push_back(
				    {.base           = heap.Base48(),
				     .size           = heap.GetSize(),
				     .table_offset   = source->bindless_sampler->table_offset,
				     .sampler        = i,
				     .mapping_offset = mapping_offset});
			}
			continue;
		}
		if (!evaluate(source_index, snapshot.samplers[i])) {
			return FailIndirect(__LINE__);
		}
		if (program.info.samplers[i].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			// MipNone always selects the base level. Explicit point gathers currently require
			// encoded-zero primary and secondary bias. A gather returns the texels of one level,
			// so linear mip filtering gathers the nearest level, as point does (GatherMip rounds):
			// the game's HiZ compute shader 0x37ab7d3ab9bff0b9 samples linearly, and rejecting it
			// skipped its dispatches every frame (the jungle floor went missing).
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires mip filtering None or Point with zero LOD biases");
			}
		}
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	// KYTY_DEBUG_RESOURCES=<hash,...>: print that shader's image and sampler descriptors
	// (first 6 refreshes).
	if (static const std::string debug = [] {
		    const char* env = std::getenv("KYTY_DEBUG_RESOURCES");
		    return env != nullptr ? std::string(env) : std::string();
	    }();
	    !debug.empty() && debug.find(fmt::format("{:016x}", program.shader_hash)) != std::string::npos) {
		static std::mutex                             printed_mutex;
		static std::unordered_map<uint64_t, uint32_t> printed_per_shader;
		bool                                          print = false;
		{
			std::lock_guard lock(printed_mutex);
			static const uint32_t limit = [] {
				const char* env = std::getenv("KYTY_DEBUG_RESOURCES_N");
				return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 3u;
			}();
			print = printed_per_shader[program.shader_hash]++ < limit;
		}
		if (print) {
			std::string text = fmt::format("[debug-res] {:016x} images={} samplers={} buffers={}\n",
			                               program.shader_hash, snapshot.images.size(),
			                               snapshot.samplers.size(), snapshot.buffers.size());
			for (size_t i = 0; i < program.info.images.size(); i++) {
				const auto& image  = program.info.images[i];
				const auto* source = Source(program, image.source);
				text += fmt::format("  info{} pc=0x{:x} dim={} read={} written={} mips={} src={} n={} ind={}",
				                    i, image.first_use_pc, static_cast<int>(image.dimension),
				                    image.read ? 1 : 0, image.written ? 1 : 0, image.mip_count,
				                    image.source, source != nullptr ? source->dword_count : 0u,
				                    source != nullptr && source->indirect_descriptor.has_value() ? 1 : 0);
				if (source != nullptr && source->indirect_descriptor.has_value()) {
					const auto& ind = *source->indirect_descriptor;
					text += fmt::format(" mat={} table={} sel={}+{} tab={}*{} shift={} key>>{}&{:x} "
					                    "bindless={} srcs={}",
					                    ind.material_source, ind.table_source, ind.selector_offset,
					                    ind.selector_stride, ind.table_offset, ind.table_stride,
					                    ind.selector_shift, ind.key_shift, ind.key_mask,
					                    ind.bindless ? 1 : 0, ind.sources.size());
				}
				text += "\n";
			}
			for (size_t i = 0; i < snapshot.images.size(); i++) {
				const auto& d = snapshot.images[i].dwords;
				const auto  w = ((d[2] >> 14u) & 0x3fffu) + 1u;
				const auto  h = ((d[2] >> 28u) | ((d[3] & 0x3ffu) << 4u)) + 1u;
				text += fmt::format(
				    "  img{} bindless={} indirect_root={} base=0x{:x} fmt={} {}x{} dw={:08x} {:08x} "
				    "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
				    i, specialization.images[i].bindless ? 1 : 0,
				    specialization.images[i].indirect_root,
				    (static_cast<uint64_t>(d[1] & 0xffu) << 40u) | (static_cast<uint64_t>(d[0]) << 8u),
				    (d[1] >> 20u) & 0x1ffu, w, h, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
				if (i < g_debug_raw_images.size()) {
					const auto& r = g_debug_raw_images[i].dwords;
					text += fmt::format("    evaluated: {:08x} {:08x} {:08x} {:08x} {:08x} "
					                    "{:08x} {:08x} {:08x} source={} active={}\n",
					                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
					                    program.info.images[i].source,
					                    program.active_sources.empty() ||
					                            program.active_sources[program.info.images[i].source]
					                        ? 1 : 0);
				}
			}
			g_debug_raw_images.clear();
			for (size_t i = 0; i < snapshot.buffers.size(); i++) {
				const auto& d = snapshot.buffers[i].dwords;
				text += fmt::format("  buf{} {:08x} {:08x} {:08x} {:08x}\n", i, d[0], d[1], d[2], d[3]);
			}
			std::fputs(text.c_str(), stderr);
		}
	}
	if (!BuildResourceSpecialization(program, snapshot, specialization,
	                                 runtime.float_image_atomics)) {
		return FailIndirect(__LINE__);
	}
	// v432: read-only formatted buffers read their format and selects from the draw.
	for (uint32_t i = 0; i < specialization.buffers.size(); i++) {
		if (DynamicBufferFormat(program.info, specialization, i)) {
			specialization.buffers[i].descriptor_format  = Prospero::BufferFormat::kInvalid;
			specialization.buffers[i].descriptor_swizzle = DstSel(4, 5, 6, 7);
		}
	}
	served = trace.FullyServed();
	trace.Succeeded();
	return true;
}

// KYTY_SRT_TRACE_VERIFY=1: a refresh served from a replay trace is evaluated again without it,
// and any difference stops the emulator.
void VerifyTracedRefresh(const ResourcePlan& program, const SrtRuntime& runtime,
                         const ResourceSnapshot& snapshot,
                         const ResourceSpecialization& specialization) {
	ResourceSnapshot       reference_snapshot;
	ResourceSpecialization reference_specialization;
	bool                   served = false;
	auto&                  suppressed = SrtTraceSession::Suppressed();
	suppressed                        = true;
	// The reference lists its GPU-filled slots apart.
	SrtRuntime reference_runtime = runtime;
	if (reference_runtime.gpu_fills != nullptr) {
		reference_runtime.gpu_fills = &reference_snapshot.gpu_fills;
	}
	const bool ok = MaterializeResourcesImpl(program, reference_runtime, reference_snapshot,
	                                         reference_specialization, served);
	suppressed    = false;
	// A GPU-filled slot's host value is a placeholder, and whether its bytes were GPU-written can
	// change between the two walks (a download in between): compare the other slots.
	auto flat           = snapshot.flattened_srt;
	auto reference_flat = reference_snapshot.flattened_srt;
	for (const std::vector<SrtGpuFill>* fills: {&snapshot.gpu_fills, static_cast<const std::vector<SrtGpuFill>*>(&reference_snapshot.gpu_fills)}) {
		for (const auto& fill: *fills) {
			if (fill.flat_offset < flat.size()) flat[fill.flat_offset] = 0u;
			if (fill.flat_offset < reference_flat.size()) reference_flat[fill.flat_offset] = 0u;
		}
	}
	auto reads           = snapshot.specialization_reads;
	auto reference_reads = reference_snapshot.specialization_reads;
	std::ranges::sort(reads);
	reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
	std::ranges::sort(reference_reads);
	reference_reads.erase(std::unique(reference_reads.begin(), reference_reads.end()),
	                      reference_reads.end());
	const char* field = nullptr;
	if (!ok) field = "reference refresh failed";
	else if (flat != reference_flat) field = "flattened_srt";
	else if (snapshot.buffers != reference_snapshot.buffers) field = "buffers";
	else if (snapshot.images != reference_snapshot.images) field = "images";
	else if (snapshot.samplers != reference_snapshot.samplers) field = "samplers";
	else if (snapshot.user_data != reference_snapshot.user_data) field = "user_data";
	else if (snapshot.buffer_write_extents != reference_snapshot.buffer_write_extents) field = "buffer_write_extents";
	else if (snapshot.bindless_heaps != reference_snapshot.bindless_heaps) field = "bindless_heaps";
	else if (snapshot.bindless_sampler_heaps != reference_snapshot.bindless_sampler_heaps) field = "bindless_sampler_heaps";
	else if (!(snapshot.uniform_fill == reference_snapshot.uniform_fill)) field = "uniform_fill";
	else if (reads != reference_reads) field = "specialization_reads";
	else if (!(specialization == reference_specialization)) field = "specialization";
	if (field != nullptr) {
		std::string detail;
		if (std::strcmp(field, "flattened_srt") == 0) {
			const auto& a = snapshot.flattened_srt;
			const auto& b = reference_snapshot.flattened_srt;
			for (size_t i = 0; i < std::max(a.size(), b.size()); i++) {
				const auto x = i < a.size() ? a[i] : 0xdeadbeefu;
				const auto y = i < b.size() ? b[i] : 0xdeadbeefu;
				if (x != y) {
					detail += fmt::format(" [{}] {:08x}/{:08x}", i, x, y);
				}
			}
			detail += fmt::format(" fills={}", snapshot.gpu_fills.size());
			for (const auto& fill: snapshot.gpu_fills) {
				detail += fmt::format(" {}@{:x}", fill.flat_offset, fill.address);
			}
		}
		EXIT("SRT trace replay differs from the walk: shader %016" PRIx64 " stage %u: %s%s\n",
		     program.shader_hash, static_cast<uint32_t>(program.stage), field, detail.c_str());
	}
	static std::atomic<uint64_t> verified {0};
	if (verified.fetch_add(1, std::memory_order_relaxed) % 100000 == 0) {
		std::fprintf(stderr, "SRT trace replay verified: %llu\n",
		             static_cast<unsigned long long>(verified.load()));
	}
}

} // namespace

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	bool       served = false;
	const bool ok     = MaterializeResourcesImpl(program, runtime, snapshot, specialization, served);
	static const bool verify = std::getenv("KYTY_SRT_TRACE_VERIFY") != nullptr;
	if (ok && served && verify) {
		VerifyTracedRefresh(program, runtime, snapshot, specialization);
	}
	return ok;
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	std::vector<uint8_t> dynamic_format(specialization.buffers.size(), 0);
	for (uint32_t index = 0; index < specialization.buffers.size() && index < original_buffer_count; ++index) {
		dynamic_format[index] = DynamicBufferFormat(program.info, specialization, index) ? 1u : 0u;
	}
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.write_tracked              = source.write_tracked;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.gpu_records_flat           = source.gpu_records_flat;
		buffer.gpu_records_add_flat       = source.gpu_records_add_flat;
		buffer.gpu_records_add            = source.gpu_records_add;
		buffer.gpu_records_ssa            = source.gpu_records_ssa;
		buffer.dynamic_format             = dynamic_format[index] != 0u;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		// Before the fields below overwrite the plan's (DynamicImageClass reads the plan's cube).
		const bool dynamic_class = index < original_image_count && DynamicImageClass(program.info, index, source);
		const bool dynamic_layer = index < original_image_count && DynamicImageLayer(program.info, index, source);
		const bool dynamic_cube  = index < original_image_count && DynamicImageCube(program.info, index, source);
		const bool dynamic_3d    = index < original_image_count && DynamicImage3D(program.info, index, source);
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.bindless                   = source.bindless;
		image.cube                       = source.cube;
		image.dynamic_class              = dynamic_class;
		image.dynamic_layer              = dynamic_layer;
		image.dynamic_cube               = dynamic_cube;
		image.dynamic_3d                 = dynamic_3d;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto original_sampler_count = samplers.size();
	for (uint32_t index = 0; index < original_sampler_count && index < specialization.samplers.size();
	     index++) {
		samplers[index].bindless                = specialization.samplers[index].bindless;
		samplers[index].bindless_mapping_offset = specialization.samplers[index].bindless_mapping_offset;
	}
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::PointInteger;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				if (memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds regardless of its address.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						for (uint32_t component = 0; component < count; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							const auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(one), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			// Enumerated candidates only sample; a bindless image is loaded per instruction and
			// serves every image operation.
			EXIT_IF(image.indirect_root == memory.resource && !image.bindless &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw);
		}
	}
	// Tracking gives an image index only to the accesses that survived its dead-code pass; an
	// access it removed keeps its T# register from translation, which need not be an image index.
	std::vector<bool> live_image_access(memory_info.size());
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (ImageOpcodeInfoOf(inst.GetOpcode()).access != ImageAccess::None) {
				live_image_access[inst.Flags<MemoryFlags>().index] = true;
			}
		}
	}
	for (uint32_t index = 0; index < memory_info.size(); index++) {
		auto& memory = memory_info[index];
		if (live_image_access[index] && memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap(memory.resource, "image access");
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap(buffer.image_alias, "buffer image alias");
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap(pair.image, "sampled pair");
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap(image.indirect_root, "indirect root");
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap(resource, "indirect resource");
		}
	}
	image_remap.Apply(images);
}

// KYTY_WRITE_BITMAP=1: buffers written at addresses the host cannot bound record the pages they
// write in a GPU bitmap, so a guest read of a page the shader did not write needs no download
// (rework.md, Phase 2 first step; compute shader eaa507075716aa3b adds into a 70 MiB range and
// caused half of all readbacks in a new game).
bool WriteBitmapEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
