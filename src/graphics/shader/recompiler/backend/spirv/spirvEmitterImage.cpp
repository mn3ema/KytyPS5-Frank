#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/frontend/decode/ImageOps.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t DmaskComponentIndex(uint32_t dmask, uint32_t component) {
	uint32_t index = 0;
	for (uint32_t i = 0; i < component; i++) {
		index += (dmask >> i) & 1u;
	}
	return index;
}

uint32_t DmaskComponent(uint32_t dmask, uint32_t index) {
	for (uint32_t component = 0; component < 4u; component++) {
		if (((dmask >> component) & 1u) != 0u && index-- == 0u) return component;
	}
	return 0u;
}

uint32_t ImageGatherComponent(uint32_t dmask) {
	switch (dmask) {
		case 0x2u: return 1;
		case 0x4u: return 2;
		case 0x8u: return 3;
		default: return 0;
	}
}

bool HasFlag(const IR::MemoryInfo& mem, uint32_t flag) {
	return (mem.image_sample_flags & flag) != 0u;
}

uint32_t AddressU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags, component);
	const auto packed = layout.bit_offset / 32u;
	if (packed >= address.NumArgs()) return ConstantU32(ctx.state, 0);
	auto value = ctx.Def(address.Arg(packed));
	if (layout.bit_width == 16u) {
		if ((layout.bit_offset & 31u) != 0u) {
			value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
			               ConstantU32(ctx.state, 16));
		}
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), value,
		               ConstantU32(ctx.state, 0xffffu));
	}
	return value;
}

uint32_t AddressF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto value = AddressU32(ctx, mem, address, component);
	return Decoder::ImageAddressComponentLayout(mem.image_sample_flags, component).bit_width == 16u
	           ? EmitF16BitsToF32(ctx.state, value)
	           : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), value);
}

ImageSampleLayout Layout(const IR::MemoryInfo& mem) {
	ImageSampleLayout layout;
	uint32_t          cursor = 0;
	const auto&       info   = ImageDimensionInfoFor(mem.image_dimension);
	if (HasFlag(mem, Decoder::ImageSampleFlagOffset)) layout.offset = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagBias)) layout.bias = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagCompare)) layout.dref = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagDerivative)) {
		layout.grad_x = cursor;
		cursor += info.spatial_components;
		layout.grad_y = cursor;
		cursor += info.spatial_components;
	}
	layout.coord = cursor;
	cursor += info.coordinate_components;
	if (HasFlag(mem, Decoder::ImageSampleFlagLod)) layout.lod = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagLodClamp)) layout.clamp = cursor++;
	return layout;
}

uint32_t ZeroF32(EmitterState& state) {
	return ConstantF32(state, 0);
}

// The view-relative mip IMAGE_GATHER4*_L reads: the instruction's LOD under the S# and T# clamps,
// rounded like a point mip filter, limited to the view's last level (`last_f32` / `last_u32`).
uint32_t GatherMipLevel(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                        const IR::Inst& address, const ImageSampleLayout& layout, uint32_t last_f32,
                        uint32_t last_u32) {
	auto& state = ctx.state;
	const auto Extract = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first),
		                           ConstantU32(state, count));
	};
	const auto LodFixed = [&](uint32_t word, uint32_t first) {
		return Binary(state, spv::OpFMul, TypeF32(state),
		              Unary(state, spv::OpConvertUToF, TypeF32(state), Extract(word, first, 12)),
		              ConstantF32Value(state, 1.0f / 256.0f));
	};
	const auto sampler_control = ctx.Arg(inst, 6);
	const auto preclamp = Binary(state, spv::OpINotEqual, TypeBool(state),
	                            Extract(sampler_control, 28, 1), ConstantU32(state, 0));
	const auto half = ConstantF32Value(state, 0.5f);
	const auto zero = ZeroF32(state);
	const auto base = Unary(state, spv::OpConvertUToF, TypeF32(state),
	                        Extract(ctx.Arg(inst, 4), 12, 4));
	const auto minimum = Binary(state, spv::OpFSub, TypeF32(state),
	                            LodFixed(ctx.Arg(inst, 3), 8), base);
	const auto last = last_f32;
	auto lod = Binary(state, spv::OpFAdd, TypeF32(state),
	                  AddressF32(ctx, mem, address, layout.lod),
	                  Select(state, TypeF32(state), preclamp, half, zero));
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
	    state, lod, LodFixed(ctx.Arg(inst, 5), 0), LodFixed(ctx.Arg(inst, 5), 12));
	// T# minimum LOD is in physical mip space and applies after the S# view-relative clamp.
	lod = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(state, lod, minimum);
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, lod, zero, last);
	lod = Binary(state, spv::OpFAdd, TypeF32(state), lod,
	             Select(state, TypeF32(state), preclamp, zero, half));
	// The clamped value is nonnegative, so conversion performs the integer mip rounding.
	const auto mip = EmitUMin32(state, Unary(state, spv::OpConvertFToU, TypeU32(state), lod),
	                           last_u32);
	const auto mip_none = Binary(state, spv::OpIEqual, TypeBool(state),
	                            Extract(sampler_control, 26, 2), ConstantU32(state, 0));
	return Select(state, TypeU32(state), mip_none, ConstantU32(state, 0), mip);
}

uint32_t GatherMip(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                   const IR::Inst& address, const ImageSampleLayout& layout, uint32_t mip_count) {
	return GatherMipLevel(ctx, inst, mem, address, layout,
	                      ConstantF32Value(ctx.state, static_cast<float>(mip_count - 1u)),
	                      ConstantU32(ctx.state, mip_count - 1u));
}

uint32_t CubeAxis(EmitterState& state, uint32_t value) {
	return Binary(state, spv::OpFSub, TypeF32(state), value, ConstantF32(state, 0x3f800000u));
}

uint32_t CubeLayer(EmitterState& state, uint32_t value) {
	const auto guest = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), guest, value);
	const auto padding = Binary(
	    state, spv::OpShiftLeftLogical, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), guest, ConstantU32(state, 3)),
	    ConstantU32(state, 1));
	const auto host   = Binary(state, spv::OpISub, TypeU32(state), guest, padding);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), result, host);
	return result;
}

// BindlessAs2D: a 1D coordinate (or derivative) as the 2D one of a height-1 view.
uint32_t Bindless2DF32(EmitterState& state, uint32_t x, float y) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), result, x,
	                          ConstantF32Value(state, y));
	return result;
}

uint32_t Bindless2DU32(EmitterState& state, uint32_t x) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2), result, x,
	                          ConstantU32(state, 0));
	return result;
}

uint32_t CoordF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  uint32_t first, uint32_t components, bool cube = false) {
	auto x = AddressF32(ctx, mem, address, first);
	if (components == 1u) return x;
	auto y = mem.image_address_components > first + 1u ? AddressF32(ctx, mem, address, first + 1u)
	                                                   : ZeroF32(ctx.state);
	if (cube) {
		x = CubeAxis(ctx.state, x);
		y = CubeAxis(ctx.state, y);
	}
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		auto z = mem.image_address_components > first + 2u
		             ? AddressF32(ctx, mem, address, first + 2u)
		             : ZeroF32(ctx.state);
		if (cube) z = CubeLayer(ctx.state, z);
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 3),
		                              result, x, y, z);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 2),
		                              result, x, y);
	}
	return result;
}

// v431 (ImageResource::dynamic_cube): the coordinates both ways, picked by the draw's cube bit.
uint32_t CoordF32ForImage(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                          uint32_t first, uint32_t components, uint32_t resource, bool cube) {
	if (!ImageResourceAt(ctx.state, resource).dynamic_cube) {
		return CoordF32(ctx, mem, address, first, components, cube);
	}
	const auto plain = CoordF32(ctx, mem, address, first, components, false);
	const auto faces = CoordF32(ctx, mem, address, first, components, true);
	return SelectVector(ctx.state, components == 1u ? TypeF32(ctx.state) : TypeF32Vector(ctx.state, components),
	                    components, DynamicCubeIs(ctx.state, resource), faces, plain);
}

uint32_t CoordU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  ImageDimension dimension) {
	const auto components = ImageDimensionInfoFor(dimension).coordinate_components;
	const auto x          = AddressU32(ctx, mem, address, 0);
	if (components == 1u) return x;
	const auto y      = mem.image_address_components > 1u ? AddressU32(ctx, mem, address, 1)
	                                                      : ConstantU32(ctx.state, 0);
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		const auto z = mem.image_address_components > 2u ? AddressU32(ctx, mem, address, 2)
		                                                 : ConstantU32(ctx.state, 0);
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 3),
		                              result, x, y, z);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 2),
		                              result, x, y);
	}
	return result;
}

uint32_t LodU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                ImageDimension dimension) {
	const auto component = ImageDimensionInfoFor(dimension).coordinate_components;
	return mem.image_has_mip && mem.image_address_components > component
	           ? AddressU32(ctx, mem, address, component)
	           : ConstantU32(ctx.state, 0);
}

uint32_t FloatBits(ValueEmitContext& ctx, uint32_t value) {
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentBits(ValueEmitContext& ctx, uint32_t value,
                              Prospero::TextureNumericClass numeric_class) {
	if (numeric_class == Prospero::TextureNumericClass::Uint) {
		return value;
	}
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentZero(EmitterState& state, Prospero::TextureNumericClass numeric_class) {
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float: return ZeroF32(state);
		case Prospero::TextureNumericClass::Uint: return ConstantU32(state, 0);
		case Prospero::TextureNumericClass::Sint: return ConstantI32(state, 0);
		case Prospero::TextureNumericClass::Unsupported: break;
	}
	EXIT("invalid sampled image numeric class");
}

uint32_t ResultVector(ValueEmitContext& ctx, uint32_t value,
                      Prospero::TextureNumericClass numeric_class, bool dref,
                      const IR::MemoryInfo& mem, bool gather = false) {
	auto value_class = numeric_class;
	if (dref) {
		value_class = Prospero::TextureNumericClass::Float;
	}
	const bool integer = value_class == Prospero::TextureNumericClass::Uint ||
	                     value_class == Prospero::TextureNumericClass::Sint;
	if (mem.data_bits == 16u) {
		uint32_t   packed[4] = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
		                        ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
		const auto scalar    = [&](uint32_t index) {
			if (dref) return value;
			const auto component = gather ? index : DmaskComponent(mem.dmask, index);
			const auto result    = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract,
			                              ImageScalarType(ctx.state, value_class), result, value,
			                              component);
			return result;
		};
		for (uint32_t word = 0; word < mem.data_dwords; word++) {
			const auto low_index  = word * 2u;
			const auto high_index = low_index + 1u;
			const auto low        = scalar(low_index);
			const auto high       = high_index < mem.component_count
			                            ? scalar(high_index)
			                            : SampledComponentZero(ctx.state, value_class);
			if (integer) {
				const auto low_bits  = SampledComponentBits(ctx, low, value_class);
				const auto high_bits = SampledComponentBits(ctx, high, value_class);
				const auto mask      = ConstantU32(ctx.state, 0xffffu);
				packed[word] =
				    Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
				           Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), low_bits, mask),
				           Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
				                  Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
				                         high_bits, mask),
				                  ConstantU32(ctx.state, 16u)));
			} else {
				const auto pair = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
				                              TypeF32Vector(ctx.state, 2), pair, low, high);
				packed[word] = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpExtInst, TypeU32(ctx.state), packed[word],
				                              GlslStd450(ctx.state), GLSLstd450PackHalf2x16, pair);
			}
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4),
		                              result, packed[0], packed[1], packed[2], packed[3]);
		return result;
	}
	uint32_t component[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		if (dref) {
			component[index] = index == 0u ? FloatBits(ctx, value) : ConstantU32(ctx.state, 0);
			continue;
		}
		const auto scalar = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(
		    spv::OpCompositeExtract, ImageScalarType(ctx.state, value_class), scalar, value, index);
		component[index] = SampledComponentBits(ctx, scalar, value_class);
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              component[0], component[1], component[2], component[3]);
	return result;
}

static uint32_t QueryDimensionsOf(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                  const IR::Inst& address, uint32_t image);

uint32_t QueryDimensions(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                         const IR::Inst& address) {
	const auto float_result = QueryDimensionsOf(ctx, mem, address, LoadImageDescriptor(ctx.state, mem.resource));
	if (!ImageResourceAt(ctx.state, mem.resource).dynamic_class) {
		return float_result;
	}
	// v429: the T# is in one of the two bindings, a 1x1 null image in the other.
	const auto uint_result =
	    QueryDimensionsOf(ctx, mem, address, LoadDynamicClassUintDescriptor(ctx.state, mem.resource));
	return SelectU32x4(ctx.state, DynamicClassIsUint(ctx.state, mem.resource), uint_result, float_result);
}

static uint32_t QueryDimensionsOf(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                  const IR::Inst& address, uint32_t image) {
	const bool  as_2d     = BindlessAs2D(ImageResourceAt(ctx.state, mem.resource));
	const auto  dimension = as_2d ? ImageDimension::Dim2D : ImageResourceAt(ctx.state, mem.resource).dimension;
	const auto& info      = ImageDimensionInfoFor(dimension);
	const auto  size      = ctx.state.builder.AllocateId();
	// A storage image (a bindless one read under POPS) has no levels to query: its view is the
	// one level it stores to.
	const bool  storage = ImageResourceAt(ctx.state, mem.resource).resource_class ==
	                      IR::ImageResourceClass::Storage;
	if (info.multisampled != 0u || storage) {
		ctx.state.builder.AddFunction(spv::OpImageQuerySize,
		                              ImageViewSizeType(ctx.state, dimension), size, image);
	} else {
		ctx.state.builder.AddFunction(spv::OpImageQuerySizeLod,
		                              ImageViewSizeType(ctx.state, dimension), size, image,
		                              AddressU32(ctx, mem, address, 0));
	}
	// A 1D image reports its width only.
	const auto components = as_2d ? 1u : info.coordinate_components;
	const bool dynamic_layer = ImageResourceAt(ctx.state, mem.resource).dynamic_layer;
	uint32_t   result[4]  = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
	                         ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
	if (info.coordinate_components == 1u) {
		result[0] = size;
	} else {
		for (uint32_t index = 0; index < components; index++) {
			result[index] = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
			                              result[index], size, index);
		}
	}
	if (dynamic_layer && components >= 3u) {
		// v430: a 2D T# reports no layers, as the specialized 2D module did (component 2 is 0).
		result[2] = Select(ctx.state, TypeU32(ctx.state), DynamicLayerIs2D(ctx.state, mem.resource),
		                   ConstantU32(ctx.state, 0u), result[2]);
	}
	if (storage) {
		result[3] = ConstantU32(ctx.state, 1);
	} else if (info.multisampled == 0u) {
		result[3] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(ctx.state), result[3],
		                              image);
	}
	const auto vector = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), vector,
	                              result[0], result[1], result[2], result[3]);
	return vector;
}

uint32_t PackedOffset(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                      const ImageSampleLayout& layout, ImageDimension dimension) {
	const auto components = ImageDimensionInfoFor(dimension).spatial_components;
	const auto zero       = ConstantI32(ctx.state, 0);
	if (layout.offset == NoImageComponent || mem.image_address_components <= layout.offset) {
		if (components == 1u) return zero;
		const auto result = ctx.state.builder.AllocateId();
		if (components == 3u) {
			ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 3),
			                              result, zero, zero, zero);
		} else {
			ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 2),
			                              result, zero, zero);
		}
		return result;
	}
	const auto packed    = Unary(ctx.state, spv::OpBitcast, TypeI32(ctx.state),
	                             AddressU32(ctx, mem, address, layout.offset));
	uint32_t   values[3] = {zero, zero, zero};
	for (uint32_t index = 0; index < components; index++) {
		values[index] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(ctx.state), values[index],
		                              packed, ConstantU32(ctx.state, index * 8u),
		                              ConstantU32(ctx.state, 6));
	}
	if (components == 1u) return values[0];
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 3),
		                              result, values[0], values[1], values[2]);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 2),
		                              result, values[0], values[1]);
	}
	return result;
}

// IMAGE_SAMPLE*_O (RDNA 2 ISA 8.2.5): one dword of six-bit signed texel offsets, x in bits 5:0,
// y in 13:8 and z in 21:16, added to the texel coordinates of the level that is read. Vulkan
// accepts only constant offsets on sample instructions (ConstOffset), within
// [minTexelOffset, maxTexelOffset]; every implementation supports -8..7. Other offsets move the
// normalized coordinate by offset / level size instead (ApplyDynamicSampleOffset).
struct SampleOffset {
	uint32_t const_offset = 0;     // ConstOffset operand id; 0 when there is none
	uint32_t packed       = 0;     // the packed offsets when they need ApplyDynamicSampleOffset
	bool     dynamic      = false;
};

int32_t SignExtend6(uint32_t value) {
	return static_cast<int32_t>((value & 0x3fu) ^ 0x20u) - 0x20;
}

// The condition masking every use of an image result: each use is a component extract whose
// every use is SelectU32(condition, component, old), i.e. the destination write under EXEC.
// Lanes where it is false discard the result, so an operand written under the same condition
// (Select(condition, value, old)) may be read as `value`.
std::optional<IR::Value> ResultExecGuard(const IR::Inst& inst) {
	std::optional<IR::Value> guard;
	if (!inst.HasUses()) {
		return std::nullopt;
	}
	for (const auto& use: inst.Uses()) {
		const auto* extract = use.user;
		if (extract == nullptr || extract->GetOpcode() != IR::ValueOpcode::CompositeExtractU32x4 ||
		    !extract->HasUses()) {
			return std::nullopt;
		}
		for (const auto& extract_use: extract->Uses()) {
			const auto* select = extract_use.user;
			if (select == nullptr || select->GetOpcode() != IR::ValueOpcode::SelectU32 ||
			    extract_use.operand != 1u) {
				return std::nullopt;
			}
			const auto condition = select->Arg(0).Resolve();
			if (guard.has_value() && !(*guard == condition)) {
				return std::nullopt;
			}
			guard = condition;
		}
	}
	return guard;
}

SampleOffset SampleOffsetFor(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const IR::Inst& address, const ImageSampleLayout& layout,
                             ImageDimension dimension, bool cube, bool bindless) {
	SampleOffset result;
	if (layout.offset == NoImageComponent || mem.image_address_components <= layout.offset) {
		return result;
	}
	if (cube) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			std::fputs("Warning: IMAGE_SAMPLE*_O offsets on a cube map are ignored.\n", stderr);
		}
		return result;
	}
	auto&      state      = ctx.state;
	const auto components = ImageDimensionInfoFor(dimension).spatial_components;
	const auto packed_index =
	    Decoder::ImageAddressComponentLayout(mem.image_sample_flags, layout.offset).bit_offset / 32u;
	auto value =
	    packed_index < address.NumArgs() ? address.Arg(packed_index).Resolve() : IR::Value(0u);
	// Offsets are literals in guest code; under a non-uniform EXEC the register holds
	// Select(exec, literal, old), which the lanes that keep the result see as the literal.
	if (!value.IsImmediate()) {
		if (const auto guard = ResultExecGuard(inst); guard.has_value()) {
			while (!value.IsImmediate()) {
				const auto* select = value.TryInstruction();
				if (select == nullptr || select->GetOpcode() != IR::ValueOpcode::SelectU32 ||
				    !(select->Arg(0).Resolve() == *guard)) {
					break;
				}
				value = select->Arg(1).Resolve();
			}
		}
	}
	if (value.IsImmediate() &&
	    (value.GetType() == IR::Type::U32 || value.GetType() == IR::Type::F32)) {
		const auto bits = value.GetType() == IR::Type::U32 ? value.U32()
		                                                   : std::bit_cast<uint32_t>(value.F32Value());
		int32_t offsets[3] = {};
		bool    in_range   = true;
		bool    any        = false;
		for (uint32_t index = 0; index < components; index++) {
			offsets[index] = SignExtend6(bits >> (index * 8u));
			in_range       = in_range && offsets[index] >= -8 && offsets[index] <= 7;
			any            = any || offsets[index] != 0;
		}
		if (!any) {
			return result;
		}
		if (in_range) {
			uint32_t ids[3] = {};
			for (uint32_t index = 0; index < components; index++) {
				ids[index] = ConstantI32(state, offsets[index]);
			}
			if (components == 1u) {
				result.const_offset = ids[0];
			} else if (components == 2u) {
				result.const_offset = state.builder.Constant(
				    spv::OpConstantComposite, TypeI32Vector(state, 2), ids[0], ids[1]);
			} else {
				result.const_offset = state.builder.Constant(
				    spv::OpConstantComposite, TypeI32Vector(state, 3), ids[0], ids[1], ids[2]);
			}
			return result;
		}
	}
	if (bindless) {
		// The dynamic path queries the image's size through its descriptor, which a bindless
		// image does not have in this form.
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			std::fputs("Warning: non-constant IMAGE_SAMPLE*_O offsets on a bindless image are "
			           "ignored.\n",
			           stderr);
		}
		return result;
	}
	result.dynamic = true;
	result.packed  = AddressU32(ctx, mem, address, layout.offset);
	return result;
}

// Moves `coord` by the packed texel offsets divided by the size of the level `level_f32`
// (relative to the view's base, clamped to the view's levels). Exact for single-level reads up
// to float rounding; with linear mip filtering the coarser level moves by half the offset.
uint32_t ApplyDynamicSampleOffset(EmitterState& state, uint32_t resource, uint32_t coord,
                                  ImageDimension dimension, uint32_t packed, uint32_t level_f32,
                                  uint32_t array_index = 0u) {
	static std::atomic_flag warned = ATOMIC_FLAG_INIT;
	if (!warned.test_and_set(std::memory_order_relaxed)) {
		std::fputs("Warning: IMAGE_SAMPLE*_O offsets that are not constant or lie outside -8..7 "
		           "move the coordinate by offset / level size.\n",
		           stderr);
	}
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto& info       = ImageDimensionInfoFor(dimension);
	const auto  components = info.spatial_components;
	const auto  image      = LoadImageDescriptor(state, resource, 0u, array_index);
	const auto  levels     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(state), levels, image);
	const auto floored = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), floored, GlslStd450(state),
	                          GLSLstd450FMax, level_f32, ConstantF32Value(state, 0.0f));
	const auto level_float = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), level_float, GlslStd450(state),
	                          GLSLstd450Floor, floored);
	const auto level_u32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), level_u32, level_float);
	const auto last  = Binary(state, spv::OpISub, TypeU32(state), levels, ConstantU32(state, 1));
	const auto level = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), level, GlslStd450(state),
	                          GLSLstd450UMin, level_u32, last);
	const auto size = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size,
	                          image, level);
	const auto packed_i32 = Unary(state, spv::OpBitcast, TypeI32(state), packed);
	uint32_t   moved[4]   = {};
	for (uint32_t index = 0; index < info.coordinate_components; index++) {
		auto component = coord;
		if (info.coordinate_components > 1u) {
			component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, coord,
			                          index);
		}
		if (index < components) {
			const auto offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), offset, packed_i32,
			                          ConstantU32(state, index * 8u), ConstantU32(state, 6));
			const auto offset_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), offset_f32, offset);
			auto extent = size;
			if (info.coordinate_components > 1u) {
				extent = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), extent, size,
				                          index);
			}
			const auto extent_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), extent_f32, extent);
			component = Binary(state, spv::OpFAdd, TypeF32(state), component,
			                   Binary(state, spv::OpFDiv, TypeF32(state), offset_f32, extent_f32));
		}
		moved[index] = component;
	}
	if (info.coordinate_components == 1u) {
		return moved[0];
	}
	const auto result = state.builder.AllocateId();
	if (info.coordinate_components == 2u) {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), result,
		                          moved[0], moved[1]);
	} else {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 3), result,
		                          moved[0], moved[1], moved[2]);
	}
	return result;
}

uint32_t HorizontalOffsets(EmitterState& state, ImageDimension dimension) {
	const auto components   = ImageDimensionInfoFor(dimension).spatial_components;
	const auto count        = ConstantU32(state, 4);
	const auto element_type = components == 1u ? TypeI32(state) : TypeI32Vector(state, 2);
	const auto array_type   = state.builder.Type(spv::OpTypeArray, element_type, count);
	uint32_t   offsets[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		const auto x = ConstantI32(state, static_cast<int32_t>(index) - 1);
		if (components == 1u) {
			offsets[index] = x;
		} else {
			offsets[index] = state.builder.Constant(
			    spv::OpConstantComposite, TypeI32Vector(state, 2), x, ConstantI32(state, 0));
		}
	}
	return state.builder.Constant(spv::OpConstantComposite, array_type, offsets[0], offsets[1],
	                              offsets[2], offsets[3]);
}

uint32_t InverseSwizzle(uint32_t swizzle, uint32_t component) {
	for (uint32_t source = 0; source < 4u; source++) {
		if (((swizzle >> (source * 3u)) & 7u) == 4u + component) return source;
	}
	return UINT32_MAX;
}

Format::BufferFormatInfo ImageConversionFormat(const EmitterState&   state,
                                               const IR::MemoryInfo& mem) {
	const auto format = ImageResourceAt(state, mem.resource).conversion_format;
	if (format == Prospero::BufferFormat::kInvalid) return {};
	const auto info = Format::GetFormatInfo(format);
	EXIT_IF(Prospero::RemapTextureFormat(format) == format || info.component_count == 0u ||
	        info.component_count > 4u);
	EXIT_IF(info.type != Format::ComponentType::Uscaled &&
	        (info.type != Format::ComponentType::Uint || !info.packed_bitfield ||
	         info.byte_size != sizeof(uint32_t)));
	return info;
}

uint32_t ImageGatherSource(const EmitterState& state, const IR::MemoryInfo& mem) {
	const auto component = ImageGatherComponent(mem.dmask);
	const auto info      = ImageConversionFormat(state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return component;
	if (info.packed_bitfield) return 0u;
	const auto selector =
	    (ImageResourceAt(state, mem.resource).shader_swizzle >> (component * 3u)) & 7u;
	return selector >= 4u ? (selector - 4u) % info.component_count : 0u;
}

uint32_t UnpackImageTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t texel) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return texel;

	const auto numeric_class = ImageResourceAt(ctx.state, mem.resource).numeric_class;
	const auto scalar_type   = ImageScalarType(ctx.state, numeric_class);
	uint32_t   packed        = 0;
	if (info.packed_bitfield) {
		packed = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, packed, texel, 0u);
	}
	uint32_t components[4] {};
	for (uint32_t component = 0; component < info.component_count; component++) {
		components[component] = ctx.state.builder.AllocateId();
		if (info.packed_bitfield) {
			ctx.state.builder.AddFunction(spv::OpBitFieldUExtract, scalar_type,
			                              components[component], packed,
			                              ConstantU32(ctx.state, info.component_bit_offset[component]),
			                              ConstantU32(ctx.state, info.component_bits[component]));
		} else {
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type,
			                              components[component], texel, component);
			// UNorm backing preserves the guest's filtering; scaling precedes swizzle constants.
			components[component] = Binary(ctx.state, spv::OpFMul, scalar_type, components[component],
			                               ConstantF32Value(ctx.state, 255.0f));
		}
	}
	for (uint32_t component = info.component_count; component < 4u; component++) {
		components[component] = components[component % info.component_count];
	}

	const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
	uint32_t   selected[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		const auto selector = (swizzle >> (component * 3u)) & 7u;
		if (selector == 1u) {
			selected[component] = info.packed_bitfield ? ConstantU32(ctx.state, 1u)
			                                          : ConstantF32Value(ctx.state, 1.0f);
		} else if (selector >= 4u) {
			selected[component] = components[selector - 4u];
		} else {
			selected[component] = info.packed_bitfield ? ConstantU32(ctx.state, 0u)
			                                          : ZeroF32(ctx.state);
		}
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              ImageVectorType(ctx.state, numeric_class, 4), result, selected[0],
	                              selected[1], selected[2], selected[3]);
	return result;
}

uint32_t UnpackImageGather(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t gathered) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return gathered;

	const auto component = ImageGatherComponent(mem.dmask);
	const auto selector =
	    (ImageResourceAt(ctx.state, mem.resource).shader_swizzle >> (component * 3u)) & 7u;
	const auto numeric_class = ImageResourceAt(ctx.state, mem.resource).numeric_class;
	const auto vector_type   = ImageVectorType(ctx.state, numeric_class, 4);
	if (selector < 4u) {
		uint32_t value;
		switch (info.type) {
			case Format::ComponentType::Uscaled:
				value = ConstantF32Value(ctx.state, selector == 1u ? 1.0f : 0.0f);
				break;
			default: value = ConstantU32(ctx.state, selector == 1u ? 1u : 0u); break;
		}
		return ctx.state.builder.Constant(spv::OpConstantComposite, vector_type, value, value,
		                                  value, value);
	}
	if (!info.packed_bitfield) {
		const auto scale  = ConstantF32Value(ctx.state, 255.0f);
		const auto scales = ctx.state.builder.Constant(spv::OpConstantComposite, vector_type,
		                                                scale, scale, scale, scale);
		return Binary(ctx.state, spv::OpFMul, vector_type, gathered, scales);
	}

	uint32_t values[4] {};
	for (uint32_t lane = 0; lane < 4u; lane++) {
		const auto packed = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), packed, gathered,
		                              lane);
		values[lane]        = ctx.state.builder.AllocateId();
		const auto physical = (selector - 4u) % info.component_count;
		ctx.state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(ctx.state), values[lane],
		                              packed,
		                              ConstantU32(ctx.state, info.component_bit_offset[physical]),
		                              ConstantU32(ctx.state, info.component_bits[physical]));
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              values[0], values[1], values[2], values[3]);
	return result;
}

uint32_t EmitOneDimensionalGatherLz(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                    uint32_t coord, Prospero::TextureNumericClass numeric_class) {
	auto& state = ctx.state;
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto image = LoadImageDescriptor(state, mem.resource);
	const auto width = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, TypeU32(state), width, image,
	                          ConstantU32(state, 0));
	const auto width_f32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), width_f32, width);
	const auto left = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), left, GlslStd450(state),
	                          GLSLstd450Floor,
	                          Binary(state, spv::OpFSub, TypeF32(state),
	                                 Binary(state, spv::OpFMul, TypeF32(state), coord, width_f32),
	                                 ConstantF32(state, 0x3f000000u)));

	const auto sampled = MakeSampledImage(state, mem.resource,
	                                     LoadSamplerDescriptor(state, mem.sampler));
	const auto vector_type = ImageVectorType(state, numeric_class, 4);
	const auto scalar_type = ImageScalarType(state, numeric_class);
	const auto component = ImageGatherSource(state, mem);
	uint32_t values[2] {};
	for (uint32_t index = 0; index < 2u; index++) {
		const auto sample_coord =
		    Binary(state, spv::OpFDiv, TypeF32(state),
		           Binary(state, spv::OpFAdd, TypeF32(state), left,
		                  ConstantF32(state, index == 0u ? 0x3f000000u : 0x3fc00000u)),
		           width_f32);
		const auto texel = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpImageSampleExplicitLod, vector_type, texel, sampled,
		                          sample_coord, spv::ImageOperandsLodMask, ZeroF32(state));
		values[index] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, values[index], texel,
		                          component);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, result, values[0], values[1],
	                          values[1], values[0]);
	return result;
}

// IMAGE_GATHER4*_L of a bindless image. Its one view holds every level of the T#, and
// OpImageGather always reads the view's base level (Vulkan has no LOD operand for it), so the
// instruction's LOD was lost and the gather read level 0. Gathers the four texels of the selected
// level instead by point reads at their centres (as EmitOneDimensionalGatherLz does): the sampler
// still applies the address modes and the border, and a centre read is exact under bilinear
// filtering. Texel order is a gather's: (i0,j1), (i1,j1), (i1,j0), (i0,j0). _O offsets move
// i0/j0 by whole texels of that level. The explicit LOD carries the level minus the sampler's LOD
// bias, which Vulkan adds to an explicit LOD too (multiples of 1/256: the sum is exact).
float SamplerLodBiasLimit() {
	return GetHostImageFeatures().max_sampler_lod_bias;
}

uint32_t EmitBindlessGatherLod(ValueEmitContext& ctx, const IR::Inst& inst,
                               const IR::MemoryInfo& mem, const IR::Inst& address,
                               const ImageSampleLayout& layout, uint32_t coord,
                               ImageDimension dimension,
                               Prospero::TextureNumericClass numeric_class) {
	auto& state = ctx.state;
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const bool array  = dimension == ImageDimension::Dim2DArray;
	const auto image  = LoadImageDescriptor(state, mem.resource);
	const auto levels = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(state), levels, image);
	const auto last_u32 = Binary(state, spv::OpISub, TypeU32(state), levels, ConstantU32(state, 1));
	const auto last_f32 = Unary(state, spv::OpConvertUToF, TypeF32(state), last_u32);
	const auto level    = GatherMipLevel(ctx, inst, mem, address, layout, last_f32, last_u32);
	const auto size     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size,
	                          image, level);
	const auto Component = [&](uint32_t vector, uint32_t type, uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, type, value, vector, index);
		return value;
	};
	const auto half = ConstantF32Value(state, 0.5f);
	uint32_t   extent[2] {};
	uint32_t   first[2] {};
	const auto packed = layout.offset != NoImageComponent &&
	                            mem.image_address_components > layout.offset
	                        ? Unary(state, spv::OpBitcast, TypeI32(state),
	                                AddressU32(ctx, mem, address, layout.offset))
	                        : 0u;
	for (uint32_t axis = 0; axis < 2u; axis++) {
		extent[axis] = Unary(state, spv::OpConvertUToF, TypeF32(state),
		                     Component(size, TypeU32(state), axis));
		const auto scaled = Binary(state, spv::OpFSub, TypeF32(state),
		                           Binary(state, spv::OpFMul, TypeF32(state),
		                                  Component(coord, TypeF32(state), axis), extent[axis]),
		                           half);
		first[axis] = EmitGlsl<GLSLstd450Floor, IR::Type::F32>(state, scaled);
		if (packed != 0u) {
			const auto offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), offset, packed,
			                          ConstantU32(state, axis * 8u), ConstantU32(state, 6));
			first[axis] = Binary(state, spv::OpFAdd, TypeF32(state), first[axis],
			                     Unary(state, spv::OpConvertSToF, TypeF32(state), offset));
		}
	}
	const auto layer = array ? Component(coord, TypeF32(state), 2) : 0u;
	// S# dword 2 bits 13:0: LOD_BIAS, signed 6.8; the sampler cache clamps it to the device limit.
	const auto bias_bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), bias_bits,
	                          Unary(state, spv::OpBitcast, TypeI32(state), ctx.Arg(inst, 6)),
	                          ConstantU32(state, 0), ConstantU32(state, 14));
	auto bias = Binary(state, spv::OpFMul, TypeF32(state),
	                   Unary(state, spv::OpConvertSToF, TypeF32(state), bias_bits),
	                   ConstantF32Value(state, 1.0f / 256.0f));
	bias = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, bias,
	                                                 ConstantF32Value(state, -SamplerLodBiasLimit()),
	                                                 ConstantF32Value(state, SamplerLodBiasLimit()));
	const auto lod = Binary(state, spv::OpFSub, TypeF32(state),
	                        Unary(state, spv::OpConvertUToF, TypeF32(state), level), bias);
	const auto sampled     = MakeSampledImage(state, mem.resource,
	                                          LoadSamplerDescriptor(state, mem.sampler));
	const auto vector_type = ImageVectorType(state, numeric_class, 4);
	const auto scalar_type = ImageScalarType(state, numeric_class);
	const auto component   = ImageGatherSource(state, mem);
	constexpr float Steps[4][2] = {{0.5f, 1.5f}, {1.5f, 1.5f}, {1.5f, 0.5f}, {0.5f, 0.5f}};
	uint32_t values[4] {};
	for (uint32_t texel = 0; texel < 4u; texel++) {
		uint32_t centre[2] {};
		for (uint32_t axis = 0; axis < 2u; axis++) {
			centre[axis] = Binary(state, spv::OpFDiv, TypeF32(state),
			                      Binary(state, spv::OpFAdd, TypeF32(state), first[axis],
			                             ConstantF32Value(state, Steps[texel][axis])),
			                      extent[axis]);
		}
		const auto point = state.builder.AllocateId();
		if (array) {
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 3), point,
			                          centre[0], centre[1], layer);
		} else {
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), point,
			                          centre[0], centre[1]);
		}
		const auto sample = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpImageSampleExplicitLod, vector_type, sample, sampled,
		                          point, spv::ImageOperandsLodMask, lod);
		values[texel] = Component(sample, scalar_type, component);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, result, values[0], values[1],
	                          values[2], values[3]);
	return result;
}

uint32_t PackImageTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t texel) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return texel;
	EXIT_IF(info.type != Format::ComponentType::Uint || !info.packed_bitfield);

	auto packed = ConstantU32(ctx.state, 0u);
	for (uint32_t component = 0; component < info.component_count; component++) {
		const auto value = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, texel,
		                              component);
		const auto maximum =
		    ConstantU32(ctx.state, info.component_bits[component] == 32u
		                               ? UINT32_MAX
		                               : (1u << info.component_bits[component]) - 1u);
		const auto within =
		    Binary(ctx.state, spv::OpULessThan, TypeBool(ctx.state), value, maximum);
		const auto clamped = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), clamped, within, value,
		                              maximum);
		const auto shifted =
		    info.component_bit_offset[component] == 0u
		        ? clamped
		        : Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), clamped,
		                 ConstantU32(ctx.state, info.component_bit_offset[component]));
		packed = Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), packed, shifted);
	}
	const auto zero   = ConstantU32(ctx.state, 0u);
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              packed, zero, zero, zero);
	return result;
}

uint32_t StoreTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t data, bool integer) {
	const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
	uint32_t   values[4] {};
	const auto dmask = mem.dmask != 0u ? mem.dmask : 1u;
	for (uint32_t component = 0; component < 4u; component++) {
		const auto source = InverseSwizzle(swizzle, component);
		uint32_t   raw    = ConstantU32(ctx.state, 0);
		if (source < 4u && ((dmask >> source) & 1u) != 0u) {
			const auto packed_index = DmaskComponentIndex(dmask, source);
			raw                     = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), raw, data,
			                              mem.data_bits == 16u ? packed_index / 2u : packed_index);
			if (mem.data_bits == 16u) {
				if ((packed_index & 1u) != 0u) {
					raw = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), raw,
					             ConstantU32(ctx.state, 16u));
				}
				raw = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), raw,
				             ConstantU32(ctx.state, 0xffffu));
			}
		}
		values[component] = integer ? raw
		                    : mem.data_bits == 16u
		                        ? EmitF16BitsToF32(ctx.state, raw)
		                        : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), raw);
	}
	const auto texel = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              integer ? TypeU32Vector(ctx.state, 4)
	                                      : TypeF32Vector(ctx.state, 4),
	                              texel, values[0], values[1], values[2], values[3]);
	return PackImageTexel(ctx, mem, texel);
}

} // namespace

// KYTY_BINDLESS_FEEDBACK_SPARSE (default on; =0 stores every lookup's word as before): a bindless
// lookup stores its feedback word only when it is nonzero (a pending key or a diagnostic value).
// Every pixel of a full-screen pass stored a zero to the same few words. Off whenever something
// reads the zeros: the usage probe (KYTY_BINDLESS_EVICT, on unless "0"; its marks are cleared by
// the draws that sample a key) and the texture dump (KYTY_DUMP_TEXTURES_TRIGGER, which arms the
// probe); also off with KYTY_KEY_CAPTURE.
static bool BindlessFeedbackSparse() {
	static const bool enabled = [] {
		if (const char* value = std::getenv("KYTY_BINDLESS_FEEDBACK_SPARSE");
		    value != nullptr && value[0] == '0') {
			return false;
		}
		const char* evict = std::getenv("KYTY_BINDLESS_EVICT");
		if (evict == nullptr || evict[0] != '0') {
			return false;
		}
		return std::getenv("KYTY_DUMP_TEXTURES_TRIGGER") == nullptr &&
		       std::getenv("KYTY_KEY_CAPTURE") == nullptr;
	}();
	return enabled;
}

// The bindless slot of the image handle's key: translation[region + key] when the key is inside
// the heap's entry count (both from the flattened SRT, patched by the host), otherwise and while
// the texture is pending, slot 0 (the placeholder). A pending key is flagged in the feedback
// buffer for the host to resolve; every lane that reads an entry sees the same translation, so
// the store never races with a different value.
static uint32_t BindlessSlot(ValueEmitContext& ctx, const IR::ImageResource& image,
                             IR::Value image_arg) {
	auto&       state  = ctx.state;
	const auto* handle = image_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0 ||
	        state.bindless_translation_variable == 0);
	const auto key  = ctx.Def(handle->Arg(0));
	const auto Load = [&](uint32_t variable, uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, variable, ConstantU32(state, 0), index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto base     = Load(state.flattened_srt_variable,
	                           ConstantU32(state, image.indirect_mapping_offset));
	const auto count    = Load(state.flattened_srt_variable,
	                           ConstantU32(state, image.indirect_mapping_offset + 1u));
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), key, count);
	const auto entry    = Binary(state, spv::OpIAdd, TypeU32(state), base, key);
	const auto index    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), index, in_range, entry,
	                          ConstantU32(state, 0u));
	const auto raw     = Load(state.bindless_translation_variable, index);
	const auto pending = Binary(state, spv::OpIEqual, TypeBool(state), raw,
	                            ConstantU32(state, IR::BindlessPending));
	// v127: a value that names no slot of the array (or carries flag bits where none belong)
	// samples slot 0 and is reported, instead of indexing past the descriptor array.
	const auto max_slots = ConstantU32(state, IR::BindlessMaxSlots);
	auto       valid     = Binary(state, spv::OpULessThan, TypeBool(state), raw, max_slots);
	if (BindlessVolumeMayBeArray(image)) {
		const auto flags    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
		                             ConstantU32(state, IR::BindlessSlotFlagMask));
		const auto is_array = Binary(state, spv::OpIEqual, TypeBool(state), flags,
		                             ConstantU32(state, IR::BindlessSlotArrayFlag));
		const auto low      = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
		                             ConstantU32(state, ~IR::BindlessSlotFlagMask));
		const auto low_ok   = Binary(state, spv::OpULessThan, TypeBool(state), low, max_slots);
		const auto array_ok = Binary(state, spv::OpLogicalAnd, TypeBool(state), is_array, low_ok);
		valid = Binary(state, spv::OpLogicalOr, TypeBool(state), valid, array_ok);
	}
	const auto bad = Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                        Unary(state, spv::OpLogicalNot, TypeBool(state), pending),
	                        Unary(state, spv::OpLogicalNot, TypeBool(state), valid));
	if (state.bindless_feedback_variable != 0) {
		const auto pending_flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), pending_flag, pending,
		                          ConstantU32(state, 1u), ConstantU32(state, 0u));
		// Diagnostics: a key outside its heap leaves itself (with the top bit set) in
		// feedback word 0, which belongs to no heap.
		const auto marked_key = Binary(state, spv::OpBitwiseOr, TypeU32(state), key,
		                               ConstantU32(state, 0x80000000u));
		const auto key_flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), key_flag, in_range, pending_flag,
		                          marked_key);
		// v127: the same store, redirected to the out-of-range word with the raw value when the
		// translation names no slot (no extra store on the common path).
		const auto flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), flag, bad, raw, key_flag);
		const auto word = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), word, bad,
		                          ConstantU32(state, IR::BindlessOutOfRangeSlotWord), index);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
		                          word);
		if (BindlessFeedbackSparse()) {
			// Only a pending key (1) or a diagnostic value is stored: the host clears each request
			// it takes (BindlessTable::TakeRequests), so the zero of a resolved key carries nothing.
			auto store_it =
			    Binary(state, spv::OpINotEqual, TypeBool(state), flag, ConstantU32(state, 0u));
			if (IR::BindlessRuntimeProbeEnabled()) {
				// v389: also while a usage probe runs (BindlessProbeActiveWord), so a sample clears
				// the probe's mark on its key.
				const auto probe_pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
				                          probe_pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
				                          ConstantU32(state, IR::BindlessProbeActiveWord));
				const auto probe = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, TypeU32(state), probe, probe_pointer);
				store_it = Binary(state, spv::OpLogicalOr, TypeBool(state), store_it,
				                  Binary(state, spv::OpINotEqual, TypeBool(state), probe, ConstantU32(state, 0u)));
			}
			EmitIfCondition(state, store_it,
			                [&]() { state.builder.AddFunction(spv::OpStore, pointer, flag); });
		} else {
			state.builder.AddFunction(spv::OpStore, pointer, flag);
		}
		// KYTY_BINDLESS_OOB_INFO=1 (diagnostics): which shader looked a key up outside its heap,
		// and the count it compared it with (BindlessOobInfoWord). In-range lookups store to a
		// scratch word after them.
		static const bool oob_info = [] {
			const char* value = std::getenv("KYTY_BINDLESS_OOB_INFO");
			return value != nullptr && value[0] == '1';
		}();
		if (oob_info) {
			const uint32_t values[3] = {
			    ConstantU32(state, static_cast<uint32_t>(state.program.shader_hash)), count, key};
			for (uint32_t i = 0; i < 3u; i++) {
				const auto info_word = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), info_word, in_range,
				                          ConstantU32(state, IR::BindlessOobInfoWord + 3u),
				                          ConstantU32(state, IR::BindlessOobInfoWord + i));
				const auto info_pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
				                          info_pointer, state.bindless_feedback_variable,
				                          ConstantU32(state, 0), info_word);
				state.builder.AddFunction(spv::OpStore, info_pointer, values[i]);
			}
		}
	}
	// KYTY_SLOT_CAPTURE=1 (v376, Crash B, diagnostics only): every bindless lookup leaves the
	// translation value it read in feedback word SlotCaptureBase + its translation index (raw |
	// 0x80000000, or 0xfffffffe for a pending key), so a device loss can compare what the GPU read
	// with what the host wrote (BindlessTable::CompareSlotCapture).
	static const bool slot_capture = [] {
		const char* value = std::getenv("KYTY_SLOT_CAPTURE");
		return value != nullptr && value[0] == '1';
	}();
	if (slot_capture && state.bindless_feedback_variable != 0) {
		const auto marked = Binary(state, spv::OpBitwiseOr, TypeU32(state), raw, ConstantU32(state, 0x80000000u));
		const auto value  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, pending, ConstantU32(state, 0xfffffffeu),
		                          marked);
		const auto fits = Binary(state, spv::OpULessThan, TypeBool(state), index,
		                         ConstantU32(state, IR::BindlessSlotCaptureEntries));
		const auto slot_word = Binary(state, spv::OpIAdd, TypeU32(state), index,
		                              ConstantU32(state, IR::BindlessSlotCaptureBase));
		const auto capture_word = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), capture_word, fits, slot_word,
		                          ConstantU32(state, IR::BindlessSlotCaptureBase + IR::BindlessSlotCaptureEntries));
		const auto capture_pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), capture_pointer,
		                          state.bindless_feedback_variable, ConstantU32(state, 0), capture_word);
		state.builder.AddFunction(spv::OpStore, capture_pointer, value);
	}
	state.bindless_translation_raw = raw;
	// Pending keys sample slot 2 (the blue placeholder) while their texture is loaded; a value
	// that names no slot samples slot 0 (v127).
	const auto safe = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), safe, bad, ConstantU32(state, 0u),
	                          raw);
	const auto slot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, pending, ConstantU32(state, 2u),
	                          safe);
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

// The bindless sampler array slot of the sampler handle's key: region + key while the key is
// inside the heap's entry count (both from the flattened SRT, patched by the host), otherwise
// slot 0, the default sampler.
static uint32_t BindlessSamplerSlot(ValueEmitContext& ctx, const IR::SamplerResource& sampler,
                                    IR::Value sampler_arg) {
	auto&       state  = ctx.state;
	const auto* handle = sampler_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0);
	const auto key  = ctx.Def(handle->Arg(0));
	const auto Load = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          ConstantU32(state, index));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto base     = Load(sampler.bindless_mapping_offset);
	const auto count    = Load(sampler.bindless_mapping_offset + 1u);
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), key, count);
	const auto entry    = Binary(state, spv::OpIAdd, TypeU32(state), base, key);
	const auto slot     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, in_range, entry,
	                          ConstantU32(state, 0u));
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

static void EmitImageOperation(ValueEmitContext& ctx, const IR::Inst& inst);
static void EmitDynamic3DOr2D(ValueEmitContext& ctx, const IR::Inst& inst);
static void EmitBindlessVolumeOrArray(ValueEmitContext& ctx, const IR::Inst& inst);

// KYTY_CANARY_DETECT=1 (v378, Crash B, diagnostics only; pair with KYTY_FREE_CANARY): a bindless
// float sample that returns the free canary's colour (0.0625, 0.9375, 0.3125, 1; 2- and 1-channel forms) records the shader, the
// translation value and the descriptor slot in the feedback buffer (BindlessCanary*), so a read of
// a destroyed image's memory during its hold is seen even when it never reaches the screen.
static void EmitCanaryDetect(EmitterState& state, uint32_t sample,
                             Prospero::TextureNumericClass numeric_class, uint32_t resource) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CANARY_DETECT");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled || state.bindless_feedback_variable == 0 || state.bindless_slot == 0) {
		return;
	}
	const auto bool_type = TypeBool(state);
	const bool integer   = numeric_class == Prospero::TextureNumericClass::Uint ||
	                     numeric_class == Prospero::TextureNumericClass::Sint;
	const auto Component = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract,
		                          integer ? ImageScalarType(state, numeric_class) : TypeF32(state), value, sample,
		                          index);
		if (numeric_class == Prospero::TextureNumericClass::Sint) {
			return Unary(state, spv::OpBitcast, TypeU32(state), value);
		}
		return value;
	};
	// The canary (vma.cpp FreeCanary) as a 3/4-channel image (0.0625, 0.9375, 0.3125, 1), a 2-channel
	// one (0.0625, 0.9375, 0, 1) and a 1-channel or depth one (0.0625, 0, 0, 1; tighter).
	const auto Near = [&](uint32_t value, float target, float within) {
		const auto diff = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeF32(state), diff, GlslStd450(state), GLSLstd450FAbs,
		                          Binary(state, spv::OpFSub, TypeF32(state), value, ConstantF32Value(state, target)));
		return Binary(state, spv::OpFOrdLessThan, bool_type, diff, ConstantF32Value(state, within));
	};
	const auto And = [&](uint32_t a, uint32_t b) { return Binary(state, spv::OpLogicalAnd, bool_type, a, b); };
	const auto Or  = [&](uint32_t a, uint32_t b) { return Binary(state, spv::OpLogicalOr, bool_type, a, b); };
	const auto r = Component(0), g = Component(1), b = Component(2), a = Component(3);
	uint32_t hit = 0, signature = 0;
	if (integer) {
		// v381: integer images hold the canary's float bit patterns (vma.cpp FreeCanary).
		const auto Is = [&](uint32_t value, uint32_t bits) {
			return Binary(state, spv::OpIEqual, bool_type, value, ConstantU32(state, bits));
		};
		const auto first = And(Is(r, 0x3d800000u), Is(g, 0x3f700000u));
		const auto four  = And(first, Is(b, 0x3ea00000u));
		const auto two   = And(first, Is(b, 0u));
		const auto one   = And(And(Is(r, 0x3d800000u), Is(g, 0u)), Is(b, 0u));
		hit       = Or(Or(four, two), one);
		signature = Select(state, TypeU32(state), four, ConstantU32(state, 1u),
		                   Select(state, TypeU32(state), two, ConstantU32(state, 2u), ConstantU32(state, 3u)));
	} else {
	const auto opaque = Binary(state, spv::OpFOrdGreaterThan, bool_type, a, ConstantF32Value(state, 0.996f));
	const auto b_zero = Binary(state, spv::OpFOrdEqual, bool_type, b, ConstantF32Value(state, 0.0f));
	const auto g_zero = Binary(state, spv::OpFOrdEqual, bool_type, g, ConstantF32Value(state, 0.0f));
	const auto rg     = And(Near(r, 0.0625f, 0.004f), Near(g, 0.9375f, 0.004f));
	const auto four   = And(And(rg, Near(b, 0.3125f, 0.004f)), opaque);
	const auto two    = And(And(rg, b_zero), opaque);
	const auto one    = And(And(And(Near(r, 0.0625f, 0.0005f), g_zero), b_zero), opaque);
	hit = Or(Or(four, two), one);
	// Which signature: 1 (3/4 channels), 2 (2 channels), 3 (1 channel or depth).
	signature = Select(state, TypeU32(state), four, ConstantU32(state, 1u),
	                   Select(state, TypeU32(state), two, ConstantU32(state, 2u), ConstantU32(state, 3u)));
	}
	const auto slot = state.bindless_slot;
	const auto raw  = state.bindless_translation_raw != 0 ? state.bindless_translation_raw : slot;
	EmitIfCondition(state, hit, [&]() {
		const auto Pointer = [&](uint32_t word) {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
			                          word);
			return pointer;
		};
		const auto hash_low  = ConstantU32(state, static_cast<uint32_t>(state.program.shader_hash));
		const auto hash_high = ConstantU32(state, static_cast<uint32_t>(state.program.shader_hash >> 32u));
		const auto count     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), count,
		                          Pointer(ConstantU32(state, IR::BindlessCanaryWord)),
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0),
		                          ConstantU32(state, 1));
		state.builder.AddFunction(spv::OpStore, Pointer(ConstantU32(state, IR::BindlessCanaryWord + 1u)), hash_low);
		state.builder.AddFunction(spv::OpStore, Pointer(ConstantU32(state, IR::BindlessCanaryWord + 2u)), hash_high);
		state.builder.AddFunction(spv::OpStore, Pointer(ConstantU32(state, IR::BindlessCanaryWord + 3u)), raw);
		const auto per_signature = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), per_signature,
		                          Pointer(Binary(state, spv::OpIAdd, TypeU32(state), signature,
		                                         ConstantU32(state, IR::BindlessCanaryWord + 3u))),
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0),
		                          ConstantU32(state, 1));
		const auto masked = Binary(state, spv::OpBitwiseAnd, TypeU32(state), slot,
		                           ConstantU32(state, IR::BindlessCanarySlotMask));
		state.builder.AddFunction(
		    spv::OpStore,
		    Pointer(Binary(state, spv::OpIAdd, TypeU32(state), masked, ConstantU32(state, IR::BindlessCanarySlots))),
		    hash_low);
		// v382: the size of the image the descriptor names (width | height << 16), to compare with
		// the view the host holds in that slot: a stale descriptor shows another image's size.
		{
			const auto& resource_info = ImageResourceAt(state, resource);
			const auto  dimension     = BindlessAs2D(resource_info) ? ImageDimension::Dim2D : resource_info.dimension;
			const auto& info          = ImageDimensionInfoFor(dimension);
			uint32_t    packed        = ConstantU32(state, 0xffffffffu);
			if (info.multisampled == 0u && resource_info.resource_class != IR::ImageResourceClass::Storage &&
			    info.coordinate_components >= 1u && info.coordinate_components <= 3u) {
				state.builder.RequireCapability(spv::CapabilityImageQuery);
				const auto image = LoadImageDescriptor(state, resource);
				const auto size  = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size, image,
				                          ConstantU32(state, 0));
				uint32_t width = size, height = ConstantU32(state, 1);
				if (info.coordinate_components > 1u) {
					width  = state.builder.AllocateId();
					height = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), width, size, 0u);
					state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), height, size, 1u);
				}
				packed = Binary(state, spv::OpBitwiseOr, TypeU32(state),
				                Binary(state, spv::OpBitwiseAnd, TypeU32(state), width, ConstantU32(state, 0xffffu)),
				                Binary(state, spv::OpShiftLeftLogical, TypeU32(state), height, ConstantU32(state, 16u)));
			}
			state.builder.AddFunction(
			    spv::OpStore,
			    Pointer(Binary(state, spv::OpIAdd, TypeU32(state), masked, ConstantU32(state, IR::BindlessCanarySize))),
			    packed);
			state.builder.AddFunction(spv::OpStore, Pointer(ConstantU32(state, IR::BindlessCanaryWord + 7u)), packed);
		}
		state.builder.AddFunction(
		    spv::OpStore,
		    Pointer(Binary(state, spv::OpIAdd, TypeU32(state), masked, ConstantU32(state, IR::BindlessCanaryRaw))),
		    Binary(state, spv::OpBitwiseOr, TypeU32(state), raw,
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), signature, ConstantU32(state, 24u))));
	});
}

void EmitImage(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto op         = inst.GetOpcode();
	const auto image_info = IR::ImageOpcodeInfoOf(op);
	auto&       state     = ctx.state;
	const auto& mem       = ctx.Memory(inst);
	const auto  image_arg = inst.Arg(0);
	ctx.ResourceIndex(image_arg, IR::ValueOpcode::GetImageResource);
	const auto& image   = ImageResourceAt(state, mem.resource);
	state.bindless_slot = image.bindless ? BindlessSlot(ctx, image, image_arg) : 0u;
	state.bindless_sampler_slot = 0u;
	if (image_info.needs_sampler && inst.NumArgs() > 1u &&
	    mem.sampler < state.program.info.samplers.size() &&
	    state.program.info.samplers[mem.sampler].bindless) {
		state.bindless_sampler_slot =
		    BindlessSamplerSlot(ctx, state.program.info.samplers[mem.sampler], inst.Arg(1));
	}
	// Diagnostics: KYTY_KEY_CAPTURE=<hash> marks the keys this shader samples (BindlessCapture*).
	static const uint64_t capture_hash = [] {
		const char* value = std::getenv("KYTY_KEY_CAPTURE");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	if (capture_hash != 0 && state.program.shader_hash == capture_hash &&
	    state.bindless_feedback_variable != 0) {
		const auto Mark = [&](IR::Value handle_arg, uint32_t first, uint32_t value) {
			const auto* handle = handle_arg.ResolveInstruction();
			if (handle == nullptr || handle->NumArgs() == 0u) {
				return;
			}
			const auto key    = ctx.Def(handle->Arg(0));
			const auto low    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), key,
			                           ConstantU32(state, 0xffffu));
			const auto index  = Binary(state, spv::OpIAdd, TypeU32(state), low, ConstantU32(state, first));
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
			                          index);
			state.builder.AddFunction(spv::OpStore, pointer, value);
		};
		// Compile time: every image instruction, its sampler and what the sampler handle's key is.
		{
			std::string what = "none";
			if (inst.NumArgs() > 1u) {
				if (const auto* handle = inst.Arg(1).Resolve().TryInstruction();
				    handle != nullptr && handle->NumArgs() != 0u) {
					const auto key = handle->Arg(0).Resolve();
					if (key.IsImmediate()) {
						what = key.GetType() == IR::Type::U32 ? fmt::format("immediate 0x{:x}", key.U32()) : "immediate";
					} else if (const auto* key_inst = key.TryInstruction(); key_inst != nullptr) {
						what = std::string(IR::ValueOpcodeName(key_inst->GetOpcode()));
					}
				}
			}
			const bool sampler_bindless = image_info.needs_sampler && mem.sampler < state.program.info.samplers.size() &&
			                              state.program.info.samplers[mem.sampler].bindless;
			std::fprintf(stderr,
			             "[key-capture] emit pc=0x%x op=%s image=%u bindless_image=%d sampler=%u bindless_sampler=%d slot=%u key=%s\n",
			             inst.Flags<IR::MemoryFlags>().pc, std::string(IR::ValueOpcodeName(op)).c_str(), mem.resource,
			             image.bindless ? 1 : 0, mem.sampler, sampler_bindless ? 1 : 0, state.bindless_sampler_slot,
			             what.c_str());
		}
		uint32_t sampler_key = ConstantU32(state, 0xffffu);
		if (state.bindless_sampler_slot != 0u) {

			if (const auto* handle = inst.Arg(1).ResolveInstruction();
			    handle != nullptr && handle->NumArgs() != 0u) {
				sampler_key = Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Def(handle->Arg(0)),
				                     ConstantU32(state, 0xffffu));
			}
			// The raw dword the sampler key is cut from (before its shift or mask), at the image
			// key's word, to compare with the material data in memory.
			if (image.bindless) {
				const auto* sampler_handle = inst.Arg(1).ResolveInstruction();
				const auto* key_inst =
				    sampler_handle != nullptr && sampler_handle->NumArgs() != 0u
				        ? sampler_handle->Arg(0).Resolve().TryInstruction()
				        : nullptr;
				if (key_inst != nullptr && key_inst->NumArgs() == 2u &&
				    (key_inst->GetOpcode() == IR::ValueOpcode::ShiftRightLogical32 ||
				     key_inst->GetOpcode() == IR::ValueOpcode::BitwiseAnd32)) {
					const auto raw = ctx.Def(key_inst->Arg(0));
					Mark(image_arg, IR::BindlessCaptureSamplers, raw);
				}
			}
		}
		if (image.bindless) {
			const auto value = Binary(state, spv::OpBitwiseOr, TypeU32(state), sampler_key,
			                          ConstantU32(state, 0x40000000u | ((mem.resource & 0xffu) << 16u)));
			Mark(image_arg, IR::BindlessCaptureImages, value);
		}
	}
	if (BindlessVolumeMayBeArray(image) && op != IR::ValueOpcode::ImageWrite) {
		EmitBindlessVolumeOrArray(ctx, inst);
		return;
	}
	if (image.dynamic_3d && op != IR::ValueOpcode::ImageWrite) {
		EmitDynamic3DOr2D(ctx, inst);
		return;
	}
	EmitImageOperation(ctx, inst);
}

// v433 (ImageResource::dynamic_3d): a 3D instruction's slot whose T# may be 2D. The instruction is
// emitted against the 3D binding and, with the image seen as 2D (the 2D group's binding, as the
// 2D-specialized module would emit it), against the 2D binding; the draw's bit selects. Both run:
// no divergent branch, so implicit-LOD derivatives stay defined.
static void EmitDynamic3DOr2D(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& mem   = ctx.Memory(inst);
	std::optional<uint32_t> predefined;
	if (const auto found = ctx.definitions.find(&inst); found != ctx.definitions.end()) {
		predefined = found->second;
		ctx.definitions.erase(found);
	}
	const auto Emit = [&](bool as_2d) {
		if (as_2d) {
			state.image_override_index     = mem.resource;
			state.image_override           = state.program.info.images.at(mem.resource);
			state.image_override.dimension = ImageDimension::Dim2D;
		}
		EmitImageOperation(ctx, inst);
		state.image_override_index = UINT32_MAX;
		const auto found = ctx.definitions.find(&inst);
		EXIT_IF(found == ctx.definitions.end());
		const auto value = found->second;
		ctx.definitions.erase(found);
		return value;
	};
	const auto value_2d = Emit(true);
	const auto value_3d = Emit(false);
	const auto is_2d    = DynamicImage3DIs2D(state, mem.resource);
	const auto type     = TypeId(state, inst.GetType());
	uint32_t   value    = 0;
	// OpSelect takes a condition of the result's vector size (the U32x2 pair is a structure).
	switch (inst.GetType()) {
		case IR::Type::U32x4: value = SelectU32x4(state, is_2d, value_2d, value_3d); break;
		case IR::Type::U32x3: value = SelectVector(state, type, 3u, is_2d, value_2d, value_3d); break;
		case IR::Type::U32x2: {
			std::array<uint32_t, 4> parts {};
			for (uint32_t i = 0; i < 2u; i++) {
				const auto a = state.builder.AllocateId();
				const auto b = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), a, value_2d, i);
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), b, value_3d, i);
				parts[i] = Select(state, TypeU32(state), is_2d, a, b);
			}
			value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, type, value, parts[0], parts[1]);
			break;
		}
		default: value = Select(state, type, is_2d, value_2d, value_3d); break;
	}
	if (predefined) {
		ctx.definitions.emplace(&inst, *predefined);
	}
	ctx.Define(inst, value);
}

// A bindless 3D image instruction whose key may name a 2D or 2D-array texture: the GPU addresses
// by the T# type, so such a texture is read as a 2D array at layer round(z) (Vulkan's array
// layer rule is the same). Both reads are emitted, each with the other view's placeholder slot,
// and the flag selects: no divergent branch, so implicit-LOD derivatives stay defined.
static void EmitBindlessVolumeOrArray(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& mem   = ctx.Memory(inst);
	const auto  raw   = state.bindless_translation_raw;
	const auto  slot  = state.bindless_slot;
	const auto  flags = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
	                           ConstantU32(state, IR::BindlessSlotFlagMask));
	const auto  as_array = Binary(state, spv::OpIEqual, TypeBool(state), flags,
	                              ConstantU32(state, IR::BindlessSlotArrayFlag));
	const auto  array_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
	                               ConstantU32(state, ~IR::BindlessSlotFlagMask));
	// v127: never past the array (BindlessSlot reported the value).
	const auto  array_index = Select(state, TypeU32(state),
	                                 Binary(state, spv::OpULessThan, TypeBool(state), array_low,
	                                        ConstantU32(state, IR::BindlessMaxSlots)),
	                                 array_low, ConstantU32(state, 0u));
	const auto  NonUniformSelect = [&](uint32_t condition, uint32_t yes, uint32_t no) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, condition, yes, no);
		state.builder.AddAnnotation(spv::OpDecorate, value, spv::DecorationNonUniform);
		return value;
	};
	const auto array_slot  = NonUniformSelect(as_array, array_index, ConstantU32(state, 0u));
	const auto volume_slot = NonUniformSelect(as_array, ConstantU32(state, 0u), slot);
	// An existing definition (a value used before this point) is re-pointed at the selection.
	std::optional<uint32_t> predefined;
	if (const auto found = ctx.definitions.find(&inst); found != ctx.definitions.end()) {
		predefined = found->second;
		ctx.definitions.erase(found);
	}
	const auto Emit = [&](bool array_view) {
		if (array_view) {
			state.image_override_index = mem.resource;
			state.image_override       = state.program.info.images.at(mem.resource);
			state.image_override.dimension = ImageDimension::Dim2DArray;
		}
		state.bindless_slot = array_view ? array_slot : volume_slot;
		EmitImageOperation(ctx, inst);
		state.image_override_index = UINT32_MAX;
		state.bindless_slot        = slot;
		const auto found = ctx.definitions.find(&inst);
		EXIT_IF(found == ctx.definitions.end());
		const auto value = found->second;
		ctx.definitions.erase(found);
		return value;
	};
	const auto array_value  = Emit(true);
	const auto volume_value = Emit(false);
	const auto value        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeId(state, inst.GetType()), value, as_array,
	                          array_value, volume_value);
	if (predefined) {
		ctx.definitions.emplace(&inst, *predefined);
	}
	ctx.Define(inst, value);
}

static void EmitImageOperation(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op         = inst.GetOpcode();
	const auto  image_info = IR::ImageOpcodeInfoOf(op);
	auto&       state      = ctx.state;
	const auto& mem        = ctx.Memory(inst);
	const auto& image      = ImageResourceAt(state, mem.resource);
	const auto  image_arg  = inst.Arg(0);
	const auto* address = ctx.ImageAddress(inst.Arg(image_info.needs_sampler ? 2 : 1));
	if (op == IR::ValueOpcode::ImageQueryDimensions) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		ctx.Define(inst, QueryDimensions(ctx, mem, *address));
		return;
	}
	if (op == IR::ValueOpcode::ImageQueryLod) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		const auto dimension = image.dimension;
		const auto sampled = MakeSampledImage(state, mem.resource,
		                                     LoadSamplerDescriptor(state, mem.sampler));
		const auto lod       = state.builder.AllocateId();
		auto       lod_coord = CoordF32ForImage(ctx, mem, *address, 0,
		                                        ImageDimensionInfoFor(dimension).spatial_components,
		                                        mem.resource, image.cube);
		if (BindlessAs2D(image)) lod_coord = Bindless2DF32(state, lod_coord, 0.5f);
		state.builder.AddFunction(spv::OpImageQueryLod, TypeF32Vector(state, 2), lod, sampled,
		                          lod_coord);
		uint32_t values[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0),
		                      ConstantU32(state, 0)};
		for (uint32_t index = 0; index < 2u; index++) {
			const auto component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, lod,
			                          index);
			values[index] = FloatBits(ctx, component);
		}
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result,
		                          values[0], values[1], values[2], values[3]);
		ctx.Define(inst, result);
		return;
	}
	if (op == IR::ValueOpcode::ImageRead) {
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  numeric_class  = image.numeric_class;
		const auto  condition      = ctx.Arg(inst, 2);
		ctx.Define(
		    inst,
		    EmitValueOrDefaultIfCondition(
		        state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4),
		        [&]() {
			        auto coord = CoordU32(ctx, mem, *address, dimension);
			        if (BindlessAs2D(image)) coord = Bindless2DU32(state, coord);
			        if (image.dynamic_layer) {
				        // v430: a 2D T# (one-layer view) is read at layer 0 whatever the layer operand,
				        // as the hardware ignores the slice of a 2D texture.
				        const auto layer = state.builder.AllocateId();
				        state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), layer, coord, 2u);
				        const auto kept = Select(state, TypeU32(state), DynamicLayerIs2D(state, mem.resource),
				                                 ConstantU32(state, 0u), layer);
				        const auto replaced = state.builder.AllocateId();
				        state.builder.AddFunction(spv::OpCompositeInsert, TypeU32Vector(state, 3), replaced,
				                                  kept, coord, 2u);
				        coord = replaced;
			        }
			        if (image.dynamic_class) {
				        // v429: fetch through both bindings (the T# is in one, a null image in the
				        // other) and keep the one the draw's class bit names. A texel fetch has no
				        // derivatives, so both run unconditionally.
				        const auto lod = LodU32(ctx, mem, *address, dimension);
				        const auto fetch = [&](uint32_t descriptor, Prospero::TextureNumericClass cls) {
					        const auto texel = state.builder.AllocateId();
					        state.builder.AddFunction(spv::OpImageFetch, ImageVectorType(state, cls, 4), texel,
					                                  descriptor, coord, spv::ImageOperandsLodMask, lod);
					        return ResultVector(ctx, UnpackImageTexel(ctx, mem, texel), cls, false, mem);
				        };
				        const auto as_float = fetch(LoadImageDescriptor(state, mem.resource),
				                                    Prospero::TextureNumericClass::Float);
				        const auto as_uint  = fetch(LoadDynamicClassUintDescriptor(state, mem.resource),
				                                    Prospero::TextureNumericClass::Uint);
				        return SelectU32x4(state, DynamicClassIsUint(state, mem.resource), as_uint, as_float);
			        }
			        const auto descriptor = LoadImageDescriptor(state, mem.resource);
			        const auto color      = state.builder.AllocateId();
			        if (image.resource_class == IR::ImageResourceClass::Storage) {
				        // Coherent under the pixel interlock (IndirectDescriptor::storage).
				        state.builder.RequireCapability(spv::CapabilityStorageImageReadWithoutFormat);
				        state.builder.AddFunction(spv::OpImageRead,
				                                  ImageVectorType(state, numeric_class, 4), color,
				                                  descriptor, coord);
			        } else if (dimension_info.multisampled != 0u) {
				        state.builder.AddFunction(
				            spv::OpImageFetch, ImageVectorType(state, numeric_class, 4), color,
				            descriptor, coord, spv::ImageOperandsSampleMask,
				            AddressU32(ctx, mem, *address, dimension_info.coordinate_components));
			        } else {
				        state.builder.AddFunction(spv::OpImageFetch,
				                                  ImageVectorType(state, numeric_class, 4), color,
				                                  descriptor, coord, spv::ImageOperandsLodMask,
				                                  LodU32(ctx, mem, *address, dimension));
			        }
			        return ResultVector(ctx, UnpackImageTexel(ctx, mem, color), numeric_class,
			                            false, mem);
		        }));
		return;
	}
	if (op == IR::ValueOpcode::ImageWrite) {
		const bool uint_image = image.numeric_class == Prospero::TextureNumericClass::Uint;
		const auto dimension  = image.dimension;
		EmitIfCondition(state, ctx.Arg(inst, 3), [&]() {
			const auto mip_lod =
			    ImageResourceAt(state, mem.resource).mip_mode == IR::ImageMipMode::Dynamic
			        ? LodU32(ctx, mem, *address, dimension)
			        : 0u;
			const auto coord = CoordU32(ctx, mem, *address, dimension);
			const auto texel = StoreTexel(ctx, mem, ctx.Arg(inst, 2), uint_image);
			EmitStorageImageWrite(state, mem.resource, mip_lod, coord, texel);
		});
		return;
	}
	if (op == IR::ValueOpcode::ImageSampleRaw || op == IR::ValueOpcode::ImageGatherRaw) {
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  layout         = Layout(mem);
		const auto  numeric_class  = image.numeric_class;
		const bool  dref           = HasFlag(mem, Decoder::ImageSampleFlagCompare);
		if (dref && ImageResourceAt(state, mem.resource).conversion_format !=
		                Prospero::BufferFormat::kInvalid) {
			ctx.Fail(inst, "uses depth comparison with a converted image");
			return;
		}
		if (op == IR::ValueOpcode::ImageGatherRaw) {
			const auto coord = CoordF32ForImage(ctx, mem, *address, layout.coord,
			                                    dimension_info.coordinate_components, mem.resource, image.cube);
			if (BindlessAs2D(image)) {
				ctx.Fail(inst, "gathers from a bindless 1D image");
				return;
			}
			if (dimension == ImageDimension::Dim1D) {
				if (dref || !HasFlag(mem, Decoder::ImageSampleFlagLevelZero) ||
				    HasFlag(mem, Decoder::ImageSampleFlagOffset) ||
				    HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
					ctx.Fail(inst, "has an unsupported 1D gather variant");
					return;
				}
				const auto sample = EmitOneDimensionalGatherLz(ctx, mem, coord, numeric_class);
				ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
				                              numeric_class, false, mem, true));
				return;
			}
			if (dimension == ImageDimension::Dim1DArray) {
				ctx.Fail(inst, "has an unsupported 1D-array gather");
				return;
			}
			if (image.bindless && image.mip_mode == IR::ImageMipMode::Dynamic && !dref &&
			    !image.cube && !HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal) &&
			    (dimension == ImageDimension::Dim2D || dimension == ImageDimension::Dim2DArray) &&
			    layout.lod != NoImageComponent) {
				const auto sample = EmitBindlessGatherLod(ctx, inst, mem, *address, layout, coord,
				                                          dimension, numeric_class);
				ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
				                              numeric_class, false, mem, true));
				return;
			}
			const auto result_numeric_class = dref ? Prospero::TextureNumericClass::Float
			                                       : numeric_class;
			const auto result_type = ImageVectorType(state, result_numeric_class, 4);
			uint32_t component_or_dref;
			if (dref) {
				component_or_dref = layout.dref != NoImageComponent
				                        ? AddressF32(ctx, mem, *address, layout.dref)
				                        : ZeroF32(state);
			} else {
				component_or_dref = ConstantU32(state, ImageGatherSource(state, mem));
			}
			uint32_t operand_mask = 0;
			uint32_t offset = 0;
			if (HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
				operand_mask = spv::ImageOperandsConstOffsetsMask;
				offset = HorizontalOffsets(state, dimension);
			} else if (layout.offset != NoImageComponent) {
				operand_mask = spv::ImageOperandsOffsetMask;
				offset = PackedOffset(ctx, mem, *address, layout, dimension);
			}
			const auto sampler_id = LoadSamplerDescriptor(state, mem.sampler);
			const auto EmitGather = [&](uint32_t mip) {
				const auto sampled = MakeSampledImage(state, mem.resource, sampler_id, mip);
				const auto sample = state.builder.AllocateId();
				std::vector<uint32_t> words {
				    static_cast<uint32_t>(dref ? spv::OpImageDrefGather : spv::OpImageGather), result_type,
				    sample, sampled, coord, component_or_dref};
				if (operand_mask != 0u) {
					words.push_back(operand_mask);
					words.push_back(offset);
				}
				state.builder.AddFunction(words);
				return sample;
			};
			const auto sample = image.mip_mode == IR::ImageMipMode::Dynamic && image.mip_count > 1u
			                        ? EmitIndexSwitch(
			                              state, GatherMip(ctx, inst, mem, *address, layout,
			                                               image.mip_count),
			                              image.mip_count, result_type, EmitGather)
			                        : EmitGather(0);
			ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
			                              result_numeric_class, false, mem, true));
			return;
		}
		const bool explicit_lod = HasFlag(mem, Decoder::ImageSampleFlagDerivative) ||
		                          HasFlag(mem, Decoder::ImageSampleFlagLod) ||
		                          HasFlag(mem, Decoder::ImageSampleFlagLevelZero) ||
		                          state.program.stage != ShaderType::Pixel;
		auto       opcode       = spv::OpImageSampleImplicitLod;
		if (explicit_lod) {
			opcode = dref ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleExplicitLod;
		} else if (dref) {
			opcode = spv::OpImageSampleDrefImplicitLod;
		}
		uint32_t result_type = ImageVectorType(state, numeric_class, 4);
		uint32_t dref_value  = 0;
		if (dref) {
			result_type = TypeF32(state);
			dref_value  = ZeroF32(state);
			if (layout.dref != NoImageComponent) {
				dref_value = AddressF32(ctx, mem, *address, layout.dref);
			}
		}
		// IMAGE_SAMPLE*_CL: the last address component is a minimum LOD. Implicit and gradient
		// samples take it as the MinLod operand; explicit-LOD forms (every sample outside pixel
		// shaders) raise their LOD to it.
		uint32_t lod_clamp = 0;
		if (layout.clamp != NoImageComponent && mem.image_address_components > layout.clamp) {
			lod_clamp = AddressF32(ctx, mem, *address, layout.clamp);
		}
		const bool gradient   = HasFlag(mem, Decoder::ImageSampleFlagDerivative);
		const bool min_lod_op = lod_clamp != 0u && (gradient || !explicit_lod) &&
		                        GetHostImageFeatures().min_lod;
		uint32_t              explicit_lod_value = 0;
		uint32_t              operand_mask       = 0;
		std::vector<uint32_t> operands;
		if (gradient) {
			operand_mask |= spv::ImageOperandsGradMask;
			for (const auto first: {layout.grad_x, layout.grad_y}) {
				auto grad = CoordF32(ctx, mem, *address, first, dimension_info.spatial_components);
				if (BindlessAs2D(image)) grad = Bindless2DF32(state, grad, 0.0f);
				operands.push_back(grad);
			}
		} else if (explicit_lod) {
			operand_mask |= spv::ImageOperandsLodMask;
			auto lod = ZeroF32(state);
			if (HasFlag(mem, Decoder::ImageSampleFlagLod) && layout.lod != NoImageComponent) {
				lod = AddressF32(ctx, mem, *address, layout.lod);
			}
			if (lod_clamp != 0u) {
				const auto clamped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
				                          GLSLstd450FMax, lod, lod_clamp);
				lod = clamped;
			}
			explicit_lod_value = lod;
			operands.push_back(lod);
		} else if (layout.bias != NoImageComponent) {
			operand_mask |= spv::ImageOperandsBiasMask;
			operands.push_back(AddressF32(ctx, mem, *address, layout.bias));
		}
		const auto sampler_id = LoadSamplerDescriptor(state, mem.sampler);
		const auto sample_offset = SampleOffsetFor(
		    ctx, inst, mem, *address, layout, BindlessAs2D(image) ? ImageDimension::Dim2D : dimension,
		    image.cube, image.bindless);
		size_t const_offset_index = SIZE_MAX;
		if (sample_offset.const_offset != 0u) {
			operand_mask |= spv::ImageOperandsConstOffsetMask;
			const_offset_index = operands.size();
			operands.push_back(sample_offset.const_offset);
		}
		if (min_lod_op) {
			state.builder.RequireCapability(spv::CapabilityMinLod);
			operand_mask |= spv::ImageOperandsMinLodMask;
			operands.push_back(lod_clamp);
		} else if (lod_clamp != 0u && !explicit_lod) {
			static std::atomic_flag warned = ATOMIC_FLAG_INIT;
			if (!warned.test_and_set(std::memory_order_relaxed)) {
				std::fputs("Warning: IMAGE_SAMPLE*_CL clamp ignored: the device lacks "
				           "shaderResourceMinLod.\n",
				           stderr);
			}
		}
		const auto EmitSample = [&](uint32_t resource, uint32_t array_index = 0u) -> uint32_t {
			const auto& candidate = ImageResourceAt(state, resource);
			const bool  offsets   = sample_offset.const_offset != 0u || sample_offset.dynamic;
			if (candidate.dynamic_cube && offsets) {
				// v431: offsets apply to a 2D array and are ignored on a cube: both samples run
				// (implicit derivatives stay defined) and the draw's cube bit picks one.
				const auto components = ImageDimensionInfoFor(candidate.dimension).coordinate_components;
				const auto with_offsets = [&]() {
					auto plain_coord = CoordF32(ctx, mem, *address, layout.coord, components, false);
					if (sample_offset.dynamic) {
						uint32_t level = explicit_lod_value != 0u ? explicit_lod_value : ZeroF32(state);
						if (explicit_lod_value == 0u && !HasFlag(mem, Decoder::ImageSampleFlagLevelZero) &&
						    state.program.stage == ShaderType::Pixel) {
							state.builder.RequireCapability(spv::CapabilityImageQuery);
							const auto query = state.builder.AllocateId();
							state.builder.AddFunction(
							    spv::OpImageQueryLod, TypeF32Vector(state, 2), query,
							    MakeSampledImage(state, resource, sampler_id, 0u, array_index),
							    CoordF32(ctx, mem, *address, layout.coord,
							             ImageDimensionInfoFor(candidate.dimension).spatial_components, false));
							level = state.builder.AllocateId();
							state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), level, query, 0u);
						}
						plain_coord = ApplyDynamicSampleOffset(state, resource, plain_coord, candidate.dimension,
						                                       sample_offset.packed, level, array_index);
					}
					return plain_coord;
				}();
				const auto emit = [&](uint32_t coord, bool keep_offset) {
					const auto            sampled = MakeSampledImage(state, resource, sampler_id, 0u, array_index);
					const auto            sample  = state.builder.AllocateId();
					std::vector<uint32_t> words {static_cast<uint32_t>(opcode), result_type, sample, sampled, coord};
					if (dref) {
						words.push_back(dref_value);
					}
					auto mask = operand_mask;
					auto list = operands;
					if (!keep_offset && const_offset_index != SIZE_MAX) {
						mask &= ~static_cast<uint32_t>(spv::ImageOperandsConstOffsetMask);
						list.erase(list.begin() + static_cast<std::ptrdiff_t>(const_offset_index));
					}
					if (mask != 0u) {
						words.push_back(mask);
						words.insert(words.end(), list.begin(), list.end());
					}
					state.builder.AddFunction(words);
					return sample;
				};
				const auto plain_sample = emit(with_offsets, true);
				const auto cube_sample  = emit(CoordF32(ctx, mem, *address, layout.coord, components, true), false);
				return SelectVector(state, result_type, dref ? 1u : 4u, DynamicCubeIs(state, resource),
				                    cube_sample, plain_sample);
			}
			auto coord = CoordF32ForImage(ctx, mem, *address, layout.coord,
			                              ImageDimensionInfoFor(candidate.dimension).coordinate_components,
			                              resource, candidate.cube);
			if (BindlessAs2D(candidate)) coord = Bindless2DF32(state, coord, 0.5f);
			if (sample_offset.dynamic) {
				uint32_t level = ZeroF32(state);
				if (explicit_lod_value != 0u) {
					level = explicit_lod_value;
				} else if (!HasFlag(mem, Decoder::ImageSampleFlagLevelZero) &&
				           state.program.stage == ShaderType::Pixel) {
					// The level an implicit or gradient sample reads, relative to the view base.
					state.builder.RequireCapability(spv::CapabilityImageQuery);
					const auto query = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpImageQueryLod, TypeF32Vector(state, 2), query,
					    MakeSampledImage(state, resource, sampler_id, 0u, array_index),
					    CoordF32(ctx, mem, *address, layout.coord,
					             ImageDimensionInfoFor(candidate.dimension).spatial_components,
					             candidate.cube));
					level = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), level, query,
					                          0u);
				}
				coord = ApplyDynamicSampleOffset(state, resource, coord, candidate.dimension,
				                                 sample_offset.packed, level, array_index);
			}
			const auto sampled = MakeSampledImage(state, resource, sampler_id, 0u, array_index);
			const auto            sample  = state.builder.AllocateId();
			std::vector<uint32_t> sample_operands {result_type, sample, sampled, coord};
			if (dref) {
				sample_operands.push_back(dref_value);
			}
			if (operand_mask != 0u) {
				sample_operands.push_back(operand_mask);
				sample_operands.insert(sample_operands.end(), operands.begin(), operands.end());
			}
			state.builder.AddFunction(opcode, sample_operands);
			return sample;
		};
		if (image.indirect_root != mem.resource || image.bindless) {
			const auto sample = EmitSample(mem.resource);
			if (image.bindless && !dref) {
				EmitCanaryDetect(state, sample, numeric_class, mem.resource);
			}
			auto       result = sample;
			if (!dref) {
				result = UnpackImageTexel(ctx, mem, sample);
			}
			ctx.Define(inst, ResultVector(ctx, result, numeric_class, dref, mem));
			return;
		}
		const auto* handle = image_arg.ResolveInstruction();
		const auto* source = image.source < state.program.descriptor_sources.size()
		                         ? &state.program.descriptor_sources[image.source]
		                         : nullptr;
		if (handle == nullptr || source == nullptr || !source->indirect_descriptor.has_value() ||
		    handle->NumArgs() == 0u) {
			ctx.Fail(inst, "has invalid indirect image key provenance");
			return;
		}
		const auto key = ctx.Def(handle->Arg(0));
		if (state.flattened_srt_variable == 0 || image.indirect_search_iterations == 0u ||
		    image.indirect_resources.size() < 2u) {
			ctx.Fail(inst, "has no indirect image runtime mapping");
			return;
		}
		const auto selected = EmitIndirectResourceIndex(
		    state, key, image.indirect_mapping_offset, image.indirect_search_iterations, 0u);
		const auto kind = *IR::DescriptorBindingForImage(image);
		const auto& binding = *IR::FindBinding(state.program.bindings, kind);
		uint32_t first_child = 0;
		bool homogeneous = true;
		for (uint32_t ordinal = 0; ordinal < image.indirect_resources.size(); ++ordinal) {
			const auto resource = image.indirect_resources[ordinal];
			const auto& candidate = ImageResourceAt(state, resource);
			if (candidate.dimension != image.dimension || candidate.cube != image.cube ||
			    IR::DescriptorBindingForImage(candidate) != kind ||
			    candidate.mip_count != 1u) {
				homogeneous = false;
				break;
			}
			if (ordinal == 1u) first_child = ResourceForDescriptor(state, kind, resource);
			const auto child_slot = first_child + ordinal - 1u;
			if (ordinal > 1u && (child_slot >= binding.resources.size() ||
			                     binding.resources[child_slot] != resource)) {
				homogeneous = false;
				break;
			}
		}
		if (homogeneous) {
			// v301: the selection comes from the per-draw mapping in the flattened SRT buffer; a
			// value past the candidate list would index past the pushed descriptor array into
			// undefined descriptor memory (old headers of freed images: Crash B). Such a value
			// samples the root instead, and bindless programs leave the shader and the value in
			// feedback words IndirectOutOfRangeWord(+1) for the host ([indirect-oob]).
			const auto candidates = static_cast<uint32_t>(image.indirect_resources.size());
			const auto in_list = Binary(state, spv::OpULessThan, TypeBool(state), selected,
			                            ConstantU32(state, candidates));
			if (state.bindless_feedback_variable != 0) {
				const auto out = Unary(state, spv::OpLogicalNot, TypeBool(state), in_list);
				EmitIfCondition(state, out, [&]() {
					const auto store_word = [&](uint32_t word, uint32_t value) {
						const auto pointer = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
						                          pointer, state.bindless_feedback_variable,
						                          ConstantU32(state, 0), ConstantU32(state, word));
						state.builder.AddFunction(spv::OpStore, pointer, value);
					};
					store_word(IR::IndirectOutOfRangeWord,
					           ConstantU32(state, static_cast<uint32_t>(state.program.shader_hash)));
					store_word(IR::IndirectOutOfRangeWord + 1u,
					           Binary(state, spv::OpBitwiseOr, TypeU32(state), selected,
					                  ConstantU32(state, candidates << 24u)));
				});
			}
			const auto safe = Select(state, TypeU32(state), in_list, selected, ConstantU32(state, 0u));
			const auto child_index = Binary(state, spv::OpIAdd, TypeU32(state), safe,
			                                ConstantU32(state, first_child - 1u));
			const auto is_root = Binary(state, spv::OpIEqual, TypeBool(state), safe,
			                            ConstantU32(state, 0u));
			const auto array_index = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), array_index, is_root,
			                          ConstantU32(state, ResourceForDescriptor(state, kind, mem.resource)),
			                          child_index);
			const auto sample = EmitSample(mem.resource, array_index);
			const auto result = dref ? sample : UnpackImageTexel(ctx, mem, sample);
			ctx.Define(inst, ResultVector(ctx, result, numeric_class, dref, mem));
			return;
		}
		const auto            default_label = state.builder.AllocateId();
		const auto            merge_label   = state.builder.AllocateId();
		std::vector<uint32_t> labels(image.indirect_resources.size() - 1u);
		std::vector<uint32_t> switch_words {spv::OpSwitch, selected, default_label};
		for (uint32_t candidate = 1; candidate < image.indirect_resources.size(); candidate++) {
			labels[candidate - 1u] = state.builder.AllocateId();
			switch_words.push_back(candidate);
			switch_words.push_back(labels[candidate - 1u]);
		}
		state.builder.AddFunction(spv::OpSelectionMerge, merge_label,
		                          spv::SelectionControlMaskNone);
		state.builder.AddFunction(switch_words);
		std::vector<uint32_t> phi_words {spv::OpPhi, result_type, state.builder.AllocateId()};
		EmitLabel(state, default_label);
		phi_words.push_back(EmitSample(image.indirect_resources[0]));
		phi_words.push_back(default_label);
		state.builder.AddFunction(spv::OpBranch, merge_label);
		for (uint32_t candidate = 1; candidate < image.indirect_resources.size(); candidate++) {
			EmitLabel(state, labels[candidate - 1u]);
			phi_words.push_back(EmitSample(image.indirect_resources[candidate]));
			phi_words.push_back(labels[candidate - 1u]);
			state.builder.AddFunction(spv::OpBranch, merge_label);
		}
		EmitLabel(state, merge_label);
		state.builder.AddFunction(phi_words);
		auto result = phi_words[2];
		if (!dref) {
			result = UnpackImageTexel(ctx, mem, result);
		}
		ctx.Define(inst, ResultVector(ctx, result, numeric_class, dref, mem));
		return;
	}
	if (image_info.access == IR::ImageAccess::Atomic) {
		const auto dimension = image.dimension;
		const auto result_type = TypeId(state, inst.GetType());
		const auto zero = image.atomic64 ? ConstantU64(state, 0) : ConstantU32(state, 0);
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		                     state, ctx.Arg(inst, inst.NumArgs() - 1), result_type, zero, [&]() {
			           const auto pointer      = state.builder.AllocateId();
			           const auto pointer_type = state.builder.Type(
			               spv::OpTypePointer, spv::StorageClassImage,
			               image.atomic64 ? TypeScalarU64(state) : TypeU32(state));
			           state.builder.AddFunction(spv::OpImageTexelPointer, pointer_type, pointer,
			                                     ImageDescriptorPointer(state, mem.resource),
			                                     CoordU32(ctx, mem, *address, dimension),
			                                     ConstantU32(state, 0));
			           if (op == IR::ValueOpcode::ImageAtomicFMin32 ||
			               op == IR::ValueOpcode::ImageAtomicFMax32) {
				           return AtomicUpdate(state, pointer, IR::ResourceKind::Image,
				                               [&](uint32_t old) {
					                               return EmitFloatAtomicReplacement(
					                                   state, old, ctx.Arg(inst, 2),
					                                   op == IR::ValueOpcode::ImageAtomicFMax32);
				                               });
			           }
			           const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			           EmitAtomicMemoryBarrier(state, IR::ResourceKind::Image);
			           return old;
		           }));
		return;
	}
	ctx.Fail(inst, "has no image SPIR-V emitter");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
