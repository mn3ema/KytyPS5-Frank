#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <atomic>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

bool UserDataDwordIndex(const EmitterState& state, IR::ScalarReg reg, uint32_t& dword_index) {
	const auto register_index = IR::RegIndex(reg);
	const auto& registers = state.program.bindings.user_data_registers;
	const auto  found     = std::lower_bound(registers.begin(), registers.end(), register_index);
	if (found == registers.end() || *found != register_index) {
		return false;
	}
	dword_index = static_cast<uint32_t>(found - registers.begin());
	return true;
}

uint32_t EmitBuiltinU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	if (kind == IR::StageInputKind::DispatchThreadCount) {
		const auto dword = state.program.bindings.dispatch_thread_dword;
		if (!DispatchDimensionsIndirect(state)) {
			return EmitShaderDataDwordLoad(state, dword + component);
		}
		// The counts stay where the guest's indirect arguments are; the shader data holds
		// their device address.
		const auto u64     = TypeScalarU64(state);
		const auto low     = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword));
		const auto high    = Unary(state, spv::OpUConvert, u64,
		                           EmitShaderDataDwordLoad(state, dword + 1u));
		const auto base    = Binary(state, spv::OpBitwiseOr, u64, low,
		                            Binary(state, spv::OpShiftLeftLogical, u64, high,
		                                   state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
		const auto address = Binary(state, spv::OpIAdd, u64, base,
		                            state.builder.Constant(spv::OpConstant, u64,
		                                                   component * 4u, 0u));
		const auto load = [&]() {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          address);
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
			                          spv::MemoryAccessAlignedMask, 4u);
			return value;
		};
		if (state.program.rt_role != IR::RtRole::None && IR::RtAddressTrap()) {
			// v443l: a callable split's launch extents, behind the address trap (spirvEmitterRt.cpp).
			const auto usable = EmitRtAddressUsable(state, base);
			if (IR::RtAddressTrapReport()) {
				EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), usable),
				                [&]() { EmitRtAddressTrapped(state, base, 1u); });
			}
			return EmitValueOrZeroIfCondition(state, usable, load);
		}
		return load();
	}
	if (kind == IR::StageInputKind::LocalInvocationIndex) {
		return EmitLocalInvocationIndex(state);
	}
	if (state.lane_count == 2 && (kind == IR::StageInputKind::LocalInvocationId ||
	                              kind == IR::StageInputKind::GlobalInvocationId)) {
		const auto* cs      = ShaderWorkgroupInput(state.program.stage, state.input_info);
		uint32_t    divisor = 1;
		for (uint32_t axis = 0; axis < component; axis++) {
			divisor *= std::max(cs->threads_num[axis], 1u);
		}
		const auto size    = std::max(cs->threads_num[component], 1u);
		const auto divided = EmitBinaryU32(state, spv::OpUDiv, EmitLocalInvocationIndex(state),
		                                   ConstantU32(state, divisor));
		const auto local   = EmitBinaryU32(state, spv::OpUMod, divided, ConstantU32(state, size));
		if (kind == IR::StageInputKind::LocalInvocationId) {
			return local;
		}
		const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, component);
		return EmitAddU32(state, local,
		                  EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, size)));
	}
	// v443i: a callable split's module takes its workgroup built-ins from the launch id
	// (EmitRtWorkgroupInput). Since v442 it declares no compute inputs (DefineInputs), and without
	// this every lane of every wave read thread 0 of group 0 from the "not declared" case below.
	if (state.program.rt_role != IR::RtRole::None &&
	    (kind == IR::StageInputKind::WorkgroupId || kind == IR::StageInputKind::NumWorkgroups ||
	     kind == IR::StageInputKind::LocalInvocationId || kind == IR::StageInputKind::GlobalInvocationId)) {
		return EmitInputComponentU32(state, kind, component);
	}
	const bool centroid = kind == IR::StageInputKind::BaryCoordSmoothCentroid;
	const auto variable = InputVariableForKind(
	    state, centroid ? IR::StageInputKind::BaryCoordSmooth : kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	if (kind == IR::StageInputKind::FrontFacing) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), value, variable);
		// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS: 1 front / 0 back when set, +1.0f / -1.0f bits when clear.
		const bool all_bits = state.input_info.pixel != nullptr &&
		                      state.input_info.pixel->ps_front_face_all_bits;
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), bits, value,
		                          ConstantU32(state, all_bits ? 1u : 0x3f800000u),
		                          ConstantU32(state, all_bits ? 0u : 0xbf800000u));
		return bits;
	}
	if (kind == IR::StageInputKind::VertexIndex || kind == IR::StageInputKind::InstanceIndex ||
	    kind == IR::StageInputKind::InvocationId || kind == IR::StageInputKind::PrimitiveId ||
	    kind == IR::StageInputKind::Layer || kind == IR::StageInputKind::SampleId) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), value, variable);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (kind == IR::StageInputKind::FragCoord || kind == IR::StageInputKind::TessCoord) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, variable, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (centroid || kind == IR::StageInputKind::BaryCoordSmooth ||
	    kind == IR::StageInputKind::BaryCoordNoPerspective) {
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		if (centroid) {
			const auto coordinates = state.builder.AllocateId();
			state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
			                          GlslStd450(state), GLSLstd450InterpolateAtCentroid, variable);
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value,
			                          coordinates, component + 1u);
		} else {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, variable, ConstantU32(state, component + 1u));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		}
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	return EmitInputComponentU32(state, kind, component);
}

uint32_t EmitDppWriteCondition(ValueEmitContext& ctx, const IR::DppMoveFlags& flags,
                               uint32_t exec) {
	auto&      state      = ctx.state;
	const auto lane       = EmitSubgroupLocalInvocationId(state);
	const auto bank_shift = state.builder.AllocateId();
	const auto row_shift  = state.builder.AllocateId();
	const auto bank       = state.builder.AllocateId();
	const auto row        = state.builder.AllocateId();
	const auto bank_bit   = state.builder.AllocateId();
	const auto row_bit    = state.builder.AllocateId();
	const auto bank_hit   = state.builder.AllocateId();
	const auto row_hit    = state.builder.AllocateId();
	const auto bank_ok    = state.builder.AllocateId();
	const auto row_ok     = state.builder.AllocateId();
	const auto masks_ok   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), bank_shift, lane,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), row_shift, lane,
	                          ConstantU32(state, 4));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), bank, bank_shift,
	                          ConstantU32(state, 3));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, row_shift,
	                          ConstantU32(state, 3));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), bank_bit,
	                          ConstantU32(state, 1), bank);
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), row_bit,
	                          ConstantU32(state, 1), row);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), bank_hit,
	                          ConstantU32(state, flags.bank_mask), bank_bit);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row_hit,
	                          ConstantU32(state, flags.row_mask), row_bit);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), bank_ok, bank_hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), row_ok, row_hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), masks_ok, bank_ok, row_ok);
	uint32_t writable = masks_ok;
	if (!flags.bound_control) {
		const auto target  = EmitDppTargetLane(state, flags);
		const auto bounded = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), bounded, writable,
		                          target.valid);
		writable = bounded;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), result, exec, writable);
	return result;
}

uint32_t EmitAttribute(EmitterState& state, uint32_t attr, uint32_t chan) {
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantU32(state, 0);
	}
	if (state.program.stage == ShaderType::Vertex || state.program.stage == ShaderType::Local) {
		return EmitVertexParameterComponentU32(state, *input, chan & 3u);
	}
	const auto load_per_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};
	if (input->per_vertex) {
		const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
		                                  ? IR::StageInputKind::BaryCoordNoPerspective
		                                  : IR::StageInputKind::BaryCoordSmooth;
		const auto barycentric      = InputVariableForKind(state, barycentric_kind);
		uint32_t   sum              = 0;
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			const auto pointer = state.builder.AllocateId();
			const auto weight  = state.builder.AllocateId();
			const auto product = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, barycentric, ConstantU32(state, vertex));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), weight, pointer);
			state.builder.AddFunction(spv::OpFMul, TypeF32(state), product, load_per_vertex(vertex),
			                          weight);
			if (vertex == 0u) {
				sum = product;
			} else {
				const auto next = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFAdd, TypeF32(state), next, sum, product);
				sum = next;
			}
		}
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, sum);
		return bits;
	}
	const auto vector    = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto bits      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), vector, input->variable_id);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, vector,
	                          chan & 3u);
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, component);
	return bits;
}

uint32_t EmitInterpolationParameter(ValueEmitContext& ctx, uint32_t attr, uint32_t chan,
                                    uint32_t mode) {
	auto&       state = ctx.state;
	const auto* input = InputBindingForParameter(state, attr);
	if (!input->per_vertex) {
		return EmitAttribute(ctx.state, attr, chan);
	}
	const auto load_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};

	const auto selected_vertex = (mode + 1u) % 3u;
	uint32_t   value           = load_vertex(selected_vertex);
	if (!PixelParameterIsCustom(state, attr) && mode < 2u) {
		const auto delta = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFSub, TypeF32(state), delta, value, load_vertex(0));
		value = delta;
	}
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

uint32_t MrtOutputMode(const EmitterState& state, const IR::ExportInfo& exp) {
	if (state.program.stage != ShaderType::Pixel || exp.kind != IR::ExportTargetKind::Mrt ||
	    exp.index >= std::size(state.input_info.pixel->target_output_mode)) {
		return 0;
	}
	return state.input_info.pixel->target_output_mode[exp.index];
}

uint32_t ExportRawComponent(ValueEmitContext& ctx, uint32_t vector, uint32_t component) {
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, vector,
	                              component);
	return value;
}

uint32_t ExportVector(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp,
                      bool uint_output) {
	auto& state = ctx.state;
	if (exp.compr && !uint_output) {
		const auto unpack =
		    MrtOutputMode(state, exp) == 5u ? GLSLstd450UnpackUnorm2x16 : GLSLstd450UnpackHalf2x16;
		uint32_t f32[4] = {ConstantF32(state, 0), ConstantF32(state, 0), ConstantF32(state, 0),
		                   ConstantF32(state, 0x3f800000u)};
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed   = state.builder.AllocateId();
			const auto unpacked = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, data, pair);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
			                          GlslStd450(state), unpack, packed);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) != 0u) {
					f32[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state),
					                          f32[component], unpacked, lane);
				}
			}
		}
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector,
		                          f32[0], f32[1], f32[2], f32[3]);
		return vector;
	}
	uint32_t raw[4] = {
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, uint_output ? 1u : 0x3f800000u),
	};
	if (exp.compr) {
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed = ExportRawComponent(ctx, data, pair);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) == 0u) {
					continue;
				}
				raw[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), raw[component],
				                          packed, ConstantU32(state, lane * 16u),
				                          ConstantU32(state, 16));
			}
		}
	} else {
		for (uint32_t component = 0; component < 4u; component++) {
			if (((exp.en >> component) & 1u) != 0u) {
				raw[component] = ExportRawComponent(ctx, data, component);
			}
		}
	}
	if (uint_output) {
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), vector,
		                          raw[0], raw[1], raw[2], raw[3]);
		return vector;
	}
	uint32_t f32[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		f32[component] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32[component], raw[component]);
	}
	const auto vector = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector, f32[0],
	                          f32[1], f32[2], f32[3]);
	return vector;
}

void EmitAuxPositionExport(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp) {
	auto& state = ctx.state;
	for (uint32_t component = 0; component < 4; component++) {
		if ((exp.en & (1u << component)) == 0) {
			continue;
		}
		const auto output = IR::DecodePositionExportComponent(
		    state.input_info.vertex->pa_cl_vs_out_cntl, exp.index, component);
		if (output.layer || output.viewport) {
			const auto raw = ExportRawComponent(ctx, data, component);
			if (output.layer) {
				const auto layer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), layer, raw,
				                          ConstantU32(state, 0x7ffu));
				const auto pointer = state.program.stage == ShaderType::Mesh
				                         ? MeshOutputPointer(state, IR::StageOutputKind::Layer)
				                         : state.layer_variable;
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
			if (output.viewport) {
				// GFX10 MISC.z packs the viewport index in bits 16..19 alongside the layer.
				const auto viewport = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), viewport, raw,
				                          ConstantU32(state, 16), ConstantU32(state, 4));
				state.builder.AddFunction(spv::OpStore, state.viewport_index_variable, viewport);
			}
			continue;
		}
		if (!output.point_size && output.clip_distance == UINT32_MAX &&
		    output.cull_distance == UINT32_MAX) {
			continue;
		}

		const auto raw = ExportRawComponent(ctx, data, component);
		const auto f32 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
		if (output.point_size) {
			state.builder.AddFunction(spv::OpStore, state.point_size_variable, f32);
			continue;
		}
		auto StoreDistance = [&](IR::StageOutputKind kind, uint32_t variable, uint32_t index) {
			if (index == UINT32_MAX) {
				return;
			}
			uint32_t pointer;
			if (state.program.stage == ShaderType::Mesh) {
				pointer = MeshOutputPointer(state, kind, index);
			} else {
				pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain,
				                          TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
				                          pointer, variable, ConstantU32(state, index));
			}
			state.builder.AddFunction(spv::OpStore, pointer, f32);
		};
		StoreDistance(IR::StageOutputKind::ClipDistance, state.clip_distance_variable, output.clip_distance);
		StoreDistance(IR::StageOutputKind::CullDistance, state.cull_distance_variable, output.cull_distance);
	}
}

// v423: the 2-bit source component of each physical component of render target `target` in the
// draw's export mapping (BindingLayout::export_mapping: 8 packed bytes in 2 dwords).
std::array<uint32_t, 4> ExportMappingSelectors(EmitterState& state, uint32_t target) {
	const auto word = EmitShaderDataDwordLoad(state, state.program.bindings.ExportMappingDword() + target / 4u);
	std::array<uint32_t, 4> selectors {};
	for (uint32_t component = 0; component < 4; component++) {
		const auto shifted = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), shifted, word,
		                          ConstantU32(state, (target % 4u) * 8u + component * 2u));
		selectors[component] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), selectors[component], shifted,
		                          ConstantU32(state, 3));
	}
	return selectors;
}

uint32_t ConvertClipCoordinate(EmitterState& state, uint32_t coordinate, uint32_t scale,
                               uint32_t offset, uint32_t half_extent) {
	const auto window  = state.builder.AllocateId();
	const auto biased  = state.builder.AllocateId();
	const auto divided = state.builder.AllocateId();
	const auto ndc     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), window, coordinate, scale);
	state.builder.AddFunction(spv::OpFAdd, TypeF32(state), biased, window, offset);
	state.builder.AddFunction(spv::OpFDiv, TypeF32(state), divided, biased, half_extent);
	state.builder.AddFunction(spv::OpFSub, TypeF32(state), ndc, divided,
	                          ConstantF32Value(state, 1.0f));
	return ndc;
}

uint32_t ConvertPositionToClipSpace(EmitterState& state, uint32_t position) {
	const auto& transform = state.input_info.vertex->clip_space;
	uint32_t    components[4] {};
	for (uint32_t i = 0; i < 4; i++) {
		components[i] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), components[i], position,
		                          i);
	}
	// v422: from the shader data when the layout reserves them (BindingLayout::clip_space):
	// scale x, y, offset x, y, half extent x, y.
	uint32_t values[6] {};
	const bool from_data = state.program.bindings.clip_space;
	for (uint32_t i = 0; i < 6; i++) {
		if (from_data) {
			const auto bits = EmitShaderDataDwordLoad(state, state.program.bindings.ClipSpaceDword() + i);
			values[i]       = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, TypeF32(state), values[i], bits);
		} else {
			const float* source[3] = {transform.scale, transform.offset, transform.half_extent};
			values[i]              = ConstantF32Value(state, source[i / 2][i % 2]);
		}
	}
	components[0] = ConvertClipCoordinate(state, components[0], values[0], values[2], values[4]);
	components[1] = ConvertClipCoordinate(state, components[1], values[1], values[3], values[5]);
	const auto converted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), converted,
	                          components[0], components[1], components[2], components[3]);
	return converted;
}

} // namespace

// v440 (spirvEmitterRt.cpp): the dispatch's thread extent on one axis, from the shader data.
uint32_t EmitDispatchThreadCount(EmitterState& state, uint32_t axis) {
	return EmitBuiltinU32(state, IR::StageInputKind::DispatchThreadCount, axis);
}
uint32_t EmitWqmU64(EmitterState& state, uint32_t value) {
	const auto shifted_one = state.builder.AllocateId();
	const auto merged_one  = state.builder.AllocateId();
	const auto shifted_two = state.builder.AllocateId();
	const auto merged_two  = state.builder.AllocateId();
	const auto quad_bits   = state.builder.AllocateId();
	const auto result      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_one, value,
	                          ConstantU64(state, 0x0000000100000001ull));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_one, value, shifted_one);
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_two, merged_one,
	                          ConstantU64(state, 0x0000000200000002ull));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_two, merged_one,
	                          shifted_two);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU64(state), quad_bits, merged_two,
	                          ConstantU64(state, 0x1111111111111111ull));
	state.builder.AddFunction(spv::OpIMul, TypeU64(state), result, quad_bits,
	                          ConstantU64(state, 0x0000000f0000000full));
	return result;
}

// Diagnostics: KYTY_PROBE_RECT=<x>,<y>,<w>,<h> (w * h <= 16384) with KYTY_PROBE_PC=<hash>@...: that
// pixel shader stores its float MRT0 export (the probed registers) for each pixel of the screen
// rectangle in the bindless feedback buffer (BindlessCaptureProbe); the host writes it to
// KYTY_PROBE_RECT_FILE. Numbers, where the on-screen probe only shows post-processed colours.
static void EmitProbeRectStore(EmitterState& state, uint32_t value) {
	struct Rect {
		uint64_t hash = 0;
		uint32_t x = 0, y = 0, w = 0, h = 0;
	};
	static const Rect rect = [] {
		Rect        result;
		const char* probe = std::getenv("KYTY_PROBE_PC");
		const char* value = std::getenv("KYTY_PROBE_RECT");
		if (probe == nullptr || value == nullptr) {
			return result;
		}
		result.hash = std::strtoull(probe, nullptr, 16);
		uint32_t numbers[4] {};
		char*    cursor = const_cast<char*>(value);
		for (auto& number: numbers) {
			number = static_cast<uint32_t>(std::strtoul(cursor, &cursor, 10));
			if (*cursor == ',') {
				cursor++;
			}
		}
		result.x = numbers[0];
		result.y = numbers[1];
		result.w = numbers[2];
		result.h = numbers[3];
		if (result.w == 0u || result.h == 0u ||
		    result.w * result.h * 4u > IR::BindlessCaptureProbeWords) {
			result.hash = 0;
		}
		return result;
	}();
	if (rect.hash == 0 || rect.hash != state.program.shader_hash ||
	    state.program.stage != ShaderType::Pixel || state.bindless_feedback_variable == 0) {
		return;
	}
	const auto frag = InputVariableForKind(state, IR::StageInputKind::FragCoord);
	if (frag == 0) {
		std::fprintf(stderr, "[probe-rect] %016llx has no FragCoord input; nothing stored\n",
		             static_cast<unsigned long long>(state.program.shader_hash));
		return;
	}
	const auto Coordinate = [&](uint32_t component, uint32_t origin) {
		const auto pointer = state.builder.AllocateId();
		const auto f32     = state.builder.AllocateId();
		const auto u32     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, frag, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), f32, pointer);
		state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), u32, f32);
		return std::pair {u32, Binary(state, spv::OpISub, TypeU32(state), u32,
		                              ConstantU32(state, origin))};
	};
	const auto [x, dx] = Coordinate(0, rect.x);
	const auto [y, dy] = Coordinate(1, rect.y);
	const auto inside  = Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                            Binary(state, spv::OpULessThan, TypeBool(state), dx,
	                                   ConstantU32(state, rect.w)),
	                            Binary(state, spv::OpULessThan, TypeBool(state), dy,
	                                   ConstantU32(state, rect.h)));
	EmitIfCondition(state, inside, [&]() {
		const auto index = Binary(state, spv::OpIAdd, TypeU32(state),
		                          Binary(state, spv::OpIMul, TypeU32(state), dy,
		                                 ConstantU32(state, rect.w)),
		                          dx);
		const auto base  = Binary(state, spv::OpIAdd, TypeU32(state),
		                          Binary(state, spv::OpIMul, TypeU32(state), index,
		                                 ConstantU32(state, 4u)),
		                          ConstantU32(state, IR::BindlessCaptureProbe));
		const auto Store = [&](uint32_t word, uint32_t bits) {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          pointer, state.bindless_feedback_variable,
			                          ConstantU32(state, 0),
			                          Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                 ConstantU32(state, word)));
			state.builder.AddFunction(spv::OpStore, pointer, bits);
		};
		Store(0, Binary(state, spv::OpBitwiseOr, TypeU32(state), x,
		                Binary(state, spv::OpShiftLeftLogical, TypeU32(state), y,
		                       ConstantU32(state, 16u))));
		for (uint32_t component = 0; component < 3u; component++) {
			const auto f32  = state.builder.AllocateId();
			const auto bits = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), f32, value,
			                          component);
			state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, f32);
			Store(1u + component, bits);
		}
	});
}

// Diagnostics: with KYTY_PROBE_RECT set, a probed shader of another stage appends its parameter-0
// export (the probed registers, Attribute.cpp) per vertex: an atomic counter at
// BindlessCaptureProbe, then records of 4 words (0x80000000 | sequence, three value bits).
static void EmitProbeAppend(EmitterState& state, uint32_t data) {
	const auto pointer = [&](uint32_t index) {
		const auto element = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          element, state.bindless_feedback_variable, ConstantU32(state, 0),
		                          index);
		return element;
	};
	const auto sequence = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), sequence,
	                          pointer(ConstantU32(state, IR::BindlessCaptureProbe)),
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1u));
	constexpr uint32_t Records = (IR::BindlessCaptureProbeWords - 4u) / 4u;
	const auto slot = Binary(state, spv::OpUMod, TypeU32(state), sequence, ConstantU32(state, Records));
	const auto base = Binary(state, spv::OpIAdd, TypeU32(state),
	                         Binary(state, spv::OpIMul, TypeU32(state), slot, ConstantU32(state, 4u)),
	                         ConstantU32(state, IR::BindlessCaptureProbe + 4u));
	const auto Store = [&](uint32_t word, uint32_t value) {
		state.builder.AddFunction(
		    spv::OpStore,
		    pointer(Binary(state, spv::OpIAdd, TypeU32(state), base, ConstantU32(state, word))), value);
	};
	Store(0, Binary(state, spv::OpBitwiseOr, TypeU32(state), sequence, ConstantU32(state, 0x80000000u)));
	for (uint32_t component = 0; component < 3u; component++) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data, component);
		Store(1u + component, value);
	}
}

// Diagnostics: KYTY_PROBE_TESS_LEVELS=1 with a probed evaluation shader. The invocation at the
// patch corner (u, v) = (0, 0) appends three consecutive records: (outer 0, 1, 2), (outer 3,
// inner 0, inner 1) as float bits, then the three probed registers (to tell patches apart).
static void EmitProbeTessLevelsAppend(EmitterState& state, uint32_t data) {
	const auto coord_variable = InputVariableForKind(state, IR::StageInputKind::TessCoord);
	if (coord_variable == 0) {
		return;
	}
	const auto coord = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 3), coord, coord_variable);
	const auto component = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value, coord, index);
		return value;
	};
	const auto is_zero = [&](uint32_t value) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFOrdEqual, TypeBool(state), result, value,
		                          ConstantF32Value(state, 0.0f));
		return result;
	};
	const auto corner = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), corner, is_zero(component(0)),
	                          is_zero(component(1)));
	EmitIfCondition(state, corner, [&]() {
		const auto pointer = [&](uint32_t index) {
			const auto element = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
			                          element, state.bindless_feedback_variable,
			                          ConstantU32(state, 0), index);
			return element;
		};
		const auto first = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), first,
		                          pointer(ConstantU32(state, IR::BindlessCaptureProbe)),
		                          ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone),
		                          ConstantU32(state, 3u));
		constexpr uint32_t Records = (IR::BindlessCaptureProbeWords - 4u) / 4u;
		const auto level = [&](uint32_t variable, uint32_t index) {
			const auto ptr = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)), ptr,
			                          variable, ConstantU32(state, index));
			const auto f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), f32, ptr);
			const auto bits = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, f32);
			return bits;
		};
		const auto probed = [&](uint32_t index) {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data, index);
			return value;
		};
		const uint32_t outer = state.probe_tess_outer_variable;
		const uint32_t inner = state.probe_tess_inner_variable;
		const std::array<std::array<uint32_t, 3>, 3> records {{
		    {level(outer, 0), level(outer, 1), level(outer, 2)},
		    {level(outer, 3), level(inner, 0), level(inner, 1)},
		    {probed(0), probed(1), probed(2)},
		}};
		for (uint32_t r = 0; r < 3u; r++) {
			const auto sequence = Binary(state, spv::OpIAdd, TypeU32(state), first, ConstantU32(state, r));
			const auto slot =
			    Binary(state, spv::OpUMod, TypeU32(state), sequence, ConstantU32(state, Records));
			const auto base = Binary(
			    state, spv::OpIAdd, TypeU32(state),
			    Binary(state, spv::OpIMul, TypeU32(state), slot, ConstantU32(state, 4u)),
			    ConstantU32(state, IR::BindlessCaptureProbe + 4u));
			const auto store = [&](uint32_t word, uint32_t value) {
				state.builder.AddFunction(
				    spv::OpStore,
				    pointer(Binary(state, spv::OpIAdd, TypeU32(state), base, ConstantU32(state, word))),
				    value);
			};
			store(0, Binary(state, spv::OpBitwiseOr, TypeU32(state), sequence,
			                ConstantU32(state, 0x80000000u)));
			for (uint32_t c = 0; c < 3u; c++) {
				store(1u + c, records[r][c]);
			}
		}
	});
}

void EmitSetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& exp   = ctx.Export(inst);
	const auto  exec  = ctx.Arg(inst, 1);
	if (state.program.stage == ShaderType::Pixel && exp.vm && state.requirements.pixel_valid_mask &&
	    state.pixel_valid_mask_variable != 0) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, exec, ConstantU32(state, 1),
		                          ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable, value);
	}
	if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) {
		return;
	}
	// Skip dormant color exports after their valid mask; MRT1 is reserved for logical alpha.
	if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
	    exp.index != 0 && state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
		return;
	}
	EmitIfCondition(state, exec, [&]() {
		const auto data = ctx.Arg(inst, 0);
		if (state.program.stage != ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Parameter &&
		    exp.index == 0 && ProbeAppendWanted(state) && state.bindless_feedback_variable != 0) {
			if (state.probe_tess_outer_variable != 0 && state.probe_tess_inner_variable != 0) {
				EmitProbeTessLevelsAppend(state, data);
			} else {
				EmitProbeAppend(state, data);
			}
		}
		if (exp.kind == IR::ExportTargetKind::Primitive) {
			if (state.program.stage == ShaderType::Mesh) {
				state.builder.AddFunction(spv::OpStore, MeshPrimitivePointer(state),
				                          ExportRawComponent(ctx, data, 0));
			}
			return;
		}
		if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0) {
			EmitAuxPositionExport(ctx, data, exp);
			return;
		}
		if (exp.kind == IR::ExportTargetKind::MrtZ) {
			if ((exp.en & 1u) != 0u && state.depth_variable != 0) {
				const auto raw = ExportRawComponent(ctx, data, 0);
				const auto f32 = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
				state.builder.AddFunction(spv::OpStore, state.depth_variable, f32);
			}
			if ((exp.en & 4u) != 0u && state.sample_mask_variable != 0) {
				const auto raw     = ExportRawComponent(ctx, data, 2);
				const auto value   = state.builder.AllocateId();
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeI32(state), value, raw);
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeI32(state)),
				    pointer, state.sample_mask_variable, ConstantU32(state, 0));
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
			return;
		}
		const auto variable =
		    state.program.stage == ShaderType::Mesh ? 0u : OutputVariableForExport(state, exp);
		if (state.program.stage != ShaderType::Mesh && variable == 0) {
			return;
		}
		const bool mrt =
		    state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt;
		const bool uint_output =
		    mrt && (state.input_info.pixel->target_uint_mask & (1u << exp.index)) != 0;
		const bool sint_output =
		    mrt && (state.input_info.pixel->target_sint_mask & (1u << exp.index)) != 0;
		const auto vector_type = uint_output   ? TypeU32Vector(state, 4)
		                         : sint_output ? TypeI32Vector(state, 4)
		                                       : TypeF32Vector(state, 4);
		auto       value       = ExportVector(ctx, data, exp, uint_output || sint_output);
		if (mrt && exp.index == 0 && !uint_output && !sint_output) {
			EmitProbeRectStore(state, value);
		}
		// KYTY_DEBUG_MAGENTA_PS=<hash>: that pixel shader's float MRT0 export becomes magenta,
		// to see where its draws land (diagnostic).
		if (static const std::string magenta = [] {
			    const char* env = std::getenv("KYTY_DEBUG_MAGENTA_PS");
			    return env != nullptr ? std::string(env) : std::string();
		    }();
		    !magenta.empty() && mrt && !uint_output && !sint_output &&
		    [&] {
			    static std::atomic<uint32_t> printed {0};
			    if (printed.fetch_add(1) < 400)
				    std::fprintf(stderr, "[magenta] ps %016llx mrt%u uint=%d\n",
				                 static_cast<unsigned long long>(state.program.shader_hash),
				                 exp.index, uint_output ? 1 : 0);
			    return true;
		    }() &&
		    magenta.find(fmt::format("{:016x}", state.program.shader_hash)) != std::string::npos) {
			value = state.builder.AllocateId();
			if (uint_output || sint_output) {
				const auto all = ConstantU32(state, 0xffffffffu);
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), value,
				                          all, ConstantU32(state, 0), all, all);
			} else {
				// The n-th hash in the list gets the n-th color (magenta, green, black, blue, ...),
				// to tell several shaders apart in one run.
				const auto pos   = magenta.find(fmt::format("{:016x}", state.program.shader_hash));
				const auto index = static_cast<size_t>(
				    std::count(magenta.begin(), magenta.begin() + static_cast<ptrdiff_t>(pos), ','));
				static constexpr float Colors[][3] = {{1, 0, 1}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1},
				                                      {1, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 0, 0}};
				const auto& c = Colors[index % 8];
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), value,
				                          ConstantF32(state, c[0]), ConstantF32(state, c[1]),
				                          ConstantF32(state, c[2]), ConstantF32(state, 1.0f));
			}
		}
		// KYTY_NAN_MAGENTA=1 (diagnostics): a float colour export with a NaN or Inf component is
		// written magenta instead, and (in shaders with the bindless feedback buffer) leaves the
		// shader's hash in BindlessNanExportWord, to find which pass produces a NaN.
		if (static const bool nan_magenta = [] {
			    const char* env = std::getenv("KYTY_NAN_MAGENTA");
			    return env != nullptr && env[0] == '1';
		    }();
		    nan_magenta && mrt && !uint_output && !sint_output) {
			const auto bool4   = state.builder.Type(spv::OpTypeVector, TypeBool(state), 4u);
			const auto is_nan  = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIsNan, bool4, is_nan, value);
			const auto is_inf  = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIsInf, bool4, is_inf, value);
			const auto bad4    = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalOr, bool4, bad4, is_nan, is_inf);
			const auto bad     = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAny, TypeBool(state), bad, bad4);
			const auto magenta = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), magenta,
			                          ConstantF32Value(state, 1.0f), ConstantF32Value(state, 0.0f),
			                          ConstantF32Value(state, 1.0f), ConstantF32Value(state, 1.0f));
			const auto chosen = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeF32Vector(state, 4), chosen, bad, magenta,
			                          value);
			value = chosen;
			if (state.bindless_feedback_variable != 0) {
				const auto word = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), word, bad,
				                          ConstantU32(state, IR::BindlessNanExportWord),
				                          ConstantU32(state, IR::BindlessNanExportWord + 1u));
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
				                          pointer, state.bindless_feedback_variable,
				                          ConstantU32(state, 0), word);
				state.builder.AddFunction(
				    spv::OpStore, pointer,
				    ConstantU32(state, static_cast<uint32_t>(state.program.shader_hash)));
			}
		}
		if (sint_output) {
			const auto signed_value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, vector_type, signed_value, value);
			value = signed_value;
			if (exp.compr) {
				const auto shift = state.builder.AllocateId();
				const auto bits  = ConstantU32(state, 16);
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), shift,
				                          bits, bits, bits, bits);
				const auto high = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpShiftLeftLogical, vector_type, high, value, shift);
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpShiftRightArithmetic, vector_type, value, high,
				                          shift);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index == 0 && !uint_output &&
		    state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
			// Broadcast logical alpha before swizzling the primary output.
			const auto blend_output =
			    OutputVariableForExport(state, {.kind = IR::ExportTargetKind::Mrt, .index = 1});
			if (blend_output != 0) {
				auto alpha = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, alpha, value, value,
				                          3u, 3u, 3u, 3u);
				uint32_t alpha_factor = 0;
				switch (state.input_info.pixel->alpha_blend_source) {
					case ShaderAlphaBlendSource::SourceAlphaOne:
						alpha_factor = ConstantF32Value(state, 1.0f);
						break;
					case ShaderAlphaBlendSource::SourceAlphaZero:
						alpha_factor = ConstantF32Value(state, 0.0f);
						break;
					default: break;
				}
				if (alpha_factor != 0 && state.program.bindings.export_mapping) {
					// v423: the first physical component the draw's mapping takes from alpha.
					const auto selectors = ExportMappingSelectors(state, 0);
					uint32_t   found     = ConstantBool(state, false);
					uint32_t   parts[4] {};
					for (uint32_t component = 0; component < 4; component++) {
						const auto is_alpha = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpIEqual, TypeBool(state), is_alpha,
						                          selectors[component], ConstantU32(state, 3));
						const auto not_found = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), not_found, found);
						const auto take = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), take, is_alpha,
						                          not_found);
						const auto alpha_part = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), alpha_part,
						                          alpha, component);
						parts[component] = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpSelect, TypeF32(state), parts[component], take,
						                          alpha_factor, alpha_part);
						const auto next = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), next, found, is_alpha);
						found = next;
					}
					const auto factors = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, factors, parts[0],
					                          parts[1], parts[2], parts[3]);
					alpha = factors;
				} else if (alpha_factor != 0) {
					const auto mapping = state.input_info.pixel->target_export_mapping[0];
					for (uint32_t component = 0; component < 4; component++) {
						if (mapping.Map(component) != 3u) continue;
						const auto factors = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeInsert, vector_type, factors,
						                          alpha_factor, alpha, component);
						alpha = factors;
						break;
					}
				}
				state.builder.AddFunction(spv::OpStore, blend_output, alpha);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size() &&
		    state.program.bindings.export_mapping) {
			// v423: the draw's mapping (BindingLayout::export_mapping), as a dynamic shuffle.
			const auto selectors = ExportMappingSelectors(state, exp.index);
			const auto element   = uint_output ? TypeU32(state) : sint_output ? TypeI32(state) : TypeF32(state);
			uint32_t parts[4] {};
			for (uint32_t component = 0; component < 4; component++) {
				parts[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorExtractDynamic, element, parts[component], value,
				                          selectors[component]);
			}
			const auto mapped = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, mapped, parts[0], parts[1],
			                          parts[2], parts[3]);
			value = mapped;
		} else if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size()) {
			const auto mapping = state.input_info.pixel->target_export_mapping[exp.index];
			if (!mapping.IsIdentity()) {
				const auto mapped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, mapped, value, value,
				                          mapping.Map(0), mapping.Map(1), mapping.Map(2),
				                          mapping.Map(3));
				value = mapped;
			}
		}
		if (exp.kind == IR::ExportTargetKind::Position &&
		    state.input_info.vertex->clip_space.enabled) {
			value = ConvertPositionToClipSpace(state, value);
		}
		if (state.program.stage == ShaderType::Mesh) {
			const auto kind = exp.kind == IR::ExportTargetKind::Position
			                      ? IR::StageOutputKind::Position
			                      : IR::StageOutputKind::Parameter;
			state.builder.AddFunction(spv::OpStore, MeshOutputPointer(state, kind, exp.index),
			                          value);
		} else if (exp.kind == IR::ExportTargetKind::Position) {
			if (state.invalid_position_clip_distance != UINT32_MAX) {
				const auto zero = state.builder.Constant(spv::OpConstantNull, TypeF32Vector(state, 4));
				const auto equal = state.builder.AllocateId();
				const auto invalid = state.builder.AllocateId();
				const auto distance = state.builder.AllocateId();
				const auto distance_pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFOrdEqual, TypeBoolVector(state, 4), equal,
				                          value, zero);
				const auto all_zero = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAll, TypeBool(state), all_zero, equal);
				// A NaN position component kills the vertex on the guest: shaders write NaN
				// (e.g. 0xff820000) to drop primitives, and AMD's clipper discards any primitive
				// with such a vertex. Host GPUs give no such guarantee (NVIDIA drew long slivers
				// across Wolverine's water). KYTY_NAN_POSITION_KILL=0: zero positions only.
				static const bool nan_kill = [] {
					const char* env = std::getenv("KYTY_NAN_POSITION_KILL");
					return env == nullptr || env[0] != '0';
				}();
				if (nan_kill) {
					const auto nan       = state.builder.AllocateId();
					const auto any_nan   = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpIsNan, TypeBoolVector(state, 4), nan, value);
					state.builder.AddFunction(spv::OpAny, TypeBool(state), any_nan, nan);
					state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), invalid, all_zero,
					                          any_nan);
					// The clip distance alone is not enough: the clipper interpolates the new
					// vertex as valid + 0 * (invalid - valid), and 0 * NaN is NaN, so the clipped
					// edge still carries a NaN and NVIDIA rasterizes it as a long spike. A finite
					// stand-in makes the clipped primitive collapse exactly onto its valid edge.
					const auto nan_mask = state.builder.AllocateId();
					const auto finite   = state.builder.AllocateId();
					const auto zero_f   = ConstantF32Value(state, 0.0f);
					const auto one_f    = ConstantF32Value(state, 1.0f);
					const auto stand_in = state.builder.Constant(
					    spv::OpConstantComposite, TypeF32Vector(state, 4), zero_f, zero_f, zero_f,
					    one_f);
					state.builder.AddFunction(spv::OpCompositeConstruct, TypeBoolVector(state, 4),
					                          nan_mask, any_nan, any_nan, any_nan, any_nan);
					state.builder.AddFunction(spv::OpSelect, TypeF32Vector(state, 4), finite,
					                          nan_mask, stand_in, value);
					value = finite;
				} else {
					state.builder.AddFunction(spv::OpCopyObject, TypeBool(state), invalid, all_zero);
				}
				// Zero at valid vertices makes a primitive containing an invalid position
				// collapse to its remaining edge, before the undefined 0/0 perspective divide.
				state.builder.AddFunction(spv::OpSelect, TypeF32(state), distance, invalid,
				                          ConstantF32Value(state, -1.0f),
				                          ConstantF32Value(state, 0.0f));
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
				    distance_pointer, state.clip_distance_variable,
				    ConstantU32(state, state.invalid_position_clip_distance));
				state.builder.AddFunction(spv::OpStore, distance_pointer, distance);
				static std::atomic<uint32_t> logged = 0;
				if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
					LOGF("Shader: emitted zero-position clip guard hash=0x%016" PRIx64 "\n",
					     state.program.shader_hash);
				}
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpAccessChain,
			    TypePointer(state, spv::StorageClassOutput, TypeF32Vector(state, 4)), pointer,
			    variable, ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		} else {
			state.builder.AddFunction(spv::OpStore, variable, value);
		}
	});
}

uint32_t EmitIdentity(ValueEmitContext&, uint32_t value) {
	return value;
}

void EmitVoid(ValueEmitContext&) {}

void EmitStoreCompletion(EmitterState& state) {
	state.builder.AddFunction(
	    spv::OpControlBarrier, ConstantU32(state, spv::ScopeSubgroup),
	    ConstantU32(state, spv::ScopeDevice),
	    ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask |
	                          spv::MemorySemanticsImageMemoryMask));
}

void EmitBarrier(EmitterState& state) {
	const auto tessellation = state.program.stage == ShaderType::TessellationControl;
	if (!tessellation && ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		// Independent graphics invocations have no native workgroup left to synchronize.
		return;
	}
	const auto memory_scope = tessellation ? spv::ScopeInvocation : LdsMemoryScope(state);
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer
	                        ? spv::MemorySemanticsUniformMemoryMask
	                        : spv::MemorySemanticsWorkgroupMemoryMask;
	const auto semantics = tessellation ? spv::MemorySemanticsMaskNone
	                                    : spv::MemorySemanticsAcquireReleaseMask | memory;
	// The ray-tracing stages have no workgroup: a barrier there can only span the subgroup.
	const auto execution_scope = state.program.rt_role != IR::RtRole::None ? spv::ScopeSubgroup : spv::ScopeWorkgroup;
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, execution_scope),
	                          ConstantU32(state, memory_scope), ConstantU32(state, semantics));
}

// One guest wave's LDS accesses are in program order (all LDS reads and writes of a wavefront
// complete in order), so a lane reads LDS another lane of its wave just wrote without a barrier.
// Host invocations do not run in lockstep: they need one, at the scope WaveLdsScope chose.
void EmitWaveLdsBarrier(EmitterState& state) {
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer
	                        ? spv::MemorySemanticsUniformMemoryMask
	                        : spv::MemorySemanticsWorkgroupMemoryMask;
	state.builder.AddFunction(
	    spv::OpControlBarrier, ConstantU32(state, state.wave_lds_scope),
	    ConstantU32(state, LdsMemoryScope(state)),
	    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | memory));
}

uint32_t EmitLaneId(EmitterState& state) {
	return state.program.stage == ShaderType::TessellationControl
	           ? EmitBuiltinU32(state, IR::StageInputKind::InvocationId, 0)
	           : EmitSubgroupLocalInvocationId(state);
}

uint32_t EmitMeshDrawParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	const auto index  = inst.Arg(0).U32();
	if (state.program.stage != ShaderType::Mesh || index >= IR::PushData::MeshDrawDwordCount) {
		ctx.Fail(inst, "invalid mesh draw parameter");
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 0),
	                          ConstantU32(state, index));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), result, pointer);
	return result;
}

uint32_t EmitGetUserData(EmitterState& state, IR::ScalarReg reg) {

	uint32_t dword = 0;
	if (!UserDataDwordIndex(state, reg, dword)) {
		return ConstantU32(state, 0);
	} else if (state.rt_relocate) {
		// v443: the register's place among the split's user-data registers, from the shader record.
		return EmitShaderDataDwordLoadAt(state, 0u, RtPlaced(state, RtRecordSlot::UserData, IR::RegIndex(reg), dword));
	} else {
		return EmitShaderDataDwordLoad(state, dword);
	}
}

uint32_t EmitGetBuiltin(ValueEmitContext& ctx, IR::Value kind, IR::Value index) {
	return EmitBuiltinU32(ctx.state, static_cast<IR::StageInputKind>(kind.U32()), index.U32());
}

uint32_t EmitGetDispatchThreadExtent(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto start = ctx.state.program.bindings.dispatch_thread_dword;
	if (start == IR::PushData::NoStart || !inst.Arg(0).IsImmediate() || inst.Arg(0).U32() >= 3u) {
		ctx.Fail(inst, "invalid dispatch thread extent");
	}
	if (DispatchDimensionsIndirect(ctx.state)) {
		return EmitBuiltinU32(ctx.state, IR::StageInputKind::DispatchThreadCount, inst.Arg(0).U32());
	}
	return EmitShaderDataDwordLoad(ctx.state, start + inst.Arg(0).U32());
}

uint32_t EmitUndefU1(EmitterState& state, const IR::Inst& inst) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUndef, TypeId(state, inst.GetType()), result);
	return result;
}

uint32_t EmitDppMoveU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto flags    = inst.Flags<IR::DppMoveFlags>();
	const auto target   = EmitDppTargetLane(state, flags);
	const auto shuffled = ctx.Shuffle(inst, 0, target.lane);
	if (flags.fetch_inactive) {
		return shuffled;
	}
	const auto ballot        = ctx.Ballot(inst.Arg(1));
	const auto source_active = EmitBallotLaneActiveBool(state, ballot, target.lane);
	const auto can_fetch     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), can_fetch, target.valid,
	                          source_active);
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, can_fetch, shuffled,
	                                                ConstantU32(state, 0));
}

uint32_t EmitDppUpdateU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto flags = inst.Flags<IR::DppMoveFlags>();
	auto       write = EmitDppWriteCondition(ctx, flags, ctx.Arg(inst, 2));
	if (!flags.bound_control && !flags.fetch_inactive && !flags.dpp8) {
		// A source lane EXEC disables is invalid like a vacated one, and without bound_ctrl the
		// receiving lane keeps its value (PS5 ISA, DPP options); so does a lane the host subgroup
		// lacks. EmitDppMoveU32 read zero from it.
		const auto target        = EmitDppTargetLane(ctx.state, flags);
		const auto source_active = EmitBallotLaneActiveBool(ctx.state, ctx.Ballot(inst.Arg(2)),
		                                                    target.lane);
		write = Binary(ctx.state, spv::OpLogicalAnd, TypeBool(ctx.state), write, source_active);
	}
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, write, ctx.Arg(inst, 0),
	                                                ctx.Arg(inst, 1));
}

// A wave64 workgroup shader on a 64-wide host subgroup (compute and mesh shaders on AMD) runs
// natively. Its VCC and EXEC branches decide for the wave as well, as the two emulated halves
// below do; a lane's own bit would split the wave. Lanes a partial workgroup lacks are not in the
// ballot, so a zero condition holds when no present lane fails it. SCC is computed from scalars
// and whole-wave mask words, so it is the same in every lane already.
static uint32_t EmitNativeWaveCondition(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto kind  = inst.Flags<CFG::BranchCondition>();
	const bool zero  = kind == CFG::BranchCondition::ExecZero || kind == CFG::BranchCondition::VccZero;
	if (!zero && kind != CFG::BranchCondition::ExecNonZero &&
	    kind != CFG::BranchCondition::VccNonZero) {
		return ctx.Arg(inst, 0);
	}
	auto predicate = ctx.Arg(inst, 0);
	if (zero) {
		predicate = Unary(state, spv::OpLogicalNot, TypeBool(state), predicate);
	}
	state.builder.RequireCapability(spv::CapabilityGroupNonUniform);
	state.builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
	const auto ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), predicate);
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	const auto any = Binary(state, spv::OpBitwiseOr, TypeU32(state), low, high);
	return Binary(state, zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(state), any,
	              ConstantU32(state, 0));
}

// A wave32 compute or mesh shader on a 32-wide host subgroup gets the same wave-uniform VCC and
// EXEC branches (KYTY_WAVE_UNIFORM_BRANCHES=0: per-lane ones, as before). On the hardware such a
// branch is a scalar decision for the whole wave; per-lane branches leave the lanes EXEC disables
// outside the region, and a region can re-enable them (S_ORN2_SAVEEXEC exec, S_MOV exec, -1) and
// read them across lanes. Wolverine's meshlet sphere CS 637ad82d enters a region with the lanes
// below the meshlet's vertex count, turns every lane on, starts the inactive ones at +-inf and
// reduces with DPP and V_READLANE 15/31: lanes left outside read as 0, so meshlets with fewer than
// 32 vertices got spheres around the origin (17..31) or all-zero ones (up to 16), and the culling
// removed them (the hexagonal holes in the skin).
//
// v446a (Mac 11-21): wave-uniform branches cost registers. Inside such a region every lane runs
// and each write is a select on EXEC, which keeps the old and the new value of every register the
// region writes live together; with per-lane branches the lanes EXEC disables skip the region and
// the driver can give both values one register, as the hardware does. NVIDIA, offline
// (tools/spv_regs): ade1bef389644fa5 168 -> 75 registers, dec083415ec94cc6 181 -> 31,
// 7cca2c0d258a643b 168 -> 128, 238da5993d101c7b 204 -> 155. Per-lane branches differ only where
// a lane EXEC disables matters inside the region: a cross-lane read (V_READLANE, DPP, permutes,
// swizzles; V_READFIRSTLANE reads an active lane) or an EXEC write that turns lanes on
// (IR::Program::exec_widens_lanes), as 637ad82d does. So by default only programs with one of
// those keep wave-uniform branches. KYTY_WAVE_UNIFORM_BRANCHES=1: every wave32 program (v445z),
// =0: none (per-lane, as before the 637ad82d fix).
// v446r: the default is 1 again. g455h (v446o, auto): the GPU hung (device lost, warps stuck in
// three shaders) 2.5 min into the New Game opening cinematic, as v446j/m did in Mac's play;
// KYTY_WAVE_UNIFORM_BRANCHES=2 brings the per-lane planning back.
static uint32_t WaveUniformBranchMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_WAVE_UNIFORM_BRANCHES");
		if (value == nullptr || value[0] == '\0') {
			return 1u;
		}
		return value[0] == '0' ? 0u : value[0] == '1' ? 1u : 2u;
	}();
	return mode;
}

// KYTY_WAVE_PERLANE_MIN=<n> (default 1000): a branch goes per-lane only when the region it splits
// holds at least n IR instructions; small regions keep their EXEC selects, which cost little, while
// many small divergent branches cost more (g453c, every region per-lane: 7cca2c0d 382 -> 731 us per
// call; g453e/g453h, 1000: 419/438 us, while ade1bef 358 -> 201/208 us, dec08341 149 -> 46/66 us
// and 032d821b 58 -> 42/38 us kept their gains). 0: every region.
static uint32_t WavePerLaneMinInstructions() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_WAVE_PERLANE_MIN");
		return text == nullptr ? 1000u : static_cast<uint32_t>(std::strtoul(text, nullptr, 0));
	}();
	return value;
}

// KYTY_WAVE_UNIFORM_HASHES=<hash,...>: those programs keep wave-uniform branches (diagnostics).
static bool WaveUniformListed(uint64_t hash) {
	static const std::string list = [] {
		const char* text = std::getenv("KYTY_WAVE_UNIFORM_HASHES");
		return text == nullptr ? std::string() : std::string(text);
	}();
	if (list.empty()) {
		return false;
	}
	const auto name = fmt::format("{:016x}", hash);
	return list.find(name) != std::string::npos;
}

// A cross-lane read or a control barrier in the block, or nullptr. Control barriers need
// subgroup-uniform control flow: the workgroup barrier, the store-completion barrier
// (EmitStoreCompletion) and, while wave LDS ordering is on (WaveLdsScope), the barrier between LDS
// accesses (EmitBlock). g453a with v446c (no barriers counted) lost the device in
// e1f7c4d1cc06bdeb, whose per-lane regions held 7 of them.
static const char* LaneSensitiveOp(const EmitterState& state, const IR::Block& block) {
	for (const auto& inst: block) {
		switch (inst.GetOpcode()) {
			case IR::ValueOpcode::ReadLane: return "ReadLane";
			case IR::ValueOpcode::WriteLane: return "WriteLane";
			case IR::ValueOpcode::DppMoveU32: return "DppMoveU32";
			case IR::ValueOpcode::DppUpdateU32: return "DppUpdateU32";
			case IR::ValueOpcode::Permlane16U32: return "Permlane16U32";
			case IR::ValueOpcode::PermuteU32: return "PermuteU32";
			case IR::ValueOpcode::BpermuteU32: return "BpermuteU32";
			case IR::ValueOpcode::SwizzleU32: return "SwizzleU32";
			case IR::ValueOpcode::WqmU64: return "WqmU64";
			case IR::ValueOpcode::Barrier: return "Barrier";
			case IR::ValueOpcode::StoreCompletion: return "StoreCompletion";
			default:
				if (state.wave_lds_scope != 0 && IR::SharedAccessOf(inst.GetOpcode()) != IR::SharedAccess::None) {
					return "LDS access (wave LDS barrier)";
				}
				break;
		}
	}
	return nullptr;
}

// Which branch blocks keep wave-uniform branches: a branch whose lanes have not rejoined before a
// lane-sensitive block (a cross-lane read, or an EXEC write that turns lanes on). With per-lane
// branches the lanes split at a branch and rejoin at its immediate post-dominator; a
// lane-sensitive block reachable from the branch's targets before that point would run without
// the lanes that took the other way.
static void PlanWaveUniformBranches(EmitterState& state) {
	const auto& program = state.program;
	const auto  mode    = WaveUniformBranchMode();
	static const bool log = std::getenv("KYTY_WAVE_UNIFORM_LOG") != nullptr;
	state.wave_uniform_ready = true;
	if (mode != 2u) {
		state.wave_per_lane_all = mode == 0u;
		return;
	}
	// v446h: the A/B variant with wave-uniform branches everywhere (KYTY_TRANSLATION_VARIANT=2).
	if ((program.info.translation_variant & 2u) != 0u || WaveUniformListed(program.shader_hash)) {
		return;
	}
	const auto  min_size = WavePerLaneMinInstructions();
	const auto& blocks = program.blocks;
	const auto  count  = blocks.size();
	std::unordered_map<const IR::Block*, uint32_t> index;
	index.reserve(count);
	for (uint32_t i = 0; i < count; i++) {
		index.emplace(blocks[i], i);
	}
	std::vector<uint8_t> sensitive(count, 0);
	bool any_sensitive = false;
	bool any_widening  = false;
	for (uint32_t i = 0; i < count; i++) {
		const char* op = LaneSensitiveOp(state, *blocks[i]);
		if (op != nullptr || blocks[i]->exec_widens) {
			sensitive[i] = 1;
			any_sensitive = true;
		}
		any_widening = any_widening || blocks[i]->exec_widens;
	}
	// The flag without a flagged block (a pass rebuilt the blocks), or a merged callable split
	// (one module's blocks at a time): every branch stays wave-uniform when anything is sensitive.
	if ((program.exec_widens_lanes && !any_widening) || program.layout_blocks != nullptr) {
		bool other = program.exec_widens_lanes;
		const auto& all = program.layout_blocks != nullptr ? *program.layout_blocks : blocks;
		for (const auto* block: all) {
			other = other || LaneSensitiveOp(state, *block) != nullptr;
		}
		state.wave_per_lane_all = !other;
		if (log) {
			LOGF("[wave-branches] %016llx %s (whole program)\n", static_cast<unsigned long long>(program.shader_hash),
			     other ? "wave-uniform" : "per-lane");
		}
		return;
	}
	if (!any_sensitive && min_size == 0u) {
		state.wave_per_lane_all = true;
		if (log) {
			LOGF("[wave-branches] %016llx per-lane (no lane-sensitive block)\n",
			     static_cast<unsigned long long>(program.shader_hash));
		}
		return;
	}
	// Immediate post-dominators (Cooper, Harvey, Kennedy) on the reversed graph, with a virtual
	// exit (index count) after every block without successors.
	const uint32_t exit = static_cast<uint32_t>(count);
	std::vector<std::vector<uint32_t>> succ(count + 1);
	for (uint32_t i = 0; i < count; i++) {
		for (const auto* next: blocks[i]->ImmSuccessors()) {
			if (const auto found = index.find(next); found != index.end()) {
				succ[i].push_back(found->second);
			}
		}
		if (succ[i].empty()) {
			succ[i].push_back(exit);
		}
	}
	// Reverse graph: edges from a block's successor back to it.
	std::vector<std::vector<uint32_t>> rsucc(count + 1);
	for (uint32_t i = 0; i < count; i++) {
		for (const auto s: succ[i]) {
			rsucc[s].push_back(i);
		}
	}
	// Postorder of the reverse graph from the exit.
	std::vector<uint32_t> order;
	std::vector<uint32_t> number(count + 1, UINT32_MAX);
	{
		std::vector<std::pair<uint32_t, size_t>> stack;
		std::vector<uint8_t>                     seen(count + 1, 0);
		stack.emplace_back(exit, 0);
		seen[exit] = 1;
		while (!stack.empty()) {
			auto& [node, next] = stack.back();
			if (next < rsucc[node].size()) {
				const auto child = rsucc[node][next++];
				if (seen[child] == 0) {
					seen[child] = 1;
					stack.emplace_back(child, 0);
				}
			} else {
				number[node] = static_cast<uint32_t>(order.size());
				order.push_back(node);
				stack.pop_back();
			}
		}
	}
	std::vector<uint32_t> ipdom(count + 1, UINT32_MAX);
	ipdom[exit] = exit;
	const auto intersect = [&](uint32_t a, uint32_t b) {
		while (a != b) {
			while (number[a] < number[b]) a = ipdom[a];
			while (number[b] < number[a]) b = ipdom[b];
		}
		return a;
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (size_t k = order.size(); k-- > 0;) {
			const auto node = order[k];
			if (node == exit) continue;
			uint32_t idom = UINT32_MAX;
			for (const auto s: succ[node]) { // predecessors of node in the reversed graph
				if (number[s] == UINT32_MAX || ipdom[s] == UINT32_MAX) continue;
				idom = idom == UINT32_MAX ? s : intersect(s, idom);
			}
			if (idom != UINT32_MAX && ipdom[node] != idom) {
				ipdom[node] = idom;
				changed = true;
			}
		}
	}
	std::vector<uint32_t> sizes(count, 0);
	for (uint32_t i = 0; i < count; i++) {
		sizes[i] = static_cast<uint32_t>(blocks[i]->Instructions().size());
	}
	uint32_t uniform = 0;
	uint32_t small   = 0;
	uint32_t branches = 0;
	std::vector<uint32_t> mark(count + 1, UINT32_MAX);
	std::vector<uint32_t> work;
	for (uint32_t b = 0; b < count; b++) {
		if (succ[b].size() < 2) continue;
		branches++;
		// A block the reverse walk never reached (an endless loop): keep the branch wave-uniform.
		const uint32_t stop = ipdom[b] == UINT32_MAX ? UINT32_MAX - 1 : ipdom[b];
		bool     hit  = ipdom[b] == UINT32_MAX;
		uint64_t size = 0;
		work.assign(succ[b].begin(), succ[b].end());
		while (!hit && !work.empty()) {
			const auto node = work.back();
			work.pop_back();
			// v446o: a branch that reaches itself is a loop branch; loops keep wave-uniform branches
			// (lanes leave a per-lane loop one by one, which the guest's wave never does).
			if (node == b) {
				hit = true;
				break;
			}
			if (node == stop || node == exit || mark[node] == b) continue;
			mark[node] = b;
			if (sensitive[node] != 0) {
				hit = true;
				break;
			}
			size += sizes[node];
			for (const auto s: succ[node]) work.push_back(s);
		}
		if (hit) {
			uniform++;
		} else if (size < min_size) {
			small++;
		} else {
			state.wave_per_lane_blocks.insert(blocks[b]);
		}
	}
	if (log) {
		LOGF("[wave-branches] %016llx %u of %u branch blocks wave-uniform (+%u small regions)\n",
		     static_cast<unsigned long long>(program.shader_hash), uniform, branches, small);
	}
}

// v446o: whether this program may take per-lane branches (then its module asks for maximal
// reconvergence: structured merges bring the lanes back together before cross-lane work).
bool MayUsePerLaneBranches(EmitterState& state) {
	if (WaveUniformBranchMode() == 1u) {
		return false;
	}
	if (!state.wave_uniform_ready) {
		PlanWaveUniformBranches(state);
	}
	return state.wave_per_lane_all || !state.wave_per_lane_blocks.empty();
}

static bool WaveUniformBranch(EmitterState& state, const IR::Inst& inst) {
	if (!state.wave_uniform_ready) {
		PlanWaveUniformBranches(state);
	}
	// A condition outside a planned branch block (not expected) stays wave-uniform.
	return !state.wave_per_lane_all && !state.wave_per_lane_blocks.contains(inst.Parent());
}

uint32_t EmitConditionRef(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.other_half == nullptr) {
		const auto* workgroup = ShaderWorkgroupInput(ctx.state.program.stage, ctx.state.input_info);
		if (workgroup != nullptr && ctx.state.program.wave_size == workgroup->host_subgroup_size &&
		    (ctx.state.program.wave_size == 64u ||
		     (ctx.state.program.wave_size == 32u && WaveUniformBranch(ctx.state, inst)))) {
			return EmitNativeWaveCondition(ctx, inst);
		}
		return ctx.Arg(inst, 0);
	}
	// A native scalar branch makes one decision for both emulated wave halves.
	if (ctx.half != 0) return ctx.other_half->Def(IR::Value(&inst));
	const auto kind = inst.Flags<CFG::BranchCondition>();
	if (kind == CFG::BranchCondition::ScalarInstruction) return ctx.Arg(inst, 0);
	const auto ballot = ctx.Ballot(inst.Arg(0));
	const auto low = ctx.state.builder.AllocateId();
	const auto high = ctx.state.builder.AllocateId();
	const auto combined = ctx.state.builder.AllocateId();
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), low, ballot, 0);
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), high, ballot, 1);
	const bool zero = kind == CFG::BranchCondition::ExecZero ||
	                  kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::SccZero;
	ctx.state.builder.AddFunction(zero ? spv::OpBitwiseAnd : spv::OpBitwiseOr,
	                              TypeU32(ctx.state), combined, low, high);
	ctx.state.builder.AddFunction(zero ? spv::OpIEqual : spv::OpINotEqual,
	                              TypeBool(ctx.state), result, combined,
	                              ConstantU32(ctx.state, zero ? ~0u : 0u));
	return result;
}

uint32_t EmitBallot(ValueEmitContext& ctx, IR::Value predicate) {
	return ctx.Ballot(predicate);
}

uint32_t EmitReadFirstLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	// KYTY_RFL_IDENTITY_HASHES=<hash,...>: in those shaders each lane keeps its own value, for
	// values the game expects to be wave-uniform that the host's lane packing makes divergent
	// (experiment).
	if (static const std::string identity = [] {
		    const char* env = std::getenv("KYTY_RFL_IDENTITY_HASHES");
		    return env != nullptr ? std::string(env) : std::string();
	    }();
	    !identity.empty() &&
	    ((identity == "ps" && ctx.state.program.stage == ShaderType::Pixel) ||
	     identity.find(fmt::format("{:016x}", ctx.state.program.shader_hash)) != std::string::npos)) {
		return ctx.Def(inst.Arg(0));
	}
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto lane   = ctx.FirstLane(ballot);
	return ctx.Shuffle(inst, 0, lane);
}

// The lanes' reduction, natively over the lanes the host subgroup has (IR::MatchLaneReduction).
static uint32_t EmitLaneReduction(ValueEmitContext& ctx, const IR::LaneReduction& reduction) {
	auto& state = ctx.state;
	// A wave64 on a 32-wide subgroup keeps lanes 32-63 in the second half.
	const auto host_lanes = state.lane_count == 2 ? 32u : state.program.wave_size;
	auto&      lane = reduction.first_lane / host_lanes == ctx.half ? ctx : *ctx.other_half;
	const auto subid = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), subid,
	                          state.subgroup_local_invocation_id_variable);
	const auto in_range =
	    Binary(state, spv::OpULessThan, TypeBool(state),
	           Binary(state, spv::OpISub, TypeU32(state), subid,
	                  ConstantU32(state, reduction.first_lane % host_lanes)),
	           ConstantU32(state, reduction.lanes));
	const auto contribution =
	    Select(state, TypeU32(state), in_range, lane.Def(reduction.source),
	           ConstantU32(state, IR::ReductionIdentity(reduction.operation)));
	spv::Op operation = spv::OpGroupNonUniformIAdd;
	switch (reduction.operation) {
		case IR::ValueOpcode::UMax32: operation = spv::OpGroupNonUniformUMax; break;
		case IR::ValueOpcode::UMin32: operation = spv::OpGroupNonUniformUMin; break;
		case IR::ValueOpcode::SMax32: operation = spv::OpGroupNonUniformSMax; break;
		case IR::ValueOpcode::SMin32: operation = spv::OpGroupNonUniformSMin; break;
		case IR::ValueOpcode::BitwiseOr32: operation = spv::OpGroupNonUniformBitwiseOr; break;
		case IR::ValueOpcode::BitwiseAnd32: operation = spv::OpGroupNonUniformBitwiseAnd; break;
		case IR::ValueOpcode::BitwiseXor32: operation = spv::OpGroupNonUniformBitwiseXor; break;
		default: break;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(operation, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          contribution);
	return result;
}

uint32_t EmitReadLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (const auto reduction = IR::MatchLaneReduction(inst, ctx.state.program.wave_size)) {
		return EmitLaneReduction(ctx, *reduction);
	}
	return ctx.Shuffle(inst, 0, ctx.Arg(inst, 1));
}

uint32_t EmitWriteLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto hit   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), hit,
	                          EmitSubgroupLocalInvocationId(state), ctx.Arg(inst, 2));
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, hit, ctx.Arg(inst, 1),
	                                                ctx.Arg(inst, 0));
}

uint32_t EmitPermlane16U32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state     = ctx.state;
	const auto flags     = inst.Flags<IR::PermlaneFlags>();
	const auto subid     = EmitSubgroupLocalInvocationId(state);
	const auto row       = state.builder.AllocateId();
	const auto row_value = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto lane8     = state.builder.AllocateId();
	const auto shift     = state.builder.AllocateId();
	const auto upper     = state.builder.AllocateId();
	const auto selected  = state.builder.AllocateId();
	const auto shifted   = state.builder.AllocateId();
	const auto index     = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	if (flags.x16) {
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), row_value, row,
		                          ConstantU32(state, 16));
	} else {
		state.builder.AddFunction(spv::OpCopyObject, TypeU32(state), row_value, row);
	}
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane8, lane,
	                          ConstantU32(state, 7));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane8,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), upper, lane,
	                          ConstantU32(state, 8));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, upper, ctx.Arg(inst, 2),
	                          ctx.Arg(inst, 1));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), shifted, selected, shift);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), index, shifted,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row_value, index);
	const auto shuffled = ctx.Shuffle(inst, 0, target);
	uint32_t   result   = shuffled;
	if (!flags.fetch_inactive) {
		const auto source_exec = ctx.Shuffle(inst, 3, target);
		result                 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, source_exec, shuffled,
		                          ConstantU32(state, 0));
	}
	return result;
}

uint32_t EmitGetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitAttribute(ctx.state, inst.Arg(0).U32(), inst.Arg(1).U32());
}

uint32_t EmitGetInterpolationParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitInterpolationParameter(ctx, inst.Arg(0).U32(), inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetShaderBase(ValueEmitContext& ctx) {
	// Guest S_GETPC values stay shader-relative in SPIR-V, matching the runtime ABI. The
	// runtime descriptor evaluator supplies the mapped shader base for host-side planning.
	return ctx.Def(IR::Value(uint64_t {0}));
}

uint32_t EmitReadClockRealtime64(ValueEmitContext& ctx) {
	// S_MEMREALTIME is a free-running counter at 100 MHz. Read the device clock once as a
	// uvec2 so the two halves cannot tear, then bring it down toward the guest's rate with a
	// shift (no 64-bit divide): a 1 GHz clock (NVIDIA) shifted by 3 lands near 125 MHz, so guest
	// timeouts expire slightly early rather than 10x early; a 100 MHz clock (AMD) is not
	// shifted. Shifting AMD's clock by 3 made the game's GPU frame pacer wait 8x its interval,
	// 133 or 267 ms a frame (DEBUGGING.md, 2026-10-01).
	auto&      state = ctx.state;
	const auto clock = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpReadClockKHR, TypeU32Vector(state, 2), clock,
	                          ConstantU32(state, spv::ScopeDevice));
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, clock, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, clock, 1);
	const uint32_t ClockShift = GetDeviceClockShift();
	if (ClockShift == 0) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU64(state), result, low, high);
		return result;
	}
	const auto low_shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low,
	                                ConstantU32(state, ClockShift));
	const auto high_carried = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high,
	                                 ConstantU32(state, 32u - ClockShift));
	const auto low_result =
	    Binary(state, spv::OpBitwiseOr, TypeU32(state), low_shifted, high_carried);
	const auto high_result = Binary(state, spv::OpShiftRightLogical, TypeU32(state), high,
	                                ConstantU32(state, ClockShift));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU64(state), result, low_result,
	                          high_result);
	return result;
}

void EmitUnreachable(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "must be lowered before SPIR-V emission");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
