#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t TessellationPointer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state          = ctx.state;
	using Attribute      = IR::TessellationAttribute;
	const auto  kind     = static_cast<Attribute>(inst.Arg(0).U32());
	const auto& tess     = state.input_info.vertex->tess;
	const auto  variable = state.tess_variables.at(static_cast<uint32_t>(kind));
	EXIT_IF(variable == 0);
	const auto input   = kind == Attribute::ControlInput || kind == Attribute::EvaluationInput;
	const auto storage = input ? spv::StorageClassInput : spv::StorageClassOutput;
	const auto pointer = state.builder.AllocateId();
	if (kind == Attribute::Factor) {
		EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
		// The factor ring holds the outer factors, then the inner ones: tri 3 + 1, quad 4 + 2,
		// isoline 2 + 0.
		const auto index       = inst.Arg(1).U32() / 4u;
		const auto outer_count = tess.domain == 2u ? 4u : tess.domain == 1u ? 3u : 2u;
		const auto inner_count = tess.domain == 2u ? 2u : tess.domain == 1u ? 1u : 0u;
		const auto outer       = index < outer_count;
		EXIT_NOT_IMPLEMENTED(index >= outer_count + inner_count);
		// KYTY_TESS_FACTOR_PERM=<outer digits>[:<inner digits>] (diagnostic, quads): the ring's
		// i-th outer (inner) factor goes to TessLevelOuter (Inner) [digit i], e.g. 1032:10.
		static const std::string perm = [] {
			const char* env = std::getenv("KYTY_TESS_FACTOR_PERM");
			return env != nullptr ? std::string(env) : std::string();
		}();
		uint32_t level = outer ? index : index - outer_count;
		if (tess.domain == 2u && !perm.empty()) {
			const auto colon = perm.find(':');
			const auto part  = outer ? perm.substr(0, colon)
			                         : (colon == std::string::npos ? std::string() : perm.substr(colon + 1));
			if (level < part.size() && part[level] >= '0' && part[level] <= '3') {
				level = static_cast<uint32_t>(part[level] - '0');
			}
		}
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, TypeF32(state)),
		                          pointer, outer ? variable : state.tess_inner_variable,
		                          ConstantU32(state, level));
		return pointer;
	}
	auto address = ctx.Arg(inst, 1);
	if (kind == Attribute::PatchOutput) {
		address =
		    EmitBinaryU32(state, spv::OpISub, address, ConstantU32(state, state.tess_patch_base));
	}
	const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
	const auto stride = local ? tess.ls_stride : tess.hs_stride;
	const auto offset = kind == Attribute::PatchOutput ? address
	                                                   : EmitBinaryU32(state, spv::OpUMod, address,
	                                                                   ConstantU32(state, stride));
	const auto attribute =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 4));
	const auto component =
	    EmitBinaryU32(state, spv::OpBitwiseAnd,
	                  EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 2)),
	                  ConstantU32(state, 3));
	const auto type = TypePointer(state, storage, TypeU32(state));
	if (kind == Attribute::LocalOutput || kind == Attribute::PatchOutput) {
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute,
		                          component);
	} else {
		const auto vertex =
		    kind == Attribute::ControlOutput
		        ? EmitLaneId(state)
		        : EmitBinaryU32(state, spv::OpUDiv, address, ConstantU32(state, stride));
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, vertex, attribute,
		                          component);
	}
	return pointer;
}

} // namespace

void DefineTessellationInterfaces(EmitterState& state) {
	using Attribute = IR::TessellationAttribute;
	std::array<bool, 6> used {};
	uint32_t            patch_begin = UINT32_MAX, patch_end = 0;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::GetTessellationAttribute &&
			    inst.GetOpcode() != IR::ValueOpcode::SetTessellationAttribute)
				continue;
			const auto kind = inst.Arg(0).U32();
			used.at(kind)   = true;
			if (kind == static_cast<uint32_t>(Attribute::PatchOutput)) {
				EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
				patch_begin = std::min(patch_begin, inst.Arg(1).U32());
				patch_end   = std::max(patch_end, inst.Arg(1).U32() + 4u);
			}
		}
	}
	if (std::ranges::none_of(used, [](bool value) { return value; })) return;
	const auto& tess  = state.input_info.vertex->tess;
	const auto  array = [&](uint32_t type, uint32_t count) {
		return state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	};
	for (uint32_t index = 0; index < used.size(); index++) {
		if (!used[index]) continue;
		const auto kind    = static_cast<Attribute>(index);
		const bool input   = kind == Attribute::ControlInput || kind == Attribute::EvaluationInput;
		const auto storage = input ? spv::StorageClassInput : spv::StorageClassOutput;
		uint32_t   type;
		if (kind == Attribute::Factor) {
			type = array(TypeF32(state), 4u);
		} else if (kind == Attribute::PatchOutput) {
			state.tess_patch_base = patch_begin;
			type = array(TypeU32Vector(state, 4), (patch_end - patch_begin + 15u) / 16u);
		} else {
			const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
			const auto stride = local ? tess.ls_stride : tess.hs_stride;
			type              = array(TypeU32Vector(state, 4), (stride + 15u) / 16u);
			if (kind != Attribute::LocalOutput) {
				type = array(type, local ? tess.input_control_points : tess.output_control_points);
			}
		}
		auto& variable = state.tess_variables[index];
		variable       = DefineInterfaceVariable(state, type, storage, "tess_attributes");
		if (kind == Attribute::Factor) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
			                            spv::BuiltInTessLevelOuter);
			state.tess_inner_variable =
			    DefineInterfaceVariable(state, array(TypeF32(state), 2u), storage, "tess_inner");
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationBuiltIn, spv::BuiltInTessLevelInner);
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationPatch);
		} else {
			state.builder.AddAnnotation(
			    spv::OpDecorate, variable, spv::DecorationLocation,
			    kind == Attribute::PatchOutput ? (tess.hs_stride + 15u) / 16u : 0u);
		}
		if (kind == Attribute::Factor || kind == Attribute::PatchOutput) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationPatch);
		}
	}
}

void DefineTessellationExecutionModes(EmitterState& state) {
	const auto& tess = state.input_info.vertex->tess;
	// VGT_TF_PARAM: TYPE 0 isoline, 1 tri, 2 quad; PARTITIONING 0 integer, 1 pow2,
	// 2 fractional odd, 3 fractional even; TOPOLOGY 0 point, 1 line, 2 tri CW, 3 tri CCW.
	// Wolverine's terrain is a quad domain with fractional-even spacing.
	EXIT_NOT_IMPLEMENTED(tess.domain > 2u || tess.partitioning > 3u || tess.output_topology > 3u);
	state.builder.RequireCapability(spv::CapabilityTessellation);
	if (state.program.stage == ShaderType::TessellationControl) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputVertices,
		                               tess.output_control_points);
	} else {
		state.builder.AddExecutionMode(state.main_func,
		                               tess.domain == 2u   ? spv::ExecutionModeQuads
		                               : tess.domain == 1u ? spv::ExecutionModeTriangles
		                                                   : spv::ExecutionModeIsolines);
		// KYTY_TESS_SPACING=<0|2|3> (diagnostic): equal, fractional odd or fractional even spacing
		// instead of the guest's partitioning.
		static const int spacing_override = [] {
			const char* env = std::getenv("KYTY_TESS_SPACING");
			return env != nullptr ? std::atoi(env) : -1;
		}();
		const auto partitioning = spacing_override >= 0 ? static_cast<uint32_t>(spacing_override)
		                                                : tess.partitioning;
		std::fprintf(stderr, "[tess-mode] domain=%u partitioning=%u (guest %u) topology=%u\n",
		             tess.domain, partitioning, tess.partitioning, tess.output_topology);
		state.builder.AddExecutionMode(state.main_func,
		                               partitioning == 2u   ? spv::ExecutionModeSpacingFractionalOdd
		                               : partitioning == 3u ? spv::ExecutionModeSpacingFractionalEven
		                                                    : spv::ExecutionModeSpacingEqual);
		if (tess.output_topology == 0u) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModePointMode);
		} else if (tess.output_topology >= 2u) {
			state.builder.AddExecutionMode(state.main_func, tess.output_topology == 2u
			                                                    ? spv::ExecutionModeVertexOrderCw
			                                                    : spv::ExecutionModeVertexOrderCcw);
		}
	}
}

uint32_t EmitGetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, 2), [&] {
		const auto value = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value,
		                              TessellationPointer(ctx, inst));
		return value;
	});
}

void EmitSetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto condition = ctx.Arg(inst, 3);
	// KYTY_TESS_FACTOR_LANE0=1 (diagnostic): only invocation 0 writes the tessellation factors.
	static const bool lane0 = std::getenv("KYTY_TESS_FACTOR_LANE0") != nullptr;
	if (lane0 && inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::Factor)) {
		auto&      st    = ctx.state;
		const auto first = st.builder.AllocateId();
		st.builder.AddFunction(spv::OpIEqual, TypeBool(st), first, EmitLaneId(st),
		                       ConstantU32(st, 0));
		const auto both = st.builder.AllocateId();
		st.builder.AddFunction(spv::OpLogicalAnd, TypeBool(st), both, condition, first);
		condition = both;
	}
	EmitIfCondition(ctx.state, condition, [&] {
		auto value = ctx.Arg(inst, 2);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::Factor)) {
			const auto floating = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpBitcast, TypeF32(ctx.state), floating, value);
			value = floating;
			// KYTY_TESS_FORCE_FACTOR=<f>: every tessellation factor becomes f (diagnostic).
			static const float forced = [] {
				const char* env = std::getenv("KYTY_TESS_FORCE_FACTOR");
				return env != nullptr ? static_cast<float>(std::atof(env)) : 0.0f;
			}();
			if (forced > 0.0f) value = ConstantF32Value(ctx.state, forced);
			// KYTY_TESS_FORCE_OUTER=<f> / KYTY_TESS_FORCE_INNER=<f>: only the outer (inner)
			// factors become f (diagnostic).
			static const float forced_outer = [] {
				const char* env = std::getenv("KYTY_TESS_FORCE_OUTER");
				return env != nullptr ? static_cast<float>(std::atof(env)) : 0.0f;
			}();
			static const float forced_inner = [] {
				const char* env = std::getenv("KYTY_TESS_FORCE_INNER");
				return env != nullptr ? static_cast<float>(std::atof(env)) : 0.0f;
			}();
			if (forced <= 0.0f && (forced_outer > 0.0f || forced_inner > 0.0f) &&
			    inst.Arg(1).IsImmediate()) {
				const auto& tess        = ctx.state.input_info.vertex->tess;
				const auto  outer_count = tess.domain == 2u ? 4u : tess.domain == 1u ? 3u : 2u;
				const bool  outer       = inst.Arg(1).U32() / 4u < outer_count;
				const float f           = outer ? forced_outer : forced_inner;
				if (f > 0.0f) value = ConstantF32Value(ctx.state, f);
			}
			// KYTY_TESS_INNER_FROM_OUTER=1 (diagnostic, quads): an inner factor store writes the
			// smaller of the two outer factors across that direction (inner 0: outer 1 and 3,
			// inner 1: outer 0 and 2) as already written, instead of the shader's value.
			static const bool inner_from_outer = std::getenv("KYTY_TESS_INNER_FROM_OUTER") != nullptr;
			if (inner_from_outer && forced <= 0.0f && inst.Arg(1).IsImmediate() &&
			    ctx.state.input_info.vertex->tess.domain == 2u && inst.Arg(1).U32() / 4u >= 4u) {
				auto&      st    = ctx.state;
				const auto inner = inst.Arg(1).U32() / 4u - 4u;
				const auto outer_variable =
				    st.tess_variables.at(static_cast<uint32_t>(IR::TessellationAttribute::Factor));
				const auto load_outer = [&](uint32_t i) {
					const auto ptr = st.builder.AllocateId();
					st.builder.AddFunction(spv::OpAccessChain,
					                       TypePointer(st, spv::StorageClassOutput, TypeF32(st)), ptr,
					                       outer_variable, ConstantU32(st, i));
					const auto loaded = st.builder.AllocateId();
					st.builder.AddFunction(spv::OpLoad, TypeF32(st), loaded, ptr);
					return loaded;
				};
				const auto a = load_outer(inner == 0u ? 1u : 0u);
				const auto b = load_outer(inner == 0u ? 3u : 2u);
				const auto lt = st.builder.AllocateId();
				st.builder.AddFunction(spv::OpFOrdLessThan, TypeBool(st), lt, a, b);
				const auto m = st.builder.AllocateId();
				st.builder.AddFunction(spv::OpSelect, TypeF32(st), m, lt, a, b);
				value = m;
			}
			// KYTY_TESS_FACTOR_FIX=<f>: a NaN or non-positive factor becomes f (diagnostic).
			static const float fix = [] {
				const char* env = std::getenv("KYTY_TESS_FACTOR_FIX");
				return env != nullptr ? static_cast<float>(std::atof(env)) : 0.0f;
			}();
			if (fix > 0.0f && forced <= 0.0f) {
				auto&      b        = ctx.state.builder;
				const auto positive = b.AllocateId();
				b.AddFunction(spv::OpFOrdGreaterThan, TypeBool(ctx.state), positive, value,
				              ConstantF32(ctx.state, 0.0f));
				const auto fixed = b.AllocateId();
				b.AddFunction(spv::OpSelect, TypeF32(ctx.state), fixed, positive, value,
				              ConstantF32Value(ctx.state, fix));
				value = fixed;
			}
		}
		ctx.state.builder.AddFunction(spv::OpStore, TessellationPointer(ctx, inst), value);
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
