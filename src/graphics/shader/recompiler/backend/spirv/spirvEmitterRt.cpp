#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include "common/assert.h"

#include <algorithm>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

// v440 callable split (IR/RtCallable.h): the two halves of a ray-tracing shader on Vulkan's
// ray-tracing pipeline. The guest compute dispatch becomes a trace of (threads x groups.x,
// groups.y, groups.z) invocations: 32 consecutive launch ids form one guest wave (subgroups in a
// ray-generation shader are 32 consecutive launch ids on the RTX 5090, rt_probe), the workgroup
// built-ins come from the launch id, and the guest register state crosses each call in the payload.

namespace {

uint32_t LaunchVariable(EmitterState& state, spv::BuiltIn builtin) {
	const auto vector  = state.builder.Type(spv::OpTypeVector, TypeU32(state), 3u);
	const auto pointer = TypePointer(state, spv::StorageClassInput, vector);
	const auto id      = state.builder.DefineGlobalVariable(pointer, spv::StorageClassInput);
	state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationBuiltIn, builtin);
	state.interface_variables.push_back(id);
	return id;
}

uint32_t LaunchComponent(EmitterState& state, uint32_t variable, uint32_t component) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeU32(state)),
	                          pointer, variable, ConstantU32(state, component));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t PayloadElement(EmitterState& state, uint32_t slot) {
	const auto storage = state.program.rt_role == IR::RtRole::Raygen ? spv::StorageClassCallableDataKHR
	                                                                 : spv::StorageClassIncomingCallableDataKHR;
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, TypeU32(state)), pointer,
	                          state.rt_payload_variable, ConstantU32(state, slot));
	return pointer;
}

} // namespace

void DefineRtInterface(EmitterState& state) {
	state.builder.RequireCapability(spv::CapabilityRayTracingKHR);
	state.builder.RequireExtension("SPV_KHR_ray_tracing");
	// v441a: the guest's wave code after a callable call (cross-lane reads, the scalar state
	// broadcast) assumes its lanes run together again: structured merges reconverge.
	if (GetRtMaximalReconvergence()) {
		state.builder.RequireExtension("SPV_KHR_maximal_reconvergence");
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeMaximallyReconvergesKHR);
	}
	const bool raygen  = state.program.rt_role == IR::RtRole::Raygen;
	const auto storage = raygen ? spv::StorageClassCallableDataKHR : spv::StorageClassIncomingCallableDataKHR;
	state.rt_payload_variable = state.builder.DefineGlobalVariable(
	    TypeU32ArrayPointer(state, storage, std::max(state.program.rt_payload_words, 1u)), storage);
	if (raygen) {
		state.builder.AddAnnotation(spv::OpDecorate, state.rt_payload_variable, spv::DecorationLocation, 0u);
	}
	state.builder.AddName(state.rt_payload_variable, "guest_registers");
	state.rt_launch_id   = LaunchVariable(state, spv::BuiltInLaunchIdKHR);
	state.rt_launch_size = LaunchVariable(state, spv::BuiltInLaunchSizeKHR);
}

uint32_t RtPlaced(EmitterState& state, RtRecordSlot kind, uint32_t key, uint32_t value) {
	if (!state.rt_relocate) {
		return ConstantU32(state, value);
	}
	const auto [slot, added] = state.rt_record_slots.try_emplace(
	    std::pair {static_cast<uint32_t>(kind), key}, static_cast<uint32_t>(state.rt_record_words.size()));
	if (added) {
		state.rt_record_words.push_back(value);
	}
	EXIT_IF(state.rt_record_words.at(slot->second) != value);
	if (state.rt_record_variable == 0) {
		state.rt_record_variable = state.builder.AllocateId();
	}
	const auto pointer = state.builder.AllocateId();
	const auto loaded  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassShaderRecordBufferKHR, TypeU32(state)), pointer,
	                          state.rt_record_variable, ConstantU32(state, 0u), ConstantU32(state, slot->second));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), loaded, pointer);
	return loaded;
}

void DefineRtRecord(EmitterState& state) {
	if (state.rt_record_variable == 0) {
		return;
	}
	EXIT_IF(state.rt_record_words.empty() || state.rt_record_words.size() > IR::RtRecordMaxWords);
	const auto array = state.builder.DecoratedType(
	    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, TypeU32(state),
	    ConstantU32(state, static_cast<uint32_t>(state.rt_record_words.size())));
	// (the same struct type as another block of one such array would be; its decorations are the same)
	const auto block = state.builder.DecoratedType(spv::OpTypeStruct,
	                                               {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                                {spv::OpDecorate, {spv::DecorationBlock}}},
	                                               array);
	state.builder.DefineGlobalVariable(state.rt_record_variable,
	                                   TypePointer(state, spv::StorageClassShaderRecordBufferKHR, block),
	                                   spv::StorageClassShaderRecordBufferKHR);
	state.builder.AddName(state.rt_record_variable, "callable_record");
}

// v443e, a callable's loop cap (IR::RtCallableLoopLimit): void report(u32 decisions, u32 pc, u32
// entry_lanes: the subgroup's lanes when the callable started). On the
// decision that reaches the cap the invocation counts itself in the binding table's counters (their
// address is in the shader data, IR/RtCallable.h) and leaves what the host needs to name it: the
// callable's code hash, the guest pc of the loop decision, and how the wave looked there. Nothing
// here depends on the split the callable is in. Defined before the entry point (as get_bda_pointer).
void DefineRtLoopCapReport(EmitterState& state) {
	const auto u32 = TypeU32(state), u64 = TypeScalarU64(state), boolean = TypeBool(state);
	const auto function_type = state.builder.Type(spv::OpTypeFunction, TypeVoid(state), u32, u32, u32);
	state.rt_loop_cap_report_function = state.builder.AllocateId();
	const auto decisions = state.builder.AllocateId(), pc = state.builder.AllocateId();
	const auto entry_lanes = state.builder.AllocateId();
	const auto entry_label = state.builder.AllocateId(), report_label = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddName(state.rt_loop_cap_report_function, "callable_loop_cap_report");
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.rt_loop_cap_report_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, u32, decisions);
	state.builder.AddFunction(spv::OpFunctionParameter, u32, pc);
	state.builder.AddFunction(spv::OpFunctionParameter, u32, entry_lanes);
	EmitLabel(state, entry_label);
	const auto dword = state.program.bindings.dispatch_thread_dword;
	uint32_t   table = ConstantDeviceAddress(state, 0);
	if (dword != IR::PushData::NoStart) {
		table = Binary(state, spv::OpBitwiseOr, u64,
		               Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 3u)),
		               Binary(state, spv::OpShiftLeftLogical, u64,
		                      Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 4u)),
		                      state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
	}
	const auto fire = Binary(state, spv::OpLogicalAnd, boolean,
	                         Binary(state, spv::OpIEqual, boolean, decisions, ConstantU32(state, state.rt_loop_cap)),
	                         Binary(state, spv::OpINotEqual, boolean, table, ConstantDeviceAddress(state, 0)));
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, fire, report_label, merge_label);
	EmitLabel(state, report_label);
	const auto word = [&](uint32_t offset) {
		const auto address = Binary(state, spv::OpIAdd, u64, table,
		                            state.builder.Constant(spv::OpConstant, u64,
		                                                   IR::RtTargetSlots * IR::RtTargetWords * 4u + offset, 0u));
		return Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address);
	};
	const auto atomic = [&](spv::Op op, uint32_t pointer, uint32_t value) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(op, u32, result, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u),
		                          value);
	};
	atomic(spv::OpAtomicExchange, word(32u), ConstantU32(state, state.program.rt_code_hash));
	atomic(spv::OpAtomicExchange, word(36u), pc);
	if (IR::RtAddressTrapReport()) {
		// v443o: the lane and the wave only with KYTY_RT_TRAP_REPORT=1 (this function is inlined at
		// every loop decision of a callable; which callable and which decision is what names it).
		uint32_t lane = ConstantU32(state, UINT32_MAX);
		if (state.subgroup_local_invocation_id_variable != 0) {
			lane = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, u32, lane, state.subgroup_local_invocation_id_variable);
		}
		state.builder.RequireCapability(spv::CapabilityGroupNonUniform);
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
		const auto ballot = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), ConstantBool(state, true));
		const auto lanes = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, u32, lanes, ballot, 0);
		atomic(spv::OpAtomicExchange, word(40u), lane);
		atomic(spv::OpAtomicExchange, word(44u), lanes);
		atomic(spv::OpAtomicExchange, word(48u), LaunchComponent(state, state.rt_launch_id, 0));
		atomic(spv::OpAtomicExchange, word(52u), entry_lanes);
	}
	atomic(spv::OpAtomicIAdd, word(28u), ConstantU32(state, 1u));
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

// v443l, the address trap for the addresses the host writes into a callable split's shader data
// (the launch extents, the binding table, the write bitmap): one that is in the first 16 KB after a
// 4 GiB boundary or is no dword's address is not dereferenced (IR/RtCallable.h; none of these
// buffers is there). True: usable. With the trap off, always.
uint32_t EmitRtAddressUsable(EmitterState& state, uint32_t address) {
	if (!IR::RtAddressTrap()) {
		return ConstantBool(state, true);
	}
	const auto u32 = TypeU32(state), boolean = TypeBool(state);
	const auto low = Unary(state, spv::OpUConvert, u32, address);
	return Binary(state, spv::OpLogicalAnd, boolean,
	              Binary(state, spv::OpUGreaterThanEqual, boolean, low, ConstantU32(state, IR::RtTrapPageBytes)),
	              Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, low, ConstantU32(state, 3u)),
	                     ConstantU32(state, 0u)));
}

// The trap's record in the binding table's counters (when the table's own address is usable):
// `site` 0 the guest address table (get_bda_pointer), 1 the launch extents, 3 the write bitmap.
void EmitRtAddressTrapped(EmitterState& state, uint32_t address, uint32_t site, uint32_t guest) {
	const auto dword = state.program.bindings.dispatch_thread_dword;
	if (dword == IR::PushData::NoStart) {
		return;
	}
	const auto u32 = TypeU32(state), u64 = TypeScalarU64(state);
	const auto table = Binary(state, spv::OpBitwiseOr, u64,
	                          Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 3u)),
	                          Binary(state, spv::OpShiftLeftLogical, u64,
	                                 Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 4u)),
	                                 state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
	EmitIfCondition(state, EmitRtAddressUsable(state, table), [&]() {
		const auto atomic = [&](spv::Op op, uint32_t word_offset, uint32_t value) {
			const auto at = Binary(state, spv::OpIAdd, u64, table,
			                       state.builder.Constant(spv::OpConstant, u64,
			                                              IR::RtTargetSlots * IR::RtTargetWords * 4u + word_offset, 0u));
			const auto pointer = Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), at);
			const auto result  = state.builder.AllocateId();
			state.builder.AddFunction(op, u32, result, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u),
			                          value);
		};
		const auto high_of = [&](uint32_t value) {
			return Unary(state, spv::OpUConvert, u32,
			             Binary(state, spv::OpShiftRightLogical, u64, value,
			                    state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
		};
		atomic(spv::OpAtomicExchange, 80u, Unary(state, spv::OpUConvert, u32, address));
		atomic(spv::OpAtomicExchange, 84u, high_of(address));
		atomic(spv::OpAtomicExchange, 88u, guest != 0 ? Unary(state, spv::OpUConvert, u32, guest) : ConstantU32(state, 0u));
		atomic(spv::OpAtomicExchange, 92u, guest != 0 ? high_of(guest) : ConstantU32(state, 0u));
		atomic(spv::OpAtomicExchange, 96u,
		       ConstantU32(state, state.program.rt_role == IR::RtRole::Callable
		                              ? state.program.rt_code_hash
		                              : static_cast<uint32_t>(state.program.shader_hash)));
		atomic(spv::OpAtomicExchange, 100u, ConstantU32(state, site));
		atomic(spv::OpAtomicIAdd, 76u, ConstantU32(state, 1u));
	});
}

bool RtTracePc(const EmitterState& state) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_RT_TRACE_PC");
		return value != nullptr && value[0] == '1';
	}();
	return enabled && state.program.rt_role == IR::RtRole::Callable && !state.program.dispatcher_fallback &&
	       state.program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
}

void EmitRtTraceMark(EmitterState& state, RtTraceMark mark, uint32_t pc) {
	const auto u32 = TypeU32(state), u64 = TypeScalarU64(state);
	const auto dword = state.program.bindings.dispatch_thread_dword;
	const auto table = Binary(state, spv::OpBitwiseOr, u64,
	                          Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 3u)),
	                          Binary(state, spv::OpShiftLeftLogical, u64,
	                                 Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 4u)),
	                                 state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
	const auto atomic = [&](spv::Op op, uint32_t offset, uint32_t value) {
		const auto address = Binary(state, spv::OpIAdd, u64, table,
		                            state.builder.Constant(spv::OpConstant, u64,
		                                                   IR::RtTargetSlots * IR::RtTargetWords * 4u + offset, 0u));
		const auto pointer = Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address);
		const auto result  = state.builder.AllocateId();
		state.builder.AddFunction(op, u32, result, pointer, ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0u),
		                          value);
	};
	switch (mark) {
		case RtTraceMark::Start: atomic(spv::OpAtomicIAdd, 56u, ConstantU32(state, 1u)); break;
		case RtTraceMark::End: atomic(spv::OpAtomicIAdd, 60u, ConstantU32(state, 1u)); break;
		case RtTraceMark::Block:
			atomic(spv::OpAtomicUMax, 64u, ConstantU32(state, pc));
			atomic(spv::OpAtomicExchange, 68u, ConstantU32(state, pc));
			atomic(spv::OpAtomicExchange, 72u, ConstantU32(state, state.program.rt_code_hash));
			break;
	}
}

uint32_t EmitRtWorkgroupInput(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	const auto* cs = state.input_info.compute;
	if (cs == nullptr) {
		return 0;
	}
	// The launch is (threads x groups, 1, 1): one-dimensional, so that 32 consecutive launch ids
	// share a subgroup (rt_probe: 2D launches pack 8x4 tiles). The group counts come from the
	// dispatch's thread extents in the shader data (BindingLayout: dispatch_thread_dword).
	const uint32_t t[3]    = {std::max(cs->threads_num[0], 1u), std::max(cs->threads_num[1], 1u),
	                          std::max(cs->threads_num[2], 1u)};
	const uint32_t threads = t[0] * t[1] * t[2];
	const auto u32 = [&](uint32_t value) { return ConstantU32(state, value); };
	const auto id  = [&]() { return LaunchComponent(state, state.rt_launch_id, 0); };
	const auto local_index = [&]() { return EmitBinaryU32(state, spv::OpUMod, id(), u32(threads)); };
	const auto count = [&](uint32_t axis) {
		// (at least 1: extents that were not read, the address trap, are 0, and these divide)
		return EmitGlsl<GLSLstd450UMax, IR::Type::U32>(
		    state, EmitBinaryU32(state, spv::OpUDiv, EmitDispatchThreadCount(state, axis), u32(t[axis])), u32(1u));
	};
	const auto group = [&](uint32_t axis) {
		const auto linear = EmitBinaryU32(state, spv::OpUDiv, id(), u32(threads));
		if (axis == 0u) {
			return EmitBinaryU32(state, spv::OpUMod, linear, count(0));
		}
		const auto above_x = EmitBinaryU32(state, spv::OpUDiv, linear, count(0));
		if (axis == 1u) {
			return EmitBinaryU32(state, spv::OpUMod, above_x, count(1));
		}
		return EmitBinaryU32(state, spv::OpUDiv, above_x, count(1));
	};
	const auto local = [&](uint32_t axis) {
		const uint32_t divisor = axis == 0u ? 1u : axis == 1u ? t[0] : t[0] * t[1];
		return EmitBinaryU32(state, spv::OpUMod, EmitBinaryU32(state, spv::OpUDiv, local_index(), u32(divisor)),
		                     u32(t[axis]));
	};
	switch (kind) {
		case IR::StageInputKind::LocalInvocationIndex: return local_index();
		case IR::StageInputKind::LocalInvocationId: return local(component);
		case IR::StageInputKind::WorkgroupId: return group(component);
		case IR::StageInputKind::NumWorkgroups: return count(component);
		case IR::StageInputKind::GlobalInvocationId:
			return EmitAddU32(state, EmitBinaryU32(state, spv::OpIMul, group(component), u32(t[component])),
			                  local(component));
		default: return 0;
	}
}

uint32_t EmitPayloadLoad(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.rt_payload_variable == 0 || !inst.Arg(0).IsImmediate()) {
		ctx.Fail(inst, "payload load outside a callable split");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, PayloadElement(state, inst.Arg(0).U32()));
	return value;
}

void EmitPayloadStore(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.rt_payload_variable == 0 || !inst.Arg(0).IsImmediate()) {
		ctx.Fail(inst, "payload store outside a callable split");
	}
	state.builder.AddFunction(spv::OpStore, PayloadElement(state, inst.Arg(0).U32()), ctx.Arg(inst, 1));
}

void EmitExecuteCallable(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.program.rt_role != IR::RtRole::Raygen || state.rt_payload_variable == 0) {
		ctx.Fail(inst, "callable call outside a ray-generation shader");
	}
	const auto index = ctx.Arg(inst, 0);
	// Only lanes the guest has active, and only for an address in the target set.
	const auto call = Binary(state, spv::OpLogicalAnd, TypeBool(state), ctx.Arg(inst, 1),
	                               Binary(state, spv::OpINotEqual, TypeBool(state), index, ConstantU32(state, UINT32_MAX)));
	// KYTY_RT_NO_CALLS=1 (diagnostics): the calls are left out.
	static const bool no_calls = [] {
		const char* value = std::getenv("KYTY_RT_NO_CALLS");
		return value != nullptr && value[0] == '1';
	}();
	// KYTY_RT_COUNT=1 (diagnostics): per lane at the call site, three counters after the target
	// table (IR/RtCallable.h): +0 lanes active at the call, +4 of them with an address outside the
	// target set, +8 lanes that call. The host prints them ([rt-count]).
	static const bool count = [] {
		const char* value = std::getenv("KYTY_RT_COUNT");
		return value != nullptr && value[0] == '1';
	}();
	const bool counting = count && state.program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	const auto u32 = TypeU32(state), u64 = TypeScalarU64(state), boolean = TypeBool(state);
	uint32_t   table = 0, table_ok = 0;
	// An atomic add to the counter word at `offset` + 4 * `index` (index 0: none).
	const auto bump = [&](uint32_t offset, uint32_t index_id = 0u, uint32_t by = 1u) {
		auto address = Binary(state, spv::OpIAdd, u64, table,
		                      state.builder.Constant(spv::OpConstant, u64,
		                                             IR::RtTargetSlots * IR::RtTargetWords * 4u + offset, 0u));
		if (index_id != 0u) {
			address = Binary(state, spv::OpIAdd, u64, address,
			                 Unary(state, spv::OpUConvert, u64,
			                       Binary(state, spv::OpShiftLeftLogical, u32, index_id, ConstantU32(state, 2u))));
		}
		const auto pointer = Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address);
		const auto result  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, u32, result, pointer, ConstantU32(state, 1u),
		                          ConstantU32(state, 0u), ConstantU32(state, by));
	};
	if (counting) {
		const auto dword = state.program.bindings.dispatch_thread_dword;
		table = Binary(state, spv::OpBitwiseOr, u64,
		               Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 3u)),
		               Binary(state, spv::OpShiftLeftLogical, u64,
		                      Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 4u)),
		                      state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
		table_ok           = EmitRtAddressUsable(state, table);
		const auto active  = Binary(state, spv::OpLogicalAnd, boolean, ctx.Arg(inst, 1), table_ok);
		const auto unknown = Binary(state, spv::OpLogicalAnd, boolean, active,
		                            Binary(state, spv::OpIEqual, boolean, index, ConstantU32(state, UINT32_MAX)));
		EmitIfCondition(state, active, [&]() { bump(0u); });
		EmitIfCondition(state, unknown, [&]() { bump(4u); });
		EmitIfCondition(state, Binary(state, spv::OpLogicalAnd, boolean, call, table_ok), [&]() { bump(8u); });
	}
	if (no_calls) {
		return;
	}
	EmitIfCondition(state, call, [&]() {
		// v443c: the lanes inside each callable (in on the way in, out on the way back) and the
		// lanes back from their call, so a trace that never ends shows which callable holds them.
		if (counting) {
			EmitIfCondition(state, table_ok, [&]() { bump(IR::RtCounterWords * 4u, index); });
		}
		state.builder.AddFunction(spv::OpExecuteCallableKHR, index, state.rt_payload_variable);
		if (counting) {
			EmitIfCondition(state, table_ok, [&]() {
				bump(IR::RtCounterWords * 4u, index, UINT32_MAX);
				bump(24u);
			});
		}
	});
	// v443: a lane whose call target the pipeline does not have notes it, and its wave then leaves
	// every loop at the loop's next exit decision (EmitStructuredTerminator) and runs to its end. The
	// target is missing while a generation with it is being built (the first one of a session, or an
	// older one that runs meanwhile). Such a lane used to go on as if the call had returned, with
	// nothing of what the callee does to its state, inside the guest's loops around the call.
	if (state.rt_missing_call_variable != 0) {
		const auto missing = Binary(state, spv::OpLogicalAnd, TypeBool(state), ctx.Arg(inst, 1),
		                            Binary(state, spv::OpIEqual, TypeBool(state), index, ConstantU32(state, UINT32_MAX)));
		EmitIfCondition(state, missing, [&]() {
			state.builder.AddFunction(spv::OpStore, state.rt_missing_call_variable, ConstantU32(state, 1u));
		});
	}
}

uint32_t EmitRtCallableIndex(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.program.rt_role != IR::RtRole::Raygen ||
	    state.program.bindings.dispatch_thread_dword == IR::PushData::NoStart) {
		ctx.Fail(inst, "callable index outside a ray-generation shader");
	}
	const auto u32 = TypeU32(state), u64 = TypeScalarU64(state), boolean = TypeBool(state);
	const auto c   = [&](uint32_t value) { return ConstantU32(state, value); };
	const auto low = ctx.Arg(inst, 0), high = ctx.Arg(inst, 1);
	const auto dword = state.program.bindings.dispatch_thread_dword;
	const auto table = Binary(state, spv::OpBitwiseOr, u64,
	                          Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 3u)),
	                          Binary(state, spv::OpShiftLeftLogical, u64,
	                                 Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword + 4u)),
	                                 state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
	const auto word = [&](uint32_t slot, uint32_t index) {
		const auto offset  = Binary(state, spv::OpIAdd, u32, Binary(state, spv::OpIMul, u32, slot, c(IR::RtTargetWords * 4u)),
		                            c(index * 4u));
		const auto address = Binary(state, spv::OpIAdd, u64, table, Unary(state, spv::OpUConvert, u64, offset));
		const auto pointer = Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address);
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, u32, value, pointer, spv::MemoryAccessAlignedMask, 4u);
		return value;
	};
	// RtTargetHash, in SPIR-V.
	auto h = Binary(state, spv::OpBitwiseXor, u32, Binary(state, spv::OpShiftRightLogical, u32, low, c(2u)),
	                Binary(state, spv::OpIMul, u32, high, c(0x9e3779b1u)));
	h = Binary(state, spv::OpBitwiseXor, u32, h, Binary(state, spv::OpShiftRightLogical, u32, h, c(15u)));
	h = Binary(state, spv::OpIMul, u32, h, c(0x85ebca6bu));
	h = Binary(state, spv::OpBitwiseXor, u32, h, Binary(state, spv::OpShiftRightLogical, u32, h, c(13u)));
	// v443l: a table address that is none (EmitRtAddressUsable) finds no target: no call.
	return EmitValueOrDefaultIfCondition(state, EmitRtAddressUsable(state, table), u32, c(UINT32_MAX), [&]() {
		uint32_t result = c(UINT32_MAX);
		for (uint32_t probe = 0; probe < IR::RtTargetProbes; probe++) {
			const auto slot  = Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpIAdd, u32, h, c(probe)),
			                          c(IR::RtTargetSlots - 1u));
			const auto match = Binary(state, spv::OpLogicalAnd, boolean,
			                          Binary(state, spv::OpIEqual, boolean, word(slot, 0u), low),
			                          Binary(state, spv::OpIEqual, boolean, word(slot, 1u), high));
			const auto next = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, u32, next, match, word(slot, 2u), result);
			result = next;
		}
		return result;
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
