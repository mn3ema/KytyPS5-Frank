#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include "common/assert.h"

#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

// v440 callable split (IR/RtCallable.h). The payload carries every SGPR, VCC_LO, M0, SCC and the
// VGPRs both sides use: a register a callee leaves alone comes back as it went, and SGPR values keep
// their scalar form (a mask register reads back through the lane's bit, lane ids being kept).

void Translator::StorePayloadRegisters(const IR::RtCallInfo& rt) {
	const auto store = [&](uint32_t slot, IR::U32 value) {
		ir.Emit(IR::ValueOpcode::PayloadStore, {IR::Value(slot), value});
	};
	for (uint32_t reg = 0; reg < IR::NumScalarRegs; reg++) {
		store(IR::PayloadSgprSlot + reg, ir.GetScalarReg(static_cast<IR::ScalarReg>(reg)));
	}
	store(IR::PayloadVccSlot, ir.GetVccLo());
	store(IR::PayloadM0Slot, ir.GetM0());
	store(IR::PayloadSccSlot, ir.Select(ir.GetScc(), IR::U32(IR::Value(1u)), IR::U32(IR::Value(0u))));
	for (uint32_t reg = 0; reg < rt.vgpr_count; reg++) {
		store(IR::PayloadVgprSlot + reg, ir.GetVectorReg(static_cast<IR::VectorReg>(reg)));
	}
	// v443c: the rest of a shared payload is zero at each call. A callable's registers above the
	// ray-generation shader's own are its temporaries; before v443 they were not in the payload and
	// read as 0 until written, as every register nothing wrote does. In v443 the callable loads all
	// it reads from the payload, so without this they were whatever the payload's memory held: never
	// written before a lane's first call, a previous callable's leftovers after it. (CutRtPayload
	// drops the stores of slots no callable reads.)
	// KYTY_RT_PAYLOAD_ZERO=0: as v443 (for a comparison).
	static const bool zero_rest = [] {
		const char* value = std::getenv("KYTY_RT_PAYLOAD_ZERO");
		return value == nullptr || value[0] != '0';
	}();
	if (rt.role == IR::RtRole::Raygen && zero_rest) {
		for (uint32_t reg = rt.vgpr_count; reg < rt.payload_vgprs; reg++) {
			store(IR::PayloadVgprSlot + reg, IR::U32(IR::Value(0u)));
		}
	}
}

void Translator::LoadPayloadRegisters(const IR::RtCallInfo& rt, IR::U1 call_exec) {
	const auto load = [&](uint32_t slot) {
		return IR::U32(ir.Emit(IR::ValueOpcode::PayloadLoad, {IR::Value(slot)}));
	};
	// After a call, the scalar state is the callee's: what it wrote reached only the lanes that ran
	// it (their payloads), but a guest SGPR write is the wave's. Every lane takes the first calling
	// lane's scalars (all calling lanes ran the same callee on the same scalars), as the guest wave
	// would see them; with no calling lane the payloads hold what was stored, the same everywhere.
	const auto scalar = [&](uint32_t slot) {
		const auto value = load(slot);
		if (rt.role != IR::RtRole::Raygen) {
			return value;
		}
		return IR::U32(ir.Emit(IR::ValueOpcode::ReadFirstLane, {value, call_exec}));
	};
	for (uint32_t reg = 0; reg < IR::NumScalarRegs; reg++) {
		const auto sgpr = static_cast<IR::ScalarReg>(reg);
		ir.SetScalarReg(sgpr, scalar(IR::PayloadSgprSlot + reg));
		ir.SetScalarMaskTag(sgpr, IR::U1(IR::Value(false)));
	}
	const auto vcc = scalar(IR::PayloadVccSlot);
	ir.SetVccLo(vcc);
	ir.SetVccHi(IR::U32(IR::Value(0u)));
	ir.SetVcc(ThreadBit({vcc, IR::U32(IR::Value(0u))}));
	ir.SetM0(scalar(IR::PayloadM0Slot));
	ir.SetScc(ir.INotEqual(scalar(IR::PayloadSccSlot), IR::U32(IR::Value(0u))));
	for (uint32_t reg = 0; reg < rt.vgpr_count; reg++) {
		ir.SetVectorReg(static_cast<IR::VectorReg>(reg), load(IR::PayloadVgprSlot + reg));
	}
}

void Translator::EmitCallableCall(const IR::RtCallInfo& rt) {
	EXIT_IF(rt.role != IR::RtRole::Raygen || program.wave_size != 32u);
	const auto lo = ir.GetScalarReg(static_cast<IR::ScalarReg>(rt.source_reg));
	const auto hi = ir.GetScalarReg(static_cast<IR::ScalarReg>(rt.source_reg + 1u));
	// The callable index of the target address, from the pipeline's target table (IR/RtCallable.h);
	// an address outside the set calls nothing.
	const auto index = IR::U32(ir.Emit(IR::ValueOpcode::RtCallableIndex, {lo, hi}));
	StorePayloadRegisters(rt);
	const auto call_exec = ir.GetExec();
	ir.Emit(IR::ValueOpcode::ExecuteCallable, {index, call_exec});
	LoadPayloadRegisters(rt, call_exec);
	// s_swappc_b64 writes the return address; nothing reads it after the call in these shaders.
	for (uint32_t half = 0; half < 2u; half++) {
		const auto reg = static_cast<IR::ScalarReg>(rt.return_reg + half);
		ir.SetScalarReg(reg, IR::U32(IR::Value(0u)));
		ir.SetScalarMaskTag(reg, IR::U1(IR::Value(false)));
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
