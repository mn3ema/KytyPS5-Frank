#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

void Translator::FailMissingTranslation(const Decoder::Instruction& inst) {
	// A non-fatal compile gives up on the shader instead: its draws or dispatches are skipped.
	// Every missing translation is named, so one log lists all a shader needs.
	if (TranslationNonFatalFlag()) {
		LOGF("shader translation: opcode %s at pc 0x%08x has no IR translation\n",
		     Decoder::InstructionToString(inst).c_str(), inst.pc);
		Profiler::Message(Profiler::MessageWarning,
		                  "Shader translation: opcode %s at pc 0x%08x has no IR translation",
		                  Decoder::InstructionToString(inst).c_str(), inst.pc);
		TranslationUnsupportedFlag() = true;
		return;
	}
	EXIT("opcode %s at pc 0x%08x has no IR translation",
	     Decoder::InstructionToString(inst).c_str(), inst.pc);
}

// v446a: EXEC writes that can turn on lanes the current EXEC lacks (IR::Program::exec_widens_lanes).
// Restoring EXEC from a saved mask (S_MOV/S_OR from an SGPR, S_OR_SAVEEXEC) is the join of a
// structured region and is not counted.
void Translator::NoteExecWidening(const Decoder::Instruction& inst) {
	using O = Decoder::Opcode;
	const auto is_exec  = [](const Decoder::Operand& operand) {
		return operand.kind == Decoder::OperandKind::ExecLo || operand.kind == Decoder::OperandKind::ExecHi;
	};
	const auto constant = [](const Decoder::Operand& operand) {
		return (operand.kind == Decoder::OperandKind::LiteralConstant ||
		        operand.kind == Decoder::OperandKind::IntegerInlineConstant ||
		        operand.kind == Decoder::OperandKind::FloatInlineConstant) &&
		       operand.value != 0u;
	};
	const auto mark = [&] {
		program.exec_widens_lanes = true;
		if (auto* current = ir.CurrentBlock(); current != nullptr) {
			current->exec_widens = true;
		}
		static const bool log = std::getenv("KYTY_WAVE_UNIFORM_LOG") != nullptr;
		if (log) {
			LOGF("[wave-branches] %016llx: %s at pc 0x%08x turns EXEC lanes on\n",
			     static_cast<unsigned long long>(program.shader_hash), Decoder::InstructionToString(inst).c_str(),
			     inst.pc);
		}
	};
	switch (inst.opcode) {
		case O::S_ORN2_SAVEEXEC_B32:
		case O::S_ORN2_SAVEEXEC_B64: mark(); return;
		default: break;
	}
	if (!is_exec(inst.dst)) {
		return;
	}
	switch (inst.opcode) {
		case O::S_ORN2_B32:
		case O::S_ORN2_B64:
		case O::S_NOT_B32:
		case O::S_NOT_B64:
		case O::S_WQM_B32:
		case O::S_WQM_B64:
		case O::S_NAND_B32:
		case O::S_NAND_B64:
		case O::S_NOR_B32:
		case O::S_NOR_B64:
		case O::S_XNOR_B32:
		case O::S_XNOR_B64: mark(); return;
		default: break;
	}
	switch (inst.opcode) {
		case O::S_MOV_B32:
		case O::S_MOV_B64:
		case O::S_CMOV_B64:
		case O::S_OR_B32:
		case O::S_OR_B64:
		case O::S_XOR_B32:
		case O::S_XOR_B64:
			// S_MOV exec, -1 and the like (S_AND with a constant cannot add lanes).
			if (constant(inst.src0) || (inst.family == Decoder::Family::SOP2 && constant(inst.src1))) {
				mark();
			}
			return;
		default: return;
	}
}

void Translator::TranslateInstruction(const Decoder::Instruction& inst) {
	current_opcode = inst.opcode;
	current_pc     = inst.pc;

	switch (inst.opcode) {
		case Decoder::Opcode::UNKNOWN:
		case Decoder::Opcode::COUNT:
			if (TranslationNonFatalFlag()) {
				LOGF("shader translation: decoded opcode at pc 0x%08x has no IR translation\n",
				     inst.pc);
				TranslationUnsupportedFlag() = true;
				return;
			}
			EXIT("decoded opcode has no IR translation at pc 0x%08x", inst.pc);
		case Decoder::Opcode::UNSUPPORTED:
			EXIT("unsupported decoded instruction: %s", Decoder::InstructionToString(inst).c_str());
		default: break;
	}

	if (inst.family == Decoder::Family::SOP1 || inst.family == Decoder::Family::SOP2) {
		NoteExecWidening(inst);
	}

	switch (inst.family) {
		case Decoder::Family::SOP1:
		case Decoder::Family::SOP2:
		case Decoder::Family::SOPK:
		case Decoder::Family::SOPC:
		case Decoder::Family::SOPP: return EmitScalar(inst);
		case Decoder::Family::VOP1:
		case Decoder::Family::VOP2:
		case Decoder::Family::VOP3:
		case Decoder::Family::VOP3P:
		case Decoder::Family::VOPC: return EmitVector(inst);
		case Decoder::Family::SMEM:
		case Decoder::Family::MUBUF:
		case Decoder::Family::MTBUF:
		case Decoder::Family::FLAT:
		case Decoder::Family::DS:
		case Decoder::Family::MIMG: return EmitMemory(inst);
		case Decoder::Family::VINTRP: return EmitInterpolation(inst);
		case Decoder::Family::EXP: return EXP(inst);
		default: return FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
