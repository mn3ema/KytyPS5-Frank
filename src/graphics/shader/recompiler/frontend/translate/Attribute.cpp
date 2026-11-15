#include "common/assert.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <array>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

namespace {

IR::ExportTargetKind ExportTargetKindFromTarget(uint32_t target, uint32_t& index) {
	index = 0;
	switch (target) {
		case 0x08u: return IR::ExportTargetKind::MrtZ;
		case 0x09u: return IR::ExportTargetKind::Null;
		case 0x14u: return IR::ExportTargetKind::Primitive;
		default: break;
	}
	if (target <= 0x07u) {
		index = target;
		return IR::ExportTargetKind::Mrt;
	}
	if (target >= 0x0cu && target <= 0x0fu) {
		index = target - 0x0cu;
		return IR::ExportTargetKind::Position;
	}
	if (target >= 0x20u && target <= 0x3fu) {
		index = target - 0x20u;
		return IR::ExportTargetKind::Parameter;
	}
	return IR::ExportTargetKind::Unknown;
}

} // namespace

IR::ExportFlags Translator::AddExportInfo(const Decoder::Instruction& inst) {
	IR::ExportInfo info;
	info.kind        = ExportTargetKindFromTarget(inst.exp.target, info.index);
	info.target      = inst.exp.target;
	info.en          = inst.exp.en;
	info.done        = inst.exp.done;
	info.compr       = inst.exp.compr;
	info.vm          = inst.exp.vm;
	const auto index = static_cast<uint32_t>(program.export_info.size());
	program.export_info.push_back(info);
	return {.index = index, .pc = inst.pc};
}

void Translator::TranslateEmbeddedFetch(const Decoder::Instruction& inst, uint32_t attribute,
                                        uint32_t component_count,
                                        const ShaderBufferResource& resource) {
	const auto format = Format::GetFormatInfo(resource.Format());
	for (uint32_t component = 0; component < component_count; component++) {
		auto source = Format::FormattedSource {Format::FormattedSourceKind::Memory, component};
		if (inst.formatted && !inst.typed) {
			source = Format::ResolveFormattedSource(
			    format, GetDstSel(resource.DstSelXYZW(), component));
			if (source.kind == Format::FormattedSourceKind::Invalid) {
				EXIT("invalid formatted vertex input %u at pc 0x%08x: component=%u format=%u "
				     "selector=%u descriptor=%08x,%08x,%08x,%08x",
				     attribute, inst.pc, component, static_cast<unsigned>(resource.RawFormat()),
				     static_cast<unsigned>(GetDstSel(resource.DstSelXYZW(), component)),
				     resource.fields[0],
				     resource.fields[1], resource.fields[2], resource.fields[3]);
			}
		}
		IR::Value value;
		if (source.kind == Format::FormattedSourceKind::Memory) {
			value = ir.Emit(IR::ValueOpcode::GetAttribute,
			                {IR::Value(attribute), IR::Value(source.component)});
			auto& required = program.info.vertex_fetch_components[attribute];
			required = static_cast<uint8_t>(std::max<uint32_t>(required, source.component + 1u));
		} else {
			value = IR::Value(Format::FormattedConstantBits(format, source.kind));
		}
		WriteOperand(OffsetOperand(inst.dst, component), value);
	}
}

void Translator::ForceDestination(const Decoder::Instruction& inst,
                                  const std::vector<uint32_t>& values) {
	for (uint32_t component = 0; component < values.size(); component++) {
		WriteRawU32(OffsetOperand(inst.dst, component), IR::U32(IR::Value(values[component])));
	}
}

void Translator::ScaleDestination(const Decoder::Instruction& inst, uint32_t factor_bits) {
	const auto operand = OffsetOperand(inst.dst, 0);
	const auto value   = IR::F32(ReadOperand(operand, IR::Type::F32));
	float      factor  = 1.0f;
	std::memcpy(&factor, &factor_bits, sizeof(factor));
	WriteOperand(operand, ir.Emit(IR::ValueOpcode::FPMul32, {value, IR::Value::F32(factor)}));
}

void Translator::ForceVectorRegisters(uint32_t first, const std::vector<uint32_t>& values) {
	Decoder::Operand reg;
	reg.kind = Decoder::OperandKind::Vgpr;
	reg.reg  = first;
	for (uint32_t component = 0; component < values.size(); component++) {
		WriteRawU32(OffsetOperand(reg, component), IR::U32(IR::Value(values[component])));
	}
}

const ProbePc& ProbePcSetting() {
	static const ProbePc setting = [] {
		ProbePc result;
		const char* value = std::getenv("KYTY_PROBE_PC");
		if (value == nullptr) {
			return result;
		}
		char* end   = nullptr;
		result.hash = std::strtoull(value, &end, 16);
		if (end != nullptr && *end == '@') {
			result.pc = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
		}
		if (end != nullptr && end[0] == ':') {
			end++;
			while (end != nullptr &&
			       (end[0] == 'v' || end[0] == 'u' || end[0] == 's' || end[0] == 'U' || end[0] == 'S' ||
			        end[0] == 'r') &&
			       result.count < 3u) {
				result.kinds[result.count]   = end[0];
				result.vgprs[result.count++] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
				// '+' too: the test chains split their settings on commas.
				if (end != nullptr && (*end == ',' || *end == '+')) {
					end++;
				}
			}
		}
		return result;
	}();
	return setting;
}

const ExportPatchUv& ExportPatchUvSetting() {
	static const ExportPatchUv setting = [] {
		ExportPatchUv result;
		const char*   value = std::getenv("KYTY_EXPORT_PATCH_UV");
		if (value == nullptr) {
			return result;
		}
		char* end   = nullptr;
		result.hash = std::strtoull(value, &end, 16);
		if (end == nullptr || *end != '@') {
			return result;
		}
		result.pc = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
		if (end == nullptr || *end != ':') {
			return result;
		}
		end++;
		for (uint32_t i = 0; i < 4u; i++) {
			if (end == nullptr || (*end != 'v' && *end != 'u')) {
				return result;
			}
			result.as_uint[i] = *end == 'u';
			result.regs[i]    = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			if (end != nullptr && (*end == ',' || *end == '+')) {
				end++;
			}
		}
		if (end == nullptr || *end != ':') {
			return result;
		}
		result.target = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
		result.valid  = true;
		return result;
	}();
	return setting;
}

void Translator::PatchUvCopy(const ExportPatchUv& setting) {
	for (uint32_t i = 0; i < 2u; i++) {
		Decoder::Operand base;
		base.kind = Decoder::OperandKind::Vgpr;
		base.reg  = setting.regs[i];
		Decoder::Operand coord;
		coord.kind = Decoder::OperandKind::Vgpr;
		coord.reg  = setting.regs[i + 2u];
		const auto read = [&](const Decoder::Operand& operand, bool as_uint) {
			if (as_uint) {
				return IR::F32(ir.Emit(IR::ValueOpcode::ConvertF32U32, {ReadRawU32(operand)}));
			}
			return IR::F32(ReadOperand(operand, IR::Type::F32));
		};
		const auto b = read(base, setting.as_uint[i]);
		const auto c = read(coord, setting.as_uint[i + 2u]);
		const auto half = ir.Emit(IR::ValueOpcode::FPMul32, {c, IR::Value::F32(0.5f)});
		Decoder::Operand target;
		target.kind = Decoder::OperandKind::Vgpr;
		target.reg  = PatchUvVgpr + i;
		WriteOperand(target, ir.Emit(IR::ValueOpcode::FPAdd32, {b, half}));
	}
}

void Translator::ProbeCopy(const ProbePc& probe) {
	std::array<IR::U32, 3> values;
	for (uint32_t i = 0; i < 3u; i++) {
		const uint32_t   at = std::min(i, probe.count - 1u);
		Decoder::Operand source;
		// r<n>: scalar register n's raw bits (a float constant, read back with KYTY_PROBE_RECT).
		const bool scalar = probe.kinds[at] == 's' || probe.kinds[at] == 'S' || probe.kinds[at] == 'r';
		source.kind = scalar ? Decoder::OperandKind::Sgpr : Decoder::OperandKind::Vgpr;
		source.reg  = probe.vgprs[at];
		if (scalar && (source.reg == 106u || source.reg == 107u)) {
			source.kind = source.reg == 106u ? Decoder::OperandKind::VccLo : Decoder::OperandKind::VccHi;
			source.reg  = 0;
		}
		values[i]   = ReadRawU32(source);
		if (probe.kinds[at] != 'v' && probe.kinds[at] != 'r') {
			const auto as_float = ir.Emit(IR::ValueOpcode::ConvertF32U32, {values[i]});
			// Lower case: 1/8 per unit; upper case (U/S): one unit is full brightness.
			const float scale =
			    (probe.kinds[at] == 'U' || probe.kinds[at] == 'S') ? 1.0f : 0.125f;
			const auto scaled =
			    ir.Emit(IR::ValueOpcode::FPMul32, {as_float, IR::Value::F32(scale)});
			values[i] = IR::U32(ir.Emit(IR::ValueOpcode::BitCastU32F32, {scaled}));
		}
	}
	for (uint32_t i = 0; i < 3u; i++) {
		Decoder::Operand target;
		target.kind = Decoder::OperandKind::Vgpr;
		target.reg  = ProbeVgpr + i;
		WriteRawU32(target, values[i]);
	}
}

void Translator::V_INTERP_P1_F32() {}

void Translator::V_INTERP_P2_F32(const Decoder::Instruction& inst) {
	const auto value = ir.Emit(IR::ValueOpcode::GetAttribute,
	                           {IR::Value(inst.src1.value), IR::Value(inst.src2.value)});
	WriteOperand(inst.dst, value);
}

void Translator::V_INTERP_MOV_F32(const Decoder::Instruction& inst) {
	if (inst.src0.value >= 3u) {
		EXIT("v_interp_mov_f32 mode %u is reserved at pc 0x%08x", inst.src0.value, inst.pc);
	}
	const auto value = ir.Emit(
	    IR::ValueOpcode::GetInterpolationParameter,
	    {IR::Value(inst.src1.value), IR::Value(inst.src2.value), IR::Value(inst.src0.value)});
	WriteOperand(inst.dst, value);
}

void Translator::EXP(const Decoder::Instruction& inst) {
	uint32_t index = 0;
	if (ExportTargetKindFromTarget(inst.exp.target, index) == IR::ExportTargetKind::Unknown) {
		EXIT("unsupported EXP target 0x%02x at pc 0x%08x", inst.exp.target, inst.pc);
	}
	std::array<IR::Value, 4> components {IR::Value(0u), IR::Value(0u), IR::Value(0u),
	                                     IR::Value(0u)};
	for (uint32_t source = 0; source < std::min(inst.src_count, 4u); source++) {
		components[source] = ReadRawU32(PlainOperand(SourceAt(inst, source)));
	}
	// Diagnostics: KYTY_FORCE_EXPORT=<shader hash>:<target>:<component>:<value> (all hex)
	// exports a constant dword in place of one component, to bisect a wrong draw.
	struct ForcedExport {
		uint64_t hash      = 0;
		uint32_t target    = 0;
		uint32_t component = 0;
		uint32_t value     = 0;
	};
	static const ForcedExport forced = [] {
		ForcedExport result;
		const char*  value = std::getenv("KYTY_FORCE_EXPORT");
		if (value != nullptr) {
			char* end        = nullptr;
			result.hash      = std::strtoull(value, &end, 16);
			result.target    = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			result.component = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			result.value     = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
		}
		return result;
	}();
	if (forced.hash != 0 && forced.hash == program.shader_hash &&
	    forced.target == inst.exp.target && forced.component < 4u) {
		components[forced.component] = IR::Value(forced.value);
	}
	const auto& patch_uv = ExportPatchUvSetting();
	if (patch_uv.valid && patch_uv.hash == program.shader_hash &&
	    patch_uv.target == inst.exp.target) {
		for (uint32_t i = 0; i < 2u; i++) {
			Decoder::Operand source;
			source.kind           = Decoder::OperandKind::Vgpr;
			source.reg            = PatchUvVgpr + i;
			components[i * 2u]    = ReadRawU32(source);
		}
	}
	const auto& probe = ProbePcSetting();
	// Pixel shaders export the probed registers as MRT0; other stages as parameter 0 (the host
	// reads them back with KYTY_PROBE_RECT, which appends each vertex's values).
	const bool  probed = probe.hash != 0 && probe.hash == program.shader_hash &&
	                    (program.stage == ShaderType::Pixel ? inst.exp.target == 0u
	                                                        : inst.exp.target == 0x20u);
	if (probed) {
		for (uint32_t i = 0; i < 3u; i++) {
			Decoder::Operand source;
			source.kind   = Decoder::OperandKind::Vgpr;
			source.reg    = ProbeVgpr + i;
			components[i] = ReadRawU32(source);
		}
		components[3] = IR::Value(program.stage == ShaderType::Pixel ? 0x3f800000u : 0u);
	}
	const auto data = ir.Emit(IR::ValueOpcode::CompositeConstructU32x4,
	                          {components[0], components[1], components[2], components[3]});
	const auto flags = AddExportInfo(inst);
	if (probed) {
		program.export_info[flags.index].compr = false;
		program.export_info[flags.index].en    = 0xfu;
	}
	ir.Emit(IR::ValueOpcode::SetAttribute, {data, ir.GetExec()}, flags);
}

void Translator::EmitInterpolation(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::V_INTERP_P1_F32: V_INTERP_P1_F32(); return;
		case Decoder::Opcode::V_INTERP_P2_F32: V_INTERP_P2_F32(inst); return;
		case Decoder::Opcode::V_INTERP_MOV_F32: V_INTERP_MOV_F32(inst); return;
		default: return FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
