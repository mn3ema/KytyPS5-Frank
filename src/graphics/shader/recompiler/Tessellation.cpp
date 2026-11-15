#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <array>
#include <optional>
#include <unordered_map>
#include <utility>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#define TESS_LOG(...) do { LOGF(__VA_ARGS__); std::fprintf(stderr, "[tess] " __VA_ARGS__); } while (0)

namespace Libs::Graphics::ShaderRecompiler {
namespace {

struct TessellationAddress {
	enum class Kind { Unknown, Affine, PackedControlPoint };
	Kind     kind        = Kind::Unknown;
	uint32_t coefficient = 0;
	uint32_t constant    = 0;
};

std::optional<uint32_t> ReflectTessellationStride(const Decoder::Program& program, bool local,
                                                  uint32_t control_points,
                                                  uint32_t input_stride) {
	using namespace Decoder;
	using Address = TessellationAddress;
	using Kind    = Address::Kind;
	std::array<Address, IR::NumVectorRegs> registers {};
	registers[local ? 3u : 1u] =
	    local ? Address {Kind::Affine, 1u, 0u} : Address {Kind::PackedControlPoint};
	const auto constant = [](uint32_t value) { return Address {Kind::Affine, 0u, value}; };
	const auto read     = [&](const Operand& operand) -> Address {
		if (operand.absolute || operand.negate || operand.sdwa_sext || operand.op_sel ||
		    operand.op_sel_hi || operand.negate_hi || operand.dpp)
			return {};
		Address value;
		if (operand.kind == OperandKind::IntegerInlineConstant ||
		    operand.kind == OperandKind::LiteralConstant) {
			value = constant(operand.value);
		} else if (operand.kind == OperandKind::Vgpr) {
			value = registers.at(operand.reg);
		}
		// The native packed HS ID's low byte is the patch ordinal. The logical
		// interface addresses one patch, as in the frontend's v1 initialization.
		if (value.kind == Kind::PackedControlPoint && operand.sdwa_sel == 0u) return constant(0u);
		return operand.sdwa_sel == 6u ? value : Address {};
	};
	const auto affine = [](uint64_t coefficient, uint64_t offset) {
		return coefficient <= UINT32_MAX && offset <= UINT32_MAX
		           ? Address {Kind::Affine, static_cast<uint32_t>(coefficient),
		                      static_cast<uint32_t>(offset)}
		           : Address {};
	};
	const auto add = [&](Address lhs, Address rhs) {
		return lhs.kind == Kind::Affine && rhs.kind == Kind::Affine
		           ? affine(uint64_t {lhs.coefficient} + rhs.coefficient,
		                    uint64_t {lhs.constant} + rhs.constant)
		           : Address {};
	};
	const auto multiply = [&](Address lhs, Address rhs) {
		if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0u) ||
		    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0u))
			return constant(0u);
		if (lhs.kind != Kind::Affine || rhs.kind != Kind::Affine) return Address {};
		if (rhs.coefficient != 0u) std::swap(lhs, rhs);
		return rhs.coefficient == 0u ? affine(uint64_t {lhs.coefficient} * rhs.constant,
		                                      uint64_t {lhs.constant} * rhs.constant)
		                             : Address {};
	};
	const auto low24 = [&](Address value) {
		if (value.kind == Kind::Affine && value.coefficient == 0u)
			return constant(value.constant & 0xffffffu);
		return value.kind == Kind::Affine &&
		               uint64_t {value.coefficient} * (control_points - 1u) + value.constant <=
		                   0xffffffu
		           ? value
		           : Address {};
	};
	uint32_t stride = 0;
	for (const auto& inst: program.instructions) {
		// Stage exits preserve the active path's definitions. An internal join
		// would require merging register definitions.
		// KYTY_TESS_LINEAR=1 (default): scan through internal branches linearly; every ring
		// access must still have an affine address and all of them must agree on one stride, so
		// a definition mixed up across paths shows as a rejection. The terrain HS
		// a9f31c39ad53f06c computes its ring addresses before its first internal branch.
		static const bool linear = [] {
			const char* value = std::getenv("KYTY_TESS_LINEAR");
			return value == nullptr || value[0] != '0';
		}();
		if (!linear && IsDirectBranch(inst.opcode) &&
		    inst.branch_target != program.instructions.back().pc) {
			TESS_LOG("%s tessellation program has an internal branch at pc 0x%08x\n",
			     local ? "LS" : "HS", inst.pc);
			return std::nullopt;
		}
		const bool local_store =
		    inst.opcode == Opcode::DS_WRITE_B32 || inst.opcode == Opcode::DS_WRITE2_B32;
		const bool buffer_store = inst.opcode == Opcode::BUFFER_STORE_DWORD ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX2 ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX3 ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX4;
		const bool control_store =
		    !local && buffer_store && inst.src2.kind == OperandKind::Sgpr && inst.src2.reg == 2u;
		const bool control_read =
		    !local && (inst.opcode == Opcode::DS_READ_B32 || inst.opcode == Opcode::DS_READ2_B32);
		if ((local && local_store) || control_store || control_read) {
			const auto address = read(inst.src0);
			if (address.kind != Kind::Affine) {
				TESS_LOG("%s tessellation address is not affine at pc 0x%08x\n", local ? "LS" : "HS",
				     inst.pc);
				return std::nullopt;
			}
			if (address.coefficient != 0u) {
				const auto expected = control_read ? input_stride : stride;
				if ((address.coefficient & 3u) != 0u ||
				    (expected != 0u && expected != address.coefficient)) {
					TESS_LOG("%s tessellation stride %u at pc 0x%08x disagrees with %u\n",
					     local ? "LS" : "HS", address.coefficient, inst.pc, expected);
					return std::nullopt;
				}
				if (!control_read) stride = address.coefficient;
			}
		}
		// Store vdata is a source, despite occupying the decoder's dst field.
		if (local_store || buffer_store || inst.dst.kind != OperandKind::Vgpr) continue;
		const auto lhs   = read(inst.src0);
		const auto rhs   = read(inst.src1);
		const auto third = read(inst.src2);
		Address    value;
		switch (inst.opcode) {
			case Opcode::V_BFE_U32:
				if (lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
				    rhs.coefficient == 0u && third.kind == Kind::Affine &&
				    third.coefficient == 0u) {
					if (rhs.constant == 8u && third.constant == 5u) value = {Kind::Affine, 1u, 0u};
					if (rhs.constant == 0u && third.constant == 8u) value = constant(0u);
				}
				break;
			case Opcode::V_AND_B32:
				// The patch ordinal (low byte of the packed HS ID) as an AND instead of a BFE
				// (Wolverine's terrain HS a9f31c39ad53f06c): one patch, so 0.
				if ((lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
				     rhs.coefficient == 0u && rhs.constant == 0xffu) ||
				    (rhs.kind == Kind::PackedControlPoint && lhs.kind == Kind::Affine &&
				     lhs.coefficient == 0u && lhs.constant == 0xffu))
					value = constant(0u);
				break;
			case Opcode::V_ADD_NC_U32: value = add(lhs, rhs); break;
			case Opcode::V_MUL_U32_U24: value = multiply(low24(lhs), low24(rhs)); break;
			case Opcode::V_MAD_U32_U24: value = add(multiply(low24(lhs), low24(rhs)), third); break;
			case Opcode::V_LSHL_ADD_U32:
				if (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant < 32u)
					value = add(multiply(lhs, constant(1u << rhs.constant)), third);
				break;
			case Opcode::V_LSHLREV_B32:
				if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant < 32u)
					value = multiply(rhs, constant(1u << lhs.constant));
				break;
			case Opcode::V_SUB_NC_U32:
				if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine &&
				    lhs.coefficient >= rhs.coefficient && lhs.constant >= rhs.constant)
					value = affine(lhs.coefficient - rhs.coefficient, lhs.constant - rhs.constant);
				break;
			default: break;
		}
		if (inst.dst.sdwa_sel != 6u || inst.dst.op_sel || inst.dst.omod || inst.dst.clamp ||
		    inst.dst.dpp)
			value = {};
		for (uint32_t index = 0;
		     index < std::max(inst.data_dwords, 1u) && inst.dst.reg + index < registers.size(); index++) {
			registers[inst.dst.reg + index] = {};
		}
		registers.at(inst.dst.reg) = value;
	}
	if (stride == 0u) {
		TESS_LOG("%s tessellation program has no attribute stride\n", local ? "LS" : "HS");
		return std::nullopt;
	}
	return stride;
}

const IR::Inst* TessellationBufferBase(const IR::Inst& inst) {
	if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::None || inst.NumArgs() <= 3u)
		return nullptr;
	const auto* base = inst.Arg(3).Resolve().TryInstruction();
	return base != nullptr && base->GetOpcode() == IR::ValueOpcode::TessellationBase ? base
	                                                                                 : nullptr;
}

IR::Value ActiveAddress(IR::Block& block, IR::Block::iterator before, IR::Value value,
                        IR::Value predicate, std::unordered_map<IR::Inst*, IR::Value>& resolved) {
	using namespace IR;
	value        = value.Resolve();
	auto* source = value.TryInstruction();
	if (source == nullptr) return value;
	if (const auto found = resolved.find(source); found != resolved.end()) return found->second;
	if (source->GetOpcode() == ValueOpcode::SelectU32) {
		return source->Arg(0).Resolve() == predicate.Resolve()
		           ? ActiveAddress(block, before, source->Arg(1), predicate, resolved)
		           : value;
	}
	if (source->GetOpcode() != ValueOpcode::IAdd32 && source->GetOpcode() != ValueOpcode::ISub32)
		return value;
	const auto lhs = ActiveAddress(block, before, source->Arg(0), predicate, resolved);
	const auto rhs = ActiveAddress(block, before, source->Arg(1), predicate, resolved);
	if (lhs != source->Arg(0).Resolve() || rhs != source->Arg(1).Resolve()) {
		auto copy = block.PrependNewInst(before, source->GetOpcode(), {lhs, rhs},
		                                 source->Flags<uint64_t>());
		value     = Value(&*copy);
	}
	resolved.emplace(source, value);
	return value;
}

// The constant an address takes on the lanes a predicated access runs on, or nothing. Exec-masked
// register writes reach an address as SelectU32(predicate, active, inactive), and definitions
// made before internal branches as phis: arms that agree, or the arm the access's own predicate
// selects, decide it. Phi operands that lead back to the phi being evaluated are skipped.
struct ActiveAddressEvaluator {
	IR::Value                                        predicate;
	std::unordered_map<const IR::Inst*, std::optional<uint32_t>> memo;
	std::unordered_map<const IR::Inst*, bool>        active_phis;
	static constexpr uint32_t                        Cycle = 0xffffffffu;

	// Cycle marks an operand that only loops back to a phi in progress.
	std::optional<uint32_t> Evaluate(const IR::Value& value, uint32_t depth = 0) {
		using namespace IR;
		const auto resolved = value.Resolve();
		if (resolved.IsImmediate()) {
			return resolved.GetType() == Type::U32 ? std::optional<uint32_t>(resolved.U32())
			                                       : std::nullopt;
		}
		const auto* source = resolved.TryInstruction();
		if (source == nullptr || depth > 64u) {
			return std::nullopt;
		}
		if (const auto found = memo.find(source); found != memo.end()) {
			return found->second;
		}
		if (active_phis.contains(source)) {
			return Cycle;
		}
		const auto result = Compute(*source, depth);
		if (!result || *result != Cycle) {
			memo[source] = result;
		}
		return result;
	}

	std::optional<uint32_t> Compute(const IR::Inst& source, uint32_t depth) {
		using namespace IR;
		const auto arg = [&](size_t index) { return Evaluate(source.Arg(index), depth + 1u); };
		const auto plain = [&](size_t index) -> std::optional<uint32_t> {
			const auto value = arg(index);
			return value && *value != Cycle ? value : std::nullopt;
		};
		switch (source.GetOpcode()) {
			case ValueOpcode::Phi: {
				active_phis[&source] = true;
				std::optional<uint32_t> agreed;
				bool                    ok = true;
				for (size_t index = 0; index < source.NumArgs() && ok; index++) {
					const auto value = arg(index);
					if (value && *value == Cycle) continue;
					if (!value || (agreed && *agreed != *value)) {
						ok = false;
					} else {
						agreed = value;
					}
				}
				active_phis.erase(&source);
				return ok && agreed ? agreed : std::nullopt;
			}
			case ValueOpcode::SelectU32: {
				if (source.Arg(0).Resolve() == predicate.Resolve()) {
					return plain(1);
				}
				const auto active   = plain(1);
				const auto inactive = plain(2);
				return active && inactive && *active == *inactive ? active : std::nullopt;
			}
			case ValueOpcode::BitFieldUExtract: {
				const auto base   = plain(0);
				const auto offset = plain(1);
				const auto count  = plain(2);
				if (!base || !offset || !count || *offset >= 32u || *count == 0u) {
					return std::nullopt;
				}
				const auto shifted = *base >> *offset;
				return *count >= 32u ? shifted : shifted & ((1u << *count) - 1u);
			}
			case ValueOpcode::IAdd32:
			case ValueOpcode::ISub32:
			case ValueOpcode::IMul32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftRightLogical32: {
				const auto lhs = plain(0);
				const auto rhs = plain(1);
				if (!lhs || !rhs) {
					return std::nullopt;
				}
				switch (source.GetOpcode()) {
					case ValueOpcode::IAdd32: return *lhs + *rhs;
					case ValueOpcode::ISub32: return *lhs - *rhs;
					case ValueOpcode::IMul32: return *lhs * *rhs;
					case ValueOpcode::BitwiseAnd32: return *lhs & *rhs;
					case ValueOpcode::BitwiseOr32: return *lhs | *rhs;
					case ValueOpcode::ShiftLeftLogical32:
						return *rhs < 32u ? std::optional<uint32_t>(*lhs << *rhs) : std::nullopt;
					default:
						return *rhs < 32u ? std::optional<uint32_t>(*lhs >> *rhs) : std::nullopt;
				}
			}
			default: return std::nullopt;
		}
	}
};

std::optional<uint32_t> ConstantActiveAddress(const IR::Value& value, const IR::Value& predicate) {
	ActiveAddressEvaluator evaluator;
	evaluator.predicate = predicate;
	const auto result   = evaluator.Evaluate(value);
	return result && *result != ActiveAddressEvaluator::Cycle ? result : std::nullopt;
}

} // namespace

bool AnalyzeTessellationPrograms(std::span<const uint32_t> local, std::span<const uint32_t> control,
                                 ShaderTessellationInputInfo& info) {
	const auto       local_program = Decoder::DecodeFrontProgram(local);
	Decoder::Program control_program;
	Decoder::DecodeProgram(control, control_program);
	const auto ls_stride =
	    ReflectTessellationStride(local_program, true, info.input_control_points, 0u);
	if (!ls_stride) {
		return false;
	}
	const auto hs_stride = ReflectTessellationStride(control_program, false,
	                                                 info.output_control_points, *ls_stride);
	if (!hs_stride) {
		return false;
	}
	info.ls_stride = *ls_stride;
	info.hs_stride = *hs_stride;
	LOGF("Tessellation interface: input_cp=%u output_cp=%u ls_stride=%u hs_stride=%u\n",
	     info.input_control_points, info.output_control_points, info.ls_stride, info.hs_stride);
	return true;
}

bool LowerTessellationMemory(IR::Program& program, const CompileOptions& options) {
	using namespace IR;
	if (options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		return true;
	}
	const auto unsupported = [&](const char* what) {
		TESS_LOG("%s tessellation lowering does not model %s\n",
		     options.stage == ShaderType::Local                 ? "LS"
		     : options.stage == ShaderType::TessellationControl ? "HS"
		                                                       : "TES",
		     what);
		return false;
	};
	const auto& tess = options.input_info.vertex->tess;
	// The HS ID is InvocationId << 8; its low byte (the patch ordinal) is 0 for the one patch
	// an invocation group runs. The terrain HS a9f31c39ad53f06c reads it with an AND 0xff or a
	// byte extract, which constant propagation does not fold: fold those to 0 here, so the
	// patch-relative ring and factor addresses become immediates.
	{
		const auto shift_amount = [](const Value& value, uint32_t& amount) {
			const auto* shl = value.Resolve().TryInstruction();
			if (shl == nullptr || shl->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
			    !shl->Arg(1).Resolve().IsImmediate())
				return false;
			amount = shl->Arg(1).Resolve().U32();
			return amount < 32u;
		};
		for (auto* block: program.blocks) {
			for (auto& inst: *block) {
				uint32_t amount = 0;
				if (inst.GetOpcode() == ValueOpcode::BitwiseAnd32) {
					for (uint32_t side = 0; side < 2u; side++) {
						const auto mask = inst.Arg(side).Resolve();
						if (mask.IsImmediate() && shift_amount(inst.Arg(1u - side), amount) &&
						    (mask.U32() >> amount) == 0u) {
							inst.ReplaceUsesWith(Value(0u));
							break;
						}
					}
				} else if (inst.GetOpcode() == ValueOpcode::BitFieldUExtract &&
				           inst.Arg(1).Resolve().IsImmediate() &&
				           inst.Arg(2).Resolve().IsImmediate() &&
				           shift_amount(inst.Arg(0), amount) &&
				           inst.Arg(1).Resolve().U32() + inst.Arg(2).Resolve().U32() <= amount) {
					inst.ReplaceUsesWith(Value(0u));
				}
			}
		}
	}
	// Ring addresses can reuse data VGPRs. Their inactive values are irrelevant to
	// a store guarded by the same EXEC predicate, but must remain intact elsewhere.
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			if (TessellationBufferBase(*it) == nullptr) continue;
			std::unordered_map<Inst*, Value> resolved;
			it->SetArg(
			    2, ActiveAddress(*block, it, it->Arg(2), it->Arg(it->NumArgs() - 1u), resolved));
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	uint32_t reads = 0, writes = 0, factors = 0;
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto&      inst   = *it;
			const auto shared = SharedAccessOf(inst.GetOpcode());
			const auto buffer = BufferAccessOf(inst.GetOpcode());
			if (shared == SharedAccess::None && buffer == BufferAccess::None) {
				continue;
			}
			const auto&           memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			bool                  write  = false;
			uint32_t              components = 0;
			TessellationAttribute kind;
			Value                 address, predicate;
			if (shared != SharedAccess::None && memory.kind == ResourceKind::Lds) {
				write = shared == SharedAccess::Write;
				if (memory.data_bits != 32u ||
				    (options.stage == ShaderType::Local
				         ? !write
				         : options.stage != ShaderType::TessellationControl ||
				               shared != SharedAccess::Read)) {
					return unsupported("this LDS access");
				}
				kind       = write ? TessellationAttribute::LocalOutput
				                   : TessellationAttribute::ControlInput;
				components = SharedComponentCount(inst.GetOpcode());
				address    = inst.Arg(0);
				predicate  = inst.Arg(inst.NumArgs() - 1u);
			} else if (buffer != BufferAccess::None && memory.kind == ResourceKind::Buffer) {
				const auto* base = TessellationBufferBase(inst);
				if (base == nullptr) {
					continue;
				}
				if (memory.data_bits != 32u || memory.formatted || memory.idxen || !memory.offen ||
				    buffer == BufferAccess::Atomic) {
					return unsupported("this ring buffer access");
				}
				write      = buffer == BufferAccess::Write;
				components = BufferComponentCount(inst.GetOpcode());
				address    = inst.Arg(2).Resolve();
				predicate  = inst.Arg(inst.NumArgs() - 1u);
				if (base->Arg(0).U32() == 1u) {
					if (!write || options.stage != ShaderType::TessellationControl) {
						return unsupported("a tessellation factor access outside the HS");
					}
					kind = TessellationAttribute::Factor;
					factors += components;
				} else {
					kind = write ? TessellationAttribute::ControlOutput
					             : TessellationAttribute::EvaluationInput;
					if (write ? options.stage != ShaderType::TessellationControl
					          : options.stage != ShaderType::TessellationEvaluation) {
						return unsupported("a control-point access in the wrong stage");
					}
					// A write past the control points is patch data (the terrain HS a9f31c39ad53f06c
					// stores its factors there too). Its address can reach here as a select on the
					// store's EXEC predicate rather than an immediate; taken as a control point it
					// lands, modulo the stride, on the first dwords of the writer's own control point.
					auto patch_address = address.IsImmediate() ? std::optional<uint32_t>(address.U32())
					                                           : std::nullopt;
					if (!patch_address && write) {
						patch_address = ConstantActiveAddress(address, predicate);
					}
					if (patch_address &&
					    *patch_address >= tess.hs_stride * tess.output_control_points) {
						if (!write) {
							return unsupported("a patch-constant read");
						}
						kind    = TessellationAttribute::PatchOutput;
						address = Value(*patch_address);
					}
				}
			} else {
				continue;
			}
			// KYTY_TESS_LOG_RING=1: every lowered ring/LDS access, with its kind and address.
			static const bool log_ring = std::getenv("KYTY_TESS_LOG_RING") != nullptr;
			if (log_ring && write && kind == TessellationAttribute::ControlOutput) {
				// The address expression, to see what keeps it from folding.
				std::string tree;
				const std::function<void(const Value&, uint32_t)> describe =
				    [&](const Value& value, uint32_t depth) {
					    const auto resolved = value.Resolve();
					    if (resolved.IsImmediate()) {
						    tree += resolved.GetType() == Type::U32 ? std::to_string(resolved.U32())
						                                            : std::string("imm");
						    return;
					    }
					    const auto* source = resolved.TryInstruction();
					    if (source == nullptr) {
						    tree += "?";
						    return;
					    }
					    tree += ValueOpcodeName(source->GetOpcode());
					    if (depth >= 5u) {
						    tree += "(..)";
						    return;
					    }
					    tree += "(";
					    for (size_t index = 0; index < source->NumArgs() && index < 4u; index++) {
						    if (index != 0u) tree += ",";
						    describe(source->Arg(index), depth + 1u);
					    }
					    tree += ")";
				    };
				describe(address, 0);
				std::fprintf(stderr, "[tess-ring] HS control write address %s\n", tree.c_str());
			}
			if (log_ring) {
				std::fprintf(stderr, "[tess-ring] %s hash=%016llx %s kind=%u comps=%u offset=%u address=%s\n",
				             options.stage == ShaderType::Local                 ? "LS"
				             : options.stage == ShaderType::TessellationControl ? "HS"
				                                                               : "TES",
				             static_cast<unsigned long long>(options.shader_hash),
				             write ? "write" : "read", static_cast<uint32_t>(kind), components,
				             memory.offset,
				             address.IsImmediate() ? std::to_string(address.U32()).c_str() : "value");
			}
			// KYTY_TESS_PATCH_OUTPUTS=0 (diagnostic): drop the HS's patch-data writes (no TES this
			// lowering supports reads them).
			static const bool drop_patch = [] {
				const char* value = std::getenv("KYTY_TESS_PATCH_OUTPUTS");
				return value != nullptr && value[0] == '0';
			}();
			if (drop_patch && write && kind == TessellationAttribute::PatchOutput) {
				inst.Invalidate();
				continue;
			}
			const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
				return Value(&*block->PrependNewInst(it, opcode, args));
			};
			std::array<Value, 4> values;
			for (uint32_t component = 0; component < components; component++) {
				const auto offset = memory.offset + 4u * component;
				const auto byte_address =
				    offset == 0u ? address : emit(ValueOpcode::IAdd32, {address, Value(offset)});
				if (write) {
					Value data = shared != SharedAccess::None ? inst.Arg(component + 1u)
					             : components == 1u
					                 ? inst.Arg(4)
					                 : emit(components == 2u   ? ValueOpcode::CompositeExtractU32x2
					                        : components == 3u ? ValueOpcode::CompositeExtractU32x3
					                                           : ValueOpcode::CompositeExtractU32x4,
					                        {inst.Arg(4), Value(component)});
					emit(ValueOpcode::SetTessellationAttribute,
					     {Value(static_cast<uint32_t>(kind)), byte_address, data, predicate});
					writes++;
				} else {
					values[component] =
					    emit(ValueOpcode::GetTessellationAttribute,
					         {Value(static_cast<uint32_t>(kind)), byte_address, predicate});
					reads++;
				}
			}
			if (!write) {
				Value replacement = values[0];
				if (components == 2u)
					replacement =
					    emit(ValueOpcode::CompositeConstructU32x2, {values[0], values[1]});
				if (components == 3u)
					replacement = emit(ValueOpcode::CompositeConstructU32x3,
					                   {values[0], values[1], values[2]});
				if (components == 4u)
					replacement = emit(ValueOpcode::CompositeConstructU32x4,
					                   {values[0], values[1], values[2], values[3]});
				inst.ReplaceUsesWith(replacement);
			}
			inst.Invalidate();
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	RemoveIdentities(program.blocks);
	EliminateDeadCode(program.blocks);
	LOGF("%s tessellation lowering: reads=%u writes=%u factors=%u\n",
	     options.stage == ShaderType::Local                 ? "LS"
	     : options.stage == ShaderType::TessellationControl ? "HS"
	                                                       : "TES",
	     reads, writes, factors);
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler
