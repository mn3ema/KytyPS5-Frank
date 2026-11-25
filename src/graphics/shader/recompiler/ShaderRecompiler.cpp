#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/RtMerge.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler {

namespace {

const char* GetDumpLabel(const CompileOptions& options) {
	return options.dump_label != nullptr ? options.dump_label : "ShaderRecompiler";
}

const char* StageName(ShaderType stage);

// KYTY_DUMP_GAVE_UP=<folder>: write the guest code of every shader that gives up to
// <folder>/<stage>_<hash>.bin, so it can be disassembled offline (llvm-mc -triple=amdgcn).
void DumpGaveUpCode(const CompileOptions& options, std::span<const uint32_t> code) {
	const char* folder = std::getenv("KYTY_DUMP_GAVE_UP");
	if (folder == nullptr || code.empty()) {
		return;
	}
	const auto path = fmt::format("{}/{}_{:016x}.bin", folder, StageName(options.stage),
	                              options.shader_hash);
	if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
		std::fwrite(code.data(), sizeof(uint32_t), code.size(), file);
		std::fclose(file);
	}
}

// A shader that gives up: the current compile phase turns red, and the timeline gets the reason.
void ReportGaveUp(const CompileOptions& options, Profiler::Phases& phases, const char* reason) {
	phases.Color(Profiler::MessageFailure);
	Profiler::Add(Profiler::Counter::ShadersGaveUp);
	Profiler::Message(Profiler::MessageFailure, "Shader gave up: %s 0x%016" PRIx64 ": %s",
	                  StageName(options.stage), options.shader_hash, reason);
	// Also to stderr (first 256): a shader that gives up skips its draws or dispatches for good.
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1) < 256) {
		std::fprintf(stderr, "[gave-up] %s 0x%016" PRIx64 ": %s\n", StageName(options.stage),
		             options.shader_hash, reason);
		std::fflush(stderr);
	}
}

std::string MakeIrDump(std::string_view cfg, const IR::Program& ir) {
	std::string dump = "CFG:\n";
	dump += cfg;
	dump += "\nIR:\n";
	dump += fmt::format("mode={} scratch_dwords={}\n",
	                    ir.dispatcher_fallback ? "dispatcher" : "structured", ir.scratch_dwords);
	dump += IR::ProgramToString(ir);
	return dump;
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "CS";
		case ShaderType::Vertex: return "VS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "TES";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Pixel: return "PS";
		default: return "unknown";
	}
}

void LogDispatcherFallback(const CompileOptions& options, const CFG::Graph& cfg, const char* phase) {
	// Also to stderr (first 256): a dispatcher shader stops after a bounded number of block
	// transitions (KYTY_DISPATCHER_MAX_TRANSITIONS), which can cut long per-pixel loops short.
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1) < 256) {
		std::fprintf(stderr, "[dispatcher] %s 0x%016" PRIx64 " (%s)\n", StageName(options.stage),
		             options.shader_hash, phase);
	}
	const auto* block        = cfg.FindBlock(cfg.failure_block);
	const auto  start        = block != nullptr ? block->start_pc : UINT32_MAX;
	const auto  end          = block != nullptr ? block->end_pc : UINT32_MAX;
	const auto  predecessors = block != nullptr ? block->predecessors.size() : 0u;
	const auto  successors   = block != nullptr ? block->successors.size() : 0u;
	LOGF("%s CFG dispatcher fallback: stage=%s hash=0x%016" PRIx64
	     " phase=%s failure=%s block=%" PRIu32 " pc=0x%08" PRIx32 "..0x%08" PRIx32 " preds=%" PRIu64
	     " succs=%" PRIu64 " blocks=%" PRIu64 " loops=%" PRIu64 " back_edges=%" PRIu64
	     " reason=%s\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash, phase,
	     CFG::FailureKindToString(cfg.failure_kind).c_str(), cfg.failure_block, start, end,
	     static_cast<uint64_t>(predecessors), static_cast<uint64_t>(successors),
	     static_cast<uint64_t>(cfg.blocks.size()), static_cast<uint64_t>(cfg.natural_loops.size()),
	     static_cast<uint64_t>(cfg.back_edges.size()), cfg.unsupported_reason.c_str());
}

enum class EmbeddedFetchValueType {
	Unknown,
	Constant,
	AttribTable,
	Attrib,
	BufferTable,
	Buffer
};

struct EmbeddedFetchSgprInfo {
	EmbeddedFetchValueType type      = EmbeddedFetchValueType::Unknown;
	int                    attrib_id = 0;
	uint32_t               value     = 0;
};

using EmbeddedFetchVectorLanes = std::map<uint64_t, EmbeddedFetchSgprInfo>;

uint64_t EmbeddedFetchVectorLaneKey(uint32_t reg, uint32_t lane) {
	return (static_cast<uint64_t>(reg) << 32u) | lane;
}

uint32_t EmbeddedFetchLane(uint32_t lane, uint32_t wave_size) {
	return wave_size == 32 || wave_size == 64 ? lane % wave_size : lane;
}

void ClearEmbeddedFetchVectorLanes(EmbeddedFetchVectorLanes* lanes, uint32_t reg) {
	const auto first = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg, 0));
	const auto last  = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg + 1u, 0));
	lanes->erase(first, last);
}

bool IsDecodedSgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Sgpr || op.kind == Decoder::OperandKind::VccLo ||
	       op.kind == Decoder::OperandKind::VccHi;
}

uint32_t DecodedSgprReg(const Decoder::Operand& op) {
	switch (op.kind) {
		case Decoder::OperandKind::VccLo: return 106u;
		case Decoder::OperandKind::VccHi: return 107u;
		default: return op.reg;
	}
}

bool IsDecodedVgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Vgpr;
}

uint32_t DecodedDstSize(const Decoder::Instruction& inst) {
	return std::max(inst.data_dwords, 1u);
}

uint32_t EmbeddedFetchDstSize(const Decoder::Instruction& inst) {
	return inst.opcode == Decoder::Opcode::V_MAD_U64_U32 ? 2u : DecodedDstSize(inst);
}

void ClearEmbeddedFetchSgprs(std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                             const Decoder::Operand& dst, uint32_t size) {
	if (!IsDecodedSgpr(dst)) {
		return;
	}
	const auto register_id = DecodedSgprReg(dst);
	for (uint32_t i = 0; i < size && register_id + i < sgprs.size(); i++) {
		sgprs[register_id + i] = {};
	}
}

bool TryDecodedOperandConstant(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                               const Decoder::Operand& op, uint32_t& value) {
	switch (op.kind) {
		case Decoder::OperandKind::LiteralConstant:
		case Decoder::OperandKind::IntegerInlineConstant:
		case Decoder::OperandKind::FloatInlineConstant: value = op.value; return true;
		case Decoder::OperandKind::Null: value = 0; return true;
		default: break;
	}
	if (IsDecodedSgpr(op) && DecodedSgprReg(op) < sgprs.size() &&
	    sgprs[DecodedSgprReg(op)].type == EmbeddedFetchValueType::Constant) {
		value = sgprs[DecodedSgprReg(op)].value;
		return true;
	}
	return false;
}

bool TryDecodedSmemOffset(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                          const Decoder::Instruction& inst, uint32_t& raw_offset) {
	uint32_t base = 0;
	if (!TryDecodedOperandConstant(sgprs, inst.src1, base)) {
		return false;
	}
	const auto value = static_cast<uint64_t>(base) + inst.offset;
	if (value > 0xffffffffull) {
		return false;
	}
	raw_offset = static_cast<uint32_t>(value);
	return true;
}

bool IsEmbeddedFetchSLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_LOAD_DWORD:
		case Decoder::Opcode::S_LOAD_DWORDX2:
		case Decoder::Opcode::S_LOAD_DWORDX4:
		case Decoder::Opcode::S_LOAD_DWORDX8:
		case Decoder::Opcode::S_LOAD_DWORDX16: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchBufferLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_X:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XY:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZ:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZW: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchAttribPropagationAlu(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_BFE_U32:
		case Decoder::Opcode::S_AND_B32:
		case Decoder::Opcode::S_ADD_I32:
		case Decoder::Opcode::S_ADD_U32:
		case Decoder::Opcode::S_LSHL_B32: return true;
		default: return false;
	}
}

int BufferTableAttribFromOffset(uint32_t raw_offset, int dword) {
	return static_cast<int>((raw_offset + static_cast<uint32_t>(dword) * 4u) / 16u);
}

Frontend::EmbeddedFetchPlan DetectEmbeddedVertexFetch(
    const Decoder::Program& decoded, const ShaderVertexInputInfo* input_info,
    uint32_t user_data_base, uint32_t user_data_count, uint32_t wave_size) {
	const uint32_t    vertex_index_reg   = input_info->logical_stage == ShaderType::Local ? 2u : 5u;
	const uint32_t    instance_index_reg = input_info->logical_stage == ShaderType::Local ? 5u : 8u;
	Frontend::EmbeddedFetchPlan data;
	data.loads.reserve(input_info->resources_num);
	int32_t vertex_offset_candidate   = -1;
	int32_t instance_offset_candidate = -1;
	bool    vertex_offset_conflict    = false;
	bool    instance_offset_conflict  = false;

	const int shift_regs = 8;
	const int attrib_reg = input_info->fetch_attrib_reg + shift_regs;
	const int buffer_reg = input_info->fetch_buffer_reg + shift_regs;

	std::array<EmbeddedFetchSgprInfo, 108> sgprs {};
	std::array<bool, 256>                 vgpr_is_index {};
	EmbeddedFetchVectorLanes               vector_lanes;
	const bool                             track_vector_lanes =
	    std::none_of(decoded.instructions.begin(), decoded.instructions.end(),
	                 [](const auto& inst) {
		                 return Decoder::IsDirectBranch(inst.opcode) ||
		                        inst.opcode == Decoder::Opcode::S_SETPC_B64;
	                 });

	if (attrib_reg >= 0 && attrib_reg < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg].type = EmbeddedFetchValueType::AttribTable;
	}
	if (attrib_reg + 1 >= 0 && attrib_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg + 1].type = EmbeddedFetchValueType::AttribTable;
	}
	if (buffer_reg >= 0 && buffer_reg < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg].type = EmbeddedFetchValueType::BufferTable;
	}
	if (buffer_reg + 1 >= 0 && buffer_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg + 1].type = EmbeddedFetchValueType::BufferTable;
	}

	for (const auto& inst: decoded.instructions) {
		// Fetch shaders accumulate the draw's vertex offset in v0. The PS5 NGG ABI
		// seeds S_NGG_VERTEX_INDEX in v5 and S_NGG_INSTANCE_INDEX in v8, then applies the
		// corresponding direct-draw offsets before fetching.
		const bool vertex_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == 0 || (user_data_base == 8 && inst.dst.reg == vertex_index_reg));
		const bool instance_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == (user_data_base == 8 ? instance_index_reg : 3u));
		uint32_t   sad_zero = 0;
		const bool index_offset_add =
		    (vertex_index_accumulator || instance_index_accumulator) && IsDecodedSgpr(inst.src0) &&
		    ((inst.opcode == Decoder::Opcode::V_ADD_I32 && IsDecodedVgpr(inst.src1) &&
		      inst.src1.reg == inst.dst.reg) ||
		     (user_data_base == 8 &&
		      (inst.dst.reg == vertex_index_reg || inst.dst.reg == instance_index_reg) &&
		      inst.opcode == Decoder::Opcode::V_SAD_U32 && IsDecodedVgpr(inst.src2) &&
		      inst.src2.reg == inst.dst.reg &&
		      TryDecodedOperandConstant(sgprs, inst.src1, sad_zero) && sad_zero == 0));
		if (data.loads.empty() && index_offset_add) {
			const auto reg = DecodedSgprReg(inst.src0);
			if (reg >= user_data_base && reg - user_data_base < user_data_count) {
				auto& candidate = vertex_index_accumulator ? vertex_offset_candidate
				                                           : instance_offset_candidate;
				auto& conflict  = vertex_index_accumulator ? vertex_offset_conflict
				                                           : instance_offset_conflict;
				if (candidate >= 0 && candidate != static_cast<int32_t>(reg)) {
					conflict = true;
				} else {
					candidate = static_cast<int32_t>(reg);
				}
			}
		}
		switch (inst.opcode) {
			case Decoder::Opcode::V_WRITELANE_B32: {
				uint32_t lane = 0;
				if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
					vgpr_is_index[inst.dst.reg] = false;
				}
				if (track_vector_lanes && IsDecodedVgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size() &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					vector_lanes[EmbeddedFetchVectorLaneKey(inst.dst.reg,
					                                        EmbeddedFetchLane(lane, wave_size))] =
					    sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedVgpr(inst.dst)) {
					ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
				}
				break;
			}
			case Decoder::Opcode::V_READLANE_B32: {
				uint32_t lane = 0;
				if (track_vector_lanes && IsDecodedSgpr(inst.dst) &&
				    DecodedSgprReg(inst.dst) < sgprs.size() && IsDecodedVgpr(inst.src0) &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					const auto found = vector_lanes.find(EmbeddedFetchVectorLaneKey(
					    inst.src0.reg, EmbeddedFetchLane(lane, wave_size)));
					sgprs[DecodedSgprReg(inst.dst)] =
					    found != vector_lanes.end() ? found->second : EmbeddedFetchSgprInfo {};
				} else if (IsDecodedSgpr(inst.dst)) {
					ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
				}
				break;
			}
			case Decoder::Opcode::S_MOV_B32:
				if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size()) {
					sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedSgpr(inst.dst)) {
					uint32_t value = 0;
					if (TryDecodedOperandConstant(sgprs, inst.src0, value)) {
						auto& dst = sgprs[DecodedSgprReg(inst.dst)];
						dst.type  = EmbeddedFetchValueType::Constant;
						dst.value = value;
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
					}
				}
				break;
			case Decoder::Opcode::S_MOVK_I32:
				if (IsDecodedSgpr(inst.dst)) {
					auto& dst = sgprs[DecodedSgprReg(inst.dst)];
					dst.type  = EmbeddedFetchValueType::Constant;
					dst.value = inst.src0.value;
				}
				break;
			default:
				if (IsEmbeddedFetchSLoad(inst)) {
					if (IsDecodedSgpr(inst.src0) && DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type ==
					        EmbeddedFetchValueType::AttribTable) {
						uint32_t raw_offset = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							const auto register_id = DecodedSgprReg(inst.dst);
							const int  index       = static_cast<int>(raw_offset / 4u);
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Attrib;
								dst.attrib_id    = index + static_cast<int>(i);
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else if (IsDecodedSgpr(inst.src0) &&
					           DecodedSgprReg(inst.src0) < sgprs.size() &&
					           sgprs[DecodedSgprReg(inst.src0)].type ==
					               EmbeddedFetchValueType::BufferTable) {
						const auto register_id = DecodedSgprReg(inst.dst);
						uint32_t   raw_offset  = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst = sgprs[register_id + i];
								dst.type  = EmbeddedFetchValueType::Buffer;
								dst.attrib_id =
								    BufferTableAttribFromOffset(raw_offset, static_cast<int>(i));
							}
						} else if (IsDecodedSgpr(inst.src1) &&
						           DecodedSgprReg(inst.src1) < sgprs.size() &&
						           sgprs[DecodedSgprReg(inst.src1)].type ==
						               EmbeddedFetchValueType::Attrib &&
						           (inst.offset & 0x3u) == 0) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Buffer;
								dst.attrib_id    = sgprs[DecodedSgprReg(inst.src1)].attrib_id;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
					}
				} else if (inst.opcode == Decoder::Opcode::V_CNDMASK_B32) {
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
						ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
					}
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size() &&
					    IsDecodedVgpr(inst.src0) && inst.src0.reg == instance_index_reg &&
					    IsDecodedVgpr(inst.src1) && inst.src1.reg == vertex_index_reg) {
						vgpr_is_index[inst.dst.reg] = true;
					}
				} else if (IsEmbeddedFetchAttribPropagationAlu(inst)) {
					if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
					    DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type == EmbeddedFetchValueType::Attrib) {
						sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
					} else if (IsDecodedSgpr(inst.dst)) {
						uint32_t src0 = 0;
						uint32_t src1 = 0;
						if (TryDecodedOperandConstant(sgprs, inst.src0, src0) &&
						    TryDecodedOperandConstant(sgprs, inst.src1, src1)) {
							auto& dst = sgprs[DecodedSgprReg(inst.dst)];
							dst.type  = EmbeddedFetchValueType::Constant;
							switch (inst.opcode) {
								case Decoder::Opcode::S_AND_B32: dst.value = src0 & src1; break;
								case Decoder::Opcode::S_LSHL_B32:
									dst.value = src0 << (src1 & 31u);
									break;
								case Decoder::Opcode::S_BFE_U32:
									dst.value = src0 >> (src1 & 31u);
									break;
								default: dst.value = src0 + src1; break;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
						}
					}
				} else if (IsEmbeddedFetchBufferLoad(inst)) {
					if (IsDecodedVgpr(inst.src0) && inst.src0.reg < vgpr_is_index.size() &&
					    vgpr_is_index[inst.src0.reg] &&
					    IsDecodedSgpr(inst.src1) && DecodedSgprReg(inst.src1) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src1)].type == EmbeddedFetchValueType::Buffer) {
						const auto& buffer = sgprs[DecodedSgprReg(inst.src1)];
						if (data.loads.empty()) {
							if (!vertex_offset_conflict) {
								data.vertex_offset_sgpr = vertex_offset_candidate;
							}
							if (!instance_offset_conflict) {
								data.instance_offset_sgpr = instance_offset_candidate;
							}
						}
						auto& load        = data.loads.emplace_back();
						load.pc           = inst.pc;
						load.attrib_id    = buffer.attrib_id;
						load.components   = DecodedDstSize(inst);
					}
				}
				break;
		}
		if (inst.opcode == Decoder::Opcode::V_MOVRELD_B32) {
			vector_lanes.clear();
		} else if (inst.opcode != Decoder::Opcode::V_WRITELANE_B32 && IsDecodedVgpr(inst.dst)) {
			for (uint32_t i = 0;
			     i < EmbeddedFetchDstSize(inst) && inst.dst.reg + i < vgpr_is_index.size();
			     i++) {
				ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg + i);
			}
		}
	}

	return data;
}

Decoder::Program DecodeFusedProgram(std::span<const uint32_t> front, std::span<const uint32_t> back,
                                    std::vector<uint32_t>& joined_code) {
	EXIT_IF(back.empty());
	auto       result      = Decoder::DecodeFrontProgram(front);
	const auto front_words = static_cast<uint32_t>(result.code.size());
	joined_code.assign(result.code.begin(), result.code.end());
	joined_code.insert(joined_code.end(), back.begin(), back.end());
	// The merged-stage ABI passes the back shader in s[6:7]. Give that handoff an
	// ordinary CFG edge, retaining both bodies in one register and LDS lifetime.
	joined_code[front_words - 1u] = 0xbf820000u; // s_branch to the following instruction
	result.instructions.back()    = {};
	Decoder::DecodeInstruction(joined_code, front_words - 1u, result.instructions.back());
	Decoder::Program back_program;
	Decoder::DecodeProgram(back, back_program);
	const auto back_pc = front_words * sizeof(uint32_t);
	for (auto& inst: back_program.instructions) {
		// A back-stage PC-relative data reference requires its guest code address.
		EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_GETPC_B64);
		inst.pc += back_pc;
		inst.branch_target += back_pc;
		result.instructions.push_back(std::move(inst));
	}
	result.code = joined_code;
	return result;
}

} // namespace

namespace {

// v440: the payload's VGPR count for a callable split: the highest VGPR the code names (wide
// operands counted generously), + 1, at most 256.
uint32_t RtPayloadVgprs(std::span<const uint32_t> code) {
	Decoder::Program decoded;
	Decoder::DecodeProgram(code, decoded);
	uint32_t top = 0;
	const auto see = [&](const Decoder::Operand& operand, uint32_t count) {
		if (operand.kind == Decoder::OperandKind::Vgpr) {
			top = std::max(top, operand.reg + std::max(count, 1u));
		}
	};
	for (const auto& inst: decoded.instructions) {
		const uint32_t count = std::max(inst.data_dwords, 2u);
		see(inst.dst, count);
		see(inst.dst2, count);
		see(inst.src0, count);
		see(inst.src1, count);
		see(inst.src2, count);
		for (const auto reg: inst.image_nsa_addr) {
			if (reg != 0) {
				top = std::max(top, reg + 1u);
			}
		}
	}
	return std::min(top, 256u);
}

// v440 M3: a whole callable split (CompileOptions::rt_split): the main code as the ray-generation
// shader, each target as a callable, merged into one program (MergeRtCallables).
TranslateResult TranslateRtSplit(std::span<const uint32_t> main_code, const CompileOptions& options) {
	const auto&    split = *options.rt_split;
	IR::RtCallInfo raygen;
	raygen.role       = IR::RtRole::Raygen;
	raygen.call_pc    = split.call_pc;
	raygen.source_reg = split.source_reg;
	raygen.return_reg = split.return_reg;
	// The payload carries the VGPRs the ray-generation code names: a callee's other VGPRs are its
	// own temporaries (what it leaves there the caller never reads), so the payload, and with it
	// the ray-generation module, does not depend on the call targets.
	raygen.vgpr_count = RtPayloadVgprs(main_code);
	for (const auto& target: split.targets) {
		raygen.targets.push_back(target.address);
	}
	IR::RtCallInfo callable = raygen;
	callable.role           = IR::RtRole::Callable;
	callable.targets.clear();
	// v443 (shared callables, IR/RtCallable.h): the payload is the whole register file in every
	// split, and a callable moves every VGPR it uses, so its module does not depend on which
	// ray-generation shader calls it.
	if (IR::RtSharedCallables()) {
		raygen.payload_vgprs   = IR::RtSharedPayloadVgprs;
		callable.payload_vgprs = IR::RtSharedPayloadVgprs;
		callable.vgpr_count    = IR::RtSharedPayloadVgprs;
	}

	CompileOptions main_options = options;
	main_options.rt_split       = nullptr;
	main_options.rt_call        = &raygen;
	auto result                 = TranslateProgram(main_code, main_options);
	if (result.unsupported) {
		std::fprintf(stderr, "[rt-callables] %016llx: the ray-generation half did not translate\n",
		             static_cast<unsigned long long>(options.shader_hash));
		return result;
	}
	std::vector<IR::Program> callees;
	callees.reserve(split.targets.size());
	for (size_t index = 0; index < split.targets.size(); index++) {
		CompileOptions callee_options = options;
		callee_options.rt_split       = nullptr;
		IR::RtCallInfo callee_info    = callable;
		callee_info.code_address      = split.targets[index].address;
		callee_info.code_hash         = IR::RtCodeHash(split.targets[index].code);
		callee_options.rt_call        = &callee_info;
		callee_options.shader_hash    = options.shader_hash ^ (split.targets[index].address * 0x9e3779b97f4a7c15ull);
		auto callee                   = TranslateProgram(split.targets[index].code, callee_options);
		if (callee.unsupported) {
			std::fprintf(stderr, "[rt-callables] %016llx: callable %zu (0x%llx) did not translate\n",
			             static_cast<unsigned long long>(options.shader_hash), index,
			             static_cast<unsigned long long>(split.targets[index].address));
			return callee;
		}
		callees.push_back(std::move(callee.program));
	}
	uint32_t callee_scratch = 0;
	for (const auto& callee: callees) {
		callee_scratch = std::max(callee_scratch, callee.scratch_dwords);
	}
	const uint32_t raygen_scratch = result.program.scratch_dwords;
	// v440t: KYTY_RT_PAYLOAD_CUT=0 keeps the whole register file in the payload traffic.
	static const bool payload_cut = [] {
		const char* value = std::getenv("KYTY_RT_PAYLOAD_CUT");
		return value == nullptr || value[0] != '0';
	}();
	if (payload_cut) {
		std::string cut_stats;
		IR::CutRtPayload(result.program, callees, cut_stats);
		std::fprintf(stderr, "[rt-callables] %016llx: %s\n", static_cast<unsigned long long>(options.shader_hash),
		             cut_stats.c_str());
	}
	std::string    reason;
	if (!IR::MergeRtCallables(result.program, callees, reason)) {
		std::fprintf(stderr, "[rt-callables] %016llx: not merged: %s\n",
		             static_cast<unsigned long long>(options.shader_hash), reason.c_str());
		TranslateResult unsupported_result;
		unsupported_result.unsupported = true;
		return unsupported_result;
	}
	// Scratch is per module (function-local): a callee's spills stay its own, as within one call.
	std::fprintf(stderr, "[rt-callables] %016llx: ray generation + %zu callables merged, payload %u VGPRs, "
	             "scratch dwords %u / %u (callables)\n",
	             static_cast<unsigned long long>(options.shader_hash), split.targets.size(), raygen.vgpr_count,
	             raygen_scratch, callee_scratch);
	return result;
}

} // namespace

TranslateResult TranslateProgram(std::span<const uint32_t> code, const CompileOptions& options) {
	if (options.rt_split != nullptr) {
		return TranslateRtSplit(code, options);
	}
	const Frontend::TranslationNonFatalScope non_fatal_scope(options.non_fatal);
	if (code.empty()) {
		EXIT("shader recompiler input is empty\n");
	}
	if (options.stage != ShaderType::Compute && options.stage != ShaderType::Vertex &&
	    options.stage != ShaderType::Pixel && options.stage != ShaderType::Mesh &&
	    options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		EXIT("shader recompiler received unsupported stage %u\n",
		     static_cast<unsigned>(options.stage));
	}

	// KYTY_PATCH_GUEST=<hash>@<pc hex>=<word hex>[+<word hex>...][/...] (diagnostics, 11-13 hex
	// holes): replaces guest code dwords of that shader before decoding, e.g. a V_MOV_B32 over a
	// culling decision. The shader keeps its hash; the switch is part of the disk cache key.
	std::vector<uint32_t> patched_code;
	if (const char* patches = std::getenv("KYTY_PATCH_GUEST"); patches != nullptr) {
		const auto        hash_text = fmt::format("{:016x}", options.shader_hash);
		std::string_view  rest(patches);
		while (!rest.empty()) {
			const auto end   = rest.find('/');
			const auto entry = rest.substr(0, end);
			rest             = end == std::string_view::npos ? std::string_view() : rest.substr(end + 1);
			const auto at    = entry.find('@');
			const auto eq    = entry.find('=');
			if (at == std::string_view::npos || eq == std::string_view::npos || eq < at ||
			    entry.substr(0, at) != hash_text) {
				continue;
			}
			if (patched_code.empty()) patched_code.assign(code.begin(), code.end());
			auto pc = std::strtoull(std::string(entry.substr(at + 1, eq - at - 1)).c_str(), nullptr, 16);
			std::string_view words = entry.substr(eq + 1);
			while (!words.empty()) {
				const auto plus = words.find('+');
				const auto word = std::strtoul(std::string(words.substr(0, plus)).c_str(), nullptr, 16);
				words = plus == std::string_view::npos ? std::string_view() : words.substr(plus + 1);
				if (pc / 4u < patched_code.size()) {
					std::fprintf(stderr, "[patch-guest] %s pc=0x%04llx %08x -> %08lx\n",
					             hash_text.c_str(), static_cast<unsigned long long>(pc),
					             patched_code[pc / 4u], static_cast<unsigned long>(word));
					patched_code[pc / 4u] = static_cast<uint32_t>(word);
				}
				pc += 4u;
			}
		}
		if (!patched_code.empty()) code = patched_code;
	}

	// The phases as Tracy zones, each with the stage and hash as its text.
	Profiler::Phases phases;
	phases.SetText("%s 0x%016" PRIx64, StageName(options.stage), options.shader_hash);
	KYTY_PROFILER_PHASE(phases, "Shader: decode", profiler::colors::Green100);

	const auto compile_begin = std::chrono::steady_clock::now();
	const auto phase_ms      = [&compile_begin]() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                                 std::chrono::steady_clock::now() - compile_begin)
		                                 .count());
	};

	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " code_words=%" PRIu64 " decode\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(code.size()));

	Decoder::Program decoded;
	std::vector<uint32_t> joined_code;
	if (!options.back_code.empty()) {
		decoded = DecodeFusedProgram(code, options.back_code, joined_code);
	} else if (options.stage == ShaderType::Local) {
		decoded = Decoder::DecodeFrontProgram(code);
		// The separately compiled hull half runs in the next Vulkan stage.
		auto& handoff     = decoded.instructions.back();
		handoff.opcode    = Decoder::Opcode::S_ENDPGM;
		handoff.src_count = 0;
	} else {
		Decoder::DecodeProgram(code, decoded);
	}
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " decode instructions=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(decoded.instructions.size()), phase_ms());

	// KYTY_DUMP_SHADER_HASHES="hash,hash,...": dump these shaders' code too (into
	// KYTY_DUMP_GAVE_UP), for shaders that compile but are skipped later.
	if (const char* list = std::getenv("KYTY_DUMP_SHADER_HASHES");
	    list != nullptr &&
	    std::string_view(list).find(fmt::format("{:016x}", options.shader_hash)) !=
	        std::string_view::npos) {
		DumpGaveUpCode(options, code);
	}
	std::string decoded_dump;
	if (options.dump_ir) {
		decoded_dump = Decoder::ProgramToString(decoded);
		if (options.early_dump) {
			LOGF("%s decoded RDNA2 (early):\n%s", GetDumpLabel(options), decoded_dump.c_str());
		}
	}

	KYTY_PROFILER_PHASE(phases, "Shader: CFG", profiler::colors::Green200);
	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " CFG BuildGraph\n", GetDumpLabel(options),
	     StageName(options.stage), options.shader_hash);
	auto native_cfg = CFG::BuildGraph(decoded);
	if (options.non_fatal && native_cfg.unsupported && !native_cfg.irreducible) {
		DumpGaveUpCode(options, code);
		ReportGaveUp(options, phases, native_cfg.unsupported_reason.c_str());
		LOGF("%s gave up hash=0x%016" PRIx64 ": %s\n", GetDumpLabel(options), options.shader_hash,
		     native_cfg.unsupported_reason.c_str());
		TranslateResult unsupported_result;
		unsupported_result.unsupported = true;
		return unsupported_result;
	}
	CFG::Graph structured_cfg;
	auto* selected_cfg = &native_cfg;
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " CFG BuildGraph blocks=%" PRIu64
	     " loops=%" PRIu64 " back_edges=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(native_cfg.blocks.size()),
	     static_cast<uint64_t>(native_cfg.natural_loops.size()),
	     static_cast<uint64_t>(native_cfg.back_edges.size()), phase_ms());
	if (native_cfg.irreducible) {
		LogDispatcherFallback(options, native_cfg, "build");
	} else {
		LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " CFG Structurize\n",
		     GetDumpLabel(options), StageName(options.stage), options.shader_hash);
		structured_cfg = CFG::Structurize(native_cfg);
		if (structured_cfg.unsupported) {
			native_cfg.unsupported = true;
			native_cfg.failure_kind = structured_cfg.failure_kind;
			native_cfg.failure_block = structured_cfg.failure_block;
			native_cfg.unsupported_reason = structured_cfg.unsupported_reason;
			LogDispatcherFallback(options, native_cfg, "structurize");
		} else {
			selected_cfg = &structured_cfg;
			LOGF("%s structured CFG success: blocks=%" PRIu64 "\n", GetDumpLabel(options),
			     static_cast<uint64_t>(selected_cfg->blocks.size()));
		}
		LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " CFG Structurize blocks=%" PRIu64
		     " loops=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
		     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
		     static_cast<uint64_t>(selected_cfg->blocks.size()),
		     static_cast<uint64_t>(selected_cfg->natural_loops.size()), phase_ms());
	}

	const auto& cfg = *selected_cfg;
	// A memory bound: each IR block carries the SSA rewrite's register tables (about 9 KB) and
	// the rewrite records goto flags per block, so a structured graph of hundreds of thousands
	// of blocks (every call target of a ray-tracing shader inlined: 485,521 blocks) took the
	// translation past 70 GB. Such a program gives up instead. KYTY_MAX_IR_BLOCKS overrides.
	static const uint64_t max_ir_blocks = [] {
		const char* value = std::getenv("KYTY_MAX_IR_BLOCKS");
		return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {100000};
	}();
	if (options.non_fatal && cfg.blocks.size() > max_ir_blocks) {
		const auto reason = fmt::format("structured CFG has {} blocks, above the {}-block bound (KYTY_MAX_IR_BLOCKS)",
		                                cfg.blocks.size(), max_ir_blocks);
		std::fprintf(stderr, "[gave-up-reason] %016llx %s\n", static_cast<unsigned long long>(options.shader_hash),
		             reason.c_str());
		DumpGaveUpCode(options, code);
		ReportGaveUp(options, phases, reason.c_str());
		TranslateResult unsupported_result;
		unsupported_result.unsupported             = true;
		unsupported_result.input_dependent_refusal = true;
		return unsupported_result;
	}
	Frontend::EmbeddedFetchPlan embedded_fetch;
	if ((options.stage == ShaderType::Vertex || options.stage == ShaderType::Local) &&
	    options.input_info.vertex != nullptr && options.input_info.vertex->fetch_embedded) {
		embedded_fetch = DetectEmbeddedVertexFetch(
		    decoded, options.input_info.vertex, options.user_data_base,
		    static_cast<uint32_t>(options.user_data.size()), options.wave_size);
		if (!embedded_fetch.loads.empty()) {
			LOGF("%s embedded vertex fetch plan: detected=%" PRIu64 "\n", GetDumpLabel(options),
			     static_cast<uint64_t>(embedded_fetch.loads.size()));
		}
	}
	Frontend::TranslateOptions translate_options {
	    .stage            = options.stage,
	    .wave_size        = options.wave_size,
	    .shader_hash      = options.shader_hash,
	    .user_data_base   = options.user_data_base,
	    .user_data_count  = static_cast<uint32_t>(options.user_data.size()),
	    .input_info       = options.input_info,
	    .embedded_fetch   = embedded_fetch.loads.empty() ? nullptr : &embedded_fetch,
	    .rt_call          = options.rt_call,
	};
	KYTY_PROFILER_PHASE(phases, "Shader: IR translate", profiler::colors::Green300);
	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " IR TranslateProgram\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash);
	auto ir = Frontend::TranslateProgram(decoded, cfg, translate_options);
	// v446h: the A/B translation variant rides on the program (ShaderInfo::translation_variant).
	// ShaderStageInputInfo is a union: read the member of this stage only.
	uint32_t variant_bits = 0;
	if (options.stage == ShaderType::Compute) {
		variant_bits = options.input_info.compute != nullptr ? options.input_info.compute->translation_variant : 0u;
	} else if (options.stage == ShaderType::Pixel) {
		variant_bits = options.input_info.pixel != nullptr ? options.input_info.pixel->translation_variant : 0u;
	} else {
		variant_bits = options.input_info.vertex != nullptr ? options.input_info.vertex->translation_variant : 0u;
	}
	const auto translation_variant = static_cast<uint8_t>(variant_bits);
	ir.info.translation_variant = translation_variant;
	// v446j: compute programs keep static buffer formats by default. In-run A/B (g454a, top 30
	// compute shaders): frame -0.5 to -0.7 ms, GPU -1.5 ms a frame. Draws do not (g454c: GPU -0.7 ms
	// but the GPU thread +0.6 ms, frame no better). KYTY_COMPUTE_STATIC_FORMATS=0 turns it off.
	// v446s: off by default again (KYTY_COMPUTE_STATIC_FORMATS=1 turns it on): with it, v446r drew
	// the text of the game's notes with blocks over letters (Mac 11-22).
	static const bool compute_static_formats = [] {
		const char* value = std::getenv("KYTY_COMPUTE_STATIC_FORMATS");
		return value != nullptr && value[0] == '1';
	}();
	ir.info.static_buffer_formats = options.stage == ShaderType::Compute && compute_static_formats ? 1u : 0u;
	if (options.non_fatal && Frontend::TranslationUnsupported()) {
		DumpGaveUpCode(options, code);
		ReportGaveUp(options, phases, "no IR translation for an instruction");
		LOGF("%s gave up hash=0x%016" PRIx64 ": no IR translation for an instruction\n",
		     GetDumpLabel(options), options.shader_hash);
		TranslateResult unsupported_result;
		unsupported_result.unsupported = true;
		return unsupported_result;
	}
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " IR TranslateProgram blocks=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(ir.blocks.size()), phase_ms());
	const auto passes_begin = std::chrono::steady_clock::now();
	const auto ms_since     = [](std::chrono::steady_clock::time_point begin) {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                                 std::chrono::steady_clock::now() - begin)
		                                 .count());
	};
	// KYTY_PASS_TIMES=1: per-pass milliseconds of the IR passes before resource tracking, printed
	// with the phase line when they add up to a second or more (large programs).
	static const bool pass_times_enabled = std::getenv("KYTY_PASS_TIMES") != nullptr;
	std::map<std::string, uint64_t> pass_times;
#define KYTY_PASS_TIME(name, call)                                                                 \
	([&]() -> decltype(auto) {                                                                     \
		const auto pass_begin_ = std::chrono::steady_clock::now();                                 \
		struct PassTimer_ {                                                                        \
			std::map<std::string, uint64_t>&      times;                                            \
			std::chrono::steady_clock::time_point begin;                                            \
			~PassTimer_() {                                                                         \
				if (pass_times_enabled) {                                                           \
					times[#name] += static_cast<uint64_t>(                                          \
					    std::chrono::duration_cast<std::chrono::milliseconds>(                       \
					        std::chrono::steady_clock::now() - begin)                               \
					        .count());                                                              \
				}                                                                                   \
			}                                                                                       \
		} pass_timer_ {pass_times, pass_begin_};                                                   \
		return call;                                                                               \
	}())
	KYTY_PROFILER_PHASE(phases, "Shader: IR passes", profiler::colors::Green400);
	const auto count_ir = [&](const char* when) {
		if (!pass_times_enabled || ir.blocks.size() < 4096u) {
			return;
		}
		size_t total = 0, phis = 0, identities = 0, phi_operands = 0;
		for (auto* block: ir.blocks) {
			for (const auto& inst: *block) {
				total++;
				if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
					phis++;
					phi_operands += inst.NumArgs();
				}
				identities += inst.GetOpcode() == IR::ValueOpcode::Identity ? 1u : 0u;
			}
		}
		std::fprintf(stderr, "[pass-times] %016llx %s: blocks=%zu insts=%zu phis=%zu phi_operands=%zu identities=%zu\n",
		             static_cast<unsigned long long>(options.shader_hash), when, ir.blocks.size(), total,
		             phis, phi_operands, identities);
	};
	count_ir("before ssa");
	KYTY_PASS_TIME(RewriteToSsa, IR::RewriteToSsa(ir.blocks));
	count_ir("after ssa");
	KYTY_PASS_TIME(ConstantPropagationPass, IR::ConstantPropagationPass(ir.blocks, ir.wave_size));
	KYTY_PASS_TIME(ResolveControlFlowIdentities, IR::ResolveControlFlowIdentities(ir));
	KYTY_PASS_TIME(RemoveIdentities, IR::RemoveIdentities(ir.blocks));
	KYTY_PASS_TIME(EliminateDeadCode, IR::EliminateDeadCode(ir.blocks));
	// Mask reads that fold can make select conditions and phis identical: fold until stable.
	for (int round = 0; round < 4 && KYTY_PASS_TIME(FoldLaneMasks, IR::FoldLaneMasks(ir)) != 0; round++) {
		KYTY_PASS_TIME(ConstantPropagationPass, IR::ConstantPropagationPass(ir.blocks, ir.wave_size));
		KYTY_PASS_TIME(ResolveControlFlowIdentities, IR::ResolveControlFlowIdentities(ir));
		KYTY_PASS_TIME(RemoveIdentities, IR::RemoveIdentities(ir.blocks));
		KYTY_PASS_TIME(EliminateDeadCode, IR::EliminateDeadCode(ir.blocks));
	}
	if (const auto folded = KYTY_PASS_TIME(SimplifyBoundedLoopRegisters, IR::SimplifyBoundedLoopRegisters(ir)); folded != 0) {
		LOGF("%s bounded-loop comparisons: hash=0x%016" PRIx64 " folded=%u\n",
		     GetDumpLabel(options), options.shader_hash, folded);
		KYTY_PASS_TIME(ConstantPropagationPass, IR::ConstantPropagationPass(ir.blocks, ir.wave_size));
		KYTY_PASS_TIME(ResolveControlFlowIdentities, IR::ResolveControlFlowIdentities(ir));
		KYTY_PASS_TIME(RemoveIdentities, IR::RemoveIdentities(ir.blocks));
		KYTY_PASS_TIME(EliminateDeadCode, IR::EliminateDeadCode(ir.blocks));
	}
	const auto read_lane_stats = KYTY_PASS_TIME(EliminateReadLane, IR::EliminateReadLane(ir, ir.wave_size));
	if (read_lane_stats.rewritten_reads != 0) {
		LOGF("%s read-lane elimination: reads=%" PRIu32 "\n", GetDumpLabel(options),
		     read_lane_stats.rewritten_reads);
		KYTY_PASS_TIME(ConstantPropagationPass, IR::ConstantPropagationPass(ir.blocks, ir.wave_size));
		KYTY_PASS_TIME(ResolveControlFlowIdentities, IR::ResolveControlFlowIdentities(ir));
		KYTY_PASS_TIME(RemoveIdentities, IR::RemoveIdentities(ir.blocks));
		KYTY_PASS_TIME(EliminateDeadCode, IR::EliminateDeadCode(ir.blocks));
	}
	if (const auto removed = KYTY_PASS_TIME(SimplifyLocalAddressStores, IR::SimplifyLocalAddressStores(ir)); removed != 0) {
		LOGF("%s local-address proof: hash=0x%016" PRIx64 " removed_global_stores=%u\n",
		     GetDumpLabel(options), options.shader_hash, removed);
		KYTY_PASS_TIME(EliminateDeadCode, IR::EliminateDeadCode(ir.blocks));
	}
	if (!LowerTessellationMemory(ir, options)) {
		if (!options.non_fatal) {
			EXIT("%s failed hash=0x%016" PRIx64 ": unsupported tessellation memory shape\n",
			     GetDumpLabel(options), options.shader_hash);
		}
		ReportGaveUp(options, phases, "unsupported tessellation memory shape");
		TranslateResult unsupported_result;
		unsupported_result.unsupported = true;
		return unsupported_result;
	}
	std::string cfg_dump;
	if (options.dump_ir) {
		cfg_dump = CFG::GraphToString(cfg);
		if (options.early_dump) {
			LOGF("%s native IR before resource tracking:\n%s", GetDumpLabel(options),
			     MakeIrDump(cfg_dump, ir).c_str());
		}
	}
	ir.bindless_images        = options.bindless_images;
	// KYTY_DUMP_IR_HASH=<hash>[,<hash>...] with KYTY_DUMP_GAVE_UP=<folder>: the IR before resource
	// tracking of those shaders, to <folder>/<stage>_<hash>_ir.txt (also for shaders that give up).
	if (const char* hashes = std::getenv("KYTY_DUMP_IR_HASH"); hashes != nullptr &&
	    std::string_view(hashes).find(fmt::format("{:016x}", options.shader_hash)) != std::string_view::npos) {
		if (const char* folder = std::getenv("KYTY_DUMP_GAVE_UP"); folder != nullptr) {
			const auto path = fmt::format("{}/{}_{:016x}_ir.txt", folder, StageName(options.stage),
			                              options.shader_hash);
			if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
				const auto dump = MakeIrDump(CFG::GraphToString(cfg), ir);
				std::fwrite(dump.data(), 1, dump.size(), file);
				std::fclose(file);
			}
			// Our own decoding with pcs (llvm-mc's gfx1030 decoding drifts on this code).
			const auto dec = fmt::format("{}/{}_{:016x}_dec.txt", folder, StageName(options.stage),
			                             options.shader_hash);
			if (FILE* file = std::fopen(dec.c_str(), "wb"); file != nullptr) {
				const auto text = Decoder::ProgramToString(decoded);
				std::fwrite(text.data(), 1, text.size(), file);
				std::fclose(file);
			}
			// The guest code too, for llvm-mc disassembly.
			const auto bin = fmt::format("{}/{}_{:016x}.bin", folder, StageName(options.stage),
			                             options.shader_hash);
			if (FILE* file = std::fopen(bin.c_str(), "wb"); file != nullptr) {
				std::fwrite(code.data(), sizeof(uint32_t), code.size(), file);
				std::fclose(file);
			}
		}
	}
	KYTY_PROFILER_PHASE(phases, "Shader: resource tracking", profiler::colors::Amber300);
	const auto tracking_begin = std::chrono::steady_clock::now();
	const bool tracked        = IR::TrackResources(ir, decoded, native_cfg);
	const auto tracking_ms    = ms_since(tracking_begin);
	// KYTY_DUMP_IR_HASH (with KYTY_DUMP_GAVE_UP): every scalar read after resource tracking, with
	// its memory info, to <folder>/<stage>_<hash>_mem.txt.
	if (const char* hashes = std::getenv("KYTY_DUMP_IR_HASH"); tracked && hashes != nullptr &&
	    std::string_view(hashes).find(fmt::format("{:016x}", options.shader_hash)) != std::string_view::npos) {
		if (const char* folder = std::getenv("KYTY_DUMP_GAVE_UP"); folder != nullptr) {
			const auto path = fmt::format("{}/{}_{:016x}_mem.txt", folder, StageName(options.stage),
			                              options.shader_hash);
			if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
				for (auto* block: ir.blocks) {
					for (auto& inst: *block) {
						if (inst.GetOpcode() != IR::ValueOpcode::ReadConstBuffer &&
						    inst.GetOpcode() != IR::ValueOpcode::LoadAddressU32) {
							continue;
						}
						const auto flags = inst.Flags<IR::MemoryFlags>();
						if (flags.index >= ir.memory_info.size()) {
							continue;
						}
						const auto& memory  = ir.memory_info[flags.index];
						const auto  dynamic = inst.NumArgs() > 1u ? inst.Arg(1).Resolve() : IR::Value(0u);
						const auto line = fmt::format(
						    "pc=0x{:04x} {} index={} kind={} resource={} offset={} planning_only={} uses={} dynamic={}\n",
						    flags.pc, IR::ValueOpcodeName(inst.GetOpcode()), flags.index,
						    static_cast<uint32_t>(memory.kind), memory.resource, memory.offset,
						    memory.planning_only ? 1 : 0, inst.Uses().size(),
						    dynamic.IsImmediate() && dynamic.GetType() == IR::Type::U32 ? fmt::format("imm 0x{:x}", dynamic.U32()) : std::string("value"));
						std::fputs(line.c_str(), file);
					}
				}
				std::fclose(file);
			}
		}
	}
	// The IR passes after translation, resource tracking among them, which the phases above miss.
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " IR passes tracking_ms=%" PRIu64
	     " passes_ms=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash, tracking_ms,
	     ms_since(passes_begin), phase_ms());
	if (pass_times_enabled && ms_since(passes_begin) >= 1000u) {
		std::string list;
		for (const auto& [name, ms]: pass_times) {
			list += fmt::format(" {}={}", name, ms);
		}
		std::fprintf(stderr, "[pass-times] %016llx%s\n", static_cast<unsigned long long>(options.shader_hash),
		             list.c_str());
	}
#undef KYTY_PASS_TIME
	if (!tracked && options.non_fatal) {
		DumpGaveUpCode(options, code);
		ReportGaveUp(options, phases, "resource tracking failed");
		LOGF("%s gave up hash=0x%016" PRIx64 ": resource tracking failed\n", GetDumpLabel(options),
		     options.shader_hash);
		if (Config::GraphicsDebugDumpEnabled()) {
			const auto path = Config::GetShaderLogFolder() / "rejected" /
			                  fmt::format("{}_{:016x}.ir", StageName(options.stage), options.shader_hash);
			Common::File::CreateDirectories(path.parent_path());
			Common::File file(path);
			if (!file.IsInvalid()) {
				const auto dump = MakeIrDump(CFG::GraphToString(cfg), ir);
				file.Write(dump.data(), dump.size());
			}
		}
		TranslateResult unsupported_result;
		unsupported_result.unsupported = true;
		unsupported_result.decoded_dump = std::move(decoded_dump);
		return unsupported_result;
	}
	// A subgroup larger than a host mesh workgroup runs in passes, cut at its barriers.
	if (options.stage == ShaderType::Mesh && options.input_info.vertex != nullptr &&
	    options.input_info.vertex->mesh.passes > 1u) {
		if (const auto* reason = Spirv::MeshPassesUnsupported(ir); reason != nullptr) {
			if (!options.non_fatal) {
				EXIT("%s failed hash=0x%016" PRIx64 ": mesh passes: %s\n",
				     GetDumpLabel(options), options.shader_hash, reason);
			}
			DumpGaveUpCode(options, code);
			ReportGaveUp(options, phases, reason);
			LOGF("%s gave up hash=0x%016" PRIx64 ": %u mesh passes, but %s\n",
			     GetDumpLabel(options), options.shader_hash, options.input_info.vertex->mesh.passes,
			     reason);
			TranslateResult unsupported_result;
			unsupported_result.unsupported = true;
			return unsupported_result;
		}
	}
	// A bound on what the driver is given: a ray-tracing shader with every hit shader inlined
	// (e6d76d24, 50,000 to 130,000 words) becomes millions of SPIR-V instructions, and the NVIDIA
	// driver took more than 30 GB compiling one before the test closed it. The largest of the
	// game's other shaders is under 700,000 SPIR-V words (14 to 21 per IR instruction).
	// KYTY_MAX_IR_INSTRUCTIONS (default 200000, a few million SPIR-V words); 0: no bound.
	if (options.non_fatal) {
		static const uint64_t max_instructions = [] {
			const char* value = std::getenv("KYTY_MAX_IR_INSTRUCTIONS");
			return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {200000};
		}();
		if (max_instructions != 0) {
			uint64_t instructions = 0;
			for (const auto* block: ir.blocks) {
				instructions += block->Instructions().size();
			}
			if (instructions > max_instructions) {
				const auto reason = fmt::format("{} IR instructions, above the {}-instruction bound (KYTY_MAX_IR_INSTRUCTIONS)",
				                                instructions, max_instructions);
				std::fprintf(stderr, "[gave-up-reason] %016llx %s\n",
				             static_cast<unsigned long long>(options.shader_hash), reason.c_str());
				DumpGaveUpCode(options, code);
				ReportGaveUp(options, phases, reason.c_str());
				TranslateResult unsupported_result;
				unsupported_result.unsupported             = true;
				unsupported_result.input_dependent_refusal = true;
				return unsupported_result;
			}
		}
	}
	TranslateResult result;
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(decoded_dump);
		result.cfg_dump     = std::move(cfg_dump);
	}
	return result;
}

CompileResult CompileProgram(TranslateResult translated, const CompileOptions& options,
                             const IR::ResourceSpecialization& specialization,
                             uint32_t push_data_start_dword) {
	Profiler::Phases phases;
	phases.SetText("%s 0x%016" PRIx64, StageName(options.stage), options.shader_hash);
	KYTY_PROFILER_PHASE(phases, "Shader: specialize and bind", profiler::colors::Green100);
	const auto emit_begin = std::chrono::steady_clock::now();
	auto& ir = translated.program;
	IR::ApplyResourceSpecialization(ir, specialization);
	// The resource plan owns host descriptor evaluation now. Keep only dependencies consumed
	// by GPU memory operations; bound descriptor dwords must not retain shader instructions.
	for (auto& inst: ir.value_storage) {
		inst.Invalidate();
	}
	for (auto* block: ir.blocks) {
		for (auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uint32_t first = 0;
			if (op == IR::ValueOpcode::GetBufferResource) {
				if (std::ranges::any_of(inst.Uses(), [&](const IR::Use& use) {
					return ir.memory_info[use.user->Flags<IR::MemoryFlags>().index].kind ==
					       IR::ResourceKind::IndirectBuffer;
				})) {
					continue;
				}
				const auto resource = inst.Flags<uint32_t>();
				first               = resource < ir.info.buffers.size() &&
				                              ir.info.buffers[resource].indirect_root == resource
				                          ? 1u
				                          : 0u;
			} else if (op == IR::ValueOpcode::GetImageResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.images.size() &&
				                ir.info.images[resource].indirect_root == resource ? 1u : 0u;
			} else if (op == IR::ValueOpcode::GetSamplerResource) {
				// A bindless sampler handle carries its GPU-computed key in dword 0 (the shader
				// indexes the mirrored sampler heap with it, BindlessSamplerSlot), as a bindless
				// image handle does. Zeroing it made every bindless sample use heap record 0.
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.samplers.size() && ir.info.samplers[resource].bindless ? 1u : 0u;
			} else {
				continue;
			}
			// A buffer bounded by its own count (gpu_records_ssa) keeps dword 2.
			const bool keep_records =
			    op == IR::ValueOpcode::GetBufferResource &&
			    inst.Flags<uint32_t>() < ir.info.buffers.size() &&
			    ir.info.buffers[inst.Flags<uint32_t>()].gpu_records_ssa;
			for (size_t index = first; index < inst.NumArgs(); index++) {
				if (keep_records && index == 2u) {
					continue;
				}
				inst.SetArg(index, IR::Value(0u));
			}
		}
	}
	ir.value_storage.clear();
	IR::RemoveIdentities(ir.blocks);
	IR::EliminateDeadCode(ir.blocks);

	const auto translation_variant   = ir.info.translation_variant; // v446h: kept across the collection
	const auto static_buffer_formats = ir.info.static_buffer_formats;
	IR::CollectShaderInfo(ir, options.input_info);
	ir.info.translation_variant   = translation_variant;
	ir.info.static_buffer_formats = static_buffer_formats;
	IR::AllocateBindings(ir, push_data_start_dword,
	                     ir.stage == ShaderType::Compute && options.input_info.compute != nullptr &&
	                         options.input_info.compute->lds_storage,
	                     ir.stage != ShaderType::Compute && ir.stage != ShaderType::Pixel &&
	                         options.input_info.vertex != nullptr &&
	                         options.input_info.vertex->clip_space.enabled);
	std::string ir_dump;
	if (options.dump_ir) {
		ir_dump = MakeIrDump(translated.cfg_dump, ir);
		if (options.early_dump) {
			LOGF("%s native IR and bindings (early):\n%s", GetDumpLabel(options), ir_dump.c_str());
		}
	}

	KYTY_PROFILER_PHASE(phases, "Shader: SPIR-V emit", profiler::colors::CyanA700);
	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " SPIR-V EmitProgram\n",
	     GetDumpLabel(options), StageName(ir.stage), ir.shader_hash);
	std::vector<uint32_t>              spirv;
	std::vector<std::vector<uint32_t>> rt_callable_spirv;
	std::vector<std::vector<uint32_t>> rt_callable_records;
	if (ir.rt_modules.empty()) {
		spirv = Spirv::EmitProgram(ir, options.input_info);
	} else {
		// A merged callable split: each module from its range of the blocks, with the program's one
		// set of resources and bindings.
		auto all_blocks = std::move(ir.blocks);
		auto all_info   = std::move(ir.block_info);
		ir.layout_blocks = &all_blocks;
		const auto merged_scratch = ir.scratch_dwords;
		for (const auto& module: ir.rt_modules) {
			ir.blocks.assign(all_blocks.begin() + module.first_block,
			                 all_blocks.begin() + module.first_block + module.block_count);
			ir.block_info.assign(all_info.begin() + module.first_block,
			                     all_info.begin() + module.first_block + module.block_count);
			ir.rt_role  = module.role;
			ir.rt_code_hash = module.code_hash;
			// v442: a module's scratch is its own (function-local).
			ir.scratch_dwords = ir.bindings.rt_fixed ? module.scratch_dwords : merged_scratch;
			// v443: a shared callable's shader record (IR/RtCallable.h) comes back with its module.
			std::vector<uint32_t> record;
			auto module_spirv = Spirv::EmitProgram(ir, options.input_info, &record);
			if (module.role == IR::RtRole::Raygen) {
				spirv = std::move(module_spirv);
			} else {
				rt_callable_spirv.push_back(std::move(module_spirv));
				rt_callable_records.push_back(std::move(record));
			}
		}
		ir.layout_blocks  = nullptr;
		ir.scratch_dwords = merged_scratch;
		ir.blocks  = std::move(all_blocks);
		ir.block_info = std::move(all_info);
		ir.rt_role = IR::RtRole::Raygen;
	}
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " SPIR-V EmitProgram words=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(ir.stage), ir.shader_hash,
	     static_cast<uint64_t>(spirv.size()),
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - emit_begin)
	                               .count()));
	CompileResult result;
	result.spirv   = std::move(spirv);
	result.rt_callable_spirv = std::move(rt_callable_spirv);
	result.rt_callable_records = std::move(rt_callable_records);
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(translated.decoded_dump);
		result.ir_dump      = std::move(ir_dump);
	}
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler
