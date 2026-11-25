#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <fmt/format.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::Decoder {
namespace {

constexpr uint32_t MaxFunctionWords  = 4096;
// v439: 262,144 (was 131,072). The ray-tracing shaders behind the glass reflections expand past
// 131,072 words with every hit shader they can call (e6d76d24: up to ~129k, 29be8047: more than
// 131k); since v438zz translates such programs in seconds, the bound only keeps runaway
// expansions out. KYTY_MAX_EXPANDED_WORDS=<n> overrides it.
uint32_t MaxExpandedWordsLimit() {
	static const uint32_t limit = [] {
		const char* value = std::getenv("KYTY_MAX_EXPANDED_WORDS");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 262144ul;
		return static_cast<uint32_t>(std::clamp<unsigned long>(n, 4096ul, 1ul << 22u));
	}();
	return limit;
}
constexpr uint64_t MaxCallTableBytes = 16u * 1024u * 1024u;
constexpr size_t   MaxCallTargets    = 256;

bool IsCall(uint32_t word) {
	return (word & 0xff80ff00u) == 0xbe802100u && ((word >> 16u) & 127u) != 125u;
}

uint32_t DestinationWords(const Instruction& inst, uint32_t wave_size) {
	const auto name = magic_enum::enum_name(inst.opcode);
	if (name.starts_with("V_CMP")) return wave_size / 32u;
	const bool pair = name.find("64") != std::string_view::npos;
	return std::max(inst.data_dwords, pair ? 2u : 1u);
}

bool Writes(const Instruction& inst, uint32_t reg, uint32_t wave_size) {
	if (IsCall(inst.raw[0])) {
		const auto dst = (inst.raw[0] >> 16u) & 127u;
		return reg == dst || reg == dst + 1u;
	}
	return (inst.dst.kind == OperandKind::Sgpr && reg >= inst.dst.reg &&
	        reg < inst.dst.reg + DestinationWords(inst, wave_size)) ||
	       (inst.dst2.kind == OperandKind::Sgpr && reg >= inst.dst2.reg &&
	        reg < inst.dst2.reg + wave_size / 32u);
}

struct Function {
	std::vector<uint32_t>    code;
	std::vector<Instruction> instructions;
};

bool ReadFunction(uint64_t address, uint32_t return_reg, const ShaderCodeReader& read,
                  Function& function, std::string& reason, uint32_t wave_size) {
	uint32_t next            = 0;
	uint32_t furthest_target = 0;
	bool     returned        = false;
	while (next + MaxInstructionRawWords <= MaxFunctionWords) {
		while (function.code.size() < next + MaxInstructionRawWords) {
			std::array<uint32_t, 64> chunk {};
			if (!read(address + function.code.size() * 4u, chunk)) {
				reason = "shader function code is not mapped";
				return false;
			}
			function.code.insert(function.code.end(), chunk.begin(), chunk.end());
		}
		if (GetInstructionFamily(function.code[next]) == Family::Unknown) {
			reason = "shader function has an unknown instruction encoding";
			return false;
		}
		Instruction inst;
		DecodeInstruction(function.code, next, inst);
		if (IsCall(inst.raw[0]) || inst.opcode == Opcode::S_ENDPGM ||
		    inst.opcode == Opcode::UNSUPPORTED || Writes(inst, return_reg, wave_size) ||
		    Writes(inst, return_reg + 1u, wave_size)) {
			reason = fmt::format("shader function needs nested calls or unsupported instructions "
			                     "(0x{:x} + 0x{:x}: {:08x} {})",
			                     address, next * 4u, inst.raw[0],
			                     IsCall(inst.raw[0]) ? "call"
			                     : inst.opcode == Opcode::S_ENDPGM ? "s_endpgm"
			                     : inst.opcode == Opcode::UNSUPPORTED ? "unsupported"
			                                                          : "writes the return pair");
			return false;
		}
		function.instructions.push_back(inst);
		next += inst.word_count;
		if (IsDirectBranch(inst.opcode)) {
			furthest_target = std::max(furthest_target, inst.branch_target);
		}
		if (inst.opcode == Opcode::S_SETPC_B64) {
			if (inst.src0.kind != OperandKind::Sgpr || inst.src0.reg != return_reg) {
				reason = "shader function returns through a different scalar pair";
				return false;
			}
			returned = true;
			if (next * 4u > furthest_target) break;
		}
	}
	if (!returned || function.instructions.back().opcode != Opcode::S_SETPC_B64) {
		reason = "shader function exceeds the bounded code range";
		return false;
	}
	function.code.resize(next);
	std::set<uint32_t> starts;
	for (const auto& inst: function.instructions)
		starts.insert(inst.pc);
	for (const auto& inst: function.instructions) {
		if (IsDirectBranch(inst.opcode) && !starts.contains(inst.branch_target)) {
			reason = "shader function branches outside its instruction range";
			return false;
		}
	}
	return true;
}

struct CallTarget {
	uint64_t address;
	Function function;
};

struct Call {
	uint32_t                source_reg;
	uint32_t                return_reg;
	bool                    dynamic;
	std::vector<CallTarget> targets;
};

bool IsScalarLoad(Opcode opcode) {
	return opcode == Opcode::S_LOAD_DWORD || opcode == Opcode::S_LOAD_DWORDX2 ||
	       opcode == Opcode::S_LOAD_DWORDX4 || opcode == Opcode::S_LOAD_DWORDX8 ||
	       opcode == Opcode::S_LOAD_DWORDX16;
}

bool IsScalarBufferLoad(Opcode opcode) {
	return opcode == Opcode::S_BUFFER_LOAD_DWORD || opcode == Opcode::S_BUFFER_LOAD_DWORDX2 ||
	       opcode == Opcode::S_BUFFER_LOAD_DWORDX4 || opcode == Opcode::S_BUFFER_LOAD_DWORDX8 ||
	       opcode == Opcode::S_BUFFER_LOAD_DWORDX16;
}

std::optional<uint32_t> Immediate(const Operand& operand) {
	if (operand.kind == OperandKind::Null) return 0;
	if (operand.kind == OperandKind::LiteralConstant ||
	    operand.kind == OperandKind::IntegerInlineConstant)
		return operand.value;
	return {};
}

std::optional<uint32_t> ScalarIndex(const Operand& operand) {
	switch (operand.kind) {
		case OperandKind::Sgpr: return operand.reg;
		case OperandKind::VccLo: return 106;
		case OperandKind::VccHi: return 107;
		case OperandKind::M0: return 124;
		case OperandKind::ExecLo: return 126;
		case OperandKind::ExecHi: return 127;
		default: return {};
	}
}

void EmitMove(std::vector<uint32_t>& code, uint32_t reg, uint32_t value) {
	code.push_back(0xbe8003ffu | (reg << 16u)); // S_MOV_B32 s[reg], literal
	code.push_back(value);
}

bool EmitOriginalPc(std::vector<uint32_t>& code, const Instruction& inst, uint64_t base,
                    std::string& reason) {
	if (inst.dst.kind == OperandKind::Null) {
		code.push_back(0xbf800000u); // S_NOP retains an instruction boundary.
		return true;
	}
	const auto reg = (inst.raw[0] >> 16u) & 127u;
	if (!((inst.dst.kind == OperandKind::Sgpr && inst.dst.reg <= 105u) ||
	      inst.dst.kind == OperandKind::VccLo || inst.dst.kind == OperandKind::M0 ||
	      inst.dst.kind == OperandKind::ExecLo)) {
		reason = "shader GETPC destination is not a supported scalar pair";
		return false;
	}
	// Expansion changes instruction positions, but GETPC must still address the
	// original shader's constants/code. MOV does not change SCC, matching GETPC.
	const auto pc = base + inst.pc + 4u;
	EmitMove(code, reg, static_cast<uint32_t>(pc));
	EmitMove(code, reg + 1u, static_cast<uint32_t>(pc >> 32u));
	return true;
}

using Branches = std::vector<std::pair<uint32_t, uint32_t>>;

// v439: long edges of large expansions in one pass. Island sites every IslandSpacing words (at
// instruction boundaries); every long edge takes one slot in each island between its source and
// target, and hops from slot to slot; each island starts with a branch over its slots. Before
// v439 islands were inserted one at a time, each insertion moving the whole program and
// rescanning every edge: seconds on the GPU thread for the glass's ray-tracing shaders (hundreds
// of callees, each returning across all the others). False when an edge would still be out of
// range (the caller then uses the incremental method).
bool RelocateBranchesBatched(std::vector<uint32_t>& code, Branches& branches) {
	constexpr int64_t  Range         = std::numeric_limits<int16_t>::max();
	constexpr uint32_t IslandSpacing = 12288;
	constexpr int64_t  LongEdge      = 16384;
	std::vector<size_t> long_edges;
	for (size_t i = 0; i < branches.size(); i++) {
		const auto delta = int64_t(branches[i].second) - int64_t(branches[i].first) - 1;
		if (delta > LongEdge || delta < -LongEdge) long_edges.push_back(i);
	}
	if (long_edges.empty()) {
		for (const auto [at, target]: branches) {
			const auto delta = int64_t(target) - int64_t(at) - 1;
			if (delta > Range || delta < -Range - 1) return false;
			code[at] = (code[at] & 0xffff0000u) | (static_cast<uint32_t>(delta) & 0xffffu);
		}
		return true;
	}
	// Island sites: the first instruction boundary at or after each multiple of the spacing.
	std::vector<uint32_t> sites;
	{
		uint32_t next_site = IslandSpacing;
		for (uint32_t word = 0; word < code.size();) {
			if (word >= next_site) {
				sites.push_back(word);
				next_site = word + IslandSpacing;
			}
			Instruction inst;
			DecodeInstruction(code, word, inst);
			if (inst.word_count == 0 || inst.word_count > code.size() - word) return false;
			word += inst.word_count;
		}
	}
	// The sites each long edge passes (strictly between source and target), and the slots.
	std::vector<std::vector<size_t>> slots_at(sites.size()); // per site: long-edge indices
	for (const auto e: long_edges) {
		const auto [at, target] = branches[e];
		const auto lo = std::min(at, target), hi = std::max(at, target);
		auto first = std::upper_bound(sites.begin(), sites.end(), lo);
		for (auto it = first; it != sites.end() && *it < hi; ++it) {
			// A site exactly at the target needs no slot (the target is its first word).
			slots_at[static_cast<size_t>(it - sites.begin())].push_back(e);
		}
	}
	// New positions: an island of 1 + k words before each site's instruction.
	std::vector<uint32_t> shift(sites.size() + 1u, 0u); // words inserted before site i
	for (size_t i = 0; i < sites.size(); i++) {
		const auto k = slots_at[i].size();
		shift[i + 1u] = shift[i] + (k == 0 ? 0u : static_cast<uint32_t>(k + 1u));
	}
	const auto moved = [&](uint32_t old) {
		const auto passed = static_cast<size_t>(std::upper_bound(sites.begin(), sites.end(), old) - sites.begin());
		return old + shift[passed];
	};
	// A branch target at a site lands on the island's bypass, which continues to the site.
	std::vector<uint32_t> result;
	result.reserve(code.size() + shift.back());
	std::vector<std::vector<uint32_t>> slot_pos(sites.size()); // new position of each slot
	size_t                             site = 0;
	for (uint32_t word = 0; word < code.size(); word++) {
		if (site < sites.size() && word == sites[site]) {
			const auto k = slots_at[site].size();
			if (k != 0) {
				result.push_back(0xbf820000u | static_cast<uint32_t>(k)); // S_BRANCH over the slots
				for (size_t j = 0; j < k; j++) {
					slot_pos[site].push_back(static_cast<uint32_t>(result.size()));
					result.push_back(0xbf820000u);
				}
			}
			++site;
		}
		result.push_back(code[word]);
	}
	const auto patch = [&](uint32_t at, uint32_t target) {
		const auto delta = int64_t(target) - int64_t(at) - 1;
		if (delta > Range || delta < -Range - 1) return false;
		result[at] = (result[at] & 0xffff0000u) | (static_cast<uint32_t>(delta) & 0xffffu);
		return true;
	};
	// Short edges keep their target (moved); a target at a site goes to the site's first word
	// after its island, as before (the island's bypass would also get there).
	std::vector<uint8_t> is_long(branches.size(), 0u);
	for (const auto e: long_edges) is_long[e] = 1u;
	for (size_t i = 0; i < branches.size(); i++) {
		if (is_long[i] != 0u) continue;
		if (!patch(moved(branches[i].first), moved(branches[i].second))) return false;
	}
	// Long edges: source -> slot in each passed island (in the direction of travel) -> target.
	for (const auto e: long_edges) {
		const auto [at, target] = branches[e];
		std::vector<uint32_t> hops;
		const bool forward = target > at;
		for (size_t i = 0; i < sites.size(); i++) {
			const auto& list = slots_at[i];
			const auto  it   = std::find(list.begin(), list.end(), e);
			if (it != list.end()) hops.push_back(slot_pos[i][static_cast<size_t>(it - list.begin())]);
		}
		if (!forward) std::reverse(hops.begin(), hops.end());
		uint32_t from = moved(at);
		for (const auto hop: hops) {
			if (!patch(from, hop)) return false;
			from = hop;
		}
		if (!patch(from, moved(target))) return false;
	}
	code = std::move(result);
	return true;
}

bool RelocateBranches(std::vector<uint32_t>& code, Branches& branches, std::string& reason) {
	{
		auto copy = code;
		if (RelocateBranchesBatched(copy, branches)) {
			if (copy.size() > MaxExpandedWordsLimit()) {
				reason = "expanded shader branch islands exceed the bounded code size";
				return false;
			}
			code = std::move(copy);
			return true;
		}
	}
	// Keep relocation symbolic until every function has been inserted. A long
	// edge uses guarded S_BRANCH islands; ordinary fallthrough skips the island.
	// Unlike GETPC/ADD/SETPC stubs, these do not clobber guest SGPRs or SCC.
	for (size_t index = 0; index < branches.size();) {
		const auto [at, target] = branches[index];
		const auto delta        = int64_t(target) - int64_t(at) - 1;
		if (delta >= std::numeric_limits<int16_t>::min() &&
		    delta <= std::numeric_limits<int16_t>::max()) {
			++index;
			continue;
		}
		if (code.size() + 2 > MaxExpandedWordsLimit() || branches.size() > MaxExpandedWordsLimit()) {
			reason = "expanded shader branch islands exceed the bounded code size";
			return false;
		}
		const auto desired   = delta > 0 ? at + 16384u : at - 16384u;
		uint32_t   insertion = 0;
		for (uint32_t word = 0; word <= desired && word < code.size();) {
			insertion = word;
			Instruction inst;
			DecodeInstruction(code, word, inst);
			if (inst.word_count == 0 || inst.word_count > code.size() - word) {
				reason = "expanded shader has an invalid branch-island boundary";
				return false;
			}
			word += inst.word_count;
		}
		if (insertion <= std::min(at, target) || insertion >= std::max(at, target)) {
			reason = "cannot place a branch island between expanded instructions";
			return false;
		}
		code.insert(code.begin() + insertion, {0xbf820000u, 0xbf820000u});
		for (auto& [source, destination]: branches) {
			if (source >= insertion) source += 2;
			if (destination >= insertion) destination += 2;
		}
		const auto onward      = branches[index].second;
		branches[index].second = insertion + 1u;
		branches.emplace_back(insertion, insertion + 2u); // Fallthrough bypass.
		branches.emplace_back(insertion + 1u, onward);    // Long edge continuation.
		// Insertion may lengthen an earlier edge beyond its range, so revisit it.
		index = 0;
	}
	for (const auto [at, target]: branches) {
		const auto delta = int64_t(target) - int64_t(at) - 1;
		code[at]         = (code[at] & 0xffff0000u) | (static_cast<uint32_t>(delta) & 0xffffu);
	}
	return true;
}

// The decoded shader: everything the expansion needs that depends only on its code.
struct ShaderAnalysis {
	std::vector<uint32_t> code;
	Program               program;
	std::set<uint32_t>    labels;
	bool                  has_calls = false;
	bool                  has_setpc = false;
};

void AnalyzeShader(std::span<const uint32_t> code, ShaderAnalysis& analysis) {
	analysis.code.assign(code.begin(), code.end());
	if (std::none_of(code.begin(), code.end(), IsCall)) return;
	DecodeProgram(analysis.code, analysis.program);
	for (const auto& inst: analysis.program.instructions) {
		analysis.has_calls |= IsCall(inst.raw[0]);
		if (IsDirectBranch(inst.opcode)) analysis.labels.insert(inst.branch_target);
		analysis.has_setpc |= inst.opcode == Opcode::S_SETPC_B64;
	}
}

// One call instruction and every address it can reach.
struct CallSite {
	uint32_t              pc         = 0;
	uint32_t              source_reg = 0;
	uint32_t              return_reg = 0;
	bool                  dynamic    = false;
	std::vector<uint64_t> addresses; // ascending

	bool operator==(const CallSite&) const = default;
};

// Resolves each call's targets from the user data and the memory holding call tables. Which
// registers it consults depends on the code alone; their values reach the result only through
// the targets.
bool ResolveCalls(const ShaderAnalysis& analysis, uint64_t base,
                  std::span<const uint32_t> user_data, const ShaderCodeReader& read,
                  uint32_t wave_size, std::vector<CallSite>& sites,
                  std::set<uint32_t>& dependencies, std::string& reason) {
	const auto& program = analysis.program;
	const auto& labels  = analysis.labels;
	const auto resolve = [&](auto&& self, uint32_t reg, size_t before, size_t first, uint32_t depth,
	                         bool load_memory = false) -> std::optional<uint32_t> {
		if (reg >= 128 || depth > 32) return {};
		const bool initial =
		    reg < user_data.size() &&
		    std::none_of(
		        program.instructions.begin(), program.instructions.begin() + before,
		        [&](const auto& inst) { return Writes(inst, reg, wave_size); });
		if (initial) {
			dependencies.insert(reg);
			return user_data[reg];
		}
		for (size_t i = before; i > first;) {
			const auto& inst = program.instructions[--i];
			if (IsDirectBranch(inst.opcode) || IsCall(inst.raw[0])) return {};
			if (Writes(inst, reg, wave_size)) {
				if (inst.opcode == Opcode::S_GETPC_B64 && inst.dst.kind == OperandKind::Sgpr) {
					const auto pc = base + inst.pc + 4u;
					return static_cast<uint32_t>(pc >> ((reg - inst.dst.reg) * 32u));
				}
				if (load_memory && IsScalarLoad(inst.opcode) &&
				    inst.src0.kind == OperandKind::Sgpr) {
					const auto low    = self(self, inst.src0.reg, i, first, depth + 1, true);
					const auto high   = self(self, inst.src0.reg + 1u, i, first, depth + 1, true);
					const auto offset = inst.src1.kind == OperandKind::Sgpr
					                        ? self(self, inst.src1.reg, i, first, depth + 1, true)
					                        : Immediate(inst.src1);
					if (!low || !high || !offset) return {};
					const uint64_t          address     = (uint64_t {*high} << 32u) | *low;
					const uint32_t          byte_offset = *offset + inst.offset;
					std::array<uint32_t, 1> value {};
					if (!read(address + byte_offset + (reg - inst.dst.reg) * 4u, value)) return {};
					return value[0];
				}
				if ((inst.opcode != Opcode::S_MOV_B32 && inst.opcode != Opcode::S_MOV_B64) ||
				    inst.dst.kind != OperandKind::Sgpr)
					return {};
				if (inst.src0.kind == OperandKind::Sgpr) {
					return self(self, inst.src0.reg + reg - inst.dst.reg, i, first, depth + 1,
					            load_memory);
				}
				if (inst.opcode == Opcode::S_MOV_B32 &&
				    (inst.src0.kind == OperandKind::LiteralConstant ||
				     inst.src0.kind == OperandKind::IntegerInlineConstant))
					return inst.src0.value;
				return {};
			}
			if (labels.contains(inst.pc)) return {};
		}
		return {};
	};
	// Follow only straight-line scalar copies back to one buffer load. The
	// runtime load and its data arguments are retained in the expanded program.
	const auto origin = [&](auto&& self, uint32_t reg, size_t before, size_t first,
	                        uint32_t depth) -> std::optional<std::pair<size_t, uint32_t>> {
		if (reg >= 105u || depth > 32) return {};
		for (size_t i = before; i > first;) {
			const auto& inst = program.instructions[--i];
			if (IsDirectBranch(inst.opcode) || IsCall(inst.raw[0])) return {};
			if (Writes(inst, reg, wave_size)) {
				if (IsScalarBufferLoad(inst.opcode)) return std::pair {i, reg - inst.dst.reg};
				if ((inst.opcode == Opcode::S_MOV_B32 || inst.opcode == Opcode::S_MOV_B64) &&
				    inst.src0.kind == OperandKind::Sgpr)
					return self(self, inst.src0.reg + reg - inst.dst.reg, i, first, depth + 1);
				return {};
			}
			if (labels.contains(inst.pc)) return {};
		}
		return {};
	};
	for (size_t i = 0; i < program.instructions.size(); i++) {
		const auto& inst = program.instructions[i];
		if (!IsCall(inst.raw[0])) continue;
		const auto src   = inst.raw[0] & 255u;
		const auto dst   = (inst.raw[0] >> 16u) & 127u;
		size_t     first = i;
		while (first > 0 && !labels.contains(program.instructions[first].pc) &&
		       !IsDirectBranch(program.instructions[first - 1].opcode))
			--first;
		const auto low  = resolve(resolve, src, i, first, 0);
		const auto high = resolve(resolve, src + 1u, i, first, 0);
		if (src >= 105u || dst >= 105u) {
			reason = "shader call uses a special scalar register pair";
			return false;
		}
		const bool         dynamic = !low || !high;
		std::set<uint64_t> addresses;
		if (!dynamic) {
			addresses.insert((uint64_t {*high} << 32u) | *low);
		} else {
			const auto lo = origin(origin, src, i, first, 0);
			const auto hi = origin(origin, src + 1u, i, first, 0);
			if (!lo || !hi || lo->first != hi->first || hi->second != lo->second + 1u) {
				reason = "shader call target is not a scalar buffer table load";
				return false;
			}
			const auto& load = program.instructions[lo->first];
			if (load.src0.kind != OperandKind::Sgpr) {
				reason = "shader call table descriptor is not a scalar register tuple";
				return false;
			}
			ShaderBufferResource descriptor;
			for (uint32_t word = 0; word < 4; ++word) {
				const auto value =
				    resolve(resolve, load.src0.reg + word, lo->first, first, 0, true);
				if (!value) {
					reason = "shader call table descriptor is not statically readable";
					return false;
				}
				descriptor.fields[word] = *value;
			}
			// Prove the dynamic byte offset's alignment from its last definition.
			// This bounds all possible target slots without assuming a runtime index.
			uint32_t   shift      = 0;
			const auto offset_reg = ScalarIndex(load.src1);
			for (size_t at = lo->first; at > first;) {
				const auto& def = program.instructions[--at];
				if (IsDirectBranch(def.opcode) || IsCall(def.raw[0])) break;
				const auto dst_reg  = ScalarIndex(def.dst);
				const auto dst2_reg = ScalarIndex(def.dst2);
				if (offset_reg && dst_reg && *offset_reg >= *dst_reg &&
				    *offset_reg < *dst_reg + DestinationWords(def, wave_size)) {
					if (def.opcode == Opcode::S_LSHL_B32 && *offset_reg == *dst_reg) {
						if (const auto amount = Immediate(def.src1)) shift = *amount & 31u;
					}
					break;
				}
				if (offset_reg && dst2_reg && *offset_reg >= *dst2_reg &&
				    *offset_reg < *dst2_reg + wave_size / 32u)
					break;
				if (labels.contains(def.pc)) break;
			}
			const uint64_t size = descriptor.GetSize();
			if (shift < 3u || shift > 16u || size == 0 || size > MaxCallTableBytes ||
			    descriptor.Base48() == 0 || descriptor.Type() != 0 || descriptor.SwizzleEnabled() ||
			    descriptor.AddTid() || (descriptor.Base48() & 3u) != 0 || (load.offset & 3u) != 0) {
				reason = "shader call table needs unsupported bounds or addressing";
				return false;
			}
			std::vector<uint32_t> table(size / 4u);
			if (!read(descriptor.Base48(), table)) {
				reason = "shader call table is not fully mapped";
				return false;
			}
			const uint64_t step       = uint64_t {1} << shift;
			const uint64_t first_slot = (load.offset % step) + lo->second * 4u;
			for (uint64_t offset = first_slot; offset + 8u <= size; offset += step) {
				const uint64_t address =
				    table[offset / 4u] | (uint64_t {table[offset / 4u + 1u]} << 32u);
				if (address == 0) continue; // A runtime null target takes the explicit trap path.
				addresses.insert(address);
				if (addresses.size() > MaxCallTargets) {
					reason = "shader call table exceeds the bounded target count";
					return false;
				}
			}
		}
		sites.push_back({inst.pc, src, dst, dynamic, {addresses.begin(), addresses.end()}});
	}
	return true;
}

// Reads the callees and emits the expanded program: a function of the code, the resolved
// targets, the consulted registers and the callee code.
bool BuildExpansion(const ShaderAnalysis& analysis, uint64_t base,
                    const std::vector<CallSite>& sites, const std::set<uint32_t>& dependencies,
                    const ShaderCodeReader& read, uint32_t wave_size,
                    std::vector<uint32_t>& expanded, std::string& reason) {
	const auto&              program = analysis.program;
	std::map<uint32_t, Call> calls;
	for (const auto& site: sites) {
		Call call {site.source_reg, site.return_reg, site.dynamic, {}};
		for (const auto address: site.addresses) {
			CallTarget target {address, {}};
			if ((address & 3u) != 0 ||
			    !ReadFunction(address, site.return_reg, read, target.function, reason, wave_size)) {
				if (reason.empty()) reason = "shader call target is not DWORD aligned";
				return false;
			}
			call.targets.push_back(std::move(target));
		}
		calls.emplace(site.pc, std::move(call));
	}
	const bool has_backedge =
	    std::any_of(program.instructions.begin(), program.instructions.end(), [](const auto& inst) {
		    return IsDirectBranch(inst.opcode) && inst.branch_target <= inst.pc;
	    });
	if (has_backedge) {
		// v439: a write of a register a call target was resolved from matters only if a call
		// site can run after it (a loop around both). The registers were not written before the
		// calls in program order (ResolveCalls), so the question is whether a path leads from
		// such a write back to a call: a walk over the main program's instructions (fallthrough
		// and direct branches; calls return to the next instruction). Before v439 any write in a
		// shader with a loop refused the expansion (384f4708, fc0537b7: the glass's ray tracing).
		const auto& insts = program.instructions;
		std::unordered_map<uint32_t, size_t> index_of_pc;
		for (size_t i = 0; i < insts.size(); i++) index_of_pc.emplace(insts[i].pc, i);
		std::vector<uint8_t> is_call(insts.size(), 0u);
		for (size_t i = 0; i < insts.size(); i++) is_call[i] = calls.contains(insts[i].pc) ? 1u : 0u;
		const auto successors = [&](size_t i, auto&& visit) {
			const auto& inst = insts[i];
			if (inst.opcode == Opcode::S_ENDPGM) return;
			if (IsDirectBranch(inst.opcode)) {
				if (const auto t = index_of_pc.find(inst.branch_target); t != index_of_pc.end()) visit(t->second);
				if (inst.opcode == Opcode::S_BRANCH) return;
			}
			if (i + 1 < insts.size()) visit(i + 1);
		};
		std::vector<uint8_t> seen(insts.size(), 0u);
		std::vector<size_t>  pending;
		for (size_t i = 0; i < insts.size(); i++) {
			bool writes_dependency = false;
			for (const auto reg: dependencies) writes_dependency |= Writes(insts[i], reg, wave_size);
			if (writes_dependency && seen[i] == 0u) {
				seen[i] = 1u;
				pending.push_back(i);
			}
		}
		// A write reaching a call (including a call that itself writes the register: its next
		// run reads the new value) refuses the expansion.
		bool reaches_call = false;
		std::vector<uint8_t> walked(insts.size(), 0u);
		while (!pending.empty() && !reaches_call) {
			const auto at = pending.back();
			pending.pop_back();
			successors(at, [&](size_t next) {
				if (walked[next] != 0u) return;
				walked[next] = 1u;
				if (is_call[next] != 0u) reaches_call = true;
				pending.push_back(next);
			});
		}
		if (reaches_call) {
			reason = "shader loop may overwrite a call-target dependency";
			return false;
		}
	}
	// A callee must not invalidate the user-data assumptions used to resolve later calls.
	for (const auto& [pc, call]: calls) {
		for (const auto& target: call.targets) {
			for (const auto& inst: target.function.instructions) {
				for (const auto reg: dependencies) {
					if (Writes(inst, reg, wave_size)) {
						reason = "shader function modifies a call-target dependency";
						return false;
					}
				}
			}
		}
	}
	std::vector<uint32_t>                      result;
	std::map<uint32_t, uint32_t>               main_pc;
	std::vector<std::pair<uint32_t, uint32_t>> main_branches;
	Branches                                   relocations;
	for (const auto& inst: program.instructions) {
		main_pc[inst.pc] = static_cast<uint32_t>(result.size());
		const auto found = calls.find(inst.pc);
		if (found == calls.end()) {
			if (inst.opcode == Opcode::S_ENDPGM) {
				result.push_back(0xbf810000u);
			} else if (inst.opcode == Opcode::S_GETPC_B64) {
				if (!EmitOriginalPc(result, inst, base, reason)) return false;
			} else {
				if (IsDirectBranch(inst.opcode))
					main_branches.emplace_back(result.size(), inst.branch_target);
				result.insert(result.end(), inst.raw, inst.raw + inst.word_count);
			}
		} else {
			const auto&           call      = found->second;
			const uint64_t        return_pc = base + inst.pc + inst.word_count * 4u;
			std::vector<uint32_t> returns;
			auto                  emit_target = [&](const CallTarget& target) -> bool {
				EmitMove(result, call.return_reg, static_cast<uint32_t>(return_pc));
				EmitMove(result, call.return_reg + 1u, static_cast<uint32_t>(return_pc >> 32u));
				std::map<uint32_t, uint32_t>               local_pc;
				std::vector<std::pair<uint32_t, uint32_t>> branches;
				for (const auto& body: target.function.instructions) {
					const auto at     = static_cast<uint32_t>(result.size());
					local_pc[body.pc] = at;
					if (body.opcode == Opcode::S_SETPC_B64) {
						returns.push_back(at);
						result.push_back(0xbf820000u); // S_BRANCH to the caller's continuation.
					} else if (body.opcode == Opcode::S_GETPC_B64) {
						if (!EmitOriginalPc(result, body, target.address, reason)) return false;
					} else {
						if (IsDirectBranch(body.opcode))
							branches.emplace_back(at, body.branch_target);
						result.insert(result.end(), body.raw, body.raw + body.word_count);
					}
				}
				for (const auto& [at, target]: branches) {
					relocations.emplace_back(at, local_pc.at(target));
				}
				return true;
			};
			if (call.dynamic) {
				// SWAPPC preserves SCC. Keep separate decision trees for its two incoming
				// values, without borrowing a guest SGPR. v439: each tree's match restores SCC
				// and branches to the callee's one body, emitted after the trees (before v439
				// every body was emitted once per tree: twice the code, and the glass's
				// ray-tracing shaders went past the size bound).
				// Group by the high address word so each low-word comparison has
				// one failure edge, rather than two branches sharing a later arm.
				const auto select_scc = static_cast<uint32_t>(result.size());
				result.push_back(0xbf850000u);
				std::vector<std::pair<uint32_t, size_t>> stubs; // (S_BRANCH position, target index)
				for (uint32_t scc = 0; scc < 2; ++scc) {
					if (scc != 0) relocations.emplace_back(select_scc, result.size());
					std::map<uint32_t, std::vector<size_t>> groups;
					for (size_t k = 0; k < call.targets.size(); k++)
						groups[static_cast<uint32_t>(call.targets[k].address >> 32u)].push_back(k);
					for (const auto& [high, targets]: groups) {
						result.push_back(0xbf06ff00u | (call.source_reg + 1u));
						result.push_back(high);
						const auto high_miss = static_cast<uint32_t>(result.size());
						result.push_back(0xbf840000u);
						for (const auto k: targets) {
							result.push_back(0xbf06ff00u | call.source_reg);
							result.push_back(static_cast<uint32_t>(call.targets[k].address));
							const auto low_miss = static_cast<uint32_t>(result.size());
							result.push_back(0xbf840000u);
							result.push_back(scc != 0 ? 0xbf068080u : 0xbf068180u);
							stubs.emplace_back(static_cast<uint32_t>(result.size()), k);
							result.push_back(0xbf820000u); // S_BRANCH to the callee's body
							relocations.emplace_back(low_miss, result.size());
						}
						result.push_back(0xbf920000u | ShaderCallMissTrapCode);
						result.push_back(0xbf810000u);
						relocations.emplace_back(high_miss, result.size());
					}
					result.push_back(0xbf920000u | ShaderCallMissTrapCode);
					result.push_back(0xbf810000u);
				}
				std::vector<uint32_t> bodies(call.targets.size());
				for (size_t k = 0; k < call.targets.size(); k++) {
					bodies[k] = static_cast<uint32_t>(result.size());
					if (!emit_target(call.targets[k])) return false;
				}
				for (const auto& [at, k]: stubs) relocations.emplace_back(at, bodies[k]);
			} else {
				for (const auto& target: call.targets)
					if (!emit_target(target)) return false;
			}
			for (const auto at: returns)
				relocations.emplace_back(at, result.size());
		}
		if (result.size() > MaxExpandedWordsLimit()) {
			// v439: with the size it had reached (callees so far) and the call targets in all.
			size_t targets = 0;
			for (const auto& [at, c]: calls) targets += c.targets.size();
			reason = "expanded shader exceeds the bounded code size (" + std::to_string(result.size()) +
			         " words at pc 0x" + fmt::format("{:x}", inst.pc) + ", " + std::to_string(targets) +
			         " call targets in all)";
			return false;
		}
	}
	for (const auto& [at, target]: main_branches) {
		const auto found = main_pc.find(target);
		if (found == main_pc.end()) {
			reason = "shader branches outside its instruction range";
			return false;
		}
		relocations.emplace_back(at, found->second);
	}
	if (!RelocateBranches(result, relocations, reason)) return false;
	expanded = std::move(result);
	return true;
}

using Reads = std::vector<std::pair<uint64_t, std::vector<uint32_t>>>;

// Merges contiguous and overlapping reads into ranges (overlapping reads saw the same memory),
// so that validating them takes a few reads instead of one per 64-word chunk.
void MergeReads(Reads& reads) {
	std::sort(reads.begin(), reads.end(),
	          [](const auto& a, const auto& b) { return a.first < b.first; });
	Reads merged;
	for (auto& [address, words]: reads) {
		if (!merged.empty()) {
			auto&      last = merged.back();
			const auto end  = last.first + last.second.size() * sizeof(uint32_t);
			if (address <= end) {
				const auto skip = (end - address) / sizeof(uint32_t);
				if (skip < words.size()) {
					last.second.insert(last.second.end(),
					                   words.begin() + static_cast<std::ptrdiff_t>(skip),
					                   words.end());
				}
				continue;
			}
		}
		merged.emplace_back(address, std::move(words));
	}
	reads = std::move(merged);
}

} // namespace

bool InlineShaderFunctions(std::span<const uint32_t> code, uint64_t base,
                           std::span<const uint32_t> user_data, const ShaderCodeReader& read,
                           std::vector<uint32_t>& expanded, std::string& reason,
                           uint32_t wave_size, std::vector<uint32_t>* consulted_user_data) {
	expanded.clear();
	reason.clear();
	if (consulted_user_data != nullptr) {
		consulted_user_data->clear();
	}
	if (wave_size != 32u && wave_size != 64u) {
		reason = "shader calls require a known wave size";
		return false;
	}
	ShaderAnalysis analysis;
	AnalyzeShader(code, analysis);
	if (!analysis.has_calls) return true;
	if (analysis.has_setpc) {
		reason = "shader calls combined with indirect branches are unsupported";
		return false;
	}
	std::vector<CallSite> sites;
	std::set<uint32_t>    dependencies;
	const bool            ok =
	    ResolveCalls(analysis, base, user_data, read, wave_size, sites, dependencies, reason) &&
	    BuildExpansion(analysis, base, sites, dependencies, read, wave_size, expanded, reason);
	if (consulted_user_data != nullptr) {
		consulted_user_data->assign(dependencies.begin(), dependencies.end());
	}
	return ok;
}

bool SplitShaderCalls(std::span<const uint32_t> code, uint64_t base,
                      std::span<const uint32_t> user_data, const ShaderCodeReader& read,
                      uint32_t wave_size, ShaderCallSplit& split, std::string& reason) {
	split = {};
	reason.clear();
	if (wave_size != 32u) {
		reason = "callable split requires wave32";
		return false;
	}
	ShaderAnalysis analysis;
	AnalyzeShader(code, analysis);
	if (!analysis.has_calls) {
		reason = "no calls";
		return false;
	}
	if (analysis.has_setpc) {
		reason = "shader calls combined with indirect branches are unsupported";
		return false;
	}
	std::vector<CallSite> sites;
	std::set<uint32_t>    dependencies;
	if (!ResolveCalls(analysis, base, user_data, read, wave_size, sites, dependencies, reason)) {
		return false;
	}
	if (sites.size() != 1u) {
		reason = fmt::format("callable split needs one call site, the shader has {}", sites.size());
		return false;
	}
	const auto& site = sites.front();
	split.call_pc    = site.pc;
	split.source_reg = site.source_reg;
	split.return_reg = site.return_reg;
	split.main_code.assign(code.begin(), code.end());
	if (site.pc % 4u != 0u || site.pc / 4u >= split.main_code.size() || !IsCall(split.main_code[site.pc / 4u])) {
		reason = "call site is not a call";
		return false;
	}
	split.main_code[site.pc / 4u] = 0xbf800000u; // s_nop 0
	// v442c: a target that cannot be read as a callable function (often an address whose code the
	// game has not streamed in, or another material's data) is left out instead of failing the whole
	// split: calls to it call nothing (KYTY_RT_SKIP_BAD_TARGETS=0: the split fails, as before).
	static const bool skip_bad = [] {
		const char* value = std::getenv("KYTY_RT_SKIP_BAD_TARGETS");
		return value == nullptr || value[0] != '0';
	}();
	split.skipped_targets = 0;
	split.skipped_reason.clear();
	for (const auto address: site.addresses) {
		Function function;
		if (!ReadFunction(address, site.return_reg, read, function, reason, wave_size)) {
			if (!skip_bad) {
				return false;
			}
			if (split.skipped_targets++ == 0) {
				split.skipped_reason = reason;
			}
			reason.clear();
			continue;
		}
		for (const auto& inst: function.instructions) {
			if (inst.opcode == Opcode::S_SETPC_B64) {
				function.code[inst.pc / 4u] = 0xbf810000u; // s_endpgm
			}
		}
		split.targets.push_back({address, std::move(function.code)});
	}
	if (split.targets.empty() && split.skipped_targets != 0) {
		reason = split.skipped_reason;
	}
	return !split.targets.empty();
}

struct ShaderFunctionExpander::Impl {
	struct Expansion {
		uint64_t              base            = 0;
		uint32_t              wave_size       = 0;
		size_t                user_data_count = 0;
		std::vector<CallSite> sites;
		Reads                 reads;  // the callee code, as read
		Reads                 failed; // reads that failed: (address, requested words)
		bool                  ok = false;
		std::string           reason;
		std::vector<uint32_t> code;
		// When StillReads last confirmed the callee code.
		std::chrono::steady_clock::time_point validated;
		uint64_t                              sites_hash = 0; // SitesHash(sites)
	};
	struct Shader {
		uint64_t               code_hash = 0;
		ShaderAnalysis         analysis;
		std::vector<Expansion> expansions; // kept; replaced in turn once full
		std::unordered_map<uint64_t, uint32_t> by_sites; // sites_hash -> index
		uint32_t               next_victim = 0;
		// The user-data registers the last complete call resolution consulted (code-dependent
		// only); valid while consulted_known.
		std::vector<uint32_t> consulted;
		uint32_t              consulted_wave  = 0;
		bool                  consulted_known = false;
	};
	// The expansions kept per shader (KYTY_EXPAND_KEEP, default 2048). A call shader's targets
	// come from per-object tables: with 8 kept, turning the camera brought more objects than that
	// into view, each expansion was built again (~3 ms each, 190 a second, 55 % of the GPU thread
	// while turning on the env3 roof, 2026-10-21); 256 still filled up (shader fc0537b7278db60c).
	static size_t MaxExpansions() {
		static const size_t keep = [] {
			const char* value = std::getenv("KYTY_EXPAND_KEEP");
			const auto  n     = value != nullptr ? std::strtoull(value, nullptr, 10) : 2048ull;
			return static_cast<size_t>(n < 1 ? 1 : n);
		}();
		return keep;
	}
	static uint64_t SitesHash(const std::vector<CallSite>& sites) {
		uint64_t h = 0xcbf29ce484222325ull;
		const auto mix = [&h](uint64_t v) { h = (h ^ v) * 0x100000001b3ull; };
		for (const auto& site: sites) {
			mix(site.pc);
			mix((uint64_t {site.source_reg} << 32u) | site.return_reg);
			mix(site.dynamic ? 1u : 0u);
			for (const auto address: site.addresses) mix(address);
			mix(site.addresses.size());
		}
		return h;
	}
	uint64_t builds = 0;

	std::unordered_map<uint64_t, Shader> shaders; // by code address
	std::vector<uint32_t>                scratch;

	bool StillReads(const Expansion& expansion, const ShaderCodeReader& read) {
		for (const auto& [address, words]: expansion.reads) {
			scratch.resize(words.size());
			if (!read(address, scratch) || scratch != words) return false;
		}
		for (const auto& [address, words]: expansion.failed) {
			scratch.resize(words.size());
			if (read(address, scratch)) return false;
		}
		return true;
	}
};

ShaderFunctionExpander::ShaderFunctionExpander(): m_impl(std::make_unique<Impl>()) {}
ShaderFunctionExpander::~ShaderFunctionExpander() = default;

bool ShaderFunctionExpander::HasCalls(std::span<const uint32_t> code) const {
	const auto found = m_impl->shaders.find(reinterpret_cast<uint64_t>(code.data()));
	if (found == m_impl->shaders.end() || found->second.analysis.code.size() != code.size()) {
		return true;
	}
	return found->second.analysis.has_calls || found->second.analysis.has_setpc;
}

const std::vector<uint32_t>* ShaderFunctionExpander::ConsultedUserData(
    std::span<const uint32_t> code, uint64_t code_hash, uint32_t wave_size) const {
	const auto found = m_impl->shaders.find(reinterpret_cast<uint64_t>(code.data()));
	if (code_hash == 0 || found == m_impl->shaders.end() || !found->second.consulted_known ||
	    found->second.consulted_wave != wave_size ||
	    found->second.code_hash != code_hash ||
	    found->second.analysis.code.size() != code.size()) {
		return nullptr;
	}
	return &found->second.consulted;
}

bool ShaderFunctionExpander::Expand(std::span<const uint32_t> code, uint64_t base,
                                    std::span<const uint32_t> user_data,
                                    const ShaderCodeReader& read, std::vector<uint32_t>& expanded,
                                    std::string& reason, uint32_t wave_size, uint64_t code_hash) {
	expanded.clear();
	reason.clear();
	if (wave_size != 32u && wave_size != 64u) {
		reason = "shader calls require a known wave size";
		return false;
	}
	auto& shader = m_impl->shaders[reinterpret_cast<uint64_t>(code.data())];
	// Comparing the whole code on every dispatch was ~4 us a dispatch (6 % of the GPU thread in
	// the save's opening fight, 2026-10-09); the content hash names the same code.
	const bool same_hash = code_hash != 0 && shader.code_hash == code_hash &&
	                       shader.analysis.code.size() == code.size();
	if (!same_hash && (shader.analysis.code.size() != code.size() ||
	                   !std::equal(code.begin(), code.end(), shader.analysis.code.begin()))) {
		shader = {};
		AnalyzeShader(code, shader.analysis);
	}
	shader.code_hash = code_hash;
	const auto& analysis = shader.analysis;
	if (!analysis.has_calls) return true;
	if (analysis.has_setpc) {
		reason = "shader calls combined with indirect branches are unsupported";
		return false;
	}
	std::vector<CallSite> sites;
	std::set<uint32_t>    dependencies;
	if (!ResolveCalls(analysis, base, user_data, read, wave_size, sites, dependencies, reason)) {
		shader.consulted_known = false;
		return false;
	}
	shader.consulted.assign(dependencies.begin(), dependencies.end());
	shader.consulted_wave  = wave_size;
	shader.consulted_known = true;
	// Re-reading every callee to confirm the kept expansion was ~2 us a dispatch for a shader
	// whose call targets depend on per-dispatch data (title CS c4df2a00067e0666, ~16,000
	// dispatches a second, all with different user data): once confirmed, the same targets reuse
	// it for KYTY_EXPAND_REUSE_MS (default 50) before the callees are read again.
	static const auto reuse = std::chrono::milliseconds([] {
		const char* value = std::getenv("KYTY_EXPAND_REUSE_MS");
		return value != nullptr ? std::strtoul(value, nullptr, 10) : 50ul;
	}());
	const auto now        = std::chrono::steady_clock::now();
	const auto sites_hash = Impl::SitesHash(sites);
	if (const auto found = shader.by_sites.find(sites_hash); found != shader.by_sites.end()) {
		auto& expansion = shader.expansions[found->second];
		if (expansion.sites_hash == sites_hash && expansion.base == base &&
		    expansion.wave_size == wave_size &&
		    expansion.user_data_count == user_data.size() && expansion.sites == sites &&
		    ((reuse.count() != 0 && now - expansion.validated < reuse) ||
		     m_impl->StillReads(expansion, read))) {
			if (now - expansion.validated >= reuse) {
				expansion.validated = now;
			}
			expanded = expansion.code;
			reason   = expansion.reason;
			return expansion.ok;
		}
	}
	Impl::Expansion expansion;
	expansion.base            = base;
	expansion.wave_size       = wave_size;
	expansion.user_data_count = user_data.size();
	expansion.sites           = std::move(sites);
	expansion.sites_hash      = sites_hash;
	const ShaderCodeReader recording = [&](uint64_t address, std::span<uint32_t> words) {
		const bool ok = read(address, words);
		(ok ? expansion.reads : expansion.failed)
		    .emplace_back(address, std::vector<uint32_t>(words.begin(), words.end()));
		return ok;
	};
	expansion.ok = BuildExpansion(analysis, base, expansion.sites, dependencies, recording,
	                              wave_size, expansion.code, expansion.reason);
	MergeReads(expansion.reads);
	expansion.validated = now;
	expanded = expansion.code;
	reason   = expansion.reason;
	uint32_t slot = 0;
	if (shader.expansions.size() < Impl::MaxExpansions()) {
		slot = static_cast<uint32_t>(shader.expansions.size());
		shader.expansions.emplace_back();
	} else {
		slot = shader.next_victim++ % static_cast<uint32_t>(shader.expansions.size());
		const auto old_hash = shader.expansions[slot].sites_hash;
		if (const auto it = shader.by_sites.find(old_hash); it != shader.by_sites.end() && it->second == slot) {
			shader.by_sites.erase(it);
		}
	}
	// KYTY_FPS_LOG: how often expansions are built ([expand], every 256 builds).
	if (static const bool log = std::getenv("KYTY_FPS_LOG") != nullptr; log && (++m_impl->builds % 256) == 0) {
		std::fprintf(stderr, "[expand] %llu built; shader %016llx keeps %zu\n",
		             static_cast<unsigned long long>(m_impl->builds),
		             static_cast<unsigned long long>(code_hash), shader.expansions.size());
	}
	shader.expansions[slot] = std::move(expansion);
	shader.by_sites[sites_hash] = slot;
	return shader.expansions[slot].ok;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
