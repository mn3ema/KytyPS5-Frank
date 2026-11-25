#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

struct SccTag {
	auto operator<=>(const SccTag&) const = default;
};
struct ThreadBitScalarReg {
	ScalarReg reg {};
	auto      operator<=>(const ThreadBitScalarReg&) const = default;
};
struct ScalarMaskTag {
	ScalarReg reg {};
	auto      operator<=>(const ScalarMaskTag&) const = default;
};
struct ExecTag {
	auto operator<=>(const ExecTag&) const = default;
};
struct ExecLoTag {
	auto operator<=>(const ExecLoTag&) const = default;
};
struct ExecHiTag {
	auto operator<=>(const ExecHiTag&) const = default;
};
struct VccTag {
	auto operator<=>(const VccTag&) const = default;
};
struct VccLoTag {
	auto operator<=>(const VccLoTag&) const = default;
};
struct VccHiTag {
	auto operator<=>(const VccHiTag&) const = default;
};
struct M0Tag {
	auto operator<=>(const M0Tag&) const = default;
};
struct GotoVariable {
	uint32_t index                                  = 0;
	auto     operator<=>(const GotoVariable&) const = default;
};

using Variable =
    std::variant<ScalarReg, ThreadBitScalarReg, ScalarMaskTag, VectorReg, GotoVariable, SccTag,
                 ExecTag, ExecLoTag, ExecHiTag, VccTag, VccLoTag, VccHiTag, M0Tag>;

// One number per variable, for the definition index below.
uint64_t VariableKey(ScalarReg reg) { return (uint64_t {0} << 32u) | RegIndex(reg); }
uint64_t VariableKey(ThreadBitScalarReg value) { return (uint64_t {1} << 32u) | RegIndex(value.reg); }
uint64_t VariableKey(ScalarMaskTag value) { return (uint64_t {2} << 32u) | RegIndex(value.reg); }
uint64_t VariableKey(VectorReg reg) { return (uint64_t {3} << 32u) | RegIndex(reg); }
uint64_t VariableKey(GotoVariable value) { return (uint64_t {4} << 32u) | value.index; }
uint64_t VariableKey(SccTag) { return uint64_t {5} << 32u; }
uint64_t VariableKey(ExecTag) { return uint64_t {6} << 32u; }
uint64_t VariableKey(ExecLoTag) { return uint64_t {7} << 32u; }
uint64_t VariableKey(ExecHiTag) { return uint64_t {8} << 32u; }
uint64_t VariableKey(VccTag) { return uint64_t {9} << 32u; }
uint64_t VariableKey(VccLoTag) { return uint64_t {10} << 32u; }
uint64_t VariableKey(VccHiTag) { return uint64_t {11} << 32u; }
uint64_t VariableKey(M0Tag) { return uint64_t {12} << 32u; }

// Every variable's definition per block, in one flat open-addressed table that lives only during
// the rewrite. The per-block register arrays this replaces held about 9 KB in every block for the
// IR's whole life: gigabytes on programs with hundreds of thousands of blocks. A bucket of eight
// slots is chosen by (variable, block / 8), the block's low bits pick the first slot: a read walks
// neighbouring blocks for one variable, so its lookups share cache lines.
struct DefTable {
	struct Entry {
		uint64_t variable = 0;
		uint32_t block    = UINT32_MAX; // ssa_index; UINT32_MAX: empty
		Value    value;
	};
	std::vector<Entry> entries;
	size_t             used = 0;

	static size_t Hash(uint64_t key) {
		key ^= key >> 33u;
		key *= 0xff51afd7ed558ccdull;
		key ^= key >> 33u;
		key *= 0xc4ceb9fe1a85ec53ull;
		key ^= key >> 33u;
		return static_cast<size_t>(key);
	}
	size_t Start(uint64_t variable, uint32_t block) const {
		const size_t buckets = entries.size() / 8u;
		return ((Hash(variable * 0x9e3779b97f4a7c15ull + (block >> 3u)) & (buckets - 1u)) * 8u) | (block & 7u);
	}
	// Linear probing from there (stepping by whole buckets would only ever see slots with the same
	// low block bits, and one block defining many variables fills that class).
	size_t Next(size_t slot) const { return (slot + 1u) & (entries.size() - 1u); }

	template <typename T>
	const Value& Get(const Block* block, T variable) const {
		static const Value empty {};
		if (entries.empty()) {
			return empty;
		}
		const uint64_t key = VariableKey(variable);
		for (size_t slot = Start(key, block->ssa_index);; slot = Next(slot)) {
			const auto& entry = entries[slot];
			if (entry.block == UINT32_MAX) {
				return empty;
			}
			if (entry.block == block->ssa_index && entry.variable == key) {
				return entry.value;
			}
		}
	}
	template <typename T>
	void Set(const Block* block, T variable, const Value& value) {
		Insert(VariableKey(variable), block->ssa_index, value);
	}
	void Insert(uint64_t key, uint32_t block, const Value& value) {
		if ((used + 1) * 4 > entries.size() * 3) {
			Grow();
		}
		for (size_t slot = Start(key, block);; slot = Next(slot)) {
			auto& entry = entries[slot];
			if (entry.block == UINT32_MAX) {
				entry = {key, block, value};
				used++;
				return;
			}
			if (entry.block == block && entry.variable == key) {
				entry.value = value;
				return;
			}
		}
	}
	void Grow() {
		std::vector<Entry> old = std::move(entries);
		entries.assign(old.empty() ? 4096 : old.size() * 2, Entry {});
		used = 0;
		for (const auto& entry: old) {
			if (entry.block != UINT32_MAX) {
				Insert(entry.variable, entry.block, entry.value);
			}
		}
	}
};

Value InitialValue(ScalarReg) {
	return Value(0u);
}
Value InitialValue(ThreadBitScalarReg) {
	return Value(false);
}
Value InitialValue(ScalarMaskTag) {
	return Value(false);
}
Value InitialValue(VectorReg) {
	return Value(0u);
}
Value InitialValue(GotoVariable) {
	return Value(false);
}
Value InitialValue(SccTag) {
	return Value(false);
}
Value InitialValue(ExecTag) {
	return Value(false);
}
Value InitialValue(ExecLoTag) {
	return Value(0u);
}
Value InitialValue(ExecHiTag) {
	return Value(0u);
}
Value InitialValue(VccTag) {
	return Value(false);
}
Value InitialValue(VccLoTag) {
	return Value(0u);
}
Value InitialValue(VccHiTag) {
	return Value(0u);
}
Value InitialValue(M0Tag) {
	return Value(0u);
}

enum class ReadStep { Start, SetValue, PushPhiArgument, CollectArgument };

struct ReadState {
	Block*   block = nullptr;
	Value    result;
	Inst*    phi  = nullptr;
	size_t   pred = 0;
	size_t   operand_base = 0; // CollectArgument: first of this block's operands in `operands`
	ReadStep step = ReadStep::Start;
};

class Pass {
public:
	// Per variable, the blocks (by ssa_index, ascending) that define it: a construct's merge reads
	// a variable none of the construct's blocks define straight from the header (see
	// Block::ssa_construct_header). Filled before the rewrite; empty for small programs.
	std::unordered_map<uint64_t, std::vector<uint32_t>> definition_blocks;
	bool                                                construct_skip = false;


	template <typename T>
	void Write(T variable, Block* block, Value value) {
		definitions.Set(block, variable, value);
	}

	template <typename T>
	Value Read(T variable, Block* root) {
		stack.assign({{}, {.block = root}});
		const auto prepare_phi = [&]() {
			auto&      state        = stack.back();
			const auto predecessors = state.block->ImmPredecessors();
			if (state.pred == predecessors.size()) {
				const auto result = TryRemoveTrivialPhi(*state.phi, InitialValue(variable));
				const auto block  = state.block;
				stack.pop_back();
				stack.back().result = result;
				Write(variable, block, result);
				return;
			}
			state.step = ReadStep::PushPhiArgument;
			stack.push_back({.block = predecessors[state.pred]});
		};

		// An early-sealed join (see RewriteToSsa): the predecessors' values first, a phi only when
		// they differ. Placing the phi first, as for other blocks, made millions of trivial phis
		// on big programs, each one an identity for RemoveIdentities. No cycle can come back to
		// the block without passing a block that defines the variable first; if one does, that
		// visit places the phi itself and this one takes it.
		const auto prepare_collect = [&]() {
			auto&      state        = stack.back();
			const auto predecessors = state.block->ImmPredecessors();
			if (state.pred < predecessors.size()) {
				state.step = ReadStep::CollectArgument;
				stack.push_back({.block = predecessors[state.pred]});
				return;
			}
			auto* const block = state.block;
			block->ssa_collecting = false;
			Value result;
			if (const auto& def = definitions.Get(block, variable); !def.IsEmpty()) {
				result = def;
			} else {
				Value same;
				bool  trivial = true;
				for (size_t index = state.operand_base; index < operands.size(); index++) {
					const auto operand = operands[index].Resolve();
					if (operand == same.Resolve()) {
						continue;
					}
					if (!same.IsEmpty()) {
						trivial = false;
						break;
					}
					same = operand;
				}
				if (trivial) {
					result = same.IsEmpty() ? InitialValue(variable) : same;
				} else {
					auto& phi = *block->PrependNewInst(block->begin(), ValueOpcode::Phi);
					phi.SetFlags(InitialValue(variable).GetType());
					for (size_t index = 0; index < predecessors.size(); index++) {
						phi.AddPhiOperand(predecessors[index], operands[state.operand_base + index]);
					}
					result = Value(&phi);
				}
				Write(variable, block, result);
			}
			operands.resize(state.operand_base);
			stack.pop_back();
			stack.back().result = result;
		};

		do {
			auto& state = stack.back();
			auto* block = state.block;
			switch (state.step) {
				case ReadStep::Start: {
					if (const auto& def = definitions.Get(block, variable); !def.IsEmpty()) {
						state.result = def;
					} else if (!block->IsSsaSealed()) {
						auto& phi = *block->PrependNewInst(block->begin(), ValueOpcode::Phi);
						phi.SetFlags(InitialValue(variable).GetType());
						incomplete_phis[block][Variable(variable)] = &phi;
						state.result                               = Value(&phi);
					} else if (const auto predecessors = block->ImmPredecessors();
					           predecessors.size() == 1) {
						state.step = ReadStep::SetValue;
						stack.push_back({.block = predecessors.front()});
						break;
					} else if (const auto* header = SkipConstruct(variable, block); header != nullptr) {
						state.step = ReadStep::SetValue;
						stack.push_back({.block = const_cast<Block*>(header)});
						break;
					} else if (block->ssa_sealed_early && !block->ssa_collecting) {
						block->ssa_collecting = true;
						state.operand_base    = operands.size();
						prepare_collect();
						break;
					} else {
						auto& phi = *block->PrependNewInst(block->begin(), ValueOpcode::Phi);
						phi.SetFlags(InitialValue(variable).GetType());
						Write(variable, block, Value(&phi));
						state.phi = &phi;
						prepare_phi();
						break;
					}
					[[fallthrough]];
				}
				case ReadStep::SetValue: {
					const auto result = state.result;
					if (!SkipPassThroughWrite(variable, block, stack.size())) {
						Write(variable, block, result);
					}
					stack.pop_back();
					stack.back().result = result;
					break;
				}
				case ReadStep::CollectArgument: {
					operands.push_back(state.result);
					state.pred++;
					prepare_collect();
					break;
				}
				case ReadStep::PushPhiArgument: {
					const auto predecessors = block->ImmPredecessors();
					state.phi->AddPhiOperand(predecessors[state.pred], state.result);
					state.pred++;
					prepare_phi();
					break;
				}
			}
		} while (stack.size() > 1);
		return stack.back().result;
	}

	void Seal(Block* block) {
		if (const auto found = incomplete_phis.find(block); found != incomplete_phis.end()) {
			for (auto& [variable, phi]: found->second) {
				std::visit([&](auto value) { AddPhiOperands(value, *phi, block); }, variable);
			}
		}
		block->SsaSeal();
	}

private:
	template <typename T>
	const Block* SkipConstruct(T variable, const Block* block) const {
		const Block* header = block->ssa_construct_header;
		if (!construct_skip || header == nullptr || !block->IsSsaSealed() ||
		    block->ImmPredecessors().size() < 2u) {
			return nullptr;
		}
		const auto found = definition_blocks.find(VariableKey(variable));
		if (found != definition_blocks.end()) {
			const auto& list = found->second;
			const auto  next = std::upper_bound(list.begin(), list.end(), header->ssa_index);
			if (next != list.end() && *next < block->ssa_index) {
				return nullptr; // defined inside the construct
			}
		}
		return header;
	}

	// A goto variable's definition is not recorded in a sealed single-predecessor block the walk
	// only passes through (not the reading block): the table is a cache, and half its entries were
	// such blocks. Unsealed blocks keep theirs (it is the incomplete phi).
	template <typename T>
	static bool SkipPassThroughWrite(T, const Block* block, size_t depth) {
		if constexpr (std::is_same_v<T, GotoVariable>) {
			return depth > 2u && block->IsSsaSealed() && block->ImmPredecessors().size() == 1u;
		} else {
			return false;
		}
	}

	template <typename T>
	Value AddPhiOperands(T variable, Inst& phi, Block* block) {
		for (auto* predecessor: block->ImmPredecessors()) {
			phi.AddPhiOperand(predecessor, Read(variable, predecessor));
		}
		return TryRemoveTrivialPhi(phi, InitialValue(variable));
	}

	Value TryRemoveTrivialPhi(Inst& phi, Value initial_value) {
		Value same;
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			const auto operand = phi.Arg(index).Resolve();
			if (operand == same.Resolve() || operand == Value(&phi)) {
				continue;
			}
			if (!same.IsEmpty()) {
				return Value(&phi);
			}
			same = operand;
		}
		if (same.IsEmpty()) {
			same = initial_value;
		}
		const auto users = phi.Uses();
		phi.ReplaceUsesWith(same);
		for (const auto& use: users) {
			if (use.user->GetOpcode() == ValueOpcode::Phi) {
				TryRemoveTrivialPhi(*use.user, initial_value);
			}
		}
		return same;
	}

	std::unordered_map<Block*, std::map<Variable, Inst*>> incomplete_phis;
	DefTable                                              definitions;
	std::vector<ReadState>                               stack;
	std::vector<Value>                                   operands;
};

void VisitInstruction(Pass& pass, Block* block, Inst& inst) {
	switch (inst.GetOpcode()) {
		case ValueOpcode::SetScalarRegister:
			pass.Write(inst.Arg(0).ScalarRegister(), block, inst.Arg(1));
			break;
		case ValueOpcode::SetThreadBitScalarRegister:
			pass.Write(ThreadBitScalarReg {inst.Arg(0).ScalarRegister()}, block, inst.Arg(1));
			break;
		case ValueOpcode::SetScalarMaskTag:
			pass.Write(ScalarMaskTag {inst.Arg(0).ScalarRegister()}, block, inst.Arg(1));
			break;
		case ValueOpcode::SetVectorRegister:
			pass.Write(inst.Arg(0).VectorRegister(), block, inst.Arg(1));
			break;
		case ValueOpcode::SetGotoVariable:
			pass.Write(GotoVariable {inst.Arg(0).U32()}, block, inst.Arg(1));
			break;
		case ValueOpcode::SetScc: pass.Write(SccTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetExec: pass.Write(ExecTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetExecLo: pass.Write(ExecLoTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetExecHi: pass.Write(ExecHiTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetVcc: pass.Write(VccTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetVccLo: pass.Write(VccLoTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetVccHi: pass.Write(VccHiTag {}, block, inst.Arg(0)); break;
		case ValueOpcode::SetM0: pass.Write(M0Tag {}, block, inst.Arg(0)); break;
		case ValueOpcode::GetScalarRegister:
			inst.ReplaceUsesWith(pass.Read(inst.Arg(0).ScalarRegister(), block));
			break;
		case ValueOpcode::GetThreadBitScalarRegister:
			inst.ReplaceUsesWith(
			    pass.Read(ThreadBitScalarReg {inst.Arg(0).ScalarRegister()}, block));
			break;
		case ValueOpcode::GetScalarMaskTag:
			inst.ReplaceUsesWith(pass.Read(ScalarMaskTag {inst.Arg(0).ScalarRegister()}, block));
			break;
		case ValueOpcode::GetVectorRegister:
			inst.ReplaceUsesWith(pass.Read(inst.Arg(0).VectorRegister(), block));
			break;
		case ValueOpcode::GetGotoVariable:
			inst.ReplaceUsesWith(pass.Read(GotoVariable {inst.Arg(0).U32()}, block));
			break;
		case ValueOpcode::GetScc: inst.ReplaceUsesWith(pass.Read(SccTag {}, block)); break;
		case ValueOpcode::GetExec: inst.ReplaceUsesWith(pass.Read(ExecTag {}, block)); break;
		case ValueOpcode::GetExecLo: inst.ReplaceUsesWith(pass.Read(ExecLoTag {}, block)); break;
		case ValueOpcode::GetExecHi: inst.ReplaceUsesWith(pass.Read(ExecHiTag {}, block)); break;
		case ValueOpcode::GetVcc: inst.ReplaceUsesWith(pass.Read(VccTag {}, block)); break;
		case ValueOpcode::GetVccLo: inst.ReplaceUsesWith(pass.Read(VccLoTag {}, block)); break;
		case ValueOpcode::GetVccHi: inst.ReplaceUsesWith(pass.Read(VccHiTag {}, block)); break;
		case ValueOpcode::GetM0: inst.ReplaceUsesWith(pass.Read(M0Tag {}, block)); break;
		default: break;
	}
}

} // namespace

void RewriteToSsa(const BlockList& blocks) {
	for (size_t index = 0; index < blocks.size(); index++) {
		blocks[index]->ssa_index = static_cast<uint32_t>(index);
	}
	Pass pass;
	// Construct skipping (see Pass::definition_blocks), for big programs only. A merge whose
	// predecessors are not all inside [header, merge) keeps the ordinary walk.
	if (blocks.size() >= 4096u) {
		for (auto* block: blocks) {
			const auto* header = block->ssa_construct_header;
			if (header == nullptr) {
				continue;
			}
			const bool inside = header->ssa_index < block->ssa_index &&
			                    std::ranges::all_of(block->ImmPredecessors(), [&](const Block* predecessor) {
				                    return predecessor->ssa_index >= header->ssa_index &&
				                           predecessor->ssa_index < block->ssa_index;
			                    });
			if (!inside) {
				block->ssa_construct_header = nullptr;
			}
		}
		const auto define = [&](uint64_t key, uint32_t index) {
			auto& list = pass.definition_blocks[key];
			if (list.empty() || list.back() != index) {
				list.push_back(index);
			}
		};
		for (auto* block: blocks) {
			for (const auto& inst: *block) {
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister: define(VariableKey(inst.Arg(0).ScalarRegister()), block->ssa_index); break;
					case ValueOpcode::SetThreadBitScalarRegister: define(VariableKey(ThreadBitScalarReg {inst.Arg(0).ScalarRegister()}), block->ssa_index); break;
					case ValueOpcode::SetScalarMaskTag: define(VariableKey(ScalarMaskTag {inst.Arg(0).ScalarRegister()}), block->ssa_index); break;
					case ValueOpcode::SetVectorRegister: define(VariableKey(inst.Arg(0).VectorRegister()), block->ssa_index); break;
					case ValueOpcode::SetGotoVariable: define(VariableKey(GotoVariable {inst.Arg(0).U32()}), block->ssa_index); break;
					case ValueOpcode::SetScc: define(VariableKey(SccTag {}), block->ssa_index); break;
					case ValueOpcode::SetExec: define(VariableKey(ExecTag {}), block->ssa_index); break;
					case ValueOpcode::SetExecLo: define(VariableKey(ExecLoTag {}), block->ssa_index); break;
					case ValueOpcode::SetExecHi: define(VariableKey(ExecHiTag {}), block->ssa_index); break;
					case ValueOpcode::SetVcc: define(VariableKey(VccTag {}), block->ssa_index); break;
					case ValueOpcode::SetVccLo: define(VariableKey(VccLoTag {}), block->ssa_index); break;
					case ValueOpcode::SetVccHi: define(VariableKey(VccHiTag {}), block->ssa_index); break;
					case ValueOpcode::SetM0: define(VariableKey(M0Tag {}), block->ssa_index); break;
					default: break;
				}
			}
		}
		pass.construct_skip = true;
	}
	// Braun et al. seal a block once no predecessor can still change what flows into it: when
	// every predecessor has been filled. Its reads then follow a single predecessor or build a
	// complete phi at once. Sealing every block only at the end made each read in a block without
	// a local definition an incomplete phi, nearly all of them trivial and removed again: half of
	// a big shader's IR passes (DEBUGGING.md, 2026-10-01). Loop headers, whose back edge is filled
	// later, are sealed at the end as before.
	std::unordered_set<const Block*> filled;
	filled.reserve(blocks.size());
	for (auto* block: blocks) {
		const auto predecessors = block->ImmPredecessors();
		if (std::ranges::all_of(predecessors,
		                        [&](const Block* predecessor) { return filled.contains(predecessor); })) {
			pass.Seal(block);
			block->ssa_sealed_early = true;
		}
		for (auto& inst: *block) {
			VisitInstruction(pass, block, inst);
		}
		filled.insert(block);
	}
	for (auto* block: blocks) {
		if (!block->IsSsaSealed()) {
			pass.Seal(block);
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
