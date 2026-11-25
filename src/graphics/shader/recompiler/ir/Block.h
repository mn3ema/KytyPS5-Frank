#pragma once

#include "graphics/shader/recompiler/ir/Reg.h"
#include "graphics/shader/recompiler/ir/Value.h"

#include <array>
#include <initializer_list>
#include <list>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// A standard list provides stable instruction addresses without adding an intrusive-container
// dependency.
class Block {
public:
	using InstructionList = std::list<Inst>;
	using iterator        = InstructionList::iterator;
	using const_iterator  = InstructionList::const_iterator;

	Block() = default;
	~Block();

	Block(const Block&)            = delete;
	Block& operator=(const Block&) = delete;
	Block(Block&&)                 = delete;
	Block& operator=(Block&&)      = delete;

	Inst&    AppendNewInst(ValueOpcode opcode, std::initializer_list<Value> args = {},
	                       uint64_t flags = 0);
	iterator PrependNewInst(iterator insertion_point, ValueOpcode opcode,
	                        std::initializer_list<Value> args = {}, uint64_t flags = 0);
	void     AddBranch(Block* block);

	[[nodiscard]] InstructionList&        Instructions();
	[[nodiscard]] const InstructionList&  Instructions() const;
	[[nodiscard]] std::span<Block* const> ImmPredecessors() const;
	[[nodiscard]] std::span<Block* const> ImmSuccessors() const;

	void               SsaSeal();
	[[nodiscard]] bool IsSsaSealed() const;

	iterator       begin();
	const_iterator begin() const;
	iterator       end();
	const_iterator end() const;
	bool           empty() const;

	// SsaRewrite: sealed while filling, once every predecessor was filled (every cycle through
	// the block passes a block sealed later); and a read collecting this block's operands.
	bool ssa_sealed_early = false;
	bool ssa_collecting   = false;
	// SsaRewrite: position in the block list.
	uint32_t ssa_index = 0;
	// Set by translation on a structured construct's merge block: the construct's header. The
	// construct's blocks lie between the two in block order, so a read at the merge of a variable
	// no block in between defines takes the header's value without walking every branch.
	Block* ssa_construct_header = nullptr;
	// v446a: translation wrote EXEC in this block in a way that can turn lanes on
	// (IR::Program::exec_widens_lanes); the emitter keeps wave-uniform branches around it.
	bool exec_widens = false;

private:
	InstructionList     instructions;
	std::vector<Block*> predecessors;
	std::vector<Block*> successors;
	bool                ssa_sealed = false;
};

using BlockList = std::vector<Block*>;

} // namespace Libs::Graphics::ShaderRecompiler::IR
