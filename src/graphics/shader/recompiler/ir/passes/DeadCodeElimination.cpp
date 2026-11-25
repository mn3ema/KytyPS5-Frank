#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

void RemoveIdentities(const BlockList& blocks) {
	// Each identity's uses move to its argument in program order, as before; its own entry in the
	// argument's use list is dropped for all identities at once at the end. Removing it one by
	// one searched and shifted lists that grow with every identity folded into the same value.
	// Removed identities are marked and erased after that (a set of them cost most of the pass
	// on programs with millions of identities).
	std::vector<Inst*> touched;
	bool               any = false;
	for (auto* block: blocks) {
		for (auto& inst: block->Instructions()) {
			if (inst.GetOpcode() != ValueOpcode::Identity) {
				continue;
			}
			const auto replacement = inst.Arg(0);
			inst.ReplaceUsesForRemoval(replacement, touched);
			any = true;
		}
	}
	if (!any) {
		return;
	}
	Inst::DropRemovedUses(touched);
	for (auto* block: blocks) {
		std::erase_if(block->Instructions(), [](const Inst& inst) { return inst.Removed(); });
	}
}

void EliminateDeadCode(const BlockList& blocks) {
	std::vector<Inst*> pending;
	for (auto* block: blocks) {
		for (auto& inst: *block) {
			// Retained resource-planning instructions live outside the block list.
			inst.live = inst.MayHaveSideEffects() || std::ranges::any_of(inst.Uses(),
			    [](const Use& use) { return use.user->Parent() == nullptr; });
			if (inst.live) pending.push_back(&inst);
		}
	}
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
			auto* input = inst->Arg(arg).TryInstruction();
			if (input != nullptr && input->Parent() != nullptr && !input->live) {
				input->live = true;
				pending.push_back(input);
			}
		}
	}
	// Unlink every dead operand before freeing nodes in a dead Phi cycle.
	for (auto* block: blocks)
		for (auto& inst: *block)
			if (!inst.live) inst.Invalidate();
	for (auto* block: blocks)
		std::erase_if(block->Instructions(), [](const Inst& inst) { return !inst.live; });
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
