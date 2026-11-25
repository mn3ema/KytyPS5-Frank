#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0,
                      bool lds_storage = false, bool clip_space = false);

struct SharedMemoryResources {
	bool lds = false;
	bool gds = false;
};

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers);
// The same over `blocks` only (one module of a merged callable split).
SharedMemoryResources CollectMemoryResources(const Program& program, const BlockList& blocks,
                                             std::vector<uint32_t>& buffers);
bool UsesFlattenedSrt(const Program& program);

// v442: whether callable splits get the fixed layout (BindingLayout::rt_fixed): the device must
// allow partially bound descriptor bindings (the host says so; assumed until it does), and
// KYTY_RT_FIXED_LAYOUT=0 turns it off.
void SetRtFixedLayoutSupported(bool supported);
bool RtFixedLayout();
// v443: callable modules that depend only on their own code (BindingLayout::rt_shared,
// IR/RtCallable.h); KYTY_RT_SHARED_CALLABLES=0 turns it off.
bool RtSharedCallables();
// The image binding kinds callable splits have had (bit i: FirstImageBinding + i), with `add`
// joined: the fixed set layout of a split made now has them all. The host seeds it with the kinds
// of earlier sessions, so every split of a session gets the same layout from the start.
uint64_t RtImageKinds(uint64_t add = 0);

// v442: the bindings of a callable split's fixed descriptor set (BindingLayout::rt_fixed), in
// binding order: every buffer binding, the sampler array and the image arrays the program has, with
// the capacities of RtFixedDescriptorCount. The same for every generation of a split until a count
// passes its capacity or a new kind of image appears.
struct RtFixedBinding {
	DescriptorBindingKind kind  = DescriptorBindingKind::Buffers;
	uint32_t              count = 0;
};
std::vector<RtFixedBinding> RtFixedSetLayout(const BindingLayout& layout);

// v443e: the loop cap of a callable (IR/RtCallable.h): after this many loop exit decisions in one
// invocation it takes every loop's exit and returns, and says so in the binding table's counters.
// A callable is a hit shader's worth of work: its loops go over a wave's lanes or a mask's bits. One
// that never ends holds the GPU until the system resets it (the device losses of g443b to g443d).
// KYTY_RT_CALLABLE_LOOP_LIMIT: the cap (default 2048; 0: none). A shader with the loop guard
// (KYTY_LOOP_GUARD_HASHES) has that instead.
uint32_t RtCallableLoopLimit();

// v443j: the address trap of a callable split's modules (IR/RtCallable.h). On unless
// KYTY_RT_TRAP_ADDR=0.
bool RtAddressTrap();
// v443o: KYTY_RT_TRAP_REPORT=1 (diagnostics; every module of a split changes): the trap also
// records what it stopped in the binding table's counters (IR/RtCallable.h), and the loop cap its
// lane and wave. Without it the trap is a few instructions with no branch of their own: the driver
// inlines the address translation at every one of a module's hundreds of table reads, and the
// record (two nested branches and eight atomics) more than tripled what it had to compile.
bool RtAddressTrapReport();

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
