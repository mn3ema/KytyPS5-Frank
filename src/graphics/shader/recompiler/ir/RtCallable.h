#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_RTCALLABLE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_RTCALLABLE_H_

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// v440: a ray-tracing shader split at its call site (Decoder::SplitShaderCalls). The main program
// becomes a ray-generation shader that calls each target as a callable shader; the guest register
// state crosses the call in the callable payload, one 32-bit slot per register:
// s0..s105, VCC_LO, M0, SCC, then v0..v[vgpr_count-1]. EXEC stays with the caller: a callee
// starts with its active lanes on, and a lane calls only where its guest exec is set.
enum class RtRole : uint8_t { None, Raygen, Callable };

constexpr uint32_t PayloadSgprSlot = 0;   // + SGPR index
constexpr uint32_t PayloadVccSlot  = 106;
constexpr uint32_t PayloadM0Slot   = 107;
constexpr uint32_t PayloadSccSlot  = 108;
constexpr uint32_t PayloadVgprSlot = 109; // + VGPR index

// v440 M3: the ray-generation shader finds a call target's callable index in a table the host
// builds per pipeline (its device address in the shader data, after the dispatch's thread extents:
// BindingLayout dispatch_thread_dword + 3 and + 4), so its code does not depend on the target set.
// Open addressing: RtTargetSlots entries of 4 words (address low, high, callable index, 0), an empty
// entry's index is UINT32_MAX, and an address is at most RtTargetProbes entries from its hash.
// v443: 4096 entries, was 512. With ~190 targets in 512 entries (g442d) a target's eight entries
// were all taken about once per table ("does not fit the target table": its calls did nothing).
constexpr uint32_t RtTargetSlots  = 4096;
constexpr uint32_t RtTargetProbes = 8;
constexpr uint32_t RtTargetWords  = 4;
// After the table, RtCounterWords words, then one word per callable.
// KYTY_RT_COUNT=1 (diagnostics): +0 lanes at the call, +4 of them outside the target set, +8
// calling, +12 / +16 / +20 the BVH's, +24 lanes back from their call; per callable: the lanes inside
// it now (v443c: a hang shows where).
// Always (v443e, the callables' loop cap, RtCallableLoopLimit): +28 the invocations that hit the cap,
// and of the last one +32 its callable's code hash (RtCodeHash), +36 the guest pc of the loop
// decision, +40 its subgroup invocation id, +44 the subgroup's lanes there (a ballot), +48 its
// launch id, +52 the subgroup's lanes when the callable started.
// KYTY_RT_TRACE_PC=1 (diagnostics, v443f; every callable module changes): +56 callable invocations
// started, +60 ended, and of the guest blocks callables completed: +64 the highest pc, +68 the last
// pc, +72 the code hash of the callable that completed it. Lanes that enter a callable and never
// leave it, in code without a loop, stopped in the block after the last one completed.
// v443j, the address trap (RtAddressTrap; KYTY_RT_TRAP_ADDR=0 turns it off; every module of a split
// changes): a guest address whose page-table entry is in the first 16 KB after a 4 GiB boundary of
// the device's address space (N * 4 GiB + less than a page: where every device loss since g441o
// read), or is not a dword's address, is not read; it reads as a page that is not there (and is
// asked for like one, v443l). +76 how many, and of the
// last one: +80 the device address low, +84 high, +88 the guest address low, +92 high, +96 the code
// hash of its module (the ray generation's is its shader hash's low word), +100 where: 0 the
// guest address table, 1 the launch extents, 3 the write bitmap (v443l: the addresses the host
// writes into the shader data are behind the same trap, EmitRtAddressUsable; a binding table whose
// own address is none finds no call target and records nothing).
constexpr uint32_t RtCounterWords = 32;
constexpr uint32_t RtTrapPageBytes = 0x4000;
// What names a callable in its own module: its guest code, not its address or its place.
[[nodiscard]] constexpr uint32_t RtCodeHash(std::span<const uint32_t> code) {
	uint32_t hash = 0x811c9dc5u;
	for (const auto word: code) {
		hash = (hash ^ word) * 0x01000193u;
	}
	return hash;
}
constexpr uint32_t RtShaderDataDwords = 5; // thread extents (3) + table address (2)
[[nodiscard]] constexpr uint32_t RtTargetHash(uint32_t low, uint32_t high) {
	uint32_t h = (low >> 2u) ^ (high * 0x9e3779b1u);
	h ^= h >> 15u;
	h *= 0x85ebca6bu;
	h ^= h >> 13u;
	return h & (RtTargetSlots - 1u);
}

// v442 (incremental generations): a callable split's target set grows while the game streams, and
// each growth makes a new program (a generation) with more callables. A module that existed before
// must come out with the same SPIR-V, and the pipeline layout must stay the same, so that the
// module's pipeline library (and the driver's cache entry for it) is taken again and only the new
// callables compile. So a callable split has a fixed layout (BindingLayout::rt_fixed):
// - the descriptor set has every non-array binding, the buffer array at RtBufferCapacity, and the
//   sampler array and each image array the program has at RtArrayCapacity(count): they change only
//   when a count passes its capacity or a new kind of image appears (the host binds what exists;
//   the rest is VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND);
// - the shader data has every optional part, with RtBufferCapacity entries per buffer part;
// - a module declares only the bindings its own code names (their ids are reserved before the code
//   and the variables declared after it: spirvEmitterModule.cpp, ReserveRtDescriptors and
//   DefineUsedRtDescriptors), with these capacities as array lengths, preloads only its own
//   buffers' data, and has its own scratch size; the workgroup built-ins come from the launch id.
// A resource keeps its index for good: the merge appends each callable's resources in target order,
// and the host appends new targets after the old ones. The ray-generation module is the same too,
// until the set of payload slots the callables read or write grows (CutRtPayload).
// v443 (shared callables): the callables of a split are the game's hit shaders, and its other
// ray-tracing shaders call the same ones; in another session the same ones come in another order.
// So a callable's module must depend on nothing but its own code (BindingLayout::rt_shared): then
// its SPIR-V, and with it its pipeline library and the driver's cache entry, is the same in every
// split and every session.
// - What depends on the callable's place in the merged program (the index of a buffer, image or
//   sampler in its descriptor array or in the shader data, the flattened-SRT words of a bindless
//   image) is read from the callable's shader record (ShaderRecordBufferKHR; the words follow the
//   group handle in the shader binding table), one word per such number in the order the module
//   first names them (RtRecordSlot). The compile returns each callable's words.
// - The payload has the whole register file (RtSharedPayloadVgprs) and a callable loads and stores
//   every VGPR it uses, whatever the ray-generation shader's own range is.
// - The user data has RtUserDataDwords in the shader data, whatever the count the ray-generation
//   shader reads, so what follows it is at the same place in every split.
// - The descriptor set has the image kinds any split has had (RtImageKinds), not only this one's.
// KYTY_RT_SHARED_CALLABLES=0: as v442.
constexpr uint32_t RtSharedPayloadVgprs = 256;
constexpr uint32_t RtUserDataDwords     = 16;
constexpr uint32_t RtRecordMaxWords     = 512; // (the host checks the device's maxShaderGroupStride)

constexpr uint32_t RtBufferCapacity   = 64; // ShaderInfo::MaxBuffers (a merge above it fails)
constexpr uint32_t RtMinArrayCapacity = 16;
[[nodiscard]] constexpr uint32_t RtArrayCapacity(uint32_t count) {
	uint32_t capacity = RtMinArrayCapacity;
	while (capacity < count) {
		capacity *= 2u;
	}
	return capacity;
}

struct RtCallInfo {
	RtRole                role       = RtRole::None;
	uint32_t              call_pc    = 0; // raygen: byte pc of the call (an s_nop in the code)
	uint32_t              source_reg = 0; // raygen: SGPR pair with the target address
	uint32_t              return_reg = 0; // raygen: SGPR pair the call writes the return address to
	std::vector<uint64_t> targets;        // raygen: the code address of each callable index
	uint32_t              vgpr_count = 0; // VGPRs this side stores and loads (v0..)
	// v443: VGPRs the payload has room for (0: vgpr_count). With shared callables the payload is
	// the whole register file for every split, and each side moves what it uses.
	uint32_t              payload_vgprs = 0;
	uint64_t              code_address = 0; // callable: its guest code address (S_GETPC returns it + pc)
	uint32_t              code_hash    = 0; // callable: RtCodeHash of its code (v443e)

	[[nodiscard]] uint32_t PayloadWords() const { return PayloadVgprSlot + std::max(vgpr_count, payload_vgprs); }
};

// v440 M3: the input of a whole callable split (CompileOptions::rt_split). TranslateProgram then
// translates the main code as the ray-generation shader and each target as a callable, and merges
// them into one program (MergeRtCallables): one resource plan and one binding layout for all of
// them (a ray-tracing pipeline has one layout), each module's blocks a range of the program's.
struct RtSplitInput {
	struct Target {
		uint64_t                  address = 0;
		std::span<const uint32_t> code;
	};
	uint32_t            call_pc    = 0;
	uint32_t            source_reg = 0;
	uint32_t            return_reg = 0;
	std::vector<Target> targets;
	uint32_t            generation = 0; // the host's (its target set's growth count)
	uint32_t            safe_from  = 0; // v442b: the oldest generation whose code matches the guest's
};

// One module of a merged callable split: its blocks are program.blocks[first_block, + block_count).
struct RtModule {
	RtRole   role        = RtRole::None;
	uint32_t first_block = 0;
	uint32_t block_count = 0;
	// v442: the module's own scratch size (function-local), so its SPIR-V does not depend on the
	// other modules' spills.
	uint32_t scratch_dwords = 0;
	uint32_t code_hash      = 0; // v443e: a callable's RtCodeHash (its loop cap's report names it)
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif
