#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RTMERGE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RTMERGE_H_

#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;

// v440 M3: merges the translated callables of a callable split into its ray-generation program
// (IR/RtCallable.h, RtModule). Each callable's resources are appended to the program's (its
// descriptor sources, memory metadata and the resource indices its instructions carry renumbered),
// its blocks become a module range of program.blocks and its planning values join the program's.
// The host then sees one shader: one resource plan, one specialization, one binding layout.
// False (with `reason`) for a callable this first version does not merge: buffers read through a
// GPU table or with GPU-sized records, SRT reads, bounded buffer writes, a uniform fill.
bool MergeRtCallables(Program& raygen, std::vector<Program>& callables, std::string& reason);

// v440t: before the merge, cuts the payload traffic to the registers some callable uses. A
// callable's store of a slot it only loaded (a register it leaves alone) is dropped; the ray
// generation side then stores only the slots a callable reads or writes and loads back only the
// slots a callable writes (the others keep the value it stored). `stats` names the counts.
void CutRtPayload(Program& raygen, std::vector<Program>& callables, std::string& stats);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif
