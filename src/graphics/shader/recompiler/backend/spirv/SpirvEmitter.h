#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info);

// Optional image features of the device, set once by the device layer.
struct HostImageFeatures {
	// shaderResourceMinLod: the MinLod image operand, used for IMAGE_SAMPLE*_CL.
	bool min_lod = false;
	// fragmentShaderPixelInterlock: ordered critical sections for POPS pixel shaders.
	bool pixel_interlock = false;
	// maxSamplerLodBias: the clamp the sampler cache applies to the S# LOD bias.
	float max_sampler_lod_bias = 0.0f;
};

// Whether a pixel shader of this program runs inside an ordered fragment interlock
// (KYTY_POPS_INTERLOCK=0 turns it off).
bool UsesPixelInterlock(const IR::Program& program);

void              SetHostImageFeatures(const HostImageFeatures& features);
HostImageFeatures GetHostImageFeatures();

// S_MEMREALTIME counts at 100 MHz. The emitter reads the device clock (OpReadClockKHR, device
// scope) and shifts it right by this many bits to approach that rate. The device layer sets it
// from the device's clock rate: 3 for a 1 GHz clock (NVIDIA, the default), 0 for a 100 MHz one
// (AMD).
void     SetDeviceClockShift(uint32_t shift);
uint32_t GetDeviceClockShift();

// A coherent (GLC) buffer load is followed by a device-scope acquire barrier, so that a load
// polling for another workgroup's write sees it. Without one, AMD's drivers keep serving the
// cached value (BufferWorkgroupPublication, ISSUES.md #16); NVIDIA reloads anyway. The device
// layer enables it on every vendor but NVIDIA; KYTY_COHERENT_LOAD_ACQUIRE=0/1 overrides that.
void SetCoherentLoadAcquire(bool enabled);
bool GetCoherentLoadAcquire();

// The device runs 32-bit float shaders in the SPIR-V DenormFlushToZero mode
// (shaderDenormFlushToZeroFloat32, with independent denormal behaviour per width). Shaders whose
// MODE flushes FP32 denormals (every one this title runs) then declare the mode and drop the
// software flushes around MAD/MAC, conversions and transcendentals: the host flushes the inputs
// and outputs of every FP32 operation itself, as the guest ALU does. KYTY_DENORM_FTZ_MODE=0 keeps
// the software flushes.
void SetDenormFlushToZero32(bool supported);
bool GetDenormFlushToZero32();

// The device bounds-checks storage buffers to the byte (robustBufferAccess2 with a 1-byte
// robustStorageBufferAccessSizeAlignment): plain loads and stores of guest buffers whose bound is
// the descriptor range then skip the shader's own check and branch, as upstream KytyPS5 ee034be4
// does. Accesses whose bound is narrower than the range (GPU-sized buffers), atomics, merged
// subword stores and write-tracked stores keep the check. KYTY_ROBUST_BUFFER_BOUNDS=0 keeps every
// shader check.
void SetRobustBufferAccess(bool supported);
bool GetRobustBufferAccess();

// Why a mesh program cannot run in passes (ShaderMeshInputInfo::passes), or nullptr.
[[nodiscard]] const char* MeshPassesUnsupported(const IR::Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
