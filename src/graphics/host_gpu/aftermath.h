#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_AFTERMATH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_AFTERMATH_H_

#include <cstddef>
#include <cstdint>

namespace Libs::Graphics::Aftermath {

// Opt-in NVIDIA Aftermath GPU crash dumps: KYTY_AFTERMATH_DLL=<path to GFSDK_Aftermath_Lib.x64.dll>.
// Kyty already enables shader debug info through VK_NV_device_diagnostics_config; the Aftermath
// runtime receives that debug info, so the crash dump can be mapped back to SPIR-V instructions.
// Writes _aftermath.nv-gpudmp and _aftermath_<id>.nvdbg files next to the log.

// Must be called before the Vulkan device is created.
void Initialize();

// Waits (a few seconds at most) for Aftermath to finish writing the crash dump after a device loss.
void WaitForCrashDump();

// True when KYTY_AFTERMATH_DLL is set (Vulkan objects get debug names then, so the dump's
// resource tracking names the image or buffer at the fault).
bool Enabled();

// Each compiled guest shader: the Aftermath hash of its SPIR-V, so the decoded dump's shader hashes
// map back to guest shader hashes ([aftermath-map] lines, and resolved in [aftermath] at a crash).
void NoteShader(uint64_t guest_hash, const uint32_t* spirv, size_t words);

} // namespace Libs::Graphics::Aftermath

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_AFTERMATH_H_
