#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Descriptor set 1 of a pipeline that samples bindless images (the host side is BindlessTable).
inline constexpr uint32_t BindlessDescriptorSet  = 1;
inline constexpr uint32_t BindlessImages2D       = 0;
inline constexpr uint32_t BindlessImages2DArray  = 1;
inline constexpr uint32_t BindlessImagesCube     = 2;
inline constexpr uint32_t BindlessImages3D       = 3;
// Storage images written at a GPU-selected key (Marvel's Wolverine's scent trail accumulates its
// density into a per-material image picked from a 20 MB material table): 2D, any format.
inline constexpr uint32_t BindlessStorage2D      = 4;
inline constexpr uint32_t BindlessTranslation    = 5;
inline constexpr uint32_t BindlessFeedback       = 6;
// Samplers mirrored from guest sampler heaps (16-byte S# records); slot 0 is the default sampler.
inline constexpr uint32_t BindlessSamplers       = 7;

// Set by the device setup when the storage array can be indexed non-uniformly and updated after
// bind. KYTY_BINDLESS_STORAGE=0 keeps store-only material images enumerated (or dropped).
inline std::atomic_bool g_bindless_storage_supported {false};
inline bool BindlessStorageImagesEnabled() {
	static const bool allowed = [] {
		const char* value = std::getenv("KYTY_BINDLESS_STORAGE");
		return value == nullptr || value[0] != '0';
	}();
	return allowed && g_bindless_storage_supported.load(std::memory_order_relaxed);
}
// v439 (address-keyed bindless images): a shader that reads its T# from a GPU-computed address
// (the ray-tracing shaders' material T#s, found by following pointers through instance and
// geometry records) looks the texture up by that address. The address space is cut into blocks of
// BindlessAddressBlockBytes; a block holds BindlessAddressBlockBytes / 32 T# records at one 32-byte
// phase (address & 31) and is an ordinary heap (BindlessTable::Heap, base = block, table_offset =
// phase), found through a directory the host keeps in the translation buffer: open addressing,
// BindlessAddressDirectoryEntries entries of four words (block low, block high, tag, region) at
// translation index BindlessAddressDirectoryBase, probed BindlessAddressProbes times from
// BindlessAddressHash. A block the directory does not know is requested in the feedback buffer at
// the same words (block low, block high, tag | resource << 8) and samples slot 0 meanwhile; the
// host then creates the heap and the directory entry, and its keys resolve as any heap's do.
inline constexpr uint32_t BindlessAddressBlockBits         = 14u;
inline constexpr uint32_t BindlessAddressBlockBytes        = 1u << BindlessAddressBlockBits;
inline constexpr uint32_t BindlessAddressKeysPerBlock      = BindlessAddressBlockBytes / 32u;
inline constexpr uint32_t BindlessAddressDirectoryBase     = 1u;
inline constexpr uint32_t BindlessAddressDirectoryEntries  = 4096u; // a power of two
inline constexpr uint32_t BindlessAddressDirectoryWords    = BindlessAddressDirectoryEntries * 4u;
inline constexpr uint32_t BindlessAddressProbes            = 4u;
inline constexpr uint32_t BindlessAddressTagValid          = 0x80000000u;
// The tag of a block at one phase as one view binding samples it: valid | binding << 24 | phase.
// A request adds the resource index the host registered for the shader's image (bits 8-23).
inline constexpr uint32_t BindlessAddressTag(uint32_t binding, uint32_t phase) {
	return BindlessAddressTagValid | ((binding & 0x7fu) << 24u) | (phase & 31u);
}
// The first directory entry probed for a block (the shader computes the same in SPIR-V).
inline constexpr uint32_t BindlessAddressHash(uint32_t block_low, uint32_t block_high, uint32_t tag) {
	uint32_t h = (block_low >> BindlessAddressBlockBits) ^ (block_high * 0x9e3779b1u) ^ (tag * 0x85ebca6bu);
	h ^= h >> 16u;
	h *= 0x7feb352du;
	h ^= h >> 15u;
	return h & (BindlessAddressDirectoryEntries - 1u);
}
// v439b (value-keyed bindless images): a sampled image whose T# the shader builds in a way no
// address explains (a descriptor Phi web that also carries other data) looks the texture up by
// the T#'s eight words. Per sampled view binding, a directory of BindlessValueEntries entries of
// eight words (the T#) at BindlessValueDirectoryBase + binding * BindlessValueWords (translation
// buffer), probed BindlessAddressProbes times from BindlessValueHash; entry e's translation is at
// BindlessValueRegionBase + binding * BindlessValueEntries + e. An unknown T# is requested in the
// feedback words of its first probed entry (its eight words) and samples slot 0 meanwhile.
inline constexpr uint32_t BindlessValueEntries        = 4096u; // a power of two
inline constexpr uint32_t BindlessValueWords          = BindlessValueEntries * 8u;
inline constexpr uint32_t BindlessValueBindings       = 5u;    // the sampled view bindings and Storage2D
inline constexpr uint32_t BindlessValueDirectoryBase  = BindlessAddressDirectoryBase + BindlessAddressDirectoryWords;
inline constexpr uint32_t BindlessValueRegionBase     = BindlessValueDirectoryBase + BindlessValueBindings * BindlessValueWords;
inline constexpr uint32_t BindlessReservedEntries     = BindlessValueRegionBase + BindlessValueBindings * BindlessValueEntries;
inline constexpr uint32_t BindlessValueHash(const uint32_t* words) {
	uint32_t h = 0x811c9dc5u;
	for (uint32_t i = 0; i < 8u; i++) {
		h = (h ^ words[i]) * 0x01000193u;
	}
	h ^= h >> 16u;
	h *= 0x7feb352du;
	h ^= h >> 15u;
	return h & (BindlessValueEntries - 1u);
}
// A translation entry for a key whose texture is not resident yet; the shader samples slot 0.
inline constexpr uint32_t BindlessPending        = 0xffffffffu;
// In a 3D binding's region: the key's T# is a 2D or 2D-array texture, held in the 2D-array
// binding at the slot in the low bits. The hardware addresses by the T# type, not the
// instruction's dimension, so a 3D sample of it reads layer round(z) (Marvel's Wolverine
// particles). Pending (all bits set) is told apart by the bit above.
inline constexpr uint32_t BindlessSlotArrayFlag  = 0x40000000u;
// v448: in a 2D-array binding's region: the key's T# is a cube (or cube array), held in the
// 2D-array binding at the slot in the low bits. The hardware addresses a cube T# by its type: a
// 2D-array instruction's coordinates are cube face coordinates (s, t in [1, 2], layer
// cube * 8 + face), which the shader converts as for a cube resource (CubeAxis, CubeLayer).
inline constexpr uint32_t BindlessSlotCubeFlag   = 0x80000000u;
inline constexpr uint32_t BindlessSlotFlagMask   = 0xc0000000u;
// v127: slots per image array (BindlessTable::MaxImagesPerArray). A translation value that is
// not pending and names no slot below this (or, outside a 3D region, carries flag bits) would
// index past the descriptor array into undefined descriptor memory; the shader samples slot 0
// instead and leaves the raw value in feedback word BindlessOutOfRangeSlotWord for the host
// ([bindless-oob]). Below the capture areas, above any heap region.
inline constexpr uint32_t BindlessMaxSlots           = 65536u;
inline constexpr uint32_t BindlessOutOfRangeSlotWord = 0xcfff0u;
// Diagnostics (KYTY_KEY_CAPTURE=<hash>): that shader marks every image key it samples in feedback
// word BindlessCaptureImages + (key & 0xffff) and every sampler key in BindlessCaptureSamplers +
// (key & 0xffff); the host prints them ([key-capture]). Above any heap region.
inline constexpr uint32_t BindlessCaptureImages   = 0xe0000u;
// Diagnostics (KYTY_SLOT_CAPTURE=1, v376): the translation value each bindless lookup read, at
// BindlessSlotCaptureBase + its translation index (indices below BindlessSlotCaptureEntries; the
// word after them takes the rest). Between the heap regions and the out-of-range word.
inline constexpr uint32_t BindlessSlotCaptureBase    = 0x80000u;
inline constexpr uint32_t BindlessSlotCaptureEntries = 0x4fff0u - 1u;
// Diagnostics (KYTY_PROBE_RECT with KYTY_PROBE_PC): the probed pixel shader's MRT0 export per
// pixel of a screen rectangle, 4 words each (x | y << 16, then three value bits); 16384 pixels.
inline constexpr uint32_t BindlessCaptureProbe    = 0xd0000u;
inline constexpr uint32_t BindlessCaptureProbeWords = 0x10000u;
inline constexpr uint32_t BindlessCaptureSamplers = 0xf0000u;
// Diagnostics (KYTY_CANARY_DETECT=1, v378, Crash B; uses the probe area, so not with
// KYTY_PROBE_RECT): a bindless sample that returns the free-canary colour (0.0625, 0.9375, 0.3125, 1) adds 1 to
// BindlessCanaryWord and leaves its shader hash (+1, +2) and translation value (+3); per descriptor
// slot (slot & 0x3fff) the shader hash low word at BindlessCanarySlots and the translation value
// at BindlessCanaryRaw (v380: | signature << 24). v380: hits per signature at +4 (3/4 channels),
// +5 (2 channels), +6 (1 channel or depth).
inline constexpr uint32_t BindlessCanaryWord  = 0xd0000u;
inline constexpr uint32_t BindlessCanarySlots = 0xd0010u;
inline constexpr uint32_t BindlessCanaryRaw   = 0xd4010u;
inline constexpr uint32_t BindlessCanarySlotMask = 0x3fffu;
// v382: per slot, the size of the image the GPU descriptor names (width | height << 16; all ones for a
// storage or multisampled image); the last hit's at BindlessCanaryWord + 7.
inline constexpr uint32_t BindlessCanarySize  = 0xd8010u;
// Diagnostics (KYTY_BINDLESS_OOB_INFO=1): a key outside its heap also leaves the sampling shader's
// hash (low 32 bits), the entry count the draw patched and the key in these three words.
inline constexpr uint32_t BindlessOobInfoWord     = 0xcfff1u;
// Diagnostics (KYTY_NAN_MAGENTA=1): a pixel shader that exports a NaN or Inf colour leaves its hash
// (low 32 bits) here; other pixels store to the word after it.
inline constexpr uint32_t BindlessNanExportWord   = 0xcfff5u;
// v301: an indirect image selection past its candidate list (the shader sampled the root
// instead): the shader hash's low word, then the value | candidates << 24 ([indirect-oob]).
inline constexpr uint32_t IndirectOutOfRangeWord  = 0xcfff6u;
// v389: while the host runs a usage probe it sets this word to 1, and bindless lookups then store
// their feedback word even when it is 0 (the sample clears the probe's mark), as without sparse
// feedback; the rest of the time sparse feedback stays sparse. KYTY_BINDLESS_PROBE_RUNTIME=0 turns
// it off (the probe then runs only with KYTY_BINDLESS_EVICT=1, which disables sparse feedback).
inline constexpr uint32_t BindlessProbeActiveWord = 0xcfff8u;
inline bool BindlessRuntimeProbeEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PROBE_RUNTIME");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}
// The S# a sampler without a usable heap entry gets: wrap on every axis, trilinear, full LOD range.
inline constexpr std::array<uint32_t, 4> BindlessDefaultSampler {
    0x00000000u,                             // wrap on every axis
    0xfffu << 12u,                           // max_lod 255.9
    (1u << 20u) | (1u << 22u) | (2u << 26u), // bilinear mag/min, linear mip
    0x00000000u};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
