#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

enum class UniformFillKind { None, Buffer, Image };

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;

	bool operator==(const UniformFill&) const = default;
};

// A guest descriptor heap a bindless image indexes: the host owns its key -> slot translation
// and patches the image's two flattened-SRT words (region base, entry count) before upload.
struct BindlessHeapUse {
	uint64_t base           = 0;
	uint64_t size           = 0;
	uint32_t table_offset   = 0;
	uint32_t image          = 0;
	uint32_t mapping_offset = 0;
	// v439: an address-keyed image (no table): the host patches its resource index, and every
	// address heap of its binding takes part in the draw.
	bool     address        = false;
	// v439b: a value-keyed image: the value heap of its binding takes part in the draw.
	bool     value          = false;

	bool operator==(const BindlessHeapUse&) const = default;
};

// A guest sampler heap a bindless sampler indexes: the host mirrors its S# records into the
// bindless sampler array and patches the sampler's two flattened-SRT words (region base, count).
struct BindlessSamplerHeapUse {
	uint64_t base           = 0;
	uint64_t size           = 0;
	uint32_t table_offset   = 0;
	uint32_t sampler        = 0;
	uint32_t mapping_offset = 0;

	bool operator==(const BindlessSamplerHeapUse&) const = default;
};

// The bytes of a written buffer a draw or dispatch can store to, relative to the descriptor's
// base. Valid when every store's address is bounded by values the host knows; otherwise the
// whole descriptor range counts as written.
struct BufferWriteExtent {
	uint64_t begin = 0;
	uint64_t end   = 0;
	bool     valid = false;

	bool operator==(const BufferWriteExtent&) const = default;
};

// A flat SRT slot the GPU fills: its bytes at `address` were GPU-written when the refresh ran, and
// only the shader uses the value, so the renderer copies them into the uploaded flat buffer on the
// GPU before the command instead of the host waiting for the GPU to read them.
struct SrtGpuFill {
	uint32_t flat_offset = 0;
	uint64_t address     = 0;

	bool operator==(const SrtGpuFill& other) const = default;
};

// KYTY_SRT_VERIFY=1 (compute): a refresh that would read bytes the GPU wrote, and so wait for
// the GPU, reads the bytes the host holds instead (what the guest wrote before the GPU work that
// is still running), and records them. The renderer copies the bytes as the GPU holds them
// before the dispatch, behind the work that wrote them, next to the expected values, and the
// shader compares the two pairs at entry: when any differs it does nothing and reports its hash,
// and the host never reads that shader's GPU-written bytes this way again. The common case is
// bytes that only share a range a GPU binding may write (a header of offsets next to GPU-written
// counts): equal, so the dispatch runs exactly as with the wait, without it.
struct SrtVerify {
	uint64_t address  = 0;
	uint32_t expected = 0;

	bool operator==(const SrtVerify& other) const = default;
};

// Programs that verify (BindingLayout::srt_verify): compute shaders that read the SRT. The
// renderer and the recompiler decide it the same way.
inline bool SrtVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Most recorded dwords per refresh; past it, reads wait as before.
inline constexpr uint32_t SrtVerifyMaxEntries = 256;

struct ResourceSnapshot {
	std::vector<BufferWriteExtent>      buffer_write_extents;
	std::vector<BindlessHeapUse>        bindless_heaps;
	std::vector<BindlessSamplerHeapUse> bindless_sampler_heaps;
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	std::vector<SrtGpuFill>     gpu_fills;
	std::vector<SrtVerify>      srt_verify;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
