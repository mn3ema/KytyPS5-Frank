#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BUFFERFORMAT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BUFFERFORMAT_H_

#include "graphics/guest_gpu/gpu_defs.h"

namespace Libs::Graphics::ShaderRecompiler::Format {

enum class ComponentType {
	Unknown,
	Uint,
	Sint,
	Unorm,
	Snorm,
	Uscaled,
	Sscaled,
	Float,
};

struct BufferFormatInfo {
	Prospero::BufferFormat format                  = Prospero::BufferFormat::kInvalid;
	ComponentType          type                    = ComponentType::Unknown;
	uint32_t               component_count         = 0;
	uint32_t               byte_size               = 0;
	uint32_t               component_bits[4]       = {};
	uint32_t               component_bit_offset[4] = {};
	bool                   packed_bitfield         = false;
};

enum class FormattedSourceKind { Memory, Zero, One, Invalid };

struct FormattedSource {
	FormattedSourceKind kind      = FormattedSourceKind::Zero;
	uint32_t            component = 0;
};

constexpr FormattedSource ResolveFormattedSource(const BufferFormatInfo& info, uint32_t selector) {
	if (selector == 0u) return {};
	if (selector == 1u) return {FormattedSourceKind::One, 0};
	if (selector < 4u || selector > 7u || info.component_count == 0u) {
		return {FormattedSourceKind::Invalid, 0};
	}
	// Formatted decoding expands source channels before applying the swizzle.
	return {FormattedSourceKind::Memory, (selector - 4u) % info.component_count};
}

constexpr uint32_t FormattedConstantBits(const BufferFormatInfo& info, FormattedSourceKind kind) {
	if (kind != FormattedSourceKind::One) return 0;
	return info.type == ComponentType::Uint || info.type == ComponentType::Sint ? 1u : 0x3f800000u;
}

constexpr Prospero::BufferFormat DecodeTBufferFormat(uint32_t data_format, uint32_t number_format) {
	return static_cast<Prospero::BufferFormat>(((number_format & 0x7u) << 4u) |
	                                           (data_format & 0xfu));
}

constexpr ComponentType GetFormatComponentType(Prospero::BufferFormat format) {
	switch (format) {
		case Prospero::BufferFormat::k8UInt:
		case Prospero::BufferFormat::k16UInt:
		case Prospero::BufferFormat::k8_8UInt:
		case Prospero::BufferFormat::k32UInt:
		case Prospero::BufferFormat::k16_16UInt:
		case Prospero::BufferFormat::k11_11_10UInt:
		case Prospero::BufferFormat::k10_11_11UInt:
		case Prospero::BufferFormat::k2_10_10_10UInt:
		case Prospero::BufferFormat::k10_10_10_2UInt:
		case Prospero::BufferFormat::k8_8_8_8UInt:
		case Prospero::BufferFormat::k32_32UInt:
		case Prospero::BufferFormat::k16_16_16_16UInt:
		case Prospero::BufferFormat::k32_32_32UInt:
		case Prospero::BufferFormat::k32_32_32_32UInt: return ComponentType::Uint;
		case Prospero::BufferFormat::k8SInt:
		case Prospero::BufferFormat::k16SInt:
		case Prospero::BufferFormat::k8_8SInt:
		case Prospero::BufferFormat::k32SInt:
		case Prospero::BufferFormat::k16_16SInt:
		case Prospero::BufferFormat::k11_11_10SInt:
		case Prospero::BufferFormat::k10_11_11SInt:
		case Prospero::BufferFormat::k2_10_10_10SInt:
		case Prospero::BufferFormat::k10_10_10_2SInt:
		case Prospero::BufferFormat::k8_8_8_8SInt:
		case Prospero::BufferFormat::k32_32SInt:
		case Prospero::BufferFormat::k16_16_16_16SInt:
		case Prospero::BufferFormat::k32_32_32SInt:
		case Prospero::BufferFormat::k32_32_32_32SInt: return ComponentType::Sint;
		case Prospero::BufferFormat::k8UNorm:
		case Prospero::BufferFormat::k16UNorm:
		case Prospero::BufferFormat::k8_8UNorm:
		case Prospero::BufferFormat::k16_16UNorm:
		case Prospero::BufferFormat::k11_11_10UNorm:
		case Prospero::BufferFormat::k10_11_11UNorm:
		case Prospero::BufferFormat::k2_10_10_10UNorm:
		case Prospero::BufferFormat::k10_10_10_2UNorm:
		case Prospero::BufferFormat::k8_8_8_8UNorm:
		case Prospero::BufferFormat::k16_16_16_16UNorm: return ComponentType::Unorm;
		case Prospero::BufferFormat::k8SNorm:
		case Prospero::BufferFormat::k16SNorm:
		case Prospero::BufferFormat::k8_8SNorm:
		case Prospero::BufferFormat::k16_16SNorm:
		case Prospero::BufferFormat::k11_11_10SNorm:
		case Prospero::BufferFormat::k10_11_11SNorm:
		case Prospero::BufferFormat::k2_10_10_10SNorm:
		case Prospero::BufferFormat::k10_10_10_2SNorm:
		case Prospero::BufferFormat::k8_8_8_8SNorm:
		case Prospero::BufferFormat::k16_16_16_16SNorm: return ComponentType::Snorm;
		case Prospero::BufferFormat::k8UScaled:
		case Prospero::BufferFormat::k16UScaled:
		case Prospero::BufferFormat::k8_8UScaled:
		case Prospero::BufferFormat::k16_16UScaled:
		case Prospero::BufferFormat::k11_11_10UScaled:
		case Prospero::BufferFormat::k10_11_11UScaled:
		case Prospero::BufferFormat::k2_10_10_10UScaled:
		case Prospero::BufferFormat::k10_10_10_2UScaled:
		case Prospero::BufferFormat::k8_8_8_8UScaled:
		case Prospero::BufferFormat::k16_16_16_16UScaled: return ComponentType::Uscaled;
		case Prospero::BufferFormat::k8SScaled:
		case Prospero::BufferFormat::k16SScaled:
		case Prospero::BufferFormat::k8_8SScaled:
		case Prospero::BufferFormat::k16_16SScaled:
		case Prospero::BufferFormat::k11_11_10SScaled:
		case Prospero::BufferFormat::k10_11_11SScaled:
		case Prospero::BufferFormat::k2_10_10_10SScaled:
		case Prospero::BufferFormat::k10_10_10_2SScaled:
		case Prospero::BufferFormat::k8_8_8_8SScaled:
		case Prospero::BufferFormat::k16_16_16_16SScaled: return ComponentType::Sscaled;
		case Prospero::BufferFormat::k16Float:
		case Prospero::BufferFormat::k32Float:
		case Prospero::BufferFormat::k16_16Float:
		case Prospero::BufferFormat::k11_11_10Float:
		case Prospero::BufferFormat::k10_11_11Float:
		case Prospero::BufferFormat::k32_32Float:
		case Prospero::BufferFormat::k16_16_16_16Float:
		case Prospero::BufferFormat::k32_32_32Float:
		case Prospero::BufferFormat::k32_32_32_32Float: return ComponentType::Float;
		default: return ComponentType::Unknown;
	}
}

constexpr BufferFormatInfo GetFormatInfo(Prospero::BufferFormat format) {
	const auto type = GetFormatComponentType(format);
	if (type == ComponentType::Unknown) {
		return {};
	}

	switch (format) {
		case Prospero::BufferFormat::k8UNorm:
		case Prospero::BufferFormat::k8SNorm:
		case Prospero::BufferFormat::k8UScaled:
		case Prospero::BufferFormat::k8SScaled:
		case Prospero::BufferFormat::k8UInt:
		case Prospero::BufferFormat::k8SInt:
			return {format, type, 1, 1, {8, 0, 0, 0}, {0, 0, 0, 0}, false};
		case Prospero::BufferFormat::k16UNorm:
		case Prospero::BufferFormat::k16SNorm:
		case Prospero::BufferFormat::k16UScaled:
		case Prospero::BufferFormat::k16SScaled:
		case Prospero::BufferFormat::k16UInt:
		case Prospero::BufferFormat::k16SInt:
		case Prospero::BufferFormat::k16Float:
			return {format, type, 1, 2, {16, 0, 0, 0}, {0, 0, 0, 0}, false};
		case Prospero::BufferFormat::k8_8UNorm:
		case Prospero::BufferFormat::k8_8SNorm:
		case Prospero::BufferFormat::k8_8UScaled:
		case Prospero::BufferFormat::k8_8SScaled:
		case Prospero::BufferFormat::k8_8UInt:
		case Prospero::BufferFormat::k8_8SInt:
			return {format, type, 2, 2, {8, 8, 0, 0}, {0, 8, 0, 0}, false};
		case Prospero::BufferFormat::k32UInt:
		case Prospero::BufferFormat::k32SInt:
		case Prospero::BufferFormat::k32Float:
			return {format, type, 1, 4, {32, 0, 0, 0}, {0, 0, 0, 0}, false};
		case Prospero::BufferFormat::k16_16UNorm:
		case Prospero::BufferFormat::k16_16SNorm:
		case Prospero::BufferFormat::k16_16UScaled:
		case Prospero::BufferFormat::k16_16SScaled:
		case Prospero::BufferFormat::k16_16UInt:
		case Prospero::BufferFormat::k16_16SInt:
		case Prospero::BufferFormat::k16_16Float:
			return {format, type, 2, 4, {16, 16, 0, 0}, {0, 16, 0, 0}, false};
		case Prospero::BufferFormat::k11_11_10UNorm:
		case Prospero::BufferFormat::k11_11_10SNorm:
		case Prospero::BufferFormat::k11_11_10UScaled:
		case Prospero::BufferFormat::k11_11_10SScaled:
		case Prospero::BufferFormat::k11_11_10UInt:
		case Prospero::BufferFormat::k11_11_10SInt:
		case Prospero::BufferFormat::k11_11_10Float:
			return {format, type, 3, 4, {11, 11, 10, 0}, {0, 11, 22, 0}, true};
		case Prospero::BufferFormat::k10_11_11UNorm:
		case Prospero::BufferFormat::k10_11_11SNorm:
		case Prospero::BufferFormat::k10_11_11UScaled:
		case Prospero::BufferFormat::k10_11_11SScaled:
		case Prospero::BufferFormat::k10_11_11UInt:
		case Prospero::BufferFormat::k10_11_11SInt:
		case Prospero::BufferFormat::k10_11_11Float:
			return {format, type, 3, 4, {10, 11, 11, 0}, {0, 10, 21, 0}, true};
		case Prospero::BufferFormat::k2_10_10_10UNorm:
		case Prospero::BufferFormat::k2_10_10_10SNorm:
		case Prospero::BufferFormat::k2_10_10_10UScaled:
		case Prospero::BufferFormat::k2_10_10_10SScaled:
		case Prospero::BufferFormat::k2_10_10_10UInt:
		case Prospero::BufferFormat::k2_10_10_10SInt:
			return {format, type, 4, 4, {2, 10, 10, 10}, {0, 2, 12, 22}, true};
		case Prospero::BufferFormat::k10_10_10_2UNorm:
		case Prospero::BufferFormat::k10_10_10_2SNorm:
		case Prospero::BufferFormat::k10_10_10_2UScaled:
		case Prospero::BufferFormat::k10_10_10_2SScaled:
		case Prospero::BufferFormat::k10_10_10_2UInt:
		case Prospero::BufferFormat::k10_10_10_2SInt:
			return {format, type, 4, 4, {10, 10, 10, 2}, {0, 10, 20, 30}, true};
		case Prospero::BufferFormat::k8_8_8_8UNorm:
		case Prospero::BufferFormat::k8_8_8_8SNorm:
		case Prospero::BufferFormat::k8_8_8_8UScaled:
		case Prospero::BufferFormat::k8_8_8_8SScaled:
		case Prospero::BufferFormat::k8_8_8_8UInt:
		case Prospero::BufferFormat::k8_8_8_8SInt:
			return {format, type, 4, 4, {8, 8, 8, 8}, {0, 8, 16, 24}, false};
		case Prospero::BufferFormat::k32_32UInt:
		case Prospero::BufferFormat::k32_32SInt:
		case Prospero::BufferFormat::k32_32Float:
			return {format, type, 2, 8, {32, 32, 0, 0}, {0, 32, 0, 0}, false};
		case Prospero::BufferFormat::k16_16_16_16UNorm:
		case Prospero::BufferFormat::k16_16_16_16SNorm:
		case Prospero::BufferFormat::k16_16_16_16UScaled:
		case Prospero::BufferFormat::k16_16_16_16SScaled:
		case Prospero::BufferFormat::k16_16_16_16UInt:
		case Prospero::BufferFormat::k16_16_16_16SInt:
		case Prospero::BufferFormat::k16_16_16_16Float:
			return {format, type, 4, 8, {16, 16, 16, 16}, {0, 16, 32, 48}, false};
		case Prospero::BufferFormat::k32_32_32UInt:
		case Prospero::BufferFormat::k32_32_32SInt:
		case Prospero::BufferFormat::k32_32_32Float:
			return {format, type, 3, 12, {32, 32, 32, 0}, {0, 32, 64, 0}, false};
		case Prospero::BufferFormat::k32_32_32_32UInt:
		case Prospero::BufferFormat::k32_32_32_32SInt:
		case Prospero::BufferFormat::k32_32_32_32Float:
			return {format, type, 4, 16, {32, 32, 32, 32}, {0, 32, 64, 96}, false};
		default: return {};
	}
}

constexpr uint32_t GetFormatComponentByteOffset(const BufferFormatInfo& info, uint32_t component) {
	if (component >= info.component_count) {
		return 0;
	}
	return info.packed_bitfield ? 0u : info.component_bit_offset[component] / 8u;
}

// v432 (dynamic buffer formats): the distinct memory layouts of the valid buffer formats (byte
// size, components, their bits and offsets), numbered from 1 in format order; 0: no valid format.
// A format is decoded at run time as its layout (the loads) and its component type (the conversion).
struct BufferLayouts {
	BufferFormatInfo info[16] {};
	uint8_t          id[128] {};
	uint32_t         count = 0;
};

constexpr bool SameBufferLayout(const BufferFormatInfo& a, const BufferFormatInfo& b) {
	if (a.byte_size != b.byte_size || a.component_count != b.component_count ||
	    a.packed_bitfield != b.packed_bitfield) {
		return false;
	}
	for (uint32_t c = 0; c < 4u; c++) {
		if (a.component_bits[c] != b.component_bits[c] ||
		    a.component_bit_offset[c] != b.component_bit_offset[c]) {
			return false;
		}
	}
	return true;
}

constexpr BufferLayouts MakeBufferLayouts() {
	BufferLayouts layouts;
	for (uint32_t value = 0; value < 128u; value++) {
		const auto info = GetFormatInfo(static_cast<Prospero::BufferFormat>(value));
		if (info.type == ComponentType::Unknown || info.component_count == 0u) continue;
		uint32_t id = 0;
		for (uint32_t i = 1; i <= layouts.count; i++) {
			if (SameBufferLayout(layouts.info[i], info)) id = i;
		}
		if (id == 0) {
			id                = ++layouts.count;
			layouts.info[id]  = info;
		}
		layouts.id[value] = static_cast<uint8_t>(id);
	}
	return layouts;
}

inline constexpr BufferLayouts kBufferLayouts = MakeBufferLayouts();
static_assert(kBufferLayouts.count <= 15u, "buffer layouts must fit 4 bits");

} // namespace Libs::Graphics::ShaderRecompiler::Format

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BUFFERFORMAT_H_ */
