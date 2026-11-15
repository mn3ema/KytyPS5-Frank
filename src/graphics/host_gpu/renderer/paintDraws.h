#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PAINTDRAWS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PAINTDRAWS_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Diagnostics: KYTY_PAINT_PS=<ps hash>[+<ps hash>...] makes every draw of those pixel shaders add a
// bright constant colour (blend ONE*CONSTANT + ONE*DST on every colour target, all channels)
// wherever its fragments land, whatever the shader outputs: shows where an effect covers the
// screen even when it is drawn fully transparent. The n-th hash of the list gets the n-th colour
// of magenta, green, cyan, yellow, red, blue, white, orange (repeating); KYTY_PAINT_GAIN scales it
// (default 4, for HDR targets before tone mapping).
namespace Libs::Graphics::PaintDraws {

inline const std::string& List() {
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_PAINT_PS");
		return value != nullptr ? std::string(value) : std::string();
	}();
	return list;
}

// The position of hash in the list (0, 1, ...) or -1.
inline int Index(uint64_t hash) {
	const auto& list = List();
	if (list.empty() || hash == 0) {
		return -1;
	}
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
	const auto at = list.find(text);
	if (at == std::string::npos) {
		return -1;
	}
	int index = 0;
	for (size_t i = 0; i < at; i++) {
		index += list[i] == '+' ? 1 : 0;
	}
	return index;
}

// KYTY_PAINT_MODE=opaque: instead of adding a colour, the listed shaders draw with blending off
// (their raw RGB output replaces the target): shows what colour a transparent effect computes.
inline bool Opaque() {
	static const bool opaque = [] {
		const char* value = std::getenv("KYTY_PAINT_MODE");
		return value != nullptr && std::string(value) == "opaque";
	}();
	return opaque;
}

inline std::array<float, 4> Color(int index) {
	static const float gain = [] {
		const char* value = std::getenv("KYTY_PAINT_GAIN");
		return value != nullptr ? static_cast<float>(std::atof(value)) : 4.0f;
	}();
	static constexpr float palette[8][3] = {{1, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0},
	                                        {1, 0, 0}, {0, 0, 1}, {1, 1, 1}, {1, 0.5f, 0}};
	const auto& c = palette[index & 7];
	return {c[0] * gain, c[1] * gain, c[2] * gain, 1.0f};
}

} // namespace Libs::Graphics::PaintDraws

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PAINTDRAWS_H_
