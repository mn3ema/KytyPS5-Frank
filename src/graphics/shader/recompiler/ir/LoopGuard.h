#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_LOOPGUARD_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_LOOPGUARD_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace Libs::Graphics::ShaderRecompiler::IR {

// KYTY_LOOP_GUARD_HASHES: which shaders get a per-invocation cap on loop-exit decisions
// (KYTY_LOOP_GUARD_LIMIT, default 65536; past it every guarded loop takes its exit):
//   "all"                 every shader;
//   "digits:<hex digits>" the shaders whose hash starts with one of those digits;
//   anything else         a list of 16-digit hashes (any separator).
// The binding layout (BindingLayout::loop_guard) and the SPIR-V emitter both ask here. Both
// variables are read in the recompiler's sources, so they are part of the shader cache key.
inline bool LoopGuardApplies(uint64_t shader_hash) {
	static const char* const list = std::getenv("KYTY_LOOP_GUARD_HASHES");
	if (list == nullptr || list[0] == '\0') {
		return false;
	}
	const std::string_view text(list);
	char                   hash[17];
	std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(shader_hash));
	if (text == "all") {
		return true;
	}
	if (text.starts_with("digits:")) {
		return text.substr(7).find(hash[0]) != std::string_view::npos;
	}
	return text.find(std::string_view(hash, 16)) != std::string_view::npos;
}

// KYTY_LOOP_GUARD_LIMIT: "<default>" and/or "<16-digit hash>:<limit>" entries, any separator
// other than hex digits and ':' (e.g. "1000000000/dc76e1223a9bf673:2048"). Default 65536.
// KYTY_LOOP_GUARD_REPORT: the decision count at which a guarded invocation reports its shader
// once (the guard-trip report); default: that shader's limit. Both are part of the cache key.
inline uint32_t LoopGuardParseLimit(const char* text, uint64_t shader_hash, uint32_t fallback) {
	if (text == nullptr) {
		return fallback;
	}
	uint32_t   result  = fallback;
	bool       found   = false;
	const auto is_tok  = [](char c) {
		return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == ':';
	};
	const std::string_view all(text);
	size_t                 i = 0;
	while (i < all.size()) {
		while (i < all.size() && !is_tok(all[i])) i++;
		size_t j = i;
		while (j < all.size() && is_tok(all[j])) j++;
		const auto token = all.substr(i, j - i);
		i                = j;
		if (token.empty()) continue;
		const auto colon = token.find(':');
		if (colon == std::string_view::npos) {
			const auto value = std::strtoul(std::string(token).c_str(), nullptr, 10);
			if (value != 0ul && !found) result = static_cast<uint32_t>(value);
			continue;
		}
		const auto hash  = std::strtoull(std::string(token.substr(0, colon)).c_str(), nullptr, 16);
		const auto value = std::strtoul(std::string(token.substr(colon + 1)).c_str(), nullptr, 10);
		if (hash == shader_hash && value != 0ul) {
			result = static_cast<uint32_t>(value);
			found  = true;
		}
	}
	return result;
}

inline uint32_t LoopGuardLimit(uint64_t shader_hash) {
	static const char* const text = std::getenv("KYTY_LOOP_GUARD_LIMIT");
	return LoopGuardParseLimit(text, shader_hash, 1u << 16u);
}

inline uint32_t LoopGuardReport(uint64_t shader_hash) {
	static const char* const text = std::getenv("KYTY_LOOP_GUARD_REPORT");
	return LoopGuardParseLimit(text, shader_hash, LoopGuardLimit(shader_hash));
}

// The guard-trip report (BindingLayout::loop_guard): a host buffer of 1 + 2 * LoopGuardRingSize
// dwords — word 0 counts trips, then (hash low, hash high) pairs in a ring at 1 + 2 * (n % size).
inline constexpr uint32_t LoopGuardRingSize = 4096u;

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_LOOPGUARD_H_
