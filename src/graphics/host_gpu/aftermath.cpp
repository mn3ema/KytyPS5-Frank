#include "graphics/host_gpu/aftermath.h"
#include "graphics/host_gpu/addressBindings.h"

#include "graphics/host_gpu/renderer/image/imageHistory.h"

#include "common/common.h"
#include "common/config.h"
#include "common/logging/log.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <ctime>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::Aftermath {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

// Declared here from the Aftermath SDK headers (GFSDK_Aftermath_GpuCrashDump*.h), so Kyty builds
// without the SDK and only loads the DLL when asked to.
constexpr uint32_t ApiVersion                 = 0x0000021B;
constexpr uint32_t WatchedApiVulkan           = 0x2;
constexpr uint32_t FeatureDeferDebugInfo      = 0x1;
constexpr uint32_t ResultSuccess              = 0x1;
constexpr uint32_t StatusNotStarted           = 0;
constexpr uint32_t StatusCollectingDataFailed = 2;
constexpr uint32_t StatusFinished             = 4;
constexpr uint32_t StatusUnknown              = 5;

struct ShaderDebugInfoIdentifier {
	uint64_t id[2];
};

using GpuCrashDumpCb    = void(__cdecl*)(const void*, uint32_t, void*);
using ShaderDebugInfoCb = void(__cdecl*)(const void*, uint32_t, void*);
using EnableFn = uint32_t(__cdecl*)(uint32_t, uint32_t, uint32_t, GpuCrashDumpCb, ShaderDebugInfoCb,
                                    void*, void*, void*);
using GetStatusFn = uint32_t(__cdecl*)(uint32_t*);
using GetShaderDebugInfoIdentifierFn =
    uint32_t(__cdecl*)(uint32_t, const void*, uint32_t, ShaderDebugInfoIdentifier*);

// GFSDK_Aftermath_GpuCrashDumpDecoding.h (2.2x): the decoder, to print the dump in-process.
using Decoder         = void*;
using CreateDecoderFn = uint32_t(__cdecl*)(uint32_t, const void*, uint32_t, Decoder*);
using DestroyDecoderFn = uint32_t(__cdecl*)(Decoder);
using GenerateJsonFn  = uint32_t(__cdecl*)(Decoder, uint32_t, uint32_t, void*, void*, void*, void*,
                                          uint32_t*);
using GetJsonFn       = uint32_t(__cdecl*)(Decoder, uint32_t, char*);
// GFSDK_Aftermath.h: the hash Aftermath gives a SPIR-V module.
struct SpirvCode {
	const void* data;
	uint32_t    size;
};
struct ShaderBinaryHash {
	uint64_t hash;
};
using GetShaderHashSpirvFn = uint32_t(__cdecl*)(uint32_t, const SpirvCode*, ShaderBinaryHash*);
constexpr uint32_t DecoderAllInfo = 0xFFF;

GetStatusFn                    g_get_status     = nullptr;
GetShaderDebugInfoIdentifierFn g_get_identifier = nullptr;
CreateDecoderFn                g_create_decoder = nullptr;
DestroyDecoderFn               g_destroy_decoder = nullptr;
GenerateJsonFn                 g_generate_json  = nullptr;
GetJsonFn                      g_get_json       = nullptr;
GetShaderHashSpirvFn           g_shader_hash    = nullptr;
bool                           g_enabled        = false;

std::mutex                             g_map_mutex;
std::unordered_map<uint64_t, uint64_t> g_shader_map; // Aftermath hash -> guest shader hash
// The SPIR-V of every shader by Aftermath hash, and the driver's shader debug info by identifier,
// for the decoder to map warp PCs to SPIR-V instructions ("Shader mapping").
std::unordered_map<uint64_t, std::vector<uint32_t>> g_spirv;
std::mutex                                          g_debug_mutex;
std::unordered_map<std::string, std::vector<uint8_t>> g_debug_info; // "<id0>-<id1>"
std::vector<uint8_t>                                g_dump;          // the last crash dump
bool                                                g_debug_missing = false;
bool                                                g_decoded_again = false;

void WriteFile(const std::string& name, const void* data, uint32_t size);

using SetDataFn = void(__cdecl*)(const void*, uint32_t);

std::string DebugInfoName(const ShaderDebugInfoIdentifier& identifier) {
	char name[64];
	std::snprintf(name, sizeof(name), "%016" PRIx64 "-%016" PRIx64, identifier.id[0], identifier.id[1]);
	return name;
}

void __cdecl LookupDebugInfo(const ShaderDebugInfoIdentifier* identifier, SetDataFn set, void* /*user*/) {
	const auto key = DebugInfoName(*identifier);
	{
		std::lock_guard lock(g_debug_mutex);
		if (const auto found = g_debug_info.find(key); found != g_debug_info.end()) {
			set(found->second.data(), static_cast<uint32_t>(found->second.size()));
			return;
		}
	}
	// Written by OnShaderDebugInfo, possibly in an earlier session of the same build.
	const std::string file_name = "_aftermath_" + key + ".nvdbg";
	if (FILE* file = std::fopen(file_name.c_str(), "rb"); file != nullptr) {
		std::vector<uint8_t> data;
		uint8_t              buffer[65536];
		size_t               n = 0;
		while ((n = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
			data.insert(data.end(), buffer, buffer + n);
		}
		std::fclose(file);
		if (!data.empty()) {
			set(data.data(), static_cast<uint32_t>(data.size()));
			return;
		}
	}
	g_debug_missing = true;
	std::fprintf(stderr, "[aftermath] no shader debug info %s yet\n", key.c_str());
}

void __cdecl LookupShader(const ShaderBinaryHash* hash, SetDataFn set, void* /*user*/) {
	std::lock_guard lock(g_map_mutex);
	if (const auto found = g_spirv.find(hash->hash); found != g_spirv.end()) {
		set(found->second.data(), static_cast<uint32_t>(found->second.size() * sizeof(uint32_t)));
		char name[64];
		std::snprintf(name, sizeof(name), "_aftermath_%016" PRIx64 ".spv", hash->hash);
		WriteFile(name, found->second.data(), static_cast<uint32_t>(found->second.size() * sizeof(uint32_t)));
		std::fprintf(stderr, "[aftermath] gave the decoder SPIR-V 0x%016" PRIx64 " (%zu words, %s)\n",
		             hash->hash, found->second.size(), name);
		return;
	}
	std::fprintf(stderr, "[aftermath] no SPIR-V for shader 0x%016" PRIx64 "\n", hash->hash);
}

// The decoded dump on stderr ([aftermath] lines, up to 3,000) and in _aftermath_<time>.json, with
// every Aftermath shader hash it mentions resolved to the guest shader ([aftermath] shader ...).
void DecodeDump(const void* dump, uint32_t size) {
	if (g_create_decoder == nullptr || g_generate_json == nullptr || g_get_json == nullptr) {
		std::fprintf(stderr, "[aftermath] decoder entry points missing; open the .nv-gpudmp in Nsight Graphics\n");
		return;
	}
	Decoder  decoder = nullptr;
	uint32_t result  = g_create_decoder(ApiVersion, dump, size, &decoder);
	if (result != ResultSuccess || decoder == nullptr) {
		std::fprintf(stderr, "[aftermath] CreateDecoder failed (0x%08x)\n", result);
		return;
	}
	uint32_t json_size = 0;
	result = g_generate_json(decoder, DecoderAllInfo, 0, reinterpret_cast<void*>(&LookupDebugInfo),
	                         reinterpret_cast<void*>(&LookupShader), nullptr, nullptr, &json_size);
	if (result != ResultSuccess || json_size == 0) {
		std::fprintf(stderr, "[aftermath] GenerateJSON failed (0x%08x)\n", result);
		if (g_destroy_decoder != nullptr) {
			g_destroy_decoder(decoder);
		}
		return;
	}
	std::string json(json_size, '\0');
	result = g_get_json(decoder, json_size, json.data());
	if (g_destroy_decoder != nullptr) {
		g_destroy_decoder(decoder);
	}
	if (result != ResultSuccess) {
		std::fprintf(stderr, "[aftermath] GetJSON failed (0x%08x)\n", result);
		return;
	}
	while (!json.empty() && json.back() == '\0') {
		json.pop_back();
	}
	static int decodes = 0;
	char       name[64];
	std::snprintf(name, sizeof(name), "_aftermath_%lld_%d.json", static_cast<long long>(std::time(nullptr)),
	              decodes++);
	WriteFile(name, json.data(), static_cast<uint32_t>(json.size()));
	std::fprintf(stderr, "[aftermath] decoded dump: %zu bytes of JSON in %s\n", json.size(), name);
	size_t lines = 0;
	for (size_t begin = 0; begin < json.size() && lines < 3000; lines++) {
		auto end = json.find('\n', begin);
		if (end == std::string::npos) {
			end = json.size();
		}
		std::fprintf(stderr, "[aftermath] %.*s\n", static_cast<int>(std::min<size_t>(end - begin, 400)),
		             json.data() + begin);
		begin = end + 1;
	}
	// v122: the host's history of every resource the fault section names (the handle Aftermath
	// prints is the VkImage handle), whatever address space it was found in.
	{
		static const std::string key = "\"Resource Pointer/Handle\": ";
		size_t                   pos = 0;
		int                      printed = 0;
		while ((pos = json.find(key, pos)) != std::string::npos && printed < 8) {
			pos += key.size();
			const auto handle = std::strtoull(json.c_str() + pos, nullptr, 10);
			if (handle != 0) {
				std::fprintf(stderr, "[aftermath] resource handle %p:\n", reinterpret_cast<void*>(handle));
				ImageHistory::Describe(reinterpret_cast<VkImage>(handle), stderr);
				printed++;
			}
		}
	}
	// v370 (Crash B plan B1): every address-valued field of the dump (the page-fault VA among
	// them; the exact key name differs between Aftermath versions, so any key containing
	// "address" with a number value counts) looked up in the driver's own binding map.
	if (AddressBindings::Enabled()) {
		std::vector<uint64_t> seen;
		size_t                pos = 0;
		while ((pos = json.find("\": ", pos)) != std::string::npos && seen.size() < 6) {
			const auto key_begin = json.rfind('"', pos - 1);
			if (key_begin == std::string::npos) {
				pos += 3;
				continue;
			}
			std::string key = json.substr(key_begin + 1, pos - key_begin - 1);
			std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			pos += 3;
			if (key.find("address") == std::string::npos) {
				continue;
			}
			const char* value = json.c_str() + pos;
			if (*value == '"') {
				value++;
			}
			const auto address = std::strtoull(value, nullptr, 0);
			if (address < 0x10000 || std::find(seen.begin(), seen.end(), address) != seen.end()) {
				continue;
			}
			seen.push_back(address);
			std::fprintf(stderr, "[aftermath] \"%s\" = 0x%016" PRIx64 " in the driver map:\n", key.c_str(), address);
			AddressBindings::Describe(address, 1, stderr);
		}
	}
	{
		std::lock_guard lock(g_map_mutex);
		for (const auto& [aftermath, guest]: g_shader_map) {
			char hex[32], hex_upper[32], dec[32];
			std::snprintf(hex, sizeof(hex), "%016" PRIx64, aftermath);
			std::snprintf(hex_upper, sizeof(hex_upper), "%016" PRIX64, aftermath);
			std::snprintf(dec, sizeof(dec), "%" PRIu64, aftermath);
			if (json.find(hex) != std::string::npos || json.find(hex_upper) != std::string::npos ||
			    json.find(dec) != std::string::npos) {
				std::fprintf(stderr, "[aftermath] shader 0x%016" PRIx64 " in the dump = guest shader %016" PRIx64 "\n",
				             aftermath, guest);
			}
		}
	}
	std::fflush(stderr);
}

void WriteFile(const std::string& name, const void* data, uint32_t size) {
	if (FILE* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
		std::fwrite(data, 1, size, file);
		std::fclose(file);
	}
}

void __cdecl OnGpuCrashDump(const void* dump, uint32_t size, void* /*user_data*/) {
	WriteFile("_aftermath.nv-gpudmp", dump, size);
	g_dump.assign(static_cast<const uint8_t*>(dump), static_cast<const uint8_t*>(dump) + size);
	std::printf("Aftermath: GPU crash dump written to _aftermath.nv-gpudmp (%u bytes)\n", size);
	std::fflush(stdout);
	std::fprintf(stderr, "[aftermath] GPU crash dump received (%u bytes)\n", size);
	DecodeDump(dump, size);
}

void __cdecl OnShaderDebugInfo(const void* info, uint32_t size, void* /*user_data*/) {
	ShaderDebugInfoIdentifier identifier {};
	if (g_get_identifier == nullptr ||
	    g_get_identifier(ApiVersion, info, size, &identifier) != ResultSuccess) {
		return;
	}
	const auto key = DebugInfoName(identifier);
	WriteFile("_aftermath_" + key + ".nvdbg", info, size);
	std::lock_guard lock(g_debug_mutex);
	g_debug_info[key].assign(static_cast<const uint8_t*>(info), static_cast<const uint8_t*>(info) + size);
	std::fprintf(stderr, "[aftermath] shader debug info %s (%u bytes)\n", key.c_str(), size);
}

} // namespace

void Initialize() {
	const char* path = std::getenv("KYTY_AFTERMATH_DLL");
	if (path == nullptr) {
		return;
	}
	HMODULE module = LoadLibraryA(path);
	if (module == nullptr) {
		LOGF("Aftermath: cannot load %s\n", path);
		return;
	}
	auto enable = reinterpret_cast<EnableFn>(
	    reinterpret_cast<void*>(GetProcAddress(module, "GFSDK_Aftermath_EnableGpuCrashDumps")));
	g_get_status = reinterpret_cast<GetStatusFn>(
	    reinterpret_cast<void*>(GetProcAddress(module, "GFSDK_Aftermath_GetCrashDumpStatus")));
	g_get_identifier = reinterpret_cast<GetShaderDebugInfoIdentifierFn>(reinterpret_cast<void*>(
	    GetProcAddress(module, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier")));
	const auto proc = [module](const char* name) {
		return reinterpret_cast<void*>(GetProcAddress(module, name));
	};
	g_create_decoder  = reinterpret_cast<CreateDecoderFn>(proc("GFSDK_Aftermath_GpuCrashDump_CreateDecoder"));
	g_destroy_decoder = reinterpret_cast<DestroyDecoderFn>(proc("GFSDK_Aftermath_GpuCrashDump_DestroyDecoder"));
	g_generate_json   = reinterpret_cast<GenerateJsonFn>(proc("GFSDK_Aftermath_GpuCrashDump_GenerateJSON"));
	g_get_json        = reinterpret_cast<GetJsonFn>(proc("GFSDK_Aftermath_GpuCrashDump_GetJSON"));
	g_shader_hash     = reinterpret_cast<GetShaderHashSpirvFn>(proc("GFSDK_Aftermath_GetShaderHashSpirv"));
	std::fprintf(stderr, "[aftermath] loaded %s: decoder %s, SPIR-V hash %s\n", path,
	             g_create_decoder != nullptr && g_generate_json != nullptr ? "yes" : "NO",
	             g_shader_hash != nullptr ? "yes" : "NO");
	if (enable == nullptr) {
		LOGF("Aftermath: GFSDK_Aftermath_EnableGpuCrashDumps not found\n");
		return;
	}
	const auto result = enable(ApiVersion, WatchedApiVulkan, FeatureDeferDebugInfo, OnGpuCrashDump,
	                           OnShaderDebugInfo, nullptr, nullptr, nullptr);
	LOGF("Aftermath: GPU crash dumps %s (result 0x%08" PRIx32 ")\n",
	     result == ResultSuccess ? "enabled" : "NOT enabled", result);
	std::fprintf(stderr, "[aftermath] GPU crash dumps %s (result 0x%08" PRIx32 ")\n",
	             result == ResultSuccess ? "enabled" : "NOT enabled", result);
	g_enabled = result == ResultSuccess;
}

bool Enabled() {
	return g_enabled;
}

void NoteShader(uint64_t guest_hash, const uint32_t* spirv, size_t words) {
	if (!g_enabled || g_shader_hash == nullptr || spirv == nullptr || words == 0) {
		return;
	}
	const SpirvCode  code {spirv, static_cast<uint32_t>(words * sizeof(uint32_t))};
	ShaderBinaryHash hash {};
	if (g_shader_hash(ApiVersion, &code, &hash) != ResultSuccess) {
		return;
	}
	std::lock_guard lock(g_map_mutex);
	if (g_spirv.find(hash.hash) == g_spirv.end()) {
		g_spirv.emplace(hash.hash, std::vector<uint32_t>(spirv, spirv + words));
	}
	if (g_shader_map.emplace(hash.hash, guest_hash).second) {
		std::fprintf(stderr, "[aftermath-map] 0x%016" PRIx64 " -> guest %016" PRIx64 "\n", hash.hash,
		             guest_hash);
	}
}

void WaitForCrashDump() {
	if (g_get_status == nullptr) {
		return;
	}
	// The dump callback decodes in-process (a few seconds); wait for it. Called also from the
	// host-exception path (09:58: a Vulkan layer's thread crashed on the device loss and took the
	// process down before the dump arrived): a crash the driver has not started collecting within
	// 5 s is not a GPU crash, and the wait ends.
	const auto start    = std::chrono::steady_clock::now();
	const auto deadline = start + std::chrono::seconds(30);
	uint32_t   status   = StatusUnknown;
	while (std::chrono::steady_clock::now() < deadline) {
		g_get_status(&status);
		if (status == StatusFinished || status == StatusCollectingDataFailed) {
			break;
		}
		if (status == StatusNotStarted &&
		    std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	std::printf("Aftermath: crash dump status %u\n", status);
	std::fflush(stdout);
	// The shader debug info can arrive after the dump callback decoded: decode once more with it.
	if (g_debug_missing && !g_decoded_again && !g_dump.empty()) {
		g_decoded_again = true;
		std::fprintf(stderr, "[aftermath] decoding again with the shader debug info received since\n");
		DecodeDump(g_dump.data(), static_cast<uint32_t>(g_dump.size()));
	}
}

#else

void Initialize() {}
void WaitForCrashDump() {}
bool Enabled() {
	return false;
}
void NoteShader(uint64_t, const uint32_t*, size_t) {}

#endif

} // namespace Libs::Graphics::Aftermath
