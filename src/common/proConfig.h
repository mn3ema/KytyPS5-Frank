#ifndef EMULATOR_SRC_COMMON_PROCONFIG_H_
#define EMULATOR_SRC_COMMON_PROCONFIG_H_

#include <cstdint>
#include <filesystem>

// PS5 Pro mode and its upscalers (--trinity, --pssr-upscaler, --dlss-folder, --dlss-preset, --fsr-folder).
// Kept apart from emulatorConfig.h, which the shader cache's source version hashes: these options
// change no shader translation, so they must not re-key the cache.
namespace Config {

enum class PssrUpscaler { Linear, DLSS, FSR };

struct ProOptions {
	bool                  pro_enabled   = false;
	PssrUpscaler          pssr_upscaler = PssrUpscaler::Linear;
	std::filesystem::path dlss_folder;
	uint32_t              dlss_preset   = 0; // 0 is the model default
	std::filesystem::path fsr_folder;
};

// The command line fills it before the emulator starts; read-only afterwards.
ProOptions& MutableProOptions();

bool                         Ps5ProEnabled();
PssrUpscaler                 GetPssrUpscaler();
const std::filesystem::path& GetDlssFolder();
uint32_t                     GetDlssPreset();
const std::filesystem::path& GetFsrFolder();

} // namespace Config

#endif // EMULATOR_SRC_COMMON_PROCONFIG_H_
