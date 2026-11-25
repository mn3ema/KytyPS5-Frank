#include "common/proConfig.h"
#include "common/archive.h"
#include "common/common.h"
#include "common/cpuAffinity.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string_view>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#if defined(KYTY_PGO_GENERATE)
#include <chrono>
#include <cstdlib>
#include <string>
#include <system_error>
#include <thread>

// compiler-rt profile runtime (instrumented builds, CMake KYTY_PGO_GENERATE).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

using namespace Common;
using namespace Emulator;

#if defined(KYTY_PGO_GENERATE)
// The profile runtime writes its counters only at a normal exit, but the run scripts may stop the
// emulator with TerminateProcess. With KYTY_PGO_FILE=<file.profraw>, a thread writes them every
// KYTY_PGO_DUMP_SECONDS (default 20) to <file>.tmp and renames that over <file>, so the file always
// holds the last complete dump.
static void StartPgoDumps() {
	const char* file = std::getenv("KYTY_PGO_FILE");
	if (file == nullptr || *file == '\0') {
		::printf("PGO: instrumented build without KYTY_PGO_FILE: no profile is written\n");
		return;
	}
	const char* seconds_text = std::getenv("KYTY_PGO_DUMP_SECONDS");
	const int   seconds =
	    (seconds_text != nullptr && std::atoi(seconds_text) > 0) ? std::atoi(seconds_text) : 20;
	static const std::string target = file;
	static const std::string temp   = target + ".tmp";
	::printf("PGO: writing the profile to %s every %d s\n", target.c_str(), seconds);
	std::thread([seconds]() {
		for (uint32_t dump = 1;; dump++) {
			std::this_thread::sleep_for(std::chrono::seconds(seconds));
			__llvm_profile_set_filename(temp.c_str());
			const int       written = __llvm_profile_write_file();
			std::error_code error;
			if (written == 0) {
				std::filesystem::rename(temp, target, error);
			}
			::printf("PGO: dump %u %s\n", dump,
			         written != 0 ? "failed to write" : (error ? "failed to rename" : "written"));
			::fflush(stdout);
		}
	}).detach();
}
#endif

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler = Debug::GetCompiler() + "-" + Debug::GetLinker();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --exact-denormals                    Flush FP32 denormals in software as the PS5\n"
	         "                                       does (exact below 1.2e-38; slower on GPUs\n"
	         "                                       without hardware flush, e.g. NVIDIA).\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --trinity                            Enable experimental PS5 Pro mode for\n"
	         "                                       supported titles. Default: off.\n");
	::printf("  --pssr-upscaler <value>              With --trinity, upscale PSSR with Linear,\n"
	         "                                       DLSS, or FSR. Default: Linear.\n");
	::printf("  --dlss-folder <folder>               DLSS model folder.\n");
	::printf("  --dlss-preset <A-O>                  DLSS preset. Default: the model's own.\n");
	::printf("  --fsr-folder <folder>                FidelityFX DLL folder (Windows).\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --output-resolution <1080|2160|auto> TV resolution the game is told. Default: auto\n"
	         "                                       (2160 unless the game asks for detection).\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --vulkan-validation-fatal <t|f>      Exit on the first validation error (default\n"
	         "                                       true); false logs errors and continues.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf(
	    "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
	::printf(
	    "  --trophy-notifications <true|false>   Show trophy unlock toasts and play their sound.\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf("  --bindless                           Sample heap-indexed images through bindless arrays.\n");
	::printf("  --no-float-image-atomics             Skip shaders that run integer image atomics on R32\n"
	         "                                       float images (they run by default).\n");
	::printf("  --skip-shaders <hash,...>            Skip the draws and dispatches of these guest shaders\n"
	         "                                       (hex hashes, as logged).\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value, magic_enum::case_insensitive);
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (arg == "--fullscreen") {
			options.config.fullscreen_enabled = true;
			continue;
		}

		if (arg == "--exact-denormals") {
			// The guest's FP32 denormal flushes, emulated in software (KYTY_DENORM_FLUSH=1).
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			_putenv_s("KYTY_DENORM_FLUSH", "1");
#else
			setenv("KYTY_DENORM_FLUSH", "1", 1);
#endif
			continue;
		}

		if (arg == "--hide-cursor") {
			options.config.hide_cursor_enabled = true;
			continue;
		}

		if (arg == "--vr") {
			options.config.vr_enabled = true;
			continue;
		}

		// PS5 Pro mode for titles that support it (PSSR, which --pssr-upscaler maps to a host upscaler).
		if (arg == "--trinity") {
			Config::MutableProOptions().pro_enabled = true;
			continue;
		}

		if (arg == "--amd-cpu") {
			options.config.amd_cpu_enabled = true;
			continue;
		}

		if (arg == "--playgo-hack") {
			options.config.playgo_hack_enabled = true;
			continue;
		}

		if (arg == "--tessellation") {
			options.config.tessellation_enabled = true;
			continue;
		}

		if (arg == "--profile") {
			options.config.profiler_enabled = true;
			continue;
		}
		if (arg == "--bindless") {
			options.config.bindless_images_enabled = true;
			continue;
		}
		if (arg == "--float-image-atomics" || arg == "--no-float-image-atomics") {
			options.config.float_image_atomics_enabled = arg == "--float-image-atomics";
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
#endif

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--pssr-upscaler") {
			if (!ParseEnum(value, Config::MutableProOptions().pssr_upscaler)) {
				::printf("invalid PSSR upscaler: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--dlss-folder") {
			Config::MutableProOptions().dlss_folder = Common::PathFromUtf8(value);
		} else if (arg == "--dlss-preset") {
			if (value.size() != 1 || value[0] < 'A' || value[0] > 'O') {
				::printf("invalid DLSS preset: %s\n", value.c_str());
				return false;
			}
			Config::MutableProOptions().dlss_preset = static_cast<uint32_t>(value[0] - 'A' + 1);
		} else if (arg == "--fsr-folder") {
			Config::MutableProOptions().fsr_folder = Common::PathFromUtf8(value);
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--output-resolution") {
			// The TV resolution the game is told (VideoOutGetOutputStatus): 1080 or 2160.
			if (value != "1080" && value != "2160" && value != "auto") {
				::printf("invalid output resolution: %s (1080, 2160 or auto)\n", value.c_str());
				return false;
			}
			if (value != "auto") {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
				_putenv_s("KYTY_OUTPUT_RESOLUTION", value.c_str());
#else
				setenv("KYTY_OUTPUT_RESOLUTION", value.c_str(), 1);
#endif
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-assisted-validation") {
			if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation-fatal") {
			if (!ParseBool(value, options.config.vulkan_validation_fatal)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--skip-shaders") {
			options.config.skip_shader_hashes = value;
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trophy-notifications") {
			if (!ParseBool(value, options.config.trophy_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

// KytyPS5-Frank: the settings tuned on Marvel's Wolverine are the defaults. Each environment
// variable already set (to anything) wins; KYTY_FRANK_DEFAULTS=0 leaves all of them unset and the
// command line options at KytyPS5's defaults.
static bool FrankDefaultsEnabled() {
	const char* value = std::getenv("KYTY_FRANK_DEFAULTS");
	return value == nullptr || value[0] != '0';
}

static void ApplyFrankEnvironmentDefaults() {
	if (!FrankDefaultsEnabled()) {
		return;
	}
	static constexpr std::pair<const char*, const char*> Defaults[] = {
	    {"KYTY_BINDLESS_EVICT", "0"},
	    // Slot reuse on (2026-11-01): with it off the 2D-array binding's 65,536 slots run out after
	    // ~25 minutes of play (black silhouettes, 3-10 fps; plan 10-23, v122 fixed the double free
	    // reuse exposed). KYTY_BINDLESS_SLOT_REUSE=0 restores the old setting.
	    {"KYTY_BINDLESS_SLOT_REUSE", "1"},
	    // Multi-draw replay on (2026-11-01): ~100 ms/s off the GPU thread in Mac's opening stretch,
	    // no artifacts in 7 runs (m87 toggled, m88 and m90-m94 always on). KYTY_MULTI_DRAW_REPLAY=0
	    // turns it off.
	    {"KYTY_MULTI_DRAW_REPLAY", "1"},
	    {"KYTY_SHARED_BUFFERS", "1"},
	    {"KYTY_WRITE_BITMAP", "1"},
	    // Ray traversal (dc76e1223a9bf673) loops without end on our side: a 2048-step cap.
	    {"KYTY_LOOP_GUARD_HASHES", "dc76e1223a9bf673"},
	    {"KYTY_LOOP_GUARD_LIMIT", "2048"},
	    // No barriers between dispatches the game's command stream does not order.
	    {"KYTY_DISPATCH_RUNS", "1"},
	    // No NVIDIA crash-diagnostic markers per draw and dispatch (they serialize the GPU).
	    {"KYTY_NV_CHECKPOINTS", "0"},
	    {"KYTY_SRT_DEFER", "1"},
	    // GPU-driven mesh draws convert their counts in batches, not one dispatch, two drains and
	    // a rendering restart each (jungle median 18.5 -> 20.5-22.3 fps).
	    {"KYTY_MESH_ARGS_RUNS", "1"},
	    // Guest-visible host buffers over the CPU-written hot ranges, instead of page faults per
	    // write (sewer standing 20.1 -> 22.6 fps, roof 19.7 -> 22.0; KYTY_HOST_ZONES=0 turns off).
	    {"KYTY_HOST_ZONES", "1"},
	    // Repeated syncs of a range with no page changed skip the tracker walk (37-52 % skipped).
	    {"KYTY_SYNC_SKIP", "1"},
	    // A tracked binding's predicted ranges synced and marked in one pass (s1tbatch, 2026-11-09,
	    // in-run A/B over 18 pairs: GPU thread 827 vs 832 ms/s, fps 26.6 vs 25.8; Mac: keep).
	    // KYTY_TRACKED_BATCH=0 turns it off.
	    {"KYTY_TRACKED_BATCH", "1"},
	    // Exact per-draw/per-dispatch cuts (s2hot, 2026-11-09, in-run A/B over 18 pairs: GPU thread
	    // 830 vs 843 ms/s, lower in 14 of 18 pairs; fps 26.5 vs 25.7). KYTY_HOTPATH=0 turns them off.
	    {"KYTY_HOTPATH", "1"},
	    // Pages the write bitmap releases stay no-access until a CPU access resolves them (the next
	    // written binding arms them again): s10lazy, 2026-11-09, in-run A/B over 18 pairs: GPU
	    // thread 807 vs 816 ms/s, fps 28.05 vs 27.55 (one release site; v366 adds the settle).
	    // KYTY_LAZY_READ_RELEASE=0 turns it off.
	    {"KYTY_LAZY_READ_RELEASE", "1"},
	    // Zero-compile (v0.4): graphics pipeline libraries (used only when the driver offers
	    // VK_EXT_graphics_pipeline_library) with per-stage descriptor sets, the startup library and
	    // compute warm-up and the sharded driver caches. KYTY_GPL=0 / KYTY_STAGE_SETS=0 turn them off.
	    {"KYTY_GPL", "1"},
	    {"KYTY_STAGE_SETS", "1"},
	    // v438g (Mac 11-15): the frame keeps its aspect ratio (16:9 on a 16:10 panel gets black
	    // bars instead of a stretch). KYTY_KEEP_ASPECT=0: fill the window.
	    {"KYTY_KEEP_ASPECT", "1"},
	    // v438zy (Mac 11-17, "make it default"): sparse textures (only the levels a view can
	    // reach get memory) on NVIDIA GPUs when KYTY_TEX_SPARSE is not set. mac6 play (v438zu):
	    // hitches 31.6 -> 4.5 per minute, 1% low 6.3 -> 12.4 fps, device memory flat at 12-13 GB.
	    // KYTY_TEX_SPARSE=0 turns it off; KYTY_TEX_SPARSE_NVIDIA=0 keeps NVIDIA off by default.
	    {"KYTY_TEX_SPARSE_NVIDIA", "2"},
	    // v0.5 (Mac's play settings since 11-21, built in):
	    // The glass reflection resolve 2feed4c0d841a3c4 only draws artifacts until the ray-tracing
	    // shaders translate; the four big ray-tracing compute shaders that never translate
	    // (unsupported SOP1 opcode 0x21, over the IR bounds) are skipped outright instead of retried
	    // (a retry froze the game for minutes). KYTY_SKIP_SHADER_HASHES= (empty) skips none.
	    {"KYTY_SKIP_SHADER_HASHES",
	     "2feed4c0d841a3c4,fc0537b7278db60c,384f47083f0d5e08,29be8047c4e86afe,e6d76d24f59f8015"},
	    // Tessellated pipelines are built whole, not linked from pipeline libraries (linked, the
	    // water and terrain tessellation drew nothing). KYTY_GPL_SKIP=0 links them.
	    {"KYTY_GPL_SKIP", "2"},
	    // No dispatch-run exclusions: 06544f382c6b0850 and 3378b4a64461d6ca overlap like the rest
	    // (in-run A/B -1.2 and -3.0 ms a frame, no GPU hang in 10 boots).
	    {"KYTY_DISPATCH_RUN_EXCLUDE", "0"},
	};
	for (const auto& [name, value]: Defaults) {
		if (std::getenv(name) == nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			_putenv_s(name, value);
#else
			setenv(name, value, 0);
#endif
		}
	}
}

static void ApplyFrankOptionDefaults(RunOptions& options) {
	if (!FrankDefaultsEnabled()) {
		return;
	}
	options.config.bindless_images_enabled = true;
	options.config.tessellation_enabled    = true;
	options.config.readback_linear_images  = true;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	options.config.red_zone_protection_enabled = true;
#endif
	(void)ParseEnum(std::string("Performance"), options.config.shader_optimization_type);
	// v0.5: the guest's vblank at 120 Hz, as in Mac's play (--vblank-frequency overrides).
	options.config.vblank_frequency = 120;
}

static int Main(int argc, char* argv[]) {
	ApplyFrankEnvironmentDefaults();
	Common::OptOutOfPowerThrottling();
	VirtualMemory::Init();
	InitializeThreads();
#if defined(KYTY_PGO_GENERATE)
	// After the address-space reservations: the thread's stack must not land in the guest range.
	StartPgoDumps();
#endif

	RunOptions options;
	bool       show_help = false;
	ApplyFrankOptionDefaults(options);

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
