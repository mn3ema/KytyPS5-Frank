// Offline batch translation of guest shaders extracted from the game files: <hash>.header (the
// AGC header, pointers self-relative), <hash>.code and manifest.csv in one folder.
//
// Each shader takes the path a draw or dispatch takes in-game: AgcCreateShader, the header's
// registers decoded into the hardware state as the PM4 handlers do, PrepareProgram, then
// TranslateProgram and ExtractResourcePlan with ProgramCache::Get's options (non-fatal, bindless
// images). Draw-time state the header does not hold gets defaults: the input primitive that fits
// the header's GE_CNTL group size (points, lines or triangles), identity pixel interpolators, the
// default export mapping, and stand-in vertex attribute and buffer tables built from the
// header's own input semantics. The host subgroup size is 32, as on
// NVIDIA. Shader function calls are not expanded: their callees live in game memory.
//
// usage: shader_batch_tool <shader folder> <log file> [first manifest row] [end row]
// Prints "START <row> <hash>" before and one tab-separated RESULT line after each shader. A
// fatal error exits with code 321 after START, so a driver can resume at the next row. The
// RESULT line lists every instruction the decoder does not support, not only the first one the
// CFG stops at. KYTY_BATCH_DUMP_DIR makes rejected shaders dump their IR (see main).

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderLedger.h"
#include "xxhash.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "libs/agc.h"
#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <malloc.h>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace Libs::Graphics {
namespace {

// KYTY_BATCH_LEDGER=<folder>: every shader's source recipe (the source part of the disk-cache key,
// with the stage input this tool builds) goes to <folder>/ledger.bin, in the emulator's format, so
// it can be compared with the recipes the game recorded (zero-compile plan, step 5: how much of a
// shader's draw-time key the header alone predicts).
std::unique_ptr<ShaderLedger::Ledger> g_ledger;

// KYTY_BATCH_PAIRING=1 (zero-compile plan step 5): a pixel shader's SPI_PS_INPUT_CNTL comes from
// the game's AGC library (AgcCreateInterpolantMapping) against the vertex-shader output layouts
// found in the folder (the game has 7), using the first layout that supplies every input semantic,
// instead of identity settings. A pixel shader with KILL_ENABLE also records the source the game
// uses with kill off (and early Z): both occur in play.
bool                                     g_pairing = false;

// KYTY_BATCH_RESOURCES=1: one RES line per buffer and image of the translated program, with its
// usage (what the instructions ask of it) and where its descriptor comes from (a structural
// signature of the descriptor's first dword: opcodes, user-data registers and immediates), for
// learning which specializations unplayed shaders get (zero-compile plan step 5).
bool g_resources = false;

std::string ValueSignature(const ShaderRecompiler::IR::Value& input, int depth) {
	using namespace ShaderRecompiler::IR;
	const Value value = input.Resolve();
	if (value.IsEmpty()) {
		return "_";
	}
	if (value.IsImmediate()) {
		return fmt::format("#{:x}", value.ImmediateBits());
	}
	if (value.GetType() == Type::ScalarReg) {
		return fmt::format("s{}", RegIndex(value.ScalarRegister()));
	}
	if (value.GetType() == Type::VectorReg) {
		return "v";
	}
	const Inst* inst = value.TryInstruction();
	if (inst == nullptr) {
		return "?";
	}
	if (value.IsPhi()) {
		return "phi";
	}
	std::string text(ValueOpcodeName(inst->GetOpcode()));
	if (depth <= 0) {
		return text + "(..)";
	}
	text += '(';
	for (size_t i = 0; i < inst->NumArgs(); i++) {
		text += (i == 0 ? "" : ",") + ValueSignature(inst->Arg(i), depth - 1);
	}
	return text + ')';
}
std::vector<std::vector<ShaderSemantic>> g_vs_layouts;

bool LayoutFits(const Shader& ps, const std::vector<ShaderSemantic>& layout) {
	for (uint32_t i = 0; i < ps.num_input_semantics; i++) {
		if (std::none_of(layout.begin(), layout.end(), [&](const ShaderSemantic& out) {
			    return out.semantic == ps.input_semantics[i].semantic;
		    })) {
			return false;
		}
	}
	return true;
}

#ifdef _WIN32
// A crash (not a fatal error, which exits with 321) would lose the buffered log. Print the
// fault and the host stack on one CRASH line before the process dies, so the driver can say
// where the shader crashed. C++ exceptions, which the recompiler uses internally, pass through.
LONG WINAPI ReportCrash(EXCEPTION_POINTERS* info) {
	const auto code = info->ExceptionRecord->ExceptionCode;
	if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_STACK_OVERFLOW &&
	    code != EXCEPTION_ILLEGAL_INSTRUCTION && code != EXCEPTION_INT_DIVIDE_BY_ZERO &&
	    code != EXCEPTION_PRIV_INSTRUCTION && code != EXCEPTION_ARRAY_BOUNDS_EXCEEDED) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	auto stack = Common::HostBacktrace();
	std::replace(stack.begin(), stack.end(), '\n', '|');
	std::printf("CRASH\t0x%08lx\t0x%016llx\t%s\n", static_cast<unsigned long>(code),
	            static_cast<unsigned long long>(
	                reinterpret_cast<uintptr_t>(info->ExceptionRecord->ExceptionAddress)),
	            stack.c_str());
	std::fflush(stdout);
	return EXCEPTION_CONTINUE_SEARCH;
}
#endif

// Shader checksum registers (COMPUTE_SHADER_CHKSUM, SPI_SHADER_PGM_CHKSUM_PS/GS): no state.
constexpr uint32_t ComputeChecksum = 0x22a;
constexpr uint32_t PixelChecksum   = 0x006;
constexpr uint32_t GeometryChecksum = 0x080;

struct ManifestRow {
	uint64_t    hash = 0;
	std::string type;
};

std::vector<ManifestRow> ReadManifest(const std::filesystem::path& folder) {
	std::ifstream            file(folder / "manifest.csv");
	std::vector<ManifestRow> rows;
	std::string              line;
	std::getline(file, line);
	while (std::getline(file, line)) {
		const auto first  = line.find(',');
		const auto second = line.find(',', first + 1);
		if (line.empty() || first == std::string::npos || second == std::string::npos) {
			continue;
		}
		rows.push_back({.hash = std::strtoull(line.substr(0, first).c_str(), nullptr, 16),
		                .type = line.substr(first + 1, second - first - 1)});
	}
	return rows;
}

std::vector<char> ReadFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Kept alive for the whole run: the shader map and the hash cache hold these addresses.
void* CopyAligned(const std::vector<char>& bytes, size_t alignment) {
	void* memory = _aligned_malloc(std::max<size_t>(bytes.size(), 1), alignment);
	std::memcpy(memory, bytes.data(), bytes.size());
	return memory;
}

struct GuestState {
	HW::Context           context;
	HW::UserConfig        user_config;
	HW::ComputeShaderInfo compute;
	HW::PixelShaderInfo   pixel;
	HW::VertexShaderInfo  vertex;
	uint32_t              dispatch_modifier = 0;
	std::string           unhandled;
};

void NoteUnhandled(GuestState& state, const char* space, uint32_t offset) {
	state.unhandled += fmt::format("{}{}:0x{:03x}", state.unhandled.empty() ? "" : " ", space, offset);
}

uint64_t SetAddressLo(uint64_t address, uint32_t value) {
	return (address & 0xFFFFFF00000000FFull) | (static_cast<uint64_t>(value) << 8u);
}

uint64_t SetAddressHi(uint64_t address, uint32_t value) {
	return (address & 0xFFFF00FFFFFFFFFFull) | ((static_cast<uint64_t>(value) & 0xffu) << 40u);
}

// The context registers the headers carry, decoded as pm4Handlers.cpp does.
void SetContextRegister(GuestState& state, uint32_t offset, uint32_t value) {
	auto& ctx = state.context;
	switch (offset) {
		case Pm4::SPI_PS_INPUT_ENA: ctx.SetPsInputEna(value); break;
		case Pm4::SPI_PS_INPUT_ADDR: ctx.SetPsInputAddr(value); break;
		case Pm4::SPI_PS_IN_CONTROL: ctx.SetPsInControl(value); break;
		case Pm4::SPI_BARYC_CNTL: ctx.SetBarycCntl(value); break;
		case Pm4::SPI_SHADER_Z_FORMAT: ctx.SetShaderZFormat(value); break;
		case Pm4::SPI_SHADER_COL_FORMAT:
			for (uint32_t i = 0; i < 8; i++) {
				ctx.SetTargetOutputMode(i, (value >> (i * 4)) & 0xFu);
			}
			break;
		case Pm4::DB_SHADER_CONTROL: {
			HW::DepthShaderControl control {};
			control.other_bits = value & 0xFFFF908Eu;
			control.conservative_z_export_value =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, CONSERVATIVE_Z_EXPORT);
			control.shader_z_behavior  = KYTY_PM4_GET(value, DB_SHADER_CONTROL, Z_ORDER);
			control.shader_kill_enable = KYTY_PM4_GET(value, DB_SHADER_CONTROL, KILL_ENABLE) != 0;
			control.shader_z_export_enable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, Z_EXPORT_ENABLE) != 0;
			control.shader_mask_export_enable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, MASK_EXPORT_ENABLE) != 0;
			control.shader_dual_export_enable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, DUAL_EXPORT_ENABLE) != 0;
			control.shader_execute_on_noop =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, EXEC_ON_NOOP) != 0;
			control.alpha_to_mask_disable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, ALPHA_TO_MASK_DISABLE) != 0;
			ctx.SetDepthShaderControl(control);
			break;
		}
		case Pm4::CB_SHADER_MASK: ctx.SetShaderMask(value); break;
		case Pm4::PA_SC_SHADER_CONTROL: ctx.SetScShaderControl(value); break;
		case Pm4::SPI_VS_OUT_CONFIG: ctx.SetVsOutConfig(value); break;
		case Pm4::SPI_SHADER_POS_FORMAT: ctx.SetShaderPosFormat(value); break;
		case Pm4::SPI_SHADER_IDX_FORMAT: ctx.SetShaderIdxFormat(value); break;
		case Pm4::PA_CL_VS_OUT_CNTL: ctx.SetClVsOutCntl(value); break;
		case Pm4::GE_NGG_SUBGRP_CNTL: ctx.SetNggSubgrpCntl(value); break;
		case Pm4::VGT_GS_INSTANCE_CNT: ctx.SetGsInstanceCnt(value); break;
		case Pm4::VGT_GS_ONCHIP_CNTL: ctx.SetGsOnchipCntl(value); break;
		case Pm4::GE_MAX_OUTPUT_PER_SUBGROUP: ctx.SetMaxOutputPerSubgroup(value); break;
		case Pm4::VGT_ESGS_RING_ITEMSIZE: ctx.SetEsgsRingItemsize(value); break;
		case Pm4::VGT_GS_MAX_VERT_OUT: ctx.SetGsMaxVertOut(value); break;
		case Pm4::VGT_SHADER_STAGES_EN: ctx.SetShaderStages(value); break;
		case Pm4::VGT_GS_OUT_PRIM_TYPE: ctx.SetGsOutPrimType(value); break;
		default: NoteUnhandled(state, "cx", offset); break;
	}
}

// The SH registers the headers carry, decoded as pm4Handlers.cpp does.
void SetShRegister(GuestState& state, uint32_t offset, uint32_t value) {
	auto& cs = state.compute.cs_regs;
	auto& ps = state.pixel.ps_regs;
	auto& vs = state.vertex;
	switch (offset) {
		case Pm4::COMPUTE_PGM_LO: cs.data_addr = SetAddressLo(cs.data_addr, value); break;
		case Pm4::COMPUTE_PGM_HI: cs.data_addr = SetAddressHi(cs.data_addr, value); break;
		case Pm4::COMPUTE_PGM_RSRC1:
			cs.vgprs =
			    (value >> Pm4::COMPUTE_PGM_RSRC1_VGPRS_SHIFT) & Pm4::COMPUTE_PGM_RSRC1_VGPRS_MASK;
			cs.priority = (value >> Pm4::COMPUTE_PGM_RSRC1_PRIORITY_SHIFT) &
			              Pm4::COMPUTE_PGM_RSRC1_PRIORITY_MASK;
			cs.float_mode = (value >> Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_MASK;
			cs.dx10_clamp = ((value >> Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_MASK) != 0u;
			cs.debug_mode = ((value >> Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_MASK) != 0u;
			cs.ieee_mode = ((value >> Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_MASK) != 0u;
			cs.fp16_overflow = ((value >> Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_SHIFT) &
			                    Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_MASK) != 0u;
			cs.threadgroup_configuration = ((value >> Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT) &
			                                Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_MASK) != 0u;
			cs.require_forward_progress = ((value >> Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT) &
			                               Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_MASK) != 0u;
			break;
		case Pm4::COMPUTE_PGM_RSRC2:
			cs.scratch_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_MASK) != 0u;
			cs.user_sgpr = (value >> Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT) &
			               Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_MASK;
			cs.tgid_x_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0u;
			cs.tgid_y_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0u;
			cs.tgid_z_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0u;
			cs.tg_size_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0u;
			cs.tidig_comp_cnt = (value >> Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT) &
			                    Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK;
			cs.lds_size =
			    (value >> Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT) & Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK;
			break;
		case Pm4::COMPUTE_PGM_RSRC3:
			cs.shared_vgprs = (value >> Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_SHIFT) &
			                  Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_MASK;
			break;
		case Pm4::COMPUTE_NUM_THREAD_X: cs.num_thread_x = value; break;
		case Pm4::COMPUTE_NUM_THREAD_Y: cs.num_thread_y = value; break;
		case Pm4::COMPUTE_NUM_THREAD_Z: cs.num_thread_z = value; break;
		case Pm4::SPI_SHADER_PGM_LO_PS: ps.data_addr = SetAddressLo(ps.data_addr, value); break;
		case Pm4::SPI_SHADER_PGM_HI_PS: ps.data_addr = SetAddressHi(ps.data_addr, value); break;
		case Pm4::SPI_SHADER_PGM_RSRC1_PS: {
			HW::PsShaderResource1 r1;
			r1.vgprs            = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, VGPRS);
			r1.priority         = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, PRIORITY);
			r1.float_mode       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, FLOAT_MODE);
			r1.dx10_clamp       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, DX10_CLAMP) != 0;
			r1.debug_mode       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, DEBUG_MODE) != 0;
			r1.ieee_mode        = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, IEEE_MODE) != 0;
			r1.cu_group_disable = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, CU_GROUP_DISABLE) != 0;
			r1.require_forward_progress =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, FWD_PROGRESS) != 0;
			r1.fp16_overflow = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, FP16_OVFL) != 0;
			ps.rsrc1         = r1;
			break;
		}
		case Pm4::SPI_SHADER_PGM_RSRC2_PS: {
			HW::PsShaderResource2 r2;
			r2.scratch_en = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, SCRATCH_EN);
			r2.user_sgpr  = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, USER_SGPR) +
			                (KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, USER_SGPR_MSB) << 5u);
			r2.wave_cnt_en    = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, WAVE_CNT_EN);
			r2.extra_lds_size = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, EXTRA_LDS_SIZE);
			r2.raster_ordered_shading =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, RASTER_ORDERED_SHADING);
			r2.shared_vgprs = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, SHARED_VGPR_CNT);
			ps.rsrc2        = r2;
			break;
		}
		case Pm4::SPI_SHADER_PGM_LO_ES:
			vs.es_regs.data_addr = SetAddressLo(vs.es_regs.data_addr, value);
			break;
		case Pm4::SPI_SHADER_PGM_HI_ES:
			vs.es_regs.data_addr = SetAddressHi(vs.es_regs.data_addr, value);
			break;
		case Pm4::SPI_SHADER_PGM_RSRC1_GS: {
			HW::GsShaderResource1 r1;
			r1.vgprs           = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, VGPRS);
			r1.priority        = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, PRIORITY);
			r1.float_mode      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, FLOAT_MODE);
			r1.dx10_clamp      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, DX10_CLAMP) != 0;
			r1.debug_mode      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, DEBUG_MODE) != 0;
			r1.ieee_mode       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, IEEE_MODE) != 0;
			r1.cu_group_enable = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, CU_GROUP_ENABLE) != 0;
			r1.require_forward_progress =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, FWD_PROGRESS) != 0;
			r1.threadgroup_configuration =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, WGP_MODE) != 0;
			r1.gs_vgpr_component_count =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, GS_VGPR_COMP_CNT);
			r1.fp16_overflow = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, FP16_OVFL) != 0;
			vs.gs_regs.rsrc1 = r1;
			break;
		}
		case Pm4::SPI_SHADER_PGM_RSRC2_GS: {
			HW::GsShaderResource2 r2;
			r2.scratch_en = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, SCRATCH_EN) != 0;
			r2.user_sgpr  = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, USER_SGPR) +
			                (KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, USER_SGPR_MSB) << 5u);
			r2.es_vgpr_component_count =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, ES_VGPR_COMP_CNT);
			r2.offchip_lds   = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, OC_LDS_EN) != 0;
			r2.lds_size      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, LDS_SIZE);
			r2.shared_vgprs  = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, SHARED_VGPR_CNT);
			vs.gs_regs.rsrc2 = r2;
			break;
		}
		case ComputeChecksum:
		case PixelChecksum:
		case GeometryChecksum: break;
		default: NoteUnhandled(state, "sh", offset); break;
	}
}

void SetUserConfigRegister(GuestState& state, uint32_t offset, uint32_t value) {
	switch (offset) {
		case Pm4::GE_CNTL: {
			HW::GeControl control;
			control.primitive_group_size = KYTY_PM4_GET(value, GE_CNTL, PRIM_GRP_SIZE);
			control.vertex_group_size    = KYTY_PM4_GET(value, GE_CNTL, VERT_GRP_SIZE);
			state.user_config.SetGeControl(control);
			break;
		}
		case Pm4::GE_USER_VGPR_EN: {
			HW::GeUserVgprEn enable;
			enable.vgpr1 = KYTY_PM4_GET(value, GE_USER_VGPR_EN, EN_USER_VGPR1) != 0;
			enable.vgpr2 = KYTY_PM4_GET(value, GE_USER_VGPR_EN, EN_USER_VGPR2) != 0;
			enable.vgpr3 = KYTY_PM4_GET(value, GE_USER_VGPR_EN, EN_USER_VGPR3) != 0;
			state.user_config.SetGeUserVgprEn(enable);
			break;
		}
		default: NoteUnhandled(state, "uc", offset); break;
	}
}

void ApplyHeaderRegisters(GuestState& state, const Shader& shader) {
	for (uint32_t i = 0; shader.cx_registers != nullptr && i < shader.num_cx_registers; i++) {
		SetContextRegister(state, shader.cx_registers[i].offset, shader.cx_registers[i].value);
	}
	for (uint32_t i = 0; shader.sh_registers != nullptr && i < shader.num_sh_registers; i++) {
		SetShRegister(state, shader.sh_registers[i].offset, shader.sh_registers[i].value);
	}
	if (shader.specials == nullptr) {
		return;
	}
	const auto& specials    = *shader.specials;
	state.dispatch_modifier = specials.dispatch_modifier;
	if (specials.ge_cntl.offset != 0) {
		SetUserConfigRegister(state, specials.ge_cntl.offset, specials.ge_cntl.value);
	}
	if (specials.vgt_shader_stages_en.offset != 0) {
		SetContextRegister(state, specials.vgt_shader_stages_en.offset,
		                   specials.vgt_shader_stages_en.value);
	}
	if (specials.vgt_gs_out_prim_type.offset != 0) {
		SetContextRegister(state, specials.vgt_gs_out_prim_type.offset,
		                   specials.vgt_gs_out_prim_type.value);
	}
	if (specials.ge_user_vgpr_en.offset != 0) {
		SetUserConfigRegister(state, specials.ge_user_vgpr_en.offset,
		                      specials.ge_user_vgpr_en.value);
	}
}

// Stand-ins for the draw's vertex attribute and buffer tables: every semantic the header declares
// gets a 32-bit float attribute with its channel count, in a buffer of its own.
struct VertexTables {
	std::array<uint32_t, 256>    attrib {};
	std::array<uint32_t, 4 * 32> buffer {};
};

bool PrepareVertexTables(const Shader& shader, HW::VertexShaderInfo& vertex, VertexTables& tables,
                         std::string& note) {
	ShaderMappedData data;
	data.type                = static_cast<Prospero::ShaderBinaryType>(shader.type);
	data.user_data           = shader.user_data;
	data.input_semantics     = shader.input_semantics;
	data.num_input_semantics = shader.num_input_semantics;
	if (shader.input_semantics != nullptr) {
		std::copy_n(shader.input_semantics,
		            std::min<uint32_t>(shader.num_input_semantics, ShaderMappedData::MaxInputSemantics),
		            data.input_semantics_snapshot.begin());
	}
	data.code_size_bytes     = shader.shader_size;
	data.scratch_size_dwords = shader.scratch_size_dw_per_thread;
	ShaderVertexMetadata metadata;
	std::string          error;
	if (!ShaderReadVertexMetadata(data, HW::UserSgprInfo::SGPRS_MAX, metadata, &error)) {
		note = "vertex metadata: " + error;
		return false;
	}
	if (metadata.vertex_attrib_reg < 0) {
		return true;
	}
	// VertexAttribFormat k32Float, k32_32Float, k32_32_32Float, k32_32_32_32Float.
	static constexpr std::array<uint32_t, 5> float_formats = {311, 88, 257, 298, 311};
	for (const auto& semantic: metadata.input_semantics) {
		const auto format = float_formats[std::min<uint32_t>(semantic.size_in_elements, 4u)];
		tables.attrib[semantic.semantic] = (semantic.semantic & 0x1fu) | (format << 5u);
	}
	for (uint32_t i = 0; i < 32; i++) {
		const uint64_t base      = 0x100000000ull + i * 0x1000000ull;
		tables.buffer[i * 4 + 0] = static_cast<uint32_t>(base);
		tables.buffer[i * 4 + 1] = static_cast<uint32_t>(base >> 32u) | (16u << 16u);
		tables.buffer[i * 4 + 2] = 0x10000u;
		tables.buffer[i * 4 + 3] = DstSel(4, 5, 6, 7);
	}
	const auto set_pointer = [&](int reg, const void* table) {
		const auto address                = reinterpret_cast<uint64_t>(table);
		vertex.gs_user_sgpr.value[reg]     = static_cast<uint32_t>(address);
		vertex.gs_user_sgpr.value[reg + 1] = static_cast<uint32_t>(address >> 32u);
	};
	set_pointer(metadata.vertex_attrib_reg, tables.attrib.data());
	set_pointer(metadata.vertex_buffer_reg, tables.buffer.data());
	note = fmt::format("stand-in vertex tables ({} semantics)", metadata.input_semantics.size());
	return true;
}

// Stand-in guest memory and user data: one 16-byte pattern laid over every address and over the
// user SGPRs. Read as a buffer descriptor (V#, 16-byte aligned) it is a raw 2 GB buffer; as an
// image descriptor (T#, 32-byte aligned) a null image, as dword 0 is zero; as a pointer, an
// address below 2^48 that reads the same pattern. Materialization then succeeds with unbound
// images, as when a game draws with nothing bound to a slot.
constexpr std::array<uint32_t, 4> MemoryPattern = {0u, 0u, 0x7ffffff0u, DstSel(4, 5, 6, 7)};

bool ReadPatternMemory(void* /*userdata*/, uint64_t address, std::span<uint32_t> values) {
	for (size_t i = 0; i < values.size(); i++) {
		values[i] = MemoryPattern[((address >> 2u) + i) & 3u];
	}
	return true;
}

void FillUserData(HW::UserSgprInfo& user_data) {
	for (uint32_t i = 0; i < HW::UserSgprInfo::SGPRS_MAX; i++) {
		user_data.value[i] = MemoryPattern[i & 3u];
	}
}

// "valid", or the first validator message (SPIRV-Tools, Vulkan 1.3, as the emulator validates).
std::string ValidateSpirv(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          first;
	tools.SetMessageConsumer(
	    [&first](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		    if (first.empty()) {
			    first = message;
		    }
	    });
	if (tools.Validate(spirv)) {
		return "valid";
	}
	std::replace(first.begin(), first.end(), '\t', ' ');
	std::replace(first.begin(), first.end(), '\n', ' ');
	return "invalid: " + first.substr(0, 300);
}

// With KYTY_BATCH_DUMP_DIR set, a shader whose SPIR-V fails validation leaves
// <folder>/invalid/<hash>.spv, its disassembly (.spvasm) and its IR (.ir) behind. With
// KYTY_BATCH_DUMP_ALL set as well, every compiled shader leaves them in <folder>/all, with its
// decoded code (.asm).
void DumpSpirv(const char* subfolder, const std::string& name, const std::vector<uint32_t>& spirv,
               const std::string& ir_dump, const std::string& decoded_dump = {}) {
	const char* folder = std::getenv("KYTY_BATCH_DUMP_DIR");
	if (folder == nullptr) {
		return;
	}
	const auto base = std::filesystem::path(folder) / subfolder;
	std::filesystem::create_directories(base);
	std::ofstream(base / (name + ".spv"), std::ios::binary)
	    .write(reinterpret_cast<const char*>(spirv.data()),
	           static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
	// KYTY_BATCH_DUMP_SPV_ONLY: the module alone, for runs over thousands of shaders.
	if (std::getenv("KYTY_BATCH_DUMP_SPV_ONLY") != nullptr) {
		return;
	}
	std::string source;
	spvtools::SpirvTools(SPV_ENV_VULKAN_1_3)
	    .Disassemble(spirv, &source, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
	std::ofstream(base / (name + ".spvasm")) << source;
	if (!ir_dump.empty()) {
		std::ofstream(base / (name + ".ir")) << ir_dump;
	}
	if (!decoded_dump.empty()) {
		std::ofstream(base / (name + ".asm")) << decoded_dump;
	}
}

void Phase(const char* name) {
	std::printf("PHASE\t%s\n", name);
	std::fflush(stdout);
}

double MillisecondsSince(std::chrono::steady_clock::time_point begin) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
	    .count();
}

void AppendNote(std::string& note, const std::string& text) {
	if (!text.empty()) {
		note += (note.empty() ? "" : "; ") + text;
	}
}

// Every distinct instruction the decoder marks unsupported, e.g. "unsupported family=SOP1
// opcode=0x21", joined with "; ".
std::string UnsupportedInstructions(std::span<const uint32_t> code) {
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
	std::vector<std::string> found;
	for (const auto& inst: decoded.instructions) {
		if (inst.opcode != ShaderRecompiler::Decoder::Opcode::UNSUPPORTED) {
			continue;
		}
		auto text = ShaderRecompiler::Decoder::InstructionToString(inst);
		if (const auto colon = text.find(": "); colon != std::string::npos) {
			text.erase(0, colon + 2);
		}
		while (!text.empty() && text.back() == ' ') {
			text.pop_back();
		}
		if (std::find(found.begin(), found.end(), text) == found.end()) {
			found.push_back(std::move(text));
		}
	}
	std::string joined;
	for (const auto& text: found) {
		AppendNote(joined, text);
	}
	return joined;
}

void RunShader(const std::filesystem::path& folder, const ManifestRow& row) {
	const auto name   = fmt::format("{:016x}", row.hash);
	auto*      header = CopyAligned(ReadFile(folder / (name + ".header")), 16);
	auto*      code   = CopyAligned(ReadFile(folder / (name + ".code")), 256);

	Shader* shader = nullptr;
	if (Gen5::AgcCreateShader(&shader, header, code) != 0 || shader == nullptr) {
		std::printf("RESULT\t%s\t%s\t-\tfailed\t0\t0\t0\t0\tAgcCreateShader failed\t\n",
		            name.c_str(), row.type.c_str());
		return;
	}

	GuestState state;
	ApplyHeaderRegisters(state, *shader);
	FillUserData(state.compute.cs_user_sgpr);
	FillUserData(state.pixel.ps_user_sgpr);
	FillUserData(state.vertex.gs_user_sgpr);
	std::string note;
	AppendNote(note, state.unhandled.empty() ? "" : "unhandled registers " + state.unhandled);

	ShaderComputeInputInfo compute_info;
	ShaderPixelInputInfo   pixel_info;
	ShaderVertexInputInfo  vertex_info;
	VertexTables           vertex_tables;
	ShaderStageInputInfo   stage_input {};
	ShaderParams           params;
	ShaderType             stage          = ShaderType::Unknown;
	uint32_t               wave_size      = 64;
	uint32_t               user_data_base = 0;

	const auto prepare_begin = std::chrono::steady_clock::now();
	switch (static_cast<Prospero::ShaderBinaryType>(shader->type)) {
		case Prospero::ShaderBinaryType::kCs:
			state.compute.cs_regs.wave_size       = Pm4::ComputeWaveSize(state.dispatch_modifier);
			state.compute.cs_user_sgpr.count      = state.compute.cs_regs.user_sgpr;
			compute_info.host_subgroup_size       = 32;
			params = PrepareProgram(state.compute, state.context.GetShaderRegisters(), compute_info);
			stage                 = ShaderType::Compute;
			wave_size             = compute_info.wave_size;
			stage_input.compute   = &compute_info;
			break;
		case Prospero::ShaderBinaryType::kPs: {
			std::array<ShaderRegister, 32> mapping {};
			bool                           mapped = false;
			if (g_pairing && shader->num_input_semantics != 0 && shader->input_semantics != nullptr) {
				for (const auto& layout: g_vs_layouts) {
					if (!LayoutFits(*shader, layout)) {
						continue;
					}
					Shader gs {};
					gs.output_semantics     = const_cast<ShaderSemantic*>(layout.data());
					gs.num_output_semantics = static_cast<uint16_t>(layout.size());
					mapped = Gen5::AgcCreateInterpolantMapping(mapping.data(), &gs, shader) == 0;
					break;
				}
				AppendNote(note, mapped ? "paired" : "no fitting vertex layout");
			}
			for (uint32_t i = 0; i < 32; i++) {
				state.context.SetPsInputSettings(i, mapped ? mapping[i].value : i);
			}
			const std::array<Prospero::ColorComponentMapping, 8> export_mapping {};
			params = PrepareProgram(state.pixel, state.context.GetShaderRegisters(), export_mapping,
			                        pixel_info);
			stage             = ShaderType::Pixel;
			wave_size         = pixel_info.wave_size;
			stage_input.pixel = &pixel_info;
			break;
		}
		case Prospero::ShaderBinaryType::kGs: {
			// The draw's input primitive, as the header's GE_CNTL groups it: meshlet shaders take
			// one point per subgroup (VERT_GRP_SIZE 1) and expand it.
			const auto vertex_group = state.user_config.GetGeControl().vertex_group_size;
			state.user_config.SetPrimitiveType(vertex_group == 1   ? Prospero::PrimitiveType::kPointList
			                                   : vertex_group == 2 ? Prospero::PrimitiveType::kLineList
			                                                       : Prospero::PrimitiveType::kTriList);
			const bool merged = (state.context.GetShaderStages() & 0x20u) != 0;
			if (!merged) {
				std::string table_note;
				if (!PrepareVertexTables(*shader, state.vertex, vertex_tables, table_note)) {
					std::printf("RESULT\t%s\t%s\tVS\tfailed\t0\t0\t0\t0\t%s\t\n", name.c_str(),
					            row.type.c_str(), table_note.c_str());
					return;
				}
				AppendNote(note, table_note);
			}
			params = PrepareProgram(state.vertex, state.context, state.user_config, vertex_info);
			stage  = vertex_info.logical_stage;
			if (stage == ShaderType::Mesh) {
				auto& mesh              = vertex_info.mesh;
				mesh.host_subgroup_size = 32;
				wave_size               = mesh.wave_size;
				// The host limits of the RTX 5090 (VK_EXT_mesh_shader), as GetGraphicsPrograms
				// applies them: a larger subgroup runs in passes, and a draw that still does not fit
				// is skipped.
				mesh.passes = mesh.PassesFor(128u);
				if (mesh.passes > 1u) {
					AppendNote(note, fmt::format("mesh passes={} (wave{} threads={})", mesh.passes,
					                             mesh.wave_size, mesh.HostThreads()));
				}
				if (mesh.passes == 0u || mesh.max_vertices > 256u || mesh.max_primitives > 256u ||
				    mesh.lds_size_dwords * 4u > 28672u) {
					AppendNote(note, fmt::format("mesh exceeds host limits: wave{} threads={} "
					                             "vertices={} primitives={} LDS={}",
					                             mesh.wave_size, mesh.HostThreads(),
					                             mesh.max_vertices, mesh.max_primitives,
					                             mesh.lds_size_dwords));
				}
			} else {
				user_data_base = 8;
				wave_size      = vertex_info.wave_size;
			}
			stage_input.vertex = &vertex_info;
			break;
		}
		default:
			std::printf(
			    "RESULT\t%s\t%s\t-\tnot standalone\t0\t0\t0\t0\tfetch shader or fused half\t\n",
			    name.c_str(), row.type.c_str());
			return;
	}
	const auto prepare_ms = MillisecondsSince(prepare_begin);
	if (params.hash != row.hash) {
		AppendNote(note, fmt::format("declared hash {:016x}", params.hash));
	}

	const char* stage_name = stage == ShaderType::Compute ? "CS"
	                         : stage == ShaderType::Pixel ? "PS"
	                         : stage == ShaderType::Mesh  ? "MS"
	                                                      : "VS";
	ShaderRecompiler::CompileOptions options;
	options.stage           = stage;
	options.shader_hash     = params.hash;
	options.user_data       = std::span<const uint32_t>(params.user_data).first(params.user_data_count);
	options.back_code       = params.back_code;
	// KYTY_BATCH_DUMP_IR: keep each shader's final IR, for the .ir of an invalid-SPIR-V dump.
	options.dump_ir         = std::getenv("KYTY_BATCH_DUMP_IR") != nullptr ||
	                  std::getenv("KYTY_BATCH_DUMP_ALL") != nullptr;
	options.early_dump      = false;
	options.dump_label      = "ShaderBatch";
	options.input_info      = stage_input;
	options.user_data_base  = user_data_base;
	options.wave_size       = wave_size;
	options.non_fatal       = true;
	options.bindless_images = true;

	uint64_t ledger_source_id = 0;
	if (g_ledger != nullptr) {
		const ShaderLedger::SourceFields fields {.stage          = static_cast<uint32_t>(stage),
		                                         .shader_hash    = params.hash,
		                                         .wave_size      = wave_size,
		                                         .user_data_base = user_data_base,
		                                         .bindless       = options.bindless_images,
		                                         .non_fatal      = options.non_fatal,
		                                         .user_data      = options.user_data,
		                                         .code           = params.code,
		                                         .back_code      = params.back_code};
		const uint64_t id = stage == ShaderType::Compute ? g_ledger->RecordSource(fields, compute_info)
		                    : stage == ShaderType::Pixel ? g_ledger->RecordSource(fields, pixel_info)
		                                                 : g_ledger->RecordSource(fields, vertex_info);
		AppendNote(note, fmt::format("source {:016x}", id));
		ledger_source_id = id;
		if (g_pairing && stage == ShaderType::Pixel && pixel_info.ps_pixel_kill_enable) {
			auto no_kill                 = pixel_info;
			no_kill.ps_pixel_kill_enable = false;
			no_kill.ps_early_z = !no_kill.ps_depth_export_enable && !no_kill.ps_sample_mask_export_enable;
			AppendNote(note, fmt::format("no-kill source {:016x}", g_ledger->RecordSource(fields, no_kill)));
		}
	}
	const auto unsupported     = UnsupportedInstructions(params.code);
	const auto translate_begin = std::chrono::steady_clock::now();
	auto       translated      = ShaderRecompiler::TranslateProgram(params.code, options);
	const auto translate_ms    = MillisecondsSince(translate_begin);
	// Upstream executes BVH dispatches now (337f4320): no "ray tracing" outcome any more.
	const char* outcome        = translated.unsupported ? "gave up" : "ok";
	const bool   ok         = !translated.unsupported;
	const bool   dispatcher = ok && translated.program.dispatcher_fallback;
	const size_t blocks     = ok ? translated.program.blocks.size() : 0;
	double       plan_ms    = 0;
	double       emit_ms    = 0;
	std::string  spirv      = "-";
	size_t       spirv_words = 0;
	if (ok) {
		// The rest of the in-game path, with every descriptor unbound: materialize the resource
		// plan, emit SPIR-V and validate it.
		const auto plan_begin = std::chrono::steady_clock::now();
		const auto plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		plan_ms               = MillisecondsSince(plan_begin);
		if (g_resources) {
			const auto source_signature = [&](uint32_t source) {
				if (source >= plan.descriptor_sources.size() || plan.descriptor_sources[source].dword_count == 0) {
					return std::string("-");
				}
				const auto& descriptor = plan.descriptor_sources[source];
				return ValueSignature(descriptor.dwords[0], 6) +
				       (descriptor.indirect_descriptor.has_value() ? " indirect" : "");
			};
			for (size_t i = 0; i < plan.info.buffers.size(); i++) {
				const auto& b = plan.info.buffers[i];
				std::printf("RES\t%s\t%016" PRIx64 "\tB\t%zu\tr%dw%da%df%ds%d alias%d ext%u\t%s\n", name.c_str(),
				            ledger_source_id, i, b.read, b.written, b.atomic, b.formatted, b.scalar,
				            b.image_alias != ShaderRecompiler::IR::BufferResource::NoImageAlias,
				            b.max_byte_extent, source_signature(b.source).c_str());
			}
			for (size_t i = 0; i < plan.info.images.size(); i++) {
				const auto& m = plan.info.images[i];
				std::printf("RES\t%s\t%016" PRIx64 "\tI\t%zu\tc%ud%um%ur%dw%da%dz%dk%dq%du%db%d\t%s\n", name.c_str(),
				            ledger_source_id, i, static_cast<uint32_t>(m.resource_class),
				            static_cast<uint32_t>(m.dimension), static_cast<uint32_t>(m.mip_mode), m.read,
				            m.written, m.atomic, m.depth_compare, m.cube, m.r128, m.unfiltered, m.bindless,
				            source_signature(m.source).c_str());
			}
		}
		// The plan fingerprint the emulator records permutations under (the source entry's payload).
		uint64_t plan_fingerprint = 0;
		if (g_ledger != nullptr) {
			bool uses_clock = false;
			for (auto* block: translated.program.blocks) {
				for (const auto& inst: *block) {
					uses_clock = uses_clock ||
					             inst.GetOpcode() == ShaderRecompiler::IR::ValueOpcode::ReadClockRealtime64;
				}
			}
			std::vector<uint8_t> record;
			if (ShaderDiskCache::EncodeSourceRecord(false, uses_clock, &plan, record)) {
				plan_fingerprint = XXH3_64bits(record.data(), record.size());
			}
		}
		Phase("materialize");
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		          .user_data                  = options.user_data,
		          .shader_base                = reinterpret_cast<uint64_t>(params.code.data()),
		          .read_memory                = ReadPatternMemory,
		          .userdata                   = nullptr,
		          .read_specialization_memory = ReadPatternMemory,
        };
		if (!ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization)) {
			spirv = fmt::format("materialization failed (line {})",
			                    ShaderRecompiler::IR::LastIndirectImageFailureLine());
		} else {
			if (g_ledger != nullptr && ledger_source_id != 0) {
				// The specialization of this tool's stand-in draw (unbound descriptors), for comparison
				// with the game's recorded permutations.
				g_ledger->RecordPermutation(ledger_source_id, 0, specialization, plan_fingerprint);
			}
			Phase("emit");
			const auto emit_begin = std::chrono::steady_clock::now();
			const auto push_data  = stage == ShaderType::Mesh
			                            ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount
			                            : 0u;
			auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options,
			                                                 specialization, push_data);
			Phase("validate");
			spirv       = ValidateSpirv(compiled.spirv);
			spirv_words = compiled.spirv.size();
			emit_ms     = MillisecondsSince(emit_begin);
			if (spirv != "valid") {
				DumpSpirv("invalid", name, compiled.spirv, compiled.ir_dump);
			}
			if (std::getenv("KYTY_BATCH_DUMP_ALL") != nullptr) {
				DumpSpirv("all", name, compiled.spirv, compiled.ir_dump, compiled.decoded_dump);
			}
		}
	}
	std::printf("RESULT\t%s\t%s\t%s\t%s\t%.2f\t%.2f\t%d\t%zu\t%s\t%s\t%s\t%zu\t%.2f\n",
	            name.c_str(), row.type.c_str(), stage_name, outcome, prepare_ms,
	            translate_ms + plan_ms, dispatcher ? 1 : 0, blocks, note.c_str(),
	            unsupported.c_str(), spirv.c_str(), spirv_words, emit_ms);
}

} // namespace
} // namespace Libs::Graphics

int main(int argc, char* argv[]) {
	using namespace Libs::Graphics;
	if (argc < 3) {
		std::fprintf(stderr, "usage: shader_batch_tool <shader folder> <log file> [first row]\n");
		return 2;
	}
	const std::filesystem::path folder = argv[1];
	const size_t                first  = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;
	const size_t                end    = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : SIZE_MAX;

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction   = Config::LogDirection::File;
	config.printf_output_file = argv[2];
	// KYTY_BATCH_DUMP_DIR=<folder>: the IR of every shader that gives up in resource tracking goes
	// to <folder>/rejected/<stage>_<hash>.ir, as with --graphics-debug-dump in the emulator.
	if (const char* dump = std::getenv("KYTY_BATCH_DUMP_DIR"); dump != nullptr) {
		config.graphics_debug_dump_enabled = true;
		config.shader_log_folder           = dump;
	}
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();
#ifdef _WIN32
	AddVectoredExceptionHandler(1, ReportCrash);
#endif

	if (const char* ledger = std::getenv("KYTY_BATCH_LEDGER"); ledger != nullptr && ledger[0] != '\0') {
		g_ledger = ShaderLedger::Ledger::Open(ledger);
	}
	// The emulator's device setup turns on bindless storage images on every GPU this build targets
	// (bindlessTable.cpp); without it the batch tool planned store-only GPU-selected images
	// differently, so their permutations carried another plan fingerprint (fr3 review, 11-14).
	// KYTY_BATCH_BINDLESS_STORAGE=0 plans them as before.
	if (const char* storage = std::getenv("KYTY_BATCH_BINDLESS_STORAGE"); storage == nullptr || storage[0] != '0') {
		ShaderRecompiler::IR::g_bindless_storage_supported.store(true, std::memory_order_relaxed);
	}
	const auto rows = ReadManifest(folder);
	if (const char* resources = std::getenv("KYTY_BATCH_RESOURCES"); resources != nullptr && resources[0] == '1') {
		g_resources = true;
	}
	if (const char* pairing = std::getenv("KYTY_BATCH_PAIRING"); pairing != nullptr && pairing[0] == '1') {
		g_pairing = true;
		for (const auto& row: rows) {
			const auto name   = fmt::format("{:016x}", row.hash);
			auto*      header = CopyAligned(ReadFile(folder / (name + ".header")), 16);
			auto*      code   = CopyAligned(ReadFile(folder / (name + ".code")), 256);
			Shader*    shader = nullptr;
			if (Gen5::AgcCreateShader(&shader, header, code) != 0 || shader == nullptr ||
			    shader->num_output_semantics == 0 || shader->output_semantics == nullptr) {
				continue;
			}
			std::vector<ShaderSemantic> layout(shader->output_semantics,
			                                   shader->output_semantics + shader->num_output_semantics);
			if (std::none_of(g_vs_layouts.begin(), g_vs_layouts.end(), [&](const auto& known) {
				    return known.size() == layout.size() &&
				           std::memcmp(known.data(), layout.data(), layout.size() * sizeof(ShaderSemantic)) == 0;
			    })) {
				g_vs_layouts.push_back(std::move(layout));
			}
		}
		std::printf("PAIRING\t%zu vertex output layouts\n", g_vs_layouts.size());
	}
	for (size_t i = first; i < std::min(end, rows.size()); i++) {
		std::printf("START\t%zu\t%016" PRIx64 "\n", i, rows[i].hash);
		std::fflush(stdout);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
		// A fatal error of the recompiler (EXIT) ends only this shader: one RESULT line with the
		// message, then the next row, instead of exit 321 and a restart by the driver.
		try {
			Common::RecoverableExitScope recoverable;
			RunShader(folder, rows[i]);
		} catch (const Common::RecoverableExit& failure) {
			auto text = failure.text;
			std::replace(text.begin(), text.end(), '\n', ' ');
			std::replace(text.begin(), text.end(), '\t', ' ');
			std::printf("RESULT\t%016" PRIx64 "\t%s\t-\tfatal\t0\t0\t0\t0\t%s\t\n", rows[i].hash,
			            rows[i].type.c_str(), text.c_str());
		}
#else
		RunShader(folder, rows[i]);
#endif
		std::fflush(stdout);
	}
	std::printf("DONE\t%zu\n", rows.size());
	return 0;
}
