#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/RtCallable.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	bool                        non_fatal = false;
	bool                        bindless_images = false;
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
	// v440: set for the two halves of a ray-tracing shader split at its call (IR/RtCallable.h).
	const IR::RtCallInfo*       rt_call = nullptr;
	// v440 M3: a whole callable split (ray generation + callables, merged; IR/RtCallable.h). The
	// code TranslateProgram is given is then the split's main code.
	const IR::RtSplitInput*     rt_split = nullptr;
};

struct TranslateResult {
	bool        unsupported = false;
	// The refusal depends on this variant's input (a size bound): the shader's other variants may
	// translate, so it is no reason for a shader-wide refusal.
	bool        input_dependent_refusal = false;
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	// A merged callable split: spirv is the ray-generation module, these the callables (callable
	// index i: the split's target i).
	std::vector<std::vector<uint32_t>> rt_callable_spirv;
	// v443 (shared callables, IR/RtCallable.h): each callable's shader record words (empty: none),
	// which the host writes after its group handle in the shader binding table.
	std::vector<std::vector<uint32_t>> rt_callable_records;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
