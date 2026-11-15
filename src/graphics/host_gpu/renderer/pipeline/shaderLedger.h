#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLEDGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLEDGER_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <vector>

// The shader ledger (plan 4.12): every shader program this machine has translated, kept as the
// inputs of the translation (a recipe), so that a new build can translate them all again at
// startup (ProgramCache::StartPrecompile) instead of one by one in the middle of play.
//
// <disk cache directory>/ledger.bin, append-only records:
//   [u32 'KLDG'][u16 version][u8 kind][u8 0][u32 payload size][u64 xxh3(payload)][payload]
// kind 'S': a source recipe: [u64 id = xxh3 of the rest][the source part of the disk-cache key,
// ShaderLedger::AppendSourceKeyTail: options, code, the stage input fields translation reads]
// (canonical bytes: the same recipe always gets the same id, a raw struct would carry padding);
// kind 'P': a permutation recipe (its source, the push-data start, the fingerprint of the
// resource plan it was specialized against, the resource specialization); kind 'I': the disk cache's keys were imported (plan 4.13; its payload is a summary, not
// replayed). Like the key, a source holds only the user-data count: translation reads nothing
// else of it (the values feed the function expander, whose output is the recorded code). Unlike
// the disk cache's entries it does not depend on the build: a new build reads it as it is. A replayed
// recipe is keyed again by the build that replays it (BuildSourceKey), so a record that decodes
// to something else than what was recorded can only produce an entry nobody asks for.
// KYTY_SHADER_LEDGER=0 turns recording off. The file holds guest shader code: it stays local.
namespace Libs::Graphics::ShaderLedger {

enum class InputKind : uint8_t { Vertex = 'V', Pixel = 'P', Compute = 'C' };

struct SourceFields {
	uint32_t                  stage          = 0;
	uint64_t                  shader_hash    = 0;
	uint32_t                  wave_size      = 64;
	uint32_t                  user_data_base = 0;
	bool                      bindless       = false;
	bool                      non_fatal      = false;
	std::span<const uint32_t> user_data;
	std::span<const uint32_t> code;
	std::span<const uint32_t> back_code;
};

struct SourceRecipe {
	uint64_t              id             = 0;
	uint32_t              stage          = 0;
	uint64_t              shader_hash    = 0;
	uint32_t              wave_size      = 64;
	uint32_t              user_data_base = 0;
	bool                  bindless       = false;
	bool                  non_fatal      = false;
	std::vector<uint32_t> user_data;
	std::vector<uint32_t> code;
	std::vector<uint32_t> back_code;
	InputKind             kind = InputKind::Compute;
	ShaderVertexInputInfo  vertex {};
	ShaderPixelInputInfo   pixel {};
	ShaderComputeInputInfo compute {};

	[[nodiscard]] SourceFields Fields() const {
		return {.stage          = stage,
		        .shader_hash    = shader_hash,
		        .wave_size      = wave_size,
		        .user_data_base = user_data_base,
		        .bindless       = bindless,
		        .non_fatal      = non_fatal,
		        .user_data      = user_data,
		        .code           = code,
		        .back_code      = back_code};
	}
};

struct PermutationRecipe {
	uint64_t             source_id        = 0;
	uint32_t             push_data_cursor = 0;
	// xxh3 of the source entry's payload (its resource plan) the specialization was computed
	// against: the replay uses the recipe only when this build's plan is the same, because a
	// specialization only fits the plan it came from (ApplyResourceSpecialization).
	uint64_t             plan_fingerprint = 0;
	std::vector<uint8_t> specialization; // ShaderDiskCache::AppendSpecialization bytes
};


class Ledger {
public:
	// Null when recording is off (KYTY_SHADER_LEDGER=0) or the directory is empty.
	[[nodiscard]] static std::unique_ptr<Ledger> Open(const std::filesystem::path& directory);

	explicit Ledger(std::filesystem::path path);
	~Ledger();
	Ledger(const Ledger&)            = delete;
	Ledger& operator=(const Ledger&) = delete;

	// Records the source if it is new; returns its id either way.
	uint64_t RecordSource(const SourceFields& fields, const ShaderVertexInputInfo& info);
	uint64_t RecordSource(const SourceFields& fields, const ShaderPixelInputInfo& info);
	uint64_t RecordSource(const SourceFields& fields, const ShaderComputeInputInfo& info);
	uint64_t RecordSource(const SourceRecipe& recipe, bool* added = nullptr);
	// True when the permutation was new. `plan_fingerprint`: see PermutationRecipe (0: not recorded).
	bool     RecordPermutation(uint64_t source_id, uint32_t push_data_cursor,
	                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                           uint64_t plan_fingerprint);

	// The recipes read at Open or at the last Reload (moved out: the second call gets nothing).
	void TakeRecipes(std::vector<SourceRecipe>& sources, std::vector<PermutationRecipe>& permutations);
	// Reads the file again (after ImportDiskCache appended to it).
	void Reload();

	// Whether the file holds an import marker: the disk cache's keys were read into it once
	// (ShaderLedger::ImportDiskCache, plan 4.13).
	[[nodiscard]] bool Imported() const { return m_imported; }
	void               MarkImported(std::span<const uint8_t> summary);

	[[nodiscard]] bool     WriteFailed() {
		std::lock_guard lock(m_mutex);
		return m_write_failed;
	}
	// After a complete replay: how many of the recipes read at startup this build cannot use
	// (permutations of another resource plan, of a refused or missing source, refused sources).
	// They stay in the file (another build may use them) but no longer count in "Shaders N".
	void SetUnusable(uint64_t count);

	[[nodiscard]] uint64_t SourceCount() const { return m_source_count; }
	[[nodiscard]] uint64_t PermutationCount() const { return m_permutation_count; }

private:
	void     Load();
	void     LoadFile(const std::filesystem::path& path, bool own);
	uint64_t RecordSourceTail(std::span<const uint8_t> tail, bool* added = nullptr);
	void     Append(uint8_t kind, std::span<const uint8_t> payload);

	std::filesystem::path          m_path;
	std::mutex                     m_mutex;
	std::FILE*                     m_file = nullptr;
	bool                           m_write_failed = false;
	std::unordered_set<uint64_t>   m_ids; // source ids and permutation ids already in the file
	std::vector<SourceRecipe>      m_sources;
	std::vector<PermutationRecipe> m_permutations;
	uint64_t                       m_source_count      = 0;
	uint64_t                       m_permutation_count = 0;
	uint64_t                       m_unusable          = 0;
	bool                           m_imported          = false;
};

} // namespace Libs::Graphics::ShaderLedger

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLEDGER_H_
