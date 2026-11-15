#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINERECIPE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINERECIPE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_set>
#include <vector>

// Graphics pipeline recipes (plan 4.14, Mac 10-26: "can the graphics pipelines be added to the
// shaders being compiled at start?"). A recipe is everything vkCreateGraphicsPipelines reads for
// one pipeline, serialized with each stage's module named by the 128-bit hash of its SPIR-V, plus
// the descriptor set layout the pipeline layout is made of. Every pipeline the emulator builds is
// recorded once in _PipelineCache/<title>.recipes; at startup, after the shader ledger's replay
// has created every known module, the recipes are built again on worker threads, and a draw whose
// prepared pipeline serializes to a prebuilt recipe takes that pipeline instead of building it.
// A recipe whose module is not in this session (a permutation the replay skipped) is skipped.
namespace Libs::Graphics {

struct GraphicContext;
struct GraphicsPipelineBuild;

namespace PipelineRecipe {

struct Hash128 {
	uint64_t low  = 0;
	uint64_t high = 0;
	bool     operator==(const Hash128&) const = default;
};

struct Hash128Hasher {
	size_t operator()(const Hash128& hash) const noexcept { return static_cast<size_t>(hash.low ^ (hash.high * 0x9e3779b97f4a7c15ull)); }
};

[[nodiscard]] Hash128 HashWords(std::span<const uint32_t> words);
[[nodiscard]] Hash128 HashBytes(std::span<const uint8_t> bytes);

// Every shader module made from the program cache's SPIR-V (ProgramCache::AddPermutation).
// With `keep`, the SPIR-V is also written to the SPIR-V store (once per hash, on a background
// thread): the startup replay of the shader ledger only checks the disk cache and creates no
// modules, so the recipes are built from the store's copies.
void RegisterModule(vk::ShaderModule module, std::span<const uint32_t> spirv, bool keep = false);
// The SPIR-V store's directory (set once at startup, before any module is registered).
void SetSpirvStore(const std::filesystem::path& directory);
[[nodiscard]] bool             FindModuleHash(vk::ShaderModule module, Hash128& hash);

// v391 (Mac 11-11, "finish 391"): module identities. A recipe names each stage by the hash of its
// SPIR-V, so a recompiler change (v389's probe) left every recipe naming SPIR-V this build no
// longer makes: the warm-up built pipelines no draw asks for, and each draw's pipeline was
// compiled cold during play (1-2 s hangs). Each module is now also recorded with a stable id of
// the permutation it was made from (the ledger's source id, push data cursor, plan fingerprint
// and specialization: the same for every build) and the recompiler session it came from, in
// <title>.modules beside the recipes. At startup, after the replay made this session's modules,
// a recipe stage made by another session is renamed to this session's module of the same
// permutation (when both declare the same interface), and the renamed recipe is built and kept.
[[nodiscard]] Hash128 StableId(uint64_t source_id, uint32_t push_data_cursor, uint64_t plan_fingerprint,
                               std::span<const uint8_t> specialization);
// Opens the module map (before any NoteModule). `session_tag`: xxh3 of the shader disk cache's
// session key (it changes exactly when the recompiler or its switches change).
void OpenModuleMap(const std::filesystem::path& path, uint64_t session_tag);
// This session made `spirv` for the permutation `stable`. With `store`, the SPIR-V also goes to
// the SPIR-V store (the replay makes no modules; the warm-up builds from the store's copies).
void NoteModule(const Hash128& stable, std::span<const uint32_t> spirv, bool store);
// Returns once every SPIR-V queued for the store is on disk (the warm-up reads the store's copies).
void WaitSpirvStore();
// Whether this session's module of `stable` is known (the replay reads a permutation it finds in
// the disk cache only when it is not: once per session key).
// v421 (ledger carry-over, Fable step 2): `old_stable` (a permutation recorded under another
// resource plan) names this session's module of `stable` (the same source, push cursor and
// specialization under this build's plan), so pipeline recipes made from the old module are
// renamed to it. False when `stable` has no module in this session or nothing changed.
[[nodiscard]] bool AliasModule(const Hash128& old_stable, const Hash128& stable);
[[nodiscard]] bool HasCurrentModule(const Hash128& stable);
struct RemapResult {
	bool     changed      = false; // a stage was renamed: `recipe` holds the renamed bytes
	bool     interface_differs = false; // a stage of the same permutation declares another interface
	uint32_t unknown      = 0;     // stages whose permutation is not in the map (recorded before v391)
};
// Renames the recipe's stages made by another session to this session's modules.
[[nodiscard]] RemapResult Remap(std::vector<uint8_t>& recipe);
[[nodiscard]] vk::ShaderModule FindModule(const Hash128& hash);
// v418 (Mac 11-13: the counts included many obsolete pipelines): whether every stage of the recipe
// is a module of this recompiler session (made by the ledger replay or by a draw). A recipe with
// a stage of another session would build, from the SPIR-V store's old copy, a pipeline no draw of
// this build asks for: the warm-up leaves it out (and out of its count). It stays in the file.
[[nodiscard]] bool StagesCurrent(std::span<const uint8_t> recipe);

// The recipe of a prepared pipeline (the build records its descriptor bindings and, for rect
// lists, the generated tessellation SPIR-V). False when a stage's module is not registered.
[[nodiscard]] bool Serialize(const GraphicsPipelineBuild& build, std::vector<uint8_t>& out);

// Builds the pipeline a recipe describes: modules by hash (registered), the rect-list stages from
// the recipe's SPIR-V, the set and pipeline layouts made and destroyed around the call. The
// rebuilt create info must serialize back to the same bytes, or nothing is built (a format
// mismatch is skipped, never misbuilt). eErrorUnknown when it cannot be rebuilt.
vk::Result Build(GraphicContext& graphics, std::span<const uint8_t> recipe, vk::PipelineCache cache,
                 vk::Pipeline* pipeline);

// The recipe file: a magic and version, then records (u32 size, 16-byte hash, bytes).
class Store {
public:
	// Reads every record (a truncated tail is ignored) and opens the file for appending.
	void Open(const std::filesystem::path& path);
	// Appends the recipe unless it is known; true when it was new.
	// `counted`: the overlay counts it as built this session (not a renamed recipe, v391).
	bool Record(const Hash128& hash, std::span<const uint8_t> recipe, bool counted = true);
	[[nodiscard]] bool Known(const Hash128& hash);

	struct Entry {
		Hash128              hash;
		std::vector<uint8_t> bytes;
	};
	// The records read at Open (moved out once).
	[[nodiscard]] std::vector<Entry> TakeLoaded();
	[[nodiscard]] size_t             Size();

	~Store();

private:
	std::mutex                                 m_mutex;
	std::unordered_set<Hash128, Hash128Hasher> m_known;
	std::vector<Entry>                         m_loaded;
	std::FILE*                                 m_file = nullptr;
	std::filesystem::path                      m_path;
};

// v394 (Mac 11-11: "why am I getting 70+ new pipelines in an area I've played more than a dozen
// times"): recipes recorded before v391 name SPIR-V of older sessions that the map never saw, so
// they could not be renamed (74 of 76 new recipes in mac_v393_1 had an old recipe's exact state
// with this session's module of a permutation the game had used for weeks). Their modules are
// found once in the disk cache's entries of other sessions (ShaderLedger::ImportModuleMap).
// The recipe stages the map does not know (after OpenModuleMap; `loaded` are the store's records),
// with their SPIR-V sizes from the SPIR-V store.
void CollectUnknownModules(std::span<const Store::Entry> loaded);
// Whether the one-time import is still wanted (not done before, and modules are unknown).
[[nodiscard]] bool ModuleImportWanted();
[[nodiscard]] bool IsUnknownModuleSize(uint64_t words);
[[nodiscard]] bool IsUnknownModule(const Hash128& module);
[[nodiscard]] size_t UnknownModuleCount();
// Records a module of another session (`session_tag`) found in the disk cache.
void NoteModuleFrom(uint64_t session_tag, const Hash128& stable, std::span<const uint32_t> spirv);
void MarkModuleImportDone();

} // namespace PipelineRecipe
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINERECIPE_H_
