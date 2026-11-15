#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLEDGERIMPORT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLEDGERIMPORT_H_

#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderLedger.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

// Seeding the shader ledger from the disk cache's keys (plan 4.13). Every disk-cache entry stores
// its whole key, and a key holds everything its translation reads: the session key, then the
// source fields below ('S'), and for a permutation the resource specialization and the push-data
// start ('P'). So the entries every earlier build wrote hold the recipes of every program those
// builds met. The import parses each key, re-encodes what it parsed with this build's writers and
// keeps it only when the bytes are the same (a key of another format, or a parser slip, is
// skipped, never misread), then records it in the ledger, whose replay compiles it for this build.
//
// This file and its .cpp are outside the recompiler's version hash (CMakeLists.txt,
// shader_cache_seeds) on purpose: changing the importer must not invalidate cache entries.
namespace Libs::Graphics::ShaderLedger {

// The source part of a disk-cache key after the session key (ProgramCache::BuildSourceKey writes
// it through this, the import re-encodes with it).
template <typename InputInfo>
void AppendSourceKeyTail(ShaderDiskCache::ByteWriter& writer, const SourceFields& fields,
                         const InputInfo& input_info) {
	writer.Put<uint8_t>('S');
	writer.Put(fields.stage);
	writer.Put(fields.shader_hash);
	writer.Put(fields.wave_size);
	writer.Put(fields.user_data_base);
	writer.Put(static_cast<uint32_t>(fields.user_data.size()));
	writer.Put(fields.bindless);
	writer.Put(fields.non_fatal);
	writer.PutWords(fields.code);
	writer.PutWords(fields.back_code);
	ShaderDiskCache::AppendInputKey(writer, input_info);
}

// The inverse of AppendSourceKeyTail: `bytes` starts with the 'S'. True when the fields parse and
// re-encode to the same bytes; `consumed` is their length. The user data comes back as zeros of
// the recorded count. `scratch` is working space.
bool ParseSourceTail(std::span<const uint8_t> bytes, SourceRecipe& recipe, size_t& consumed,
                     ShaderDiskCache::ByteWriter& scratch);

struct ImportStats {
	uint64_t files            = 0; // entry files found
	uint64_t unreadable       = 0; // could not be opened, or no entry header
	uint64_t rejected         = 0; // a key that does not parse and re-encode to itself:
	uint64_t rejected_session = 0; //   no session switch list found
	uint64_t rejected_source  = 0; //   the source part does not
	uint64_t old_permutations = 0; // permutation keys whose suffix does not (source recorded)
	uint64_t no_source_entry  = 0; // permutation keys whose source entry is missing (not recorded)
	uint64_t sources          = 0; // keys of source entries
	uint64_t permutations     = 0; // keys of permutation entries
	uint64_t new_sources      = 0; // source recipes the ledger did not have
	uint64_t new_permutations = 0; // permutation recipes the ledger did not have
	uint64_t moved            = 0; // entries of other sessions moved to `prune_to`
	double   seconds          = 0.0;
	bool     stopped          = false;

	[[nodiscard]] std::string Summary() const;
};

// Reads the key of every entry in `directory` on `threads` threads (below-normal priority) and
// records the recipes in `ledger`. `files_done` / `files_total` follow the progress; `stop` ends
// it early. With `prune_to` set (KYTY_SHADER_CACHE_PRUNE=1), every entry whose session prefix is
// not `session_key` is moved there once its recipe is recorded, and so is every key this build
// cannot parse (another format). Nothing is deleted. When recording, sibling folders named
// "<directory>_..." (where clean-ups moved entries) are read too, never changed. A permutation is
// recorded with the fingerprint of its source entry's payload (PermutationRecipe::plan_fingerprint),
// found next to it; without that entry it is not recorded. With `record` false (the ledger has been
// seeded already, and Get records every program it builds since), keys are not parsed: only
// their session prefix is read, for the move.
ImportStats ImportDiskCache(const std::filesystem::path& directory, Ledger& ledger, uint32_t threads,
                            const std::atomic<bool>& stop, std::atomic<uint32_t>& files_done,
                            std::atomic<uint32_t>& files_total, std::span<const uint8_t> session_key,
                            const std::filesystem::path& prune_to, bool record);

// v394: the one-time search of the disk cache (and its "_..." siblings) for the modules of earlier
// sessions that pipeline recipes name but the module map lacks (PipelineRecipe::CollectUnknownModules).
// Per entry it reads the header and the payload's SPIR-V word count; only an entry whose count is
// one of the unknown modules' has its SPIR-V hashed, and only a match has its key parsed for the
// permutation's stable id (PipelineRecipe::NoteModuleFrom). Marks the import done unless stopped.
// Returns the modules found.
uint64_t ImportModuleMap(const std::filesystem::path& directory, uint32_t threads, const std::atomic<bool>& stop,
                         std::atomic<uint32_t>& files_done, std::atomic<uint32_t>& files_total,
                         std::span<const uint8_t> session_key);

} // namespace Libs::Graphics::ShaderLedger

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLEDGERIMPORT_H_
