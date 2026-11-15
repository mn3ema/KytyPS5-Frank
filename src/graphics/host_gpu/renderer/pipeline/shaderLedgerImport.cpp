#include "graphics/host_gpu/renderer/pipeline/shaderLedgerImport.h"

#include "common/cpuAffinity.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineRecipe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics::ShaderLedger {
namespace {

using ShaderDiskCache::ByteReader;
using ShaderDiskCache::ByteWriter;

// The disk cache's entry file: [u32 magic "KSDC"][u32 format][u64 key size][u64 payload size]
// [u64 payload hash][key][payload] (ShaderDiskCache::Store::Save). Only the key is read.
constexpr uint32_t EntryMagic      = 0x4344534bu;
constexpr size_t   EntryHeaderSize = 4 + 4 + 8 + 8 + 8;
constexpr uint64_t MaxKeySize      = 16ull << 20u;
// ShaderParams::user_data (shaderCompiler.h) holds 40 words.
constexpr uint32_t MaxUserData = 40;

// The inverses of ShaderDiskCache::AppendInputKey, field for field (the kind byte is read by the
// caller). Fields the key does not hold keep their defaults: translation does not read them.
bool ReadInputKey(ByteReader& in, ShaderComputeInputInfo& info) {
	for (auto& value: info.threads_num) {
		if (!in.Get(value)) return false;
	}
	for (auto* value: {&info.lds_size_dwords, &info.scratch_size_dwords, &info.host_subgroup_size,
	                   &info.wave_size}) {
		if (!in.Get(*value)) return false;
	}
	if (!in.Get(info.float_mode)) return false;
	for (auto& value: info.group_id) {
		if (!in.Get(value)) return false;
	}
	int32_t thread_ids_num     = 0;
	int32_t workgroup_register = 0;
	if (!in.Get(info.dispatch_thread_dimensions) || !in.Get(info.lds_storage) ||
	    !in.Get(info.dispatch_dimensions_indirect) || !in.Get(thread_ids_num) ||
	    !in.Get(workgroup_register) || !in.Get(info.tg_size_en)) {
		return false;
	}
	info.thread_ids_num     = thread_ids_num;
	info.workgroup_register = workgroup_register;
	return true;
}

bool ReadInputKey(ByteReader& in, ShaderPixelInputInfo& info) {
	if (!in.Get(info.input_num)) return false;
	const auto inputs = std::min<uint32_t>(info.input_num, std::size(info.interpolator_settings));
	for (uint32_t i = 0; i < inputs; i++) {
		if (!in.Get(info.interpolator_settings[i])) return false;
	}
	for (auto* value: {&info.wave_size, &info.ps_system_input_base, &info.custom_interpolation_mask,
	                   &info.ps_perspective_center_vgpr, &info.ps_perspective_centroid_vgpr,
	                   &info.target_uint_mask, &info.target_sint_mask, &info.scratch_size_dwords}) {
		if (!in.Get(*value)) return false;
	}
	for (auto& mode: info.target_output_mode) {
		if (!in.Get(mode)) return false;
	}
	for (auto& mapping: info.target_export_mapping) {
		if (!in.Get(mapping.packed)) return false;
	}
	for (bool* flag: {&info.ps_pos_x, &info.ps_pos_y, &info.ps_pos_z, &info.ps_pos_w,
	                  &info.ps_front_face, &info.ps_ancillary, &info.ps_no_perspective,
	                  &info.ps_pixel_kill_enable, &info.ps_depth_export_enable,
	                  &info.ps_sample_mask_export_enable, &info.ps_sample_shading,
	                  &info.dual_source_blending, &info.ps_early_z, &info.ps_execute_on_noop}) {
		if (!in.Get(*flag)) return false;
	}
	if (info.ps_front_face && !in.Get(info.ps_front_face_all_bits)) return false;
	uint8_t alpha = 0;
	if (!in.Get(alpha)) return false;
	info.alpha_blend_source = static_cast<ShaderAlphaBlendSource>(alpha);
	return true;
}

bool ReadInputKey(ByteReader& in, ShaderVertexInputInfo& info) {
	uint32_t logical_stage    = 0;
	int32_t  resources_num    = 0;
	int32_t  fetch_attrib_reg = 0;
	int32_t  fetch_buffer_reg = 0;
	if (!in.Get(logical_stage) || !in.Get(resources_num) || !in.Get(fetch_attrib_reg) ||
	    !in.Get(fetch_buffer_reg) || !in.Get(info.wave_size) || !in.Get(info.scratch_size_dwords) ||
	    !in.Get(info.pa_cl_vs_out_cntl) || !in.Get(info.clip_space.enabled)) {
		return false;
	}
	info.logical_stage    = static_cast<ShaderType>(logical_stage);
	info.resources_num    = resources_num;
	info.fetch_attrib_reg = fetch_attrib_reg;
	info.fetch_buffer_reg = fetch_buffer_reg;
	if (info.clip_space.enabled) {
		for (int i = 0; i < 2; i++) {
			if (!in.Get(info.clip_space.scale[i]) || !in.Get(info.clip_space.offset[i]) ||
			    !in.Get(info.clip_space.half_extent[i])) {
				return false;
			}
		}
	}
	auto& mesh = info.mesh;
	for (auto& value: mesh.threads_num) {
		if (!in.Get(value)) return false;
	}
	for (auto* value: {&mesh.lds_size_dwords, &mesh.scratch_size_dwords, &mesh.host_subgroup_size,
	                   &mesh.wave_size, &mesh.input_primitive, &mesh.primitives_per_group,
	                   &mesh.vertices_per_group, &mesh.max_vertices, &mesh.max_primitives,
	                   &mesh.provoking_vertex, &mesh.passes}) {
		if (!in.Get(*value)) return false;
	}
	if (!in.Get(mesh.fast_launch)) return false;
	auto& tess = info.tess;
	for (auto* value: {&tess.input_control_points, &tess.output_control_points, &tess.ls_stride,
	                   &tess.hs_stride, &tess.domain, &tess.partitioning, &tess.output_topology}) {
		if (!in.Get(*value)) return false;
	}
	if (!in.Get(info.fetch_external) || !in.Get(info.fetch_embedded)) return false;
	const int resources = std::clamp(info.resources_num, 0, ShaderVertexInputInfo::RES_MAX);
	for (int i = 0; i < resources; i++) {
		uint32_t field3        = 0;
		int32_t  registers_num = 0;
		int32_t  attr_id       = 0;
		if (!in.Get(field3) || !in.Get(registers_num) || !in.Get(attr_id)) return false;
		info.resources[i].fields[3]       = field3;
		info.resources_dst[i].registers_num = registers_num;
		info.resources_dst[i].attr_id       = attr_id;
	}
	return true;
}

struct ParsedKey {
	SourceRecipe recipe;
	size_t       source_size      = 0; // the source key's length (session prefix + source part)
	bool         permutation      = false;
	uint32_t     push_data_cursor = 0;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
};

bool KindMatchesStage(uint8_t kind, uint32_t stage) {
	switch (static_cast<ShaderType>(stage)) {
		case ShaderType::Vertex:
		case ShaderType::Mesh:
		case ShaderType::Local:
		case ShaderType::TessellationControl:
		case ShaderType::TessellationEvaluation: return kind == 'V';
		case ShaderType::Pixel: return kind == 'P';
		case ShaderType::Compute: return kind == 'C';
		default: return false;
	}
}

enum class Parse { Ok, NoSession, BadSource, BadPermutation };

// `tail` starts at the 'S' after the session key. Ok when it parses, re-encodes to the same bytes
// and the rest is nothing (a source key) or a permutation suffix that does too; BadPermutation
// when only the source part does (an older specialization layout: the source is still usable).
Parse ParseTail(std::span<const uint8_t> tail, ParsedKey& out, ByteWriter& scratch) {
	size_t consumed = 0;
	if (!ParseSourceTail(tail, out.recipe, consumed, scratch)) {
		return Parse::BadSource;
	}
	out.source_size = consumed; // ParseKey adds the session prefix
	const auto rest = tail.subspan(consumed);
	out.permutation = !rest.empty();
	if (!out.permutation) {
		return Parse::Ok;
	}
	if (rest.size() < 1 + sizeof(uint32_t) || rest[0] != 'P') {
		return Parse::BadPermutation;
	}
	const auto spec = rest.subspan(1, rest.size() - 1 - sizeof(uint32_t));
	std::memcpy(&out.push_data_cursor, rest.data() + rest.size() - sizeof(uint32_t), sizeof(uint32_t));
	out.specialization = {};
	if (!ShaderDiskCache::DecodeSpecialization(spec, out.specialization)) {
		return Parse::BadPermutation;
	}
	scratch.bytes.clear();
	ShaderDiskCache::AppendSpecialization(scratch, out.specialization);
	return scratch.bytes.size() == spec.size() &&
	               std::memcmp(scratch.bytes.data(), spec.data(), spec.size()) == 0
	           ? Parse::Ok
	           : Parse::BadPermutation;
}

bool IsSwitchEntry(std::span<const uint8_t> key, size_t offset) {
	if (offset > key.size() || key.size() - offset < 8 + 5) {
		return false;
	}
	uint64_t length = 0;
	std::memcpy(&length, key.data() + offset, sizeof(length));
	return length >= 5 && length <= 128 && key.size() - offset - 8 >= length &&
	       std::memcmp(key.data() + offset + 8, "KYTY_", 5) == 0;
}

// The end of the session key's switch list (SessionKey: a fixed part, then for every recompiler
// switch [u64 name length][name][u8 set]([u64 value length][value])), or 0.
size_t SwitchListEnd(std::span<const uint8_t> key) {
	size_t offset = 0;
	for (size_t start = 8; start < std::min<size_t>(key.size(), 160); start++) {
		if (IsSwitchEntry(key, start)) {
			offset = start;
			break;
		}
	}
	if (offset == 0) {
		return 0;
	}
	while (offset + 8 <= key.size() && IsSwitchEntry(key, offset)) {
		uint64_t length = 0;
		std::memcpy(&length, key.data() + offset, sizeof(length));
		size_t at = offset + 8 + length;
		if (at >= key.size() || key[at] > 1u) {
			return 0;
		}
		const bool set = key[at] != 0;
		at++;
		if (set) {
			uint64_t value_length = 0;
			if (key.size() - at < 8) return 0;
			std::memcpy(&value_length, key.data() + at, sizeof(value_length));
			if (value_length > 4096 || key.size() - at - 8 < value_length) return 0;
			at += 8 + value_length;
		}
		offset = at;
	}
	return offset;
}

// `session_size` gets the length of the key's session prefix (everything before the 'S').
Parse ParseKey(std::span<const uint8_t> key, ParsedKey& out, ByteWriter& scratch,
               size_t& session_size) {
	const size_t end = SwitchListEnd(key);
	if (end == 0) {
		return Parse::NoSession;
	}
	// After the switches: the GPU-sized-written hash (u64) in current builds, none in older ones.
	for (const size_t at: {end + 8, end}) {
		if (at < key.size() && key[at] == 'S') {
			const auto result = ParseTail(key.subspan(at), out, scratch);
			if (result == Parse::Ok || result == Parse::BadPermutation) {
				session_size = at;
				out.source_size += at;
				return result;
			}
		}
	}
	return Parse::BadSource;
}

// Moves an entry of another session out of the cache directory (plan 4.13: the clean-up is a
// move; deleting the folder is left to the user).
bool MoveOut(const std::filesystem::path& from, const std::filesystem::path& to_directory) {
	std::error_code error;
	std::filesystem::rename(from, to_directory / from.filename(), error);
	return !error;
}

// The entry's key, or its first `limit` bytes.
bool ReadEntryKey(const std::filesystem::path& path, std::vector<uint8_t>& key,
                  size_t limit = SIZE_MAX) {
	std::FILE* file = nullptr;
#ifdef _WIN32
	if (_wfopen_s(&file, path.c_str(), L"rb") != 0) {
		file = nullptr;
	}
#else
	file = std::fopen(path.c_str(), "rb");
#endif
	if (file == nullptr) {
		return false;
	}
	uint8_t header[EntryHeaderSize];
	bool    ok = std::fread(header, 1, sizeof(header), file) == sizeof(header);
	uint32_t magic    = 0;
	uint64_t key_size = 0;
	if (ok) {
		std::memcpy(&magic, header, sizeof(magic));
		std::memcpy(&key_size, header + 8, sizeof(key_size));
		ok = magic == EntryMagic && key_size > 0 && key_size <= MaxKeySize;
	}
	if (ok) {
		key.resize(std::min(static_cast<size_t>(key_size), limit));
		ok = std::fread(key.data(), 1, key.size(), file) == key.size();
	}
	std::fclose(file);
	return ok;
}

// The plan fingerprint of a permutation key's source (PermutationRecipe::plan_fingerprint): xxh3
// of the payload of the source entry the same build wrote next to it, or 0 when there is none.
uint64_t SourcePlanFingerprint(const std::filesystem::path& directory, std::span<const uint8_t> source_key,
                               std::vector<uint8_t>& scratch) {
	const auto hash = XXH3_128bits(source_key.data(), source_key.size());
	const auto path = directory / fmt::format("{:016x}{:016x}.bin", hash.high64, hash.low64);
	std::FILE* file = nullptr;
#ifdef _WIN32
	if (_wfopen_s(&file, path.c_str(), L"rb") != 0) {
		file = nullptr;
	}
#else
	file = std::fopen(path.c_str(), "rb");
#endif
	if (file == nullptr) {
		return 0;
	}
	uint8_t  header[EntryHeaderSize];
	uint32_t magic        = 0;
	uint64_t key_size     = 0;
	uint64_t payload_size = 0;
	uint64_t payload_hash = 0;
	bool     ok           = std::fread(header, 1, sizeof(header), file) == sizeof(header);
	if (ok) {
		std::memcpy(&magic, header, 4);
		std::memcpy(&key_size, header + 8, 8);
		std::memcpy(&payload_size, header + 16, 8);
		std::memcpy(&payload_hash, header + 24, 8);
		ok = magic == EntryMagic && key_size == source_key.size() && payload_size <= (64ull << 20u);
	}
	if (ok) {
		scratch.resize(static_cast<size_t>(key_size + payload_size));
		ok = std::fread(scratch.data(), 1, scratch.size(), file) == scratch.size() &&
		     std::memcmp(scratch.data(), source_key.data(), source_key.size()) == 0;
	}
	std::fclose(file);
	if (!ok) {
		return 0;
	}
	const uint64_t fingerprint = XXH3_64bits(scratch.data() + key_size, payload_size);
	return fingerprint == payload_hash ? fingerprint : 0;
}

// "<32 hex digits>.bin": an entry (HexKey), not the ledger or a temporary file.
bool IsEntryName(const std::filesystem::path& path, std::array<char, 32>& name) {
	const auto file = path.filename().string();
	if (file.size() != 36 || file.compare(32, 4, ".bin") != 0) {
		return false;
	}
	for (size_t i = 0; i < 32; i++) {
		const char c = file[i];
		if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
			return false;
		}
		name[i] = c;
	}
	return true;
}

} // namespace

bool ParseSourceTail(std::span<const uint8_t> bytes, SourceRecipe& recipe, size_t& consumed,
                     ShaderDiskCache::ByteWriter& scratch) {
	ByteReader in(bytes);
	uint8_t    marker          = 0;
	uint32_t   user_data_count = 0;
	uint8_t    kind            = 0;
	if (!in.Get(marker) || marker != 'S' || !in.Get(recipe.stage) || !in.Get(recipe.shader_hash) ||
	    !in.Get(recipe.wave_size) || !in.Get(recipe.user_data_base) || !in.Get(user_data_count) ||
	    !in.Get(recipe.bindless) || !in.Get(recipe.non_fatal) || !in.GetWords(recipe.code) ||
	    !in.GetWords(recipe.back_code) || !in.Get(kind)) {
		return false;
	}
	if (recipe.code.empty() || user_data_count > MaxUserData || !KindMatchesStage(kind, recipe.stage)) {
		return false;
	}
	recipe.kind = static_cast<InputKind>(kind);
	bool ok     = false;
	switch (kind) {
		case 'V': recipe.vertex = ShaderVertexInputInfo {}; ok = ReadInputKey(in, recipe.vertex); break;
		case 'P': recipe.pixel = ShaderPixelInputInfo {}; ok = ReadInputKey(in, recipe.pixel); break;
		case 'C': recipe.compute = ShaderComputeInputInfo {}; ok = ReadInputKey(in, recipe.compute); break;
		default: break;
	}
	if (!ok || !in.Ok()) {
		return false;
	}
	recipe.user_data.assign(user_data_count, 0u);
	scratch.bytes.clear();
	switch (kind) {
		case 'V': AppendSourceKeyTail(scratch, recipe.Fields(), recipe.vertex); break;
		case 'P': AppendSourceKeyTail(scratch, recipe.Fields(), recipe.pixel); break;
		default: AppendSourceKeyTail(scratch, recipe.Fields(), recipe.compute); break;
	}
	if (scratch.bytes.size() > bytes.size() ||
	    std::memcmp(scratch.bytes.data(), bytes.data(), scratch.bytes.size()) != 0) {
		return false;
	}
	consumed = scratch.bytes.size();
	return true;
}

std::string ImportStats::Summary() const {
	return fmt::format("{} files: {} source keys, {} permutation keys ({} without their source "
	                   "entry), {} permutation keys of an older layout (source kept), {} rejected "
	                   "({} session, {} source part), {} unreadable; {} sources and {} permutations "
	                   "new to the ledger; {} entries of other sessions moved out; {:.1f} s{}",
	                   files, sources, permutations, no_source_entry, old_permutations, rejected,
	                   rejected_session, rejected_source, unreadable, new_sources, new_permutations,
	                   moved, seconds, stopped ? " (stopped)" : "");
}

ImportStats ImportDiskCache(const std::filesystem::path& directory, Ledger& ledger, uint32_t threads,
                            const std::atomic<bool>& stop, std::atomic<uint32_t>& files_done,
                            std::atomic<uint32_t>& files_total, std::span<const uint8_t> session_key,
                            const std::filesystem::path& prune_to, bool record) {
	const auto  begin = std::chrono::steady_clock::now();
	ImportStats stats;
	bool        prune = !prune_to.empty() && !session_key.empty();
	if (prune) {
		std::error_code error;
		std::filesystem::create_directories(prune_to, error);
		prune = std::filesystem::is_directory(prune_to, error);
	}
	// The cache directory, and when recording, the folders earlier clean-ups moved entries to
	// (siblings named "<directory>_..."): they are read, never changed.
	std::vector<std::filesystem::path> directories {directory};
	if (record) {
		std::error_code error;
		const auto      prefix = directory.filename().string() + "_";
		const auto parent = directory.has_parent_path() ? directory.parent_path() : std::filesystem::path(".");
		std::filesystem::directory_iterator it(parent, error);
		const std::filesystem::directory_iterator end;
		for (; !error && it != end; it.increment(error)) {
			const auto name = it->path().filename().string();
			if (it->is_directory(error) && name.size() > prefix.size() && name.starts_with(prefix)) {
				directories.push_back(it->path());
			}
		}
	}
	struct Name {
		std::array<char, 32> hex {};
		uint8_t              directory = 0;
	};
	std::vector<Name> names;
	for (size_t d = 0; d < directories.size() && d < 255; d++) {
		std::error_code error;
		std::filesystem::directory_iterator it(directories[d], error);
		const std::filesystem::directory_iterator end;
		for (; !error && it != end; it.increment(error)) {
			Name name;
			name.directory = static_cast<uint8_t>(d);
			if (IsEntryName(it->path(), name.hex)) {
				names.push_back(name);
			}
			if (stop.load(std::memory_order_relaxed)) {
				break;
			}
		}
	}
	stats.files = names.size();
	files_done.store(0);
	files_total.store(static_cast<uint32_t>(names.size()));

	std::atomic<size_t> next {0};
	std::mutex          stats_mutex;
	auto work = [&] {
		Common::LowerBackgroundThreadPriority();
		ImportStats           local;
		std::vector<uint8_t>  key;
		ByteWriter            scratch;
		std::vector<uint8_t>  source_file;
		// Plan fingerprints by source entry (xxh3 of the source key), this thread's.
		std::unordered_map<uint64_t, uint64_t> fingerprints;
		ParsedKey             parsed;
		for (;;) {
			if (stop.load(std::memory_order_acquire)) {
				local.stopped = true;
				break;
			}
			const size_t index = next.fetch_add(1);
			if (index >= names.size()) {
				break;
			}
			const auto& name    = names[index];
			const auto& in      = directories[name.directory];
			auto        path    = in / (std::string(name.hex.data(), name.hex.size()) + ".bin");
			// Moves are for the cache directory only.
			const bool  movable = prune && name.directory == 0;
			size_t      session_size = 0;
			if (!record) {
				// The session check only (the move is all this pass is for).
				if (!ReadEntryKey(path, key, session_key.size())) {
					local.unreadable++;
				} else if (movable && (key.size() != session_key.size() ||
				                     std::memcmp(key.data(), session_key.data(), key.size()) != 0) &&
				           MoveOut(path, prune_to)) {
					local.moved++;
				}
			} else if (!ReadEntryKey(path, key)) {
				local.unreadable++;
			} else if (const auto result = ParseKey(key, parsed, scratch, session_size);
			           result == Parse::NoSession || result == Parse::BadSource) {
				local.rejected++;
				(result == Parse::NoSession ? local.rejected_session : local.rejected_source)++;
				// A key this build cannot read is of another format: this build never asks for
				// it (unless it starts with this session's key: then it stays, whatever it is).
				const bool this_session =
				    key.size() >= session_key.size() &&
				    std::memcmp(key.data(), session_key.data(), session_key.size()) == 0;
				if (movable && !this_session && MoveOut(path, prune_to)) {
					local.moved++;
				}
			} else {
				bool       added = false;
				const auto id    = ledger.RecordSource(parsed.recipe, &added);
				local.new_sources += added ? 1u : 0u;
				if (result == Parse::BadPermutation) {
					// The source is recorded; the variant is of an older layout and is lost.
					local.old_permutations++;
				} else if (parsed.permutation) {
					local.permutations++;
					const auto source_key = std::span<const uint8_t>(key).first(parsed.source_size);
					const auto source_id  = XXH3_64bits(source_key.data(), source_key.size());
					auto       found      = fingerprints.find(source_id);
					if (found == fingerprints.end()) {
						// Next to it, or in another of the folders (a clean-up may have split
						// a session's entries between them).
						uint64_t fingerprint = SourcePlanFingerprint(in, source_key, source_file);
						for (size_t d = 0; fingerprint == 0 && d < directories.size(); d++) {
							if (d != name.directory) {
								fingerprint = SourcePlanFingerprint(directories[d], source_key, source_file);
							}
						}
						found = fingerprints.emplace(source_id, fingerprint).first;
					}
					if (found->second == 0) {
						local.no_source_entry++;
					} else {
						local.new_permutations += ledger.RecordPermutation(id, parsed.push_data_cursor,
						                                                   parsed.specialization,
						                                                   found->second)
						                              ? 1u
						                              : 0u;
					}
				} else {
					local.sources++;
				}
				// The recipe is in the ledger now (unless the ledger cannot be written): an
				// entry of another session can go, the replay writes this session's.
				const bool other_session =
				    session_size != session_key.size() ||
				    std::memcmp(key.data(), session_key.data(), session_size) != 0;
				if (movable && other_session && !ledger.WriteFailed() && MoveOut(path, prune_to)) {
					local.moved++;
				}
			}
			files_done.fetch_add(1, std::memory_order_relaxed);
		}
		std::lock_guard lock(stats_mutex);
		stats.unreadable += local.unreadable;
		stats.rejected += local.rejected;
		stats.sources += local.sources;
		stats.permutations += local.permutations;
		stats.new_sources += local.new_sources;
		stats.new_permutations += local.new_permutations;
		stats.moved += local.moved;
		stats.rejected_session += local.rejected_session;
		stats.rejected_source += local.rejected_source;
		stats.old_permutations += local.old_permutations;
		stats.no_source_entry += local.no_source_entry;
		stats.stopped = stats.stopped || local.stopped;
	};
	std::vector<std::thread> workers;
	for (uint32_t t = 1; t < std::max(threads, 1u); t++) {
		workers.emplace_back(work);
	}
	work();
	for (auto& worker: workers) {
		worker.join();
	}
	stats.stopped = stats.stopped || stop.load();
	stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	return stats;
}

uint64_t ImportModuleMap(const std::filesystem::path& directory, uint32_t threads, const std::atomic<bool>& stop,
                         std::atomic<uint32_t>& files_done, std::atomic<uint32_t>& files_total,
                         std::span<const uint8_t> session_key) {
	if (!PipelineRecipe::ModuleImportWanted()) {
		return 0;
	}
	std::vector<std::filesystem::path> directories {directory};
	{
		std::error_code error;
		const auto      prefix = directory.filename().string() + "_";
		const auto parent = directory.has_parent_path() ? directory.parent_path() : std::filesystem::path(".");
		std::filesystem::directory_iterator it(parent, error);
		const std::filesystem::directory_iterator end;
		for (; !error && it != end; it.increment(error)) {
			const auto name = it->path().filename().string();
			if (it->is_directory(error) && name.size() > prefix.size() && name.starts_with(prefix)) {
				directories.push_back(it->path());
			}
		}
	}
	std::vector<std::pair<std::filesystem::path, uint8_t>> files;
	for (size_t d = 0; d < directories.size() && d < 255; d++) {
		std::error_code error;
		std::filesystem::directory_iterator it(directories[d], error);
		const std::filesystem::directory_iterator end;
		std::array<char, 32> hex {};
		for (; !error && it != end && !stop.load(std::memory_order_relaxed); it.increment(error)) {
			if (IsEntryName(it->path(), hex)) {
				files.emplace_back(it->path(), static_cast<uint8_t>(d));
			}
		}
	}
	files_done.store(0);
	files_total.store(static_cast<uint32_t>(files.size()));
	std::atomic<size_t>   next {0};
	std::atomic<uint64_t> found {0};
	std::atomic<bool>     stopped {false};
	auto work = [&] {
		Common::LowerBackgroundThreadPriority();
		std::vector<uint8_t>  key;
		std::vector<uint32_t> words;
		std::vector<uint8_t>  source_file;
		ByteWriter            scratch;
		ParsedKey             parsed;
		for (;;) {
			if (stop.load(std::memory_order_acquire)) {
				stopped = true;
				break;
			}
			const size_t index = next.fetch_add(1);
			if (index >= files.size() || PipelineRecipe::UnknownModuleCount() == 0) {
				break;
			}
			const auto& [path, dir] = files[index];
			files_done.fetch_add(1, std::memory_order_relaxed);
			std::FILE* file = nullptr;
#ifdef _WIN32
			if (_wfopen_s(&file, path.c_str(), L"rb") != 0) {
				file = nullptr;
			}
#else
			file = std::fopen(path.c_str(), "rb");
#endif
			if (file == nullptr) {
				continue;
			}
			uint8_t  header[EntryHeaderSize];
			uint32_t magic        = 0;
			uint64_t key_size     = 0;
			uint64_t payload_size = 0;
			uint64_t count        = 0;
			bool     ok = std::fread(header, 1, sizeof(header), file) == sizeof(header);
			if (ok) {
				std::memcpy(&magic, header, 4);
				std::memcpy(&key_size, header + 8, 8);
				std::memcpy(&payload_size, header + 16, 8);
				ok = magic == EntryMagic && key_size > 0 && key_size <= MaxKeySize && payload_size >= 8 &&
				     payload_size <= (64ull << 20u);
			}
			// The payload of a permutation starts with its SPIR-V word count (EncodePermutationRecord).
			ok = ok && std::fseek(file, static_cast<long>(EntryHeaderSize + key_size), SEEK_SET) == 0 &&
			     std::fread(&count, sizeof(count), 1, file) == 1 && count != 0 && count * 4 + 8 <= payload_size &&
			     PipelineRecipe::IsUnknownModuleSize(count);
			if (ok) {
				words.resize(static_cast<size_t>(count));
				ok = std::fread(words.data(), 4, words.size(), file) == words.size() &&
				     PipelineRecipe::IsUnknownModule(PipelineRecipe::HashWords(words));
			}
			if (ok) {
				key.resize(static_cast<size_t>(key_size));
				ok = std::fseek(file, static_cast<long>(EntryHeaderSize), SEEK_SET) == 0 &&
				     std::fread(key.data(), 1, key.size(), file) == key.size();
			}
			std::fclose(file);
			if (!ok) {
				continue;
			}
			size_t session_size = 0;
			if (ParseKey(key, parsed, scratch, session_size) != Parse::Ok || !parsed.permutation ||
			    static_cast<ShaderType>(parsed.recipe.stage) == ShaderType::Compute) {
				continue;
			}
			if (key.size() >= session_key.size() &&
			    std::memcmp(key.data(), session_key.data(), session_key.size()) == 0) {
				continue; // this session's: the replay and Get record those
			}
			const auto source_key  = std::span<const uint8_t>(key).first(parsed.source_size);
			uint64_t   fingerprint = SourcePlanFingerprint(directories[dir], source_key, source_file);
			for (size_t d = 0; fingerprint == 0 && d < directories.size(); d++) {
				if (d != dir) {
					fingerprint = SourcePlanFingerprint(directories[d], source_key, source_file);
				}
			}
			if (fingerprint == 0) {
				continue;
			}
			// The ledger's source id: xxh3 of the source part after the session prefix (RecordSourceTail).
			const auto tail      = std::span<const uint8_t>(key).subspan(session_size, parsed.source_size - session_size);
			const auto source_id = XXH3_64bits(tail.data(), tail.size());
			// The specialization bytes as the key holds them ('P', bytes, u32 cursor; re-encoded equal by ParseTail).
			const auto rest = std::span<const uint8_t>(key).subspan(parsed.source_size);
			const auto spec = rest.subspan(1, rest.size() - 1 - sizeof(uint32_t));
			const auto stable = PipelineRecipe::StableId(source_id, parsed.push_data_cursor, fingerprint, spec);
			const auto tag    = XXH3_64bits(key.data(), session_size) | 1u;
			PipelineRecipe::NoteModuleFrom(tag, stable, words);
			found.fetch_add(1, std::memory_order_relaxed);
		}
	};
	std::vector<std::thread> workers;
	for (uint32_t t = 1; t < std::max(threads, 1u); t++) {
		workers.emplace_back(work);
	}
	work();
	for (auto& worker: workers) {
		worker.join();
	}
	if (!stopped.load() && !stop.load()) {
		PipelineRecipe::MarkModuleImportDone();
	}
	return found.load();
}

} // namespace Libs::Graphics::ShaderLedger
