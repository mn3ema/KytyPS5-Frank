#include "graphics/host_gpu/renderer/pipeline/shaderLedger.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderLedgerImport.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompileProgress.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <system_error>
#include <xxhash.h>

namespace Libs::Graphics {
enum class ZeroCompileEvent : uint32_t;
void ZeroCompileNote(ZeroCompileEvent event, uint64_t hash, double ms); // pipelineCache.cpp
} // namespace Libs::Graphics

namespace Libs::Graphics::ShaderLedger {
namespace {

constexpr uint32_t RecordMagic   = 0x47444c4bu; // 'KLDG'
// 3: permutations carry their plan fingerprint; 2: source recipes as key tails; 1: raw structs.
constexpr uint16_t RecordVersion = 3;
constexpr uint8_t  KindSource      = 'S';
constexpr uint8_t  KindPermutation = 'P';
constexpr uint8_t  KindImported    = 'I';
constexpr size_t   HeaderSize      = 4 + 2 + 1 + 1 + 4 + 8;
// A record larger than this is damage (the largest guest program is well under 1 MB).
constexpr uint32_t MaxPayload = 64u << 20u;

template <typename... Args>
void LedgerLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadWholeFile(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
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
	bytes.clear();
	uint8_t chunk[1u << 16u];
	for (;;) {
		const auto read = std::fread(chunk, 1, sizeof(chunk), file);
		bytes.insert(bytes.end(), chunk, chunk + read);
		if (read < sizeof(chunk)) {
			break;
		}
	}
	std::fclose(file);
	return true;
}

void Publish(uint64_t total) {
	auto& progress = GetShaderPrecompileProgress();
	progress.ledger_total.store(static_cast<uint32_t>(total), std::memory_order_relaxed);
	progress.ledger_known.store(true, std::memory_order_release);
}

bool DecodeSource(std::span<const uint8_t> payload, SourceRecipe& recipe) {
	if (payload.size() < sizeof(uint64_t)) {
		return false;
	}
	std::memcpy(&recipe.id, payload.data(), sizeof(uint64_t));
	const auto                  tail     = payload.subspan(sizeof(uint64_t));
	size_t                      consumed = 0;
	ShaderDiskCache::ByteWriter scratch;
	return XXH3_64bits(tail.data(), tail.size()) == recipe.id &&
	       ParseSourceTail(tail, recipe, consumed, scratch) && consumed == tail.size();
}

bool DecodePermutation(std::span<const uint8_t> payload, PermutationRecipe& recipe, uint64_t& id) {
	ShaderDiskCache::ByteReader reader(payload);
	uint64_t                    size = 0;
	if (!reader.Get(id) || !reader.Get(recipe.source_id) || !reader.Get(recipe.push_data_cursor) ||
	    !reader.Get(recipe.plan_fingerprint) || !reader.GetCount(size)) {
		return false;
	}
	recipe.specialization.resize(size);
	for (auto& byte: recipe.specialization) {
		if (!reader.Get(byte)) {
			return false;
		}
	}
	return reader.AtEnd();
}

} // namespace

std::unique_ptr<Ledger> Ledger::Open(const std::filesystem::path& directory) {
	if (const char* value = std::getenv("KYTY_SHADER_LEDGER"); value != nullptr && value[0] == '0') {
		LedgerLog("Shader ledger: off (KYTY_SHADER_LEDGER=0)");
		return nullptr;
	}
	if (directory.empty()) {
		return nullptr;
	}
	std::error_code error;
	std::filesystem::create_directories(directory, error);
	auto ledger = std::make_unique<Ledger>(directory / "ledger.bin");
	LedgerLog("Shader ledger: {} sources, {} permutations ({})", ledger->m_source_count,
	          ledger->m_permutation_count, (directory / "ledger.bin").generic_string());
	return ledger;
}

Ledger::Ledger(std::filesystem::path path): m_path(std::move(path)) {
	Load();
}

Ledger::~Ledger() {
	if (m_file != nullptr) {
		std::fclose(m_file);
	}
}

void Ledger::Load() {
	// v438f (Fable 11-15): a shipped list is its own read-only file, list.bin next to ledger.bin,
	// read first (its order is the warm-up order: played content first); only ledger.bin is
	// appended to, so a release never overwrites a player's recordings. KYTY_SHADER_LIST=0: not read.
	const char* list_value = std::getenv("KYTY_SHADER_LIST");
	if (list_value == nullptr || list_value[0] != '0') {
		LoadFile(m_path.parent_path() / "list.bin", false);
	}
	LoadFile(m_path, true);
	Publish(m_source_count + m_permutation_count - std::min(m_unusable, m_source_count + m_permutation_count));
}

void Ledger::LoadFile(const std::filesystem::path& path, bool own) {
	std::vector<uint8_t> bytes;
	if (!ReadWholeFile(path, bytes)) {
		return;
	}
	const auto sources_before      = m_source_count;
	const auto permutations_before = m_permutation_count;
	size_t offset = 0;
	while (bytes.size() - offset >= HeaderSize) {
		uint32_t magic = 0;
		uint16_t version = 0;
		uint8_t  kind = 0;
		uint32_t size = 0;
		uint64_t hash = 0;
		std::memcpy(&magic, bytes.data() + offset, 4);
		std::memcpy(&version, bytes.data() + offset + 4, 2);
		std::memcpy(&kind, bytes.data() + offset + 6, 1);
		std::memcpy(&size, bytes.data() + offset + 8, 4);
		std::memcpy(&hash, bytes.data() + offset + 12, 8);
		if (magic != RecordMagic || size > MaxPayload || bytes.size() - offset - HeaderSize < size) {
			break;
		}
		const std::span<const uint8_t> payload(bytes.data() + offset + HeaderSize, size);
		if (XXH3_64bits(payload.data(), payload.size()) != hash) {
			break;
		}
		offset += HeaderSize + size;
		if (version != RecordVersion) {
			continue; // a record of another ledger version: kept in the file, not replayed
		}
		if (kind == KindSource) {
			SourceRecipe recipe;
			if (DecodeSource(payload, recipe) && m_ids.insert(recipe.id).second) {
				m_sources.push_back(std::move(recipe));
				m_source_count++;
			}
		} else if (kind == KindPermutation) {
			PermutationRecipe recipe;
			uint64_t          id = 0;
			if (DecodePermutation(payload, recipe, id) && m_ids.insert(id).second) {
				m_permutations.push_back(std::move(recipe));
				m_permutation_count++;
			}
		} else if (kind == KindImported) {
			m_imported = true;
		}
	}
	if (offset != bytes.size() && own) {
		// A damaged or partly written tail (the emulator was closed mid-append): cut it off, so
		// that new records do not land behind it.
		LedgerLog("Shader ledger: dropping {} damaged bytes at the end of {}", bytes.size() - offset,
		          path.generic_string());
		std::error_code error;
		std::filesystem::resize_file(path, offset, error);
	} else if (offset != bytes.size()) {
		LedgerLog("Shader ledger: {} is damaged after byte {} (the rest is not read)", path.generic_string(), offset);
	}
	if (!own) {
		LedgerLog("Shader ledger: shipped list {}: {} sources, {} permutations", path.generic_string(),
		          m_source_count - sources_before, m_permutation_count - permutations_before);
	}
}

void Ledger::Reload() {
	std::lock_guard lock(m_mutex);
	// Closed, not just flushed: a file _wfopen_s opened for writing cannot be opened again for
	// reading (Append opens it again at the next record).
	if (m_file != nullptr) {
		std::fclose(m_file);
		m_file = nullptr;
	}
	m_ids.clear();
	m_sources.clear();
	m_permutations.clear();
	m_source_count      = 0;
	m_permutation_count = 0;
	m_unusable          = 0;
	m_imported          = false;
	Load();
}

void Ledger::SetUnusable(uint64_t count) {
	std::lock_guard lock(m_mutex);
	m_unusable = std::min(count, m_source_count + m_permutation_count);
	Publish(m_source_count + m_permutation_count - m_unusable);
}

void Ledger::MarkImported(std::span<const uint8_t> summary) {
	std::lock_guard lock(m_mutex);
	Append(KindImported, summary);
	m_imported = !m_write_failed;
}

void Ledger::TakeRecipes(std::vector<SourceRecipe>& sources, std::vector<PermutationRecipe>& permutations) {
	std::lock_guard lock(m_mutex);
	sources      = std::move(m_sources);
	permutations = std::move(m_permutations);
	m_sources.clear();
	m_permutations.clear();
}

void Ledger::Append(uint8_t kind, std::span<const uint8_t> payload) {
	// Under m_mutex.
	if (m_write_failed) {
		return;
	}
	if (m_file == nullptr) {
#ifdef _WIN32
		if (_wfopen_s(&m_file, m_path.c_str(), L"ab") != 0) {
			m_file = nullptr;
		}
#else
		m_file = std::fopen(m_path.c_str(), "ab");
#endif
		if (m_file == nullptr) {
			m_write_failed = true;
			LedgerLog("Shader ledger: cannot write {}", m_path.generic_string());
			return;
		}
	}
	uint8_t header[HeaderSize] {};
	const uint32_t magic   = RecordMagic;
	const uint16_t version = RecordVersion;
	const uint32_t size    = static_cast<uint32_t>(payload.size());
	const uint64_t hash    = XXH3_64bits(payload.data(), payload.size());
	std::memcpy(header, &magic, 4);
	std::memcpy(header + 4, &version, 2);
	std::memcpy(header + 6, &kind, 1);
	std::memcpy(header + 8, &size, 4);
	std::memcpy(header + 12, &hash, 8);
	if (std::fwrite(header, 1, sizeof(header), m_file) != sizeof(header) ||
	    std::fwrite(payload.data(), 1, payload.size(), m_file) != payload.size() ||
	    std::fflush(m_file) != 0) {
		m_write_failed = true;
		LedgerLog("Shader ledger: write to {} failed; recording stops", m_path.generic_string());
	}
}

uint64_t Ledger::RecordSourceTail(std::span<const uint8_t> tail, bool* added) {
	const uint64_t id = XXH3_64bits(tail.data(), tail.size());
	std::lock_guard lock(m_mutex);
	const bool fresh = m_ids.insert(id).second;
	if (added != nullptr) {
		*added = fresh;
	}
	if (!fresh) {
		return id;
	}
	ShaderDiskCache::ByteWriter payload;
	payload.Put(id);
	payload.PutBytes(tail);
	Append(KindSource, payload.bytes);
	m_source_count++;
	ZeroCompileNote(static_cast<ZeroCompileEvent>(1), id, 0.0); // a source the list lacked (after the startup wait)
	Publish(m_source_count + m_permutation_count - std::min(m_unusable, m_source_count + m_permutation_count));
	return id;
}

uint64_t Ledger::RecordSource(const SourceFields& fields, const ShaderVertexInputInfo& info) {
	ShaderDiskCache::ByteWriter tail;
	AppendSourceKeyTail(tail, fields, info);
	return RecordSourceTail(tail.bytes);
}

uint64_t Ledger::RecordSource(const SourceFields& fields, const ShaderPixelInputInfo& info) {
	ShaderDiskCache::ByteWriter tail;
	AppendSourceKeyTail(tail, fields, info);
	return RecordSourceTail(tail.bytes);
}

uint64_t Ledger::RecordSource(const SourceFields& fields, const ShaderComputeInputInfo& info) {
	ShaderDiskCache::ByteWriter tail;
	AppendSourceKeyTail(tail, fields, info);
	return RecordSourceTail(tail.bytes);
}

uint64_t Ledger::RecordSource(const SourceRecipe& recipe, bool* added) {
	ShaderDiskCache::ByteWriter tail;
	switch (recipe.kind) {
		case InputKind::Vertex: AppendSourceKeyTail(tail, recipe.Fields(), recipe.vertex); break;
		case InputKind::Pixel: AppendSourceKeyTail(tail, recipe.Fields(), recipe.pixel); break;
		default: AppendSourceKeyTail(tail, recipe.Fields(), recipe.compute); break;
	}
	return RecordSourceTail(tail.bytes, added);
}

bool Ledger::RecordPermutation(uint64_t source_id, uint32_t push_data_cursor,
                               const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                               uint64_t plan_fingerprint) {
	if (plan_fingerprint == 0) {
		return false;
	}
	ShaderDiskCache::ByteWriter spec;
	ShaderDiskCache::AppendSpecialization(spec, specialization);
	ShaderDiskCache::ByteWriter body;
	body.Put(source_id);
	body.Put(push_data_cursor);
	body.Put(plan_fingerprint);
	body.Put<uint64_t>(spec.bytes.size());
	body.PutBytes(spec.bytes);
	const uint64_t id = XXH3_64bits(body.bytes.data(), body.bytes.size()) ^ 0x5045524du; // 'PERM'
	std::lock_guard lock(m_mutex);
	if (!m_ids.insert(id).second) {
		return false;
	}
	ShaderDiskCache::ByteWriter payload;
	payload.Put(id);
	payload.PutBytes(body.bytes);
	Append(KindPermutation, payload.bytes);
	m_permutation_count++;
	ZeroCompileNote(static_cast<ZeroCompileEvent>(1), id, 0.0); // a permutation the list lacked
	Publish(m_source_count + m_permutation_count - std::min(m_unusable, m_source_count + m_permutation_count));
	return true;
}

} // namespace Libs::Graphics::ShaderLedger
