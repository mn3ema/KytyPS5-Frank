#include "graphics/host_gpu/renderer/pipeline/pipelineRecipe.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStats.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompileProgress.h"

#include <cstdio>
#include <string>
#include <thread>
#include <deque>
#include <condition_variable>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <xxhash.h>

namespace Libs::Graphics::PipelineRecipe {

namespace {

constexpr uint32_t RecipeVersion = 1;
constexpr char     FileMagic[8]  = {'K', 'y', 'P', 'R', 'e', 'c', '0', '1'};

// The SPIR-V store: <directory>/<hash>.spv, written by one background thread.
struct SpirvStore {
	std::mutex                                 mutex;
	std::condition_variable                    wake;
	std::filesystem::path                      directory;
	std::unordered_set<Hash128, Hash128Hasher> stored; // on disk or queued
	std::deque<std::pair<Hash128, std::vector<uint32_t>>> queue;
	std::thread                                writer;
	bool                                       enabled = false;
	uint32_t                                   in_flight = 0; // queued or being written
};
SpirvStore& Spirv() {
	static SpirvStore store;
	return store;
}

std::string HashName(const Hash128& hash) {
	char text[40];
	std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hash.high),
	              static_cast<unsigned long long>(hash.low));
	return text;
}

bool LoadSpirv(const Hash128& hash, std::vector<uint32_t>& words) {
	auto& store = Spirv();
	if (!store.enabled) {
		return false;
	}
	const auto path = store.directory / (HashName(hash) + ".spv");
	std::FILE* file = std::fopen(path.string().c_str(), "rb");
	if (file == nullptr) {
		return false;
	}
	std::fseek(file, 0, SEEK_END);
	const auto size = std::ftell(file);
	std::fseek(file, 0, SEEK_SET);
	bool ok = size > 0 && size % 4 == 0;
	if (ok) {
		words.resize(static_cast<size_t>(size) / 4u);
		ok = std::fread(words.data(), 1, static_cast<size_t>(size), file) == static_cast<size_t>(size);
	}
	std::fclose(file);
	return ok && HashWords(words) == hash;
}

std::mutex                                                        g_modules_mutex;
std::unordered_map<VkShaderModule, Hash128>                       g_module_hashes;
std::unordered_map<Hash128, vk::ShaderModule, Hash128Hasher>      g_modules_by_hash;

class Writer {
public:
	explicit Writer(std::vector<uint8_t>& out): m_out(out) {}
	template <typename T>
	void Put(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
		m_out.insert(m_out.end(), bytes, bytes + sizeof(T));
	}
	void PutWords(std::span<const uint32_t> words) {
		Put(static_cast<uint32_t>(words.size()));
		const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
		m_out.insert(m_out.end(), bytes, bytes + words.size_bytes());
	}

private:
	std::vector<uint8_t>& m_out;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> in): m_in(in) {}
	template <typename T>
	bool Get(T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (m_pos + sizeof(T) > m_in.size()) {
			m_ok = false;
			return false;
		}
		std::memcpy(&value, m_in.data() + m_pos, sizeof(T));
		m_pos += sizeof(T);
		return true;
	}
	bool GetWords(std::vector<uint32_t>& words) {
		uint32_t count = 0;
		if (!Get(count) || m_pos + uint64_t {count} * 4u > m_in.size()) {
			m_ok = false;
			return false;
		}
		words.resize(count);
		std::memcpy(words.data(), m_in.data() + m_pos, uint64_t {count} * 4u);
		m_pos += uint64_t {count} * 4u;
		return true;
	}
	[[nodiscard]] bool Ok() const { return m_ok; }
	[[nodiscard]] bool AtEnd() const { return m_pos == m_in.size(); }

private:
	std::span<const uint8_t> m_in;
	size_t                   m_pos = 0;
	bool                     m_ok  = true;
};

template <typename Flags>
uint32_t FlagBits(Flags flags) {
	return static_cast<uint32_t>(static_cast<typename Flags::MaskType>(flags));
}

uint32_t FloatBits(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

float BitsFloat(uint32_t bits) {
	float value = 0.0f;
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

bool ChainHas(const void* chain, const void* target) {
	for (const auto* node = static_cast<const vk::BaseInStructure*>(chain); node != nullptr;
	     node = reinterpret_cast<const vk::BaseInStructure*>(node->pNext)) {
		if (node == target) {
			return true;
		}
	}
	return false;
}

} // namespace

Hash128 HashWords(std::span<const uint32_t> words) {
	const auto hash = XXH3_128bits(words.data(), words.size_bytes());
	return {hash.low64, hash.high64};
}

Hash128 HashBytes(std::span<const uint8_t> bytes) {
	const auto hash = XXH3_128bits(bytes.data(), bytes.size());
	return {hash.low64, hash.high64};
}

void SetSpirvStore(const std::filesystem::path& directory) {
	auto&           store = Spirv();
	std::lock_guard lock(store.mutex);
	if (store.enabled) {
		return;
	}
	std::error_code error;
	std::filesystem::create_directories(directory, error);
	store.directory = directory;
	for (const auto& entry: std::filesystem::directory_iterator(directory, error)) {
		const auto name = entry.path().stem().string();
		if (entry.path().extension() != ".spv" || name.size() != 32) {
			continue;
		}
		Hash128 hash;
		hash.high = std::strtoull(name.substr(0, 16).c_str(), nullptr, 16);
		hash.low  = std::strtoull(name.substr(16).c_str(), nullptr, 16);
		store.stored.insert(hash);
	}
	store.enabled = true;
	store.writer  = std::thread([&store] {
        for (;;) {
            std::pair<Hash128, std::vector<uint32_t>> item;
            {
                std::unique_lock lock(store.mutex);
                store.wake.wait(lock, [&store] { return !store.queue.empty(); });
                item = std::move(store.queue.front());
                store.queue.pop_front();
            }
            const auto path = store.directory / (HashName(item.first) + ".spv");
            auto       temp = path;
            temp += ".tmp";
            if (std::FILE* file = std::fopen(temp.string().c_str(), "wb"); file != nullptr) {
                const bool ok = std::fwrite(item.second.data(), 4, item.second.size(), file) == item.second.size();
                std::fclose(file);
                std::error_code error;
                if (ok) {
                    std::filesystem::rename(temp, path, error);
                } else {
                    std::filesystem::remove(temp, error);
                }
            }
            {
                std::lock_guard lock(store.mutex);
                store.in_flight--;
            }
        }
    });
	store.writer.detach();
}

void RegisterModule(vk::ShaderModule module, std::span<const uint32_t> spirv, bool keep) {
	const auto hash = HashWords(spirv);
	{
		std::lock_guard lock(g_modules_mutex);
		g_module_hashes[static_cast<VkShaderModule>(module)] = hash;
		g_modules_by_hash.try_emplace(hash, module);
	}
	if (auto& store = Spirv(); keep && store.enabled) {
		std::lock_guard lock(store.mutex);
		if (store.stored.insert(hash).second) {
			store.queue.emplace_back(hash, std::vector<uint32_t>(spirv.begin(), spirv.end()));
			store.in_flight++;
			store.wake.notify_one();
		}
	}
}

bool FindModuleHash(vk::ShaderModule module, Hash128& hash) {
	std::lock_guard lock(g_modules_mutex);
	const auto      found = g_module_hashes.find(static_cast<VkShaderModule>(module));
	if (found == g_module_hashes.end()) {
		return false;
	}
	hash = found->second;
	return true;
}

vk::ShaderModule FindModule(const Hash128& hash) {
	std::lock_guard lock(g_modules_mutex);
	const auto      found = g_modules_by_hash.find(hash);
	return found == g_modules_by_hash.end() ? vk::ShaderModule {} : found->second;
}

namespace {

// The interface a module declares: every variable of an interface storage class with its
// decorations (set, binding, location, component, built-in) and array length, sorted and hashed.
// Two modules of one permutation with the same signature fit the same pipeline layout and vertex
// input (a recompiler change that added a binding or an input is never renamed).
uint64_t InterfaceSignature(std::span<const uint32_t> spirv) {
	if (spirv.size() < 5) {
		return 0;
	}
	struct Decor {
		uint32_t set = ~0u, binding = ~0u, location = ~0u, component = ~0u, builtin = ~0u;
	};
	std::unordered_map<uint32_t, Decor>                         decor;
	std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> pointers;  // id -> (class, pointee)
	std::unordered_map<uint32_t, uint32_t>                      arrays;    // id -> length constant id (0: runtime)
	std::unordered_map<uint32_t, uint32_t>                      constants; // id -> value
	std::vector<std::pair<uint32_t, uint32_t>>                  variables; // (type, id) + class below
	std::vector<uint32_t>                                       classes;
	for (size_t i = 5; i < spirv.size();) {
		const uint32_t count  = spirv[i] >> 16u;
		const uint32_t opcode = spirv[i] & 0xffffu;
		if (count == 0 || i + count > spirv.size()) {
			return 0;
		}
		const auto* w = &spirv[i];
		switch (opcode) {
			case 71: // OpDecorate target decoration literal
				if (count >= 4) {
					auto& d = decor[w[1]];
					switch (w[2]) {
						case 34: d.set = w[3]; break;
						case 33: d.binding = w[3]; break;
						case 30: d.location = w[3]; break;
						case 31: d.component = w[3]; break;
						case 11: d.builtin = w[3]; break;
						default: break;
					}
				}
				break;
			case 32: // OpTypePointer id class type
				if (count >= 4) {
					pointers[w[1]] = {w[2], w[3]};
				}
				break;
			case 28: // OpTypeArray id element length
				if (count >= 4) {
					arrays[w[1]] = w[3];
				}
				break;
			case 29: // OpTypeRuntimeArray id element
				if (count >= 3) {
					arrays[w[1]] = 0;
				}
				break;
			case 43: // OpConstant type id value
				if (count >= 4) {
					constants[w[2]] = w[3];
				}
				break;
			case 59: // OpVariable type id class
				if (count >= 4) {
					const uint32_t storage = w[3];
					// UniformConstant, Input, Uniform, Output, PushConstant, StorageBuffer
					if (storage == 0 || storage == 1 || storage == 2 || storage == 3 || storage == 9 || storage == 12) {
						variables.emplace_back(w[1], w[2]);
						classes.push_back(storage);
					}
				}
				break;
			default: break;
		}
		i += count;
	}
	std::vector<std::array<uint32_t, 7>> rows;
	rows.reserve(variables.size());
	for (size_t v = 0; v < variables.size(); v++) {
		const auto [type, id] = variables[v];
		const auto d          = decor.contains(id) ? decor[id] : Decor {};
		uint32_t   length     = 1;
		if (const auto pointer = pointers.find(type); pointer != pointers.end()) {
			if (const auto array = arrays.find(pointer->second.second); array != arrays.end()) {
				length = array->second == 0 ? ~0u
				         : constants.contains(array->second) ? constants[array->second]
				                                              : ~1u;
			}
		}
		rows.push_back({classes[v], d.set, d.binding, d.location, d.component, d.builtin, length});
	}
	std::sort(rows.begin(), rows.end());
	return XXH3_64bits(rows.data(), rows.size() * sizeof(rows[0])) | 1u; // never 0
}

struct ModuleMap {
	struct Known {
		Hash128  stable;
		uint64_t signature = 0;
		bool     ambiguous = false; // the same SPIR-V was recorded for two permutations: see Remap
		// v423b: every stable id the module was recorded under (aliases included, AliasModule);
		// Remap renames when all of them that have a module in this session agree on it.
		std::vector<Hash128> stables;
	};
	struct Current {
		Hash128  module;
		uint64_t signature = 0;
	};
	std::mutex                                           mutex;
	bool                                                 open = false;
	uint64_t                                             tag  = 0;
	std::FILE*                                           file = nullptr;
	std::unordered_map<Hash128, Known, Hash128Hasher>    by_module;  // any session
	std::unordered_map<Hash128, Current, Hash128Hasher>  current;    // this session, by stable id
	std::unordered_set<Hash128, Hash128Hasher>           current_modules; // the modules of `current`
	std::unordered_set<Hash128, Hash128Hasher>           written;    // xor of (tag, stable, module)
	bool                                                 imported = false; // v394: the marker record was read or written
	std::unordered_set<Hash128, Hash128Hasher>           unknown;    // v394: recipe stages not in the map
	std::unordered_set<uint64_t>                         unknown_sizes; // their SPIR-V word counts
};
ModuleMap& Modules() {
	static ModuleMap map;
	return map;
}
constexpr char ModuleMagic[8] = {'K', 'y', 'M', 'o', 'd', '0', '0', '1'};

struct ModuleRecord {
	uint64_t tag = 0;
	Hash128  stable;
	Hash128  module;
	uint64_t signature = 0;
};
static_assert(sizeof(ModuleRecord) == 48);

Hash128 RecordKey(const ModuleRecord& record) {
	return HashBytes({reinterpret_cast<const uint8_t*>(&record), offsetof(ModuleRecord, signature)});
}

// Under the map's mutex.
void AddRecord(ModuleMap& map, const ModuleRecord& record, bool write) {
	if (!map.written.insert(RecordKey(record)).second) {
		return;
	}
	if (record.tag == 0) {
		// v394: the marker of a finished import of the disk cache's other sessions.
		map.imported = true;
		if (write && map.file != nullptr) {
			std::fwrite(&record, sizeof(record), 1, map.file);
			std::fflush(map.file);
		}
		return;
	}
	map.unknown.erase(record.module);
	if (const auto known = map.by_module.find(record.module); known != map.by_module.end()) {
		if (!(known->second.stable == record.stable)) {
			known->second.ambiguous = true;
			if (std::find(known->second.stables.begin(), known->second.stables.end(), record.stable) ==
			        known->second.stables.end() &&
			    known->second.stables.size() < 64) {
				known->second.stables.push_back(record.stable);
			}
		}
	} else {
		map.by_module[record.module] = {record.stable, record.signature, false, {record.stable}};
	}
	if (record.tag == map.tag) {
		map.current[record.stable] = {record.module, record.signature};
		map.current_modules.insert(record.module);
	}
	if (write && map.file != nullptr) {
		std::fwrite(&record, sizeof(record), 1, map.file);
		std::fflush(map.file);
	}
}

} // namespace

Hash128 StableId(uint64_t source_id, uint32_t push_data_cursor, uint64_t plan_fingerprint,
                 std::span<const uint8_t> specialization) {
	std::vector<uint8_t> bytes;
	Writer               w(bytes);
	w.Put(uint32_t {0x6d6f6431u}); // 'mod1'
	w.Put(source_id);
	w.Put(push_data_cursor);
	w.Put(plan_fingerprint);
	w.Put(static_cast<uint64_t>(specialization.size()));
	bytes.insert(bytes.end(), specialization.begin(), specialization.end());
	return HashBytes(bytes);
}

void OpenModuleMap(const std::filesystem::path& path, uint64_t session_tag) {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	if (map.open) {
		return;
	}
	map.tag = session_tag;
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	uint64_t valid_end = 0;
	size_t   records   = 0;
	if (std::FILE* in = std::fopen(path.string().c_str(), "rb"); in != nullptr) {
		char magic[8] {};
		if (std::fread(magic, 1, sizeof(magic), in) == sizeof(magic) &&
		    std::memcmp(magic, ModuleMagic, sizeof(magic)) == 0) {
			valid_end = sizeof(magic);
			ModuleRecord record;
			while (std::fread(&record, sizeof(record), 1, in) == 1) {
				valid_end += sizeof(record);
				AddRecord(map, record, false);
				records++;
			}
		}
		std::fclose(in);
	}
	if (valid_end == 0) {
		map.file = std::fopen(path.string().c_str(), "wb");
		if (map.file != nullptr) {
			std::fwrite(ModuleMagic, 1, sizeof(ModuleMagic), map.file);
			std::fflush(map.file);
		}
	} else {
		std::filesystem::resize_file(path, valid_end, error); // a torn record is cut off
		map.file = std::fopen(path.string().c_str(), "ab");
	}
	map.open = true;
	LOGF("Pipeline modules: %zu records (%zu of this session) in %s\n", records, map.current.size(),
	     path.string().c_str());
}

void NoteModule(const Hash128& stable, std::span<const uint32_t> spirv, bool store) {
	const auto hash = HashWords(spirv);
	if (auto& spirv_store = Spirv(); store && spirv_store.enabled) {
		std::lock_guard lock(spirv_store.mutex);
		if (spirv_store.stored.insert(hash).second) {
			spirv_store.queue.emplace_back(hash, std::vector<uint32_t>(spirv.begin(), spirv.end()));
			spirv_store.in_flight++;
			spirv_store.wake.notify_one();
		}
	}
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	if (!map.open) {
		return;
	}
	if (const auto found = map.current.find(stable); found != map.current.end() && found->second.module == hash) {
		return;
	}
	AddRecord(map, {map.tag, stable, hash, InterfaceSignature(spirv)}, true);
}

void WaitSpirvStore() {
	auto& store = Spirv();
	for (;;) {
		{
			std::lock_guard lock(store.mutex);
			if (store.in_flight == 0) {
				return;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
}

void CollectUnknownModules(std::span<const Store::Entry> loaded) {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	if (!map.open || map.imported) {
		return;
	}
	for (const auto& entry: loaded) {
		const auto& recipe = entry.bytes;
		constexpr size_t header = 3 * sizeof(uint32_t);
		if (recipe.size() < header) {
			continue;
		}
		uint32_t stage_count = 0;
		std::memcpy(&stage_count, recipe.data() + 2 * sizeof(uint32_t), sizeof(stage_count));
		size_t pos = header;
		for (uint32_t i = 0; i < stage_count && i < 4 && pos + 8 <= recipe.size(); i++) {
			uint32_t kind = 0;
			std::memcpy(&kind, recipe.data() + pos + 4, sizeof(kind));
			pos += 8;
			if (kind == 0) {
				if (pos + sizeof(Hash128) > recipe.size()) {
					break;
				}
				Hash128 hash;
				std::memcpy(&hash, recipe.data() + pos, sizeof(hash));
				if (!map.by_module.contains(hash)) {
					map.unknown.insert(hash);
				}
				pos += sizeof(Hash128);
			}
			pos += 4;
		}
	}
	auto& store = Spirv();
	for (const auto& hash: map.unknown) {
		std::error_code error;
		const auto      size = std::filesystem::file_size(store.directory / (HashName(hash) + ".spv"), error);
		if (!error && size % 4 == 0) {
			map.unknown_sizes.insert(size / 4);
		}
	}
	LOGF("Pipeline modules: %zu recipe stages of earlier sessions not in the map (%zu sizes)\n", map.unknown.size(),
	     map.unknown_sizes.size());
}

bool ModuleImportWanted() {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	return map.open && !map.imported && !map.unknown.empty();
}

bool IsUnknownModuleSize(uint64_t words) {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	return map.unknown_sizes.contains(words);
}

bool IsUnknownModule(const Hash128& module) {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	return map.unknown.contains(module);
}

size_t UnknownModuleCount() {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	return map.unknown.size();
}

void NoteModuleFrom(uint64_t session_tag, const Hash128& stable, std::span<const uint32_t> spirv) {
	if (session_tag == 0) {
		return;
	}
	const auto      hash = HashWords(spirv);
	const auto      signature = InterfaceSignature(spirv);
	auto&           map  = Modules();
	std::lock_guard lock(map.mutex);
	if (map.open) {
		AddRecord(map, {session_tag, stable, hash, signature}, true);
	}
}

void MarkModuleImportDone() {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	if (map.open && !map.imported) {
		ModuleRecord marker;
		marker.module = {0x6d6f64696d706f72ull, 0x7420646f6e65ull}; // "modimport done"
		AddRecord(map, marker, true);
	}
}

bool AliasModule(const Hash128& old_stable, const Hash128& stable) {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	if (!map.open) {
		return false;
	}
	const auto found = map.current.find(stable);
	if (found == map.current.end()) {
		return false;
	}
	if (const auto old = map.current.find(old_stable);
	    old != map.current.end() && old->second.module == found->second.module) {
		return false;
	}
	AddRecord(map, {map.tag, old_stable, found->second.module, found->second.signature}, true);
	return true;
}

bool HasCurrentModule(const Hash128& stable) {
	auto&           map = Modules();
	std::lock_guard lock(map.mutex);
	return map.current.contains(stable);
}

RemapResult Remap(std::vector<uint8_t>& recipe) {
	RemapResult result;
	auto&       map = Modules();
	// version, flags, stage count; per stage: stage, kind, [hash], subgroup (SerializeWith).
	constexpr size_t header = 3 * sizeof(uint32_t);
	if (recipe.size() < header) {
		return result;
	}
	uint32_t stage_count = 0;
	std::memcpy(&stage_count, recipe.data() + 2 * sizeof(uint32_t), sizeof(stage_count));
	if (stage_count > 4) {
		return result;
	}
	std::vector<std::pair<size_t, Hash128>> renames;
	{
		std::lock_guard lock(map.mutex);
		if (!map.open) {
			return result;
		}
		size_t pos = header;
		for (uint32_t i = 0; i < stage_count; i++) {
			uint32_t kind = 0;
			if (pos + 2 * sizeof(uint32_t) > recipe.size()) {
				return {};
			}
			std::memcpy(&kind, recipe.data() + pos + sizeof(uint32_t), sizeof(kind));
			pos += 2 * sizeof(uint32_t);
			if (kind == 0) {
				if (pos + sizeof(Hash128) > recipe.size()) {
					return {};
				}
				Hash128 hash;
				std::memcpy(&hash, recipe.data() + pos, sizeof(hash));
				const auto known = map.by_module.find(hash);
				// v423b: a module recorded under several stable ids (the same SPIR-V for several
				// permutations, or AliasModule's old ids) is renamed when every one of them that has
				// a module in this session names the same module.
				const ModuleMap::Current* agreed = nullptr;
				bool                      disagree = false;
				if (known != map.by_module.end() && known->second.ambiguous) {
					for (const auto& stable: known->second.stables) {
						const auto now = map.current.find(stable);
						if (now == map.current.end()) {
							continue;
						}
						if (agreed == nullptr) {
							agreed = &now->second;
						} else if (!(agreed->module == now->second.module)) {
							disagree = true;
							break;
						}
					}
				}
				if (known == map.by_module.end() || (known->second.ambiguous && (disagree || agreed == nullptr))) {
					result.unknown++;
				} else if (known->second.ambiguous) {
					if (!(agreed->module == hash)) {
						if (agreed->signature != known->second.signature || known->second.signature == 0) {
							result.interface_differs = true;
							return result; // never half-renamed
						}
						renames.emplace_back(pos, agreed->module);
					}
				} else if (const auto now = map.current.find(known->second.stable); now != map.current.end() &&
				                                                                    !(now->second.module == hash)) {
					if (now->second.signature != known->second.signature || known->second.signature == 0) {
						result.interface_differs = true;
						return result; // never half-renamed
					}
					renames.emplace_back(pos, now->second.module);
				}
				pos += sizeof(Hash128);
			}
			pos += sizeof(uint32_t);
		}
	}
	for (const auto& [offset, hash]: renames) {
		std::memcpy(recipe.data() + offset, &hash, sizeof(hash));
	}
	result.changed = !renames.empty();
	return result;
}

namespace {
using ModuleResolver = bool (*)(void* context, vk::ShaderModule module, Hash128& hash);
bool SerializeWith(const GraphicsPipelineBuild& build, std::vector<uint8_t>& out, ModuleResolver resolve,
                   void* context);
} // namespace

bool Serialize(const GraphicsPipelineBuild& build, std::vector<uint8_t>& out) {
	return SerializeWith(
	    build, out, [](void*, vk::ShaderModule module, Hash128& hash) { return FindModuleHash(module, hash); },
	    nullptr);
}

namespace {
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool SerializeWith(const GraphicsPipelineBuild& build, std::vector<uint8_t>& out, ModuleResolver resolve,
                   void* context) {
	out.clear();
	Writer      w(out);
	const auto& info = build.pipeline_info;
	w.Put(RecipeVersion);
	w.Put(FlagBits(info.flags));
	w.Put(info.stageCount);
	for (uint32_t i = 0; i < info.stageCount; i++) {
		const auto& stage = build.stages[i];
		w.Put(FlagBits(vk::ShaderStageFlags(stage.stage)));
		if (build.tess_control != nullptr && stage.module == build.tess_control) {
			w.Put(uint32_t {1});
		} else if (build.tess_eval != nullptr && stage.module == build.tess_eval) {
			w.Put(uint32_t {2});
		} else {
			Hash128 hash;
			if (!resolve(context, stage.module, hash)) {
				return false;
			}
			w.Put(uint32_t {0});
			w.Put(hash);
		}
		w.Put(stage.pNext == &build.subgroup_sizes[i] ? build.subgroup_sizes[i].requiredSubgroupSize : 0u);
	}
	w.PutWords(build.rect_tcs_spirv);
	w.PutWords(build.rect_tes_spirv);

	w.Put(static_cast<uint8_t>(info.pVertexInputState != nullptr));
	if (info.pVertexInputState != nullptr) {
		const auto& vi = build.vertex_input_info;
		w.Put(vi.vertexBindingDescriptionCount);
		for (uint32_t i = 0; i < vi.vertexBindingDescriptionCount; i++) {
			const auto& b = build.input_desc[i];
			w.Put(b.binding);
			w.Put(b.stride);
			w.Put(static_cast<uint32_t>(b.inputRate));
		}
		w.Put(vi.vertexAttributeDescriptionCount);
		for (uint32_t i = 0; i < vi.vertexAttributeDescriptionCount; i++) {
			const auto& a = build.input_attr[i];
			w.Put(a.location);
			w.Put(a.binding);
			w.Put(static_cast<uint32_t>(a.format));
			w.Put(a.offset);
		}
	}
	w.Put(static_cast<uint8_t>(info.pInputAssemblyState != nullptr));
	if (info.pInputAssemblyState != nullptr) {
		w.Put(static_cast<uint32_t>(build.input_assembly.topology));
		w.Put(build.input_assembly.primitiveRestartEnable);
	}
	w.Put(static_cast<uint8_t>(info.pTessellationState != nullptr));
	if (info.pTessellationState != nullptr) {
		w.Put(build.tessellation.patchControlPoints);
	}
	w.Put(static_cast<uint8_t>(build.viewport_state.pNext == &build.depth_clip_control));
	w.Put(build.depth_clip_control.negativeOneToOne);

	const auto& r = build.rasterizer;
	w.Put(r.depthClampEnable);
	w.Put(static_cast<uint8_t>(ChainHas(r.pNext, &build.clip_ext)));
	w.Put(static_cast<uint8_t>(ChainHas(r.pNext, &build.provoking_vertex)));
	w.Put(build.clip_ext.depthClipEnable);
	w.Put(static_cast<uint32_t>(build.provoking_vertex.provokingVertexMode));
	w.Put(FlagBits(r.cullMode));
	w.Put(static_cast<uint32_t>(r.frontFace));
	w.Put(static_cast<uint32_t>(r.polygonMode));
	w.Put(FloatBits(r.lineWidth));
	w.Put(r.rasterizerDiscardEnable);
	w.Put(r.depthBiasEnable);

	const auto& ms = build.multisampling;
	w.Put(ms.sampleShadingEnable);
	w.Put(static_cast<uint32_t>(ms.rasterizationSamples));
	w.Put(FloatBits(ms.minSampleShading));

	const auto& cb = build.color_blending;
	w.Put(cb.attachmentCount);
	for (uint32_t i = 0; i < cb.attachmentCount; i++) {
		const auto& a = build.color_blend[i];
		w.Put(a.blendEnable);
		w.Put(static_cast<uint32_t>(a.srcColorBlendFactor));
		w.Put(static_cast<uint32_t>(a.dstColorBlendFactor));
		w.Put(static_cast<uint32_t>(a.colorBlendOp));
		w.Put(static_cast<uint32_t>(a.srcAlphaBlendFactor));
		w.Put(static_cast<uint32_t>(a.dstAlphaBlendFactor));
		w.Put(static_cast<uint32_t>(a.alphaBlendOp));
		w.Put(FlagBits(a.colorWriteMask));
	}
	w.Put(cb.logicOpEnable);
	w.Put(static_cast<uint32_t>(cb.logicOp));
	w.Put(static_cast<uint8_t>(cb.pNext == &build.color_write));
	w.Put(build.color_write.attachmentCount);
	for (uint32_t i = 0; i < build.color_write.attachmentCount; i++) {
		w.Put(build.color_write_enable[i]);
	}

	w.Put(static_cast<uint8_t>(info.pDepthStencilState != nullptr));
	if (info.pDepthStencilState != nullptr) {
		w.Put(build.depth_stencil.depthBoundsTestEnable);
		w.Put(FloatBits(build.depth_stencil.minDepthBounds));
		w.Put(FloatBits(build.depth_stencil.maxDepthBounds));
	}
	w.Put(static_cast<uint32_t>(build.dynamic_states.size()));
	for (const auto state: build.dynamic_states) {
		w.Put(static_cast<uint32_t>(state));
	}
	w.Put(build.rendering_info.colorAttachmentCount);
	for (uint32_t i = 0; i < build.rendering_info.colorAttachmentCount; i++) {
		w.Put(static_cast<uint32_t>(build.rendering.color_formats[i]));
	}
	w.Put(static_cast<uint32_t>(build.rendering_info.depthAttachmentFormat));
	w.Put(static_cast<uint32_t>(build.rendering_info.stencilAttachmentFormat));

	w.Put(static_cast<uint32_t>(build.descriptor_bindings.size()));
	for (const auto& binding: build.descriptor_bindings) {
		w.Put(binding.binding);
		w.Put(static_cast<uint32_t>(binding.descriptorType));
		w.Put(binding.descriptorCount);
		w.Put(FlagBits(binding.stageFlags));
	}
	w.Put(static_cast<uint8_t>(build.uses_bindless));
	w.Put(FlagBits(build.push_stages));
	w.Put(build.push_size);
	return true;
}
} // namespace

bool StagesCurrent(std::span<const uint8_t> recipe) {
	// version, flags, stage count; per stage: stage, kind, [hash], subgroup (SerializeWith).
	constexpr size_t header = 3 * sizeof(uint32_t);
	if (recipe.size() < header) {
		return false;
	}
	uint32_t stage_count = 0;
	std::memcpy(&stage_count, recipe.data() + 2 * sizeof(uint32_t), sizeof(stage_count));
	if (stage_count > 4) {
		return false;
	}
	std::vector<Hash128> hashes;
	size_t               pos = header;
	for (uint32_t i = 0; i < stage_count; i++) {
		uint32_t kind = 0;
		if (pos + 2 * sizeof(uint32_t) > recipe.size()) {
			return false;
		}
		std::memcpy(&kind, recipe.data() + pos + sizeof(uint32_t), sizeof(kind));
		pos += 2 * sizeof(uint32_t);
		if (kind == 0) {
			if (pos + sizeof(Hash128) > recipe.size()) {
				return false;
			}
			Hash128 hash;
			std::memcpy(&hash, recipe.data() + pos, sizeof(hash));
			hashes.push_back(hash);
			pos += sizeof(Hash128);
		}
		pos += sizeof(uint32_t);
	}
	{
		auto&           map = Modules();
		std::lock_guard lock(map.mutex);
		if (!map.open) {
			return true; // no map: nothing is known, nothing is left out
		}
		std::erase_if(hashes, [&](const Hash128& hash) { return map.current_modules.contains(hash); });
	}
	// Modules this session made outside the ledger's permutations (registered by draws).
	return std::all_of(hashes.begin(), hashes.end(), [](const Hash128& hash) { return FindModule(hash) != nullptr; });
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
vk::Result Build(GraphicContext& graphics, std::span<const uint8_t> recipe, vk::PipelineCache cache,
                 vk::Pipeline* pipeline) {
	auto  owner = std::make_unique<GraphicsPipelineBuild>();
	auto& b     = *owner;
	b.device    = graphics.device;
	Reader   r(recipe);
	uint32_t version = 0, flags = 0, stage_count = 0;
	r.Get(version);
	r.Get(flags);
	r.Get(stage_count);
	if (!r.Ok() || version != RecipeVersion || stage_count > b.stages.size()) {
		return vk::Result::eErrorUnknown;
	}
	struct StageRecord {
		uint32_t stage = 0, kind = 0, subgroup = 0;
		Hash128  hash;
	};
	std::array<StageRecord, 4> stage_records {};
	for (uint32_t i = 0; i < stage_count; i++) {
		auto& s = stage_records[i];
		r.Get(s.stage);
		r.Get(s.kind);
		if (s.kind == 0) {
			r.Get(s.hash);
		}
		r.Get(s.subgroup);
	}
	r.GetWords(b.rect_tcs_spirv);
	r.GetWords(b.rect_tes_spirv);
	if (!r.Ok()) {
		return vk::Result::eErrorUnknown;
	}
	// Stage modules not in this session are made from the SPIR-V store for the build only.
	struct TempModules {
		vk::Device                    device;
		std::vector<vk::ShaderModule> modules;
		~TempModules() {
			for (auto module: modules) {
				device.destroyShaderModule(module, nullptr);
			}
		}
		std::vector<std::pair<vk::ShaderModule, Hash128>> hashes;
	} temps {graphics.device, {}, {}};
	for (uint32_t i = 0; i < stage_count; i++) {
		const auto& s = stage_records[i];
		vk::ShaderModule module;
		if (s.kind == 0) {
			module = FindModule(s.hash);
			if (module == nullptr) {
				std::vector<uint32_t> words;
				if (LoadSpirv(s.hash, words)) {
					module = CompileSPV(words, graphics.device);
					temps.modules.push_back(module);
					temps.hashes.emplace_back(module, s.hash);
				}
			}
		} else if (s.kind == 1) {
			if (b.tess_control == nullptr && !b.rect_tcs_spirv.empty()) {
				b.tess_control = CompileSPV(b.rect_tcs_spirv, graphics.device);
			}
			module = b.tess_control;
		} else if (s.kind == 2) {
			if (b.tess_eval == nullptr && !b.rect_tes_spirv.empty()) {
				b.tess_eval = CompileSPV(b.rect_tes_spirv, graphics.device);
			}
			module = b.tess_eval;
		}
		if (module == nullptr) {
			return vk::Result::eErrorUnknown; // not in this session
		}
		b.stages[i] = {.stage  = static_cast<vk::ShaderStageFlagBits>(s.stage),
		               .module = module,
		               .pName  = "main"};
		if (s.subgroup != 0) {
			b.subgroup_sizes[i].requiredSubgroupSize = s.subgroup;
			b.stages[i].pNext                        = &b.subgroup_sizes[i];
		}
	}
	auto& info      = b.pipeline_info;
	info.flags      = vk::PipelineCreateFlags(flags);
	info.stageCount = stage_count;
	info.pStages    = b.stages.data();

	uint8_t has = 0;
	r.Get(has);
	if (has != 0) {
		uint32_t count = 0;
		r.Get(count);
		if (count > b.input_desc.size()) {
			return vk::Result::eErrorUnknown;
		}
		for (uint32_t i = 0; i < count; i++) {
			uint32_t rate = 0;
			r.Get(b.input_desc[i].binding);
			r.Get(b.input_desc[i].stride);
			r.Get(rate);
			b.input_desc[i].inputRate = static_cast<vk::VertexInputRate>(rate);
		}
		b.vertex_input_info.vertexBindingDescriptionCount = count;
		b.vertex_input_info.pVertexBindingDescriptions    = b.input_desc.data();
		r.Get(count);
		if (count > b.input_attr.size()) {
			return vk::Result::eErrorUnknown;
		}
		for (uint32_t i = 0; i < count; i++) {
			uint32_t format = 0;
			r.Get(b.input_attr[i].location);
			r.Get(b.input_attr[i].binding);
			r.Get(format);
			r.Get(b.input_attr[i].offset);
			b.input_attr[i].format = static_cast<vk::Format>(format);
		}
		b.vertex_input_info.vertexAttributeDescriptionCount = count;
		b.vertex_input_info.pVertexAttributeDescriptions    = b.input_attr.data();
		info.pVertexInputState                              = &b.vertex_input_info;
	}
	r.Get(has);
	if (has != 0) {
		uint32_t topology = 0;
		r.Get(topology);
		r.Get(b.input_assembly.primitiveRestartEnable);
		b.input_assembly.topology = static_cast<vk::PrimitiveTopology>(topology);
		info.pInputAssemblyState  = &b.input_assembly;
	}
	r.Get(has);
	if (has != 0) {
		r.Get(b.tessellation.patchControlPoints);
		info.pTessellationState = &b.tessellation;
	}
	r.Get(has);
	r.Get(b.depth_clip_control.negativeOneToOne);
	if (has != 0) {
		b.viewport_state.pNext = &b.depth_clip_control;
	}
	info.pViewportState = &b.viewport_state;

	auto&    rs        = b.rasterizer;
	uint8_t  with_clip = 0, with_provoking = 0;
	uint32_t value     = 0;
	r.Get(rs.depthClampEnable);
	r.Get(with_clip);
	r.Get(with_provoking);
	r.Get(b.clip_ext.depthClipEnable);
	r.Get(value);
	b.provoking_vertex.provokingVertexMode = static_cast<vk::ProvokingVertexModeEXT>(value);
	if (with_clip != 0) {
		rs.pNext = &b.clip_ext;
	}
	if (with_provoking != 0) {
		b.provoking_vertex.pNext = rs.pNext;
		rs.pNext                 = &b.provoking_vertex;
	}
	r.Get(value);
	rs.cullMode = vk::CullModeFlags(value);
	r.Get(value);
	rs.frontFace = static_cast<vk::FrontFace>(value);
	r.Get(value);
	rs.polygonMode = static_cast<vk::PolygonMode>(value);
	r.Get(value);
	rs.lineWidth = BitsFloat(value);
	r.Get(rs.rasterizerDiscardEnable);
	r.Get(rs.depthBiasEnable);
	info.pRasterizationState = &rs;

	auto& ms = b.multisampling;
	r.Get(ms.sampleShadingEnable);
	r.Get(value);
	ms.rasterizationSamples = static_cast<vk::SampleCountFlagBits>(value);
	r.Get(value);
	ms.minSampleShading    = BitsFloat(value);
	info.pMultisampleState = &ms;

	auto&    cb    = b.color_blending;
	uint32_t count = 0;
	r.Get(count);
	if (count > b.color_blend.size()) {
		return vk::Result::eErrorUnknown;
	}
	for (uint32_t i = 0; i < count; i++) {
		auto& a = b.color_blend[i];
		r.Get(a.blendEnable);
		r.Get(value);
		a.srcColorBlendFactor = static_cast<vk::BlendFactor>(value);
		r.Get(value);
		a.dstColorBlendFactor = static_cast<vk::BlendFactor>(value);
		r.Get(value);
		a.colorBlendOp = static_cast<vk::BlendOp>(value);
		r.Get(value);
		a.srcAlphaBlendFactor = static_cast<vk::BlendFactor>(value);
		r.Get(value);
		a.dstAlphaBlendFactor = static_cast<vk::BlendFactor>(value);
		r.Get(value);
		a.alphaBlendOp = static_cast<vk::BlendOp>(value);
		r.Get(value);
		a.colorWriteMask = vk::ColorComponentFlags(value);
	}
	cb.attachmentCount = count;
	cb.pAttachments    = b.color_blend.data();
	r.Get(cb.logicOpEnable);
	r.Get(value);
	cb.logicOp = static_cast<vk::LogicOp>(value);
	r.Get(has);
	r.Get(count);
	if (count > b.color_write_enable.size()) {
		return vk::Result::eErrorUnknown;
	}
	for (uint32_t i = 0; i < count; i++) {
		r.Get(b.color_write_enable[i]);
	}
	b.color_write.attachmentCount    = count;
	b.color_write.pColorWriteEnables = b.color_write_enable.data();
	if (has != 0) {
		cb.pNext = &b.color_write;
	}
	info.pColorBlendState = &cb;

	r.Get(has);
	if (has != 0) {
		r.Get(b.depth_stencil.depthBoundsTestEnable);
		r.Get(value);
		b.depth_stencil.minDepthBounds = BitsFloat(value);
		r.Get(value);
		b.depth_stencil.maxDepthBounds = BitsFloat(value);
		info.pDepthStencilState        = &b.depth_stencil;
	}
	r.Get(count);
	if (!r.Ok() || count > 64) {
		return vk::Result::eErrorUnknown;
	}
	for (uint32_t i = 0; i < count; i++) {
		r.Get(value);
		b.dynamic_states.push_back(static_cast<vk::DynamicState>(value));
	}
	b.dynamic_state.dynamicStateCount = count;
	b.dynamic_state.pDynamicStates    = b.dynamic_states.data();
	info.pDynamicState                = &b.dynamic_state;

	r.Get(count);
	if (count > b.rendering.color_formats.size()) {
		return vk::Result::eErrorUnknown;
	}
	for (uint32_t i = 0; i < count; i++) {
		r.Get(value);
		b.rendering.color_formats[i] = static_cast<vk::Format>(value);
	}
	b.rendering.color_count = count;
	r.Get(value);
	b.rendering.depth_format = static_cast<vk::Format>(value);
	r.Get(value);
	b.rendering.stencil_format                = static_cast<vk::Format>(value);
	b.rendering_info.colorAttachmentCount    = count;
	b.rendering_info.pColorAttachmentFormats = b.rendering.color_formats.data();
	b.rendering_info.depthAttachmentFormat   = b.rendering.depth_format;
	b.rendering_info.stencilAttachmentFormat = b.rendering.stencil_format;
	info.pNext                               = &b.rendering_info;

	r.Get(count);
	if (!r.Ok() || count > 4096) {
		return vk::Result::eErrorUnknown;
	}
	b.descriptor_bindings.resize(count);
	for (auto& binding: b.descriptor_bindings) {
		r.Get(binding.binding);
		r.Get(value);
		binding.descriptorType = static_cast<vk::DescriptorType>(value);
		r.Get(binding.descriptorCount);
		r.Get(value);
		binding.stageFlags = vk::ShaderStageFlags(value);
	}
	uint8_t bindless = 0;
	r.Get(bindless);
	b.uses_bindless = bindless != 0;
	r.Get(value);
	b.push_stages = vk::ShaderStageFlags(value);
	r.Get(b.push_size);
	if (!r.Ok() || !r.AtEnd()) {
		return vk::Result::eErrorUnknown;
	}
	// The rebuilt create info must say exactly what was recorded.
	std::vector<uint8_t> check;
	const bool           serialized = SerializeWith(
        b, check,
        [](void* context, vk::ShaderModule module, Hash128& hash) {
            for (const auto& [temp, temp_hash]: static_cast<TempModules*>(context)->hashes) {
                if (temp == module) {
                    hash = temp_hash;
                    return true;
                }
            }
            return FindModuleHash(module, hash);
        },
        &temps);
	if (!serialized || check.size() != recipe.size() ||
	    std::memcmp(check.data(), recipe.data(), check.size()) != 0) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 4) {
			LOGF("Pipeline recipes: a recipe did not rebuild to itself (%zu vs %zu bytes); skipped\n",
			     check.size(), recipe.size());
		}
		return vk::Result::eErrorUnknown;
	}

	// The layouts, as PrepareGraphicsPipeline makes them (CreateDescriptorLayout, FillSetLayouts;
	// KYTY_STAGE_SETS: the pre-rasterization set 0, set 1, the pixel shader's set 2).
	if (b.uses_bindless && graphics.bindless_layout == nullptr) {
		return vk::Result::eErrorUnknown;
	}
	// As PrepareGraphicsPipeline: split whenever the pipeline has a pixel shader.
	const bool stage_sets = ShaderRecompiler::IR::StageDescriptorSets() &&
	                        std::ranges::any_of(std::span(b.stages.data(), info.stageCount), [](const auto& stage) {
		                        return stage.stage == vk::ShaderStageFlagBits::eFragment;
	                        });
	std::vector<vk::DescriptorSetLayout> owned;
	const auto make_set = [&](std::span<const vk::DescriptorSetLayoutBinding> bindings, bool push) {
		vk::DescriptorSetLayoutCreateInfo set_info {};
		set_info.flags        = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
		                             : vk::DescriptorSetLayoutCreateFlags {};
		set_info.bindingCount = static_cast<uint32_t>(bindings.size());
		set_info.pBindings    = bindings.data();
		vk::DescriptorSetLayout set_layout;
		if (graphics.device.createDescriptorSetLayout(&set_info, nullptr, &set_layout) != vk::Result::eSuccess) {
			return vk::DescriptorSetLayout {};
		}
		owned.push_back(set_layout);
		return set_layout;
	};
	const auto destroy_sets = [&]() {
		for (const auto set_layout: owned) {
			graphics.device.destroyDescriptorSetLayout(set_layout, nullptr);
		}
	};
	std::array<vk::DescriptorSetLayout, 3> set_layouts {};
	uint32_t                               set_count = 0;
	if (stage_sets) {
		std::vector<vk::DescriptorSetLayoutBinding> pre;
		std::vector<vk::DescriptorSetLayoutBinding> ps;
		bool                                        pre_push = false;
		bool                                        ps_push  = false;
		SplitStageSets(graphics.max_push_descriptors, b.descriptor_bindings, pre, ps, pre_push, ps_push);
		set_layouts = {make_set(pre, pre_push),
		               b.uses_bindless ? graphics.bindless_layout : EmptySetLayout(graphics),
		               make_set(ps, ps_push)};
		set_count   = 3;
	} else {
		uint32_t descriptor_count = 0;
		for (const auto& binding: b.descriptor_bindings) {
			descriptor_count += binding.descriptorCount;
		}
		set_layouts[0] = make_set(b.descriptor_bindings, descriptor_count <= graphics.max_push_descriptors);
		set_layouts[1] = graphics.bindless_layout;
		set_count      = b.uses_bindless ? 2u : 1u;
	}
	if (std::ranges::any_of(std::span(set_layouts.data(), set_count), [](auto l) { return !l; })) {
		destroy_sets();
		return vk::Result::eErrorUnknown;
	}
	const vk::PushConstantRange  push {b.push_stages, 0, b.push_size};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = set_count;
	layout_info.pSetLayouts            = set_layouts.data();
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	vk::PipelineLayout layout;
	if (graphics.device.createPipelineLayout(&layout_info, nullptr, &layout) != vk::Result::eSuccess) {
		destroy_sets();
		return vk::Result::eErrorUnknown;
	}
	info.layout            = layout;
	info.basePipelineIndex = -1;
	info.flags |= PipelineStats::CaptureFlags(); // KYTY_PIPE_STATS, reported when a draw takes it
	const auto result      = graphics.device.createGraphicsPipelines(cache, 1, &info, nullptr, pipeline);
	graphics.device.destroyPipelineLayout(layout, nullptr);
	destroy_sets();
	return result;
}

void Store::Open(const std::filesystem::path& path) {
	std::lock_guard lock(m_mutex);
	m_path = path;
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	uint64_t valid_end = 0;
	if (std::FILE* in = std::fopen(path.string().c_str(), "rb"); in != nullptr) {
		char magic[8] {};
		if (std::fread(magic, 1, sizeof(magic), in) == sizeof(magic) &&
		    std::memcmp(magic, FileMagic, sizeof(magic)) == 0) {
			valid_end = sizeof(magic);
			for (;;) {
				uint32_t size = 0;
				Hash128  hash;
				if (std::fread(&size, sizeof(size), 1, in) != 1 || std::fread(&hash, sizeof(hash), 1, in) != 1 ||
				    size == 0 || size > (1u << 24u)) {
					break;
				}
				std::vector<uint8_t> bytes(size);
				if (std::fread(bytes.data(), 1, size, in) != size || !(HashBytes(bytes) == hash)) {
					break;
				}
				valid_end += sizeof(size) + sizeof(hash) + size;
				if (m_known.insert(hash).second) {
					m_loaded.push_back({hash, std::move(bytes)});
				}
			}
		}
		std::fclose(in);
	}
	if (valid_end == 0) {
		m_file = std::fopen(path.string().c_str(), "wb");
		if (m_file != nullptr) {
			std::fwrite(FileMagic, 1, sizeof(FileMagic), m_file);
			std::fflush(m_file);
		}
	} else {
		// A torn record at the end (a crash mid-append) is cut off.
		std::filesystem::resize_file(path, valid_end, error);
		m_file = std::fopen(path.string().c_str(), "ab");
	}
	LOGF("Pipeline recipes: %zu in %s\n", m_loaded.size(), path.string().c_str());
}

bool Store::Record(const Hash128& hash, std::span<const uint8_t> recipe, bool counted) {
	std::lock_guard lock(m_mutex);
	if (!m_known.insert(hash).second) {
		return false;
	}
	if (m_file != nullptr) {
		const auto size = static_cast<uint32_t>(recipe.size());
		std::fwrite(&size, sizeof(size), 1, m_file);
		std::fwrite(&hash, sizeof(hash), 1, m_file);
		std::fwrite(recipe.data(), 1, recipe.size(), m_file);
		std::fflush(m_file);
	}
	if (counted) {
		GetShaderPrecompileProgress().pipelines_new.fetch_add(1, std::memory_order_relaxed);
	}
	return true;
}

bool Store::Known(const Hash128& hash) {
	std::lock_guard lock(m_mutex);
	return m_known.contains(hash);
}

std::vector<Store::Entry> Store::TakeLoaded() {
	std::lock_guard lock(m_mutex);
	return std::move(m_loaded);
}

size_t Store::Size() {
	std::lock_guard lock(m_mutex);
	return m_known.size();
}

Store::~Store() {
	if (m_file != nullptr) {
		std::fclose(m_file);
	}
}

} // namespace Libs::Graphics::PipelineRecipe
