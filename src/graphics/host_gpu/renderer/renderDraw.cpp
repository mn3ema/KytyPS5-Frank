#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/host_gpu/renderer/drawPipe.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/shaderBindings.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "common/tracePoints.h"
#include "graphics/host_gpu/renderer/reuseProbe.h"

#include "graphics/host_gpu/renderer/image/imageHistory.h"
#include "graphics/host_gpu/timeline.h"
#include "common/fpsLog.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/imgRef.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/barrierSink.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/dispatchRun.h"
#include "graphics/host_gpu/renderer/indirectDispatch.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "graphics/host_gpu/renderer/htileWatch.h"
#include "graphics/host_gpu/renderer/presentRt.h"
#include "graphics/host_gpu/renderer/rtDrawLog.h"
#include "graphics/host_gpu/renderer/imgTrace.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/rtSnap.h"
#include "graphics/host_gpu/renderer/drawStats.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/paintDraws.h"
#include "graphics/host_gpu/renderer/renderStats.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

static bool EnvSwitchOn(const char* name); // below, with ReplayDiagnosticsActive

namespace {

// Diagnostics: KYTY_MESH_WALK=1 with KYTY_LOG_PS_DRAWS walks the title's generic meshlet mesh
// shader's data path on the CPU for each logged mesh draw and prints what the shader will read:
// the entity record (user data 0..3, one 16-byte record per draw index), the mesh record it names
// (64 bytes in the ring slot 0x620 table), the meshlet descriptors (ring slot 0x5a0 data), the
// first meshlet's indices and vertices, the instance transform (ring slot 0x660) and the camera
// position, plus whether the buffer cache holds those bytes as GPU-written. A mesh whose
// triangles rasterize nothing (KYTY_DRAW_STATS frag=0) shows here whether its data is missing,
// stale or read from the wrong place.
struct WalkDescriptor {
	uint64_t base    = 0;
	uint32_t stride  = 0;
	uint32_t records = 0;
	uint32_t dword3  = 0;
	[[nodiscard]] uint64_t Bytes() const noexcept {
		return stride == 0 ? records : static_cast<uint64_t>(records) * stride;
	}
	[[nodiscard]] bool Readable(uint64_t offset, uint64_t size) const noexcept {
		return GuestRange {base + offset, size}.Valid();
	}
};

WalkDescriptor ReadWalkDescriptor(const uint32_t* words) {
	return {words[0] | (static_cast<uint64_t>(words[1] & 0xffffu) << 32u), (words[1] >> 16u) & 0x3fffu,
	        words[2], words[3]};
}

uint32_t WalkDword(const WalkDescriptor& d, uint64_t offset) {
	if (!d.Readable(offset, 4)) {
		return 0xdeadbeefu;
	}
	uint32_t value = 0;
	std::memcpy(&value, reinterpret_cast<const void*>(d.base + offset), 4);
	return value;
}

std::string WalkDwords(const WalkDescriptor& d, uint64_t offset, uint32_t count) {
	std::string text;
	for (uint32_t i = 0; i < count; i++) {
		text += fmt::format("{}{:08x}", i == 0 ? "" : " ", WalkDword(d, offset + i * 4u));
	}
	return text;
}

std::string WalkFormat(const WalkDescriptor& d) {
	return fmt::format("{:x}+{:x}(stride {} records {} dw3 {:08x})", d.base, d.Bytes(), d.stride,
	                   d.records, d.dword3);
}

const char* WalkModified(BufferCache& cache, uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return "invalid";
	}
	const bool gpu = cache.IsRegionGpuModified(address, size);
	const bool cpu = cache.IsRegionCpuModified(address, size);
	return gpu && cpu ? "gpu+cpu" : gpu ? "gpu" : cpu ? "cpu" : "clean";
}

// The GPU's copy of the bytes (the cached buffer's, which the BDA page table points into),
// against guest memory: "gpu=same" or the differing dwords.
std::string GpuCopyText(BufferCache& cache, uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid() || size == 0 || size % 4 != 0) {
		return {};
	}
	std::vector<uint32_t> gpu;
	if (!cache.PeekGpuCopy(address, size, gpu)) {
		return " gpu=no-buffer";
	}
	std::vector<uint32_t> cpu(gpu.size());
	std::memcpy(cpu.data(), reinterpret_cast<const void*>(address), size);
	if (cpu == gpu) {
		return " gpu=same";
	}
	std::string text = " gpu=DIFF:";
	for (size_t i = 0; i < gpu.size(); i++) {
		text += fmt::format(" {:08x}", gpu[i]);
	}
	return text;
}

// KYTY_MESH_WALK=2: GpuCopyText for the mesh walk.
std::string WalkGpuCopy(BufferCache& cache, uint64_t address, uint64_t size) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MESH_WALK");
		return value != nullptr && value[0] == '2';
	}();
	return enabled ? GpuCopyText(cache, address, size) : std::string();
}

// KYTY_SLOT_DUMP=<slot>:<offset>:<dwords>[:<record>:<dwords>][+...] (all hex, slot = NGG
// user-data slot of a V#, 0 = s8): with KYTY_LOG_PS_DRAWS, every logged draw also prints its
// vertex stage's user data and, per entry, the V# at that slot with the dwords at <offset> of its
// memory (guest copy, and the buffer cache's GPU copy when it differs). With the optional pair,
// the four dwords at <offset> are themselves a V# (a descriptor the shader loads from a table),
// and the dwords at <record> × its stride of that buffer are printed too.
struct SlotDump {
	uint32_t slot            = 0;
	uint32_t offset          = 0;
	uint32_t dwords          = 0;
	uint32_t indirect_record = 0;
	uint32_t indirect_dwords = 0;
	// <record> as @<slot>:<offset>: the record index is the dword at <offset> of that slot's buffer.
	int32_t  record_slot   = -1;
	uint32_t record_offset = 0;
};

const std::vector<SlotDump>& SlotDumps() {
	static const std::vector<SlotDump> dumps = [] {
		std::vector<SlotDump> result;
		const char*           value = std::getenv("KYTY_SLOT_DUMP");
		if (value == nullptr) {
			return result;
		}
		const char* p = value;
		while (*p != '\0') {
			char*    end = nullptr;
			SlotDump d;
			d.slot = static_cast<uint32_t>(std::strtoul(p, &end, 16));
			if (end == nullptr || *end != ':') break;
			d.offset = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			if (end == nullptr || *end != ':') break;
			d.dwords = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			if (end != nullptr && *end == ':') {
				if (end[1] == '@') {
					d.record_slot = static_cast<int32_t>(std::strtoul(end + 2, &end, 16));
					if (end == nullptr || *end != ':') break;
					d.record_offset = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
				} else {
					d.indirect_record = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
				}
				if (end == nullptr || *end != ':') break;
				d.indirect_dwords = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			}
			result.push_back(d);
			if (end == nullptr || (*end != '+' && *end != ',')) break;
			p = end + 1;
		}
		return result;
	}();
	return dumps;
}

void LogSlotDumps(uint64_t flip, const uint32_t* users, uint32_t user_count, BufferCache& cache) {
	std::string ud;
	for (uint32_t i = 0; i < user_count; i++) {
		ud += fmt::format(" {:08x}", users[i]);
	}
	std::fprintf(stderr, "[ps-draw-vs-ud] flip=%llu n=%u%s\n", static_cast<unsigned long long>(flip),
	             user_count, ud.c_str());
	for (const auto& d: SlotDumps()) {
		if (d.slot + 3u >= user_count) {
			std::fprintf(stderr, "[slot-dump] flip=%llu slot=%x beyond %u user slots\n",
			             static_cast<unsigned long long>(flip), d.slot, user_count);
			continue;
		}
		const auto     desc  = ReadWalkDescriptor(users + d.slot);
		const uint64_t bytes = static_cast<uint64_t>(d.dwords) * 4u;
		std::fprintf(stderr, "[slot-dump] flip=%llu slot=%x off=%x v#=%s |%s%s\n",
		             static_cast<unsigned long long>(flip), d.slot, d.offset,
		             WalkFormat(desc).c_str(), WalkDwords(desc, d.offset, d.dwords).c_str(),
		             GpuCopyText(cache, desc.base + d.offset, bytes).c_str());
		if (d.indirect_dwords == 0) {
			continue;
		}
		uint32_t words[4];
		for (uint32_t i = 0; i < 4u; i++) {
			words[i] = WalkDword(desc, d.offset + i * 4u);
		}
		const auto inner  = ReadWalkDescriptor(words);
		uint32_t   record = d.indirect_record;
		if (d.record_slot >= 0 && static_cast<uint32_t>(d.record_slot) + 3u < user_count) {
			record = WalkDword(ReadWalkDescriptor(users + d.record_slot), d.record_offset);
		}
		const uint64_t inner_offset =
		    static_cast<uint64_t>(record) * (inner.stride == 0 ? 1u : inner.stride);
		const uint64_t inner_bytes = static_cast<uint64_t>(d.indirect_dwords) * 4u;
		std::fprintf(stderr, "[slot-dump] flip=%llu slot=%x off=%x -> v#=%s rec=%x off=%llx |%s%s\n",
		             static_cast<unsigned long long>(flip), d.slot, d.offset,
		             WalkFormat(inner).c_str(), record,
		             static_cast<unsigned long long>(inner_offset),
		             WalkDwords(inner, inner_offset, d.indirect_dwords).c_str(),
		             GpuCopyText(cache, inner.base + inner_offset, inner_bytes).c_str());
	}
}

// KYTY_MESH_WALK=2: the GPU's BDA page table entry for the page holding address, against the
// device address of the cached buffer that should be there.
std::string WalkBdaEntry(BufferCache& cache, uint64_t address) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MESH_WALK");
		return value != nullptr && value[0] == '2';
	}();
	if (!enabled) {
		return {};
	}
	uint64_t entry = 0, expected = 0;
	if (!cache.PeekBdaEntry(address, entry, expected)) {
		return " bda=none";
	}
	return entry == expected ? fmt::format(" bda=ok({:x})", entry)
	                         : fmt::format(" bda=MISMATCH(table {:x} buffer {:x})", entry, expected);
}

bool MeshWalkEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MESH_WALK");
		return value != nullptr && (value[0] == '1' || value[0] == '2');
	}();
	return enabled;
}

// KYTY_MESH_NEAR=<x>:<y>:<z>:<radius> (diagnostics, floats): every meshlet draw (generic meshlet
// shader layout, see LogMeshWalk) whose instance translation lies within <radius> of the point is
// logged as "[mesh-near]" with its shaders, counts, the instance's 3x4 transform and the mesh record
// (every 100th guest flip, 400 lines at most). Finds the draws of objects at a known world position.
void LogMeshNear(uint64_t flip, const uint32_t* users, uint32_t user_count, uint32_t count,
                 uint32_t first_instance, uint64_t vs_hash, uint64_t ps_hash) {
	static const std::vector<float> near_point = [] {
		std::vector<float> v;
		if (const char* value = std::getenv("KYTY_MESH_NEAR"); value != nullptr) {
			std::string text(value);
			size_t      start = 0;
			while (start <= text.size() && v.size() < 4) {
				const auto end = text.find(':', start);
				v.push_back(std::strtof(text.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str(), nullptr));
				if (end == std::string::npos) break;
				start = end + 1;
			}
		}
		return v;
	}();
	if (near_point.size() < 4 || user_count < 21 || (flip % 100u) != 0) { // every 100th guest flip
		return;
	}
	const auto entities   = ReadWalkDescriptor(users + 0);
	const auto ring       = ReadWalkDescriptor(users + 16);
	const auto draw_index = users[20];
	const auto slot       = [&](uint32_t offset) {
		uint32_t words[4] = {};
		for (uint32_t i = 0; i < 4; i++) {
			words[i] = WalkDword(ring, offset + i * 4u);
		}
		return ReadWalkDescriptor(words);
	};
	const auto     meshes    = slot(0x620);
	const auto     instances = slot(0x660);
	const uint64_t eoff      = static_cast<uint64_t>(draw_index) * 16u;
	const uint32_t e1        = WalkDword(entities, eoff + 4);
	const uint32_t e2        = WalkDword(entities, eoff + 8);
	const uint64_t toff      = static_cast<uint64_t>(e1 & 0xffffffu) * 128u;
	float          row[12]   = {};
	for (uint32_t i = 0; i < 12; i++) {
		const uint32_t bits = WalkDword(instances, toff + i * 4u);
		std::memcpy(&row[i], &bits, 4);
	}
	const float dx = row[3] - near_point[0];
	const float dy = row[7] - near_point[1];
	const float dz = row[11] - near_point[2];
	if (!(dx * dx + dy * dy + dz * dz <= near_point[3] * near_point[3])) {
		return;
	}
	static std::atomic<uint64_t> last_flip {0};
	static std::atomic<uint32_t> lines {0};
	if (last_flip.exchange(flip) != flip) lines.store(0);
	if (lines.fetch_add(1) >= 400) {
		return;
	}
	const uint64_t moff = (static_cast<uint64_t>(e2) << 6u) & 0xffffc0u;
	std::fprintf(stderr,
	             "[mesh-near] flip=%llu vs=%016llx ps=%016llx count=%u first_inst=%u draw_index=%u "
	             "xf=[%g %g %g %g] [%g %g %g %g] [%g %g %g %g] mesh: %s\n",
	             static_cast<unsigned long long>(flip), static_cast<unsigned long long>(vs_hash),
	             static_cast<unsigned long long>(ps_hash), count, first_instance, draw_index, row[0], row[1],
	             row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
	             WalkDwords(meshes, moff, 16).c_str());
}

void LogMeshWalk(uint64_t flip, const uint32_t* users, uint32_t user_count, uint32_t count,
                 uint32_t first_instance, BufferCache& cache) {
	if (user_count < 21) {
		std::fprintf(stderr, "[mesh-walk] flip=%llu only %u user data slots\n",
		             static_cast<unsigned long long>(flip), user_count);
		return;
	}
	const auto entities   = ReadWalkDescriptor(users + 0);  // s[8:11]
	const auto view_cb    = ReadWalkDescriptor(users + 4);  // s[12:15]
	const auto misc_cb    = ReadWalkDescriptor(users + 8);  // s[16:19]
	const auto control    = ReadWalkDescriptor(users + 12); // s[20:23]
	const auto ring       = ReadWalkDescriptor(users + 16); // s[24:27]
	const auto draw_index = users[20];                      // s28
	std::fprintf(stderr,
	             "[mesh-walk] flip=%llu count=%u first_inst=%u draw_index=%u entities=%s control=%s "
	             "ring=%s view=%s misc=%s\n",
	             static_cast<unsigned long long>(flip), count, first_instance, draw_index,
	             WalkFormat(entities).c_str(), WalkFormat(control).c_str(), WalkFormat(ring).c_str(),
	             WalkFormat(view_cb).c_str(), WalkFormat(misc_cb).c_str());
	std::fprintf(stderr, "[mesh-walk]   control: %s | camera(view+0xf0): %s\n",
	             WalkDwords(control, 0, 4).c_str(), WalkDwords(view_cb, 0xf0, 4).c_str());
	// The ring slots the shader loads descriptors from.
	struct Slot {
		uint32_t    offset;
		const char* name;
	};
	static constexpr Slot slots[] = {{0x540, "visible"}, {0x620, "meshes"},  {0x5a0, "data"},
	                                 {0x2e0, "stream0"}, {0x2c0, "stream1"}, {0x580, "attrA"},
	                                 {0x2a0, "attrB"},   {0x4e0, "prev"},    {0x660, "instances"},
	                                 {0x640, "materials"}};
	std::string slot_text;
	for (const auto& slot: slots) {
		uint32_t words[4] = {};
		for (uint32_t i = 0; i < 4; i++) {
			words[i] = WalkDword(ring, slot.offset + i * 4u);
		}
		slot_text += fmt::format(" {}={}", slot.name, WalkFormat(ReadWalkDescriptor(words)));
	}
	std::fprintf(stderr, "[mesh-walk]   ring slots:%s\n", slot_text.c_str());
	const auto slot = [&](uint32_t offset) {
		uint32_t words[4] = {};
		for (uint32_t i = 0; i < 4; i++) {
			words[i] = WalkDword(ring, offset + i * 4u);
		}
		return ReadWalkDescriptor(words);
	};
	const auto meshes    = slot(0x620);
	const auto data      = slot(0x5a0);
	const auto stream0   = slot(0x2e0);
	const auto instances = slot(0x660);
	// The entity record of this draw index (Y = 0: one instance).
	const uint64_t entity_offset = static_cast<uint64_t>(draw_index) * 16u;
	const uint32_t e0            = WalkDword(entities, entity_offset);
	const uint32_t e1            = WalkDword(entities, entity_offset + 4);
	const uint32_t e2            = WalkDword(entities, entity_offset + 8);
	const uint32_t e3            = WalkDword(entities, entity_offset + 12);
	const uint32_t transform     = e1 & 0xffffffu;
	const uint64_t mesh_offset   = (static_cast<uint64_t>(e2) << 6u) & 0xffffc0u;
	std::fprintf(stderr,
	             "[mesh-walk]   entity[%u] @%llx: %08x %08x %08x %08x (%s%s) -> mesh record %x "
	             "transform %u\n",
	             draw_index,
	             static_cast<unsigned long long>(entities.base + entity_offset), e0, e1, e2, e3,
	             WalkModified(cache, entities.base + entity_offset, 16),
	             WalkGpuCopy(cache, entities.base + entity_offset, 16).c_str(), e2, transform);
	std::fprintf(stderr, "[mesh-walk]   mesh @%llx (%s%s): %s\n",
	             static_cast<unsigned long long>(meshes.base + mesh_offset),
	             WalkModified(cache, meshes.base + mesh_offset, 64),
	             WalkGpuCopy(cache, meshes.base + mesh_offset, 64).c_str(),
	             WalkDwords(meshes, mesh_offset, 16).c_str());
	const uint32_t vertex_base   = WalkDword(meshes, mesh_offset + 0x00);
	const uint32_t table         = WalkDword(meshes, mesh_offset + 0x14);
	const uint32_t meshlets      = WalkDword(meshes, mesh_offset + 0x18);
	const uint32_t flags         = WalkDword(meshes, mesh_offset + 0x1c);
	const uint32_t off_x         = WalkDword(meshes, mesh_offset + 0x20);
	const uint32_t off_y         = WalkDword(meshes, mesh_offset + 0x24);
	const uint32_t off_z         = WalkDword(meshes, mesh_offset + 0x28);
	const uint32_t scale_bits    = WalkDword(meshes, mesh_offset + 0x3c);
	float          scale         = 0.0f;
	std::memcpy(&scale, &scale_bits, 4);
	std::fprintf(stderr,
	             "[mesh-walk]   mesh: vertex_base=%u table=0x%x meshlets=%u flags=%08x (layout %s, "
	             "offsets %d) offsets=%d,%d,%d scale=%g\n",
	             vertex_base, table, meshlets, flags, (flags & 1u) != 0 ? "streams" : "raw",
	             (flags & 16u) != 0 ? 1 : 0, static_cast<int32_t>(off_x << 16u) >> 8,
	             static_cast<int32_t>(off_y << 16u) >> 8, static_cast<int32_t>(off_z << 16u) >> 8,
	             scale);
	// The instance transform: three rows of a 3x4 matrix at 128 bytes per instance.
	{
		const uint64_t offset = static_cast<uint64_t>(transform) * 128u;
		float          row[12] = {};
		for (uint32_t i = 0; i < 12; i++) {
			const uint32_t bits = WalkDword(instances, offset + i * 4u);
			std::memcpy(&row[i], &bits, 4);
		}
		std::fprintf(stderr,
		             "[mesh-walk]   instance %u @%llx (%s%s): [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] "
		             "+0x30..: %s\n",
		             transform, static_cast<unsigned long long>(instances.base + offset),
		             WalkModified(cache, instances.base + offset, 128),
		             WalkGpuCopy(cache, instances.base + offset, 48).c_str(), row[0], row[1], row[2], row[3],
		             row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
		             WalkDwords(instances, offset + 0x30, 12).c_str());
	}
	// The meshlet descriptors and the first meshlet's indices and vertices.
	if (meshlets == 0 || meshlets > 4096 || table == 0xdeadbeefu) {
		std::fprintf(stderr, "[mesh-walk]   no meshlets to walk\n");
		return;
	}
	uint64_t    prims_total = 0;
	std::string meshlet_text;
	for (uint32_t m = 0; m < meshlets; m++) {
		const uint32_t desc   = WalkDword(data, table + static_cast<uint64_t>(m) * 4u);
		const uint32_t vcount = ((desc >> 17u) & 0x7fu) + 1u;
		const uint32_t pcount = (desc >> 24u) + 1u;
		prims_total += pcount;
		if (m < 6) {
			meshlet_text += fmt::format(" [{}: v{} p{} idx 0x{:x}]", m, vcount, pcount,
			                            (desc << 4u) & 0x1ffff0u);
		}
	}
	std::fprintf(stderr, "[mesh-walk]   meshlet table @%llx (%s%s%s): prims_total=%llu%s\n",
	             static_cast<unsigned long long>(data.base + table),
	             WalkModified(cache, data.base + table, static_cast<uint64_t>(meshlets) * 4u),
	             WalkGpuCopy(cache, data.base + table, std::min<uint64_t>(meshlets, 8u) * 4u).c_str(),
	             WalkBdaEntry(cache, data.base + table).c_str(),
	             static_cast<unsigned long long>(prims_total), meshlet_text.c_str());
	const uint32_t desc0      = WalkDword(data, table);
	const uint32_t vcount0    = ((desc0 >> 17u) & 0x7fu) + 1u;
	const uint32_t pcount0    = (desc0 >> 24u) + 1u;
	const uint64_t index_base = static_cast<uint64_t>(table) + static_cast<uint64_t>(meshlets) * 24u +
	                            ((desc0 << 4u) & 0x1ffff0u);
	std::string prim_text;
	std::string vertex_text;
	const auto  index16 = [&](uint64_t byte_offset) {
		const uint32_t word = WalkDword(data, byte_offset & ~3ull);
		return (word >> ((byte_offset & 3u) * 8u)) & 0xffffu;
	};
	for (uint32_t t = 0; t < std::min(pcount0, 4u); t++) {
		const uint32_t packed = index16(index_base + t * 2u);
		prim_text += fmt::format(" {}/{}/{}", packed & 31u, (packed >> 5u) & 31u, (packed >> 10u) & 31u);
	}
	std::fprintf(stderr,
	             "[mesh-walk]   meshlet 0: indices @%llx (%s%s) prims:%s\n",
	             static_cast<unsigned long long>(data.base + index_base),
	             WalkModified(cache, data.base + index_base, static_cast<uint64_t>(pcount0 + vcount0) * 2u),
	             WalkGpuCopy(cache, data.base + (index_base & ~3ull), 16).c_str(), prim_text.c_str());
	for (uint32_t t = 0; t < std::min(vcount0, 4u); t++) {
		const uint32_t index  = index16(index_base + static_cast<uint64_t>(pcount0) * 2u + t * 2u);
		const uint64_t vertex = static_cast<uint64_t>(vertex_base) + index;
		uint32_t       words[4] = {};
		uint64_t       address  = 0;
		if ((flags & 1u) != 0) {
			address = stream0.base + vertex * 2u * stream0.stride;
			for (uint32_t i = 0; i < 4; i++) {
				words[i] = WalkDword(stream0, vertex * 2u * stream0.stride + i * 4u);
			}
		} else {
			address = data.base + vertex * 16u;
			for (uint32_t i = 0; i < 4; i++) {
				words[i] = WalkDword(data, vertex * 16u + i * 4u);
			}
		}
		const auto     x   = static_cast<int32_t>(static_cast<int16_t>(words[0] & 0xffffu));
		const auto     y   = static_cast<int32_t>(static_cast<int16_t>(words[0] >> 16u));
		const auto     z   = static_cast<int32_t>(static_cast<int16_t>(words[1] & 0xffffu));
		const uint32_t w   = words[1] >> 16u;
		const uint32_t exp = (w >> 5u) & 0xfu;
		const bool     add = (flags & 16u) != 0;
		const float    px  = static_cast<float>((x << exp) + (add ? (static_cast<int32_t>(off_x << 16u) >> 8) : 0)) * scale;
		const float    py  = static_cast<float>((y << exp) + (add ? (static_cast<int32_t>(off_y << 16u) >> 8) : 0)) * scale;
		const float    pz  = static_cast<float>((z << exp) + (add ? (static_cast<int32_t>(off_z << 16u) >> 8) : 0)) * scale;
		vertex_text += fmt::format(" [v{} idx {} @{:x} ({}{}{}): {:08x} {:08x} {:08x} {:08x} -> {:g} {:g} {:g}]",
		                           t, index, address, WalkModified(cache, address, 16),
		                           WalkGpuCopy(cache, address, 16),
		                           t == 0 ? WalkBdaEntry(cache, address) : std::string(), words[0],
		                           words[1], words[2], words[3], px, py, pz);
	}
	std::fprintf(stderr, "[mesh-walk]   vertices:%s\n", vertex_text.c_str());
	// The whole mesh's extent in local and in camera-relative world space (every vertex of every
	// meshlet, as the shader decodes them): a mesh the game built collapsed shows up here.
	{
		float    lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
		float    wlo[3] = {1e30f, 1e30f, 1e30f}, whi[3] = {-1e30f, -1e30f, -1e30f};
		uint32_t total = 0, zero_words = 0;
		float    row[12] = {}, cam[3] = {};
		{
			const uint64_t offset = static_cast<uint64_t>(transform) * 128u;
			for (uint32_t i = 0; i < 12; i++) {
				const uint32_t bits = WalkDword(instances, offset + i * 4u);
				std::memcpy(&row[i], &bits, 4);
			}
			for (uint32_t i = 0; i < 3; i++) {
				const uint32_t bits = WalkDword(view_cb, 0xf0 + i * 4u);
				std::memcpy(&cam[i], &bits, 4);
			}
		}
		for (uint32_t m = 0; m < meshlets && m < 256; m++) {
			const uint32_t desc   = WalkDword(data, table + static_cast<uint64_t>(m) * 4u);
			const uint32_t vcount = ((desc >> 17u) & 0x7fu) + 1u;
			const uint32_t pcount = (desc >> 24u) + 1u;
			const uint64_t base   = static_cast<uint64_t>(table) + static_cast<uint64_t>(meshlets) * 24u +
			                      ((desc << 4u) & 0x1ffff0u) + static_cast<uint64_t>(pcount) * 2u;
			for (uint32_t t = 0; t < vcount; t++) {
				const uint32_t index  = index16(base + t * 2u);
				const uint64_t vertex = static_cast<uint64_t>(vertex_base) + index;
				uint32_t       words[2] = {};
				for (uint32_t i = 0; i < 2; i++) {
					words[i] = (flags & 1u) != 0
					               ? WalkDword(stream0, vertex * 2u * stream0.stride + i * 4u)
					               : WalkDword(data, vertex * 16u + i * 4u);
				}
				if (words[0] == 0 && words[1] == 0) {
					zero_words++;
				}
				const auto     x   = static_cast<int32_t>(static_cast<int16_t>(words[0] & 0xffffu));
				const auto     y   = static_cast<int32_t>(static_cast<int16_t>(words[0] >> 16u));
				const auto     z   = static_cast<int32_t>(static_cast<int16_t>(words[1] & 0xffffu));
				const uint32_t exp = ((words[1] >> 16u) >> 5u) & 0xfu;
				const bool     add = (flags & 16u) != 0;
				const float    p[3] = {
				    static_cast<float>((x << exp) + (add ? (static_cast<int32_t>(off_x << 16u) >> 8) : 0)) * scale,
				    static_cast<float>((y << exp) + (add ? (static_cast<int32_t>(off_y << 16u) >> 8) : 0)) * scale,
				    static_cast<float>((z << exp) + (add ? (static_cast<int32_t>(off_z << 16u) >> 8) : 0)) * scale};
				const float w[3] = {row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3] - cam[0],
				                    row[4] * p[0] + row[5] * p[1] + row[6] * p[2] + row[7] - cam[1],
				                    row[8] * p[0] + row[9] * p[1] + row[10] * p[2] + row[11] - cam[2]};
				for (int i = 0; i < 3; i++) {
					lo[i]  = std::min(lo[i], p[i]);
					hi[i]  = std::max(hi[i], p[i]);
					wlo[i] = std::min(wlo[i], w[i]);
					whi[i] = std::max(whi[i], w[i]);
				}
				total++;
			}
		}
		std::fprintf(stderr,
		             "[mesh-walk]   extent: %u vertices (%u zero) local [%g %g %g]..[%g %g %g] "
		             "camera-relative [%g %g %g]..[%g %g %g]\n",
		             total, zero_words, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], wlo[0], wlo[1],
		             wlo[2], whi[0], whi[1], whi[2]);
	}
}

} // namespace

void NoteTickWork(uint64_t tick, char kind, uint64_t a, uint64_t b); // diagnosticCheckpoints.cpp
void NoteTickWorkViews(const uint64_t* views, size_t count, const uint32_t* regions,
                       size_t num_regions); // diagnosticCheckpoints.cpp

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

// Diagnostics for facing (changes nothing unless asked):
// - KYTY_LOG_FACE_STATE=<pixel shader hash>: the first 8 draws of that pixel shader log the
//   registers that decide facing: PA_SU_SC_MODE_CNTL (cull, FACE), the primitive type,
//   VGT_TF_PARAM (tessellator topology), SPI_BARYC_CNTL (FRONT_FACE_ALL_BITS) and viewport 0.
// - KYTY_DIAG_FLIP_TESS_FACE_FILE=<file>: while that file exists, tessellated (patch) draws get
//   the opposite VkFrontFace, which flips gl_FrontFacing and culling for them only. A test of
//   whether a tessellated surface's facing matches the guest's; not a fix.
// - KYTY_DIAG_FLIP_ALL_FACE_FILE=<file>: the same for every draw while that file exists.
// - KYTY_FACE_CENSUS=1: every 5 s, how many draws had each culling / FACE / tessellation state.
static void LogAndDiagnoseFacing(const CommandBuffer& buffer, const CommandRecorder& vk_buffer,
                                 uint64_t ps_hash) {
	static const uint64_t log_ps = [] {
		const char* value = std::getenv("KYTY_LOG_FACE_STATE");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	static const char* flip_file     = std::getenv("KYTY_DIAG_FLIP_TESS_FACE_FILE");
	static const char* flip_all_file = std::getenv("KYTY_DIAG_FLIP_ALL_FACE_FILE");
	static const bool  census        = [] {
		const char* value = std::getenv("KYTY_FACE_CENSUS");
		return value != nullptr && value[0] == '1';
	}();
	if (log_ps == 0 && flip_file == nullptr && flip_all_file == nullptr && !census) {
		return;
	}
	const auto& ctx   = buffer.GetRegisters();
	const auto& mode  = ctx.GetModeControl();
	const auto  prim  = buffer.GetUserConfig().GetPrimType();
	const bool  patch = prim == Prospero::PrimitiveType::kPatch;
	if (log_ps != 0 && ps_hash == log_ps) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			const auto& sh = ctx.GetShaderRegisters();
			const auto& vp = ctx.GetScreenViewport().viewports[0];
			std::fprintf(stderr,
			             "[face-state] ps=%016" PRIx64 " prim=%u patch=%d cull_front=%d cull_back=%d "
			             "face=%d(%s front) tf_param=0x%08x topology=%u baryc_cntl=0x%08x "
			             "front_face_all_bits=%d vp_xscale=%f vp_yscale=%f vp_yoffset=%f "
			             "db_shader_control: z_order=%u kill=%d other_bits=0x%08x z=%d zw=%d zfunc=%u\n",
			             log_ps, static_cast<uint32_t>(prim), patch ? 1 : 0, mode.cull_front ? 1 : 0,
			             mode.cull_back ? 1 : 0, mode.face ? 1 : 0, mode.face ? "CW" : "CCW",
			             sh.m_vgtTfParam, (sh.m_vgtTfParam >> 5u) & 0x7u, sh.baryc_cntl,
			             (sh.baryc_cntl & 0x01000000u) != 0 ? 1 : 0, vp.xscale, vp.yscale, vp.yoffset,
			             sh.db_shader_control.shader_z_behavior, sh.db_shader_control.shader_kill_enable ? 1 : 0,
			             sh.db_shader_control.other_bits, ctx.GetDepthControl().z_enable ? 1 : 0,
			             ctx.GetDepthControl().z_write_enable ? 1 : 0, ctx.GetDepthControl().zfunc);
		}
	}
	const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	if (census) {
		// [patch][cull_front][cull_back][face]
		static std::atomic<uint64_t> counts[2][2][2][2] {};
		static std::atomic<int64_t>  printed_ms {0};
		counts[patch ? 1 : 0][mode.cull_front ? 1 : 0][mode.cull_back ? 1 : 0][mode.face ? 1 : 0]
		    .fetch_add(1, std::memory_order_relaxed);
		auto last = printed_ms.load(std::memory_order_relaxed);
		if (now_ms - last > 5000 && printed_ms.compare_exchange_strong(last, now_ms)) {
			std::string line;
			for (int t = 0; t < 2; t++)
				for (int f = 0; f < 2; f++)
					for (int b = 0; b < 2; b++)
						for (int c = 0; c < 2; c++) {
							const auto n = counts[t][f][b][c].exchange(0, std::memory_order_relaxed);
							if (n != 0) {
								line += fmt::format(" {}{}{}{}={}", t ? "tess:" : "", f ? "cullF" : "",
								                    b ? "cullB" : "", c ? "/CW" : "/CCW", n);
							}
						}
			std::fprintf(stderr, "[face-census]%s\n", line.c_str());
		}
	}
	if (!PipelineDynamicRasterStateEnabled() || (flip_file == nullptr && flip_all_file == nullptr)) {
		return;
	}
	static std::atomic<int64_t> checked_ms {-1};
	static std::atomic<bool>    flip_tess {false};
	static std::atomic<bool>    flip_all {false};
	if (now_ms - checked_ms.load(std::memory_order_relaxed) > 500) {
		checked_ms.store(now_ms, std::memory_order_relaxed);
		std::error_code error;
		const bool      tess_on = flip_file != nullptr && std::filesystem::exists(flip_file, error);
		const bool all_on = flip_all_file != nullptr && std::filesystem::exists(flip_all_file, error);
		if (tess_on != flip_tess.exchange(tess_on, std::memory_order_relaxed)) {
			std::fprintf(stderr, "[face-diag] tessellated front face %s\n",
			             tess_on ? "FLIPPED" : "normal");
		}
		if (all_on != flip_all.exchange(all_on, std::memory_order_relaxed)) {
			std::fprintf(stderr, "[face-diag] all draws' front face %s\n", all_on ? "FLIPPED" : "normal");
		}
	}
	if (flip_all.load(std::memory_order_relaxed) ||
	    (patch && flip_tess.load(std::memory_order_relaxed))) {
		vk_buffer.setFrontFace(mode.face ? vk::FrontFace::eCounterClockwise
		                                 : vk::FrontFace::eClockwise);
	}
}

static void SetGraphicsDynamicParams(const CommandBuffer& buffer, const CommandRecorder& vk_buffer,
                                     const ShaderVertexInputInfo& vs_input_info,
                                     const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	vk_buffer.setViewportWithCount(viewport_count, viewports.data());
	vk_buffer.setScissorWithCount(viewport_count, scissors.data());

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	vk_buffer.setLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	vk_buffer.setBlendConstants(blend_constants.data());
	// Tests only on aspects the pass has: a monolithic pipeline without them had no depth/stencil
	// state at all, but a linked pipeline's fragment library always has one (it cannot see the
	// formats), and depth/stencil on with no such attachment faulted the GPU (v436, fixed address
	// reads). Same values as before for every pass that has the aspect.
	const bool pass_depth   = rendering.depth_stencil_attachment.has_depth;
	const bool pass_stencil = rendering.depth_stencil_attachment.has_stencil;
	vk_buffer.setDepthTestEnable(depth.depth_test_enable && pass_depth ? VK_TRUE : VK_FALSE);
	vk_buffer.setDepthWriteEnable(depth.depth_write_enable && pass_depth ? VK_TRUE : VK_FALSE);
	vk_buffer.setDepthCompareOp(depth.depth_compare_op);

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = (use_front || use_back) && pass_depth;
	vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	vk_buffer.setStencilTestEnable(depth.stencil_test_enable && pass_stencil ? VK_TRUE : VK_FALSE);
	if (depth.stencil_test_enable && pass_stencil) {
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state) {
			vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp, state.compareOp);
			vk_buffer.setStencilCompareMask(face, state.compareMask);
			vk_buffer.setStencilWriteMask(face, state.writeMask);
			vk_buffer.setStencilReference(face, state.reference);
		};
		set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front);
		set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back);
	}

	if (PipelineDynamicRasterStateEnabled()) {
		// Formerly pipeline-key fields (PipelineCache::TryGetGraphicsPipeline), from the same
		// registers and with the same rule: rect lists are drawn without culling.
		const bool        rect_list = Prospero::IsRectList(buffer.GetUserConfig().GetPrimType());
		vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
		if (!rect_list && mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
		if (!rect_list && mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		vk_buffer.setCullMode(cull_mode);
		vk_buffer.setFrontFace(mode.face ? vk::FrontFace::eClockwise
		                                 : vk::FrontFace::eCounterClockwise);
#if !defined(__APPLE__)
		// Without a depth attachment the static pipeline had no depth/stencil state at all.
		const bool bounds_test =
		    depth.depth_bounds_test_enable && rendering.depth_stencil_attachment.has_depth;
		vk_buffer.setDepthBoundsTestEnable(bounds_test ? VK_TRUE : VK_FALSE);
		if (bounds_test) {
			vk_buffer.setDepthBounds(depth.depth_min_bounds, depth.depth_max_bounds);
		}
#endif
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	if (rendering.num_color_attachments != 0) {
		vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable);
	}
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

// v438zk: KYTY_DRAW_STATE_REUSE (off by default; =1 on; =verify also checks every reset).
// A draw's state is 12 KB, most of it three stages' arrays of 32 entries, built whole for every
// draw though a draw fills few entries (the idea is Senaxx/KytyPS5's, renderDraw.cpp; GPL-2.0-only).
// With the switch the GPU thread keeps one state and puts back only what the last draw can have
// written: everything outside the stages' arrays, and of those the entries up to each count and
// the one after it. That nothing writes past them is an assumption about every user of the
// state, so the switch stays opt-in until =verify (the reset state compared with a fresh one,
// byte for byte, at every draw; an exit on a difference) has run clean in a soak.
static int DrawStateReuseMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_DRAW_STATE_REUSE");
		if (value == nullptr || value[0] == '0' || value[0] == 0) {
			return 0;
		}
		return std::strcmp(value, "verify") == 0 ? 2 : 1;
	}();
	return mode;
}

static void ResetDrawRenderState(DrawRenderState& state, const DrawRenderState& fresh) {
	static_assert(std::is_trivially_copyable_v<DrawRenderState>);
	const auto at = [](const auto& member, const auto& whole) {
		return static_cast<size_t>(reinterpret_cast<const char*>(&member) - reinterpret_cast<const char*>(&whole));
	};
	// Up to the stages' arrays, and from the pixel stage's inputs to the end.
	std::memcpy(static_cast<void*>(&state), &fresh, at(fresh.vertex_info, fresh));
	std::memcpy(static_cast<void*>(&state.ps_input_info), &fresh.ps_input_info,
	            sizeof(DrawRenderState) - at(fresh.ps_input_info, fresh));
	for (size_t i = 0; i < state.vertex_info.size(); i++) {
		auto&       info  = state.vertex_info[i];
		const auto& clean = fresh.vertex_info[i];
		const auto  used  = [](int count) {
			return static_cast<size_t>(
			    std::clamp(count + 1, 0, static_cast<int>(ShaderVertexInputInfo::RES_MAX)));
		};
		const auto resources = used(info.resources_num);
		const auto buffers   = used(info.buffers_num);
		std::memcpy(static_cast<void*>(info.resources), clean.resources, resources * sizeof(info.resources[0]));
		std::memcpy(static_cast<void*>(info.resources_dst), clean.resources_dst, resources * sizeof(info.resources_dst[0]));
		std::memcpy(static_cast<void*>(info.buffers), clean.buffers, buffers * sizeof(info.buffers[0]));
		// The rest of the stage, after its arrays.
		std::memcpy(static_cast<void*>(&info.stage), &clean.stage, sizeof(ShaderVertexInputInfo) - at(clean.stage, clean));
	}
}

// A draw's state: the thread's reused one (KYTY_DRAW_STATE_REUSE) when no draw up the stack holds
// it, else one of its own, built as before.
// File-scope thread_locals (a class's inline static thread_local unique_ptr hit an lld-link
// "Associative COMDAT symbol does not exist" error with clang-cl).
struct DrawRenderStateFree {
	void operator()(DrawRenderState* state) const { ::operator delete(state); } // trivially destructible
};
static thread_local std::unique_ptr<DrawRenderState, DrawRenderStateFree> t_draw_state_shared;
static thread_local bool                                                   t_draw_state_shared_in_use = false;

class DrawRenderStateRef {
public:
	DrawRenderStateRef() {
		if (DrawStateReuseMode() == 0 || t_draw_state_shared_in_use) {
			m_state = &m_own.emplace();
			return;
		}
		static const auto fresh = std::make_unique<const DrawRenderState>();
		if (t_draw_state_shared == nullptr) {
			t_draw_state_shared.reset(static_cast<DrawRenderState*>(::operator new(sizeof(DrawRenderState))));
			std::memcpy(static_cast<void*>(t_draw_state_shared.get()), fresh.get(), sizeof(DrawRenderState));
		} else {
			ResetDrawRenderState(*t_draw_state_shared, *fresh);
		}
		if (DrawStateReuseMode() == 2 && std::memcmp(t_draw_state_shared.get(), fresh.get(), sizeof(DrawRenderState)) != 0) {
			EXIT("KYTY_DRAW_STATE_REUSE=verify: the reset draw state differs from a fresh one (a draw wrote an "
			     "array entry past its count)\n");
		}
		t_draw_state_shared_in_use = true;
		m_shared        = true;
		m_state         = t_draw_state_shared.get();
	}
	~DrawRenderStateRef() {
		if (m_shared) {
			t_draw_state_shared_in_use = false;
		}
	}
	KYTY_CLASS_NO_COPY(DrawRenderStateRef);
	[[nodiscard]] DrawRenderState& operator*() const { return *m_state; }

private:
	std::optional<DrawRenderState> m_own;
	DrawRenderState*               m_state  = nullptr;
	bool                           m_shared = false;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

// KYTY_MESH_ARGS_RUNS: the mesh-tasks command of the DRAW_INDIRECT being recorded, from the
// current batch, or converted now together with the run's later DRAW_INDIRECTs (see
// MeshArgsRunState). False: no run here; the caller converts this draw's counts alone.
static bool ConvertMeshArgsRun(RenderContext& context, MeshIndirectArgs& converter,
                               CommandBuffer& buffer, uint64_t args, vk::Buffer& command_buffer,
                               vk::DeviceSize& command_offset) {
	auto& run = MeshArgsRun();
	const bool replay_configured = RenderStats::IndirectReplayConfigured();
	if (replay_configured) {
		run.last_used_packet = nullptr;
		run.last_used_generation = 0;
	}
	if (run.packet == nullptr) {
		return false;
	}
	if (const auto* entry = run.Take(run.packet); entry != nullptr) {
		if (entry->args != args) {
			// The command processor collected another address (should not happen): drop the batch.
			run.converted.clear();
			run.next = 0;
			return false;
		}
		command_buffer = entry->buffer;
		command_offset = entry->offset;
		if (replay_configured) {
			run.last_used_packet = run.packet;
			run.last_used_generation = run.generation;
		}
		run.hits++;
		return true;
	}
	if (run.upcoming.size() < 2 || run.upcoming.front().packet != run.packet ||
	    run.upcoming.front().args != args) {
		return false;
	}
	// Synchronize and register every argument range first: uploads record copies and barriers of
	// their own, which must come before the conversions read the counts.
	std::vector<std::pair<Buffer*, uint64_t>> sources;
	sources.reserve(run.upcoming.size());
	// Obtaining a later range can merge buffers obtained before it (the earlier Buffer* is then a
	// deleted buffer whose bytes stop following the guest's): obtain them all once, then again
	// for the pointers the conversions use (as FillFlatSlotsOnGpu does).
	if (run.upcoming.size() > 1u) {
		for (const auto& entry: run.upcoming) {
			(void)context.GetBufferCache().ObtainBuffer(entry.args, 16u, false, false, {}, true);
		}
	}
	for (const auto& entry: run.upcoming) {
		const auto source = context.GetBufferCache().ObtainBuffer(entry.args, 16u, false, false, {},
		                                                          true);
		if (source.first == nullptr || !source.first->HasDeviceAddress()) {
			break;
		}
		sources.push_back(source);
	}
	if (sources.size() < 2) {
		return false;
	}
	context.GetCommandScheduler().EndRendering();
	const auto recorder = buffer.Recorder();
	vk::MemoryBarrier counts_ready {};
	counts_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
	                             vk::AccessFlagBits::eTransferWrite |
	                             vk::AccessFlagBits::eIndirectCommandRead;
	counts_ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eComputeShader, {}, 1, &counts_ready, 0,
	                         nullptr, 0, nullptr);
	run.converted.clear();
	run.next = 0;
	for (size_t i = 0; i < sources.size(); i++) {
		const auto converted = converter.Convert(
		    recorder, sources[i].first->BufferDeviceAddress() + sources[i].second, false, false);
		run.converted.push_back(
		    {run.upcoming[i].packet, run.upcoming[i].args, converted.buffer, converted.offset});
	}
	if (replay_configured) run.generation++;
	vk::MemoryBarrier command_ready {};
	command_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	command_ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                         vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &command_ready, 0,
	                         nullptr, 0, nullptr);
	run.batches++;
	run.batched += run.converted.size();
	run.upcoming.clear();
	const auto* entry = run.Take(run.packet);
	EXIT_IF(entry == nullptr);
	command_buffer = entry->buffer;
	command_offset = entry->offset;
	if (replay_configured) {
		run.last_used_packet = run.packet;
		run.last_used_generation = run.generation;
	}
	if (static const bool log = Common::FpsLogDiagnostics(); log) {
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		if (now - last >= std::chrono::seconds(5)) {
			const double s = std::chrono::duration<double>(now - last).count();
			last           = now;
			std::fprintf(stderr,
			             "[mesh-args-runs] batches %.0f/s converted in batches %.0f/s, used %.0f/s, "
			             "single %.0f/s\n",
			             static_cast<double>(run.batches) / s, static_cast<double>(run.batched) / s,
			             static_cast<double>(run.hits) / s, static_cast<double>(run.single) / s);
			run.batches = run.batched = run.hits = run.single = 0;
		}
	}
	return true;
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	const CpuCost::Phase cpu_phase_("AcquireRenderTargets");
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		// v393 (11-11, mac_v390_1): this check let through what FindRenderTarget refuses, and the
		// game exited there ("color target requires rediscovery") at a level change with no word of
		// which condition held. Same conditions now (a target without guest memory is never
		// registered, as FindTexture allows), and the image is described before the exit.
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) || owner->depth_id ||
		    owner->binding.needs_rebind) {
			if (owner != nullptr) {
				std::fprintf(stderr,
				             "[rt-rediscover] slot %u id %u: registered %d, empty %d, depth_id %u, needs_rebind %d; "
				             "image 0x%llx+0x%llx %ux%u fmt %u; wanted 0x%llx+0x%llx %ux%u\n",
				             target.target_slot, static_cast<uint32_t>(target.image_id.index),
				             static_cast<int>(owner->registered), static_cast<int>(owner->info.data.Empty()),
				             static_cast<uint32_t>(owner->depth_id.index), static_cast<int>(owner->binding.needs_rebind),
				             static_cast<unsigned long long>(owner->info.data.address),
				             static_cast<unsigned long long>(owner->info.data.size), owner->info.extent.width,
				             owner->info.extent.height, static_cast<uint32_t>(owner->backing.format),
				             static_cast<unsigned long long>(target.desc.info.data.address),
				             static_cast<unsigned long long>(target.desc.info.data.size), target.desc.info.extent.width,
				             target.desc.info.extent.height);
				std::fflush(stderr);
			}
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		if (ImgTrace::Overlaps(target.desc.info.data.address, target.desc.info.data.size)) {
			ImgTrace::Log("rt slot%u want 0x%llx %ux%u -> id=%u img 0x%llx+0x%llx %ux%u fmt%u vk=%p",
			              target.target_slot,
			              static_cast<unsigned long long>(target.desc.info.data.address),
			              target.desc.info.extent.width, target.desc.info.extent.height,
			              static_cast<uint32_t>(target.image_id.index),
			              static_cast<unsigned long long>(image.info.data.address),
			              static_cast<unsigned long long>(image.info.data.size), image.info.extent.width,
			              image.info.extent.height, static_cast<uint32_t>(image.backing.format),
			              static_cast<const void*>(static_cast<VkImage>(image.backing.image)));
		}
		if (PresentRt::Get().enabled && target.guest_mip_level == 0) {
			PresentRt::NoteTarget(target.image_id, target.desc.info.data.address,
			                      target.desc.info.extent.width, target.desc.info.extent.height,
			                      static_cast<uint32_t>(target.desc.info.pixel_format),
			                      target.target_slot);
		}
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		// v446x: and within the cache image the view really is on (it can be smaller than the
		// guest target asked for: a render area past an attachment's end is invalid and the GPU
		// writes outside the image, VUID-VkRenderingInfo-pNext-06079).
		state.width  = std::min(state.width, std::max(image.backing.extent.width >> view.base_level, 1u));
		state.height = std::min(state.height, std::max(image.backing.extent.height >> view.base_level, 1u));
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (metadata.kind == ImageMetadataKind::Htile) {
			HtileWatch::NoteDepthBind(depth.desc.info.data.address, metadata.range.address,
			                          depth.desc.view_info.base_layer, depth.depth_clear_enable,
			                          meta_clear, depth.depth_clear_value, depth.depth_write_enable);
		}
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Without VK_EXT_attachment_feedback_loop_* (the AMD Windows driver), the draw samples
			// the depth target it writes in the GENERAL layout: not defined by Vulkan, but what
			// the hardware does anyway, and better than ending the emulator.
			static std::atomic_bool warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				LOGF("Warning: depth attachment feedback loop without host support; "
				     "using the GENERAL layout\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		// v446x: and within the cache image the depth view is on. Mac 11-22, the truck-fight
		// cinematics: a 59x264 D16 target resolved to a 48x264 cache image, the render area stayed
		// 59 wide and the GPU wrote past the image (device losses, WriteInvalid; validation:
		// VUID-VkRenderingInfo-pNext-06079).
		state.width  = std::min(state.width, std::max(image.backing.extent.width >> view.base_level, 1u));
		state.height = std::min(state.height, std::max(image.backing.extent.height >> view.base_level, 1u));
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (sh_regs.target_output_mode[slot] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
	// Nonzero: a draw whose counts the GPU reads from these guest DrawIndirectArgs.
	uint64_t gpu_args_address = 0;
	// Nonzero: a vertex-pipeline draw whose instance count the GPU reads from this guest dword.
	uint64_t gpu_instance_count_address = 0;
	// With gpu_args_address: START_INST_LOC's user SGPR + 1 (0 = none), see DrawAutoArgs.
	uint32_t start_instance_sgpr = 0;
	const uint32_t* indirect_packet = nullptr;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		// A buffer only constant-only embedded fetches refer to is never read: bind the null
		// buffer instead of acquiring whatever its unconsumed descriptor names.
		bool fetched = false;
		for (int a = 0; a < vs_input_info.resources_num && !fetched; a++) {
			fetched = vs_input_info.resources_dst[a].buffer_index == i &&
			          VertexAttributeFetched(vs_input_info, a);
		}
		const auto size = fetched ? VertexBufferDescriptorSize(i, vs_input_info) : 0;
		sizes[i]        = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase, uint64_t shaders = 0) {
	// v446s: with the shaders (vertex hash low 32 bits << 32 | pixel hash low 32 bits) in the last
	// argument, a device loss's NVIDIA checkpoint names the draw (KYTY_NV_CHECKPOINTS=1).
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, shaders != 0 ? shaders : draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// The programs of a draw for these registers (GetGraphicsPrograms with the SRT refresh).
static void SelectDrawPrograms(RenderContext& context, const HW::Context& ctx, const HW::Shader& sh_ctx,
                               const HW::UserConfig& ucfg, bool indexed, const char* name,
                               uint32_t color_output_mask, bool ps_active,
                               PipelineCache::GraphicsPrograms&      programs,
                               std::array<ShaderVertexInputInfo, 3>& vertex_info,
                               ShaderPixelInputInfo&                 ps_input_info) {
	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	programs      = {};
	ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	auto& pipeline_cache = context.GetPipelineCache();
	if (indexed) {
		LogDrawPhase(name, "GetGraphicsPrograms");
	}
	programs = pipeline_cache.GetGraphicsPrograms(vertex_shader_info, pixel_shader_info, shader_regs,
	                                              ctx, ucfg, target_export_mapping, ps_active,
	                                              vertex_info, ps_input_info);
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	// KYTY_DRAW_PIPE, on the draw thread: the command processor selected them for these registers.
	if (const auto* prepared = DrawPipe::CurrentPrepared(); prepared != nullptr) {
		EXIT_IF(prepared->ps_active != state.ps_active ||
		        prepared->color_output_mask != color_output_mask);
		state.programs      = prepared->programs;
		state.vertex_info   = prepared->vertex_info;
		state.ps_input_info = prepared->ps_input_info;
		return;
	}
	SelectDrawPrograms(buffer.GetContext(), buffer.GetRegisters(), buffer.GetShaders(),
	                   buffer.GetUserConfig(), draw.IsIndexed(), draw.Name(), color_output_mask,
	                   state.ps_active, state.programs, state.vertex_info, state.ps_input_info);
}

void RenderExecutor::RunPipeDraw(uint64_t submit_id, CommandBuffer& buffer,
                                 const DrawIndexArgs* index_args, const DrawAutoArgs* auto_args) {
	if (index_args != nullptr) {
		DrawIndex(submit_id, buffer, *index_args);
	} else {
		DrawAuto(submit_id, buffer, *auto_args);
	}
}

// KYTY_DRAW_PIPE, the command processor: RefreshShaders for a draw the draw thread will run, with
// copies of the snapshots the program cache will overwrite. False: no valid vertex shader.
bool RenderExecutor::PrepareDrawProgramsForPipe(RenderContext& context, const HW::Context& ctx,
                                                const HW::Shader& sh_ctx, const HW::UserConfig& ucfg,
                                                DrawPipe::PreparedPrograms& out) {
	if (!DrawHasValidVertexShader(sh_ctx)) {
		return false;
	}
	const auto& shader_regs = ctx.GetShaderRegisters();
	out.color_output_mask   = DrawColorOutputMask(ctx);
	out.ps_active           = sh_ctx.GetPs().ps_regs.data_addr != 0 &&
	                (out.color_output_mask != 0 || PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	{
		// Off the renderer mutex the draw thread holds: guest reads of bytes no GPU work wrote are
		// served from memory, the rest takes the mutex (ReadShaderGuestMemory).
		RenderContext::MarkOffLock(true);
		SelectDrawPrograms(context, ctx, sh_ctx, ucfg, false, "pipe", out.color_output_mask,
		                   out.ps_active, out.programs, out.vertex_info, out.ps_input_info);
		RenderContext::MarkOffLock(false);
	}
	const auto writes = [](const ShaderStageRuntime& runtime) {
		return runtime.program != nullptr &&
		       (HasShaderBufferWrites(runtime) || runtime.program->has_address_writes ||
		        std::ranges::any_of(runtime.program->info.images, [](const auto& image) {
			        return image.written;
		        }));
	};
	// The guest bytes the stage's buffer writes can reach: the store extent inside a written V#
	// (as NativeStorageBuffer marks them), else the whole V#. Image writes, image atomics and
	// raw-pointer stores have no range.
	const auto add_ranges = [&out](const ShaderStageRuntime& runtime) {
		if (runtime.program == nullptr) {
			return;
		}
		const auto& program   = *runtime.program;
		const auto& resources = *runtime.resources;
		if (program.has_address_writes ||
		    std::ranges::any_of(program.info.images, [](const auto& image) {
			    return image.written || image.atomic || image.atomic64;
		    })) {
			out.writes_unbounded = true;
		}
		const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
		for (size_t i = 0; i < count; i++) {
			const auto& buffer = program.info.buffers[i];
			if (!buffer.written && !buffer.atomic) {
				continue;
			}
			const auto& value = resources.buffers[i];
			if (value.dword_count < 4) {
				out.writes_unbounded = true;
				continue;
			}
			ShaderBufferResource descriptor;
			std::memcpy(descriptor.fields, value.dwords.data(), sizeof(descriptor.fields));
			const uint64_t address = descriptor.Base48();
			const uint64_t size    = descriptor.GetSize();
			if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
				continue; // bound as null: the writes are dropped
			}
			uint64_t begin = 0;
			uint64_t end   = size;
			if (!buffer.atomic && i < resources.buffer_write_extents.size()) {
				if (const auto& extent = resources.buffer_write_extents[i]; extent.valid) {
					begin = std::min(extent.begin, size);
					end   = std::min(extent.end, size);
				}
			}
			if (end > begin) {
				out.write_ranges.emplace_back(address + begin, address + end);
			}
		}
	};
	out.writes_memory    = false;
	out.writes_unbounded = false;
	out.write_ranges.clear();
	for (uint32_t i = 0; i < 3; i++) {
		auto& stage = out.vertex_info[i].stage;
		if (stage.resources != nullptr) {
			out.resources[i] = *stage.resources;
			stage.resources  = &out.resources[i];
			if (writes(stage)) {
				out.writes_memory = true;
				add_ranges(stage);
			}
		}
	}
	if (auto& stage = out.ps_input_info.stage; stage.resources != nullptr) {
		out.resources[3]  = *stage.resources;
		stage.resources   = &out.resources[3];
		if (writes(stage)) {
			out.writes_memory = true;
			add_ranges(stage);
		}
	}
	// writes() counts image writes but not image atomics; HasShaderBufferWrites counts both.
	if (out.writes_memory && out.write_ranges.empty() && !out.writes_unbounded) {
		out.writes_unbounded = true; // writes nothing a range names: keep the old drain
	}
	return true;
}

bool DrawPrepHasVertexShader(const HW::Shader& sh_ctx) {
	return DrawHasValidVertexShader(sh_ctx);
}

bool DrawPrepPixelState(const HW::Context& ctx, const HW::Shader& sh_ctx,
                        std::array<Prospero::ColorComponentMapping, 8>& target_export_mapping) {
	static_assert(RENDER_COLOR_ATTACHMENTS_MAX == 8);
	const auto& shader_regs       = ctx.GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(ctx);
	const bool  ps_active         = sh_ctx.GetPs().ps_regs.data_addr != 0 &&
	                       (color_output_mask != 0 ||
	                        PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	target_export_mapping = {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	return ps_active;
}

// KYTY_LOG_DRAWS=<first_frame>:<count>: from that frame on, log count draws with their shader
// hashes, counts and whether their programs exist (-1: not resolved, the draw is empty). A census
// of what a scene submits.
static void LogDrawCensusLine(CommandBuffer& buffer, const char* kind, uint32_t count,
                              uint32_t instances, int vs_program, int ps_program, int ps_active) {
	static const auto setting = [] {
		struct Setting {
			uint64_t first_frame = 0;
			uint32_t count       = 0;
		} result;
		if (const char* value = std::getenv("KYTY_LOG_DRAWS"); value != nullptr) {
			char* end          = nullptr;
			result.first_frame = std::strtoull(value, &end, 10);
			if (end != nullptr && *end == ':') {
				result.count = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
			}
		}
		return result;
	}();
	static std::atomic<uint32_t> used {0};
	// KYTY_LOG_DRAWS_WITH_COUNT=<n>: log the first 64 draws of exactly n vertices/indices.
	static const uint32_t with_count = [] {
		const char* value = std::getenv("KYTY_LOG_DRAWS_WITH_COUNT");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (with_count != 0 && count == with_count) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 64) {
			const auto& sh = buffer.GetShaders();
			const auto  h  = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
			std::fprintf(stderr,
			             "[draw-count] %s es=%016" PRIx64 " gs=%016" PRIx64 " ps=%016" PRIx64
			             " count=%u instances=%u vs_program=%d ps_program=%d\n",
			             kind, h(sh.GetVs().es_regs.data_addr), h(sh.GetVs().gs_regs.data_addr),
			             h(sh.GetPs().ps_regs.data_addr), count, instances, vs_program, ps_program);
			std::fflush(stderr);
		}
	}
	// KYTY_LOG_GS_SLOTS=<es hash>: each draw of that ES/VS shader with its pixel shader and GS
	// user-data slots 24..30 as the context holds them at the draw (first 200 draws).
	if (static const uint64_t watch = [] {
		    const char* value = std::getenv("KYTY_LOG_GS_SLOTS");
		    return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	    }();
	    watch != 0) {
		const auto& sh = buffer.GetShaders();
		const bool  es_match =
		    sh.GetVs().es_regs.data_addr != 0 && ShaderDeclaredHash(sh.GetVs().es_regs.data_addr) == watch;
		// Or the pixel shader's hash; with KYTY_TRACE_GS_SLOT_SEQ the lines come from its budget, with
		// the last write to the traced slot this thread saw.
		const bool ps_match =
		    !es_match && sh.GetPs().ps_regs.data_addr != 0 && ShaderDeclaredHash(sh.GetPs().ps_regs.data_addr) == watch;
		static std::atomic<uint32_t> logged {0};
		if ((es_match || ps_match) && std::getenv("KYTY_TRACE_GS_SLOT_SEQ") != nullptr) {
			// The command-buffer dwords these draws' slot came from, read again a few frames later:
			// does the game write them after the submission was parsed?
			struct Pending {
				const uint32_t* where;
				uint32_t        value;
				uint64_t        frame;
			};
			static std::mutex           pending_mutex;
			static std::vector<Pending> pending;
			const uint64_t frame_now = buffer.GetContext().GetGpu().GetFrameNum();
			{
				std::lock_guard lock(pending_mutex);
				for (auto it = pending.begin(); it != pending.end();) {
					if (frame_now >= it->frame + 3) {
						const uint32_t now = *it->where;
						if (TakeGsSlotSeqLine()) {
							std::fprintf(stderr, "[gs-seq] later at=%p parsed=%08x now=%08x (%llu frames later)%s\n",
							             static_cast<const void*>(it->where), it->value, now,
							             static_cast<unsigned long long>(frame_now - it->frame),
							             now != it->value ? " CHANGED" : "");
						}
						it = pending.erase(it);
					} else {
						++it;
					}
				}
				const auto w = GetGsSlotLastWriter();
				if (w.source == 1 && w.where != nullptr && pending.size() < 256) {
					pending.push_back({w.where, w.value, frame_now});
				}
			}
			if (TakeGsSlotSeqLine()) {
				const auto  writer = GetGsSlotLastWriter();
				const auto& users  = sh.GetVs().gs_user_sgpr.value;
				std::fprintf(stderr,
				             "[gs-seq] draw frame=%llu flip=%llu count=%u user_sgpr=%u s24=%08x s36=%08x last=#%llu src=%u value=%08x"
				             " at=%p now=%08x set=%u+%u\n",
				             static_cast<unsigned long long>(buffer.GetContext().GetGpu().GetFrameNum()),
				             static_cast<unsigned long long>(Sync::GuestFlipCount()), count,
				             static_cast<uint32_t>(sh.GetVs().gs_regs.rsrc2.user_sgpr), users[16], users[28],
				             static_cast<unsigned long long>(writer.seq), writer.source, writer.value,
				             static_cast<const void*>(writer.where),
				             writer.where != nullptr ? *writer.where : 0u, writer.start, writer.count);
			}
		} else if (es_match || ps_match) {
			if (logged.fetch_add(1) < 200u) {
				const auto& users = sh.GetVs().gs_user_sgpr.value;
				std::fprintf(stderr,
				             "[gs-slots] frame=%llu ps=%016llx count=%u instances=%u user_sgpr=%u"
				             " s32..s38: %08x %08x %08x %08x %08x %08x %08x\n",
				             static_cast<unsigned long long>(
				                 buffer.GetContext().GetGpu().GetFrameNum()),
				             static_cast<unsigned long long>(
				                 sh.GetPs().ps_regs.data_addr != 0
				                     ? ShaderDeclaredHash(sh.GetPs().ps_regs.data_addr)
				                     : 0ull),
				             count, instances, static_cast<uint32_t>(sh.GetVs().gs_regs.rsrc2.user_sgpr),
				             users[24], users[25], users[26], users[27], users[28], users[29], users[30]);
			}
		}
	}
	// KYTY_TRACE_GS_SLOT_SEQ with KYTY_TRACE_GS_SLOT=<slot> and KYTY_TRACE_GS_SLOT_VALUE=<hex>: every
	// draw whose GS shader reads that slot and sees that value, with the write that left it.
	if (static const struct BadSlot {
		    int32_t  slot  = -1;
		    uint32_t value = 0;
	    } bad = [] {
		    BadSlot     b;
		    const char* seq   = std::getenv("KYTY_TRACE_GS_SLOT_SEQ");
		    const char* slot  = std::getenv("KYTY_TRACE_GS_SLOT");
		    const char* value = std::getenv("KYTY_TRACE_GS_SLOT_VALUE");
		    if (seq != nullptr && slot != nullptr && value != nullptr) {
			    b.slot  = static_cast<int32_t>(std::strtol(slot, nullptr, 10));
			    b.value = static_cast<uint32_t>(std::strtoul(value, nullptr, 16));
		    }
		    return b;
	    }();
	    bad.slot >= 0 && bad.slot < 32) {
		// The dword the bad value came from, read again 1, 4, 16 and 40 ms later (checked at every
		// draw): does the game write it after the submission was parsed?
		struct Timed {
			const uint32_t*                       where;
			uint32_t                              parsed;
			std::chrono::steady_clock::time_point at;
			uint32_t                              seen[4];
			int                                   next;
		};
		static std::vector<Timed> timed;
		if (!timed.empty()) {
			const auto now = std::chrono::steady_clock::now();
			static constexpr int ages_ms[4] = {1, 4, 16, 40};
			for (auto it = timed.begin(); it != timed.end();) {
				while (it->next < 4 && now - it->at >= std::chrono::milliseconds(ages_ms[it->next])) {
					it->seen[it->next++] = *it->where;
				}
				if (it->next == 4) {
					std::fprintf(stderr, "[gs-seq] timed at=%p parsed=%08x +1ms=%08x +4ms=%08x +16ms=%08x +40ms=%08x\n",
					             static_cast<const void*>(it->where), it->parsed, it->seen[0], it->seen[1],
					             it->seen[2], it->seen[3]);
					it = timed.erase(it);
				} else {
					++it;
				}
			}
		}
		const auto& sh = buffer.GetShaders();
		if (sh.GetVs().gs_regs.rsrc2.user_sgpr > static_cast<uint32_t>(bad.slot) &&
		    sh.GetVs().gs_user_sgpr.value[bad.slot] == bad.value && TakeGsSlotSeqLine()) {
			const auto writer = GetGsSlotLastWriter();
			if (writer.where != nullptr && timed.size() < 64) {
				timed.push_back({writer.where, writer.value, std::chrono::steady_clock::now(), {}, 0});
			}
			const auto h      = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
			// The command-buffer dwords after the write (the packets up to this draw).
			std::string next;
			if (writer.where != nullptr) {
				const auto addr = reinterpret_cast<uint64_t>(writer.where);
				next += fmt::format(" gpu-written={} gpu-value={:08x} |",
				                    Libs::LibKernel::Memory::IsGpuBufferWritten(addr, 4) ? 1 : 0,
				                    Libs::LibKernel::Memory::PeekGpuDword(addr));
			}
			if (writer.source == 1 && writer.where != nullptr) {
				for (int i = 1; i <= 40; i++) {
					next += fmt::format(" {:08x}", writer.where[i]);
				}
				// And the 48 dwords before the SET_SH_REG packet's header.
				const auto* header = writer.where - (bad.slot - static_cast<int32_t>(writer.start)) - 2;
				next += " | before:";
				for (int i = -48; i <= 0; i++) {
					next += fmt::format(" {:08x}", header[i]);
				}
			}
			std::fprintf(stderr,
			             "[gs-seq] bad-draw frame=%llu es=%016llx ps=%016llx count=%u instances=%u last=#%llu src=%u"
			             " value=%08x at=%p next:%s\n",
			             static_cast<unsigned long long>(buffer.GetContext().GetGpu().GetFrameNum()),
			             static_cast<unsigned long long>(h(sh.GetVs().es_regs.data_addr)),
			             static_cast<unsigned long long>(h(sh.GetPs().ps_regs.data_addr)), count, instances,
			             static_cast<unsigned long long>(writer.seq), writer.source, writer.value,
			             static_cast<const void*>(writer.where), next.c_str());
		}
	}
	// KYTY_LOG_DRAW_PAIRS=1: each new (es, gs, ps) combination once, with its first draw's shape.
	if (static const bool log_pairs = std::getenv("KYTY_LOG_DRAW_PAIRS") != nullptr; log_pairs) {
		static std::mutex                   pairs_mutex;
		static std::unordered_set<uint64_t> pairs;
		const auto& sh = buffer.GetShaders();
		const auto  h  = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
		const uint64_t es = h(sh.GetVs().es_regs.data_addr);
		const uint64_t gs = h(sh.GetVs().gs_regs.data_addr);
		const uint64_t ps = h(sh.GetPs().ps_regs.data_addr);
		bool fresh = false;
		{
			std::lock_guard lock(pairs_mutex);
			fresh = pairs.insert(es ^ (gs * 31u) ^ (ps * 0x9e3779b97f4a7c15ull)).second;
		}
		if (fresh) {
			const auto& regs = buffer.GetRegisters();
			std::fprintf(stderr,
			             "[draw-pair] %s es=%016" PRIx64 " gs=%016" PRIx64 " ps=%016" PRIx64
			             " count=%u instances=%u vs_program=%d ps_program=%d ps_active=%d rt_mask=0x%x"
			             " cb_shader_mask=0x%x\n",
			             kind, es, gs, ps, count, instances, vs_program, ps_program, ps_active,
			             regs.GetRenderTargetMask(), regs.GetShaderRegisters().m_cbShaderMask);
			for (uint32_t i = 0; i < 8; i++) {
				const auto& rt = regs.GetRenderTarget(i);
				const auto& bc = regs.GetBlendControl(i);
				if (rt.base.addr == 0 || ((regs.GetRenderTargetMask() >> (i * 4u)) & 0xfu) == 0) {
					continue;
				}
				std::fprintf(stderr,
				             "[draw-pair]   rt%u addr=0x%010" PRIx64 " fmt=%u nfmt=%u order=%u %ux%u bypass=%d"
				             " blend=%d src=%u dst=%u comb=%u sep=%d asrc=%u adst=%u acomb=%u\n",
				             i, rt.base.addr, static_cast<uint32_t>(rt.info.format),
				             static_cast<uint32_t>(rt.info.channel_type),
				             static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
				             rt.attrib2.height + 1, rt.info.blend_bypass ? 1 : 0, bc.enable ? 1 : 0,
				             bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
				             bc.separate_alpha_blend ? 1 : 0, bc.alpha_srcblend, bc.alpha_destblend,
				             bc.alpha_comb_fcn);
			}
			{
				const auto& vs = sh.GetVs();
				std::string users;
				for (uint32_t i = 0; i < HW::UserSgprInfo::SGPRS_MAX; i++) {
					users += fmt::format(" {:08x}", vs.gs_user_sgpr.value[i]);
				}
				std::fprintf(stderr, "[draw-pair]   gs user_sgpr=%u written=%u:%s\n",
				             static_cast<uint32_t>(vs.gs_regs.rsrc2.user_sgpr), vs.gs_user_sgpr.count,
				             users.c_str());
			}
			const auto& dc = regs.GetDepthControl();
			std::fprintf(stderr, "[draw-pair]   depth z=%d zw=%d zfunc=%u\n", dc.z_enable ? 1 : 0,
			             dc.z_write_enable ? 1 : 0, static_cast<uint32_t>(dc.zfunc));
			std::fflush(stderr);
		}
	}
	if (setting.count == 0) {
		return;
	}
	const auto frame = static_cast<uint64_t>(buffer.GetContext().GetGpu().GetFrameNum());
	// KYTY_LOG_DRAWS_TRIGGER=<file>: also wait until that file exists (checked once per frame), so
	// the census can start at a screen reached by hand.
	static const char*           trigger    = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	// Polled every 1024 draws: the GPU frame number does not advance in every mode.
	static std::atomic<uint32_t> checked    = 0;
	static std::atomic_bool      triggered  = trigger == nullptr;
	if (!triggered && (checked.fetch_add(1, std::memory_order_relaxed) & 1023u) == 0u) {
		if (FILE* file = std::fopen(trigger, "rb"); file != nullptr) {
			std::fclose(file);
			triggered = true;
			LOGF("Draw census: triggered at frame %" PRIu64 "\n", frame);
		}
	}
	// KYTY_LOG_DRAWS_FLIP=<guest flip>: start at that guest flip instead of the GPU frame.
	static const uint64_t first_flip = [] {
		const char* value = std::getenv("KYTY_LOG_DRAWS_FLIP");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
	}();
	if (first_flip != 0 && Sync::GuestFlipCount() < first_flip) {
		return;
	}
	if (!triggered || (first_flip == 0 && frame < setting.first_frame) ||
	    used.fetch_add(1, std::memory_order_relaxed) >= setting.count) {
		return;
	}
	const auto& sh_ctx = buffer.GetShaders();
	const auto  hash   = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
	const auto& regs   = buffer.GetRegisters();
	const auto& dc     = regs.GetDepthControl();
	const auto& drt    = regs.GetDepthRenderTarget();
	std::fprintf(stderr,
	             "Draw census: frame=%" PRIu64 " %s es=%016" PRIx64 " gs=%016" PRIx64
	             " ps=%016" PRIx64 " count=%u instances=%u vs_program=%d ps_program=%d ps_active=%d"
	             " ls=%016" PRIx64 " hs=%016" PRIx64 " rt_mask=0x%x z=%d zw=%d zfunc=%u"
	             " zbase=0x%" PRIx64 " zfmt=%u se=%d sf=%u/%u sops=%u%u%u ref=%u mask=%x wmask=%x opval=%u"
	             " sbase=0x%" PRIx64 " swbase=0x%" PRIx64 " sfmt=%u htile=0x%" PRIx64 "\n",
	             frame, kind, hash(sh_ctx.GetVs().es_regs.data_addr),
	             hash(sh_ctx.GetVs().gs_regs.data_addr), hash(sh_ctx.GetPs().ps_regs.data_addr),
	             count, instances, vs_program, ps_program, ps_active,
	             hash(sh_ctx.GetVs().ls_regs.data_addr), hash(sh_ctx.GetVs().hs_regs.data_addr),
	             regs.GetRenderTargetMask(), dc.z_enable ? 1 : 0, dc.z_write_enable ? 1 : 0,
	             static_cast<unsigned>(dc.zfunc), drt.z_read_base_addr,
	             static_cast<unsigned>(drt.z_info.format), dc.stencil_enable ? 1 : 0,
	             static_cast<unsigned>(dc.stencilfunc), static_cast<unsigned>(dc.stencilfunc_bf),
	             static_cast<unsigned>(regs.GetStencilControl().stencil_fail),
	             static_cast<unsigned>(regs.GetStencilControl().stencil_zpass),
	             static_cast<unsigned>(regs.GetStencilControl().stencil_zfail),
	             static_cast<unsigned>(regs.GetStencilMask().stencil_testval),
	             static_cast<unsigned>(regs.GetStencilMask().stencil_mask),
	             static_cast<unsigned>(regs.GetStencilMask().stencil_writemask),
	             static_cast<unsigned>(regs.GetStencilMask().stencil_opval),
	             drt.stencil_read_base_addr, drt.stencil_write_base_addr,
	             static_cast<unsigned>(drt.stencil_info.format), drt.htile_data_base_addr);
	std::fflush(stderr);
}

void SeqNote(char kind, uint64_t shader_addr, int queue, uint64_t frame, uint64_t presented); // graphicsRun.cpp

// KYTY_FULL_BARRIER_PS=<hex,...> (experiment): draws of those pixel shaders get a full memory
// barrier (all commands, all memory) before and after them, outside the render pass. Shows
// whether a draw reads or writes something a missing barrier leaves racing.
static bool FullBarrierPs(uint64_t hash) {
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_FULL_BARRIER_PS");
		return value != nullptr ? std::string(value) : std::string();
	}();
	return !list.empty() && hash != 0 &&
	       (list == "all" || list.find(fmt::format("{:016x}", hash)) != std::string::npos);
}

static void FullMemoryBarrier(const CommandRecorder& vk_buffer) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                          vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0, nullptr,
	                          0, nullptr);
}


static void LogDrawCensus(CommandBuffer& buffer, const DrawCallInfo& draw,
                          const DrawRenderState& state) {
	SeqNote('D', buffer.GetShaders().GetPs().ps_regs.data_addr, 0,
	        static_cast<uint64_t>(buffer.GetContext().GetGpu().GetFrameNum()),
	        buffer.GetContext().GetGraphics().presented_frames.load(std::memory_order_relaxed));
	LogDrawCensusLine(buffer, draw.Name(), draw.index_count, draw.instance_count,
	                  state.programs.vertex[0] ? 1 : 0, state.programs.pixel ? 1 : 0,
	                  state.ps_active ? 1 : 0);
}

// Diagnostics: the draw parameters of the watched shader's draws in the watched frame (see
// InWatchedShaderFrame); each line precedes that draw's WatchShader bindings.
void RenderExecutor::LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
                                    uint32_t index_offset, int32_t base_vertex,
                                    uint32_t first_vertex, int32_t host_vertex_offset,
                                    uint32_t host_first_instance, bool indirect) {
	static std::atomic<uint32_t> logged {0};
	const auto*                  program = state.vertex_info[0].stage.program;
	if (program == nullptr ||
	    !InWatchedShaderFrame(program->shader_hash,
	                          m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                          logged)) {
		return;
	}
	LOGF("WatchShader draw: %s count=%u instances=%u first_instance=%u index_offset=%u "
	     "base_vertex=%d first_vertex=%u host_vertex_offset=%d host_first_instance=%u "
	     "indirect=%d\n",
	     draw.Name(), draw.index_count, draw.instance_count, draw.first_instance, index_offset,
	     base_vertex, first_vertex, host_vertex_offset, host_first_instance, indirect ? 1 : 0);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const CpuCost::Phase cpu_phase_("PrepareDrawRenderState");
	KYTY_PROFILER_FUNCTION();
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	RefreshShaders(buffer, draw, color_output_mask, state);
	LogDrawCensus(buffer, draw, state);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel)) {
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);

	// A draw that writes nothing still counts its samples for an occlusion query (the game's
	// reference draw with depth testing off): it rasterizes without attachments.
	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active &&
	    !Occlusion::CountsDraw(buffer)) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(const CommandRecorder&       vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandRecorder&     vk_buffer,
                              const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandRecorder& vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().depth_color_copy_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.first_vertex != 0 || emit.first_instance != 0 ||
	                     state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.depth_load_clear_enable) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("[depth-copy] first depth-to-color copy: %ux%u\n",
			     static_cast<uint32_t>(rect.right - rect.left),
			     static_cast<uint32_t>(rect.bottom - rect.top));
		}
		const auto command = buffer.Recorder();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

// KYTY_WATCH_STENCIL_PS=<hex,...>: while the KYTY_WATCH_SHADER_TRIGGER file exists, after each draw
// of a listed pixel shader (first 12 frames) the stencil plane of its depth target is copied out
// to _ShaderDump/wimg_<frame>_<hash>_host.bin.
static void WatchStencilAfterDraw(RenderContext& context, const CommandBuffer& buffer,
                                  const DrawRenderState& state) {
	static const std::string hashes = [] {
		const char* value = std::getenv("KYTY_WATCH_STENCIL_PS");
		return value != nullptr ? std::string(value) : std::string();
	}();
	static const char* trigger = std::getenv("KYTY_WATCH_SHADER_TRIGGER");
	if (hashes.empty() || trigger == nullptr || !state.depth_info.image_id) {
		return;
	}
	const auto ps = buffer.GetShaders().GetPs().ps_regs.data_addr;
	if (ps == 0) {
		return;
	}
	const auto hash = ShaderDeclaredHash(ps);
	const auto name = fmt::format("{:016x}", hash);
	if (hashes.find(name) == std::string::npos) {
		return;
	}
	const auto frame = context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	static uint64_t checked = UINT64_MAX;
	static bool     on      = false;
	static uint64_t first   = 0;
	if (checked != frame) {
		checked = frame;
		std::error_code error;
		const bool now_on = std::filesystem::exists(trigger, error);
		if (now_on && !on) {
			first = frame;
		}
		on = now_on;
	}
	if (!on || frame >= first + 12) {
		return;
	}
	auto& cache = context.GetTextureCache();
	if (!cache.GetImage(state.depth_info.image_id).info.HasStencil()) {
		return;
	}
	cache.WatchImageDump(state.depth_info.image_id, frame, ("_" + name.substr(0, 8)).c_str());
}

// KYTY_POPS_NOINTERLOCK=1 (v438zh): what became of each draw whose pixel shader has a fragment
// interlock (ExecutePreparedDraw). [pops-nointerlock]: every distinct combination of shaders,
// draw shape and outcome once (the first 64), and the totals every 5 s.
namespace PopsNoInterlock {

enum Outcome : uint32_t {
	Applied,
	MeshPipeline,
	GpuArguments,
	Tessellation,
	NotTriangleList,
	PrimitiveRestart,
	CountNotTriangles,
	PolygonMode,
	Multisampled,
	NoSharedWrites,
	VertexStageWrites,
	NoWriteBarrier,
	BarrierSink,
	TooManyTriangles,
	TooManyPerInstance,
	QueryAroundDraw,
	ReadsPrimitiveId,
	PipelineNotReady,
	OutcomeCount,
};

static constexpr const char* OutcomeText[OutcomeCount] = {
    "no interlock",
    "mesh pipeline",
    "GPU-read arguments",
    "tessellated",
    "not a triangle list",
    "primitive restart",
    "count not a multiple of 3",
    "polygon mode not fill",
    "multisampled",
    "pixel shader writes no shared memory",
    "vertex stage writes memory",
    "no barrier after writing draws",
    "barrier sink",
    "more triangles than KYTY_POPS_NOINTERLOCK_MAX",
    "more triangles per instance than KYTY_POPS_NOINTERLOCK_TRIS",
    "query around the draw",
    "pixel shader reads the primitive id",
    "twin pipeline not ready",
};

static void Note(Outcome outcome, uint64_t ps, uint64_t vs, const DrawCallInfo& draw, uint64_t triangles) {
	static std::mutex                             mutex;
	static std::unordered_set<uint64_t>           seen;
	static std::array<uint64_t, OutcomeCount>     counts {};
	static uint64_t                               recorded = 0;
	static std::chrono::steady_clock::time_point last     = std::chrono::steady_clock::now();
	std::lock_guard lock(mutex);
	counts[outcome]++;
	if (outcome == Applied) {
		recorded += triangles;
	}
	uint64_t key = 1469598103934665603ull;
	for (const uint64_t value: {ps, vs, uint64_t {draw.index_count}, uint64_t {draw.instance_count},
	                            uint64_t {draw.IsIndexed() ? 1u : 0u}, uint64_t {outcome}}) {
		key = (key ^ value) * 1099511628211ull;
	}
	if (seen.size() < 64u && seen.insert(key).second) {
		if (outcome == Applied) {
			std::fprintf(stderr,
			             "[pops-nointerlock] ps=%016llx vs=%016llx %s count=%u inst=%u: no interlock, %llu "
			             "single-triangle draw%s\n",
			             static_cast<unsigned long long>(ps), static_cast<unsigned long long>(vs), draw.Name(),
			             draw.index_count, draw.instance_count, static_cast<unsigned long long>(triangles),
			             triangles == 1 ? "" : "s with the write barrier between");
		} else {
			std::fprintf(stderr, "[pops-nointerlock] ps=%016llx vs=%016llx %s count=%u inst=%u: interlock kept (%s)\n",
			             static_cast<unsigned long long>(ps), static_cast<unsigned long long>(vs), draw.Name(),
			             draw.index_count, draw.instance_count, OutcomeText[outcome]);
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - last).count();
	last                 = now;
	uint64_t    kept = 0;
	std::string reasons;
	for (uint32_t i = Applied + 1u; i < OutcomeCount; i++) {
		if (counts[i] != 0) {
			kept += counts[i];
			reasons += fmt::format("{}{} {}", reasons.empty() ? "" : ", ", OutcomeText[i], counts[i]);
		}
	}
	std::fprintf(stderr,
	             "[pops-nointerlock] %.1f s: %llu draws without interlock (as %llu single-triangle draws); "
	             "interlock kept on %llu%s%s\n",
	             seconds, static_cast<unsigned long long>(counts[Applied]), static_cast<unsigned long long>(recorded),
	             static_cast<unsigned long long>(kept), reasons.empty() ? "" : ": ", reasons.c_str());
	counts   = {};
	recorded = 0;
}

} // namespace PopsNoInterlock

// KYTY_POPS_BUFFER_COHERENT=0 (v438zi): the draws of pixel shaders that have a plain-buffer twin
// (pipelineCache.h). [pops-buffers]: each shader pair once (the first 64), totals every 5 s.
static void NotePopsPlainBuffers(bool taken, const char* kept, uint64_t ps, uint64_t vs) {
	static std::mutex                             mutex;
	static std::unordered_set<uint64_t>           seen;
	static uint64_t                               with_twin = 0, original = 0;
	static std::chrono::steady_clock::time_point last      = std::chrono::steady_clock::now();
	std::lock_guard lock(mutex);
	(taken ? with_twin : original)++;
	if (seen.size() < 64u && seen.insert((ps * 0x9e3779b97f4a7c15ull) ^ vs ^ (taken ? 1u : 0u)).second) {
		std::fprintf(stderr, "[pops-buffers] ps=%016llx vs=%016llx: %s%s%s\n", static_cast<unsigned long long>(ps),
		             static_cast<unsigned long long>(vs), taken ? "plain-buffer twin" : "original module (",
		             taken ? "" : kept, taken ? "" : ")");
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	std::fprintf(stderr, "[pops-buffers] %.1f s: %llu draws with the plain-buffer twin, %llu with the original module\n",
	             std::chrono::duration<double>(now - last).count(), static_cast<unsigned long long>(with_twin),
	             static_cast<unsigned long long>(original));
	last      = now;
	with_twin = 0;
	original  = 0;
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	const CpuCost::Phase cpu_phase_("ExecutePreparedDraw");
	KYTY_PROFILER_FUNCTION();
	auto& ucfg = buffer.GetUserConfig();
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	// KYTY_VERTEX_VERIFY=<n>: every n-th draw's bound ranges are compared with guest memory
	// (BufferCache::VertexVerifyRange). Disarmed on every way out, so no dispatch is checked.
	struct VertexVerifyScope {
		BufferCache* cache = nullptr;
		~VertexVerifyScope() {
			if (cache != nullptr) {
				cache->VertexVerifyDisarm();
			}
		}
	} vertex_verify;
	if (BufferCache::VertexVerifyInterval() != 0) {
		vertex_verify.cache = &m_context.GetBufferCache();
		vertex_verify.cache->VertexVerifyArm();
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		// Counts read on the GPU (DRAW_INDIRECT): one group per vertex, as fast launch runs.
		EXIT_NOT_IMPLEMENTED(emit.gpu_args_address != 0 && !mesh.fast_launch);
	}
	if (mesh_active && emit.gpu_args_address == 0) {
		const auto& mesh       = state.vertex_info[0].mesh;
		const auto  primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (draw.instance_count > limits.maxMeshWorkGroupCount[1]) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	vk::Buffer     mesh_command_buffer = nullptr;
	vk::DeviceSize mesh_command_offset = 0;
	Buffer*        gpu_args_buffer = nullptr;
	if (emit.gpu_args_address != 0) {
		KYTY_PROFILER_BLOCK("Draw::GpuArgs");
		const CpuCost::Phase cpu_draw_phase_("Draw::GpuArgs");
		// The mesh shader reads the start vertex there, and the counts are copied from there:
		// synchronize and register the arguments like the guest indices below.
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    emit.gpu_args_address, 16u, false, false, {}, true);
		gpu_args_buffer = args_buffer;
		if (mesh_active && args_buffer != nullptr && args_buffer->HasDeviceAddress()) {
			// The counts become the mesh-tasks command on the GPU, validated against the device's
			// limits (MeshIndirectArgs), before the draw's bindings are committed.
			if (m_mesh_args == nullptr) {
				m_mesh_args = std::make_unique<MeshIndirectArgs>(m_context.GetGraphics(),
				                                                 m_context.GetCommandScheduler());
			}
			if (!ConvertMeshArgsRun(m_context, *m_mesh_args, buffer, emit.gpu_args_address,
			                        mesh_command_buffer, mesh_command_offset)) {
				m_context.GetCommandScheduler().EndRendering();
				const auto converted = m_mesh_args->Convert(
				    buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset);
				mesh_command_buffer = converted.buffer;
				mesh_command_offset = converted.offset;
				MeshArgsRun().single++;
			}
		}
	}
	if (emit.gpu_instance_count_address != 0) {
		(void)m_context.GetBufferCache().ObtainBuffer(emit.gpu_instance_count_address, 4u, false,
		                                              false, {}, true);
	}
	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		const auto& clip               = state.vertex_info[i].clip_space;
		bindings.vertex[i].clip_space = {clip.scale[0],  clip.scale[1],       clip.offset[0],
		                                 clip.offset[1], clip.half_extent[0], clip.half_extent[1]};
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		auto& mapping = bindings.pixel->export_mapping;
		mapping       = {0u, 0u};
		for (uint32_t target = 0; target < 8u; target++) {
			mapping[target / 4u] |= static_cast<uint32_t>(state.ps_input_info.target_export_mapping[target].packed)
			                        << ((target % 4u) * 8u);
		}
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	{
		KYTY_PROFILER_BLOCK("Draw::PrepareGraphicsBindings");
		const CpuCost::Phase cpu_draw_phase_("Draw::PrepareGraphicsBindings");
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		KYTY_PROFILER_BLOCK("Draw::VertexIndexBuffers");
		const CpuCost::Phase cpu_draw_phase_("Draw::VertexIndexBuffers");
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	// KYTY_REBIND_AFTER_JOIN: a lookup after the bindings (another stage's, the vertex or index
	// range) joined a buffer they resolved; resolve them and the per-draw ranges again.
	if (RebindAfterJoin(stages, m_rebind_epoch, "draw") && !mesh_active) {
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
		(void)RebindAfterJoin(stages, m_rebind_epoch, "draw (after its vertex/index ranges)");
	}
	if (vertex_verify.cache != nullptr && vertex_verify.cache->VertexVerifyArmed() && !mesh_active) {
		const auto& input  = state.vertex_info[0];
		const auto  shader = input.stage.program->shader_hash;
		for (uint32_t i = 0; i < vertex_bindings.count; i++) {
			if (vertex_bindings.sizes[i] != 0) {
				vertex_verify.cache->VertexVerifyRange("vertex buffer", i, shader, input.buffers[i].addr,
				                                       vertex_bindings.sizes[i], vertex_bindings.buffers[i],
				                                       vertex_bindings.offsets[i]);
			}
		}
		// An index buffer the host built (host_data) has no guest bytes to compare with.
		if (index_source.host_data == nullptr && index_source.size != 0) {
			vertex_verify.cache->VertexVerifyRange("index buffer", 0, shader, index_source.address,
			                                       index_source.size, index_binding.buffer,
			                                       index_binding.offset);
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// A draw that writes no memory may be skipped while its new pipeline compiles in the
	// background; its render targets are not touched. One that writes buffers or storage images
	// waits for the pipeline, as others may read what it writes.
	const auto writes_memory = [](const ShaderStageRuntime& runtime) {
		return HasShaderBufferWrites(runtime) ||
		       std::ranges::any_of(runtime.program->info.images, [](const auto& image) {
			       return image.written &&
			              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		       });
	};
	bool may_defer = !(state.ps_active && writes_memory(state.ps_input_info.stage));
	for (const auto& stage: vertex_stages) {
		may_defer = may_defer && !writes_memory(stage.stage);
	}
	// Glyphs and cached layers: drawn once into small targets (SmallColorTargetDraw).
	may_defer = may_defer && !SmallColorTargetDraw(buffer.GetRegisters());
	// KYTY_POPS_NOINTERLOCK=1 (v438zh, experiment, off by default). A fragment interlock orders
	// and excludes the fragments that land on one pixel. One triangle, rasterized with one sample
	// per pixel in fill mode, makes at most one fragment per pixel, so within a draw of one
	// triangle the interlock has nothing to order or exclude; what the draw sees of earlier draws
	// is settled by the barrier a draw that writes shared memory gets after it (below), with or
	// without the interlock. So a triangle-list draw of such a shader is recorded as one draw per
	// triangle, in primitive order (instance by instance, triangle by triangle: the order the
	// ordered interlock gives inside the single draw), with that same barrier between them, and
	// with the pipeline of the shader's twin module without the interlock (PopsNoInterlockEnabled,
	// pipelineCache.h). Each triangle is the same primitive as before: the same indices (the
	// first index moves by 3), the same gl_VertexIndex and gl_InstanceIndex, the same provoking
	// vertex; the recompiler gives a pixel shader no primitive id (a twin whose SPIR-V reads one
	// is only taken for single-triangle draws).
	// The interlock is kept, and the draw recorded as before, for: mesh pipelines, GPU-read
	// arguments, tessellation, any topology but a triangle list (strips and fans share vertices,
	// rect lists are tessellated, quads are fans), primitive restart, polygon modes other than
	// fill (a triangle's lines and points can touch one pixel twice), more than one sample per
	// pixel (clipped pieces of one triangle may both cover a pixel), a vertex stage that writes
	// memory (it would run more often), more than KYTY_POPS_NOINTERLOCK_TRIS triangles per instance
	// (default 2) or KYTY_POPS_NOINTERLOCK_MAX over all instances (default 16), a counted
	// occlusion query or draw statistics around more than one triangle (a
	// query cannot span rendering instances), runs without the barrier after writing draws, the
	// barrier sink, and while the twin's pipeline is not ready. [pops-nointerlock] says which.
	uint64_t                        pops_triangles = 0; // > 0: one draw per triangle, no interlock
	PipelineCache::GraphicsPrograms pops_programs;
	if (PopsNoInterlockEnabled() && state.ps_active && state.programs.pixel.no_interlock_module) {
		static const uint64_t max_triangles = [] {
			const char* value = std::getenv("KYTY_POPS_NOINTERLOCK_MAX");
			return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {16};
		}();
		// v438zi: at most this many triangles per instance (default 2: a quad). m15: with every
		// list of up to 16 triangles taken, the decal volumes (boxes, 12 triangles, a few fragments
		// each) became ~1,900 one-triangle draws a frame, each behind a barrier that drains the
		// one before: about +10 ms of GPU a frame where they are drawn, for nothing. A box
		// overlaps itself by design; the draws worth taking are the screen-sized ones, which are
		// drawn as one or two triangles.
		static const uint64_t max_per_instance = [] {
			const char* value = std::getenv("KYTY_POPS_NOINTERLOCK_TRIS");
			return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {2};
		}();
		static const bool write_barrier = [] {
			const char* value = std::getenv("KYTY_DRAW_WRITE_BARRIER");
			return value == nullptr || value[0] != '0';
		}();
		const auto samples = [&] {
			// As the pipeline's rasterization samples (PipelineCache::TryGetGraphicsPipeline).
			uint32_t count = 0;
			for (uint32_t i = 0; i < state.color_count; i++) {
				count = std::max<uint32_t>(count, state.color_info[i].desc.info.samples);
			}
			const bool with_depth = state.depth_info.desc.view_info.format != vk::Format::eUndefined &&
			                        static_cast<bool>(state.depth_info.image_id);
			if (with_depth) {
				count = std::max<uint32_t>(count, state.depth_info.desc.info.samples);
			}
			if (state.color_count == 0 && !with_depth) {
				count = render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
			}
			return count;
		};
		const uint64_t triangles = uint64_t {draw.index_count / 3u} * draw.instance_count;
		auto           outcome   = PopsNoInterlock::Applied;
		if (mesh_active) {
			outcome = PopsNoInterlock::MeshPipeline;
		} else if (emit.gpu_args_address != 0 || emit.gpu_instance_count_address != 0) {
			outcome = PopsNoInterlock::GpuArguments;
		} else if (vertex_stages.size() != 1) {
			outcome = PopsNoInterlock::Tessellation;
		} else if (ucfg.GetPrimType() != Prospero::PrimitiveType::kTriList ||
		           topology != vk::PrimitiveTopology::eTriangleList) {
			outcome = PopsNoInterlock::NotTriangleList;
		} else if (primitive_restart_enable) {
			outcome = PopsNoInterlock::PrimitiveRestart;
		} else if (triangles == 0 || draw.index_count % 3u != 0) {
			outcome = PopsNoInterlock::CountNotTriangles;
		} else if (DrawDynamicRaster(buffer.GetRegisters(), false).polygon_mode != vk::PolygonMode::eFill) {
			outcome = PopsNoInterlock::PolygonMode;
		} else if (samples() != 1u) {
			outcome = PopsNoInterlock::Multisampled;
		} else if (!HasShaderBufferWrites(state.ps_input_info.stage)) {
			// Without them the draw gets no barrier after it.
			outcome = PopsNoInterlock::NoSharedWrites;
		} else if (writes_memory(vertex_stages[0].stage) || vertex_stages[0].stage.program->has_address_writes) {
			outcome = PopsNoInterlock::VertexStageWrites;
		} else if (!write_barrier || SyncToggle::SkipDrawWriteBarrier()) {
			outcome = PopsNoInterlock::NoWriteBarrier;
		} else if (BarrierSinkEnabled()) {
			outcome = PopsNoInterlock::BarrierSink;
		} else if (draw.index_count / 3u > max_per_instance) {
			outcome = PopsNoInterlock::TooManyPerInstance;
		} else if (triangles > max_triangles) {
			outcome = PopsNoInterlock::TooManyTriangles;
		} else if (triangles > 1 && (Occlusion::CountsDraw(buffer) || DrawStats::Enabled())) {
			outcome = PopsNoInterlock::QueryAroundDraw;
		} else if (draw.index_count > 3u &&
		           (state.programs.pixel.no_interlock_flags & PopsTwinReadsPrimitiveId) != 0) {
			outcome = PopsNoInterlock::ReadsPrimitiveId;
		}
		if (outcome == PopsNoInterlock::Applied) {
			pops_triangles = triangles;
			pops_programs  = state.programs;
			auto& pixel    = pops_programs.pixel;
			pixel.module   = pixel.no_interlock_module;
			pixel.id |= PopsNoInterlockIdBit;
			pixel.no_interlock_module  = nullptr;
			pixel.plain_buffers_module = nullptr;
		} else {
			PopsNoInterlock::Note(outcome, state.ps_input_info.stage.program->shader_hash,
			                      vertex_stages.back().stage.program->shader_hash, draw, 0);
		}
	}
	const auto get_pipeline = [&](const PipelineCache::GraphicsPrograms& programs) {
		KYTY_PROFILER_BLOCK("Draw::GetPipeline");
		const CpuCost::Phase cpu_draw_phase_("Draw::GetPipeline");
		return m_context.GetPipelineCache().TryGetGraphicsPipeline(
		    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages,
		    buffer, state.ps_active ? &state.ps_input_info : nullptr, topology,
		    primitive_restart_enable, programs, may_defer);
	};
	// KYTY_POPS_BUFFER_COHERENT=0 (v438zi, experiment, off by default): every other draw of a
	// primitive-ordered pixel shader that only reads its storage buffers takes the twin module
	// whose buffer arrays are not Coherent (pipelineCache.h). The interlock stays and the draw is
	// recorded as it is: mesh pipelines, GPU-read arguments and any topology included. Not when a
	// vertex stage of the draw writes memory (then something in the draw does write).
	bool        pops_plain_buffers = false;
	const char* pops_plain_kept    = nullptr;
	if (pops_triangles == 0 && PopsPlainBuffersEnabled() && state.ps_active &&
	    state.programs.pixel.plain_buffers_module) {
		for (const auto& stage: vertex_stages) {
			if (writes_memory(stage.stage) || stage.stage.program->has_address_writes) {
				pops_plain_kept = "a vertex stage writes memory";
			}
		}
		if (pops_plain_kept == nullptr) {
			pops_plain_buffers = true;
			pops_programs      = state.programs;
			auto& pixel        = pops_programs.pixel;
			pixel.module       = pixel.plain_buffers_module;
			pixel.id |= PopsPlainBuffersIdBit;
			pixel.no_interlock_module  = nullptr;
			pixel.plain_buffers_module = nullptr;
		}
	}
	auto* deferred_pipeline =
	    get_pipeline(pops_triangles != 0 || pops_plain_buffers ? pops_programs : state.programs);
	if (pops_triangles != 0) {
		if (deferred_pipeline == nullptr) {
			// The twin's pipeline did not compile within the wait: this draw is recorded as
			// before, with the interlock.
			pops_triangles    = 0;
			deferred_pipeline = get_pipeline(state.programs);
		}
		PopsNoInterlock::Note(pops_triangles != 0 ? PopsNoInterlock::Applied : PopsNoInterlock::PipelineNotReady,
		                      state.ps_input_info.stage.program->shader_hash,
		                      vertex_stages.back().stage.program->shader_hash, draw, pops_triangles);
	} else if (pops_plain_buffers || pops_plain_kept != nullptr) {
		if (pops_plain_buffers && deferred_pipeline == nullptr) {
			pops_plain_buffers = false;
			pops_plain_kept    = "twin pipeline not ready";
			deferred_pipeline  = get_pipeline(state.programs);
		}
		NotePopsPlainBuffers(pops_plain_buffers, pops_plain_kept != nullptr ? pops_plain_kept : "",
		                     state.ps_input_info.stage.program->shader_hash,
		                     vertex_stages.back().stage.program->shader_hash);
	}
	// Only the generic meshlet shader has the user-data layout LogMeshNear walks; other mesh shaders
	// would make it read through unrelated descriptors.
	if (static const bool mesh_near = std::getenv("KYTY_MESH_NEAR") != nullptr;
	    mesh_near && mesh_active && state.vertex_info[0].stage.program->shader_hash == 0x8fb896cb9412ab56ull) {
		const auto& vs = buffer.GetShaders().GetVs();
		LogMeshNear(Sync::GuestFlipCount(), vs.gs_user_sgpr.value, vs.gs_user_sgpr.count, draw.index_count,
		            draw.first_instance, state.vertex_info[0].stage.program->shader_hash,
		            state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0);
	}
	// KYTY_LOG_PS_DRAWS=<ps hash> (diagnostics): every draw of that pixel shader, with the guest
	// flip, its counts, whether the GPU reads its arguments, and the NGG user data slots 16..18
	// (30 lines a flip at most).
	if (static const std::string log_ps = [] {
		    const char* value = std::getenv("KYTY_LOG_PS_DRAWS");
		    return value != nullptr ? std::string(value) : std::string();
	    }();
	    !log_ps.empty() && state.ps_active &&
	    log_ps.find(fmt::format("{:016x}", state.ps_input_info.stage.program->shader_hash)) !=
	        std::string::npos) {
		static std::atomic<uint64_t> last_flip {0};
		static std::atomic<uint32_t> lines {0};
		const auto                   flip = Sync::GuestFlipCount();
		if (last_flip.exchange(flip) != flip) lines.store(0);
		if (lines.fetch_add(1) < 30) {
			const auto& users = buffer.GetShaders().GetVs().gs_user_sgpr.value;
			std::fprintf(stderr,
			             "[ps-draw] flip=%llu %s count=%u instances=%u first_instance=%u gpu_args=0x%llx "
			             "start_sgpr=%u vs=%016llx mesh=%d s16=%08x s17=%08x s18=%08x\n",
			             static_cast<unsigned long long>(flip), draw.Name(), draw.index_count,
			             draw.instance_count, draw.first_instance,
			             static_cast<unsigned long long>(emit.gpu_args_address), emit.start_instance_sgpr,
			             static_cast<unsigned long long>(state.vertex_info[0].stage.program->shader_hash),
			             mesh_active ? 1 : 0, users[16], users[17], users[18]);
			{
				const auto& regs = buffer.GetRegisters();
				const auto& dc   = regs.GetDepthControl();
				const auto& mc   = regs.GetModeControl();
				const auto& po   = regs.GetPolyOffset();
				const auto& drt  = regs.GetDepthRenderTarget();
				const auto& ds   = state.depth_info;
				std::fprintf(stderr,
				             "[ps-draw-ds] flip=%llu z=%d zw=%d zfunc=%u bounds=%d stencil=%d sfunc=%u/%u "
				             "cull=%d/%d face=%d poly_off=%d/%d scale=%g/%g offset=%g/%g clamp=%g "
				             "neg_bits=%d float=%d zbase=0x%llx zfmt=%u host_zfmt=%u test=%d write=%d op=%u "
				             "minmax=%g/%g\n",
				             static_cast<unsigned long long>(flip), dc.z_enable ? 1 : 0,
				             dc.z_write_enable ? 1 : 0, static_cast<unsigned>(dc.zfunc),
				             dc.depth_bounds_enable ? 1 : 0, dc.stencil_enable ? 1 : 0,
				             static_cast<unsigned>(dc.stencilfunc), static_cast<unsigned>(dc.stencilfunc_bf),
				             mc.cull_front ? 1 : 0, mc.cull_back ? 1 : 0, mc.face ? 1 : 0,
				             mc.poly_offset_front_enable ? 1 : 0, mc.poly_offset_back_enable ? 1 : 0,
				             po.front_scale, po.back_scale, po.front_offset, po.back_offset, po.clamp,
				             po.neg_num_db_bits, po.db_is_float_fmt ? 1 : 0,
				             static_cast<unsigned long long>(drt.z_read_base_addr),
				             static_cast<unsigned>(drt.z_info.format),
				             static_cast<unsigned>(ds.desc.view_info.format), ds.depth_test_enable ? 1 : 0,
				             ds.depth_write_enable ? 1 : 0, static_cast<unsigned>(ds.depth_compare_op),
				             ds.depth_min_bounds, ds.depth_max_bounds);
				for (uint32_t i = 0; i < 8; i++) {
					const auto& rt = regs.GetRenderTarget(i);
					const auto& bc = regs.GetBlendControl(i);
					if (rt.base.addr == 0 || ((regs.GetRenderTargetMask() >> (i * 4u)) & 0xfu) == 0) continue;
					std::fprintf(stderr,
					             "[ps-draw-rt] flip=%llu rt%u addr=0x%llx fmt=%u nfmt=%u %ux%u mask=%x bypass=%d "
					             "blend=%d src=%u dst=%u comb=%u sep=%d asrc=%u adst=%u acomb=%u\n",
					             static_cast<unsigned long long>(flip), i,
					             static_cast<unsigned long long>(rt.base.addr),
					             static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
					             rt.attrib2.width + 1, rt.attrib2.height + 1,
					             (regs.GetRenderTargetMask() >> (i * 4u)) & 0xfu, rt.info.blend_bypass ? 1 : 0,
					             bc.enable ? 1 : 0, bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
					             bc.separate_alpha_blend ? 1 : 0, bc.alpha_srcblend, bc.alpha_destblend,
					             bc.alpha_comb_fcn);
				}
			}
			const auto& ps_regs  = buffer.GetShaders().GetPs();
			const auto& ps_users = ps_regs.ps_user_sgpr.value;
			std::fprintf(stderr,
			             "[ps-draw-ud] flip=%llu n=%u rsrc2=%u s24..31: %08x %08x %08x %08x %08x %08x %08x %08x\n",
			             static_cast<unsigned long long>(flip), ps_regs.ps_user_sgpr.count,
			             static_cast<uint32_t>(ps_regs.ps_regs.rsrc2.user_sgpr), ps_users[24], ps_users[25],
			             ps_users[26], ps_users[27], ps_users[28], ps_users[29], ps_users[30], ps_users[31]);
			// What each stage binds, and the dwords of its first (small, per-draw) buffer.
			{
				auto& cache = m_context.GetTextureCache();
				for (const auto* stage: stages) {
					if (stage == nullptr || stage->runtime == nullptr || stage->runtime->program == nullptr) continue;
					std::string text;
					for (const auto& binding: stage->images) {
						const auto* image = cache.TryGetImage(binding.image_id);
						if (image == nullptr) continue;
						text += fmt::format(" img{}:{:x}:{}x{}:f{}", 
						                    binding.desc.type == TextureCache::BindingType::Storage ? "S" : "T",
						                    image->info.data.address, image->info.extent.width,
						                    image->info.extent.height, static_cast<uint32_t>(image->backing.format));
					}
					for (const auto& src: stage->buffer_sources) {
						text += fmt::format(" buf:{:x}+{:x}", src.address, src.size);
					}
					if (!stage->buffer_sources.empty() && stage->buffer_sources[0].size <= 0x100 &&
					    stage->buffer_sources[0].address != 0) {
						const auto* words =
						    reinterpret_cast<const uint32_t*>(stage->buffer_sources[0].address);
						text += " |";
						for (uint64_t i = 0; i < stage->buffer_sources[0].size / 4; i++) {
							text += fmt::format(" {:08x}", words[i]);
						}
					}
					std::fprintf(stderr, "[ps-draw-res] flip=%llu sh=%016llx%s\n",
					             static_cast<unsigned long long>(flip),
					             static_cast<unsigned long long>(stage->runtime->program->shader_hash),
					             text.c_str());
				}
			}
			if (mesh_active && MeshWalkEnabled()) {
				const auto& vs = buffer.GetShaders().GetVs();
				LogMeshWalk(flip, vs.gs_user_sgpr.value, vs.gs_user_sgpr.count, draw.index_count,
				            draw.first_instance, m_context.GetBufferCache());
			}
			// KYTY_WATCH_DRAW=<flip>:<ES user sgpr>:<offset> (dec:hex:hex): at the first logged draw
			// from that guest flip on, put a hardware write watch on the buffer the V# in that user
			// sgpr pair points at, + offset (who on the CPU writes this draw's constants).
			if (static const char* watch_draw = std::getenv("KYTY_WATCH_DRAW"); watch_draw != nullptr) {
				static std::atomic_bool done {false};
				unsigned long long      from = 0;
				unsigned                sgpr = 0;
				unsigned long long      off  = 0;
				if (!done.load() && std::sscanf(watch_draw, "%llu:%x:%llx", &from, &sgpr, &off) == 3 &&
				    flip >= from) {
					const auto& vs = buffer.GetShaders().GetVs();
					if (sgpr + 1 < vs.gs_user_sgpr.count && !done.exchange(true)) {
						const uint64_t base = vs.gs_user_sgpr.value[sgpr] |
						                      (static_cast<uint64_t>(vs.gs_user_sgpr.value[sgpr + 1] & 0xffffu) << 32u);
						Common::TracePoints::SetWatch(base + off, 4, 0x60);
					}
				}
			}
			if (!SlotDumps().empty()) {
				const auto& vs = buffer.GetShaders().GetVs();
				LogSlotDumps(flip, vs.gs_user_sgpr.value, vs.gs_user_sgpr.count,
				             m_context.GetBufferCache());
			}
		}
	}
	if (const auto watched = RtDrawLog::Address(); watched != 0) {
		uint32_t slots = 0;
		for (uint32_t i = 0; i < state.color_count; i++) {
			if (state.color_info[i].desc.info.data.address == watched) {
				slots |= 1u << state.color_info[i].target_slot;
			}
		}
		if (slots != 0) {
			const auto& d  = state.depth_info;
			const auto& sf = d.stencil_front;
			char        text[320];
			std::snprintf(
			    text, sizeof(text),
			    "slots=%x n=%u depth=0x%llx test=%d write=%d op=%d bounds=%d stencil=%d cmp=%d "
			    "ref=%u rmask=%x wmask=%x pass=%d fail=%d zfail=%d %s",
			    slots, state.color_count,
			    static_cast<unsigned long long>(d.desc.info.data.address), d.depth_test_enable ? 1 : 0,
			    d.depth_write_enable ? 1 : 0, static_cast<int>(d.depth_compare_op),
			    d.depth_bounds_test_enable ? 1 : 0, d.stencil_test_enable ? 1 : 0,
			    static_cast<int>(sf.compareOp), sf.reference, sf.compareMask, sf.writeMask,
			    static_cast<int>(sf.passOp), static_cast<int>(sf.failOp),
			    static_cast<int>(sf.depthFailOp), draw.Name());
			uint64_t state_hash = 1469598103934665603ull;
			for (const char* c = text; *c != '\0'; c++) {
				state_hash = (state_hash ^ static_cast<uint8_t>(*c)) * 1099511628211ull;
			}
			RtDrawLog::NoteDraw(state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0,
			                    state_hash, deferred_pipeline == nullptr, text);
		}
	}
	if (deferred_pipeline == nullptr) {
		if (Common::FrameSkips::Enabled()) {
			Common::FrameSkips::Skip(Common::FrameSkips::NoPipeline);
			if (state.ps_active && Common::FrameSkips::Watched(state.ps_input_info.stage.program->shader_hash)) {
				Common::FrameSkips::g_watched_nopipe.fetch_add(1, std::memory_order_relaxed);
			}
		}
		return;
	}
	auto& pipeline = *deferred_pipeline;
	if (Common::FrameSkips::Enabled()) {
		Common::FrameSkips::g_draws.fetch_add(1, std::memory_order_relaxed);
		if (state.ps_active && Common::FrameSkips::Watched(state.ps_input_info.stage.program->shader_hash)) {
			Common::FrameSkips::g_watched.fetch_add(1, std::memory_order_relaxed);
		}
	}
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering = [&] {
		KYTY_PROFILER_BLOCK("Draw::AcquireRenderTargets");
		return AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
		                            feedback_aspects, stages);
	}();

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. Recorded through the recording thread when it runs.
	const auto vk_buffer = buffer.Recorder();
	KYTY_PROFILER_BLOCK("Draw::Commit");
	// KYTY_CPU_COST: the commit of the bindings and state up to the recording, then the recording.
	std::optional<CpuCost::Phase> cpu_commit_phase_(std::in_place, "Draw::Commit");
	const uint64_t debug_shaders =
	    (vertex_stages.back().stage.program->shader_hash << 32u) |
	    (state.ps_active ? state.ps_input_info.stage.program->shader_hash & 0xffffffffull : 0ull);
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u, debug_shaders);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u, debug_shaders);
	}
	// KYTY_GPL: a linked pipeline (independent-sets layout) is bound before its sets and push
	// constants too: they were recorded against the previous pipeline (often a monolithic one
	// with an incompatible layout), and linked pipelines read garbage set addresses (v436q,
	// fixed-address faults). KYTY_GPL_BIND_FIRST=0 restores the old order.
	static const bool bind_first = [] {
		const char* value = std::getenv("KYTY_GPL_BIND_FIRST");
		return value == nullptr || value[0] != '0';
	}();
	if (bind_first && pipeline.independent_sets) {
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.Bound());
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (mesh_active) {
		// GPU arguments: dword 3 = 0xFFFFFFFF tells the fast-launch entry to read the start vertex
		// at dwords 4-5's address + 8 (the shader's other draw dwords are unused then).
		const bool     gpu_args = emit.gpu_args_address != 0;
		const uint32_t draw_data[] {
		    gpu_args ? 0u : draw.index_count,
		    gpu_args ? 0u
		             : draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset)
		                                : emit.first_vertex,
		    gpu_args ? emit.start_instance_sgpr : emit.first_instance,
		    gpu_args ? 0xFFFFFFFFu : index_source.guest_element_size,
		    static_cast<uint32_t>(gpu_args ? emit.gpu_args_address : index_source.address),
		    static_cast<uint32_t>((gpu_args ? emit.gpu_args_address : index_source.address) >> 32u),
		    0u};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        pipeline.push_stages ? pipeline.push_stages
		                                             : vk::ShaderStageFlagBits::eMeshEXT |
		                                                   vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), state.depth_info, rendering);
	// Diagnostics: KYTY_LOG_TESS_DRAWS=1 prints each distinct tessellated draw once ([tess-draw]:
	// every vertex-stage hash, the pixel shader, depth state, colour targets, sizes), then counts
	// per combination every 10 s, to find all passes that draw a tessellated surface.
	if (static const bool log_tess = std::getenv("KYTY_LOG_TESS_DRAWS") != nullptr;
	    log_tess && buffer.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kPatch) {
		std::string stages_text;
		for (const auto& stage: vertex_stages) {
			stages_text += fmt::format(
			    "{}{:016x}", stages_text.empty() ? "" : "/",
			    stage.stage.program != nullptr ? stage.stage.program->shader_hash : 0ull);
		}
		const uint64_t ps = state.ps_active && state.ps_input_info.stage.program != nullptr
		                        ? state.ps_input_info.stage.program->shader_hash
		                        : 0ull;
		const auto& depth = state.depth_info;
		const auto  key   = fmt::format("vs={} ps={:016x} ztest={} zwrite={} zop={} colors={} "
		                                "idx={} inst={}",
		                                stages_text, ps, depth.depth_test_enable ? 1 : 0,
		                                depth.depth_write_enable ? 1 : 0,
		                                static_cast<int>(depth.depth_compare_op), state.color_count,
		                                draw.index_count, draw.instance_count);
		static std::mutex                               tess_mutex;
		static std::map<std::string, uint64_t>          tess_counts;
		static int64_t                                  tess_printed_ms = 0;
		std::lock_guard                                 lock(tess_mutex);
		if (tess_counts[key]++ == 0) {
			std::fprintf(stderr, "[tess-draw] new %s\n", key.c_str());
		}
		const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		if (now - tess_printed_ms > 10000) {
			tess_printed_ms = now;
			for (auto& [text, count]: tess_counts) {
				if (count != 0) {
					std::fprintf(stderr, "[tess-draw] %llu x %s\n",
					             static_cast<unsigned long long>(count), text.c_str());
				}
			}
		}
	}
	LogAndDiagnoseFacing(buffer, vk_buffer,
	                     state.ps_active && state.ps_input_info.stage.program != nullptr
	                         ? state.ps_input_info.stage.program->shader_hash
	                         : 0ull);
	if (state.ps_active) {
		if (const int paint = PaintDraws::Index(state.ps_input_info.stage.program->shader_hash);
		    paint >= 0) {
			const auto color = PaintDraws::Color(paint);
			vk_buffer.setBlendConstants(color.data());
		}
	}
	if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
		vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	}

	// GPU arguments: the group counts (vertex count, instance count, 1) go into a ring entry for
	// drawMeshTasksIndirectEXT, outside the render pass; the arguments are never read on the CPU.
	vk::Buffer     mesh_indirect_buffer = nullptr;
	vk::DeviceSize mesh_indirect_offset = 0;
	constexpr uint32_t MeshIndirectEntries = 4096;
	if (mesh_active && emit.gpu_args_address != 0 && mesh_command_buffer) {
		mesh_indirect_buffer = mesh_command_buffer;
		mesh_indirect_offset = mesh_command_offset;
	} else if (mesh_active && emit.gpu_args_address != 0) {
		auto&              cache = m_context.GetBufferCache();
		auto&              args  = cache.GetBuffer(cache.FindBuffer(emit.gpu_args_address, 16u));
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal,
			    0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
			    MeshIndirectEntries * 16u);
		}
		mesh_indirect_buffer = m_mesh_indirect->Handle();
		mesh_indirect_offset = (m_mesh_indirect_next++ % MeshIndirectEntries) * 16u;
		m_context.GetCommandScheduler().EndRendering();
		// The arguments' writer (a dispatch or a copy) and the entry's last indirect read come
		// before the copy.
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eIndirectCommandRead;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		const vk::BufferCopy copy {args.Offset(emit.gpu_args_address), mesh_indirect_offset, 8u};
		vk_buffer.copyBuffer(args.Handle(), mesh_indirect_buffer, 1, &copy);
		vk_buffer.fillBuffer(mesh_indirect_buffer, mesh_indirect_offset + 8u, 4u, 1u);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0,
		                          nullptr, 0, nullptr);
	}

	// Vertex pipeline, counts on the GPU: the arguments' writer (a dispatch or a copy) comes
	// before the indirect read; the arguments are read where the cache holds them.
	vk::Buffer     vertex_indirect_buffer = nullptr;
	vk::DeviceSize vertex_indirect_offset = 0;
	if (!mesh_active && emit.gpu_instance_count_address != 0) {
		// Only the instance count is on the GPU: a ring entry gets the known counts and offsets,
		// and the GPU copies the instance count into it.
		auto& cache  = m_context.GetBufferCache();
		auto& source = cache.GetBuffer(cache.FindBuffer(emit.gpu_instance_count_address, 4u));
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<Buffer>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal,
			    0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
			    MeshIndirectEntries * 16u);
		}
		vertex_indirect_buffer = m_mesh_indirect->Handle();
		vertex_indirect_offset = (m_mesh_indirect_next++ % MeshIndirectEntries) * 16u;
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eIndirectCommandRead;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
		                          0, nullptr);
		const uint32_t command[4] {draw.index_count, 0u, emit.first_vertex, emit.first_instance};
		vk_buffer.updateBuffer(vertex_indirect_buffer, vertex_indirect_offset, sizeof(command),
		                       command);
		const vk::BufferCopy copy {source.Offset(emit.gpu_instance_count_address),
		                           vertex_indirect_offset + sizeof(uint32_t), sizeof(uint32_t)};
		vk_buffer.copyBuffer(source.Handle(), vertex_indirect_buffer, 1, &copy);
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0,
		                          nullptr, 0, nullptr);
	} else if (!mesh_active && emit.gpu_args_address != 0) {
		auto& cache = m_context.GetBufferCache();
		auto& args  = cache.GetBuffer(cache.FindBuffer(emit.gpu_args_address, 16u));
		vertex_indirect_buffer = args.Handle();
		vertex_indirect_offset = args.Offset(emit.gpu_args_address);
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier ready {};
		ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		ready.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &ready, 0,
		                          nullptr, 0, nullptr);
	}

	const bool full_barrier =
	    state.ps_active && FullBarrierPs(state.ps_input_info.stage.program->shader_hash);
	if (full_barrier) {
		m_context.GetCommandScheduler().EndRendering();
		FullMemoryBarrier(vk_buffer);
	}
	if (SyncToggle::NoGuestBarriers()) {
		// Without the guest's global barriers (upstream 70668fbd), a draw whose shaders write
		// buffers waits for earlier accesses to that memory, as dispatches do.
		vk::PipelineStageFlags hazard_stages = {};
		for (const auto& stage: vertex_stages) {
			if (HasShaderBufferWrites(stage.stage)) {
				hazard_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
			}
		}
		if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
			hazard_stages |= vk::PipelineStageFlagBits::eFragmentShader;
		}
		if (hazard_stages) {
			m_context.GetCommandScheduler().EndRendering();
			ShaderWriteHazardBarrier(vk_buffer, hazard_stages);
		}
	}
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u, debug_shaders);
	}
	// v446t, diagnostics: KYTY_LOG_DRAW_TARGETS=<vertex or pixel hash,...>: the render targets of
	// that shader's draws (first 64), for a device loss checkpoints pin on such a draw.
	if (static const std::string targets_hashes = [] {
		    const char* env = std::getenv("KYTY_LOG_DRAW_TARGETS");
		    return env != nullptr ? std::string(env) : std::string();
	    }();
	    !targets_hashes.empty()) {
		const auto vs = vertex_stages.back().stage.program->shader_hash;
		const auto ps = state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0ull;
		// KYTY_LOG_DRAW_TARGETS_NOPS=1: only the draws without a pixel shader.
		static const bool only_no_ps = std::getenv("KYTY_LOG_DRAW_TARGETS_NOPS") != nullptr;
		if ((targets_hashes.find(fmt::format("{:016x}", vs)) != std::string::npos ||
		     (ps != 0 && targets_hashes.find(fmt::format("{:016x}", ps)) != std::string::npos)) &&
		    (!only_no_ps || ps == 0)) {
			// Printed when the targets differ from the last printed draw's (up to 64 prints).
			static std::atomic<uint32_t> printed {0};
			static uint64_t              last_key = 0;
			uint64_t key = (static_cast<uint64_t>(state.depth_info.image_id.index) << 32u) ^
			               (static_cast<uint64_t>(state.depth_info.desc.view_info.base_level) << 24u) ^
			               (static_cast<uint64_t>(state.depth_info.desc.view_info.base_layer) << 16u) ^
			               state.color_count;
			for (uint32_t i = 0; i < state.color_count; i++) {
				key = key * 31u + state.color_info[i].image_id.index;
			}
			const bool changed = key != last_key;
			last_key           = key;
			if (changed && printed.fetch_add(1, std::memory_order_relaxed) < 64u) {
				auto&       cache = m_context.GetTextureCache();
				std::string text  = fmt::format(
                    "[draw-targets] vs={:016x} ps={:016x} n={} inst={} colors={} depth={}\n", vs, ps,
                    draw.index_count, draw.instance_count, state.color_count,
                    static_cast<bool>(state.depth_info.image_id) ? 1 : 0);
				const auto describe = [&](const char* what, ImageId id, const TextureCache::ImageDesc& desc) {
					const auto* image = static_cast<bool>(id) ? cache.TryGetImage(id) : nullptr;
					text += fmt::format(
					    "  {} id={} guest=0x{:x}+0x{:x} stencil=0x{:x}+0x{:x} {}x{} fmt={} vkimage={} sparse={} "
					    "registered={}\n",
					    what, static_cast<bool>(id) ? static_cast<uint32_t>(id.index) : 0xffffffffu,
					    desc.info.data.address, desc.info.data.size, desc.info.stencil.address,
					    desc.info.stencil.size, desc.info.extent.width, desc.info.extent.height,
					    static_cast<int>(desc.info.pixel_format),
					    image != nullptr ? fmt::format("{}", static_cast<const void*>(static_cast<VkImage>(image->backing.image))) : std::string("none"),
					    image != nullptr && image->sparse.sparse ? 1 : 0,
					    image != nullptr && image->registered ? 1 : 0);
				};
				for (uint32_t i = 0; i < state.color_count; i++) {
					describe("color", state.color_info[i].image_id, state.color_info[i].desc);
				}
				if (static_cast<bool>(state.depth_info.image_id)) {
					describe("depth", state.depth_info.image_id, state.depth_info.desc);
					const auto& dv = state.depth_info.desc.view_info;
					text += fmt::format("  depth view level={}+{} layer={}+{} | rendering {}x{} layers={}\n",
					                    dv.base_level, dv.level_count, dv.base_layer, dv.layer_count,
					                    rendering.width, rendering.height, rendering.num_layers);
					text += fmt::format("  depth write={} test={} stencil test={} clear={} \n",
					                    state.depth_info.depth_write_enable ? 1 : 0,
					                    state.depth_info.depth_test_enable ? 1 : 0,
					                    state.depth_info.stencil_test_enable ? 1 : 0,
					                    state.depth_info.stencil_clear_enable ? 1 : 0);
				}
				std::fputs(text.c_str(), stderr);
			}
		}
	}
	KYTY_PROFILER_BLOCK("Draw::Record");
	cpu_commit_phase_.reset();
	const CpuCost::Phase cpu_record_phase_("Draw::Record");
	NoteTickWork(m_context.GetCommandScheduler().CurrentTick(), 'D',
	             vertex_stages.back().stage.program->shader_hash,
	             state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0);
	if (ImageHistory::Enabled()) {
		const auto tick = m_context.GetCommandScheduler().CurrentTick();
		const auto hash = state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0;
		for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
			ImageHistory::NoteBindView(static_cast<VkImageView>(rendering.color_attachments[i].image_view),
			                           tick, hash, 'A');
		}
		if (rendering.depth_stencil_attachment.image_view) {
			ImageHistory::NoteBindView(static_cast<VkImageView>(rendering.depth_stencil_attachment.image_view),
			                           tick, hash, 'Z');
		}
	}
	if (static const bool fault_map = EnvSwitchOn("KYTY_FAULT_MAP"); fault_map) {
		std::array<uint64_t, 24> views {};
		size_t                   count = 0;
		for (const auto& info: m_descriptor_images) {
			if (count < views.size()) views[count++] = reinterpret_cast<uint64_t>(static_cast<VkImageView>(info.imageView));
		}
		for (uint32_t i = 0; i < rendering.num_color_attachments && count < views.size(); i++) {
			views[count++] = reinterpret_cast<uint64_t>(static_cast<VkImageView>(rendering.color_attachments[i].image_view));
		}
		if (rendering.depth_stencil_attachment.image_view && count < views.size()) {
			views[count++] = reinterpret_cast<uint64_t>(static_cast<VkImageView>(rendering.depth_stencil_attachment.image_view));
		}
		std::array<uint32_t, 6> regions {};
		size_t                  num_regions = 0;
		for (const auto& stage: m_draw_bindless_regions) {
			if (num_regions < regions.size()) regions[num_regions++] = stage;
		}
		NoteTickWorkViews(views.data(), count, regions.data(), num_regions);
		NoteFaultMapBindings(false);
	}
	DrawStats::Prepare(m_context, buffer);
	Occlusion::Prepare(m_context, buffer);
	m_context.GetCommandScheduler().BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.Bound());
	if (pipeline.eds3_raster) {
		const auto raster = DrawDynamicRaster(
		    buffer.GetRegisters(), Prospero::IsRectList(buffer.GetUserConfig().GetPrimType()));
		vk_buffer.setDepthClipEnableEXT(raster.depth_clip_enable ? VK_TRUE : VK_FALSE);
		vk_buffer.setDepthClipNegativeOneToOneEXT(raster.negative_one_to_one ? VK_TRUE : VK_FALSE);
		vk_buffer.setPolygonModeEXT(raster.polygon_mode);
	}
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u, debug_shaders);
	}
	// KYTY_RT_SIZES=1: draws per first color target size, every 10 s ([rt-sizes]): the game's
	// internal render resolution. Counts full draws only (a replayed multi-draw sub-draw shares its
	// base's targets), so it no longer turns replay off (v438c).
	if (static const bool log = EnvSwitchOn("KYTY_RT_SIZES"); log) {
		static std::mutex                             mutex;
		static std::unordered_map<uint64_t, uint32_t> sizes;
		static auto                                   last = std::chrono::steady_clock::now();
		const uint64_t key = state.color_count != 0
		                         ? (uint64_t {state.color_info[0].Extent().width} << 32u) |
		                               state.color_info[0].Extent().height
		                         : 0u;
		std::lock_guard lock(mutex);
		sizes[key]++;
		if (const auto now = std::chrono::steady_clock::now(); now - last > std::chrono::seconds(10)) {
			last = now;
			std::vector<std::pair<uint64_t, uint32_t>> list(sizes.begin(), sizes.end());
			std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			std::string text;
			for (size_t i = 0; i < list.size() && i < 10; i++) {
				text += fmt::format(" {}x{}:{}", list[i].first >> 32u, list[i].first & 0xffffffffu, list[i].second);
			}
			std::fprintf(stderr, "[rt-sizes] draws by first color target:%s\n", text.c_str());
			sizes.clear();
		}
	}
	GpuTiming::Before(m_context, buffer);
	int64_t gpu_zone = -1;
	if (Profiler::g_counters) {
		// The draw's program, as its GPU zone's name and its command zone's text.
		char      name[48];
		const int size = std::snprintf(
		    name, sizeof(name), "VS %016" PRIx64 " PS %016" PRIx64,
		    vertex_stages.back().stage.program->shader_hash,
		    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0);
		gpu_zone = GpuProfiler::Begin(m_context, buffer, name, static_cast<size_t>(size), 0x4e79a7);
		Profiler::CommandZone::Annotate("%s", name);
	}
	// KYTY_BARRIER_SINK: a pending guest barrier may pass this draw when it writes only its own
	// attachments and samples none of them (barrierSink.h (b)). Attachments in GENERAL or a
	// feedback-loop layout are (or may be) sampled; read-only depth layouts are not written.
	bool draw_safe = false;
	if (BarrierSinkEnabled()) {
		const auto writes = [&](const ShaderStageRuntime& runtime) {
			return writes_memory(runtime) || runtime.program->has_address_writes;
		};
		draw_safe = !feedback_aspects && !(state.ps_active && writes(state.ps_input_info.stage));
		for (const auto& stage: vertex_stages) {
			draw_safe = draw_safe && !writes(stage.stage);
		}
		auto& cache = m_context.GetTextureCache();
		for (uint32_t i = 0; draw_safe && i < state.color_count; i++) {
			draw_safe = cache.GetImage(state.color_info[i].image_id).binding.attachment_layout ==
			            vk::ImageLayout::eColorAttachmentOptimal;
		}
		if (draw_safe && state.depth_info.image_id) {
			const auto layout = cache.GetImage(state.depth_info.image_id).binding.attachment_layout;
			draw_safe = layout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
			            layout == vk::ImageLayout::eDepthAttachmentOptimal ||
			            layout == vk::ImageLayout::eStencilAttachmentOptimal ||
			            layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
			            layout == vk::ImageLayout::eDepthReadOnlyOptimal ||
			            layout == vk::ImageLayout::eStencilReadOnlyOptimal ||
			            layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal ||
			            layout == vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
		}
		for (const auto* stage: stages) {
			if (!draw_safe) break;
			draw_safe = stage->gds.buffer == nullptr && stage->shared_memory.buffer == nullptr;
			for (const auto& binding: stage->images) {
				bool target = state.depth_info.image_id && binding.image_id == state.depth_info.image_id;
				for (uint32_t i = 0; !target && i < state.color_count; i++) {
					target = binding.image_id == state.color_info[i].image_id;
				}
				draw_safe = draw_safe && !target;
			}
		}
	}
	SetNextDrawSafe(draw_safe);
	// KYTY_XRAY_PS=<hash+...>|blend|all (diagnostics; changes the picture): these pixel shaders'
	// draws (blend: every draw that blends into its first target) skip the depth and stencil
	// tests and culling, to see an effect that draws but is hidden or rejected.
	// KYTY_XRAY_VS=<hash+...>: also requires the vertex stage's program to be listed.
	// KYTY_XRAY_MODE=<bits>: 1 depth test, 2 stencil test, 4 culling (default 7: all).
	// Every 100 guest flips [xray] lists the (vs, ps) pairs X-rayed in that flip with their draws.
	if (static const std::string xray = [] {
		    const char* value = std::getenv("KYTY_XRAY_PS");
		    return value != nullptr ? std::string(value) : std::string();
	    }();
	    !xray.empty() && state.ps_active) {
		static const std::string xray_vs = [] {
			const char* value = std::getenv("KYTY_XRAY_VS");
			return value != nullptr ? std::string(value) : std::string();
		}();
		const uint64_t vs_hash = vertex_stages.back().stage.program->shader_hash;
		const uint64_t ps_hash = state.ps_input_info.stage.program->shader_hash;
		bool           match =
		    xray == "all" ||
		    (xray == "blend" ? buffer.GetRegisters().GetBlendControl(0).enable
		                     : xray.find(fmt::format("{:016x}", ps_hash)) != std::string::npos);
		if (match && !xray_vs.empty()) {
			match = xray_vs.find(fmt::format("{:016x}", vs_hash)) != std::string::npos;
		}
		static const uint32_t mode = [] {
			const char* value = std::getenv("KYTY_XRAY_MODE");
			return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 7u;
		}();
		if (match) {
			if ((mode & 1u) != 0) vk_buffer.setDepthTestEnable(VK_FALSE);
			if ((mode & 2u) != 0) vk_buffer.setStencilTestEnable(VK_FALSE);
			if ((mode & 4u) != 0 && PipelineDynamicRasterStateEnabled()) {
				vk_buffer.setCullMode(vk::CullModeFlagBits::eNone);
			}
			static std::mutex                                                census_mutex;
			static std::map<std::pair<uint64_t, uint64_t>, std::pair<uint32_t, uint64_t>> census;
			static uint64_t                                                  census_flip = ~0ull;
			const auto flip = Sync::GuestFlipCount();
			std::lock_guard lock(census_mutex);
			if (flip / 100 != census_flip / 100) {
				if (!census.empty()) {
					std::string text;
					for (const auto& [key, value]: census) {
						text += fmt::format(" {:016x}/{:016x}:{}x{}", key.first, key.second,
						                    value.first, value.second);
					}
					std::fprintf(stderr, "[xray] flip=%llu%s\n",
					             static_cast<unsigned long long>(census_flip), text.c_str());
					census.clear();
				}
				census_flip = flip;
			}
			if (flip == census_flip) {
				auto& entry = census[{vs_hash, ps_hash}];
				entry.first++;
				entry.second = draw.index_count;
			}
		}
	}
	// KYTY_POPS_SPLIT_INSTANCES=1 (v438zf, experiment, off by default): an instanced draw whose
	// pixel shader writes shared memory (storage images or buffers: the primitive-ordered shaders)
	// is recorded as one draw per instance, in instance order, with the barrier this kind of draw
	// already gets after it (below) between the instances. The pixel shader and its interlock are
	// unchanged. The order is the one the interlock gives inside the single draw (instance k
	// before k+1 at every pixel); the instances no longer overlap in flight. Each instance sees
	// the same gl_InstanceIndex as before (first instance + k), which is also what instance-rate
	// vertex bindings step by. Not split: mesh pipelines and GPU-read arguments (their instances
	// are not the host draw's), more instances than KYTY_POPS_SPLIT_MAX (default 8), quad lists
	// (their draws are already per quad), a counted occlusion query or draw statistics around the
	// draw (a query cannot span rendering instances), and runs without the barrier after such
	// draws. [pops-split]: the first 16 candidates and what happened to them.
	static const bool split_enabled = [] {
		const char* value = std::getenv("KYTY_POPS_SPLIT_INSTANCES");
		return value != nullptr && value[0] == '1';
	}();
	static const uint32_t split_max = [] {
		const char* value = std::getenv("KYTY_POPS_SPLIT_MAX");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 8u;
	}();
	static const bool split_barrier = [] {
		const char* value = std::getenv("KYTY_DRAW_WRITE_BARRIER");
		return value == nullptr || value[0] != '0';
	}();
	const bool split_candidate = split_enabled && draw.instance_count > 1 && state.ps_active &&
	                             HasShaderBufferWrites(state.ps_input_info.stage);
	const auto note_split = [&](const char* what) {
		static std::atomic<uint32_t> logged {0};
		if (logged.load(std::memory_order_relaxed) < 16u && logged.fetch_add(1) < 16u) {
			std::fprintf(stderr, "[pops-split] ps=%016llx %s count=%u inst=%u: %s\n",
			             static_cast<unsigned long long>(
			                 state.ps_input_info.stage.program->shader_hash),
			             draw.Name(), draw.index_count, draw.instance_count, what);
		}
	};
	DrawStats::Begin(m_context, buffer);
	Occlusion::Begin(m_context, buffer);
	if (mesh_active) {
		if (split_candidate) note_split("mesh pipeline");
		if (mesh_indirect_buffer) {
			vk_buffer.drawMeshTasksIndirectEXT(mesh_indirect_buffer, mesh_indirect_offset, 1, 16u);
		} else {
			// A draw with more groups than the host allows in one dispatch (RADV: 65,535 in x) is
			// split; draw dword 6 gives each part its first group.
			constexpr auto base_offset = (ShaderRecompiler::IR::PushData::MeshDrawDwordCount - 1u) *
			                             sizeof(uint32_t);
			const auto& limits   = m_context.GetGraphics().mesh_shader_properties;
			const auto  per_draw = std::min<uint64_t>(
			    limits.maxMeshWorkGroupCount[0],
			    std::max<uint64_t>(1u, limits.maxMeshWorkGroupTotalCount /
			                               std::max(draw.instance_count, 1u)));
			uint32_t groups_done = 0;
			do {
				const auto groups =
				    static_cast<uint32_t>(std::min<uint64_t>(mesh_groups - groups_done, per_draw));
				if (groups_done != 0) {
					// The same stages as the draw dwords' push above (the layout's range).
					vk_buffer.pushConstants(pipeline.pipeline_layout,
					                        pipeline.push_stages ? pipeline.push_stages
					                                             : vk::ShaderStageFlagBits::eMeshEXT |
					                                                   vk::ShaderStageFlagBits::eFragment,
					                        base_offset, sizeof(groups_done), &groups_done);
				}
				vk_buffer.drawMeshTasksEXT(groups, draw.instance_count, 1);
				groups_done += groups;
			} while (groups_done < mesh_groups);
		}
	} else if (vertex_indirect_buffer) {
		if (split_candidate) note_split("GPU-read arguments");
		vk_buffer.drawIndirect(vertex_indirect_buffer, vertex_indirect_offset, 1, 16u);
	} else if (pops_triangles != 0) {
		// KYTY_POPS_NOINTERLOCK (above): one draw per triangle in primitive order, the bound
		// pipeline has no interlock. Between two triangles: the end of rendering and the barrier
		// a draw that writes shared memory gets after it (below), then the same attachments
		// again, loaded (a clear ran once).
		auto resume = rendering;
		for (auto& attachment: resume.color_attachments) {
			attachment.is_clear = false;
		}
		resume.depth_stencil_attachment.is_clear      = false;
		resume.depth_stencil_attachment.depth_clear   = false;
		resume.depth_stencil_attachment.stencil_clear = false;
		bool first = true;
		for (uint32_t instance = 0; instance < draw.instance_count; instance++) {
			for (uint32_t index = 0; index < draw.index_count; index += 3u) {
				if (!first) {
					m_context.GetCommandScheduler().EndRendering();
					ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eFragmentShader);
					m_context.GetCommandScheduler().BeginRendering(resume);
				}
				first = false;
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(3u, 1u, index, emit.vertex_offset, emit.first_instance + instance);
				} else {
					vk_buffer.draw(3u, 1u, emit.first_vertex + index, emit.first_instance + instance);
				}
			}
		}
	} else {
		bool split = split_candidate;
		if (split) {
			const char* skipped = nullptr;
			if (draw.instance_count > split_max) {
				skipped = "instance count above KYTY_POPS_SPLIT_MAX";
			} else if (emit.gpu_instance_count_address != 0) {
				skipped = "GPU-read instance count";
			} else if (ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy) {
				skipped = "quad list";
			} else if (Occlusion::CountsDraw(buffer) || DrawStats::Enabled()) {
				skipped = "query around the draw";
			} else if (!split_barrier || SyncToggle::SkipDrawWriteBarrier()) {
				skipped = "no barrier after writing draws";
			} else if (!buffer.IsRendering()) {
				skipped = "not rendering";
			}
			note_split(skipped != nullptr ? skipped : "split");
			split = skipped == nullptr;
		}
		if (split) {
			// The rendering instance to resume: the same attachments, loaded (a clear ran once).
			auto resume = buffer.CurrentRenderState();
			for (auto& attachment: resume.color_attachments) {
				attachment.is_clear = false;
			}
			resume.depth_stencil_attachment.is_clear      = false;
			resume.depth_stencil_attachment.depth_clear   = false;
			resume.depth_stencil_attachment.stencil_clear = false;
			vk::PipelineStageFlags write_stages = vk::PipelineStageFlagBits::eFragmentShader;
			for (const auto& stage: vertex_stages) {
				if (HasShaderBufferWrites(stage.stage)) {
					write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
				}
			}
			DrawCallInfo one = draw;
			one.instance_count = 1;
			DrawEmitInfo at = emit;
			for (uint32_t instance = 0; instance < draw.instance_count; instance++) {
				if (instance != 0) {
					m_context.GetCommandScheduler().EndRendering();
					ShaderWriteBarrier(vk_buffer, write_stages);
					m_context.GetCommandScheduler().BeginRendering(resume);
				}
				at.first_instance = emit.first_instance + instance;
				EmitDrawPrimitives(ucfg, vk_buffer, one, at);
			}
		} else {
			EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
		}
	}
	Occlusion::End(m_context, buffer);
	if (DrawStats::Enabled()) {
		DrawStats::Info info;
		info.vs        = vertex_stages.back().stage.program->shader_hash;
		info.ps        = state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0;
		info.kind      = draw.Name();
		info.count     = draw.index_count;
		info.instances = draw.instance_count;
		info.gpu_args  = static_cast<bool>(mesh_indirect_buffer) || static_cast<bool>(vertex_indirect_buffer);
		if (state.color_count > 0 && state.color_info[0].image_id) {
			const auto& image = m_context.GetTextureCache().GetImage(state.color_info[0].image_id);
			info.rt0   = image.info.data.address;
			info.rt0_w = image.info.extent.width;
			info.rt0_h = image.info.extent.height;
		}
		const auto& bc = buffer.GetRegisters().GetBlendControl(0);
		info.blend     = bc.enable;
		info.blend_src = static_cast<uint32_t>(bc.color_srcblend);
		info.blend_dst = static_cast<uint32_t>(bc.color_destblend);
		DrawStats::End(m_context, buffer, info);
	}
	SetNextDrawSafe(false);
	if (RtSnap::Enabled()) {
		RtSnap::AfterCommand(m_context, "draw", stages);
	}
	if (RtSnap::ProbeEnabled()) {
		std::array<ImageId, RENDER_COLOR_ATTACHMENTS_MAX> probe_colors {};
		for (uint32_t i = 0; i < state.color_count && i < probe_colors.size(); i++) {
			probe_colors[i] = state.color_info[i].image_id;
		}
		RtSnap::ProbeAfterDraw(m_context, std::span {probe_colors.data(), state.color_count},
		                       state.depth_info.image_id, vertex_stages.back().stage.program->shader_hash,
		                       state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0,
		                       draw.Name());
	}
	Profiler::Add(Profiler::Counter::Draws);
	GpuProfiler::End(buffer, gpu_zone);
	GpuTiming::After(m_context, buffer,
	                 state.ps_active ? state.ps_input_info.stage.program->shader_hash
	                                 : vertex_stages.back().stage.program->shader_hash,
	                 GpuTiming::Draw);

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u, debug_shaders);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	// KYTY_DRAW_WRITE_BARRIER=0 (experiment): no barrier after a draw that writes buffers; only
	// the guest's own sync orders its writes, as on the console.
	static const bool draw_write_barrier = [] {
		const char* value = std::getenv("KYTY_DRAW_WRITE_BARRIER");
		return value == nullptr || value[0] != '0';
	}();
	if (shader_write_stages && draw_write_barrier && !SyncToggle::SkipDrawWriteBarrier()) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	if (full_barrier) {
		m_context.GetCommandScheduler().EndRendering();
		FullMemoryBarrier(vk_buffer);
	}
	if (m_replay.census_indexed) {
		// KYTY_RENDER_STATS ([multi-packets]): an indexed multi-draw packet's sub-draw, by class.
		bool census_writes = shader_write_stages || full_barrier ||
		                     (state.ps_active && (writes_memory(state.ps_input_info.stage) ||
		                                          state.ps_input_info.stage.program->has_address_writes));
		for (const auto& stage: vertex_stages) {
			census_writes = census_writes || writes_memory(stage.stage) ||
			                stage.stage.program->has_address_writes;
		}
		RenderStats::NoteMultiPacket(mesh_active ? RenderStats::MultiPacketNote::IndexedMesh
		                                         : RenderStats::MultiPacketNote::IndexedVertex);
		if (census_writes) {
			RenderStats::NoteMultiPacket(RenderStats::MultiPacketNote::IndexedWrites);
		}
	}
	if (m_replay.active ||
	    (RenderStats::IndirectReplayActive() && emit.gpu_args_address != 0)) {
		bool writes = shader_write_stages || full_barrier ||
		              (state.ps_active && (writes_memory(state.ps_input_info.stage) ||
		                                   state.ps_input_info.stage.program->has_address_writes));
		for (const auto& stage: vertex_stages) {
			writes = writes || writes_memory(stage.stage) || stage.stage.program->has_address_writes;
		}
		const auto& run = MeshArgsRun();
		// Captured only while replay is active, so the toggle's off windows pay nothing for it.
		const bool indirect_base = !m_replay.active && emit.gpu_args_address != 0 &&
		                           RenderStats::IndirectReplayActive() &&
		                           state.vertex_info[0].mesh.fast_launch &&
		                           mesh_indirect_buffer != nullptr && emit.indirect_packet != nullptr &&
		                           run.last_used_packet == emit.indirect_packet &&
		                           run.last_used_generation == run.generation;
		const bool cpu_base = emit.gpu_args_address == 0 && m_replay.active && !mesh_indirect_buffer;
		if (mesh_active && !draw.IsIndexed() && (cpu_base || indirect_base) &&
		    vertex_stages.size() == 1 && buffer.IsRendering() &&
		    buffer.CurrentRenderState() == rendering) {
			CaptureReplayBase(buffer, draw, state, emit, pipeline.pipeline_layout, writes,
			                 mesh_indirect_buffer, mesh_indirect_offset, gpu_args_buffer);
		} else {
			if (emit.gpu_args_address != 0) {
				const auto reason = !mesh_active || !state.vertex_info[0].mesh.fast_launch
				                        ? RenderStats::IndirectReplayNote::BaseNotFastLaunch
				                        : RenderStats::IndirectReplayNote::BaseArgsRun;
				m_replay.indirect_invalid_reason = static_cast<uint32_t>(reason);
				RenderStats::NoteIndirectReplay(reason);
			} else {
				RenderStats::NoteReplay(RenderStats::ReplayNote::BaseNotMesh);
			}
		}
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u, debug_shaders);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	m_replay.valid = false;

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	// KYTY_CPU_COST: the indexed draw's own path up to its render state (the renderer mutex,
	// the metadata/copy/resolve checks, the index source), and what follows the draw.
	std::optional<CpuCost::Phase> cpu_begin_phase_(std::in_place, "DrawIndex::Begin");
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	// The zone is the wait for the renderer mutex (another thread holding it).
	Common::LockGuard lock = [&]() -> Common::LockGuard {
		KYTY_PROFILER_BLOCK("Renderer mutex wait");
		return Common::LockGuard(m_context.GetMutex());
	}();
	if (args.index_count == 0 || args.instance_count == 0) {
		LogDrawCensusLine(buffer, "DrawIndex-empty", args.index_count, args.instance_count, -1, -1,
		                  -1);
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}
	if (ReuseProbe::Enabled()) {
		ReuseProbe::NoteDraw(sh_ctx, buffer.GetRegisters(), ucfg, args.render_target_slice_offset, 2u);
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	const DrawRenderStateRef state_ref; // v438zk: KYTY_DRAW_STATE_REUSE
	DrawRenderState&         state = *state_ref;
	cpu_begin_phase_.reset();
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		Common::FrameSkips::Skip(Common::FrameSkips::PrepFailed);
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;
	LogWatchedDraw(draw, state, ucfg.GetIndexOffset(), args.base_vertex, 0, emit.vertex_offset,
	               emit.first_instance, indirect);

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	const CpuCost::Phase cpu_end_phase_("DrawIndex::End");
	WatchStencilAfterDraw(m_context, buffer, state);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	m_replay.valid = false;

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	std::optional<CpuCost::Phase> cpu_begin_phase_(std::in_place, "DrawAuto::Begin"); // as in DrawIndex
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);
	// KYTY_LOG_GPU_ARGS_DRAWS=1: each new (es, gs, ps) of a DrawIndexAuto with GPU-written
	// arguments, once, with the arguments' address.
	static const bool log_gpu_args = std::getenv("KYTY_LOG_GPU_ARGS_DRAWS") != nullptr;
	if (log_gpu_args && (args.gpu_args_address != 0 || args.gpu_instance_count_address != 0)) {
		static std::mutex                   seen_mutex;
		static std::unordered_set<uint64_t> seen;
		const auto h  = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
		const auto es = h(sh_ctx.GetVs().es_regs.data_addr);
		const auto gs = h(sh_ctx.GetVs().gs_regs.data_addr);
		const auto ps = h(sh_ctx.GetPs().ps_regs.data_addr);
		const auto key = es ^ (gs * 0x9e3779b97f4a7c15ull) ^ (ps * 0xc2b2ae3d27d4eb4full);
		std::lock_guard lock(seen_mutex);
		if (seen.insert(key).second) {
			std::fprintf(stderr,
			             "[gpu-args-draw] submit=%llu es=%016llx gs=%016llx ps=%016llx args=0x%llx "
			             "inst=0x%llx prim=%u\n",
			             static_cast<unsigned long long>(submit_id),
			             static_cast<unsigned long long>(es), static_cast<unsigned long long>(gs),
			             static_cast<unsigned long long>(ps),
			             static_cast<unsigned long long>(args.gpu_args_address),
			             static_cast<unsigned long long>(args.gpu_instance_count_address),
			             static_cast<uint32_t>(ucfg.GetPrimType()));
			std::fflush(stderr);
		}
	}

	// The zone is the wait for the renderer mutex (another thread holding it).
	Common::LockGuard lock = [&]() -> Common::LockGuard {
		KYTY_PROFILER_BLOCK("Renderer mutex wait");
		return Common::LockGuard(m_context.GetMutex());
	}();
	if (args.gpu_args_address == 0 &&
	    (args.vertex_count == 0 ||
	     (args.instance_count == 0 && args.gpu_instance_count_address == 0))) {
		LogDrawCensusLine(buffer, "DrawIndexAuto-empty", args.vertex_count, args.instance_count, -1,
		                  -1, -1);
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}
	if (ReuseProbe::Enabled()) {
		ReuseProbe::NoteDraw(sh_ctx, buffer.GetRegisters(), ucfg, args.render_target_slice_offset, 1u);
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	const DrawRenderStateRef state_ref; // v438zk: KYTY_DRAW_STATE_REUSE
	DrawRenderState&         state = *state_ref;
	cpu_begin_phase_.reset();
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		Common::FrameSkips::Skip(Common::FrameSkips::PrepFailed);
		ResetBindings();
		return;
	}

	// Counts read on the GPU work for fast-launch mesh draws and vertex-pipeline draws with one
	// draw call; others read the guest arguments here (waiting for the GPU when it wrote them).
	std::optional<CpuCost::Phase> cpu_args_phase_(std::in_place, "DrawAuto::Args");
	DrawAutoArgs drawn = args;
	// KYTY_GPU_ARGS_MESH / KYTY_GPU_ARGS_VERTEX=0: read that kind's GPU-written counts here.
	// Mesh draws read them here by default since 2026-10-10: with the counts read on the GPU,
	// ~1 boot in 4 lost the device (shader read fault) in the title scene's first GPU-driven
	// draws; read here, 0 in 22 (the counts read were all sane). Read here, they need the GPU
	// drained first (the page protection of a just-marked page is not in force yet), ~25 drains a
	// frame; on the GPU they are now validated against the device's limits (MeshIndirectArgs), so
	// they are read there by default again. KYTY_GPU_ARGS_MESH=0 reads them here.
	static const bool gpu_args_mesh = [] {
		const char* value = std::getenv("KYTY_GPU_ARGS_MESH");
		return value == nullptr || value[0] != '0';
	}();
	static const bool gpu_args_vertex = [] {
		const char* value = std::getenv("KYTY_GPU_ARGS_VERTEX");
		return value == nullptr || value[0] != '0';
	}();
	if (args.gpu_args_address != 0) {
		const bool mesh   = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		const bool gpu_ok = mesh ? gpu_args_mesh && state.vertex_info[0].mesh.fast_launch
		                         : gpu_args_vertex &&
		                               ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy;
		if (!gpu_ok) {
			uint32_t words[4] {};
			m_context.GetBufferCache().WaitSharedWrites(args.gpu_args_address, sizeof(words));
			static const bool args_debug = std::getenv("KYTY_ARGS_DEBUG") != nullptr;
			const auto        before =
			    args_debug ? m_context.GetBufferCache().DescribeWriteState(args.gpu_args_address)
			                      : std::string {};
			std::memcpy(words, reinterpret_cast<const void*>(args.gpu_args_address), sizeof(words));
			if (args_debug && (words[0] > (1u << 22u) || words[1] > 65535u)) {
				std::fprintf(stderr, "[gpu-args]   before the read: %s\n", before.c_str());
			}
			drawn.vertex_count     = words[0];
			drawn.instance_count   = words[1];
			drawn.first_vertex     = words[2];
			drawn.first_instance   = words[3];
			drawn.gpu_args_address = 0;
			if (words[0] > (1u << 22u) || words[1] > 65535u) {
				static std::atomic<uint32_t> odd {0};
				if (odd.fetch_add(1) < 32) {
					std::fprintf(stderr, "[gpu-args] odd counts at 0x%llx: %u %u %u %u mesh=%d\n",
					             static_cast<unsigned long long>(args.gpu_args_address), words[0],
					             words[1], words[2], words[3], mesh ? 1 : 0);
					std::fprintf(stderr, "[gpu-args]   %s\n",
					             m_context.GetBufferCache()
					                 .DescribeWriteState(args.gpu_args_address)
					                 .c_str());
					std::fflush(stderr);
				}
			}
			if (drawn.vertex_count == 0 || drawn.instance_count == 0) {
				ResetBindings();
				return;
			}
		}
	}
	if (args.gpu_instance_count_address != 0) {
		const bool mesh   = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		const bool gpu_ok = !mesh && ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy;
		if (!gpu_ok) {
			m_context.GetBufferCache().WaitSharedWrites(args.gpu_instance_count_address,
			                                            sizeof(uint32_t));
			std::memcpy(&drawn.instance_count,
			            reinterpret_cast<const void*>(args.gpu_instance_count_address),
			            sizeof(uint32_t));
			drawn.gpu_instance_count_address = 0;
			if (drawn.instance_count == 0) {
				ResetBindings();
				return;
			}
		}
	}
	const DrawCallInfo drawn_info {CommandBufferDebugOp::DrawIndexAuto, drawn.vertex_count,
	                               drawn.instance_count, drawn.first_instance};
	cpu_args_phase_.reset();

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = drawn.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, drawn.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(drawn.first_vertex));
	emit.first_instance = instance_offset;
	emit.gpu_args_address = drawn.gpu_args_address;
	emit.gpu_instance_count_address = drawn.gpu_instance_count_address;
	emit.start_instance_sgpr = drawn.gpu_args_address != 0 ? drawn.start_instance_sgpr : 0u;
	emit.indirect_packet = drawn.gpu_args_address != 0 ? drawn.indirect_packet : nullptr;
	LogWatchedDraw(drawn_info, state, ucfg.GetIndexOffset(), 0, drawn.first_vertex,
	               static_cast<int32_t>(emit.first_vertex), emit.first_instance, indirect);

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, drawn_info, state, topology, emit, index_source, false);
	const CpuCost::Phase cpu_end_phase_("DrawAuto::End");
	WatchStencilAfterDraw(m_context, buffer, state);
	ResetBindings();
}

void RenderExecutor::BeginMultiDraw() {
	m_replay.active = true;
	m_replay.valid  = false;
}

void RenderExecutor::EndMultiDraw() {
	m_replay.active = false;
	m_replay.valid  = false;
}

// Per-draw diagnostics that a replayed draw would skip: the replay is off while any runs.
// The shader registers as bytes for indirect replay, with the GS and PS user SGPR values zeroed:
// replay checks those itself (vertex parameters and the plan's resource words, PS user data), and
// compares everything else (programs, resource registers, SGPR types and counts) exactly.
static void ShaderBytesForReplay(const HW::Shader& sh, std::vector<uint8_t>& out) {
	out.resize(sizeof(sh));
	std::memcpy(out.data(), &sh, sizeof(sh));
	const auto* base = reinterpret_cast<const uint8_t*>(&sh);
	const auto  zero = [&](const void* field, size_t size) {
		std::memset(out.data() + (reinterpret_cast<const uint8_t*>(field) - base), 0, size);
	};
	zero(sh.GetVs().gs_user_sgpr.value, sizeof(sh.GetVs().gs_user_sgpr.value));
	zero(sh.GetPs().ps_user_sgpr.value, sizeof(sh.GetPs().ps_user_sgpr.value));
}

// KYTY_LOG_INDIRECT_DRAWS (graphicsRun.cpp) logs per indirect draw: replay would skip those lines.
static bool LogIndirectDraw() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LOG_INDIRECT_DRAWS");
		return value != nullptr && std::strtoul(value, nullptr, 10) != 0;
	}();
	return enabled;
}

// A diagnostic switch is on when set to anything but empty or "0" (fps plan 11-15: KYTY_FAULT_MAP=0
// in the fps launchers counted as set and turned multi-draw replay off in every run).
static bool EnvSwitchOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

static bool ReplayDiagnosticsActive(const CommandBuffer& buffer, const DrawRenderState& state) {
	static const bool env_diagnostics = [] {
		for (const char* name: {"KYTY_FAULT_MAP", "KYTY_XRAY_PS", "KYTY_XRAY_VS",
		                        "KYTY_LOG_PS_DRAWS", "KYTY_LOG_TESS_DRAWS",
		                        "KYTY_MESH_NEAR", "KYTY_WATCH_STENCIL_PS", "KYTY_WATCH_DRAW",
		                        "KYTY_LOG_FACE_STATE", "KYTY_FACE_CENSUS", "KYTY_LOG_DRAWS",
		                        "KYTY_LOG_DRAWS_WITH_COUNT", "KYTY_LOG_GS_SLOTS",
		                        "KYTY_TRACE_GS_SLOT_SEQ", "KYTY_TRACE_GS_SLOT",
		                        "KYTY_TRACE_GS_SLOT_VALUE", "KYTY_LOG_DRAW_PAIRS",
		                        "KYTY_LOG_DRAWS_FLIP", "KYTY_FULL_BARRIER_PS",
		                        "KYTY_DRAW_WRITE_BARRIER", "KYTY_LOG_GPU_ARGS_DRAWS",
		                        "KYTY_ARGS_DEBUG", "KYTY_DIAG_FLIP_TESS_FACE_FILE",
		                        "KYTY_DIAG_FLIP_ALL_FACE_FILE", "KYTY_LOG_DRAWS_TRIGGER",
		                        "KYTY_WATCH_SHADER_TRIGGER", "KYTY_XRAY_MODE"}) {
			if (EnvSwitchOn(name)) {
				return true;
			}
		}
		return false;
	}();
	return env_diagnostics || graphics_debug_dump_enabled() || DrawStats::Enabled() ||
	       Profiler::g_counters || RtSnap::Enabled() || ImageHistory::Enabled() ||
	       GpuTiming::Detail::g_active.load(std::memory_order_relaxed) || BarrierSinkEnabled() ||
	       Occlusion::CountsDraw(buffer) || RtDrawLog::Address() != 0 ||
	       Common::FrameSkips::Enabled() || !SlotDumps().empty() || MeshWalkEnabled() ||
	       ImgRef::Address() != 0 || LogIndirectDraw() || SyncToggle::ForceUpload() ||
	       (state.ps_active && PaintDraws::Index(state.ps_input_info.stage.program->shader_hash) >= 0);
}

void RenderExecutor::CaptureReplayBase(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawRenderState& state, const DrawEmitInfo& emit,
                                       vk::PipelineLayout layout, bool writes,
                                       vk::Buffer mesh_indirect_buffer,
                                       vk::DeviceSize mesh_indirect_offset,
                                       const void* gpu_args_buffer) {
	const CpuCost::Phase cpu_phase_("Replay::CaptureBase");
	(void)draw;
	auto& base = m_replay;
	base.valid = false;
	base.indirect = false;
	base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::NoBase);
	if (writes) {
		if (emit.gpu_args_address != 0) {
			base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::BaseWrites);
			RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::BaseWrites);
		} else {
			RenderStats::NoteReplay(RenderStats::ReplayNote::BaseWrites);
		}
		return;
	}
	if (ReplayDiagnosticsActive(buffer, state)) {
		if (emit.gpu_args_address != 0) {
			base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::Diagnostics);
			RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::Diagnostics);
		} else {
			RenderStats::NoteReplay(RenderStats::ReplayNote::BaseDiagnostics);
		}
		return;
	}
	if (emit.gpu_args_address != 0 &&
	    (mesh_indirect_buffer == nullptr || gpu_args_buffer == nullptr || emit.indirect_packet == nullptr ||
	     MeshArgsRun().last_used_packet != emit.indirect_packet ||
	     MeshArgsRun().last_used_generation != MeshArgsRun().generation)) {
		base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::BaseArgsRun);
		RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::BaseArgsRun);
		return;
	}
	const auto& vs_stage  = state.vertex_info[0].stage;
	uint64_t    mask      = ~0ull;
	uint32_t    plan_base = 0;
	if (!PipelineCache::PlanUserDataReads(vs_stage.resources, mask, plan_base) ||
	    plan_base != vs_stage.program->user_data_base) {
		if (emit.gpu_args_address != 0) {
			base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::BasePlanUnknown);
			RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::BasePlanUnknown);
		} else {
			RenderStats::NoteReplay(RenderStats::ReplayNote::BasePlanUnknown);
		}
		return;
	}
	// The user data exactly as the draw's refresh saw it (PrepareProgram is what
	// GetGraphicsPrograms describes the stage with); a mismatch means the comparison below would
	// not be the refresh's own input.
	auto&                 sh = buffer.GetShaders();
	ShaderVertexInputInfo described;
	const auto params = PrepareProgram(sh.GetVs(), buffer.GetRegisters(), buffer.GetUserConfig(),
	                                   described);
	const auto& snapshot_user_data = vs_stage.resources->user_data;
	if (params.user_data_count != snapshot_user_data.size() ||
	    params.user_data_count > base.vs_user.size() ||
	    !std::equal(snapshot_user_data.begin(), snapshot_user_data.end(), params.user_data.begin())) {
		if (emit.gpu_args_address != 0) {
			base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::BaseUserData);
			RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::BaseUserData);
		} else {
			RenderStats::NoteReplay(RenderStats::ReplayNote::BaseUserData);
		}
		return;
	}
	const auto& ps      = sh.GetPs();
	base.ps_user_count  = std::min<uint32_t>(ps.ps_regs.rsrc2.user_sgpr, base.ps_user.size());
	std::copy_n(ps.ps_user_sgpr.value, base.ps_user_count, base.ps_user.begin());
	base.vs_user_count  = params.user_data_count;
	std::copy_n(params.user_data.begin(), params.user_data_count, base.vs_user.begin());
	base.vs_params_hash = params.hash;
	base.vs_plan_reads  = mask;
	base.vs_program     = vs_stage.program;
	base.vs_shader_hash = vs_stage.program->shader_hash;
	base.ps_active      = state.ps_active;
	base.ps_shader_hash = state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0;
	base.vs_shader_data = m_graphics_bindings.vertex[0].shader_data;
	base.bound_data     = base.vs_shader_data;
	base.tick           = m_context.GetCommandScheduler().CurrentTick();
	base.buffer         = &buffer;
	base.rendering      = buffer.CurrentRenderState();
	base.commit_generation = m_last_commit.generation;
	base.layout         = layout;
	base.mesh           = state.vertex_info[0].mesh;
	base.indirect       = emit.gpu_args_address != 0;
	base.args_run_generation = base.indirect ? MeshArgsRun().generation : 0;
	base.buffer_unregister_epoch = m_context.GetBufferCache().UnregisterEpoch();
	base.gpu_args_buffer = base.indirect ? gpu_args_buffer : nullptr;
	base.mesh_indirect_buffer = base.indirect ? mesh_indirect_buffer : nullptr;
	base.mesh_indirect_offset = base.indirect ? mesh_indirect_offset : 0;
	base.start_instance_sgpr = base.indirect ? emit.start_instance_sgpr : 0;
	base.indirect_invalid_reason = static_cast<uint32_t>(RenderStats::IndirectReplayNote::NoBase);
	base.valid          = true;
	if (base.indirect) {
		const auto& ctx  = buffer.GetRegisters();
		const auto& ucfg = buffer.GetUserConfig();
		base.ctx_snapshot.resize(sizeof(ctx));
		std::memcpy(base.ctx_snapshot.data(), &ctx, sizeof(ctx));
		base.ucfg_snapshot.resize(sizeof(ucfg));
		std::memcpy(base.ucfg_snapshot.data(), &ucfg, sizeof(ucfg));
		ShaderBytesForReplay(buffer.GetShaders(), base.sh_snapshot);
		RenderStats::g_indirect_base_maybe.store(true, std::memory_order_relaxed);
	}
	if (base.indirect) {
		RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::BaseCaptured);
	} else {
		RenderStats::NoteReplay(RenderStats::ReplayNote::BaseCaptured);
	}
}

void RenderExecutor::InvalidateIndirectReplay(uint32_t reason) {
	if (m_replay.valid && m_replay.indirect) {
		m_replay.valid = false;
		m_replay.indirect_invalid_reason = reason;
	}
	RenderStats::g_indirect_base_maybe.store(false, std::memory_order_relaxed);
}

// KYTY_RENDER_STATS census ([multi-packets]): the state a replayed packet ends with, kept only
// while the base of its last full sub-draw still stands (what a replay of one more sub-draw
// would have checked first).
void RenderExecutor::CensusMultiDrawEnd(CommandBuffer& buffer) {
	auto&             base = m_replay;
	Common::LockGuard lock(m_context.GetMutex());
	base.census_valid = base.active && base.valid && !base.indirect && base.buffer == &buffer &&
	                    base.tick == m_context.GetCommandScheduler().CurrentTick() &&
	                    buffer.IsRendering() && buffer.CurrentRenderState() == base.rendering &&
	                    m_last_commit.generation == base.commit_generation;
	if (!base.census_valid) {
		return;
	}
	base.census_tick              = base.tick;
	base.census_commit_generation = m_last_commit.generation;
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	base.census_ctx.resize(sizeof(ctx));
	std::memcpy(base.census_ctx.data(), &ctx, sizeof(ctx));
	base.census_ucfg.resize(sizeof(ucfg));
	std::memcpy(base.census_ucfg.data(), &ucfg, sizeof(ucfg));
	ShaderBytesForReplay(buffer.GetShaders(), base.census_sh);
}

// Before a replayed packet's first sub-draw (its draw-index register and start instance are
// written): would the previous packet's base have replayed it? Counted, nothing else. The base
// fields are still that base's when the commit generation is unchanged (a full draw commits).
void RenderExecutor::CensusMultiDrawFirst(CommandBuffer& buffer) {
	using Note = RenderStats::MultiPacketNote;
	RenderStats::NoteMultiPacket(Note::FirstDraws);
	auto& base = m_replay;
	if (!base.census_valid) {
		RenderStats::NoteMultiPacket(Note::FirstNoBase);
		return;
	}
	base.census_valid = false;
	Common::LockGuard lock(m_context.GetMutex());
	if (m_context.GetCommandScheduler().CurrentTick() != base.census_tick || &buffer != base.buffer ||
	    !buffer.IsRendering() || !(buffer.CurrentRenderState() == base.rendering) ||
	    m_last_commit.generation != base.census_commit_generation) {
		RenderStats::NoteMultiPacket(Note::FirstOtherCommit);
		return;
	}
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	auto&       sh   = buffer.GetShaders();
	ShaderBytesForReplay(sh, base.census_scratch);
	const auto& ps   = sh.GetPs();
	bool        same = base.census_ctx.size() == sizeof(ctx) &&
	                   std::memcmp(base.census_ctx.data(), &ctx, sizeof(ctx)) == 0 &&
	                   base.census_ucfg.size() == sizeof(ucfg) &&
	                   std::memcmp(base.census_ucfg.data(), &ucfg, sizeof(ucfg)) == 0 &&
	                   base.census_sh == base.census_scratch &&
	                   std::min<uint32_t>(ps.ps_regs.rsrc2.user_sgpr, base.ps_user.size()) ==
	                       base.ps_user_count &&
	                   std::equal(base.ps_user.begin(), base.ps_user.begin() + base.ps_user_count,
	                              ps.ps_user_sgpr.value);
	if (same) {
		ShaderVertexInputInfo described;
		const auto params = PrepareProgram(sh.GetVs(), ctx, ucfg, described);
		same = params.hash == base.vs_params_hash && params.user_data_count == base.vs_user_count;
		for (uint32_t i = 0; same && i < params.user_data_count; i++) {
			same = params.user_data[i] == base.vs_user[i] || ((base.vs_plan_reads >> i) & 1u) == 0;
		}
	}
	RenderStats::NoteMultiPacket(same ? Note::FirstSameState : Note::FirstStateChanged);
}

static bool ReplayIndirectDiagnosticsActive(const CommandBuffer& buffer, bool ps_active,
                                           uint64_t ps_shader_hash) {
	static const bool env_diagnostics = [] {
		for (const char* name: {"KYTY_FAULT_MAP", "KYTY_XRAY_PS", "KYTY_XRAY_VS",
		                        "KYTY_LOG_PS_DRAWS", "KYTY_LOG_TESS_DRAWS",
		                        "KYTY_MESH_NEAR", "KYTY_WATCH_STENCIL_PS", "KYTY_WATCH_DRAW",
		                        "KYTY_LOG_FACE_STATE", "KYTY_FACE_CENSUS", "KYTY_LOG_DRAWS",
		                        "KYTY_LOG_DRAWS_WITH_COUNT", "KYTY_LOG_GS_SLOTS",
		                        "KYTY_TRACE_GS_SLOT_SEQ", "KYTY_TRACE_GS_SLOT",
		                        "KYTY_TRACE_GS_SLOT_VALUE", "KYTY_LOG_DRAW_PAIRS",
		                        "KYTY_LOG_DRAWS_FLIP", "KYTY_FULL_BARRIER_PS",
		                        "KYTY_DRAW_WRITE_BARRIER", "KYTY_LOG_GPU_ARGS_DRAWS",
		                        "KYTY_ARGS_DEBUG", "KYTY_DIAG_FLIP_TESS_FACE_FILE",
		                        "KYTY_DIAG_FLIP_ALL_FACE_FILE", "KYTY_LOG_DRAWS_TRIGGER",
		                        "KYTY_WATCH_SHADER_TRIGGER", "KYTY_XRAY_MODE"}) {
			if (EnvSwitchOn(name)) return true;
		}
		return false;
	}();
	return env_diagnostics || graphics_debug_dump_enabled() || DrawStats::Enabled() ||
	       Profiler::g_counters || RtSnap::Enabled() || ImageHistory::Enabled() ||
	       GpuTiming::Detail::g_active.load(std::memory_order_relaxed) || BarrierSinkEnabled() ||
	       Occlusion::CountsDraw(buffer) || RtDrawLog::Address() != 0 ||
	       Common::FrameSkips::Enabled() || !SlotDumps().empty() || MeshWalkEnabled() ||
	       ImgRef::Address() != 0 || LogIndirectDraw() || SyncToggle::ForceUpload() ||
	       (ps_active && PaintDraws::Index(ps_shader_hash) >= 0);
}

bool RenderExecutor::ReplayIndirectDraw(uint64_t submit_id, CommandBuffer& buffer,
                                        uint64_t gpu_args_address, uint32_t start_instance_sgpr,
                                        const uint32_t* packet) {
	const CpuCost::Phase cpu_phase_("Replay::IndirectDraw");
	auto& base = m_replay;
	if (!base.valid || !base.indirect) {
		const auto reason = base.indirect_invalid_reason <
		                            static_cast<uint32_t>(RenderStats::IndirectReplayNote::Count)
		                        ? static_cast<RenderStats::IndirectReplayNote>(base.indirect_invalid_reason)
		                        : RenderStats::IndirectReplayNote::NoBase;
		RenderStats::NoteIndirectReplay(reason);
		return false;
	}
	auto fail = [&](RenderStats::IndirectReplayNote reason) {
		base.valid = false;
		base.indirect_invalid_reason = static_cast<uint32_t>(reason);
		RenderStats::NoteIndirectReplay(reason);
		return false;
	};
	if (gpu_args_address == 0 || start_instance_sgpr != base.start_instance_sgpr) {
		return fail(RenderStats::IndirectReplayNote::StartInstance);
	}
	if (packet == nullptr || MeshArgsRun().packet != packet ||
	    MeshArgsRun().generation != base.args_run_generation) {
		return fail(RenderStats::IndirectReplayNote::ArgsRun);
	}
	const auto* entry = MeshArgsRun().Peek(packet);
	if (entry == nullptr || entry->args != gpu_args_address || entry->buffer == nullptr ||
	    entry->buffer != base.mesh_indirect_buffer) {
		return fail(RenderStats::IndirectReplayNote::ArgsRun);
	}
	EXIT_IF(buffer.IsInvalid());
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.PopPendingOperations();
	Common::LockGuard lock = [&]() -> Common::LockGuard {
		KYTY_PROFILER_BLOCK("Renderer mutex wait");
		return Common::LockGuard(m_context.GetMutex());
	}();
	if (scheduler.CurrentTick() != base.tick || &buffer != base.buffer ||
	    !buffer.IsRendering() || !(buffer.CurrentRenderState() == base.rendering)) {
		return fail(RenderStats::IndirectReplayNote::CommandChanged);
	}
	if (m_last_commit.generation != base.commit_generation) {
		return fail(RenderStats::IndirectReplayNote::CommitChanged);
	}
	if (ReplayIndirectDiagnosticsActive(buffer, base.ps_active, base.ps_shader_hash)) {
		return fail(RenderStats::IndirectReplayNote::Diagnostics);
	}
	auto& cache = m_context.GetBufferCache();
	const auto [args_buffer, args_offset] =
	    cache.ObtainBuffer(gpu_args_address, 16u, false, false, {}, true);
	(void)args_offset;
	if (args_buffer == nullptr || args_buffer != base.gpu_args_buffer ||
	    !args_buffer->HasDeviceAddress()) {
		return fail(RenderStats::IndirectReplayNote::ArgsBuffer);
	}
	if (cache.UnregisterEpoch() != base.buffer_unregister_epoch) {
		return fail(RenderStats::IndirectReplayNote::BufferCacheChanged);
	}
	// ObtainBuffer can wait, and a wait may run another queue's slice that invalidates the base.
	if (!base.valid || !base.indirect) {
		return fail(RenderStats::IndirectReplayNote::PacketBoundary);
	}
	// Context/user-config register packets may sit between the draws (the command processor keeps
	// the base across them); the full path derives pipeline, render targets and render state from
	// these registers, so they must be byte-identical to the base's (a padding difference only
	// costs a fallback).
	{
		const auto& ctx  = buffer.GetRegisters();
		const auto& ucfg = buffer.GetUserConfig();
		if (base.ctx_snapshot.size() != sizeof(ctx) || base.ucfg_snapshot.size() != sizeof(ucfg) ||
		    std::memcmp(base.ctx_snapshot.data(), &ctx, sizeof(ctx)) != 0 ||
		    std::memcmp(base.ucfg_snapshot.data(), &ucfg, sizeof(ucfg)) != 0) {
			return fail(RenderStats::IndirectReplayNote::ContextChanged);
		}
		thread_local std::vector<uint8_t> sh_bytes;
		ShaderBytesForReplay(buffer.GetShaders(), sh_bytes);
		if (sh_bytes != base.sh_snapshot) {
			return fail(RenderStats::IndirectReplayNote::ContextChanged);
		}
	}
	if (scheduler.CurrentTick() != base.tick || &buffer != base.buffer ||
	    !buffer.IsRendering() || !(buffer.CurrentRenderState() == base.rendering)) {
		return fail(RenderStats::IndirectReplayNote::CommandChanged);
	}
	if (m_last_commit.generation != base.commit_generation) {
		return fail(RenderStats::IndirectReplayNote::CommitChanged);
	}
	auto&       sh = buffer.GetShaders();
	const auto& ps = sh.GetPs();
	if (std::min<uint32_t>(ps.ps_regs.rsrc2.user_sgpr, base.ps_user.size()) != base.ps_user_count ||
	    !std::equal(base.ps_user.begin(), base.ps_user.begin() + base.ps_user_count,
	                ps.ps_user_sgpr.value)) {
		return fail(RenderStats::IndirectReplayNote::PsUserData);
	}
	ShaderVertexInputInfo described;
	const auto params = PrepareProgram(sh.GetVs(), buffer.GetRegisters(), buffer.GetUserConfig(),
	                                   described);
	if (params.hash != base.vs_params_hash || params.user_data_count != base.vs_user_count) {
		return fail(RenderStats::IndirectReplayNote::VsProgram);
	}
	uint64_t changed = 0;
	for (uint32_t i = 0; i < params.user_data_count; i++) {
		if (params.user_data[i] != base.vs_user[i]) changed |= 1ull << i;
	}
	if ((changed & base.vs_plan_reads) != 0) {
		return fail(RenderStats::IndirectReplayNote::PlanReads);
	}
	const std::vector<uint32_t>* data = &base.vs_shader_data;
	if (changed != 0) {
		base.scratch = base.vs_shader_data;
		const auto& registers = base.vs_program->bindings.user_data_registers;
		const auto  data_base = base.vs_program->user_data_base;
		bool        any = false;
		for (size_t i = 0; i < registers.size() && i < base.scratch.size(); i++) {
			if (registers[i] < data_base || registers[i] - data_base >= params.user_data_count) continue;
			const auto index = registers[i] - data_base;
			if (((changed >> index) & 1u) != 0) {
				base.scratch[i] = params.user_data[index];
				any = true;
			}
		}
		if (any) data = &base.scratch;
	}
	if (*data != base.bound_data) {
		if (!RecommitShaderData(buffer, *data)) {
			return fail(RenderStats::IndirectReplayNote::CommandChanged);
		}
		base.bound_data = *data;
	}
	const auto* consumed = MeshArgsRun().Take(packet);
	if (consumed == nullptr || consumed->args != gpu_args_address ||
	    consumed->buffer != entry->buffer || consumed->offset != entry->offset) {
		return fail(RenderStats::IndirectReplayNote::ArgsRun);
	}
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    0, 0, 0, 0, 0);
	const auto vk_buffer = buffer.Recorder();
	const uint32_t draw_data[] {0u,
	                            0u,
	                            start_instance_sgpr,
	                            0xFFFFFFFFu,
	                            static_cast<uint32_t>(gpu_args_address),
	                            static_cast<uint32_t>(gpu_args_address >> 32u),
	                            0u};
	static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
	// The commit's stages (a linked pipeline's push range names every graphics stage).
	const auto mesh_push_stages = m_last_commit.push_stages;
	vk_buffer.pushConstants(base.layout, mesh_push_stages, 0, sizeof(draw_data), draw_data);
	vk_buffer.drawMeshTasksIndirectEXT(consumed->buffer, consumed->offset, 1, 16u);
	NoteTickWork(scheduler.CurrentTick(), 'D', base.vs_shader_hash,
	             base.ps_shader_hash);
	SetNextDrawSafe(false);
	Profiler::Add(Profiler::Counter::Draws);
	RenderStats::NoteIndirectReplay(RenderStats::IndirectReplayNote::Replayed);
	return true;
}

bool RenderExecutor::ReplayMultiDraw(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t vertex_count, uint32_t instance_count,
                                     uint32_t first_vertex, uint32_t first_instance) {
	const CpuCost::Phase cpu_phase_("Replay::MultiDraw");
	auto& base = m_replay;
	if (!base.active || !base.valid) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::NoBase);
		return false;
	}
	EXIT_IF(buffer.IsInvalid());
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.PopPendingOperations();
	Common::LockGuard lock = [&]() -> Common::LockGuard {
		KYTY_PROFILER_BLOCK("Renderer mutex wait");
		return Common::LockGuard(m_context.GetMutex());
	}();
	// As DrawAuto: a draw without vertices or instances records nothing.
	if (vertex_count == 0 || instance_count == 0) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::Empty);
		return true;
	}
	if (scheduler.CurrentTick() != base.tick || &buffer != base.buffer || !buffer.IsRendering() ||
	    !(buffer.CurrentRenderState() == base.rendering)) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::CommandChanged);
		base.valid = false;
		return false;
	}
	if (m_last_commit.generation != base.commit_generation) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::CommitChanged);
		base.valid = false;
		return false;
	}
	if (GpuTiming::Detail::g_active.load(std::memory_order_relaxed)) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::BaseDiagnostics);
		return false;
	}
	auto&       sh = buffer.GetShaders();
	const auto& ps = sh.GetPs();
	if (std::min<uint32_t>(ps.ps_regs.rsrc2.user_sgpr, base.ps_user.size()) != base.ps_user_count ||
	    !std::equal(base.ps_user.begin(), base.ps_user.begin() + base.ps_user_count,
	                ps.ps_user_sgpr.value)) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::PsUserData);
		return false;
	}
	ShaderVertexInputInfo described;
	const auto params = PrepareProgram(sh.GetVs(), buffer.GetRegisters(), buffer.GetUserConfig(),
	                                   described);
	if (params.hash != base.vs_params_hash || params.user_data_count != base.vs_user_count) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::VsProgram);
		return false;
	}
	uint64_t changed = 0;
	for (uint32_t i = 0; i < params.user_data_count; i++) {
		if (params.user_data[i] != base.vs_user[i]) {
			changed |= 1ull << i;
		}
	}
	// A resource the refresh would resolve differently: the draw needs its own refresh.
	if ((changed & base.vs_plan_reads) != 0) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::PlanReads);
		return false;
	}
	const auto& limits = m_context.GetGraphics().mesh_shader_properties;
	if (instance_count > limits.maxMeshWorkGroupCount[1]) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::Limits);
		return false; // the full path reports it
	}
	const auto primitives = base.mesh.InputPrimitiveCount(vertex_count);
	if (primitives == 0) {
		RenderStats::NoteReplay(RenderStats::ReplayNote::Empty);
		return true; // as ExecutePreparedDraw: nothing is recorded
	}
	// The shader data PrepareBindings would build: the base's, with the user-data words that
	// changed taken from the registers now.
	const std::vector<uint32_t>* data = &base.vs_shader_data;
	if (changed != 0) {
		base.scratch           = base.vs_shader_data;
		const auto& registers  = base.vs_program->bindings.user_data_registers;
		const auto  data_base  = base.vs_program->user_data_base;
		bool        any        = false;
		for (size_t i = 0; i < registers.size() && i < base.scratch.size(); i++) {
			if (registers[i] < data_base || registers[i] - data_base >= params.user_data_count) {
				continue;
			}
			const auto index = registers[i] - data_base;
			if (((changed >> index) & 1u) != 0) {
				base.scratch[i] = params.user_data[index];
				any             = true;
			}
		}
		if (any) {
			data = &base.scratch;
		}
	}
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    vertex_count, 0, first_vertex, instance_count, first_instance);
	if (*data != base.bound_data) {
		if (!RecommitShaderData(buffer, *data)) {
			RenderStats::NoteReplay(RenderStats::ReplayNote::UploadMoved);
			base.valid = false;
			return false;
		}
		base.bound_data = *data;
		RenderStats::NoteReplay(RenderStats::ReplayNote::Replayed);
	} else {
		RenderStats::NoteReplay(RenderStats::ReplayNote::ReplayedSameData);
	}
	const auto vk_buffer = buffer.Recorder();
	// ExecutePreparedDraw's draw dwords for a non-indexed draw with CPU arguments.
	const uint32_t draw_data[] {vertex_count, first_vertex, first_instance, 0u, 0u, 0u, 0u};
	static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
	// The commit's stages (a linked pipeline's push range names every graphics stage).
	const auto mesh_push_stages = m_last_commit.push_stages;
	vk_buffer.pushConstants(base.layout, mesh_push_stages, 0, sizeof(draw_data), draw_data);
	NoteTickWork(scheduler.CurrentTick(), 'D', base.vs_shader_hash, base.ps_shader_hash);
	const uint32_t mesh_groups = (primitives - 1u) / base.mesh.primitives_per_group + 1u;
	constexpr auto base_offset =
	    (ShaderRecompiler::IR::PushData::MeshDrawDwordCount - 1u) * sizeof(uint32_t);
	const auto per_draw = std::min<uint64_t>(
	    limits.maxMeshWorkGroupCount[0],
	    std::max<uint64_t>(1u, limits.maxMeshWorkGroupTotalCount / std::max(instance_count, 1u)));
	uint32_t groups_done = 0;
	do {
		const auto groups =
		    static_cast<uint32_t>(std::min<uint64_t>(mesh_groups - groups_done, per_draw));
		if (groups_done != 0) {
			vk_buffer.pushConstants(base.layout, mesh_push_stages, base_offset, sizeof(groups_done),
			                        &groups_done);
		}
		vk_buffer.drawMeshTasksEXT(groups, instance_count, 1);
		groups_done += groups;
	} while (groups_done < mesh_groups);
	Profiler::Add(Profiler::Counter::Draws);
	return true;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
