#include "common/assert.h"
#include <fmt/format.h>
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/chainWatch.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/barrierReason.h"
#include "graphics/host_gpu/renderer/dispatchRun.h"
#include "graphics/host_gpu/renderer/syncToggle.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <deque>
#include <chrono>
#include <xxhash.h>
#include "graphics/host_gpu/renderer/reuseProbe.h"
#include "graphics/host_gpu/renderer/htileWatch.h"
#include "graphics/host_gpu/renderer/rtSnap.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

std::string DeviceAddressState(uint64_t address); // streamBuffer.cpp
// v443l: the guest address of the last address-trap report, for the table read before the next
// callable split dispatch (DispatchDirectRecord).
static thread_local uint64_t t_rt_trap_peek = 0;

namespace {
// KYTY_CS_WATCH=<guest address> (hex): every dispatch binding a buffer that contains the
// address logs its hash, group counts and buffer sources, with the dwords at the address
// before it runs ([cs-watch]). Finds the compute shader that writes a buffer a draw reads.
uint64_t CsWatchAddress() {
	static const uint64_t address = [] {
		const char* value = std::getenv("KYTY_CS_WATCH");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	return address;
}

void LogCsWatch(uint64_t hash, uint32_t gx, uint32_t gy, uint32_t gz, const PreparedBindings& bindings,
                const uint32_t* users, uint32_t user_count) {
	const uint64_t watched = CsWatchAddress();
	if (watched == 0) return;
	int hit = -1;
	for (size_t i = 0; i < bindings.buffer_sources.size(); i++) {
		const auto& src = bindings.buffer_sources[i];
		if (src.address != 0 && watched >= src.address && watched < src.address + src.size) {
			hit = static_cast<int>(i);
			break;
		}
	}
	if (hit < 0) return;
	static std::atomic<uint32_t> lines {0};
	if (lines.fetch_add(1) >= 4000) return;
	std::string text;
	for (uint32_t i = 0; i < user_count && i < 16u; i++) {
		text += fmt::format("{}{:08x}", i == 0 ? " ud:" : " ", users[i]);
	}
	for (const auto& src: bindings.buffer_sources) {
		text += fmt::format(" buf:{:x}+{:x}", src.address, src.size);
	}
	// KYTY_CS_WATCH_DUMP=<source>:<offset>:<dwords>[:<user sgpr>:<stride>] (hex): the dwords at
	// <offset> (+ user data register × stride) of buffer source <source> of the dispatch.
	static const auto dump = [] {
		std::array<uint32_t, 5> result {0, 0, 0, UINT32_MAX, 0};
		const char*             value = std::getenv("KYTY_CS_WATCH_DUMP");
		if (value != nullptr) {
			char* end = nullptr;
			result[0] = static_cast<uint32_t>(std::strtoul(value, &end, 16));
			if (end != nullptr && *end == ':') result[1] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			if (end != nullptr && *end == ':') result[2] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			if (end != nullptr && *end == ':') result[3] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
			if (end != nullptr && *end == ':') result[4] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
		}
		return result;
	}();
	if (dump[2] != 0 && dump[0] < bindings.buffer_sources.size()) {
		const auto& src    = bindings.buffer_sources[dump[0]];
		uint64_t    offset = dump[1];
		if (dump[3] < user_count) {
			offset += static_cast<uint64_t>(users[dump[3]]) * dump[4];
		}
		text += fmt::format(" dump[{}]+{:x}:", dump[0], offset);
		for (uint32_t i = 0; i < dump[2]; i++) {
			const uint64_t at = src.address + offset + i * 4u;
			uint32_t       w  = 0xdeadbeefu;
			if (src.address != 0 && offset + i * 4u + 4u <= src.size && GuestRange {at, 4}.Valid()) {
				std::memcpy(&w, reinterpret_cast<const void*>(at), 4);
			}
			text += fmt::format(" {:08x}", w);
		}
	}
	// KYTY_CS_WATCH_EMITTERS=<record offset>:<dwords> (hex): source 0 is a table of 0x50-byte
	// entries (count at +192, record index at +204, from +0); print each new record of source 1
	// (stride 0x300) once, the dwords at <record offset> ([cs-emitter]).
	static const std::array<uint32_t, 2> emitters = [] {
		std::array<uint32_t, 2> result {0, 0};
		if (const char* value = std::getenv("KYTY_CS_WATCH_EMITTERS"); value != nullptr) {
			char* end = nullptr;
			result[0] = static_cast<uint32_t>(std::strtoul(value, &end, 16));
			if (end != nullptr && *end == ':') result[1] = static_cast<uint32_t>(std::strtoul(end + 1, &end, 16));
		}
		return result;
	}();
	// The buffer sources' descriptors (stride, records, dword 3), once per shader.
	std::vector<ShaderBufferResource> descriptors;
	if (bindings.runtime != nullptr && bindings.runtime->program != nullptr &&
	    bindings.runtime->resources != nullptr &&
	    !bindings.runtime->program->bindings.descriptors.empty()) {
		for (const auto resource: bindings.runtime->program->bindings.descriptors.front().resources) {
			descriptors.push_back(
			    DecodeNativeDescriptor<ShaderBufferResource>(bindings.runtime->resources->buffers[resource]));
		}
		static std::unordered_set<uint64_t> described;
		if (described.insert(hash).second) {
			std::string d;
			for (size_t i = 0; i < descriptors.size(); i++) {
				d += fmt::format(" [{}]{:x}:s{}:n{:x}:{:08x}", i, descriptors[i].Base48(), descriptors[i].Stride(),
				                 descriptors[i].NumRecords(), descriptors[i].fields[3]);
			}
			std::fprintf(stderr, "[cs-desc] hash=%016llx%s\n", static_cast<unsigned long long>(hash), d.c_str());
		}
	}
	if (emitters[1] != 0 && bindings.buffer_sources.size() > 1) {
		static std::unordered_set<uint64_t> seen;
		const auto& table  = bindings.buffer_sources[0];
		const auto& record = bindings.buffer_sources[1];
		for (uint64_t at = 192; at + 64 <= table.size; at += 0x50) {
			uint32_t entry[16] = {};
			if (!GuestRange {table.address + at, 64}.Valid()) break;
			std::memcpy(entry, reinterpret_cast<const void*>(table.address + at), 64);
			if (entry[0] == 0 || entry[0] > 0x100000u || entry[3] >= 0x800u) continue;
			const uint64_t base = static_cast<uint64_t>(entry[3]) * 0x300u + emitters[0];
			std::string    values;
			uint64_t       key = entry[3] * 0x100000001b3ull ^ entry[15];
			for (uint32_t i = 0; i < emitters[1]; i++) {
				uint32_t w = 0xdeadbeefu;
				if (base + i * 4u + 4u <= record.size && GuestRange {record.address + base + i * 4u, 4}.Valid()) {
					std::memcpy(&w, reinterpret_cast<const void*>(record.address + base + i * 4u), 4);
				}
				key = key * 1099511628211ull ^ w;
				values += fmt::format(" {:08x}", w);
			}
			if (seen.size() < 20000 && seen.insert(key).second) {
				// Curve keys (16 bytes, in the 16-byte-stride sources): for each curve block of the
				// simulation (record + 424, 456, ..., 744: index, key count), the first two keys at
				// record[304] + index.
				const uint64_t rec  = record.address + entry[3] * 0x300u;
				uint32_t       base = 0;
				std::string    keys;
				if (GuestRange {rec, 0x300}.Valid()) {
					std::memcpy(&base, reinterpret_cast<const void*>(rec + 304u), 4);
					static constexpr uint32_t kCurves[] = {424, 456, 488, 520, 552, 584, 616, 648, 680, 712, 744};
					for (size_t i = 2; i < descriptors.size() && i < bindings.buffer_sources.size(); i++) {
						const auto& src = bindings.buffer_sources[i];
						if (descriptors[i].Stride() != 16 || src.address != 0x1210000000ull) continue;
						for (const auto field: kCurves) {
							uint32_t index = 0, count = 0;
							std::memcpy(&index, reinterpret_cast<const void*>(rec + field), 4);
							std::memcpy(&count, reinterpret_cast<const void*>(rec + field + 4u), 4);
							if (count == 0 || count > 64) continue;
							const uint64_t at = static_cast<uint64_t>(base + index) * 16u;
							keys += fmt::format(" c{}n{}:", field, count);
							for (uint32_t w = 0; w < 8u; w++) {
								uint32_t v = 0xdeadbeefu;
								if (at + w * 4u + 4u <= src.size && GuestRange {src.address + at + w * 4u, 4}.Valid()) {
									std::memcpy(&v, reinterpret_cast<const void*>(src.address + at + w * 4u), 4);
								}
								keys += fmt::format(" {:08x}", v);
							}
						}
						break;
					}
				}
				std::fprintf(stderr, "[cs-emitter] hash=%016llx emitter=%x count=%u base=%08x %08x rows=%08x |%s | keys%s\n",
				             static_cast<unsigned long long>(hash), entry[3], entry[0], entry[1], entry[2],
				             entry[15], values.c_str(), keys.c_str());
			}
		}
	}
	uint32_t words[8] = {};
	if (GuestRange {watched, sizeof(words)}.Valid()) {
		std::memcpy(words, reinterpret_cast<const void*>(watched), sizeof(words));
	}
	std::fprintf(stderr,
	             "[cs-watch] hash=%016llx groups=%ux%ux%u hit=%d%s | %08x %08x %08x %08x %08x %08x %08x %08x\n",
	             static_cast<unsigned long long>(hash), gx, gy, gz, hit, text.c_str(), words[0], words[1],
	             words[2], words[3], words[4], words[5], words[6], words[7]);
}
} // namespace

void NoteTickWork(uint64_t tick, char kind, uint64_t a, uint64_t b); // diagnosticCheckpoints.cpp

// KYTY_LOG_SKIPPED_DISPATCHES=<n>: log the first n dispatches of compute shaders that gave up,
// with their user data, to see which guest buffers they would have written.
static bool EarlyDispatchSkip() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EARLY_DISPATCH_SKIP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

static void LogSkippedDispatch(const HW::ComputeShaderInfo& cs, const char* groups) {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_LOG_SKIPPED_DISPATCHES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static std::atomic<uint32_t> used {0};
	if (budget == 0 || used.fetch_add(1, std::memory_order_relaxed) >= budget) {
		return;
	}
	std::string user_data;
	for (uint32_t i = 0; i < cs.cs_user_sgpr.count && i < HW::UserSgprInfo::SGPRS_MAX; i++) {
		char word[12];
		std::snprintf(word, sizeof(word), " %08x", cs.cs_user_sgpr.value[i]);
		user_data += word;
	}
	LOGF("Skipped dispatch: hash=0x%016" PRIx64 " groups=%s user_data=%s\n",
	     ShaderDeclaredHash(cs.cs_regs.data_addr), groups, user_data.c_str());
}
namespace {

// A host GPU zone for a dispatch (gpuProfiler.h), named by its shader.
int64_t BeginGpuZone(RenderContext& context, CommandBuffer& buffer, uint64_t hash) {
	if (!Profiler::g_counters) {
		return -1;
	}
	char      name[32];
	const int size = std::snprintf(name, sizeof(name), "CS %016" PRIx64, hash);
	Profiler::CommandZone::Annotate("CS %016" PRIx64, hash);
	return GpuProfiler::Begin(context, buffer, name, static_cast<size_t>(size), 0xf28e2b);
}

// The rollback below drains the GPU twice per dispatch and almost never fires once a shader's
// pages are resident (one retry in a whole run, against ~2500 protected dispatches a frame). A
// shader is trusted after trust_runs clean attempts and dispatched without it; any fault that
// unprotected work reports re-arms every shader. KYTY_DISPATCH_RECOVERY=always keeps it on every
// dispatch; KYTY_DISPATCH_RECOVERY_TRUST=<n> sets the clean runs needed.
struct DispatchRecoveryPolicy {
	std::unordered_map<uint64_t, uint32_t> clean_runs;
	uint64_t                               seen_faults          = 0;
	uint32_t                               trust_runs           = 4;
	bool                                   always               = false;
	uint64_t                               protected_dispatches = 0;
	uint64_t                               trusted_dispatches   = 0;
	uint64_t                               retries              = 0;
	uint64_t                               rearms               = 0;

	uint64_t saved_bytes = 0, saved_buffers = 0;
	double   drain_ms    = 0;
	uint64_t window_protected = 0, window_trusted = 0, window_rearms = 0, window_retries = 0;
	uint64_t window_saved_bytes = 0, window_saved_buffers = 0;
	double   window_drain_ms = 0;
	std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();

	// [dispatch-recovery] every 5 s on stderr: what the rollback protection cost.
	void Report() {
		const auto now = std::chrono::steady_clock::now();
		if (now - window_start < std::chrono::seconds(5)) {
			return;
		}
		const double seconds = std::chrono::duration<double>(now - window_start).count();
		std::fprintf(stderr,
		             "[dispatch-recovery] %.0f s: protected %" PRIu64 " trusted %" PRIu64
		             " rearms %" PRIu64 " retries %" PRIu64 " saved %" PRIu64 " buffers %" PRIu64
		             " MB drain %.0f ms (unattributed faults so far %" PRIu64 ")\n",
		             seconds, protected_dispatches - window_protected,
		             trusted_dispatches - window_trusted, rearms - window_rearms,
		             retries - window_retries, saved_buffers - window_saved_buffers,
		             (saved_bytes - window_saved_bytes) >> 20u, drain_ms - window_drain_ms,
		             seen_faults);
		window_protected     = protected_dispatches;
		window_trusted       = trusted_dispatches;
		window_rearms        = rearms;
		window_retries       = retries;
		window_saved_bytes   = saved_bytes;
		window_saved_buffers = saved_buffers;
		window_drain_ms      = drain_ms;
		window_start         = now;
	}

	void Count(bool trusted) {
		(trusted ? trusted_dispatches : protected_dispatches)++;
		Report();
		if (((protected_dispatches + trusted_dispatches) & 0x3fff) == 0) {
			LOGF("DispatchRecovery: protected=%" PRIu64 " trusted=%" PRIu64 " retries=%" PRIu64
			     " rearms=%" PRIu64 " shaders=%zu\n",
			     protected_dispatches, trusted_dispatches, retries, rearms, clean_runs.size());
		}
	}
};

// Dispatches are recorded on the GPU thread only.
DispatchRecoveryPolicy& RecoveryPolicy() {
	static DispatchRecoveryPolicy policy = [] {
		DispatchRecoveryPolicy p;
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY")) {
			p.always = std::strcmp(value, "always") == 0;
		}
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY_TRUST")) {
			p.trust_runs = static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
		}
		return p;
	}();
	return policy;
}

// A missing BDA page returns zero during the first attempt. Preserve every
// externally writable buffer, including atomic destinations, so that attempt
// can be rolled back before any dependent dispatch observes its results.
static bool SingleRebind() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SINGLE_REBIND");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

class DispatchBufferRecovery {
public:
	DispatchBufferRecovery(RenderContext& context, const PreparedBindings& bindings,
	                       uint64_t shader_address)
	    : m_context(context), m_shader_address(shader_address) {
		const auto& program = *bindings.runtime->program;
		m_enabled           = program.info.uses_dma &&
		                      std::none_of(program.info.images.begin(), program.info.images.end(),
		                                   [](const auto& image) { return image.written; });
		if (!m_enabled) return;
		auto& cache     = context.GetBufferCache();
		auto& scheduler = context.GetCommandScheduler();
		auto& policy    = RecoveryPolicy();
		if (!policy.always) {
			if (const auto faults = cache.UnattributedFaults(); faults != policy.seen_faults) {
				policy.seen_faults = faults;
				policy.clean_runs.clear();
				policy.rearms++;
			}
			if (policy.clean_runs[shader_address] >= policy.trust_runs) {
				m_enabled = false;
				policy.Count(true);
				return;
			}
		}
		policy.Count(false);
		// Reports from earlier work cannot be attributed to this transaction.
		const auto drain_start = std::chrono::steady_clock::now();
		cache.ProcessFaultBuffer();
		(KYTY_SUBMIT_REASON(), scheduler.Finish());
		policy.drain_ms += std::chrono::duration<double, std::milli>(
		                       std::chrono::steady_clock::now() - drain_start)
		                       .count();
		RangeSet ranges;
		// buffer_sources follows the binding layout's buffers, which can be fewer than the
		// shader's buffer resources (only the live ones are bound).
		if (!bindings.buffer_sources.empty() && !program.bindings.descriptors.empty()) {
			const auto& bound = program.bindings.descriptors.front().resources;
			for (size_t i = 0; i < bindings.buffer_sources.size() && i < bound.size(); ++i) {
				const auto& source = bindings.buffer_sources[i];
				if (!program.info.buffers[bound[i]].written || source.size == 0) continue;
				const auto start = source.address & ~uint64_t {3};
				const auto end   = (source.address + source.size + 3) & ~uint64_t {3};
				ranges.Add(start, end - start);
			}
		}
		ranges.ForEach([&](uint64_t start, uint64_t end) {
			const auto id = cache.FindBuffer(start, end - start);
			cache.SynchronizeBuffersInRange(start, end - start);
			const auto& source = cache.GetBuffer(id);
			Save(source, source.Offset(start), start, end - start, false);
		});
		if (std::any_of(program.bindings.descriptors.begin(), program.bindings.descriptors.end(),
		                [](const auto& binding) {
			                return binding.kind == ShaderRecompiler::IR::DescriptorBindingKind::Gds;
		                })) {
			const auto& source = *cache.GetGdsBuffer();
			Save(source, 0, 0, source.Size(), true);
		}
	}

	bool Retry() {
		if (!m_enabled) return false;
		auto&      cache  = m_context.GetBufferCache();
		const auto report = cache.CollectFaults();
		auto&      policy = RecoveryPolicy();
		if (report.page_count == 0) {
			if (report.trap.claimed != 0) {
				const auto hash =
				    (uint64_t {report.trap.shader_hash_high} << 32) | report.trap.shader_hash_low;
				EXIT("GPU shader trap after page residency: hash=0x%016" PRIx64
				     " pc=0x%08x code=0x%02x values=%08x %08x %08x\n",
				     hash, report.trap.pc, report.trap.code, report.trap.reserved[0],
				     report.trap.reserved[1], report.trap.reserved[2]);
			}
			if (m_attempts == 0) {
				policy.clean_runs[m_shader_address]++;
			}
			return false;
		}
		policy.clean_runs[m_shader_address] = 0;
		policy.retries++;
		// KYTY_RECOVERY_STRICT=1: a dispatch whose pages never become resident ends the session
		// (as before). Otherwise, after 16 attempts it keeps its last result (the missing pages read
		// zero) and the GPU thread goes on: one unbackable page must not end a test run.
		static const bool strict = [] {
			const char* value = std::getenv("KYTY_RECOVERY_STRICT");
			return value != nullptr && value[0] == '1';
		}();
		if (++m_attempts > (strict ? 128u : 16u)) {
			if (strict) {
				EXIT("GPU shader paging did not converge: shader=0x%016" PRIx64 " pages=%" PRIu64 "\n",
				     m_shader_address, report.page_count);
			}
			static std::atomic<uint32_t> gave_up {0};
			if (gave_up.fetch_add(1, std::memory_order_relaxed) < 64) {
				std::fprintf(stderr,
				             "[dispatch-recovery] gave up: shader=0x%016" PRIx64 " pages=%" PRIu64
				             " after %u attempts\n",
				             m_shader_address, report.page_count, m_attempts - 1);
			}
			return false;
		}
		static uint32_t reports = 0;
		if (reports++ < 32) {
			::printf("GPU dispatch retry: shader=0x%016" PRIx64 " attempt=%u missing_pages=%" PRIu64
			         "\n",
			         m_shader_address, m_attempts, report.page_count);
		}
		for (const auto& saved: m_saved) {
			if (saved.gds) {
				cache.GetGdsBuffer()->CopyFrom(m_context.GetCommandScheduler().Current(),
				                               *saved.buffer, 0, 0, saved.buffer->Size());
			} else {
				// Resolving a fault can merge owners; never retain a native destination
				// handle or BufferId across CollectFaults().
				auto [destination, offset] =
				    cache.ObtainBuffer(saved.address, saved.buffer->Size(), true);
				destination->CopyFrom(m_context.GetCommandScheduler().Current(), *saved.buffer, 0,
				                      offset, saved.buffer->Size());
			}
		}
		m_context.PrepareBda();
		return true;
	}

private:
	void Save(const Buffer& source, uint64_t offset, uint64_t address, uint64_t size, bool gds) {
		RecoveryPolicy().saved_bytes += size;
		RecoveryPolicy().saved_buffers++;
		auto saved =
		    std::make_unique<Buffer>(m_context.GetGraphics(), m_context.GetCommandScheduler(),
		                             MemoryUsage::DeviceLocal, 0, AllFlags, size);
		saved->CopyFrom(m_context.GetCommandScheduler().Current(), source, offset, 0, size);
		m_saved.push_back({address, gds, std::move(saved)});
	}
	struct Saved {
		uint64_t                address;
		bool                    gds;
		std::unique_ptr<Buffer> buffer;
	};
	RenderContext&     m_context;
	uint64_t           m_shader_address;
	bool               m_enabled  = false;
	uint32_t           m_attempts = 0;
	std::vector<Saved> m_saved;
};

} // namespace

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

static bool ConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                    const CommandBuffer&          buffer);

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const bool consumed = ConsumeComputeMetaClear(input, buffer);
	if (HtileWatch::MetaClearLog()) {
		const auto& program   = *input.stage.program;
		const auto& resources = *input.stage.resources;
		for (uint32_t i = 0; i < program.info.buffers.size() && i < resources.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (!resource.written) continue;
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			HtileWatch::NoteComputeWrite(program.shader_hash, descriptor.Base48(),
			                             descriptor.GetSize(), resource.read, consumed,
			                             program.info.has_bitwise_xor);
		}
	}
	return consumed;
}

static bool ConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                    const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                 CommandBuffer& command, uint32_t group_x,
                                                 uint32_t group_y, uint32_t group_z,
                                                 uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX)
				return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1])
			return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue                  clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0, bool whole_range = false) {
	if (ShaderRecompiler::IR::FindBinding(input.stage.program->bindings,
	        ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) == nullptr) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	if (whole_range) {
		// v440p: a callable split's GPU-converted indirect launch (group counts unknown here).
		bindings.shared_memory = {storage.Handle(), 0, limit & ~uint64_t {3}};
		return;
	}
	if (indirect_args != 0) {
		// The group counts are GPU data: reading them here made the GPU thread submit and wait
		// for the GPU (CommandScheduler::Wait, forced) on every such dispatch. The shader indexes
		// the buffer by its workgroup id; bind the whole range instead (KYTY_LDS_INDIRECT_READ=1:
		// read the counts as before).
		static const bool read_counts = [] {
			const char* value = std::getenv("KYTY_LDS_INDIRECT_READ");
			return value != nullptr && value[0] == '1';
		}();
		if (!read_counts) {
			bindings.shared_memory = {storage.Handle(), 0, limit & ~uint64_t {3}};
			return;
		}
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

// KYTY_DISPATCH_RUNS: the run's last dispatch orders the whole run for what follows, also when it
// records nothing itself (skipped, empty).
class DispatchRunPost {
public:
	DispatchRunPost(RenderContext& context, CommandBuffer& buffer)
	    : m_context(context), m_buffer(buffer),
	      m_enabled(DispatchRunsEnabled() && DispatchSyncEnabled()) {}
	~DispatchRunPost() {
		if (!m_enabled) return;
		auto& run = DispatchRun();
		if (run.more || !run.pending_post) return;
		run.pending_post = false;
		Common::LockGuard lock(m_context.GetMutex());
		if (!m_buffer.IsInvalid()) {
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnRunPost);
			ShaderAccessBarrier(m_buffer.Recorder(), vk::PipelineStageFlagBits::eComputeShader);
		}
	}
	KYTY_CLASS_NO_COPY(DispatchRunPost);

private:
	RenderContext& m_context;
	CommandBuffer& m_buffer;
	bool           m_enabled;
};

// Diagnostics: KYTY_BVH_CHECK=<hash>:<counter buffer>:<leaf buffer> (bad108e74fb72e9f:5:0). The
// bottom-up BVH refit counts arrivals in a counter per node (bits 0-2; parent index in bits 3-29,
// slot in the parent above) and re-arms each counter when its node completes, so between refits
// every counter of a blob holds the value it started with. At the shader's first dispatch of each
// presented frame the GPU copies of the counters of the blobs refit in the last frames are compared
// with their reference ([bvh-check] on stderr):
// - only count bits differ: a refit left nodes incomplete (printed at once, with the last refits of
//   the blob and the workgroup and thread counts they ran with);
// - a valid armed tree with other links: the blob was instanced again (new reference, silently);
// - anything else: the memory holds other data now; printed only if the shader refits it again.
namespace {
struct BvhRefitNote {
	uint64_t frame     = 0;
	uint64_t seq       = 0;
	bool     continues = false;
	bool     batched   = false;
	uint32_t slot      = UINT32_MAX;
	uint32_t leaves    = 0;
	std::array<uint32_t, 6> counts {}; // groups, threads (filled by the first check after it)
	bool     counts_known = false;
};
struct BvhRecord {
	uint64_t                                    frame = 0, seq = 0, run = 0;
	const uint32_t*                             packet = nullptr;
	bool                                        continues = false, batched = false;
	uint32_t                                    slot = UINT32_MAX;
	std::array<std::pair<uint64_t, uint64_t>, 8> ranges {};
};
struct BvhBlob {
	uint64_t                 size = 0;
	std::vector<uint32_t>    ref;
	std::vector<uint32_t>    garbage; // the last snapshot, when it was neither the tree nor a tree
	std::deque<BvhRefitNote> notes;
	uint64_t                 last_refit = 0;
	bool                     reported   = false;
};
class BvhCheck {
public:
	static BvhCheck& Get() {
		static BvhCheck check;
		return check;
	}
	bool     enabled     = false;
	uint64_t hash        = 0;
	uint32_t counter_idx = 5;
	uint32_t leaf_idx    = 0;

	void OnDispatch(RenderContext& context, const PreparedBindings& bindings, bool batched,
	                vk::Buffer ring, uint64_t ring_bytes, vk::Buffer indirect_buffer,
	                uint64_t indirect_offset) {
		if (counter_idx >= bindings.buffer_sources.size()) return;
		const auto frame = context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		std::lock_guard lock(m_mutex);
		if (frame != m_last_frame) {
			m_last_frame = frame;
			Check(context, frame, ring, ring_bytes);
		}
		const auto& source = bindings.buffer_sources[counter_idx];
		if (source.address == 0 || source.size == 0) return;
		auto& blob = m_blobs[source.address];
		const auto size = std::min<uint64_t>(source.size & ~uint64_t {3}, 0x2000u);
		if (blob.size != size) {
			blob = {};
			blob.size = size;
		}
		if (!blob.garbage.empty()) {
			if (m_reports < 60) {
				m_reports++;
				Print("refit reads foreign counters", frame, source.address, blob, blob.garbage);
			}
			blob.garbage.clear();
			blob.ref.clear(); // re-captured by the next check
		}
		BvhRefitNote note;
		note.frame     = frame;
		note.seq       = ++m_seq;
		note.continues = DispatchRunsEnabled() && DispatchRun().continues;
		note.batched   = batched;
		note.slot      = indirect_buffer == ring ? static_cast<uint32_t>(indirect_offset / 32u) : UINT32_MAX;
		note.leaves    = leaf_idx < bindings.buffer_sources.size()
		                     ? static_cast<uint32_t>(bindings.buffer_sources[leaf_idx].size / 4u)
		                     : 0u;
		blob.notes.push_back(note);
		if (blob.notes.size() > 6) blob.notes.pop_front();
		blob.last_refit = frame;
		if (!note.continues) m_run++;
		BvhRecord record;
		record.frame     = frame;
		record.seq       = note.seq;
		record.run       = m_run;
		record.packet    = DispatchRunsEnabled() ? DispatchRun().packet : nullptr;
		record.continues = note.continues;
		record.batched   = batched;
		record.slot      = note.slot;
		for (size_t i = 0; i < record.ranges.size() && i < bindings.buffer_sources.size(); i++) {
			record.ranges[i] = {bindings.buffer_sources[i].address, bindings.buffer_sources[i].size};
		}
		m_records.push_back(record);
		if (m_records.size() > 8192) m_records.pop_front();
	}

private:
	void Print(const char* what, uint64_t frame, uint64_t address, const BvhBlob& blob,
	           const std::vector<uint32_t>& now) {
		std::string changed;
		uint32_t    count = 0;
		for (size_t k = 0; k < now.size() && k < blob.ref.size(); k++) {
			if (now[k] != blob.ref[k] && count++ < 10) {
				changed += fmt::format(" [{}] {:08x}->{:08x}", k, blob.ref[k], now[k]);
			}
		}
		std::fprintf(stderr, "[bvh-check] %s: frame=%llu blob=0x%llx bytes=0x%llx words changed=%u:%s\n", what,
		             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(address),
		             static_cast<unsigned long long>(blob.size), count, changed.c_str());
		PrintRecords(frame, address, blob.size, 0);
		for (const auto& note: blob.notes) {
			const std::string counts =
			    note.counts_known ? fmt::format(" groups={}x{}x{} threads={}x{}x{}", note.counts[0], note.counts[1],
			                                    note.counts[2], note.counts[3], note.counts[4], note.counts[5])
			                      : std::string(note.slot == UINT32_MAX ? " (direct groups)" : " (counts not captured)");
			std::fprintf(stderr, "[bvh-check]   refit frame=%llu seq=%llu continues=%d batched=%d leaves=%u%s\n",
			             static_cast<unsigned long long>(note.frame), static_cast<unsigned long long>(note.seq),
			             note.continues ? 1 : 0, note.batched ? 1 : 0, note.leaves, counts.c_str());
		}
	}

	// Every recent dispatch whose bindings overlap [address, address + size) (any binding).
	void PrintRecords(uint64_t frame, uint64_t address, uint64_t size, uint64_t focus) {
		uint32_t listed = 0;
		for (auto it = m_records.rbegin(); it != m_records.rend() && listed < 12; ++it) {
			if (it->frame + 4 < frame) break;
			int hit = -1;
			for (size_t b = 0; b < it->ranges.size(); b++) {
				const auto [a, n] = it->ranges[b];
				if (n != 0 && a < address + size && address < a + n) {
					hit = static_cast<int>(b);
					break;
				}
			}
			if (hit < 0) continue;
			listed++;
			std::string bufs;
			for (size_t b = 0; b < it->ranges.size(); b++) {
				if (it->ranges[b].second != 0) {
					bufs += fmt::format(" b{}=0x{:x}+0x{:x}", b, it->ranges[b].first, it->ranges[b].second);
				}
			}
			std::fprintf(stderr,
			             "[bvh-check]   %sdispatch frame=%llu seq=%llu run=%llu packet=%p continues=%d batched=%d slot=%u "
			             "hit=b%d%s\n",
			             it->seq == focus ? "* " : "", static_cast<unsigned long long>(it->frame), static_cast<unsigned long long>(it->seq),
			             static_cast<unsigned long long>(it->run), static_cast<const void*>(it->packet),
			             it->continues ? 1 : 0, it->batched ? 1 : 0, it->slot, hit, bufs.c_str());
		}
	}

	static bool ValidArmed(const std::vector<uint32_t>& words) {
		for (const auto w: words) {
			if (w == 0) continue;
			const auto count  = w & 7u;
			const auto parent = (w >> 3u) & 0x7ffffffu;
			if (count == 0 || count > 4 || parent >= words.size()) return false;
		}
		return true;
	}

	void Check(RenderContext& context, uint64_t frame, vk::Buffer ring, uint64_t ring_bytes) {
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		std::vector<uint64_t>                      keys;
		for (const auto& [address, blob]: m_blobs) {
			if (blob.last_refit + 3 >= frame) {
				ranges.emplace_back(address, blob.size);
				keys.push_back(address);
			}
		}
		if (ranges.empty()) return;
		context.GetBufferCache().SnapshotGpu(
		    ranges, ring, 0, ring_bytes,
		    [this, frame, keys = std::move(keys)](std::vector<std::vector<uint32_t>>& words,
		                                          std::vector<uint32_t>&              ring_words) {
			    std::lock_guard lock(m_mutex);
			    m_checks++;
			    for (size_t i = 0; i < keys.size() && i < words.size(); i++) {
				    auto found = m_blobs.find(keys[i]);
				    if (found == m_blobs.end() || words[i].empty()) continue;
				    auto& blob = found->second;
				    if (words[i].size() * 4u != blob.size) continue;
				    // Counts of the refits since the last check: their ring slots are current.
				    for (auto& note: blob.notes) {
					    if (!note.counts_known && note.slot != UINT32_MAX && note.frame + 2 >= frame &&
					        (note.slot + 1u) * 8u <= ring_words.size()) {
						    const auto* e = &ring_words[note.slot * 8u];
						    note.counts       = {e[0], e[1], e[2], e[4], e[5], e[6]};
						    note.counts_known = true;
						    // Four threads a leaf: thread counts that disagree with the leaf list the
						    // dispatch bound ran this refit on another instance's arguments.
						    if (note.leaves != 0 && note.counts[3] != note.leaves * 4u) {
							    m_args_mismatch++;
							    if (m_mismatch_reports < 40) {
								    m_mismatch_reports++;
								    std::fprintf(stderr,
								                 "[bvh-check] refit arguments disagree with its leaf list: frame=%llu "
								                 "seq=%llu blob=0x%llx leaves=%u threads=%u groups=%u\n",
								                 static_cast<unsigned long long>(note.frame),
								                 static_cast<unsigned long long>(note.seq),
								                 static_cast<unsigned long long>(keys[i]), note.leaves, note.counts[3],
								                 note.counts[0]);
								    PrintRecords(frame, keys[i], blob.size, note.seq);
							    }
						    }
					    }
				    }
				    const auto& now = words[i];
				    if (blob.ref.empty()) {
					    if (ValidArmed(now)) blob.ref = now;
					    continue;
				    }
				    uint32_t count_only = 0, structural = 0, decremented = 0;
				    for (size_t k = 0; k < now.size(); k++) {
					    if (now[k] == blob.ref[k]) continue;
					    ((now[k] & ~7u) == (blob.ref[k] & ~7u) ? count_only : structural)++;
					    // Lowered by a few arrivals, borrowing into the parent index or not.
					    if (blob.ref[k] - now[k] <= 8u) decremented++;
				    }
				    if (structural != 0 && decremented == count_only + structural) {
					    // Refit arrivals beyond a node's count: the parent links are now wrong.
					    m_underflow++;
					    if (m_reports < 60) {
						    m_reports++;
						    Print("counters decremented past zero", frame, keys[i], blob, now);
					    }
					    blob.garbage = now;
					    continue;
				    }
				    if (count_only == 0 && structural == 0) {
					    m_matches++;
					    continue;
				    }
				    if (structural != 0) {
					    const bool valid = ValidArmed(now);
					    if (m_reports < 60) {
						    m_reports++;
						    Print(valid ? "counters now a different valid tree" : "counters now foreign", frame,
						          keys[i], blob, now);
					    }
					    if (valid) {
						    m_reinstanced++;
						    blob.ref = now;
					    } else {
						    m_foreign++;
						    blob.garbage = now;
					    }
					    continue;
				    }
				    m_partial++;
				    if (!blob.reported && m_reports < 60) {
					    blob.reported = true;
					    m_reports++;
					    Print("counters left incomplete", frame, keys[i], blob, now);
				    }
			    }
			    const auto t = std::chrono::steady_clock::now();
			    if (t - m_last_report > std::chrono::seconds(10)) {
				    m_last_report = t;
				    std::fprintf(stderr,
				                 "[bvh-check] blobs=%zu checks=%llu match=%llu incomplete=%llu reinstanced=%llu "
				                 "foreign=%llu underflow=%llu args-mismatch=%llu\n",
				                 m_blobs.size(), static_cast<unsigned long long>(m_checks),
				                 static_cast<unsigned long long>(m_matches), static_cast<unsigned long long>(m_partial),
				                 static_cast<unsigned long long>(m_reinstanced),
				                 static_cast<unsigned long long>(m_foreign),
				                 static_cast<unsigned long long>(m_underflow),
				                 static_cast<unsigned long long>(m_args_mismatch));
			    }
		    });
	}

	BvhCheck() {
		if (const char* value = std::getenv("KYTY_BVH_CHECK"); value != nullptr) {
			char* end = nullptr;
			hash      = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') counter_idx = std::strtoul(end + 1, &end, 10);
			if (end != nullptr && *end == ':') leaf_idx = std::strtoul(end + 1, &end, 10);
			enabled = hash != 0;
		}
	}
	std::recursive_mutex                  m_mutex;
	std::unordered_map<uint64_t, BvhBlob> m_blobs;
	uint64_t                              m_last_frame = UINT64_MAX;
	uint64_t                              m_seq        = 0;
	uint64_t m_checks = 0, m_matches = 0, m_partial = 0, m_reinstanced = 0, m_foreign = 0, m_underflow = 0;
	uint64_t                              m_run = 0;
	uint64_t                              m_args_mismatch = 0;
	uint32_t                              m_mismatch_reports = 0;
	std::deque<BvhRecord>                 m_records;
	uint32_t                              m_reports = 0;
	std::chrono::steady_clock::time_point m_last_report = std::chrono::steady_clock::now();
};
} // namespace

// Dispatches whose ordering the guest stream leaves out but which still need it (they keep their
// barriers inside runs). Marvel's Wolverine: 06544f382c6b0850 and 3378b4a64461d6ca (cs 0x11d04f8000
// and 0x11d04f3600, 14-26 groups each); overlapped inside a run they hung the GPU about 20 s into
// gameplay in about 1 boot in 3-4 (0 in 10 with their barriers kept, at the same frame rate).
// KYTY_DISPATCH_RUN_EXCLUDE=<hex shader hashes or cs addresses> replaces the list (empty: none);
// KYTY_DISPATCH_RUN_ONLY=<...> keeps runs for the listed shaders only (diagnostic).
static thread_local bool t_run_excluded = false;

static void NoteRunShader(uint64_t cs_addr, uint64_t shader_hash) {
	struct Lists {
		std::vector<uint64_t> exclude, only;
	};
	static const Lists lists = [] {
		Lists l;
		const auto parse = [](const char* value, std::vector<uint64_t>& out) {
			if (value == nullptr) return;
			for (const char* p = value; *p != '\0';) {
				char* end = nullptr;
				const auto v = std::strtoull(p, &end, 16);
				if (end == p) {
					p++;
					continue;
				}
				out.push_back(v);
				p = end;
			}
		};
		const char* exclude = std::getenv("KYTY_DISPATCH_RUN_EXCLUDE");
		// v445y (Mac 11-21, applied): no exclusions by default. With the barriers of v445 and later
		// these two overlap safely: no GPU hang in 10 boots (g451e-l, g452a/b), in-run A/B -1.2 and
		// -3.0 ms a frame (GPU -2.0 and -4.3 ms). KYTY_DISPATCH_RUN_EXCLUDE restores a list.
		parse(exclude != nullptr ? exclude : "", l.exclude);
		parse(std::getenv("KYTY_DISPATCH_RUN_ONLY"), l.only);
		return l;
	}();
	const auto listed = [cs_addr, shader_hash](const std::vector<uint64_t>& list) {
		return std::find(list.begin(), list.end(), cs_addr) != list.end() ||
		       std::find(list.begin(), list.end(), shader_hash) != list.end();
	};
	// v445w: "runoverlap" sync-toggle windows ignore the exclusion list (scene-matched A/B).
	t_run_excluded = (listed(lists.exclude) && !SyncToggle::RunOverlapWindow()) ||
	                 (!lists.only.empty() && !listed(lists.only));
}

// Whether the barrier before this dispatch can be left out (an earlier dispatch of its run was
// recorded last), and the one after it (another dispatch of the run follows).
static bool RunSkipsBarrierBefore() {
	return DispatchRunsEnabled() && DispatchRun().continues && !t_run_excluded;
}
static bool RunSkipsBarrierAfter() {
	if (!DispatchRunsEnabled() || !DispatchRun().more || t_run_excluded) return false;
	DispatchRun().pending_post = true;
	DispatchRun().skipped_barriers++;
	return true;
}
// KYTY_BARRIER_REASON_STATS (barrierReason.h): the dispatch about to be closed by GpuTiming::After.
static void NoteBarrierReasonDispatch(RenderContext& context) {
	if (!BarrierReason::Enabled()) return;
	const bool runs = DispatchRunsEnabled() && DispatchSyncEnabled();
	BarrierReason::SetDispatch(context.GetCommandScheduler().CurrentTick(),
	                           runs ? DispatchRun().why : static_cast<uint32_t>(BarrierReason::WhyRunsOff),
	                           t_run_excluded);
}

// The renderer mutex for a dispatch, at a tracked depth (RenderContext::LockRenderer) so the
// async-compute thread's program-lookup window can release it.
class DispatchRendererLock {
public:
	explicit DispatchRendererLock(RenderContext& context): m_context(context) {
		KYTY_PROFILER_BLOCK("Renderer mutex wait");
		m_context.LockRenderer();
	}
	~DispatchRendererLock() { m_context.UnlockRenderer(); }
	KYTY_CLASS_NO_COPY(DispatchRendererLock);

private:
	RenderContext& m_context;
};

// KYTY_ASYNC_THREAD: on the async-compute thread, the program lookup and SRT refresh
// (GetComputeProgram, with that thread's own program cache) run without the renderer mutex.
// Afterwards the thread's registers are bound to the scheduler's command buffer again.
class ProgramLookupWindow {
public:
	explicit ProgramLookupWindow(RenderContext& context): m_context(context) {
		if (GuestGpu::IsAsyncThread() && context.GetPipelineCache().HasAsyncProgramCache()) {
			m_active = true;
			m_depth  = context.ReleaseRenderer();
		}
	}
	~ProgramLookupWindow() {
		if (m_active) {
			KYTY_PROFILER_BLOCK("Renderer mutex wait (after program lookup)");
			m_context.ReacquireRenderer(m_depth);
			GuestGpu::RebindCurrentProcessor();
		}
	}
	KYTY_CLASS_NO_COPY(ProgramLookupWindow);

private:
	RenderContext& m_context;
	uint32_t       m_depth  = 0;
	bool           m_active = false;
};

void QueueStatsNoteDispatch(uint64_t program_ns, uint64_t total_ns); // graphicsRun.cpp
void QueueStatsNoteDispatchParts(uint64_t prepare_ns, uint64_t record_ns); // graphicsRun.cpp
void QueueStatsNoteBindings(uint64_t shader_hash, uint64_t bindings_hash, uint64_t srt_hash); // graphicsRun.cpp
uint32_t QueueStatsQueue(); // graphicsRun.cpp
namespace {
// KYTY_QUEUE_STATS, [dispatch-delta]: for a dispatch whose shader is the same as the previous
// dispatch's on its queue, how many binding slots differ from that dispatch's (buffers, images,
// samplers) and how many flattened-SRT words. The fps plan's delta path (rebind only what changed)
// pays only if most such dispatches change few slots.
struct DispatchDeltaState {
	uint64_t                                          shader = 0;
	std::vector<ShaderRecompiler::IR::DescriptorValue> buffers, images, samplers;
	std::vector<uint32_t>                             srt;
};
struct DispatchDeltaStats {
	std::atomic<uint64_t> dispatches {0}, same {0}, changed_buffers {0}, changed_images {0}, changed_samplers {0};
	std::atomic<uint64_t> srt_changed {0}, srt_words {0}, slots {0}, srt_only {0};
	std::array<std::atomic<uint64_t>, 6> histogram {}; // changed slots: 0, 1, 2, 3, 4, 5+
	std::atomic<int64_t> last_report {0};
};
DispatchDeltaStats g_delta;

template <typename T>
uint32_t CountChanged(std::vector<T>& previous, const std::vector<T>& current) {
	uint32_t changed = static_cast<uint32_t>(previous.size() > current.size() ? previous.size() - current.size()
	                                                                          : current.size() - previous.size());
	const size_t n = std::min(previous.size(), current.size());
	for (size_t i = 0; i < n; ++i) changed += previous[i] == current[i] ? 0u : 1u;
	previous = current;
	return changed;
}

void NoteDispatchDelta(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSnapshot& r) {
	thread_local std::array<DispatchDeltaState, 64> states;
	auto&      state = states[QueueStatsQueue() % states.size()];
	const bool same  = state.shader == shader_hash;
	state.shader     = shader_hash;
	const auto b     = CountChanged(state.buffers, r.buffers);
	const auto i     = CountChanged(state.images, r.images);
	const auto s     = CountChanged(state.samplers, r.samplers);
	const auto w     = CountChanged(state.srt, r.flattened_srt);
	g_delta.dispatches.fetch_add(1, std::memory_order_relaxed);
	g_delta.slots.fetch_add(r.buffers.size() + r.images.size() + r.samplers.size(), std::memory_order_relaxed);
	if (same) {
		const auto changed = b + i + s;
		g_delta.same.fetch_add(1, std::memory_order_relaxed);
		g_delta.changed_buffers.fetch_add(b, std::memory_order_relaxed);
		g_delta.changed_images.fetch_add(i, std::memory_order_relaxed);
		g_delta.changed_samplers.fetch_add(s, std::memory_order_relaxed);
		g_delta.histogram[std::min(changed, 5u)].fetch_add(1, std::memory_order_relaxed);
		if (w != 0) {
			g_delta.srt_changed.fetch_add(1, std::memory_order_relaxed);
			g_delta.srt_words.fetch_add(w, std::memory_order_relaxed);
			g_delta.srt_only.fetch_add(changed == 0 ? 1u : 0u, std::memory_order_relaxed);
		}
	}
	const auto now  = std::chrono::steady_clock::now().time_since_epoch();
	const auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
	auto       last = g_delta.last_report.load(std::memory_order_relaxed);
	if (last == 0) {
		g_delta.last_report.compare_exchange_strong(last, ms);
		return;
	}
	if (ms - last < 5000 || !g_delta.last_report.compare_exchange_strong(last, ms)) return;
	const double seconds  = static_cast<double>(ms - last) / 1000.0;
	const auto   all      = g_delta.dispatches.exchange(0);
	const auto   same_n   = g_delta.same.exchange(0);
	const double per_same = same_n != 0 ? 1.0 / static_cast<double>(same_n) : 0.0;
	std::array<double, 6> h {};
	for (size_t k = 0; k < h.size(); ++k) h[k] = 100.0 * static_cast<double>(g_delta.histogram[k].exchange(0)) * per_same;
	const auto srt_changed = g_delta.srt_changed.exchange(0);
	std::fprintf(stderr,
	             "[dispatch-delta] %.0f dispatches/s (%.1f slots each), %.0f/s same shader as the previous on "
	             "the queue: changed slots 0 %.0f%% 1 %.0f%% 2 %.0f%% 3 %.0f%% 4 %.0f%% 5+ %.0f%%; changed per "
	             "dispatch buffers %.2f images %.2f samplers %.2f; SRT changed in %.0f%% (%.1f words), SRT only "
	             "%.0f%%\n",
	             static_cast<double>(all) / seconds,
	             all != 0 ? static_cast<double>(g_delta.slots.exchange(0)) / static_cast<double>(all) : 0.0,
	             static_cast<double>(same_n) / seconds, h[0], h[1], h[2], h[3], h[4], h[5],
	             static_cast<double>(g_delta.changed_buffers.exchange(0)) * per_same,
	             static_cast<double>(g_delta.changed_images.exchange(0)) * per_same,
	             static_cast<double>(g_delta.changed_samplers.exchange(0)) * per_same,
	             100.0 * static_cast<double>(srt_changed) * per_same,
	             srt_changed != 0 ? static_cast<double>(g_delta.srt_words.exchange(0)) / static_cast<double>(srt_changed)
	                              : 0.0,
	             100.0 * static_cast<double>(g_delta.srt_only.exchange(0)) * per_same);
}
} // namespace

static void NoteDispatchBindings(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSnapshot& r) {
	static const bool enabled = std::getenv("KYTY_QUEUE_STATS") != nullptr;
	if (!enabled) return;
	NoteDispatchDelta(shader_hash, r);
	uint64_t h = XXH3_64bits(r.buffers.data(), r.buffers.size() * sizeof(r.buffers[0]));
	h ^= XXH3_64bits(r.images.data(), r.images.size() * sizeof(r.images[0])) * 0x9e3779b97f4a7c15ull;
	h ^= XXH3_64bits(r.samplers.data(), r.samplers.size() * sizeof(r.samplers[0])) * 0xc2b2ae3d27d4eb4full;
	const uint64_t srt = XXH3_64bits(r.flattened_srt.data(), r.flattened_srt.size() * 4u);
	QueueStatsNoteBindings(shader_hash, h, srt);
}
namespace {
// KYTY_QUEUE_STATS: a dispatch's whole time and its program-lookup part, per guest queue. The
// clock is read only when the stats are on (two to four clock reads a dispatch otherwise, ~1% of
// the GPU thread at 50k dispatches a second).
bool QueueTimingEnabled() {
	static const bool enabled = std::getenv("KYTY_QUEUE_STATS") != nullptr;
	return enabled;
}
struct QueueDispatchTimer {
	bool                                  on    = QueueTimingEnabled();
	std::chrono::steady_clock::time_point begin = on ? std::chrono::steady_clock::now()
	                                                 : std::chrono::steady_clock::time_point {};
	uint64_t                              program_ns = 0;
	~QueueDispatchTimer() {
		if (!on) {
			return;
		}
		QueueStatsNoteDispatch(program_ns,
		                       static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                 std::chrono::steady_clock::now() - begin)
		                                                 .count()));
	}
};
struct ScopedNs {
	uint64_t&                             out;
	bool                                  on    = QueueTimingEnabled();
	std::chrono::steady_clock::time_point begin = on ? std::chrono::steady_clock::now()
	                                                 : std::chrono::steady_clock::time_point {};
	~ScopedNs() {
		if (!on) {
			return;
		}
		out += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now() - begin)
		                                 .count());
	}
};
} // namespace

// v445b, KYTY_DISPATCH_DELTA (render.h, DispatchDelta). A guest dispatch of the shader the previous
// dispatch on its queue ran, in the same command buffer, builds on what that one resolved:
// - read-only buffer bindings whose source is the same and CPU-clean are taken over
//   (RenderExecutor::DeltaSlotReusable); written bindings, record-time copies and everything else
//   are resolved as before;
// - the pipeline lookup is left out (same program, same dispatch kind);
// - when nothing touched the compute bind point since (CommandScheduler::ComputeStateSerial), the
//   pipeline is not bound again and only the descriptors that differ are pushed
//   (RenderExecutor::PushDescriptorsDelta). Push constants are always pushed again.
// Not for callable splits (ray-tracing pipelines, bound sets), not without the recording thread
// (a direct recorder cannot note what it binds), not after a recovery retry.
// KYTY_DISPATCH_DELTA=0: off. KYTY_DISPATCH_DELTA_VERIFY=1: each delta dispatch also resolves its
// buffers and looks its pipeline up in full; a difference is logged ([dispatch-delta-verify]), and
// the dispatch then commits and binds in full, so the run draws what the full path draws.
bool DispatchDeltaEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DISPATCH_DELTA");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}
bool DispatchDeltaVerify() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DISPATCH_DELTA_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
namespace {
// [dispatch-delta-path], with KYTY_QUEUE_STATS or the verify, every 5 s.
struct DispatchDeltaPathStats {
	std::atomic<uint64_t> dispatches {0}, based {0}, slots {0}, reused {0}, writes {0}, pushed {0}, kept {0};
	std::atomic<uint64_t> verified {0}, mismatches {0};
	std::atomic<int64_t>  last_report {0};
};
DispatchDeltaPathStats g_delta_path;
void ReportDispatchDeltaPath() {
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count();
	auto last = g_delta_path.last_report.load(std::memory_order_relaxed);
	if (last == 0) {
		g_delta_path.last_report.compare_exchange_strong(last, ms);
		return;
	}
	if (ms - last < 5000 || !g_delta_path.last_report.compare_exchange_strong(last, ms)) return;
	const double seconds = static_cast<double>(ms - last) / 1000.0;
	const auto   all     = g_delta_path.dispatches.exchange(0);
	const auto   based   = g_delta_path.based.exchange(0);
	const auto   slots   = g_delta_path.slots.exchange(0);
	const auto   writes  = g_delta_path.writes.exchange(0);
	std::fprintf(stderr,
	             "[dispatch-delta-path] %.0f dispatches/s, %.0f/s on a base (%.0f%%): buffer slots taken over "
	             "%.0f%%, bind point kept %.0f%%, descriptors pushed %.0f%% | verify: %llu compared, %llu "
	             "different\n",
	             static_cast<double>(all) / seconds, static_cast<double>(based) / seconds,
	             all != 0 ? 100.0 * static_cast<double>(based) / static_cast<double>(all) : 0.0,
	             slots != 0 ? 100.0 * static_cast<double>(g_delta_path.reused.exchange(0)) / static_cast<double>(slots)
	                        : 0.0,
	             based != 0 ? 100.0 * static_cast<double>(g_delta_path.kept.exchange(0)) / static_cast<double>(based)
	                        : 0.0,
	             writes != 0 ? 100.0 * static_cast<double>(g_delta_path.pushed.exchange(0)) / static_cast<double>(writes)
	                         : 0.0,
	             static_cast<unsigned long long>(g_delta_path.verified.exchange(0)),
	             static_cast<unsigned long long>(g_delta_path.mismatches.exchange(0)));
}
bool DispatchDeltaStats() {
	return QueueTimingEnabled() || DispatchDeltaVerify();
}
} // namespace

const RenderExecutor::DispatchDelta* RenderExecutor::DispatchDeltaBase(const void* owner,
                                                                      const ShaderComputeInputInfo& input,
                                                                      const ShaderProgram& program) const {
	const auto& base      = m_dispatch_delta;
	auto&       scheduler = m_context.GetCommandScheduler();
	if (!DispatchDeltaEnabled() || !base.valid || !scheduler.Threaded() || base.owner != owner ||
	    base.program != static_cast<const void*>(input.stage.program) || base.program_id != program.id ||
	    program.rt_callables != nullptr || base.thread_dimensions != input.dispatch_thread_dimensions ||
	    base.dimensions_indirect != input.dispatch_dimensions_indirect ||
	    base.tick != scheduler.CurrentTick() || base.command_serial != scheduler.CommandSerial() ||
	    base.unregister_epoch != m_context.GetBufferCache().UnregisterEpoch()) {
		return nullptr;
	}
	return &base;
}

// Whether the compute bind point still holds what the base's dispatch left: its pipeline bound and
// its descriptors pushed, in this command buffer.
bool RenderExecutor::DispatchDeltaStateKept(const DispatchDelta* delta, const void* pipeline,
                                            vk::Pipeline handle) const {
	const auto& scheduler = m_context.GetCommandScheduler();
	return delta != nullptr && delta->pipeline == pipeline && delta->bound_pipeline == handle && handle &&
	       delta->tick == scheduler.CurrentTick() && delta->command_serial == scheduler.CommandSerial() &&
	       delta->compute_serial == scheduler.ComputeStateSerial();
}

// v445d. The base describes the command buffer and the bind point as they were when its dispatch
// bound its pipeline, not as they are when the dispatch function ends: the recovery of a
// dispatch that reads through the address table collects its faults after the dispatch
// (DispatchBufferRecovery::Retry), which records a pass of its own and flushes. Taken at the end,
// the tick and the serial were the new command buffer's, in which nothing is bound, and the next
// dispatch of the shader kept a bind point that did not exist (g448c).
RenderExecutor::DispatchDeltaMark RenderExecutor::MarkDispatchDelta(uint64_t unregister_epoch) const {
	const auto& scheduler = m_context.GetCommandScheduler();
	return {scheduler.CurrentTick(), scheduler.CommandSerial(), scheduler.ComputeStateSerial(), unregister_epoch,
	        true};
}

void RenderExecutor::RebindBuffersDelta(PreparedBindings& bindings, const DispatchDelta*& delta,
                                        uint64_t shader_hash) {
	m_delta_reused = 0;
	if (delta == nullptr) {
		RebindBuffers(bindings);
		return;
	}
	m_delta_reuse = delta;
	RebindBuffers(bindings);
	m_delta_reuse = nullptr;
	if (DispatchDeltaStats()) {
		g_delta_path.slots.fetch_add(bindings.buffers.size(), std::memory_order_relaxed);
		g_delta_path.reused.fetch_add(m_delta_reused, std::memory_order_relaxed);
	}
	if (!DispatchDeltaVerify()) {
		return;
	}
	// The full path, over the same sources; its result is what the dispatch uses.
	m_delta_verify_buffers = bindings.buffers;
	m_delta_verify_data    = bindings.shader_data;
	const auto reused      = m_delta_reused;
	RebindBuffers(bindings);
	g_delta_path.verified.fetch_add(1, std::memory_order_relaxed);
	// A record-time copy (BufferCache::ObtainBuffer: a CPU-modified range of at most a cache page is
	// copied into the stream buffer, a new slice each time) is the same binding when both resolves
	// copied the same guest range: the slice's place differs, and with it the slot's offset byte
	// in the shader data, which is taken from the full path before the data is compared.
	auto&      verify_cache = m_context.GetBufferCache();
	const auto stream       = verify_cache.GetUtilityBuffer(MemoryUsage::Stream).Handle();
	const auto upload       = verify_cache.GetUtilityBuffer(MemoryUsage::Upload).Handle();
	const auto copy_equal   = [&](const vk::DescriptorBufferInfo& d, const vk::DescriptorBufferInfo& f) {
		return d.buffer == f.buffer && (d.buffer == stream || d.buffer == upload) && d.range == f.range;
	};
	const auto slot_equal = [&](const vk::DescriptorBufferInfo& d, const vk::DescriptorBufferInfo& f) {
		return (d.buffer == f.buffer && d.offset == f.offset && d.range == f.range) || copy_equal(d, f);
	};
	bool same = m_delta_verify_buffers.size() == bindings.buffers.size() &&
	            m_delta_verify_data.size() == bindings.shader_data.size();
	if (same && bindings.runtime != nullptr && *bindings.runtime) {
		const auto offset_dword = bindings.runtime->program->bindings.memory_offset_dword;
		for (size_t i = 0; i < bindings.buffers.size(); i++) {
			const size_t dword = offset_dword + i / 4u;
			if (copy_equal(m_delta_verify_buffers[i], bindings.buffers[i]) && dword < bindings.shader_data.size()) {
				const uint32_t mask        = 0xffu << ((i % 4u) * 8u);
				m_delta_verify_data[dword] = (m_delta_verify_data[dword] & ~mask) | (bindings.shader_data[dword] & mask);
			}
		}
	}
	same = same && m_delta_verify_data == bindings.shader_data;
	for (size_t i = 0; same && i < bindings.buffers.size(); i++) {
		same = slot_equal(m_delta_verify_buffers[i], bindings.buffers[i]);
	}
	if (!same) {
		g_delta_path.mismatches.fetch_add(1, std::memory_order_relaxed);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
			std::string text;
			for (size_t i = 0; i < bindings.buffers.size() && i < m_delta_verify_buffers.size(); i++) {
				const auto& d = m_delta_verify_buffers[i];
				const auto& f = bindings.buffers[i];
				if (!slot_equal(d, f)) {
					text += fmt::format(" slot {} (guest 0x{:x}+0x{:x}): delta 0x{:x}+0x{:x}/0x{:x}, full 0x{:x}+0x{:x}/0x{:x};",
					                    i, i < bindings.buffer_sources.size() ? bindings.buffer_sources[i].address : 0u,
					                    i < bindings.buffer_sources.size() ? bindings.buffer_sources[i].size : 0u,
					                    reinterpret_cast<uint64_t>(static_cast<VkBuffer>(d.buffer)), d.offset, d.range,
					                    reinterpret_cast<uint64_t>(static_cast<VkBuffer>(f.buffer)), f.offset, f.range);
				}
			}
			std::fprintf(stderr,
			             "[dispatch-delta-verify] %016llx: the delta path's bindings differ from the full path's "
			             "(%u slots taken over, shader data %s):%s\n",
			             static_cast<unsigned long long>(shader_hash), reused,
			             m_delta_verify_data == bindings.shader_data ? "same" : "DIFFERENT", text.c_str());
		}
	}
	delta = nullptr;
}

void RenderExecutor::CaptureDispatchDelta(const void* owner, const ShaderComputeInputInfo& input,
                                          const ShaderProgram& program, void* pipeline, vk::Pipeline handle,
                                          const PreparedBindings& bindings, const DispatchDeltaMark& mark,
                                          bool usable) {
	auto& base = m_dispatch_delta;
	if (!DispatchDeltaEnabled()) {
		return;
	}
	// Nothing of the base stands when the command buffer its dispatch was recorded in is not the
	// current one any more (a flush between the bind and here): its bindings were obtained for
	// that tick, and its bind point went with that command buffer.
	base.valid = usable && mark.set && program.rt_callables == nullptr &&
	             mark.tick == m_context.GetCommandScheduler().CurrentTick() &&
	             mark.command_serial == m_context.GetCommandScheduler().CommandSerial();
	if (!base.valid) {
		base.writes_valid = false;
		return;
	}
	auto& scheduler          = m_context.GetCommandScheduler();
	base.owner               = owner;
	base.program             = static_cast<const void*>(input.stage.program);
	base.program_id          = program.id;
	base.pipeline            = pipeline;
	base.thread_dimensions   = input.dispatch_thread_dimensions;
	base.dimensions_indirect = input.dispatch_dimensions_indirect;
	base.tick                = mark.tick;
	base.command_serial      = mark.command_serial;
	base.unregister_epoch    = mark.unregister_epoch;
	base.compute_serial      = mark.compute_serial;
	base.bound_pipeline      = handle;
	base.sources             = bindings.buffer_sources;
	base.buffers             = bindings.buffers;
	base.shader_data         = bindings.shader_data;
	if (DispatchDeltaStats()) {
		ReportDispatchDeltaPath();
	}
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	QueueDispatchTimer queue_timer;
	DispatchRunPost run_post(m_context, buffer);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	if (ReuseProbe::Enabled()) {
		ReuseProbe::NoteDispatch(sh_ctx, thread_group_x, thread_group_y, thread_group_z, 0);
	}
	// The zone is the wait for the renderer mutex (another thread holding it).
	DispatchRendererLock lock(m_context);
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}
	// The skip list's ray-tracing dispatches are most of this title's dispatches in the jungle
	// (~2,600 skip-list lookups a frame); drop them before preparing their inputs and program.
	// KYTY_EARLY_DISPATCH_SKIP=0 drops them in ProgramCache::Get as before.
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(sh_ctx.GetCs().cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		ResetBindings();
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions      = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	ShaderProgram compute_program;
	{
		KYTY_PROFILER_BLOCK("Dispatch::GetComputeProgram");
		ScopedNs program_time {queue_timer.program_ns};
		ProgramLookupWindow window(m_context);
		compute_program =
		    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	if (!compute_program && PipelineCache::SrtDeferPending()) {
		// Suspended until the SRT bytes it reads land (KYTY_SRT_DEFER); not skipped.
		ResetBindings();
		return;
	}
	if (!compute_program) {
		char groups[48];
		std::snprintf(groups, sizeof(groups), "%ux%ux%u", thread_group_x, thread_group_y,
		              thread_group_z);
		LogSkippedDispatch(cs_regs, groups);
		ResetBindings();
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0] = thread_group_x;
		input_info.dispatch_threads_num[1] = thread_group_y;
		input_info.dispatch_threads_num[2] = thread_group_z;
	}

	DispatchDirectRecord(submit_id, buffer, cs_regs, input_info, compute_program, thread_group_x,
	                     thread_group_y, thread_group_z, mode);
}

void RenderExecutor::DispatchDirectRecord(uint64_t submit_id, CommandBuffer& buffer,
                                          const HW::ComputeShaderInfo& cs_regs,
                                          ShaderComputeInputInfo&      input_info,
                                          const ShaderProgram&         compute_program,
                                          uint32_t thread_group_x, uint32_t thread_group_y,
                                          uint32_t thread_group_z, uint32_t mode) {
	(void)submit_id;
	const bool use_thread_dimensions = input_info.dispatch_thread_dimensions;
	const auto& program   = *input_info.stage.program;
	{
		const CpuCost::Phase notes_phase_("Dispatch::Notes");
		NoteRunShader(cs_regs.cs_regs.data_addr, program.shader_hash);
		NoteDispatchBindings(program.shader_hash, *input_info.stage.resources);
		NoteTickWork(m_context.GetCommandScheduler().CurrentTick(), 'C', program.shader_hash, cs_regs.cs_regs.data_addr);
	}
	const auto& resources = *input_info.stage.resources;
	// KYTY_LOG_CS_IMAGES=1: each compute shader's written images (extent, format), once per shape.
	if (static const bool log_images = std::getenv("KYTY_LOG_CS_IMAGES") != nullptr; log_images) {
		static std::unordered_set<uint64_t> seen;
		static std::mutex                   seen_mutex; // v440y: the GPU and async-compute threads both dispatch
		std::lock_guard                     seen_lock(seen_mutex);
		for (uint32_t i = 0; i < program.info.images.size() && i < resources.images.size(); i++) {
			// v440w: reads too, with the guest address (KYTY_LOG_CS_IMAGES=2: written only, as before).
			static const bool written_only = std::string_view(std::getenv("KYTY_LOG_CS_IMAGES")) == "2";
			if (!program.info.images[i].written && written_only) continue;
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const uint64_t key = program.shader_hash ^ (uint64_t {r.Width5()} << 1u) ^
			                     (uint64_t {r.Height5()} << 17u) ^ (uint64_t {static_cast<uint32_t>(r.Format())} << 33u) ^ (uint64_t {i} << 45u) ^
			                     (r.Base40() * 0x9e3779b97f4a7c15ull);
			if (seen.insert(key).second) {
				std::fprintf(stderr, "[cs-image] hash=%016llx image[%u] %s addr=%llx %ux%ux%u type=%u fmt=%u groups=%ux%ux%u\n",
				             static_cast<unsigned long long>(program.shader_hash), i,
				             program.info.images[i].written ? (program.info.images[i].read ? "rw" : "w") : "r",
				             static_cast<unsigned long long>(r.Base40()),
				             static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
				             static_cast<uint32_t>(r.Depth()) + 1u, static_cast<uint32_t>(r.Type()),
				             static_cast<uint32_t>(r.Format()), thread_group_x, thread_group_y, thread_group_z);
			}
		}
	}
	{
		KYTY_PROFILER_BLOCK("Dispatch::TryConsumeClears");
		if (resources.specialization_reads.empty() &&
		    (TryConsumeComputeMetaClear(input_info, buffer) ||
		     TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
		                                 thread_group_z, mode))) {
			ResetBindings();
			return;
		}
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	// KYTY_QUEUE_STATS: the parts after the program lookup ([queue-stats2]).
	struct PartsNote {
		using Clock = std::chrono::steady_clock;
		bool              on    = QueueTimingEnabled();
		Clock::time_point begin = on ? Clock::now() : Clock::time_point {};
		Clock::time_point mid {};
		void              Mid() {
			if (on) mid = Clock::now();
		}
		~PartsNote() {
			if (!on || mid == Clock::time_point {}) return;
			const auto end = Clock::now();
			QueueStatsNoteDispatchParts(
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(mid - begin).count()),
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count()));
		}
	} parts_note;
	{
		const CpuCost::Phase end_phase_("Dispatch::EndRendering");
		buffer.EndRendering();
	}
	// v445b (KYTY_DISPATCH_DELTA): the previous dispatch recorded here, when this one may build on it.
	std::optional<CpuCost::Phase> pipeline_phase_(std::in_place, "Dispatch::PipelineLookup");
	const DispatchDelta* delta = DispatchDeltaBase(&cs_regs, input_info, compute_program);
	using ComputePipeline = std::remove_reference_t<
	    decltype(m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program))>;
	auto& pipeline = delta != nullptr && !DispatchDeltaVerify()
	                     ? *static_cast<ComputePipeline*>(delta->pipeline)
	                     : m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	pipeline_phase_.reset();
	if (delta != nullptr && delta->pipeline != static_cast<void*>(&pipeline)) {
		// KYTY_DISPATCH_DELTA_VERIFY: the lookup the delta path leaves out gave another pipeline.
		std::fprintf(stderr, "[dispatch-delta-verify] %016llx: the base's pipeline is not the lookup's\n",
		             static_cast<unsigned long long>(program.shader_hash));
		delta = nullptr;
	}
	if (DispatchDeltaEnabled() && (QueueTimingEnabled() || DispatchDeltaVerify())) {
		g_delta_path.dispatches.fetch_add(1, std::memory_order_relaxed);
		g_delta_path.based.fetch_add(delta != nullptr ? 1u : 0u, std::memory_order_relaxed);
	}
	if (pipeline.rt_ready != nullptr && !pipeline.rt_ready->load(std::memory_order_acquire)) {
		// v440 M3: its ray-tracing pipeline is still being built (KYTY_RT_ASYNC).
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::fprintf(stderr, "[rt-callables] %016llx: dispatch skipped while its pipeline builds\n",
			             static_cast<unsigned long long>(program.shader_hash));
		}
		ResetBindings();
		return;
	}
	if (t_rt_trap_peek != 0 && pipeline.ray_tracing) {
		// v443l: what the GPU's address table holds for the page of the last trapped address and its
		// neighbours, against what the buffer cache has there. Read here, before anything of this
		// dispatch is recorded (the read submits and waits). The first six only.
		// KYTY_RT_TRAP_PEEK=0: not read.
		static const bool peek = [] {
			const char* value = std::getenv("KYTY_RT_TRAP_PEEK");
			return value == nullptr || value[0] != '0';
		}();
		static std::atomic<uint32_t> peeked {0};
		const auto                   guest = std::exchange(t_rt_trap_peek, 0);
		if (peek && peeked.fetch_add(1, std::memory_order_relaxed) < 6) {
			auto& cache = m_context.GetBufferCache();
			for (int64_t step = -1; step <= 1; step++) {
				const uint64_t page = (guest & ~uint64_t {BufferCache::CACHING_PAGESIZE - 1}) +
				                      static_cast<uint64_t>(step * static_cast<int64_t>(BufferCache::CACHING_PAGESIZE));
				uint64_t entry = 0, expected = 0;
				if (!cache.PeekBdaEntry(page, entry, expected)) {
					std::fprintf(stderr, "[rt-callables] address trap: guest page 0x%llx is not guest memory the table covers\n",
					             static_cast<unsigned long long>(page));
					continue;
				}
				std::fprintf(stderr,
				             "[rt-callables] address trap: guest page 0x%llx: the GPU's table holds 0x%016llx (%s), the buffer "
				             "cache has %s there (0x%016llx)%s\n",
				             static_cast<unsigned long long>(page), static_cast<unsigned long long>(entry),
				             entry == 0 ? "no page" : DeviceAddressState(entry).c_str(),
				             expected != 0 ? "a buffer" : "no buffer", static_cast<unsigned long long>(expected),
				             entry == expected ? "" : ": THEY DIFFER");
			}
		}
	}
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
		if (pipeline.ray_tracing) {
			// v440 M3: a callable split's 1D launch finds its workgroup ids from its thread extents
			// (whole groups). v440p: the shader reads them by device address: GPU-converted from an
			// indirect dispatch's group counts, or written here.
			IndirectDispatchGroups::RtLaunch launch {};
			if (m_rt_indirect_launch != nullptr) {
				launch = *m_rt_indirect_launch;
			} else {
				if (m_indirect_groups == nullptr) {
					m_indirect_groups = std::make_unique<IndirectDispatchGroups>(m_context.GetGraphics(),
					                                                             m_context.GetCommandScheduler());
				}
				const uint32_t groups[3] = {thread_group_x, thread_group_y, thread_group_z};
				std::array<uint32_t, 3> extents {};
				for (uint32_t axis = 0; axis < 3u; axis++) {
					extents[axis] = groups[axis] * std::max(input_info.threads_num[axis], 1u);
				}
				launch = m_indirect_groups->WriteRtExtents(buffer.Recorder(), extents);
			}
			bindings.shader_data[program.bindings.dispatch_thread_dword + 0u] = static_cast<uint32_t>(launch.extents);
			bindings.shader_data[program.bindings.dispatch_thread_dword + 1u] =
			    static_cast<uint32_t>(launch.extents >> 32u);
			bindings.shader_data[program.bindings.dispatch_thread_dword + 2u] = 0;
			bindings.shader_data[program.bindings.dispatch_thread_dword + 3u] =
			    static_cast<uint32_t>(pipeline.rt_target_table);
			bindings.shader_data[program.bindings.dispatch_thread_dword + 4u] =
			    static_cast<uint32_t>(pipeline.rt_target_table >> 32u);
		}
	}
	FindBuffers(bindings);
	LogCsWatch(program.shader_hash, thread_group_x, thread_group_y, thread_group_z, bindings,
	           cs_regs.cs_user_sgpr.value, cs_regs.cs_user_sgpr.count);
	if (program.info.uses_dma) {
		KYTY_PROFILER_BLOCK("Dispatch::PrepareBda");
		m_context.PrepareBda();
	}
	// The dispatch loop below resolves the images again before recording, and nothing in between
	// reads them: with single rebind, this first pass only repeated each image's view lookup
	// (KYTY_HOTPATH).
	if (!SingleRebind() || !SyncToggle::HotPath()) RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings, 0, m_rt_indirect_launch != nullptr);
	// The dispatch loop below binds the buffers again before recording; this first pass only
	// repeated every buffer's sync and the flattened SRT upload (KYTY_SINGLE_REBIND=0: as before).
	if (!SingleRebind()) RebindBuffers(bindings);

	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, cs_regs.cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	parts_note.Mid();
	std::optional<CpuCost::Phase> recovery_phase_(std::in_place, "Dispatch::RecoveryBegin");
	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	recovery_phase_.reset();
	bool              delta_retried = false;
	DispatchDeltaMark delta_mark; // the scheduler's state at this dispatch's bind
	do {
		if (std::exchange(delta_retried, true)) {
			delta = nullptr; // a recovery retry: everything again, in full
		}
		RebindImages(bindings);
		uint64_t join_epoch = m_context.GetBufferCache().UnregisterEpoch();
		RebindBuffersDelta(bindings, delta, program.shader_hash);
		{
			const CpuCost::Phase join_phase_("Dispatch::AfterJoin");
			PreparedBindings* join_stage = &bindings;
			(void)RebindAfterJoin(std::span {&join_stage, 1u}, join_epoch, "dispatch");
			ChainWatch::Note(program.shader_hash,
			                 m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
			                 cs_regs.cs_user_sgpr.value, cs_regs.cs_user_sgpr.count, bindings.buffer_sources);
		}
		const auto        vk_buffer        = buffer.Recorder();
		PreparedBindings* descriptor_stage = &bindings;
		const auto        bind_point       = pipeline.ray_tracing ? vk::PipelineBindPoint::eRayTracingKHR
		                                                          : vk::PipelineBindPoint::eCompute;
		// v445b: the bind point as the base left it (nothing bound or recorded raw since, same
		// command buffer): only the descriptors that differ are pushed, and the pipeline stays bound.
		const bool delta_kept = DispatchDeltaStateKept(delta, &pipeline, pipeline.pipeline);
		m_delta_push          = delta_kept;
		CommitBindings(buffer, bind_point, pipeline, std::span {&descriptor_stage, 1u});
		m_delta_push = false;
		if (delta != nullptr && DispatchDeltaStats()) {
			g_delta_path.kept.fetch_add(delta_kept ? 1u : 0u, std::memory_order_relaxed);
			g_delta_path.writes.fetch_add(m_delta_writes, std::memory_order_relaxed);
			g_delta_path.pushed.fetch_add(m_delta_pushed, std::memory_order_relaxed);
		}
		// KYTY_CPU_COST: from here to the end of the pass, the recording: the barrier bookkeeping
		// ("Dispatch::BarrierBefore"/"After" inside it), the bind, the timing marks and the dispatch.
		const CpuCost::Phase record_phase_("Dispatch::Record");
		NoteFaultMapBindings(true);
		KYTY_PROFILER_BLOCK("Dispatch::Record");
		std::optional<CpuCost::Phase> barrier_phase_(std::in_place, "Dispatch::BarrierBefore");
		bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
		    HasShaderBufferWrites(input_info.stage);
		has_storage_writes =
		    std::any_of(program.info.images.begin(), program.info.images.end(),
		                [](const auto& image) {
			                return image.written &&
			                       image.resource_class ==
			                           ShaderRecompiler::IR::ImageResourceClass::Storage;
		                }) ||
		    has_storage_writes;
		if (has_storage_writes) {
			// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
			// while allowing the queue to execute asynchronously.
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnPre);
			if (DispatchSyncEnabled() && !SyncToggle::SkipDispatchBarrier() && !RunSkipsBarrierBefore()) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		if (t_run_excluded && DispatchRunsEnabled() && DispatchRun().continues &&
		    DispatchSyncEnabled()) {
			// The dispatch before it left out its barrier after (KYTY_DISPATCH_RUN_EXCLUDE).
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnExcl);
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		barrier_phase_.reset();
		// v440 M3: a callable split runs in the ray-tracing stages, which the compute barriers around
		// dispatches do not name: everything before it is finished first, and it is finished before
		// anything after it.
		vk::MemoryBarrier rt_barrier {};
		rt_barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		rt_barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		if (pipeline.ray_tracing) {
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnRt);
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
			                          {}, 1, &rt_barrier, 0, nullptr, 0, nullptr);
		}
		if (!(delta_kept && DispatchDeltaStateKept(delta, &pipeline, pipeline.pipeline))) {
			vk_buffer.bindPipeline(bind_point, pipeline.pipeline);
		}
		delta_mark = MarkDispatchDelta(join_epoch);
		if (pipeline.ray_tracing && pipeline.rt_stack_bytes != 0) {
			vk_buffer.setRayTracingPipelineStackSizeKHR(pipeline.rt_stack_bytes); // v443q (a test)
		}
		GpuTiming::Before(m_context, buffer);
		const auto gpu_zone = BeginGpuZone(m_context, buffer, program.shader_hash);
		{
			// Name each compute shader once by its code address, so a device-fault checkpoint
			// (which prints the address) can be matched to a hash for KYTY_SKIP_SHADER_HASHES.
			// Repeated dispatches of one shader skip the set (the common case in dispatch runs).
			static std::unordered_set<uint64_t> named;
			static thread_local uint64_t        last_named = ~uint64_t {0};
			if (const auto addr = static_cast<uint64_t>(cs_regs.cs_regs.data_addr);
			    addr != last_named && (last_named = addr, named.insert(addr).second)) {
				std::fprintf(stderr, "[cs] addr=0x%016" PRIx64 " hash=0x%016" PRIx64 "\n",
				             static_cast<uint64_t>(cs_regs.cs_regs.data_addr), program.shader_hash);
			}
		}
		if (pipeline.ray_tracing) {
			const uint64_t threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
			                         std::max(input_info.threads_num[1], 1u) * std::max(input_info.threads_num[2], 1u);
			const uint64_t width = uint64_t {thread_group_x} * thread_group_y * thread_group_z * threads;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr, "[rt-callables] %016llx: trace %llu invocations (groups %ux%ux%u x %llu threads)\n",
				             static_cast<unsigned long long>(program.shader_hash), static_cast<unsigned long long>(width),
				             thread_group_x, thread_group_y, thread_group_z, static_cast<unsigned long long>(threads));
			}
			if (m_rt_indirect_launch != nullptr) {
				vk_buffer.traceRaysIndirectKHR(pipeline.rt_raygen_region, pipeline.rt_callable_region,
				                               m_rt_indirect_launch->command);
			} else if (width != 0 && width <= 0x3fffffffu) {
				vk_buffer.traceRaysKHR(pipeline.rt_raygen_region, pipeline.rt_callable_region,
				                       static_cast<uint32_t>(width), 1, 1);
			}
			if (pipeline.rt_counters != nullptr && pipeline.rt_counters[7] != pipeline.rt_cap_hits) {
				// v443e: callables that hit their loop cap (IR::RtCallableLoopLimit) since the last look.
				// Each returned early, so what it shades may be wrong; without the cap it held the GPU.
				const uint32_t hits = pipeline.rt_counters[7];
				const uint32_t code = pipeline.rt_counters[8];
				std::string    where = "a callable not in this pipeline";
				if (compute_program.rt_callables != nullptr) {
					const auto& known = compute_program.rt_callables->code_hashes;
					for (size_t i = 0; i < known.size() && i < compute_program.rt_callables->targets.size(); i++) {
						if (known[i] == code) {
							where = fmt::format("callable {} (guest code 0x{:x})", i, compute_program.rt_callables->targets[i]);
							break;
						}
					}
				}
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 200) {
					std::fprintf(stderr,
					             "[rt-callables] %016llx gen %u: %u callable invocations hit the loop cap (%u new); the last: %s, "
					             "code hash %08x, loop decision at pc 0x%x, lane %u of subgroup lanes %08x (%08x when the "
					             "callable started), launch id %u\n",
					             static_cast<unsigned long long>(program.shader_hash),
					             compute_program.rt_callables != nullptr ? compute_program.rt_callables->generation : 0u, hits,
					             hits - pipeline.rt_cap_hits, where.c_str(), code, pipeline.rt_counters[9],
					             pipeline.rt_counters[10], pipeline.rt_counters[11], pipeline.rt_counters[13], pipeline.rt_counters[12]);
				}
				pipeline.rt_cap_hits = hits;
			}
			if (pipeline.rt_counters != nullptr && pipeline.rt_counters[19] != pipeline.rt_trap_hits) {
				// v443j, the address trap (IR/RtCallable.h): a guest address whose page-table entry put
				// it at N * 4 GiB plus less than a page was not read. Whether a buffer really is there
				// (then the trap took a real page away, and KYTY_RT_TRAP_ADDR=0 is the answer) is in
				// the line: the buffer cache's own buffer for that guest address.
				const uint32_t hits   = pipeline.rt_counters[19];
				const uint64_t device = (uint64_t {pipeline.rt_counters[21]} << 32u) | pipeline.rt_counters[20];
				const uint64_t guest  = (uint64_t {pipeline.rt_counters[23]} << 32u) | pipeline.rt_counters[22];
				const uint32_t code   = pipeline.rt_counters[24];
				std::string    where  = "the ray generation or a callable not in this pipeline";
				if (compute_program.rt_callables != nullptr) {
					const auto& known = compute_program.rt_callables->code_hashes;
					for (size_t i = 0; i < known.size() && i < compute_program.rt_callables->targets.size(); i++) {
						if (known[i] == code) {
							where = fmt::format("callable {} (guest code 0x{:x})", i, compute_program.rt_callables->targets[i]);
							break;
						}
					}
				}
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 200) {
					// v443l: which address (IR/RtCallable.h, +100).
					const uint32_t site = pipeline.rt_counters[25];
					const char*    what = site == 0   ? "guest address table"
					                      : site == 1 ? "launch extents"
					                      : site == 3 ? "write bitmap"
					                                  : "unknown site";
					std::fprintf(stderr,
					             "[rt-callables] %016llx gen %u: address trap, %u reads not made (%u new); the last (%s): guest "
					             "0x%llx -> dev 0x%llx in %s, code hash %08x; this pipeline's binding table is at 0x%llx\n",
					             static_cast<unsigned long long>(program.shader_hash),
					             compute_program.rt_callables != nullptr ? compute_program.rt_callables->generation : 0u, hits,
					             hits - pipeline.rt_trap_hits, what, static_cast<unsigned long long>(guest),
					             static_cast<unsigned long long>(device), where.c_str(), code,
					             static_cast<unsigned long long>(pipeline.rt_target_table));
				}
				pipeline.rt_trap_hits = hits;
				if (pipeline.rt_counters[25] == 0) {
					t_rt_trap_peek = guest; // looked at before the next split's dispatch is recorded
				}
			}
			if (pipeline.rt_counters != nullptr && pipeline.rt_count) {
				// v440x (KYTY_RT_COUNT=1): the call-site counters, at most every 2 s per shader.
				static std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> printed;
				const auto now  = std::chrono::steady_clock::now();
				auto&      last = printed[program.shader_hash];
				if (now - last > std::chrono::seconds(2)) {
					last = now;
					std::fprintf(stderr, "[rt-count] %016llx gen %u: lanes at the call %u, outside the target set %u, calling %u, back %u; "
					             "intersections %u, no child/invalid %u, triangle missed %u\n",
					             static_cast<unsigned long long>(program.shader_hash),
					             compute_program.rt_callables != nullptr ? compute_program.rt_callables->generation : 0u,
					             pipeline.rt_counters[0], pipeline.rt_counters[1], pipeline.rt_counters[2], pipeline.rt_counters[6],
					             pipeline.rt_counters[3], pipeline.rt_counters[4], pipeline.rt_counters[5]);
				}
			}
			if (compute_program.rt_callables != nullptr) {
				auto& scheduler = m_context.GetCommandScheduler();
				m_context.GetPipelineCache().NoteRtDispatch(
				    program.shader_hash, compute_program.rt_callables->generation, compute_program.id,
				    scheduler.CurrentTick(), [&scheduler](uint64_t tick) { return scheduler.IsFree(tick); });
			}
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
			                          {}, 1, &rt_barrier, 0, nullptr, 0, nullptr);
		} else {
			vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
		}
		RtSnap::BufferProbeAfterDispatch(m_context, program.shader_hash, bindings, cs_regs.cs_user_sgpr.value,
		                                 cs_regs.cs_user_sgpr.count);
		if (RtSnap::Enabled()) {
			PreparedBindings* const snap_stage = &bindings;
			RtSnap::AfterCommand(m_context, "dispatch", std::span(&snap_stage, 1));
		}
		Profiler::Add(Profiler::Counter::Dispatches);
		GpuProfiler::End(buffer, gpu_zone);
		NoteBarrierReasonDispatch(m_context);
		GpuTiming::After(m_context, buffer, program.shader_hash, GpuTiming::Dispatch);
		Timeline::Mark("dispatch", program.shader_hash,
		               (static_cast<uint64_t>(thread_group_x) << 32u) |
		                   (thread_group_y << 16u) | thread_group_z);

		// The removed host fence also ordered read-only dispatches before later writers.
		const CpuCost::Phase barrier_after_phase_("Dispatch::BarrierAfter");
		if (DispatchSyncEnabled() && !SyncToggle::SkipDispatchBarrier() && !RunSkipsBarrierAfter()) {
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnPost);
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
			if (DispatchRunsEnabled()) DispatchRun().pending_post = false;
		}
	} while ([&recovery] {
		const CpuCost::Phase retry_phase_("Dispatch::RecoveryEnd");
		return recovery.Retry();
	}());
	const CpuCost::Phase capture_phase_("Dispatch::CaptureDelta");
	CaptureDispatchDelta(&cs_regs, input_info, compute_program, &pipeline, pipeline.pipeline, bindings, delta_mark,
	                     !pipeline.ray_tracing && m_rt_indirect_launch == nullptr);
	ResetBindings();
}

namespace {
// G4 stage 1: the async-compute thread's queued direct dispatches (QueueDispatchDirect).
struct QueuedDispatch {
	uint64_t                                 submit_id = 0;
	HW::ComputeShaderInfo                    cs_regs {};
	ShaderComputeInputInfo                   input_info {};
	ShaderProgram                            program {};
	ShaderRecompiler::IR::ResourceSnapshot   resources;
	uint32_t                                 groups[3] {};
	uint32_t                                 mode = 0;
	bool                                     indirect  = false;
	uint64_t                                 args_addr = 0;
	// Nothing to record (the dispatch was dropped when queued), only its place in its dispatch
	// run: a run's last dispatch closes the run (DispatchRunPost) even when it is dropped.
	bool                                     post_only = false;
	// DispatchRun() as the command processor set it for this dispatch.
	const uint32_t*                          run_packet    = nullptr;
	bool                                     run_continues = false;
	bool                                     run_more      = false;
	uint32_t                                 run_why       = 0; // KYTY_BARRIER_REASON_STATS
	uint64_t                                 program_ns    = 0;
};
struct DispatchQueue {
	std::vector<std::unique_ptr<QueuedDispatch>> entries; // reused, so the snapshots keep their storage
	size_t                                       count       = 0;
	uint64_t                                     last_shader = 0; // CS address of the last entry
};
thread_local DispatchQueue t_dispatch_queue;

QueuedDispatch& NextQueuedEntry(uint64_t shader_addr) {
	auto& queue = t_dispatch_queue;
	if (queue.count == queue.entries.size()) {
		queue.entries.push_back(std::make_unique<QueuedDispatch>());
	}
	queue.last_shader = shader_addr;
	auto& entry       = *queue.entries[queue.count];
	entry.post_only   = false;
	return entry;
}

// A dropped dispatch keeps its place in its dispatch run (QueuedDispatch::post_only).
void QueueRunPostOnly(uint64_t shader_addr) {
	if (!DispatchRunsEnabled() || !DispatchSyncEnabled()) {
		return;
	}
	auto&       entry   = NextQueuedEntry(shader_addr);
	const auto& run     = DispatchRun();
	entry.post_only     = true;
	entry.run_packet    = run.packet;
	entry.run_continues = run.continues;
	entry.run_more      = run.more;
	entry.run_why       = run.why;
	entry.program_ns    = 0;
	t_dispatch_queue.count++;
}

// Marks the thread off-lock unless it already is (the command processor marks the whole packet).
struct OffLockMark {
	bool marked = !RenderContext::RendererReleased();
	OffLockMark() {
		if (marked) RenderContext::MarkOffLock(true);
	}
	~OffLockMark() {
		if (marked) RenderContext::MarkOffLock(false);
	}
	KYTY_CLASS_NO_COPY(OffLockMark);
};
} // namespace

uint64_t RenderExecutor::LastQueuedShader() noexcept {
	return t_dispatch_queue.count != 0 ? t_dispatch_queue.last_shader : 0;
}

size_t RenderExecutor::QueuedDispatchCount() noexcept {
	return t_dispatch_queue.count;
}

void RenderExecutor::QueueDispatchDirect(uint64_t submit_id, const HW::ComputeShaderInfo& cs_regs,
                                         const HW::ShaderRegisters& sh_regs, uint32_t thread_group_x,
                                         uint32_t thread_group_y, uint32_t thread_group_z,
                                         uint32_t mode) {
	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 " (queued)\n",
			     thread_group_x, thread_group_y, thread_group_z, mode, cs_regs.cs_regs.data_addr);
		}
		QueueRunPostOnly(cs_regs.cs_regs.data_addr);
		return;
	}
	if (cs_regs.cs_regs.data_addr == 0) {
		QueueRunPostOnly(0);
		return;
	}
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(cs_regs.cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		QueueRunPostOnly(cs_regs.cs_regs.data_addr);
		return;
	}
	auto& queue = t_dispatch_queue;
	auto& entry = NextQueuedEntry(cs_regs.cs_regs.data_addr);

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	ShaderComputeInputInfo& input_info = entry.input_info;
	input_info                         = {};
	const bool use_thread_dimensions   = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0]        = thread_group_x;
	input_info.workgroup_counts[1]        = thread_group_y;
	input_info.workgroup_counts[2]        = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size    = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	// v445o (KYTY_TIMERS_CUT): program_ns is read by [queue-stats] only (QueueTimingEnabled).
	const bool program_timed = QueueTimingEnabled() || !SyncToggle::TimersCut();
	if (!program_timed) {
		SyncToggle::Bump(SyncToggle::Hits().program_untimed);
	}
	const auto program_begin = program_timed ? std::chrono::steady_clock::now()
	                                         : std::chrono::steady_clock::time_point {};
	ShaderProgram compute_program;
	{
		KYTY_PROFILER_BLOCK("Dispatch::GetComputeProgram (queued)");
		const OffLockMark off_lock;
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	const auto program_ns =
	    program_timed
	        ? static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                    std::chrono::steady_clock::now() - program_begin)
	                                    .count())
	        : uint64_t {0};
	if (!compute_program) {
		if (!PipelineCache::SrtDeferPending()) {
			char groups[48];
			std::snprintf(groups, sizeof(groups), "%ux%ux%u", thread_group_x, thread_group_y,
			              thread_group_z);
			LogSkippedDispatch(cs_regs, groups);
		}
		// Suspended until the SRT bytes it reads land (KYTY_SRT_DEFER), or skipped.
		if (QueueTimingEnabled()) QueueStatsNoteDispatch(program_ns, program_ns);
		QueueRunPostOnly(cs_regs.cs_regs.data_addr);
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0] = thread_group_x;
		input_info.dispatch_threads_num[1] = thread_group_y;
		input_info.dispatch_threads_num[2] = thread_group_z;
	}
	// The snapshot lives in the program cache's entry, which the next lookup of the shader
	// overwrites: the queued dispatch keeps a copy.
	entry.resources             = *input_info.stage.resources;
	input_info.stage.resources  = &entry.resources;
	entry.submit_id             = submit_id;
	entry.cs_regs               = cs_regs;
	entry.program               = compute_program;
	entry.groups[0]             = thread_group_x;
	entry.groups[1]             = thread_group_y;
	entry.groups[2]             = thread_group_z;
	entry.mode                  = mode;
	entry.indirect              = false;
	entry.args_addr             = 0;
	const auto& run             = DispatchRun();
	entry.run_packet            = run.packet;
	entry.run_continues         = run.continues;
	entry.run_more              = run.more;
	entry.run_why               = run.why;
	entry.program_ns            = program_ns;
	queue.count++;
}

void RenderExecutor::QueueDispatchIndirect(uint64_t submit_id, const HW::ComputeShaderInfo& cs_regs,
                                           const HW::ShaderRegisters& sh_regs, uint64_t args_addr,
                                           uint32_t mode) {
	EXIT_IF(args_addr == 0 || (args_addr & 3u) != 0);
	if (cs_regs.cs_regs.data_addr == 0) {
		QueueRunPostOnly(0);
		return;
	}
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(cs_regs.cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		QueueRunPostOnly(cs_regs.cs_regs.data_addr);
		return;
	}
	auto& queue = t_dispatch_queue;
	auto& entry = NextQueuedEntry(cs_regs.cs_regs.data_addr);

	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	ShaderComputeInputInfo& input_info      = entry.input_info;
	input_info                              = {};
	input_info.dispatch_thread_dimensions   = use_thread_dimensions;
	input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	// v445o (KYTY_TIMERS_CUT): program_ns is read by [queue-stats] only (QueueTimingEnabled).
	const bool program_timed = QueueTimingEnabled() || !SyncToggle::TimersCut();
	if (!program_timed) {
		SyncToggle::Bump(SyncToggle::Hits().program_untimed);
	}
	const auto program_begin = program_timed ? std::chrono::steady_clock::now()
	                                         : std::chrono::steady_clock::time_point {};
	ShaderProgram compute_program;
	{
		KYTY_PROFILER_BLOCK("Dispatch::GetComputeProgram (queued indirect)");
		const OffLockMark off_lock;
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	const auto program_ns =
	    program_timed
	        ? static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                    std::chrono::steady_clock::now() - program_begin)
	                                    .count())
	        : uint64_t {0};
	if (!compute_program) {
		if (!PipelineCache::SrtDeferPending()) {
			LogSkippedDispatch(cs_regs, "indirect");
		}
		if (QueueTimingEnabled()) QueueStatsNoteDispatch(program_ns, program_ns);
		QueueRunPostOnly(cs_regs.cs_regs.data_addr);
		return;
	}
	entry.resources            = *input_info.stage.resources;
	input_info.stage.resources = &entry.resources;
	entry.submit_id            = submit_id;
	entry.cs_regs              = cs_regs;
	entry.program              = compute_program;
	entry.groups[0] = entry.groups[1] = entry.groups[2] = 0;
	entry.mode                 = mode;
	entry.indirect             = true;
	entry.args_addr            = args_addr;
	const auto& run            = DispatchRun();
	entry.run_packet           = run.packet;
	entry.run_continues        = run.continues;
	entry.run_more             = run.more;
	entry.run_why              = run.why;
	entry.program_ns           = program_ns;
	queue.count++;
}

void RenderExecutor::CommitQueuedDispatches(CommandBuffer& buffer) {
	auto& queue = t_dispatch_queue;
	if (queue.count == 0) {
		return;
	}
	// Restored after: the command processor's current packet may be a dispatch of the run.
	auto&      run            = DispatchRun();
	const auto saved_packet   = run.packet;
	const auto saved_continue = run.continues;
	const auto saved_more     = run.more;
	const auto saved_why      = run.why;
	for (size_t i = 0; i < queue.count; i++) {
		auto& entry = *queue.entries[i];
		EXIT_IF(buffer.IsInvalid());
		if (entry.post_only) {
			run.packet    = entry.run_packet;
			run.continues = entry.run_continues;
			run.more      = entry.run_more;
			run.why       = entry.run_why;
			const DispatchRunPost run_post(m_context, buffer);
			continue;
		}
		const auto commit_begin = QueueTimingEnabled() ? std::chrono::steady_clock::now()
		                                               : std::chrono::steady_clock::time_point {};
		run.packet    = entry.run_packet;
		run.continues = entry.run_continues;
		run.more      = entry.run_more;
		run.why       = entry.run_why;
		{
			DispatchRunPost run_post(m_context, buffer);
			m_context.GetCommandScheduler().PopPendingOperations();
			if (entry.indirect) {
				buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect),
				                    entry.submit_id, static_cast<uint32_t>(entry.args_addr),
				                    static_cast<uint32_t>(entry.args_addr >> 32u), 0, entry.mode,
				                    entry.cs_regs.cs_regs.data_addr);
				DispatchRendererLock lock(m_context);
				DispatchIndirectRecord(entry.submit_id, buffer, entry.cs_regs, entry.input_info,
				                       entry.program, entry.args_addr, entry.mode);
			} else {
				buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect),
				                    entry.submit_id, entry.groups[0], entry.groups[1], entry.groups[2],
				                    entry.mode, entry.cs_regs.cs_regs.data_addr);
				DispatchRendererLock lock(m_context);
				DispatchDirectRecord(entry.submit_id, buffer, entry.cs_regs, entry.input_info,
				                     entry.program, entry.groups[0], entry.groups[1], entry.groups[2],
				                     entry.mode);
			}
		}
		if (QueueTimingEnabled()) {
			const auto commit_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                                 std::chrono::steady_clock::now() - commit_begin)
			                                                 .count());
			QueueStatsNoteDispatch(entry.program_ns, entry.program_ns + commit_ns);
		}
	}
	queue.count = 0;
	run.packet    = saved_packet;
	run.continues = saved_continue;
	run.more      = saved_more;
	run.why       = saved_why;
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
                                      uint32_t mode) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	QueueDispatchTimer queue_timer;
	DispatchRunPost run_post(m_context, buffer);
	// The arguments are thread counts: IndirectDispatchGroups converts them on the GPU, and the
	// shader bounds its threads by the counts it reads from the same memory.
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	// The zone is the wait for the renderer mutex (another thread holding it).
	DispatchRendererLock lock(m_context);
	const auto&       cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	if (ReuseProbe::Enabled()) {
		ReuseProbe::NoteDispatch(buffer.GetShaders(), 0, 0, 0, args_addr);
	}
	if (EarlyDispatchSkip() && PipelineCache::IsSkipListed(cs_regs.cs_regs.data_addr)) {
		Profiler::Add(Profiler::Counter::SkippedSkipList);
		ResetBindings();
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions   = use_thread_dimensions;
	input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	ShaderProgram compute_program;
	{
		ScopedNs program_time {queue_timer.program_ns};
		const auto&         sh_regs = buffer.GetRegisters().GetShaderRegisters();
		ProgramLookupWindow window(m_context);
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	if (!compute_program && PipelineCache::SrtDeferPending()) {
		// Suspended until the SRT bytes it reads land (KYTY_SRT_DEFER); not skipped.
		ResetBindings();
		return;
	}
	if (!compute_program) {
		LogSkippedDispatch(cs_regs, "indirect");
		ResetBindings();
		return;
	}
	DispatchIndirectRecord(submit_id, buffer, cs_regs, input_info, compute_program, args_addr, mode);
}

void RenderExecutor::DispatchIndirectRecord(uint64_t submit_id, CommandBuffer& buffer,
                                            const HW::ComputeShaderInfo& cs_regs,
                                            ShaderComputeInputInfo&      input_info,
                                            const ShaderProgram&         compute_program,
                                            uint64_t args_addr, uint32_t mode) {
	(void)submit_id;
	if (compute_program.rt_callables != nullptr && args_addr == 0) {
		// v441b: no argument address (g441j: the host read of a null range ended the emulator).
		static std::atomic<uint32_t> logged_null {0};
		if (logged_null.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::fprintf(stderr, "[rt-callables] %016llx: indirect dispatch without an argument address, skipped\n",
			             static_cast<unsigned long long>(input_info.stage.program->shader_hash));
		}
		ResetBindings();
		return;
	}
	if (compute_program.rt_callables != nullptr) {
		// v440 M3 (KYTY_RT_CALLABLES): a callable split launches a 1D ray-tracing grid whose size and
		// workgroup counts the host computes, so its group counts are read here (the GPU work before
		// it finishes first) and it is recorded as a direct dispatch. Thread-count arguments are not
		// handled yet (skipped).
		const bool thread_counts = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		// v440p: the launch is converted on the GPU (vkCmdTraceRaysIndirectKHR) unless
		// KYTY_RT_HOST_INDIRECT=1 or the device lacks indirect ray-tracing launches.
		static const bool host_indirect = [] {
			const char* value = std::getenv("KYTY_RT_HOST_INDIRECT");
			return value != nullptr && value[0] == '1';
		}();
		if (!thread_counts && !host_indirect && m_context.GetGraphics().rt_trace_indirect_enabled) {
			buffer.EndRendering();
			const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
			    args_addr, sizeof(vk::DispatchIndirectCommand), false, false, {}, true);
			if (args_buffer == nullptr || !args_buffer->HasDeviceAddress()) {
				ResetBindings();
				return;
			}
			if (m_indirect_groups == nullptr) {
				m_indirect_groups = std::make_unique<IndirectDispatchGroups>(m_context.GetGraphics(),
				                                                             m_context.GetCommandScheduler());
			}
			const std::array<uint32_t, 3> local_size {std::max(cs_regs.cs_regs.num_thread_x, 1u),
			                                          std::max(cs_regs.cs_regs.num_thread_y, 1u),
			                                          std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			const auto launch = m_indirect_groups->ConvertRt(buffer.Recorder(),
			                                                 args_buffer->BufferDeviceAddress() + args_offset,
			                                                 local_size, input_info.stage.program->shader_hash);
			static std::atomic<uint32_t> logged_gpu {0};
			if (logged_gpu.fetch_add(1, std::memory_order_relaxed) < 4) {
				std::fprintf(stderr, "[rt-callables] %016llx: indirect dispatch, launch converted on the GPU\n",
				             static_cast<unsigned long long>(input_info.stage.program->shader_hash));
			}
			input_info.workgroup_counts[0] = 0;
			input_info.workgroup_counts[1] = 0;
			input_info.workgroup_counts[2] = 0;
			m_rt_indirect_launch = &launch;
			DispatchDirectRecord(submit_id, buffer, cs_regs, input_info, compute_program, 1, 1, 1, mode);
			m_rt_indirect_launch = nullptr;
			return;
		}
		uint32_t   groups[3]     = {0, 0, 0};
		if (!thread_counts) {
			m_context.GetBufferCache().ReadMemory(args_addr, sizeof(groups));
			std::memcpy(groups, reinterpret_cast<const void*>(args_addr), sizeof(groups));
		}
		const uint64_t total = uint64_t {groups[0]} * groups[1] * groups[2];
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::fprintf(stderr, "[rt-callables] %016llx: indirect dispatch, %s %ux%ux%u\n",
			             static_cast<unsigned long long>(input_info.stage.program->shader_hash),
			             thread_counts ? "thread counts (skipped)" : "groups read on the host", groups[0], groups[1],
			             groups[2]);
		}
		if (thread_counts || total == 0 || total > (uint64_t {1} << 22u)) {
			ResetBindings();
			return;
		}
		input_info.workgroup_counts[0] = groups[0];
		input_info.workgroup_counts[1] = groups[1];
		input_info.workgroup_counts[2] = groups[2];
		DispatchDirectRecord(submit_id, buffer, cs_regs, input_info, compute_program, groups[0], groups[1],
		                     groups[2], mode);
		return;
	}
	static const bool merged_env = [] {
		const char* value = std::getenv("KYTY_INDIRECT_MERGED_BARRIERS");
		return value != nullptr && value[0] == '1';
	}();
	// v305: also in the sync toggle's on windows with mode "mbarrier" (constant within a frame).
	const bool merged_barriers = merged_env || SyncToggle::MergedBarrierWindow();
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	{
		// Repeated dispatches of one shader skip the set (the common case in dispatch runs).
		static std::unordered_set<uint64_t> named;
		static thread_local uint64_t        last_named = ~uint64_t {0};
		if (const auto addr = static_cast<uint64_t>(cs_regs.cs_regs.data_addr);
		    addr != last_named && (last_named = addr, named.insert(addr).second)) {
			std::fprintf(stderr, "[cs-indirect] addr=0x%016" PRIx64 " hash=0x%016" PRIx64 "\n",
			             static_cast<uint64_t>(cs_regs.cs_regs.data_addr),
			             input_info.stage.program->shader_hash);
		}
	}
	// KYTY_INDIRECT_CLAMP_GROUPS (default 1): group-count arguments go through the same GPU
	// conversion, which empties dispatches no real workload needs (garbage counts launched
	// billions of groups and stopped the queue for good: the "GPU hang" on loads).
	static const bool clamp_groups = [] {
		const char* value = std::getenv("KYTY_INDIRECT_CLAMP_GROUPS");
		return value == nullptr || value[0] != '0';
	}();
	const bool convert_groups = !use_thread_dimensions && clamp_groups;
	if ((use_thread_dimensions || convert_groups) && m_indirect_groups == nullptr) {
		m_indirect_groups = std::make_unique<IndirectDispatchGroups>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler());
	}
	// KYTY_CPU_COST: the set-up of the GPU conversion of the arguments (a run's conversions
	// recorded together; the per-dispatch conversion is "Indirect::Convert" below).
	std::optional<CpuCost::Phase> conversion_phase_(std::in_place, "Indirect::ConversionSetup");
	buffer.EndRendering();
	// KYTY_DISPATCH_RUNS: the thread counts of the run's indirect dispatches are converted together
	// at its first one, between one barrier before and one after, instead of a conversion and its
	// barriers per dispatch.
	std::optional<DispatchRunConverted> run_converted;
	if (use_thread_dimensions && DispatchRunsEnabled() && DispatchSyncEnabled()) {
		auto& run = DispatchRun();
		if (const auto found = run.converted.find(run.packet);
		    run.packet != nullptr && found != run.converted.end()) {
			run_converted = found->second;
		} else if (run.upcoming.size() > 1 && run.upcoming.front().packet == run.packet) {
			run.converted.clear();
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnConvert);
			const std::array<uint32_t, 3> current {std::max(cs_regs.cs_regs.num_thread_x, 1u),
			                                       std::max(cs_regs.cs_regs.num_thread_y, 1u),
			                                       std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			vk::MemoryBarrier ready {};
			ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
			                      vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderRead;
			ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
			                      vk::AccessFlagBits::eIndirectCommandRead;
			buffer.Recorder().pipelineBarrier(
			    vk::PipelineStageFlagBits::eAllCommands,
			    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
			    {}, 1, &ready, 0, nullptr, 0, nullptr);
			for (const auto& entry: run.upcoming) {
				auto local = entry.local;
				for (uint32_t i = 0; i < 3; i++) {
					if (local[i] == 0) local[i] = current[i];
				}
				const auto [entry_buffer, entry_offset] = m_context.GetBufferCache().ObtainBuffer(
				    entry.args, sizeof(vk::DispatchIndirectCommand), false, false, {}, true);
				if (entry_buffer == nullptr || (entry_offset & 3u) != 0 ||
				    !entry_buffer->HasDeviceAddress()) {
					break;
				}
				const auto converted = m_indirect_groups->Convert(
				    buffer.Recorder(), entry_buffer->BufferDeviceAddress() + entry_offset, local,
				    false, false,
				    entry.packet == run.packet ? input_info.stage.program->shader_hash : 0u);
				run.converted[entry.packet] = {converted.groups_buffer, converted.groups_offset,
				                               converted.threads};
			}
			vk::MemoryBarrier groups_ready {};
			groups_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
			groups_ready.dstAccessMask =
			    vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
			buffer.Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
			                                  vk::PipelineStageFlagBits::eDrawIndirect |
			                                      vk::PipelineStageFlagBits::eComputeShader,
			                                  {}, 1, &groups_ready, 0, nullptr, 0, nullptr);
			run.batched += run.converted.size();
			run.upcoming.clear();
			if (const auto again = run.converted.find(run.packet); again != run.converted.end()) {
				run_converted = again->second;
			}
		}
	}
	conversion_phase_.reset();
	// v445b (KYTY_DISPATCH_DELTA), as in DispatchDirectRecord.
	std::optional<CpuCost::Phase> pipeline_phase_(std::in_place, "Dispatch::PipelineLookup");
	const DispatchDelta* delta = DispatchDeltaBase(&cs_regs, input_info, compute_program);
	using ComputePipeline = std::remove_reference_t<
	    decltype(m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program))>;
	auto& pipeline = delta != nullptr && !DispatchDeltaVerify()
	                     ? *static_cast<ComputePipeline*>(delta->pipeline)
	                     : m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	pipeline_phase_.reset();
	if (delta != nullptr && delta->pipeline != static_cast<void*>(&pipeline)) {
		std::fprintf(stderr, "[dispatch-delta-verify] %016llx: the base's pipeline is not the lookup's (indirect)\n",
		             static_cast<unsigned long long>(input_info.stage.program->shader_hash));
		delta = nullptr;
	}
	if (DispatchDeltaEnabled() && (QueueTimingEnabled() || DispatchDeltaVerify())) {
		g_delta_path.dispatches.fetch_add(1, std::memory_order_relaxed);
		g_delta_path.based.fetch_add(delta != nullptr ? 1u : 0u, std::memory_order_relaxed);
	}
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	FindBuffers(bindings);
	const auto& program = *input_info.stage.program;
	LogCsWatch(program.shader_hash, 0u, 0u, 0u, bindings, cs_regs.cs_user_sgpr.value,
	           cs_regs.cs_user_sgpr.count);
	{
		const CpuCost::Phase notes_phase_("Dispatch::Notes");
		NoteRunShader(cs_regs.cs_regs.data_addr, program.shader_hash);
		NoteDispatchBindings(program.shader_hash, *input_info.stage.resources);
		NoteTickWork(m_context.GetCommandScheduler().CurrentTick(), 'C', program.shader_hash, cs_regs.cs_regs.data_addr);
	}
	if (static const bool log_images = std::getenv("KYTY_LOG_CS_IMAGES") != nullptr; log_images) {
		static std::unordered_set<uint64_t> seen;
		static std::mutex                   seen_mutex;
		std::lock_guard                     seen_lock(seen_mutex);
		const auto& resources = *input_info.stage.resources;
		for (uint32_t i = 0; i < program.info.images.size() && i < resources.images.size(); i++) {
			static const bool written_only = std::string_view(std::getenv("KYTY_LOG_CS_IMAGES")) == "2";
			if (!program.info.images[i].written && written_only) continue;
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const uint64_t key = program.shader_hash ^ (uint64_t {r.Width5()} << 1u) ^
			                     (uint64_t {r.Height5()} << 17u) ^ (uint64_t {static_cast<uint32_t>(r.Format())} << 33u) ^ (uint64_t {i} << 45u) ^
			                     (r.Base40() * 0x9e3779b97f4a7c15ull);
			if (seen.insert(key).second) {
				std::fprintf(stderr, "[cs-image] hash=%016llx image[%u] %s addr=%llx %ux%ux%u type=%u fmt=%u indirect\n",
				             static_cast<unsigned long long>(program.shader_hash), i,
				             program.info.images[i].written ? (program.info.images[i].read ? "rw" : "w") : "r",
				             static_cast<unsigned long long>(r.Base40()),
				             static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
				             static_cast<uint32_t>(r.Depth()) + 1u, static_cast<uint32_t>(r.Type()),
				             static_cast<uint32_t>(r.Format()));
			}
		}
	}
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	// As in DispatchDirectRecord: the loop below resolves the images again (KYTY_HOTPATH).
	if (!SingleRebind() || !SyncToggle::HotPath()) RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; the loop below finalizes the shader bindings
	// afterward (KYTY_SINGLE_REBIND=0: also bind them here first, as before).
	if (!SingleRebind()) RebindBuffers(bindings);
	std::optional<CpuCost::Phase> recovery_phase_(std::in_place, "Dispatch::RecoveryBegin");
	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	recovery_phase_.reset();
	bool              delta_retried = false;
	DispatchDeltaMark delta_mark; // the scheduler's state at this dispatch's bind
	do {
		if (std::exchange(delta_retried, true)) {
			delta = nullptr; // a recovery retry: everything again, in full
		}
		RebindImages(bindings);
		// KYTY_REBIND_AFTER_JOIN: the argument lookup or a binding's can join a buffer another
		// binding resolved (the arguments are only read, so only the bindings are resolved again).
		uint64_t       join_epoch      = m_context.GetBufferCache().UnregisterEpoch();
		vk::Buffer     indirect_buffer = nullptr;
		vk::DeviceSize indirect_offset = 0;
		if (run_converted) {
			indirect_buffer  = run_converted->groups_buffer;
			indirect_offset  = run_converted->groups_offset;
			const auto dword = program.bindings.dispatch_thread_dword;
			if (dword != ShaderRecompiler::IR::PushData::NoStart) {
				bindings.shader_data[dword]      = static_cast<uint32_t>(run_converted->threads);
				bindings.shader_data[dword + 1u] = static_cast<uint32_t>(run_converted->threads >> 32u);
			}
		} else {
		const CpuCost::Phase convert_phase_("Indirect::Convert");
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    args_addr, sizeof(vk::DispatchIndirectCommand), false, false, {},
		    use_thread_dimensions || convert_groups);
		EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
		indirect_buffer = args_buffer->Handle();
		indirect_offset = args_offset;
		if (use_thread_dimensions) {
			EXIT_IF(!args_buffer->HasDeviceAddress());
			const std::array<uint32_t, 3> local_size {
			    std::max(cs_regs.cs_regs.num_thread_x, 1u),
			    std::max(cs_regs.cs_regs.num_thread_y, 1u),
			    std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			// KYTY_INDIRECT_MERGED_BARRIERS=1: one barrier orders the thread counts' writers and the
			// guest dispatch's write hazards before the conversion, and the conversion's own barrier
			// orders the groups before the indirect read, instead of four barriers around the pair
			// (the async ray-tracing chain: ~1,300 tiny indirect dispatches a frame on the title).
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnConvert);
			if (merged_barriers) {
				vk::MemoryBarrier ready {};
				ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
				                      vk::AccessFlagBits::eTransferWrite |
				                      vk::AccessFlagBits::eShaderRead;
				ready.dstAccessMask = vk::AccessFlagBits::eShaderRead |
				                      vk::AccessFlagBits::eShaderWrite |
				                      vk::AccessFlagBits::eIndirectCommandRead;
				buffer.Recorder().pipelineBarrier(
				    vk::PipelineStageFlagBits::eAllCommands,
				    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
				    {}, 1, &ready, 0, nullptr, 0, nullptr);
			}
			const auto converted = m_indirect_groups->Convert(
			    buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset, local_size,
			    !merged_barriers, true, program.shader_hash);
			indirect_buffer = converted.groups_buffer;
			indirect_offset = converted.groups_offset;
			// The shader reads the thread counts through this device address (upstream's
			// dispatch-thread words hold the address instead of the counts).
			const auto dword = program.bindings.dispatch_thread_dword;
			if (dword != ShaderRecompiler::IR::PushData::NoStart) {
				bindings.shader_data[dword]      = static_cast<uint32_t>(converted.threads);
				bindings.shader_data[dword + 1u] = static_cast<uint32_t>(converted.threads >> 32u);
			}
		} else if (convert_groups && args_buffer->HasDeviceAddress()) {
			const auto converted = m_indirect_groups->Convert(
			    buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset, {1u, 1u, 1u},
			    true, true, program.shader_hash);
			indirect_buffer = converted.groups_buffer;
			indirect_offset = converted.groups_offset;
		}
		}
		RebindBuffersDelta(bindings, delta, program.shader_hash);
		PreparedBindings* descriptor_stage = &bindings;
		(void)RebindAfterJoin(std::span {&descriptor_stage, 1u}, join_epoch, "indirect dispatch");
		ChainWatch::Note(program.shader_hash,
		                 m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
		                 cs_regs.cs_user_sgpr.value, cs_regs.cs_user_sgpr.count, bindings.buffer_sources);
		// v445b: as in DispatchDirectRecord. A conversion recorded for this dispatch bound its own
		// pipeline, so the bind point is then not the base's and everything is pushed and bound.
		const bool delta_kept = DispatchDeltaStateKept(delta, &pipeline, pipeline.pipeline);
		m_delta_push          = delta_kept;
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		m_delta_push = false;
		if (delta != nullptr && DispatchDeltaStats()) {
			g_delta_path.kept.fetch_add(delta_kept ? 1u : 0u, std::memory_order_relaxed);
			g_delta_path.writes.fetch_add(m_delta_writes, std::memory_order_relaxed);
			g_delta_path.pushed.fetch_add(m_delta_pushed, std::memory_order_relaxed);
		}
		const CpuCost::Phase record_phase_("Dispatch::Record"); // as in DispatchDirectRecord
		NoteFaultMapBindings(true);
		const auto vk_buffer = buffer.Recorder();
		std::optional<CpuCost::Phase> barrier_phase_(std::in_place, "Dispatch::BarrierBefore");
		const bool has_storage_writes =
		    bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
		    std::any_of(
		        program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			        return image.written && image.resource_class ==
			                                    ShaderRecompiler::IR::ImageResourceClass::Storage;
		        });
		const bool merged = (merged_barriers && use_thread_dimensions) || run_converted.has_value();
		if (has_storage_writes && !merged) {
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnPre);
			if (DispatchSyncEnabled() && !SyncToggle::SkipDispatchBarrier() && !RunSkipsBarrierBefore()) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		if (!merged) {
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnConvert);
			vk::MemoryBarrier barrier {};
			barrier.srcAccessMask =
			    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
			        vk::PipelineStageFlagBits::eTransfer,
			    vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
		}
		barrier_phase_.reset();
		if (auto& check = BvhCheck::Get(); check.enabled && program.shader_hash == check.hash) {
			check.OnDispatch(m_context, bindings, run_converted.has_value(),
			                 m_indirect_groups ? m_indirect_groups->RingHandle() : vk::Buffer {},
			                 IndirectDispatchGroups::RingBytes(), indirect_buffer, indirect_offset);
		}
		if (t_run_excluded && DispatchRunsEnabled() && DispatchRun().continues &&
		    DispatchSyncEnabled()) {
			// The dispatch before it left out its barrier after (KYTY_DISPATCH_RUN_EXCLUDE).
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnExcl);
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		if (!(delta_kept && DispatchDeltaStateKept(delta, &pipeline, pipeline.pipeline))) {
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		}
		delta_mark = MarkDispatchDelta(join_epoch);
		GpuTiming::Before(m_context, buffer);
		const auto gpu_zone = BeginGpuZone(m_context, buffer, program.shader_hash);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		RtSnap::BufferProbeAfterDispatch(m_context, program.shader_hash, bindings, cs_regs.cs_user_sgpr.value,
		                                 cs_regs.cs_user_sgpr.count);
		if (RtSnap::Enabled()) {
			PreparedBindings* const snap_stage = &bindings;
			RtSnap::AfterCommand(m_context, "dispatch", std::span(&snap_stage, 1));
		}
		Profiler::Add(Profiler::Counter::Dispatches);
		GpuProfiler::End(buffer, gpu_zone);
		NoteBarrierReasonDispatch(m_context);
		GpuTiming::After(m_context, buffer, program.shader_hash, GpuTiming::Dispatch);
		Timeline::Mark("dispatch-indirect", program.shader_hash, 0);
		const CpuCost::Phase barrier_after_phase_("Dispatch::BarrierAfter");
		if (DispatchSyncEnabled() && !SyncToggle::SkipDispatchBarrier() && !RunSkipsBarrierAfter()) {
			const BarrierReason::Tag reason_tag_(BarrierReason::OwnPost);
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
			if (DispatchRunsEnabled()) DispatchRun().pending_post = false;
		}
	} while ([&recovery] {
		const CpuCost::Phase retry_phase_("Dispatch::RecoveryEnd");
		return recovery.Retry();
	}());
	const CpuCost::Phase capture_phase_("Dispatch::CaptureDelta");
	CaptureDispatchDelta(&cs_regs, input_info, compute_program, &pipeline, pipeline.pipeline, bindings, delta_mark,
	                     true);
	ResetBindings();
}

} // namespace Libs::Graphics
