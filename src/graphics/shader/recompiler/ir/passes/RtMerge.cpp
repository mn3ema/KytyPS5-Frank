#include "graphics/shader/recompiler/ir/passes/RtMerge.h"

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

// The instructions whose flags are MemoryFlags (an index into memory_info), as the translator emits
// them (Translator::AddMemoryInfo) and Program.cpp's validator checks them.
bool CarriesMemoryInfo(ValueOpcode op) {
	return op == ValueOpcode::ReadConstBuffer || op == ValueOpcode::GetScratchResource ||
	       BufferAccessOf(op) != BufferAccess::None || SharedAccessOf(op) != SharedAccess::None ||
	       AddressOpcodeInfoOf(op).access != AddressAccess::None ||
	       ImageOpcodeInfoOf(op).access != ImageAccess::None;
}

bool IsBufferKind(ResourceKind kind) {
	return kind == ResourceKind::Buffer || kind == ResourceKind::ScalarBuffer ||
	       kind == ResourceKind::IndirectBuffer;
}

const char* Unmergeable(const Program& callee) {
	if (!callee.resource_tracking_complete || !callee.srt_plan_complete) {
		return "its resources are not fully tracked";
	}
	if (callee.dispatcher_fallback) {
		return "it uses the dispatcher fallback";
	}
	if (!callee.srt_reads.empty()) {
		return "it reads the flattened SRT";
	}
	if (!callee.buffer_writes.empty()) {
		return "it has bounded buffer writes";
	}
	if (callee.uniform_fill.fill.kind != UniformFillKind::None) {
		return "it has a uniform fill";
	}
	if (callee.rt_role != RtRole::Callable || !callee.rt_modules.empty()) {
		return "it is not a single callable";
	}
	for (const auto& buffer: callee.info.buffers) {
		if (buffer.indirect_root != BufferResource::NoIndirectBuffer || buffer.table_read ||
		    buffer.gpu_records_flat != BufferResource::NoGpuRecords || buffer.gpu_records_ssa ||
		    buffer.indirect_search_iterations != 0u) {
			return "it reads a buffer through a GPU table or with GPU-sized records";
		}
	}
	for (const auto& image: callee.info.images) {
		if (image.indirect_search_iterations != 0u) {
			return "it enumerates an image table";
		}
	}
	for (const auto& source: callee.descriptor_sources) {
		if (source.indirect_descriptor.has_value() && !source.indirect_descriptor->bindless) {
			return "it enumerates an indirect descriptor";
		}
	}
	return nullptr;
}

bool ImmediateSource(const DescriptorSource& source, std::array<uint32_t, 8>& words) {
	if (source.indirect_descriptor.has_value() || source.bindless_sampler.has_value() || source.dword_count == 0) {
		return false;
	}
	for (uint32_t i = 0; i < source.dword_count; i++) {
		const auto value = source.dwords[i].Resolve();
		if (!value.IsImmediate() || value.GetType() != Type::U32) {
			return false;
		}
		words[i] = value.U32();
	}
	return true;
}

// An image of the program equal to the callee's image k: both looked up on the GPU by address or
// by value (nothing the host evaluates), with the same binding and use.
uint32_t SharedImage(const Program& program, const Program& callee, uint32_t k) {
	const auto& image  = callee.info.images[k];
	const auto& source = callee.descriptor_sources[image.source];
	if (!source.indirect_descriptor.has_value() || !source.indirect_descriptor->bindless ||
	    !(source.indirect_descriptor->address || source.indirect_descriptor->value) ||
	    image.indirect_root != ImageResource::NoIndirectImage || !image.indirect_resources.empty()) {
		return UINT32_MAX;
	}
	const auto& indirect = *source.indirect_descriptor;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto other = program.info.images[i];
		const auto& other_source = program.descriptor_sources[other.source];
		if (!other_source.indirect_descriptor.has_value()) continue;
		const auto& other_indirect = *other_source.indirect_descriptor;
		if (other_indirect.bindless != indirect.bindless || other_indirect.address != indirect.address ||
		    other_indirect.value != indirect.value || other_indirect.storage != indirect.storage ||
		    other_indirect.key_shift != indirect.key_shift || other_indirect.key_mask != indirect.key_mask) {
			continue;
		}
		other.source       = image.source;
		other.first_use_pc = image.first_use_pc;
		if (other == image) {
			return i;
		}
	}
	return UINT32_MAX;
}

// A sampler of the program with the same constant S# and use as the callee's sampler k.
uint32_t SharedSampler(const Program& program, const Program& callee, uint32_t k) {
	const auto& sampler = callee.info.samplers[k];
	std::array<uint32_t, 8> words {};
	if (sampler.bindless || !ImmediateSource(callee.descriptor_sources[sampler.source], words)) {
		return UINT32_MAX;
	}
	const auto count = callee.descriptor_sources[sampler.source].dword_count;
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		auto other = program.info.samplers[i];
		std::array<uint32_t, 8> other_words {};
		if (other.bindless || program.descriptor_sources[other.source].dword_count != count ||
		    !ImmediateSource(program.descriptor_sources[other.source], other_words) || other_words != words) {
			continue;
		}
		other.source         = sampler.source;
		other.first_use_pc   = sampler.first_use_pc;
		other.snapshot_index = sampler.snapshot_index;
		if (other == sampler) {
			return i;
		}
	}
	return UINT32_MAX;
}

} // namespace

bool MergeRtCallables(Program& raygen, std::vector<Program>& callables, std::string& reason) {
	if (raygen.rt_role != RtRole::Raygen || !raygen.rt_modules.empty()) {
		reason = "the main program is not a ray-generation shader";
		return false;
	}
	for (size_t i = 0; i < callables.size(); i++) {
		if (const char* why = Unmergeable(callables[i]); why != nullptr) {
			reason = fmt::format("callable {}: {}", i, why);
			return false;
		}
	}
	if (raygen.block_info.size() != raygen.blocks.size()) {
		reason = "the ray-generation program's block metadata does not match its blocks";
		return false;
	}
	for (const auto& callee: callables) {
		if (callee.block_info.size() != callee.blocks.size()) {
			reason = "a callable's block metadata does not match its blocks";
			return false;
		}
	}
	raygen.rt_modules.push_back({.role           = RtRole::Raygen,
	                             .first_block    = 0,
	                             .block_count    = static_cast<uint32_t>(raygen.blocks.size()),
	                             .scratch_dwords = raygen.scratch_dwords});
	raygen.buffer_writes_bounded.resize(raygen.info.buffers.size(), 0u);
	for (auto& callee: callables) {
		const auto buffer_base  = static_cast<uint32_t>(raygen.info.buffers.size());
		const auto source_base  = static_cast<uint32_t>(raygen.descriptor_sources.size());
		const auto memory_base  = static_cast<uint32_t>(raygen.memory_info.size());
		const auto remap_buffer = [&](uint32_t& index) { index += buffer_base; };

		// Images and samplers: one the program already has an equal of is shared (callables of
		// one shader sample through the same kind of GPU-addressed image with the same constant
		// sampler; 70 callables would otherwise pass every limit).
		std::vector<uint32_t> image_map(callee.info.images.size());
		for (uint32_t k = 0; k < callee.info.images.size(); k++) {
			auto image = callee.info.images[k];
			image.source += source_base;
			const auto shared = SharedImage(raygen, callee, k);
			if (shared != UINT32_MAX) {
				image_map[k] = shared;
				continue;
			}
			image_map[k] = static_cast<uint32_t>(raygen.info.images.size());
			raygen.info.images.push_back(std::move(image));
		}
		const auto remap_image = [&](uint32_t& index) {
			if (index < image_map.size()) index = image_map[index];
		};
		for (uint32_t k = 0; k < callee.info.images.size(); k++) {
			auto& image = raygen.info.images[image_map[k]];
			if (image.source < source_base) {
				continue; // shared with an earlier one
			}
			if (image.indirect_root != ImageResource::NoIndirectImage) {
				remap_image(image.indirect_root);
			}
			for (auto& child: image.indirect_resources) {
				remap_image(child);
			}
		}
		std::vector<uint32_t> sampler_map(callee.info.samplers.size());
		for (uint32_t k = 0; k < callee.info.samplers.size(); k++) {
			const auto shared = SharedSampler(raygen, callee, k);
			if (shared != UINT32_MAX) {
				sampler_map[k] = shared;
				continue;
			}
			auto sampler = callee.info.samplers[k];
			sampler.source += source_base;
			sampler_map[k]         = static_cast<uint32_t>(raygen.info.samplers.size());
			sampler.snapshot_index = sampler_map[k];
			raygen.info.samplers.push_back(sampler);
		}
		const auto remap_sampler = [&](uint32_t& index) {
			if (index < sampler_map.size()) index = sampler_map[index];
		};

		for (auto buffer: callee.info.buffers) {
			buffer.source += source_base;
			if (buffer.image_alias != BufferResource::NoImageAlias) {
				remap_image(buffer.image_alias);
			}
			for (auto& child: buffer.indirect_resources) {
				remap_buffer(child);
			}
			raygen.info.buffers.push_back(std::move(buffer));
		}
		raygen.buffer_writes_bounded.resize(raygen.info.buffers.size(), 0u);
		for (auto pair: callee.info.sampled_pairs) {
			remap_image(pair.image);
			remap_sampler(pair.sampler);
			if (std::ranges::none_of(raygen.info.sampled_pairs, [&](const SampledResourcePair& other) {
				    return other.image == pair.image && other.sampler == pair.sampler;
			    })) {
				raygen.info.sampled_pairs.push_back(pair);
			}
		}
		raygen.info.uses_dma = raygen.info.uses_dma || callee.info.uses_dma;

		for (auto source: callee.descriptor_sources) {
			if (source.indirect_descriptor.has_value()) {
				auto& indirect = *source.indirect_descriptor;
				if (indirect.material_source != UINT32_MAX) {
					indirect.material_source += source_base;
				}
				indirect.table_source += source_base;
				for (auto& index: indirect.sources) {
					index += source_base;
				}
			}
			raygen.descriptor_sources.push_back(std::move(source));
		}
		for (auto memory: callee.memory_info) {
			if (IsBufferKind(memory.kind)) {
				remap_buffer(memory.resource);
			} else if (memory.kind == ResourceKind::Image) {
				remap_image(memory.resource);
				remap_sampler(memory.sampler);
			} else if (memory.kind == ResourceKind::Sampler) {
				remap_sampler(memory.resource);
				remap_sampler(memory.sampler);
			}
			if (memory.table_resource != UINT32_MAX) {
				remap_buffer(memory.table_resource);
			}
			raygen.memory_info.push_back(memory);
		}

		const auto first_block = static_cast<uint32_t>(raygen.blocks.size());
		for (auto* block: callee.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::GetBufferResource) {
					inst.SetFlags(inst.Flags<uint32_t>() + buffer_base);
				} else if (op == ValueOpcode::GetImageResource) {
					auto index = inst.Flags<uint32_t>();
					remap_image(index);
					inst.SetFlags(index);
				} else if (op == ValueOpcode::GetSamplerResource) {
					auto index = inst.Flags<uint32_t>();
					remap_sampler(index);
					inst.SetFlags(index);
				} else if (CarriesMemoryInfo(op)) {
					auto flags = inst.Flags<MemoryFlags>();
					flags.index += memory_base;
					inst.SetFlags(flags);
				}
			}
			raygen.blocks.push_back(block);
		}
		for (auto& info: callee.block_info) {
			raygen.block_info.push_back(std::move(info));
		}
		callee.block_info.clear();
		for (auto& storage: callee.block_storage) {
			raygen.block_storage.push_back(std::move(storage));
		}
		callee.block_storage.clear();
		callee.blocks.clear();
		raygen.value_storage.splice(raygen.value_storage.end(), callee.value_storage);
		raygen.rt_modules.push_back({.role           = RtRole::Callable,
		                             .first_block    = first_block,
		                             .block_count    = static_cast<uint32_t>(raygen.blocks.size()) - first_block,
		                             .scratch_dwords = callee.scratch_dwords,
		                             .code_hash      = callee.rt_code_hash});

		raygen.requires_specialization_memory |= callee.requires_specialization_memory;
		raygen.capture_specialization_reads |= callee.capture_specialization_reads;
		raygen.scratch_dwords     = std::max(raygen.scratch_dwords, callee.scratch_dwords);
		raygen.has_address_writes = raygen.has_address_writes || callee.has_address_writes;
	}
	if (raygen.info.buffers.size() > ShaderInfo::MaxBuffers || raygen.info.images.size() > ShaderInfo::MaxImages ||
	    raygen.info.samplers.size() > ShaderInfo::MaxSamplers ||
	    raygen.info.sampled_pairs.size() > ShaderInfo::MaxSampledPairs) {
		reason = fmt::format("{} buffers, {} images, {} samplers, {} sampled pairs together: above the shader limits",
		                     raygen.info.buffers.size(), raygen.info.images.size(), raygen.info.samplers.size(),
		                     raygen.info.sampled_pairs.size());
		return false;
	}
	return true;
}

namespace {

bool PayloadSlot(const Inst& inst, uint32_t& slot) {
	const auto value = inst.Arg(0).Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) {
		return false;
	}
	slot = value.U32();
	return true;
}

void EraseInvalidated(Program& program, const std::unordered_set<const Inst*>& removed) {
	for (auto* block: program.blocks) {
		std::erase_if(block->Instructions(), [&](const Inst& inst) { return removed.contains(&inst); });
	}
}

} // namespace

void CutRtPayload(Program& raygen, std::vector<Program>& callables, std::string& stats) {
	std::unordered_set<uint32_t> read_slots;
	std::unordered_set<uint32_t> written_slots;
	size_t                       identity_stores = 0;
	for (auto& callee: callables) {
		std::unordered_set<const Inst*> removed;
		for (auto* block: callee.blocks) {
			for (auto& inst: block->Instructions()) {
				uint32_t slot = 0;
				if (inst.GetOpcode() != ValueOpcode::PayloadStore || !PayloadSlot(inst, slot)) {
					continue;
				}
				const auto* source = inst.Arg(1).Resolve().TryInstruction();
				uint32_t    source_slot = 0;
				if (source != nullptr && source->GetOpcode() == ValueOpcode::PayloadLoad &&
				    PayloadSlot(*source, source_slot) && source_slot == slot) {
					inst.Invalidate();
					removed.insert(&inst);
				}
			}
		}
		identity_stores += removed.size();
		EraseInvalidated(callee, removed);
		EliminateDeadCode(callee.blocks);
		for (auto* block: callee.blocks) {
			for (auto& inst: block->Instructions()) {
				uint32_t slot = 0;
				if (inst.GetOpcode() == ValueOpcode::PayloadLoad && PayloadSlot(inst, slot)) {
					read_slots.insert(slot);
				} else if (inst.GetOpcode() == ValueOpcode::PayloadStore && PayloadSlot(inst, slot)) {
					written_slots.insert(slot);
				} else if (inst.GetOpcode() == ValueOpcode::PayloadStore || inst.GetOpcode() == ValueOpcode::PayloadLoad) {
					stats = "a payload slot is not an immediate: not cut";
					return;
				}
			}
		}
	}
	// The ray generation side: the values stored before the call (per slot) stand in for the loads
	// after it of the slots no callable writes. Lanes that do not call keep what they stored, so a
	// slot a callable writes is stored too.
	std::unordered_map<uint32_t, Value> stored;
	std::unordered_set<const Inst*>     removed;
	std::unordered_set<uint32_t>        raygen_loads; // v443c: the slots the ray generation side loads back
	size_t                              kept_stores = 0;
	size_t                              kept_loads  = 0;
	size_t                              calls       = 0;
	for (auto* block: raygen.blocks) {
		stored.clear();
		bool after_call = false;
		for (auto& inst: block->Instructions()) {
			uint32_t slot = 0;
			if (inst.GetOpcode() == ValueOpcode::ExecuteCallable) {
				after_call = true;
				calls++;
				continue;
			}
			if (inst.GetOpcode() == ValueOpcode::PayloadStore && PayloadSlot(inst, slot)) {
				if (after_call) {
					stats = "a payload store follows the call in its block: not cut";
					return;
				}
				stored[slot] = inst.Arg(1);
				continue;
			}
			if (inst.GetOpcode() == ValueOpcode::PayloadLoad && PayloadSlot(inst, slot)) {
				if (!after_call || !stored.contains(slot)) {
					stats = "a payload load without its store before the call: not cut";
					return;
				}
				raygen_loads.insert(slot);
			}
		}
	}
	for (auto* block: raygen.blocks) {
		stored.clear();
		for (auto& inst: block->Instructions()) {
			uint32_t slot = 0;
			if (inst.GetOpcode() == ValueOpcode::PayloadStore && PayloadSlot(inst, slot)) {
				stored[slot] = inst.Arg(1);
				// (v443c: a slot this side never loads back, a callable's temporary it zeroes, is
				// stored only for a callable that reads it.)
				if (!read_slots.contains(slot) && (!written_slots.contains(slot) || !raygen_loads.contains(slot))) {
					inst.Invalidate();
					removed.insert(&inst);
				} else {
					kept_stores++;
				}
			} else if (inst.GetOpcode() == ValueOpcode::PayloadLoad && PayloadSlot(inst, slot)) {
				if (!written_slots.contains(slot)) {
					inst.ReplaceUsesWith(stored.at(slot));
				} else {
					kept_loads++;
				}
			}
		}
	}
	EraseInvalidated(raygen, removed);
	EliminateDeadCode(raygen.blocks);
	stats = fmt::format("payload cut: {} call(s), callables read {} / write {} slots, {} identity stores dropped, "
	                    "ray generation stores {} and loads {} slots",
	                    calls, read_slots.size(), written_slots.size(), identity_stores, kept_stores, kept_loads);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
