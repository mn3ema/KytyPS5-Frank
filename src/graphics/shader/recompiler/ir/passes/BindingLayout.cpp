#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/LoopGuard.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

[[noreturn]] void BindingFail(const char* message) {
	EXIT("shader binding layout failed: %s", message);
	std::abort();
}

void CollectShaderData(const Program& program, BindingLayout& layout) {
	std::array<bool, NumScalarRegs> registers {};
	bool uses_dispatch_threads = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (!inst.HasUses()) {
				continue;
			}
			uses_dispatch_threads |= inst.GetOpcode() == ValueOpcode::GetDispatchThreadExtent;
			if (inst.GetOpcode() != ValueOpcode::GetUserData) {
				continue;
			}
			if (inst.Arg(0).GetType() != Type::ScalarReg) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			const auto index = RegIndex(inst.Arg(0).ScalarRegister());
			if (index >= NumScalarRegs) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			registers[index] = true;
		}
	}
	for (uint32_t index = 0; index < registers.size(); index++) {
		if (registers[index]) {
			layout.user_data_registers.push_back(index);
		}
	}
	layout.memory_offset_dword = static_cast<uint32_t>(layout.user_data_registers.size());
	if (uses_dispatch_threads) {
		layout.dispatch_thread_dword = layout.memory_offset_dword;
		layout.memory_offset_dword += 3u;
	}
}

void AddBinding(BindingLayout& layout, DescriptorBindingKind kind,
                std::vector<uint32_t> resources = {}) {
	layout.descriptors.push_back({kind, std::move(resources)});
}

} // namespace

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers) {
	std::array<bool, ShaderInfo::MaxBuffers> live_buffers {};
	SharedMemoryResources shared;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (BufferAccessOf(op) == BufferAccess::None &&
			    SharedAccessOf(op) == SharedAccess::None &&
			    AddressOpcodeInfoOf(op).access == AddressAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				BindingFail("typed shader contains invalid memory metadata");
			}
			const auto& memory = program.memory_info[index];
			if (memory.planning_only) {
				continue;
			}
			shared.lds |= memory.kind == ResourceKind::FlatLocal;
			if (SharedAccessOf(op) != SharedAccess::None) {
				if (memory.kind != ResourceKind::Lds && memory.kind != ResourceKind::Gds) {
					BindingFail("typed shader contains invalid shared-memory metadata");
				}
				shared.gds |= memory.kind == ResourceKind::Gds;
				shared.lds |= memory.kind == ResourceKind::Lds;
			} else if (memory.kind == ResourceKind::Buffer || memory.kind == ResourceKind::ScalarBuffer) {
				EXIT_IF(memory.resource >= program.info.buffers.size());
				live_buffers.at(memory.resource) = true;
				for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
					live_buffers.at(child) = true;
				}
			}
		}
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		// G6 (KYTY_G6): a buffer read only through the GPU address table has no Buffer memory
		// operation, but the host needs its range to keep it present (RenderExecutor::RebindBuffers).
		if (live_buffers.at(i) || program.info.buffers[i].table_read) {
			buffers.push_back(i);
		}
	}
	return shared;
}

bool UsesFlattenedSrt(const Program& program) {
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_search_iterations != 0u;
	};
	return std::ranges::any_of(program.blocks, [](const Block* block) {
		return std::ranges::any_of(*block, [](const Inst& inst) {
			return inst.GetOpcode() == ValueOpcode::ReadConst;
		});
	}) || std::ranges::any_of(program.info.buffers, uses_mapping) ||
	       std::ranges::any_of(program.info.images, [&](const ImageResource& image) {
		       return uses_mapping(image) || image.bindless;
	       }) ||
	       std::ranges::any_of(program.info.samplers, &SamplerResource::bindless) ||
	       std::ranges::any_of(program.info.buffers, [](const BufferResource& buffer) {
		       return buffer.gpu_records_flat != BufferResource::NoGpuRecords;
	       });
}

bool ClipSpaceData() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CLIP_SPACE_DATA");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool ExportMappingData() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EXPORT_MAPPING_DATA");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool PushDataEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PUSH_DATA");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool StageDescriptorSets() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STAGE_SETS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool ShaderDataUniform() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_DATA_UBO");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool BufferStrideAsData() {
	// v425 turned it off: at 128 bytes of push constants one data dword per buffer pushed most
	// stages' shader data into a storage buffer (about 10% fewer fps). v428: on again; with 256
	// bytes (v426) it costs nothing measurable (g6u_str64b 26.4 fps against 25.8-26.8 without).
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFFER_STRIDE_DATA");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void AllocateBindings(Program& program, uint32_t push_data_start_dword, bool lds_storage,
                      bool clip_space) {
	if (!program.shader_info_complete || program.binding_layout_complete) {
		EXIT("shader binding layout failed: %s", !program.shader_info_complete
		                                             ? "shader info is not ready"
		                                             : "binding layout already allocated");
	}
	BindingLayout next;
	std::vector<uint32_t> buffers;
	const auto shared = CollectMemoryResources(program, buffers);
	CollectShaderData(program, next);
	next.memory_offset_count       = static_cast<uint32_t>(buffers.size());
	next.write_tracking            = std::ranges::any_of(buffers, [&](uint32_t resource) {
		           return resource < program.info.buffers.size() &&
		                  program.info.buffers[resource].write_tracked;
	           });
	next.srt_verify = SrtVerifyEnabled() && program.stage == ShaderType::Compute &&
	                  !program.srt_reads.empty();
	next.loop_guard = !program.dispatcher_fallback && LoopGuardApplies(program.shader_hash);
	next.clip_space = clip_space && ClipSpaceData();
	next.buffer_stride_count = BufferStrideAsData() ? static_cast<uint8_t>(std::min<size_t>(program.info.buffers.size(), 0xffu)) : uint8_t {0};
	next.export_mapping = program.stage == ShaderType::Pixel && ExportMappingData() &&
	                      std::ranges::any_of(program.export_info, [](const ExportInfo& exp) {
		                      return exp.kind == ExportTargetKind::Mrt;
	                      });
	next.image_flags = 0;
	for (const auto& image: program.info.images) {
		next.image_flags |= (image.dynamic_class ? BindingLayout::ImageClassFlags : 0u) |
		                    (image.dynamic_layer ? BindingLayout::ImageLayerFlags : 0u) |
		                    (image.dynamic_cube ? BindingLayout::ImageCubeFlags : 0u) |
		                    (image.dynamic_3d ? BindingLayout::ImageDim3DFlags : 0u);
	}
	next.push_data_start_dword =
	    PushDataEnabled() ? PushData::StartFor(push_data_start_dword, next.ShaderDataDwords())
	                      : PushData::NoStart;

	if (!buffers.empty()) {
		// Draw binding accesses this first group directly when memory_offset_count is nonzero.
		AddBinding(next, DescriptorBindingKind::Buffers, std::move(buffers));
	}

	std::array<std::vector<uint32_t>, ImageBindingCount> image_groups;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			EXIT("shader binding layout failed: image %u has an invalid binding class", i);
		}
		const auto group = ImageBindingIndex(*kind);
		if (group >= image_groups.size()) {
			EXIT("shader binding layout failed: image %u has an unmapped binding class", i);
		}
		auto&      resources = image_groups[group];
		const auto dynamic   = program.info.images[i].mip_mode == ImageMipMode::Dynamic;
		const auto count     = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			EXIT("shader binding layout failed: image %u has invalid specialized mip count %u", i,
			     program.info.images[i].mip_count);
		}
		resources.insert(resources.end(), count, i);
		if (program.info.images[i].dynamic_class) {
			// v429: also in the uint group of its dimension (the second, unsigned-integer binding).
			auto as_uint          = program.info.images[i];
			as_uint.numeric_class = Prospero::TextureNumericClass::Uint;
			const auto uint_kind  = DescriptorBindingForImage(as_uint);
			if (!uint_kind.has_value() || ImageBindingIndex(*uint_kind) >= image_groups.size()) {
				EXIT("shader binding layout failed: image %u has no uint binding", i);
			}
			image_groups[ImageBindingIndex(*uint_kind)].push_back(i);
		}
		if (program.info.images[i].dynamic_3d) {
			// v433: also in the 2D group of its class (the second binding, for a 2D T#).
			auto as_2d      = program.info.images[i];
			as_2d.dimension = Decoder::ImageDimension::Dim2D;
			const auto kind_2d = DescriptorBindingForImage(as_2d);
			if (!kind_2d.has_value() || ImageBindingIndex(*kind_2d) >= image_groups.size()) {
				EXIT("shader binding layout failed: image %u has no 2D binding", i);
			}
			image_groups[ImageBindingIndex(*kind_2d)].push_back(i);
		}
	}
	for (uint32_t i = 0; i < image_groups.size(); i++) {
		if (!image_groups[i].empty()) {
			AddBinding(next, static_cast<DescriptorBindingKind>(FirstImageBinding + i),
			           std::move(image_groups[i]));
		}
	}

	if (!program.info.samplers.empty()) {
		std::vector<uint32_t> resources(program.info.samplers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Samplers, std::move(resources));
	}
	if (shared.gds) {
		AddBinding(next, DescriptorBindingKind::Gds);
	}
	if (shared.lds && lds_storage) {
		AddBinding(next, DescriptorBindingKind::SharedMemory);
	}
	if (program.info.uses_dma) {
		AddBinding(next, DescriptorBindingKind::BdaPagetable);
		AddBinding(next, DescriptorBindingKind::FaultBuffer);
	}
	if (UsesFlattenedSrt(program) || next.srt_verify) {
		AddBinding(next, DescriptorBindingKind::FlattenedSrt);
	}

	if (next.ShaderDataDwords() != 0 && !next.UsesPushData()) {
		AddBinding(next, DescriptorBindingKind::ShaderData);
	}

	program.bindings                = std::move(next);
	program.binding_layout_complete = true;
}

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind) {
	for (const auto& binding: layout.descriptors) {
		if (binding.kind == kind) {
			return &binding;
		}
	}
	return nullptr;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
