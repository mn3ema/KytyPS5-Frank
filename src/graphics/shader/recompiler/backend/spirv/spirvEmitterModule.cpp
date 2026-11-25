#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

// KYTY_NO_FLOAT_CONTROLS="all" or "hash,hash,...": leave out SignedZeroInfNanPreserve for these
// shaders. A diagnostic only: on the RTX 5090 (driver 616.92) NVIDIA's code for c7c9148c7b0fa86c is
// the same with or without the mode (255 registers, 107,264 bytes).
bool FloatControlsDisabled(uint64_t shader_hash) {
	static const auto setting = [] {
		struct Setting {
			bool                  all = false;
			std::vector<uint64_t> hashes;
		} result;
		const char* value = std::getenv("KYTY_NO_FLOAT_CONTROLS");
		if (value == nullptr) {
			return result;
		}
		const std::string_view list(value);
		if (list == "all") {
			result.all = true;
			return result;
		}
		size_t start = 0;
		while (start < list.size()) {
			const auto end = std::min(list.find(',', start), list.size());
			if (end > start) {
				result.hashes.push_back(
				    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
			}
			start = end + 1;
		}
		return result;
	}();
	return setting.all || std::ranges::find(setting.hashes, shader_hash) != setting.hashes.end();
}

uint32_t TypeVoid(EmitterState& state) {
	if (state.void_type == 0) {
		state.void_type = state.builder.Type(spv::OpTypeVoid);
	}
	return state.void_type;
}

uint32_t TypeBool(EmitterState& state) {
	if (state.bool_type == 0) {
		state.bool_type = state.builder.Type(spv::OpTypeBool);
	}
	return state.bool_type;
}

uint32_t TypeBoolVector(EmitterState& state, uint32_t components) {
	auto& type = state.bool_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeBool(state), components);
	}
	return type;
}

uint32_t TypeU32(EmitterState& state) {
	if (state.u32_type == 0) {
		state.u32_type = state.builder.Type(spv::OpTypeInt, 32, 0);
	}
	return state.u32_type;
}

uint32_t TypeU64(EmitterState& state) {
	return TypeU32Vector(state, 2);
}

uint32_t TypeScalarU64(EmitterState& state) {
	if (state.native_u64_type == 0) {
		state.native_u64_type = state.builder.Type(spv::OpTypeInt, 64, 0);
	}
	return state.native_u64_type;
}

uint32_t TypeU32Pair(EmitterState& state) {
	if (state.u32_pair_type == 0) {
		const auto element = TypeU32(state);
		state.u32_pair_type = state.builder.Type(spv::OpTypeStruct, element, element);
	}
	return state.u32_pair_type;
}

uint32_t TypeI32(EmitterState& state) {
	if (state.i32_type == 0) {
		state.i32_type = state.builder.Type(spv::OpTypeInt, 32, 1);
	}
	return state.i32_type;
}

uint32_t TypeI32Pair(EmitterState& state) {
	if (state.i32_pair_type == 0) {
		const auto element = TypeI32(state);
		state.i32_pair_type = state.builder.Type(spv::OpTypeStruct, element, element);
	}
	return state.i32_pair_type;
}

uint32_t TypeF32(EmitterState& state) {
	if (state.f32_type == 0) {
		state.f32_type = state.builder.Type(spv::OpTypeFloat, 32);
	}
	return state.f32_type;
}

uint32_t TypeF64(EmitterState& state) {
	if (state.f64_type == 0) {
		state.f64_type = state.builder.Type(spv::OpTypeFloat, 64);
	}
	return state.f64_type;
}

uint32_t TypeU32Vector(EmitterState& state, uint32_t components) {
	auto& type = state.u32_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeU32(state), components);
	}
	return type;
}

uint32_t TypeU32Composite(EmitterState& state, uint32_t components) {
	EXIT_IF(components < 1u || components > 4u);
	if (components == 1u) return TypeU32(state);
	return components == 2u ? TypeU32Pair(state) : TypeU32Vector(state, components);
}

uint32_t TypeI32Vector(EmitterState& state, uint32_t components) {
	auto& type = state.i32_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeI32(state), components);
	}
	return type;
}

uint32_t TypeF32Vector(EmitterState& state, uint32_t components) {
	auto& type = state.f32_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeF32(state), components);
	}
	return type;
}

uint32_t TypePointer(EmitterState& state, spv::StorageClass storage_class, uint32_t pointee) {
	return state.builder.Type(spv::OpTypePointer, storage_class, pointee);
}

uint32_t TypeFunction(EmitterState& state) {
	if (state.function_type == 0) {
		state.function_type = state.builder.Type(spv::OpTypeFunction, TypeVoid(state));
	}
	return state.function_type;
}

uint32_t TypeStorageBufferElement(EmitterState& state, uint32_t bits) {
	return bits == 32u ? TypeU32(state) : bits == 64u ? TypeScalarU64(state)
	                                                   : state.builder.Type(spv::OpTypeInt, bits, 0);
}

uint32_t StorageRuntimeArrayType(EmitterState& state, uint32_t bits) {
	return state.builder.DecoratedType(
	    spv::OpTypeRuntimeArray,
	    {{spv::OpDecorate, {spv::DecorationArrayStride, bits / 8u}}},
	    TypeStorageBufferElement(state, bits));
}

uint32_t StorageBufferType(EmitterState& state, uint32_t bits = 32) {
	return state.builder.DecoratedType(spv::OpTypeStruct,
	                                   {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                    {spv::OpDecorate, {spv::DecorationBlock}}},
	                                   StorageRuntimeArrayType(state, bits));
}

uint32_t TypeStorageBufferPointer(EmitterState& state, uint32_t bits) {
	return TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state, bits));
}

uint32_t TypeStorageBufferElementPointer(EmitterState& state, uint32_t bits) {
	return TypePointer(state, spv::StorageClassStorageBuffer, TypeStorageBufferElement(state, bits));
}

bool DispatchDimensionsIndirect(const EmitterState& state) {
	// v440p: a callable split always reads its thread extents through the address in the shader
	// data (the host writes them, or the GPU converts the guest's indirect group counts).
	return state.program.stage == ShaderType::Compute && state.input_info.compute != nullptr &&
	       (state.input_info.compute->dispatch_dimensions_indirect || state.program.rt_role != IR::RtRole::None) &&
	       state.program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
}

bool UsesPhysicalAddresses(const EmitterState& state) {
	return state.program.info.uses_dma || DispatchDimensionsIndirect(state) ||
	       state.program.bindings.write_tracking || state.program.bindings.srt_verify ||
	       state.program.bindings.loop_guard || state.program.rt_role != IR::RtRole::None;
}

uint32_t TypePhysicalU32Pointer(EmitterState& state) {
	return TypePointer(state, spv::StorageClassPhysicalStorageBuffer, TypeU32(state));
}

uint32_t TypePushConstantElementPointer(EmitterState& state) {
	return TypePointer(state, spv::StorageClassPushConstant, TypeU32(state));
}

uint32_t TypeU32ArrayPointer(EmitterState& state, spv::StorageClass storage_class,
                             uint32_t dwords) {
	const auto count = ConstantU32(state, std::max(dwords, 1u));
	const auto array = state.builder.Type(spv::OpTypeArray, TypeU32(state), count);
	return TypePointer(state, storage_class, array);
}

uint32_t TypeU32ElementPointer(EmitterState& state, spv::StorageClass storage_class) {
	return TypePointer(state, storage_class, TypeU32(state));
}

namespace {

uint32_t PushConstantArrayType(EmitterState& state) {
	const auto count = ConstantU32(state, IR::PushData::DwordCount);
	return state.builder.DecoratedType(
	    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, sizeof(uint32_t)}}},
	    TypeU32(state), count);
}

uint32_t PushConstantBlockType(EmitterState& state) {
	return state.builder.DecoratedType(spv::OpTypeStruct,
	                                   {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                    {spv::OpDecorate, {spv::DecorationBlock}}},
	                                   PushConstantArrayType(state));
}

// The position output is Invariant (11-12): the guest GPU runs one shader binary per stage, so a
// depth pre-pass and a later pass with an EQUAL/GEQUAL depth test over the same geometry compute
// bit-identical positions. Host pipelines compile the same SPIR-V separately (different fragment
// stages, varyings removed in a depth-only pass), and without Invariant the driver may contract or
// schedule the position math differently per pipeline. KYTY_INVARIANT_POSITION=0 leaves it out.
static bool InvariantPosition() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_INVARIANT_POSITION");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint32_t PerVertexType(EmitterState& state) {
	if (InvariantPosition()) {
		return state.builder.DecoratedType(
		    spv::OpTypeStruct,
		    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInPosition}},
		     {spv::OpMemberDecorate, {0, spv::DecorationInvariant}},
		     {spv::OpDecorate, {spv::DecorationBlock}}},
		    TypeF32Vector(state, 4));
	}
	return state.builder.DecoratedType(
	    spv::OpTypeStruct,
	    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInPosition}},
	     {spv::OpDecorate, {spv::DecorationBlock}}},
	    TypeF32Vector(state, 4));
}

uint32_t SampleMaskArrayType(EmitterState& state) {
	return state.builder.Type(spv::OpTypeArray, TypeI32(state), ConstantU32(state, 1));
}

uint32_t F32ArrayType(EmitterState& state, uint32_t count) {
	return state.builder.Type(spv::OpTypeArray, TypeF32(state), ConstantU32(state, count));
}

// The set-1 runtime array (BindlessBindings.h) of an image type, aliasing the binding of its
// dimension.
uint32_t DefineBindlessImageArray(EmitterState& state, const IR::ImageResource& image, uint32_t type) {
	uint32_t binding = IR::BindlessImages2D;
	if (image.resource_class == IR::ImageResourceClass::Storage) {
		binding = IR::BindlessStorage2D;
	} else if (image.cube) {
		binding = IR::BindlessImagesCube;
	} else if (image.dimension == ImageDimension::Dim3D) {
		binding = IR::BindlessImages3D;
	} else if (image.dimension == ImageDimension::Dim2DArray ||
	           image.dimension == ImageDimension::Dim2DMsaaArray) {
		binding = IR::BindlessImages2DArray;
	}
	const auto array    = state.builder.Type(spv::OpTypeRuntimeArray, type);
	const auto variable = state.builder.DefineGlobalVariable(
	    TypePointer(state, spv::StorageClassUniformConstant, array),
	    spv::StorageClassUniformConstant);
	state.builder.AddName(variable, "bindless_images");
	if (image.resource_class == IR::ImageResourceClass::Storage &&
	    UsesPixelInterlock(state.program)) {
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationCoherent);
	}
	state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet,
	                            IR::BindlessDescriptorSet);
	state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
	state.bindless_image_variables[type] = variable;
	state.builder.RequireCapability(spv::CapabilityRuntimeDescriptorArray);
	state.builder.RequireCapability(image.resource_class == IR::ImageResourceClass::Storage
	                                    ? spv::CapabilityStorageImageArrayNonUniformIndexing
	                                    : spv::CapabilitySampledImageArrayNonUniformIndexing);
	state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
	state.builder.RequireVersion(0x00010500u);
	return variable;
}

// The set-1 sampler array, the key -> slot translation buffer and the feedback buffer. `id` is 0
// (a new id) or the id reserved for the variable (a callable split's module).
uint32_t DefineBindlessSamplers(EmitterState& state, uint32_t id) {
	const auto array   = state.builder.Type(spv::OpTypeRuntimeArray,
	                                        state.builder.Type(spv::OpTypeSampler));
	const auto pointer = TypePointer(state, spv::StorageClassUniformConstant, array);
	if (id == 0) {
		id = state.builder.DefineGlobalVariable(pointer, spv::StorageClassUniformConstant);
	} else {
		state.builder.DefineGlobalVariable(id, pointer, spv::StorageClassUniformConstant);
	}
	state.builder.AddName(id, "bindless_samplers");
	state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationDescriptorSet,
	                            IR::BindlessDescriptorSet);
	state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationBinding, IR::BindlessSamplers);
	state.builder.RequireCapability(spv::CapabilityRuntimeDescriptorArray);
	state.builder.RequireCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
	state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
	state.builder.RequireVersion(0x00010500u);
	return id;
}

uint32_t DefineBindlessBuffer(EmitterState& state, uint32_t id, const char* name, uint32_t binding,
                              bool read_only) {
	const auto pointer = TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state));
	if (id == 0) {
		id = state.builder.DefineGlobalVariable(pointer, spv::StorageClassStorageBuffer);
	} else {
		state.builder.DefineGlobalVariable(id, pointer, spv::StorageClassStorageBuffer);
	}
	state.builder.AddName(id, name);
	state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationDescriptorSet,
	                            IR::BindlessDescriptorSet);
	state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationBinding, binding);
	if (read_only) {
		state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationNonWritable);
	}
	return id;
}

// Set 1 (BindlessBindings.h): one runtime array per image type, aliasing the binding of its
// dimension, and the translation buffer. Declared before the entry point collects globals.
void DefineBindlessImages(EmitterState& state) {
	std::vector<IR::ImageResource> images;
	for (const auto& image: state.program.info.images) {
		if (!image.bindless) {
			continue;
		}
		images.push_back(image);
		// A 3D image instruction may get a 2D or 2D-array T# (BindlessSlotArrayFlag): it reads
		// the 2D-array array too.
		if (BindlessVolumeMayBeArray(image)) {
			auto as_array      = image;
			as_array.dimension = ImageDimension::Dim2DArray;
			images.push_back(as_array);
		}
	}
	for (const auto& image: images) {
		const auto type = ImageType(state, image);
		if (state.bindless_image_variables.contains(type)) {
			continue;
		}
		DefineBindlessImageArray(state, image, type);
	}
	if (state.bindless_sampler_variable == 0 &&
	    std::ranges::any_of(state.program.info.samplers, &IR::SamplerResource::bindless)) {
		state.bindless_sampler_variable = DefineBindlessSamplers(state, 0);
	}
	if ((!state.bindless_image_variables.empty() || ProbeAppendWanted(state)) &&
	    state.bindless_translation_variable == 0) {
		state.bindless_translation_variable =
		    DefineBindlessBuffer(state, 0, "bindless_translation", IR::BindlessTranslation, true);
		state.bindless_feedback_variable =
		    DefineBindlessBuffer(state, 0, "bindless_feedback", IR::BindlessFeedback, false);
	}
}

// The three host-owned buffers every shader only reads (all their accesses are loads; none is guest
// memory, so no guest device address reaches them, and the GPU fills the flattened SRT's slots in
// an earlier command): NonWritable. KYTY_HOST_BUFFERS_NONWRITABLE=0 leaves them undecorated.
bool HostBuffersNonWritable() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOST_BUFFERS_NONWRITABLE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// v442, a module of a callable split with the fixed layout (BindingLayout::rt_fixed,
// IR/RtCallable.h): an id for every descriptor variable the module may name, in an order that
// depends on nothing but the module (its requirements, and whether its own code uses LDS or GDS).
// DefineUsedRtDescriptors declares the ones the emitted code read.
void ReserveRtDescriptors(EmitterState& state) {
	auto& builder = state.builder;
	state.storage_buffer_variable = builder.AllocateId();
	if (state.requirements.buffer_u8) {
		state.storage_buffer_u8_variable = builder.AllocateId();
	}
	if (state.requirements.buffer_u16) {
		state.storage_buffer_u16_variable = builder.AllocateId();
	}
	if (state.requirements.buffer_int64_atomics) {
		state.storage_buffer_u64_variable = builder.AllocateId();
	}
	for (auto& variable: state.image_variables) {
		variable = builder.AllocateId();
	}
	state.sampler_variable = builder.AllocateId();
	if (state.rt_shared.gds) {
		state.gds_variable = builder.AllocateId();
	}
	if (state.rt_shared.lds && state.lds_storage_class == spv::StorageClassStorageBuffer) {
		state.lds_variable = builder.AllocateId();
		if (state.requirements.shared_int64_atomics) {
			state.lds_u64_variable = builder.AllocateId();
		}
	}
	state.bda_pagetable_variable = builder.AllocateId();
	state.fault_buffer_variable  = builder.AllocateId();
	state.flattened_srt_variable = builder.AllocateId();
	if (IR::ShaderDataUniform()) {
		state.shader_data_uniform_variable = builder.AllocateId();
	} else {
		state.shader_data_storage_variable = builder.AllocateId();
	}
	// Set 1, when the session samples bindless images at all (not: when some module does).
	if (state.program.bindless_images || ProbeAppendWanted(state)) {
		state.bindless_translation_variable = builder.AllocateId();
		state.bindless_feedback_variable    = builder.AllocateId();
		state.bindless_sampler_variable     = builder.AllocateId();
	}
}

void DefineDescriptors(EmitterState& state) {
	if (state.program.bindings.rt_fixed) {
		ReserveRtDescriptors(state);
		return;
	}
	DefineBindlessImages(state);
	if (state.program.bindings.UsesPushData() || state.program.stage == ShaderType::Mesh) {
		const auto type              = PushConstantBlockType(state);
		state.push_constant_variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassPushConstant, type), spv::StorageClassPushConstant);
		state.builder.AddName(type, "BufferResource");
		state.builder.AddName(state.push_constant_variable, "vsharp");
	}
	for (const auto& binding: state.program.bindings.descriptors) {
		const auto Define = [&](uint32_t type, const char* name,
		                        spv::StorageClass storage = spv::StorageClassStorageBuffer) {
			const auto variable =
			    state.builder.DefineGlobalVariable(TypePointer(state, storage, type), storage);
			state.builder.AddName(variable, name);
			state.builder.AddAnnotation(
			    spv::OpDecorate, variable, spv::DecorationDescriptorSet,
			    IR::StageDescriptorSets() && state.program.stage == ShaderType::Pixel ? IR::PixelDescriptorSet : 0u);
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding,
			                            IR::NativeBinding(state.program.stage, binding.kind));
			return variable;
		};
		const auto DecorateHostReadOnly = [&](uint32_t variable) {
			if (HostBuffersNonWritable()) {
				state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationNonWritable);
			}
		};
		const auto ArrayType = [&](uint32_t type) {
			return state.builder.Type(
			    spv::OpTypeArray, type,
			    ConstantU32(state, static_cast<uint32_t>(binding.resources.size())));
		};
		switch (binding.kind) {
			case IR::DescriptorBindingKind::Buffers:
				state.storage_buffer_variable =
				    Define(ArrayType(StorageBufferType(state)), "buffers");
				if (state.requirements.buffer_u8) {
					state.storage_buffer_u8_variable =
					    Define(ArrayType(StorageBufferType(state, 8)), "buffers_u8");
				}
				if (state.requirements.buffer_u16) {
					state.storage_buffer_u16_variable =
					    Define(ArrayType(StorageBufferType(state, 16)), "buffers_u16");
				}
				if (state.requirements.buffer_int64_atomics) {
					state.storage_buffer_u64_variable =
					    Define(ArrayType(StorageBufferType(state, 64)), "buffers_u64");
				}
				for (const auto variable: {state.storage_buffer_variable.Id(), state.storage_buffer_u8_variable.Id(),
				                           state.storage_buffer_u16_variable.Id(), state.storage_buffer_u64_variable.Id()}) {
					if (variable == 0) continue;
					if (state.storage_buffer_u8_variable != 0 || state.storage_buffer_u16_variable != 0 ||
					    state.storage_buffer_u64_variable != 0) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationAliased);
					}
					// RDNA2 stores publish to L2 even without GLC; every alias of the buffer
					// must participate in visibility for cache-bypassing polling loads.
					if (state.requirements.coherent_buffers || UsesPixelInterlock(state.program)) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationCoherent);
					}
				}
				break;
			case IR::DescriptorBindingKind::BdaPagetable:
				state.bda_pagetable_variable = Define(StorageBufferType(state, 64), "bda_pagetable");
				DecorateHostReadOnly(state.bda_pagetable_variable);
				break;
			case IR::DescriptorBindingKind::FaultBuffer:
				state.fault_buffer_variable = Define(StorageBufferType(state), "fault_buffer");
				break;
			case IR::DescriptorBindingKind::ShaderData:
				if (IR::ShaderDataUniform()) {
					// uvec4 elements (std140 arrays are 16-byte strided), one dword per component.
					const auto count = std::max(1u, (state.program.bindings.ShaderDataDwords() + 3u) / 4u);
					const auto array = state.builder.DecoratedType(
					    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 16u}}},
					    TypeU32Vector(state, 4), ConstantU32(state, count));
					const auto block = state.builder.DecoratedType(
					    spv::OpTypeStruct,
					    {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}}, {spv::OpDecorate, {spv::DecorationBlock}}},
					    array);
					state.shader_data_uniform_variable = Define(block, "shader_data", spv::StorageClassUniform);
					break;
				}
				state.shader_data_storage_variable =
				    Define(StorageBufferType(state), "shader_data");
				DecorateHostReadOnly(state.shader_data_storage_variable);
				break;
			case IR::DescriptorBindingKind::FlattenedSrt:
				state.flattened_srt_variable = Define(StorageBufferType(state), "flattened_srt");
				DecorateHostReadOnly(state.flattened_srt_variable);
				break;
			case IR::DescriptorBindingKind::Samplers:
				state.sampler_variable = Define(ArrayType(state.builder.Type(spv::OpTypeSampler)),
				                                "samplers", spv::StorageClassUniformConstant);
				break;
			case IR::DescriptorBindingKind::Gds:
				state.gds_variable = Define(StorageBufferType(state), "gds");
				break;
			case IR::DescriptorBindingKind::SharedMemory:
				state.lds_variable = Define(StorageBufferType(state), "lds_dwords");
				state.builder.AddAnnotation(spv::OpDecorate, state.lds_variable, spv::DecorationCoherent);
				if (state.requirements.shared_int64_atomics) {
					state.lds_u64_variable = Define(StorageBufferType(state, 64), "lds_qwords");
					state.builder.AddAnnotation(spv::OpDecorate, state.lds_u64_variable, spv::DecorationCoherent);
					for (const auto variable: {state.lds_variable.Id(), state.lds_u64_variable.Id()}) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationAliased);
					}
				}
				break;
			default: {
				EXIT_IF(IR::ImageBindingResourceClass(binding.kind) ==
				        IR::ImageResourceClass::None);
				auto image = state.program.info.images.at(binding.resources.front());
				// v429: a dynamic-class image's second binding is its uint group.
				if (image.dynamic_class && IR::DescriptorBindingForImage(image) != binding.kind) {
					image.numeric_class = Prospero::TextureNumericClass::Uint;
				}
				// v433: a dynamic-3D image's second binding is the 2D group of its class.
				if (image.dynamic_3d && IR::DescriptorBindingForImage(image) != binding.kind) {
					image.dimension = ImageDimension::Dim2D;
				}
				const auto  name  = "image_" + std::to_string(static_cast<uint32_t>(binding.kind));
				state.image_variables[IR::ImageBindingIndex(binding.kind)] =
				    Define(ArrayType(ImageType(state, image)), name.c_str(),
				           spv::StorageClassUniformConstant);
				// POPS: a pixel shader reads and writes its targets as storage images inside the
				// ordered interlock (decals blended into a packed G-buffer). Without Coherent
				// the read may come from a cache line loaded before the previous overlapping
				// fragment's write reached memory, and that write is lost: the console's
				// ordering only holds when the critical section's accesses are coherent.
				if (image.resource_class == IR::ImageResourceClass::Storage &&
				    UsesPixelInterlock(state.program)) {
					state.builder.AddAnnotation(spv::OpDecorate,
					                            state.image_variables[IR::ImageBindingIndex(binding.kind)],
					                            spv::DecorationCoherent);
				}
				if (image.dimension == ImageDimension::Dim1D ||
				    image.dimension == ImageDimension::Dim1DArray) {
					state.builder.RequireCapability(image.resource_class ==
					                                        IR::ImageResourceClass::Sampled
					                                    ? spv::CapabilitySampled1D
					                                    : spv::CapabilityImage1D);
				}
				break;
			}
		}
	}
}

} // namespace

uint32_t BindlessImageVariable(EmitterState& state, const IR::ImageResource& image) {
	const auto type = ImageType(state, image);
	if (const auto found = state.bindless_image_variables.find(type);
	    found != state.bindless_image_variables.end()) {
		return found->second;
	}
	// Everywhere but in a callable split's module the arrays are declared with the module.
	EXIT_IF(!state.program.bindings.rt_fixed);
	return DefineBindlessImageArray(state, image, type);
}

void DefineUsedRtDescriptors(EmitterState& state) {
	if (!state.program.bindings.rt_fixed) {
		return;
	}
	using Kind = IR::DescriptorBindingKind;
	const auto& bindings = state.program.bindings;
	const auto  Define   = [&](const DescriptorVariable& variable, Kind kind, uint32_t type, const char* name,
	                          spv::StorageClass storage = spv::StorageClassStorageBuffer) {
		const auto id = variable.Id();
		state.builder.DefineGlobalVariable(id, TypePointer(state, storage, type), storage);
		state.builder.AddName(id, name);
		state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationDescriptorSet, 0u);
		state.builder.AddAnnotation(spv::OpDecorate, id, spv::DecorationBinding,
		                            IR::NativeBinding(state.program.stage, kind));
		return id;
	};
	// The array lengths are the fixed set layout's (RtFixedDescriptorCount), not the program's counts.
	const auto ArrayType = [&](Kind kind, uint32_t type) {
		const auto* binding = IR::FindBinding(bindings, kind);
		const auto  count   = binding != nullptr ? static_cast<uint32_t>(binding->resources.size()) : 0u;
		return state.builder.Type(spv::OpTypeArray, type,
		                          ConstantU32(state, IR::RtFixedDescriptorCount(kind, count)));
	};
	const auto Decorate = [&](uint32_t id, spv::Decoration decoration) {
		state.builder.AddAnnotation(spv::OpDecorate, id, decoration);
	};
	const auto DecorateHostReadOnly = [&](uint32_t id) {
		if (HostBuffersNonWritable()) {
			Decorate(id, spv::DecorationNonWritable);
		}
	};

	struct BufferArray {
		const DescriptorVariable* variable;
		uint32_t                  bits;
		const char*               name;
	};
	const BufferArray buffer_arrays[] = {{&state.storage_buffer_variable, 32u, "buffers"},
	                                     {&state.storage_buffer_u8_variable, 8u, "buffers_u8"},
	                                     {&state.storage_buffer_u16_variable, 16u, "buffers_u16"},
	                                     {&state.storage_buffer_u64_variable, 64u, "buffers_u64"}};
	const auto used_buffer_arrays = std::ranges::count_if(
	    buffer_arrays, [](const BufferArray& array) { return array.variable->Used(); });
	for (const auto& array: buffer_arrays) {
		if (!array.variable->Used()) {
			continue;
		}
		const auto id = Define(*array.variable, Kind::Buffers,
		                       ArrayType(Kind::Buffers, StorageBufferType(state, array.bits)), array.name);
		if (used_buffer_arrays > 1) {
			Decorate(id, spv::DecorationAliased);
		}
		if (state.requirements.coherent_buffers) {
			Decorate(id, spv::DecorationCoherent);
		}
	}
	for (uint32_t index = 0; index < state.image_variables.size(); index++) {
		if (!state.image_variables[index].Used()) {
			continue;
		}
		const auto  kind    = static_cast<Kind>(IR::FirstImageBinding + index);
		const auto* binding = IR::FindBinding(bindings, kind);
		EXIT_IF(binding == nullptr || binding->resources.empty());
		auto image = state.program.info.images.at(binding->resources.front());
		// As DefineDescriptors: the second binding of a dynamic-class or dynamic-3D image.
		if (image.dynamic_class && IR::DescriptorBindingForImage(image) != kind) {
			image.numeric_class = Prospero::TextureNumericClass::Uint;
		}
		if (image.dynamic_3d && IR::DescriptorBindingForImage(image) != kind) {
			image.dimension = ImageDimension::Dim2D;
		}
		const auto name = "image_" + std::to_string(static_cast<uint32_t>(kind));
		Define(state.image_variables[index], kind, ArrayType(kind, ImageType(state, image)), name.c_str(),
		       spv::StorageClassUniformConstant);
		if (image.dimension == ImageDimension::Dim1D || image.dimension == ImageDimension::Dim1DArray) {
			state.builder.RequireCapability(image.resource_class == IR::ImageResourceClass::Sampled
			                                    ? spv::CapabilitySampled1D
			                                    : spv::CapabilityImage1D);
		}
	}
	if (state.sampler_variable.Used()) {
		Define(state.sampler_variable, Kind::Samplers,
		       ArrayType(Kind::Samplers, state.builder.Type(spv::OpTypeSampler)), "samplers",
		       spv::StorageClassUniformConstant);
	}
	if (state.gds_variable.Used()) {
		Define(state.gds_variable, Kind::Gds, StorageBufferType(state), "gds");
	}
	if (state.lds_variable.Used() || state.lds_u64_variable.Used()) {
		const bool both = state.lds_variable.Used() && state.lds_u64_variable.Used();
		if (state.lds_variable.Used()) {
			const auto id = Define(state.lds_variable, Kind::SharedMemory, StorageBufferType(state), "lds_dwords");
			Decorate(id, spv::DecorationCoherent);
			if (both) Decorate(id, spv::DecorationAliased);
		}
		if (state.lds_u64_variable.Used()) {
			const auto id =
			    Define(state.lds_u64_variable, Kind::SharedMemory, StorageBufferType(state, 64), "lds_qwords");
			Decorate(id, spv::DecorationCoherent);
			if (both) Decorate(id, spv::DecorationAliased);
		}
	}
	if (state.bda_pagetable_variable.Used()) {
		DecorateHostReadOnly(
		    Define(state.bda_pagetable_variable, Kind::BdaPagetable, StorageBufferType(state, 64), "bda_pagetable"));
	}
	if (state.fault_buffer_variable.Used()) {
		Define(state.fault_buffer_variable, Kind::FaultBuffer, StorageBufferType(state), "fault_buffer");
	}
	if (state.flattened_srt_variable.Used()) {
		DecorateHostReadOnly(
		    Define(state.flattened_srt_variable, Kind::FlattenedSrt, StorageBufferType(state), "flattened_srt"));
	}
	if (state.shader_data_uniform_variable.Used()) {
		// uvec4 elements (std140 arrays are 16-byte strided), one dword per component.
		const auto count = std::max(1u, (bindings.ShaderDataDwords() + 3u) / 4u);
		const auto array = state.builder.DecoratedType(
		    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 16u}}},
		    TypeU32Vector(state, 4), ConstantU32(state, count));
		const auto block = state.builder.DecoratedType(
		    spv::OpTypeStruct,
		    {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}}, {spv::OpDecorate, {spv::DecorationBlock}}},
		    array);
		Define(state.shader_data_uniform_variable, Kind::ShaderData, block, "shader_data", spv::StorageClassUniform);
	}
	if (state.shader_data_storage_variable.Used()) {
		DecorateHostReadOnly(
		    Define(state.shader_data_storage_variable, Kind::ShaderData, StorageBufferType(state), "shader_data"));
	}
	if (state.bindless_sampler_variable.Used()) {
		DefineBindlessSamplers(state, state.bindless_sampler_variable.Id());
	}
	if (state.bindless_translation_variable.Used()) {
		DefineBindlessBuffer(state, state.bindless_translation_variable.Id(), "bindless_translation",
		                     IR::BindlessTranslation, true);
	}
	if (state.bindless_feedback_variable.Used()) {
		DefineBindlessBuffer(state, state.bindless_feedback_variable.Id(), "bindless_feedback",
		                     IR::BindlessFeedback, false);
	}
}

uint32_t ConstantU32(EmitterState& state, uint32_t value) {
	return state.builder.Constant(spv::OpConstant, TypeU32(state), value);
}

uint32_t ConstantI32(EmitterState& state, int32_t value) {
	return state.builder.Constant(spv::OpConstant, TypeI32(state), static_cast<uint32_t>(value));
}

uint32_t ConstantF32(EmitterState& state, uint32_t bits) {
	return state.builder.Constant(spv::OpConstant, TypeF32(state), bits);
}

uint32_t ConstantF32Value(EmitterState& state, float value) {
	return ConstantF32(state, std::bit_cast<uint32_t>(value));
}

uint32_t ConstantBool(EmitterState& state, bool value) {
	return state.builder.Constant(value ? spv::OpConstantTrue : spv::OpConstantFalse,
	                              TypeBool(state));
}

uint32_t ConstantU64(EmitterState& state, uint64_t value) {
	return state.builder.Constant(spv::OpConstantComposite, TypeU64(state),
	                              ConstantU32(state, static_cast<uint32_t>(value)),
	                              ConstantU32(state, static_cast<uint32_t>(value >> 32u)));
}

uint32_t ConstantU32CompositeZero(EmitterState& state, uint32_t components) {
	EXIT_IF(components < 1u || components > 4u);
	const auto            zero = ConstantU32(state, 0);
	if (components == 1u) return zero;
	std::vector<uint32_t> values(components, zero);
	return state.builder.Constant(spv::OpConstantComposite, TypeU32Composite(state, components),
	                              values);
}

uint32_t GlslStd450(EmitterState& state) {
	return state.builder.Import("GLSL.std.450");
}

VertexInputScalarKind VertexParameterScalarKind(const EmitterState& state, uint32_t location) {
	if ((state.program.stage != ShaderType::Vertex && state.program.stage != ShaderType::Local) ||
	    location >= ShaderVertexInputInfo::RES_MAX ||
	    location >= static_cast<uint32_t>(state.input_info.vertex->resources_num)) {
		return VertexInputScalarKind::Float;
	}

	switch (state.input_info.vertex->resources[location].Format()) {
		case Prospero::BufferFormat::k8UInt:
		case Prospero::BufferFormat::k16UInt:
		case Prospero::BufferFormat::k8_8UInt:
		case Prospero::BufferFormat::k32UInt:
		case Prospero::BufferFormat::k16_16UInt:
		case Prospero::BufferFormat::k8_8_8_8UInt:
		case Prospero::BufferFormat::k32_32UInt:
		case Prospero::BufferFormat::k16_16_16_16UInt:
		case Prospero::BufferFormat::k32_32_32UInt:
		case Prospero::BufferFormat::k32_32_32_32UInt: return VertexInputScalarKind::Uint;
		case Prospero::BufferFormat::k8SInt:
		case Prospero::BufferFormat::k16SInt:
		case Prospero::BufferFormat::k8_8SInt:
		case Prospero::BufferFormat::k32SInt:
		case Prospero::BufferFormat::k16_16SInt:
		case Prospero::BufferFormat::k8_8_8_8SInt:
		case Prospero::BufferFormat::k32_32SInt:
		case Prospero::BufferFormat::k16_16_16_16SInt:
		case Prospero::BufferFormat::k32_32_32SInt:
		case Prospero::BufferFormat::k32_32_32_32SInt: return VertexInputScalarKind::Sint;
		default: return VertexInputScalarKind::Float;
	}
}

uint32_t VertexParameterComponentCount(const InputBinding& input) {
	return std::clamp(input.component_count, 1u, 4u);
}

uint32_t VertexParameterScalarType(EmitterState& state, VertexInputScalarKind kind) {
	switch (kind) {
		case VertexInputScalarKind::Sint: return TypeI32(state);
		case VertexInputScalarKind::Uint: return TypeU32(state);
		case VertexInputScalarKind::Float:
		default: return TypeF32(state);
	}
}

uint32_t DefineInterfaceVariable(EmitterState& state, uint32_t type, spv::StorageClass storage,
                                 const char* name) {
	const auto variable =
	    state.builder.DefineGlobalVariable(TypePointer(state, storage, type), storage);
	state.interface_variables.push_back(variable);
	state.builder.AddName(variable, name);
	return variable;
}

namespace {

uint32_t BuiltInForInput(IR::StageInputKind kind) {
	switch (kind) {
		case IR::StageInputKind::VertexIndex: return spv::BuiltInVertexIndex;
		case IR::StageInputKind::InvocationId: return spv::BuiltInInvocationId;
		case IR::StageInputKind::PrimitiveId: return spv::BuiltInPrimitiveId;
		case IR::StageInputKind::TessCoord: return spv::BuiltInTessCoord;
		case IR::StageInputKind::InstanceIndex: return spv::BuiltInInstanceIndex;
		case IR::StageInputKind::FragCoord: return spv::BuiltInFragCoord;
		case IR::StageInputKind::FrontFacing: return spv::BuiltInFrontFacing;
		case IR::StageInputKind::Layer: return spv::BuiltInLayer;
		case IR::StageInputKind::SampleId: return spv::BuiltInSampleId;
		case IR::StageInputKind::BaryCoordSmooth: return spv::BuiltInBaryCoordKHR;
		case IR::StageInputKind::BaryCoordNoPerspective: return spv::BuiltInBaryCoordNoPerspKHR;
		case IR::StageInputKind::WorkgroupId: return spv::BuiltInWorkgroupId;
		case IR::StageInputKind::NumWorkgroups: return spv::BuiltInNumWorkgroups;
		case IR::StageInputKind::LocalInvocationId: return spv::BuiltInLocalInvocationId;
		case IR::StageInputKind::LocalInvocationIndex: return spv::BuiltInLocalInvocationIndex;
		case IR::StageInputKind::GlobalInvocationId: return spv::BuiltInGlobalInvocationId;
		default: return UINT32_MAX;
	}
}

void DefineInputs(EmitterState& state) {
	state.inputs.reserve(state.program.info.inputs.size());
	// v442: a callable split's module takes the workgroup built-ins from the launch id
	// (EmitRtWorkgroupInput); the compute inputs (every module's, merged) are not declared, so a
	// later call target's input does not change this module.
	const bool rt_fixed = state.program.bindings.rt_fixed;
	const auto rt_derived = [](IR::StageInputKind kind) {
		return kind == IR::StageInputKind::WorkgroupId || kind == IR::StageInputKind::NumWorkgroups ||
		       kind == IR::StageInputKind::LocalInvocationId ||
		       kind == IR::StageInputKind::LocalInvocationIndex ||
		       kind == IR::StageInputKind::GlobalInvocationId;
	};
	for (const auto& input: state.program.info.inputs) {
		if (rt_fixed && rt_derived(input.kind)) {
			continue;
		}
		state.inputs.push_back({input});
	}
	const auto add_builtin = [&](IR::StageInputKind kind, uint32_t components,
	                             const char* name) {
		if (std::ranges::none_of(state.inputs, [kind](const InputBinding& input) {
			    return input.kind == kind;
		    })) {
			state.inputs.push_back({{kind, 0, components, name}});
		}
	};
	if (!rt_fixed &&
	    IR::FindBinding(state.program.bindings, IR::DescriptorBindingKind::SharedMemory) != nullptr) {
		add_builtin(IR::StageInputKind::WorkgroupId, 3, "gl_WorkGroupID");
		add_builtin(IR::StageInputKind::NumWorkgroups, 3, "gl_NumWorkGroups");
	}
	if (state.lane_count == 2 && !rt_fixed) {
		add_builtin(IR::StageInputKind::LocalInvocationIndex, 1, "gl_LocalInvocationIndex");
		if (std::ranges::any_of(state.inputs, [](const InputBinding& input) {
			    return input.kind == IR::StageInputKind::GlobalInvocationId;
		    })) {
			add_builtin(IR::StageInputKind::WorkgroupId, 3, "gl_WorkGroupID");
		}
	}
	for (auto& input: state.inputs) {
		if (state.program.stage == ShaderType::Pixel &&
		    input.kind == IR::StageInputKind::Parameter) {
			const auto location = PixelParameterLocation(state, input.location);
			const auto alias = std::ranges::find_if(state.inputs, [&](const InputBinding& other) {
				return other.kind == IR::StageInputKind::Parameter && other.variable_id != 0 &&
				       PixelParameterLocation(state, other.location) == location;
			});
			if (alias != state.inputs.end()) {
				EXIT_IF(alias->per_vertex != input.per_vertex);
				input.variable_id = alias->variable_id;
				continue;
			}
		}
		uint32_t type = TypeU32(state);
		switch (input.kind) {
			case IR::StageInputKind::VertexIndex:
			case IR::StageInputKind::InvocationId:
			case IR::StageInputKind::PrimitiveId:
			case IR::StageInputKind::InstanceIndex:
			case IR::StageInputKind::Layer:
			case IR::StageInputKind::SampleId: type = TypeI32(state); break;
			case IR::StageInputKind::WorkgroupId:
			case IR::StageInputKind::NumWorkgroups:
			case IR::StageInputKind::LocalInvocationId:
			case IR::StageInputKind::GlobalInvocationId: type = TypeU32Vector(state, 3); break;
			case IR::StageInputKind::FragCoord: type = TypeF32Vector(state, 4); break;
			case IR::StageInputKind::TessCoord:
			case IR::StageInputKind::BaryCoordSmooth:
			case IR::StageInputKind::BaryCoordNoPerspective: type = TypeF32Vector(state, 3); break;
			case IR::StageInputKind::FrontFacing: type = TypeBool(state); break;
			case IR::StageInputKind::Parameter:
				if (state.program.stage == ShaderType::Vertex ||
				    state.program.stage == ShaderType::Local) {
					type = VertexParameterScalarType(
					    state, VertexParameterScalarKind(state, input.location));
					const auto components = VertexParameterComponentCount(input);
					if (components > 1u) {
						type = state.builder.Type(spv::OpTypeVector, type, components);
					}
				} else if (input.per_vertex) {
					type = state.builder.Type(spv::OpTypeArray, TypeF32Vector(state, 4),
					                          ConstantU32(state, 3));
				} else {
					type = TypeF32Vector(state, 4);
				}
				break;
			default: break;
		}
		input.variable_id =
		    DefineInterfaceVariable(state, type, spv::StorageClassInput, input.debug_name.c_str());
		if (input.kind == IR::StageInputKind::Layer || input.kind == IR::StageInputKind::SampleId) {
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationFlat);
		}
		if (input.kind == IR::StageInputKind::Parameter) {
			const auto flat = PixelParameterIsFlat(state, input.location);
			if (input.per_vertex) {
				state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
				                            spv::DecorationPerVertexKHR);
			} else if (flat) {
				state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
				                            spv::DecorationFlat);
			}
			if (state.program.stage == ShaderType::Pixel &&
			    state.input_info.pixel->ps_no_perspective && !flat && !input.per_vertex) {
				state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
				                            spv::DecorationNoPerspective);
			}
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationLocation,
			                            PixelParameterLocation(state, input.location));
		} else if (const auto builtin = BuiltInForInput(input.kind); builtin != UINT32_MAX) {
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationBuiltIn,
			                            builtin);
		}
	}
	// Diagnostics (KYTY_PROBE_TESS_LEVELS=1, probed evaluation shader): the tessellation levels the
	// tessellator used, appended per patch by the probe (spirvEmitterFlow.cpp).
	if (state.program.stage == ShaderType::TessellationEvaluation && ProbeTessLevelsWanted() &&
	    ProbeAppendWanted(state)) {
		const auto array = [&](uint32_t count) {
			return state.builder.Type(spv::OpTypeArray, TypeF32(state), ConstantU32(state, count));
		};
		state.probe_tess_outer_variable =
		    DefineInterfaceVariable(state, array(4), spv::StorageClassInput, "gl_TessLevelOuter");
		state.builder.AddAnnotation(spv::OpDecorate, state.probe_tess_outer_variable,
		                            spv::DecorationBuiltIn, spv::BuiltInTessLevelOuter);
		state.builder.AddAnnotation(spv::OpDecorate, state.probe_tess_outer_variable,
		                            spv::DecorationPatch);
		state.probe_tess_inner_variable =
		    DefineInterfaceVariable(state, array(2), spv::StorageClassInput, "gl_TessLevelInner");
		state.builder.AddAnnotation(spv::OpDecorate, state.probe_tess_inner_variable,
		                            spv::DecorationBuiltIn, spv::BuiltInTessLevelInner);
		state.builder.AddAnnotation(spv::OpDecorate, state.probe_tess_inner_variable,
		                            spv::DecorationPatch);
	}
	if (state.requirements.subgroup_local_invocation_id) {
		const auto variable = DefineInterfaceVariable(state, TypeU32(state), spv::StorageClassInput,
		                                              "gl_SubgroupInvocationID");
		state.subgroup_local_invocation_id_variable = variable;
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
		                            spv::BuiltInSubgroupLocalInvocationId);
		if (state.program.stage == ShaderType::Pixel) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationFlat);
		}
	}
}

void DefineOutputs(EmitterState& state) {
	state.outputs.reserve(state.program.info.outputs.size());
	uint32_t clip_distance_count = 0;
	uint32_t cull_distance_count = 0;
	for (const auto& output: state.program.info.outputs) {
		state.outputs.push_back({output});
		if (output.kind == IR::StageOutputKind::ClipDistance) {
			clip_distance_count = std::max(clip_distance_count, output.index + 1);
		} else if (output.kind == IR::StageOutputKind::CullDistance) {
			cull_distance_count = std::max(cull_distance_count, output.index + 1);
		}
	}
	if (state.program.stage == ShaderType::Mesh) {
		DefineMeshOutputs(state, clip_distance_count, cull_distance_count);
		return;
	}
	// KYTY_ZERO_POSITION_GUARD=0 leaves the zero-position clip plane out (2026-10-08: prompt glyphs
	// went missing in every new game whose session translated a vertex shader with it).
	static const bool zero_position_guard = [] {
		const char* value = std::getenv("KYTY_ZERO_POSITION_GUARD");
		return value == nullptr || value[0] != '0';
	}();
	// The tessellation evaluation shader is the last stage before rasterization in tessellated
	// draws (Wolverine's water DS kills vertices with a NaN position), so it gets the guard too.
	if (zero_position_guard &&
	    (state.program.stage == ShaderType::Vertex ||
	     state.program.stage == ShaderType::TessellationEvaluation) &&
	    clip_distance_count + cull_distance_count < 8u &&
	    std::ranges::any_of(state.outputs, [](const OutputBinding& output) {
		    return output.kind == IR::StageOutputKind::Position;
	    })) {
		// Reserve one plane for the enabled PA_CL_CLIP_CNTL clipping-error cull.
		state.invalid_position_clip_distance = clip_distance_count++;
		state.outputs.push_back({{IR::StageOutputKind::ClipDistance,
		                          state.invalid_position_clip_distance, 0, "gl_ClipDistance"}});
	}
	const auto BuiltIn = [&](uint32_t& variable, uint32_t type, const char* name,
	                         spv::BuiltIn builtin) {
		if (variable == 0) {
			variable = DefineInterfaceVariable(state, type, spv::StorageClassOutput, name);
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn, builtin);
		}
		return variable;
	};
	for (auto& binding: state.outputs) {
		switch (binding.kind) {
			case IR::StageOutputKind::Position:
				if (state.per_vertex_variable == 0) {
					const auto type = PerVertexType(state);
					state.builder.AddName(type, "gl_PerVertex");
					state.per_vertex_variable = DefineInterfaceVariable(
					    state, type, spv::StorageClassOutput, "outPerVertex");
				}
				binding.variable_id = state.per_vertex_variable;
				break;
			case IR::StageOutputKind::PointSize:
				binding.variable_id = BuiltIn(state.point_size_variable, TypeF32(state),
				                              "gl_PointSize", spv::BuiltInPointSize);
				break;
			case IR::StageOutputKind::ClipDistance:
				binding.variable_id =
				    BuiltIn(state.clip_distance_variable, F32ArrayType(state, clip_distance_count),
				            "gl_ClipDistance", spv::BuiltInClipDistance);
				break;
			case IR::StageOutputKind::CullDistance:
				binding.variable_id =
				    BuiltIn(state.cull_distance_variable, F32ArrayType(state, cull_distance_count),
				            "gl_CullDistance", spv::BuiltInCullDistance);
				break;
			case IR::StageOutputKind::Layer:
				binding.variable_id =
				    BuiltIn(state.layer_variable, TypeU32(state), "gl_Layer", spv::BuiltInLayer);
				break;
			case IR::StageOutputKind::ViewportIndex:
				binding.variable_id = BuiltIn(state.viewport_index_variable, TypeU32(state),
				                              "gl_ViewportIndex", spv::BuiltInViewportIndex);
				break;
			case IR::StageOutputKind::Depth:
				binding.variable_id = BuiltIn(state.depth_variable, TypeF32(state), "gl_FragDepth",
				                              spv::BuiltInFragDepth);
				break;
			case IR::StageOutputKind::SampleMask:
				binding.variable_id =
				    BuiltIn(state.sample_mask_variable, SampleMaskArrayType(state), "gl_SampleMask",
				            spv::BuiltInSampleMask);
				break;
			case IR::StageOutputKind::Parameter:
			case IR::StageOutputKind::Mrt: {
				const bool mrt = binding.kind == IR::StageOutputKind::Mrt &&
				                 state.program.stage == ShaderType::Pixel;
				const bool uint_output =
				    mrt && (state.input_info.pixel->target_uint_mask & (1u << binding.index)) != 0;
				const bool sint_output =
				    mrt && (state.input_info.pixel->target_sint_mask & (1u << binding.index)) != 0;
				const auto type     = uint_output   ? TypeU32Vector(state, 4)
				                      : sint_output ? TypeI32Vector(state, 4)
				                                    : TypeF32Vector(state, 4);
				binding.variable_id = DefineInterfaceVariable(state, type, spv::StorageClassOutput,
				                                              binding.debug_name.c_str());
				const bool dual_source = binding.kind == IR::StageOutputKind::Mrt &&
				                         state.program.stage == ShaderType::Pixel &&
				                         state.input_info.pixel->dual_source_blending;
				EXIT_NOT_IMPLEMENTED(dual_source && binding.index > 1);
				state.builder.AddAnnotation(spv::OpDecorate, binding.variable_id,
				                            spv::DecorationLocation,
				                            dual_source ? 0u : binding.location);
				if (dual_source) {
					state.builder.AddAnnotation(spv::OpDecorate, binding.variable_id,
					                            spv::DecorationIndex, binding.index);
				}
				break;
			}
		}
	}
}

} // namespace

void DefineModule(EmitterState& state) {
	state.interface_variables.reserve(state.program.info.inputs.size() +
	                                  state.program.info.outputs.size());
	DefineInputs(state);
	DefineOutputs(state);
	DefineTessellationInterfaces(state);
	DefineDescriptors(state);
	if (state.requirements.function_lds) {
		state.lds_variable = state.builder.AllocateId();
	}
	if (state.requirements.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.scratch_variable[half] = state.builder.AllocateId();
		}
	}
	state.main_func = state.builder.AllocateId();
	if (state.program.rt_role != IR::RtRole::None) {
		DefineRtInterface(state);
	}
	if (state.program.stage == ShaderType::Mesh) {
		state.mesh_guest_func = state.builder.AllocateId();
		state.builder.RequireCapability(spv::CapabilityMeshShadingEXT); // MeshShadingEXT
		state.builder.RequireExtension("SPV_EXT_mesh_shader");
		state.builder.AddExecutionMode(state.main_func,
		                               spv::ExecutionModeOutputTrianglesEXT); // OutputTrianglesEXT
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputVertices,
		                               state.input_info.vertex->mesh.max_vertices);
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputPrimitivesEXT,
		                               state.input_info.vertex->mesh.max_primitives);
		if (state.mesh_passes > 1u) {
			state.mesh_pass_variable = state.builder.DefineGlobalVariable(
			    TypePointer(state, spv::StorageClassPrivate, TypeU32(state)),
			    spv::StorageClassPrivate);
			state.builder.AddName(state.mesh_pass_variable, "mesh_pass");
		}
	}
	if (state.program.stage == ShaderType::TessellationControl ||
	    state.program.stage == ShaderType::TessellationEvaluation) {
		DefineTessellationExecutionModes(state);
	}
	state.entry_label = state.builder.AllocateId();

	state.builder.RequireCapability(spv::CapabilityShader);
	const bool float_controls = state.float_controls;
	if (float_controls) {
		state.builder.RequireCapability(spv::CapabilitySignedZeroInfNanPreserve);
	}
	if (HostFlushesF32Denorms(state)) {
		state.builder.RequireCapability(spv::CapabilityDenormFlushToZero);
	}
	if (state.requirements.buffer_u8) {
		state.builder.RequireExtension("SPV_KHR_8bit_storage");
		state.builder.RequireCapability(spv::CapabilityStorageBuffer8BitAccess);
	}
	if (state.requirements.buffer_u16) {
		state.builder.RequireCapability(spv::CapabilityStorageBuffer16BitAccess);
	}
	if (UsesPhysicalAddresses(state)) {
		state.builder.RequireCapability(spv::CapabilityInt64);
		state.builder.RequireCapability(spv::CapabilityPhysicalStorageBufferAddresses);
		state.builder.RequireExtension("SPV_KHR_physical_storage_buffer");
	}
	if (state.requirements.buffer_int64_atomics || state.requirements.shared_int64_atomics) {
		state.builder.RequireCapability(spv::CapabilityInt64);
		state.builder.RequireCapability(spv::CapabilityInt64Atomics);
	}
	if (state.requirements.shared_int64_atomics &&
	    state.lds_storage_class == spv::StorageClassWorkgroup) {
		state.builder.RequireVersion(0x00010400u);
		state.builder.RequireExtension("SPV_KHR_workgroup_memory_explicit_layout");
		state.builder.RequireCapability(spv::CapabilityWorkgroupMemoryExplicitLayoutKHR);
	}
	if (state.requirements.shader_clock) {
		state.builder.RequireCapability(spv::CapabilityShaderClockKHR);
		state.builder.RequireExtension("SPV_KHR_shader_clock");
	}
	if (state.clip_distance_variable != 0) {
		state.builder.RequireCapability(spv::CapabilityClipDistance);
	}
	if (state.cull_distance_variable != 0) {
		state.builder.RequireCapability(spv::CapabilityCullDistance);
	}
	if (state.layer_variable != 0 || InputVariableForKind(state, IR::StageInputKind::Layer) != 0) {
		state.builder.RequireVersion(0x00010500u);
		state.builder.RequireCapability(spv::CapabilityShaderLayer);
	}
	if (state.viewport_index_variable != 0) {
		state.builder.RequireVersion(0x00010500u);
		state.builder.RequireCapability(spv::CapabilityShaderViewportIndex);
	}
	if (InputVariableForKind(state, IR::StageInputKind::SampleId) != 0) {
		state.builder.RequireCapability(spv::CapabilitySampleRateShading);
	}
	if (state.requirements.image_gather_extended) {
		state.builder.RequireCapability(spv::CapabilityImageGatherExtended);
	}
	if (state.lane_count == 2 || state.requirements.subgroup_barrier ||
	    state.requirements.subgroup_ballot || state.requirements.subgroup_shuffle ||
	    state.requirements.subgroup_local_invocation_id) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniform);
	}
	if (state.lane_count == 2 || state.requirements.subgroup_ballot) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
	}
	if (state.requirements.subgroup_shuffle) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformShuffle);
	}
	if (state.requirements.subgroup_arithmetic) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
	}
	if (state.requirements.compute_derivatives && state.program.stage == ShaderType::Compute) {
		state.builder.RequireCapability(spv::CapabilityComputeDerivativeGroupQuadsKHR);
		state.builder.RequireExtension("SPV_KHR_compute_shader_derivatives");
	}
	const bool fragment_barycentric =
	    state.program.stage == ShaderType::Pixel &&
	    std::any_of(state.inputs.begin(), state.inputs.end(), [](const InputBinding& input) {
		    return input.per_vertex || input.kind == IR::StageInputKind::BaryCoordSmooth ||
		           input.kind == IR::StageInputKind::BaryCoordNoPerspective;
	    });
	if (fragment_barycentric) {
		state.builder.RequireCapability(spv::CapabilityFragmentBarycentricKHR);
		state.builder.RequireExtension("SPV_KHR_fragment_shader_barycentric");
	}
	if (float_controls) {
		state.builder.RequireExtension("SPV_KHR_float_controls");
	}
	state.builder.AddMemoryModel(UsesPhysicalAddresses(state)
	                                 ? spv::AddressingModelPhysicalStorageBuffer64
	                                 : spv::AddressingModelLogical,
	                             spv::MemoryModelGLSL450);
	// GCN/RDNA arithmetic preserves 32-bit signed zero, infinity, and NaN. Declaring that
	// contract prevents host compilers from treating synthesized IEEE values as finite.
	if (float_controls) {
		state.builder.AddExecutionMode(state.main_func,
		                               spv::ExecutionModeSignedZeroInfNanPreserve, 32u);
	}
	if (HostFlushesF32Denorms(state)) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDenormFlushToZero, 32u);
	}
	if (state.requirements.float64) {
		EXIT_NOT_IMPLEMENTED(state.program.stage == ShaderType::Compute &&
		                     state.input_info.compute->float_mode != 0xc0);
		// Use native rounding for MODE=0xc0, consistent with ordinary FP32 arithmetic.
		state.builder.RequireCapability(spv::CapabilityFloat64);
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeSignedZeroInfNanPreserve,
		                               64u);
		// FP64 denormal preservation is temporarily disabled.
		// state.builder.RequireCapability(spv::CapabilityDenormPreserve);
		// state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDenormPreserve, 64u);
	}
	if (const auto* cs = ShaderWorkgroupInput(state.program.stage, state.input_info);
	    cs != nullptr && state.program.rt_role == IR::RtRole::None) {
		uint32_t    local_x = state.requirements.compute_derivatives ? 2u : 1u;
		uint32_t    local_y = state.requirements.compute_derivatives ? 2u : 1u;
		uint32_t    local_z = 1u;
		local_x             = cs->threads_num[0] != 0u ? cs->threads_num[0] : local_x;
		local_y             = cs->threads_num[1] != 0u ? cs->threads_num[1] : local_y;
		local_z             = cs->threads_num[2] != 0u ? cs->threads_num[2] : local_z;
		if (state.lane_count == 2) {
			local_x = ((local_x * local_y * local_z + 63u) / 64u) * 32u;
			local_y = local_z = 1u;
			if (state.requirements.compute_derivatives) {
				local_y = local_x / 2u;
				local_x = 2u;
			}
		}
		// A pass runs 1/mesh_passes of the waves (mesh threads are one-dimensional).
		local_x /= state.mesh_passes;
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeLocalSize, local_x,
		                               local_y, local_z);
		// v446o: per-lane branches: lanes reconverge at structured merges (maximal reconvergence),
		// so the wave code after a split region sees its whole wave again. v446j lost the device
		// once in a cinematic with per-lane branches and no such guarantee.
		if (state.program.wave_size == 32u && state.lane_count == 1 && GetRtMaximalReconvergence() &&
		    MayUsePerLaneBranches(state)) {
			state.builder.RequireExtension("SPV_KHR_maximal_reconvergence");
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeMaximallyReconvergesKHR);
		}
	}
	if (state.program.stage == ShaderType::Pixel) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOriginUpperLeft);
		if (UsesPixelInterlock(state.program)) {
			state.builder.RequireExtension("SPV_EXT_fragment_shader_interlock");
			state.builder.RequireCapability(spv::CapabilityFragmentShaderPixelInterlockEXT);
			state.builder.AddExecutionMode(state.main_func,
			                               spv::ExecutionModePixelInterlockOrderedEXT);
		}
		if (state.depth_variable != 0) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDepthReplacing);
		}
		if (state.input_info.pixel->ps_early_z && !state.input_info.pixel->ps_pixel_kill_enable &&
		    !state.input_info.pixel->ps_depth_export_enable &&
		    !state.input_info.pixel->ps_sample_mask_export_enable) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeEarlyFragmentTests);
		}
	}
	if (state.requirements.compute_derivatives && state.program.stage == ShaderType::Compute) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDerivativeGroupQuadsKHR);
	}
	state.builder.AddName(state.main_func, "main");
	if (state.requirements.function_lds) {
		state.builder.AddName(state.lds_variable, "lds_dwords");
	}
	if (state.requirements.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddName(state.scratch_variable[half], "scratch_dwords");
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
