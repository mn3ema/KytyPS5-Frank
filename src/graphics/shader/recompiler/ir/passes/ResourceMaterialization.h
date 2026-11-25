#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		// Written at addresses the host cannot bound: the shader records the pages it writes in
		// the GPU write bitmap (BufferCache::WriteBitmap, rework.md Phase 2 first step).
		bool                   write_tracked                   = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		// KYTY_GPU_SIZED_DYNAMIC: see BufferResource::gpu_records_flat.
		uint32_t               gpu_records_flat                = BufferResource::NoGpuRecords;
		uint32_t               gpu_records_add_flat            = BufferResource::NoGpuRecords;
		uint32_t               gpu_records_add                 = 0;
		bool                   gpu_records_ssa                 = false;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          bindless                   = false;
		bool                          bindless_address           = false; // v439
		bool                          bindless_value             = false; // v439b
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		bool     bindless                = false;
		uint32_t bindless_mapping_offset = 0;
		bool     operator==(const Sampler&) const = default;
	};

	std::vector<Buffer>  buffers;
	std::vector<Image>   images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// KYTY_WRITE_BITMAP=1 (ResourceSpecialization::Buffer::write_tracked).
[[nodiscard]] bool WriteBitmapEnabled();

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// Diagnostics: the source line of the indirect-image check that failed the last
// MaterializeResources call on this thread, or 0.
int LastIndirectImageFailureLine();

// v424: the canonical packed stride (stride, swizzle enable, index stride, ADD_TID) of a
// materialized buffer descriptor, as BindingLayout::buffer_stride_count data words carry it.
[[nodiscard]] uint32_t BufferStrideData(const DescriptorValue& value);

// v432: the stride word of a DynamicBufferFormat buffer: stride (bits 0-13), the format's memory
// layout (bits 14-17, Format::kBufferLayouts; 0: not a valid format), its component type (bits
// 18-20, Format::ComponentType) and its four destination selects (bits 21-31, base 6 per component: 0 zero, 1 one,
// 2 + n source component n, already wrapped by the format's component count).
[[nodiscard]] uint32_t BufferFormatData(const DescriptorValue& value);

// v432: a read-only, untracked, directly bound formatted buffer without swizzling
// (KYTY_BUFFER_DYNAMIC_FORMAT, default on, with KYTY_BUFFER_STRIDE_DATA): its format and
// destination selects are per-draw data (BufferResource::dynamic_format).
[[nodiscard]] bool DynamicBufferFormat(const ShaderInfo& info, const ResourceSpecialization& specialization,
                                       uint32_t index);

// v432: the canonical form of a specialization recorded before its per-draw fields became data
// (dynamic buffer formats, image class, layer and cube), as materialization builds it.
void CanonicalizeDynamicFields(const ShaderInfo& info, ResourceSpecialization& specialization);

// v429: a load-only, sampled, non-bindless, direct image slot whose T# is float or unsigned
// integer (KYTY_IMAGE_DYNAMIC_CLASS, default on): bound in both class groups, the class is
// per-draw data (ImageResource::dynamic_class).
[[nodiscard]] bool DynamicImageClass(const ShaderInfo& info, uint32_t index,
                                     const ResourceSpecialization::Image& image);
// v430: a load-only slot of an array instruction with a 2D or 2D-array T# (KYTY_IMAGE_DYNAMIC_LAYER,
// default on): declared as a 2D array, the T#'s type is per-draw data (ImageResource::dynamic_layer).
[[nodiscard]] bool DynamicImageLayer(const ShaderInfo& info, uint32_t index,
                                     const ResourceSpecialization::Image& image);
// v431: an array instruction's sampled slot with a 2D-array or cube T# (KYTY_IMAGE_DYNAMIC_CUBE,
// default on): the cube flag is per-draw data (ImageResource::dynamic_cube).
[[nodiscard]] bool DynamicImageCube(const ShaderInfo& info, uint32_t index,
                                    const ResourceSpecialization::Image& image);

// v433: a 3D instruction's sampled, read-only, direct slot whose T# is a 3D or 2D texture and whose
// class is fixed (KYTY_IMAGE_DYNAMIC_3D, default on): bound as 3D and 2D, the T#'s type is per-draw
// data (ImageResource::dynamic_3d).
[[nodiscard]] bool DynamicImage3D(const ShaderInfo& info, uint32_t index,
                                  const ResourceSpecialization::Image& image);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
