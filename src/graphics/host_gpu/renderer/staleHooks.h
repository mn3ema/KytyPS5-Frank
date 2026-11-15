#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STALEHOOKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STALEHOOKS_H_

#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {

// KYTY_STALE_HOOKS=1 (diagnostics for GPU faults in freed image memory): hooks the Vulkan
// dispatcher and reports, as [stale-use], every image view or image handed to the GPU after it
// was destroyed, by any path: push descriptors, descriptor set writes, binding a descriptor set
// that still holds a destroyed view, rendering attachments, copies, blits, clears, resolves and
// barriers, and the same for buffers (descriptors, vertex/index/indirect binds, copies). Also reports image memory freed while views of it are still alive, and views, images
// or descriptor sets holding them destroyed while a submission using them is still pending.
void InstallStaleHooks(vk::Device device);
bool StaleHooksEnabled();
// vma.cpp: images are created by VMA, outside the dispatcher.
void StaleHooksNoteImage(VkImage image, const vk::ImageCreateInfo* info, bool created);
// streamBuffer.cpp: VMA buffers, likewise (buffers made through the dispatcher are hooked).
void StaleHooksNoteBuffer(VkBuffer buffer, uint64_t size, const char* kind, bool created);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STALEHOOKS_H_ */
