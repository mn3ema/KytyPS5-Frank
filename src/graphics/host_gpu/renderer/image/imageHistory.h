#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEHISTORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEHISTORY_H_

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

namespace Libs::Graphics::ImageHistory {

// KYTY_FAULT_MAP=1 (v122): what the host did with each VkImage, kept past its destruction, so the
// device-fault report can print, for the image at the fault address, every bindless slot that ever
// held one of its views (written when, by which heap key, replaced when and by what), when it was
// last bound through set 0 and by which shader, when it was last an attachment, and when it was
// created and destroyed. Handles are reused by the driver at once, so a handle keeps a short list
// of generations, each with its guest address and extent.
bool Enabled();

void NoteCreate(VkImage image, uint64_t guest, uint64_t guest_size, uint32_t width, uint32_t height,
                uint32_t layers, uint32_t levels, uint32_t format, uint64_t tick);
void NoteDestroy(VkImage image, uint64_t tick);
void NoteView(VkImageView view, VkImage image); // a view created on the image
VkImage ImageOfView(VkImageView view);            // null when unknown
// kind: 'D' set-0 descriptor, 'A' colour attachment, 'Z' depth attachment, 'S' storage.
void NoteBind(VkImage image, uint64_t tick, uint64_t shader_hash, char kind);
void NoteBindView(VkImageView view, uint64_t tick, uint64_t shader_hash, char kind);
// A bindless slot write: the slot now holds a view of `image` (null for a placeholder); `previous`
// is the image whose view the slot held before (null for none/placeholder).
void NoteSlotWrite(uint32_t binding, uint32_t slot, VkImage image, VkImage previous, uint64_t tick,
                   uint64_t heap_base, uint32_t key);
// Every generation of the handle, newest first, with its slot events.
void Describe(VkImage image, FILE* out);
// v375: the guest ranges of the image's generations destroyed within the last within_ms.
void RecentGuestRanges(VkImage image, double within_ms, std::vector<std::pair<uint64_t, uint64_t>>* out);

} // namespace Libs::Graphics::ImageHistory

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGEHISTORY_H_
