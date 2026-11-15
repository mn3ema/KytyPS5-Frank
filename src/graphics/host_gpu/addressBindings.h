#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_ADDRESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_ADDRESSBINDINGS_H_

#include <cstdint>
#include <cstdio>
#include <vector>

namespace Libs::Graphics::AddressBindings {

// KYTY_ADDRESS_BINDING_REPORT=1 (v370, Crash B plan B1): the driver's own map of GPU virtual
// addresses, from VK_EXT_device_address_binding_report. Every bind and unbind of an image, buffer
// or memory object is kept (unbound ranges in a ring of the last 262,144), so a fault report can
// say what really lived at the address and when it went away, whatever Kyty's own model of the
// address space says (it disagreed with Aftermath for compressed surfaces, 10-24).
bool Enabled();
// object_type: VkObjectType; internal: VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT.
void Note(bool bind, int32_t object_type, uint64_t handle, uint64_t address, uint64_t size, bool internal);
// Every live or unbound range containing [address & ~(precision-1), +precision), and the nearest
// live ranges within 32 MiB below and above it.
void Describe(uint64_t address, uint64_t precision, FILE* out);
// v375: the image handles the driver unbound from a range containing the address within the last
// within_s seconds (newest first, at most 32).
std::vector<uint64_t> RecentlyUnboundImages(uint64_t address, double within_s);

} // namespace Libs::Graphics::AddressBindings

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_ADDRESSBINDINGS_H_
