#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESTATS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// Diagnostics, KYTY_PIPE_STATS (off by default): the driver's own compile statistics for each
// pipeline (VK_KHR_pipeline_executable_properties; on NVIDIA register count, local-memory spill
// size, shared memory, binary size and the like), printed once per pipeline as [pipe-stats]
// lines, so a slow shader can be checked for register pressure or spills without a profiler.
// KYTY_PIPE_STATS=1 reports every pipeline; a list of hex shader hashes (space, ',' or '+'
// separated) only the pipelines that use one of them. Pipelines are then created with
// VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR, which may miss the driver's pipeline cache.
namespace Libs::Graphics {

namespace PipelineStats {

// Whether the device was created with the extension (set by device creation).
bool DeviceEnabled();
void SetDeviceEnabled(bool enabled);
// Whether KYTY_PIPE_STATS asks for the statistics (before the device exists).
bool Requested();
// The flag to add at creation time (never to a recipe): eCaptureStatisticsKHR when enabled.
vk::PipelineCreateFlags CaptureFlags();
// kind 'G' (vs, ps) or 'C' (cs, 0).
void Report(vk::Device device, vk::Pipeline pipeline, char kind, uint64_t hash0, uint64_t hash1);

} // namespace PipelineStats
} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESTATS_H_ */
