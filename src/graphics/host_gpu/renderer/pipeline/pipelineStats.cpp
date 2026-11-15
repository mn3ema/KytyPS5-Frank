#include "graphics/host_gpu/renderer/pipeline/pipelineStats.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::PipelineStats {

namespace {

std::atomic<bool> g_device_enabled {false};

struct Filter {
	bool                  requested = false;
	bool                  all       = false;
	std::vector<uint64_t> hashes;
};

const Filter& GetFilter() {
	static const Filter filter = [] {
		Filter      result;
		const char* value = std::getenv("KYTY_PIPE_STATS");
		if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
			return result;
		}
		result.requested = true;
		if (std::strcmp(value, "1") == 0) {
			result.all = true;
			return result;
		}
		std::string text = value;
		for (auto& c: text) {
			if (c == ',' || c == '+' || c == ';') {
				c = ' ';
			}
		}
		size_t pos = 0;
		while (pos < text.size()) {
			while (pos < text.size() && text[pos] == ' ') {
				pos++;
			}
			size_t end = pos;
			while (end < text.size() && text[end] != ' ') {
				end++;
			}
			if (end > pos) {
				result.hashes.push_back(std::strtoull(text.substr(pos, end - pos).c_str(), nullptr, 16));
			}
			pos = end;
		}
		result.all = result.hashes.empty();
		return result;
	}();
	return filter;
}

bool Wanted(uint64_t hash0, uint64_t hash1) {
	const auto& filter = GetFilter();
	if (filter.all) {
		return true;
	}
	return std::ranges::find(filter.hashes, hash0) != filter.hashes.end() ||
	       (hash1 != 0 && std::ranges::find(filter.hashes, hash1) != filter.hashes.end());
}

std::string Value(const VkPipelineExecutableStatisticKHR& stat) {
	char text[64];
	switch (stat.format) {
		case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
			return stat.value.b32 != VK_FALSE ? "true" : "false";
		case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
			std::snprintf(text, sizeof(text), "%" PRId64, stat.value.i64);
			return text;
		case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
			std::snprintf(text, sizeof(text), "%" PRIu64, stat.value.u64);
			return text;
		case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
			std::snprintf(text, sizeof(text), "%.3f", stat.value.f64);
			return text;
		default: return "?";
	}
}

} // namespace

bool DeviceEnabled() {
	return g_device_enabled.load(std::memory_order_relaxed);
}

void SetDeviceEnabled(bool enabled) {
	g_device_enabled.store(enabled, std::memory_order_relaxed);
}

bool Requested() {
	return GetFilter().requested;
}

vk::PipelineCreateFlags CaptureFlags() {
	return DeviceEnabled() ? vk::PipelineCreateFlags(vk::PipelineCreateFlagBits::eCaptureStatisticsKHR)
	                       : vk::PipelineCreateFlags {};
}

void Report(vk::Device device_handle, vk::Pipeline pipeline, char kind, uint64_t hash0,
            uint64_t hash1) {
	if (!DeviceEnabled() || pipeline == nullptr || !Wanted(hash0, hash1)) {
		return;
	}
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkGetPipelineExecutablePropertiesKHR == nullptr ||
	    d.vkGetPipelineExecutableStatisticsKHR == nullptr) {
		return;
	}
	const VkDevice device = device_handle;
	VkPipelineInfoKHR info {};
	info.sType    = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
	info.pipeline = pipeline;
	uint32_t count = 0;
	if (d.vkGetPipelineExecutablePropertiesKHR(device, &info, &count, nullptr) != VK_SUCCESS ||
	    count == 0) {
		return;
	}
	std::vector<VkPipelineExecutablePropertiesKHR> executables(count);
	for (auto& e: executables) {
		e.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
	}
	if (d.vkGetPipelineExecutablePropertiesKHR(device, &info, &count, executables.data()) !=
	    VK_SUCCESS) {
		return;
	}
	std::string out;
	for (uint32_t i = 0; i < count; i++) {
		VkPipelineExecutableInfoKHR exec {};
		exec.sType           = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
		exec.pipeline        = pipeline;
		exec.executableIndex = i;
		uint32_t stat_count  = 0;
		if (d.vkGetPipelineExecutableStatisticsKHR(device, &exec, &stat_count, nullptr) !=
		    VK_SUCCESS) {
			continue;
		}
		std::vector<VkPipelineExecutableStatisticKHR> stats(stat_count);
		for (auto& s: stats) {
			s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
		}
		if (d.vkGetPipelineExecutableStatisticsKHR(device, &exec, &stat_count, stats.data()) !=
		    VK_SUCCESS) {
			continue;
		}
		char head[160];
		std::snprintf(head, sizeof(head),
		              "[pipe-stats] %c %016" PRIx64 " %016" PRIx64 " %s (subgroup %u):", kind,
		              hash0, hash1, executables[i].name, executables[i].subgroupSize);
		out += head;
		for (uint32_t s = 0; s < stat_count; s++) {
			out += " ";
			out += stats[s].name;
			out += "=";
			out += Value(stats[s]);
			out += ";";
		}
		out += "\n";
	}
	static std::mutex print_mutex;
	std::lock_guard   lock(print_mutex);
	std::fputs(out.c_str(), stderr);
	std::fflush(stderr);
}

} // namespace Libs::Graphics::PipelineStats
