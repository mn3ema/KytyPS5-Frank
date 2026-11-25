// spv_regs: offline register probe for translated shaders (Stage 3 tooling, 11-21).
//
// usage: spv_regs <file.spv> [more.spv ...]
//
// For each SPIR-V module (compute entry points only for now) it builds a pipeline layout from the
// module's own descriptor decorations, creates the compute pipeline on the first Vulkan device with
// VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR, and prints the driver's executable statistics
// (register count, local memory, shared memory, binary size) in one line:
//   REGS <file> <stat>=<value>; ...
// No game, no emulator: a shader translated offline (shader_batch_tool, KYTY_BATCH_DUMP_DIR) or
// dumped in-game (KYTY_DUMP_SPIRV_HASHES) goes straight to the driver, so a translator change can
// be judged in seconds by what the NVIDIA compiler makes of it.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> ReadFile(const char* path) {
	std::ifstream in(path, std::ios::binary | std::ios::ate);
	if (!in) {
		return {};
	}
	const auto size = static_cast<size_t>(in.tellg());
	std::vector<uint32_t> words(size / 4);
	in.seekg(0);
	in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
	return words;
}

// SPIR-V opcodes used here.
enum : uint32_t {
	OpEntryPoint = 15, OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27, OpTypeArray = 28,
	OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43, OpVariable = 59,
	OpDecorate = 71, OpTypeAccelerationStructureKHR = 5341
};
enum : uint32_t { DecBlock = 2, DecBufferBlock = 3, DecBinding = 33, DecDescriptorSet = 34 };
enum : uint32_t {
	ScUniformConstant = 0, ScUniform = 2, ScPushConstant = 9, ScStorageBuffer = 12
};

struct Module {
	uint32_t                                       model = UINT32_MAX;
	std::string                                    entry;
	std::map<uint32_t, std::vector<uint32_t>>      types;  // id -> instruction words
	std::map<uint32_t, uint32_t>                   constants;
	std::map<uint32_t, uint32_t>                   binding, set;
	std::map<uint32_t, bool>                       block, buffer_block;
	std::vector<std::pair<uint32_t, uint32_t>>     variables;  // (pointer type, id)
	std::map<uint32_t, uint32_t>                   variable_class;
};

Module Parse(const std::vector<uint32_t>& w) {
	Module m;
	for (size_t i = 5; i < w.size();) {
		const uint32_t op = w[i] & 0xffffu, n = w[i] >> 16u;
		if (n == 0 || i + n > w.size()) {
			break;
		}
		const uint32_t* a = &w[i];
		switch (op) {
			case OpEntryPoint:
				if (m.model == UINT32_MAX) {
					m.model = a[1];
					m.entry = reinterpret_cast<const char*>(&a[3]);
				}
				break;
			case OpTypeImage: case OpTypeSampler: case OpTypeSampledImage: case OpTypeArray:
			case OpTypeRuntimeArray: case OpTypeStruct: case OpTypePointer: case OpTypeAccelerationStructureKHR:
				m.types[a[1]] = std::vector<uint32_t>(a, a + n);
				break;
			case OpConstant:
				if (n >= 4) m.constants[a[2]] = a[3];
				break;
			case OpVariable:
				m.variables.emplace_back(a[1], a[2]);
				m.variable_class[a[2]] = a[3];
				break;
			case OpDecorate:
				if (a[2] == DecBinding) m.binding[a[1]] = a[3];
				if (a[2] == DecDescriptorSet) m.set[a[1]] = a[3];
				if (a[2] == DecBlock) m.block[a[1]] = true;
				if (a[2] == DecBufferBlock) m.buffer_block[a[1]] = true;
				break;
			default: break;
		}
		i += n;
	}
	return m;
}

// Descriptor type and count of a resource variable.
bool Describe(const Module& m, uint32_t pointer_type, uint32_t storage, VkDescriptorType* type, uint32_t* count) {
	auto it = m.types.find(pointer_type);
	if (it == m.types.end() || it->second.size() < 4) return false;
	uint32_t pointee = it->second[3];
	*count = 1;
	for (int depth = 0; depth < 4; depth++) {
		auto t = m.types.find(pointee);
		if (t == m.types.end()) break;
		const uint32_t op = t->second[0] & 0xffffu;
		if (op == OpTypeArray) {
			auto c = m.constants.find(t->second[3]);
			*count *= c != m.constants.end() ? c->second : 1u;
			pointee = t->second[2];
			continue;
		}
		if (op == OpTypeRuntimeArray) {
			*count = 4096;  // bindless arrays: partially bound
			pointee = t->second[2];
			continue;
		}
		if (op == OpTypeImage) {
			const uint32_t dim = t->second[3], sampled = t->second[7];
			*type = dim == 5 /*Buffer*/ ? (sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
			                            : (sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
			return true;
		}
		if (op == OpTypeSampler) { *type = VK_DESCRIPTOR_TYPE_SAMPLER; return true; }
		if (op == OpTypeSampledImage) { *type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; return true; }
		if (op == OpTypeAccelerationStructureKHR) { *type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; return true; }
		if (op == OpTypeStruct) {
			const bool bb = m.buffer_block.count(pointee) != 0;
			*type = storage == ScStorageBuffer || bb ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			return true;
		}
		break;
	}
	return false;
}

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::fprintf(stderr, "%s failed: %d\n", #x, r_); return 1; } } while (0)

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: spv_regs <file.spv> [...]\n");
		return 2;
	}
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	VkInstance instance;
	CHECK(vkCreateInstance(&ici, nullptr, &instance));
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> pds(count);
	vkEnumeratePhysicalDevices(instance, &count, pds.data());
	VkPhysicalDevice pd = VK_NULL_HANDLE;
	for (auto d : pds) {  // the discrete GPU (the laptop's NVIDIA), not the Intel iGPU
		VkPhysicalDeviceProperties p;
		vkGetPhysicalDeviceProperties(d, &p);
		if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = d; std::printf("DEVICE %s\n", p.deviceName); break; }
	}
	if (pd == VK_NULL_HANDLE) { std::fprintf(stderr, "no discrete GPU\n"); return 1; }

	// Every feature the device has, so modules that need int64, descriptor indexing, BDA... compile.
	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exec {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
	VkPhysicalDeviceVulkan13Features f13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	f13.pNext = &exec;
	VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	f12.pNext = &f13;
	VkPhysicalDeviceVulkan11Features f11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
	f11.pNext = &f12;
	VkPhysicalDeviceFeatures2 features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	features.pNext = &f11;
	vkGetPhysicalDeviceFeatures2(pd, &features);
	if (!exec.pipelineExecutableInfo) { std::fprintf(stderr, "no pipelineExecutableInfo\n"); return 1; }

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;
	const char* exts[] = {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
	VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext = &features;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = 1;
	dci.ppEnabledExtensionNames = exts;
	VkDevice device;
	CHECK(vkCreateDevice(pd, &dci, nullptr, &device));
	auto getProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
	auto getStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(pd, &props);

	for (int f = 1; f < argc; f++) {
		const auto words = ReadFile(argv[f]);
		if (words.size() < 5 || words[0] != 0x07230203u) { std::printf("SKIP %s not SPIR-V\n", argv[f]); continue; }
		const Module m = Parse(words);
		if (m.model != 5 /*GLCompute*/) { std::printf("SKIP %s execution model %u (compute only for now)\n", argv[f], m.model); continue; }
		std::map<uint32_t, std::map<uint32_t, VkDescriptorSetLayoutBinding>> sets;
		bool push = false;
		for (const auto& [ptype, id] : m.variables) {
			const uint32_t sc = m.variable_class.at(id);
			if (sc == ScPushConstant) { push = true; continue; }
			if (sc != ScUniformConstant && sc != ScUniform && sc != ScStorageBuffer) continue;
			if (!m.binding.count(id)) continue;
			VkDescriptorType type;
			uint32_t n = 1;
			if (!Describe(m, ptype, sc, &type, &n)) continue;
			auto& b = sets[m.set.count(id) ? m.set.at(id) : 0u][m.binding.at(id)];
			b.binding = m.binding.at(id);
			b.descriptorType = type;
			b.descriptorCount = n;
			b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		std::vector<VkDescriptorSetLayout> layouts;
		uint32_t max_set = 0;
		for (auto& [s, _] : sets) max_set = s > max_set ? s : max_set;
		bool ok = true;
		for (uint32_t s = 0; s <= max_set && ok; s++) {
			std::vector<VkDescriptorSetLayoutBinding> bindings;
			std::vector<VkDescriptorBindingFlags>     flags;
			if (sets.count(s)) for (auto& [_, b] : sets[s]) { bindings.push_back(b); flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT); }
			VkDescriptorSetLayoutBindingFlagsCreateInfo fci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
			fci.bindingCount = static_cast<uint32_t>(flags.size());
			fci.pBindingFlags = flags.data();
			VkDescriptorSetLayoutCreateInfo lci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
			lci.pNext = &fci;
			lci.bindingCount = static_cast<uint32_t>(bindings.size());
			lci.pBindings = bindings.data();
			VkDescriptorSetLayout l;
			if (vkCreateDescriptorSetLayout(device, &lci, nullptr, &l) != VK_SUCCESS) { ok = false; break; }
			layouts.push_back(l);
		}
		VkPushConstantRange range {VK_SHADER_STAGE_COMPUTE_BIT, 0, props.limits.maxPushConstantsSize};
		VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		plci.setLayoutCount = static_cast<uint32_t>(layouts.size());
		plci.pSetLayouts = layouts.data();
		plci.pushConstantRangeCount = push ? 1u : 0u;
		plci.pPushConstantRanges = &range;
		VkPipelineLayout pl;
		if (!ok || vkCreatePipelineLayout(device, &plci, nullptr, &pl) != VK_SUCCESS) { std::printf("FAIL %s layout\n", argv[f]); continue; }
		VkShaderModuleCreateInfo smci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		smci.codeSize = words.size() * 4;
		smci.pCode = words.data();
		VkShaderModule sm;
		if (vkCreateShaderModule(device, &smci, nullptr, &sm) != VK_SUCCESS) { std::printf("FAIL %s module\n", argv[f]); continue; }
		VkComputePipelineCreateInfo cpci {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
		cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, m.entry.c_str(), nullptr};
		cpci.layout = pl;
		VkPipeline pipe;
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) { std::printf("FAIL %s pipeline\n", argv[f]); continue; }
		VkPipelineInfoKHR pinfo {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
		pinfo.pipeline = pipe;
		uint32_t ne = 0;
		getProps(device, &pinfo, &ne, nullptr);
		std::vector<VkPipelineExecutablePropertiesKHR> eps(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
		getProps(device, &pinfo, &ne, eps.data());
		for (uint32_t e = 0; e < ne; e++) {
			VkPipelineExecutableInfoKHR einfo {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
			einfo.pipeline = pipe;
			einfo.executableIndex = e;
			uint32_t ns = 0;
			getStats(device, &einfo, &ns, nullptr);
			std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
			getStats(device, &einfo, &ns, st.data());
			std::printf("REGS %s %s ", argv[f], eps[e].name);
			for (const auto& s : st) {
				switch (s.format) {
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::printf("%s=%llu; ", s.name, static_cast<unsigned long long>(s.value.u64)); break;
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::printf("%s=%lld; ", s.name, static_cast<long long>(s.value.i64)); break;
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::printf("%s=%.2f; ", s.name, s.value.f64); break;
					default: std::printf("%s=%d; ", s.name, s.value.b32); break;
				}
			}
			std::printf("\n");
		}
		vkDestroyPipeline(device, pipe, nullptr);
		vkDestroyShaderModule(device, sm, nullptr);
		vkDestroyPipelineLayout(device, pl, nullptr);
		for (auto l : layouts) vkDestroyDescriptorSetLayout(device, l, nullptr);
		std::fflush(stdout);
	}
	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);
	return 0;
}
