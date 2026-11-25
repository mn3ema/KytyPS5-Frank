#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {
class RenderContext;
enum class PageFaultAccess;
} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Memory";
	static constexpr auto        initialize = Libs::LibKernel::Memory::Initialize;
	static constexpr auto        shutdown   = Libs::LibKernel::Memory::Shutdown;
};

using callback_func_t = void (*)(uintptr_t addr, size_t size);

constexpr uint32_t KERNEL_MAXIMUM_NAME_LENGTH = 32;
constexpr uint64_t kExtendedMemoryBase       = 0x080000000000ull;
constexpr uint64_t kExtendedMemorySize       = 512ull * 1024 * 1024 * 1024;

struct VirtualQueryInfo {
	uintptr_t start;
	uintptr_t end;
	uint64_t  offset;
	int32_t   protection;
	int32_t   memory_type;
	uint32_t  is_flexible  : 1;
	uint32_t  is_direct    : 1;
	uint32_t  is_stack     : 1;
	uint32_t  is_pooled    : 1;
	uint32_t  is_committed : 1;
	uint32_t  is_gpu_prt   : 1;
	uint32_t  amm_usage    : 1;
	uint32_t  reserved     : 1;
	char      name[KERNEL_MAXIMUM_NAME_LENGTH];
	uint8_t   gpu_mask_id;
	uint8_t   reserved2;
};

static_assert(sizeof(VirtualQueryInfo) == 72, "VirtualQueryInfo struct size is incorrect");

struct KernelBatchMapEntry {
	void*         start;
	uint64_t      offset;
	uint64_t      length;
	unsigned char protection;
	unsigned char type;
	int16_t       reserved;
	int32_t       operation;
};

static_assert(sizeof(KernelBatchMapEntry) == 32, "KernelBatchMapEntry struct size is incorrect");

struct KernelMemoryPoolBatchEntry {
	uint32_t op;
	uint32_t flags;
	union {
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} commit;
		struct {
			void*    addr;
			uint64_t len;
		} decommit;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
		} protect;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} type_protect;
		struct {
			void*    dst;
			void*    src;
			uint64_t len;
		} move;
		uintptr_t padding[3];
	};
};

static_assert(sizeof(KernelMemoryPoolBatchEntry) == 32,
              "KernelMemoryPoolBatchEntry struct size is incorrect");

struct KernelMemoryPoolBlockStats {
	int32_t available_flushed_blocks;
	int32_t available_cached_blocks;
	int32_t allocated_flushed_blocks;
	int32_t allocated_cached_blocks;
};

static_assert(sizeof(KernelMemoryPoolBlockStats) == 16,
              "KernelMemoryPoolBlockStats struct size is incorrect");

void                   RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func);
void                   SetFlexibleMemorySize(uint64_t size);
int AllocateDirectMemory(int64_t search_start, int64_t search_end, size_t size, size_t alignment,
                         int memory_type, int64_t* phys_addr_out, bool automatic = false);
int MapAutomaticMemory(uint64_t vaddr, size_t size, int type, int prot);
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
// Diagnostics: the virtual mappings of the direct memory behind a range (aliases included).
std::vector<std::pair<uint64_t, uint64_t>> FindDirectMemoryAliases(uint64_t vaddr, uint64_t size);
bool                   TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);
// GPU thread: some byte of [vaddr, vaddr + size) was written by GPU work into a buffer, so the
// GPU holds it and a host read would wait for the GPU.
[[nodiscard]] bool     IsGpuBufferWritten(uint64_t vaddr, uint64_t size);
// GPU thread: IsGpuBufferWritten, or some byte is downloading (its download not yet published), so
// a host read would wait for the GPU either way (v446p, SRT diagnostics).
[[nodiscard]] bool     IsGpuBufferPending(uint64_t vaddr, uint64_t size);
// GPU thread: starts the download of the GPU-written bytes of [vaddr, vaddr + size) without
// waiting for it (KYTY_SRT_DEFER). Returns the scheduler tick whose completion (and priority
// operations) publish them, or 0 when nothing is pending and an ordinary read may proceed.
[[nodiscard]] uint64_t StartGpuBufferReadback(uint64_t vaddr, uint64_t size);
// GPU thread: the value of a dword as of an earlier, already completed copy (it never waits; 0
// before the first copy lands), and a new copy of its current value is recorded (KYTY_GPU_SIZED_DYNAMIC).
[[nodiscard]] uint32_t PeekGpuDword(uint64_t vaddr);
bool                   TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
// GPU thread: the backing bytes of [vaddr, vaddr + size) when one mapping holds them and no GPU
// work has written any of them, else nullptr. Current until the GPU thread marks them
// GPU-written or the guest remaps them.
[[nodiscard]] const uint8_t* FindGpuCleanBacking(uint64_t vaddr, uint64_t size);
// Lets the calling thread make the GPU thread's clean-backing queries (FindGpuCleanBacking,
// TryReadGpuCleanBacking) while the GPU thread waits for it and makes only the same read-only
// queries itself: the stage-preparation helper (pipelineCache.cpp, StagePrepWorker).
void SetGpuReadDelegate(bool delegate);
// GPU thread or delegate: copies [vaddr, vaddr + size) from the backing store when no GPU work has
// written any of it. Never downloads, faults or waits; false otherwise.
[[nodiscard]] bool TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size);
// Any thread, for a read that faulted at fault_vaddr on a page protected because the GPU wrote to
// it: reads [vaddr, vaddr + size) from the backing store when none of those bytes is GPU-written.
bool TryReadCleanFaultingBytes(uint64_t fault_vaddr, uint64_t vaddr, void* data, uint64_t size);
// v445s (KYTY_FAST_FINISH_CLEANREAD): RenderContext::TryLiftCleanReadPage, 0 without a renderer.
int  TryLiftCleanFaultingPage(uint64_t fault_vaddr) noexcept;
// Crash diagnostics: prints the guest mapping, the host state and the recent guest mapping changes
// at an address whose fault nothing handled.
void DescribeUnhandledFault(uint64_t vaddr);
// The GPU side of unmapping these ranges, all at once, ahead of the remaps that follow (an AMM
// batch). The remaps' own unmaps then find the ranges no longer GPU-mapped and skip the GPU.
void PrepareGpuUnmap(std::span<const std::pair<uint64_t, uint64_t>> ranges);

// Diagnostics: this thread's running totals of time in the GPU side of maps and unmaps.
struct MapCost {
	uint64_t gpu_map_ns   = 0;
	uint64_t gpu_unmap_ns = 0;
	uint64_t gpu_unmaps   = 0;
	uint64_t lock_wait_ns = 0; // waiting for the memory-operation lock (KernelMapDirectMemory2)
	uint64_t replace_ns   = 0; // replacing the old mapping with a reservation
	uint64_t backing_ns   = 0; // host mapping of the backing view
	uint64_t steps[8] {};      // can-map, query, physical map, find, publish (to the end of the
	                           // map), set physical type, set range type
	uint64_t munmap_ns = 0;    // KernelMunmap, all of it
	uint64_t munmaps   = 0;
	uint64_t unmap_steps[6] {}; // lock wait, query, backing unmap, owner (physical/flexible/pooled)
	                            // release, range table, free callback
	uint64_t splits      = 0; // partial unmaps of a host view: count, bytes mapped again, time
	uint64_t split_bytes = 0;
	uint64_t split_ns    = 0;
	uint64_t resync_ns   = 0; // page-tracker protection of the parts mapped again
	uint64_t commit_ns       = 0; // host backing: committing the section, mapping views, protecting them
	uint64_t commit_bytes    = 0;
	uint64_t mapview_ns      = 0;
	uint64_t mapviews        = 0;
	uint64_t view_protect_ns = 0;
};
MapCost MapCostSnapshot();

// Another module's part of that report (the AMM engine's queued maps).
void SetUnhandledFaultHook(void (*hook)(uint64_t vaddr));
// v400: a guest access to memory a still-queued AMM map covers runs that map now (the hardware
// finishes a map in microseconds; the emulator's ring was 1.5 s behind). True when it did.
void SetQueuedMapResolver(bool (*resolver)(uint64_t vaddr));
[[nodiscard]] bool ResolveQueuedMap(uint64_t vaddr);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
// Like ClampRangeSize, but 0 for a range that starts outside committed memory instead of EXIT.
[[nodiscard]] uint64_t TryClampRangeSize(uint64_t vaddr, uint64_t size);
// Whether every byte of the range is committed guest memory, by the kernel's own table of guest
// mappings (what sceKernelVirtualQuery reports), without asking the host. An empty range is, and
// so is every range before the kernel's memory is set up (unit tests).
[[nodiscard]] bool     IsCommittedRange(uint64_t vaddr, uint64_t size);
// Diagnostics: every committed guest mapping (start, size), from the kernel's table.
[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>> CommittedRanges();
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept;
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
void                   InstallGpuResources(Graphics::RenderContext* renderer) noexcept;
[[nodiscard]] bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;
// A fault on a page the tracker left stricter than needed (KYTY_LAZY_READ_RELEASE): fixed, retry.
[[nodiscard]] bool ResolveGpuStaleFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;

int KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                                               const char* name);
int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags);
int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name);
int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len);
int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len);
size_t KYTY_SYSV_ABI KernelGetDirectMemorySize();
int KYTY_SYSV_ABI    KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                     size_t alignment, int64_t* phys_addr_out,
                                                     size_t* size_out);
int KYTY_SYSV_ABI    KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                             int* gpu_available);
int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name);
int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len);
int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len);
int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled();
int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot);
int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size);
int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size);
int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end);
int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment);
int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot);
int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot);
int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out);
int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size);

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name);
void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name,
                               bool fixed = false);
uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size,
                                  Common::VirtualMemory::Mode mode, const char* name);
bool     ProtectGuestMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
                            Common::VirtualMemory::Mode* old_mode = nullptr);
// Transient PageManager watch state; does not change the guest mapping's semantic protection.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void     TestBeforeNextBackingMap(callback_func_t callback);
void     TestFailNextPhysicalMemoryUnmap();
void     TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps);
void     TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps);
void     TestFailNextFixedReserveRangeRegistration();
void     TestFailNextVirtualRangeReplacement();
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
bool     TestGuestFreeRangeBounds();
#endif

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_ */
