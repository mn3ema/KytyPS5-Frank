#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMGTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMGTRACE_H_

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// KYTY_IMG_TRACE=<hex guest address> (diagnostics, changes nothing): which cache image each use of
// the image at that address resolves to ([img-trace] as render target, as texture or storage
// image, and every copy into or out of it), with the guest flip it happened in. A pass that
// reads or writes another cache image than the one the previous pass wrote works on stale or
// lost data. At most 400 lines a flip.
namespace Libs::Graphics::ImgTrace {

inline uint64_t Address() {
	static const uint64_t address = [] {
		const char* value = std::getenv("KYTY_IMG_TRACE");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : uint64_t {0};
	}();
	return address;
}

uint64_t CurrentFlip();

inline void Log(const char* format, ...) {
	static std::atomic<uint64_t> flip_seen {0};
	static std::atomic<uint32_t> lines {0};
	const auto                   flip = CurrentFlip();
	if (flip_seen.exchange(flip, std::memory_order_relaxed) != flip) {
		lines.store(0, std::memory_order_relaxed);
	}
	if (lines.fetch_add(1, std::memory_order_relaxed) >= 400) {
		return;
	}
	char    text[512];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	std::fprintf(stderr, "[img-trace] flip=%llu %s\n", static_cast<unsigned long long>(flip), text);
}

inline bool Overlaps(uint64_t address, uint64_t size) {
	const auto watched = Address();
	return watched != 0 && address <= watched && watched < address + size;
}

} // namespace Libs::Graphics::ImgTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMGTRACE_H_
