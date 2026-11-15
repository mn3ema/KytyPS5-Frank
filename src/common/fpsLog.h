#ifndef EMULATOR_SRC_COMMON_FPSLOG_H_
#define EMULATOR_SRC_COMMON_FPSLOG_H_

#include <cstdlib>
#include "common/fpsLog.h"
#include <cstring>

namespace Common {

// KYTY_FPS_LOG=1: the frame rate ([fps]) plus every subsystem's diagnostics ([page-protect],
// [sync-skip], [draw-prep], ...). With KYTY_FPS_ONLY=1 too, the frame rate only, so a benchmark
// does not pay for the diagnostics (the page-protection stats alone hash every armed page). A
// separate switch rather than a KYTY_FPS_LOG value: the shader disk cache keys on KYTY_FPS_LOG's
// value (the recompiler names it), and a benchmark should use the same cache either way.
inline bool FpsLogDiagnostics() {
	static const bool enabled = [] {
		const char* only = std::getenv("KYTY_FPS_ONLY");
		return std::getenv("KYTY_FPS_LOG") != nullptr &&
		       !(only != nullptr && std::strcmp(only, "0") != 0);
	}();
	return enabled;
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_FPSLOG_H_
