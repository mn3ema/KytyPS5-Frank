#ifndef KYTY_COMMON_TRACE_POINTS_H_
#define KYTY_COMMON_TRACE_POINTS_H_

#include "common/common.h"

namespace Common::TracePoints {

// Guest-code trace points (Windows only, diagnostic).
//
// KYTY_TRACE_PC="<rva>[:<reg>:<off>:<dwords>[:<max>[:<deref off>]]][,...]"  ('/' also separates entries)
// An entry may end in "?<reg>=<hex>" (log only when the register equals the value) or
// "?<reg>:<off>=<hex>" (only when the dword at <reg>+<off> equals it); "~" may replace "=".
// Hits that fail the condition do not count against <max>.
// KYTY_TRACE_DELAY=<seconds> arms the points that long after load.
// An entry may also carry "!<off>" after its fields: its first (passing) hit arms the hardware watch
// (KYTY_WATCH_LEN, KYTY_WATCH_MAX, KYTY_WATCH_RW) on <reg>+<off>.
//
// Each entry plants an int3 at module base + rva. When the game reaches it, one line is written
// to stderr with the thread, the integer registers, xmm0-3 as floats and, when <reg> is given,
// <dwords> dwords read at <reg> + <off> (or, with <deref off>, at *(u64*)(<reg> + <off>) +
// <deref off>). The original instruction then runs (single step) and the int3 is re-armed, until
// the point has fired <max> times (default 200); after that it stays disarmed, so a trace point
// costs nothing once its budget is spent. <reg> is one of rax rbx rcx rdx rsi rdi rbp rsp r8 r9
// r10 r11 r12 r13 r14 r15. All numbers are hex; <off> may be negative ("-1a0").
void Install(uint64_t module_base, uint64_t module_size, const char* module_name);

// A hardware write watchpoint set at run time (first call wins; needs KYTY_WATCH_DRAW or
// KYTY_WATCH_WRITE in the environment so the exception handler is installed at load).
// KYTY_WATCH_VAL=<hex dword> filters the watch: only writes that leave that value count.
// KYTY_WATCH_RW=1 makes the watch trigger on reads as well (DR7 RW=11).
void SetWatch(uint64_t address, uint32_t length, uint32_t max_hits);

} // namespace Common::TracePoints

#endif /* KYTY_COMMON_TRACE_POINTS_H_ */
