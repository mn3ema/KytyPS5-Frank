#ifndef EMULATOR_SRC_COMMON_GAMESTARTED_H_
#define EMULATOR_SRC_COMMON_GAMESTARTED_H_

#include <atomic>
#include <chrono>
#include <cstdio>

// v0.5 FG (Mac 11-24): the title-screen panels (shader preparation, frame generation hint) stay until
// the player loads a save or starts a new game. The signal, game-neutral: the game writes a file
// inside a mounted save (/savedataN/...) 10 s or more after its first save access. Wolverine (Mac's
// trace, 11-24): at boot it reads settings, the last game save and writes prefs within 0.2 s; the
// menus only read settings; loading a save reads it and writes prefs at once; gameplay autosaves.
// [savedata-trace] lines (stderr, the first 200) show the save calls and file opens with times
// (t = seconds since the first save access).
namespace Common::GameStarted {

inline std::atomic<bool>& Flag() {
	static std::atomic<bool> started {false};
	return started;
}

inline bool Started() {
	return Flag().load(std::memory_order_acquire);
}

inline double Seconds() {
	static const auto start = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

inline void Trace(const char* what, const char* detail, unsigned value) {
	static std::atomic<int> lines {0};
	if (lines.fetch_add(1, std::memory_order_relaxed) < 200) {
		std::fprintf(stderr, "[savedata-trace] t=%.1f %s %s 0x%x%s\n", Seconds(), what, detail != nullptr ? detail : "",
		             value, Started() ? "" : " (title panels up)");
	}
}

// Back at the menus: the panels show again until the next save write (Mac 11-24: "they didn't come
// back when I quit play and returned to menu"). Wolverine reads saves only at the menus (gameplay
// writes autosaves), so a save read after the game started means the player is back at a menu.
// When the panels last came back (seconds), for Note.
inline std::atomic<double>& LastReset() {
	static std::atomic<double> at {-1000.0};
	return at;
}

inline void Reset(const char* why) {
	bool expected = true;
	if (Flag().compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
		LastReset().store(Seconds(), std::memory_order_release);
		std::fprintf(stderr, "[game-started] t=%.1f %s: the title panels show again\n", Seconds(), why);
	}
}

// Mac 11-25: back at the menu Wolverine reads the saves and writes its prefs in the same instant
// (58.4 s: read, read, prefs write), so the panels flashed and went away again. A write within
// 2 s of the panels coming back is part of returning to the menu, not a load or a new game.
inline void Note(const char* why) {
	if (Seconds() - LastReset().load(std::memory_order_acquire) < 2.0) {
		return;
	}
	bool expected = false;
	if (Flag().compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
		std::fprintf(stderr, "[game-started] t=%.1f %s: the title panels hide\n", Seconds(), why);
	}
}

} // namespace Common::GameStarted

#endif // EMULATOR_SRC_COMMON_GAMESTARTED_H_
