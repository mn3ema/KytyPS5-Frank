# KytyPS5-Frank

> **Experimental, no support promised.** This is a personal test build. It may crash, and issues may go unanswered.

Hello, this emulator is built with Claude using the open source code from various emulators (I'm not a developer, just trying to help).
I'm testing it only on Marvel's Wolverine for now, the tests are all done on my laptop which has a Core Ultra 9 275HX - RTX 5090 Laptop - 64GB DDR5-5600.
I'm sharing this for anyone who wants to try it and add to it, if you can help us with the issues we're facing that would be awesome.
This is where it stands right now:

## Where it stands (November 2026, v0.4)

**Game:** Marvel's Wolverine (PPSA03671, version 01.001.005).

| Scene (RTX 5090 Laptop, Performance mode) | Typical FPS | Lowest 1-second average |
|---|---|---|
| Title screen (with a save) | 21–25 | 21 |
| Opening cinematic, first shot | 49–53 | 49 |
| Opening cinematic, the rest | 28–38 | 9 (at scene cuts) |
| First gameplay, Trask Outpost (jungle at night, rain) | 18–21 | 10 |

Measured with v0.4 on the test laptop (new game, first start with the shipped shader list). The jungle gameplay is the limit right now: the emulator's GPU thread is at its limit there, and that is what the next versions go after.

v0.3 measured other spots (see "New in v0.3" below); the closest match, grass / jungle standing, was 19 FPS.

Target: 60 FPS in Performance mode, 30 in Fidelity.

### New in v0.4

- **Shader compile stutter mostly gone, even on a first play.** The download ships a shader list for Wolverine; the first start builds it before the title screen ("Preparing pipelines", about 45–50 minutes on the test laptop, resumable; Esc or Cross skips it, with a warning that the game may stutter). In a 10-minute cold run of the opening and first gameplay, 7 shaders still compiled during play, a few milliseconds each (875 in a 40-minute session before). A restart compiles nothing.
- **Graphics pipeline libraries**: one piece per shader stage, prepared ahead and linked in microseconds; fewer shader variants (more state passed per draw); caches carried over between builds.
- **Correct aspect ratio** (black bars instead of a stretched picture) and **borderless full screen** with F11 or Alt+Enter.
- **Fixed:** the "device lost" crash every 20–40 minutes, holes in Wolverine's skin, flickering decals/dirt/cracks, black silhouettes after ~25 minutes, exploded character bodies, an upgrade-menu crash, a crash after ~45 minutes, the PlayGo stop at "Reached end of initial game content installed".
- **Faster:** multi-draw replay working again, readbacks on the copy engine (+15%), merged barriers and other GPU-thread savings.

The full list is in the [v0.4 release notes](../../releases/tag/v0.4).

### Gameplay video (v0.3)

[![Marvel's Wolverine on PC – PS5 Emulator (KytyPS5-Frank) – RTX 5090 Laptop Gameplay](https://img.youtube.com/vi/kV5F9csrAUk/hqdefault.jpg)](https://youtu.be/kV5F9csrAUk)

About 11 minutes of gameplay with v0.3 on the RTX 5090 Laptop (FPS in the window title). It ends with one of the "device lost" crashes listed under known issues.

### Screenshots

From v0.2 (FPS is in each window title, Performance mode, RTX 5090 Laptop). Click a picture for full size.

| | |
|---|---|
| ![Title screen, 28 FPS](docs/screenshots/frank/01_title.jpg) | ![Cutscene, 17 FPS](docs/screenshots/frank/02_cutscene.jpg) |
| Title screen, 28 FPS (the red dots are the particle bug below) | Cutscene, 17 FPS |
| ![Cutscene, 20 FPS](docs/screenshots/frank/03_cutscene.jpg) | ![Jungle gameplay, 16 FPS](docs/screenshots/frank/04_jungle.jpg) |
| Cutscene, 20 FPS | Jungle gameplay, 16 FPS |
| ![Jungle gameplay, 16 FPS](docs/screenshots/frank/05_jungle_cliff.jpg) | ![Climbing, 25 FPS](docs/screenshots/frank/06_climbing.jpg) |
| Jungle gameplay, 16 FPS | Climbing, 25 FPS |

**Working:** boots, menus, cutscenes and gameplay, with correct terrain, foliage, lighting and ray-traced passes.

### New in v0.3 (October 21)

- **Faster, most of all when turning the camera.** The game's call shaders (shaders that call functions through per-object tables) were re-resolved on every dispatch: the reuse window (50 ms) was shorter than one frame at 18 FPS, and a shader that can't be compiled was re-processed for every new object only to be refused again. Turning the camera on the first rooftop went from ~13 to ~25 FPS.
- **Host zones and sync skip on by default.** The memory ranges the game writes every frame get a buffer shared with the GPU instead of a page fault per write, and repeated buffer syncs with nothing changed are skipped. Standing FPS up 7–18% depending on the scene. `KYTY_HOST_ZONES=0` / `KYTY_SYNC_SKIP=0` turn them off.
- **Freeze when loading some saves fixed** (the sewer area). The NVIDIA driver never finishes compiling one pipeline there (VS `d1ea70c7117d4d3a`, PS `85892b3b3d0a61ba`). Compiles now run on worker threads with a time limit, and a pipeline that compiles for 30 s is listed in `_ShaderDump/slow_pipelines.txt` and its draws are skipped from then on. Side effect: the blue scent trail in that area doesn't draw yet.
- **Safety net for one kind of GPU hang.** Indirect dispatch sizes are checked on the GPU; a size past the device limits or 4M workgroups becomes an empty dispatch (`[indirect-clamp]` in the log) instead of a job that never ends.
- **Video memory plan follows your card's own VRAM** instead of assuming 16 GB (`KYTY_VRAM_TARGET_MB=14336` restores the old plan).
- **Write-only depth targets** no longer stop the emulator.
- **Diagnostics** (all off unless asked for): the GPU-hang report names the draws and dispatches the GPU was stuck on, a freeze recorder samples busy threads when no frame came for 15 s, and a sampling profiler (`KYTY_SAMPLE`).

### New in v0.2

- **One cause of the GPU hang ~20 s into gameplay fixed.** Two compute shaders (`06544f382c6b0850`, `3378b4a64461d6ca`) overlapped when their barriers were left out; they keep them now (0 hangs in 20 boots with the fix, same FPS). The other rare GPU crash (a shader reading outside any buffer) is still open: in a v0.1/v0.2 comparison on the test laptop, each build crashed once in three runs.
- **Red squares and streaks instead of particles fixed** (title screen, rain, embers). GS user-data slot 28 is zeroed for the particle shader `4efbcf79c36d8adf` (see the old known issue below for the details). `KYTY_PARTICLE_SLOT28=0` turns it off.
- **AMD: black window fixed.** AMD's Windows driver offers Mailbox presentation but shows nothing with it; on AMD GPUs the build now presents with Fifo (found on an RX 7900 XT). `KYTY_AMD_MAILBOX=1` keeps Mailbox.
- **Depth-to-color copies** (from upstream KytyPS5). An AMD tester's build stopped with "unsupported depth register state" at the Sony logo; this may fix it, and if not, the log now names the exact setting. Please send your `kyty_log.txt` if you still hit it.

### What this build changes compared with KytyPS5

Visual fixes:
- **Terrain rendered as flat sand.** Bindless samplers lost their texture key and all used sampler record 0 (point filtering, clamped). The key is now kept.
- **Particle masks missing.** The particle shader samples 2D-array textures through 3D image instructions, as the hardware allows. They were rejected and replaced by a blank texture, so particles drew as squares. They are now read as 2D arrays.
- **Wrong placeholder textures** after bindless slots were freed.

Speed:
- **Hardware denormal flushing** instead of software flushes in every float operation. The heaviest pixel shader went from 5.9 ms to 0.96 ms per frame.
- **No barriers between compute dispatches the game doesn't order**, and no NVIDIA crash-marker per draw. Title screen ~10 → ~20 FPS.
- **Faster tracking of GPU-written memory** (pages processed in runs). Title ~19.5 → ~23, gameplay ~15 → ~17 FPS.
- **Ray-traversal shader loop cap** (dc76e1223a9bf673) so runaway rays stop early. Gameplay ~7 → ~13 FPS at the time.
- **GPU-sized buffers, resource-table reads and shader function expansion** reworked to avoid waiting on the GPU and to reuse work. Function expansion alone was 8% of the command thread.

Stability:
- **Render-thread crash in streaming scenes fixed.** The game asks the memory manager (AMM) to map memory and writes into it shortly after. Each map took ~0.35 ms here, because our table of memory ranges was copied and re-sorted on every change. Batches queued up for half a second, and the game wrote into memory that wasn't mapped yet. The table is now an ordered map, and each batch's unmaps go to the GPU thread in one trip.
- **Fewer stalls when the game frees memory in batches.** Each unmap in a batch made its own round trip to the GPU thread (and its own GPU submission). They now go in one trip per batch: a 512-unmap batch went from 58 ms to about 7 ms.
- **Game threads no longer wait on buffer copies.** The GPU thread held the memory locks while it copied whole buffers from guest memory, up to 64 ms, and the game's main, render and job threads waited on those locks. The copy now runs after the locks are released.
- **BVH refit "device lost" fixed.** It came from reads of data still being downloaded.
- **Many diagnostics, all off by default:** frame timeline, GPU timing, crash reports that name the guest memory mapping, shader probes.

## Known issues and caveats

- **Tested only on Marvel's Wolverine, and only on an NVIDIA RTX 5090 Laptop.** Nothing new for AMD in v0.4. Intel GPUs and other games are untested.
- **Short hitches (100–250 ms) when a new area or a lot of new content appears**, also on a restart. They come from the emulator's GPU thread being at its limit, not from shader compiles.
- **First launch is long** and writes a large cache next to the emulator (about 25 GB); caches are per GPU and driver, so a driver update means a new warm-up. Set the NVIDIA shader cache size to Unlimited.
- **A rare GPU hang** in a chain of compute shaders is still open (seen once in a 16-minute session).
- **Cards with 16 GB or less** will run with texture trimming much of the time (the jungle gameplay uses 15–17 GB here).
- **Occasional freeze in some scene changes** (reported in v0.3, for example the drop-off scene; frame counter stuck). Not seen in the v0.4 test runs, but not confirmed fixed. Restarting gets past it.
- **The blue scent trail in the sewer area doesn't draw** (its pipeline never finishes compiling on NVIDIA and is skipped).
- **Red squares and streaks instead of some particles** are worked around since v0.2 (particle shader `4efbcf79c36d8adf`, GS user-data slot 28); the root cause is still open.
- **The download includes a shader list made from the game** (`_ShaderList`, shader code taken from Wolverine). You still need your own legally dumped copy of the game; no game files are otherwise provided or linked.

## Where we're struggling, and what's next

Measured in the jungle gameplay on the test laptop:
- **The GPU now takes about 34–36 ms per frame**, and the emulator's GPU thread (which walks the game's command stream in order) is right behind it at 25–32 ms. Both have to drop for 60 FPS.
- Of the GPU frame, about 11 ms is gaps between the game's work (barriers, copies, sync), about 18 ms is draws and about 5 ms compute. One pixel shader alone takes about 7 ms: the emulator holds the GPU's draw-order lock for the whole shader, where the game needs it only for its last few percent.
- When a new area appears, the GPU thread is already near its limit, so the extra work for the new content shows up as 100–250 ms frames.

**Next (v0.5):**
1. Hold the draw-order lock only where the game asks for it (the 7 ms shader).
2. Drop barriers between compute passes that don't depend on each other.
3. Name and fix the work behind the 100 ms+ frames (v0.4 already logs it with `KYTY_HITCH_COST_MS=100`).
4. Prepared shaders for parts of the game nobody has recorded yet.

## How to run

Needs 64-bit Windows 10/11, a CPU with AVX2 (Intel Haswell from 2013 or AMD Zen, and newer) and a Vulkan 1.3 GPU with a recent driver.

1. Download the latest `KytyPS5-Frank-...-win64.zip` from the [Releases](../../releases) page and unzip it to a folder on a drive with plenty of free space (the first start writes a large cache next to the emulator).
2. You need your own legally dumped copy of the game, version 01.001.005. No games or system files are provided or linked.
3. Open `Play.bat` in a text editor (right-click > Edit) and set `GAME` to your game folder, the one that contains `eboot.bin`, for example `set GAME=D:\Games\Wolverine`. If you used v0.3, delete its `_ShaderCache` and `_PipelineCache` folders.
4. Double-click `Play.bat`. The first start shows "Preparing pipelines" for a long time (about 45–50 minutes on the test laptop); leave it running and the game starts by itself when it's done. To play at once instead, press Esc or Cross and confirm with Y / Circle (the game may stutter until the rest is prepared in the background).
5. Later starts take under a minute. The log goes to `kyty_log.txt` next to `Play.bat`; please attach it when you report a problem.

Settings are environment variables: to change one, add a line such as `set KYTY_KEEP_ASPECT=0` to `Play.bat` before the line that starts `kyty_emulator.exe`. `launcher.exe`, the KytyPS5 launcher, works too, but it doesn't restart a first start that a driver crash cut off.

### Controls

A game controller should work too (only the keyboard was tested for this release). The keyboard is mapped like this:

| PS5 button | Key | | PS5 button | Key |
|---|---|---|---|---|
| Cross | J | | L1 / R1 | Q / E |
| Circle | L | | L2 / R2 | Z / C |
| Square | K | | L3 / R3 | Left Shift / Left Ctrl |
| Triangle | I | | Options | Enter |
| D-pad | Arrow keys | | Touchpad (left / right) | Backspace / Tab |
| Left stick | W A S D | | Right stick | T F G H |

F7 uses the mouse as the right stick (F7 again releases it). F11 or Alt+Enter: full screen.

The settings tuned for Wolverine are built in. To start with plain KytyPS5 settings, set the environment variable `KYTY_FRANK_DEFAULTS=0`.

### Building from source

Same as KytyPS5 (see their build instructions); this repository is a fork of it. The tuned changes are on the default branch. On Windows, `build-release.bat` (tuned for your own CPU) and `build-portable.bat` (any AVX2 CPU, used for the download) build with Visual Studio Build Tools, clang-cl and Ninja; set `KYTY_QT_DIR` (Qt 6) and `KYTY_GLSLANG` (glslangValidator) first.

## Credits

- [KytyPS5](https://github.com/KytyPS5/KytyPS5): the base emulator.
- AnyPS5 and the Jetsku fork of KytyPS5: parts of the graphics work.
- Changes in this build were made by Claude (Anthropic) and tested by Mahmoud Neama.

Licensed under GPL-2.0, like KytyPS5.

Not affiliated with Sony Interactive Entertainment or PlayStation.

---

The original KytyPS5 README is in [README-KytyPS5.md](README-KytyPS5.md).
