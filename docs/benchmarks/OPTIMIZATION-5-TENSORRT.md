# Optimization 5: ONNX Runtime TensorRT backend

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20

## Decision

**Promote TensorRT active-model-only pool 1 for production.** The objective is
minimum safe page latency with stable memory, not maximum concurrent Runs.
Against the same 1660×1400 input, lossless grayscale PNG compression 3, and
fresh-process harness, warmed TensorRT at the selected 832/64 policy averaged
3.486 s.
CUDA 320/32/8 averaged about 5.55 s before its VRAM staircase crossed the
10 GiB watchdog on page 17. TensorRT completed all 21 uncached requests and
plateaued around 1.4 GiB.

Production settings are `ExecutionProvider=TensorRt`, `GpuParallelism=1`,
`TensorRtSessionPoolSize=1`, core 832, context 64, TensorRT profile
min/opt/max 8/960/960, `PreloadOnStartup=true`, and PNG compression 3. CUDA
remains supported as an explicit configuration, but is not the measured
production winner on this host.

All results below are measured on the target host unless marked fake-backed.
Artifacts are retained under `/devtmp/komascaler-acceptance/tensorrt`.

## Host, libraries, and provider placement

```bash
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
  sh scripts/engines/tensorrt-preflight.sh
```

| Component | Measured value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060, compute 8.6, 12,288 MiB |
| Driver | 595.58.03 |
| CUDA / cuDNN | 12.9 / 9.24.0.43 |
| TensorRT | 10.14.1.48-1+cuda12.9 |
| ONNX Runtime GPU | 1.26.0 |
| .NET SDK / libvips | 10.0.302 / 8.16.1 |
| Host RAM | 15,941 MiB, no swap |
| Acceptance filesystem | `/devtmp`, btrfs, not tmpfs |

`ldd` reported no missing library. ORT exposed
`TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`.
Profiling assigned the fused convolutional graph to one TensorRT node with
zero CUDA fallback, zero CPU nodes, and zero CPU compute nodes. The cold
single-model engine build took 47.79 s and peaked at 4,051 MiB. The final
cached preflight passed in 5.29 s; inference was 152.610 ms, in-test VRAM was
1,371 MiB, minimum available RAM was 8,978 MiB, and final VRAM was 123 MiB.
Profiles now write below `/devtmp`, not the working tree.

The first cold build warned that no timing cache existed, as expected.
TensorRT logged that `NVIDIA_TF32_OVERRIDE=0` disabled its builder TF32 flag.
There were no provider-loading errors.

The initial sequential creation of all seven cold engines took 286.88 s and
sampled 4,821 MiB peak. That result and the earlier 16.433-second cached
all-seven validation used the superseded opt/max 384/704 profile; neither is
production-960 evidence. The exact production-profile replacement is recorded
in "Production engine-cache deployment gate" below.

## Exact provider policy

TensorRT registers before CUDA; CPU remains available only for shape/control
work. Readiness rejects zero TensorRT nodes or CPU `Conv`, `Gemm`, or `MatMul`,
so registration alone and wholesale CUDA fallback cannot pass.

| ORT 1.26 option | Value |
| --- | --- |
| FP16 / INT8 / TF32 | off / off / off |
| engine and timing caches | enabled |
| sequential engine build | enabled |
| auxiliary streams | 0 |
| context-memory sharing | enabled |
| builder workspace | 2,147,483,648 bytes |
| min / opt / max input | configurable; production 1×3×8×8 / 1×3×960×960 / 1×3×960×960 |

The former 704 maximum covered the initial profile. The accepted production
profile is exactly 960 for its full interior tile extent. Builder workspace
controls tactic-build scratch space; it is not a runtime VRAM cap.

Engine identity includes model SHA-256, ORT and expected TensorRT versions,
precision/TF32 policy, dynamic profile, workspace, auxiliary streams, context
sharing, and timing-cache policy. Pool size is not engine-affecting and is
excluded from engine identity. Provider and pool size remain in application
result-cache identity, so CUDA and TensorRT encoded results never collide.

## Active-model-only lifecycle

The initial eager prototype kept one TensorRT session for each of seven models
and consumed about 8.74 GiB. The retained lifecycle instead keeps persistent
engine files for all models but only one routed model pool resident:

1. startup builds/loads and profiles each engine sequentially;
2. the final warmup pool is disposed, leaving no active model;
3. a request loads its selected model from the persistent engine;
4. a route change holds the lifecycle lock and drains every GPU permit;
5. the old pool is fully disposed before the replacement is constructed;
6. the replacement is atomically published and retained until idle or switch.

Admission rechecks the active model after obtaining a permit, preventing a
waiter from running against a replaced pool. Fault cleanup, cancellation,
idle unload, and shutdown use the same full-permit drain. Old and new model
pools cannot overlap, partial drain cancellation releases acquired permits,
and double disposal remains safe. CUDA keeps its existing all-seven-session
policy.

The alternating sequence 1200→1200→1300→1200→1300 measured:

| Operation | Measured latency / VRAM |
| --- | --- |
| first cached 1200 load + 320×320 run | 3.217 s, 1,371 MiB |
| repeated same-model run | 121.087 ms, 1,371 MiB |
| cached model switches + run | 2.231–2.319 s |
| switch sampled peak | 1,371 MiB |
| idle drain / process final | 123 / 123 MiB |

The 100 ms trace repeatedly fell to 115–123 MiB between models before the new
pool rose to 1,371 MiB. This proves old model memory was released before new
construction. Minimum available RAM was 9,060 MiB.

The dedicated idle test measured 1,361 MiB loaded, 123 MiB after idle disposal,
a 2.337 s cached reload, and 123 MiB after process exit.

All seven routes then passed exact selection, loading, finite output, and exact
2× dimensions. Cached switch-plus-8×8-run times were 2.185–2.307 s; the already
active 1200 route was 11.107 ms. Peak was 1,363 MiB and final VRAM 125 MiB.

## Pool-size result

| Pool maximum | Residency strategy | Peak | Concurrent Runs proven | Decision |
| ---: | --- | ---: | ---: | --- |
| 1 | active model only | 1,435 MiB sustained | 1 | production |
| 2 | prior all-seven base + lazy selected expansion | 10,003 MiB | no | rejected prototype |
| 4 / 6 / 8 | not run | not measured | not measured | stopped for safety |

Pool 2 was not retried after active-only pool 1: pool 1 already beat CUDA
end-to-end, while the prior pool-2 experiment failed to prove two active Runs.
Extra contexts are unnecessary complexity unless a future isolated test proves
both concurrency and meaningful page-latency benefit.

## Numerical correctness separated by cause

Three distinct comparisons were made:

| Comparison | Maximum error | MAE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: |
| identical 320×320 single tile, CUDA FP32 vs TensorRT FP32 | 1.997e-6 float | 2.008e-7 float | 130.873 dB | 0.999999997 after byte quantization |
| identical complete 640×384 tiled 320/32 CUDA vs TensorRT | 1 byte | 0.0000366 byte | 92.494 dB | 0.999999998 |
| TensorRT tiled vs untiled CUDA reference, 640×384 | 48 bytes | 0.2843 byte | 46.190 dB | 0.999901 |

The first row isolates provider tactics before blending/quantization. The
second isolates provider contribution after identical tiling and blending. The
third intentionally includes tiling/context boundary error. TensorRT’s own
contribution is negligible; the larger third-row error is tiling, not the EP.

The tiled-versus-untiled matrix also covered 384×640 and 640×640 high-frequency
grayscale inputs, plus core 512/576 and contexts 32/48/64. The initial 320/32
whole-image PSNR was 46.19–49.38 dB, SSIM 0.999901–0.999952, and seam MAE
0.808–1.285 bytes. Full max/MAE/RMSE/PSNR/SSIM and seam results are in
`/devtmp/komascaler-acceptance/tensorrt/incremental/pool-1-quality/quality.json`.
Color pages remain an exact-byte application bypass and never enter inference.

## Direct sustained HTTP comparison

Both providers used a fresh process, identical 1660×1400 single-band input,
30 baseline 320/32 tiles, PNG compression 3, the same publish output, 200 ms
sampling, and distinct uncached inputs.

```bash
# CUDA baseline
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
KOMASCALER_INCLUDE_CUDA_BASELINE=0 KOMASCALER_CUDA_SUSTAINED=1 \
KOMASCALER_INCLUDE_TENSORRT=0 sh scripts/benchmarks/tensorrt-benchmark.sh

# Active-model-only TensorRT
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
KOMASCALER_TENSORRT_FACTORS=1 KOMASCALER_TENSORRT_SUSTAINED_FACTOR=1 \
KOMASCALER_INCLUDE_CUDA_BASELINE=0 KOMASCALER_INCLUDE_TENSORRT=1 \
  sh scripts/benchmarks/tensorrt-benchmark.sh
```

| Result | CUDA 320/32/8 | TensorRT active-only pool 1 |
| --- | ---: | ---: |
| completed uncached pages | 17/21, watchdog stop | 21/21 |
| HTTP result | 17 `upscaled` | 21 `upscaled` |
| warmed mean HTTP | about 5.55 s | 4.267 s |
| first request | 5.682 s | 6.679 s including selected-model load |
| typical inference stage | about 5.25 s wall / 36.3 s aggregate | about 3.94 s |
| encoded size | 4,722,886 bytes | 4,722,882 bytes |
| post-page VRAM | 5,295→7,343→8,389→9,423→10,375 MiB | 1,371–1,427 MiB |
| sampled peak | 10,447 MiB | 1,435 MiB |
| minimum available RAM | 7,176 MiB | 8,342 MiB |
| post-warmup slope | continuing staircase; run terminated | +1.439 MiB/page overall; −1.018 MiB/page over last 10 |
| final VRAM | released on termination | 3 MiB |
| cache hit | run stopped before check | 68.601 ms, byte-identical |

TensorRT’s small initial allocator growth reached a plateau; its last-ten-page
slope is negative and its range is only 56 MiB. CUDA crossed the 10 GiB
watchdog before completing the required 20 post-warmup pages. An experimental
dedicated-thread CUDA dispatcher was also rejected after it forced eight Runs
aggressively, reached 11,429 MiB on page 2, and caused an ORT allocation fault;
production retains the previously accepted bounded `Task.Run` dispatcher.

## Verification and production gate

Real GPU coverage passed provider placement, all seven routes, exact 2× output,
320/32 baseline tiling, numerical isolation, same-model reuse, alternating model
switches, idle unload/reload, double disposal, process-exit release, 21-page
sustained HTTP, and cache identity/hit behavior. Ordinary native-independent
tests cover bounded concurrency, deterministic blending, one-image queueing,
fault handling, drain cancellation, and idempotence. No native fault was
deliberately injected into the stable target driver.

The reusable scripts publish once under `/devtmp`, preserve persistent engines
and failure logs, run fresh service processes, enforce service/HTTP timeouts,
sample RAM/VRAM, and terminate above 10 GiB VRAM or below 3 GiB available RAM.
Preflight clears stale status and captures `readelf`/`ldd` loader evidence.

TensorRT satisfies the revised promotion rule:

- lower measured warmed HTTP latency than CUDA+PNG;
- negligible isolated numerical difference;
- stable sustained memory with large headroom;
- safe model switching, idle disposal, and process release;
- proven TensorRT compute with no CUDA or CPU fallback;
- every request remains well inside the 28-second deadline.

The exact 960 acceptance cache is complete. Deployment must still ensure
`/var/cache/komascaler/tensorrt` is writable by `komascaler`; startup then
builds any identity-missing engines before readiness rather than inside HTTP.

## TensorRT tile-limit revisit

The CUDA-derived 320-core limit was not carried forward. Active-model-only
TensorRT pool 1 was measured sequentially at each requested policy. All HTTP
cases used the same 1660×1400 representative page, fresh uncached cache keys,
PNG compression 3, one active session, one Run at a time, and 200 ms RAM/VRAM
sampling. The 576/64 case ran first against the existing 704 profile; only
after it passed were separate 832 and 960 profiles built.

```bash
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
KOMASCALER_TILE_CORE=832 KOMASCALER_TILE_CONTEXT=64 \
KOMASCALER_TENSORRT_PROFILE_OPTIMUM=960 \
KOMASCALER_TENSORRT_PROFILE_MAXIMUM=960 \
KOMASCALER_PRELOAD_ON_STARTUP=false KOMASCALER_SMOKE_PAGES=1 \
KOMASCALER_TENSORRT_FACTORS=1 KOMASCALER_TENSORRT_SUSTAINED_FACTOR=1 \
KOMASCALER_INCLUDE_CUDA_BASELINE=0 KOMASCALER_INCLUDE_TENSORRT=1 \
  sh scripts/benchmarks/tensorrt-benchmark.sh
```

The same command was repeated with each core/context/profile tuple. Measured
HTTP and resource results were:

Raw runs are retained at `tensorrt/runs/20260720T194016Z-3779013` (320),
`20260720T193528Z-3770910` (512), `20260720T191300Z-3703158` (576),
`20260720T192336Z-3731125` (704), and `20260720T193013Z-3754790` (832), all
relative to `/devtmp/komascaler-acceptance`.

| core/context | extent | tiles | first page, s | warmed mean, s | Run mean (range), ms | peak VRAM | slope, MiB/page | min available RAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320/32 | 384 | 30 | 7.401 | 4.281 | 128.022 (114.100–146.326) | 1,423 MiB | +1.728 | 8,642 MiB |
| 512/64 | 640 | 12 | 7.564 | 4.359 | 325.769 (291.287–379.245) | 1,433 MiB | +0.988 | 8,779 MiB |
| 576/64 | 704 | 9 | 7.348 | 4.099 | 406.043 (374.115–464.302) | 1,477 MiB | +1.325 | 8,634 MiB |
| 704/64 | 832 | 6 | 6.963 | 3.781 | 555.273 (531.776–591.174) | 1,885 MiB | −0.039 | 8,671 MiB |
| 832/64 | 960 | 4 | 6.724 | **3.486** | 771.015 (763.800–797.895) | 2,401 MiB | +1.374 | 8,670 MiB |

Each sustained row is one warmup request plus 20 distinct uncached pages. The
slopes are tiny allocator noise over only 20 pages, not a staircase: the final
832 profile varied within a narrow warmed band and remained 7.6 GiB below the
watchdog. Every request returned `upscaled`; exact tile counts were logged by
the conversion stage. The 512 candidate is slower than 576 despite fewer
tiles, demonstrating why the largest or fewest-tile policy was not selected by
assumption.

Engine construction and cached selected-model startup were measured
separately:

| profile / candidates | cold engine build | cold-build peak | cached selected-model first page | cached runtime peak |
| --- | ---: | ---: | ---: | ---: |
| opt 384, max 704 / 320, 512, 576 | 47.790 s | 4,051 MiB | 7.348–7.564 s | 1,477 MiB |
| opt=max 832 / 704 | 76.330 s | 5,587 MiB | 6.963 s | 1,885 MiB |
| opt=max 960 / 832 | 89.293 s | 7,379 MiB | 6.724 s | 2,401 MiB |

Cold builds are deployment/prewarm work and must not occur inside a production
request deadline. Persistent engine identity includes min/opt/max, so neither
expanded profile can load an incompatible 704 engine. Cached normal startup
and page processing remain well inside the 28-second deadline.

Quality used the same deterministic 960×640 high-frequency grayscale input and
untiled CUDA FP32 output as reference. Metrics are after channel-zero byte
quantization; the seam band is eight output pixels wide.

| core/context | quality tiles | whole PSNR | whole SSIM | seam MAE |
| ---: | ---: | ---: | ---: | ---: |
| 320/32 | 6 | 49.6569 dB | 0.999955 | 0.809199 byte |
| 512/64 | 4 | 60.5135 dB | 0.999996 | 0.039456 byte |
| 576/64 | 4 | 60.5135 dB | 0.999996 | 0.039456 byte |
| 704/64 | 2 | 63.5641 dB | 0.999998 | 0.016458 byte |
| 832/64 | 2 | 63.5641 dB | 0.999998 | 0.016458 byte |

The larger 64-pixel context materially reduces seam error. Provider-only
numerical differences remain those in the earlier separated comparison and
are negligible; this table primarily measures tiling/context error.

Lifecycle was repeated per engine-profile family. The 704-profile family had
already proven old-pool release between switches. At extent 832, cached
switch-plus-run was 3.00–3.06 s, peak residency 1,865 MiB, and idle drain
returned to 149 MiB. At extent 960 it was 3.28–3.29 s, peak residency 2,409
MiB, and idle drain again returned to 149 MiB. First construction of the
second model took 75.639 s and 88.753 s respectively; these were cold engine
builds, not cached switch latency. Old and new pools never overlapped, and
process exit released native residency.

### Revised production knee

832/64 is retained because it is the minimum-latency tested policy, not merely
because it is the largest. Relative to 704/64 it saves 0.295 s/page (7.8%),
keeps the same excellent reference quality, and costs about 516 MiB additional
warmed VRAM while leaving roughly 7.6 GiB beneath the safety cutoff. The
512→576 non-monotonic result and measured cold-build cost were explicitly
considered. A still-larger profile is not inferred safe or faster and is not
promoted without another isolated sustained and quality run.

## Production engine-cache deployment gate

This section supersedes every earlier all-seven startup/routing number for the
production decision. Before this gate, only models 1200 and 1300 had been
exercised with opt/max 960 during profile-specific lifecycle tests. The earlier
seven-model 286.88-second cold build, 16.433-second cached startup, and
2.185–2.307-second routed switches all used opt/max 384/704.

The production cache was generated on disk, sequentially, with one active
session and one Run at a time:

```bash
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
KOMASCALER_TENSORRT_CACHE=/devtmp/komascaler-acceptance/tensorrt/production-engines-960 \
  sh scripts/engines/tensorrt-production-cache.sh
```

The script publishes once, constructs each missing engine in manifest order,
waits for application readiness, stops the service, verifies and hashes all
seven identity directories, then launches a fresh process for cached all-seven
routing. It samples every 200 ms and terminates above 10 GiB VRAM or below
3 GiB available RAM.

| Exact 8/960/960 cold-cache result | Measured value |
| --- | ---: |
| all-seven time to readiness | 605.982 s |
| application sequential-prewarm log | 604.243 s |
| peak VRAM | 7,383 MiB |
| minimum available host RAM | 8,521 MiB |
| engine count | 7 |
| engine bytes | 136,736,484–137,292,500 each |
| VRAM after service termination | 3 MiB |

Per-model cold session-construction times were 88.565, 85.307, 85.240,
85.519, 86.426, 86.119, and 85.203 seconds in model order 1200 through 2048.
The trace drops between models and old/new pools do not overlap.

The exact inventory, including model hashes, identity directories, engine
filenames, byte lengths, and engine SHA-256 values, is retained in
`/devtmp/komascaler-acceptance/tensorrt/production-cache/20260720T200035Z-3811435/engine-inventory.csv`.
The seven identity directories are:

| model | production identity directory |
| --- | --- |
| 1200 | `509e085e42a472fe71d5ab5865d5e6bb9a81addd6888702c821f9b791fa1df2e` |
| 1300 | `9d0224e0dd6182e9b5346f8267634fe7512877fa34bcaa0aa9b47d234fa3e112` |
| 1400 | `a4e4eb62cee189cf40c60393c500a2445c2d69f46a15c3b0f263ae1c147200cc` |
| 1500 | `a566695a8dd45ee814632db901457b8b27c700ce111bc6739b315a1a354687e1` |
| 1600 | `bb83a9aa1997b8434349c1f1cbe50e8ded905fa70e34ecd53be9424a2c372520` |
| 1920 | `3c6c4a2fda3d252be80b43249c1910544ac95a994751c8d25af99d3b563d92c5` |
| 2048 | `70d752fb2d6999f8503dfac1c5e1c195687c26b5e2a46f861c8fc7d24e739709` |

A fresh cached process first preloaded all seven exact identities, disposed the
last warm pool, and then passed route heights 1200, 1251, 1351, 1451, 1551,
1761, and 1985 with the expected model each time:

| model | cached switch + 8×8 Run | residency |
| --- | ---: | ---: |
| 1200, already active | 11.004 ms | 2,299 MiB |
| 1300 | 2.345 s | 2,299 MiB |
| 1400 | 2.324 s | 2,299 MiB |
| 1500 | 2.310 s | 2,299 MiB |
| 1600 | 2.298 s | 2,299 MiB |
| 1920 | 2.338 s | 2,299 MiB |
| 2048 | 2.335 s | 2,299 MiB |

The routed process peaked at 2,299 MiB, retained at least 8,534 MiB available
RAM, returned VRAM to 3 MiB after exit, produced finite exact-2× outputs, and
preserved the CUDA/TensorRT parity tolerance. Its total test time was 36.971 s.

### Enforced production policy

Production uses `PreloadOnStartup=true`. Its meaning is strict in TensorRT
mode: before Kestrel accepts HTTP, the application sequentially creates/loads,
runs, and placement-validates all seven engines for the configured immutable
identity. It verifies that all seven `.engine` files exist, disposes the final
pool, and only then proceeds to HTTP readiness. `PreloadOnStartup=false` is
rejected for production TensorRT configuration.

After successful startup, active-model routing may only load an existing exact
identity. If an engine is deleted or unavailable after readiness, the service
faults and stops with exit code 70 for a clean systemd restart; it never starts
an approximately 89-second engine build inside the HTTP request. The restart
rebuilds the complete missing identity before readiness.

This was exercised against the completed cache by hiding the 1200 engine after
all-seven prewarm. The request was rejected in 3.673 ms, state became
`Faulted`, no engine build began, the file was restored in `finally`, all seven
engine files remained present, and post-test VRAM was 3 MiB.

`TimeoutStartSec=15min` and the readiness probe's 840-second retry window cover
the measured 10.10-minute empty-cache deployment. Readiness remains
`/health/ready`; there is no `/ready` alias.

An engine must be rebuilt when any identity input changes: ONNX model content
hash, ONNX Runtime version, expected TensorRT version, min/opt/max profile,
precision policy, TF32 policy, workspace, auxiliary-stream policy,
context-memory sharing, or timing-cache policy. Changed identities use new
directories, so incompatible engines are not loaded accidentally. The only
remaining deployment action is ordinary ownership: the installer creates
`/var/cache/komascaler/tensorrt` writable by `komascaler`; operators should
expect the first start after an identity change to take approximately ten
minutes on this host.
