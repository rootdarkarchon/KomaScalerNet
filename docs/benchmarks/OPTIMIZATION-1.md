# Optimization 1: cold-path latency

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20
Target: Debian 13, NVIDIA GeForce RTX 3060 12 GiB, CUDA 12, cuDNN 9
Fixed constraints during this historical factor-1 experiment: FP32, TF32
disabled, one GPU `Run`, all seven sessions in one set, 320-pixel maximum tile
core with 32-pixel context, and lossless WebP effort 2.

This document records the factor-1 optimization stage. The later measured
intra-image tile-parallel design in `OPTIMIZATION-2.md` supersedes the
serialized-Run constraint while retaining a single image conversion worker.

## Objective and plan

The acceptance cold conversion took 23.330710 seconds against a 28-second
application deadline. This iteration followed a measurement-first plan:

1. Add wall, tile preparation, inference, blending, normalization, encoding,
   cache-write, individual CUDA `Run`, and managed output-copy timing.
2. Move session construction and first-Run initialization before HTTP
   readiness by pre-warming all seven models at startup.
3. Remove repeated per-pixel trigonometry by calculating the horizontal and
   vertical blend weights once per tile.
4. Experiment with one-tile CPU/GPU lookahead while preserving the serialized
   GPU lane; retain it only if wall time and VRAM improve.
5. Measure output-copy cost to decide whether ONNX Runtime I/O binding and
   reusable shape-keyed buffers are justified.
6. Run ordinary verification, isolated real-GPU acceptance, and repeated real
   HTTP conversions. Retain only changes supported by those results.

All build and test output remained disk-backed under
`/devtmp/komascaler-acceptance`; GPU cases ran sequentially with the existing
180-second per-case timeout and continuous host/GPU sampling.

## Implementation retained

`TiledUpscaler.UpscaleMeasuredAsync` now reports tile count and elapsed wall,
preparation, inference, blending, and normalization time. `ConversionQueue`
adds encoding and cache-write time in one structured information log. Debug
logging on `GpuUpscaler` separates each ONNX Runtime call from the subsequent
managed output copy. These timings contain no image bytes, filenames, or
secrets.

`Models:PreloadOnStartup` defaults to `true`. After inventory validation and
before the HTTP listener starts, the singleton runs an 8×8 input through each
of the seven sessions. This constructs the required complete session set and
performs each model's first CUDA Run. If inventory is not ready, the server
still starts so the established bypass/readiness behavior remains available.
If inventory is valid but CUDA initialization fails, startup fails rather than
silently serving CPU inference.

Blending now computes each tile's one-dimensional X and Y weights once and
uses their product in the pixel loop. The previous implementation evaluated
two half-sine functions for every overlapping output pixel. A deterministic
nearest-neighbor backend regression verifies exact output bytes across tile
boundaries, and a timing regression verifies all stages are populated.

## Measurements

The production HTTP fixture was the same neutral 1660×1400 PNG used during
target acceptance. It routes to the real 1400 model, creates 30 tiles under
the fixed 320/32 policy, and produces an exact 3320×2800 lossless WebP of
4,427,946 bytes.

| variant | HTTP time | tiled wall | encode | peak VRAM | disposition |
|---|---:|---:|---:|---:|---|
| acceptance baseline, cold sessions | 23.330710 s | not instrumented | about 2.7 s in separate benchmark | 1,989 MiB observed after request | superseded |
| pre-warm + weights + experimental lookahead | 20.738010 s | 17.445285 s | 2.966402 s | 3,131 MiB | rejected |
| pre-warm + weights, sequential | 18.284759 s | 15.587869 s | 2.359206 s | 2,061 MiB | retained |
| sequential confirmation with debug timing | 18.582661 s | 15.857641 s | 2.362755 s | not separately sampled | retained |

The two final sequential runs averaged 18.433710 seconds. Relative to the
23.330710-second baseline, the measured reduction is 4.897 seconds, or 21.0%.
The first optimized response is 9.715 seconds below the 28-second application
deadline. A repeated request returned `X-Upscaler-Result: cache` in 87.977 ms
and was byte-identical to the first optimized result. The optimized result was
also byte-identical to the earlier acceptance output.

The final instrumented confirmation broke down as follows:

| stage | measured time | share of 18.583 s HTTP time |
|---|---:|---:|
| 30 serialized tile inferences, including copies | 15.714522 s | 84.6% |
| lossless WebP effort-2 encode | 2.362755 s | 12.7% |
| blending | 85.613 ms | 0.46% |
| normalization | 34.651 ms | 0.19% |
| cache write | 33.242 ms | 0.18% |
| tile input preparation | 17.599 ms | 0.09% |
| remaining HTTP/decode/inspection/key/result overhead | about 334 ms | 1.8% |

Pre-warming all seven models took 9,973.712 ms in the debug run. It shifts
session construction and first-Run costs to managed service startup; it does
not reduce total boot-plus-first-page work. The systemd readiness probe cannot
succeed until this warm-up and HTTP startup complete.

Across the 30 production tiles, managed `ToArray` output copies totaled
37.764 ms, averaged 1.259 ms, and peaked at 2.773 ms. This is 0.24% of the
15.715-second inference stage. ONNX Runtime itself, not the managed copy,
dominates the tile time.

The isolated real-GPU suite executed after the initial implementation and
continued to prove CUDA provider use, all seven models, exact 2× output,
320/32 tiling, serialization, idle reload, and disposal. Maximum observed
active GPU Runs remained exactly one. The experimental pipeline's small
production-profile run was 6.978 seconds versus the earlier 6.571-second
acceptance result; it provided no supporting speed signal.

The final-code isolated rerun passed all six cases. Its 1280×320 320/32 case
took 6.631 seconds, reported 1,989 MiB inside the test (1,477 MiB sampled at
one-second resolution), and produced the exact 2560×640 result. The concurrent
case again reported maximum active GPU Runs = 1. Provider profiling again
recorded 1,035 CUDA nodes and 20 intentional CPU shape/control nodes.

## Rejected routes

The one-tile lookahead experiment started preparation for the next tile while
the current Run was active and blended the completed tile while the next Run
waited for/acquired the existing GPU semaphore. It never allowed two active
GPU Runs. Direct timing showed preparation, blending, and normalization total
only about 128 ms per production page, placing a hard upper bound below 1% on
useful overlap. The experimental page was slower than the retained sequential
variant and its measured VRAM was 1,070 MiB higher. The experiment was removed.

I/O binding and reusable output buffers were not implemented. Eliminating all
managed output copies could save at most the measured 37.764 ms (0.20% of HTTP
wall time) while adding dynamic-shape buffer ownership and idle/fault disposal
complexity. This does not meet a reasonable benefit-to-risk threshold.

Larger tiles are excluded: 320 is the previously established VRAM limit.
Parallel GPU Runs were excluded during this factor-1 experiment; the measured
bounded tile design in `OPTIMIZATION-2.md` supersedes that restriction. FP16
and TF32 still violate production constraints. WebP effort remains 2. cuDNN
exhaustive search and CUDA graph capture were not attempted:
both may retain additional workspace/graph memory, the host previously
experienced severe pressure during acceptance, and the current trace does not
separate kernel compute from launch overhead sufficiently to justify that
risk. They require their own isolated VRAM-capped experiment if revisited.

## Remaining opportunities

Inference is now the clear target. Safe follow-up work should begin with a
CUDA profiler trace for one representative 341×344 tile, using the same hard
timeout and VRAM/RAM sampling. It should distinguish kernel execution,
host/device transfers, and launch gaps before changing provider options.
Potential experiments, one at a time, are:

- compare ORT/cuDNN-supported convolution algorithm settings under an explicit
  GPU memory ceiling, rejecting any option that raises peak allocation beyond
  the established host budget;
- test CUDA graph capture only for repeated fixed tile shapes and only after
  measuring graph-retained VRAM; dynamic edge shapes require separate graph
  identities and reduce the likely benefit;
- investigate whether an export with the same required FP32 numerical contract
  can remove redundant graph operations, backed by source-framework parity and
  new model hashes. This is a model/export project, not a runtime toggle.

The 28-second application deadline remains appropriate. The optimized average
has about 9.57 seconds of application margin, while increasing the deadline
would move closer to Suwayomi's approximately 30-second caller timeout.

## Verification

Commands used included:

```sh
dotnet format KomaScaler.Net.sln --verify-no-changes --no-restore
dotnet build KomaScaler.Net.sln --configuration Release --no-restore
dotnet test KomaScaler.Net.sln --configuration Release --no-build
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
sh scripts/acceptance/gpu-acceptance.sh
```

The final ordinary gate has 41 unit tests and 10 integration tests. The real
GPU suite remains opt-in and is reported separately from fake-backed tests.
