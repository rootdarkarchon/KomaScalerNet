# Optimization 3: tile-size and parallelism matrix

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20
Target: NVIDIA GeForce RTX 3060 12 GiB, driver 595.58.03, CUDA 12, cuDNN 9
Runtime: .NET SDK 10.0.302, ONNX Runtime GPU 1.26.0, libvips 8.16.1

## Objective

Measure core tile sizes 320, 384, 448, 512, and 576 with intra-image GPU
parallelism factors 1, 4, 6, and 8. Image-level GPU work remains serialized:
one queued image conversion at a time, with configurable bounded intra-image
GPU tile parallelism.

The test answers which combination is fastest for the established 1660×1400
production fixture. It does not by itself replace the seven-model production
acceptance that established 320 as the conservative VRAM-safe default.

## Harness and safety controls

The reusable harness is `scripts/benchmarks/gpu-tile-matrix-benchmark.sh`. It:

- publishes once to disk-backed `/devtmp`, then reuses that output for all 20
  cases;
- refuses tmpfs and requires at least 3 GiB free;
- starts a separate, pre-warmed process with a fresh cache for each combination;
- runs combinations sequentially, never constructing competing session sets;
- gives each process a 150-second hard limit and each HTTP conversion a
  60-second limit;
- samples VRAM, GPU utilization, available RAM, and shmem once per second;
- terminates a case above 10,240 MiB VRAM or below 3 GiB available RAM;
- stops each process and checks VRAM release before starting the next case.

The follow-up requested 576 core with 64 pixels of context, a 704-pixel maximum
tile extent. Configuration validation previously capped extent at 384, so the
bound was raised to 704 while retaining the production default of 320/32.
Tests cover 576/64 as the valid boundary and reject 577/64.

Command:

```bash
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_BENCHMARK_IMAGE=/devtmp/komascaler-acceptance/http-fixtures/neutral.png \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
  sh scripts/benchmarks/gpu-tile-matrix-benchmark.sh
```

Raw results are under:

```text
/devtmp/komascaler-acceptance/tile-matrix/runs/20260720T160510Z-3520066
```

## Measured matrix

The input was the same neutral 1660×1400 PNG used by earlier target acceptance.
It routed to `mangajanai-v1-1400p-2x-fp32`. Every case returned HTTP 200 with
`X-Upscaler-Result: upscaled` and an exact 3320×2800 lossless WebP.

| core | factor | tiles | HTTP seconds | tile wall ms | encode ms | sampled peak VRAM MiB | minimum available RAM MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 1 | 30 | 19.374 | 16,064 | 3,105 | 3,097 | 6,818 |
| 320 | 4 | 30 | 10.476 | 7,237 | 3,039 | 4,199 | 6,058 |
| 320 | 6 | 30 | 8.945 | 5,751 | 2,995 | 5,265 | 6,074 |
| 320 | 8 | 30 | 8.866 | 5,669 | 2,995 | 5,307 | 5,901 |
| 384 | 1 | 20 | 11.941 | 9,324 | 2,447 | 3,097 | 6,869 |
| 384 | 4 | 20 | 7.871 | 5,212 | 2,480 | 5,223 | 6,231 |
| 384 | 6 | 20 | 8.103 | 5,460 | 2,456 | 6,277 | 6,128 |
| 384 | 8 | 20 | 7.595 | 4,932 | 2,462 | 6,319 | 6,224 |
| 448 | 1 | 16 | 10.388 | 7,719 | 2,490 | 3,097 | 6,946 |
| 448 | 4 | 16 | 7.394 | 4,666 | 2,500 | 4,185 | 6,388 |
| 448 | 6 | 16 | 7.183 | 4,565 | 2,433 | 5,253 | 6,446 |
| 448 | 8 | 16 | 7.066 | 4,431 | 2,428 | 6,319 | 6,610 |
| 512 | 1 | 12 | 8.784 | 6,108 | 2,467 | 3,085 | 7,190 |
| 512 | 4 | 12 | 6.932 | 4,264 | 2,489 | 4,185 | 6,586 |
| 512 | 6 | 12 | 6.886 | 4,180 | 2,506 | 4,229 | 6,639 |
| 512 | 8 | 12 | 6.798 | 4,092 | 2,508 | 4,261 | 6,907 |
| 576 | 1 | 9 | 8.215 | 5,473 | 2,563 | 3,097 | 6,928 |
| 576 | 4 | 9 | 6.725 | 4,003 | 2,505 | 6,223 | 6,909 |
| **576** | **6** | **9** | **6.570** | **3,886** | **2,480** | **4,219** | **6,654** |
| 576 | 8 | 9 | 6.794 | 4,146 | 2,446 | 5,285 | 6,667 |

The sampled VRAM figures are one-second observations, not allocator high-water
marks. Short-lived peaks can be missed; this explains some non-monotonic values
such as 576/4 exceeding 576/6. They must not be interpreted as proof that a
larger tile always consumes less VRAM.

No pressure threshold fired, request timed out, CUDA call failed, or NVIDIA
Xid/OOM event appeared. VRAM returned to the 3 MiB host baseline after every
process. All seven CUDA sessions were preloaded for every case; only the 1400p
model performed page inference.

## Determinism and image comparison

Within a fixed tile size, factors 1, 4, 6, and 8 produced byte-identical WebP
files. This confirms deterministic ordered blending under parallel execution.
The SHA-256 values were:

| core | output bytes | SHA-256 |
| ---: | ---: | --- |
| 320 | 4,427,946 | `6e77bac99078f0d87511ad64da254e22427103d28993d951dcf62c9dc007ec98` |
| 384 | 4,403,904 | `cd460e6872a95340f7ac6738ac5d68673f5f7666b29a6d8c3e976d8be9d6b0ca` |
| 448 | 4,417,696 | `4bbf58801e62e82027373c3e9714c44dcab63882932bece8f1dbc852dcbeca4a` |
| 512 | 4,423,598 | `88701dec5e07786b071ffca5f3601959596d5891d135bf1c42290753b9b3965b` |
| 576 | 4,410,670 | `4973ae57de3045233bbe47ccd79cc869f017ab30dde4847b3c795b2387c3e12f` |

Outputs across tile sizes are intentionally not byte-identical because tile
partition and blending policy are part of cache identity. Using the accepted
320 output as a comparison—not as independent ground truth—ImageMagick reported:

| core | PSNR versus 320 | SSIM versus 320 |
| ---: | ---: | ---: |
| 384 | 17.407 dB | 0.7603 |
| 448 | 15.390 dB | 0.6299 |
| 512 | 15.160 dB | 0.6166 |
| 576 | 15.162 dB | 0.6292 |

These large differences mean timing alone cannot establish visual parity. They
do not prove that 320 is intrinsically more accurate, because 320 is not a
full-frame reference, but they do prove that changing tile size materially
changes pixels and therefore needs renewed seam/reference validation.

## Interpretation

The dominant speedup comes from reducing the number of tiles: 30 at core 320,
20 at 384, 16 at 448, 12 at 512, and 9 at 576. Encoding remains roughly
2.4–3.1 seconds and becomes the largest fixed cost once tile inference is near
four seconds.

The fastest measured combination is **576/6 at 6.570 seconds**. It is 2.296
seconds (25.9%) faster than the current 320/8 result. Factor 8 regresses at core
576 because nine tiles form an eight-plus-one batch; factor 6 produces a more
balanced six-plus-three schedule and less concurrent contention. The result is
also only a single measurement, and the 224 ms advantage over 576/8 is small
enough that repetition could change their order.

## Recommendation

Use **576/6 as the leading optimization candidate**, but do **not** change the
production 320/8 default from this matrix alone.

Before promoting 576/6, repeat it and 512/6 or 512/8 across representative page
dimensions and all seven routed models, increase pressure sampling frequency,
perform the established reference/seam inspection, and rerun idle unload,
fault recovery, sustained conversion, and process-exit VRAM release. This is
required because the prior production work established 320 as the effective
VRAM-safe limit and this one fixture cannot invalidate worst-case evidence.

If that validation passes, 576/6 is the optimal measured production setting.
If it does not, retain **320/8**, which remains fully accepted and still
completes this page with about 19 seconds of margin under the 28-second
application deadline. A conservative intermediate experiment is 512/6:
6.886 seconds here, only 316 ms behind the winner, with a balanced two batches
of six tiles and lower sampled VRAM than most 384/448 high-parallelism cases.

## Direct-ORT quality continuation

This continuation supersedes the provisional promotion path above. Larger
tiles are **not approved for production**.

### Method

`scripts/benchmarks/gpu-quality-reference.sh` runs an explicitly opt-in GPU test with a
360-second hard timeout and 200 ms pressure sampling. It creates three
deterministic monochrome inputs small enough for untiled inference:

| input | dimensions | orientation |
| --- | ---: | --- |
| landscape | 640×384 | landscape |
| portrait | 384×640 | portrait |
| square | 640×640 | square |

For each input, one direct `InferenceSession.Run` through the real 1200p FP32
model produces the untiled CUDA reference. The direct session is disposed
before the seven-session application set is loaded, avoiding simultaneous
session-set pressure. Channel zero is quantized exactly as the production
tiler. Cores 512 and 576 are then compared at contexts 32, 48, and 64.

Whole-image metrics cover all output pixels. Seam metrics cover the union of
fixed eight-output-pixel bands around every internal balanced-partition
boundary. The measured SSIM is a global luminance statistic; PSNR, MAE, RMSE,
and maximum absolute byte error are also retained in the raw JSON.

Command:

```bash
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
  sh scripts/benchmarks/gpu-quality-reference.sh
```

Raw evidence:

```text
/devtmp/komascaler-acceptance/quality-reference/runs/20260720T162653Z-3532996
```

The test passed in 22.65 seconds. Peak sampled VRAM was 7,193 MiB, minimum
available host RAM was 7,620 MiB, and VRAM returned to 3 MiB after exit.

### Untiled-reference results

The core-512 and core-576 results are identical on these inputs. This is
expected, not a benchmark defect: both cores produce the same partition count
and therefore the same balanced boundary coordinates for dimensions up to 640.

| context | whole PSNR range | minimum whole SSIM | seam PSNR range | seam MAE range, bytes |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 49.379–52.495 dB | 0.999952 | 40.546–45.369 dB | 0.784–0.959 |
| 48 | 54.816–57.437 dB | 0.999986 | 51.510–56.454 dB | 0.137–0.226 |
| 64 | **62.077–65.522 dB** | **0.999997** | **63.484–67.072 dB** | **0.013–0.029** |

Quality improves monotonically with context. Context 64 is the quality winner:
its worst seam MAE is below 0.03 of one eight-bit gray level, versus almost one
gray level at context 32. The first result at 512/32 took 4.70 seconds because
it also lazily constructed and warmed the seven-session set; subsequent quality
runs were 0.54–0.85 seconds and should not be compared with full HTTP timings.

### Explanation of the earlier cross-core differences

The earlier 15–17 dB PSNR and 0.62–0.76 SSIM values used the accepted 320
tiled output as the reference. That comparison measured disagreement between
two tiling policies, not error against ground truth or untiled inference.

`BalancedTiler` derives boundaries from the number of tiles. On the 1660×1400
page, cores 320, 384, 448, 512, and 576 create respectively 30, 20, 16, 12,
and 9 tiles with different boundary locations and input extents. The model has
a finite receptive field, and the direct-reference experiment proves that
32-pixel context leaves substantially more boundary error than 48 or 64.
Moving every boundary therefore moves where that residual error occurs, so two
context-32 tiled outputs can disagree strongly even though neither is a valid
quality reference for the other.

The small untiled experiment establishes the direction and scale of the context
effect, but it does not establish an absolute quality ordering for the full
1660×1400 page: untiled inference at that size is outside the safe VRAM
envelope. Consequently the original cross-core scores cannot support promotion.
Promotion would require representative crops/full frames with an independent
untiled reference and seam inspection; the sustained-memory results below make
that promotion moot for the current runtime design.

## High-frequency sustained-run continuation

`scripts/benchmarks/gpu-leading-repeat-benchmark.sh` uses one reusable publish, fresh
cache identities, separate processes per combination, 200 ms RAM/VRAM samples,
a 10,240 MiB VRAM cutoff, a 3 GiB available-RAM cutoff, and hard process/request
timeouts. Metadata-distinct PNGs force real inference while preserving decoded
pixels, allowing exact output-determinism checks.

The requested larger candidates were tested first on the landscape page. The
first shape was sufficient to reject every candidate, so portrait and square
runs were deliberately not attempted after the same safety condition recurred.

| core/context | factor | successful pages before stop | HTTP seconds | highest sampled VRAM MiB | result |
| ---: | ---: | ---: | --- | ---: | --- |
| 512/64 | 6 | 2 | 7.814, 8.006 | 11,397 | watchdog stop |
| 512/64 | 8 | 2 | 7.583, 7.325 | 10,415 | watchdog stop |
| 576/64 | 6 | 3 | 7.109, 6.884, 7.062 | 11,409 | watchdog stop after response |
| 576/64 | 8 | 2 | 7.430, 7.138 | 10,415 | watchdog stop |

Every completed repeat within a combination was byte-identical. No system-wide
hang was allowed: the sampler terminated the service as soon as the configured
VRAM threshold was crossed.

The retained memory is not caused by simultaneous construction of the seven
sessions. Readiness completed before sampling the page runs. VRAM increased in
steps during repeated inference using the same selected 1400p session and the
same decoded shape. Parallel factor 6 accelerated the growth, but a 512/32
factor-1 control also grew from roughly 3,085 to 4,133 and 5,157 MiB across
three pages. The evidence identifies CUDA EP arena allocation retained across
repeated Runs; parallel Runs cause it to reach the device limit sooner.

Three mitigations were measured and rejected:

- `arena_extend_strategy=kSameAsRequested` slowed growth but still crossed
  10,305 MiB on the third 512/6 page;
- `cudnn_conv_use_max_workspace=0` reproduced the original 11,419 MiB result;
- `cudnn_conv_algo_search=DEFAULT` roughly doubled latency to 15.6–15.7
  seconds and still crossed 10,385 MiB;
- a 6 GiB CUDA arena limit held total use near 7.3 GiB, but the third page
  exhausted the arena, returned recoverable fallback, and fault-stopped the
  service. It bounded failure rather than enabling memory reuse.

All experimental provider changes were reverted. Production continues to use
HEURISTIC search with no artificial per-session arena cap.

### Production control

The actual 320/32 factor-8 profile was repeated for five uncached landscape
pages under the same 200 ms sampler:

| page | HTTP seconds | output |
| ---: | ---: | --- |
| 1 | 8.328 | exact deterministic WebP |
| 2 | 8.091 | exact |
| 3 | 8.006 | exact |
| 4 | 8.735 | exact |
| 5 | 8.456 | exact |

VRAM warmed in stages and then plateaued at 8,389 MiB, minimum available RAM
was 4,016 MiB, all requests succeeded, and process exit returned VRAM to 3 MiB.
This is substantially less headroom than the one-second single-page benchmark
suggested, but it passed the five-page sustained control.

## Final recommendation

Retain **320/32 with factor 8** for production. Do not promote 512 or 576 at
factor 6 or 8.

Context 64 is the unambiguous quality winner against untiled ORT, but combining
it with either leading large core crosses the safe VRAM threshold within two or
three pages. Context 32 is less accurate at seams, while context 48 is a useful
quality midpoint, but large-core repeated allocation remains unsafe even at
context 32. Latency cannot outweigh that lifecycle failure.

Future work should first eliminate or explicitly recycle the CUDA EP arena at a
page boundary, then rerun sustained multi-shape and all-seven-model acceptance.
Until such a design proves bounded steady-state memory without adding CPU
fallback, the 6.57-second 576/6 result is a single-page optimization only and
is not a production candidate.
