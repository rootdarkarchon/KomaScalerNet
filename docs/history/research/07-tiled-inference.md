# MangaJaNai × Suwayomi — Step 7: Tiled Inference

Investigation date: 2026-07-20
Status: complete; measured defaults selected for the target RTX 3060

## Decision

Use this normal production profile by default. The maximum core size and
per-side context are operator-configurable through reloadable `appsettings`.
A valid change applies to the next page without restarting ASP.NET or reloading
the ONNX sessions.

| Property | Default value |
|---|---|
| Maximum core size | **320×320 input pixels** |
| Partitioning | Balanced rows and columns; each core is no larger than 320 |
| Context per interior side | **32 input pixels** |
| Adjacent shared input | 64 input pixels |
| Adjacent shared output at 2× | 128 output pixels |
| Stitching | Clipped half-sine blend, separable horizontally and vertically |
| Page-edge padding | None from the tiler |
| Model-validity padding | ONNX graph reflect-pads right/bottom to a multiple of 4; replicate only when reflection is impossible; crop back to exact 2× |
| Precision/runtime | FP32 ONNX Runtime CUDA; TF32 disabled |
| Residency | All seven sessions in one in-process ASP.NET singleton GPU service |
| GPU concurrency | Exactly one serialized inference lane |
| Output size | Exactly `2H × 2W` |
| Low-memory fallback | 192 maximum core with 16-pixel context; no session reload is needed for a normal profile change |

Default policy/version identifier:

```text
mangajanai-v1-fp32-balanced-{MaximumCoreSize}-context-{ContextPixelsPerSide}-halfsine-v1
```

The selected 320/32 quality profile peaked at 10876.75 MiB and retained 1411.25
MiB of measured headroom. The 384/16 profile is rejected despite being faster
because it left only 29.25 MiB free. The 448 profile failed with an ONNX Runtime
CUDA arena allocation error.

## Source parity

The pinned MangaJaNaiConverterGui implementation partitions a page into balanced
rows and columns, expands each interior side by 16 pixels, and blends the shared
outputs. The behavior is implemented in
[`auto_split.py`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/nodes/impl/upscale/auto_split.py)
and
[`tile_blending.py`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/nodes/impl/upscale/tile_blending.py).

The blend function is:

```text
sinBlend(x)  = (sin(πx − π/2) + 1) / 2
halfSin(i)   = sinBlend(clamp(2i − 0.5, 0, 1))
```

This leaves each neighboring tile fully weighted near its core and transitions
through the middle half of the shared area. Production preserves the reference
partition and blend but deliberately increases context from 16 to 32 pixels:
target-host measurements show better absolute seam/full-frame fidelity for a
small latency cost that remains within the request budget.

## Target-host evidence

Definitive archive:

```text
20260720T111652Z-STEP7-RETURN.zip
SHA-256 f468a8b78926caee45729f9953033bccf30229d9d924c243fa8d7b429c2d49c1
```

Focused production-overlap archive:

```text
20260720T114141Z-STEP7-OVERLAP-RETURN.zip
SHA-256 d88a85f1982eb491a887fa59d2a5bdac953908e39dec6f7079c6c665e2bb99a5
```

Every file in both extracted archives passed its included SHA-256 manifest.

| Environment | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB |
| Driver | 595.58.03 |
| ONNX Runtime | 1.26.0 |
| PyTorch used by the offline harness | 2.4.1+cu121 |
| CUDA/cuDNN | CUDA 12.1; cuDNN 9.1 |
| Precision | FP32 |
| TF32 | Disabled |
| Initial GPU usage | 380.6875 MiB |
| Seven-session usage before shape warming | 1456.75 MiB |
| Seven-session load time | 4.289 s |

The first Step 7 archive contained no measurements because isolated workers
loaded ONNX Runtime before the PyTorch-bundled CUDA libraries. Version 2 fixed
the preload order. The definitive run rejected CPU fallback and required
`CUDAExecutionProvider` to be the active first provider for every session.

## Core-size benchmark

Each core-size profile used the original 16-pixel context and processed the
same 15.920352 input megapixels across nine complete-page cases: all seven
routing bands, a 2401×1600 landscape spread, and the official 1660×1400 demo.
All seven FP32 sessions remained resident.

| Maximum core | Tiles | GPU/ORT `Run` time | Complete tiled-loop time | Python extraction/blend | Python share | Peak used VRAM | Free headroom | Slowest page |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 510 | 110.190 s | 110.893 s | 0.702 s | 0.633% | 4220.75 MiB | 8067.25 MiB | 18.134 s |
| 256 | 296 | 86.470 s | 87.098 s | 0.628 s | 0.722% | 7292.75 MiB | 4995.25 MiB | 14.484 s |
| 320 | 191 | 68.394 s | 69.006 s | 0.612 s | 0.887% | 7804.75 MiB | 4483.25 MiB | 10.959 s |
| 384 | 151 | 57.170 s | 57.776 s | 0.607 s | 1.050% | 12258.75 MiB | 29.25 MiB | 9.822 s |
| 448 | — | — | Failed | — | — | at/near capacity | — | CUDA OOM |

The 384 profile saves only 1.137 seconds on the slowest measured page relative
to 320 but consumes another 4454 MiB of headroom. It is not robust against
normal CUDA arena variation, another process, driver use, or unseen page
shapes. The 448 worker later failed while requesting a 168,820,736-byte
convolution buffer, confirming that this is a real capacity boundary.

The 256 profile is 26.2% slower in aggregate than 320 while saving only 512 MiB
of measured peak usage. It is not a useful low-memory fallback. The 192 profile
is materially slower but saves 3584 MiB relative to 320, so it is the meaningful
fallback tier.

Exact fixed-size padded tiles are not selected. Balanced bounded cores preserve
the reference application's edge semantics and avoid inventing reflect,
replicate, or zero pixels beyond the page. The measured dynamic-shape profile
already meets the latency target; there is no
evidence that fixed page-edge padding would justify its output change.

## Focused 320-core overlap benchmark

The focused run compared context 16 and 32 with fresh workers, the same nine
pages, the same 191 balanced cores, all seven sessions resident, FP32 CUDA, and
TF32 disabled.

| Context | GPU/ORT time | Complete tiled-loop | CPU overhead | Python share | Peak used VRAM | Free headroom | Slowest page |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 68.352 s | 68.951 s | 0.600 s | 0.870% | 7804.75 MiB | 4483.25 MiB | 11.004 s |
| **32** | **69.810 s** | **70.498 s** | **0.688 s** | **0.976%** | **10876.75 MiB** | **1411.25 MiB** | **11.735 s** |

Context 32 costs 2.13% more GPU/ORT time in aggregate and 0.732 seconds on the
slowest page. Its material cost is 3072 MiB of additional CUDA arena/workspace
usage. On the measured mostly idle 12 GiB device, the remaining 1411.25 MiB is
accepted for the normal quality profile, with a startup free-memory check and
the measured 192/16 low-memory profile retained for contention or recovery.

## Selected-profile page latency

The selected 320/32/half-sine profile measured:

| Case | Input | Model | Tiles | GPU/ORT time | Complete tiled-loop time | Python tiling overhead |
|---|---:|---:|---:|---:|---:|---:|
| Route lower band | 769×1250 | 1200p | 12 | 5.463 s | 5.496 s | 0.033 s |
| First 1300p height | 801×1251 | 1300p | 12 | 5.118 s | 5.162 s | 0.044 s |
| First 1400p height | 833×1351 | 1400p | 15 | 6.158 s | 6.197 s | 0.039 s |
| First 1500p height | 865×1451 | 1500p | 15 | 6.724 s | 6.778 s | 0.055 s |
| First 1600p height | 897×1551 | 1600p | 15 | 6.213 s | 6.263 s | 0.049 s |
| First 1920p height | 1001×1761 | 1920p | 24 | 8.961 s | 9.036 s | 0.076 s |
| First 2048p height | 1137×1985 | 2048p | 28 | 10.563 s | 10.658 s | 0.095 s |
| Landscape spread | 2401×1600 | 1600p | 40 | 11.553 s | 11.735 s | 0.183 s |
| Official demo | 1660×1400 | 1400p | 30 | 9.057 s | 9.171 s | 0.114 s |

These figures exclude image decode and final encoding, but the slowest warm
tiled page remains well below the roughly 25-second internal request deadline.
Even a cold seven-session startup plus the slowest page was approximately 16.02
seconds before decode/encode.

## Python CPU-time separation

The pinned CPU core observed during the run did not determine the benchmark
outcome. For the selected profile:

```text
complete tiled-loop time        70.497623 s
sum of ONNX Runtime Run calls   69.809754 s
Python tile/blend overhead       0.687868 s
Python share                         0.976%
```

`tileSeconds` begins immediately before `InferenceSession.Run` and ends after
the output has returned, so it includes ONNX dispatch, host/device transfers,
GPU execution, and synchronization. `totalSeconds` additionally includes the
Python tile loop and half-sine accumulation. Fixture generation and PNG preview
encoding are outside that timer.

The production C# implementation can parallelize decode, tile preparation, and
CPU blending around the single serialized GPU lane. That may reduce CPU
consumption and the small measured overhead, but it should not be used to claim
a lower GPU time than the recorded ONNX `Run` totals. ONNX Runtime thread
spinning may also be disabled later if CPU utilization matters, after confirming
that it does not harm latency.

## Seam and full-frame comparison

The seam matrix compared crop and half-sine stitching at 8, 16, 24, and 32
input pixels of context. Four native-resolution fixtures crossed several tile
boundaries: high-contrast lines, screentones, noisy JPEG texture, and an
official real-manga crop.

All candidates failed the original combined numerical gate. This is retained
as a real finding, not relabeled as a pass: a tiled result is not numerically
equivalent to full-frame inference at these context sizes. The 23-block RRDB
network's effective receptive field extends well beyond the tested overlap, so
context truncation changes texture reconstruction throughout a tile, especially
for JPEG noise and dense patterns.

That combined gate was not a valid seam-only test because its absolute seam
band includes the same broad context-truncation difference found away from the
boundary. The already-declared **seam penalty**—seam-band MAE minus nearby-band
MAE—isolates boundary localization more directly.

| Candidate | Worst global MAE | Worst seam penalty | Seam penalty in 8-bit levels | Minimum SSIM | Decision |
|---|---:|---:|---:|---:|---|
| Crop, context 8 | 0.086422 | 0.010044 | 2.561 | 0.703371 | Reject |
| Crop, context 16 | 0.078450 | 0.009355 | 2.386 | 0.737178 | Reject |
| Crop, context 24 | 0.074612 | 0.007831 | 1.997 | 0.747728 | Reject |
| Crop, context 32 | 0.071322 | 0.009495 | 2.421 | 0.765401 | Reject |
| Half-sine, context 8 | 0.085129 | 0.001932 | 0.493 | 0.708565 | Marginal |
| Half-sine, context 16 | 0.076519 | **0.000674** | **0.172** | 0.745107 | Reference-compatible baseline |
| Half-sine, context 24 | 0.072283 | 0.001434 | 0.366 | 0.759202 | Intermediate |
| **Half-sine, context 32** | **0.068226** | 0.004552 | 1.161 | **0.782048** | **Selected quality default** |

The predeclared seam-penalty limit was 0.5 of one 8-bit code value. Half-sine
16 passed it and had the lowest worst seam penalty. That metric measures how
much the tiled-to-full-frame error at a boundary exceeds the error in nearby
pixels; it is not a direct measure of the discontinuity within the candidate
image. Consequently, it is useful evidence but is not decisive by itself.

Half-sine 32 has the lowest worst global MAE and highest minimum SSIM of the
tested profiles. A derived cross-boundary gradient-error check on the returned
native PNGs also favored context 32 over 16: aggregate mean `0.048246` versus
`0.056830`, and p99 `0.458824` versus `0.547806`. These measures align with the
lower absolute seam-band MAE observed for context 32 (`0.073916` versus
`0.093279` for context 16).

Native candidate and amplified-difference inspection found broad texture and
dot-phase differences relative to full-frame output, but no straight horizontal
or vertical boundary in either half-sine finalist. The complete official demo
and landscape previews likewise showed no visible grid. Combined with the
focused production run's small latency cost and acceptable memory headroom,
this selects half-sine 32 as the quality default. It does **not** claim that
tiled and full-frame outputs are numerically interchangeable.

## Application configuration

The C# service should bind the following settings through `IOptionsMonitor`:

```json
{
  "Upscaling": {
    "Tiling": {
      "MaximumCoreSize": 320,
      "ContextPixelsPerSide": 32,
      "LowMemoryMaximumCoreSize": 192,
      "LowMemoryContextPixelsPerSide": 16
    }
  }
}
```

`ContextPixelsPerSide` is the precise configuration name for the seam/overlap
setting. With the default value 32, two adjacent tiles share 64 input pixels
and 128 output pixels at 2×.

Validate every initial value and reload: all four values must be positive and
context must be smaller than its corresponding maximum core. For the measured
RTX 3060 deployment, `320/32` is the normal tested profile and `192/16` is the
tested low-memory profile. A custom normal profile whose maximum inference
extent `MaximumCoreSize + 2 × ContextPixelsPerSide` exceeds 384 pixels must be
rejected unless a separately validated hardware policy explicitly allows it;
`384/16` and `448/16` are unsafe on this device.

Publish a valid reload as one immutable tiling-policy object. Each page takes
one snapshot before partitioning and uses it through completion, so a file
change cannot alter geometry halfway through a page. Invalid reloads are logged
and surfaced as unhealthy configuration while the last-known-good policy stays
active. Core/context changes do not alter model weights or session options and
therefore do not require session disposal. Include the effective values in the
pipeline/cache identity and expose both configured and effective values in
health and metrics.

## In-process ASP.NET lifetime

A dedicated inference process is not required for normal operation. Register a
singleton GPU service inside ASP.NET that owns all seven `InferenceSession`
instances and a single `SemaphoreSlim(1, 1)` (or equivalent async gate). The
gate covers each complete `Run` call and guarantees that only one CUDA
inference is active, while decode, tile preparation, and encoding may proceed
on bounded CPU resources.

Dispose every per-request `OrtValue`, input/output binding, tensor owner, and
pooled buffer deterministically. Keep the seven sessions resident while the GPU
service is active; changing only tiling geometry does not justify unloading
them. If idle VRAM reclamation is desired later, the singleton can enter a
draining state, wait for the GPU gate, dispose all sessions, and lazily recreate
them on the next request. This is an optional residency policy, independent of
core/context reload.

Session disposal releases allocations owned by ONNX Runtime, but CUDA may keep
a process-level runtime context or allocator state. Therefore an in-process
dispose/recreate cycle is best-effort recovery, not proof of complete VRAM
release. Only process exit guarantees destruction of all CUDA state belonging
to the application.

## Production algorithm

For an oriented `W × H` source and configured maximum core
`M = MaximumCoreSize` (default 320):

```text
countX = ceil(W / M)
countY = ceil(H / M)
coreW  = ceil(W / countX)
coreH  = ceil(H / countY)
```

For each core region:

1. expand by up to `ContextPixelsPerSide` pixels (default 32) on every side
   that remains inside the page;
2. add no synthetic context beyond the page edge;
3. create a contiguous FP32 `1×3×tileH×tileW` tensor by repeating luminance;
4. call only the already-selected model session on the serialized GPU lane;
5. take output channel 0;
6. combine horizontal and vertical overlaps with the clipped half-sine weights;
7. verify finite values and exact final dimensions `2H × 2W`;
8. clip to `[0,1]`, round to 8-bit, and encode according to the output policy.

The maximum actual inference input is normally larger than the core because it
includes context. With the 320/32 defaults, a fully interior tile can reach 384
pixels on one axis. The implementation and configuration should call the core
setting `MaximumCoreSize`, not `TileInputSize`, to avoid a dangerous mismatch.

## Memory failure policy

Do not catch a CUDA OOM and immediately continue using the same sessions. Arena
state and the active request are no longer trustworthy enough for a
latency-sensitive service.

```text
CUDA OOM or CUDA execution fault
→ mark current conversion as fallback
→ return the original valid image
→ mark the singleton GPU service faulted and block new Run calls
→ dispose all request objects and all seven sessions after the active call exits
→ verify released VRAM before an optional in-process reinitialization at 192/16
→ if release/reinitialization cannot be verified, exit ASP.NET cleanly
→ let the service supervisor restart exactly one application instance
```

The recovered service should remain on the low-memory profile until a valid
configuration change or explicit health action restores the normal profile. Do
not retry the failed page synchronously unless the remaining request deadline
is measured and sufficient; returning the original is the safe default. This
rare fault policy is the only reason process restart may be required; ordinary
core/context changes remain entirely in-process.

## Implementation and observability requirements

- Bind and validate `MaximumCoreSize` and `ContextPixelsPerSide` from
  `appsettings`; default them to 320 and 32.
- Apply a valid reload atomically to future pages and retain the last-known-good
  policy after an invalid reload.
- Snapshot the effective core/context/stitch policy once per page and include
  its values and policy version in cache keys.
- Keep all seven sessions in the ASP.NET singleton and serialize every `Run`
  call.
- Allow CPU decode/preparation for other requests, but bound memory and never
  permit two GPU inference calls concurrently.
- Record page dimensions, selected model, core grid, tile count, distinct tile
  shapes, sum of ORT time, total tiling time, encode time, and result status.
- Expose the active normal/low-memory profile in health and metrics.
- Expose GPU-service state as `Uninitialized`, `Ready`, `Draining`, or `Faulted`.
- Treat the measured 384/16 and 448/16 profiles as unsupported on the 12 GiB
  deployment, not as opportunistic settings.
- Use pooled CPU buffers in C# and avoid retaining complete per-tile outputs
  after their contribution has been stitched.
- Preserve exact 2× shape checks and return the original on any invalid result.

## Step 7 acceptance

Step 7 is complete with a measured RTX 3060 policy:

```text
balanced maximum core 320
+ 32 px context per interior side
+ clipped half-sine blend
+ FP32 CUDA / TF32 off
+ seven resident in-process sessions
+ one serialized GPU lane
+ in-process reloadable 192/16 fallback
```

These are the measured defaults, not hard-coded constants: production reads
the core and per-side context sizes from reloadable `appsettings`, snapshots a
valid policy for each page, and reflects it in cache identity and observability.
No dedicated worker process or application restart is required for a geometry
change. The default profile meets the practical seam, VRAM, and page-latency
requirements on the target host. The investigation phase can now move to the
final verified findings, runtime decision, architecture specification,
implementation prompt, and consolidated acceptance criteria requested by the
work brief.
