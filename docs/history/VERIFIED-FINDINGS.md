# Verified Findings

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

This is the consolidated research outcome. Detailed sources, commit pins,
tables, and caveats are preserved in `research/`.

## Models

The production set is the seven official MangaJaNai V1 monochrome 2× models at
nominal input heights 1200, 1300, 1400, 1500, 1600, 1920, and 2048. Discord
beta/IllustrationJaNai models are excluded.

All seven checkpoints are old-style ESRGAN RRDBNet state dictionaries with the
same graph and different weights:

- 3 external input and output channels;
- effective 2× scale;
- pixel-unshuffle factor 2 followed by internal 4× upsampling;
- 64 features, 23 RRDB blocks, growth width 32;
- 16,703,171 parameters stored as FP16 checkpoint tensors;
- reconstructed/exported and executed as FP32 for production.

The model license is CC BY-NC 4.0. The service may reference local files, but
this package does not redistribute weights or derived ONNX artifacts.

## Decode and preprocessing

The reference application decodes through libvips, converts embedded profiles
to sRGB, detects visually monochrome RGB, converts to luminance, applies its
default auto-level operation, repeats grayscale to three channels, and feeds
gamma-encoded `[0,1]` NCHW tensors. It selects output channel 0 and clips to
`[0,1]`.

Production deliberately corrects four reference edge defects:

- physically apply EXIF orientation before dimensions/routing;
- use correct RGB luminance coefficients;
- composite supported alpha over white rather than silently discarding it;
- classify all-black/all-white images as monochrome rather than color.

Color and unsupported/animated images are returned unchanged in the first
version. Lossy WebP Q80 is an application default, not a model requirement;
production starts with lossless WebP.

## Model selection

Selection uses only EXIF-oriented source height. The exact midpoint bands are:

| Oriented height | Model |
|---:|---:|
| 1–1250 | 1200p |
| 1251–1350 | 1300p |
| 1351–1450 | 1400p |
| 1451–1550 | 1500p |
| 1551–1760 | 1600p |
| 1761–1984 | 1920p |
| 1985+ | 2048p |

Exact ties choose the lower model. Width, longest side, DPI, and portrait versus
landscape do not alter routing.

## ONNX export and runtime

All graphs export at opset 17 with fixed batch 1 and dynamic height/width:

```text
input  float32 [1,3,H,W] named input
output float32 [1,3,2H,2W] named output
```

The graph reflect-pads right/bottom to a multiple of four, replicates only when
reflection is impossible, and crops to exact 2×. The conventional operator set
runs under CUDA Execution Provider.

FP16 CUDA fit in VRAM but failed the strict fidelity gate. FP32 ONNX Runtime
CUDA passed 91/91 PyTorch-reference cases on the RTX 3060 with TF32 disabled:

| Metric | Observed worst | Required |
|---|---:|---:|
| Maximum absolute error | 0.0002120137 | ≤ 0.0003 |
| Mean absolute error | 0.0000040405 | ≤ 0.00001 |
| Minimum PSNR | 101.8087 dB | ≥ 90 dB |
| Minimum SSIM | 0.99999976 | ≥ 0.99999 |
| Black/white-region MAE | 0.0000005539 | ≤ 0.00002 |
| Halftone energy relative error | 0.0000002518 | ≤ 0.0001 |

The accepted target stack was ONNX Runtime 1.26.0, CUDA 12.1, cuDNN 9.1,
NVIDIA driver 595.58.03, and RTX 3060 12 GiB.

All seven FP32 sessions loaded in about 4.16 seconds. A warmed 256×256 Run took
about 92–93 ms per model. The seven-session warmed process used 4220.75 MiB
total GPU memory in the Step 5 probe.

## Tiling

Balanced bounded cores plus clipped half-sine blending preserve the reference
tiling semantics. The selected reloadable defaults are core 320 and 32 pixels
of context per interior side. Adjacent tiles therefore share 64 input pixels
and 128 output pixels at 2×.

On nine full-page cases, 320/32 processed 191 tiles in 69.810 seconds of ORT
time and 70.498 seconds total. Python extraction/blending was 0.688 seconds,
0.976% of the tiled loop. Peak use was 10876.75 MiB, leaving 1411.25 MiB. The
slowest page was 11.735 seconds.

Context 32 cost about 2.13% more ORT time than 16 and 3072 MiB more peak VRAM,
but improved worst global MAE, minimum SSIM, absolute seam-band MAE, and a
derived cross-boundary gradient error. No straight grid was visible. Tiled
output is not numerically identical to full-frame output.

Core 384 with context 16 left only 29.25 MiB and is unsafe. Core 448 failed
CUDA allocation. The measured low-memory fallback is 192/16.

## Suwayomi contract

Suwayomi uses `server.serveConversions` and sends a synchronous POST multipart
request with one part named `image`. The filename is generated and arbitrary;
no manga/chapter/page identifier is included. The response body is sniffed by
magic bytes, while status and declared response MIME are not reliable fallback
signals.

Therefore the converter must return HTTP 200 plus a complete valid image body
for upscaled output, bypass, and recoverable failure. Never return 204, JSON, or
text for a handled page. Suwayomi does not persist the served conversion, so the
upscaler needs its own content-addressed cache and in-flight deduplication.

The WebUI can prefetch several pages concurrently. HTTP handling is concurrent.
One queued image conversion at a time, with configurable bounded
intra-image GPU tile parallelism. Production runs up to eight tiles and blends
completed results in deterministic tile order. Because the shared client can hit a roughly
30-second read timeout, the service uses an internal response deadline of 28
seconds and returns the original on a miss.

## Final architecture change

A separate worker process is not required in the first implementation. One
ASP.NET singleton owns all seven sessions. Tiling settings reload live and are
snapshotted per page. Idle disposal may reclaim session-owned allocations while
ASP.NET remains alive. Complete CUDA-context release is not guaranteed in
process; a clean systemd restart remains the last resort for an unrecoverable
CUDA fault and can be replaced by process isolation later if operational data
justifies it.
