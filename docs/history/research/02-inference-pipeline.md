# MangaJaNai × Suwayomi — Step 2: Existing Inference Pipeline

Status: complete
Source snapshot: MangaJaNaiConverterGui commit `e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222`
Relevant runtime dependency: Spandrel `v0.4.1`

## Outcome

The normal black-and-white path is now traced from image bytes to encoded output. The MangaJaNai-specific behavior lives mainly in `run_upscale.py` and the GUI's workflow configuration. Model reconstruction, tensor conversion, GPU execution, automatic tiling, overlap blending, and out-of-memory retry are generic vendored chaiNNer/Spandrel behavior.

For the production service, the revised worker design is:

- exactly one GPU worker process may exist at a time;
- it preloads all seven official 2× V1 ONNX sessions;
- it selects the appropriate resident session for each request;
- inference is serialized, so only one page/model has live activation tensors at a time;
- the worker process is terminated after the idle timeout, releasing all sessions and CUDA state together.

The accepted production artifacts are FP32 and occupy about **449 MiB on disk** across all seven models. Target-host measurement supersedes the earlier estimate: loading all seven sessions raises GPU use by `1076.06 MiB`; after every session has warmed a 256×256 shape and retained its cuDNN workspace, total GPU use is `4220.75 MiB`. Preloading all seven is therefore accepted for the RTX 3060.

## End-to-end reference path

1. Decode with libvips and convert the embedded profile to sRGB.
2. Convert the decoded image to a NumPy array.
3. Detect whether the page is effectively grayscale.
4. Select a workflow/model from the original image dimensions.
5. For a grayscale page, convert it to a single luminance channel.
6. Apply MangaJaNai's automatic black/white level adjustment.
7. Normalize to floating point `[0, 1]`.
8. Repeat the one grayscale channel to three channels.
9. Convert HWC to batched NCHW and run FP16 inference.
10. If needed, split into overlapping tiles and blend their outputs.
11. Clamp the model result to `[0, 1]`.
12. Keep output channel 0 and restore a one-channel grayscale image.
13. Convert to 8-bit with rounding.
14. Encode with libvips.

There is no default pre-inference resize in the built-in 2× workflows. With a requested scale of exactly 2×, there is also no post-inference resize.

## Decode, orientation, color, and alpha

The two decode entry points in [`run_upscale.py`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/run_upscale.py) are:

```python
pyvips.Image.new_from_file(path, access="sequential", fail=True).icc_transform("srgb").numpy()
pyvips.Image.new_from_buffer(stream, "", access="sequential").icc_transform("srgb").numpy()
```

Supported file extensions are PNG, JPEG, WebP, BMP, and AVIF.

| Concern | Existing behavior | Production decision |
|---|---|---|
| ICC/profile | Converts decoded pixels to sRGB | Preserve |
| EXIF orientation | Not applied; libvips `autorotate` defaults to false, and metadata is lost after conversion to NumPy | Correct: orient pixels before inspecting dimensions or inference |
| Channel order | libvips supplies RGB/RGBA, but local variables and OpenCV conversion constants treat it as BGR/BGRA | Correct the luminance conversion; grayscale detection itself is symmetric and unaffected |
| Alpha on grayscale input | BGRA-to-gray conversion discards alpha | Define an explicit policy; composite over white is the recommended manga default |
| Alpha on generic color input | Vendored chaiNNer has separate alpha handling, including a two-pass reconstruction path | Not part of the monochrome service path; color images should initially be bypassed |

The reference grayscale conversion uses `cv2.COLOR_BGR2GRAY` or `cv2.COLOR_BGRA2GRAY` on data originating as RGB/RGBA. This swaps the red and blue luminance coefficients. It is invisible for truly neutral pixels but can change tinted scans. This should be treated as a legacy defect, not a compatibility requirement.

## Grayscale detection

`cv_image_is_grayscale` in `run_upscale.py` operates as follows:

- A one-channel image is grayscale.
- For three or more channels, only the first three channels are inspected; alpha is ignored.
- It computes all three pairwise absolute channel differences.
- It subtracts 12 from those differences using saturating subtraction.
- Exact black `(0,0,0)` and white `(255,255,255)` pixels are excluded from the denominator.
- The remaining residual difference is averaged across non-black/non-white pixels and three channel pairs.
- The page is grayscale when that average is less than or equal to `user_threshold / 12`; the default threshold of 12 therefore compares against 1.

An entirely pure black/white image has a zero denominator and is classified as color by the current implementation. This is another edge-case defect to correct in production.

The new service's initial policy remains: images that fail the monochrome test are returned unchanged. IllustrationJaNai and Discord beta models are out of scope.

## Model selection

The built-in workflows are declared in [`MainWindowViewModel.cs`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/ViewModels/MainWindowViewModel.cs). Selection occurs before grayscale conversion or any optional resize and uses original decoded dimensions.

For the default 2× chains:

- selection is based on source image height;
- width constraints are zero/unbounded;
- there is no DPI, aspect-ratio, file-size, or longest-side input;
- the resolution bands and filenames are those recorded in the Step 1 model inventory;
- all seven V1 sessions can share one runtime graph definition and differ only in weights.

The production service must apply EXIF orientation before this height test, fixing the reference decoder's orientation omission.

## Automatic level adjustment

Every default black-and-white workflow enables `AutoAdjustLevels`.

The algorithm converts the page to 8-bit luminance, builds a 256-bin histogram, and estimates black and white points from edge-region peaks:

- The initial black peak is the largest bin from 0 through 30. The search then continues upward from 31 while tracking a larger peak and stops after the existing two-decrease rule.
- The initial white peak is the largest bin from 225 through 255. The analogous search continues downward from 224.
- Pixels are transformed with `(pixel - blackPoint) / (whitePoint - blackPoint)` and clipped to `[0, 1]`.

When automatic levels are disabled, the generic normalization path converts integer samples to `float32`, divides by the integer type's maximum value, and clips to `[0, 1]`.

The model receives gamma-encoded sRGB values. There is no linear-light conversion, mean subtraction, standard-deviation normalization, or `[-1, 1]` remapping.

Auto-level behavior should be reproduced for the first parity implementation, then evaluated as a separate visual-quality switch. A guard is required for degenerate histograms where the estimated black and white points are equal.

## Tensor conversion and model call

The vendored tensor adapter is [`nodes/impl/pytorch/auto_split.py`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/nodes/impl/pytorch/auto_split.py); the generic upscale node calls it from [`upscale_image.py`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/packages/chaiNNer_pytorch/pytorch/processing/upscale_image.py).

The exact normal path is:

| Stage | Representation |
|---|---|
| Input image | Contiguous NumPy HWC, normally one channel and `float32` in `[0,1]` |
| Grayscale adaptation | Repeat the single channel to RGB; no channel swap |
| Torch input | Batch dimension added, HWC → NCHW, channels-last memory format |
| Device/dtype | CUDA FP16 when enabled; GPU autocast is active |
| Model output | Three channels at 2× spatial dimensions |
| Range | Spandrel clamps in-place to `[0,1]` |
| Grayscale restoration | Select output channel 0, then squeeze to a 2D array |

The output is not averaged back to luminance. Channel 0 is used. Since all three model inputs are identical grayscale planes and the MangaJaNai models are trained for this path, preserving that behavior is the correct baseline.

## Padding

Spandrel identifies these ESRGAN models as having a minimum input size of 2 and a spatial multiple requirement of 4 because the architecture uses pixel unshuffle.

Before the model call, the descriptor pads only the right and bottom edges to the next valid multiple. Reflection padding is preferred; replication is used only when the requested reflection cannot be formed from a very small input. The descriptor crops the corresponding scaled padding from the output.

The reconstructed RRDB model also contains its factor-2 pixel-unshuffle preparation. With descriptor-aligned inputs, that internal padding is normally redundant.

For the ONNX implementation, the cleanest deterministic contract is likely to keep service-side tiles aligned to the graph's required multiple and to make right/bottom padding, crop, and output clamp explicit in C#. Step 4 must verify this against exported-model behavior.

## Tiling and stitching

The default `ModelTileSize` is **Auto (Estimate)**. Generic chaiNNer code estimates a starting tile size from model size and available CUDA memory, then retries with progressively smaller tiles after CUDA out-of-memory errors. The lower tile-size limit is 16 input pixels.

Important semantics:

- If the entire image fits the starting tile bounds, full-page inference is attempted first.
- Otherwise, the page is partitioned into balanced rows and columns; it is not simply a grid of maximum-size tiles plus a small tail tile.
- The default overlap value is 16 input pixels on every available side of an interior core tile.
- Adjacent inference inputs therefore share 32 input pixels in total.
- At 2× scale, each side contributes 32 output pixels of context and adjacent inference results overlap across 64 output pixels.
- Page edges receive no synthetic context from the tiler; only the model-validity padding described above may be applied on the bottom/right.
- The final output dimensions are exactly `2W × 2H`.

Overlaps are blended rather than cropped. `TileBlender` combines tiles horizontally and vertically using a clipped half-sine transition. The transition occupies the middle portion of the shared area, leaving each tile fully weighted near its core.

The 2024 commit labelled `test fix for batch upscale` does not modify the tiling modules. The fork contains a vendored chaiNNer snapshot and has drifted from current upstream, but the tiling path is generic backend infrastructure rather than a MangaJaNai-specific algorithm.

For production, fixed or tightly bounded tile sizes are preferable to exception-driven auto-sizing because they make latency and CUDA memory predictable. The reference 16-pixel overlap and half-sine blend form the parity baseline; Step 7 should compare alternative overlaps and crop-vs-blend behavior on seam-sensitive pages.

## Output conversion and encoding

After inference:

- Spandrel clamps to `[0,1]`.
- The grayscale result is converted with `(value × 255).round()` and cast to `uint8`.
- At the target scale of 2×, no final resize occurs.
- The NumPy array is wrapped in a new libvips image and encoded.

The GUI supports several output formats. Its current default is lossy WebP at quality 80, with lossless compression disabled. Because wrapping a NumPy array creates a fresh libvips image, original EXIF and ICC metadata are not carried into the output.

Lossy WebP Q80 is an application default, not a model requirement. The service should initially use lossless WebP (or PNG during validation), then benchmark near-lossless/lossy choices separately. MIME type must match the bytes returned to Suwayomi.

## MangaJaNai-specific versus generic backend behavior

| Area | Owner | Classification |
|---|---|---|
| Decode and sRGB conversion | `run_upscale.py` | MangaJaNai wrapper |
| Grayscale detection | `run_upscale.py` | MangaJaNai-specific |
| Resolution-band/model selection | GUI workflow + `run_upscale.py` | MangaJaNai-specific |
| Grayscale conversion and auto levels | `run_upscale.py` | MangaJaNai-specific |
| Optional pre/post resize | `run_upscale.py` | MangaJaNai wrapper; inactive for default 2× path |
| Model cache/lifetime | `run_upscale.py` | MangaJaNai wrapper |
| State-dict architecture detection | Spandrel 0.4.1 | Generic dependency |
| NumPy/Torch tensor conversion | vendored chaiNNer | Generic backend |
| CUDA/FP16/autocast | vendored chaiNNer + PyTorch | Generic backend |
| Tile estimation and OOM retry | vendored chaiNNer | Generic backend |
| Overlap expansion and half-sine blending | vendored chaiNNer | Generic backend |
| Generic RGBA reconstruction | vendored chaiNNer | Generic backend; bypassed by the normal grayscale path |
| Output encoding and metadata loss | `run_upscale.py` + libvips | MangaJaNai wrapper |
| Custom FDAT registration | `spandrel_custom` | IllustrationJaNai-only; out of scope |

The existing Python process already caches each model after first use. The new architecture changes that from lazy loading to eager loading of all seven sessions, while retaining one process and one serialized GPU execution lane.

## Compatibility decisions for the C# service

| Behavior | Decision |
|---|---|
| Embedded profile → sRGB | Preserve |
| Height-based seven-model selection | Preserve |
| Default auto-level algorithm | Reproduce for parity, expose to validation |
| `[0,1]` gamma-encoded input | Preserve |
| One grayscale channel repeated to RGB | Preserve |
| Output channel 0 restored to grayscale | Preserve initially |
| Right/bottom model-validity padding and crop | Preserve explicitly |
| 16 px input context + half-sine blend | Parity baseline; benchmark alternatives |
| EXIF orientation ignored | Correct before model selection |
| RGB pixels converted with BGR coefficients | Correct |
| Alpha silently discarded | Correct with explicit white composite or bypass policy |
| Pure black/white page classified as color | Correct |
| Lossy WebP Q80 default | Do not inherit; start lossless for validation |
| Production inference precision | FP32 ONNX Runtime CUDA; TF32 disabled |
| Resident model policy | Keep all seven sessions in one worker; serialize GPU inference |
| Exception-driven tile sizing | Replace with a measured deterministic default plus safe fallback |

## Step 2 acceptance result

All requested parts of the normal black-and-white inference path have been identified down to their implementing files and behaviors. No MangaJaNai-specific tiling modification needs to be ported. The C# implementation needs to reproduce a small set of wrapper policies around a standard RRDB/ESRGAN graph, while deliberately correcting four decoder/preprocessing edge cases: EXIF orientation, RGB luminance coefficients, alpha handling, and all-black/all-white detection.

The next investigation step is Suwayomi's exact serve-conversion HTTP contract.
