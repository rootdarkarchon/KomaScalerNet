# MangaJaNai × Suwayomi — Step 4: ONNX Export Proof

Status: export, RTX 3060 CUDA parity, seven-session residency, warm latency, and worker-release proof complete
Official model release: MangaJaNai `1.0.0`
Reference architecture runtime: Spandrel `0.4.1`

## Outcome

All seven official V1 2× `.pth` files were exported successfully to both FP32 and FP16 ONNX. Every graph passed ONNX validation, loaded in ONNX Runtime, ran with multiple dynamic spatial shapes, and produced the expected exact `2H × 2W` output.

The export path is proven. There are no custom operators or unsupported PyTorch fallbacks in the generated files. The production choice is one FP32, dynamic-spatial ONNX graph per model, with stable FP32 NCHW input and output. FP16 artifacts remain available but fail the stricter Step 5 target-host fidelity gate.

The export wrapper now includes the exact Spandrel descriptor boundary:

```text
FP32 RGB [1,3,H,W], range [0,1]
  → reflect-pad right/bottom to a multiple of 4
  → factor-2 pixel unshuffle
  → 64-feature / 23-block RRDB body
  → clamp to [0,1]
  → crop to [1,3,2H,2W]
```

This supersedes the provisional Step 2 suggestion to implement model-validity padding and clamp in C#. The ONNX graph can own those parity-sensitive operations; C# still owns page preprocessing, tile overlap, tile blending, grayscale restoration, and encoding.

## Delivered tools

- `mangajanai_onnx_export.py`: validates official source hashes and architecture, exports FP32/FP16, validates dynamic shapes, and writes `models.json`.
- `mangajanai-onnx-export-requirements.txt`: reproducible CPU export/reference environment.
- `mangajanai_onnx_gpu_probe.py`: keeps all seven precision-qualified sessions alive, measures session load time and VRAM with NVML, and measures cold and warm serialized CUDA tile inference.
- `mangajanai-onnx-gpu-probe-requirements.txt`: current CUDA probe dependencies.
- `mangajanai-step-4-models.json`: complete results, graph metadata, hashes, operator counts, timings, and per-shape parity measurements from this run.

The exporter uses Spandrel's restricted checkpoint unpickler rather than unrestricted `torch.load`, verifies the exact official SHA-256 values, and rejects an unexpected architecture, scale, channel count, parameter count, or size requirement.

## Reproducible export commands

```bash
python3 -m venv .venv-export
. .venv-export/bin/activate
python -m pip install -r mangajanai-onnx-export-requirements.txt

python mangajanai_onnx_export.py \
  /path/to/MangaJaNai-V1-models \
  ./onnx-models \
  --precision both \
  --provider cpu
```

For production FP32 files only:

```bash
python mangajanai_onnx_export.py \
  /path/to/MangaJaNai-V1-models \
  ./onnx-models \
  --precision fp32 \
  --provider cpu
```

FP16 files may still be generated with `--precision fp16` for diagnostics or a deliberately accepted fallback, but they are not the production baseline.

## Graph contract

| Property | Verified value |
|---|---|
| Opset | 17 |
| Batch | Fixed at 1 |
| Input name | `input` |
| Input shape | `[1, 3, height, width]` |
| Input type | FP32 for production graph |
| Input layout | NCHW |
| Input range | `[0,1]`, gamma-encoded sRGB |
| Spatial dimensions | Dynamic, each at least 4 |
| Output name | `output` |
| Output shape | `[1, 3, height×2, width×2]` |
| Output type | FP32 |
| Output range | Clamped to `[0,1]` |
| Internal edge handling | Reflect-pad right/bottom to multiple of 4; exact crop |
| Model file format | Single-file ONNX; no external tensor-data files |

Batch 1 is intentional because the production GPU lane is serialized and page tiles can have different shapes. Dynamic height and width avoid exporting a graph per tile size.

Fixed tile dimensions are not needed for ONNX Runtime CUDA. They may be worth a separate graph/profile only if TensorRT or CUDA Graph capture is adopted later.

## Generated operator set

Every FP32 graph has 1,269 nodes. Every FP16 graph has 1,275 nodes because the precision conversion adds six casts. All models share one identical operator-count signature.

The graph contains only the standard `ai.onnx` domain:

```text
Add, Cast, Clip, Concat, Constant, ConstantOfShape, Conv,
Gather, LeakyRelu, Mod, Mul, Pad, Reshape, Resize, Shape,
Slice, Squeeze, Sub, Transpose, Unsqueeze
```

There are no `aten`, contrib, custom-domain, Python, or Spandrel operators. Pixel unshuffle is represented using standard reshape/transpose operations. The RRDB nearest-neighbor upscales are standard `Resize` nodes.

## Numerical validation

Each of the seven graphs was tested at all of these shapes:

```text
[1,3,17,19] → [1,3,34,38]   # both dimensions odd
[1,3,16,20] → [1,3,32,40]   # aligned to multiple of four
[1,3,31,33] → [1,3,62,66]   # dynamic and odd
```

Inputs were deterministic random FP32 samples in `[0,1]`. The FP32 reference was the Spandrel 0.4.1 `ImageModelDescriptor` path, including its padding, clamp, and crop.

| Export | Worst maximum absolute error across all models/shapes | Worst mean absolute error | Result |
|---|---:|---:|---|
| FP32 ONNX vs PyTorch FP32 | `3.9190e-6` | below `4.0e-7` | Pass |
| FP16 ONNX vs PyTorch FP32 | `0.0034552` | `0.0002907` | Pass |

The FP16 comparison includes expected half-precision input, weight, activation, and output rounding. All 42 checks—seven models × three shapes × two precisions—passed.

ONNX Runtime `1.18.1` performed the original CPU export proof. The definitive target-host checks use ONNX Runtime `1.26.0`, CUDA `12.1`, and cuDNN `9.1` on the RTX 3060. FP16 CUDA completed but failed the production-fidelity gate; FP32 CUDA passed all 91 cases.

## Production FP32 artifacts

The seven FP32 files occupy approximately 449 MiB on disk. These target-generated artifacts produced the accepted CUDA parity and residency results:

| Model | FP32 ONNX SHA-256 |
|---:|---|
| 1200p | `001d55ee9bf9fe43cdcce817cc2d13fed30f9f95dcba502cf7f69384d1596474` |
| 1300p | `ac04401fb18715135166ca3a28cfb6e8e46d1bd6c3f0db95583a5a74baa6f8f1` |
| 1400p | `3d813490698f6208a1c0273b1c1170fad560eb3c3aaca9097578542c39521ec0` |
| 1500p | `c16a4416e0c509c0239a0c50f6b378142d1e2bdb6ab7622921449ff0e886a38d` |
| 1600p | `fca9f830f481ec43faff5076e3e9d678ad1c28ef037abdc74bc8e10c09919dd5` |
| 1920p | `5c73f796e98e6af3ca6a3ecbc583a2f47ec0252f291d51024164d6197f55da7b` |
| 2048p | `78e095c7c2af0c074e6ba56b6b7f0084cecd73153dca7d6908ba138c65999569` |

These hashes identify this exact exporter/toolchain output. Changing the exporter or converter versions requires regenerating hashes and rerunning parity.

## Load-time observations

On this CPU-only investigation host:

- restricted PyTorch/Spandrel checkpoint loading took 0.26–0.74 seconds per model;
- ONNX Runtime 1.18.1 FP16 session creation took 3.33–3.61 seconds per model;
- sequential creation of all seven FP16 CPU sessions took about 24.5 seconds;
- the complete seven-model FP32+FP16 export and validation run took about 198 seconds.

These are proof-run diagnostics, not RTX 3060 performance predictions. CUDA session creation includes graph optimization, CUDA allocator setup, and cuDNN algorithm/workspace choices and must be measured on the target host.

The cold worker should therefore be started proactively with the service or allowed to warm asynchronously. A first Suwayomi request must not wait past the Step 3 response deadline while seven sessions initialize; returning the original page during a missed cold-start deadline is preferable.

## CUDA and .NET deployment

The current C# package is [`Microsoft.ML.OnnxRuntime.Gpu`](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu). ONNX Runtime's [C# documentation](https://onnxruntime.ai/docs/get-started/with-csharp.html) supports creation of a CUDA-backed `InferenceSession`, and its [CUDA Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) documents the required CUDA/cuDNN major-version matching.

Recommended production baseline at implementation time:

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.26.0" />
```

Version `1.26.0` is the last ONNX Runtime line targeting CUDA 12; `1.27` moves to CUDA 13. Use the CUDA 12/cuDNN 9 runtime combination required by `1.26.0` and verify it against the installed NVIDIA driver. Do not install the CPU and GPU ONNX Runtime NuGet packages in the same application.

The exported operator set is conventional for CUDA EP: convolution, elementwise arithmetic, nearest-neighbor resize, reflection pad, reshape/transpose, slice, and shape arithmetic. The successful standard ONNX/ORT validation removes graph/export uncertainty, and the later target-host probe confirms that all seven graphs load and execute under `CUDAExecutionProvider` on the RTX 3060.

## RTX 3060 acceptance probe

The supplied probe was run on the Debian target with ONNX Runtime `1.26.0`, CUDA `12.1`, cuDNN `9.1`, and the NVIDIA `595.58.03` driver:

```bash
python3 -m venv .venv-gpu-probe
. .venv-gpu-probe/bin/activate
python -m pip install -r mangajanai-onnx-gpu-probe-requirements.txt

python mangajanai_onnx_gpu_probe.py \
  ./onnx-models \
  --tile 256x256 \
  --cudnn-algo HEURISTIC \
  --output gpu-probe-256.json
```

The probe:

1. creates all seven sessions sequentially and keeps them resident;
2. confirms that CUDA is the primary execution provider;
3. measures load time, resident VRAM delta, and sampled peak VRAM for every session;
4. runs one model at a time, matching the single serialized worker design;
5. verifies FP16 type, exact output shape, and finite output;
6. writes machine-readable results.

Measured result for all seven FP16 sessions plus one serialized 256×256 inference per session:

- all seven sessions loaded with CUDA as the primary provider;
- no unsupported-node, session-creation, CPU-fallback, shape, finite-value, or out-of-memory failure occurred;
- session creation took `3.06 s` total and increased GPU usage by `628.06 MiB`;
- after every session ran once and retained its 256×256 cuDNN workspace, GPU usage was `4220.75 MiB`, or `3840.06 MiB` above baseline;
- first-run tile times were `0.73–0.89 s` per model and include shape-specific plan/workspace initialization.

This accepts the all-resident FP16 memory design, although FP16 later failed the stricter Step 5 fidelity gate.

The subsequent FP32 probe also passed:

- all seven FP32 sessions loaded in `4.16 s` and increased residency by `1076.06 MiB`;
- after every session warmed at 256×256, GPU usage was `4220.75 MiB`, or `3840.06 MiB` above baseline;
- median warm inference was `92.43–93.21 ms` per 256×256 tile;
- after the probe worker exited, GPU usage returned exactly to the pre-probe baseline in all twenty samples;
- the corrected PyTorch FP32 CUDA versus ONNX Runtime FP32 CUDA validation passed all `91/91` cases under the frozen Step 5 thresholds.

This freezes FP32 ONNX Runtime CUDA, TF32 disabled, with all seven sessions resident in one serialized worker. Full-page tiled latency and seam behavior remain Step 7 measurements.

## `models.json` handoff

The generated metadata contains:

- source filename, size, SHA-256, nominal height, and selection range;
- ONNX filename, size, SHA-256, IR/opset, I/O types, node count, and operator counts;
- exact graph contract;
- package/runtime versions;
- per-shape output shape and numerical error;
- model/session/export timings;
- model license and conversion notice.

The C# service should use a trimmed deployment version of this file for selection and integrity checks. Validation-only timings and operator counts may remain in build artifacts.

## Step 4 acceptance result

The `.pth` → exact architecture → dynamic ONNX → FP16 conversion path is proven for all seven official models. Stable names/shapes, opset 17, dynamic height/width, output cropping, and numerical parity are established. The graphs also execute under the current ONNX Runtime CPU package.

Both hardware gates are complete. FP16 execution fits but fails the Step 5 production-fidelity threshold. FP32 passes `91/91` CUDA parity cases, keeps all seven sessions resident at `4220.75 MiB` total warmed GPU usage, delivers approximately `93 ms` warm 256×256 inference, and releases its CUDA allocations completely when the worker process exits. FP32 ONNX Runtime CUDA is the accepted production path; no graph redesign or alternative runtime is indicated.

The next investigation step is **Step 5: Reference Validation**: run representative manga pages through the converted models and record shape/range checks, MAE/max error, PSNR/SSIM, and visual differences—especially line art, screentones, gradients, and tile-boundary behavior. Model-selection thresholds follow in Step 6, with tiled-inference design in Step 7. The verified 25-second request deadline and single all-model worker remain constraints for those later runtime steps.
