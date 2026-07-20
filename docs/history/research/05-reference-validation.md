# MangaJaNai × Suwayomi — Step 5: Reference Validation

Investigation date: 2026-07-20
Status: FP32 ONNX CUDA accepted for production; all-seven-session RTX 3060 memory, warm latency, and worker-release gates passed

## Outcome

The exported ONNX graph is faithful to the reconstructed PyTorch/Spandrel graph in FP32. All 19 representative FP32 cases passed strict shape, range, numerical, level, halftone, and structural checks.

The decisive FP16 CUDA comparison has now run on the target RTX 3060. All 91 cases used PyTorch FP16 CUDA as the reference and ONNX Runtime FP16 CUDA as the candidate. Every output had the exact expected shape, finite values, and `[0,1]` range, and all seven ONNX sessions used `CUDAExecutionProvider`. However, only 23 of 91 cases passed the predeclared numerical gate. Worst maximum error was `0.6218262`, worst MAE was `0.0121682`, minimum PSNR was `31.4867 dB`, and minimum SSIM was `0.9959154`.

Native candidate/reference images are visually very close, but the `32×` difference images show structured changes in gradients, dot phase, and detailed screentone regions. The gate is therefore not being weakened after seeing the result: FP16 is not accepted as the production-fidelity format.

The corrected FP32 CUDA-to-CUDA run passed all 91 cases without changing the predeclared thresholds. Worst maximum error was `0.0002120`, worst MAE was `0.0000040405`, minimum PSNR was `101.8087 dB`, minimum SSIM was `0.99999976`, maximum black/white-region MAE was `0.0000005539`, and maximum halftone-energy relative error was `0.0000002518`. Every model used `CUDAExecutionProvider`; the PyTorch reference and ONNX candidate both ran in FP32 CUDA with TF32 disabled.

The FP32 hardware probe also accepted the all-resident worker design. Seven sessions loaded in `4.16 s` total and increased GPU usage by `1076.06 MiB`. After all sessions warmed at 256×256, total usage was `4220.75 MiB`, or `3840.06 MiB` above the `380.69 MiB` baseline. Median warm tile latency was `92.43–93.21 ms` across the seven models. After the probe process exited, all twenty samples returned exactly to the `380.69 MiB` baseline.

These results separate graph correctness from precision/runtime behavior:

- FP32 proves that graph reconstruction, padding, pixel unshuffle, clamp, crop, dynamic dimensions, and ONNX conversion are correct.
- FP16 CUDA results prove the GPU path is operational and structurally stable, but do not meet the frozen numerical threshold.
- FP32 CUDA passes the frozen fidelity gate and is the production precision.
- One worker can keep all seven FP32 models resident while preserving substantial headroom on the 12 GiB RTX 3060.
- Process termination is measured to release the worker's CUDA allocations completely.
- The project can continue to Step 6 model selection with precision and model residency frozen; Step 7 still owns full-page tiling, seam, and page-latency validation.

## Validation tool

`mangajanai_reference_validate.py`:

- verifies every official source-model SHA-256;
- loads the exact Spandrel `0.4.1` reference architecture;
- supports FP32 and true FP16 PyTorch references;
- supports ONNX Runtime CPU or CUDA providers;
- applies one shared preprocessing path to both runtimes;
- calculates shape, finite/range, MAE, max error, error percentiles, PSNR, and windowed SSIM;
- measures black/white-region error and halftone high-frequency-energy error;
- writes input, reference, candidate, and amplified-difference PNGs;
- emits a contact sheet and machine-readable JSON;
- can run one shared fixture per model, the full category suite on one model, or all models × all fixtures.

The shared preprocessing used in these model-parity tests is:

```text
EXIF transpose
  → RGB decode
  → corrected sRGB luminance coefficients
  → normalize to [0,1]
  → repeat luminance to three channels
  → identical tensor sent to both runtimes
```

The correction of the legacy red/blue luminance swap is intentional and was established in Step 2. Automatic-level behavior is kept outside this comparison so it cannot hide or manufacture a runtime difference; both runtimes receive byte-for-byte identical model input.

## Fixtures

The deterministic suite covers:

| Fixture | Brief category exercised |
|---|---|
| `high-contrast-line-art` | Pure black/white line art and multiple line widths |
| `screentone-heavy` | Multiple dot frequencies and a dense transition band |
| `grayscale-gradient` | Smooth gradients and precision-sensitive reconstructed dots |
| `noisy-jpeg-scan` | Noise plus JPEG quality-24 artifacts |
| `visually-monochrome-rgb` | RGB source with small channel differences |
| `landscape-double-page` | Odd-width landscape spread and center gutter |
| `odd-complete-page` | Odd portrait dimensions and mixed page structures |
| `small-odd-image` | Near-minimum odd dimensions |
| Four official-demo crops | Real manga line art, blur, JPEG residue, and screentones |
| Official-demo composite | Bounded complete-page surrogate |
| `all-model-mixed-tile` | Common structured input for all seven weights |

The real-content fixtures come from the repository's official [`mangajanaiv1demo.webp`](https://github.com/the-database/MangaJaNai/blob/main/mangajanaiv1demo.webp), SHA-256 `455169da3c0efef7d5ae12d19978db3cd7df34c9db76c1decb6bb1d551c0f4fc`. The project describes the models as restoring degraded halftone dots and removing JPEG/moiré artifacts, so those are explicit validation targets rather than generic image metrics alone ([MangaJaNai README](https://github.com/the-database/MangaJaNai)).

No redistributable corpus of original complete pages was found in the official repositories. The demo composite is therefore a bounded real-content surrogate, not a substitute for a user-owned real-page corpus on the target machine.

## Acceptance thresholds

Every case must first satisfy these hard invariants:

- exact output shape `1 × 3 × 2H × 2W`;
- finite reference and candidate values;
- candidate range entirely within `[0,1]`.

### FP32 graph/export gate

| Metric | Requirement |
|---|---:|
| Maximum absolute error | ≤ `0.0003` |
| Mean absolute error | ≤ `0.00001` |
| PSNR | ≥ `90 dB` |
| SSIM | ≥ `0.99999` |
| Black/white-region MAE | ≤ `0.00002` |
| Halftone energy relative error | ≤ `0.0001` |

### FP16 production gate

| Metric | Requirement |
|---|---:|
| Maximum absolute error | ≤ `0.005` |
| Mean absolute error | ≤ `0.001` |
| PSNR | ≥ `50 dB` |
| SSIM | ≥ `0.999` |
| Black/white-region MAE | ≤ `0.0015` |
| Halftone energy relative error | ≤ `0.02` |

The FP16 comparison must use the same effective precision on both sides. Comparing ONNX FP16 directly to PyTorch FP32 measures the model family's sensitivity to reduced precision, not conversion fidelity. Maximum error is retained as a safety bound, but MAE, PSNR, SSIM, levels, and halftone structure are more representative for dot-pattern outputs because a small phase change can create a large single-pixel difference.

## Results

### FP32 reference versus FP32 ONNX

Nineteen of nineteen cases passed.

| Observed worst case | Result | Required |
|---|---:|---:|
| Maximum absolute error | `0.0002262592` | ≤ `0.0003` |
| Mean absolute error | `0.0000056451` | ≤ `0.00001` |
| Minimum PSNR | `98.8334 dB` | ≥ `90 dB` |
| Minimum SSIM | `0.99999994` | ≥ `0.99999` |
| Maximum black/white-region MAE | `0.0000009328` | ≤ `0.00002` |
| Maximum halftone energy relative error | `0.0000001629` | ≤ `0.0001` |

This is sufficient to accept the ONNX graph/export path.

### FP16 PyTorch CPU versus FP16 ONNX Runtime CPU

Four of nineteen cases met the strict production thresholds. All nineteen met the hard shape, finite, and range invariants, but this CPU comparison is not valid for final CUDA acceptance.

| Observed worst case | Result | Production requirement |
|---|---:|---:|
| Maximum absolute error | `0.0966797` | ≤ `0.005` |
| Mean absolute error | `0.0029133` | ≤ `0.001` |
| Minimum PSNR | `45.3554 dB` | ≥ `50 dB` |
| Minimum SSIM | `0.99978161` | ≥ `0.999` |
| Maximum black/white-region MAE | `0.0005683` | ≤ `0.0015` |
| Maximum halftone energy relative error | `0.0003989` | ≤ `0.02` |

The structural, black/white-level, and halftone measures pass. Peak/average numerical error and PSNR do not. The focused visual sheet shows the PyTorch and ONNX outputs plus a `32×` amplified absolute difference for the two most revealing fixtures.

### FP16-versus-FP32 precision sensitivity

A separate 91-case diagnostic ran all seven models against all thirteen fixture categories with PyTorch FP32 as the baseline and ONNX FP16 as the candidate. It is not a conversion acceptance test. Its purpose is to detect model/content combinations that are unusually sensitive to half precision.

The strongest sensitivity occurs on smooth gradients, especially for the 1920p and 2048p weights. Across this diagnostic, worst MAE was `0.02093`, minimum SSIM was `0.98860`, and maximum error was `0.73710`. Real halftone/line-art crops were generally much closer. This finding makes the CUDA FP16 comparison mandatory and argues against deleting FP32 deployment artifacts until the target test passes.

### FP16 PyTorch CUDA versus FP16 ONNX Runtime CUDA

The target-host run completed all seven models × thirteen fixtures in `88.50 s`. It is recorded in `20260720T102248Z-RETURN.zip`, SHA-256 `e186854f4cc423e45d2d44a75c3893f560d219b456529c96661c0d051657cb01`.

| Observed worst case | Result | Production requirement | Outcome |
|---|---:|---:|---|
| Maximum absolute error | `0.6218262` | ≤ `0.005` | Fail |
| Mean absolute error | `0.0121682` | ≤ `0.001` | Fail |
| Minimum PSNR | `31.4867 dB` | ≥ `50 dB` | Fail |
| Minimum SSIM | `0.9959154` | ≥ `0.999` | Fail |
| Maximum black/white-region MAE | `0.0021289` | ≤ `0.0015` | Fail in two cases |
| Maximum halftone energy relative error | `0.0010041` | ≤ `0.02` | Pass |

Pass/failure structure:

- exact shape, finite values, and range passed in all 91 cases;
- all halftone-energy checks passed;
- all seven models passed `high-contrast-line-art`, `landscape-double-page`, and `small-odd-image`;
- numerical checks failed in 68 cases, including every smooth-gradient and official-demo fixture;
- the largest errors were concentrated in gradient/dot reconstruction and fine face/screentone detail;
- SSIM remained high, so the result is visually plausible but is not numerically faithful enough for the frozen production gate.

The repeated cuDNN Frontend `No execution plans support the graph` messages were rejected candidate-plan diagnostics rather than inference failures. The validator produced every expected output and the separate probe completed successfully. Future tooling suppresses this nonfatal log flood while preserving thrown exceptions.

### Seven-session RTX 3060 probe

| Measurement | Result |
|---|---:|
| Baseline GPU usage | `380.69 MiB` |
| Usage after loading seven sessions | `1008.75 MiB` |
| Session-only resident increase | `628.06 MiB` |
| Usage after one 256×256 run per session | `4220.75 MiB` |
| Fully warmed increase over baseline | `3840.06 MiB` |
| Seven session-load time | `3.06 s` |
| First 256×256 runs, serialized | `0.73–0.89 s` each |

The run proves that one worker can keep all seven FP16 models resident simultaneously and execute them serially with substantial VRAM headroom. The per-run timings include first-shape cuDNN plan/workspace initialization and are not steady-state throughput measurements. Worker termination/VRAM release and full-page tiled latency remain Step 7 operational measurements.

## RTX 3060 FP32 acceptance result

The FP32 gate is complete. The definitive parity archive is `20260720T104702Z-RETURN.zip`, SHA-256 `3fe926b69fbbdfc9a365da63bef0607bb9eebda91748d888ee640951a07d2b4a`. It compares PyTorch FP32 CUDA against ONNX Runtime FP32 CUDA with TF32 disabled.

| Metric | Observed worst | Required | Outcome |
|---|---:|---:|---|
| Cases | `91/91` | `91/91` | Pass |
| Maximum absolute error | `0.0002120137` | ≤ `0.0003` | Pass |
| Mean absolute error | `0.0000040405` | ≤ `0.00001` | Pass |
| Minimum PSNR | `101.8087 dB` | ≥ `90 dB` | Pass |
| Minimum SSIM | `0.99999976` | ≥ `0.99999` | Pass |
| Maximum black/white-region MAE | `0.0000005539` | ≤ `0.00002` | Pass |
| Maximum halftone energy relative error | `0.0000002518` | ≤ `0.0001` | Pass |

All shape, finite-value, range, numerical, level, and halftone checks passed for every model/fixture pair. Smooth gradients remained the most sensitive content, as expected, but stayed within every frozen limit. Native and amplified-difference visual review found no production-significant dot-phase, edge-ringing, or level shift.

The hardware archive is `20260720T104145Z-RETURN.zip`, SHA-256 `388ce4adccee1845ca598822633bfa9853e0d6ed32577b8dd720a5cef786a254`.

| FP32 hardware measurement | Result |
|---|---:|
| Seven-session load time | `4.16 s` |
| Session-load resident increase | `1076.06 MiB` |
| Fully warmed resident increase | `3840.06 MiB` |
| Total GPU usage after warming | `4220.75 MiB` |
| Cold 256×256 shape initialization | `0.543–0.687 s` per model |
| Median warm 256×256 inference | `0.09243–0.09321 s` per model |
| VRAM after worker exit | exactly baseline in `20/20` samples |

This accepts FP32, all-seven-session residency, serialized inference, and disposable-process CUDA release. The measured fixed-tile latency is an input to Step 7; complete-page latency still depends on the final tile size, overlap, stitching, decode, and encode behavior.

The ONNX Runtime [CUDA Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) remains the authoritative compatibility reference for CUDA/cuDNN setup.

## Deferred validation that depends on later implementation

Two brief requirements cannot honestly be completed before tiled inference exists:

- full nominal-height pages below/above every selection boundary;
- tile-seam visibility and stitched-output parity.

Step 7 must feed its tiled candidate output back through this metric and visual-difference framework. The target corpus should include user-owned pages near 1200, 1300, 1400, 1500, 1600, 1920, and 2048 pixels high; pages below 1200 and above 2048; portrait and landscape spreads; noisy JPEGs; and screentone-heavy pages. Seam bands must be checked at every internal tile boundary against full-frame reference inference wherever full-frame inference fits.

## Step 5 decision

The reference-validation method, fixtures, metrics, tolerances, and reproducible tooling are defined. FP32 ONNX fidelity is accepted. FP16 CUDA execution and all-seven-session residency are operationally proven, but FP16 fails the frozen fidelity gate and is not accepted as the default production precision.

The production baseline is FP32 ONNX Runtime CUDA with TF32 disabled. Its target-host fidelity, seven-session residency, cold/warm fixed-tile execution, and process-termination VRAM release are accepted. A mixed-precision or FP16 fallback may be reconsidered only with explicit visual-quality acceptance and a documented reason that FP32 cannot meet the later full-page operational envelope.

Work can proceed to **Step 6: Determine the Model-Selection Rule**. Precision, CUDA runtime, and the all-resident single-worker policy are frozen. Step 7 seam, full-page, and end-to-end page-latency gates remain tracked pre-deployment requirements.
