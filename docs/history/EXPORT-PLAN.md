# Offline `.pth` to FP32 ONNX Export Plan

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Production does not load PyTorch checkpoints. Export is a maintenance task run
on the Debian GPU host where the official `.pth` files are already present.

## Inputs and outputs

Expected inputs are the seven official filenames and SHA-256 values in
`models/models.production.json`. The exporter reconstructs the common
old-style RRDBNet graph through Spandrel, loads the state dictionary under
restricted checkpoint handling, and exports one FP32 opset-17 ONNX file per
model.

Expected output filenames also appear in the production manifest. Do not
replace the manifest hashes until the complete parity suite passes.

## Environment

Use Python 3.10–3.12 in a dedicated virtual environment. The supplied scripts
and requirements are under `tools/model-export/`. CUDA/PyTorch and ONNX Runtime
must be a mutually compatible set; the accepted target run used CUDA 12.1,
cuDNN 9.1, PyTorch 2.4.1+cu121, and ONNX Runtime GPU 1.26.0.

```bash
cd tools/model-export
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
```

If PyTorch is installed separately for the host CUDA stack, do that before the
requirements file and verify `torch.cuda.is_available()`.

## Export

Place or point the scripts at the official `.pth` directory, then use the
supplied wrapper or inspect its help for explicit paths:

```bash
./export-fp32-models.sh /absolute/path/to/pth-models
```

The export contract is fixed batch 1, dynamic H/W, input/output names
`input`/`output`, FP32 NCHW, exact 2×, and graph-owned validity padding/crop.

## Validation gate

Run the full FP32 CUDA validation wrapper against the freshly exported files:

```bash
./run-fp32-validation.sh /absolute/path/to/pth-models
```

Require CUDA as the first provider and reject CPU fallback. Validate all seven
models across all 13 fixtures (91 cases), exact shapes, finite/range values,
MAE/max error, PSNR, SSIM, black/white preservation, and halftone energy using
the frozen Step 5 thresholds.

After parity passes:

1. calculate SHA-256 for every `.pth` and `.onnx` file;
2. compare source hashes to the official inventory;
3. update `models.production.json` only if the exporter output intentionally
   changed;
4. rerun the Step 7 tile/core/context benchmark when graph/runtime changes can
   affect CUDA memory, algorithms, or output;
5. archive environment, logs, JSON metrics, visual summary, and a manifest;
6. copy the seven ONNX files into the service's external model directory.

Any changed model, exporter, Spandrel/PyTorch/ONNX version, graph padding, or
precision reopens parity. Never infer that a successful export is production
equivalent without the CUDA validation gate.
