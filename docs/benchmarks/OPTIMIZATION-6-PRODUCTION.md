# Optimization 6: final TensorRT production profile

> **Historical record:** This document records the accepted target-host outcome supplied after Optimization 5. It is not a live benchmark rerun or current operator guide; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20

## Accepted result

The production selection is ONNX Runtime TensorRT with active-model-only residency, one TensorRT session, and one GPU `Run`. The tile/profile knee is:

- core/context: `832/64`;
- TensorRT input profile min/opt/max: `8/960/960`;
- FP32, TF32 disabled;
- single-band lossless PNG compression 3;
- persistent prebuilt engines for all seven model identities;
- replace the resident model only after draining and disposal when routing changes.

The live Suwayomi `serveConversions` path completed acceptance. A representative 900×1291 page produced four tiles and measured approximately 1.74 seconds tiled wall time on the RTX 3060 host.

These numbers are retained measured findings from target-host acceptance, not measurements repeated during the later repository cleanup. Optimization 5 contains the detailed CUDA/TensorRT, quality, memory, engine-cache, model-switch, and sustained-test evidence leading to the selection.

## Consequence

CUDA EP is no longer a production backend or fallback. The production cache identity is TensorRT-only and cannot collide with historical CUDA results. CUDA runtime, cuDNN, and the NVIDIA driver are still required dependencies of TensorRT. Direct CUDA EP code remains only in isolated opt-in numerical-reference tests.
