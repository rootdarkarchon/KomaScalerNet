# Runtime decision record

## Current decision

Production is one ASP.NET Core/.NET 10 application using ONNX Runtime 1.26's TensorRT Execution Provider. The accepted host stack is CUDA 12.9, cuDNN 9, TensorRT 10.14, libvips, and an NVIDIA driver compatible with that stack.

Inference remains FP32 with TF32 explicitly disabled. Production registers TensorRT only: it has no automatic CUDA EP fallback and rejects zero TensorRT compute placement or CPU `Conv`, `Gemm`, and `MatMul`. TensorRT nevertheless requires the NVIDIA driver and CUDA/cuDNN runtime.

The measured production policy is:

- one queued image conversion at a time;
- one active model/session and one GPU `Run`;
- drain and fully dispose the old model before a routed replacement;
- core/context `832/64`;
- TensorRT dynamic profile min/opt/max `8/960/960`;
- prebuilt persistent engines for all seven model identities before readiness;
- lossless single-band PNG, compression 3;
- TF32 disabled with `NVIDIA_TF32_OVERRIDE=0`.

`PreloadOnStartup=true` sequentially builds or validates all seven engine files, validates provider placement, then unloads. HTTP processing cannot initiate cold engine construction. During normal service only the selected model stays resident. Idle timeout, fault cleanup, switching, and shutdown drain the sole GPU permit before native disposal.

## Basis

Target-host Optimization 5 measurements found active-model-only TensorRT stable and materially faster end-to-end than the comparable CUDA EP configuration. Optimization 6 selected the latency/quality/memory knee at 832/64 with one Run. A representative 900×1291 page used four tiles and measured about 1.74 seconds tiled wall time on the RTX 3060. See the dated [production result](../benchmarks/OPTIMIZATION-6-PRODUCTION.md) and detailed [TensorRT report](../benchmarks/OPTIMIZATION-5-TENSORRT.md).

## Rejected production alternatives

- CUDA EP: retained only as isolated opt-in correctness-reference code in GPU tests; it is not configurable production behavior.
- CPU inference fallback: would conceal deployment failures and violate latency requirements.
- Parallel images or tiles: not selected; one TensorRT Run was the fastest safe measured production choice.
- FP16/INT8/TF32: outside the accepted FP32 quality policy. TF32 is a separate future experiment.
- Separate worker process or custom TensorRT wrapper: unnecessary for the proven single-process lifecycle.

Historical CUDA and concurrency measurements are retained under `docs/benchmarks/` and `docs/history/`; they are not current operator guidance.
