# Final production architecture

This is the normative architecture. Dated reports under `docs/history/` and `docs/benchmarks/` explain how it was selected but do not override it.

## Topology

KomaScaler.Net is one ASP.NET Core/.NET 10 service. Suwayomi sends a synchronous multipart `POST /convert`; decode, color-page bypass, cache lookup, and encoding happen in-process. A bounded queue admits exactly one image conversion consumer. There is no worker process or IPC protocol.

Successful upscale, cache hit, byte-identical color bypass, queue/deadline fallback, and recoverable conversion failure return valid image bytes with HTTP 200. Once a valid source is captured, fallback preserves the exact original bytes. `/health/live` reports process liveness; `/health/ready` reports validated model/engine/provider readiness.

## Model and GPU lifecycle

The inventory contains seven dynamic-spatial FP32 NCHW opset-17 models with exact 2× output. Oriented source height selects the model; width affects tiling only. Every file is hash-verified.

Production uses ONNX Runtime 1.26 with TensorRT 10.14. Only `TensorrtExecutionProvider` is registered. Readiness/first-run profiling requires TensorRT nodes, rejects any CUDA-provider node, and rejects CPU `Conv`, `Gemm`, or `MatMul`. FP16 and INT8 are disabled; TF32 remains disabled in provider policy and `NVIDIA_TF32_OVERRIDE=0`. CUDA 12.9, cuDNN 9, and a compatible NVIDIA driver remain required TensorRT dependencies.

Exactly one model/session and one GPU `Run` may be active. A routed model switch serializes lifecycle work, drains the sole permit, fully disposes the previous pool, loads/warms the selected cached engine, and atomically publishes it. Old and new sessions never run concurrently. Idle unload, faults, cancellation cleanup, shutdown, and double disposal preserve the same drain-before-release contract.

All seven engine files must be prepared sequentially before readiness using input profile min/opt/max `8/960/960`. Engine identity includes model hash, ORT/TensorRT versions, precision and TF32 policy, profile, and engine-affecting provider options. HTTP requests cannot build a missing or incompatible engine. `PreloadOnStartup=true` is the production policy and may take about ten minutes on a cold deployment.

## Image pipeline

The endpoint bounds encoded upload size and decoded pixels, applies orientation and sRGB normalization, composites supported alpha over white, and detects color pages. Color pages return their original bytes. Monochrome pages use the selected model with a per-page immutable tiling/encoder snapshot.

The production tile policy is core `832`, context `64`; maximum submitted extent is `960`. Tiles run sequentially and blend in deterministic order. Output is single-band lossless PNG compression 3. Cache identity includes source hash, model hash, preprocessing/selection/tiling policy, fixed TensorRT pipeline identity, and encoder policy; this prevents collision with historical CUDA results.

Valid configuration reloads affect future pages. Invalid tiling reloads retain the last-known-good policy. Startup-only paths/provider properties are not hot-reloaded.

## Ownership boundaries

- `src/KomaScaler.Api`: ASP.NET host, endpoint, queue, TensorRT lifecycle, hosted services.
- `src/KomaScaler.Core`: model, image, tiling, cache, concurrency, and policy contracts/algorithms.
- `tests`: native-independent unit/integration tests plus isolated opt-in target-host GPU tests.
- `tools/model-export`: offline Python checkpoint inspection/export/validation only.
- `scripts`: installation, engine preparation, acceptance, and dated benchmarks.
- `deploy`: systemd and Suwayomi examples.

The production project references only `Microsoft.ML.OnnxRuntime.Gpu`; it never combines CPU and GPU ORT packages. The small direct CUDA EP helper in opt-in GPU tests exists solely to compare numerical correctness against the established reference and is not reachable from application configuration.
