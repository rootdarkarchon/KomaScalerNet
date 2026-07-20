# Repository guidance

Maintain the application specified by `docs/design/FINAL-ARCHITECTURE.md`.
Documents under `docs/history/` and `docs/benchmarks/` are evidence, not current
production instructions.

## Non-negotiable behavior

- Production is one ASP.NET Core application; do not create a separate worker
  project or IPC protocol.
- Use C# for production. Keep Python under `tools/model-export` only.
- Use only `Microsoft.ML.OnnxRuntime.Gpu`; never add the CPU ONNX Runtime
  package to the same application.
- Require TensorRT compute placement. Do not register automatic CUDA EP
  fallback or silently fall back to CPU inference.
- Process one queued image and one GPU tile Run at a time. HTTP, decode, and
  encoding work may remain concurrent.
- Keep exactly one routed model/session resident. Drain and dispose it before
  loading a replacement; keep all seven compatible engines cached on disk.
- Snapshot one immutable tiling policy per page. Valid `appsettings` reloads
  affect future pages; invalid reloads retain the last-known-good policy.
- Always return valid image bytes with HTTP 200 for success, bypass, queue
  fallback, deadline fallback, and recoverable conversion failure.
- Preserve exact original input bytes when returning the original.
- Include model hash, preprocessing/selection/tiling policy, and encoder policy
  in cache identity.
- Never commit model weights, ONNX files, cache contents, secrets, or local
  images unless they are explicit redistributable test fixtures.

## Engineering rules

- Prefer small cohesive types and dependency injection over static global
  state. The GPU singleton and in-flight deduplication registry are intentional
  singleton state.
- Use cancellation tokens for waiting and queued work. Do not assume a GPU
  `Run` can be safely preempted once started.
- Dispose every `OrtValue`, binding, decoded image, pooled buffer owner, stream,
  and session deterministically.
- Use bounded queues and bounded decoded dimensions. Never buffer an
  unvalidated unbounded upload.
- Use atomic cache writes in the destination filesystem.
- Log structured fields without logging image bytes, auth tokens, or uploaded
  filenames as trusted identifiers.
- Keep public methods testable through interfaces where a fake inference
  backend can prove queueing, deduplication, fallback, and serialization.

## Verification after each milestone

Run formatting, build, unit tests, and relevant integration tests. For native
changes, also run the opt-in TensorRT tests on the RTX 3060. Do not
claim GPU acceptance from fake-backend tests.

Expected baseline commands:

```bash
dotnet format --verify-no-changes
dotnet build --configuration Release
dotnet test --configuration Release
```

Before finishing, run the acceptance script or equivalent documented commands,
inspect the generated diff, and update deployment documentation for every new
setting or operational dependency.
