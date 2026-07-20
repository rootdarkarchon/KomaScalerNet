# KomaScaler.Net acceptance report

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20 (Europe/Zurich)

## Outcome

The .NET 10 production solution, ordinary automated test suite, publish output,
deployment artifacts, and operator documentation are complete. Formatting,
Release build, unit tests, integration tests, publish, liveness, missing-model
readiness behavior, and metrics smoke checks passed on this host.

Target-host GPU acceptance and the Suwayomi/systemd operational end-to-end are
not executed here. The seven licensed ONNX files are deliberately absent and
`nvidia-smi` cannot communicate with an active driver in this environment.
Those items remain operator acceptance gates and are not inferred from the
fake backend.

## Delivered implementation

- One ASP.NET Core executable named `komascaler`; no worker project or IPC.
- `POST /convert` with bounded multipart capture, optional fixed-time token
  authentication, exact-byte bypass/fallback, correlation and result headers.
- NetVips decode/orientation/sRGB/alpha/monochrome/auto-level/grayscale WebP
  pipeline with encoded-byte and decoded-pixel limits.
- Strict seven-model inventory/schema/range/hash validation and oriented-height
  routing; readiness remains false while exact-byte pass-through stays live.
- Immutable live tiling snapshots, balanced cores, bounded context, graph-owned
  edge validity, channel-0 half-sine blending, finite/range/exact-shape checks.
- Content-addressed atomic filesystem cache, validation/quarantine, cleanup,
  full policy identity, in-flight deduplication, and bounded unique-work queue.
- Lazy all-seven CUDA session set, CPU-EP fallback disabled, TF32 disabled,
  one complete-Run semaphore, idle drain/reload, fault disposal, and exit code
  70 for systemd restart after unverified CUDA state.
- Liveness/readiness/Prometheus-text metrics, structured logging, hardened
  systemd unit, install/acceptance scripts, and Suwayomi HOCON example.

## Versions

| Component | Version |
|---|---:|
| .NET SDK used | 10.0.302 |
| Target framework / ASP.NET runtime | net10.0 / 10.0.10 |
| Microsoft.ML.OnnxRuntime.Gpu | 1.26.0 |
| NetVips | 3.2.0 |
| Microsoft.AspNetCore.Mvc.Testing | 10.0.10 |
| Microsoft.NET.Test.Sdk | 18.8.1 |
| xunit | 2.9.3 |
| xunit.runner.visualstudio | 3.1.5 |

The publish dependency graph contains `Microsoft.ML.OnnxRuntime.Gpu` plus its
GPU platform and managed components. It contains no CPU ONNX Runtime package.

## Commands and results

Executed from the repository root with CLI/NuGet scratch directories redirected
to `/tmp` because the execution environment's home directory is read-only:

```bash
dotnet restore KomaScaler.Net.sln
dotnet format KomaScaler.Net.sln --verify-no-changes --no-restore
dotnet build KomaScaler.Net.sln --configuration Release --no-restore
dotnet test KomaScaler.Net.sln --configuration Release --no-build
sh scripts/acceptance/acceptance.sh
dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj \
  --configuration Release --no-restore --output /tmp/komascaler-publish
```

Results:

- Restore: passed.
- Formatter verification: passed.
- Release build: passed with 0 warnings and 0 errors.
- Unit tests: 38 passed, 0 failed, 0 skipped. Native NetVips tests exercised
  JPEG, PNG, WebP, and BMP decoding plus lossless WebP encoding.
- Integration tests: 9 passed, 0 failed, covering multipart, arbitrary
  filename/MIME, auth, exact bypass/fallback, fake 2x success, persistent cache,
  concurrent deduplication, response deadline, recoverable inference failure,
  and liveness.
- GPU tests: 1 skipped with the prerequisite message; no GPU result claimed.
- Publish: passed and included the seven-model manifest and CUDA provider native
  library; no ONNX weights were included.
- Published-host smoke: `/health/live` returned 200; `/health/ready` returned
  the expected 503 listing all seven absent models; `/metrics` returned the
  Prometheus-compatible queue/in-flight/GPU-state series.

The workspace mount does not permit direct executable launch even with the
executable bit, so the acceptance script was invoked through `sh`. This is a
property of this workspace mount, not the deployed Debian filesystem.

## Not executed and remaining risks

- GPU: actual CUDA-provider session load, all seven hashes/sessions, real graph
  output/parity, serialized concurrent Runs, 320/32 VRAM, latency, idle unload,
  reload, fault recovery, and process-exit VRAM. Reason: ONNX files are absent
  and the driver is inactive.
- Native AVIF decode, EXIF-oriented routing fixture, and embedded ICC fixture.
  The implementation uses libvips for these paths, but suitable fixtures/plugin
  coverage was not available in this run.
- Real-page visual seam review and parity gates. These require operator-owned
  pages and the licensed models.
- Suwayomi reader/cache/failure end-to-end and Debian systemd boot/restart.
  These require the operator's running Suwayomi and root service installation.
- Metrics expose core result/cache/queue/lifecycle state, but target-host GPU
  memory and fine-grained CUDA timing collection still depends on the target
  acceptance run.

## Target-host acceptance command

After installing CUDA 12, cuDNN 9, libvips, and copying the models:

```bash
export KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models
sh scripts/acceptance/acceptance.sh
curl --fail http://127.0.0.1:9999/health/ready
curl --fail -F 'image=@/path/to/operator-owned-monochrome-page.jpg' \
  -H 'X-Upscaler-Token: your-token' http://127.0.0.1:9999/convert \
  --output /tmp/komascaler-result.webp
```

Then enable the supplied Suwayomi `server.serveConversions.default` block and
verify a first request is `upscaled`, the second is `cache`, a color page is
byte-identical, and a simulated stopped/faulted GPU still serves the original.

## Smallest next operator action

Copy `models/models.production.json` and the seven hash-matching FP32 `.onnx`
files into `/var/lib/komascaler/models`, then run the target-host acceptance
command above on the RTX 3060.
