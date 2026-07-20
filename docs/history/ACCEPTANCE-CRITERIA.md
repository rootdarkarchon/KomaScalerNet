# Acceptance Criteria

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Codex should convert this checklist into automated tests and an operator-facing
acceptance report. Items marked GPU require the Debian RTX 3060 and local ONNX
files; all others must run in native-independent local validation with a fake backend.

## Build and repository

- [ ] `dotnet restore`, Release build, formatter verification, and ordinary
  tests pass from a clean checkout.
- [ ] Production projects contain no Python runtime dependency and no separate
  inference-worker executable.
- [ ] ONNX model files, user pages, cache files, and secrets are Git-ignored.
- [ ] NuGet versions are centrally pinned and the CPU ONNX Runtime package is
  absent.

## Suwayomi HTTP contract

- [ ] `POST /convert` accepts multipart part `image` with arbitrary filename.
- [ ] Missing/wrong part, oversize upload, and auth failure are controlled.
- [ ] A supported fake-backend page returns a valid image at exact 2× size.
- [ ] Success, cache, bypass, deadline fallback, queue fallback, and recoverable
  conversion failure return HTTP 200 plus valid image bytes.
- [ ] No handled page path returns 204, JSON, or a text error body.
- [ ] Bypass/fallback bytes are byte-for-byte equal to the source.
- [ ] `X-Upscaler-Result` accurately identifies the result path.

## Image pipeline

- [ ] JPEG, PNG, WebP, BMP, and AVIF static images are independently detected
  and decoded when libvips supports them.
- [ ] EXIF rotations are physically applied before dimension/model selection.
- [ ] Embedded profiles are converted to sRGB.
- [ ] Alpha is composited over white for the monochrome path.
- [ ] Neutral RGB and pure black/white images are treated as monochrome.
- [ ] Meaningfully color images and animated/unsupported inputs are returned
  unchanged.
- [ ] Declared MIME and multipart filename cannot override decoded reality.
- [ ] Decoded pixel limits stop decompression bombs before large output
  allocation.

## Model metadata and routing

- [ ] Missing, extra, discontinuous, overlapping, or invalid selection bands
  fail inventory readiness.
- [ ] Missing or hash-mismatched ONNX files fail GPU readiness without making
  the pass-through HTTP path unavailable.
- [ ] Heights `1,1250` select 1200; `1251,1350` select 1300;
  `1351,1450` select 1400; `1451,1550` select 1500;
  `1551,1760` select 1600; `1761,1984` select 1920; and `1985,10000`
  select 2048.
- [ ] A rotated page routes by oriented height.
- [ ] A wide landscape page routes by height, not width/longest side.
- [ ] Selected model ID/hash and policy versions participate in cache identity.

## Tiling

- [ ] Balanced partitioning covers every source pixel exactly once as a core.
- [ ] Cores never exceed configured `MaximumCoreSize`.
- [ ] Context never reads beyond page edges and introduces no synthetic tiler
  padding.
- [ ] Default policy is 320 core and 32 pixels of context per side.
- [ ] Output is finite, clipped to `[0,1]`, and exactly `2H × 2W` for odd, even,
  small, portrait, and landscape inputs.
- [ ] Half-sine accumulation has no zero-weight output pixel and normalized
  weights are numerically stable.
- [ ] A valid configuration reload affects only pages that start after
  publication; a page cannot mix two geometries.
- [ ] Invalid reload retains last-known-good values and reports the error.
- [ ] Unsafe maximum extent over 384 is rejected by the RTX 3060 policy.

## Cache, queue, and concurrency

One queued image conversion at a time, with configurable bounded intra-image
GPU tile parallelism.

- [ ] A repeated identical request is served from persistent cache without a
  second inference.
- [ ] Concurrent identical requests execute exactly one conversion.
- [ ] Different source/model/tiling/preprocess/encoder values produce different
  keys.
- [ ] Cache writes are atomic; a killed or failed writer exposes no partial
  entry.
- [ ] Corrupt cache entries become misses and do not reach Suwayomi.
- [ ] The unique-work channel is bounded and queue-full behavior returns the
  original image.
- [ ] Fake-backend instrumentation proves active GPU Runs never exceed the
  configured intra-image tile factor and no second image is processed in
  parallel.
- [ ] Followers may time out independently without canceling a producer still
  needed by other followers.
- [ ] The 28-second response deadline returns the original rather than holding
  the HTTP connection indefinitely.

## Lifecycle and failure

- [ ] Production startup pre-warms one seven-session set before readiness;
  disabling pre-warm retains lazy initialization for diagnostic use.
- [ ] Model-height changes select a resident session without reinitialization.
- [ ] Core/context changes do not reinitialize sessions.
- [ ] Idle unload occurs only with no queued/active job, disposes sessions, and
  leaves ASP.NET live.
- [ ] A later request reloads sessions successfully.
- [ ] CUDA/OOM fault moves the GPU service to `Faulted`, blocks new Runs, and
  returns originals.
- [ ] Unverified in-process recovery exits for systemd restart rather than
  repeatedly running on corrupt state.
- [ ] Shutdown drains or cancels safely and disposes native resources.

## Observability and security

- [ ] Liveness does not require loaded models; readiness reports config,
  model/cache/codec, and GPU-service state accurately.
- [ ] Metrics cover result paths, cache, dedupe, queue, model load, GPU Run,
  tiling, encoding, latency, lifecycle, and faults without source-hash labels.
- [ ] Logs include correlation, dimensions, selected model, effective tiling,
  tile count, and timings but exclude image bytes and secrets.
- [ ] Default bind is loopback and the optional shared token uses constant-time
  comparison.
- [ ] systemd runs an unprivileged user, has only required filesystem access,
  and restarts one instance on failure.

## Target-host GPU acceptance

- [ ] GPU: ORT reports CUDA as first/active provider for all seven sessions.
- [ ] GPU: all production model SHA-256 values match the manifest.
- [ ] GPU: FP32 output remains inside the Step 5 parity gates on supplied
  validation fixtures.
- [ ] GPU: representative full pages for all seven routing bands and a spread
  produce exact 2× valid images.
- [ ] GPU: 320/32 with production tile factor 8 has no visible straight tile
  grid, preserves factor-1 bytes, and remains within the measured 12 GiB
  envelope.
- [ ] GPU: record cold load, warm page latency, queue wait, peak VRAM, cache-hit
  latency, idle disposal, reload latency, and process-exit VRAM.

## Operational end-to-end

- [ ] Suwayomi `server.serveConversions.default` sends a real page through the
  service and the reader displays the 2× result.
- [ ] The second request is a cache hit.
- [ ] Color/unsupported page remains readable and byte-identical.
- [ ] Simulated GPU failure still displays the original page.
- [ ] Service starts at boot on Debian 13 and survives an intentional restart.
