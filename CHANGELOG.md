# Changelog

All notable changes to KomaScaler.Net are documented here. Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Simplified production inference to mandatory TensorRT with one active routed
  model/session and one GPU Run; removed configurable CUDA EP fallback,
  production session pools, and tile parallelism.
- Enabled Meziantou.Analyzer 3.0.123 centrally for every production and test
  project and applied its semantics-preserving culture, comparison, async,
  locking, layout, and API-usage recommendations.
- Reorganized scripts and documentation by role, removed generated/raw evidence
  and duplicate configuration, and added a local native-independent verification
  entry point. Hosted automation is deliberately omitted.
- Corrected Suwayomi's empty `downloadConversions` value to the required HOCON
  object (`{}`) and added a regression check.

### Added

- MIT licensing for KomaScaler.Net's original source, public-readiness README,
  support policy, third-party notices, model hash list, and repository-readiness
  checklist.

## [1.3.0] - 2026-07-20

### Changed

- Revisited the CUDA-derived tile limit under active-model-only TensorRT pool
  1 and selected core 832, context 64, with an isolated opt/max 960 engine
  profile after sequential 20-page latency, quality, RAM/VRAM, and lifecycle
  acceptance.
- Made TensorRT optimization-profile extents configurable, validated through
  960, and part of provider, engine, and application cache identity.
- Made all-seven TensorRT preloading mandatory before production HTTP
  readiness, prohibited request-time engine construction after startup, and
  extended systemd's first-build startup window to 15 minutes.
- Made installation authentication explicit: authenticated installs securely
  generate or preserve a root-only systemd credential, while explicit
  tokenless installs omit `LoadCredential` and remove stale token files.

### Validation

- Compared 320/32, 512/64, 576/64, 704/64, and 832/64 with exact representative
  tile counts, per-Run timings, persistent/peak VRAM, memory slope, untiled CUDA
  quality references, cold/cached startup, and profile-specific model switches.
- Built, inventoried, and routed all seven exact 8/960/960 engines sequentially
  under watchdogs; the empty-cache build completed in 605.982 seconds.
- Added isolated installation coverage for authenticated token generation and
  preservation plus tokenless unit rendering and credential removal.

## [1.2.0] - 2026-07-20

### Added

- Optional, acceptance-gated ONNX Runtime TensorRT backend with conservative
  FP32 provider settings, isolated engine caches, exclusive session leasing,
  lazy selected-model pool expansion, mandatory node-placement profiling, and
  CUDA fallback for unsupported subgraphs.
- Disk-backed TensorRT preflight, lifecycle acceptance, correctness, and
  sustained CUDA/TensorRT benchmark harnesses with hard resource watchdogs.

### Deployment

- The installer creates `/var/cache/komascaler/tensorrt`; systemd explicitly
  disables TF32 process-wide. The measured production profile is TensorRT
  active-model-only pool 1; CUDA remains a supported explicit alternative.

## [1.1.0] - 2026-07-20

### Added

- Configurable bounded intra-image GPU tile parallelism while retaining one
  queued image conversion at a time.
- Target-host tile-size, parallelism, untiled-reference quality, seam-error,
  context, timing, RAM, and VRAM benchmark coverage.
- Configurable lossless `png` and `webp` output encoders and an isolated
  3320×2800 grayscale encoding benchmark.
- Production deployment and operation guidance.

### Changed

- Selected GPU tile parallelism 8 with the established VRAM-safe 320-pixel
  core and 32-pixel context production policy.
- Selected single-band PNG compression level 3 as the output default. It
  reduced measured median encode time from 2,445.254 ms to 221.115 ms and the
  localhost transfer-plus-decode proxy from 164.664 ms to 50.088 ms versus
  lossless WebP effort 2.
- Made output format and all encoder parameters part of cache identity, made
  cache filenames format-neutral, and derive cached response MIME from bytes.
- Raised only the validation ceiling for experimental quality measurements;
  production tile dimensions remain unchanged.

### Validation

- Compared tile cores 320/384/448/512/576 and parallelism 1/4/6/8 with bounded
  sequential target-host runs and continuous resource sampling.
- Compared leading 512/576 candidates at contexts 32/48/64 against untiled
  direct CUDA Execution Provider output on landscape, portrait, and square
  inputs. Quality evidence retained the production 320/32 policy.
- Verified all PNG/WebP candidates losslessly reproduce the common grayscale
  pixels. PNG remains one-band through decode; tested WebP expands to RGB.
- Live Suwayomi serve-conversion verification remains operator-deferred; the
  running instance was inspected read-only and left unchanged.

## [1.0.0] - 2026-07-20

- Initial production implementation and target-host acceptance release.
