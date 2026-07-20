# MangaJaNai × Suwayomi On-Demand Upscaling
## Research and Architecture Brief for ChatGPT Work

## Objective

Turn the current concept into a **verified, Codex-ready implementation plan** for a local, on-demand manga upscaling service.

The final production service should preferably be written in **C#/.NET**, run on Debian, use the server's NVIDIA RTX 3060, and integrate with Suwayomi's page conversion hook.

Do not build the full application yet. First verify the model details, existing preprocessing and tiling behavior, Suwayomi's exact HTTP contract, and the runtime architecture. Then produce a concrete implementation handoff for Codex.

---

## Intended Result

```text
Reader
  ↓
Suwayomi
  ↓
Suwayomi serve-conversion HTTP request
  ↓
Manga Upscaler API
  ├─ inspect image
  ├─ bypass unsupported or unnecessary images
  ├─ select appropriate MangaJaNai model
  ├─ check persistent cache
  ├─ deduplicate concurrent requests
  └─ queue uncached inference
          ↓
     GPU inference worker
       ├─ lazily load model
       ├─ perform tiled 2× inference
       ├─ stitch output
       ├─ encode result
       └─ return/cache result
```

When no new work has arrived for approximately 5–10 minutes:

```text
queue empty
+ no active inference
+ idle timeout elapsed
    ↓
dispose the model/session or terminate the inference worker
    ↓
release GPU memory
```

If upscaling fails, the service should return the original image rather than break reading.

---

## Environment and Preferences

- Host OS: Debian/Linux
- GPU: NVIDIA GeForce RTX 3060
- Main implementation language: C#
- Preferred framework: ASP.NET Core on a current supported .NET release
- Suwayomi and the upscaler run locally or on the same LAN
- Target upscale factor: 2×
- GPU is mostly idle
- Python in production is strongly undesirable
- A one-time Python script for `.pth → ONNX` export and reference validation is acceptable
- Persistent disk caching is expected
- Primary use case: on-demand reading, not pre-upscaling the entire collection

---

## Current Technical Direction

The expected production path is:

```text
Official MangaJaNai .pth weights
    ↓
one-time export script
    ↓
validated ONNX models
    ↓
C# service using ONNX Runtime CUDA
```

Do not spend excessive time debating whether `.pth → ONNX` is theoretically possible. For an ESRGAN/RRDB-style architecture this should be a normal one-time engineering task.

The important questions are:

1. What exact architecture do the MangaJaNai models use?
2. What exact preprocessing and postprocessing are required?
3. How does model selection work?
4. What tiling and overlap behavior is required to avoid seams?
5. Does ONNX output numerically and visually match the reference runtime?
6. What exact request/response contract does Suwayomi expect?

TorchSharp, LibTorch, TensorRT, or another runtime should remain fallback options if ONNX export or execution exposes a concrete problem.

---

## Assumptions That Must Be Verified

These are working assumptions, not established facts:

- MangaJaNai provides several 2× `.pth` models for different approximate source page heights.
- The models are ESRGAN/RRDB-family networks.
- MangaJaNaiConverterGui uses a modified chaiNNer backend.
- chaiNNer or its supporting libraries reconstruct the network architecture from the `.pth` weights.
- The MangaJaNai `.pth` files are probably state dictionaries or checkpoint dictionaries.
- The 2× models likely share one graph and differ primarily in trained weights.
- Model selection is based mainly on source image height.
- Full-page inference should use tiles or strips with overlap.
- The production service can use ONNX Runtime CUDA from C#.
- Model unload from an in-process runtime may not release all CUDA allocations immediately; a disposable worker process may be preferable.

---

# Required Investigation

## 1. Inspect the Official MangaJaNai Models

Obtain the current official MangaJaNai 2× model files.

For each model, determine:

- filename;
- intended source resolution/page height;
- file size;
- top-level checkpoint structure;
- whether it contains:
  - a plain `state_dict`,
  - a checkpoint dictionary,
  - a complete pickled model,
  - or another structure;
- tensor data types;
- input/output channels;
- scale factor;
- feature count;
- block count;
- architecture family;
- whether all 2× models share an identical graph;
- licensing and redistribution requirements.

Produce a compact model inventory table.

Also inspect the state-dictionary keys and tensor shapes. Record enough information to reconstruct the network for export without depending on the complete MangaJaNai application.

---

## 2. Trace the Existing Inference Pipeline

Inspect the official MangaJaNai, MangaJaNaiConverterGui, and relevant chaiNNer/backend repositories.

Determine the exact normal black-and-white manga inference path:

- image decoding;
- EXIF orientation;
- alpha handling;
- grayscale handling;
- RGB/BGR order;
- integer-to-float conversion;
- input range;
- normalization;
- padding;
- tile size;
- tile overlap;
- edge tile handling;
- model selection;
- output clipping;
- output range conversion;
- grayscale restoration;
- output encoding.

Find the implementation, not merely README claims.

Record exact repository files, classes, functions, and commits where practical.

Particular attention should be paid to any MangaJaNai-specific modifications to chaiNNer. Determine whether those modifications affect:

- model selection;
- preprocessing;
- tiling;
- postprocessing;
- memory management;
- or only the GUI/workflow.

---

## 3. Verify Suwayomi's Conversion Contract

Inspect the current Suwayomi Server documentation and implementation.

Determine exactly:

- the configuration key for conversion while serving pages;
- request method;
- multipart field name;
- filename behavior;
- request headers;
- timeout behavior;
- whether extra form fields are sent;
- how response MIME type is handled;
- whether response filename or extension matters;
- what happens on:
  - non-2xx responses,
  - invalid images,
  - timeouts,
  - unchanged/original-image responses,
  - changed output format;
- whether conversion applies to downloaded pages;
- whether Suwayomi caches converted responses;
- likely page-prefetch concurrency.

Produce a verified minimal configuration example.

---

## 4. Prove the ONNX Export Path

Create or specify a one-time export workflow:

```text
.pth
  ↓
instantiate exact PyTorch architecture
  ↓
load state dictionary
  ↓
export ONNX
  ↓
optionally convert graph/weights to FP16
  ↓
validate against reference output
```

Verify:

- correct opset;
- dynamic height and width support;
- whether fixed tile dimensions are preferable;
- CUDA Execution Provider compatibility;
- unsupported operators;
- FP16 behavior;
- model load time;
- inference memory usage;
- whether output tensor names and shapes are stable;
- whether all resolution-specific models can use the same export code.

A simple export script is sufficient. It does not need to become part of the production service.

Deliver:

- proposed export script structure;
- required Python packages;
- expected command line;
- expected output files;
- model metadata file format, if needed.

Example desired output:

```text
models/
  mangajanai-1200-2x.onnx
  mangajanai-1300-2x.onnx
  mangajanai-1400-2x.onnx
  mangajanai-1500-2x.onnx
  mangajanai-1600-2x.onnx
  mangajanai-1920-2x.onnx
  mangajanai-2048-2x.onnx
  models.json
```

---

## 5. Define Reference Validation

A plausible-looking image is not sufficient proof.

Compare the same decoded input against:

```text
reference PyTorch/chaiNNer path
              vs
candidate ONNX Runtime path
```

At minimum compare:

- output shape;
- min/max values;
- mean absolute error;
- maximum absolute error;
- PSNR;
- SSIM;
- visual difference image;
- black and white level preservation;
- halftone reconstruction;
- tile seam visibility.

Test:

- individual tiles;
- complete pages;
- grayscale pages;
- RGB pages that are visually monochrome;
- noisy JPEG scans;
- high-contrast line art;
- screentone-heavy pages;
- double-page spreads;
- odd dimensions;
- images smaller and larger than the nominal model heights.

The validation plan should define an acceptable numerical tolerance, especially for FP16 execution.

---

## 6. Determine the Model-Selection Rule

Identify MangaJaNai's actual model-selection logic.

Questions:

- Is selection based on height, width, longest side, DPI, or another metric?
- Are exact thresholds hardcoded?
- Are thresholds simple midpoints between trained resolutions?
- Is the input resized before inference?
- How are unusually small or unusually large pages handled?
- How are landscape double-page spreads handled?
- What happens when no model is an obvious match?

If the existing logic is unnecessarily coupled to the original application, define a simpler rule that preserves expected quality.

A likely fallback rule is nearest target source height:

```text
1200
1300
1400
1500
1600
1920
2048
```

Do not finalize this rule without checking the original implementation and testing representative pages.

---

## 7. Define Tiled Inference

Design a tiled inference algorithm suitable for a 12 GB RTX 3060.

Determine experimentally:

- suitable tile size;
- overlap/padding size;
- whether reflective, replicated, or zero padding is used;
- output crop rules;
- stitching behavior;
- peak VRAM usage;
- throughput;
- seam visibility.

Potential starting values:

```text
input tile: 256×256 or 384×384
overlap:    16–32 input pixels
precision:  FP16
GPU jobs:   one at a time
```

The final recommendation must be based on measured output and VRAM behavior.

Prefer fixed tile dimensions if that materially improves ONNX Runtime or future TensorRT behavior.

---

# Proposed Production Architecture

## API Process

Responsibilities:

- expose Suwayomi-compatible HTTP endpoint;
- validate multipart request;
- read image metadata;
- decide whether to bypass;
- select model;
- compute cache key;
- serve cache hits;
- deduplicate in-flight requests;
- queue GPU work;
- return original input on failure;
- expose health and metrics endpoints;
- manage worker lifecycle.

Candidate project:

```text
MangaUpscaler.Api
```

---

## Inference Worker

Responsibilities:

- own ONNX Runtime CUDA session;
- lazily load selected model;
- run tiled inference;
- stitch results;
- encode output;
- report structured errors;
- unload after idle timeout.

Candidate project:

```text
MangaUpscaler.Worker
```

A separate process is preferred initially because process termination guarantees release of the CUDA context and associated VRAM. Work should evaluate whether this is worth the additional IPC complexity.

Possible IPC options:

- localhost HTTP;
- Unix-domain socket;
- named pipe;
- simple length-prefixed binary protocol.

For the first implementation, localhost HTTP or a Unix-domain socket is acceptable. Avoid unnecessary custom protocol work unless benchmarks justify it.

---

## Model Lifecycle

Expected behavior:

```text
first uncached request
  ↓
start worker if absent
  ↓
load selected model
  ↓
process queued pages
  ↓
reuse same model while requests continue
```

When a request needs a different model:

```text
finish active inference
  ↓
dispose current session
  ↓
load requested model
  ↓
continue queue
```

Initial implementation should keep only one model loaded at a time unless measurements show that keeping two or more models resident is clearly beneficial.

Idle behavior:

- reset timeout after completed or newly accepted work;
- never unload during active inference;
- never unload while queued work exists;
- terminate worker after 5–10 minutes of true inactivity;
- restart transparently on the next request.

---

## Concurrency

The HTTP service may accept concurrent requests, but GPU inference should initially be serialized.

Recommended shape:

```text
HTTP handlers
  ├─ cache hit → immediate response
  ├─ identical in-flight request → await existing task
  └─ new cache miss → bounded channel
                           ↓
                   one GPU consumer
```

Use a bounded queue to prevent unlimited reader prefetch from consuming memory.

Define expected behavior when the queue is full:

- wait with timeout;
- bypass and return original;
- or reject with a controlled error.

For reading reliability, returning the original image is likely preferable.

---

## Cache

Use a persistent content-addressed cache.

Suggested key material:

```text
SHA-256(
    original image bytes
    + selected model file hash
    + scale factor
    + tile settings
    + preprocessing version
    + encoder settings
)
```

Example layout:

```text
cache/
  ab/
    cd/
      abcdef....webp
```

Requirements:

- atomic writes;
- no partially visible files;
- in-flight request deduplication;
- configurable maximum size;
- configurable expiry or LRU cleanup;
- cache versioning;
- metrics for hits, misses, failures, and bytes.

The cache should live in the API process or a shared library, not inside the disposable worker.

---

## Bypass Rules

Define conservative bypass behavior.

Potential bypass cases:

- unsupported or animated format;
- tiny thumbnail;
- page already above a configured target size;
- image decode failure;
- model unavailable;
- worker unavailable;
- queue timeout;
- GPU out-of-memory;
- color page when no color model is enabled.

For the first milestone, color images may be returned unchanged rather than incorrectly processed with a monochrome model.

Determine a robust color-detection method. Do not rely only on file format or channel count because many monochrome pages are stored as RGB.

---

## Output Encoding

Evaluate:

- lossless WebP;
- near-lossless WebP;
- high-quality lossy WebP;
- PNG;
- high-quality JPEG.

The purpose of MangaJaNai is partly to recover linework and halftones, so overly aggressive lossy output may negate some gains.

Start with lossless or near-lossless WebP and benchmark:

- encode time;
- file size;
- decoding compatibility;
- visual quality;
- Suwayomi/browser behavior.

---

# Non-Functional Requirements

## Reliability

- A failed conversion must not make the manga unreadable.
- Return the original image on recoverable failure.
- Worker crashes must not crash the API.
- One retry after worker restart is acceptable.
- Log model load, unload, inference, cache, and fallback events.

## Security

- Bind locally by default.
- Support an optional shared authorization token/header.
- Limit request size.
- Validate decoded dimensions to avoid decompression bombs.
- Do not trust uploaded filenames.
- Do not expose arbitrary file paths to the worker.

## Observability

Expose:

- `/health`;
- `/ready`;
- optional Prometheus metrics;
- active model;
- worker state;
- queue length;
- cache hit rate;
- average conversion time;
- model load time;
- GPU inference time;
- encode time;
- failures and fallbacks.

## Configuration

Use normal .NET configuration sources.

Suggested settings:

```text
ModelsPath
CachePath
IdleTimeout
QueueCapacity
MaxUploadBytes
MaxDecodedPixels
TileSize
TileOverlap
UseFp16
OutputFormat
OutputQuality
BypassHeight
AuthorizationToken
CudaDeviceId
```

---

# Benchmark Plan

Benchmark on the actual RTX 3060 server.

Use representative pages around:

- 1200 px high;
- 1400 px high;
- 1600 px high;
- 1920 px high;
- 2048 px high;
- double-page spreads;
- noisy scans;
- heavy screentones.

Measure:

- cold worker startup;
- cold model load;
- first-page latency;
- warm page latency;
- tiles per second;
- complete pages per minute;
- peak VRAM;
- CPU usage;
- input decode time;
- output encode time;
- cache-hit latency;
- worker shutdown and VRAM release.

The key practical question is whether prefetch plus serialized inference keeps ahead of normal reading.

---

# Expected Deliverables from ChatGPT Work

Produce the following before handing the task to Codex:

## 1. Verified Findings

A research report containing:

- exact model inventory;
- architecture details;
- preprocessing/postprocessing;
- model-selection logic;
- tiling behavior;
- Suwayomi request/response contract;
- licensing constraints;
- primary-source citations.

## 2. Runtime Decision

A concise decision record:

```text
Chosen runtime:
Why:
Rejected alternatives:
Known risks:
Fallback:
```

Expected default: C# + ONNX Runtime CUDA, unless concrete evidence argues against it.

## 3. Export Plan

A concrete `.pth → ONNX` export procedure and validation method.

## 4. Architecture Specification

A final component diagram and responsibilities for:

- API;
- cache;
- request deduplication;
- queue;
- worker lifecycle;
- inference;
- image codec layer;
- configuration;
- logging/metrics.

## 5. Codex Implementation Prompt

Generate a detailed prompt suitable for Codex that tells it exactly what repository to create, what milestones to implement, what packages to use, and what tests must pass.

## 6. Acceptance Criteria

At minimum:

- Suwayomi can send an image through `serveConversions`.
- A supported page is upscaled 2×.
- Correct model is selected.
- Repeated request is served from cache.
- Concurrent identical requests run one conversion.
- GPU inference is serialized.
- Worker/model loads lazily.
- Worker unloads after idle timeout.
- VRAM is released after worker termination.
- Conversion failure returns original image.
- ONNX output matches reference within defined tolerances.
- Tiled output has no visible seams.
- Service runs on Debian with the RTX 3060.

---

# Suggested Codex Milestones

Work should turn these into a final implementation plan.

## Milestone 1 — Repository Skeleton

```text
MangaUpscaler.sln
src/
  MangaUpscaler.Api/
  MangaUpscaler.Core/
  MangaUpscaler.Worker/
tests/
  MangaUpscaler.UnitTests/
  MangaUpscaler.IntegrationTests/
tools/
  model-export/
```

## Milestone 2 — Suwayomi-Compatible Pass-Through

- Accept multipart `image`.
- Decode and validate input.
- Return original image.
- Add health endpoint.
- Add integration test using an in-memory ASP.NET server.

## Milestone 3 — Model Metadata and Selection

- Load `models.json`.
- Inspect image dimensions.
- Select nearest valid model.
- Unit-test thresholds and landscape pages.

## Milestone 4 — ONNX Worker

- Start worker on demand.
- Load ONNX Runtime CUDA session.
- Run one fixed-size tile.
- Return tensor result.
- Validate against reference fixture.

## Milestone 5 — Tiling and Stitching

- Split page into overlapping tiles.
- Run serialized inference.
- Crop overlap.
- Stitch output.
- Test odd dimensions and seam behavior.

## Milestone 6 — Cache and Deduplication

- Content-addressed cache.
- Atomic writes.
- In-flight task deduplication.
- Cache tests.

## Milestone 7 — Idle Lifecycle

- Start worker lazily.
- Reuse active model.
- Switch models safely.
- Terminate after idle timeout.
- Verify VRAM release operationally.

## Milestone 8 — Fallback and Hardening

- Queue bounds.
- Request limits.
- Timeout handling.
- Worker crash recovery.
- Return-original fallback.
- Structured logs and metrics.

## Milestone 9 — Deployment

- Debian setup.
- NVIDIA/CUDA requirements.
- systemd unit or container configuration.
- Suwayomi configuration.
- Benchmark script.

---

# Scope Control

Do not add these to the first implementation unless required:

- GUI;
- full manga-library batch processing;
- chapter management;
- archive extraction;
- multiple simultaneous GPU workers;
- distributed workers;
- arbitrary AI model support;
- color IllustrationJaNai support;
- TensorRT engine generation;
- administrative web dashboard.

The initial product is a narrow, reliable Suwayomi conversion service.

---

# Main Engineering Principle

Keep the production runtime simple:

```text
validated ONNX model
+ C# service
+ CUDA inference
+ persistent cache
+ reliable fallback
```

Reuse MangaJaNai's trained models and verified image-processing behavior, not the surrounding Python/chaiNNer application stack.
