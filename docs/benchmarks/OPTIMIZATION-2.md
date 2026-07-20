# Optimization 2: intra-image GPU tile parallelism

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20
Target: NVIDIA GeForce RTX 3060 12 GiB, CUDA 12, cuDNN 9
Scope: One queued image conversion at a time, with configurable bounded
intra-image GPU tile parallelism. Parallel image inference remains forbidden;
`ConversionQueue` is still a single-reader channel consumer.

## Plan

1. Add one validated `Upscaling:Models:GpuParallelism` setting while retaining
   factor-1 compatibility.
2. Submit bounded tile batches concurrently, but consume and blend their
   outputs in the original deterministic tile order.
3. Give the singleton GPU service the same number of Run permits. Make idle
   drain, fault cleanup, and shutdown acquire every permit before disposing
   any session.
4. Add fake-backed concurrency, limit-validation, and byte-parity tests.
5. Benchmark factors incrementally in separate pre-warmed processes with a
   fresh cache, hard timeouts, continuous VRAM/RAM sampling, and automatic
   termination above 10 GiB VRAM or below 3 GiB available RAM.
6. Compare every factor's result bytes to the first result and select the
   production knee rather than the fastest result regardless of memory.

## Implementation

`GpuParallelism` accepts 1 through 16. `TiledUpscaler` prepares at most one
bounded batch of that many tiles, starts their backend calls concurrently, then
awaits the batch and blends it in source tile order. Only one image can enter
this code because `ConversionQueue` retains `SingleReader=true` and executes
one work item to completion before reading the next.

`GpuUpscaler` uses the configured factor as its semaphore capacity. Active Run
instrumentation therefore measures actual overlapping tile calls. Lifecycle
operations acquire all permits before session disposal. A partial acquisition
is released if cancellation interrupts an idle drain, avoiding lost permits.
Fault cleanup waits for every already-running tile before disposing the shared
seven-session set and requesting a systemd restart.

The factor is operational and is not included in cache identity: blending is
ordered and every measured factor produced byte-identical output. Tiling,
model, preprocessing, and encoder policies remain part of cache identity as
before.

The reusable harness is `scripts/benchmarks/gpu-parallelism-benchmark.sh`. It refuses
tmpfs, requires 3 GiB disk space, publishes once under `/devtmp`, uses a unique
results directory, bounds each service process to 120 seconds, bounds each
HTTP request to 35 seconds, and samples pressure once per second. It never
copies the ONNX Runtime native assets per factor.

## Production-page measurements

The fixture is the same 1660×1400 neutral page used by target acceptance and
Optimization 1. It selects the real 1400 model, uses 30 tiles under the fixed
320/32 policy, and returns a 3320×2800 lossless effort-2 WebP of 4,427,946
bytes. Sessions were pre-warmed before request timing. Each factor ran in a
new process with a fresh cache.

| factor | HTTP seconds | peak sampled VRAM MiB | minimum available RAM MiB | result |
|---:|---:|---:|---:|---|
| 1 | 18.596, 18.712, 18.759 | 3,097 | 6,866 | exact reference |
| 2 | 13.670 | 3,131 | 6,555 | exact |
| 3 | 10.859 | 4,165 | 6,567 | exact |
| 4 | 10.298 | 4,185 | 6,256 | exact |
| 5 | 8.884 | 5,233 | 6,035 | exact |
| 6 | 8.721 | 5,253 | 5,933 | exact |
| 7 | 8.929 | 5,285 | 6,105 | exact; regression |
| 8 | 8.305, 8.372 | 5,295 | 6,001 | exact |
| 10 | 8.134, 8.163 | 6,373 | 6,089 | exact |
| 12 | 8.051 | 6,405 | 5,995 | exact |
| 15 | 8.010, 8.039 | 7,493 | 6,478 | exact |
| 16 | 8.081 | 7,503 | 6,583 | exact; regression |

No pressure watchdog fired, no request timed out, no CUDA fault occurred, and
VRAM returned to the host baseline after each process. The non-monotonic points
are consistent with normal run variance and batching 30 tiles into different
group sizes; they are why a single fastest number is not sufficient for the
production choice.

## Production selection

Production uses factor **8**.

The three factor-1 runs average 18.689 seconds; the two factor-8 runs average
8.339 seconds. Factor 8 therefore reduces page latency by 10.350 seconds, or
55.4%, while leaving about 19.6 seconds below the 28-second application
deadline.

Factor 15 is the absolute measured latency minimum, averaging 8.024 seconds,
but improves on factor 8 by only 314 ms (3.8%) while its worst sampled VRAM is
2,198 MiB higher. Factor 10 similarly spends about another 1,078 MiB for only
190 ms average improvement. Since 320 is already the established per-tile VRAM
limit and the device may face allocator variance or unrelated load, factor 8
is the better production latency/headroom knee.

The setting remains configurable so an operator with a dedicated GPU can
repeat the supplied harness and deliberately choose a different factor.
Values above 8 should not be deployed on this host without repeating fault,
idle-disposal, process-release, and sustained-load acceptance.

## Verification requirements

The final gate must prove:

- factor-8 production tiling reaches the expected concurrent Run count while
  exact dimensions and provider evidence remain unchanged;
- eight direct tile calls never exceed eight active Runs;
- idle drain, reload, double disposal, and process termination still release
  the multi-permit lifecycle safely;
- ordinary fake-backed tests continue to prove one-image queue processing and
  deterministic output at configured parallelism;
- factor-8 HTTP output is byte-identical to factor 1 and a repeated request is
  a cache hit.

## Final verification results

All requirements above passed on the target host. The isolated factor-8 GPU
suite recorded 1,035 CUDA nodes, loaded and ran all seven models with zero
wrapper/direct difference, and preserved exact routing and 2× dimensions. The
1280×320 production tiling case created four tiles, observed four active Runs,
completed in 5.494 seconds, and reported 2,051 MiB VRAM inside the test. The
dedicated eight-call case observed exactly eight active Runs. Idle drain
reduced VRAM from 1,093 to 137 MiB, reload succeeded in 4.849 seconds, and
double disposal passed with the multi-permit lifecycle.

The final configured real HTTP run returned `upscaled` in 9.068 seconds. Its
instrumented tile wall time was 5.733 seconds and effort-2 WebP encoding took
2.987 seconds. The next request returned `cache` in 68 ms. Both response bodies
were identical, and the factor-8 upscale was byte-identical to the retained
factor-1 Optimization-1 output. The service shut down cleanly and GPU memory
returned to the 3 MiB host baseline.
