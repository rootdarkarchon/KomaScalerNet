# Target-host acceptance report

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

Date: 2026-07-20 (Europe/Zurich)
Target: Debian 13.4, Linux 6.12.74+deb13+1, NVIDIA GeForce RTX 3060 12 GiB
Result: all locally executable application and GPU acceptance gates pass. Two
operator gates remain: privileged installation and configuring the separately
contained Suwayomi instance.

This is a second report. `docs/history/ACCEPTANCE-REPORT.md` was preserved (SHA-256
`11c5c9c9babb6efa0b8e4cda93b1fe3cd106562f668baafaa87d45e999e47db2`).

## What is measured

Unless explicitly labelled otherwise, results below were measured on this
host against the seven files in `/var/lib/komascaler/models`. Fake inference
results are confined to the ordinary integration-test paragraph. No model
weights were copied into the repository.

Acceptance output was consolidated at `/devtmp/komascaler-acceptance` on the
disk-backed Btrfs root filesystem. Before the final run it had 32 GiB free.
`/tmp` is a 7.8 GiB tmpfs and finished with only 96 KiB used; acceptance used
3.6 GiB on `/devtmp` including the one reusable NuGet/build/publish tree.

## Components and inventory

Commands used included:

```sh
dotnet --info
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
dpkg-query -W 'cuda-cudart-12-*' 'libcudnn9-cuda-12'
vips --version
sha256sum /var/lib/komascaler/models/*.onnx
```

Measured versions: .NET SDK 10.0.302, ASP.NET runtime 10.0.10, NVIDIA driver
595.58.03, CUDA 12.9 cudart 12.9.79, cuDNN 9.24.0.43 for CUDA 12, and libvips
8.16.1. Driver CUDA compatibility was reported as 13.2. Baseline GPU use was
3 MiB of 12,288 MiB.

The directory contains exactly seven regular FP32 ONNX files, each 67,201,339
bytes. Their measured SHA-256 values match `models/models.production.json`:

| model | SHA-256 |
|---|---|
| 1200 | `001d55ee9bf9fe43cdcce817cc2d13fed30f9f95dcba502cf7f69384d1596474` |
| 1300 | `ac04401fb18715135166ca3a28cfb6e8e46d1bd6c3f0db95583a5a74baa6f8f1` |
| 1400 | `3d813490698f6208a1c0273b1c1170fad560eb3c3aaca9097578542c39521ec0` |
| 1500 | `c16a4416e0c509c0239a0c50f6b378142d1e2bdb6ab7622921449ff0e886a38d` |
| 1600 | `fca9f830f481ec43faff5076e3e9d678ad1c28ef037abdc74bc8e10c09919dd5` |
| 1920 | `5c73f796e98e6af3ca6a3ecbc583a2f47ec0252f291d51024164d6197f55da7b` |
| 2048 | `78e095c7c2af0c074e6ba56b6b7f0084cecd73153dca7d6908ba138c65999569` |

The repository manifest hash is
`3ce6a72c92efe048e49c530e3ab161e04867db6693606c77d4f5146d7164ea50`.
The target directory is owned `nobody:nogroup`, and the manifest is not yet
beside the models because this login has no noninteractive sudo. Tests loaded
the repository manifest by absolute path while still hashing every target
model. This is an operator gate, not an application failure:

```sh
sudo install -o nobody -g nogroup -m 0644 \
  /path/to/komascaler/models/models.production.json \
  /var/lib/komascaler/models/models.production.json
```

## Build, formatting, tests, publish

The corrected complete gate was:

```sh
export DOTNET_CLI_HOME=/devtmp/komascaler-acceptance/dotnet-home
export NUGET_PACKAGES=/devtmp/komascaler-acceptance/nuget
dotnet format KomaScaler.Net.sln --verify-no-changes --no-restore
dotnet build KomaScaler.Net.sln --configuration Release --no-restore
dotnet test KomaScaler.Net.sln --configuration Release --no-build
dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj --configuration Release \
  --no-restore --output /devtmp/komascaler-acceptance/publish
```

Formatting passed. Release build passed with zero warnings and zero errors.
Forty unit tests and nine integration tests passed. The six opt-in GPU tests
were correctly skipped in the ordinary run and then executed explicitly as
described below. Native libvips coverage included actual JPEG, PNG, WebP,
BMP, AVIF/HEIF decode, alpha, EXIF orientation, monochrome detection, decoded
dimension limits, and WebP encoding. The integration tests use a labelled fake
inference backend to prove queue, deadline/recoverable fallback, cache,
deduplication, and serialization behavior; they are not cited as GPU proof.

## Hang diagnosis and bounded GPU rerun

The original combined GPU suite is an acceptance failure. It created three
approximately 1.3 GiB native-asset trees plus package/publish data on `/tmp`,
which is tmpfs with no swap. `/tmp` reached about 6.1 GiB, available RAM fell
to about 3.6 GiB, and journald recorded `Under memory pressure, flushing
caches` at 15:23:06, 15:24:25, and 15:25:08. Kernel and NVIDIA logs contained
no OOM kill and no NVIDIA Xid. The test process disappeared and VRAM returned
to 3 MiB. The evidence identifies simultaneous duplicated tmpfs artifacts and
RAM pressure, not cuDNN workspace exhaustion, inference saturation, or a
driver fault.

The fixed runner verifies a non-tmpfs destination and 3 GiB free, performs one
reusable build, runs cases sequentially, wraps each in `timeout --signal=TERM
--kill-after=15s 180s`, and samples VRAM, GPU utilization, MemAvailable, and
Shmem every second:

```sh
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/devtmp/komascaler-acceptance \
sh scripts/acceptance/gpu-acceptance.sh
```

All six isolated cases passed without a host stall:

| case | measured result | peak VRAM | minimum available RAM |
|---|---|---:|---:|
| provider/profile | 1,035 CUDA nodes; 20 CPU shape/control nodes | 265 MiB | 9,629.0 MiB |
| seven sessions/routing/parity | all seven passed; cold load + first run 4,660.789 ms | 1,155 MiB | 7,608.1 MiB |
| production tiling 320/32 | 1280×320 to 2560×640, four tiles, 6,571.035 ms | 1,989 MiB | 9,347.5 MiB |
| eight concurrent callers | maximum active GPU Runs exactly 1, 5,557.487 ms | 1,093 MiB | 9,579.7 MiB |
| idle drain/reload | 3 -> 1,093 -> 137 MiB; reload 4,641.862 ms | 1,093 MiB | 9,097.5 MiB |
| repeated disposal | double dispose after idle drain passed | 1,093 MiB | 9,569.7 MiB |

Provider enumeration was
`TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`.
An ONNX Runtime profile, rather than provider enumeration alone, proved the
model graph executed on `CUDAExecutionProvider`. ORT deliberately placed 20
shape/control nodes on CPU for performance; 1,035 compute nodes were CUDA.
The application explicitly appends CUDA and does not install the CPU ORT
package or retry inference on a CPU session, so there is no whole-inference
CPU fallback.

Production routing heights 1200, 1251, 1351, 1451, 1551, 1761, and 1985
selected the 1200, 1300, 1400, 1500, 1600, 1920, and 2048 models respectively.
Every selected model loaded and performed a real 8×8 -> 16×16 finite FP32
inference. Timings were 17.559, 765.007, 767.390, 771.742, 766.287, 778.959,
and 781.273 ms; the first model was already warm. The wrapper/direct-ORT
maximum absolute difference was exactly `0.00000000E+000`. This is measured
runtime parity. Source-framework parity was not re-inferred because the source
PyTorch weights/reference tensors are not installed; prior hash-linked export
evidence remains documentary rather than a new target-host measurement.

The sequential cases distinguish failure modes: construction of all seven
sessions stabilized at 1,155 MiB VRAM; the largest tiled inference peaked at
1,989 MiB; repeated construction/disposal and reload completed; concurrent
requests serialized. After each test process and after both HTTP processes
terminated, VRAM returned to the 3 MiB host baseline.

## Real HTTP results

The published application ran with the target model directory, absolute
repository manifest, a fresh `/devtmp` cache, five-second idle unload, and a
35-second client cap. Commands were multipart `curl --max-time` requests to
`http://127.0.0.1:18889/convert`, followed by `cmp` and `vipsheader`.

| request | result header | time | bytes/dimensions | verification |
|---|---|---:|---|---|
| cold neutral 1660×1400 PNG | `upscaled` | 23.330710 s | 4,427,946; 3320×2800 WebP | exact 2× |
| same page | `cache` | 0.069966 s | 4,427,946 | byte-identical to first result |
| color PNG | `bypass` | 0.002240 s | 293 | byte-identical to input |

A separate process used a fresh cache and a one-second response deadline. A
small neutral page returned HTTP 200 `fallback` in 1.205423 s with all 3,385
original PNG bytes unchanged. The non-preemptible queued GPU operation safely
completed afterward; the next request returned `cache` in 0.009831 s with a
valid 132×112 WebP. This is measured deadline recovery through the real GPU
path. Deliberately inducing CUDA OOM/Xid was not attempted because it risks
another system-wide disruption; recoverable exception state transitions are
also covered by the fake-backed integration test and are labelled as such.

## Fixes made during target acceptance

- Removed `session.disable_cpu_ep_fallback=1`; ORT requires its intentional
  CPU assignment for shape/control nodes. CUDA remains mandatory and profiled.
- Replaced duplicated `/tmp` GPU builds with the reusable, disk-backed,
  sequential, sampled, hard-timeout runner.
- Split the hanging GPU suite into six isolated cases and routed all seven real
  runs through production selection bands.
- Added idempotent GPU singleton disposal after idle drain and its real-GPU
  regression test; this fixed shutdown `ObjectDisposedException`.
- Added output `Effort` (validated 0..6) to configuration, encoding, and cache
  identity. Production effort 2 reduced the measured large-page lossless WebP
  encode; effort 6 had measured about 38.4 seconds on the same output.
- Raised the production response deadline from 25 to 28 seconds. The measured
  23.330710-second cold conversion was only 1.669 seconds below 25 seconds;
  28 seconds provides 4.669 seconds of application margin while remaining
  below Suwayomi's approximately 30-second caller timeout. The dominant cost
  is tiled GPU inference; seven-session cold construction measured 4.661
  seconds and effort-2 lossless WebP encoding measured about 2.7 seconds. The
  320-pixel core is an established VRAM limit and must not be increased. The
  implemented follow-up is documented in `docs/benchmarks/OPTIMIZATION-1.md`. Startup
  pre-warming and cached one-dimensional blend weights were retained. Direct
  timing showed preparation/blending/normalization and managed output copies
  were too small to justify CPU/GPU pipelining or ONNX I/O binding.
- Expanded native libvips tests to actual AVIF/HEIF and EXIF-orientation paths.

## systemd and Suwayomi

`systemctl status komascaler` reports that the unit is not installed.
`systemd-analyze verify deploy/systemd/komascaler.service` parsed the supplied
unit and only reported the expected absent staged executable
`/usr/lib/komascaler/komascaler`. Installing or changing the running host needs
interactive root access:

```sh
cd /path/to/komascaler
sudo scripts/install/install.sh /devtmp/komascaler-acceptance/publish
sudo systemctl daemon-reload
sudo systemctl enable --now komascaler
sudo systemctl status komascaler --no-pager
curl --fail http://127.0.0.1:9999/health/ready
```

Suwayomi/Tachidesk is running (Java PID 3955) inside another container-like
environment whose `/home/suwayomi` path is not visible from this host login.
Mutating it was therefore neither possible nor safe. Apply
`deploy/suwayomi/server.conf.example` to that instance's active HOCON file,
use the same optional token as KomaScaler, restart Suwayomi, and send one real
reader page. The standalone endpoint has already demonstrated the exact HTTP
contract (`upscaled`, `cache`, exact `bypass`, exact recoverable `fallback`);
only the caller-side wiring remains an operator gate.
