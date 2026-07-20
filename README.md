# KomaScaler.Net

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

KomaScaler.Net adds transparent, on-demand MangaJaNai upscaling to Suwayomi.

When Suwayomi serves a black-and-white manga page, it sends the page to KomaScaler through its `serveConversions` hook. KomaScaler selects the MangaJaNai 2× model best suited to the page resolution, upscales it on an NVIDIA GPU using TensorRT, and returns a lossless grayscale PNG. The result is cached, so subsequent requests are effectively immediate.

Color pages are returned unchanged. Models are loaded only when needed, replaced when another resolution requires a different model, and unloaded after an idle period to release VRAM.

The project is intended for people running their own Suwayomi server with an NVIDIA GPU who want improved reader-page quality without permanently pre-upscaling and storing an entire manga library.

```text
Suwayomi reader
    → original manga page
    → KomaScaler /convert
    → monochrome detection
    → resolution-based MangaJaNai model selection
    → TensorRT 2× tiled inference
    → lossless PNG cache
    → Suwayomi reader
```

## What it does

- Integrates transparently through Suwayomi's `serveConversions` hook.
- Automatically selects among seven MangaJaNai 2× models by page resolution.
- Upscales monochrome manga pages using TensorRT on an NVIDIA GPU.
- Returns color pages unchanged, preserving their original bytes.
- Produces single-channel, lossless PNG output.
- Stores converted results in a persistent cache for fast repeat requests.
- Keeps only one model active and releases its VRAM after an idle period.
- Supports a shared authentication token for container-to-host requests.

## What it is not

- It is not a general-purpose image-upscaling UI.
- It is not a batch converter for permanently rewriting a manga library.
- It is not intended to upscale color covers or illustrations.
- It is not a hosted service.
- It does not include or redistribute MangaJaNai model files.

## Requirements

- Linux; validated on Debian 13.
- An NVIDIA RTX GPU; validated on an RTX 3060 with 12 GiB VRAM.
- CUDA 12, cuDNN 9, and TensorRT 10.
- .NET 10 and the ASP.NET Core runtime.
- libvips.
- Suwayomi.
- The seven upstream MangaJaNai 2× models.

Exact validated component versions and host checks are documented in the [deployment guide](docs/operations/DEPLOYMENT.md).

## Quick start

1. Install the NVIDIA, TensorRT, .NET, and libvips dependencies.
2. Download the seven MangaJaNai 2× source models.
3. Export the `.pth` checkpoints to FP32 ONNX using the included tooling.
4. Publish and install KomaScaler.Net.
5. Prepare all seven host-specific TensorRT engines.
6. Add the Suwayomi `serveConversions` configuration.
7. Start the service and begin reading.

Follow the [deployment guide](docs/operations/DEPLOYMENT.md) for commands, engine preparation, systemd installation, readiness checks, and troubleshooting.

KomaScaler.Net is distributed as source. Run `sh scripts/verify.sh` for local formatting, Release build, native-independent tests, and script checks. TensorRT acceptance is a separate target-host operation described in the deployment guide.

## Suwayomi configuration

When Suwayomi runs in Docker Compose, make the host reachable from the container:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

KomaScaler must listen on an address reachable from that container. Configure Suwayomi with the same token stored in `/etc/komascaler/token`:

```hocon
server.serveConversions = {
  default = {
    target = "http://host.docker.internal:9999/convert"
    callTimeout = 2m
    connectTimeout = 5s
    headers = {
      "X-Upscaler-Token" = "replace-with-the-shared-token"
    }
  }
}

# This must be a Config/object, not a list.
server.downloadConversions = {}
```

This converts pages while they are served to a reader. It does not pre-convert pages when chapters are downloaded. See the [complete Suwayomi and networking setup](docs/operations/DEPLOYMENT.md#suwayomi-serveconversions).

## Models

Download the seven MangaJaNai 2× checkpoints for heights 1200, 1300, 1400, 1500, 1600, 1920, and 2048 from the [MangaJaNai 1.0.0 release](https://github.com/the-database/MangaJaNai/releases/tag/1.0.0).

The upstream release provides PyTorch `.pth` files. The included [model-export tool](tools/model-export/README.md) converts them into the seven FP32 ONNX files expected by KomaScaler. The manifest and expected hashes are stored under [`models/`](models/).

KomaScaler.Net does not distribute or own the models and is not affiliated with, endorsed by, or supported by MangaJaNai's authors.

## Performance

On the validated RTX 3060 host, a real 900×1291 Suwayomi page used four tiles and completed tiled upscaling in approximately 1.7 seconds. A subsequent identical request was served from KomaScaler's result cache.

This is a representative measurement, not a performance guarantee. Hardware, page content, model selection, and system load affect results. See the [benchmark history](docs/benchmarks/) for the measurements behind the production profile.

## Project status

> Personal, unsupported project provided as-is. There is no support commitment or roadmap, and GitHub Issues are disabled.

See [SUPPORT.md](SUPPORT.md) for the complete project and support policy.

## Documentation

- [Deployment and installation](docs/operations/DEPLOYMENT.md)
- [Configuration example](config/appsettings.example.json)
- [Production architecture](docs/design/FINAL-ARCHITECTURE.md)
- [Benchmark history](docs/benchmarks/)
- [Troubleshooting](docs/operations/DEPLOYMENT.md#verification-and-troubleshooting)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

KomaScaler.Net's original source code is licensed under the [MIT License](LICENSE), copyright (c) 2026 rootdarkarchon.

The license applies only to this project's original source code. It does not grant rights to MangaJaNai models, generated model derivatives, NVIDIA components, or other third-party material. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
