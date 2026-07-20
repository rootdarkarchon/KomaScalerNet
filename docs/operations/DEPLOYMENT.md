# Deployment and operations

This is the current operator guide. Documents under `docs/history/` and `docs/benchmarks/` are dated evidence, not configuration instructions.

## Required host

The validated target is Debian 13 with an RTX 3060 12 GiB, .NET/ASP.NET Core 10, an NVIDIA driver supporting CUDA 12.9, CUDA 12.9 runtime, cuDNN 9, TensorRT 10.14, ONNX Runtime 1.26, and libvips. TensorRT is the only production execution provider. Its CUDA runtime/driver dependencies are still mandatory.

Run preflight before installation:

```bash
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/var/lib/komascaler/acceptance \
sh scripts/engines/tensorrt-preflight.sh
```

The script reports provider/library load failures, versions, disk, RAM, VRAM, and cache writability. It does not install packages.

## Models and engines

Obtain the seven PyTorch checkpoints from the [MangaJaNai 1.0.0 release](https://github.com/the-database/MangaJaNai/releases/tag/1.0.0), export with `tools/model-export/export-fp32-models.sh`, and install only the generated ONNX files. This project neither owns nor distributes the source or derived models.

```bash
sudo install -d -o komascaler -g komascaler -m 0750 /var/lib/komascaler/models
sudo cp /path/to/export/*.fp32.onnx /var/lib/komascaler/models/
sudo cp models/models.production.json /var/lib/komascaler/models/
cd /var/lib/komascaler/models && sha256sum -c /path/to/repository/models/models.sha256
```

Prepare all seven engines sequentially before serving requests. Production uses exact profile min/opt/max `8/960/960`; persistent engines live under `/var/cache/komascaler/tensorrt`.

```bash
sudo install -d -o komascaler -g komascaler -m 0750 /var/cache/komascaler/tensorrt /var/lib/komascaler/engine-preparation
sudo -u komascaler env \
  KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
  KOMASCALER_TENSORRT_CACHE=/var/cache/komascaler/tensorrt \
  KOMASCALER_ACCEPTANCE_ROOT=/var/lib/komascaler/engine-preparation \
  sh scripts/engines/tensorrt-production-cache.sh
```

Cold all-model generation may take roughly ten minutes. It is sequential and covered by RAM/VRAM watchdogs. Model hash, ORT or TensorRT version, precision, TF32 state, profile, or engine-affecting provider options change cache identity and require rebuilding. A missing/stale engine is never built by an HTTP request: readiness fails until startup/deployment preparation completes.

`PreloadOnStartup=true` is the required production value. Startup sequentially builds or validates every engine, proves TensorRT placement, unloads the final session, then permits readiness. The systemd timeout is 15 minutes. At runtime one routed model/session/Run is resident; a model change drains and disposes it before loading the replacement.

## Publish and install

```bash
dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj -c Release -o artifacts/publish
sudo scripts/install/install.sh artifacts/publish
```

The idempotent installer checks the NVIDIA driver and required TensorRT, CUDA, cuDNN, and ORT provider libraries; creates the service user and directories; preserves existing operator configuration and manifest; makes the engine cache service-writable; and configures authentication deliberately.

Authenticated mode is the default. It creates `/etc/komascaler/token` only when absent, root:root `0600`, and installs the `LoadCredential` unit directive. Retrieve it for Suwayomi with `sudo cat /etc/komascaler/token`. Reruns never replace it.

Tokenless mode must be deliberate:

```bash
sudo env KOMASCALER_AUTH_MODE=disabled scripts/install/install.sh artifacts/publish
```

It removes the `LoadCredential` directive from the installed unit and disables application authentication through the environment. The source unit remains an authenticated template. Both paths are covered by `scripts/install/test-install-auth.sh`.

Start and inspect:

```bash
sudo systemctl start komascaler
systemctl status komascaler
journalctl -u komascaler -f
curl --fail http://127.0.0.1:9999/health/live
curl --fail http://127.0.0.1:9999/health/ready
```

## Suwayomi `serveConversions`

If Suwayomi runs in Docker Compose:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Bind Kestrel to a host address reachable from the container and restrict port 9999 with the firewall. Host `127.0.0.1` is not container loopback. Use the same root-only token value:

```hocon
server.serveConversions = {
  default = {
    target = "http://host.docker.internal:9999/convert"
    callTimeout = 2m
    connectTimeout = 5s
    headers = { "X-Upscaler-Token" = "replace-with-/etc/komascaler/token" }
  }
}

# This must be a Config/object. [] causes ConfigException$WrongType and breaks
# the web-settings GraphQL query.
server.downloadConversions = {}
```

The repository example is `deploy/suwayomi/server.conf.example`.

## Production configuration

Keep these measured settings unless separately revalidated:

```text
Tiling.MaximumCoreSize              832
Tiling.ContextPixelsPerSide          64
Models.TensorRtProfileOptimumExtent 960
Models.TensorRtProfileMaximumExtent 960
Models.PreloadOnStartup             true
Output.Format                        png
Output.Lossless                     true
Output.PngCompression                  3
Queue.ResponseDeadline             28 s
```

Concurrency is not configurable in production: one queued image conversion, one active model/session, and one GPU Run. Operators may change directories, idle timeout, queue capacity/deadline, input/cache limits, bind URL, and authentication. Invalid tiling reloads retain the last-known-good policy.

## Cache and lifecycle

Application results and TensorRT engines have distinct identities/locations. Stop the service before cleanup. Result cache entries can be deleted and recomputed. Engine deletion causes a readiness-blocking rebuild at the next startup. Idle timeout disposes the resident model; switching routes drains the sole GPU permit and frees the old model before loading the new engine. Shutdown/fault cleanup also drains before native disposal.

## Verification and troubleshooting

```bash
sh scripts/install/test-install-auth.sh
sh scripts/install/test-suwayomi-config.sh
systemd-analyze verify deploy/systemd/komascaler.service
KOMASCALER_GPU_MODEL_DIR=/var/lib/komascaler/models \
KOMASCALER_ACCEPTANCE_ROOT=/var/lib/komascaler/acceptance \
sh scripts/acceptance/tensorrt-acceptance.sh
```

- `243/CREDENTIALS`: create the authenticated token by rerunning the installer, or deliberately select tokenless mode. Never leave a missing credential reference.
- `noexec`: move published executables to `/usr/lib/komascaler` or another executable mount.
- TensorRT mismatch: inspect preflight output and use versions compatible with ORT 1.26; do not guess an arbitrary newest package.
- Missing/stale cache: regenerate exact `8/960/960` engines as the service user.
- Readiness delay: monitor the journal during sequential generation; first deployment can approach ten minutes.
- VRAM not released: wait for idle unload or stop the process, then use `nvidia-smi` to identify any remaining owner.
