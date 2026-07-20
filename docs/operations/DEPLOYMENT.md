# Deployment and operations

This is the current operator guide. Documents under `docs/history/` and `docs/benchmarks/` are dated evidence, not configuration instructions.

## 1. Build machine

Building KomaScaler.Net requires the **.NET 10 SDK**. The tested host uses
Debian 13 and SDK 10.0.302; these are tested versions, not universal minimums.
Microsoft's [Debian installation guide](https://learn.microsoft.com/dotnet/core/install/linux-debian)
documents its Debian 13 package feed. On a fresh host, add that feed before
installing .NET:

```bash
sudo apt-get update
sudo apt-get install --yes ca-certificates curl

dotnet_repo_tmp=$(mktemp -d)
cleanup_dotnet_repo_tmp() {
  rm -rf -- "$dotnet_repo_tmp"
}
trap cleanup_dotnet_repo_tmp EXIT HUP INT TERM

curl --fail --location \
  --output "$dotnet_repo_tmp/packages-microsoft-prod.deb" \
  https://packages.microsoft.com/config/debian/13/packages-microsoft-prod.deb
sudo dpkg -i "$dotnet_repo_tmp/packages-microsoft-prod.deb"
sudo apt-get update
```

The trap removes only the directory returned by `mktemp -d`. Install the build
dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  dotnet-sdk-10.0 \
  git \
  ca-certificates \
  curl
```

`wget` may replace `curl` for downloads, but the installed production unit uses
`curl` for its readiness probe. `jq` is additionally required by the production
engine-preparation script:

```bash
sudo apt-get install --yes jq
```

The solution consumes NetVips through NuGet and loads the libvips shared
library dynamically. It does not compile or link native libvips code, so
`libvips-dev` is **not** a build requirement. No repository build step invokes a
C/C++ compiler, therefore `build-essential` is also unnecessary. The current
scripts do not use `unzip`.

Run the repository's local verification entry point rather than maintaining a
second command list here:

```bash
sh scripts/verify.sh
```

The target-host engine preparation and GPU acceptance scripts also call
`dotnet publish`, `build`, and `test`. Consequently the SDK and `jq` are needed
on the target while those scripts run. They are build/validation tools, not
dependencies of the installed service.

## 2. Production runtime

The supplied publish command creates a framework-dependent deployment. A host
that only runs already-published output therefore needs:

```bash
sudo apt-get install --yes \
  aspnetcore-runtime-10.0 \
  libvips-tools \
  ca-certificates \
  curl \
  openssl
```

On Debian 13, `libvips-tools` pulls in `libvips42t64`, the actual shared-library
runtime used by NetVips. Installing `libvips42t64` directly is also valid, but
the tools package provides `vips --version` for preflight and verification.
Neither `libvips-dev` nor .NET SDK files are needed merely to run the installed
service. See Debian's [libvips42t64 package](https://packages.debian.org/stable/libs/libvips42t64)
and the [libvips installation documentation](https://www.libvips.org/install.html).

`curl` is used by `ExecStartPost` for readiness. `openssl` is used by the
installer to generate an authenticated deployment token; it is not used by the
steady-state application. ONNX Runtime 1.26 is supplied by the pinned
`Microsoft.ML.OnnxRuntime.Gpu` NuGet package in the publish output, not by a
Debian system package.

## 3. NVIDIA driver and GPU verification

The kernel NVIDIA driver is separate from CUDA, cuDNN, and TensorRT userspace
libraries. Install a driver appropriate for the GPU and Debian host using the
operator's normal driver-management policy; do not let a CUDA meta-package
silently replace a known-working driver.

The tested system is an RTX 3060 12 GiB with driver **595.58.03**. Verify the
driver before adding userspace libraries:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

If these commands fail, repair the driver first. Installing CUDA libraries
cannot fix a nonfunctional kernel driver. Refer to NVIDIA's
[CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
for NVIDIA's driver/toolkit relationship and pre-installation checks.

## 4. CUDA 12.9, cuDNN 9, and TensorRT 10 runtime

The known-good target used CUDA **12.9.2**, cuDNN **9.24.0.43**, TensorRT
**10.14.1.48**, ONNX Runtime **1.26**, and driver **595.58.03**. These are the
tested versions, not general minimum-version claims. TensorRT is the mandatory
production execution provider, but it still depends on the NVIDIA driver,
CUDA, and cuDNN.

Do **not** install the unversioned `cuda` or `cuda-toolkit` meta-packages. They
track the repository's current release and may install CUDA 13 or alter the
working driver stack. KomaScaler needs the versioned runtime package
`cuda-libraries-12-9`; it does not need the full CUDA toolkit, compiler,
samples, or development headers.

### Debian 13 / CUDA 12.9 compatibility repository

> At the time of this tested deployment, NVIDIA's Debian 13 repository
> publishes CUDA 13 packages and does not provide the required CUDA 12.9
> versioned packages. CUDA 12.9, its matching TensorRT build, and the CUDA-12
> cuDNN package therefore come from NVIDIA's **Debian 12 CUDA repository**.
> This is a deliberately narrow, version-pinned compatibility workaround.
> Never add ordinary Debian 12 distribution repositories to Debian 13.

NVIDIA's Debian 12 repository and keyring package are visible in its
[official repository index](https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/).
If the Debian 13 `cuda-keyring` package is already installed, do not install the
Debian 12 package over it: both use the same package/file identity. Extract the
second key under a distinct name instead:

```bash
compat_tmp=$(mktemp -d)
cleanup_compat_tmp() {
  rm -rf -- "$compat_tmp"
}
trap cleanup_compat_tmp EXIT HUP INT TERM

curl --fail --location \
  --output "$compat_tmp/cuda-keyring_1.1-1_all.deb" \
  https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb

dpkg-deb -x \
  "$compat_tmp/cuda-keyring_1.1-1_all.deb" \
  "$compat_tmp/extracted"

sudo install -o root -g root -m 0644 \
  "$compat_tmp/extracted/usr/share/keyrings/cuda-archive-keyring.gpg" \
  /usr/share/keyrings/cuda-debian12-archive-keyring.gpg

printf '%s\n' \
  'deb [signed-by=/usr/share/keyrings/cuda-debian12-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' \
  | sudo tee /etc/apt/sources.list.d/cuda-debian12-x86_64.list >/dev/null

sudo apt-get update
```

The trap removes only the resolved directory returned by `mktemp -d`. Keep a
Debian 13 NVIDIA source, if present, bound to its own Debian 13 keyring. Separate
keyring filenames prevent one `cuda-keyring` package from overwriting the
other repository's key and avoid the missing-key/signature failure observed on
the tested host.

Before installing anything, inspect the candidates and available TensorRT
versions:

```bash
apt-cache policy \
  cuda-libraries-12-9 \
  cudnn9-cuda-12 \
  libnvinfer10 \
  libnvinfer-plugin10 \
  libnvonnxparsers10

apt-cache madison \
  libnvinfer10 \
  libnvinfer-plugin10 \
  libnvonnxparsers10
```

Simulate the exact tested runtime transaction first:

```bash
sudo apt-get --simulate install \
  cuda-libraries-12-9=12.9.2-1 \
  cudnn9-cuda-12=9.24.0.43-1 \
  libnvinfer10=10.14.1.48-1+cuda12.9 \
  libnvinfer-plugin10=10.14.1.48-1+cuda12.9 \
  libnvonnxparsers10=10.14.1.48-1+cuda12.9
```

Stop if APT proposes CUDA 13, a different TensorRT ABI, or removal/replacement
of the working NVIDIA driver. When the simulation is correct, run the same
version-pinned transaction without `--simulate`:

```bash
sudo apt-get install \
  cuda-libraries-12-9=12.9.2-1 \
  cudnn9-cuda-12=9.24.0.43-1 \
  libnvinfer10=10.14.1.48-1+cuda12.9 \
  libnvinfer-plugin10=10.14.1.48-1+cuda12.9 \
  libnvonnxparsers10=10.14.1.48-1+cuda12.9
```

APT metadata on the tested host shows that `libnvinfer-plugin10` and
`libnvonnxparsers10` depend on the exact matching `libnvinfer10`; those three
packages provide the TensorRT libraries KomaScaler loads. No TensorRT
development, samples, Python, lean/dispatch, or `trtexec` package is required.
CUDA and cuDNN are intentionally listed separately rather than assumed to come
from TensorRT. See NVIDIA's [cuDNN Linux guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)
and [TensorRT Debian package guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-debian.html).

## 5. Optional model export tooling

Model export is a one-time preparation task and is not a production-service
dependency. The pinned Python stack is supported with Python **3.10-3.12**;
Debian 13's default Python 3.13 must not be presented as compatible with that
environment. On the tested Debian 13 APT sources, `python3.12` has no candidate.

Use the repository's supported Python 3.10-3.12 environment/bootstrap method
and point `PYTHON_BIN` at that environment. Do not add an ordinary Debian 12
repository merely to obtain Python. The detailed procedure and pinned packages
are in [the export plan](../history/EXPORT-PLAN.md) and
[`tools/model-export`](../../tools/model-export/README.md). Python, PyTorch,
Spandrel, ONNX export packages, and validation libraries can be removed from a
production-only host after verified ONNX files have been produced.

## 6. Post-install verification

Confirm the framework, image runtime, driver, shared libraries, and installed
package versions:

```bash
dotnet --info
vips --version
nvidia-smi

ldconfig -p | grep -E \
  'libcudart\.so\.12|libcudnn\.so\.9|libnvinfer\.so\.10|libnvonnxparser\.so\.10'

dpkg-query -W \
  'cuda-*12-9*' \
  'cudnn9-cuda-12' \
  'libnvinfer10' \
  'libnvinfer-plugin10' \
  'libnvonnxparsers10'
```

Then run the existing commands for each repository operation rather than
copying variants from this package section:

- local build and native-independent checks: `sh scripts/verify.sh`;
- publish and service installation: [Publish and install](#publish-and-install);
- sequential engine preparation: [Models and engines](#models-and-engines);
- read-only provider/host preflight:
  `sh scripts/engines/tensorrt-preflight.sh` with the documented environment;
- real target-host GPU acceptance: [Verification and troubleshooting](#verification-and-troubleshooting).

The preflight reports versions, provider/library loading, disk, RAM, VRAM, and
cache writability. It does not install or modify packages.

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
