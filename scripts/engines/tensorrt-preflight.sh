#!/bin/sh
set -eu

root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
models=${KOMASCALER_GPU_MODEL_DIR:-/var/lib/komascaler/models}
publish="$root/tensorrt/publish"
results="$root/tensorrt/preflight"
test_artifacts="$root/tensorrt/test-artifacts"
dotnet_home="$root/tensorrt/dotnet-home"
packages="$root/nuget"
mkdir -p "$publish" "$results" "$test_artifacts" "$dotnet_home" "$packages"
: > "$results/missing.txt"
printf 'RUNNING\n' > "$results/status.txt"

fstype=$(findmnt -no FSTYPE -T "$root")
case "$fstype" in tmpfs|ramfs) echo "refusing memory-backed acceptance root: $root ($fstype)" >&2; exit 2;; esac
available_kib=$(df -Pk "$root" | awk 'NR==2 {print $4}')
test "$available_kib" -ge 3145728 || { echo "less than 3 GiB free under $root" >&2; exit 2; }

{
  date -u +%FT%TZ
  printf 'acceptance_root=%s fstype=%s free_kib=%s\n' "$root" "$fstype" "$available_kib"
  printf 'dotnet='; dotnet --version
  printf 'kernel='; uname -srmo
  printf 'vips='; vips --version
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.used --format=csv,noheader
  free -m
  dpkg-query -W -f='${Package}\t${Version}\n' 2>/dev/null | grep -E '(^|\t)(cuda|cudnn|libcudnn|tensorrt|libnvinfer|libnvonnx)' || true
} | tee "$results/host.txt"

env DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  dotnet restore src/KomaScaler.Api/KomaScaler.Api.csproj \
  --configfile config/NuGet.Config --artifacts-path "$test_artifacts"
env DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj -c Release --no-restore \
  --artifacts-path "$test_artifacts" -o "$publish"
trt_provider="$publish/runtimes/linux-x64/native/libonnxruntime_providers_tensorrt.so"
test -f "$trt_provider" || { echo "ORT TensorRT provider library missing: $trt_provider" >&2; exit 2; }
readelf -d "$trt_provider" | grep NEEDED > "$results/provider-needed.txt"
ldd "$trt_provider" > "$results/provider-ldd.txt"
missing=0
for library in libnvinfer.so.10 libnvonnxparser.so.10; do
  if ! /sbin/ldconfig -p | grep -q "$library"; then echo "missing=$library" | tee -a "$results/missing.txt"; missing=1; fi
done
test -w "$results"
test -d "$models" && test "$(find "$models" -maxdepth 1 -name '*.onnx' | wc -l)" -eq 7
if test "$missing" -ne 0; then
  echo "BLOCKED: ORT 1.26.0 TensorRT provider requires TensorRT 10 libraries; no GPU session was attempted." | tee "$results/status.txt"
  exit 2
fi

env DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  dotnet restore tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj \
  --configfile config/NuGet.Config --artifacts-path "$test_artifacts"
env DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  dotnet build tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj -c Release \
  --no-restore --artifacts-path "$test_artifacts"

sampler_pid=
cleanup() { test -z "$sampler_pid" || kill "$sampler_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM HUP
printf 'timestamp,vram_mib,available_ram_mib\n' > "$results/resources.csv"
(
  while :; do
    stamp=$(date +%s.%N)
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    ram=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    printf '%s,%s,%s\n' "$stamp" "$vram" "$ram"
    sleep 0.2
  done
) >> "$results/resources.csv" & sampler_pid=$!

if timeout --signal=TERM --kill-after=10s 180s env \
  DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  NVIDIA_TF32_OVERRIDE=0 \
  KOMASCALER_TENSORRT_ACCEPTANCE=1 \
  KOMASCALER_GPU_PROVIDER=TensorRt \
  KOMASCALER_GPU_MODEL_DIR="$models" \
  KOMASCALER_ACCEPTANCE_ROOT="$root" \
  KOMASCALER_TENSORRT_CACHE="$root/tensorrt/engines" \
  dotnet test tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj -c Release --no-build --no-restore \
    --artifacts-path "$test_artifacts" \
    --filter FullyQualifiedName~TensorRtProviderPreflightAndPlacement \
    --logger "console;verbosity=detailed" > "$results/provider-test.log" 2>&1; then status=0; else status=$?; fi
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
cat "$results/provider-test.log"
awk -F, 'NR>1 && NF==3 {if($2>peak)peak=$2;if(minram==0||$3<minram)minram=$3} END {printf "peak_vram_mib=%d min_available_ram_mib=%d\n",peak,minram}' "$results/resources.csv" | tee "$results/resources-summary.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ' | tee "$results/final-vram-mib.txt"
if test "$status" -eq 0; then printf 'PASSED\n' > "$results/status.txt"; else printf 'FAILED status=%s\n' "$status" > "$results/status.txt"; fi
exit "$status"
