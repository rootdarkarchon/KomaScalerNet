#!/bin/sh
set -eu

model_directory=${KOMASCALER_GPU_MODEL_DIR:?Set KOMASCALER_GPU_MODEL_DIR}
acceptance_root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
results_root="$acceptance_root/quality-reference/runs/$run_id"
test_filter=${KOMASCALER_GPU_TEST_FILTER:-FullyQualifiedName~UntiledDirectOrtQualityReferenceMatrix}
mkdir -p "$results_root"

filesystem_type=$(findmnt -n -o FSTYPE -T "$results_root")
if test "$filesystem_type" = tmpfs; then echo "Refusing quality output on tmpfs." >&2; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader >"$results_root/gpu-before.csv"

mkdir -p "$acceptance_root/dotnet-home" "$acceptance_root/nuget"
export DOTNET_CLI_HOME="$acceptance_root/dotnet-home"
export NUGET_PACKAGES="$acceptance_root/nuget"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
artifacts="$acceptance_root/quality-reference/artifacts"
dotnet build tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj \
  --configuration Release --artifacts-path "$artifacts"

test_pid=
sampler_pid=
cleanup() {
  if test -n "$sampler_pid"; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; fi
  if test -n "$test_pid"; then kill -TERM "$test_pid" 2>/dev/null || true; wait "$test_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

timeout --signal=TERM --kill-after=15s 360s \
  env "KOMASCALER_GPU_MODEL_DIR=$model_directory" \
      "KOMASCALER_ACCEPTANCE_ROOT=$acceptance_root" \
      "KOMASCALER_QUALITY_OUTPUT=$results_root/quality.json" \
  dotnet test tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj \
    --configuration Release --no-build --no-restore --artifacts-path "$artifacts" \
    --filter "$test_filter" \
    --logger "console;verbosity=detailed" >"$results_root/test.log" 2>&1 &
test_pid=$!

(
  echo "utc\tgpu_used_mib\tgpu_util_percent\tram_available_kib\tshmem_kib"
  while kill -0 "$test_pid" 2>/dev/null; do
    utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ' | tr ',' '\t')
    ram=$(awk '/MemAvailable:/ { available=$2 } /Shmem:/ { shmem=$2 } END { print available "\t" shmem }' /proc/meminfo)
    printf '%s\t%s\t%s\n' "$utc" "$gpu" "$ram"
    gpu_used=$(printf '%s' "$gpu" | cut -f 1)
    ram_available=$(printf '%s' "$ram" | cut -f 1)
    if test "$gpu_used" -gt 10240 || test "$ram_available" -lt 3145728; then
      echo "unsafe gpu_used_mib=$gpu_used ram_available_kib=$ram_available" >"$results_root/unsafe-pressure"
      kill -TERM "$test_pid" 2>/dev/null || true
      break
    fi
    sleep 0.2
  done
) >"$results_root/samples.tsv" &
sampler_pid=$!

status=0
wait "$test_pid" || status=$?
test_pid=
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
sleep 2
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader >"$results_root/gpu-after.csv"

if test -f "$results_root/unsafe-pressure"; then cat "$results_root/unsafe-pressure" >&2; exit 1; fi
if test "$status" -ne 0; then cat "$results_root/test.log" >&2; exit "$status"; fi
echo "RESULTS_ROOT=$results_root"
cat "$results_root/test.log"
trap - EXIT HUP INT TERM
