#!/bin/sh
set -eu

model_directory=${KOMASCALER_GPU_MODEL_DIR:?Set KOMASCALER_GPU_MODEL_DIR}
fixture=${KOMASCALER_BENCHMARK_IMAGE:?Set KOMASCALER_BENCHMARK_IMAGE to a monochrome production page}
acceptance_root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
benchmark_root="$acceptance_root/parallelism"
run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
results_root="$benchmark_root/runs/$run_id"
manifest=${KOMASCALER_MODEL_MANIFEST:-$(pwd)/models/models.production.json}

mkdir -p "$results_root" "$acceptance_root/dotnet-home" "$acceptance_root/nuget"
filesystem_type=$(findmnt -n -o FSTYPE -T "$benchmark_root")
if test "$filesystem_type" = tmpfs; then
  echo "Refusing benchmark output on tmpfs: $benchmark_root" >&2
  exit 1
fi
available_kib=$(df -Pk "$benchmark_root" | awk 'NR == 2 { print $4 }')
if test "$available_kib" -lt 3145728; then
  echo "At least 3 GiB free is required; found ${available_kib} KiB." >&2
  exit 1
fi

export DOTNET_CLI_HOME="$acceptance_root/dotnet-home"
export NUGET_PACKAGES="$acceptance_root/nuget"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
publish="$benchmark_root/publish"
dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj --configuration Release --output "$publish"

service_pid=
sampler_pid=
cleanup() {
  if test -n "$sampler_pid"; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; fi
  if test -n "$service_pid"; then kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

factors=${*:-1 2 3 4}
for factor in $factors; do
  case "$factor" in *[!0-9]*|'') echo "Factor must be an integer: $factor" >&2; exit 1 ;; esac
  if test "$factor" -lt 1 || test "$factor" -gt 16; then echo "Factor must be 1..16: $factor" >&2; exit 1; fi
  run_root="$results_root/factor-$factor"
  mkdir -p "$run_root"
  port=$((18900 + factor))
  log="$run_root/service.log"
  timeout --signal=TERM --kill-after=15s 120s \
    env "Kestrel__Endpoints__Http__Url=http://127.0.0.1:$port" \
      "Upscaling__Models__Directory=$model_directory" \
      "Upscaling__Models__InventoryFile=$manifest" \
      "Upscaling__Models__GpuParallelism=$factor" \
      "Upscaling__Models__IdleSessionUnloadAfter=00:10:00" \
      "Upscaling__Cache__Directory=$run_root/cache" \
      dotnet "$publish/komascaler.dll" >"$log" 2>&1 &
  service_pid=$!

  ready=0
  for attempt in $(seq 1 45); do
    if curl --silent --fail --max-time 1 "http://127.0.0.1:$port/health/ready" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$service_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if test "$ready" -ne 1; then
    echo "Factor $factor failed to become ready; see $log" >&2
    exit 1
  fi

  samples="$run_root/samples.tsv"
  unsafe="$run_root/unsafe-pressure"
  (
    echo "utc\tgpu_used_mib\tgpu_util_percent\tram_available_kib\tshmem_kib"
    while kill -0 "$service_pid" 2>/dev/null; do
      utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
      gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ' | tr ',' '\t')
      ram=$(awk '/MemAvailable:/ { available=$2 } /Shmem:/ { shmem=$2 } END { print available "\t" shmem }' /proc/meminfo)
      printf '%s\t%s\t%s\n' "$utc" "$gpu" "$ram"
      gpu_used=$(printf '%s' "$gpu" | cut -f 1)
      ram_available=$(printf '%s' "$ram" | cut -f 1)
      if test "$gpu_used" -gt 10240 || test "$ram_available" -lt 3145728; then
        echo "unsafe gpu_used_mib=$gpu_used ram_available_kib=$ram_available" >"$unsafe"
        kill -TERM "$service_pid" 2>/dev/null || true
        break
      fi
      sleep 1
    done
  ) >"$samples" &
  sampler_pid=$!

  curl --silent --show-error --max-time 35 \
    --dump-header "$run_root/headers" --output "$run_root/output.webp" \
    --write-out '%{http_code}\t%{time_total}\t%{size_download}\n' \
    --form "image=@$fixture" "http://127.0.0.1:$port/convert" >"$run_root/curl.tsv" || true

  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  sampler_pid=
  kill -TERM "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
  service_pid=

  if test -f "$unsafe"; then cat "$unsafe" >&2; exit 1; fi
  result=$(awk 'BEGIN { IGNORECASE=1 } /^X-Upscaler-Result:/ { gsub("\\r", "", $2); print $2 }' "$run_root/headers")
  if test "$result" != upscaled; then echo "Factor $factor did not upscale: $result" >&2; exit 1; fi
  if ! test -f "$results_root/reference.webp"; then cp "$run_root/output.webp" "$results_root/reference.webp"
  else cmp "$results_root/reference.webp" "$run_root/output.webp"
  fi
  summary=$(awk -F '\t' 'NR > 1 { if ($2 > max_gpu) max_gpu=$2; if (min_ram == 0 || $4 < min_ram) min_ram=$4 } END { printf "peakVramMiB=%d minimumRamAvailableMiB=%.1f", max_gpu, min_ram/1024 }' "$samples")
  printf 'factor=%s\t' "$factor"
  cat "$run_root/curl.tsv"
  printf '%s\n' "$summary"
done

trap - EXIT HUP INT TERM
