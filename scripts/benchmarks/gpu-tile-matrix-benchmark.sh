#!/bin/sh
set -eu

model_directory=${KOMASCALER_GPU_MODEL_DIR:?Set KOMASCALER_GPU_MODEL_DIR}
fixture=${KOMASCALER_BENCHMARK_IMAGE:?Set KOMASCALER_BENCHMARK_IMAGE to a monochrome production page}
acceptance_root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
benchmark_root="$acceptance_root/tile-matrix"
run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
results_root="$benchmark_root/runs/$run_id"
manifest=${KOMASCALER_MODEL_MANIFEST:-$(pwd)/models/models.production.json}
tile_sizes=${KOMASCALER_TILE_SIZES:-"320 384 448 512 576"}
factors=${KOMASCALER_GPU_FACTORS:-"1 4 6 8"}

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
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$results_root/gpu-before.csv"
free -k >"$results_root/ram-before.txt"

export DOTNET_CLI_HOME="$acceptance_root/dotnet-home"
export NUGET_PACKAGES="$acceptance_root/nuget"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
publish="$benchmark_root/publish"
dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj --configuration Release --output "$publish"

printf 'tile\tfactor\thttp\tseconds\tbytes\tresult\tpeak_vram_mib\tmin_ram_available_mib\tpost_vram_mib\toutput_sha256\n' >"$results_root/summary.tsv"
service_pid=
sampler_pid=
cleanup() {
  if test -n "$sampler_pid"; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; fi
  if test -n "$service_pid"; then kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

index=0
for tile in $tile_sizes; do
  for factor in $factors; do
    index=$((index + 1))
    run_root="$results_root/tile-$tile-factor-$factor"
    mkdir -p "$run_root"
    port=$((19100 + index))
    log="$run_root/service.log"
    echo "BEGIN tile=$tile factor=$factor"

    timeout --signal=TERM --kill-after=15s 150s \
      env "Kestrel__Endpoints__Http__Url=http://127.0.0.1:$port" \
        "Upscaling__Models__Directory=$model_directory" \
        "Upscaling__Models__InventoryFile=$manifest" \
        "Upscaling__Models__GpuParallelism=$factor" \
        "Upscaling__Models__IdleSessionUnloadAfter=00:10:00" \
        "Upscaling__Tiling__MaximumCoreSize=$tile" \
        "Upscaling__Tiling__ContextPixelsPerSide=32" \
        "Upscaling__Queue__ResponseDeadline=00:00:55" \
        "Upscaling__Cache__Directory=$run_root/cache" \
        dotnet "$publish/komascaler.dll" >"$log" 2>&1 &
    service_pid=$!

    ready=0
    for attempt in $(seq 1 50); do
      if curl --silent --fail --max-time 1 "http://127.0.0.1:$port/health/ready" >"$run_root/ready.json" 2>/dev/null; then ready=1; break; fi
      if ! kill -0 "$service_pid" 2>/dev/null; then break; fi
      sleep 1
    done

    if test "$ready" -eq 1; then
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

      curl --silent --show-error --max-time 60 \
        --dump-header "$run_root/headers" --output "$run_root/output.webp" \
        --write-out '%{http_code}\t%{time_total}\t%{size_download}\n' \
        --form "image=@$fixture" "http://127.0.0.1:$port/convert" >"$run_root/curl.tsv" 2>"$run_root/curl.stderr" || true

      kill "$sampler_pid" 2>/dev/null || true
      wait "$sampler_pid" 2>/dev/null || true
      sampler_pid=
    else
      echo '000\t0\t0' >"$run_root/curl.tsv"
      : >"$run_root/samples.tsv"
    fi

    kill -TERM "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
    service_pid=
    sleep 2

    http=$(cut -f 1 "$run_root/curl.tsv")
    seconds=$(cut -f 2 "$run_root/curl.tsv")
    bytes=$(cut -f 3 "$run_root/curl.tsv")
    result=$(awk 'BEGIN { IGNORECASE=1 } /^X-Upscaler-Result:/ { gsub("\r", "", $2); print $2 }' "$run_root/headers" 2>/dev/null || true)
    if test -f "$run_root/unsafe-pressure"; then result=unsafe-pressure; fi
    if test "$ready" -ne 1; then result=not-ready; fi
    if test -z "$result"; then result=no-result; fi
    peak_gpu=$(awk -F '\t' 'NR > 1 && $2 > max { max=$2 } END { print max+0 }' "$run_root/samples.tsv")
    min_ram=$(awk -F '\t' 'NR > 1 && (min == 0 || $4 < min) { min=$4 } END { printf "%.1f", min/1024 }' "$run_root/samples.tsv")
    post_gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if test -f "$run_root/output.webp"; then output_hash=$(sha256sum "$run_root/output.webp" | cut -d ' ' -f 1); else output_hash=-; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tile" "$factor" "$http" "$seconds" "$bytes" "$result" "$peak_gpu" "$min_ram" "$post_gpu" "$output_hash" >>"$results_root/summary.tsv"
    echo "END tile=$tile factor=$factor result=$result seconds=$seconds peakVramMiB=$peak_gpu postVramMiB=$post_gpu"
  done
done

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$results_root/gpu-after.csv"
free -k >"$results_root/ram-after.txt"
echo "RESULTS_ROOT=$results_root"
cat "$results_root/summary.tsv"
trap - EXIT HUP INT TERM
