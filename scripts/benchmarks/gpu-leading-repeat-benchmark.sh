#!/bin/sh
set -eu

model_directory=${KOMASCALER_GPU_MODEL_DIR:?Set KOMASCALER_GPU_MODEL_DIR}
fixture=${KOMASCALER_BENCHMARK_IMAGE:?Set KOMASCALER_BENCHMARK_IMAGE}
acceptance_root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
benchmark_root="$acceptance_root/leading-repeat"
run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
results_root="$benchmark_root/runs/$run_id"
manifest=${KOMASCALER_MODEL_MANIFEST:-$(pwd)/models/models.production.json}
context=${KOMASCALER_TILE_CONTEXT:-64}
combinations=${KOMASCALER_LEADING_COMBINATIONS:-"512:6 512:8 576:6 576:8"}
shapes=${KOMASCALER_PAGE_SHAPES:-"landscape portrait square"}
repeats=${KOMASCALER_REPEATS:-3}
mkdir -p "$results_root/fixtures"

test "$(findmnt -n -o FSTYPE -T "$results_root")" != tmpfs || { echo "Refusing tmpfs." >&2; exit 1; }
test "$(df -Pk "$results_root" | awk 'NR == 2 { print $4 }')" -ge 3145728 || { echo "At least 3 GiB free is required." >&2; exit 1; }
command -v magick >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader >"$results_root/gpu-before.csv"

# Three page shapes and three byte-distinct variants avoid cache hits while
# keeping decoded pixels identical within each shape.
magick "$fixture" "$results_root/fixtures/landscape-base.png"
magick "$fixture" -rotate 90 "$results_root/fixtures/portrait-base.png"
magick "$fixture" -gravity center -crop 1400x1400+0+0 +repage "$results_root/fixtures/square-base.png"
for shape in landscape portrait square; do
  for repeat in $(seq 1 "$repeats"); do
    magick "$results_root/fixtures/$shape-base.png" -set comment "repeat-$repeat" "$results_root/fixtures/$shape-$repeat.png"
  done
done

export DOTNET_CLI_HOME="$acceptance_root/dotnet-home"
export NUGET_PACKAGES="$acceptance_root/nuget"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
publish="$benchmark_root/publish"
dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj --configuration Release --output "$publish"

printf 'core\tfactor\tshape\trepeat\thttp\tseconds\tbytes\tresult\tsha256\n' >"$results_root/requests.tsv"
printf 'core\tfactor\tpeak_vram_mib\tminimum_ram_available_mib\tpost_vram_mib\tsamples\n' >"$results_root/pressure.tsv"
service_pid=
sampler_pid=
cleanup() {
  if test -n "$sampler_pid"; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; fi
  if test -n "$service_pid"; then kill -TERM "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

index=0
for combination in $combinations; do
  core=${combination%:*}
  factor=${combination#*:}
  index=$((index + 1))
  run_root="$results_root/core-$core-factor-$factor"
  mkdir -p "$run_root"
  port=$((19300 + index))
  echo "BEGIN core=$core factor=$factor context=$context"
  timeout --signal=TERM --kill-after=15s 240s \
    env "Kestrel__Endpoints__Http__Url=http://127.0.0.1:$port" \
      "Upscaling__Models__Directory=$model_directory" \
      "Upscaling__Models__InventoryFile=$manifest" \
      "Upscaling__Models__GpuParallelism=$factor" \
      "Upscaling__Tiling__MaximumCoreSize=$core" \
      "Upscaling__Tiling__ContextPixelsPerSide=$context" \
      "Upscaling__Queue__ResponseDeadline=00:00:55" \
      "Upscaling__Cache__Directory=$run_root/cache" \
      dotnet "$publish/komascaler.dll" >"$run_root/service.log" 2>&1 &
  service_pid=$!

  ready=0
  for attempt in $(seq 1 50); do
    if curl --silent --fail --max-time 1 "http://127.0.0.1:$port/health/ready" >"$run_root/ready.json" 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$service_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  test "$ready" -eq 1 || { echo "Not ready: core=$core factor=$factor" >&2; exit 1; }

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
        echo "unsafe gpu_used_mib=$gpu_used ram_available_kib=$ram_available" >"$run_root/unsafe-pressure"
        kill -TERM "$service_pid" 2>/dev/null || true
        break
      fi
      sleep 0.2
    done
  ) >"$run_root/samples.tsv" &
  sampler_pid=$!

  for shape in $shapes; do
    for repeat in $(seq 1 "$repeats"); do
      request="$run_root/$shape-$repeat"
      curl --silent --show-error --max-time 60 \
        --dump-header "$request.headers" --output "$request.webp" \
        --write-out '%{http_code}\t%{time_total}\t%{size_download}\n' \
        --form "image=@$results_root/fixtures/$shape-$repeat.png" \
        "http://127.0.0.1:$port/convert" >"$request.curl" 2>"$request.stderr" || true
      http=$(cut -f 1 "$request.curl")
      seconds=$(cut -f 2 "$request.curl")
      bytes=$(cut -f 3 "$request.curl")
      result=$(awk 'BEGIN { IGNORECASE=1 } /^X-Upscaler-Result:/ { gsub("\r", "", $2); print $2 }' "$request.headers")
      if test -f "$request.webp"; then hash=$(sha256sum "$request.webp" | cut -d ' ' -f 1); else hash=-; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$core" "$factor" "$shape" "$repeat" "$http" "$seconds" "$bytes" "$result" "$hash" >>"$results_root/requests.tsv"
      test "$http" = 200 && test "$result" = upscaled || { echo "Request failed: core=$core factor=$factor shape=$shape repeat=$repeat" >&2; exit 1; }
    done
  done

  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  sampler_pid=
  kill -TERM "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
  service_pid=
  sleep 2
  test ! -f "$run_root/unsafe-pressure" || { cat "$run_root/unsafe-pressure" >&2; exit 1; }
  post_gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  awk -F '\t' -v core="$core" -v factor="$factor" -v post="$post_gpu" 'NR > 1 { if ($2 > max) max=$2; if (min == 0 || $4 < min) min=$4 } END { printf "%s\t%s\t%d\t%.1f\t%s\t%d\n", core, factor, max, min/1024, post, NR-1 }' "$run_root/samples.tsv" >>"$results_root/pressure.tsv"
  echo "END core=$core factor=$factor postVramMiB=$post_gpu"
done

echo "RESULTS_ROOT=$results_root"
cat "$results_root/requests.tsv"
cat "$results_root/pressure.tsv"
trap - EXIT HUP INT TERM
