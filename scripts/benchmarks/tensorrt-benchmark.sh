#!/bin/sh
set -eu

root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
models=${KOMASCALER_GPU_MODEL_DIR:-/var/lib/komascaler/models}
fixture=${KOMASCALER_BENCHMARK_IMAGE:-$root/http-fixtures/neutral.png}
publish="$root/tensorrt/publish"
engines=${KOMASCALER_TENSORRT_CACHE:-$root/tensorrt/engines}
factors=${KOMASCALER_TENSORRT_FACTORS:-1 2 4 6 8}
candidate=${KOMASCALER_TENSORRT_SUSTAINED_FACTOR:-}
include_cuda=${KOMASCALER_INCLUDE_CUDA_BASELINE:-1}
include_tensor_rt=${KOMASCALER_INCLUDE_TENSORRT:-1}
cuda_sustained=${KOMASCALER_CUDA_SUSTAINED:-0}
tile_core=${KOMASCALER_TILE_CORE:-320}
tile_context=${KOMASCALER_TILE_CONTEXT:-32}
profile_optimum=${KOMASCALER_TENSORRT_PROFILE_OPTIMUM:-384}
profile_maximum=${KOMASCALER_TENSORRT_PROFILE_MAXIMUM:-704}
preload=${KOMASCALER_PRELOAD_ON_STARTUP:-true}
smoke_pages=${KOMASCALER_SMOKE_PAGES:-3}
run_root="$root/tensorrt/runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$publish" "$engines" "$run_root"

case "$(findmnt -no FSTYPE -T "$root")" in tmpfs|ramfs) echo "refusing memory-backed root" >&2; exit 2;; esac
test "$(df -Pk "$root" | awk 'NR==2 {print $4}')" -ge 3145728 || { echo "less than 3 GiB free" >&2; exit 2; }
test -f "$fixture"
test "$(find "$models" -maxdepth 1 -name '*.onnx' | wc -l)" -eq 7
dotnet_home="$root/dotnet-home"
packages="$root/nuget/packages"
artifacts="$root/tensorrt/publish-artifacts"
mkdir -p "$dotnet_home" "$packages" "$artifacts"
env DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  dotnet restore src/KomaScaler.Api/KomaScaler.Api.csproj --configfile config/NuGet.Config --artifacts-path "$artifacts"
env DOTNET_CLI_HOME="$dotnet_home" NUGET_PACKAGES="$packages" \
  dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj -c Release --no-restore \
    --artifacts-path "$artifacts" -o "$publish"

service_pid=
sampler_pid=
cleanup() {
  test -z "$sampler_pid" || kill "$sampler_pid" 2>/dev/null || true
  test -z "$service_pid" || kill "$service_pid" 2>/dev/null || true
  test -z "$service_pid" || wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

sample() {
  destination=$1
  while :; do
    stamp=$(date +%s.%N)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    ram=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    state=$(test -z "$service_pid" && echo none || ps -o stat= -p "$service_pid" 2>/dev/null | tr -d ' ')
    printf '%s,%s,%s,%s\n' "$stamp" "$gpu" "$ram" "${state:-exited}" >> "$destination"
    vram=$(printf '%s' "$gpu" | cut -d, -f1)
    if test "$vram" -gt 10240 || test "$ram" -lt 3072; then kill "$service_pid" 2>/dev/null || true; exit 70; fi
    sleep 0.2
  done
}

run_case() {
  provider=$1
  factor=$2
  pages=$3
  label=${4:-smoke}
  case_root="$run_root/${provider}-c${tile_core}-x${tile_context}-p${factor}-${label}"
  cache="$case_root/result-cache"
  mkdir -p "$case_root" "$cache"
  printf 'timestamp,vram_mib,gpu_percent,available_ram_mib,process_state\n' > "$case_root/resources.csv"
  port=$((21000 + factor + $(test "$provider" = TensorRt && echo 20 || echo 0)))
  env NVIDIA_TF32_OVERRIDE=0 \
    Kestrel__Endpoints__Http__Url="http://127.0.0.1:$port" \
    Upscaling__Models__Directory="$models" \
    Upscaling__Models__ExecutionProvider="$provider" \
    Upscaling__Models__GpuParallelism="$factor" \
    Upscaling__Models__TensorRtSessionPoolSize="$factor" \
    Upscaling__Models__TensorRtCacheDirectory="$engines" \
    Upscaling__Models__TensorRtProfileOptimumExtent="$profile_optimum" \
    Upscaling__Models__TensorRtProfileMaximumExtent="$profile_maximum" \
    Upscaling__Models__PreloadOnStartup="$preload" \
    Upscaling__Tiling__MaximumCoreSize="$tile_core" \
    Upscaling__Tiling__ContextPixelsPerSide="$tile_context" \
    Upscaling__Cache__Directory="$cache" \
    Logging__LogLevel__KomaScaler.Inference.GpuUpscaler=Debug \
    timeout --signal=TERM --kill-after=10s 900s "$publish/komascaler" > "$case_root/service.log" 2>&1 &
  service_pid=$!
  sample "$case_root/resources.csv" & sampler_pid=$!
  ready=0
  for attempt in $(seq 1 240); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health/ready" > "$case_root/ready.json" 2>/dev/null; then ready=1; break; fi
    kill -0 "$service_pid" 2>/dev/null || break
    sleep 1
  done
  test "$ready" -eq 1 || { echo "$provider/$factor did not become ready" >&2; return 1; }
  printf 'page,start_epoch,end_epoch,http_seconds,bytes,result,post_page_vram_mib,available_ram_mib\n' > "$case_root/pages.csv"
  page=0
  while test "$page" -lt "$pages"; do
    unique="$case_root/input-$page.png"
    cp "$fixture" "$unique"
    printf '%08d' "$page" >> "$unique"
    started=$(date +%s.%N)
    metrics=$(curl -fsS --max-time 28 -w '%{time_total},%{size_download}' -o "$case_root/output-$page.image" -D "$case_root/headers-$page" -F "image=@$unique" "http://127.0.0.1:$port/convert")
    ended=$(date +%s.%N)
    result=$(awk 'BEGIN{IGNORECASE=1} /^X-Upscaler-Result:/ {gsub("\r",""); print $2}' "$case_root/headers-$page")
    post_vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    available_ram=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$page" "$started" "$ended" "$metrics" "$result" "$post_vram" "$available_ram" >> "$case_root/pages.csv"
    page=$((page + 1))
  done
  cache_started=$(date +%s.%N)
  cache_metrics=$(curl -fsS --max-time 28 -w '%{time_total},%{size_download}' \
    -o "$case_root/cache-hit.image" -D "$case_root/cache-hit.headers" \
    -F "image=@$case_root/input-0.png" "http://127.0.0.1:$port/convert")
  cache_ended=$(date +%s.%N)
  cache_result=$(awk 'BEGIN{IGNORECASE=1} /^X-Upscaler-Result:/ {gsub("\r",""); print $2}' "$case_root/cache-hit.headers")
  printf 'start_epoch,end_epoch,http_seconds,bytes,result\n%s,%s,%s,%s\n' \
    "$cache_started" "$cache_ended" "$cache_metrics" "$cache_result" > "$case_root/cache-hit.csv"
  kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; sampler_pid=
  kill "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; service_pid=
  sleep 2
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$case_root/post-exit-vram.txt"
  awk -F, 'NR>2 {n++; x=n; y=$7; sx+=x; sy+=y; sxy+=x*y; sx2+=x*x} END {if(n>1) printf "post_warmup_vram_slope_mib_per_page=%.6f\n",(n*sxy-sx*sy)/(n*sx2-sx*sx); else print "post_warmup_vram_slope_mib_per_page=not-enough-pages"}' "$case_root/pages.csv" > "$case_root/slope.txt"
}

test "$include_cuda" -eq 0 || run_case Cuda 8 3
test "$cuda_sustained" -eq 0 || run_case Cuda 8 21 sustained
if test "$include_tensor_rt" -ne 0; then
  for factor in $factors; do
    run_case TensorRt "$factor" "$smoke_pages"
    peak=$(awk -F, 'NR>1 && $2+0>m {m=$2+0} END {print m+0}' "$run_root/TensorRt-c${tile_core}-x${tile_context}-p$factor-smoke/resources.csv")
    printf 'factor=%s sampled_peak_vram_mib=%s\n' "$factor" "$peak" >> "$run_root/incremental-pools.txt"
    if test "$peak" -ge 9500; then
      printf 'stopped_before_next_factor=1 reason=peak_vram_approaching_watchdog\n' >> "$run_root/incremental-pools.txt"
      break
    fi
  done
  test -z "$candidate" || run_case TensorRt "$candidate" 21 sustained
fi
printf '%s\n' "$run_root"
