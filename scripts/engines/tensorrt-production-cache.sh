#!/bin/sh
set -eu

models=${KOMASCALER_GPU_MODEL_DIR:?Set KOMASCALER_GPU_MODEL_DIR}
root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
engines=${KOMASCALER_TENSORRT_CACHE:-$root/tensorrt/engines}
results="$root/tensorrt/production-cache/$(date -u +%Y%m%dT%H%M%SZ)-$$"
publish="$root/tensorrt/publish"
artifacts="$root/tensorrt/publish-artifacts"
manifest="$models/models.production.json"
mkdir -p "$results" "$engines" "$root/dotnet-home" "$root/nuget/packages"

test "$(findmnt -n -o FSTYPE -T "$results")" != tmpfs || { echo "Refusing to use tmpfs." >&2; exit 1; }
test -f "$manifest"
command -v jq >/dev/null

export DOTNET_CLI_HOME="$root/dotnet-home"
export NUGET_PACKAGES="$root/nuget/packages"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

dotnet publish src/KomaScaler.Api/KomaScaler.Api.csproj --configuration Release \
  --artifacts-path "$artifacts" --output "$publish"
dotnet build tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj --configuration Release \
  --artifacts-path "$artifacts"
test_dll="$artifacts/bin/KomaScaler.GpuTests/release/KomaScaler.GpuTests.dll"

subject_pid=
sampler_pid=
cleanup() {
  test -z "$sampler_pid" || { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
  test -z "$subject_pid" || { kill -TERM "$subject_pid" 2>/dev/null || true; wait "$subject_pid" 2>/dev/null || true; }
}
trap cleanup EXIT HUP INT TERM

sample() {
  destination=$1
  watched=$2
  printf 'timestamp,vram_mib,gpu_percent,available_ram_mib,process_state\n' > "$destination"
  while kill -0 "$watched" 2>/dev/null; do
    stamp=$(date +%s.%N)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    vram=$(printf '%s' "$gpu" | cut -d, -f1)
    utilization=$(printf '%s' "$gpu" | cut -d, -f2)
    ram=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)
    state=$(ps -o stat= -p "$watched" 2>/dev/null | tr -d ' ' || true)
    printf '%s,%s,%s,%s,%s\n' "$stamp" "$vram" "$utilization" "$ram" "$state" >> "$destination"
    if test "$vram" -gt 10240 || test "$ram" -lt 3072; then
      printf 'watchdog vram_mib=%s available_ram_mib=%s\n' "$vram" "$ram" > "$results/watchdog-failure.txt"
      kill -TERM "$watched" 2>/dev/null || true
      break
    fi
    sleep 0.2
  done
}

start_epoch=$(date +%s.%N)
env NVIDIA_TF32_OVERRIDE=0 \
  Kestrel__Endpoints__Http__Url=http://127.0.0.1:19999 \
  Upscaling__Models__Directory="$models" \
  Upscaling__Models__TensorRtCacheDirectory="$engines" \
  Upscaling__Models__TensorRtProfileOptimumExtent=960 \
  Upscaling__Models__TensorRtProfileMaximumExtent=960 \
  Upscaling__Models__PreloadOnStartup=true \
  Upscaling__Tiling__MaximumCoreSize=832 \
  Upscaling__Tiling__ContextPixelsPerSide=64 \
  Upscaling__Cache__Directory="$results/result-cache" \
  timeout --signal=TERM --kill-after=15s 900s "$publish/komascaler" > "$results/cold-build.log" 2>&1 &
subject_pid=$!
sample "$results/cold-build-resources.csv" "$subject_pid" & sampler_pid=$!

ready=0
for unused in $(seq 1 900); do
  if curl -fsS --max-time 1 http://127.0.0.1:19999/health/ready > "$results/ready.json" 2>/dev/null; then ready=1; break; fi
  kill -0 "$subject_pid" 2>/dev/null || break
  sleep 1
done
test "$ready" -eq 1 || { wait "$subject_pid" || true; subject_pid=; echo "Production cache build did not reach readiness; see $results" >&2; exit 1; }
ready_epoch=$(date +%s.%N)
kill -TERM "$subject_pid" 2>/dev/null || true
wait "$subject_pid" 2>/dev/null || true
subject_pid=
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
sleep 2

identity='tensorrt|ort=1.26.0|trt=10.14|device=0|fp16=0|int8=0|tf32=0|workspace=2147483648|aux=0|contextSharing=True|timingCache=True|pool=1|profile=input:1x3x8x8-1x3x960x960-1x3x960x960'
printf 'model_id,model_sha256,identity_directory,engine_file,engine_bytes,engine_sha256\n' > "$results/engine-inventory.csv"
jq -r '.models[] | [.id,.sha256] | @tsv' "$manifest" | while IFS="$(printf '\t')" read -r model_id model_hash; do
  cache_id=$(printf '%s' "$model_hash|$identity" | sha256sum | cut -d' ' -f1)
  directory="$engines/$cache_id"
  engine=$(find "$directory" -maxdepth 1 -type f -name '*.engine' -print -quit)
  test -n "$engine" || { echo "Missing production engine for $model_id" >&2; exit 1; }
  bytes=$(stat -c %s "$engine")
  engine_hash=$(sha256sum "$engine" | cut -d' ' -f1)
  printf '%s,%s,%s,%s,%s,%s\n' "$model_id" "$model_hash" "$cache_id" "$(basename "$engine")" "$bytes" "$engine_hash" >> "$results/engine-inventory.csv"
done
test "$(($(wc -l < "$results/engine-inventory.csv") - 1))" -eq 7

printf 'cold_build_seconds=%s\n' "$(awk -v start="$start_epoch" -v end="$ready_epoch" 'BEGIN{printf "%.3f",end-start}')" > "$results/summary.txt"
awk -F, 'NR>1{if($2>peak)peak=$2;if(min==0||$4<min)min=$4}END{print "cold_build_peak_vram_mib="peak;print "cold_build_min_available_ram_mib="min}' "$results/cold-build-resources.csv" >> "$results/summary.txt"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | sed 's/^/post_cold_process_vram_mib=/' >> "$results/summary.txt"

timeout --signal=TERM --kill-after=15s 360s env \
  NVIDIA_TF32_OVERRIDE=0 \
  KOMASCALER_TENSORRT_ACCEPTANCE=1 \
  KOMASCALER_GPU_PROVIDER=TensorRt \
  KOMASCALER_GPU_PARALLELISM=1 \
  KOMASCALER_GPU_MODEL_DIR="$models" \
  KOMASCALER_ACCEPTANCE_ROOT="$root" \
  KOMASCALER_TENSORRT_CACHE="$engines" \
  KOMASCALER_TENSORRT_PROFILE_OPTIMUM=960 \
  KOMASCALER_TENSORRT_PROFILE_MAXIMUM=960 \
  KOMASCALER_PRELOAD_ON_STARTUP=true \
  dotnet test "$test_dll" --filter FullyQualifiedName~AllSevenRoutedSwitchesUseSingleActiveTensorRtSession \
    --logger 'console;verbosity=detailed' > "$results/all-seven-routing.log" 2>&1 &
subject_pid=$!
sample "$results/all-seven-resources.csv" "$subject_pid" & sampler_pid=$!
status=0
wait "$subject_pid" || status=$?
subject_pid=
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=
test "$status" -eq 0 || { cat "$results/all-seven-routing.log" >&2; exit "$status"; }
awk -F, 'NR>1{if($2>peak)peak=$2;if(min==0||$4<min)min=$4}END{print "routing_peak_vram_mib="peak;print "routing_min_available_ram_mib="min}' "$results/all-seven-resources.csv" >> "$results/summary.txt"
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | sed 's/^/post_routing_process_vram_mib=/' >> "$results/summary.txt"

cat "$results/summary.txt"
cat "$results/engine-inventory.csv"
cat "$results/all-seven-routing.log"
echo "RESULTS_ROOT=$results"
trap - EXIT HUP INT TERM
