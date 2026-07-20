#!/bin/sh
set -eu

root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
models=${KOMASCALER_GPU_MODEL_DIR:-/var/lib/komascaler/models}
results="$root/tensorrt/acceptance"
test_dll="$root/tensorrt/test-artifacts/bin/KomaScaler.GpuTests/release/KomaScaler.GpuTests.dll"
mkdir -p "$results"

KOMASCALER_ACCEPTANCE_ROOT="$root" KOMASCALER_GPU_MODEL_DIR="$models" sh scripts/engines/tensorrt-preflight.sh

monitor_pid=
test_pid=
cleanup() {
  test -z "$monitor_pid" || kill "$monitor_pid" 2>/dev/null || true
  test -z "$test_pid" || kill "$test_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

monitor() {
  while kill -0 "$test_pid" 2>/dev/null; do
    stamp=$(date +%s.%N)
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    ram=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    printf '%s,%s,%s\n' "$stamp" "$vram" "$ram" >> "$results/resources.csv"
    if test "$vram" -gt 10240 || test "$ram" -lt 3072; then kill "$test_pid" 2>/dev/null || true; exit 70; fi
    sleep 0.2
  done
}

filter='FullyQualifiedName~AllSevenRoutedSwitchesUseSingleActiveTensorRtSession|FullyQualifiedName~ProductionTiling832By64IsSequential|FullyQualifiedName~TensorRtActiveModelSwitchLifecycle|FullyQualifiedName~TensorRtProviderNumericalIsolation|FullyQualifiedName~IdleDrainAndReload|FullyQualifiedName~DisposalIsIdempotentAfterIdleDrain'
printf 'timestamp,vram_mib,available_ram_mib\n' > "$results/resources.csv"
timeout --signal=TERM --kill-after=15s 900s env \
  NVIDIA_TF32_OVERRIDE=0 \
  KOMASCALER_TENSORRT_ACCEPTANCE=1 \
  KOMASCALER_GPU_PROVIDER=TensorRt \
  KOMASCALER_GPU_MODEL_DIR="$models" \
  KOMASCALER_ACCEPTANCE_ROOT="$root" \
  KOMASCALER_TENSORRT_CACHE="$root/tensorrt/engines" \
  dotnet test "$test_dll" \
    --filter "$filter" --logger "console;verbosity=detailed" > "$results/lifecycle.log" 2>&1 &
test_pid=$!
monitor & monitor_pid=$!
wait "$test_pid"
test_pid=
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

timeout --signal=TERM --kill-after=15s 900s env \
  NVIDIA_TF32_OVERRIDE=0 \
  KOMASCALER_TENSORRT_ACCEPTANCE=1 \
  KOMASCALER_GPU_PROVIDER=TensorRt \
  KOMASCALER_GPU_MODEL_DIR="$models" \
  KOMASCALER_ACCEPTANCE_ROOT="$root" \
  KOMASCALER_TENSORRT_CACHE="$root/tensorrt/engines" \
  KOMASCALER_QUALITY_OUTPUT="$results/quality.json" \
  dotnet test "$test_dll" \
    --filter FullyQualifiedName~UntiledDirectOrtQualityReferenceMatrix \
    --logger "console;verbosity=detailed" > "$results/quality.log" 2>&1 &
test_pid=$!
monitor & monitor_pid=$!
wait "$test_pid"
test_pid=
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
