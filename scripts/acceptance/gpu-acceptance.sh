#!/bin/sh
set -eu

model_directory=${KOMASCALER_GPU_MODEL_DIR:?Set KOMASCALER_GPU_MODEL_DIR to the seven-model directory}
acceptance_root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
mkdir -p "$acceptance_root"
filesystem_type=$(findmnt -n -o FSTYPE -T "$acceptance_root")
if test "$filesystem_type" = tmpfs; then
  echo "Refusing GPU acceptance output on tmpfs: $acceptance_root" >&2
  exit 1
fi
available_kib=$(df -Pk "$acceptance_root" | awk 'NR == 2 { print $4 }')
if test "$available_kib" -lt 3145728; then
  echo "At least 3 GiB free is required for one reusable GPU test build; found ${available_kib} KiB." >&2
  exit 1
fi

mkdir -p "$acceptance_root/dotnet-home" "$acceptance_root/nuget" "$acceptance_root/samples"
export DOTNET_CLI_HOME="$acceptance_root/dotnet-home"
export NUGET_PACKAGES="$acceptance_root/nuget"
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
export KOMASCALER_ACCEPTANCE_ROOT="$acceptance_root"

dotnet build tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj \
  --configuration Release --artifacts-path "$acceptance_root/artifacts"

sampler_pid=
cleanup_sampler() {
  if test -n "$sampler_pid"; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap cleanup_sampler EXIT HUP INT TERM

for gpu_case in \
  ProviderProfileSingleModel \
  AllSevenRoutedSwitchesUseSingleActiveTensorRtSession \
  ProductionTiling832By64IsSequential \
  ConcurrentTileRunsRespectConfiguredParallelism \
  IdleDrainAndReload \
  DisposalIsIdempotentAfterIdleDrain
do
  sample_file="$acceptance_root/samples/${gpu_case}.tsv"
  (
    echo "utc\tgpu_used_mib\tgpu_util_percent\tram_available_kib\tshmem_kib"
    while :; do
      utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
      gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ' | tr ',' '\t')
      ram=$(awk '/MemAvailable:/ { available=$2 } /Shmem:/ { shmem=$2 } END { print available "\t" shmem }' /proc/meminfo)
      printf '%s\t%s\t%s\n' "$utc" "$gpu" "$ram"
      sleep 1
    done
  ) >"$sample_file" &
  sampler_pid=$!
  echo "Running isolated GPU case: $gpu_case"
  timeout --signal=TERM --kill-after=15s 180s \
    dotnet test tests/KomaScaler.GpuTests/KomaScaler.GpuTests.csproj \
      --configuration Release --no-build --no-restore \
      --artifacts-path "$acceptance_root/artifacts" \
      --filter "FullyQualifiedName~$gpu_case" \
      --logger 'console;verbosity=detailed'
  cleanup_sampler
  sampler_pid=
  awk -F '\t' 'NR > 1 { if ($2 > max_gpu) max_gpu=$2; if (min_ram == 0 || $4 < min_ram) min_ram=$4; if ($5 > max_shmem) max_shmem=$5 } END { printf "case=%s peakVramMiB=%d minimumRamAvailableMiB=%.1f peakShmemMiB=%.1f samples=%d\n", file, max_gpu, min_ram/1024, max_shmem/1024, NR-1 }' file="$gpu_case" "$sample_file"
done

trap - EXIT HUP INT TERM
