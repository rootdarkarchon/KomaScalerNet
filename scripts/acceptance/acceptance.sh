#!/bin/sh
set -eu

acceptance_root=${KOMASCALER_ACCEPTANCE_ROOT:-/devtmp/komascaler-acceptance}
repository=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repository"
mkdir -p "$acceptance_root"
KOMASCALER_TEST_ROOT="$acceptance_root" sh scripts/install/test-install-auth.sh
sh scripts/install/test-suwayomi-config.sh

dotnet restore KomaScaler.Net.sln
dotnet format KomaScaler.Net.sln --verify-no-changes --no-restore
dotnet build KomaScaler.Net.sln --configuration Release --no-restore
dotnet test tests/KomaScaler.UnitTests/KomaScaler.UnitTests.csproj --configuration Release --no-build
dotnet test tests/KomaScaler.IntegrationTests/KomaScaler.IntegrationTests.csproj --configuration Release --no-build

if test -n "${KOMASCALER_GPU_MODEL_DIR:-}"; then
  sh scripts/acceptance/gpu-acceptance.sh
else
  echo "GPU acceptance not run: set KOMASCALER_GPU_MODEL_DIR and ensure CUDA, cuDNN, libvips, and the RTX 3060 are available."
fi
