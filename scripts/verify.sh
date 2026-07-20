#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repository"

dotnet restore KomaScaler.Net.sln
dotnet format KomaScaler.Net.sln --verify-no-changes --no-restore
dotnet build KomaScaler.Net.sln --configuration Release --no-restore
dotnet test tests/KomaScaler.UnitTests/KomaScaler.UnitTests.csproj --configuration Release --no-build
dotnet test tests/KomaScaler.IntegrationTests/KomaScaler.IntegrationTests.csproj --configuration Release --no-build

find scripts tools -type f -name '*.sh' -print | while IFS= read -r script; do
  case "$(head -n 1 "$script")" in
    *bash*) bash -n "$script" ;;
    *) sh -n "$script" ;;
  esac
done

sh scripts/install/test-install-auth.sh
sh scripts/install/test-suwayomi-config.sh
if command -v systemd-analyze >/dev/null 2>&1; then
  sh scripts/install/verify-systemd-unit.sh
fi

echo "Local native-independent verification passed."
echo "Real TensorRT acceptance is separate; see README.md and docs/operations/DEPLOYMENT.md."
