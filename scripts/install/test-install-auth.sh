#!/bin/sh
set -eu

repository=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
base=${KOMASCALER_TEST_ROOT:-${TMPDIR:-/tmp}}
test_directory=$(mktemp -d "$base/komascaler-install-auth.XXXXXX")
cleanup() { rm -rf "$test_directory"; }
trap cleanup EXIT HUP INT TERM

owner=$(id -un)
group=$(id -gn)
unit="$test_directory/komascaler.service"
token="$test_directory/token"

sh "$repository/scripts/install/configure-install-auth.sh" authenticated \
  "$repository/deploy/systemd/komascaler.service" "$unit" "$token" "$owner" "$group"
test -f "$token"
test "$(stat -c %a "$token")" = 600
test "$(tr -d '\n' < "$token" | wc -c)" -eq 64
tr -d '\n' < "$token" | grep -Eq '^[0-9a-f]{64}$'
grep -q '^LoadCredential=Upscaling__Security__Token:' "$unit"
first_hash=$(sha256sum "$token" | cut -d' ' -f1)

sh "$repository/scripts/install/configure-install-auth.sh" authenticated \
  "$repository/deploy/systemd/komascaler.service" "$unit" "$token" "$owner" "$group"
test "$first_hash" = "$(sha256sum "$token" | cut -d' ' -f1)"

sh "$repository/scripts/install/configure-install-auth.sh" disabled \
  "$repository/deploy/systemd/komascaler.service" "$unit" "$token" "$owner" "$group"
test ! -e "$token"
! grep -q '^LoadCredential=Upscaling__Security__Token:' "$unit"
grep -q '^Environment=Upscaling__Security__Token=$' "$unit"
grep -q '^ExecStart=' "$unit"

echo "Authenticated and tokenless installation tests passed."
