#!/bin/sh
set -eu

repository=${1:-"$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"}
base=${KOMASCALER_TEST_ROOT:-${TMPDIR:-/tmp}}
temporary=$(mktemp -d "$base/komascaler-systemd-verify.XXXXXX")
cleanup() { rm -rf "$temporary"; }
trap cleanup EXIT HUP INT TERM

# systemd-analyze requires referenced executables to exist. Substitute only
# those paths in a temporary copy so this remains a syntax/sandboxing check and
# never executes or installs the service.
sed \
  -e 's#^ExecStart=.*#ExecStart=/bin/true#' \
  -e 's#^ExecStartPost=.*#ExecStartPost=/bin/true#' \
  "$repository/deploy/systemd/komascaler.service" > "$temporary/komascaler.service"

systemd-analyze verify "$temporary/komascaler.service"
