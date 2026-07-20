#!/bin/sh
set -eu

repository=${1:-"$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"}
example="$repository/deploy/suwayomi/server.conf.example"

grep -Fq 'server.downloadConversions = {}' "$example"
if grep -Eq 'server\.downloadConversions[[:space:]]*=[[:space:]]*\[' "$example"; then
  echo "downloadConversions must be a HOCON object, not a list" >&2
  exit 1
fi

echo "Suwayomi conversion configuration shape is valid."
