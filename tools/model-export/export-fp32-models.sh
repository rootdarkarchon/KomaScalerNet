#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUT="$ROOT/models/onnx-fp32"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/extracted/official/models" >&2
  exit 2
fi
SOURCE_ROOT="$(realpath -- "$1")"

if [[ ! -x "$PY" ]]; then
  echo "The Python environment is missing. Use the existing kit's .venv or set PYTHON_BIN." >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Model directory does not exist: $SOURCE_ROOT" >&2
  exit 2
fi

mapfile -d '' MODEL_FILES < <(
  find "$SOURCE_ROOT" -type f -name '2x_MangaJaNai_*p_V1_ESRGAN_*.pth' -print0 | sort -z
)
if [[ ${#MODEL_FILES[@]} -ne 7 ]]; then
  echo "Expected exactly seven official 2x MangaJaNai V1 .pth files; found ${#MODEL_FILES[@]}." >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESOLVED="$ROOT/work/fp32-export-sources-$STAMP"
mkdir -p "$RESOLVED" "$OUT"
for source in "${MODEL_FILES[@]}"; do
  ln -s "$(realpath -- "$source")" "$RESOLVED/$(basename -- "$source")"
done

echo "Exporting seven FP32 ONNX models to: $OUT"
"$PY" "$ROOT/tools/mangajanai_onnx_export.py" \
  "$RESOLVED" \
  "$OUT" \
  --precision fp32 \
  --provider cpu \
  2>&1 | tee "$OUT/fp32-export.log"

mapfile -t ONNX_FILES < <(find "$OUT" -maxdepth 1 -type f -name '*.fp32.onnx' | sort)
if [[ ${#ONNX_FILES[@]} -ne 7 ]]; then
  echo "Export did not produce exactly seven FP32 ONNX files." >&2
  exit 1
fi
sha256sum "${ONNX_FILES[@]}" > "$OUT/fp32-sha256s.txt"

echo
echo "FP32 export complete. Continue with:"
echo "  ./run-fp32-validation.sh '$SOURCE_ROOT'"
