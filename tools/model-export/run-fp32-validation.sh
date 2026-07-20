#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ONNX_DIR="$ROOT/models/onnx-fp32"
DEMO="$ROOT/fixtures/mangajanaiv1demo.webp"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/extracted/official/models [device-id]" >&2
  exit 2
fi
SOURCE_ROOT="$(realpath -- "$1")"
DEVICE_ID="${2:-0}"

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
mapfile -t ONNX_FILES < <(
  find "$ONNX_DIR" -maxdepth 1 -type f -name '2x_MangaJaNai_*p_V1_ESRGAN_*.fp32.onnx' | sort
)
if [[ ${#MODEL_FILES[@]} -ne 7 ]]; then
  echo "Expected exactly seven official .pth files; found ${#MODEL_FILES[@]}." >&2
  exit 2
fi
if [[ ${#ONNX_FILES[@]} -ne 7 ]]; then
  echo "Expected seven locally exported FP32 ONNX files; found ${#ONNX_FILES[@]}." >&2
  echo "Run ./export-fp32-models.sh first." >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/$STAMP"
RESOLVED="$OUT/resolved-pytorch-models"
mkdir -p "$RESOLVED" "$OUT/environment"
for source in "${MODEL_FILES[@]}"; do
  ln -s "$(realpath -- "$source")" "$RESOLVED/$(basename -- "$source")"
done

echo "Writing FP32 results under: $OUT"
"$PY" "$ROOT/tools/gpu_preflight.py" \
  --device-id "$DEVICE_ID" \
  --output "$OUT/environment/preflight.json" \
  2>&1 | tee "$OUT/preflight.log"
nvidia-smi > "$OUT/environment/nvidia-smi-before.txt" 2>&1
"$PY" -m pip freeze > "$OUT/environment/pip-freeze.txt"
sha256sum "$RESOLVED"/*.pth "${ONNX_FILES[@]}" > "$OUT/environment/model-sha256s.txt"

set +e
"$PY" "$ROOT/tools/mangajanai_reference_validate.py" \
  "$RESOLVED" \
  "$ONNX_DIR" \
  "$OUT/validation" \
  --provider cuda \
  --precision fp32 \
  --cudnn-algo HEURISTIC \
  --official-demo "$DEMO" \
  --all-models-all-fixtures \
  2>&1 | tee "$OUT/validation.log"
VALIDATION_EXIT=${PIPESTATUS[0]}

"$PY" "$ROOT/tools/mangajanai_onnx_gpu_probe.py" \
  "$ONNX_DIR" \
  --precision fp32 \
  --device-id "$DEVICE_ID" \
  --tile 256x256 \
  --cudnn-algo HEURISTIC \
  --warmup-runs 2 \
  --timed-runs 5 \
  --output "$OUT/gpu-probe.json" \
  2>&1 | tee "$OUT/gpu-probe.log"
PROBE_EXIT=${PIPESTATUS[0]}

POST_EXIT=2
if [[ -f "$OUT/gpu-probe.json" ]]; then
  "$PY" "$ROOT/tools/gpu_post_exit.py" \
    "$OUT/gpu-probe.json" \
    --device-id "$DEVICE_ID" \
    --output "$OUT/gpu-post-exit.json" \
    2>&1 | tee "$OUT/gpu-post-exit.log"
  POST_EXIT=${PIPESTATUS[0]}
else
  echo "GPU probe JSON was not produced; post-exit release comparison is unavailable." | tee "$OUT/gpu-post-exit.log"
fi
nvidia-smi > "$OUT/environment/nvidia-smi-after.txt" 2>&1
set -e

"$PY" "$ROOT/tools/make_results_bundle.py" \
  "$OUT" \
  --validation-exit "$VALIDATION_EXIT" \
  --probe-exit "$PROBE_EXIT" \
  --post-exit "$POST_EXIT"

RETURN_ZIP="$ROOT/results/${STAMP}-RETURN.zip"
echo
echo "Run complete. Send back this one file:"
echo "  $RETURN_ZIP"
echo
if [[ $VALIDATION_EXIT -ne 0 ]]; then
  echo "The strict FP32 parity gate did not pass; the archive contains the diagnostics."
fi
if [[ $PROBE_EXIT -ne 0 ]]; then
  echo "The all-seven-session FP32 probe reported an error; the archive contains its log."
fi
if [[ $POST_EXIT -ne 0 ]]; then
  echo "VRAM release was not confirmed within tolerance; the archive contains the measurements."
fi
