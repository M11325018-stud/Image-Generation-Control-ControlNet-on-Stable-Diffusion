#!/usr/bin/env bash
# Usage: bash hw2_2.sh <NOISE_DIR> <OUTPUT_DIR> <WEIGHT_PATH>
#  $1: path to predefined noises (e.g., hw2_data/face/noise)
#  $2: path to output images directory
#  $3: path to pretrained weight (e.g., hw2_data/face/UNet.pt)

set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: bash hw2_2.sh <NOISE_DIR> <OUTPUT_DIR> <WEIGHT_PATH>"
  exit 1
fi

NOISE_DIR="$1"
OUTPUT_DIR="$2"
WEIGHT_PATH="$3"

# 基本檢查
[ -d "$NOISE_DIR" ]   || { echo "[ERR] noise dir not found: $NOISE_DIR"; exit 2; }
[ -f "$WEIGHT_PATH" ] || { echo "[ERR] weight not found:   $WEIGHT_PATH"; exit 3; }

# 以腳本所在目錄為基準找 inference 檔
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INF_PY="$SCRIPT_DIR/inference_hw2.2.py"
[ -f "$INF_PY" ] || { echo "[ERR] inference script not found: $INF_PY"; exit 4; }

# 準備輸出資料夾
mkdir -p "$OUTPUT_DIR"

echo "[P2] running inference..."
echo "     noise  : $NOISE_DIR"
echo "     output : $OUTPUT_DIR"
echo "     weight : $WEIGHT_PATH"

# 5 分鐘時限（系統有 timeout 才套用）
if command -v timeout >/dev/null 2>&1; then
  timeout 300s python3 "$INF_PY" \
    --noise_dir "$NOISE_DIR" \
    --out_dir   "$OUTPUT_DIR" \
    --weight    "$WEIGHT_PATH" \
    --steps 50
else
  python3 "$INF_PY" \
    --noise_dir "$NOISE_DIR" \
    --out_dir   "$OUTPUT_DIR" \
    --weight    "$WEIGHT_PATH" \
    --steps 50
fi

echo "[P2] done. Images should be 00.png ~ 09.png in $OUTPUT_DIR"
