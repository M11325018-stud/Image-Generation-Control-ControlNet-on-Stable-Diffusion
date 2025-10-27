
#!/usr/bin/env bash
# HW2 Problem 1 - only inference, single-arg interface
# Usage: bash hw2_1.sh <OUTPUT_DIR>
set -euo pipefail

# 1) check arg
if [[ $# -ne 1 ]]; then
  echo "Usage: bash hw2_1.sh <OUTPUT_DIR>"
  exit 1
fi
OUT_DIR="$1"

# 2) locate files (relative paths only)
CKPT="p1_ddpm_cond_unet.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "[Error] missing checkpoint: ${CKPT}"
  exit 2
fi

ENTRY=""
for f in "inference_hw2.1.py" "inference_p1.py" "inference.py"; do
  if [[ -f "$f" ]]; then ENTRY="$f"; break; fi
done
if [[ -z "${ENTRY}" ]]; then
  echo "[Error] cannot find inference script."
  exit 3
fi

# 3) run inference (generates 500 images; even→mnistm, odd→svhn)
python3 "${ENTRY}" \
  --output_dir "${OUT_DIR}" \
  --ckpt "${CKPT}" \
  --seed 2027 \
  --num_steps 1000 \
  --sample_steps 350 \
  --cfg_w 1.8 \
  --trace_digits 0,1 --trace_count 6

