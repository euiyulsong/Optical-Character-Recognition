#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data output results
rm -f results/*.json

run() {
  echo
  echo "======================================================================"
  echo "[$1]"
  echo "======================================================================"
  shift
  "$@"
}

# Separate datasets are intentional:
# - recognition: cropped word images -> train_raw/val_raw.jsonl
# - detection/unwarping: full document-like SynthText images -> annotations.json
run "1/7 Download recognition crops" python3 downlaod.py
run "2/7 Download detection/unwarping images" python3 download_det.py
run "3/7 Train detection - Mini DBNet" python3 dbnet.py
run "4/7 Train recognition - SVTR-LCNet" python3 svtr_lcnet_rec.py
run "5/7 Train unwarping - Mini UVDoc" python3 train_uv.py
run "6/7 Evaluate unwarping" python3 eval_uvdoc.py
run "7/7 Final result table" python3 summarize_results.py
