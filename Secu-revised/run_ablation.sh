#!/usr/bin/env bash
# =====================================================================
# run_ablation.sh — train the 6 remaining ablation variants, then score
# each with eval_internal.py (label-free Silhouette / DBI / CHI).
#
# Results accumulate in internal_metrics.csv (one row per --tag), which
# maps directly onto the paper's Table 1 + ablation (a)(b)(c).
#
# The main model (DINOv2 + size-mml, with medoid) is already done:
#   checkpoint  -> model/best_dinov2-mml.pth.tar
#   metrics row -> internal_metrics.csv (tag dinov2-mml)
#
# Usage (from Secu-revised/):
#   bash run_ablation.sh             # full 201-epoch runs (hours)
#   EPOCHS=2 ONLY=rn18-size bash run_ablation.sh   # quick smoke of one variant
# =====================================================================
set -u
cd "$(dirname "$0")"

PY=../venv/Scripts/python.exe
DATA=./data/adaptive_output
EPOCHS="${EPOCHS:-201}"
ONLY="${ONLY:-}"                 # set to a tag to run just that variant
export MPLBACKEND=Agg
export USE_LIBUV=0               # README: avoids libuv error on Windows DDP

mkdir -p logs

# tag | backbone | secu-cst | use-medoid
VARIANTS=(
  "rn18-size|resnet18|size|1"
  "rn18-mml|resnet18|size-mml|1"
  "vit-size|vit|size|1"
  "vit-mml|vit|size-mml|1"
  "dinov2-size|dinov2|size|1"
  "dinov2-nomedoid|dinov2|size-mml|0"
)

port=1240
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r tag bb cst medoid <<< "$v"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ]; then continue; fi
  port=$((port+1))

  echo "==================================================================="
  echo ">>> [$tag] backbone=$bb  cst=$cst  medoid=$medoid  epochs=$EPOCHS"
  echo "==================================================================="

  # Clear any stale best_model so a failed run can't be mis-renamed.
  rm -f model/best_model.pth.tar

  "$PY" main.py "$DATA" -j 4 -p 50 --lr 0.01 --epochs "$EPOCHS" \
    --secu-num-ins 4305 --secu-alpha 517 --secu-k 8 9 10 \
    --clr 0.001 --min-crop 0.2 --log "secu-$tag" \
    --dist-url "tcp://localhost:$port" \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --secu-tx 0.07 --use-medoid "$medoid" --secu-lratio 0.7 --warm-up 30 \
    -b 64 --backbone "$bb" --secu-cst "$cst" --data-name custom \
    > "logs/train_$tag.log" 2>&1
  rc=$?

  if [ $rc -ne 0 ] || [ ! -f model/best_model.pth.tar ]; then
    echo "!!! [$tag] TRAINING FAILED (rc=$rc) — see logs/train_$tag.log; skipping eval"
    continue
  fi

  mv -f model/best_model.pth.tar "model/best_$tag.pth.tar"
  echo ">>> [$tag] trained OK -> model/best_$tag.pth.tar; evaluating..."

  "$PY" eval_internal.py \
    --model-path "model/best_$tag.pth.tar" \
    --backbone "$bb" --secu-cst "$cst" \
    --secu-num-ins 4305 --secu-alpha 517 --secu-k 8 9 10 \
    --secu-tx 0.07 --secu-lratio 0.7 \
    --data-name custom --data-path "$DATA" \
    --tag "$tag" \
    > "logs/eval_$tag.log" 2>&1
  echo ">>> [$tag] eval done; metrics appended to internal_metrics.csv"
  grep -A6 "Internal metrics" "logs/eval_$tag.log" || true
done

echo ""
echo "=== ALL DONE. internal_metrics.csv ==="
cat internal_metrics.csv
