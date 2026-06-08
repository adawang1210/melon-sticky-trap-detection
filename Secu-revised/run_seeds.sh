#!/usr/bin/env bash
# run_seeds.sh — extra seeds for the medoid ablation, to put error bars on the
# +2% medoid effect (Item 5). Trains DINOv2 size-mml with and without medoid
# for seeds 1 and 2 (the original run is the seed-0/None data point), evals
# each, and appends to internal_metrics_seeds.csv.
set -u
cd "$(dirname "$0")"
PY=../venv/Scripts/python.exe
DATA=./data/adaptive_output
export MPLBACKEND=Agg
export USE_LIBUV=0
mkdir -p logs

# tag | backbone | secu-cst | use-medoid
VARIANTS=( "dinov2-mml|dinov2|size-mml|1" "dinov2-nomedoid|dinov2|size-mml|0" )

port=1300
for seed in 1 2; do
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r tag bb cst medoid <<< "$v"
    port=$((port+1))
    name="${tag}-s${seed}"
    echo "=== [$name] seed=$seed medoid=$medoid ==="
    rm -f model/best_model.pth.tar
    timeout 9000 "$PY" main.py "$DATA" -j 4 -p 50 --lr 0.01 --epochs 201 \
      --secu-num-ins 4305 --secu-alpha 517 --secu-k 8 9 10 \
      --clr 0.001 --min-crop 0.2 --log "secu-$name" --seed "$seed" \
      --dist-url "tcp://localhost:$port" \
      --multiprocessing-distributed --world-size 1 --rank 0 \
      --secu-tx 0.07 --use-medoid "$medoid" --secu-lratio 0.7 --warm-up 30 \
      -b 64 --backbone "$bb" --secu-cst "$cst" --data-name custom \
      > "logs/train_$name.log" 2>&1
    if [ $? -ne 0 ] || [ ! -f model/best_model.pth.tar ]; then
      echo "!!! [$name] TRAIN FAILED — see logs/train_$name.log"; continue
    fi
    mv -f model/best_model.pth.tar "model/best_$name.pth.tar"
    "$PY" eval_internal.py --model-path "model/best_$name.pth.tar" \
      --backbone "$bb" --secu-cst "$cst" \
      --secu-num-ins 4305 --secu-alpha 517 --secu-k 8 9 10 \
      --secu-tx 0.07 --secu-lratio 0.7 \
      --data-name custom --data-path "$DATA" \
      --tag "$name" --out internal_metrics_seeds.csv > "logs/eval_$name.log" 2>&1
    echo ">>> [$name] done"
  done
done
echo "=== ALL SEED RUNS DONE ==="
cat internal_metrics_seeds.csv
