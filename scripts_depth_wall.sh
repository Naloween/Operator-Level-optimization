#!/usr/bin/env bash
# E02 depth wall, 1-seed pass. Two model families, same grid, lambda fixed at 1e-4.
# Runs sequentially: both families share one GPU, so parallelism would only contend.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

GRID_DEPTH="model.depth=2,4,8,16,32,64,128,256"
GRID_METHOD="optim.type=als,adam,heavyball,muon,kfac,shampoo,soap"
GRID_LR="optim.lr=1.0,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001"

echo "=== CReLU / looks-linear (conditioning removed by construction) ==="
python -m olo.run configs/experiments/e02_depth_wall.yaml --quiet \
  --sweep "$GRID_DEPTH" --sweep "$GRID_METHOD" --sweep "$GRID_LR"

echo "=== ReLU / Xavier (collapse present) ==="
python -m olo.run configs/experiments/e02_depth_wall.yaml --quiet \
  --set model.type=relu_mlp --set model.init=xavier --set name=e02_relu_xavier \
  --sweep "$GRID_DEPTH" --sweep "$GRID_METHOD" --sweep "$GRID_LR"

echo "=== done: $(ls -d runs/e02_* 2>/dev/null | wc -l) run directories ==="
