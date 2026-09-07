#!/usr/bin/env bash
# E02 depth wall on MNIST, STAGE A: parameter-space baselines only.
#
# Baselines cost ~0.0025 s/step against ALS's ~0.48 s/step at depth 32, so they can be
# swept across the full depth range for the price of a fraction of one ALS cell. The point
# is to locate where (or whether) standard methods actually collapse, before committing
# exact-solve compute to a depth range that may turn out to be uneventful. Stage B runs
# ALS only at the depths this identifies.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

DEPTHS="model.depth=2,4,8,16,32,64,128,256"
BASELINES="optim.type=adam,heavyball,muon,kfac,shampoo,soap"
LRS="optim.lr=0.1,0.03,0.01,0.003,0.001"

echo "=== CReLU / looks-linear (conditioning removed by construction) ==="
python -m olo.run configs/experiments/e02_depth_wall.yaml --quiet \
  --sweep "$DEPTHS" --sweep "$BASELINES" --sweep "$LRS"

echo "=== ReLU / Xavier (collapse present at init) ==="
python -m olo.run configs/experiments/e02_depth_wall.yaml --quiet \
  --set model.type=relu_mlp --set model.init=xavier --set name=e02_relu_xavier \
  --sweep "$DEPTHS" --sweep "$BASELINES" --sweep "$LRS"

echo "=== stage A done: $(ls -d runs/e02_* 2>/dev/null | wc -l) run directories ==="
