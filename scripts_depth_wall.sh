#!/usr/bin/env bash
# E02 depth wall on MNIST, STAGE A: parameter-space baselines only.
#
# Baselines cost ~0.0025 s/step against ALS's ~0.48 s/step at depth 32, so they can be
# swept across the full depth range for the price of a fraction of one ALS cell. The point
# is to locate where (or whether) standard methods actually collapse before committing
# exact-solve compute. Stage B runs ALS only at the depths this identifies.
#
# One (family, depth) per invocation. `olo.run` skips a run whose directory already
# exists, so progress is durable and re-running resumes where it stopped; chunking keeps
# any single invocation short, so an interruption costs minutes rather than the sweep.
#
# NOTE the explicit per-chunk `name`. Values passed with --set do not enter the run name
# (only multi-valued --sweep axes do), so without this every depth would map to the same
# directory, all but the first would be skipped as already-existing, and the sweep would
# report success while containing one depth repeated. `olo.run` now refuses that case
# outright, but the names still have to be right.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

BASELINES="optim.type=adam,heavyball,muon,kfac,shampoo,soap"
LRS="optim.lr=0.1,0.03,0.01,0.003,0.001"
CFG=configs/experiments/e02_depth_wall.yaml

for depth in 2 4 8 16 32 64 128 256; do
  python -m olo.run "$CFG" --quiet --set "model.depth=$depth" \
    --set "name=e02_crelu_L${depth}" \
    --sweep "$BASELINES" --sweep "$LRS" || exit 1

  python -m olo.run "$CFG" --quiet --set "model.depth=$depth" \
    --set model.type=relu_mlp --set model.init=xavier \
    --set "name=e02_relu_L${depth}" \
    --sweep "$BASELINES" --sweep "$LRS" || exit 1

  echo "done depth $depth: $(ls -d runs/e02_* 2>/dev/null | wc -l) runs total"
done
echo "=== stage A complete: $(ls -d runs/e02_* 2>/dev/null | wc -l) run directories ==="
