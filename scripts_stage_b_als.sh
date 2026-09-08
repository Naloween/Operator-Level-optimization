#!/usr/bin/env bash
# E02 depth wall, STAGE B: ALS at the depths Stage A identified.
#
#   ./scripts_stage_b_als.sh 8 32 128
#
# Stage A swept the six parameter-space baselines across depths 2..256 because they are
# ~200x cheaper per step than the exact solve. This adds ALS only where the answer is in
# doubt, since one ALS cell at depth 128 costs about as much as an entire depth of
# baselines.
#
# Cost, measured on this configuration (1500 steps, MNIST, width 32):
#   depth   2      0.2 min/cell        depth  32     12.1 min/cell
#   depth   8      2.5 min/cell        depth  64    ~25   min/cell
#   depth  16      5.7 min/cell        depth 128    ~50   min/cell
# times the number of learning rates. Budget before launching.
#
# ALS solves every square layer exactly and hands the 784-wide input layer to Adam; the
# split is recorded in each run's meta.json.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <depth> [depth ...]" >&2
  echo "  run scripts_depth_wall.sh first, then read the curve to pick depths" >&2
  exit 2
fi

LRS="optim.lr=0.1,0.03,0.01,0.003,0.001"
CFG=configs/experiments/e02_depth_wall.yaml

for depth in "$@"; do
  echo "--- CReLU/looks-linear ALS depth $depth ---"
  python -m olo.run "$CFG" --quiet --set "model.depth=$depth" \
    --set optim.type=als --set "name=e02_crelu_L${depth}" \
    --sweep "$LRS" || exit 1

  echo "--- ReLU/Xavier ALS depth $depth ---"
  python -m olo.run "$CFG" --quiet --set "model.depth=$depth" \
    --set model.type=relu_mlp --set model.init=xavier \
    --set optim.type=als --set "name=e02_relu_L${depth}" \
    --sweep "$LRS" || exit 1

  echo "done ALS depth $depth: $(ls -d runs/e02_* 2>/dev/null | wc -l) runs total"
done

# Close any cell whose optimum landed on a grid edge, in either family, before the curve
# is read: an untuned cell is a lower bound, not a result.
echo "--- extending learning-rate grids where the optimum is at an edge ---"
python -m olo.extend_lr "e02_*" --metric eval_val_accuracy --max-rounds 3

echo "=== stage B complete: $(ls -d runs/e02_* 2>/dev/null | wc -l) run directories ==="
