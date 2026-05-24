#!/bin/bash
# Sprint 6 Cluster B launcher: ratio-sanity cells (B.2) + ensemble training (B.4).
#
# Modes:
#   sanity             Train 2 ratio-sanity cells at seed=20260601 (ratios 0.25, 0.50)
#                      on the expanded 10c real corpus. ~1.7 hr wall on A100.
#   ensemble <RATIO>   Train 5 ensemble members at seeds {20260601..20260605} at RATIO.
#                      ~4.2 hr wall on A100. RATIO is locked from B.3 decision.
#
# Resume-skip: cells with best.pt already present are skipped, so this script is
# safe to re-run after interrupts. Per-cell logs go to artifacts/sprint6_ensemble/logs/.
#
# Evaluation (per-member tier1_metrics_b0.001_c0.5.json) is a separate phase (B.5);
# see scripts/evaluate_ml_detector.py.

set -uo pipefail

MODE="${1:-}"
ENSEMBLE_RATIO="${2:-}"

OUTDIR=artifacts/sprint6_ensemble
DATA_DIR=data/training_dataset_v2
REAL_DATA_DIR=data/tier2_train_v2_10c
VAL_DATA_DIR=data/tier2_val_v2

case "$MODE" in
  sanity)
    declare -a CELLS=("20260601:0.25" "20260601:0.50")
    ;;
  ensemble)
    if [ -z "$ENSEMBLE_RATIO" ]; then
      echo "Usage: $0 ensemble <ratio>" >&2
      exit 2
    fi
    declare -a CELLS=(
      "20260601:${ENSEMBLE_RATIO}"
      "20260602:${ENSEMBLE_RATIO}"
      "20260603:${ENSEMBLE_RATIO}"
      "20260604:${ENSEMBLE_RATIO}"
      "20260605:${ENSEMBLE_RATIO}"
    )
    ;;
  *)
    echo "Usage: $0 sanity | ensemble <ratio>" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTDIR/logs"

TOTAL_CELLS=${#CELLS[@]}
echo "=== Sprint 6 Cluster B launcher (mode=$MODE) ==="
echo "Cells: $TOTAL_CELLS"
echo "Output: $OUTDIR"
echo "Synthetic data: $DATA_DIR"
echo "Real data:      $REAL_DATA_DIR"
echo "Val data:       $VAL_DATA_DIR"
echo "Started: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

CELL_NUM=0
for cell in "${CELLS[@]}"; do
  CELL_NUM=$((CELL_NUM + 1))
  seed="${cell%%:*}"
  ratio="${cell##*:}"
  cell_name="unet_seed${seed}_ratio${ratio}"
  log_path="${OUTDIR}/logs/${cell_name}.log"
  best_pt="${OUTDIR}/${cell_name}/best.pt"

  echo "[$(date -u +'%H:%M:%SZ')] Cell ${CELL_NUM}/${TOTAL_CELLS}: ${cell_name}"

  if [ -f "$best_pt" ]; then
    echo "  -> skipped (best.pt exists)"
    continue
  fi

  if ! python scripts/train_ml_detector.py \
      --data-dir "$DATA_DIR" \
      --real-data-dir "$REAL_DATA_DIR" \
      --val-data-dir "$VAL_DATA_DIR" \
      --synthetic-ratio "$ratio" \
      --architecture unet \
      --unet-base-channels 64 \
      --epochs 25 \
      --batch-size 64 \
      --n-samples-per-epoch 20000 \
      --device cuda \
      --num-workers 16 \
      --pin-memory \
      --prefetch-factor 4 \
      --learning-rate 0.001 \
      --lr-schedule cosine \
      --seed "$seed" \
      --output-dir "$OUTDIR" \
      > "$log_path" 2>&1; then
    echo "  -> FAILED (see $log_path); continuing"
  else
    echo "  -> done"
  fi
done

echo
echo "=== B launcher complete at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
ls -1 "$OUTDIR"/*/best.pt 2>/dev/null | wc -l | xargs -I{} echo "Cells with best.pt: {} / $TOTAL_CELLS"
