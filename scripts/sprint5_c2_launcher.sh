#!/bin/bash
# Sprint 5 Cluster C2 ratio-sweep launcher.
# 7 ratios x 3 seeds = 21 U-Net cells. Each ~50 min on A100 at base_channels=64.
# Sequential execution; per-cell logs under <out_dir>/logs/<cell>.log.
# Resume-skip: cells with tier1_metrics.json already present are skipped, so
# this is safe to re-run after interrupts.
set -uo pipefail

SEEDS=(20260512 20260513 20260514)
RATIOS=(0.00 0.25 0.38 0.50 0.75 0.90 1.00)
OUTDIR=artifacts/sprint5_ratio_sweep

mkdir -p "$OUTDIR/logs"

TOTAL_CELLS=$(( ${#SEEDS[@]} * ${#RATIOS[@]} ))
echo "=== Sprint 5 Cluster C2 sweep ==="
echo "Cells: $TOTAL_CELLS (${#SEEDS[@]} seeds x ${#RATIOS[@]} ratios)"
echo "Output: $OUTDIR"
echo "Started: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

CELL_NUM=0
for seed in "${SEEDS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    CELL_NUM=$((CELL_NUM + 1))
    cell_name="unet_seed${seed}_ratio${ratio}"
    log_path="${OUTDIR}/logs/${cell_name}.log"
    metrics_path="${OUTDIR}/${cell_name}/tier1_metrics.json"

    echo "[$(date -u +'%H:%M:%SZ')] Cell ${CELL_NUM}/${TOTAL_CELLS}: ${cell_name}"

    if [ -f "$metrics_path" ]; then
      echo "  -> skipped (tier1_metrics.json already exists)"
      continue
    fi

    if ! python scripts/train_ml_detector.py \
        --data-dir data/training_dataset_v2 \
        --real-data-dir data/tier2_train_v2 \
        --val-data-dir data/tier2_val_v2 \
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
        --seed "$seed" \
        --output-dir "$OUTDIR" \
        > "$log_path" 2>&1; then
      echo "  -> FAILED (see $log_path); continuing"
    else
      echo "  -> done"
    fi
  done
done

echo
echo "=== Sweep complete at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
ls -1 "$OUTDIR"/*/tier1_metrics.json 2>/dev/null | wc -l | xargs -I{} echo "Cells with tier1_metrics.json: {} / $TOTAL_CELLS"