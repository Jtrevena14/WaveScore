#!/usr/bin/env bash
# Full training workflow on the instance, run from inside tmux.
#   bash scripts/aws/run_training.sh [s3://my-bucket]
# LIMIT=3 bash scripts/aws/run_training.sh    quick smoke test (3 files per piano and subset)
# Each step logs to logs/<step>.log; the script stops at the first failure.
set -eo pipefail

cd "$(dirname "$0")/../.."
source scripts/aws/activate_env.sh

S3_BUCKET=${1:-}
export SHEETCREATOR_MAPS=${SHEETCREATOR_MAPS:-/data/MAPS}
SCHEME=maps-full
WORKERS=${WORKERS:-$(nproc)}
limit=()
if [ -n "${LIMIT:-}" ]; then
    limit=(--limit "$LIMIT")
fi
mkdir -p logs

step() {
    local name=$1
    shift
    echo "=== $name: $*"
    "$@" 2>&1 | tee "logs/$name.log"
}

# 1. One ByteDance pass over every file (fills data/cache/), plus the baseline scores.
step 1_transcribe python -m eval.evaluate --backend piano --filter none default --split all --scheme $SCHEME "${limit[@]}"
step 2_tune python -m eval.tune_filters --backend piano --scheme $SCHEME ${LIMIT:+--limit $LIMIT}
step 3_classifier python -m eval.train_classifier --backend piano --scheme $SCHEME "${limit[@]}"
step 4_cnn python -m eval.train_cnn --backend piano --scheme $SCHEME --workers "$WORKERS" "${limit[@]}"
step 5_evaluate python -m eval.evaluate --backend piano --filter none default tuned learned cnn --split test --scheme $SCHEME "${limit[@]}"

if [ -n "$S3_BUCKET" ]; then
    aws s3 sync data/cache "$S3_BUCKET/cache" --only-show-errors
    aws s3 sync models "$S3_BUCKET/models" --exclude piano_note_pedal.pth --only-show-errors
    aws s3 sync logs "$S3_BUCKET/logs" --only-show-errors
fi
echo "Done. Models:"
ls -la models
