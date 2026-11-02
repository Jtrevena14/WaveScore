#!/usr/bin/env bash
# Restore extracted MAPS from S3 onto a fresh instance or volume.
#   bash restore_maps.sh s3://my-bucket/MAPS
set -euo pipefail

S3_SRC=${1:?usage: restore_maps.sh <s3://bucket/MAPS>}
MAPS_DIR=${MAPS_DIR:-/data/MAPS}
mkdir -p "$MAPS_DIR"
aws s3 sync "$S3_SRC" "$MAPS_DIR" --only-show-errors
echo "MAPS in $MAPS_DIR: $(find "$MAPS_DIR" -name '*.wav' | wc -l) wav files"
