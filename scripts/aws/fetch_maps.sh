#!/usr/bin/env bash
# Download MAPS one zip at a time, extract it without ISOL, back it up to S3.
#   bash fetch_maps.sh urls.txt s3://my-bucket/MAPS
# urls.txt: one direct zip URL per line (blank lines and # comments ignored).
# Optional: MAPS_USER / MAPS_PASSWORD for sources that need a login.
# Re-running skips zips already extracted, and wget -c resumes partial downloads.
set -euo pipefail

URLS=${1:?usage: fetch_maps.sh <urls.txt> <s3://bucket/MAPS>}
S3_DEST=${2:?usage: fetch_maps.sh <urls.txt> <s3://bucket/MAPS>}
MAPS_DIR=${MAPS_DIR:-/data/MAPS}

# Zips go on the instance's local NVMe when it is mounted, to keep the EBS volume free.
if mountpoint -q /opt/dlami/nvme 2>/dev/null; then
    ZIP_DIR=${ZIP_DIR:-/opt/dlami/nvme/zips}
else
    ZIP_DIR=${ZIP_DIR:-/data/zips}
fi
mkdir -p "$ZIP_DIR" "$MAPS_DIR" "$MAPS_DIR/.done"

auth=()
if [ -n "${MAPS_USER:-}" ]; then
    auth=(--user "$MAPS_USER" --password "${MAPS_PASSWORD:-}")
fi

while IFS= read -r url || [ -n "$url" ]; do
    url=${url%%#*}
    url=$(echo "$url" | xargs)
    [ -z "$url" ] && continue
    name=$(basename "${url%%\?*}")
    if [ -e "$MAPS_DIR/.done/$name" ]; then
        echo "skip $name (already extracted)"
        continue
    fi
    echo "== $name"
    wget -c "${auth[@]}" -O "$ZIP_DIR/$name" "$url"
    unzip -n -q "$ZIP_DIR/$name" -x '*/ISOL/*' -d "$MAPS_DIR"
    # Some MAPS piano zips contain an inner zip per piano; extract those the same way.
    find "$MAPS_DIR" -name '*.zip' -print0 | while IFS= read -r -d '' inner; do
        unzip -n -q "$inner" -x '*/ISOL/*' -d "$(dirname "$inner")" && rm -f "$inner"
    done
    rm -f "$ZIP_DIR/$name"
    touch "$MAPS_DIR/.done/$name"
    aws s3 sync "$MAPS_DIR" "$S3_DEST" --only-show-errors
    df -h "$MAPS_DIR" | tail -1
done < "$URLS"

echo "MAPS in $MAPS_DIR: $(find "$MAPS_DIR" -name '*.wav' | wc -l) wav files"
