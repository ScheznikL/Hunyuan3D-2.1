#!/usr/bin/env bash
set -e

UUID_LOG="rendered_uuids.txt"
MODEL_COUNT=0
SIZE_LIMIT_GB=650
MIN_MODELS_BEFORE_CHECK=1000

DATASET_STORE="/dcs/large/u5745134/dataset"
OBJ_MODELS_DIR="$DATASET_STORE/raw/3dmodels"
OUTPUT_ROOT="$DATASET_STORE/preprocessed"

# Blender binary
BLENDER_BIN="$(which blender)"

# Collect all OBJ paths
OBJ_LIST=$(find "$OBJ_MODELS_DIR" -type f -name "*.obj")

# Write paths to temp file
TMP_OBJ_LIST=$(mktemp)
printf "%s\n" $OBJ_LIST > "$TMP_OBJ_LIST"

# Launch Blender once for the batch
$BLENDER_BIN -b -P batch_render_blender.py -- \
    --file_list "$TMP_OBJ_LIST" \
    --output_root "$OUTPUT_ROOT" \
    --uuid_log "$UUID_LOG" \
    --size_limit "$SIZE_LIMIT_GB" \
    --min_models "$MIN_MODELS_BEFORE_CHECK" \
    --geo_mode --resolution 512 --view 12

rm "$TMP_OBJ_LIST"
