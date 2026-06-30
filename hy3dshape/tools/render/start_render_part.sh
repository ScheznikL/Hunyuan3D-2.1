#!/usr/bin/env bash
set -e

############################
# CONFIG
############################
UUID_LOG="rendered_uuids_secondtry.txt"

SIZE_LIMIT_GB=650
MIN_MODELS_BEFORE_CHECK=1000

DATASET_STORE="/dcs/large/u5745134/dataset"
OBJ_MODELS_DIR="$DATASET_STORE/raw/3dmodels"
OUTPUT_ROOT="$DATASET_STORE/preprocessed"

BLENDER_BIN="$(which blender)"

# SHARDING (for multi-GPU)
SHARD_ID=${SHARD_ID:-0}        # 0,1,2,...
NUM_SHARDS=${NUM_SHARDS:-1}    # total workers

VIEWS=12
RESOLUTION=512

############################
# COLLECT & SHARD OBJECTS
############################
TMP_OBJ_LIST=$(mktemp)

find "$OBJ_MODELS_DIR" -type f -name "*.obj" \
  | sort \
  | awk "NR % $NUM_SHARDS == $SHARD_ID" \
  > "$TMP_OBJ_LIST"

echo "[INFO] Shard $SHARD_ID / $NUM_SHARDS"
echo "[INFO] Objects to process: $(wc -l < "$TMP_OBJ_LIST")"

############################
# RUN BLENDER ONCE
############################
$BLENDER_BIN -b -P partially_render.py -- \
    --file_list "$TMP_OBJ_LIST" \
    --output_root "$OUTPUT_ROOT" \
    --uuid_log "${UUID_LOG%.txt}_shard${SHARD_ID}.txt" \
    --size_limit "$SIZE_LIMIT_GB" \
    --min_models "$MIN_MODELS_BEFORE_CHECK" \
    --views "$VIEWS" \
    --resolution "$RESOLUTION" \
    --geo_mode

rm "$TMP_OBJ_LIST"
