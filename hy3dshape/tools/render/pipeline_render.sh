#!/usr/bin/env bash
set -e

UUID_LOG="rendered_uuids.txt"
> "$UUID_LOG"

MODEL_COUNT=0
SIZE_LIMIT_GB=650

DATASET_STORE="/dcs/large/u5745134/dataset"

OBJ_MODELS_DIR="$DATASET_STORE/raw/3dmodels"
OUTPUT_ROOT="$DATASET_STORE/preprocessed"

BLENDER_BIN="$(which blender)"

render_obj () {
    INPUT_FILE="$1"

    FILENAME=$(basename -- "$INPUT_FILE")
    UUID="${FILENAME%.*}"

    RENDER_DIR="$OUTPUT_ROOT/$UUID/render_cond"

    if [ ! -d "$RENDER_DIR" ]; then
        echo "ERROR: Render directory does not exist: $RENDER_DIR" >&2
        exit 1
    fi

    echo "[RENDER] $UUID"

    blender -b -P hy3dshape/tools/render/render.py -- \
        --object "$INPUT_FILE" \
        --output_folder "$RENDER_DIR" \
        --geo_mode \
        --resolution 512 \
        --views 12
}

# ---- MAIN LOOP (no subshell!) ----
while IFS= read -r -d '' obj; do
    MODEL_COUNT=$((MODEL_COUNT + 1))

    FILENAME=$(basename -- "$obj")
    UUID="${FILENAME%.*}"

    render_obj "$obj"

    # If render succeeded
    echo "$UUID" >> "$UUID_LOG"

    # ---- Disk usage check after 1000 models ----
    if [ "$MODEL_COUNT" -ge 700 ]; then
        DATASET_SIZE_GB=$(du -hd1 "$DATASET_STORE" | grep -E '^[0-9]+G' | awk '{print $1}' | sed 's/G//')

        if [ -n "$DATASET_SIZE_GB" ] && [ "$DATASET_SIZE_GB" -ge "$SIZE_LIMIT_GB" ]; then
            echo "ERROR: Dataset size ${DATASET_SIZE_GB}GB ≥ limit ${SIZE_LIMIT_GB}GB" >&2
            echo "Stopping. Rendered UUIDs saved to $UUID_LOG"
            exit 1
        fi
    fi

done < <(find "$OBJ_MODELS_DIR" -type f -name "*.obj" -print0)
