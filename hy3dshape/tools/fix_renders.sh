#!/bin/bash
set -e

SOURCE_ROOT="/dcs/large/u5745134/dataset/preprocessed"
SOURCE_ARCHIVE="/dcs/large/u5745134/dataset/renders19.zip" 

echo "[INFO] Starting update process (Clean & Replace)..."

find "$SOURCE_ROOT" -type d -name "render_cond" | while read -r RENDER_DIR; do
    
    UUID=$(basename "$(dirname "$RENDER_DIR")")

    # 1. DELETE OLD PNGs (Safety Check: only delete if they look like the old format)
    # This removes 000.png, 001.png, etc.
    rm -f "$RENDER_DIR"/*.png

    # 2. EXTRACT NEW PNGs and JSONs
    # This puts 00000.png, 00001.png, and new.json into the folder
    unzip -o -j -q "$SOURCE_ARCHIVE" "*$UUID/*.png" "*$UUID/*.json" -d "$RENDER_DIR" || echo "[WARN] No match for $UUID"

done

echo "[SUCCESS] Folders cleaned and updated."

# find "$ROOT_DIR" -maxdepth 1 -mindepth 1 -type d | while read -r MODEL_OUT_DIR; do

#     # Extract UUID (Filename without extension)
#     UUID=$(basename "$MODEL_OUT_DIR")

#     echo "------------------------------------------------"
#     echo "Processing: $UUID"

#     # Define Output Directories
#     # GEO_DIR="$MODEL_OUT_DIR/geo_data"
#     RENDER_DIR="$MODEL_OUT_DIR/render_cond"
    

#     # --- A. there is 20 renders in each plus ${UUID}/transforms_train.json
#     NEW_RENDERS_DIR="$TEMP_RENDERS_ZIP/${UUID}/" 
#     find "$NEW_RENDERS_DIR" -maxdepth 1 -mindepth 1 -type d | while read -r RENDER; do
        
#     if [ -f "$RENDER" ]; then
#         cp "$RENDER" "$RENDER_DIR/"
#         echo "   [OK] Prompt copied from temp."
#     else
#         echo "   [WARN] obj with $UUID wasn't identified with ENG prompt!" >> non_eng_log.log        # Check if prompt already exists
#         if [ -f "$PROMPT_DIR/${UUID}_prompt" ]; then
#             echo "   [OK] Prompt already exists, keeping it."
#         else
#             echo "   [WARN] No prompt found anywhere for $UUID"
#         fi
#     fi
# done