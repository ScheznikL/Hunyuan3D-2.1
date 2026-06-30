#!/bin/bash
set -e

# ================= CONFIGURATION =================
# Root folder where all your {uuid} folders live
# Example: /dcs/large/u5745134/dataset/preprocessed
SOURCE_ROOT="/dcs/large/u5745134/dataset/preprocessed"

# The final destination file
FINAL_ARCHIVE="$SOURCE_ROOT/render_conds.zip"

# Temporary directory to build the structure before final zipping
TEMP_STAGING_DIR="$SOURCE_ROOT/temp_zip_staging"

# ================= EXECUTION =================

# 1. Setup Staging Area
echo "[INFO] Creating staging directory..."
mkdir -p "$TEMP_STAGING_DIR"

# 2. Iterate through all 'render_cond' folders
echo "[INFO] Searching for render_cond folders..."

# Find folders named 'render_cond' and process them
find "$SOURCE_ROOT" -type d -name "render_cond" | while read -r RENDER_DIR; do
    
    # Extract UUID (Parent directory name)
    # RENDER_DIR example: .../preprocessed/B08G217VYH/render_cond
    UUID_DIR=$(dirname "$RENDER_DIR")
    UUID=$(basename "$UUID_DIR")

    echo "Processing: $UUID"

    # 3. Create individual UUID zip containing ONLY PNGs
    # -j (junk-paths) ensures PNGs are at the root of the zip, not nested in folders
    # We zip directly into the staging directory with name {uuid}.zip
    zip -j -q "$TEMP_STAGING_DIR/$UUID.zip" "$RENDER_DIR"/*.png

done

echo "[INFO] Individual zipping complete. Creating final archive..."

# 4. Create the final 'render_conds.zip' from the staging area
# -j again to make sure the internal zips are flat inside the main zip
cd "$TEMP_STAGING_DIR"
zip -0 -q "$FINAL_ARCHIVE" *.zip

# 5. Cleanup
echo "[INFO] Cleaning up staging directory..."
cd "$SOURCE_ROOT"
rm -rf "$TEMP_STAGING_DIR"

echo "------------------------------------------------"
echo "[SUCCESS] Archive created at: $FINAL_ARCHIVE"
echo "Structure check:"
unzip -l "$FINAL_ARCHIVE" | head -n 10
echo "..."
echo "------------------------------------------------"