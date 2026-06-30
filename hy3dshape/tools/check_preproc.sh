OUTPUT_ROOT="/dcs/large/u5745134/dataset/preprocessed"

for UUID_DIR in "$OUTPUT_ROOT"/*; do
    [ -d "$UUID_DIR" ] || continue

    UUID_="$(basename "$UUID_DIR")"
    GEO_DIR="$UUID_DIR/geo_data"
    SURFACE_FILE="$GEO_DIR/${UUID_}_surface.npz"
    SDF_FILE="$GEO_DIR/${UUID_}_sdf.npz"

    # Only one check is needed: do the required files exist?
    if [[ -f "$SURFACE_FILE" && -f "$SDF_FILE" ]]; then
        echo "[OK] $UUID_ - Geometry files verified."
    else
        # If files are missing, the directory is a failure. 
        # This covers cases where the dir is empty OR just missing the npz files.
        echo "[DELETE] $UUID_ - Missing required .npz files. Purging..."
        # rm -rf "$UUID_DIR"
    fi
done