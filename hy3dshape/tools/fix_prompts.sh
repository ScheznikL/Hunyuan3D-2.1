export TEMP_PROMPT_DIR="/dcs/large/u5745134/dataset/temp_prompts_staging"
export ROOT_DIR="/dcs/large/u5745134/dataset/preprocessed/"

find "$ROOT_DIR" -maxdepth 1 -mindepth 1 -type d | while read -r MODEL_OUT_DIR; do

    # Extract UUID (Filename without extension)
    UUID=$(basename "$MODEL_OUT_DIR")

    echo "------------------------------------------------"
    echo "Processing: $UUID"

    # Define Output Directories
    # GEO_DIR="$MODEL_OUT_DIR/geo_data"
    # RENDER_DIR="$MODEL_OUT_DIR/render_cond"
    PROMPT_DIR="$MODEL_OUT_DIR/prompt"

    # --- A. LINK PROMPT ---
    GENERATED_PROMPT="$TEMP_PROMPT_DIR/${UUID}_prompt"

    if [ -f "$GENERATED_PROMPT" ]; then
        cp "$GENERATED_PROMPT" "$PROMPT_DIR/${UUID}_prompt"
        echo "   [OK] Prompt copied from temp."
    else
        echo "   [WARN] obj with $UUID wasn't identified with ENG prompt!" >> non_eng_log.log        # Check if prompt already exists
        if [ -f "$PROMPT_DIR/${UUID}_prompt" ]; then
            echo "   [OK] Prompt already exists, keeping it."
        else
            echo "   [WARN] No prompt found anywhere for $UUID"
        fi
    fi
done