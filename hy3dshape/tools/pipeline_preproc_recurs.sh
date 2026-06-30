#!/usr/bin/env bash
set -e


# ================= CONFIGURATION =================
# 1. Project & Dataset Locations
# Adjust these two paths to match your actual folders
export PROJECT_ROOT="/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1"
export DATASET_STORE="/dcs/large/u5745134/dataset" 

# 2. Output Locations
export OUTPUT_ROOT="/dcs/large/u5745134/dataset/preprocessed"
export TEMP_PROMPT_DIR="/dcs/large/u5745134/dataset/temp_prompts_staging"

# 3. Tools
export BLENDER_PATH="/d/Blender_Foundation/blender.exe"
export OPENCV_IO_ENABLE_OPENEXR=1
# ================= the function =================
process_obj() {
    INPUT_FILE="$1"
    

    FILENAME=$(basename -- "$INPUT_FILE")
    UUID="${FILENAME%.*}"

    MODEL_OUT_DIR="$OUTPUT_ROOT/$UUID"
    GEO_DIR="$MODEL_OUT_DIR/geo_data"
    RENDER_DIR="$MODEL_OUT_DIR/render_cond"
    PROMPT_DIR="$MODEL_OUT_DIR/prompt"
    
    SURFACE_FILE="$GEO_DIR/${UUID}_surface.npz"
    SDF_FILE="$GEO_DIR/${UUID}_sdf.npz"

    # -------- SKIP CHECK --------
    if [[ -f "$SURFACE_FILE" && -f "$SDF_FILE" ]]; then
        echo "[SKIP] $UUID already processed"
        return 0
    fi

    echo "------------------------------------------------"
    echo "Processing: $UUID"   

    mkdir -p "$GEO_DIR" "$RENDER_DIR" "$PROMPT_DIR"

    GENERATED_PROMPT="$TEMP_PROMPT_DIR/${UUID}_prompt"

    if [ -f "$GENERATED_PROMPT" ]; then
        cp "$GENERATED_PROMPT" "$PROMPT_DIR/${UUID}_prompt"
    else
        echo "{ \"UUID\": \"$UUID\", \"error\": \"No metadata found\", \"item_dimensions\": {} }" \
            > "$PROMPT_DIR/${UUID}_prompt"
    fi

    python hy3dshape/tools/watertight/watertight_and_sample.py \
        --input_obj "$INPUT_FILE" \
        --output_prefix "$GEO_DIR/$UUID"
}

export -f process_obj

# ================= STEP 1: PREPARATION =================
echo ">>> [Step 1] Preparing Metadata and Prompts..."
echo "DONE ... "
# Run the Python script to extract data and generate prompt JSONs
# python prepare_and_generate.py \
#     --dataset_root "$DATASET_STORE" \
#     --prompt_output "$TEMP_PROMPT_DIR"

# echo ">>> [Step 1.1] Converting all models to triangle OBJ..."

 #SOURCE_MODELS_DIR="${DATASET_STORE}/3dmodels/original"
#  OBJ_MODELS_DIR="${DATASET_STORE}/raw/3dmodels"

# mkdir -p "$OBJ_MODELS_DIR"

# find "$SOURCE_MODELS_DIR" -type f \( \
#     -name "*.glb" -o \
#     -name "*.gltf" -o \
#     -name "*.fbx" -o \
#     -name "*.obj" \
# \) | while read INPUT_FILE; do

#     FILENAME=$(basename -- "$INPUT_FILE")
#     UUID="${FILENAME%.*}"
#     OUTPUT_OBJ="$OBJ_MODELS_DIR/${UUID}.obj"

#     if [ -f "$OUTPUT_OBJ" ]; then
#         echo "   [SKIP] Already converted: $UUID"
#         continue
#     fi

#     echo "   [CONVERT] $UUID → OBJ"

#     "$BLENDER_PATH" -b --python convert_to_obj.py -- \
#         "$INPUT_FILE" \
#         "$OUTPUT_OBJ"

# done

echo ">>> Conversion Complete."

# ================= STEP 2: PROCESSING =================
echo ">>> [Step 2] Processing 3D Models..."

# ABO specific path: models are in 3dmodels/original/ inside the dataset store
#SOURCE_MODELS_DIR="${DATASET_STORE}/3dmodels/original"

# if [ ! -d "$OBJ_MODELS_DIR" ]; then
#     echo "CRITICAL ERROR: Model directory not found at:"
#     echo "$OBJ_MODELS_DIR"
#     echo "Ensure the tar file extracted correctly."
#     exit 1
# fi


#---------------using function ---------------------

# find "$OBJ_MODELS_DIR" -type f -name "*.obj" -print0 \
#   | parallel -0 -j 12 --halt soon,fail=1 process_obj {}

#WAS -> find "$OBJ_MODELS_DIR" -type f -name "*.obj" -print0 | parallel -0 -j (nproc) --halt soon,fail=1 process_obj {}

# Find all GLB/OBJ files in the source directory
# using 'find' to handle subdirectories if they exist 

##find "$SOURCE_MODELS_DIR" -type f \( -name "*.glb" -o -name "*.obj" \) | while read INPUT_FILE; do
    
# find "$OBJ_MODELS_DIR" -type f \( -name "*.obj" \) | while read INPUT_FILE; do

#     # Extract UUID (Filename without extension)
#     FILENAME=$(basename -- "$INPUT_FILE")
#     UUID="${FILENAME%.*}"

#     echo "------------------------------------------------"
#     echo "Processing: $UUID"

#     # Define Output Directories
#     MODEL_OUT_DIR="$OUTPUT_ROOT/$UUID"
#     GEO_DIR="$MODEL_OUT_DIR/geo_data"
#     RENDER_DIR="$MODEL_OUT_DIR/render_cond"
#     PROMPT_DIR="$MODEL_OUT_DIR/prompt"

#     # Create Structure
#     mkdir -p "$GEO_DIR" "$RENDER_DIR" "$PROMPT_DIR"

    # --- A. LINK PROMPT ---
    # Copy the generated prompt file to the final folder
    GENERATED_PROMPT="$TEMP_PROMPT_DIR/${UUID}_prompt"
    
    if [ -f "$GENERATED_PROMPT" ]; then
        cp "$GENERATED_PROMPT" "$PROMPT_DIR/${UUID}_prompt"
        echo "   [OK] Prompt linked."
    else
        rm "$PROMPT_DIR/${UUID}_prompt"
        echo "   [WARN] No metadata found. creating placeholder."
        # echo "{ \"UUID\": \"$UUID\", \"error\": \"No metadata found\", \"item_dimensions\": {} }" > "$PROMPT_DIR/${UUID}_prompt"
    fi

#     # --- B. EXECUTE WATERTIGHT SCRIPT ---
#     # The watertight script is assumed to handle the geometry processing
#     # Adjust path to watertight_and_sample.py if it's not in current dir
#     echo "   [EXEC] Running Watertight Processing..."
    
#     python watertight/watertight_and_sample.py \
        # --input_obj "$INPUT_FILE" \
        # --output_prefix "$GEO_DIR/$UUID"

    #--- C. RENDER (Optional) ---
    # echo "   [EXEC] Rendering..."
    # "$BLENDER_PATH" -b -P render/render.py -- \
    #    --object "$INPUT_FILE" \
    #    --output_folder "$RENDER_DIR" \
    #    --geo_mode --resolution 512

#   done

echo ">>> Pipeline Complete."