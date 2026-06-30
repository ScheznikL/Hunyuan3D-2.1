#!/usr/bin/env bash
set -e

# ================= CONFIGURATION =================
# Adjust to match your setup
export OUTPUT_ROOT="${OUTPUT_ROOT:-/dcs/large/u5745134/dataset/preprocessed}"
export PYTHON="${PYTHON:-python}"

# ================= SCRIPT =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "  PROMPT LANGUAGE CHECKER"
echo "=============================================="
echo ""

# Check if dataset exists
if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "ERROR: Dataset directory not found: $OUTPUT_ROOT"
    exit 1
fi

# Mode selection
MODE="${1:-scan}"

case "$MODE" in
    scan|check)
        echo ">>> Scanning all prompts for language issues..."
        $PYTHON "$SCRIPT_DIR/check_and_fix_prompts.py" \
            --dataset "$OUTPUT_ROOT" \
            --create-plan "$OUTPUT_ROOT/translation_plan.json"
        ;;
    
    translate)
        echo ">>> Automatically translating non-English prompts..."
        
        # Check for API key
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo ""
            echo "WARNING: ANTHROPIC_API_KEY not set!"
            echo "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            echo ""
            echo "Running in DRY-RUN mode (no actual changes)..."
            echo ""
            
            $PYTHON "$SCRIPT_DIR/auto_translate_prompts.py" \
                --dataset "$OUTPUT_ROOT" \
                --dry-run
        else
            echo "API key found. Running translation..."
            echo "Add --dry-run flag to this script to preview changes first."
            echo ""
            
            $PYTHON "$SCRIPT_DIR/auto_translate_prompts.py" \
                --dataset "$OUTPUT_ROOT"
        fi
        ;;
    
    apply)
        if [ ! -f "$OUTPUT_ROOT/translation_plan.json" ]; then
            echo "ERROR: translation_plan.json not found!"
            echo "Run '$0 scan' first to create it."
            exit 1
        fi
        
        echo ">>> Applying translations from translation_plan.json..."
        echo ""
        echo "IMPORTANT: Make sure you've edited translation_plan.json"
        echo "and filled in the 'translated_prompt' fields!"
        echo ""
        read -p "Continue? (y/N) " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $PYTHON "$SCRIPT_DIR/check_and_fix_prompts.py" \
                --apply "$OUTPUT_ROOT/translation_plan.json"
        else
            echo "Cancelled."
        fi
        ;;
    
    *)
        echo "Usage: $0 {scan|translate|apply}"
        echo ""
        echo "Commands:"
        echo "  scan      - Scan prompts and create translation plan"
        echo "  translate - Auto-translate using Claude API (requires ANTHROPIC_API_KEY)"
        echo "  apply     - Apply manual translations from translation_plan.json"
        echo ""
        echo "Environment variables:"
        echo "  OUTPUT_ROOT         - Path to preprocessed dataset (default: /dcs/large/u5745134/dataset/preprocessed)"
        echo "  ANTHROPIC_API_KEY   - API key for automatic translation"
        echo "  PYTHON              - Python interpreter (default: python)"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "Done!"