#!/usr/bin/env bash

# --- НАЛАШТУВАННЯ ШЛЯХІВ ---
LOG_FILE="non_eng_log.log" 
SOURCE_DIR="/dcs/large/u5745134/dataset/preprocessed"
DEST_DIR="/dcs/large/u5745134/dataset/non_eng_lang"

# Створюємо цільову папку, якщо її ще немає
mkdir -p "$DEST_DIR"

echo ">>> Scanning log for non-ENG UIDs..."

# 1. Витягуємо UID з рядків, що містять [WARN] та "wasn't identified"
# Використовуємо grep з регулярним виразом для пошуку ID після слова 'with'
uids=$(grep "\[WARN\] obj with" "$LOG_FILE" | awk '{print $4}')

count=0
for id in $uids; do
    src_path="$SOURCE_DIR/$id"
    
    if [ -d "$src_path" ]; then
        echo " [MOVE] Moving folder for $id to $DEST_DIR"
        mv "$src_path" "$DEST_DIR/"
        count=$((count + 1))
    else
        echo " [SKIP] Folder for $id not found in source directory."
    fi
done

echo "---------------------------------------"
echo ">>> Done! Moved $count folders to $DEST_DIR."