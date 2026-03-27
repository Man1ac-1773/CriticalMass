#!/bin/bash

# 1. Prep the arena
mkdir -p competitors
touch competitors/__init__.py

# Clear out any bots from previous runs to avoid ghost matches
rm -f competitors/*_bot.py

# 2. Define your combatants here
# You can use branch names, tags, or raw commit hashes
BRANCHES=(
    "master" # Usually your stable baseline
    "HEAD"  # Whatever you are currently working on
)

echo "--- Assembling Competitors ---"

for BRANCH in "${BRANCHES[@]}"; do
    # Swap slashes for dashes so the filenames are clean and importable
    SAFE_NAME=$(echo "$BRANCH" | tr '/' '-')
    OUT_FILE="competitors/${SAFE_NAME}_bot.py"
    
    # git show lets us read a file from another branch without checking it out
    if git show "$BRANCH:my_bot.py" > "$OUT_FILE" 2>/dev/null; then
        echo "Extracted: $BRANCH"
    else
        echo "Warning: Could not find my_bot.py in branch '$BRANCH'. Skipping."
        rm -f "$OUT_FILE" # Clean up the empty file if git show failed
    fi
done

echo -e "\n--- Let the games begin ---"
python3 tournament.py
