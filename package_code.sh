#!/bin/bash
# Script to package only code files, excluding large binaries and data

OUTPUT_FILE="project1_code_only.tar.gz"
EXCLUDE_PATTERNS=(
  "venv"
  "*.pt"
  "*.pth"
  "*.mov"
  "*.gif"
  "*.png"
  "*.jpg"
  "*.jpeg"
  "pre-trained_models"
  "**/experiment*/model.pt"
  "**/experiment*/best_model.pt"
  "**/experiment*/checkpoints"
  "*.pdf"
  ".DS_Store"
  "__pycache__"
  "*.pyc"
  "Anaconda*.sh"
)

echo "Creating code-only archive..."
echo "This may take a minute..."

# Build exclude arguments
EXCLUDE_ARGS=""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude='$pattern'"
done

# Create archive with only code files
tar -czf "$OUTPUT_FILE" \
  --exclude='venv' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.mov' \
  --exclude='*.gif' \
  --exclude='pre-trained_models' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='Anaconda*.sh' \
  --exclude='*.pdf' \
  ml-equivariant-neural-rendering-main/ \
  *.py *.md *.txt *.json .gitignore 2>/dev/null

if [ -f "$OUTPUT_FILE" ]; then
  SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
  echo "✓ Created: $OUTPUT_FILE ($SIZE)"
  echo ""
  echo "You can now share this file via:"
  echo "  - Email (if < 25MB)"
  echo "  - Google Drive / Dropbox"
  echo "  - WeTransfer (wetransfer.com)"
  echo "  - GitHub (as a release)"
else
  echo "✗ Failed to create archive"
  exit 1
fi

