#!/usr/bin/env bash
# Deploy NadirClaw patches from this fork to pip site-packages
# Usage: ./deploy-patches.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be copied."
fi

# Find site-packages directory
SITE_PACKAGES=$(python -c "import nadirclaw; import os; print(os.path.dirname(nadirclaw.__file__))")

if [[ -z "$SITE_PACKAGES" ]]; then
    echo "ERROR: nadirclaw not found. Install it first: pip install nadirclaw==0.14.3"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/nadirclaw"

# Files to patch (only the ones we modified)
PATCH_FILES=(
    "server.py"
    "routing.py"
    "web_dashboard.py"
    "compress.py"
    "quota.py"
    "settings.py"
)

echo "Target: $SITE_PACKAGES"
echo ""

for f in "${PATCH_FILES[@]}"; do
    SRC="$PATCHES_DIR/$f"
    DST="$SITE_PACKAGES/$f"

    if [[ ! -f "$SRC" ]]; then
        echo "SKIP: $f (not found in fork)"
        continue
    fi

    if [[ ! -f "$DST" ]]; then
        echo "SKIP: $f (not found in site-packages, install nadirclaw first)"
        continue
    fi

    if $DRY_RUN; then
        echo "WOULD COPY: $f"
    else
        cp "$SRC" "$DST"
        echo "PATCHED: $f"
    fi
done

echo ""
echo "Done! Restart NadirClaw to apply changes."
echo "  nadirclaw serve --verbose"
