#!/bin/bash

# Navigate to the script's directory (PhasePix project root)
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# One-click auto-update: fast-forward this folder (code, SOPs and Methodology
# files) when it is a Git clone and the machine is online. PhasePix rewrites
# parameter_presets.json / light_calibration_presets.json whenever a lab recipe
# is saved, so they are parked for the fast-forward and restored afterwards —
# local preset edits can never block the update, and no recipe is ever lost.
# ---------------------------------------------------------------------------
if [ -d ".git" ] && command -v git >/dev/null 2>&1; then
    echo "Checking for PhasePix updates (git pull --ff-only origin main)..."
    PRESET_STASH="$(mktemp -d)"
    for preset in parameter_presets.json light_calibration_presets.json; do
        [ -f "$preset" ] && cp "$preset" "$PRESET_STASH/$preset"
    done
    if ! git pull --ff-only origin main 2>/dev/null; then
        git checkout -- parameter_presets.json light_calibration_presets.json 2>/dev/null || true
        git pull --ff-only origin main 2>/dev/null || echo "  Update check skipped (offline or local changes)."
    fi
    for preset in parameter_presets.json light_calibration_presets.json; do
        [ -f "$PRESET_STASH/$preset" ] && cp "$PRESET_STASH/$preset" "$preset"
    done
    rm -rf "$PRESET_STASH"
    echo ""
fi

echo "=================================================="
echo " Starting PhasePix (SEM Phase Analyzer)..."
echo "=================================================="

# Check for Python virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found in .venv or venv."
    echo "Attempting to run using system Streamlit..."
fi

# Run the Streamlit application
streamlit run app.py
