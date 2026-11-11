#!/bin/bash

# Navigate to the script's directory (PhasePix project root)
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# One-click auto-update: fast-forward this folder (code, SOPs and Methodology
# files) when it is a Git clone and the machine is online. Every failure —
# offline, local commits, Git not installed — is ignored, so startup never blocks.
# ---------------------------------------------------------------------------
if [ -d ".git" ] && command -v git >/dev/null 2>&1; then
    echo "Checking for PhasePix updates (git pull --ff-only origin main)..."
    git pull --ff-only origin main 2>/dev/null || echo "  Update check skipped (offline or local changes)."
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
