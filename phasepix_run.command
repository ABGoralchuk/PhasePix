#!/bin/bash

# Navigate to the script's directory (PhasePix project root)
cd "$(dirname "$0")"

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
