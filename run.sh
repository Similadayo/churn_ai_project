#!/usr/bin/env bash
# ============================================================
#  Customer Churn Prediction - One-Click Launcher (Mac/Linux)
#  Usage:  double-click (after making executable) or run:
#            bash run.sh
#  It will create a private environment, install everything,
#  and open the dashboard in your browser.
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo "============================================================"
echo "  Starting the Customer Churn Prediction Dashboard"
echo "============================================================"
echo

# --- 1. Locate a Python interpreter ---
if command -v python3.11 >/dev/null 2>&1; then
    PY=python3.11
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ERROR] Python was not found."
    echo "Please install Python 3.11 from https://www.python.org/downloads/"
    exit 1
fi

# --- 2. Create the virtual environment (first run only) ---
if [ ! -x "venv/bin/python" ]; then
    echo "[setup] Creating a private Python environment (first run, please wait)..."
    "$PY" -m venv venv
fi

# --- 3. Install dependencies ---
echo "[setup] Installing required packages (first run can take a few minutes)..."
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/python -m pip install -r requirements.txt

# --- 4. Launch the dashboard ---
echo
echo "============================================================"
echo "  All set! Opening the dashboard in your browser..."
echo "  (Keep this window open while you use the app.)"
echo "  To stop: press Ctrl+C."
echo "============================================================"
echo
exec ./venv/bin/python -m streamlit run app/streamlit_app.py
