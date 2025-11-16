#!/usr/bin/env bash
# Exit on error
set -o errexit

# Set a specific, stable Python version
export PYTHON_VERSION=3.11.5

# Upgrade pip
pip install --upgrade pip

# Forcefully re-install all dependencies from requirements.txt, ignoring any cached versions
echo "Installing dependencies with a clean cache..."
pip install --no-cache-dir --force-reinstall -r requirements.txt

# Run your data pipeline to build the FAISS index
echo "Building Knowledge Base..."
python build.py

echo "Build finished successfully."
