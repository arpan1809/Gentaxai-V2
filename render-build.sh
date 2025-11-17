#!/usr/bin/env bash
# Exit on any error
set -o errexit

echo "Using Python version: $PYTHON_VERSION"

# Upgrade pip safely
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies WITHOUT cache to avoid OOM from unpacked wheels
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Skipping FAISS index build (using prebuilt index from repo)."

echo "Build completed successfully."
