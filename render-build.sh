#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Install dependencies with a clean cache
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt

# 2. Run your data pipeline to build the FAISS index
# (This assumes you have your data/raw folder populated, 
# or you can use a post-deploy job for this)
echo "Building Knowledge Base..."
python 04_build_kb.py

echo "Build finished."
