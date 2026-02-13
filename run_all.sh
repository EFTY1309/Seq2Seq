#!/bin/bash

# Single script to run entire Seq2Seq pipeline
# Usage: ./run_all.sh

set -e  # Exit on any error

echo "======================================"
echo "Seq2Seq Code Generation - Full Pipeline"
echo "======================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.9+"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# Step 2: Train all models
echo ""
echo "[2/4] Training all models (this will take ~10-15 minutes)..."
python train.py --config config_quick.yaml --model all

# Step 3: Evaluate models
echo ""
echo "[3/4] Evaluating models..."
python evaluate.py --model all --split test

# Step 4: Visualize attention
echo ""
echo "[4/4] Generating attention visualizations..."
python visualize_attention.py --num_examples 5 --summary

# Done
echo ""
echo "======================================"
echo "✅ Pipeline completed successfully!"
echo "======================================"
echo ""
echo "Check your results:"
echo "  - Trained models: checkpoints/"
echo "  - Evaluation results: results/"
echo "  - Visualizations: visualizations/attention/"
echo ""
