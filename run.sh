#!/bin/bash

# Seq2Seq Code Generation - Quick Start Script

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Seq2Seq Code Generation - Text to Python                     ║"
echo "║                  RNN | LSTM | Attention                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    USE_DOCKER=true
    echo "✅ Docker detected - will use containerized environment"
else
    USE_DOCKER=false
    echo "ℹ️  Docker not found - will run locally"
fi

echo ""
echo "Choose an option:"
echo "  1. Train all models"
echo "  2. Train specific model (vanilla/lstm/attention)"
echo "  3. Evaluate all models"
echo "  4. Visualize attention"
echo "  5. Run complete pipeline (train + evaluate + visualize)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Training all models..."
        if [ "$USE_DOCKER" = true ]; then
            docker-compose up --build seq2seq
        else
            python train.py --model all
        fi
        ;;
    2)
        read -p "Enter model name (vanilla/lstm/attention): " model
        echo "Training $model model..."
        if [ "$USE_DOCKER" = true ]; then
            docker-compose run --rm seq2seq python train.py --model $model
        else
            python train.py --model $model
        fi
        ;;
    3)
        echo "Evaluating all models..."
        if [ "$USE_DOCKER" = true ]; then
            docker-compose run --rm evaluate
        else
            python evaluate.py --model all
        fi
        ;;
    4)
        echo "Visualizing attention..."
        if [ "$USE_DOCKER" = true ]; then
            docker-compose run --rm visualize
        else
            python visualize_attention.py --num_examples 10 --summary
        fi
        ;;
    5)
        echo "Running complete pipeline..."
        if [ "$USE_DOCKER" = true ]; then
            docker-compose up --build seq2seq
            docker-compose run --rm evaluate
            docker-compose run --rm visualize
        else
            python train.py --model all
            python evaluate.py --model all
            python visualize_attention.py --num_examples 10 --summary
        fi
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                       Execution Completed                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results are available in:"
echo "  📁 checkpoints/    - Trained model weights"
echo "  📁 results/        - Evaluation metrics"
echo "  📁 visualizations/ - Attention heatmaps"
echo "  📁 logs/           - Training logs"
echo ""
