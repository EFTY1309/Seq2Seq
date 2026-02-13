# Seq2Seq Code Generation - Quick Start Script (PowerShell)

Write-Host "╔══════════════════════════════════════════════════════════════════════╗"
Write-Host "║         Seq2Seq Code Generation - Text to Python                     ║"
Write-Host "║                  RNN | LSTM | Attention                              ║"
Write-Host "╚══════════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Check if Docker is available
$dockerAvailable = $false
try {
    $null = docker --version 2>$null
    $null = docker-compose --version 2>$null
    $dockerAvailable = $true
    Write-Host "✅ Docker detected - will use containerized environment" -ForegroundColor Green
} catch {
    Write-Host "ℹ️  Docker not found - will run locally" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Choose an option:"
Write-Host "  1. Train all models"
Write-Host "  2. Train specific model (vanilla/lstm/attention)"
Write-Host "  3. Evaluate all models"
Write-Host "  4. Visualize attention"
Write-Host "  5. Run complete pipeline (train + evaluate + visualize)"
Write-Host ""

$choice = Read-Host "Enter choice [1-5]"

switch ($choice) {
    "1" {
        Write-Host "Training all models..." -ForegroundColor Cyan
        if ($dockerAvailable) {
            docker-compose up --build seq2seq
        } else {
            python train.py --model all
        }
    }
    "2" {
        $model = Read-Host "Enter model name (vanilla/lstm/attention)"
        Write-Host "Training $model model..." -ForegroundColor Cyan
        if ($dockerAvailable) {
            docker-compose run --rm seq2seq python train.py --model $model
        } else {
            python train.py --model $model
        }
    }
    "3" {
        Write-Host "Evaluating all models..." -ForegroundColor Cyan
        if ($dockerAvailable) {
            docker-compose run --rm evaluate
        } else {
            python evaluate.py --model all
        }
    }
    "4" {
        Write-Host "Visualizing attention..." -ForegroundColor Cyan
        if ($dockerAvailable) {
            docker-compose run --rm visualize
        } else {
            python visualize_attention.py --num_examples 10 --summary
        }
    }
    "5" {
        Write-Host "Running complete pipeline..." -ForegroundColor Cyan
        if ($dockerAvailable) {
            docker-compose up --build seq2seq
            docker-compose run --rm evaluate
            docker-compose run --rm visualize
        } else {
            python train.py --model all
            python evaluate.py --model all
            python visualize_attention.py --num_examples 10 --summary
        }
    }
    default {
        Write-Host "Invalid choice!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════════╗"
Write-Host "║                       Execution Completed                            ║"
Write-Host "╚══════════════════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "Results are available in:"
Write-Host "  📁 checkpoints/    - Trained model weights"
Write-Host "  📁 results/        - Evaluation metrics"
Write-Host "  📁 visualizations/ - Attention heatmaps"
Write-Host "  📁 logs/           - Training logs"
Write-Host ""
