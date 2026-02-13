# Single script to run entire Seq2Seq pipeline (PowerShell)
# Usage: .\run_all.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Seq2Seq Code Generation - Full Pipeline" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Step 1: Install dependencies
Write-Host ""
Write-Host "[1/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Step 2: Train all models
Write-Host ""
Write-Host "[2/4] Training all models (this will take ~10-15 minutes)..." -ForegroundColor Yellow
python train.py --config config_quick.yaml --model all

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Training failed" -ForegroundColor Red
    exit 1
}

# Step 3: Evaluate models
Write-Host ""
Write-Host "[3/4] Evaluating models..." -ForegroundColor Yellow
python evaluate.py --model all --split test

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Evaluation failed" -ForegroundColor Red
    exit 1
}

# Step 4: Visualize attention
Write-Host ""
Write-Host "[4/4] Generating attention visualizations..." -ForegroundColor Yellow
python visualize_attention.py --num_examples 5 --summary

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Visualization failed" -ForegroundColor Red
    exit 1
}

# Done
Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "✅ Pipeline completed successfully!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check your results:" -ForegroundColor Cyan
Write-Host "  - Trained models: checkpoints/"
Write-Host "  - Evaluation results: results/"
Write-Host "  - Visualizations: visualizations/attention/"
Write-Host ""
