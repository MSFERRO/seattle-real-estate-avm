# PowerShell execution helper for Windows environments
param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("install", "train", "predict", "test", "api", "docker-build", "docker-run", "docker-stop")]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"

switch ($Task) {
    "install" {
        Write-Host "Installing requirements..." -ForegroundColor Cyan
        pip install -r requirements.txt
    }
    "train" {
        Write-Host "Executing model training and 5-fold CV benchmark..." -ForegroundColor Cyan
        python -m src.models.train
    }
    "predict" {
        Write-Host "Running batch inference on future_unseen_examples.csv..." -ForegroundColor Cyan
        python -m src.models.predict
    }
    "test" {
        Write-Host "Running automated tests with pytest..." -ForegroundColor Cyan
        pytest -v tests/
    }
    "api" {
        Write-Host "Starting FastAPI server on http://localhost:8000..." -ForegroundColor Cyan
        uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
    }
    "docker-build" {
        Write-Host "Building Docker image..." -ForegroundColor Cyan
        docker build -t seattle-house-price-api:latest .
    }
    "docker-run" {
        Write-Host "Starting docker-compose..." -ForegroundColor Cyan
        docker-compose up -d
    }
    "docker-stop" {
        Write-Host "Stopping docker-compose..." -ForegroundColor Cyan
        docker-compose down
    }
}
