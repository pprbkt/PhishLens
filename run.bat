@echo off
title PhishLens Server
setlocal enabledelayedexpansion

echo ============================================================
echo Starting PhishLens Application
echo ============================================================
echo.

:: Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)

:: Check if dependencies are installed
python -c "import fastapi, uvicorn, rapidocr_onnxruntime, sklearn" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: Check if model artifacts exist; train if missing
if not exist "models\classifier.joblib" (
    echo [INFO] Trained model artifacts not found. Running training pipeline...
    python training\train.py
    if errorlevel 1 (
        echo [ERROR] Training pipeline failed.
        pause
        exit /b 1
    )
)

echo [INFO] Server starting at http://localhost:8080
echo [INFO] Swagger API Docs at http://localhost:8080/docs
echo.
echo Press Ctrl+C in this window to stop the server.
echo.

:: Open browser after a brief delay in background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080"

:: Start Uvicorn Server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error.
    pause
)
