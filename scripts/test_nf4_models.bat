@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  TokenPowerBench NF4 Model Testing - Windows Batch Script
REM ─────────────────────────────────────────────────────────────────────────────
REM
REM  This batch file runs the PowerShell NF4 testing suite
REM
REM  Usage:
REM    test_nf4_models.bat                     # Run with defaults
REM    test_nf4_models.bat check               # Check availability only
REM    test_nf4_models.bat download            # Download models only
REM    test_nf4_models.bat benchmark           # Benchmark only
REM    test_nf4_models.bat full                # Full workflow (default)
REM ─────────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

REM ── Configuration ────────────────────────────────────────────────────────────
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "LOG_DIR=%PROJECT_DIR%\logs"
set "MODEL_DIR=%USERPROFILE%\models"

REM ── Colors and formatting ────────────────────────────────────────────────────
set "ESC=[0m"

REM ── Parse command line argument ──────────────────────────────────────────────
set MODE=full
if not "%1"=="" (
    set MODE=%1
)

REM ── Title ────────────────────────────────────────────────────────────────────
title TokenPowerBench NF4 Model Testing
color 0B
cls

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║           TokenPowerBench NF4 Model Testing Suite                  ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo Mode: %MODE%
echo Project Dir: %PROJECT_DIR%
echo Model Dir: %MODEL_DIR%
echo.

REM ── Check prerequisites ──────────────────────────────────────────────────────
echo [*] Checking prerequisites...

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Python not found in PATH
    echo Please install Python 3.9+ or add it to your system PATH
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found: %PYTHON_VERSION%

REM ── Check required modules ───────────────────────────────────────────────────
echo [*] Checking Python dependencies...

python -c "import vllm, bitsandbytes, torch" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Required Python packages not installed
    echo Please run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo [OK] All dependencies found

REM ── Create directories ───────────────────────────────────────────────────────
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

REM ── Execute PowerShell script with parameters ────────────────────────────────
echo.
echo [*] Starting PowerShell script...
echo.

if "%MODE%"=="check" (
    echo [MODE] Check only - verifying model availability
    powershell -NoProfile -ExecutionPolicy Bypass ^
        -Command "& '!SCRIPT_DIR!test_nf4_models.ps1' -CheckOnly"
) else if "%MODE%"=="download" (
    echo [MODE] Download only - fetching missing models
    powershell -NoProfile -ExecutionPolicy Bypass ^
        -Command "& '!SCRIPT_DIR!test_nf4_models.ps1' -DownloadOnly"
) else if "%MODE%"=="benchmark" (
    echo [MODE] Benchmark only - testing available models
    powershell -NoProfile -ExecutionPolicy Bypass ^
        -Command "& '!SCRIPT_DIR!test_nf4_models.ps1' -BenchmarkOnly"
) else (
    echo [MODE] Full workflow - check, download, and benchmark
    powershell -NoProfile -ExecutionPolicy Bypass ^
        -Command "& '!SCRIPT_DIR!test_nf4_models.ps1'"
)

set SCRIPT_EXIT=%ERRORLEVEL%

REM ── Report results ───────────────────────────────────────────────────────────
echo.
echo ╔════════════════════════════════════════════════════════════════════╗
if %SCRIPT_EXIT% EQU 0 (
    echo ║                    ✅ TESTING COMPLETED SUCCESSFULLY             ║
) else (
    echo ║                    ❌ TESTING FAILED (Exit: %SCRIPT_EXIT%)               ║
)
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo Results saved to: %PROJECT_DIR%\results\
echo.

REM ── Keep window open ─────────────────────────────────────────────────────────
pause

exit /b %SCRIPT_EXIT%
