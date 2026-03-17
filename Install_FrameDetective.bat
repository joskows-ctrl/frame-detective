@echo off
title Frame Detective V1 - Installer
echo ============================================
echo   Frame Detective V1 - GPU Installer
echo   by Jonah Oskow
echo ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv frame_detective_env
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Installing PyTorch with CUDA support (this may take a few minutes)...
frame_detective_env\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [WARNING] CUDA install failed. Falling back to CPU-only PyTorch...
    frame_detective_env\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo [3/4] Installing dependencies...
frame_detective_env\Scripts\pip.exe install opencv-python-headless ccvfi Pillow
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [4/4] Creating launcher...
(
echo @echo off
echo title Frame Detective V1
echo "%~dp0frame_detective_env\Scripts\python.exe" "%~dp0frame_detective_gui.py"
) > "%~dp0Run_FrameDetective.bat"

echo.
echo ============================================
echo   Installation complete!
echo   Double-click Run_FrameDetective.bat to launch.
echo ============================================
echo.
pause
