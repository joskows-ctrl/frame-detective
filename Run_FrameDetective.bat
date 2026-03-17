@echo off
title Frame Detective V1
if not exist "%~dp0frame_detective_env\Scripts\python.exe" (
    echo Frame Detective is not installed yet.
    echo Please run Install_FrameDetective.bat first.
    pause
    exit /b 1
)
"%~dp0frame_detective_env\Scripts\python.exe" "%~dp0frame_detective_gui.py"
