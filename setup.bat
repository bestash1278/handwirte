@echo off
rem Created 2026-09-21 KST
python --version >nul 2>&1
if errorlevel 1 (
  echo Install Python with pip and Tcl/Tk, and enable Add Python to PATH.
  exit /b 1
)
python -c "import torch, numpy, PIL, tkinter" >nul 2>&1
if errorlevel 1 (
  python -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
if not exist models\mnist.pt (
  python train_model.py
  if errorlevel 1 exit /b 1
)
exit /b 0
