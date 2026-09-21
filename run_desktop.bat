@echo off
rem Created 2026-09-21 KST
cd /d "%~dp0"
call setup.bat
if errorlevel 1 goto failed
python desktop_version\digit_recognition.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Application failed. Read the error above.
pause
exit /b 1
