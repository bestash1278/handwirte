@echo off
rem Created 2026-09-21 KST
cd /d "%~dp0"
call setup.bat
if errorlevel 1 goto failed
echo Open http://127.0.0.1:8000 in your browser.
python web_version\server.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Application failed. Read the error above.
pause
exit /b 1
