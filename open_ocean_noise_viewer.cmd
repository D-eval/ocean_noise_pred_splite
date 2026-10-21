@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=F:\Applications\miniconda\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" -m source.day_noise_viewer --open
if errorlevel 1 pause
