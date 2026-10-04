@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run prototypes\gpu-tracker\setup.cmd first.
  exit /b 1
)
if "%~1"=="" (
  echo Usage: prototypes\gpu-tracker\start.cmd "C:\path\to\match.mp4"
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0server.py" --video "%~1" %2 %3 %4 %5 %6 %7 %8 %9
