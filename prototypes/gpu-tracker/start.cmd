@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run prototypes\gpu-tracker\setup.cmd first.
  exit /b 1
)
if "%~1"=="" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0server.py"
  exit /b %errorlevel%
)
if "%~1"=="--port" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0server.py" %*
  exit /b %errorlevel%
)
"%~dp0.venv\Scripts\python.exe" "%~dp0server.py" --video "%~1" %2 %3 %4 %5 %6 %7 %8 %9
