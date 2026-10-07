@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run prototypes\gpu-tracker\setup.cmd first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m unittest discover -s "%~dp0tests" -t "%~dp0tests" %*
exit /b %errorlevel%
