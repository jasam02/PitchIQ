@echo off
rem Speed pass 3 verification on the GPU machine: unit tests, the two reference windows with the committed code
rem (a git worktree of gpu-tracking-new) and this code, exact_diff / compare, the 60 s broadcast timing back to
rem back, two consecutive runs against two fresh ones, the ball decode side-by-side on real frames, the ball
rem clips. Outputs go to the scratch folder (default C:\scratch\speed-pass-3); nothing under runs\ is written.
rem
rem   tools\verify_speed_pass_3.cmd                       everything
rem   tools\verify_speed_pass_3.cmd --only tests,windows,diffs
rem   tools\verify_speed_pass_3.cmd --dry-run             print the commands only
rem   tools\verify_speed_pass_3.cmd --scratch D:/scratch/sp3
rem
rem Check nvidia-smi first: the timing steps need an idle GPU.
setlocal
set "HERE=%~dp0.."
if not exist "%HERE%\.venv\Scripts\python.exe" (
  echo Run prototypes\gpu-tracker\setup.cmd first.
  exit /b 1
)
"%HERE%\.venv\Scripts\python.exe" "%HERE%\tools\verify_speed_pass_3.py" %*
exit /b %errorlevel%
