$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $venvPython)) {
    $basePython = $env:PITCHIQ_PYTHON
    if (!$basePython) {
        try {
            $candidate = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $basePython = $candidate }
        } catch {
            # A missing launcher or Python version can use the bundled runtime below.
        }
    }
    if (!$basePython) {
        $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundled) { $basePython = $bundled }
    }
    if (!$basePython) { throw 'Install Python 3.12, or set PITCHIQ_PYTHON to its python.exe path, then retry.' }
    & $basePython -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
}
& $venvPython -m pip install --disable-pip-version-check torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Prototype dependencies could not be installed.' }
& $venvPython (Join-Path $root 'download_model.py')
if ($LASTEXITCODE -ne 0) { throw 'Soccer model download failed.' }
Write-Host 'Ready. Run start.cmd "C:\path\to\match.mp4".'
