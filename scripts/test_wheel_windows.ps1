# Test a cross-built Windows wheel on a Windows host (PowerShell).
#   1. In WSL: bash scripts/build_wheel_windows.sh   (-> dist\augrs-*-cp39-abi3-win_amd64.whl)
#   2. On Windows: powershell -File scripts\test_wheel_windows.ps1 -Python C:\path\to\python.exe -VenvDir C:\tmp\augrs-venv
# Creates a fresh venv (never touches the global Python), installs the wheel + test deps,
# runs the smoke test and the compat test suite against the installed wheel.
param(
    [string]$Python = "python",  # any Python >= 3.9 (only used to create the venv)
    [string]$VenvDir = "$env:TEMP\augrs_win_venv"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$wheel = Get-ChildItem "$root\dist\augrs-*-win_amd64.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $wheel) { throw "no Windows wheel in $root\dist (run scripts/build_wheel_windows.sh in WSL first)" }
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) { & $Python -m venv $VenvDir }
$py = "$VenvDir\Scripts\python.exe"
& $py -m pip install --disable-pip-version-check --quiet --force-reinstall --no-deps $wheel.FullName
& $py -m pip install --disable-pip-version-check --quiet numpy pytest pyyaml "albumentations==2.0.8" opencv-python-headless
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
Push-Location $VenvDir
try {
    & $py "$root\scripts\smoke_test.py"
    if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }
    & $py -m pytest "$root\compat-tests" -q -p no:cacheprovider --rootdir $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
} finally { Pop-Location }
