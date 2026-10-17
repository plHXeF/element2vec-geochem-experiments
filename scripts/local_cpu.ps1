# Run repository commands from the repository root with the locked CPU environment.
# Example: ./scripts/local_cpu.ps1 scripts/experiment.py list
# This wrapper does not activate or modify the existing root .venv.
$ErrorActionPreference = 'Stop'
$researchRepo = Split-Path -Parent $PSScriptRoot
$researchPython = Join-Path $researchRepo 'environments/local-cpu/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $researchPython)) {
    throw 'Create the environment first: uv sync --locked --project environments/local-cpu'
}
foreach ($researchThreadVariable in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) {
    [Environment]::SetEnvironmentVariable($researchThreadVariable, '1', 'Process')
}
$env:PYTHONUTF8 = '1'
$env:PYTHONNOUSERSITE = '1'
if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path $researchRepo 'environments/local-cpu/.venv/.cache/matplotlib'
}
Push-Location -LiteralPath $researchRepo
try {
    & $researchPython @args
    $researchExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $researchExitCode
