# Run from PowerShell: .\scripts\run_windows_gpu.ps1 -Ks 1 -Seeds 42 -Jobs 1 -Epochs 1
# Default: all k values, three seeds, two concurrent GPU training processes.
param(
    [int[]]$Ks = @(1, 2, 4, 8),
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateRange(1, 16)][int]$Jobs = 2,
    [ValidateRange(1, 100000)][int]$Epochs = 100,
    [switch]$BuildOnly,
    [switch]$SkipBuild,
    [switch]$DryRun,
    [string]$Python = ""
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $root '.venv\Scripts\python.exe'
}
if (-not (Test-Path $Python)) {
    throw "Python not found at $Python. Create .venv as described in README.md, or pass -Python."
}

$arguments = @((Join-Path $PSScriptRoot 'run_parallel.py'), '--ks')
$arguments += @($Ks | ForEach-Object { [string]$_ })
$arguments += '--seeds'
$arguments += @($Seeds | ForEach-Object { [string]$_ })
$arguments += @('--jobs', [string]$Jobs, '--epochs', [string]$Epochs)
if ($BuildOnly) { $arguments += '--build-only' }
if ($SkipBuild) { $arguments += '--skip-build' }
if ($DryRun) { $arguments += '--dry-run' }

& $Python @arguments
exit $LASTEXITCODE
