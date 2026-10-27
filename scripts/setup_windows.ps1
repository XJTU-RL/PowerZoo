#Requires -Version 5.1
<#
.SYNOPSIS
Create an isolated Windows environment and validate the actual PowerZoo runtime.
.EXAMPLE
powershell -File scripts/setup_windows.ps1 -TorchFlavor cu124
#>
[CmdletBinding()]
param(
    [string]$EnvironmentName = "powerzoo-win",
    [ValidateSet("cpu", "cu124")]
    [string]$TorchFlavor = "cpu"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Invoke-Conda {
    # Propagate native command failures on both Windows PowerShell and PowerShell 7.
    & conda @args
    if ($LASTEXITCODE -ne 0) { throw "Conda command failed (exit $LASTEXITCODE)." }
}

Push-Location $projectRoot
try {
    $environments = (& conda env list --json | ConvertFrom-Json).envs
    if ($LASTEXITCODE -ne 0) { throw "Cannot list Conda environments." }
    $exists = $environments | Where-Object { (Split-Path -Leaf $_) -eq $EnvironmentName }
    if (-not $exists) {
        Invoke-Conda create --yes --name $EnvironmentName --override-channels --channel conda-forge "python=3.10" "pip>=24" "setuptools>=68,<81" wheel
    }
    Invoke-Conda run --no-capture-output --name $EnvironmentName python -c "import sys; assert sys.version_info[:2] == (3, 10), 'Use Python 3.10 for the Windows constraints'"
    Invoke-Conda env config vars set --name $EnvironmentName PYTHONUTF8=1
    Invoke-Conda run --no-capture-output --name $EnvironmentName python -m pip install "torch==2.6.0+$TorchFlavor" --index-url "https://download.pytorch.org/whl/$TorchFlavor"
    Invoke-Conda run --no-capture-output --name $EnvironmentName python -m pip install -r requirements-windows.txt
    Invoke-Conda run --no-capture-output --name $EnvironmentName python -m pip install --no-deps -e .
    Invoke-Conda run --no-capture-output --name $EnvironmentName python -m pip check
    $smokeArgs = @("scripts/check_runtime.py", "--train")
    if ($TorchFlavor -ne "cpu") { $smokeArgs += "--require-cuda" }
    Invoke-Conda run --no-capture-output --name $EnvironmentName python @smokeArgs
    Write-Host "Ready: conda activate $EnvironmentName"
}
finally {
    Pop-Location
}
