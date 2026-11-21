#Requires -Version 5.1
<#
.SYNOPSIS
    Build the quant_toolkit.finance._core C++ extension modules and install them in-tree.

.DESCRIPTION
    PowerShell twin of bin/bash/build_core.sh. Requires the dev env to be active (python, pybind11,
    cmake, ninja: pip install -e '.[cpp]') and a C++ toolchain on PATH (run from a
    "Developer PowerShell for VS" so Ninja finds MSVC). The .pyd lands in
    packages/finance/src/quant_toolkit/finance/_core/, next to the editable install.

.EXAMPLE
    bin\powershell\build_core.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)" }
}

Push-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
try {
    $python = (Get-Command python).Source
    $pybind11Dir = (python -m pybind11 --cmakedir)
    if ($LASTEXITCODE -ne 0) { throw "pybind11 not found; pip install -e '.[cpp]'" }

    Invoke-Native cmake @("-S", ".", "-B", "build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DPython_EXECUTABLE=$python", "-Dpybind11_DIR=$pybind11Dir")
    Invoke-Native cmake @("--build", "build")
    Invoke-Native cmake @("--install", "build", "--prefix", ".")

    # expose the compilation database for clangd/CLion (copied: symlinks need Developer Mode on Windows)
    if (Test-Path "build\compile_commands.json") {
        Copy-Item "build\compile_commands.json" "compile_commands.json" -Force
    }
} finally {
    Pop-Location
}
