#Requires -Version 5.1
<#
.SYNOPSIS
    Editable-install every portion of the quant_toolkit family into the active environment.

.DESCRIPTION
    PowerShell twin of bin/bash/dev_install.sh. Idempotent. Without -Deps, --no-deps leaves
    numpy/scipy/pandas to the caller (conda or pip); with -Deps the dev extras (tests, lint,
    notebooks) are installed as well.

.EXAMPLE
    bin\powershell\dev_install.ps1 -Deps
#>
[CmdletBinding()]
param(
    [switch]$Deps
)

$ErrorActionPreference = "Stop"
Push-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
try {
    if ($Deps) {
        pip install -e packages/common -e packages/finance -e ".[dev,notebook,accel]"
    } else {
        pip install --no-deps -e packages/common -e packages/finance
    }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }

    # a broken namespace shows up immediately: all portion roots must be listed
    $check = @'
import quant_toolkit
paths = list(quant_toolkit.__path__)
print("quant_toolkit.__path__")
for p in paths:
    print("  ", p)
'@
    python -c $check
    if ($LASTEXITCODE -ne 0) { throw "quant_toolkit namespace check failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}
