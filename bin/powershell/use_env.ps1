#Requires -Version 5.1
<#
.SYNOPSIS
    Create (if needed) and activate a conda environment, then editable-install the repo into it.

.DESCRIPTION
    PowerShell twin of bin/bash/use_env.sh. Must be dot-sourced so the activation sticks to the
    calling session, and conda must be hooked into PowerShell (conda init powershell).

    Environments are named conda/<prefix>/<prefix>-<YYYYMM>.yml. If the environment does not exist
    it is created from its env file. If the suffix is 'latest' or omitted, the name is read from
    conda/<prefix>/<prefix>-latest.txt. With -Prune an existing environment is pruned against its
    env file after activation.

.EXAMPLE
    . bin\powershell\use_env.ps1 pricing
.EXAMPLE
    . bin\powershell\use_env.ps1 pricing-202611 -Prune
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Name,
    [switch]$Prune
)

if ($MyInvocation.InvocationName -ne ".") {
    Write-Host "This script is intended to be dot-sourced, not executed"
    Write-Host "    . $($MyInvocation.MyCommand.Path) <name> [-Prune]"
    exit 1
}

# dot-sourcing runs in the caller's scope: keep helpers prefixed and remove them on the way out
try {
    if (-not $Name) {
        Write-Host "No environment name specified. Usage: . bin\powershell\use_env.ps1 <name> [-Prune]"
        return
    }
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Write-Host "conda is not on PATH; run 'conda init powershell' (from the Miniforge Prompt) and restart the shell"
        return
    }

    $__ue_root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

    # split env name into prefix and suffix
    $__ue_prefix, $__ue_suffix = $Name -split "-", 2
    $__ue_dir = Join-Path $__ue_root "conda\$__ue_prefix"

    # use latest if suffix empty or 'latest'
    $__ue_name = $Name
    if (-not $__ue_suffix -or $__ue_suffix -eq "latest") {
        $__ue_latest = Join-Path $__ue_dir "$__ue_prefix-latest.txt"
        if (-not (Test-Path $__ue_latest)) {
            Write-Host "No latest file found for dir $__ue_dir and prefix $__ue_prefix for env name $Name"
            return
        }
        $__ue_name = (Get-Content $__ue_latest -TotalCount 1).Trim()
    }

    $__ue_exists = [bool](conda env list | ForEach-Object { ($_ -split "\s+")[0] } | Where-Object { $_ -eq $__ue_name })
    $__ue_file = Join-Path $__ue_dir "$__ue_name.yml"

    if (-not $__ue_exists) {
        if (-not (Test-Path $__ue_file)) {
            Write-Host "Environment file $__ue_file does not exist."
            return
        }
        Write-Host "Creating environment $__ue_name from $__ue_file..."
        conda env create -n $__ue_name -f $__ue_file
        if ($LASTEXITCODE -ne 0) { Write-Host "conda env create failed"; return }
    } else {
        Write-Host "Environment $__ue_name already exists"
    }

    Write-Host "Activating environment $__ue_name..."
    conda activate $__ue_name

    # the conda env only supplies Python; the library comes from the repo's editable installs
    $__ue_install = Join-Path $__ue_root "bin\powershell\dev_install.ps1"
    & $__ue_install -Deps

    if ($Prune -and $__ue_exists) {
        Write-Host "Pruning environment $__ue_name..."
        conda env update -n $__ue_name -f $__ue_file --prune
        & $__ue_install -Deps
    }
} finally {
    Remove-Variable -Name __ue_* -ErrorAction SilentlyContinue
}
