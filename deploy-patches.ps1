# Deploy NadirClaw patches from this fork to pip site-packages
# Usage: .\deploy-patches.ps1 [-DryRun]

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Find site-packages directory
$sitePackages = python -c "import nadirclaw; import os; print(os.path.dirname(nadirclaw.__file__))" 2>$null

if (-not $sitePackages) {
    Write-Host "ERROR: nadirclaw not found. Install it first: pip install nadirclaw==0.14.3" -ForegroundColor Red
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$patchesDir = Join-Path $scriptDir "nadirclaw"

$patchFiles = @(
    "server.py",
    "routing.py",
    "web_dashboard.py",
    "compress.py",
    "quota.py",
    "settings.py"
)

Write-Host "Target: $sitePackages"
Write-Host ""

foreach ($f in $patchFiles) {
    $src = Join-Path $patchesDir $f
    $dst = Join-Path $sitePackages $f

    if (-not (Test-Path $src)) {
        Write-Host "SKIP: $f (not found in fork)" -ForegroundColor Yellow
        continue
    }

    if (-not (Test-Path $dst)) {
        Write-Host "SKIP: $f (not found in site-packages)" -ForegroundColor Yellow
        continue
    }

    if ($DryRun) {
        Write-Host "WOULD COPY: $f" -ForegroundColor Cyan
    } else {
        Copy-Item $src $dst -Force
        Write-Host "PATCHED: $f" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Done! Restart NadirClaw to apply changes."
Write-Host "  nadirclaw serve --verbose"
