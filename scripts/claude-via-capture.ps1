# Launch Claude Code pointed at the local capture proxy.
#
# Why this exists:
#   ~/.claude/settings.json has ANTHROPIC_BASE_URL hard-coded to the
#   real middleware. Claude Code's settings.json env > process env, so
#   just setting $env:ANTHROPIC_BASE_URL in a shell doesn't win.
#
# What this script does:
#   1. Verifies the capture proxy is actually listening on --proxy-url.
#   2. Makes a throw-away copy of settings.json with the URL swapped
#      to the proxy.
#   3. Points Claude Code at that throw-away settings via the
#      CLAUDE_CONFIG_DIR env (so the real one is untouched).
#   4. On exit, cleans up the temp dir.
#
# Usage:
#   .\scripts\claude-via-capture.ps1
#   .\scripts\claude-via-capture.ps1 -ProxyUrl http://localhost:9100

param(
    [string]$ProxyUrl = "http://localhost:9100",
    [string]$RealSettings = "$env:USERPROFILE\.claude\settings.json"
)

$ErrorActionPreference = "Stop"

# --- Preflight: proxy must be up, otherwise Claude Code just hangs ---
try {
    $status = Invoke-RestMethod -Uri "$ProxyUrl/__capture_status" -TimeoutSec 3
    Write-Host "proxy OK: upstream=$($status.upstream) out=$($status.out)" -ForegroundColor Green
} catch {
    Write-Host "proxy not reachable at $ProxyUrl — start it first" -ForegroundColor Red
    exit 1
}

# --- Build throw-away config dir ---
$tmpDir = Join-Path $env:TEMP ("claude-capture-" + [guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
Write-Host "temp config dir: $tmpDir"

# Rewrite settings.json so settings.env.ANTHROPIC_BASE_URL points at
# the proxy. Preserve everything else; never mutate the original file.
if (-not (Test-Path $RealSettings)) {
    Write-Host "real settings file not found: $RealSettings" -ForegroundColor Red
    exit 1
}
$cfg = Get-Content -Raw -Path $RealSettings -Encoding UTF8 | ConvertFrom-Json
if ($null -eq $cfg.env) {
    $cfg | Add-Member -NotePropertyName env -NotePropertyValue ([pscustomobject]@{})
}
# Assigning to an existing NoteProperty works; for a missing one, force-add.
if ($cfg.env.PSObject.Properties['ANTHROPIC_BASE_URL']) {
    $cfg.env.ANTHROPIC_BASE_URL = $ProxyUrl
} else {
    $cfg.env | Add-Member -NotePropertyName ANTHROPIC_BASE_URL -NotePropertyValue $ProxyUrl
}
($cfg | ConvertTo-Json -Depth 32) | Set-Content -Encoding UTF8 -Path (Join-Path $tmpDir "settings.json")
Write-Host "patched settings written; ANTHROPIC_BASE_URL -> $ProxyUrl" -ForegroundColor Green

try {
    $env:CLAUDE_CONFIG_DIR = $tmpDir
    Write-Host "starting claude against $ProxyUrl ..." -ForegroundColor Cyan
    claude @args
} finally {
    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
    Remove-Item Env:\CLAUDE_CONFIG_DIR -ErrorAction SilentlyContinue
}
