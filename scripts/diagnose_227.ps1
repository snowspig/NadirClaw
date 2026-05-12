# Windows PowerShell 版 227 诊断脚本（专攻 reasoning rate-limit 问题）
# 用法（在 227 的 PowerShell 里）：
#   cd 到 NadirClaw 安装目录
#   powershell -ExecutionPolicy Bypass -File .\diagnose_227.ps1 | Tee-Object -FilePath .\nadirclaw_diag.txt
# 然后把 nadirclaw_diag.txt 贴回来。
#
# 只读：不改配置、不重启服务、不动 key。

$ErrorActionPreference = 'Continue'
function Sep($t) { Write-Host ""; Write-Host "===== $t =====" -ForegroundColor Cyan }

$envFile = Join-Path $env:USERPROFILE '.nadirclaw\.env'
$db      = Join-Path $env:USERPROFILE '.nadirclaw\logs\requests.db'
$log     = Join-Path $env:USERPROFILE '.nadirclaw\logs\server.log'

Sep "META"
Get-Date
"host=$env:COMPUTERNAME user=$env:USERNAME"
"powershell=$($PSVersionTable.PSVersion)"

Sep "H1: 部署位置与分支"
$repoCandidates = @(
  (Get-Location).Path,
  (Join-Path $env:USERPROFILE 'NadirClaw'),
  'C:\NadirClaw', 'D:\NadirClaw'
)
$repo = $null
foreach ($d in $repoCandidates) {
  if (Test-Path (Join-Path $d '.git')) { $repo = $d; break }
}
if (-not $repo) { "NOT FOUND — 在 NadirClaw 源码目录里跑" }
else {
  "repo=$repo"
  Push-Location $repo
  git rev-parse --abbrev-ref HEAD
  git log --oneline -5
  "--- anthropic_api include? ---"
  Select-String -Path (Join-Path $repo 'nadirclaw\server.py') -Pattern 'anthropic_api|include_router' | Select-Object -First 4
  "--- claude-* -> auto rewrite still present? ---"
  $ap = Join-Path $repo 'nadirclaw\anthropic_api.py'
  if (Test-Path $ap) { Select-String -Path $ap -Pattern 'startswith\("claude-"\)' }
  Pop-Location
}

Sep "服务进程与端口"
Get-Process | Where-Object { $_.ProcessName -match 'python|uvicorn|nadirclaw' } | Select-Object Id,ProcessName,StartTime,CPU | Format-Table -AutoSize
Get-NetTCPConnection -LocalPort 8856 -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,State,OwningProcess | Format-Table -AutoSize

Sep "H2: .env 关键配置（脱敏）"
if (Test-Path $envFile) {
  "ls: $((Get-Item $envFile).LastWriteTime)  size=$((Get-Item $envFile).Length)"
  Get-Content $envFile | Where-Object { $_ -match '^(NADIRCLAW_|ANTHROPIC_|OPENAI_|GEMINI_|ZAI_|KIMI_|MINIMAX_|GOOGLE_)' } | ForEach-Object {
    $kv = $_ -split '=', 2
    if ($kv.Length -eq 2) {
      $v = $kv[1].Trim('"')
      if ($v.Length -gt 6) { "$($kv[0])=$($v.Substring(0,6))***(len=$($v.Length))" }
      else { "$($kv[0])=<empty-or-short>" }
    }
  }
} else { "no $envFile" }

Sep "===== 最关键：reasoning tier 的 fallback_reasons（决定根因） ====="
if (Test-Path $db) {
  # 需要 sqlite3.exe 在 PATH，否则用 python
  $sqlite = Get-Command sqlite3 -ErrorAction SilentlyContinue
  if ($sqlite) {
    $sqlQuery = @"
.headers on
.mode column
.width 20 28 10 14 60 50
SELECT datetime(ts,'unixepoch','localtime') AS t,
       selected_model,
       tier,
       fallback_used,
       substr(COALESCE(fallback_reasons,''),1,200) AS fallback_reasons,
       substr(COALESCE(error,''),1,150) AS error
FROM requests
WHERE tier='reasoning'
ORDER BY ts DESC LIMIT 8;
"@
    $sqlQuery | & sqlite3.exe $db
  } else {
    # 回退：用 python 的内置 sqlite3
    $py = @"
import sqlite3, json, sys
conn = sqlite3.connect(r'$db')
conn.row_factory = sqlite3.Row
cur = conn.execute(
  "SELECT datetime(ts,'unixepoch','localtime') t, selected_model, tier, "
  "fallback_used, fallback_reasons, substr(COALESCE(error,''),1,200) err "
  "FROM requests WHERE tier='reasoning' ORDER BY ts DESC LIMIT 8"
)
for row in cur:
    print('---')
    for k in row.keys():
        v = row[k]
        if k == 'fallback_reasons' and v:
            try:
                parsed = json.loads(v)
                print(f'  {k}:')
                for e in parsed:
                    print(f'    - model={e.get("model")} error_type={e.get("error_type")} msg={(e.get("message") or "")[:120]}')
                continue
            except Exception: pass
        print(f'  {k}: {v}')
"@
    $py | python
  }
} else { "no $db" }

Sep "按 error_type 分桶最近 24h reasoning 失败"
if (Test-Path $db) {
  $py = @"
import sqlite3, json, time
conn = sqlite3.connect(r'$db')
cutoff = time.time() - 86400
buckets = {}
for (fr,) in conn.execute("SELECT fallback_reasons FROM requests WHERE tier='reasoning' AND ts>?", (cutoff,)):
    if not fr: continue
    try: rows = json.loads(fr)
    except Exception: continue
    for e in rows:
        et = e.get('error_type','?')
        buckets[et] = buckets.get(et, 0) + 1
for k,v in sorted(buckets.items(), key=lambda x:-x[1]):
    print(f'  {k:30s} {v}')
if not buckets: print('  (no fallback_reasons in last 24h)')
"@
  $py | python
}

Sep "reasoning tier 最近 24h 打向的 model"
if (Test-Path $db) {
  $py = @"
import sqlite3, time
conn = sqlite3.connect(r'$db')
cutoff = time.time() - 86400
for (m,n) in conn.execute("SELECT selected_model, COUNT(*) FROM requests WHERE tier='reasoning' AND ts>? GROUP BY selected_model ORDER BY 2 DESC", (cutoff,)):
    print(f'  {m:40s} {n}')
"@
  $py | python
}

Sep "直连上游（绕过 NadirClaw）"
if (Test-Path $envFile) {
  $base = (Get-Content $envFile | Where-Object { $_ -match '^ANTHROPIC_API_BASE=' } | Select-Object -First 1) -replace '^ANTHROPIC_API_BASE=', '' -replace '"',''
  $key  = (Get-Content $envFile | Where-Object { $_ -match '^ANTHROPIC_API_KEY='  } | Select-Object -First 1) -replace '^ANTHROPIC_API_KEY=',  '' -replace '"',''
  if ($base -and $key) {
    "upstream: $base"
    $body = @{ model = "claude-opus-4-6-20250918"; max_tokens = 16; messages = @(@{ role = "user"; content = "hi" }) } | ConvertTo-Json -Depth 5
    try {
      $sw = [Diagnostics.Stopwatch]::StartNew()
      $resp = Invoke-WebRequest -Uri "$base/v1/messages" -Method POST `
        -Headers @{ "x-api-key" = $key; "anthropic-version" = "2023-06-01"; "content-type" = "application/json" } `
        -Body $body -TimeoutSec 30 -UseBasicParsing
      $sw.Stop()
      "status=$($resp.StatusCode) time=$($sw.Elapsed.TotalSeconds)s len=$($resp.Content.Length)"
      $resp.Content.Substring(0, [Math]::Min(300, $resp.Content.Length))
    } catch {
      "EXCEPTION: $($_.Exception.Message)"
      if ($_.Exception.Response) {
        $reader = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
        $body = $reader.ReadToEnd()
        "response body (first 400): $($body.Substring(0, [Math]::Min(400, $body.Length)))"
      }
    }
  } else { "ANTHROPIC_API_BASE or ANTHROPIC_API_KEY not set" }
}

Sep "本地 /v1/messages（走 NadirClaw）"
$tok = $null
if (Test-Path $envFile) {
  $tok = (Get-Content $envFile | Where-Object { $_ -match '^NADIRCLAW_AUTH_TOKEN=' } | Select-Object -First 1) -replace '^NADIRCLAW_AUTH_TOKEN=','' -replace '"',''
}
try {
  $body = @{ model = "claude-opus-4-6-20250918"; max_tokens = 16; messages = @(@{ role = "user"; content = "hi" }) } | ConvertTo-Json -Depth 5
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $headers = @{ "anthropic-version" = "2023-06-01"; "content-type" = "application/json" }
  if ($tok) { $headers["Authorization"] = "Bearer $tok" }
  $resp = Invoke-WebRequest -Uri "http://localhost:8856/v1/messages" -Method POST -Headers $headers -Body $body -TimeoutSec 60 -UseBasicParsing
  $sw.Stop()
  "status=$($resp.StatusCode) time=$($sw.Elapsed.TotalSeconds)s"
  "X-Routed-Model: $($resp.Headers['X-Routed-Model'])"
  "X-Routed-Tier: $($resp.Headers['X-Routed-Tier'])"
  $resp.Content.Substring(0, [Math]::Min(300, $resp.Content.Length))
} catch { "EXCEPTION: $($_.Exception.Message)" }

Sep "最近 80 行含 reasoning/ERROR/WARNING/opus 的日志"
if (Test-Path $log) {
  Get-Content $log -Tail 500 | Select-String -Pattern 'reasoning|ERROR|WARNING|anthropic|opus' | Select-Object -Last 80
} else { "no $log" }

Sep DONE
