#!/usr/bin/env bash
# 聚焦"reasoning tier 全失败"的窄诊断
# 用法：ssh SNOWSPIG@192.168.8.227 'bash -s' < scripts/diagnose_reasoning_failures.sh
set -u
sep() { printf '\n===== %s =====\n' "$*"; }

DB=~/.nadirclaw/logs/requests.db
ENV=~/.nadirclaw/.env

sep "最近 reasoning tier 的请求是否都失败，失败原因是什么"
if [ -f "$DB" ]; then
  sqlite3 "$DB" ".headers on" ".mode column" <<'SQL'
SELECT datetime(ts,'unixepoch','localtime') AS t,
       selected_model,
       fallback_used,
       substr(COALESCE(error, ''), 1, 120) AS error,
       substr(COALESCE(fallback_reasons, ''), 1, 300) AS fallback_reasons
FROM requests
WHERE tier = 'reasoning'
ORDER BY ts DESC
LIMIT 10;
SQL
else
  echo "no $DB"
fi

sep "按 error_type 分桶看最近 24h 的 reasoning 失败"
if [ -f "$DB" ]; then
  # json_each 需要 sqlite3 >= 3.38，回退方案用 like 匹配
  sqlite3 "$DB" <<'SQL'
SELECT
  CASE
    WHEN fallback_reasons LIKE '%RateLimitExhausted%' OR fallback_reasons LIKE '%RateLimitError%' THEN 'rate_limit'
    WHEN fallback_reasons LIKE '%ReadTimeout%' OR fallback_reasons LIKE '%ConnectTimeout%' THEN 'timeout'
    WHEN fallback_reasons LIKE '%ConnectError%' OR fallback_reasons LIKE '%APIConnectionError%' THEN 'connection_error'
    WHEN fallback_reasons LIKE '%AuthenticationError%' OR fallback_reasons LIKE '%401%' THEN 'auth'
    WHEN fallback_reasons LIKE '%RemoteProtocolError%' THEN 'disconnected'
    WHEN fallback_reasons IS NULL OR fallback_reasons = '' OR fallback_reasons = '[]' THEN 'no_fallback'
    ELSE 'other'
  END AS bucket,
  COUNT(*) AS n
FROM requests
WHERE tier = 'reasoning' AND ts > strftime('%s', 'now', '-24 hours')
GROUP BY bucket
ORDER BY n DESC;
SQL
fi

sep "reasoning tier 打向的是哪些 model（看 selected_model 分布）"
if [ -f "$DB" ]; then
  sqlite3 "$DB" <<'SQL'
SELECT selected_model, COUNT(*) AS n
FROM requests
WHERE tier = 'reasoning' AND ts > strftime('%s', 'now', '-24 hours')
GROUP BY selected_model
ORDER BY n DESC;
SQL
fi

sep ".env 里 Anthropic 相关配置（脱敏）"
if [ -f "$ENV" ]; then
  grep -E '^(ANTHROPIC_|NADIRCLAW_REASONING|NADIRCLAW_REASONING_FALLBACK|NADIRCLAW_COMPLEX_MODEL)' "$ENV" \
    | awk -F= '{v=$2; n=length(v); if(n>6){print $1"="substr(v,1,6)"***(len="n")"} else {print $1"=<empty-or-short>"}}'
fi

sep "直连中间商 /v1/messages 能否走通（绕过 NadirClaw）"
# 从 .env 取出 ANTHROPIC_API_BASE 和 ANTHROPIC_API_KEY
if [ -f "$ENV" ]; then
  BASE=$(grep '^ANTHROPIC_API_BASE=' "$ENV" | cut -d= -f2- | tr -d '"')
  KEY=$(grep '^ANTHROPIC_API_KEY=' "$ENV" | cut -d= -f2- | tr -d '"')
fi
if [ -n "${BASE:-}" ] && [ -n "${KEY:-}" ]; then
  echo "direct upstream: $BASE"
  curl -sS -o /tmp/nc_up.json -D /tmp/nc_up.hdr \
    -w "http=%{http_code} total=%{time_total}s connect=%{time_connect}s tls=%{time_appconnect}s\n" \
    --max-time 30 \
    -X POST "$BASE/v1/messages" \
    -H "x-api-key: $KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d '{"model":"claude-opus-4-6-20250918","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'
  echo "--- status line ---"
  head -1 /tmp/nc_up.hdr
  echo "--- first 300 bytes of body ---"
  head -c 300 /tmp/nc_up.json; echo
else
  echo "ANTHROPIC_API_BASE or ANTHROPIC_API_KEY not set in $ENV — 跳过直连测试"
fi

sep "服务最近 80 行日志（只筛 reasoning/ERROR/WARNING）"
LOG=~/.nadirclaw/logs/server.log
if [ -f "$LOG" ]; then
  tail -n 500 "$LOG" | grep -iE 'reasoning|ERROR|WARNING|anthropic|opus' | tail -n 80
elif systemctl list-units --type=service 2>/dev/null | grep -q nadirclaw; then
  journalctl -u nadirclaw --no-pager -n 500 | grep -iE 'reasoning|ERROR|WARNING|anthropic|opus' | tail -n 80
fi

sep DONE
