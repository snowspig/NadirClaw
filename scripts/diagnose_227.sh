#!/usr/bin/env bash
# NadirClaw 远端诊断脚本
# 用法：
#   scp scripts/diagnose_227.sh SNOWSPIG@192.168.8.227:~/diagnose_nadirclaw.sh
#   ssh SNOWSPIG@192.168.8.227 "bash ~/diagnose_nadirclaw.sh" 2>&1 | tee nadirclaw_diag.txt
#
# 或者直接：
#   ssh SNOWSPIG@192.168.8.227 'bash -s' < scripts/diagnose_227.sh 2>&1 | tee nadirclaw_diag.txt
#
# 它只做读取，不改配置、不重启服务。

set -u
HOST_LABEL="$(hostname)"
sep() { printf '\n===== %s =====\n' "$*"; }

sep "META"
date
echo "host=$HOST_LABEL user=$(id -un)"
echo "uname=$(uname -a)"

sep "H1: 部署位置与分支"
# 尝试几个常见位置；真实路径改这里
for d in ~/NadirClaw /opt/NadirClaw /srv/NadirClaw /usr/local/NadirClaw ./NadirClaw; do
  [ -d "$d/.git" ] && REPO="$d" && break
done
if [ -z "${REPO:-}" ]; then
  echo "NOT FOUND — 手动告诉我 NadirClaw 安装目录"
else
  echo "repo=$REPO"
  (cd "$REPO" && git rev-parse --abbrev-ref HEAD && git log --oneline -5)
  echo "--- include_router 是否挂 anthropic_api ---"
  grep -nE "anthropic_api|include_router" "$REPO/nadirclaw/server.py" 2>/dev/null | head
  echo "--- /v1/messages 在 server.py 或 anthropic_api.py 吗 ---"
  grep -nE '@router\.post|@app\.post' "$REPO/nadirclaw/anthropic_api.py" 2>/dev/null | head
fi

sep "H0: claude-* → auto 强制改写是否仍在"
if [ -n "${REPO:-}" ]; then
  grep -n 'startswith("claude-")' "$REPO/nadirclaw/anthropic_api.py" 2>/dev/null || echo "no match"
fi

sep "服务进程与端口"
ps -ef | grep -E 'nadirclaw|uvicorn' | grep -v grep
echo "--- 监听 8856 ---"
(ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E ':8856|nadirclaw' || echo "no listener on 8856"

sep "H2: .env 里的关键配置（脱敏）"
ENV=~/.nadirclaw/.env
if [ -f "$ENV" ]; then
  echo "ls: $(ls -la $ENV)"
  # 只打印 key 名和 value 前 6 个字符，尾随 '***'
  grep -E '^(NADIRCLAW_|ANTHROPIC_|OPENAI_|GEMINI_|ZAI_|KIMI_|MINIMAX_|GOOGLE_)' "$ENV" \
    | awk -F= '{v=$2; n=length(v); if(n>6){print $1"="substr(v,1,6)"***(len="n")"} else {print $1"=<empty-or-short>"}}'
else
  echo "no $ENV"
fi

sep "进程环境里是否有 ANTHROPIC_API_KEY=local 这种"
# 只看 nadirclaw 进程的 environ
PIDS=$(pgrep -f 'nadirclaw|uvicorn' 2>/dev/null)
for pid in $PIDS; do
  echo "--- pid=$pid ---"
  for k in ANTHROPIC_API_KEY ANTHROPIC_API_BASE NADIRCLAW_COMPLEX_MODEL NADIRCLAW_MODELS NADIRCLAW_FALLBACK_CHAIN; do
    v=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep "^$k=" | head -1 | cut -d= -f2-)
    n=${#v}
    if [ -z "$v" ]; then
      echo "  $k=(unset)"
    elif [ "$n" -gt 8 ]; then
      echo "  $k=${v:0:8}***(len=$n)"
    else
      echo "  $k=$v"
    fi
  done
done

sep "H3/H4: 端点健康检查"
# 本地不知 AUTH_TOKEN 就先读 env
TOK=$(grep '^NADIRCLAW_AUTH_TOKEN=' "$ENV" 2>/dev/null | cut -d= -f2- | tr -d '"')
[ -z "$TOK" ] && TOK="${NADIRCLAW_AUTH_TOKEN:-}"
echo "token_len=${#TOK}"

echo "--- /health ---"
curl -sS -o /dev/null -w "http=%{http_code} time=%{time_total}s\n" http://localhost:8856/health

echo "--- /v1/messages (Anthropic 客户端实际走的路径) ---"
curl -sS -o /tmp/nc_msg.json -D /tmp/nc_msg.hdr -w "http=%{http_code} time=%{time_total}s\n" \
  -X POST http://localhost:8856/v1/messages \
  -H "Authorization: Bearer $TOK" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-4-6-20250918","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}'
echo "--- response headers ---"
grep -iE '^(HTTP/|x-routed|x-complexity|retry-after|content-type)' /tmp/nc_msg.hdr 2>/dev/null
echo "--- response body (前 400 字符) ---"
head -c 400 /tmp/nc_msg.json; echo

echo "--- /v1/chat/completions with model=opus ---"
curl -sS -o /tmp/nc_cc.json -D /tmp/nc_cc.hdr -w "http=%{http_code} time=%{time_total}s\n" \
  -X POST http://localhost:8856/v1/chat/completions \
  -H "Authorization: Bearer $TOK" \
  -H "content-type: application/json" \
  -d '{"model":"opus","messages":[{"role":"user","content":"say hi"}],"max_tokens":32,"stream":false}'
grep -iE '^(HTTP/|x-routed|x-complexity|retry-after)' /tmp/nc_cc.hdr 2>/dev/null
head -c 400 /tmp/nc_cc.json; echo

sep "最近 10 条真实请求落在哪个模型上（SQLite）"
DB=~/.nadirclaw/logs/requests.db
if [ -f "$DB" ]; then
  sqlite3 "$DB" ".headers on" \
    "SELECT datetime(ts,'unixepoch','localtime') AS t, tier, selected_model, fallback_reason, status
     FROM requests ORDER BY ts DESC LIMIT 10;" 2>&1
else
  echo "no $DB"
fi

sep "最近 200 行服务日志"
LOG=~/.nadirclaw/logs/server.log
if [ -f "$LOG" ]; then
  tail -n 200 "$LOG"
elif systemctl list-units --type=service 2>/dev/null | grep -q nadirclaw; then
  journalctl -u nadirclaw --no-pager -n 200
else
  echo "no server.log and no systemd unit — 你用什么启动的？"
fi

sep "DONE"
