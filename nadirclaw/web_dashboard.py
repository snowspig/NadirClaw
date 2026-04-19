"""Web-based dashboard for NadirClaw.

Serves a single-page HTML dashboard at /dashboard that shows:
- Real-time routing stats (requests, tier distribution)
- Cost tracking and savings
- Model usage breakdown
- Recent request log

Auto-refreshes every 5 seconds via fetch().
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from nadirclaw.auth import UserSession, validate_local_auth
from nadirclaw.settings import settings

router = APIRouter()


def _load_recent_logs(limit: int = 200) -> List[Dict[str, Any]]:
    """Load recent log entries."""
    log_path = settings.LOG_DIR / "requests.jsonl"
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().split("\n")
    recent = lines[-limit:] if len(lines) > limit else lines
    entries = []
    for line in reversed(recent):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


@router.get("/dashboard/api/stats")
async def dashboard_stats(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """API endpoint for dashboard data."""
    from nadirclaw.budget import get_budget_tracker
    from nadirclaw.savings import calculate_actual_cost, get_model_cost

    entries = _load_recent_logs(500)
    completions = [e for e in entries if e.get("type") in ("completion", "anthropic_messages") and e.get("status") == "ok"]

    # Tier distribution
    tiers: Dict[str, int] = {}
    for e in completions:
        tier = e.get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1

    # Model usage
    models: Dict[str, Dict[str, Any]] = {}
    for e in completions:
        model = e.get("selected_model", "unknown")
        if model not in models:
            models[model] = {"requests": 0, "tokens": 0, "cost": 0.0, "avg_latency_ms": 0, "latencies": []}
        models[model]["requests"] += 1
        tokens = (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0)
        models[model]["tokens"] += tokens
        cost = e.get("cost", 0) or 0
        models[model]["cost"] += cost
        lat = e.get("total_latency_ms", 0) or 0
        if lat > 0:
            models[model]["latencies"].append(lat)

    # Calculate avg latency
    for m in models.values():
        lats = m.pop("latencies")
        m["avg_latency_ms"] = round(sum(lats) / len(lats)) if lats else 0

    # Recent requests (last 20)
    recent = []
    for e in completions[:20]:
        prompt = (e.get("prompt", "") or "")[:80]
        response = (e.get("response_preview", "") or "")[:120]
        recent.append({
            "time": e.get("timestamp", ""),
            "model": e.get("selected_model", ""),
            "tier": e.get("tier", ""),
            "latency_ms": e.get("total_latency_ms", 0),
            "tokens": (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0),
            "cost": e.get("cost", 0),
            "prompt": prompt,
            "response": response,
            "fallback": e.get("fallback_used"),
            "tokens_saved": e.get("tokens_saved", 0) or 0,
        })

    # Budget
    budget = get_budget_tracker().get_status()

    # Fallback stats
    fallbacks = sum(1 for e in completions if e.get("fallback_used"))

    # Optimization stats
    total_tokens_saved = sum(e.get("tokens_saved", 0) or 0 for e in completions)
    total_original_tokens = sum(e.get("original_tokens", 0) or 0 for e in completions if e.get("original_tokens"))
    opt_savings_pct = (total_tokens_saved / max(total_original_tokens, 1) * 100) if total_original_tokens else 0
    optimized_requests = sum(1 for e in completions if e.get("optimization_mode") and e.get("optimization_mode") != "off")

    # Compression stats
    comp_stats = {}
    comp_config = {"enabled": False}
    try:
        from nadirclaw.compress import get_compression_stats, get_compression_config
        comp_stats = get_compression_stats()
        comp_config = get_compression_config()
    except ImportError:
        pass

    # Quota status - PPChat model suspension tracking
    quota_data = {}
    try:
        from nadirclaw.quota import get_quota_tracker, PPCHAT_MODELS
        quota = get_quota_tracker()
        paid_disabled = quota.is_paid_models_disabled()
        ppchat_suspended = "ppchat" in quota.get_suspended_providers()

        countdown_seconds = 0
        if paid_disabled or ppchat_suspended:
            from datetime import datetime, timezone, timedelta
            beijing_tz = timezone(timedelta(hours=8))
            now = datetime.now(beijing_tz)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            countdown_seconds = int((tomorrow - now).total_seconds())

        ppchat_models_status = {}
        for model in PPCHAT_MODELS:
            ppchat_models_status[model] = {
                "suspended": ppchat_suspended,
                "reason": "ppchat quota exhausted" if ppchat_suspended else "active",
            }

        quota_data = {
            "paid_models_disabled": paid_disabled,
            "quota_used": quota._quota_used if hasattr(quota, "_quota_used") else 0,
            "quota_limit": quota.daily_quota if hasattr(quota, "daily_quota") else 0,
            "countdown_seconds": countdown_seconds,
            "ppchat_suspended": ppchat_suspended,
            "suspended_providers": list(quota.get_suspended_providers()),
            "ppchat_models": ppchat_models_status,
        }
    except ImportError:
        pass

    return {
        "total_requests": len(completions),
        "tier_distribution": tiers,
        "model_usage": dict(sorted(models.items(), key=lambda x: x[1]["requests"], reverse=True)),
        "recent_requests": recent,
        "budget": budget,
        "fallback_count": fallbacks,
        "simple_model": settings.SIMPLE_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
        "optimization": {
            "total_tokens_saved": total_tokens_saved,
            "savings_pct": round(opt_savings_pct, 1),
            "optimized_requests": optimized_requests,
        },
        "compression": comp_stats,
        "compression_config": comp_config,
        "quota": quota_data,
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the web dashboard HTML."""
    return DASHBOARD_HTML


@router.post("/dashboard/api/ppchat/restore")
async def restore_ppchat(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Manually restore PPChat access after quota reset."""
    from nadirclaw.quota import get_quota_tracker

    quota = get_quota_tracker()
    if "ppchat" in quota.get_suspended_providers():
        quota._suspended_providers.discard("ppchat")
        quota._save_state()
        return {"status": "restored", "message": "PPChat access has been restored"}
    return {"status": "already_active", "message": "PPChat is not suspended"}


class CompressionToggleRequest(BaseModel):
    enabled: str = "false"


@router.post("/dashboard/api/compression")
async def toggle_compression(
    request: CompressionToggleRequest,
    current_user: UserSession = Depends(validate_local_auth),
):
    """Toggle context compression on/off at runtime."""
    try:
        import nadirclaw.compress as comp_module
        new_val = request.enabled.lower() in ("true", "1", "yes")
        comp_module._COMPRESS_ENABLED = new_val
        return {"status": "ok", "enabled": new_val}
    except Exception as e:
        return {"status": "error", "error": str(e)}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NadirClaw Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; }
.header { padding: 1.5rem 2rem; border-bottom: 1px solid #1e1e2e; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.3rem; font-weight: 700; color: #fff; }
.header h1 span { color: #a78bfa; }
.header .status { font-size: 0.8rem; color: #6b7280; }
.header .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #34d399; margin-right: 6px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; padding: 1.5rem 2rem; }
.card { background: #1a1a24; border-radius: 12px; padding: 1.25rem; }
.card-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.5rem; }
.card-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
.card-value.green { color: #34d399; }
.card-value.purple { color: #a78bfa; }
.card-value.amber { color: #fbbf24; }
.card-sub { font-size: 0.78rem; color: #6b7280; margin-top: 0.25rem; }
.section { padding: 0 2rem 1.5rem; }
.section-title { font-size: 0.85rem; font-weight: 600; color: #9ca3af; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.table-wrap { background: #1a1a24; border-radius: 12px; overflow: hidden; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th { text-align: left; padding: 0.75rem 1rem; color: #6b7280; font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid #252535; }
td { padding: 0.6rem 1rem; border-bottom: 1px solid #1e1e2e; }
tr:last-child td { border: none; }
.tier-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; }
.tier-simple { background: #064e3b; color: #34d399; }
.tier-complex { background: #4c1d95; color: #a78bfa; }
.tier-reasoning { background: #78350f; color: #fbbf24; }
.tier-direct { background: #1e293b; color: #94a3b8; }
.tier-free { background: #1e3a2f; color: #6ee7b7; }
.bar-wrap { display: flex; gap: 4px; height: 24px; border-radius: 6px; overflow: hidden; }
.bar-seg { transition: width 0.3s ease; }
.bar-simple { background: #34d399; }
.bar-complex { background: #a78bfa; }
.bar-reasoning { background: #fbbf24; }
.bar-other { background: #4b5563; }
.model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
.model-card { background: #1a1a24; border-radius: 12px; padding: 1rem 1.25rem; }
.model-name { font-size: 0.82rem; font-weight: 600; color: #fff; margin-bottom: 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; font-size: 0.75rem; color: #9ca3af; }
.model-stats span { color: #e0e0e0; font-weight: 500; }
.fallback-tag { color: #f87171; font-size: 0.7rem; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
.quota-banner { display: flex; align-items: center; gap: 1rem; padding: 1rem 2rem; background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); border-bottom: 1px solid #dc2626; }
.quota-banner .quota-icon { font-size: 1.5rem; }
.quota-banner .quota-info { flex: 1; }
.quota-banner .quota-title { font-weight: 600; font-size: 0.9rem; color: #fecaca; }
.quota-banner .quota-detail { font-size: 0.75rem; color: #fca5a5; margin-top: 0.2rem; }
.quota-banner .quota-countdown { font-family: 'SF Mono', 'Monaco', monospace; font-size: 1.2rem; font-weight: 700; color: #fff; background: #991b1b; padding: 0.5rem 1rem; border-radius: 8px; min-width: 100px; text-align: center; }
.quota-banner .restore-btn { background: #dc2626; color: #fff; border: none; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.75rem; }
.quota-banner .restore-btn:hover { background: #b91c1c; }
.ppchat-status { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.ppchat-model-tag { padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.75rem; }
.ppchat-model-tag.active { background: #064e3b; }
.ppchat-model-tag.suspended { background: #7f1d1d; }
.ppchat-model-tag .icon { margin-right: 4px; }
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } .two-col { grid-template-columns: 1fr; } }
.compression-card { background: rgba(26,26,36,0.9); border-radius: 8px; padding: 0.8rem 1rem; border: 1px solid #2a2a3e; }
.compression-toggle { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; }
.compression-toggle > span { font-size: 0.82rem; color: #e0e0e0; }
.compression-toggle .comp-switch { position: relative; width: 40px; height: 22px; display: inline-block; cursor: pointer; }
.compression-toggle .comp-switch input { opacity: 0; width: 0; height: 0; position: absolute; }
.compression-toggle .comp-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: #3a3a4e; border-radius: 22px; transition: 0.3s; }
.compression-toggle .comp-slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 3px; bottom: 3px; background: #e0e0e0; border-radius: 50%; transition: 0.3s; }
.compression-toggle .comp-switch input:checked + .comp-slider { background: #a78bfa; }
.compression-toggle .comp-switch input:checked + .comp-slider:before { transform: translateX(18px); }
.compression-stats { display: flex; gap: 1.2rem; flex-wrap: wrap; }
.comp-stat { text-align: center; }
.comp-stat-value { font-size: 1rem; font-weight: 700; color: #a78bfa; font-family: 'Courier New', monospace; }
.comp-stat-label { font-size: 0.65rem; color: #6b7280; margin-top: 2px; }
.compression-desc { font-size: 0.72rem; color: #6b7280; margin-bottom: 0.5rem; }
@media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="header">
  <h1>Nadir<span>Claw</span> Dashboard</h1>
  <div class="status"><span class="dot"></span>Live &mdash; refreshing every 5s</div>
</div>

<div id="quota-banner" class="quota-banner" style="display: none;">
  <div class="quota-icon">&#128683;</div>
  <div class="quota-info">
    <div class="quota-title" id="quota-status">Paid Models Disabled</div>
    <div class="quota-detail" id="quota-usage">Daily quota exhausted. Using free models until midnight.</div>
  </div>
  <button class="restore-btn" onclick="restorePPChat()">Restore</button>
  <div class="quota-countdown" id="quota-countdown">--:--:--</div>
</div>

<div class="grid" id="stats-cards">
  <div class="card"><div class="card-label">Total Requests</div><div class="card-value" id="total-reqs">-</div></div>
  <div class="card"><div class="card-label">Today's Spend</div><div class="card-value green" id="daily-spend">-</div><div class="card-sub" id="daily-budget"></div></div>
  <div class="card"><div class="card-label">Monthly Spend</div><div class="card-value purple" id="monthly-spend">-</div><div class="card-sub" id="monthly-budget"></div></div>
  <div class="card"><div class="card-label">Fallbacks</div><div class="card-value amber" id="fallback-count">-</div><div class="card-sub">auto-recovered</div></div>
</div>

<div class="section">
  <div class="section-title">Routing Distribution</div>
  <div class="bar-wrap" id="tier-bar" style="margin-bottom: 0.5rem;"></div>
  <div id="tier-legend" style="font-size: 0.75rem; color: #9ca3af;"></div>
</div>

<div class="section" id="ppchat-models-section"></div>

<div class="section two-col">
  <div>
    <div class="section-title">Models</div>
    <div class="model-grid" id="model-grid"></div>
  </div>
  <div>
    <div class="section-title">Recent Requests</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Time</th><th>Model</th><th>Tier</th><th>Latency</th><th>Tokens</th><th>Response</th></tr></thead>
        <tbody id="recent-body"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title">上下文压缩</div>
  <div class="compression-card">
    <p class="compression-desc">截断旧的工具输出以减少 prompt 大小。保留最近约10轮对话。</p>
    <div class="compression-toggle">
      <span>压缩状态</span>
      <label class="comp-switch">
        <input type="checkbox" id="compression-toggle" onchange="toggleCompression(this.checked)">
        <span class="comp-slider"></span>
      </label>
    </div>
    <div class="compression-stats" id="compression-stats"></div>
  </div>
</div>

<script>
const TIER_COLORS = { simple: '#34d399', complex: '#a78bfa', reasoning: '#fbbf24', direct: '#94a3b8', free: '#6ee7b7' };
const TIER_CLASSES = { simple: 'tier-simple', complex: 'tier-complex', reasoning: 'tier-reasoning', direct: 'tier-direct', free: 'tier-free' };

async function refresh() {
  try {
    const res = await fetch('/dashboard/api/stats');
    const d = await res.json();

    document.getElementById('total-reqs').textContent = d.total_requests.toLocaleString();
    document.getElementById('daily-spend').textContent = '$' + (d.budget.daily_spend || 0).toFixed(4);
    document.getElementById('monthly-spend').textContent = '$' + (d.budget.monthly_spend || 0).toFixed(4);
    document.getElementById('fallback-count').textContent = d.fallback_count;

    if (d.budget.daily_budget) document.getElementById('daily-budget').textContent = 'of $' + d.budget.daily_budget.toFixed(2) + ' budget';
    if (d.budget.monthly_budget) document.getElementById('monthly-budget').textContent = 'of $' + d.budget.monthly_budget.toFixed(2) + ' budget';

    // Quota banner - show when paid models disabled or ppchat suspended
    const quota = d.quota || {};
    const anyDisabled = quota.paid_models_disabled || quota.ppchat_suspended;
    const banner = document.getElementById('quota-banner');
    if (anyDisabled) {
      banner.style.display = 'flex';
      let statusMsg = '';
      if (quota.paid_models_disabled) {
        statusMsg = '\u26D4 Daily quota exhausted - paid models disabled';
      } else if (quota.ppchat_suspended) {
        statusMsg = '\u274C PPChat quota exhausted - Opus/Sonnet/GPT suspended';
      }
      let secs = quota.countdown_seconds || 0;
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      const s = secs % 60;
      const countdown = String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
      document.getElementById('quota-status').textContent = statusMsg;
      document.getElementById('quota-countdown').textContent = countdown;
    } else {
      banner.style.display = 'none';
    }
    if (quota.quota_limit > 0) {
      const usagePct = ((quota.quota_used / quota.quota_limit) * 100).toFixed(1);
      document.getElementById('quota-usage').textContent = quota.quota_used.toLocaleString() + ' / ' + quota.quota_limit.toLocaleString() + ' (' + usagePct + '%)';
    }

    // PPChat models status
    const ppchatModels = quota.ppchat_models || {};
    const ppchatSection = document.getElementById('ppchat-models-section');
    if (ppchatSection && Object.keys(ppchatModels).length > 0) {
      let html = '<div class="section-title">PPChat Models Status</div><div class="ppchat-status">';
      for (const [model, status] of Object.entries(ppchatModels)) {
        const cls = status.suspended ? 'suspended' : 'active';
        const icon = status.suspended ? '\u274C' : '\u2705';
        const shortName = model.split('/').pop();
        html += '<div class="ppchat-model-tag ' + cls + '"><span class="icon">' + icon + '</span><span style="font-weight:500;color:#e0e0e0">' + shortName + '</span> <span style="color:#9ca3af;font-size:0.7rem;margin-left:0.5rem">' + status.reason + '</span></div>';
      }
      html += '</div>';
      ppchatSection.innerHTML = html;
    } else {
      ppchatSection.innerHTML = '';
    }

    // Tier bar
    const total = Object.values(d.tier_distribution).reduce((a,b) => a+b, 0) || 1;
    const bar = document.getElementById('tier-bar');
    const legend = document.getElementById('tier-legend');
    bar.innerHTML = '';
    legend.innerHTML = '';
    for (const [tier, count] of Object.entries(d.tier_distribution).sort((a,b) => b[1]-a[1])) {
      const pct = (count / total * 100);
      const seg = document.createElement('div');
      seg.className = 'bar-seg';
      seg.style.width = pct + '%';
      seg.style.background = TIER_COLORS[tier] || '#4b5563';
      bar.appendChild(seg);
      legend.innerHTML += '<span style="color:' + (TIER_COLORS[tier]||'#4b5563') + '">' + tier + ' ' + count + ' (' + pct.toFixed(0) + '%)</span>  ';
    }

    // Model cards
    const mg = document.getElementById('model-grid');
    mg.innerHTML = '';
    for (const [name, info] of Object.entries(d.model_usage)) {
      mg.innerHTML += '<div class="model-card"><div class="model-name">' + name + '</div><div class="model-stats">' +
        '<div>Requests <span>' + info.requests + '</span></div>' +
        '<div>Tokens <span>' + info.tokens.toLocaleString() + '</span></div>' +
        '<div>Cost <span>$' + info.cost.toFixed(4) + '</span></div>' +
        '<div>Avg latency <span>' + info.avg_latency_ms + 'ms</span></div>' +
        '</div></div>';
    }

    // Recent
    const rb = document.getElementById('recent-body');
    rb.innerHTML = '';
    for (const r of d.recent_requests) {
      const t = r.time ? new Date(r.time).toLocaleTimeString() : '-';
      const tc = TIER_CLASSES[r.tier] || 'tier-direct';
      const fb = r.fallback ? ' <span class="fallback-tag">⚡fallback</span>' : '';
      const resp = (r.response || '').replace(/</g,'&lt;').substring(0, 150) || (r.prompt||'').replace(/</g,'&lt;').substring(0, 80);
      rb.innerHTML += '<tr><td>' + t + '</td><td style="font-size:0.75rem">' + r.model + fb + '</td><td><span class="tier-badge ' + tc + '">' + r.tier + '</span></td><td>' + (r.latency_ms||0) + 'ms</td><td>' + r.tokens + '</td><td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + resp + '</td></tr>';
    }

    // Compression toggle and stats
    const compToggle = document.getElementById('compression-toggle');
    const compStatsEl = document.getElementById('compression-stats');
    if (d.compression_config) {
      compToggle.checked = d.compression_config.enabled;
    }
    if (d.compression) {
      const tokensSaved = (d.compression.total_tokens_before || 0) - (d.compression.total_tokens_after || 0);
      compStatsEl.innerHTML = '<div class="comp-stat"><div class="comp-stat-value">' + tokensSaved.toLocaleString() + '</div><div class="comp-stat-label">tokens saved</div></div>' +
        '<div class="comp-stat"><div class="comp-stat-value">' + (d.compression.total_requests_compressed || 0) + '</div><div class="comp-stat-label">requests processed</div></div>' +
        '<div class="comp-stat"><div class="comp-stat-value">' + (d.compression.total_deduped || 0) + '</div><div class="comp-stat-label">deduplicated</div></div>' +
        '<div class="comp-stat"><div class="comp-stat-value">' + (d.compression.total_compressed || 0) + '</div><div class="comp-stat-label">summarized</div></div>';
    }
  } catch(e) { console.error('Dashboard refresh error:', e); }
}

async function toggleCompression(enabled) {
  const newVal = enabled ? 'true' : 'false';
  try {
    const res = await fetch('/dashboard/api/compression', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: newVal })
    });
    const data = await res.json();
    if (data.status !== 'ok') {
      document.getElementById('compression-toggle').checked = !enabled;
    }
  } catch(e) {
    document.getElementById('compression-toggle').checked = !enabled;
  }
}

async function restorePPChat() {
  try {
    const res = await fetch('/dashboard/api/ppchat/restore', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'restored') refresh();
  } catch(e) {}
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""
