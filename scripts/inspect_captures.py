"""Summarize a captures.jsonl written by capture_proxy.py.

Usage:
    python scripts/inspect_captures.py captures.jsonl           # summary
    python scripts/inspect_captures.py captures.jsonl --full    # one block per request
    python scripts/inspect_captures.py captures.jsonl --id <id> # single request, raw JSON

No network calls. No secrets are printed — the JSONL was already written
with auth values redacted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warn: line {i} is not valid JSON: {e}", file=sys.stderr)


def _fmt_ms(n):
    if n is None:
        return "-"
    if n < 1000:
        return f"{n}ms"
    return f"{n/1000:.2f}s"


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return (s[: n - 1] + "…") if len(s) > n else s


def print_summary(records: list[dict]) -> None:
    print(f"{len(records)} captured request(s)\n")
    header = f"{'#':>3} {'time':19} {'method':6} {'path':28} {'status':6} {'model':24} {'ttfb':>7} {'total':>7} {'stream':6}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(records, 1):
        req = r.get("request") or {}
        resp = r.get("response") or {}
        body = req.get("body") or {}
        ms = body.get("messages_summary") or {}
        timing = r.get("timing_ms") or {}
        ts = (r.get("captured_at") or "")[:19].replace("T", " ")
        method = req.get("method", "?")
        path = _short(req.get("path", ""), 28)
        status = str(resp.get("status") if resp.get("status") is not None else "-")
        model = _short(ms.get("model") or "-", 24)
        stream = "yes" if ms.get("stream") else "no"
        ttfb = _fmt_ms(timing.get("send_to_first_byte"))
        total = _fmt_ms(timing.get("total"))
        print(f"{i:>3} {ts:19} {method:6} {path:28} {status:>6} {model:24} {ttfb:>7} {total:>7} {stream:6}")


def print_full(records: list[dict]) -> None:
    for i, r in enumerate(records, 1):
        req = r.get("request") or {}
        resp = r.get("response") or {}
        body = req.get("body") or {}
        ms = body.get("messages_summary") or {}
        timing = r.get("timing_ms") or {}
        print(f"\n=== #{i} id={r.get('id')}  {r.get('captured_at')} ===")
        print(f"  {req.get('method')} {req.get('path')}  ->  {req.get('target')}")
        print(f"  status={resp.get('status')}  error={resp.get('error')}")
        print(f"  timing: received->send={_fmt_ms(timing.get('received_to_send'))}  "
              f"send->first_byte={_fmt_ms(timing.get('send_to_first_byte'))}  "
              f"first_byte->done={_fmt_ms(timing.get('first_byte_to_done'))}  "
              f"total={_fmt_ms(timing.get('total'))}")
        if ms:
            print(f"  request body:")
            print(f"    model={ms.get('model')}  stream={ms.get('stream')}  "
                  f"max_tokens={ms.get('max_tokens')}  temperature={ms.get('temperature')}")
            print(f"    messages={ms.get('message_count')}  roles={ms.get('message_roles')}")
            print(f"    tools_defined={ms.get('tool_definitions')}  "
                  f"tool_use_blocks={ms.get('tool_use_blocks')}  "
                  f"tool_result_blocks={ms.get('tool_result_blocks')}  "
                  f"images={ms.get('image_blocks')}")
            print(f"    system_chars={ms.get('system_chars')}  text_total_chars={ms.get('text_total_chars')}")
            if ms.get("tool_names"):
                print(f"    tool_names={ms['tool_names']}")
            if ms.get("first_user_excerpt"):
                print(f"    first_user: {_short(ms['first_user_excerpt'], 200)}")
            if ms.get("last_user_excerpt"):
                print(f"    last_user:  {_short(ms['last_user_excerpt'], 200)}")
        rbody = resp.get("body")
        if rbody:
            print(f"  response body: {rbody.get('format')} bytes={rbody.get('byte_size')}"
                  f" stop_reason={rbody.get('stop_reason')}")
            if rbody.get("text_excerpt"):
                print(f"    text: {_short(rbody['text_excerpt'], 200)}")
        sse = resp.get("sse")
        if sse:
            print(f"  sse: chunks={sse.get('chunks')} bytes={sse.get('bytes')}")
            print(f"       events={sse.get('event_counts')}")
        usage = resp.get("usage")
        if usage:
            print(f"  usage: {usage}")
        headers_r = req.get("headers") or {}
        interesting = {k: v for k, v in headers_r.items()
                       if k.lower() in {"authorization", "x-api-key", "anthropic-version",
                                        "anthropic-beta", "user-agent", "content-type"}}
        if interesting:
            print(f"  notable request headers: {interesting}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--full", action="store_true", help="detailed block per request")
    ap.add_argument("--id", help="print raw JSON for a single record")
    ap.add_argument("--limit", type=int, default=50, help="cap number of records shown")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    records = list(_load(args.path))

    if args.id:
        matches = [r for r in records if r.get("id", "").startswith(args.id)]
        if not matches:
            print(f"no record matches id prefix {args.id}", file=sys.stderr)
            return 1
        print(json.dumps(matches[0], indent=2, ensure_ascii=False))
        return 0

    records = records[-args.limit:]
    if args.full:
        print_full(records)
    else:
        print_summary(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
