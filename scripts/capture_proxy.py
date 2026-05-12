"""Transparent capture proxy for Claude Code → Anthropic (or any upstream).

Run this locally, point Claude Code at it, and it will forward every
request to the real upstream while logging the full request/response
structure and timings to a JSONL file.

Quick start
-----------
1. pick an upstream (the real Anthropic API or whatever endpoint your
   Claude Code normally talks to), e.g. https://api.anthropic.com

2. run:

       python scripts/capture_proxy.py --upstream https://api.anthropic.com \
           --port 9100 --out captures.jsonl

3. point Claude Code at http://localhost:9100 instead of the real host.
   the simplest way for Claude Code:

       ANTHROPIC_BASE_URL=http://localhost:9100 claude

   (or set it in Claude Code's settings; the proxy passes the real
   Authorization / x-api-key / anthropic-version headers through untouched.)

4. use Claude Code normally. each request appends one JSON object to
   captures.jsonl with:

   - request: method, path, query, headers (auth values redacted),
     body parsed (for /v1/messages: model, message count, tool count,
     system-prompt length, first/last user turn excerpt, whether stream)
   - response: status, headers, timing breakdown (connect / first-byte /
     total), body excerpt, sse event count + first/last event types for
     streaming, upstream-reported token usage when present
   - timing: wall-clock timestamps at each stage

5. look at the JSONL, or use scripts/inspect_captures.py for a summary.

The proxy is fully streaming-transparent: SSE events are forwarded
byte-for-byte to Claude Code as they arrive, and the capture is built
up in parallel without adding latency to the critical path.

Never records auth values, API keys, or OAuth tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


logger = logging.getLogger("capture_proxy")

# Header names we never log raw. We record their presence + length so you
# can still confirm the client sent them.
_SENSITIVE_HEADERS = frozenset({
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "cookie",
    "set-cookie",
    "proxy-authorization",
})

# Hop-by-hop headers — not forwarded to the upstream, not returned to
# the client. See RFC 7230 section 6.1.
_HOP_BY_HOP = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
})


def _redact_headers(headers: dict[str, str]) -> dict[str, Any]:
    """Return a redacted, log-safe copy of the headers."""
    out: dict[str, Any] = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in _SENSITIVE_HEADERS:
            out[k] = {"present": True, "length": len(v)}
        else:
            out[k] = v
    return out


def _summarize_messages(body: dict[str, Any]) -> dict[str, Any]:
    """Extract structural info from an Anthropic /v1/messages body."""
    messages = body.get("messages") or []
    roles = [m.get("role", "?") for m in messages]
    # Count how many tool_use / tool_result blocks appear
    tool_use_blocks = 0
    tool_result_blocks = 0
    image_blocks = 0
    text_total_chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            text_total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "tool_use":
                    tool_use_blocks += 1
                elif btype == "tool_result":
                    tool_result_blocks += 1
                elif btype == "image":
                    image_blocks += 1
                elif btype == "text":
                    text_total_chars += len(block.get("text", "") or "")

    def _first_user_excerpt() -> str:
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c[:500]
                if isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "text":
                            return (b.get("text") or "")[:500]
        return ""

    def _last_user_excerpt() -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c[-500:]
                if isinstance(c, list):
                    for b in reversed(c):
                        if isinstance(b, dict) and b.get("type") == "text":
                            return (b.get("text") or "")[-500:]
        return ""

    system = body.get("system")
    if isinstance(system, list):
        system_len = sum(len((s or {}).get("text", "") or "") for s in system if isinstance(s, dict))
    else:
        system_len = len(system or "") if isinstance(system, str) else 0

    tools = body.get("tools") or []

    return {
        "model": body.get("model"),
        "stream": bool(body.get("stream")),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "message_count": len(messages),
        "message_roles": roles,
        "tool_definitions": len(tools),
        "tool_names": [t.get("name") for t in tools if isinstance(t, dict)][:20],
        "tool_use_blocks": tool_use_blocks,
        "tool_result_blocks": tool_result_blocks,
        "image_blocks": image_blocks,
        "text_total_chars": text_total_chars,
        "system_chars": system_len,
        "first_user_excerpt": _first_user_excerpt(),
        "last_user_excerpt": _last_user_excerpt(),
    }


def _summarize_request_body(raw: bytes, content_type: str, full_body: bool = False) -> dict[str, Any]:
    """Best-effort structural summary of the request body.

    When `full_body` is true, also include the raw body decoded as UTF-8
    (with replacement for undecodable bytes) under `raw_body`.
    """
    info: dict[str, Any] = {"byte_size": len(raw)}
    if full_body and raw:
        info["raw_body"] = raw.decode("utf-8", errors="replace")
    if not raw:
        return info
    if "json" not in (content_type or "").lower():
        info["format"] = "non-json"
        info["excerpt"] = raw[:400].decode("utf-8", errors="replace")
        return info
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        info["format"] = "invalid-json"
        info["parse_error"] = str(e)
        info["excerpt"] = raw[:400].decode("utf-8", errors="replace")
        return info
    info["format"] = "json"
    if isinstance(body, dict) and "messages" in body:
        info["messages_summary"] = _summarize_messages(body)
    info["top_level_keys"] = list(body.keys()) if isinstance(body, dict) else []
    if full_body:
        info["body"] = body
    return info


def _parse_sse_event_types(chunk: bytes, counter: dict[str, int]) -> None:
    """Update counter with any `event: <name>` lines found in `chunk`."""
    for line in chunk.splitlines():
        if line.startswith(b"event:"):
            name = line[6:].strip().decode("ascii", errors="replace")
            counter[name] = counter.get(name, 0) + 1


def _maybe_parse_usage_from_sse(chunk: bytes, usage_acc: dict[str, Any]) -> None:
    """Try to extract Anthropic `usage` from message_start / message_delta events."""
    for line in chunk.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            payload = json.loads(data)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        etype = payload.get("type", "")
        if etype == "message_start":
            msg = payload.get("message") or {}
            usage = msg.get("usage") or {}
            if usage:
                usage_acc["input_tokens"] = usage.get("input_tokens")
                usage_acc["cache_creation_input_tokens"] = usage.get("cache_creation_input_tokens")
                usage_acc["cache_read_input_tokens"] = usage.get("cache_read_input_tokens")
        elif etype == "message_delta":
            usage = payload.get("usage") or {}
            if "output_tokens" in usage:
                usage_acc["output_tokens"] = usage["output_tokens"]


class CaptureWriter:
    """Append-only JSONL writer. One line per captured request."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        # Open lazily in the event loop; create parent directory now.
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def build_app(
    upstream: str, out_path: Path, max_body_bytes: int, full_body: bool = False,
) -> FastAPI:
    app = FastAPI(title="capture-proxy")
    writer = CaptureWriter(out_path)
    upstream_url = upstream.rstrip("/")

    # Shared client — warm connection pool, sane timeouts, no buffering.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    @app.get("/__capture_status")
    async def status() -> dict[str, Any]:
        return {
            "upstream": upstream_url,
            "out": str(out_path),
            "out_size_bytes": out_path.stat().st_size if out_path.exists() else 0,
        }

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def catch_all(full_path: str, request: Request) -> Any:
        req_id = uuid.uuid4().hex[:12]
        t_received = time.time()
        raw = await request.body()
        if len(raw) > max_body_bytes:
            return JSONResponse(
                {"error": f"body exceeds max_body_bytes={max_body_bytes}"},
                status_code=413,
            )

        # Build the forwarded URL + headers.
        target = f"{upstream_url}/{full_path.lstrip('/')}"
        if request.url.query:
            target += f"?{request.url.query}"
        fwd_headers: dict[str, str] = {}
        for k, v in request.headers.items():
            if k.lower() in _HOP_BY_HOP:
                continue
            fwd_headers[k] = v

        content_type = request.headers.get("content-type", "")
        req_summary = _summarize_request_body(raw, content_type, full_body=full_body)
        client_stream_requested = False
        if req_summary.get("format") == "json":
            ms = req_summary.get("messages_summary") or {}
            client_stream_requested = bool(ms.get("stream"))

        # ---- Non-streaming: buffer upstream, log, return whole body ----
        if not client_stream_requested:
            t_send = time.time()
            try:
                resp = await client.request(
                    request.method, target, headers=fwd_headers, content=raw,
                )
                body = resp.content
                t_done = time.time()
                record = _build_record(
                    req_id=req_id, request=request, target=target,
                    raw=raw, req_summary=req_summary,
                    status=resp.status_code,
                    resp_headers=dict(resp.headers),
                    resp_body=body, sse_events=None, usage=None,
                    t_received=t_received, t_send=t_send,
                    t_first_byte=t_done, t_done=t_done,
                    error=None, full_body=full_body,
                )
            except Exception as exc:
                t_done = time.time()
                record = _build_record(
                    req_id=req_id, request=request, target=target,
                    raw=raw, req_summary=req_summary,
                    status=None, resp_headers=None, resp_body=None,
                    sse_events=None, usage=None,
                    t_received=t_received, t_send=t_send,
                    t_first_byte=None, t_done=t_done,
                    error=repr(exc), full_body=full_body,
                )
                await writer.write(record)
                return JSONResponse({"error": str(exc)}, status_code=502)
            await writer.write(record)
            resp_h = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
            }
            return httpx_response_to_fastapi(resp.status_code, resp_h, body)

        # ---- Streaming: byte-for-byte passthrough + parallel capture ----
        t_send = time.time()
        upstream_req = client.build_request(
            request.method, target, headers=fwd_headers, content=raw,
        )

        # Open the upstream stream once. We keep the response object
        # around so we can both pipe its body to the client and pull
        # headers/status for the downstream StreamingResponse.
        try:
            probe = await client.send(upstream_req, stream=True)
        except Exception as exc:
            t_done = time.time()
            record = _build_record(
                req_id=req_id, request=request, target=target,
                raw=raw, req_summary=req_summary,
                status=None, resp_headers=None, resp_body=None,
                sse_events=None, usage=None,
                t_received=t_received, t_send=t_send,
                t_first_byte=None, t_done=t_done,
                error=repr(exc), full_body=full_body,
            )
            await writer.write(record)
            return JSONResponse({"error": str(exc)}, status_code=502)

        status_code_initial = probe.status_code
        headers_passthrough = {
            k: v for k, v in probe.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
        }

        async def gen():
            sse_counter: dict[str, int] = {}
            usage_acc: dict[str, Any] = {}
            first_byte_at: float | None = None
            chunks_total = 0
            bytes_total = 0
            collected = bytearray() if full_body else None
            error: str | None = None
            try:
                async for chunk in probe.aiter_raw():
                    if first_byte_at is None:
                        first_byte_at = time.time()
                    chunks_total += 1
                    bytes_total += len(chunk)
                    _parse_sse_event_types(chunk, sse_counter)
                    _maybe_parse_usage_from_sse(chunk, usage_acc)
                    if collected is not None:
                        collected.extend(chunk)
                    yield chunk
            except Exception as exc:
                error = repr(exc)
            finally:
                try:
                    await probe.aclose()
                except Exception:
                    pass
                t_done = time.time()
                record = _build_record(
                    req_id=req_id, request=request, target=target,
                    raw=raw, req_summary=req_summary,
                    status=status_code_initial,
                    resp_headers=dict(probe.headers),
                    resp_body=None,
                    sse_events={
                        "event_counts": sse_counter,
                        "chunks": chunks_total,
                        "bytes": bytes_total,
                    },
                    usage=usage_acc or None,
                    t_received=t_received, t_send=t_send,
                    t_first_byte=first_byte_at, t_done=t_done,
                    error=error, full_body=full_body,
                    sse_raw=bytes(collected) if collected is not None else None,
                )
                await writer.write(record)

        return StreamingResponse(
            gen(),
            status_code=status_code_initial,
            headers=headers_passthrough,
            media_type=probe.headers.get("content-type"),
        )

    return app


def httpx_response_to_fastapi(
    status: int, headers: dict[str, str], body: bytes
) -> "JSONResponse":
    """Pass the upstream response through to FastAPI's wire format."""
    # Avoid JSONResponse so binary bodies stay intact.
    from starlette.responses import Response
    return Response(content=body, status_code=status, headers=headers)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()


def _build_record(
    *,
    req_id: str,
    request: Request,
    target: str,
    raw: bytes,
    req_summary: dict[str, Any],
    status: int | None,
    resp_headers: dict[str, str] | None,
    resp_body: bytes | None,
    sse_events: dict[str, Any] | None,
    usage: dict[str, Any] | None,
    t_received: float,
    t_send: float,
    t_first_byte: float | None,
    t_done: float,
    error: str | None,
    full_body: bool = False,
    sse_raw: bytes | None = None,
) -> dict[str, Any]:
    resp_body_info: dict[str, Any] | None = None
    if resp_body is not None:
        resp_body_info = {
            "byte_size": len(resp_body),
        }
        ct = (resp_headers or {}).get("content-type", "")
        if "json" in ct.lower():
            try:
                parsed = json.loads(resp_body)
                resp_body_info["format"] = "json"
                if isinstance(parsed, dict):
                    resp_body_info["top_level_keys"] = list(parsed.keys())
                    usage = parsed.get("usage") or usage
                    resp_body_info["stop_reason"] = parsed.get("stop_reason")
                    # Extract small text excerpt for sanity-check.
                    content = parsed.get("content") or []
                    if isinstance(content, list):
                        text_parts = [
                            (b.get("text") or "")[:300]
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        if text_parts:
                            resp_body_info["text_excerpt"] = "".join(text_parts)[:600]
                if full_body:
                    resp_body_info["body"] = parsed
            except Exception:
                resp_body_info["format"] = "invalid-json"
                resp_body_info["excerpt"] = resp_body[:400].decode("utf-8", errors="replace")
                if full_body:
                    resp_body_info["raw_body"] = resp_body.decode("utf-8", errors="replace")
        else:
            resp_body_info["format"] = ct or "unknown"
            resp_body_info["excerpt"] = resp_body[:400].decode("utf-8", errors="replace")
            if full_body:
                resp_body_info["raw_body"] = resp_body.decode("utf-8", errors="replace")

    sse_info: dict[str, Any] | None = None
    if sse_events is not None:
        sse_info = dict(sse_events)
        if full_body and sse_raw is not None:
            sse_info["raw_stream"] = sse_raw.decode("utf-8", errors="replace")

    return {
        "id": req_id,
        "captured_at": _iso(t_received),
        "request": {
            "method": request.method,
            "path": "/" + request.url.path.lstrip("/"),
            "query": request.url.query,
            "target": target,
            "client": request.client.host if request.client else None,
            "headers": _redact_headers(dict(request.headers)),
            "body": req_summary,
        },
        "response": {
            "status": status,
            "headers": _redact_headers(resp_headers) if resp_headers else None,
            "body": resp_body_info,
            "sse": sse_info,
            "usage": usage,
            "error": error,
        },
        "timing_ms": {
            "received_to_send": int((t_send - t_received) * 1000),
            "send_to_first_byte": int((t_first_byte - t_send) * 1000) if t_first_byte else None,
            "first_byte_to_done": int((t_done - t_first_byte) * 1000) if t_first_byte else None,
            "total": int((t_done - t_received) * 1000),
        },
        "timestamps_iso": {
            "received": _iso(t_received),
            "sent_upstream": _iso(t_send),
            "first_byte": _iso(t_first_byte),
            "done": _iso(t_done),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Transparent capture proxy")
    ap.add_argument("--upstream", required=True, help="e.g. https://api.anthropic.com")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="captures.jsonl", help="JSONL output path")
    ap.add_argument(
        "--max-body-bytes", type=int, default=10 * 1024 * 1024,
        help="reject requests with bodies larger than this (default 10 MiB)",
    )
    ap.add_argument(
        "--full-body", action="store_true",
        help="record full request/response bodies verbatim (incl. raw SSE "
             "stream for streaming responses). Auth headers stay redacted, "
             "but any secrets that appear inside prompt/response text WILL "
             "be written to disk.",
    )
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    out_path = Path(args.out).resolve()
    app = build_app(args.upstream, out_path, args.max_body_bytes, full_body=args.full_body)
    logger.info("capture proxy ready: %s -> %s", f"http://{args.host}:{args.port}", args.upstream)
    logger.info("writing captures to: %s", out_path)
    if args.full_body:
        logger.warning("--full-body ON: request/response bodies will be written verbatim")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
