"""Anthropic Messages API compatibility layer for NadirClaw.

Provides /v1/messages endpoint that accepts Anthropic SDK format requests,
routes them through NadirClaw's smart routing, and returns responses in
Anthropic format.

For providers with Anthropic-compatible endpoints (e.g. third-party proxies),
the original Anthropic request body is forwarded directly — no format
conversion needed. This avoids the double-conversion overhead and
format-compatibility issues that arise from Anthropic→OpenAI→Anthropic.

For non-Anthropic providers (vLLM, Ollama, OpenAI), the request is converted
to OpenAI format and sent through LiteLLM.

Uses the proven "fake streaming" approach: waits for complete response from
the upstream model, then emits Anthropic SSE events.
"""

import json
import logging
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.anthropic_api")

router = APIRouter()

# Same limits as /v1/chat/completions (server.py)
_MAX_CONTENT_LENGTH = 1_000_000  # 1 MB
_MAX_ANTHROPIC_TOKENS = 128_000  # Cap for max_tokens parameter

# Provider env-var mapping for Anthropic-compatible endpoints.
# Format: provider_name → (api_base_env_var, api_key_env_var)
_ANTHROPIC_COMPAT_PROVIDERS: Dict[str, Tuple[str, str]] = {
    "zai": ("ZAI_API_BASE", "ZAI_API_KEY"),
    "kimi": ("KIMI_API_BASE", "KIMI_API_KEY"),
    "minimax": ("MINIMAX_API_BASE", "MINIMAX_API_KEY"),
    "anthropic": ("ANTHROPIC_API_BASE", "ANTHROPIC_API_KEY"),
}

# Providers that serve Anthropic-native endpoints — no format conversion needed.
_ANTHROPIC_NATIVE_PROVIDERS = set(_ANTHROPIC_COMPAT_PROVIDERS)


def get_anthropic_compat_endpoint(
    provider: str,
) -> Optional[Tuple[str, str]]:
    """Return (api_base, api_key) if the provider has an Anthropic-compatible endpoint.

    Reads from .env file first (via dotenv_values) to bypass process environment
    overrides — e.g. Claude Code injects ANTHROPIC_API_KEY=local into the shell,
    which would shadow the real key in the .env file.
    Falls back to os.getenv for any key not present in the file.
    """
    if provider not in _ANTHROPIC_COMPAT_PROVIDERS:
        return None
    base_env, key_env = _ANTHROPIC_COMPAT_PROVIDERS[provider]

    # Read from .env file first — file values take priority over process env
    # to avoid Claude Code's ANTHROPIC_API_KEY=local override.
    from dotenv import dotenv_values
    _env_file = Path.home() / ".nadirclaw" / ".env"
    file_vals = dotenv_values(_env_file) if _env_file.exists() else {}

    api_base = file_vals.get(base_env, "") or os.getenv(base_env, "")
    api_key = file_vals.get(key_env, "") or os.getenv(key_env, "")

    if api_key in ("local", "dummy", "sk-placeholder", ""):
        return None

    if not api_base or not api_key:
        return None
    api_base = api_base.rstrip("/")
    return api_base, api_key


# Anthropic built-in tools that need conversion for other providers
_BUILTIN_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for up-to-date information. Returns search results with titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    "computer": {
        "name": "computer",
        "description": "Control a computer screen (click, type, screenshot, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action to perform"},
                "coordinate": {"type": "array", "items": {"type": "integer"}, "description": "x, y coordinates"},
                "text": {"type": "string", "description": "Text to type"},
            },
            "required": ["action"],
        },
    },
    "str_replace_based_edit": {
        "name": "str_replace_based_edit",
        "description": "Edit a file by replacing text. Commands: view, create, str_replace, insert.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command: view, create, str_replace, insert"},
                "path": {"type": "string", "description": "File path"},
                "file_text": {"type": "string", "description": "Full file content for create"},
                "old_str": {"type": "string", "description": "Text to replace"},
                "new_str": {"type": "string", "description": "Replacement text"},
                "insert_line": {"type": "integer", "description": "Line number for insert"},
                "new_str_to_insert": {"type": "string", "description": "Text to insert"},
            },
            "required": ["command", "path"],
        },
    },
}


def _convert_builtin_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic built-in tools to standard function tools.

    Anthropic built-in tools like web_search use special types
    (e.g. "web_search_20250305") that non-Anthropic providers don't understand.
    Convert them to standard function tools so GLM/Kimi/MiniMax can call them.
    """
    tool_type = tool.get("type", "")

    # Already a standard function tool
    if tool_type in ("", "function", "custom"):
        return tool

    # Check if it's a known built-in tool
    tool_name = tool.get("name", "")
    if tool_name in _BUILTIN_TOOL_SCHEMAS:
        converted = {
            "type": "custom",
            "name": tool_name,
            **_BUILTIN_TOOL_SCHEMAS[tool_name],
        }
        logger.debug("Converted built-in tool: %s (%s → custom)", tool_name, tool_type)
        return converted

    # Unknown built-in tool — convert to generic function tool
    if "_" in tool_type or tool_type not in ("", "function", "custom"):
        converted = {
            "type": "custom",
            "name": tool_name,
            "description": f"Built-in tool: {tool_name}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": f"Input for {tool_name}"},
                },
            },
        }
        logger.debug("Converted unknown built-in tool: %s (%s → custom)", tool_name, tool_type)
        return converted

    return tool


def _get_ppchat_domains() -> list[str]:
    """Get alternate ppchat domains for 429 failover."""
    raw = os.getenv("NADIRCLAW_PPCHAT_DOMAINS", "")
    if not raw:
        return []
    prefix = "https://"
    return [d if d.startswith(prefix) else prefix + d for d in raw.split(",") if d.strip()]


async def call_anthropic_direct(
    api_base: str,
    api_key: str,
    model: str,
    body: Dict[str, Any],
    provider: Optional[str] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Call an Anthropic-compatible endpoint directly, no format conversion.

    Forwards the original Anthropic request body verbatim and returns the
    raw Anthropic response.  This avoids the Anthropic→OpenAI→Anthropic
    double-conversion that can cause format issues on some providers.

    On 429 (rate limit), retries with exponential backoff and rotates
    through alternate domains from NADIRCLAW_PPCHAT_DOMAINS.
    """
    from nadirclaw import minimax_recorder

    max_retries = int(os.getenv("NADIRCLAW_429_RETRIES", "3"))
    base_backoff = float(os.getenv("NADIRCLAW_429_BACKOFF", "15"))

    # Build candidate URLs: primary domain first, then alternates
    primary_url = f"{api_base}/v1/messages?beta=true"
    alt_domains = [d for d in _get_ppchat_domains() if d != api_base.rstrip("/")]
    candidate_urls = [primary_url] + [f"{d}/v1/messages?beta=true" for d in alt_domains]

    # Open MiniMax trace context (no-op for non-MiniMax or when disabled)
    trace_ctx = minimax_recorder.begin_turn(
        provider=provider or "",
        model=model,
        body=body,
    )

    # Build the Anthropic request body from the original
    ant_body: Dict[str, Any] = {"model": model, "max_tokens": body.get("max_tokens", 4096)}
    if body.get("messages"):
        ant_body["messages"] = _sanitize_messages(body["messages"])
    if body.get("system"):
        ant_body["system"] = body["system"]
    if body.get("tools"):
        tools = body["tools"]
        if provider and provider != "anthropic":
            tools = [_convert_builtin_tool(t) for t in tools]
        ant_body["tools"] = tools
    if body.get("tool_choice"):
        ant_body["tool_choice"] = body["tool_choice"]
    if body.get("thinking"):
        ant_body["thinking"] = body["thinking"]
    if body.get("temperature") is not None:
        ant_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        ant_body["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        ant_body["stop_sequences"] = body["stop_sequences"]
    if body.get("metadata"):
        ant_body["metadata"] = body["metadata"]
    if body.get("output_config"):
        ant_body["output_config"] = body["output_config"]

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    beta = body.get("anthropic_beta", "")
    if beta:
        headers["anthropic-beta"] = beta

    logger.debug("Direct Anthropic call: model=%s url=%s beta=%s", model, primary_url, headers.get("anthropic-beta", "(none)")[:60])
    import pathlib
    _dump_path = pathlib.Path.home() / ".nadirclaw" / "debug_last_request.json"
    _dump_path.write_text(json.dumps(ant_body, default=str, ensure_ascii=False)[:100000])
    client_timeout = httpx.Timeout(timeout, connect=5.0)

    resp = None
    for attempt in range(max_retries + 1):
        url = candidate_urls[attempt % len(candidate_urls)]
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, headers=headers, json=ant_body)
        except Exception as exc:
            if attempt < max_retries:
                backoff = base_backoff * (2 ** attempt)
                logger.info(
                    "Retry %s (network error, attempt %d/%d, %.0fs): %s",
                    model, attempt + 1, max_retries, backoff, exc,
                )
                await asyncio.sleep(backoff)
                continue
            minimax_recorder.end_turn(trace_ctx, raw_response={}, error=str(exc))
            raise

        if resp.status_code < 400:
            raw_json = resp.json()
            minimax_recorder.end_turn(trace_ctx, raw_response=raw_json, error=None)
            if attempt > 0:
                logger.info("Retrying %s succeeded on attempt %d", model, attempt + 1)
            return raw_json

        # 429: retry with exponential backoff + domain rotation
        if resp.status_code == 429 and attempt < max_retries:
            backoff = base_backoff * (2 ** attempt)
            host = url.split("//")[1].split("/")[0]
            logger.warning(
                "429 on %s via %s (attempt %d/%d, wait %.0fs): %.80s",
                model, host, attempt + 1, max_retries, backoff, resp.text[:80],
            )
            await asyncio.sleep(backoff)
            continue

        # Non-retriable error (400, 401, 500, or final 429)
        break

    # If we get here, resp is set and status >= 400
    if resp is not None and resp.status_code >= 400:
        error_text = resp.text[:500]
        logger.warning(
            "Direct Anthropic call failed (%s): %s", resp.status_code, error_text,
        )
        minimax_recorder.end_turn(
            trace_ctx,
            raw_response={"status_code": resp.status_code},
            error=error_text,
        )
        from litellm.exceptions import (
            AuthenticationError as LiteLLMAuthError,
            BadRequestError as LiteLLMBadRequestError,
            InternalServerError as LiteLLMInternalServerError,
            RateLimitError as LiteLLMRateLimitError,
        )
        if resp.status_code == 400:
            raise LiteLLMBadRequestError(
                message=error_text, model=model, llm_provider="anthropic",
            )
        if resp.status_code == 401:
            raise LiteLLMAuthError(
                message=error_text, model=model, llm_provider="anthropic",
            )
        if resp.status_code == 429:
            raise LiteLLMRateLimitError(
                message=error_text, model=model, llm_provider="anthropic",
            )
        raise LiteLLMInternalServerError(
            message=error_text, model=model, llm_provider="anthropic",
        )


# Pattern to strip system-reminder tags from display text.
# Handles both closed (<system-reminder>...</system-reminder>) and
# unclosed tags (<system-reminder>...\n\nReal text after).
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>"   # closed tags
    r"|<system-reminder>.*?(?=\n\n\S|$)",       # unclosed: strip to blank-line boundary
    re.DOTALL,
)


def _clean_display_text(text: str) -> str:
    """Strip <system-reminder> blocks from text for log display."""
    if not text:
        return text
    cleaned = _SYSTEM_REMINDER_RE.sub("", text).strip()
    return cleaned if cleaned else ""


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix known provider quirks for Anthropic-compatible endpoints.

    - ZAI/GLM: tool_result needs 'id' field (SDK bug: ClaudeContentBlockToolResult.id)
    - ZAI/GLM: tool_use.input must be a dict
    """
    import copy
    msgs = copy.deepcopy(messages)
    tr_fixed = 0
    tu_fixed = 0
    for msg in msgs:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            # ZAI bug: tool_result needs an 'id' field
            if block.get("type") == "tool_result" and "id" not in block:
                block["id"] = f"toolr_{block.get('tool_use_id', 'unknown')}"
                tr_fixed += 1
            # Ensure tool_use.input is a dict
            if block.get("type") == "tool_use":
                if "id" not in block:
                    import hashlib
                    h = hashlib.md5(
                        f"{block.get('name','')}:{block.get('input','')}".encode()
                    ).hexdigest()[:24]
                    block["id"] = f"toolu_{h}"
                    tu_fixed += 1
                inp = block.get("input")
                if not isinstance(inp, dict):
                    block["input"] = {"value": inp} if inp is not None else {}
                    tu_fixed += 1
    if tr_fixed or tu_fixed:
        logger.info("_sanitize: tool_result id fixed=%d, tool_use fixed=%d", tr_fixed, tu_fixed)
    return msgs


def anthropic_response_to_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stats from a raw Anthropic response for logging/telemetry."""
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "stop_reason": data.get("stop_reason", ""),
        "model": data.get("model", ""),
    }


# ---------------------------------------------------------------------------
# Format conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def anthropic_to_openai_messages(
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Convert Anthropic Messages API format to OpenAI chat completion format."""
    result: List[Dict[str, Any]] = []

    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = [
                b.get("text", "")
                for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                result.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for i, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)
            else:
                result.append({"role": "assistant", "content": content})

        elif role == "user":
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            tc_texts = [
                                tc.get("text", "")
                                for tc in tool_content
                                if isinstance(tc, dict) and tc.get("type") == "text"
                            ]
                            tool_content = "\n".join(tc_texts)
                        result.append({
                            "role": "tool",
                            "content": str(tool_content),
                            "tool_call_id": block.get("tool_use_id", ""),
                        })
                if text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
            else:
                result.append({"role": "user", "content": content})
        else:
            result.append({"role": role, "content": str(content) if content else ""})

    return result


def anthropic_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function format.

    Note: Anthropic built-in tools (bash_20250124, text_editor_20250124, etc.)
    have no ``input_schema`` and are silently dropped. Only tools with
    ``type="custom"`` or an explicit ``input_schema`` are forwarded.
    """
    result = []
    dropped = []
    for tool in tools:
        if tool.get("type") == "custom" or "input_schema" in tool:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        else:
            dropped.append(tool.get("name", tool.get("type", "?")))
    if dropped:
        logger.warning(
            "Dropped %d Anthropic built-in tools (no input_schema): %s",
            len(dropped), dropped,
        )
    return result


# ---------------------------------------------------------------------------
# Format conversion: OpenAI → Anthropic
# ---------------------------------------------------------------------------

def openai_response_to_anthropic(
    response_data: Dict[str, Any],
    model: str,
    request_id: str,
) -> Dict[str, Any]:
    """Convert internal OpenAI-style response to Anthropic Messages API format."""
    content_blocks = []

    text = response_data.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in response_data.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", str(uuid.uuid4())),
            "name": func.get("name", ""),
            "input": input_data,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish = response_data.get("finish_reason", "stop")
    if finish == "tool_calls":
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": response_data.get("prompt_tokens", 0),
            "output_tokens": response_data.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Fake streaming: build SSE from complete response
# ---------------------------------------------------------------------------

def _build_anthropic_streaming_response(
    request_id: str,
    model: str,
    response_data: Dict[str, Any],
):
    """Build Anthropic-compatible SSE stream from a completed response.

    This is the proven "fake streaming" approach from 0.11.0:
    wait for the complete response, then emit all SSE events at once.
    """
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        content = response_data.get("content", "") or ""
        tool_calls = response_data.get("tool_calls", [])
        input_tokens = response_data.get("prompt_tokens", 0)
        output_tokens = response_data.get("completion_tokens", 0)
        finish = response_data.get("finish_reason", "stop")

        if finish == "tool_calls":
            stop_reason = "tool_use"
        elif finish == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        msg_id = f"msg_{request_id}"

        # Event: message_start
        yield {
            "event": "message_start",
            "data": json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }),
        }

        block_index = 0

        # Text content block
        if content:
            yield {
                "event": "content_block_start",
                "data": json.dumps({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                }),
            }
            yield {
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": content},
                }),
            }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({
                    "type": "content_block_stop",
                    "index": block_index,
                }),
            }
            block_index += 1

        # Tool use blocks
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                input_data = {}

            yield {
                "event": "content_block_start",
                "data": json.dumps({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": func.get("name", ""),
                        "input": {},
                    },
                }),
            }
            yield {
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(input_data),
                    },
                }),
            }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({
                    "type": "content_block_stop",
                    "index": block_index,
                }),
            }
            block_index += 1

        # Event: message_delta (stop reason + final usage)
        yield {
            "event": "message_delta",
            "data": json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }),
        }

        # Event: message_stop
        yield {
            "event": "message_stop",
            "data": json.dumps({"type": "message_stop"}),
        }

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


def build_anthropic_sse_events(
    request_id: str,
    model: str,
    response_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build Anthropic SSE event list from a completed OpenAI-format response.

    Public helper for testing and diagnostics. Returns a list of
    ``{"event": ..., "data": ...}`` dicts following Anthropic's SSE protocol.
    """
    content = response_data.get("content", "") or ""
    tool_calls = response_data.get("tool_calls", [])
    input_tokens = response_data.get("prompt_tokens", 0)
    output_tokens = response_data.get("completion_tokens", 0)
    finish = response_data.get("finish_reason", "stop")

    stop_reason = (
        "tool_use" if finish == "tool_calls"
        else ("max_tokens" if finish == "length" else "end_turn")
    )
    msg_id = f"msg_{request_id}"
    events: List[Dict[str, str]] = []

    events.append({
        "event": "message_start",
        "data": json.dumps({
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }),
    })

    block_idx = 0

    if content:
        events.append({"event": "content_block_start", "data": json.dumps({
            "type": "content_block_start", "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })})
        events.append({"event": "content_block_delta", "data": json.dumps({
            "type": "content_block_delta", "index": block_idx,
            "delta": {"type": "text_delta", "text": content},
        })})
        events.append({"event": "content_block_stop", "data": json.dumps({
            "type": "content_block_stop", "index": block_idx,
        })})
        block_idx += 1

    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {}

        events.append({"event": "content_block_start", "data": json.dumps({
            "type": "content_block_start", "index": block_idx,
            "content_block": {
                "type": "tool_use", "id": tc.get("id", str(uuid.uuid4())),
                "name": func.get("name", ""), "input": {},
            },
        })})
        events.append({"event": "content_block_delta", "data": json.dumps({
            "type": "content_block_delta", "index": block_idx,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(input_data)},
        })})
        events.append({"event": "content_block_stop", "data": json.dumps({
            "type": "content_block_stop", "index": block_idx,
        })})
        block_idx += 1

    events.append({"event": "message_delta", "data": json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })})
    events.append({"event": "message_stop", "data": json.dumps({
        "type": "message_stop",
    })})

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract text from the last user message in Anthropic format."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
            ]
            if parts:
                return "\n".join(parts)
    return ""


def _extract_display_prompt(messages: List[Dict[str, Any]]) -> str:
    """Extract user text for log display, skipping system-reminder-only messages."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "\n".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        elif isinstance(content, str):
            text = content
        else:
            continue
        cleaned = _SYSTEM_REMINDER_RE.sub("", text).strip()
        if cleaned:
            return cleaned[:2000]
    return ""


def _format_response_for_log(data: Dict[str, Any]) -> str:
    """Extract text from an Anthropic response for log display."""
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            parts.append(f"[tool:{block.get('name', '')}]")
    text = "\n".join(parts)
    return _clean_display_text(text)[:500]


def _extract_anthropic_text(data: Dict[str, Any]) -> str:
    """Extract concatenated text from an Anthropic response."""
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_anthropic_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool_calls in OpenAI-compatible format from an Anthropic response."""
    result = []
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            result.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
    return result


def _build_anthropic_streaming_response_from_raw(
    request_id: str,
    model: str,
    raw_response: Dict[str, Any],
):
    """Build SSE stream from a raw Anthropic response (direct path)."""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        usage = raw_response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        msg_id = raw_response.get("id", f"msg_{request_id}")

        yield {
            "event": "message_start",
            "data": json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }),
        }

        block_index = 0
        for block in raw_response.get("content", []):
            block_type = block.get("type", "text")
            yield {
                "event": "content_block_start",
                "data": json.dumps({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": block,
                }),
            }
            if block_type == "text":
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": block.get("text", "")},
                    }),
                }
            elif block_type == "tool_use":
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input", {})),
                        },
                    }),
                }
            yield {
                "event": "content_block_stop",
                "data": json.dumps({"type": "content_block_stop", "index": block_index}),
            }
            block_index += 1

        yield {
            "event": "message_delta",
            "data": json.dumps({
                "type": "message_delta",
                "delta": {
                    "stop_reason": raw_response.get("stop_reason", "end_turn"),
                    "stop_sequence": raw_response.get("stop_sequence"),
                },
                "usage": {"output_tokens": output_tokens},
            }),
        }
        yield {
            "event": "message_stop",
            "data": json.dumps({"type": "message_stop"}),
        }

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/v1/messages")
async def anthropic_messages(raw_request: Request):
    """Anthropic Messages API compatibility endpoint.

    Two call paths based on target provider:
    - Path A (Anthropic-compatible): Direct httpx call, original body forwarded verbatim.
      No format conversion needed — avoids Anthropic→OpenAI→Anthropic double-conversion.
    - Path B (non-Anthropic): Convert to OpenAI format, call through LiteLLM.

    Both paths share the same routing pipeline for model selection.

    NOTE: Internally, all calls are made with stream=False (non-streaming). The
    upstream provider returns the complete response, which is then wrapped into
    Anthropic SSE events for fake streaming when the client requested stream=True.
    This avoids the complexity of real streaming through format conversion layers.
    """
    from nadirclaw.server import (
        _call_with_fallback,
        _dispatch_model,
        _extract_request_metadata,
        _log_request,
        _smart_route_full,
        ChatMessage,
        ChatCompletionRequest,
        UserSession,
    )

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        body = await raw_request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate auth — support both Bearer token and x-api-key
    auth_header = raw_request.headers.get("authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    x_api_key = raw_request.headers.get("x-api-key", "")
    effective_token = token or x_api_key
    if settings.AUTH_TOKEN and effective_token != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # --- Rate limiting (parity with /v1/chat/completions) ---
    from nadirclaw.server import _rate_limiter
    user_id = effective_token or "anonymous"
    retry_after = _rate_limiter.check(user_id)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # Forward anthropic-beta header from Claude Code into body
    ant_beta_header = raw_request.headers.get("anthropic-beta", "")
    if ant_beta_header and not body.get("anthropic_beta"):
        body["anthropic_beta"] = ant_beta_header

    # Extract Anthropic fields
    ant_model = body.get("model", "")
    # ALL claude-* models → "auto" for smart routing (proven approach from 0.11.0)
    if ant_model.startswith("claude-"):
        ant_model = "auto"
    ant_stream = body.get("stream", False)
    ant_max_tokens = body.get("max_tokens", 4096)
    ant_tools = body.get("tools", [])

    # Cap max_tokens to prevent abuse
    if ant_max_tokens > _MAX_ANTHROPIC_TOKENS:
        logger.warning(
            "max_tokens=%d exceeds cap, clamping to %d",
            ant_max_tokens, _MAX_ANTHROPIC_TOKENS,
        )
        ant_max_tokens = _MAX_ANTHROPIC_TOKENS

    prompt_text = _extract_last_user_text(body.get("messages", []))
    display_prompt = _extract_display_prompt(body.get("messages", []))

    # Convert to OpenAI format
    openai_messages = anthropic_to_openai_messages(
        body.get("messages", []),
        body.get("system"),
    )
    openai_tools = anthropic_tools_to_openai(ant_tools) if ant_tools else []

    # --- Input size validation (parity with /v1/chat/completions) ---
    total_content_len = sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str)
        else len(json.dumps(m.get("content", "")))
        for m in openai_messages
    )
    if total_content_len > _MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Request content too large ({total_content_len:,} chars). "
                   f"Maximum is {_MAX_CONTENT_LENGTH:,} chars.",
        )

    # Build ChatCompletionRequest
    chat_messages = []
    for m in openai_messages:
        kwargs: Dict[str, Any] = {"role": m["role"], "content": m.get("content")}
        if m["role"] == "tool":
            kwargs["tool_call_id"] = m.get("tool_call_id", "")
            kwargs["content"] = m.get("content", "")
        if m["role"] == "assistant" and "tool_calls" in m:
            kwargs["tool_calls"] = m["tool_calls"]
        chat_messages.append(ChatMessage(**kwargs))

    req_data: Dict[str, Any] = {
        "messages": [{"role": m.role, "content": m.content, **(m.model_extra or {})} for m in chat_messages],
        "model": ant_model,
        "max_tokens": ant_max_tokens,
        "stream": False,  # Always non-streaming internally
    }
    if openai_tools:
        req_data["tools"] = openai_tools

    request = ChatCompletionRequest(**req_data)
    req_meta = _extract_request_metadata(request)

    # --- Full routing pipeline (same as /v1/chat/completions) ---
    from nadirclaw.routing import (
        apply_routing_modifiers,
        get_session_cache,
        resolve_alias,
        resolve_profile,
        get_pool_for_model,
        select_from_pool,
    )

    profile = resolve_profile(request.model)

    if profile == "eco":
        selected_model = settings.SIMPLE_MODEL
        analysis_info: Dict[str, Any] = {
            "strategy": "profile:eco", "selected_model": selected_model,
            "tier": "simple", "confidence": 1.0, "complexity_score": 0,
        }
    elif profile == "premium":
        selected_model = settings.COMPLEX_MODEL
        analysis_info = {
            "strategy": "profile:premium", "selected_model": selected_model,
            "tier": "complex", "confidence": 1.0, "complexity_score": 0,
        }
    elif profile == "free":
        selected_model = settings.FREE_MODEL
        analysis_info = {
            "strategy": "profile:free", "selected_model": selected_model,
            "tier": "free", "confidence": 1.0, "complexity_score": 0,
        }
    elif profile == "reasoning":
        selected_model = settings.REASONING_MODEL
        analysis_info = {
            "strategy": "profile:reasoning", "selected_model": selected_model,
            "tier": "reasoning", "confidence": 1.0, "complexity_score": 0,
        }
    elif request.model and request.model != "auto" and profile is None:
        resolved = resolve_alias(request.model)
        if resolved:
            selected_model = resolved
            analysis_info = {
                "strategy": "alias", "selected_model": selected_model,
                "alias_from": request.model, "tier": "direct",
                "confidence": 1.0, "complexity_score": 0,
            }
        else:
            selected_model = request.model
            analysis_info = {
                "strategy": "direct", "selected_model": selected_model,
                "tier": "direct", "confidence": 1.0, "complexity_score": 0,
            }
    else:
        # Smart routing (auto / claude-* → auto)
        session_cache = get_session_cache()
        cached = session_cache.get(request.messages)
        if cached:
            cached_model, cached_tier = cached
            selected_model = cached_model
            analysis_info = {
                "strategy": "session-cache", "selected_model": selected_model,
                "tier": cached_tier, "confidence": 1.0, "complexity_score": 0,
            }
            selected_model, final_tier, routing_info = apply_routing_modifiers(
                base_model=selected_model, base_tier=cached_tier,
                request_meta=req_meta, messages=request.messages,
                simple_model=settings.SIMPLE_MODEL, complex_model=settings.COMPLEX_MODEL,
                reasoning_model=settings.REASONING_MODEL, free_model=settings.FREE_MODEL,
                sonnet_model=settings.SONNET_MODEL,
                explore_model=settings.EXPLORE_MODEL, subagent_model=settings.SUBAGENT_MODEL,
                review_model=settings.REVIEW_MODEL,
            )
            if final_tier != cached_tier:
                analysis_info["tier"] = final_tier
                analysis_info["selected_model"] = selected_model
                analysis_info["routing_modifiers"] = routing_info
        else:
            selected_model, analysis_info = await _smart_route_full(
                request.messages, UserSession({"id": "anthropic_api"})
            )
            selected_model, final_tier, routing_info = apply_routing_modifiers(
                base_model=selected_model, base_tier=analysis_info.get("tier", "simple"),
                request_meta=req_meta, messages=request.messages,
                simple_model=settings.SIMPLE_MODEL, complex_model=settings.COMPLEX_MODEL,
                reasoning_model=settings.REASONING_MODEL, free_model=settings.FREE_MODEL,
                sonnet_model=settings.SONNET_MODEL,
                explore_model=settings.EXPLORE_MODEL, subagent_model=settings.SUBAGENT_MODEL,
                review_model=settings.REVIEW_MODEL,
            )
            analysis_info["tier"] = final_tier
            analysis_info["selected_model"] = selected_model
            analysis_info["routing_modifiers"] = routing_info
            session_cache.put(request.messages, selected_model, final_tier)

    # Pool selection — skip for capability-critical tiers
    _tier = analysis_info.get("tier", "")
    if _tier not in ("sonnet", "reasoning", "review", "long_context"):
        pool_name = get_pool_for_model(selected_model)
        if pool_name:
            pool_model = select_from_pool(pool_name)
            if pool_model:
                logger.info("Pool %s: %s → %s", pool_name, selected_model, pool_model)
                selected_model = pool_model

    # --- Dedup-only compression for old messages ---
    if settings.CONTEXT_COMPRESSION and len(body.get("messages", [])) > settings.COMPRESS_MIN_MESSAGES:
        from nadirclaw.compress import dedup_old_messages
        body["messages"] = dedup_old_messages(body["messages"])

    # Call model — two paths based on provider type
    from nadirclaw.credentials import detect_provider
    provider = detect_provider(selected_model)

    # Build fallback chain for this tier
    tier = analysis_info.get("tier", "simple")
    fallback_chain = settings.get_tier_fallback_chain(tier)
    # Remove the primary model from the chain
    fallback_chain = [m for m in fallback_chain if m != selected_model]

    try:
        from nadirclaw.telemetry import record_llm_call, trace_span

        with trace_span("anthropic_messages", {"nadirclaw.tier": tier}) as span:
            # Try direct Anthropic call first, then fallback chain
            raw_response = None
            fallback_from = None
            fallback_reasons: list[dict[str, str]] = []
            final_model = selected_model

            # Build candidate list: primary + fallback chain
            # Resolve pool members once, deduplicate to avoid cascade
            candidates = [selected_model]
            seen = {selected_model}
            for c in fallback_chain:
                pool_name = get_pool_for_model(c)
                if pool_name:
                    pool_model = select_from_pool(pool_name)
                    if pool_model and pool_model not in seen:
                        logger.info("Pool fallback %s: %s → %s", pool_name, c, pool_model)
                        candidates.append(pool_model)
                        seen.add(pool_model)
                elif c not in seen:
                    candidates.append(c)
                    seen.add(c)

            for candidate_model in candidates:
                candidate_provider = detect_provider(candidate_model)
                candidate_endpoint = (
                    get_anthropic_compat_endpoint(candidate_provider)
                    if candidate_provider else None
                )

                if candidate_endpoint:
                    # Path A: Direct Anthropic call
                    try:
                        api_base, api_key = candidate_endpoint
                        raw_response = await call_anthropic_direct(
                            api_base=api_base,
                            api_key=api_key,
                            model=candidate_model,
                            body=body,
                            provider=candidate_provider,
                        )
                        final_model = candidate_model
                        if candidate_model != selected_model:
                            fallback_from = selected_model
                        break
                    except Exception as e:
                        err_msg = str(e)[:200]
                        logger.warning(
                            "Direct Anthropic call failed for %s: %s — trying next",
                            candidate_model, err_msg,
                        )
                        fallback_reasons.append({
                            "model": candidate_model,
                            "reason": err_msg,
                            "path": "direct_anthropic",
                        })
                        continue
                else:
                    # Path B: Convert to OpenAI and use LiteLLM (single try).
                    # Use _dispatch_model directly — one try per candidate.
                    # Do NOT use _call_with_fallback here; it has its own
                    # internal chain that would consume all LiteLLM candidates
                    # and bypass the outer loop's direct-Anthropic path.
                    try:
                        logger.info(
                            "LiteLLM call for %s (provider=%s)",
                            candidate_model, candidate_provider,
                        )
                        response_data = await _dispatch_model(
                            candidate_model, request, candidate_provider,
                        )
                        final_model = candidate_model
                        if final_model != selected_model:
                            fallback_from = selected_model
                        # response_data is in OpenAI format — convert to Anthropic for return
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        stats = {
                            "prompt_tokens": response_data.get("prompt_tokens", 0),
                            "completion_tokens": response_data.get("completion_tokens", 0),
                        }
                        record_llm_call(
                            span, model=final_model, provider=candidate_provider,
                            prompt_tokens=stats["prompt_tokens"],
                            completion_tokens=stats["completion_tokens"],
                            tier=tier, latency_ms=elapsed_ms,
                        )
                        _log_request({
                            "type": "anthropic_messages",
                            "request_id": request_id,
                            "prompt": display_prompt,
                            "response": _clean_display_text(response_data.get("content") or "")[:500],
                            "selected_model": final_model,
                            "tier": tier,
                            "fallback_used": fallback_from,
                            "fallback_reasons": fallback_reasons or None,
                            "total_latency_ms": elapsed_ms,
                            **stats,
                            "status": "ok",
                            "call_path": "litellm_fallback",
                            **req_meta,
                        })
                        display_model = body.get("model", "") or final_model
                        if ant_stream:
                            return _build_anthropic_streaming_response(
                                request_id, display_model, response_data,
                            )
                        return JSONResponse(
                            content=openai_response_to_anthropic(
                                response_data, display_model, request_id,
                            ),
                            headers={"content-type": "application/json"},
                        )
                    except Exception as e:
                        err_msg = str(e)[:200]
                        logger.warning(
                            "LiteLLM call failed for %s: %s — trying next",
                            candidate_model, err_msg,
                        )
                        fallback_reasons.append({
                            "model": candidate_model,
                            "reason": err_msg,
                            "path": "litellm",
                        })
                        continue

            if raw_response is None:
                raise RuntimeError(f"All models failed in fallback chain for tier={tier}")

            # --- Success via direct Anthropic path ---
            stats = anthropic_response_to_stats(raw_response)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Some providers (GLM, Kimi) report inaccurate input_tokens.
            # Use our own estimate when upstream reports suspiciously low values.
            reported_pt = stats["prompt_tokens"]
            # Estimate from raw body: system + messages + tools
            est_chars = len(json.dumps(body.get("system", "")))
            est_chars += len(json.dumps(body.get("messages", [])))
            est_chars += len(json.dumps(body.get("tools", [])))
            estimated_pt = est_chars // 4
            prompt_tokens = max(reported_pt, estimated_pt)

            record_llm_call(
                span, model=final_model, provider=detect_provider(final_model),
                prompt_tokens=prompt_tokens,
                completion_tokens=stats["completion_tokens"],
                tier=tier, latency_ms=elapsed_ms,
            )

            _log_request({
                "type": "anthropic_messages",
                "request_id": request_id,
                "prompt": display_prompt,
                "response": _format_response_for_log(raw_response),
                "selected_model": final_model,
                "tier": tier,
                "fallback_used": fallback_from,
                "fallback_reasons": fallback_reasons or None,
                "total_latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": stats["completion_tokens"],
                "status": "ok",
                "call_path": "direct_anthropic",
                **req_meta,
            })

            display_model = body.get("model", "") or final_model
            if ant_stream:
                return _build_anthropic_streaming_response_from_raw(
                    request_id, display_model, raw_response,
                )
            return JSONResponse(
                content=raw_response, headers={"content-type": "application/json"},
            )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error("Anthropic messages error: %s", e, exc_info=True)
        err_extra = req_meta if "req_meta" in dir() else {}
        _log_request({
            "type": "anthropic_messages",
            "request_id": request_id,
            "prompt": display_prompt,
            "selected_model": selected_model,
            "tier": analysis_info.get("tier", ""),
            "status": "error",
            "error": str(e)[:500],
            "fallback_reasons": fallback_reasons if "fallback_reasons" in dir() else None,
            "total_latency_ms": elapsed_ms,
            **err_extra,
        })
        raise HTTPException(status_code=500, detail=f"Internal error. Request ID: {request_id}")
