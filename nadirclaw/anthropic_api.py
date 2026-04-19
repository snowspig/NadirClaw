"""Anthropic Messages API compatibility layer for NadirClaw.

Provides /v1/messages endpoint that accepts Anthropic SDK format requests,
converts them to internal OpenAI format, routes through NadirClaw's smart
routing, and returns responses in Anthropic format.

This enables tools that use the Anthropic SDK (e.g., Claude Code) to work
with NadirClaw as a transparent proxy.

Uses the proven "fake streaming" approach: waits for complete response from
the upstream model, then emits Anthropic SSE events. This avoids the complex
and error-prone real-time OpenAI→Anthropic stream conversion.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw.anthropic_api")

router = APIRouter()


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
    """Convert Anthropic tool definitions to OpenAI function format."""
    result = []
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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/v1/messages")
async def anthropic_messages(raw_request: Request):
    """Anthropic Messages API compatibility endpoint.

    Uses the proven "fake streaming" approach from 0.11.0:
    - All claude-* models → "auto" for smart routing
    - Wait for complete response, then emit SSE events
    - Reuses the same routing pipeline as /v1/chat/completions
    """
    from nadirclaw.server import (
        _call_with_fallback,
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

    # Extract Anthropic fields
    ant_model = body.get("model", "")
    # ALL claude-* models → "auto" for smart routing (proven approach from 0.11.0)
    if ant_model.startswith("claude-"):
        ant_model = "auto"
    ant_stream = body.get("stream", False)
    ant_max_tokens = body.get("max_tokens", 4096)
    ant_tools = body.get("tools", [])

    prompt_text = _extract_last_user_text(body.get("messages", []))

    # Convert to OpenAI format
    openai_messages = anthropic_to_openai_messages(
        body.get("messages", []),
        body.get("system"),
    )
    openai_tools = anthropic_tools_to_openai(ant_tools) if ant_tools else []

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
                execution_model=settings.EXECUTION_MODEL, review_model=settings.REVIEW_MODEL,
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
                execution_model=settings.EXECUTION_MODEL, review_model=settings.REVIEW_MODEL,
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

    # Call model with fallback (non-streaming)
    from nadirclaw.credentials import detect_provider
    provider = detect_provider(selected_model)

    try:
        from nadirclaw.telemetry import record_llm_call, trace_span

        with trace_span("anthropic_messages", {"nadirclaw.tier": analysis_info.get("tier")}) as span:
            response_data, selected_model, analysis_info = await _call_with_fallback(
                selected_model, request, provider, analysis_info,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            record_llm_call(
                span, model=selected_model, provider=provider,
                prompt_tokens=response_data.get("prompt_tokens", 0),
                completion_tokens=response_data.get("completion_tokens", 0),
                tier=analysis_info.get("tier"), latency_ms=elapsed_ms,
            )
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error("Anthropic messages error: %s", e, exc_info=True)
        _log_request({
            "type": "anthropic_messages", "request_id": request_id,
            "status": "error", "error": str(e), "total_latency_ms": elapsed_ms,
        })
        raise HTTPException(status_code=500, detail=f"Internal error. Request ID: {request_id}")

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Log with fallback indicator
    _log_request({
        "type": "anthropic_messages",
        "request_id": request_id,
        "prompt": prompt_text[:2000],
        "selected_model": selected_model,
        "tier": analysis_info.get("tier"),
        "fallback_used": analysis_info.get("fallback_from"),
        "total_latency_ms": elapsed_ms,
        "prompt_tokens": response_data.get("prompt_tokens", 0),
        "completion_tokens": response_data.get("completion_tokens", 0),
        "status": "ok",
        **req_meta,
    })

    # Return in Anthropic format
    display_model = body.get("model", "") or selected_model

    if ant_stream:
        return _build_anthropic_streaming_response(request_id, display_model, response_data)

    return JSONResponse(
        content=openai_response_to_anthropic(response_data, display_model, request_id),
        headers={"content-type": "application/json"},
    )
