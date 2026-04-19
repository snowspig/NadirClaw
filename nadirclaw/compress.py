"""Context deduplication and compression for NadirClaw.

Compresses old conversation turns (>10 rounds ago) by:
1. Deduplicating consecutive identical/similar tool results
2. Summarizing large tool outputs (keep first N lines + metadata)
3. Preserving all tool_use messages (skeleton) with id/name/input intact

Recent messages (last 10 rounds) are always preserved intact.

Designed to reduce token usage for long agentic sessions (e.g., Claude Code)
where tool output can accumulate to hundreds of thousands of tokens.
"""

import logging
import os
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("nadirclaw.compress")

_COMPRESS_ENABLED = os.getenv("NADIRCLAW_CONTEXT_COMPRESSION", "false").lower() in ("true", "1", "yes")
_COMPRESS_MIN_MESSAGES = int(os.getenv("NADIRCLAW_COMPRESS_MIN_MESSAGES", "30"))
_COMPRESS_RECENT_WINDOW = int(os.getenv("NADIRCLAW_COMPRESS_RECENT_WINDOW", "20"))
_COMPRESS_TOOL_OUTPUT_MAX_LINES = int(os.getenv("NADIRCLAW_COMPRESS_TOOL_MAX_LINES", "10"))

_compression_stats: Dict[str, int] = {
    "total_requests_compressed": 0,
    "total_tokens_before": 0,
    "total_tokens_after": 0,
    "total_deduped": 0,
    "total_compressed": 0,
}


def is_compression_enabled() -> bool:
    return _COMPRESS_ENABLED


def get_compression_stats() -> Dict[str, int]:
    return dict(_compression_stats)


def get_compression_config() -> Dict[str, Any]:
    return {
        "enabled": _COMPRESS_ENABLED,
        "min_messages": _COMPRESS_MIN_MESSAGES,
        "recent_window": _COMPRESS_RECENT_WINDOW,
        "tool_output_max_lines": _COMPRESS_TOOL_OUTPUT_MAX_LINES,
    }


def _content_signature(content: Any) -> int:
    """Hash for deduplication — first 200 chars of string representation."""
    return hash(str(content)[:200])


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from content (handles string and list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        for ib in inner:
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                parts.append(ib.get("text", ""))
                    elif isinstance(inner, str):
                        parts.append(inner)
        return "\n".join(parts)
    return str(content)


def _is_tool_result_content(content: Any) -> bool:
    """Check if content contains tool_result blocks."""
    if isinstance(content, list):
        return any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content)
    return False


def _has_tool_use(content: Any) -> bool:
    """Check if content contains tool_use blocks."""
    if isinstance(content, list):
        return any(isinstance(c, dict) and c.get("type") == "tool_use" for c in content)
    return False


def _summarize_tool_result(content: Any, max_lines: int) -> Tuple[Any, bool]:
    """Compress large tool output, keeping first N lines + summary.

    Preserves the tool_result wrapper structure (type, tool_use_id, etc).
    Returns (compressed_content, was_compressed).
    """
    if not isinstance(content, list):
        # Simple string content
        text = content if isinstance(content, str) else str(content)
        lines = text.split("\n")
        if len(lines) <= max_lines:
            return content, False
        kept = "\n".join(lines[:max_lines])
        summary = f"\n[... {len(lines) - max_lines} lines omitted]"
        return f"{kept}{summary}", True

    # List of blocks — only compress tool_result blocks
    new_blocks = []
    compressed = False
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            new_blocks.append(block)
            continue

        result_content = block.get("content", "")
        tool_use_id = block.get("tool_use_id", "")

        if isinstance(result_content, str):
            lines = result_content.split("\n")
            if len(lines) <= max_lines:
                new_blocks.append(block)
                continue
            kept = "\n".join(lines[:max_lines])
            summary = f"\n[... {len(lines) - max_lines} lines omitted]"
            new_blocks.append({**block, "content": f"{kept}{summary}"})
            compressed = True
        elif isinstance(result_content, list):
            text_parts = []
            for rc in result_content:
                if isinstance(rc, dict) and rc.get("type") == "text":
                    text_parts.append(rc.get("text", ""))
            full_text = "\n".join(text_parts)
            lines = full_text.split("\n")
            if len(lines) <= max_lines:
                new_blocks.append(block)
                continue
            kept = "\n".join(lines[:max_lines])
            summary = f"\n[... {len(lines) - max_lines} lines omitted]"
            new_blocks.append({**block, "content": f"{kept}{summary}"})
            compressed = True
        else:
            new_blocks.append(block)

    return new_blocks, compressed


def _summarize_assistant_text(content: Any, max_lines: int) -> Tuple[Any, bool]:
    """Compress long assistant text messages, keeping first N lines."""
    text = _extract_text_from_content(content)
    if not text:
        return content, False
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return content, False
    kept = "\n".join(lines[:max_lines])
    summary = f"\n[... {len(lines) - max_lines} lines omitted]"
    if isinstance(content, str):
        return f"{kept}{summary}", True
    # If content was a list of blocks, reconstruct with compressed text
    return f"{kept}{summary}", True


def compress_messages(messages: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Compress old conversation turns by dedup + summarize tool output.

    Strategy:
    - Recent window (last N messages): always preserved intact
    - Old turns:
      1. Dedup consecutive identical tool results (skip duplicates)
      2. Summarize large tool_result content (keep first K lines)
      3. Summarize long assistant text without tool_calls (keep first K lines)
      4. Always preserve: system/developer/user messages, tool_use blocks,
         tool_call_id, name fields

    Args:
        messages: List of message dicts with role/content fields.

    Returns:
        (compressed_messages, stats_dict)
    """
    if len(messages) <= _COMPRESS_MIN_MESSAGES:
        return messages, {"skipped": True}

    compressed = []
    total_before = 0
    total_after = 0
    deduped_count = 0
    compressed_count = 0
    prev_signature = None
    prev_text = None

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        is_recent = i >= len(messages) - _COMPRESS_RECENT_WINDOW

        # Check for tool_calls (in content or model_extra)
        has_tool_calls = _has_tool_use(content)
        has_tool_calls_extra = "tool_calls" in msg

        # Always keep: recent, system/developer/user, messages with tool_calls
        if is_recent or role in ("system", "developer", "user") or has_tool_calls or has_tool_calls_extra:
            compressed.append(msg)
            total_before += len(str(content))
            total_after += len(str(content))
            prev_signature = None
            prev_text = None
            continue

        content_str = str(content)
        total_before += len(content_str)

        # --- Dedup: skip consecutive similar old content ---
        sig = _content_signature(content)
        text = _extract_text_from_content(content)

        if prev_text is not None and sig == prev_signature and len(content_str) > 100:
            # Exact same hash — likely identical tool output
            deduped_count += 1
            prev_signature = sig
            prev_text = text
            total_after += 0
            continue

        # Check prefix similarity for near-duplicates (e.g. same file read twice)
        if prev_text and len(text) > 200 and len(prev_text) > 200:
            if text[:100] == prev_text[:100]:
                deduped_count += 1
                prev_signature = sig
                prev_text = text
                total_after += 0
                continue

        # --- Compress large tool_result content ---
        if _is_tool_result_content(content):
            new_content, was_compressed = _summarize_tool_result(
                content, _COMPRESS_TOOL_OUTPUT_MAX_LINES
            )
            if was_compressed:
                compressed_count += 1
                new_msg = {**msg, "content": new_content}
                compressed.append(new_msg)
                total_after += len(str(new_content))
            else:
                compressed.append(msg)
                total_after += len(content_str)
            prev_signature = sig
            prev_text = text
            continue

        # --- Compress long assistant text (no tool_calls) ---
        if role == "assistant" and len(content_str) > 1000:
            new_content, was_compressed = _summarize_assistant_text(
                content, _COMPRESS_TOOL_OUTPUT_MAX_LINES
            )
            if was_compressed:
                compressed_count += 1
                new_msg = {**msg, "content": new_content}
                compressed.append(new_msg)
                total_after += len(str(new_content))
                prev_signature = sig
                prev_text = text
                continue

        # Keep as-is
        compressed.append(msg)
        total_after += len(content_str)
        prev_signature = sig
        prev_text = text

    stats = {
        "messages_before": len(messages),
        "messages_after": len(compressed),
        "deduped": deduped_count,
        "compressed": compressed_count,
        "chars_before": total_before,
        "chars_after": total_after,
        "compression_ratio": round(total_after / total_before, 2) if total_before > 0 else 1.0,
    }

    _compression_stats["total_requests_compressed"] += 1
    _compression_stats["total_tokens_before"] += total_before // 4
    _compression_stats["total_tokens_after"] += total_after // 4
    _compression_stats["total_deduped"] += deduped_count
    _compression_stats["total_compressed"] += compressed_count

    return compressed, stats
