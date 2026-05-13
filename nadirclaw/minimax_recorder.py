"""No-op MiniMax trace recorder.

Provides stubs for begin_turn / end_turn so anthropic_api.py can import
unconditionally.  Replace with real recording logic if needed.
"""


def begin_turn(provider: str = "", model: str = "", body: dict | None = None):
    return None


def end_turn(trace_ctx, raw_response: dict | None = None, error: str | None = None):
    pass
