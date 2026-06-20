import tiktoken
from fastapi import FastAPI

app = FastAPI()
enc = tiktoken.get_encoding("o200k_base")

# ── Config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL   = "openai/gpt-4o-mini"


# ── Reused components ─────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Token count for a single string using o200k_base."""
    return len(enc.encode(text))


def build_prompt(system: str, history: list[dict], new_message: str) -> str:
    """
    Flatten system + history + new user message into one prompt string.
    Format:
        system: ...
        user: ...
        assistant: ...
        user: <new_message>
    """
    parts = []
    if system:
        parts.append(f"system: {system}")
    for msg in history:
        parts.append(f"{msg['role']}: {msg['content']}")
    parts.append(f"user: {new_message}")
    return "\n".join(parts)


def sliding_window(history: list[dict], max_tokens: int, system: str) -> list[dict]:
    """
    Drop oldest messages until the total token count of
    system + remaining history fits within max_tokens.
    The most recent message is never dropped.
    """
    if not history:
        return history

    system_tokens = count_tokens(system) if system else 0
    budget = max_tokens - system_tokens

    # Walk backwards, accumulating until we blow the budget
    kept: list[dict] = []
    running = 0
    for msg in reversed(history):
        msg_tokens = count_tokens(msg["content"])
        if running + msg_tokens > budget and kept:
            # budget blown and we already have at least the latest message
            break
        running += msg_tokens
        kept.append(msg)

    kept.reverse()
    return kept
