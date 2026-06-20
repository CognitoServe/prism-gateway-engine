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
