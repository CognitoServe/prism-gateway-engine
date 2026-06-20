import os
import json
import httpx
import tiktoken
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
enc = tiktoken.get_encoding("o200k_base")

# ── Config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
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


def build_structured_prompt(user_query: str, required_keys: list[str]) -> str:
    """
    Wrap the user's query with explicit JSON format instructions
    so the model is more likely to return parseable output.
    """
    key_list = ", ".join(f'"{k}"' for k in required_keys)
    return (
        f"Respond ONLY with valid JSON containing these keys: {key_list}.\n"
        f"No markdown fences, no explanation outside the JSON object.\n\n"
        f"User query: {user_query}"
    )


def parse_or_heal(raw_llm_output: str) -> dict:
    """
    Try direct JSON parse.  If that fails, strip markdown fences and
    extract the first {{ ... }} block.  Last resort: wrap raw text in
    a fallback dict so the caller always gets valid JSON back.
    """
    text = raw_llm_output.strip()

    # Attempt 1 — direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2 — strip ```json ... ``` fences
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]   # drop opening fence line
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3 — extract first { ... } block
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Give up — return raw text so caller never crashes
    return {"reply": raw_llm_output}


# ── New: OpenRouter caller ────────────────────────────────────────────────

async def call_openrouter(prompt: str) -> str:
    """POST to OpenRouter, return the model's reply string."""
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── New: /chat endpoint ──────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: dict):
    """
    Body: {"history": [...], "system": "...", "max_tokens": 500}
    Returns healed JSON from the model.
    """
    history    = request.get("history", [])
    system     = request.get("system", "")
    max_tokens = request.get("max_tokens", 4000)

    # 1 — prune history
    trimmed = sliding_window(history, max_tokens=max_tokens, system=system)

    # 2 — structured prompt from latest user message
    latest = trimmed[-1]["content"] if trimmed else ""
    prompt = build_structured_prompt(latest, required_keys=["reply"])

    # 3 — live LLM call
    raw = await call_openrouter(prompt)

    # 4 — parse / heal
    result = parse_or_heal(raw)

    return result
