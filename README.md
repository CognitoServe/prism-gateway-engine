# Prism Gateway Engine

A FastAPI service that wraps LLM API calls with token-aware context
management, structured prompt enforcement, and self-healing JSON parsing.

## What it does
- Counts tokens locally before sending any request (tiktoken, o200k_base)
- Prunes conversation history with a sliding-window strategy to stay
  within context limits
- Forces structured JSON output via strict system prompting
- Self-heals malformed LLM responses (markdown fences, trailing commas)
  instead of crashing
- Streams and calls a live LLM via OpenRouter

## Architecture
```mermaid
graph TD
    classDef client fill:#eef2f7,stroke:#94a3b8,stroke-width:2px;
    classDef process fill:#f0fdf4,stroke:#22c55e,stroke-width:2px;
    classDef ext fill:#fff7ed,stroke:#f97316,stroke-width:2px;
    classDef healer fill:#fef2f2,stroke:#ef4444,stroke-width:2px;

    Client["[Client] POST /chat"]:::client --> Receive["1. Receive Payload<br>(history, system, max_tokens)"]:::process
    
    Receive --> Trim["2. sliding_window()<br>Count tokens with tiktoken<br>Prune oldest history first"]:::process
    
    Trim --> BuildPrompt["3. build_structured_prompt()<br>Wrap user query with<br>JSON instructions"]:::process
    
    BuildPrompt --> CallLLM["4. call_openrouter()<br>Async POST to OpenRouter"]:::ext
    
    CallLLM --> ParseJSON["5. parse_or_heal()"]:::healer
    
    subgraph Healer [Self-Healing Parser Pipeline]
        ParseJSON --> TryDirect["Attempt 1: Direct JSON parse"]:::healer
        TryDirect -- "Success" --> ReturnResp["Return dict"]:::process
        
        TryDirect -- "Fail" --> StripFence["Attempt 2: Strip markdown fences<br>(```json ... ```)"]:::healer
        StripFence -- "Success" --> ReturnResp
        
        StripFence -- "Fail" --> ExtractBraces["Attempt 3: Extract outermost { ... }"]:::healer
        ExtractBraces -- "Success" --> ReturnResp
        
        ExtractBraces -- "Fail" --> Fallback["Attempt 4: Wrap raw response<br>in safe fallback object"]:::healer
        Fallback --> ReturnResp
    end
    
    ReturnResp --> SendClient["[Client] Final JSON Response"]:::client
```


## Run it
1. `pip install -r requirements.txt`
2. Add your OpenRouter key to a `.env` file: `OPENROUTER_API_KEY=...`
3. `uvicorn main:app --reload`
4. Visit `http://127.0.0.1:8000/docs`

## Example
POST /chat
{
  "history": [{"role": "user", "content": "..."}],
  "system": "...",
  "max_tokens": 500
}
