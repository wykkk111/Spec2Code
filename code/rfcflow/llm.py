import os
import json
import asyncio
from typing import Any, Dict, List
import re


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it or use --mode quick.")
    return key


def get_async_openai_client():
    """
    Lazy import so `--mode quick` works without installing openai.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai package is not installed. Install dependencies (pip install -r requirements.txt) "
            "or use --mode quick."
        ) from e
    return AsyncOpenAI()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON from model output, compatible with the original repo style:
    - allow ```json ... ``` blocks
    - otherwise try to locate the first {...} block
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    # fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # try find first {...}
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        return json.loads(m2.group(1))
    raise ValueError("Failed to parse JSON from model output")


def _is_o_series_model(model: str) -> bool:
    # Heuristic: OpenAI o-series models (e.g., o1, o3-mini) reject some legacy params.
    m = (model or "").strip().lower()
    return m.startswith("o")


def _chat_create_kwargs(model: str, max_tokens: int, temperature: int = 0) -> Dict[str, Any]:
    """
    Build kwargs for `client.chat.completions.create` across model families.
    - o-series: use `max_completion_tokens`, do NOT send `temperature`
    - others: use legacy `max_tokens` + `temperature`
    """
    if _is_o_series_model(model):
        return {"max_completion_tokens": int(max_tokens)}
    return {"max_tokens": int(max_tokens), "temperature": temperature}


async def openai_json(
    client: Any,
    model: str,
    prompt: str,
) -> Dict[str, Any]:
    """
    Enforce JSON object output.
    """
    # Keep options minimal for compatibility across model families.
    # Prefer JSON-mode if supported to avoid parse errors.
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    return extract_json_from_text(completion.choices[0].message.content)


async def openai_chat_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    kwargs = _chat_create_kwargs(model, max_tokens=max_tokens, temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Prefer JSON-mode if supported to avoid parse errors.
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
    except Exception:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
    return extract_json_from_text(completion.choices[0].message.content or "")


async def gather_limited(coros: List, limit: int) -> List:
    sem = asyncio.Semaphore(limit)

    async def _wrap(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in coros])



