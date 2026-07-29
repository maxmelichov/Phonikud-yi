"""Thin client for the Vercel AI Gateway (OpenAI-compatible chat completions).

Reads AI_GATEWAY_API_KEY / AI_GATEWAY_BASE_URL from .env at the repo root.
Supports text-only and audio(base64 mp3)+text messages, with retry/backoff.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Verified against GET /models on 2026-07-28. The gateway does not expose
# "google/gemini-3.1-pro" or "google/gemini-flash-3.6"; these are the closest
# real slugs.
MODEL_PRO = "google/gemini-3.1-pro-preview"
MODEL_FLASH = "google/gemini-3.6-flash"

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class GatewayError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str:
    load_dotenv(REPO_ROOT / ".env")
    val = os.environ.get(name, default)
    if not val:
        raise GatewayError(f"missing env var {name} (expected in {REPO_ROOT / '.env'})")
    return val


class Gateway:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or _env("AI_GATEWAY_API_KEY")
        self.base_url = (base_url or _env("AI_GATEWAY_BASE_URL")).rstrip("/")
        # Gemini Pro reasoning over audio routinely exceeds 300s; default high,
        # overridable via env for slow/fast model mixes.
        self.timeout = timeout or int(os.environ.get("GATEWAY_TIMEOUT_S", "900"))
        self.max_retries = max_retries
        self.session = requests.Session()

    # -- low level -----------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in _RETRY_STATUS:
                    raise GatewayError(f"{r.status_code}: {r.text[:800]}")
                last = GatewayError(f"{r.status_code}: {r.text[:400]}")
            except requests.RequestException as exc:  # network-level
                last = exc
            sleep = min(60.0, 2.0**attempt) + random.uniform(0, 1.0)
            time.sleep(sleep)
        raise GatewayError(f"exhausted {self.max_retries} retries: {last}")

    def list_models(self) -> list[str]:
        r = self.session.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return [m.get("id") for m in (data.get("data") if isinstance(data, dict) else data)]

    # -- chat ----------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str = MODEL_FLASH,
        temperature: float = 0.0,
        json_object: bool = False,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        # gemini-3.x are reasoning models: reasoning tokens are billed against
        # max_tokens, so a small budget yields empty content.
        if max_tokens is not None:
            max_tokens = max(max_tokens, 1024)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # Cap thinking: unconstrained dynamic reasoning on audio is what blew
        # the budget on the 3.1-pro run. Env override: GATEWAY_REASONING_EFFORT.
        effort = reasoning_effort or os.environ.get("GATEWAY_REASONING_EFFORT")
        if effort:
            payload["reasoning_effort"] = effort
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        try:
            data = self._post("/chat/completions", payload)
        except GatewayError as exc:
            # Some models reject response_format / reasoning_effort; retry without.
            retriable = [
                k for k in ("response_format", "reasoning_effort")
                if k in payload and k in str(exc)
            ]
            if retriable:
                for k in retriable:
                    payload.pop(k)
                data = self._post("/chat/completions", payload)
            else:
                raise
        choices = data.get("choices") or []
        if not choices:
            raise GatewayError(f"no choices in response: {json.dumps(data)[:500]}")
        return choices[0].get("message", {}).get("content") or ""

    def chat_json(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        """chat() but parses the reply as JSON, tolerating ```json fences."""
        raw = self.chat(messages, json_object=kw.pop("json_object", True), **kw)
        return parse_json_loose(raw)


# -- message helpers ---------------------------------------------------
def text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": text}


_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "m4a": "audio/mp4"}


def audio_message(
    audio_path: str | Path,
    text: str,
    role: str = "user",
    fmt: str = "mp3",
    style: str = "file",
) -> dict[str, Any]:
    """Multimodal message: base64-encoded audio part + text part.

    style="file" (default) uses OpenAI's `file`/`file_data` data-URL part. This is
    the ONLY audio shape the Vercel AI Gateway accepts for google/gemini-* as of
    2026-07-28; `input_audio` and `audio_url` both return 400 Invalid input.
    style="input_audio" emits the classic OpenAI audio part for other backends.
    """
    path = Path(audio_path)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    if style == "input_audio":
        part: dict[str, Any] = {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": fmt},
        }
    else:
        mime = _MIME.get(fmt, "audio/mpeg")
        part = {
            "type": "file",
            "file": {"filename": path.name, "file_data": f"data:{mime};base64,{b64}"},
        }
    return {"role": role, "content": [part, {"type": "text", "text": text}]}


# Unescaped ASCII " between Hebrew letters (acronyms like תשפ"ו) breaks JSON strings.
_HEB_NAKED_QUOTE = re.compile(r'(?<=[֐-׿])"(?=[֐-׿])')


def parse_json_loose(raw: str) -> Any:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    start = min([i for i in (s.find("{"), s.find("[")) if i != -1], default=-1)
    end = max(s.rfind("}"), s.rfind("]"))
    if start != -1 and end > start:
        s = s[start : end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_HEB_NAKED_QUOTE.sub("״", s))


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
