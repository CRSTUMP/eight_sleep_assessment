"""LLM client wrappers.

Both generation and chat run through Groq (Llama 3.3-70b):
- generate_json: cluster labeling, Slack alert generation
- groq_stream:   Part 2 RAG chatbot (streaming)

Embeddings are handled separately via sentence-transformers (local, no API
rate limits). See shared/embedder.py.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Iterator

from groq import Groq

from .config import settings

logger = logging.getLogger(__name__)

_groq = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "rate_limit" in str(exc).lower()


def _parse_retry_delay(exc: Exception, default: float = 30.0) -> float:
    match = re.search(r'"retryDelay":\s*"(\d+)s"', str(exc))
    return float(match.group(1)) + 2.0 if match else default


# ---------------------------------------------------------------------------
# Structured JSON generation
# ---------------------------------------------------------------------------

def generate_json(prompt: str) -> dict[str, Any]:
    """Generate a JSON response via Groq Llama 3.3-70b.

    Strips markdown code fences from the response before parsing.
    Retries up to 3 times with backoff on rate limits or transient errors.
    """
    if _groq is None:
        raise RuntimeError("GROQ_API_KEY not configured")

    json_prompt = (
        prompt
        + "\n\nIMPORTANT: Return ONLY a valid JSON object. No explanation, no markdown, no code fences."
    )
    for attempt in range(3):
        try:
            response = _groq.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": json_prompt}],
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text.strip())
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < 2:
                wait = _parse_retry_delay(exc)
                logger.warning(f"Groq rate-limited — waiting {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
            elif attempt < 2:
                logger.warning(f"generate_json attempt {attempt+1} failed: {exc!r}")
                time.sleep(2 ** attempt)
            else:
                raise

    raise RuntimeError("generate_json: max retries exceeded")


def safe_generate_json(prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Like generate_json but returns fallback on any failure (never raises)."""
    try:
        return generate_json(prompt)
    except Exception as e:
        logger.warning(f"generate_json failed, using fallback: {e}")
        return fallback


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

def groq_stream(messages: list[dict[str, str]], temperature: float = 0.3) -> Iterator[str]:
    """Stream chat completions via Groq — yields text chunks."""
    if _groq is None:
        raise RuntimeError("GROQ_API_KEY not configured — required for Part 2")
    stream = _groq.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
