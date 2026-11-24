"""Thin wrapper around Gemini (google-genai) with model fallback and retries."""
import logging
import os
import time

log = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised when no model could produce an answer. Message is user-safe."""


def get_api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def model_chain():
    """GEMINI_MODEL (if set) first, then the built-in fallbacks, de-duplicated."""
    chain = []
    preferred = os.environ.get("GEMINI_MODEL", "").strip()
    for name in ([preferred] if preferred else []) + DEFAULT_MODELS:
        if name and name not in chain:
            chain.append(name)
    return chain


def _status_code(exc):
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return getattr(exc, "status_code", None)


def _call_model(model, prompt):
    from google import genai

    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()
    if not text:
        raise ValueError("empty response from model")
    return text


def generate(prompt):
    """Return generated text, trying each model with retry + exponential backoff.

    Retries 429/5xx, falls through to the next model on 404/empty/other errors,
    and fails fast on authentication problems.
    """
    if not get_api_key():
        raise LLMError("The AI service is not configured: set GEMINI_API_KEY in the server environment.")

    retries = int(os.environ.get("LLM_RETRIES", "3"))
    base_delay = float(os.environ.get("LLM_BACKOFF_SECONDS", "2"))
    last = transient = "unknown error"

    for model in model_chain():
        for attempt in range(retries + 1):
            try:
                return _call_model(model, prompt)
            except Exception as exc:  # noqa: BLE001 - classify below
                code = _status_code(exc)
                last = f"{model}: HTTP {code}" if code else f"{model}: {type(exc).__name__}"
                if code in RETRYABLE or not code:
                    transient = last  # prefer reporting overload over "model not found"
                if code in (401, 403) or "API_KEY_INVALID" in str(exc):
                    raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY.") from None
                if code in RETRYABLE and attempt < retries:
                    delay = base_delay * (2 ** attempt)
                    log.warning("Gemini %s returned %s; retrying in %.1fs", model, code, delay)
                    time.sleep(delay)
                    continue
                log.warning("Gemini model %s failed (%s); trying next model", model, last)
                break
    raise LLMError(
        "The AI service is temporarily unavailable (it may be overloaded or rate limited). "
        f"Please try again in a minute. [{transient if transient != 'unknown error' else last}]"
    )
