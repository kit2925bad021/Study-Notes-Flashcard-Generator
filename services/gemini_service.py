"""
services/gemini_service.py

Central AI service for Mentora AI (Google Gemini via the google-genai SDK).

Every Step module talks to the AI ONLY through generate_content(prompt).
This is the single place that knows about the Gemini client, the model
name, the API key, retries and error messages. There is exactly ONE
Gemini client in the whole app (created lazily by _get_client()).

SETUP
  1. pip install -U google-genai python-dotenv
  2. Put your key in a .env file next to app.py (never in Python code):
         GEMINI_API_KEY=your_key_here
  3. Optional: change the model in ONE place, either by editing
     DEFAULT_MODEL below or by adding  GEMINI_MODEL=...  to .env.

ERROR HANDLING (each kind of failure has its OWN message)
  HTTP 503  -> "Gemini is temporarily experiencing high demand. ..."
  HTTP 429  -> "Gemini API rate limit or quota has been reached."
  HTTP 404  -> "Gemini model was not found or is unavailable ..."
  HTTP 401 / 403 -> "Gemini authentication or permission error."
  HTTP 500 / 504 -> temporary Gemini server problem (own message)
  HTTP 400  -> the request itself was rejected (own message)
  network / timeout -> "Unable to connect to the Gemini service. ..."
  bad JSON  -> "Gemini responded, but the generated format could not
               be parsed."  (see parse_json_response)
  The words "check your internet connection" are used ONLY for real
  network / connection failures - never for an HTTP error from Gemini.

RETRIES (one loop, no nesting)
  * At most MAX_RETRIES (3) attempts per generate_content() call.
  * Only temporary failures are retried: HTTP 429, 500, 503, 504 and
    real network problems (timeouts, dropped connections).
  * Never retried: 400, 401, 403, 404 and any other error - repeating
    the same request cannot fix those.
  * A short wait sits BETWEEN attempts (RETRY_DELAYS_SECONDS), so the
    API is never hit in a tight loop:
        attempt 1 -> wait -> attempt 2 -> wait -> attempt 3

DEBUG MODE
  While DEBUG_ERRORS is on (default: ON), every request and every failure
  is printed to the terminal that runs "python -m streamlit run app.py",
  prefixed with [GEMINI DEBUG]:
      request attempt 1/3 / model=... / prompt size
      HTTP status / API status / API message   (on an API error)
  The same detail is appended to the error text shown in the page.
  The API key is NEVER printed: it is replaced by [API_KEY_HIDDEN]
  everywhere. Turn debug off with  GEMINI_DEBUG=0  in .env.

DESIGN NOTES
  * The API key is read from the environment only. It is never hardcoded,
    printed, logged, or included in any error message.
  * The client is created lazily on the first AI call, so a missing key
    does NOT stop the app from starting.
  * Unexpected NON-network exceptions (TypeError, ValueError, ...) are
    reported as a generic failure and are not retried.
  * COOLDOWN: some Step modules re-ask when a call fails. To stop those
    loops from stacking on top of this file's own retries, a failed call
    starts a short cooldown. During it, generate_content() raises the
    same readable error immediately WITHOUT contacting Gemini (and prints
    a [GEMINI DEBUG] line saying so).
  * Every failure is converted to a RuntimeError with a short, readable
    message - never a stack trace, never None.
"""

import json
import logging
import os
import re
import sys
import time
import traceback
from typing import Optional

from dotenv import load_dotenv

# Load variables from the project's .env file (safe to call more than once).
load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings - the ONLY place the model name is defined.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL_NAME = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

MAX_RETRIES = 5                      # total attempts for a temporary failure
REQUEST_TIMEOUT_MS = 90_000           # per-request timeout (milliseconds)
RETRY_CODES = (429, 500, 503, 504)    # temporary errors worth retrying
RETRY_DELAYS_SECONDS = (3, 8)         # wait after attempt 1, then after attempt 2

# TEMPORARY DEBUGGING: show the real exception details (never the API key).
DEBUG_ERRORS = os.getenv("GEMINI_DEBUG", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

# Cooldowns (seconds) after a FAILED call - see COOLDOWN in the notes above.
COOLDOWN_QUOTA_SECONDS = 60     # HTTP 429 (after its retries)
COOLDOWN_KEY_SECONDS = 60       # HTTP 401 / 403
COOLDOWN_SERVER_SECONDS = 15    # 500 / 503 / 504 or network failure, after retries

# ---------------------------------------------------------------------------
# Messages - one per kind of failure, so errors are never masked.
# ---------------------------------------------------------------------------
MSG_NO_KEY = (
    "GEMINI_API_KEY is not configured. Add it to the .env file in the "
    "project folder and restart the app."
)
MSG_HIGH_DEMAND = (
    "Gemini is temporarily experiencing high demand. Please try again in a moment."
)
MSG_QUOTA = "Gemini API rate limit or quota has been reached."
MSG_NOT_FOUND = "Gemini model was not found or is unavailable for this API key."
MSG_AUTH = "Gemini authentication or permission error."
MSG_SERVER_500 = (
    "The Gemini service had a temporary internal error (HTTP 500). "
    "Please try again in a moment."
)
MSG_SERVER_504 = (
    "The Gemini service took too long to respond (HTTP 504). "
    "Please try again in a moment."
)
MSG_BAD_REQUEST = (
    "Gemini rejected the request (HTTP 400). The prompt or the request "
    "settings are not valid."
)
MSG_NETWORK = (
    "Unable to connect to the Gemini service. Check your internet connection."
)
MSG_BAD_FORMAT = (
    "Gemini responded, but the generated format could not be parsed."
)
MSG_EMPTY = "Gemini returned an empty response. Please try again."
MSG_FAILED = "Unable to generate the learning content right now. Please try again."

# Backward-compatible names in case another module imports the old constants.
# They now point at the specific, correct messages (never "check your internet").
MSG_UNAVAILABLE = MSG_HIGH_DEMAND
MSG_BAD_KEY = MSG_AUTH

# Exceptions that really mean "the network / connection had a problem".
# Only these are reported as "unable to connect" (and retried).
try:
    import httpx

    _NETWORK_ERRORS = (httpx.TransportError, ConnectionError, TimeoutError, OSError)
except ImportError:  # httpx ships with google-genai, so this is just a safety net
    _NETWORK_ERRORS = (ConnectionError, TimeoutError, OSError)


_client = None
_cooldown_until = 0.0           # time.monotonic() value; 0 = no cooldown
_cooldown_message = ""


# ---------------------------------------------------------------------------
# Debug helpers (never expose the API key)
# ---------------------------------------------------------------------------
def _redact(text) -> str:
    """Replace the API key (and anything shaped like one) with a placeholder."""
    text = str(text)
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "[API_KEY_HIDDEN]")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[API_KEY_HIDDEN]", text)


def _describe_error(error: BaseException) -> str:
    """One readable line: exception type, HTTP status, API status/message/details."""
    parts = [f"type={type(error).__module__}.{type(error).__name__}"]

    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    message = getattr(error, "message", None)
    details = getattr(error, "details", None)
    if code is not None:
        parts.append(f"http_status={code}")
    if status:
        parts.append(f"api_status={status}")
    if message:
        parts.append(f"api_message={message}")
    if details:
        parts.append(f"api_details={details}")
    parts.append(f"exception_message={error}")

    cause = error.__cause__ or error.__context__
    if cause is not None:
        parts.append(f"caused_by={type(cause).__module__}.{type(cause).__name__}: {cause}")

    return _redact(" | ".join(parts))[:1800]


def _debug_print(text: str) -> None:
    """Print a debug line to the terminal running Streamlit (key redacted)."""
    if DEBUG_ERRORS:
        print(f"[GEMINI DEBUG] {_redact(text)}", file=sys.stderr, flush=True)


def _debug_print_traceback(error: BaseException) -> None:
    if DEBUG_ERRORS:
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        print(f"[GEMINI DEBUG] traceback:\n{_redact(tb)}", file=sys.stderr, flush=True)


def _debug_print_api_error(error: BaseException, attempt: int, elapsed: float) -> None:
    """Print HTTP status, API status and API message of a Gemini API error."""
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    message = getattr(error, "message", None) or str(error)
    message = _redact(message)
    if len(message) > 700:
        message = message[:700] + "..."
    _debug_print(
        f"API error on attempt {attempt}/{MAX_RETRIES} (after {elapsed:.1f}s)\n"
        f"  HTTP status : {code}\n"
        f"  API status  : {status}\n"
        f"  API message : {message}"
    )


def _make_error(readable: str, error: Optional[BaseException] = None, extra: str = "") -> RuntimeError:
    """Build the RuntimeError shown to the student (with real cause in debug mode)."""
    if not DEBUG_ERRORS or (error is None and not extra):
        return RuntimeError(readable)
    detail = _describe_error(error) if error is not None else ""
    if extra:
        detail = f"{extra} | {detail}" if detail else extra
    return RuntimeError(f"{readable}  [DEBUG: {_redact(detail)}]")


def _describe_empty_response(response) -> str:
    """Why did Gemini return no text? (safety block, no candidates, ...)"""
    bits = []
    try:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            bits.append(f"prompt_feedback={feedback}")
        candidates = getattr(response, "candidates", None) or []
        bits.append(f"candidates={len(candidates)}")
        for candidate in candidates[:1]:
            bits.append(f"finish_reason={getattr(candidate, 'finish_reason', None)}")
    except Exception as inner:  # debugging must never crash the request
        bits.append(f"could_not_inspect_response={type(inner).__name__}")
    return "empty response: " + ", ".join(bits)


# ---------------------------------------------------------------------------
# Error classification (HTTP status -> message / retry / cooldown)
# ---------------------------------------------------------------------------
def _classify_api_error(code):
    """
    Map an HTTP status code to (readable message, retryable, cooldown seconds).

    Every code gets its own message so one kind of failure is never
    reported as another.
    """
    if code == 503:
        return MSG_HIGH_DEMAND, True, COOLDOWN_SERVER_SECONDS
    if code == 429:
        return MSG_QUOTA, True, COOLDOWN_QUOTA_SECONDS
    if code == 500:
        return MSG_SERVER_500, True, COOLDOWN_SERVER_SECONDS
    if code == 504:
        return MSG_SERVER_504, True, COOLDOWN_SERVER_SECONDS
    if code == 404:
        return MSG_NOT_FOUND, False, 0
    if code in (401, 403):
        return MSG_AUTH, False, COOLDOWN_KEY_SECONDS
    if code == 400:
        return MSG_BAD_REQUEST, False, 0
    if code is not None:
        return f"Gemini returned an unexpected error (HTTP {code}).", False, 0
    return MSG_FAILED, False, 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
def _start_cooldown(message: str, seconds: float) -> None:
    """Remember a failure so repeat calls fail fast for a short while."""
    global _cooldown_until, _cooldown_message
    _cooldown_until = time.monotonic() + seconds
    _cooldown_message = message


def _raise_if_cooling_down() -> None:
    """Fail immediately (no API request) while a cooldown is active."""
    if _cooldown_until and time.monotonic() < _cooldown_until:
        remaining = _cooldown_until - time.monotonic()
        _debug_print(
            f"cooldown active ({remaining:.0f}s left) - NOT contacting Gemini, "
            "repeating the previous error"
        )
        raise RuntimeError(_cooldown_message)


def _wait_before_retry(attempt: int, reason: str) -> None:
    """Sleep between attempts. `attempt` is the attempt that just failed (1-based)."""
    index = min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)
    delay = RETRY_DELAYS_SECONDS[index]
    _debug_print(f"{reason} - temporary error, waiting {delay}s before attempt {attempt + 1}/{MAX_RETRIES}")
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Client and config
# ---------------------------------------------------------------------------
def _get_client():
    """Create the ONE Gemini client on first use and reuse it afterwards."""
    global _client
    if _client is not None:
        return _client

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(MSG_NO_KEY)

    from google import genai
    from google.genai import types

    _client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    return _client


def _build_config(options):
    """
    Translate the optional per-call `options` dict into a Gemini config.

    Older Step modules (Study Material, step 20) pass Ollama-style options.
    Only "temperature" has a direct Gemini equivalent and is honoured.
    "num_ctx" (context window) is ignored because Gemini's window is far
    larger. "num_predict" (output cap) is also ignored on purpose: newer
    Gemini models count internal reasoning toward the output limit, so a
    small cap could cut an answer short or leave it empty.
    Unknown keys are ignored. Returns None when there is nothing to set.
    """
    if not isinstance(options, dict):
        return None
    temperature = options.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        from google.genai import types

        return types.GenerateContentConfig(temperature=float(temperature))
    return None


# ---------------------------------------------------------------------------
# JSON helper (for Step modules that ask Gemini for JSON)
# ---------------------------------------------------------------------------
def parse_json_response(text: str):
    """
    Parse JSON that Gemini returned as text.

    Removes a surrounding ```json ... ``` fence if present. If the text is
    not valid JSON, raises RuntimeError(MSG_BAD_FORMAT) - a different
    message from every connection / API error, so a formatting problem is
    never reported as a network problem.
    """
    cleaned = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        _debug_print(f"invalid JSON from Gemini (first 200 chars): {cleaned[:200]!r}")
        raise RuntimeError(MSG_BAD_FORMAT) from None


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def generate_content(prompt: str, options: Optional[dict] = None) -> str:
    """
    Send ONE prompt to Gemini and return the response text.

    Same interface the Step modules always used: text in, text out.
    `options` is optional (see _build_config); callers that pass only a
    prompt behave exactly as before.
    Raises RuntimeError with a short, readable message on any problem
    (never a traceback, never returns None or an empty string).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Cannot generate content from an empty prompt.")

    _raise_if_cooling_down()

    try:
        client = _get_client()
        config = _build_config(options)
    except RuntimeError:
        raise
    except Exception as error:  # e.g. SDK import problem or bad config object
        _debug_print(f"could not create client/config: {_describe_error(error)}")
        _debug_print_traceback(error)
        raise _make_error(MSG_FAILED, error) from None

    from google.genai import errors

    config_text = (
        f"temperature={getattr(config, 'temperature', None)}" if config is not None else "none"
    )

    # ONE loop, at most MAX_RETRIES attempts. Every failure either
    # `continue`s (temporary, attempts left) or raises - nothing else loops.
    for attempt in range(1, MAX_RETRIES + 1):
        is_last_try = attempt == MAX_RETRIES
        started = time.monotonic()
        _debug_print(
            f"request attempt {attempt}/{MAX_RETRIES}\n"
            f"  model={MODEL_NAME}\n"
            f"  prompt_chars={len(prompt)} config={config_text} timeout_ms={REQUEST_TIMEOUT_MS}"
        )

        try:
            if config is not None:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=config,
                )
            else:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )
            text = (response.text or "").strip()
            if not text:
                # e.g. blocked by safety filters or no candidates returned
                empty_info = _describe_empty_response(response)
                _debug_print(empty_info)
                raise _make_error(MSG_EMPTY, extra=empty_info)
            _debug_print(f"OK in {time.monotonic() - started:.1f}s, {len(text)} chars returned")
            return text

        except RuntimeError:
            raise

        except errors.APIError as error:
            code = getattr(error, "code", None)
            elapsed = time.monotonic() - started
            logger.warning("Gemini API error (HTTP %s)", code)
            _debug_print_api_error(error, attempt, elapsed)

            message, retryable, cooldown = _classify_api_error(code)
            if retryable and code in RETRY_CODES and not is_last_try:
                _wait_before_retry(attempt, f"HTTP {code}")
                continue

            failure = _make_error(message, error)
            if cooldown:
                _start_cooldown(failure.args[0], cooldown)
            raise failure from None

        except _NETWORK_ERRORS as error:  # timeout, connection refused, DNS, dropped connection
            elapsed = time.monotonic() - started
            logger.warning("Gemini request failed: %s", type(error).__name__)
            _debug_print(f"Network error after {elapsed:.1f}s: {_describe_error(error)}")
            if not is_last_try:
                _wait_before_retry(attempt, f"network error ({type(error).__name__})")
                continue
            _debug_print_traceback(error)
            failure = _make_error(MSG_NETWORK, error)
            _start_cooldown(failure.args[0], COOLDOWN_SERVER_SECONDS)
            raise failure from None

        except Exception as error:  # anything else is NOT a connection problem
            logger.warning("Gemini request failed unexpectedly: %s", type(error).__name__)
            _debug_print(f"UNEXPECTED {type(error).__name__}: {_describe_error(error)}")
            _debug_print_traceback(error)
            # Not a network problem, so it is not retried and gets no cooldown.
            raise _make_error(MSG_FAILED, error) from None

    # Unreachable (every path above returns, continues or raises).
    raise RuntimeError(MSG_FAILED)