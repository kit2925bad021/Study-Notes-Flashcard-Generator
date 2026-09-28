"""
Ollama service layer for the Study-Notes-Flashcard-Generator.

All communication with the local Ollama server lives in this one file.

Public functions
----------------
generate_content(prompt)                 -> plain text answer (existing interface)
stream_content(prompt, options=None)     -> text answer, piece by piece (existing)
generate_json(prompt, schema_type=None)  -> parsed JSON (dict or list)
generate_feature(prompt, expected_key)   -> validated JSON for a feature
check_ollama_connection()                -> True / False
check_model_available()                  -> True / False

Every failure is raised as a RuntimeError with a short, readable message,
so the UI never has to show a Python traceback.

IMPORTANT: none of these functions should be called during a normal
Streamlit rerun. Call them only from inside a Generate / Create /
Regenerate button handler, and keep the result in st.session_state.

--------------------------------------------------------------------
SPEED NOTES (this revision) - service-level only; the public functions,
their names, parameters and return types, MODEL_NAME and every prompt in
Steps 1-21 are unchanged:
--------------------------------------------------------------------
1. ONE SHARED HTTP SESSION. Requests used to be sent with requests.post(),
   which opens a new connection every time. A single lazily-created
   requests.Session now keeps the connection to Ollama open and reuses it.
   The session ignores proxy settings from the environment (trust_env is
   False): Ollama is always on this machine, and looking up system proxy
   settings before every call is pure overhead (noticeably slow on
   Windows).

2. 127.0.0.1 INSTEAD OF "localhost". On many Windows machines "localhost"
   resolves to the IPv6 address first, fails, and only then falls back to
   IPv4 - a delay on every new connection. Ollama listens on 127.0.0.1 by
   default, so the address is used directly. Same server, no lookup.

3. MODEL PRE-LOAD. Loading the model into memory is the slowest part of the
   FIRST request. When this module is first imported, a small background
   thread asks Ollama to load the model (an empty-prompt /api/generate call
   is Ollama's documented way to do that) and keep it loaded (KEEP_ALIVE).
   It never blocks the app, never raises, and is silently skipped if Ollama
   is not running. Set the environment variable MENTORA_OLLAMA_WARMUP=0 to
   turn it off.

4. LENGTH CAP THAT DOES NOT CUT ANSWERS OFF. DEFAULT_NUM_PREDICT is 768 (it
   was 256). num_predict is only a ceiling - the model still stops on its
   own as soon as its answer is complete, so a higher ceiling costs
   nothing on normal replies. 256 was too low for this app: a 10-card
   Flashcards reply and the Pop Culture JSON need roughly 300-700 tokens.
   A reply cut off mid-way has no closing brace, which fails every JSON
   parser, and the Flashcards page then made extra "top-up" requests to
   replace the missing cards - which is slower, not faster.
   SAFETY NET: if a reply is still cut off by the cap (and the caller did
   not choose its own num_predict), ONE extra request is made with a larger
   cap (up to MAX_NUM_PREDICT) instead of returning a truncated answer.

5. FEWER USELESS RETRIES. Retries are for genuinely temporary problems only
   (Ollama not reachable yet, connect timeout, dropped connection, HTTP 5xx
   such as "token repeat limit reached"). A READ TIMEOUT is no longer
   retried: the model was already given the full READ_TIMEOUT_SECONDS to
   answer, and repeating that up to three times made a slow model look
   frozen for minutes. The pause between retries is
   0.5 seconds (was 1.0). Bad JSON / validation problems are never retried
   here - each feature file retries those itself with a fresh prompt.

6. INTERNAL TIMING. Every generation logs one line such as
   "Generation completed in 4.82 seconds (model load 0.00s, 212 tokens)"
   through the logging module (INFO level). Nothing is shown to the
   student.

Earlier fixes kept as they were:
- repeat_penalty 1.15 / repeat_last_n 32 / temperature 0.15 keep short
  structured replies focused without distorting them.
- The JSON extractor tolerates a trailing comma before a closing "}" or
  "]" (a common small-model artifact).
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 127.0.0.1 rather than "localhost": avoids the IPv6-first lookup delay that
# many Windows setups have (see SPEED NOTES above). Same local server.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"  # kept: other files may import it
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"  # lists installed models
MODEL_NAME = "llama3.2:3b"

# Fail fast if Ollama is not running (connect), but give the model enough
# time to answer once it is (read). The first request is slower because the
# model has to be loaded into memory (the pre-load below usually hides that).
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 120
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

# Short timeout for the quick "is Ollama alive?" checks.
CHECK_TIMEOUT_SECONDS = 3

# Keep the model loaded in memory between requests. Reloading is the slowest
# part of a request, so every request after the first becomes much faster.
KEEP_ALIVE = "30m"

# Generation settings.
#
# This app mostly asks the model for small structured JSON objects (a quiz
# question, a Truth & Trick round, an emoji challenge, a scenario, a dynamic
# question, a Pop Culture connection) and for a 10-card flashcard set. The
# defaults below are conservative for sampling, but the length cap leaves
# room for the largest of those replies:
#
#   - num_predict: 768. A CEILING, not a target: the model stops by itself
#     when its answer is complete, so this costs no time on normal replies.
#     It only prevents a runaway/repetitive reply from going on forever.
#     Do not lower it without checking the Flashcards (10 Q/A pairs) and
#     Pop Culture replies still fit - a truncated reply is invalid JSON.
#   - repeat_penalty 1.15 / repeat_last_n 32: discourage repetition without
#     distorting short structured output.
#   - temperature 0.15: focused, consistent answers (good for JSON).
#   - num_ctx 2048: a stable context window for a trimmed study-material
#     excerpt plus the prompt instructions, without an unnecessarily large
#     (slower, memory-hungry) window.
#
# A caller that genuinely needs a longer answer (or different sampling) can
# still override any of these per call via generate_content(prompt,
# options={...}) / stream_content(prompt, options={...}).
DEFAULT_TEMPERATURE = 0.15
DEFAULT_NUM_PREDICT = 768  # ceiling for one reply; see the note above
DEFAULT_REPEAT_PENALTY = 1.15
DEFAULT_REPEAT_LAST_N = 32
DEFAULT_NUM_CTX = 2048
DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": DEFAULT_TEMPERATURE,
    "num_predict": DEFAULT_NUM_PREDICT,
    "repeat_penalty": DEFAULT_REPEAT_PENALTY,
    "repeat_last_n": DEFAULT_REPEAT_LAST_N,
    "num_ctx": DEFAULT_NUM_CTX,
}

# If a reply is still cut off by the cap above, ONE extra request is made
# with a larger cap, never above this value.
MAX_NUM_PREDICT = 1536

# Retry handling: at most 2 retries (3 attempts in total), short pause between.
# This is for TRANSIENT service-level failures only (Ollama not reachable
# yet, a connect timeout, a dropped connection, a 5xx from Ollama) - it is
# not, and must not become, a place where bad JSON gets retried. Each
# feature file already retries its own generation (with a fresh prompt)
# when the reply fails ITS validation; this service must not multiply those
# attempts by retrying underneath them too. A read timeout is NOT retried.
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 0.5

# Clean messages that are safe to show to the user.
MSG_NOT_RUNNING = (
    "Ollama is not running. Start the Ollama app (or run 'ollama serve') "
    "and try again."
)
MSG_MODEL_MISSING = (
    f"Required model {MODEL_NAME} is not installed. "
    f"Run this in a terminal, then try again: ollama pull {MODEL_NAME}"
)
MSG_UNAVAILABLE = "AI service is temporarily unavailable."
MSG_TIMEOUT = "The AI took too long to respond. Please try again."
MSG_INTERRUPTED = "The connection to Ollama was interrupted. Please try again."
MSG_INVALID = "Could not generate valid content. Please try again."
MSG_EMPTY = "The AI returned an empty response. Please try again."
MSG_EMPTY_PROMPT = "Please provide some text to generate content from."

# Used whenever Ollama itself reports a concrete error (an HTTP 4xx/5xx
# response, or a 200 response whose body contains an "error" field - for
# example "prediction aborted, token repeat limit reached"). The real
# detail is appended so it ends up in the raised RuntimeError's message
# instead of being swallowed - callers/logs see exactly what Ollama said,
# not just a generic "unavailable" message.
MSG_GENERATION_FAILED = "Ollama generation failed ({detail})."

# Extra hint added to JSON prompts so the small model copies the right shape.
# To support a new feature, add another entry here (name -> example JSON).
SCHEMA_HINTS: Dict[str, str] = {
    "flashcards": (
        '{"flashcards": [{"question": "...", "answer": "..."}]}'
    ),
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OllamaServiceError(RuntimeError):
    """A clean, user-safe error. It is still a RuntimeError, so existing
    `except RuntimeError` code keeps working."""


class _TransientOllamaError(OllamaServiceError):
    """A temporary failure that is worth retrying (internal use only)."""


# ---------------------------------------------------------------------------
# Shared HTTP session (connection reuse)
# ---------------------------------------------------------------------------

_session_lock = threading.Lock()
_http_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """
    Return the one shared requests.Session, creating it on first use.
    It keeps the connection to Ollama open between requests instead of
    reconnecting every time, and skips system proxy lookups (Ollama is
    local, so a proxy is never wanted) which are slow on some machines.
    """
    global _http_session
    if _http_session is None:
        with _session_lock:
            if _http_session is None:
                session = requests.Session()
                session.trust_env = False
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                _http_session = session
    return _http_session


# ---------------------------------------------------------------------------
# Low-level request helpers (the only place that talks to Ollama)
# ---------------------------------------------------------------------------

def _validate_prompt(prompt: Any) -> str:
    """Make sure the prompt is a non-empty string and return it trimmed."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise OllamaServiceError(MSG_EMPTY_PROMPT)
    return prompt.strip()


def _merge_options(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Default speed options, overridden by anything the caller passes in."""
    merged = dict(DEFAULT_OPTIONS)
    if options:
        merged.update(options)
    return merged


def _build_payload(
    prompt: str,
    stream: bool,
    options: Optional[Dict[str, Any]] = None,
    json_mode: bool = False,
) -> Dict[str, Any]:
    """Build the JSON body sent to Ollama's /api/generate endpoint."""
    data: Dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": KEEP_ALIVE,
        "options": _merge_options(options),
    }
    if json_mode:
        data["format"] = "json"  # makes Ollama return valid JSON only
    return data


def _extract_error_detail(response: requests.Response) -> str:
    """
    Pull the actual error text out of an Ollama response, e.g. "prediction
    aborted, token repeat limit reached". Ollama usually reports errors as
    a JSON body like {"error": "..."}; fall back to the raw response text
    (trimmed) if the body isn't JSON or has no "error" field.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"]).strip()
    text = (response.text or "").strip()
    return text[:300] if text else f"HTTP {response.status_code}"


def _post_once(data: Dict[str, Any], stream: bool = False) -> requests.Response:
    """
    Send ONE request to Ollama.

    Raises _TransientOllamaError for problems that may go away on their own
    (worth retrying) and OllamaServiceError for problems that will not.
    """
    try:
        response = _get_session().post(
            OLLAMA_URL, json=data, timeout=REQUEST_TIMEOUT, stream=stream
        )
    except requests.exceptions.ReadTimeout:
        # Ollama accepted the request but did not answer within the read
        # timeout. Asking again would just wait that long again (up to 3
        # times), so this is NOT retried.
        raise OllamaServiceError(MSG_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise _TransientOllamaError(MSG_NOT_RUNNING)
    except requests.exceptions.Timeout:
        raise _TransientOllamaError(MSG_TIMEOUT)
    except requests.exceptions.RequestException as error:
        logger.warning("Ollama request failed: %s", error)
        raise _TransientOllamaError(MSG_UNAVAILABLE)

    if response.status_code == 404:
        # Ollama answers 404 when the requested model is not installed.
        response.close()
        raise OllamaServiceError(MSG_MODEL_MISSING)

    if not response.ok:
        detail = _extract_error_detail(response)
        # Full detail always goes to the log...
        logger.warning("Ollama returned HTTP %s: %s", response.status_code, detail)
        server_side = response.status_code >= 500
        response.close()
        # ...and also into the raised error itself, so callers (and their
        # own error logs) see exactly what Ollama reported - e.g. "Ollama
        # returned HTTP 500 error: prediction aborted, token repeat limit
        # reached" - rather than a generic "unavailable" message that
        # hides what actually went wrong.
        message = f"Ollama returned HTTP {response.status_code} error: {detail}"
        if server_side:
            # 5xx (including the "token repeat limit reached" abort) is
            # treated as transient: _post() will retry it, up to
            # MAX_RETRIES times, before giving up.
            raise _TransientOllamaError(message)
        raise OllamaServiceError(message)

    return response


def _post(data: Dict[str, Any], stream: bool = False) -> requests.Response:
    """Send a request to Ollama, retrying temporary failures (max 2 retries)."""
    last_error: Optional[OllamaServiceError] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _post_once(data, stream=stream)
        except _TransientOllamaError as error:
            last_error = error
            logger.info("Ollama attempt %d failed: %s", attempt + 1, error)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error or OllamaServiceError(MSG_UNAVAILABLE)


def _log_timing(body: Any, started: float, label: str = "Generation") -> None:
    """Internal timing only (INFO log) - never shown to the student."""
    elapsed = time.perf_counter() - started
    load_seconds = 0.0
    tokens: Any = "?"
    if isinstance(body, dict):
        load_ns = body.get("load_duration")
        if isinstance(load_ns, (int, float)):
            load_seconds = load_ns / 1e9
        if body.get("eval_count") is not None:
            tokens = body.get("eval_count")
    logger.info(
        "%s completed in %.2f seconds (model load %.2fs, %s tokens)",
        label, elapsed, load_seconds, tokens,
    )


def _request_generation(
    prompt: str,
    json_mode: bool,
    options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Send one non-streaming request and return Ollama's parsed body."""
    response = _post(
        _build_payload(prompt, stream=False, options=options, json_mode=json_mode)
    )
    try:
        body = response.json()
    except ValueError:
        logger.warning("Ollama sent a response that was not valid JSON.")
        raise OllamaServiceError(MSG_UNAVAILABLE)
    finally:
        response.close()

    if not isinstance(body, dict) or body.get("error"):
        detail = str(body.get("error")).strip() if isinstance(body, dict) and body.get("error") else "unexpected response shape"
        logger.warning("Ollama reported an error: %s", body)
        raise OllamaServiceError(MSG_GENERATION_FAILED.format(detail=detail))
    return body


def _was_cut_off(body: Dict[str, Any], cap: Any) -> bool:
    """True if the reply stopped because it hit the num_predict ceiling."""
    if body.get("done_reason") == "length":
        return True
    eval_count = body.get("eval_count")
    return (
        isinstance(eval_count, int)
        and isinstance(cap, int)
        and cap > 0
        and eval_count >= cap
    )


def _generate(
    prompt: str,
    json_mode: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Run one full (non-streaming) generation and return the text."""
    started = time.perf_counter()
    body = _request_generation(prompt, json_mode, options)

    # Safety net: a reply cut off by the length ceiling is incomplete (for
    # JSON, it has no closing brace). If the caller did not pick its own
    # num_predict, ask once more with a larger ceiling instead of handing
    # back a truncated answer.
    caller_chose_cap = bool(options) and "num_predict" in options
    cap = _merge_options(options).get("num_predict")
    if not caller_chose_cap and _was_cut_off(body, cap):
        bigger = min(int(cap) * 2, MAX_NUM_PREDICT) if isinstance(cap, int) else MAX_NUM_PREDICT
        if not isinstance(cap, int) or bigger > cap:
            logger.info("Reply hit the %s-token cap; retrying once with %s.", cap, bigger)
            retry_options = dict(options or {})
            retry_options["num_predict"] = bigger
            body = _request_generation(prompt, json_mode, retry_options)

    _log_timing(body, started)

    text = str(body.get("response", "")).strip()
    if not text:
        raise OllamaServiceError(MSG_EMPTY)
    return text


# ---------------------------------------------------------------------------
# Model pre-load (hides the slow first-request model load)
# ---------------------------------------------------------------------------

def _warm_up_model() -> None:
    """
    Ask Ollama to load MODEL_NAME into memory and keep it there. An
    /api/generate call with no prompt loads the model without generating
    anything. Runs in a background thread; never raises.
    """
    try:
        started = time.perf_counter()
        response = _get_session().post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "keep_alive": KEEP_ALIVE},
            timeout=REQUEST_TIMEOUT,
        )
        response.close()
        logger.info("Model pre-load finished in %.2f seconds.", time.perf_counter() - started)
    except Exception as error:  # Ollama not running, etc. - just skip it.
        logger.info("Model pre-load skipped: %s", error)


_warm_up_started = False


def _start_warm_up_once() -> None:
    """Start the background pre-load once per process (unless disabled)."""
    global _warm_up_started
    if _warm_up_started:
        return
    _warm_up_started = True
    if os.environ.get("MENTORA_OLLAMA_WARMUP", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    threading.Thread(target=_warm_up_model, name="ollama-warm-up", daemon=True).start()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove markdown fences such as ```json ... ``` around a response."""
    return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip()).strip()


def _extract_json_block(text: str) -> Optional[str]:
    """Return the part of the text from the first { or [ to the last } or ]."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if not starts or not ends:
        return None
    start, end = min(starts), max(ends)
    return text[start:end + 1] if end > start else None


def _remove_trailing_commas(text: str) -> str:
    """
    Small models occasionally leave a trailing comma right before a closing
    "}" or "]" (e.g. '..."c"],}'), which otherwise valid JSON.loads()
    rejects outright. Stripped here as one extra candidate before giving
    up - never as the only thing attempted.
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def _parse_json(raw_text: str) -> Any:
    """Safely turn the model's text into a Python dict or list."""
    cleaned = _strip_code_fences(raw_text)

    candidates = [cleaned]
    extracted = _extract_json_block(cleaned)
    if extracted and extracted != cleaned:
        candidates.append(extracted)

    # For every candidate so far, also try a trailing-comma-repaired
    # version, without discarding the original (some replies are already
    # clean, and a blind regex should never run ahead of a working parse).
    for candidate in list(candidates):
        repaired = _remove_trailing_commas(candidate)
        if repaired != candidate:
            candidates.append(repaired)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:  # JSONDecodeError is a kind of ValueError
            continue
        if isinstance(data, (dict, list)):
            return data

    logger.warning("Could not parse JSON from model output: %s", raw_text[:300])
    raise OllamaServiceError(MSG_INVALID)


def _add_json_instructions(prompt: str, schema_type: Optional[str]) -> str:
    """Add short, strict JSON instructions (small models need clear rules)."""
    instructions = "Return ONLY valid JSON. No explanations. No markdown."
    hint = SCHEMA_HINTS.get(schema_type) if schema_type else None
    if hint:
        instructions += f"\nUse exactly this shape:\n{hint}"
    return f"{prompt}\n\n{instructions}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_ollama_connection() -> bool:
    """Return True if the Ollama server is reachable, otherwise False."""
    try:
        response = _get_session().get(OLLAMA_BASE_URL, timeout=CHECK_TIMEOUT_SECONDS)
        return response.ok
    except requests.exceptions.RequestException:
        return False


def check_model_available() -> bool:
    """Return True if MODEL_NAME is installed locally, otherwise False."""
    try:
        response = _get_session().get(OLLAMA_TAGS_URL, timeout=CHECK_TIMEOUT_SECONDS)
        if not response.ok:
            return False
        models = response.json().get("models", [])
    except (requests.exceptions.RequestException, ValueError, AttributeError):
        return False

    def with_tag(name: str) -> str:
        # Ollama treats "llama3" as "llama3:latest".
        return name if ":" in name else f"{name}:latest"

    wanted = with_tag(MODEL_NAME)
    installed = set()
    for model in models:
        if isinstance(model, dict):
            installed.add(with_tag(str(model.get("name") or model.get("model") or "")))
    return wanted in installed


def generate_content(prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate plain text with the local Ollama model (one request).

    Returns only the generated text. Raises RuntimeError with a short,
    readable message on any problem (never a traceback, and never None).

    options: optional Ollama options that override the defaults, e.g.
    {"num_ctx": 4096} to give a long prompt enough context window, or
    {"temperature": 0.35} for more varied wording. Existing callers that
    pass only a prompt keep exactly the same behaviour as before.
    """
    return _generate(_validate_prompt(prompt), options=options)


def stream_content(
    prompt: str, options: Optional[Dict[str, Any]] = None
) -> Iterator[str]:
    """
    Same single Ollama request as generate_content(), but yields the answer
    piece by piece so the page can show text immediately.

    Use with st.write_stream(stream_content(prompt)). Raises RuntimeError
    with a clean message on failure, exactly like generate_content().

    options: optional Ollama options that override the defaults, e.g.
    {"num_predict": 1400} to allow a longer answer.
    """
    prompt = _validate_prompt(prompt)
    started = time.perf_counter()
    response = _post(_build_payload(prompt, stream=True, options=options), stream=True)

    received_text = False
    final_chunk: Any = None
    try:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                detail = str(chunk["error"]).strip()
                logger.warning("Ollama stream error: %s", detail)
                raise OllamaServiceError(MSG_GENERATION_FAILED.format(detail=detail))
            piece = chunk.get("response", "")
            if piece:
                received_text = True
                yield piece
            if chunk.get("done"):
                final_chunk = chunk
                break
    except requests.exceptions.RequestException as error:
        logger.warning("Ollama stream interrupted: %s", error)
        raise OllamaServiceError(MSG_INTERRUPTED)
    finally:
        response.close()

    if not received_text:
        raise OllamaServiceError(MSG_EMPTY)
    _log_timing(final_chunk, started, label="Streamed generation")


def generate_json(prompt: str, schema_type: Optional[str] = None) -> Any:
    """
    Ask Ollama for strict JSON (one request) and return it as a Python
    dict or list.

    schema_type: optional name from SCHEMA_HINTS (for example "flashcards")
    that tells the model exactly which JSON shape to produce. Unknown names
    are ignored.

    Raises RuntimeError("Could not generate valid content. ...") if the
    answer is not valid JSON. Never performs its own extra retries - a
    single generation attempt per call, exactly like generate_content();
    any retrying across attempts is each feature file's own responsibility.
    """
    prompt = _validate_prompt(prompt)
    raw_text = _generate(_add_json_instructions(prompt, schema_type), json_mode=True)
    return _parse_json(raw_text)


def generate_feature(prompt: str, expected_key: Optional[str] = None) -> Any:
    """
    One helper for every AI feature: generate JSON, parse it and check it.

    If expected_key is given (for example "flashcards"), the result must
    contain that key with a non-empty value; a bare list is wrapped as
    {expected_key: list}. Returns the full validated data.

    This only checks that the expected top-level key exists and is
    non-empty - it does not, and must not, know anything about a
    feature's own schema (exact item counts, required sub-fields, etc.).
    That validation belongs in each feature file.
    """
    data = generate_json(prompt, schema_type=expected_key)

    if expected_key is not None:
        if isinstance(data, list):
            data = {expected_key: data}
        if not isinstance(data, dict) or not data.get(expected_key):
            logger.warning("Expected key %r missing from model output.", expected_key)
            raise OllamaServiceError(MSG_INVALID)

    return data


# Start the background model pre-load once, when this module is first
# imported (see SPEED NOTES). Never blocks and never raises.
_start_warm_up_once()