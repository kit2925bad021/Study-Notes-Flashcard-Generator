"""
Step 5 — Emoji Decoder
--------------------------
A quick learning game: Ollama turns ONE academic concept from the current
study material into a short, meaningful emoji clue.

TWO WAYS THIS FILE IS USED
----------------------------
1. Standalone page (show_emoji_decoder(), used by the "🪄 Emoji Decode"
   sidebar nav item): a manual "Generate" button, an optional hint, a
   "Show Answer" button, a typed answer + "Submit Answer", and a
   "Reveal Answer" flow. This is a free-play page a student can open any
   time - it is NOT part of the graded 5-step Practice flow and keeps its
   own "😀 Generate New Emoji Challenge" button.

2. Practice hub (steps/step21_practice.py), stage 3 of 5
   ("Practice 3/5 — Emoji Decode"): the hub does NOT call any function in
   this file to render a page. It only imports and calls three internal
   functions directly:

       _generate_emoji_challenge(content, used_concepts)
       _normalize_concept(text)
       _significant_words(text)

   and does its own generation-once / render / submit / "Next Round"
   sequencing (see step21_practice.py). This file must not rename these
   three functions or change their signatures without also updating
   step21_practice.py.

   The dict returned by _generate_emoji_challenge() still contains the
   legacy keys the hub reads (concept, emoji_sequence, hint, answer,
   explanation) AND the new round keys (emoji, question). Nothing the hub
   already reads was removed or renamed.

FIX (this revision) — "Could not create a valid Emoji Decode round"
---------------------------------------------------------------------
That message is shown by the caller (step21_practice.py) whenever
_generate_emoji_challenge() returns None. Reasons it returned None too
often, all fixed here:
  * The reply parser cut from the FIRST "{" to the LAST "}". Any trailing
    text containing a brace (or a reasoning "<think>" block) broke it.
    Now: <think> blocks and Markdown fences are removed and the first
    complete JSON object is decoded safely with json.JSONDecoder.raw_decode.
  * The reply had to use one exact set of key names. Now the round schema
    is {emoji, question, answer, hint, explanation}; key names are matched
    case-insensitively and common aliases (e.g. emoji_sequence) are
    accepted; a round wrapped in {"round": {...}} or a one-item list is
    unwrapped.
  * Every retry re-sent the same long prompt. Now attempt 1 uses the
    concise prompt and attempts 2-3 use a much shorter retry prompt with a
    smaller slice of the study material (faster and more reliable).

VALIDATION (a round is only accepted when ALL of these hold)
  1. emoji, question, answer, hint, explanation exist and are non-empty
     (placeholder values such as "None"/"N/A" count as empty).
  2. emoji really contains emoji symbols (any stray letters are removed).
  3. answer is at most 3 words.
  4. answer is not a placeholder / not a copy of the question.
  5. explanation is a real sentence that mentions the answer (so the
     explanation supports the answer).
  Malformed or incomplete rounds are never returned or displayed.
  NOTE: whether the emoji "really" depict the answer is judged by the
  model that wrote the round; we do NOT spend an extra slow Ollama call to
  double-check it. The checks above (real emoji, no answer text leaking
  into the emoji field, answer <= 3 words, explanation names the answer)
  are the cheap safety net.

RETRY: max MAX_GENERATION_ATTEMPTS (3) Ollama calls per generation. On the
final attempt an answer that is a little too long is shortened to 3 words
instead of being thrown away. If every attempt fails the caller receives
None and the standalone page shows CHALLENGE_FAILURE_MESSAGE. Nothing
raw (None, empty fields, JSON, tracebacks) is ever displayed.

MODEL RULE: this file never hardcodes a model name. Every generation call
goes through generate_content(prompt) from services/ollama_service.py, and
the model used is controlled ONLY by MODEL_NAME inside that file. This
file never creates its own Ollama client.

SOURCE OF TRUTH: every challenge is generated only from
st.session_state["study_material"]["content"].

Embedding: pass embedded=True to skip this page's own header/topic bar.

Imported in app.py with:
    from steps.step5_emoji_decoder import show_emoji_decoder
No other app.py changes are required for this feature.
"""

import json
import re
import unicodedata

import streamlit as st

from services.ollama_service import generate_content
from steps.step19_ui import show_page_header, show_topic_bar

# The study material is trimmed so the small model gets a short, simple
# prompt instead of being overwhelmed with a huge block of text.
MAX_CONTENT_CHARS = 1200
# Retry prompts use an even smaller slice so retries are fast.
RETRY_CONTENT_CHARS = 600

# Full attempt budget for ONE generation (never unlimited).
MAX_GENERATION_ATTEMPTS = 3

# The five, and only, fields a valid round must have.
REQUIRED_FIELDS = ("emoji", "question", "answer", "hint", "explanation")

# Accepted alternative key names a model may use (matched after
# lower-casing and replacing spaces/hyphens with underscores).
_FIELD_ALIASES = {
    "emoji": ("emoji", "emojis", "emoji_sequence", "emoji_clue"),
    "question": ("question", "prompt"),
    "answer": ("answer", "correct_answer"),
    "hint": ("hint", "clue"),
    "explanation": ("explanation", "reason", "rationale"),
    "concept": ("concept", "topic"),
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "as", "by", "from", "that", "this", "these", "those", "it", "its",
}

# Values a model sometimes writes instead of real content.
_PLACEHOLDER_VALUES = {
    "none", "null", "nil", "n/a", "na", "undefined", "tbd", "unknown",
    "answer", "concept", "hint", "explanation", "question", "emoji",
    "...", "-", "--",
}

# Shown by the standalone page if a round could not be generated after
# every retry. No raw exception text or Ollama error ever reaches the page.
CHALLENGE_FAILURE_MESSAGE = "Unable to generate this round. Please try again."

# The Emoji Decoder answer is a short concept/phrase, never a sentence.
MAX_ANSWER_WORDS = 3

# Sanity cap on emoji symbols in one clue.
MAX_EMOJI_SYMBOLS = 10

_JSON_DECODER = json.JSONDecoder()


# ---------------------------------------------------------------------------
# Prompt building — short and simple, ONE round per request.
# ---------------------------------------------------------------------------

def _avoid_text(avoid_concepts):
    """Soft 'do not reuse' line (last 5 concepts only, to keep prompts short)."""
    recent = [str(c) for c in (avoid_concepts or []) if str(c).strip()][-5:]
    if not recent:
        return ""
    return "Do not reuse these concepts: " + "; ".join(recent) + "\n"


def _build_emoji_prompt(content, avoid_concepts, retry=False):
    """
    Build the prompt for exactly ONE Emoji Decode round.

    retry=False -> the concise first-attempt prompt.
    retry=True  -> an even shorter prompt (smaller study-material slice)
                   used for attempts 2 and 3.

    Previously-used concepts are only a soft "avoid" hint - they are never
    used to reject a reply afterward.
    """
    if retry:
        trimmed = (content or "")[:RETRY_CONTENT_CHARS]
        return (
            "Return ONLY one JSON object with exactly these keys: "
            "emoji, question, answer, hint, explanation.\n"
            "emoji: 2 to 4 emojis only, no letters. "
            "answer: maximum 3 words. "
            "explanation: one short sentence that includes the answer.\n"
            + _avoid_text(avoid_concepts)
            + "No Markdown. No extra text.\n\n"
            "Study material:\n" + trimmed + "\n"
        )

    trimmed = (content or "")[:MAX_CONTENT_CHARS]
    example = (
        '{"emoji": "😊📚🧠", '
        '"question": "What concept do these emojis represent?", '
        '"answer": "Learning", '
        '"hint": "Think about gaining knowledge", '
        '"explanation": "The emojis represent learning through study and gaining knowledge."}'
    )
    return (
        "Create ONE Emoji Decode round from the supplied study material.\n\n"
        "Return ONLY valid JSON.\n\n"
        "Required fields:\n"
        "emoji, question, answer, hint, explanation.\n\n"
        "The answer must be maximum 3 words.\n"
        "The emoji must clearly represent the answer.\n"
        "The explanation must be one short sentence that mentions the answer.\n"
        "Use a concept from the study material, not from the format example.\n"
        + _avoid_text(avoid_concepts)
        + "\nNo Markdown.\nNo extra text.\n\n"
        "Format example:\n" + example + "\n\n"
        "Study material:\n" + trimmed + "\n"
    )


# ---------------------------------------------------------------------------
# Cleaning + parsing
# ---------------------------------------------------------------------------

def _clean_json_text(text):
    """
    Clean a raw model reply before parsing:
      1. remove <think>...</think> reasoning blocks
      2. remove ```json / ``` Markdown fences
      3. trim whitespace

    Returns the cleaned string, or None if `text` is not a string.
    (Locating the JSON object itself is done safely in _load_json_object.)
    """
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```[ \t]*(?:json|JSON)?", "", cleaned)
    return cleaned.strip()


def _decode_first_objects(text):
    """
    Yield every complete JSON object found in `text`, scanning left to
    right. Uses raw_decode, so leading/trailing prose (even prose that
    contains braces) cannot break extraction. Never uses eval().
    """
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            return
        try:
            obj, end = _JSON_DECODER.raw_decode(text, start)
        except (ValueError, TypeError):
            pos = start + 1
            continue
        if isinstance(obj, dict):
            yield obj
        pos = max(end, start + 1)


def _looks_like_round(obj):
    """True if the dict has at least one recognisable round key."""
    keys = {_normalize_key(k) for k in obj}
    wanted = {alias for aliases in _FIELD_ALIASES.values() for alias in aliases}
    return bool(keys & wanted)


def _load_json_object(text):
    """
    Clean the reply, then safely extract and parse the JSON object.
    Returns a dict or None. Never raises, never uses eval().
    """
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        for item in text:
            if isinstance(item, dict):
                return item
        return None

    cleaned = _clean_json_text(text)
    if not cleaned:
        return None

    # Small models occasionally leave a trailing comma before } or ] - try
    # the text as-is first, then with those commas removed.
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", cleaned)
    for candidate in (cleaned, without_trailing_commas):
        first_dict = None
        for obj in _decode_first_objects(candidate):
            if _looks_like_round(obj):
                return obj
            if first_dict is None:
                first_dict = obj
        if first_dict is not None:
            return first_dict
    return None


# ---------------------------------------------------------------------------
# Duplicate-avoidance / matching helpers
# ---------------------------------------------------------------------------

def _normalize_concept(text):
    """
    Lowercase, strip punctuation, collapse spaces - for exact-duplicate
    checks.

    STABLE INTERFACE: steps/step21_practice.py (the Practice hub) calls
    this function directly (for lenient free-text answer matching). Do
    not change its name or signature without also updating that file.
    """
    normalized = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return " ".join(normalized.split())


def _significant_words(text):
    """
    Normalized, stopword-free word set used to compare two pieces of text.

    STABLE INTERFACE: steps/step21_practice.py (the Practice hub) calls
    this function directly (for lenient free-text answer matching). Do
    not change its name or signature without also updating that file.
    """
    normalised = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {w for w in normalised.split() if w and w not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Answer shortening — the answer must never be longer than 3 words.
# ---------------------------------------------------------------------------

def _clean_answer_text(text):
    """
    Trim whitespace and remove unnecessary punctuation from an answer.
    Apostrophes and hyphens inside words are kept (e.g. "Newton's Laws").
    """
    cleaned = re.sub(r"[^\w\s'\u2019\-]", " ", str(text or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned.strip(" \t\"'\u2019-")


def _shorten_answer(answer, concept=""):
    """
    Return an answer of at most MAX_ANSWER_WORDS words.

    1. Trim whitespace (and stray quotes / end punctuation).
    2. If it already fits, keep it as is.
    3. Otherwise use the concept if that is short enough.
    4. Otherwise keep the first few meaningful words of the answer.
    No extra Ollama call is made.
    """
    text = _clean_answer_text(answer)
    words = text.split()
    if len(words) <= MAX_ANSWER_WORDS:
        return text

    concept_text = _clean_answer_text(concept)
    if concept_text and len(concept_text.split()) <= MAX_ANSWER_WORDS:
        return concept_text

    meaningful = [w for w in words if w.lower().strip(".,;:!?") not in _STOPWORDS]
    chosen = meaningful[:MAX_ANSWER_WORDS] or words[:MAX_ANSWER_WORDS]
    return " ".join(chosen).strip(" \t\"'.,;:!?")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _normalize_key(key):
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _as_text(value, join_with=" "):
    """
    Convert a scalar (or a flat list of scalars) to a trimmed string.
    Returns "" for None, dicts, nested lists or placeholder words.
    """
    if value is None or isinstance(value, (dict, bool)):
        return ""
    if isinstance(value, (list, tuple)):
        if not all(isinstance(v, (str, int, float)) for v in value):
            return ""
        text = join_with.join(str(v).strip() for v in value)
    else:
        text = str(value)
    text = text.strip()
    if text.lower() in _PLACEHOLDER_VALUES:
        return ""
    return text


def _collect_fields(data):
    """
    Map the model's dict onto the canonical field names (case-insensitive,
    alias-aware). Unwraps {"round": {...}} style wrappers. Returns a dict
    of canonical_name -> raw value, or None if data is unusable.
    """
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None)
    if not isinstance(data, dict):
        return None

    # Unwrap a single nested object, e.g. {"round": {...}}.
    if not _looks_like_round(data) and len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, (dict, list)):
            return _collect_fields(only)

    by_key = {_normalize_key(k): v for k, v in data.items()}
    found = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in by_key and _as_text(by_key[alias], "" if canonical == "emoji" else " "):
                found[canonical] = by_key[alias]
                break
    return found


def _extract_emoji(text):
    """
    Keep only emoji-type characters (symbols, ZWJ, variation selectors,
    keycap/skin-tone modifiers). Letters, digits, spaces and punctuation
    are dropped. Returns (emoji_string, symbol_count).
    """
    kept = []
    symbols = 0
    for ch in text:
        if ch in "\u200d\ufe0f\u20e3":
            kept.append(ch)
            continue
        category = unicodedata.category(ch)
        if category in ("So", "Sk"):
            kept.append(ch)
            if category == "So":
                symbols += 1
    return "".join(kept), symbols


def _explanation_supports_answer(answer, explanation):
    """
    Cheap, deterministic support check: the explanation must be a real
    sentence (3+ words) and must mention the answer - at least one
    significant word of the answer (matched on its first 5 letters so
    "learn" / "learning" / "learned" all count).
    """
    if len(explanation.split()) < 3:
        return False
    answer_stems = {w[:5] for w in _significant_words(answer)}
    if not answer_stems:
        return True
    explanation_stems = {w[:5] for w in _significant_words(explanation)}
    return bool(answer_stems & explanation_stems)


def _validate_challenge(data, allow_shorten=False):
    """
    Return a validated round dict, or None if anything is missing/invalid.

    The returned dict has the new keys (emoji, question, answer, hint,
    explanation) AND the legacy keys used by step21_practice.py
    (concept, emoji_sequence).

    allow_shorten=True (used on the final attempt only) shortens an answer
    that is over 3 words instead of rejecting the round.
    """
    fields = _collect_fields(data)
    if not fields:
        return None

    question = _as_text(fields.get("question"))
    hint = _as_text(fields.get("hint"))
    explanation = _as_text(fields.get("explanation"))
    answer_raw = _as_text(fields.get("answer"))
    emoji_raw = _as_text(fields.get("emoji"), join_with="")
    concept_raw = _as_text(fields.get("concept"))

    # 1. all five fields present and non-empty
    if not (emoji_raw and question and answer_raw and hint and explanation):
        return None

    # 2. emoji must really be emoji (stray letters are stripped out)
    emoji, symbol_count = _extract_emoji(emoji_raw)
    if symbol_count < 1 or symbol_count > MAX_EMOJI_SYMBOLS:
        return None

    # 3. answer: max 3 words
    answer = _clean_answer_text(answer_raw)
    if not answer:
        return None
    if len(answer.split()) > MAX_ANSWER_WORDS:
        if not allow_shorten:
            return None
        answer = _shorten_answer(answer_raw, concept_raw)
        if not answer or len(answer.split()) > MAX_ANSWER_WORDS:
            return None

    # 4. answer must not just repeat the question
    if _normalize_concept(answer) == _normalize_concept(question):
        return None

    # 5. explanation must support (mention) the answer
    if not _explanation_supports_answer(answer, explanation):
        return None

    concept = _clean_answer_text(concept_raw)
    if not concept or len(concept.split()) > 12:
        concept = answer

    return {
        "emoji": emoji,
        "question": question,
        "answer": answer,
        "hint": hint,
        "explanation": explanation,
        # legacy keys still read by steps/step21_practice.py
        "concept": concept,
        "emoji_sequence": emoji,
    }


# ---------------------------------------------------------------------------
# Generation — ONE Ollama call per attempt, ONE round at a time.
# ---------------------------------------------------------------------------

def _generate_emoji_challenge(content, used_concepts):
    """
    Try up to MAX_GENERATION_ATTEMPTS times to generate ONE valid round.
    Each attempt is exactly ONE Ollama call:
      attempt 1  -> concise prompt
      attempt 2+ -> shorter retry prompt (smaller study-material slice)
    A failed or malformed Ollama call counts as a failed attempt and is
    retried - it never raises out of this function.

    Returns a validated round dict, or None if every attempt failed.

    STABLE INTERFACE: steps/step21_practice.py calls this function
    directly with exactly this 2-argument signature. Do not change the
    name, argument order, or count without also updating that file.
    """
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        is_retry = attempt > 0
        is_last = attempt == MAX_GENERATION_ATTEMPTS - 1

        prompt = _build_emoji_prompt(content, used_concepts, retry=is_retry)
        try:
            result_text = generate_content(prompt)  # <- the ONLY function used to call Ollama
        except Exception:
            # Timeout, connection error, etc. = failed attempt, not a crash.
            continue

        try:
            data = _load_json_object(result_text)
            challenge = _validate_challenge(data, allow_shorten=is_last)
        except Exception:
            # Parsing/validation must never crash the app.
            challenge = None

        if challenge is not None:
            return challenge

    return None


# ---------------------------------------------------------------------------
# Standalone page (embedded=False) — generate / hint / show answer / submit.
# ---------------------------------------------------------------------------

def _init_session_state():
    defaults = {
        "emoji_challenge": None,
        "emoji_hint_visible": False,
        "emoji_answer_revealed": False,
        "emoji_answer_shown": False,
        "emoji_used_concepts": [],
        "emoji_round_id": 0,
        "emoji_submitted": False,
        "emoji_submitted_text": "",
        "emoji_last_correct": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _is_correct_answer(user_text, answer):
    """Lenient free-text match between the student's answer and the answer."""
    user_norm = _normalize_concept(user_text)
    answer_norm = _normalize_concept(answer)
    if not user_norm or not answer_norm:
        return False
    if user_norm == answer_norm:
        return True
    user_words = _significant_words(user_text)
    answer_words = _significant_words(answer)
    if not user_words or not answer_words:
        return False
    return answer_words <= user_words


def _request_new_challenge(content):
    """
    Fetch a brand-new round from Ollama and reset the game state.
    Returns True on success, False on failure (so the caller only reruns
    on success and the error message stays visible).

    Only runs when the student clicks a Generate button - never on a plain
    rerun. On failure only CHALLENGE_FAILURE_MESSAGE is shown.
    """
    with st.spinner("Creating an emoji challenge from your study material..."):
        try:
            challenge = _generate_emoji_challenge(content, st.session_state.emoji_used_concepts)
        except Exception:
            challenge = None

        if challenge is None:
            st.error(f"⚠️ {CHALLENGE_FAILURE_MESSAGE}")
            return False

        st.session_state.emoji_challenge = challenge
        st.session_state.emoji_hint_visible = False
        st.session_state.emoji_answer_revealed = False
        st.session_state.emoji_answer_shown = False
        st.session_state.emoji_submitted = False
        st.session_state.emoji_submitted_text = ""
        st.session_state.emoji_last_correct = False
        # New widget key per round so the typed-answer box starts empty.
        st.session_state.emoji_round_id += 1
        st.session_state.emoji_used_concepts.append(challenge["concept"])
        return True


def _show_standalone(content):
    challenge = st.session_state.emoji_challenge

    # ---- No round loaded yet: show the generate button ----
    if challenge is None:
        if st.button("😀 Generate New Emoji Challenge", type="primary"):
            if _request_new_challenge(content):
                st.rerun()
        return

    # ---- A round is loaded ----
    with st.container(border=True):
        # The question is shown here (the answer is never shown before the
        # student asks for it).
        st.markdown(f"#### 💡 {challenge['question']}")
        st.markdown(
            f"<div style='font-size:2rem; text-align:center; padding: 0.5rem 0;'>"
            f"{challenge['emoji']}</div>",
            unsafe_allow_html=True,
        )

        hint_col, reveal_col = st.columns(2)

        with hint_col:
            hint_label = "🙈 Hide Hint" if st.session_state.emoji_hint_visible else "👁️ Show Hint"
            if st.button(hint_label, use_container_width=True):
                st.session_state.emoji_hint_visible = not st.session_state.emoji_hint_visible
                st.rerun()

        with reveal_col:
            if not st.session_state.emoji_answer_revealed:
                if st.button("🔍 Reveal Answer", use_container_width=True, type="primary"):
                    st.session_state.emoji_answer_revealed = True
                    st.rerun()

        if st.session_state.emoji_hint_visible and not st.session_state.emoji_answer_revealed:
            st.info(f"💡 Hint: {challenge['hint']}")

        # ---- Show Answer: shows only the short (max 3 words) answer. ----
        short_answer = _shorten_answer(challenge["answer"], challenge["concept"])
        if not st.session_state.emoji_answer_revealed:
            if st.session_state.get("emoji_answer_shown", False):
                st.markdown(f"**Answer:** `{short_answer}`")
            elif st.button("👁️ Show Answer", use_container_width=True, key="emoji_show_answer"):
                st.session_state.emoji_answer_shown = True
                st.rerun()

        # ---- Student's own answer + Submit Answer. Stays available even
        # after Show Answer has been clicked. ----
        answer_key = f"emoji_user_answer_{st.session_state.emoji_round_id}"
        if not st.session_state.emoji_submitted:
            st.text_input("Your answer", key=answer_key, placeholder="Type your answer here")
            if st.button("✅ Submit Answer", use_container_width=True, key="emoji_submit_answer"):
                typed = str(st.session_state.get(answer_key, "")).strip()
                if not typed:
                    st.warning("Please type an answer before submitting.")
                else:
                    st.session_state.emoji_submitted_text = typed
                    st.session_state.emoji_last_correct = _is_correct_answer(typed, challenge["answer"])
                    st.session_state.emoji_submitted = True
                    st.rerun()
        else:
            typed = st.session_state.emoji_submitted_text
            if st.session_state.emoji_last_correct:
                st.success(f"✅ Correct! Your answer: {typed}")
            else:
                st.error(f"❌ Not quite. Your answer: {typed}")
                st.markdown(f"**Answer:** `{short_answer}`")
            if not st.session_state.emoji_answer_revealed:
                st.markdown(f"**Explanation:** {challenge['explanation']}")

        if st.session_state.emoji_answer_revealed:
            st.markdown("---")
            st.markdown("### Decoded Concept Solution")
            st.markdown(f"**Concept:** {challenge['concept']}")
            st.markdown(f"**Answer:** `{short_answer}`")
            st.markdown(f"**Explanation:** {challenge['explanation']}")

    st.markdown("")

    # ---- New round control (after the answer was revealed or submitted) ----
    if st.session_state.emoji_answer_revealed or st.session_state.emoji_submitted:
        if st.button("😀 Generate New Challenge"):
            if _request_new_challenge(content):
                st.rerun()


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def show_emoji_decoder(embedded=False):
    """
    Display the standalone Emoji Decoder page.

    NOTE: the Practice hub (steps/step21_practice.py) does NOT call this
    function - it calls _generate_emoji_challenge()/_normalize_concept()/
    _significant_words() directly. `embedded` only skips this page's own
    header/topic bar.
    """
    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("No study material generated yet. Please generate study material from the Dashboard first.")
        return

    subject = study_material["subject"]
    topic = study_material["topic"]
    content = study_material["content"]

    if not embedded:
        show_page_header("😀", "Emoji Decoder", "Decode the concept from the emojis!")
        show_topic_bar(subject, topic)
        st.markdown("---")

    _init_session_state()
    _show_standalone(content)