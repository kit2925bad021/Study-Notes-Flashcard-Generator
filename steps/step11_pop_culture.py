"""
Step 11 — Pop Culture Connections (Tamil Cinema Lens)
-------------------------------------------------------
Renders the "🎬 Pop Culture Connections" page inside the Streamlit app.

This page explains the academic concept from the student's study material
using a familiar, real Tamil movie as a memory-aid analogy. The academic
explanation is always complete and correct on its own — the movie is only
a memory aid, never a replacement for the real academic content.

FLOW enforced end to end (prompt -> parsing -> validation -> UI):

    study material -> identify the actual academic concept -> write the
    full, correct academic explanation -> find ONE real, well-known Tamil
    movie that has a genuine, meaningful connection to that concept ->
    describe the movie-level idea/situation (never an invented scene or
    quote) -> connect it back to the academic explanation -> a short
    learning takeaway.

Gemini must return ONLY this JSON shape (all 8 fields required, every
value a non-empty, non-placeholder string):

    {
      "subject": "...",
      "topic": "...",
      "concept": "...",
      "academic_explanation": "...",
      "pop_culture_type": "Tamil Movie",
      "reference": "...",
      "connection": "...",
      "learning_takeaway": "..."
    }

The model is also asked for ONE optional extra string,
"topic_movie_connection" (why this movie fits this topic: academic topic
-> movie idea -> learning connection), shown under
"🎬 How This Topic Connects to the Movie". It is NOT one of the required
fields: validation, retry logic and the 8-field JSON check are unchanged,
so a reply without it is accepted exactly as before (the section then
falls back to a short line built from the existing fields).

--------------------------------------------------------------------
WHY THE OLD PARSING WAS UNRELIABLE (root-cause notes, kept here on
purpose so a future change doesn't reintroduce the same two bugs):
--------------------------------------------------------------------
1. EXTRACTION BUG. The previous parser took "the first '{' to the last
   '}' in the whole response" as the JSON to decode. Small local models
   very often "show their work" by echoing the example JSON template
   from the prompt (with "..." placeholders) before writing their real,
   filled-in answer below it. That means the response actually contains
   TWO '{...}' objects — and grabbing from the first '{' to the last '}'
   splices them into one unparseable blob, which json.loads() correctly
   rejects. This was previously misdiagnosed as "the model's JSON is
   unreliable" when the model's real answer was often fine on its own.

   FIX: _iter_json_objects() below is a string/escape-aware, brace-depth
   scanner that yields every complete top-level {...} object found in
   the text, in order. _parse_pop_culture_response() then tries each
   candidate and keeps the LAST one that is structurally complete
   (a chatty model's real, final answer normally comes after any echoed
   template or reasoning) rather than blindly spanning every candidate.

2. OVER-STRICT VALIDATION BUG. A structurally complete, well-written
   response could still be thrown away by _is_academic_explanation_grounded()
   — a keyword-overlap heuristic that was being used as a hard, blocking
   gate. A correct explanation that simply paraphrases the material
   (using different but accurate wording) can legitimately use few of
   the material's most-frequent literal words, so this heuristic could
   reject a perfectly good answer, trigger a wasted retry, and then
   reject the retry's answer for the same soft reason — landing on the
   raw-text fallback even though nothing was actually structurally
   wrong.

   FIX: grounding is still computed, but only as an informational signal
   (visible in the debug expander on the page) — it can no longer, by
   itself, discard a structurally valid, complete response. The retry
   now only fires on a genuine structural failure (no valid object
   found, or required fields missing/empty), not on this soft heuristic.

Both existing prompts were also tightened (same two prompts — no third
prompt was added) to explicitly tell the model not to echo the template
and not to use raw double quotes inside string values, since that is the
other common source of decode failures. No new Gemini client, no direct
`google.genai` import, and no hardcoded model name were introduced — this file
still calls only the existing services.gemini_service.generate_content().

3. TWO MORE DECODE-FAILURE MODES were confirmed and are now tolerated in
   _decode_json_object() (same "decode_failed" stage as before — this is
   about making that single decode step itself more forgiving, not a new
   pipeline stage):
     - a literal, unescaped newline/tab typed inside a string value
       instead of "\\n"/"\\t" — very common when a local model writes a
       multi-sentence academic_explanation. json.loads' default
       strict=True rejects any raw control character inside a string;
       strict=False allows it.
     - "smart"/curly quotes ("..."/'...') used as the JSON delimiters
       themselves instead of straight quotes — some local models default
       to typographic quotes in ordinary prose mode. These are now
       normalized to straight quotes as one of the decode attempts.
--------------------------------------------------------------------

Wire this up in app.py the same way Steps 1, 2, 9, and 10 are wired up:

    from steps.step11_pop_culture import show_pop_culture
    ...
    elif st.session_state.current_page == "🎬 Pop Culture Connections":
        show_pop_culture()

This file does NOT modify st.session_state.study_material, flashcards,
quiz data, or any other step. It only reads study_material and writes to
its own session_state keys (all prefixed with "pop_culture").

Gemini is only ever called from inside the "Explain / Regenerate" button
handler below — never during a normal Streamlit rerun.
"""

import json
import re
from collections import Counter
from html import escape

import streamlit as st

from services.gemini_service import generate_content
from steps.step19_ui import require_study_material, show_page_header, show_topic_bar


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

# The exact 8 keys Gemini must return, in the exact shape described above.
# Every value must be a non-empty, non-placeholder string.
REQUIRED_FIELDS = [
    "subject",
    "topic",
    "concept",
    "academic_explanation",
    "pop_culture_type",
    "reference",
    "connection",
    "learning_takeaway",
]

# Optional 9th key: why this movie fits this topic. Deliberately NOT in
# REQUIRED_FIELDS, so validation and retry behave exactly as before.
TOPIC_MOVIE_CONNECTION_FIELD = "topic_movie_connection"

# Fixed literal for "pop_culture_type" — this feature only ever produces a
# Tamil-movie connection, so the value is normalized to this exact string
# after a successful parse, regardless of small wording/casing differences
# a local model might return.
POP_CULTURE_TYPE = "Tamil Movie"

# If the model echoes the prompt's example schema, its placeholder values
# look like these — a field holding one of these is NOT real content, even
# though it is technically a non-empty string.
_PLACEHOLDER_VALUES = {"...", "…", ""}

# How many recently-used movie references to remember, so the model is
# nudged toward variety instead of picking the same film every time.
REFERENCE_HISTORY_LIMIT = 6


# ----------------------------------------------------------------------
# 1. KEY-TERM GROUNDING (informational only — see root-cause notes above)
# ----------------------------------------------------------------------
# A small local model can drift into a generic, templated answer that
# barely touches the actual uploaded material. To flag (not reject) that,
# we pull a handful of the material's own recurring, meaningful words and
# check whether the academic_explanation the model produced uses them.

_STOPWORDS = set(
    """
    a an the of and or but if then else for to in on at by with from as is
    are was were be been being this that these those it its into your you
    student college notes study material subject topic chapter which who
    whom what when where why how not no yes can will would should could
    also each every some any all many much more most other another such
    only own same so than too very just about above below over under
    between within without across during before after again further once
    here there their they them theirs he she his her we our us movie
    tamil cinema film scene character
    """.split()
)


def _extract_key_terms(content, max_terms=10):
    """Cheap, dependency-free extraction of the material's own recurring,
    meaningful words — used only to sanity-check that the academic
    explanation is plausibly about this material."""
    if not content:
        return []

    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", content)
    if not words:
        return []

    counts = Counter()
    first_index = {}
    original_case = {}
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        counts[lw] += 1
        if lw not in first_index:
            first_index[lw] = i
            original_case[lw] = w

    if not counts:
        return []

    ranked = sorted(counts.keys(), key=lambda lw: (-counts[lw], first_index[lw]))
    return [original_case[lw] for lw in ranked[:max_terms]]


def _is_academic_explanation_grounded(data, content):
    """True if the model's academic_explanation uses enough of the
    material's own vocabulary to plausibly be about THIS material.
    INFORMATIONAL ONLY — see root-cause notes at the top of this file for
    why this must never be used to reject an otherwise valid response."""
    key_terms = _extract_key_terms(content)
    if not key_terms:
        return True  # too little material to extract terms from

    explanation = str(data.get("academic_explanation", "")) if data else ""
    text_lower = explanation.lower()
    matches = sum(1 for term in key_terms if term.lower() in text_lower)
    required = min(max(2, len(key_terms) // 4), len(key_terms))
    return matches >= required


# ----------------------------------------------------------------------
# 2. PROMPT BUILDERS
# ----------------------------------------------------------------------

def _build_prompt(subject, topic, content, avoid_references):
    """
    Main prompt: work out the concept and the correct, complete academic
    explanation from the study material FIRST, then find ONE genuinely
    relevant, real Tamil movie to use as a memory-aid analogy.
    """

    avoid_text = ""
    if avoid_references:
        avoid_list = ", ".join(avoid_references)
        avoid_text = (
            "\nTo keep things varied, try to avoid reusing these recently "
            f"used references if a different well-known Tamil movie fits "
            f"just as well: {avoid_list}.\n"
        )

    return f"""
You are helping a Tamil/Indian college student understand an academic
concept by connecting it to a Tamil movie they are likely to know.

Subject: {subject}
Topic: {topic}

Study material (this is the ONLY source of academic facts you may use —
do not invent facts that are not supported by it):
{content}

Follow this exact order of reasoning before you answer:
1. Identify the actual academic concept that this study material is
   teaching.
2. Write a complete, correct, standalone academic explanation of that
   concept, based ONLY on the study material above. It must make full
   sense on its own even if every movie reference were removed — the
   movie is only a memory aid, never a substitute for the real answer.
3. Only AFTER step 2, think of ONE real, well-known Tamil movie that has
   a genuine, meaningful connection to this concept (for example:
   Artificial Intelligence / Robotics -> Enthiran's robot; a topic about
   teamwork or friendship -> a well-known Tamil movie where a group
   succeeds by working together). Choose the concept first and fit the
   movie to it — never the other way round. Do NOT force in a random or
   unrelated movie just to have one — the connection must be genuine.
4. Describe the movie-level idea, character, or situation that connects
   to the concept, in your own words. If you are not fully sure of an
   exact scene or line, describe the general, well-known premise instead
   — never invent a specific scene, character, or quotation you are not
   sure about.
5. Explain clearly how that movie idea maps onto the academic concept.
6. Write one short, practical takeaway that helps the student remember
   the concept using this connection.
{avoid_text}
Rules:
- Use ONLY a real, well-known Tamil movie — never an invented title.
- Do NOT quote movie dialogue, even briefly — paraphrase in your own
  words, and never claim an exact scene/line you are not sure about.
- The academic explanation must be correct, complete, and fully
  understandable without the movie reference.
- Keep the language simple, clear, and beginner-friendly for a college
  student.
- Avoid offensive, political, or otherwise inappropriate content.
- Never change the factual meaning of the study material.
- "topic_movie_connection" must be 2-3 short sentences explaining WHY this
  movie was chosen for this topic and HOW it helps the student understand
  it, in the order: academic topic -> the movie's general idea, character,
  technology or theme -> what that helps the student picture or
  understand. Do NOT just retell the movie's plot, and do NOT invent
  scenes, dialogue or events. Style example only (do not copy it): "The
  topic of Artificial Intelligence is connected to Enthiran because the
  movie revolves around an intelligent humanoid robot. It gives a familiar
  picture of machines performing tasks that need human-like intelligence."

Respond with ONLY ONE valid JSON object — nothing else. Do NOT repeat or
echo the empty template shown below in your answer, and do NOT show your
reasoning steps as text — think through steps 1-6 silently, then output
only the final, filled-in JSON object, using EXACTLY these 9 keys:

{{
  "subject": "{subject}",
  "topic": "{topic}",
  "concept": "the specific academic concept from the study material",
  "academic_explanation": "the correct, complete explanation, based only on the study material, standing fully on its own without the movie",
  "pop_culture_type": "Tamil Movie",
  "reference": "the specific, real Tamil movie (and character/element if relevant)",
  "connection": "how the movie idea connects to the academic concept",
  "topic_movie_connection": "2-3 short sentences: why this movie fits this topic and how it helps the student understand it (topic -> movie idea -> learning connection)",
  "learning_takeaway": "a short, memorable takeaway for the student"
}}

Every one of these 9 values must be a real, non-empty string (never the
literal text "..." from this template). Do not use double-quote (") marks
inside any of the text values — if you need to quote something, use
single quotes (') instead, since a double quote inside a value breaks
the JSON.
"""


def _build_prompt_simple(subject, topic, content, avoid_references):
    """
    Shorter, stricter fallback prompt used for the ONE retry when the
    first attempt doesn't come back as a valid, complete JSON object.
    Small local models sometimes need a much stricter, shorter
    instruction to comply.
    """

    avoid_text = f" Avoid reusing: {', '.join(avoid_references)}." if avoid_references else ""

    return f"""
Study material (this is the ONLY source of facts — subject: {subject},
topic: {topic}):
{content}

First identify the concept and write a complete, correct academic
explanation using ONLY the study material above — it must stand fully on
its own. Only after that, pick ONE real, well-known Tamil movie that
genuinely connects to that concept — concept first, movie second. Do not
quote dialogue; describe the general, well-known premise if you are not
sure of an exact scene.{avoid_text}

Output ONLY one JSON object and nothing else — no explanations, no
markdown, no code fences, and do NOT show this template itself in your
answer. Fill in all 9 values as real, non-empty strings (never leave the
literal "..." below), and never use a double-quote (") mark inside a
value — use single quotes (') instead:
{{
  "subject": "{subject}",
  "topic": "{topic}",
  "concept": "...",
  "academic_explanation": "...",
  "pop_culture_type": "Tamil Movie",
  "reference": "...",
  "connection": "...",
  "topic_movie_connection": "2-3 short sentences on why this movie fits this topic and how it helps understand it - no invented scenes",
  "learning_takeaway": "..."
}}
"""


# ----------------------------------------------------------------------
# 3. STAGE-BY-STAGE PARSING + VALIDATION
# ----------------------------------------------------------------------
# See the root-cause notes at the top of this file for exactly why this
# replaces the old "first '{' to last '}'" + single json.loads() attempt.

# Matches a ```json / ``` fence marker anywhere in the text, not only at
# the very start — a response can have a sentence before a fenced block.
_CODE_FENCE_PATTERN = re.compile(r"```[a-zA-Z]*")

# A trailing comma right before a closing brace/bracket ("..., }" or
# "..., ]") is a common small-model glitch that json.loads rejects outright
# even though the content is otherwise perfectly readable.
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")

# "Smart"/curly quotes used as the JSON delimiters themselves (some local
# models default to typographic quotes) — json.loads can't recognize
# these as string boundaries at all, so they're normalized to straight
# quotes as one of the decode variants tried below.
_SMART_QUOTE_MAP = str.maketrans({
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
})


def _iter_json_objects(text):
    """
    Yield every complete, top-level '{ ... }' substring found in text, in
    the order they appear, using a brace-depth scan that is aware of
    quoted strings (so a brace or quote *inside* a string value never
    throws off the boundaries of the object itself).

    This is what fixes the extraction bug described at the top of this
    file: if the model echoes the empty example template and then writes
    its real, filled-in answer below it, this yields BOTH objects
    separately instead of splicing them into one unparseable blob.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            start = i
            depth = 0
            in_string = False
            escape = False
            j = i
            closed = False
            while j < n:
                ch = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            yield text[start:j + 1]
                            i = j
                            closed = True
                            break
                j += 1
            if not closed:
                # Ran off the end of the text with an unbalanced object —
                # nothing further can be reliably scanned.
                return
        i += 1


def _decode_json_object(candidate):
    """
    Try to decode one candidate '{...}' substring, trying several
    increasingly-tolerant variants — small local models routinely:
      - leave a stray trailing comma before a closing '}'/']',
      - put a literal, unescaped newline/tab inside a string value
        instead of writing \\n/\\t (json.loads' default strict=True
        rejects any raw control character inside a string),
      - use "smart"/curly quotes ("..."/'...') as the JSON delimiters
        themselves instead of straight quotes.
    Returns a dict, or None if no variant can be decoded.
    """
    variants = [candidate]

    trailing_fixed = _TRAILING_COMMA_PATTERN.sub(r"\1", candidate)
    if trailing_fixed != candidate:
        variants.append(trailing_fixed)

    smart_fixed = candidate.translate(_SMART_QUOTE_MAP)
    if smart_fixed != candidate:
        variants.append(smart_fixed)
        smart_trailing_fixed = _TRAILING_COMMA_PATTERN.sub(r"\1", smart_fixed)
        if smart_trailing_fixed != smart_fixed:
            variants.append(smart_trailing_fixed)

    for attempt_text in variants:
        for strict in (True, False):
            try:
                data = json.loads(attempt_text, strict=strict)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, dict):
                return data

    return None


def _is_real_value(value):
    """A required field must be a non-empty string that isn't just the
    literal placeholder text left over from an echoed template."""
    return isinstance(value, str) and value.strip() and value.strip() not in _PLACEHOLDER_VALUES


def _classify_object(data):
    """
    Structural validation of one already-decoded dict. Returns
    (stage, detail):
      - stage "missing_fields": one or more of the 8 required keys is
        absent entirely from the dict.
      - stage "empty_values": every key is present, but one or more
        values is empty, not a string, or a leftover template placeholder
        (e.g. the literal "...").
      - stage "ok": structurally complete and ready to display.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return "missing_fields", f"Missing field(s): {', '.join(missing)}."

    bad = [f for f in REQUIRED_FIELDS if not _is_real_value(data.get(f))]
    if bad:
        return "empty_values", f"Empty, non-string, or placeholder value(s): {', '.join(bad)}."

    return "ok", ""


def _parse_pop_culture_response(raw):
    """
    Parse + structurally validate Gemini's response for the Pop Culture
    Lens, stage by stage, so a failure can be attributed to the EXACT
    stage that caused it instead of one generic "couldn't parse" bucket.

    Stages (matching the pipeline: generate_content -> raw response ->
    JSON extraction -> json.loads -> schema validation -> returned
    connection):
      A "no_response"       — Gemini returned unusable/empty output.
      B "extraction_failed" — no '{ ... }' object could be found at all.
      C "decode_failed"     — one or more '{ ... }'-shaped blocks were
                              found, but none of them is valid JSON.
      D "missing_fields"    — valid JSON, but required key(s) are absent.
      E "empty_values"      — valid JSON with all keys present, but one
                              or more values is empty/not a
                              string/a leftover template placeholder.
      "ok"                  — a complete, usable connection.

    When MULTIPLE '{...}' objects are found (e.g. the model echoed the
    empty example template before its real answer), the LAST one that is
    fully "ok" is used — a chatty model's real, final answer normally
    comes after any echoed template or reasoning. If none is fully "ok",
    the last candidate's own failure stage/detail is reported, since
    that is the one most likely to be the model's intended final answer.

    Returns {"stage": ..., "data": dict-or-None, "detail": "..."}.
    Never raises.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {"stage": "no_response", "data": None, "detail": "Gemini returned an empty response."}

    cleaned = _CODE_FENCE_PATTERN.sub("", raw).strip()

    candidates = list(_iter_json_objects(cleaned))
    if not candidates:
        return {
            "stage": "extraction_failed",
            "data": None,
            "detail": "No '{ ... }' JSON object was found anywhere in the response.",
        }

    classified = []  # (dict, stage, detail) for every candidate that decoded
    for candidate in candidates:
        obj = _decode_json_object(candidate)
        if obj is None:
            continue
        stage, detail = _classify_object(obj)
        classified.append((obj, stage, detail))

    if not classified:
        return {
            "stage": "decode_failed",
            "data": None,
            "detail": f"Found {len(candidates)} JSON-like block(s), but none of them decoded as valid JSON.",
        }

    # Prefer the LAST fully valid object.
    for obj, stage, detail in reversed(classified):
        if stage == "ok":
            return {"stage": "ok", "data": obj, "detail": ""}

    # Nothing fully qualified — report the last candidate's own reason,
    # since that is the one most likely to be the model's real answer.
    _, stage, detail = classified[-1]
    return {"stage": stage, "data": None, "detail": detail}


# Any double-quoted run of 60+ characters is treated as a likely verbatim
# quote (e.g. movie dialogue) and swapped for a safe paraphrase, as a
# defensive net on top of the "never quote dialogue" prompt rule.
_LONG_QUOTE_PATTERN = re.compile(r'"([^"]{60,})"')


def _defang_long_quotes(text):
    return _LONG_QUOTE_PATTERN.sub('"a well-known moment from the movie"', text)


def _sanitize_data(data):
    """Apply the long-quote guard to the free-text fields, and normalize
    pop_culture_type to the one fixed value this feature ever produces —
    regardless of small wording/casing differences a model might return —
    before we store/display the result."""
    sanitized = dict(data)

    for field in ("concept", "academic_explanation", "reference", "connection", "learning_takeaway",
                  TOPIC_MOVIE_CONNECTION_FIELD):
        value = sanitized.get(field)
        if isinstance(value, str):
            sanitized[field] = _defang_long_quotes(value)

    sanitized["pop_culture_type"] = POP_CULTURE_TYPE

    return sanitized


# ----------------------------------------------------------------------
# 4. REFERENCE HISTORY (so we nudge the model away from repeats)
# ----------------------------------------------------------------------

def _remember_reference(reference):
    if not reference or not reference.strip():
        return
    key = reference.strip()

    history = st.session_state.get("pop_culture_reference_history", [])
    # Drop any existing entry for the same reference, then re-add at the
    # end so the history stays ordered oldest -> most-recent.
    history = [r for r in history if r.lower() != key.lower()]
    history.append(key)
    st.session_state.pop_culture_reference_history = history[-REFERENCE_HISTORY_LIMIT:]


# ----------------------------------------------------------------------
# 5. GENERATION (with one retry ONLY on a genuine structural failure)
# ----------------------------------------------------------------------

def _generate_pop_culture_explanation(subject, topic, content):
    """
    Build the prompt, call Gemini, parse + structurally validate the
    response. Retries ONCE, with a shorter/stricter prompt, but ONLY when
    the first attempt failed for a real structural reason (stages
    no_response / extraction_failed / decode_failed / missing_fields /
    empty_values) — never because of the soft, informational grounding
    heuristic, which used to cause a valid response to be discarded and
    replaced by a second call that could fail the exact same way. Returns:

        {
            "data": {...} or None,
            "raw_text": "...",         # last raw Gemini response, for fallback display
            "debug_stage": "...",      # exact stage the LAST attempt landed on
            "debug_detail": "...",     # human-readable reason for that stage
        }

    Any exception from generate_content() is left to the caller
    (show_pop_culture) to catch and display gracefully — this function
    makes at most 2 Gemini calls, both from within a single button click.
    """
    avoid_references = st.session_state.get("pop_culture_reference_history", [])

    prompt = _build_prompt(subject, topic, content, avoid_references)
    raw_text = generate_content(prompt)
    parsed = _parse_pop_culture_response(raw_text)

    if parsed["stage"] != "ok":
        simple_prompt = _build_prompt_simple(subject, topic, content, avoid_references)
        raw_text = generate_content(simple_prompt)
        parsed = _parse_pop_culture_response(raw_text)

    if parsed["stage"] != "ok":
        return {
            "data": None,
            "raw_text": raw_text,
            "debug_stage": parsed["stage"],
            "debug_detail": parsed["detail"],
        }

    data = _sanitize_data(parsed["data"])
    _remember_reference(data.get("reference", ""))

    # Grounding is informational only from here on (see root-cause notes
    # at the top of this file) — it is recorded for the debug expander,
    # but a structurally complete result is always returned and shown.
    grounded = _is_academic_explanation_grounded(data, content)

    return {
        "data": data,
        "raw_text": raw_text,
        "debug_stage": "ok" if grounded else "ok_low_overlap",
        "debug_detail": (
            ""
            if grounded
            else "The explanation reused few of the material's own key words "
                 "(informational only — this did not block the result)."
        ),
    }


# ----------------------------------------------------------------------
# 6. PAGE SECTION UI
# ----------------------------------------------------------------------

def _topic_movie_connection_text(data, concept, reference, connection):
    """
    Text for "🎬 How This Topic Connects to the Movie". Uses the model's
    own topic_movie_connection when it gave a real one; otherwise falls
    back to a short line built from the existing concept / movie /
    connection fields, so the section still appears. Returns "" (section
    hidden) only if there is no movie reference at all.
    """
    value = data.get(TOPIC_MOVIE_CONNECTION_FIELD)
    if _is_real_value(value):
        return value.strip()

    if not reference:
        return ""
    text = f"The topic of {concept} is connected to {reference}."
    if connection:
        text += f" {connection}"
    return text


def _render_explanation(data):
    """Render the Tamil cinema explanation inside a bordered container,
    following study concept -> academic explanation -> movie connection
    -> learning takeaway."""

    subject = str(data.get("subject", "")).strip()
    topic = str(data.get("topic", "")).strip()
    concept = str(data.get("concept", "")).strip() or topic or "This topic"
    academic_explanation = str(data.get("academic_explanation", "")).strip() or "Not provided."
    reference = str(data.get("reference", "")).strip()
    connection = str(data.get("connection", "")).strip()
    learning_takeaway = str(data.get("learning_takeaway", "")).strip() or "Not provided."
    topic_movie_connection = _topic_movie_connection_text(data, concept, reference, connection)

    with st.container(border=True):
        st.markdown(f"### 🎬 {escape(concept)}")

        chips = ""
        if subject:
            chips += f'<span class="sf-chip">📚 {escape(subject)}</span>'
        if topic:
            chips += f'<span class="sf-chip">📖 {escape(topic)}</span>'
        if reference:
            chips += f'<span class="sf-chip">🎥 {escape(reference)}</span>'
        if chips:
            st.markdown(f'<div class="sf-chip-row">{chips}</div>', unsafe_allow_html=True)

        if topic_movie_connection:
            st.markdown("**🎬 How This Topic Connects to the Movie**")
            st.write(topic_movie_connection)

        st.markdown("**📘 The academic explanation**")
        st.write(academic_explanation)

        if reference:
            st.markdown(f"**🎭 Tamil cinema connection — {escape(reference)}**")
        else:
            st.markdown("**🎭 Tamil cinema connection**")
        st.write(connection if connection else "Not provided.")

        st.info(f"🧠 **Takeaway:** {learning_takeaway}")


def _show_debug_expander(debug_stage, debug_detail):
    """Small, honest diagnostic panel: shows exactly which pipeline stage
    the last attempt landed on, instead of only a generic message. Only
    rendered when there's something to explain."""
    if not debug_stage or debug_stage == "ok":
        return
    with st.expander("Why this couldn't be shown as a structured connection"):
        st.write(f"**Stage:** `{debug_stage}`")
        if debug_detail:
            st.write(debug_detail)


# ========================================================================
# ============================  PAGE ENTRY  =============================
# ========================================================================

def show_pop_culture():
    """Render the full Pop Culture Connections page (Step 11)."""

    show_page_header(
        "🎬", "Pop Culture Connections",
        "Understand your topic through a familiar Tamil movie connection.",
    )

    # ------------------------------------------------------------
    # Require study material from the Dashboard first.
    # ------------------------------------------------------------
    study_material = require_study_material()
    if study_material is None:
        return

    subject = study_material.get("subject", "")
    topic = study_material.get("topic", "")
    content = study_material.get("content", "")

    show_topic_bar(subject, topic)
    st.markdown("---")

    existing = st.session_state.get("pop_culture_result")
    has_result = bool(existing and existing.get("data"))
    generate_clicked = st.button(
        "🔄 Regenerate" if has_result else "🎬 Explain with Tamil Cinema",
        type="secondary" if has_result else "primary",
        key="pop_culture_generate_btn",
    )

    if generate_clicked:
        with st.spinner("Finding a Tamil cinema connection..."):
            try:
                result = _generate_pop_culture_explanation(subject, topic, content)
                st.session_state.pop_culture_result = result

                if result.get("data"):
                    st.success("✅ Connection ready!")
                else:
                    st.warning(
                        "Unable to create the Tamil cinema connection right "
                        "now. Please try again."
                    )
            except Exception as e:
                # Handle Gemini / API errors gracefully — never crash the page.
                st.error(f"Sorry, we couldn't create a pop culture connection: {e}")
                return

    # ------------------------------------------------------------
    # Nothing generated yet.
    # ------------------------------------------------------------
    stored = st.session_state.get("pop_culture_result")
    if not stored:
        st.info("Click **🎬 Explain with Tamil Cinema** to get started.")
        return

    data = stored.get("data")
    raw_text = stored.get("raw_text", "")

    # Safe fallback: if we couldn't parse JSON, show the raw text plus a
    # precise, stage-labelled reason instead of crashing or guessing.
    if not data:
        st.markdown("---")
        # Never show the raw model reply to the student - just a friendly
        # message. (The stage-labelled reason stays in the debug expander.)
        _show_debug_expander(stored.get("debug_stage"), stored.get("debug_detail"))
        st.info("Click **🎬 Explain with Tamil Cinema** to try again.")
        return

    st.markdown("---")
    _render_explanation(data)
    _show_debug_expander(stored.get("debug_stage"), stored.get("debug_detail"))