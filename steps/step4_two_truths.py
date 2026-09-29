"""
Step 4 — Truth & Trick
--------------------------------
A single-round activity inside the sequential Practice flow:

    Quiz -> Next Round -> Truth & Trick -> Next Round -> Emoji Decode ->
    Next Round -> Scenario -> Next Round -> Dynamic Challenge -> Final Score

Truth & Trick generates EXACTLY ONE round: 3 statements (2 true, 1 false)
about the current study material. The student picks the false statement
and is worth exactly 1 point. There is no "Generate New Round" /
"New Round" button - after the round is answered, "Next Round" only
advances the Practice flow to the next activity (Emoji Decode). It never
triggers another Truth & Trick generation.

MODEL RULE: this file never hardcodes a model name or creates its own
Gemini client. Every generation call goes through
generate_content(prompt) from services/gemini_service.py.

GENERATION RULE: generate_content() is called AT MOST during the first
render of this activity (when no question exists yet in session state),
and up to MAX_GENERATION_ATTEMPTS times if Gemini's reply is malformed.
It is never called again on a plain Streamlit rerun (selecting an
option, submitting an answer, etc.).

SOURCE OF TRUTH: st.session_state["study_material"]["content"].
No other content/source variable is introduced.

ROBUSTNESS: the raw Gemini reply is cleaned (strip ```json / ``` fences,
extract the {...} block), parsed with json.loads(), then structurally
validated (exactly 3 non-empty statements, lie in {0, 1, 2}, a non-empty
explanation, no duplicate/near-duplicate statements within the round).
If any step fails, generation is retried (max 3 attempts total). If it
still fails, a clean, non-technical message is shown.

FIX (this revision): lie quality. Previously nothing stopped the model
from generating a "same-topic" lie - a slightly changed fact, a related
misconception, or a partially-correct statement from the same chapter
(e.g. "Plants perform photosynthesis at night" for a photosynthesis
round). That makes the round confusing/tricky rather than a clean
"which one doesn't belong" exercise, and risks a lie that could
reasonably be read as true. Two things changed to fix this:
  1. The prompt (_build_prompt) now explicitly requires the two TRUE
     statements to be different facts directly supported by the study
     material, and the LIE to be completely unrelated to the topic -
     from a different everyday domain (sports, music, computers,
     transportation, geography, food, movies, everyday objects) - with a
     concrete good/bad example, and asks the model to vary the lie's
     position instead of always using index 0.
  2. _validate_and_clean() (called via the new _validate_and_clean(data,
     content) signature) now also rejects a round if either TRUE
     statement isn't grounded in the study material, or if the LIE
     shares meaningful vocabulary with the study material or either TRUE
     statement - see _tokenize() below. This is a plain-Python text
     check, not a model call, so it adds no extra Gemini requests and
     never runs on a plain rerun or answer submission.

FIX (this revision): the lie index is now fully auto-detected. Gemini
may still return a "lie" field in its JSON reply, but that value is
never read or trusted anywhere in this file anymore - see
_auto_classify_lie(). Instead, after the 3 statements are generated:
  1. Statement 1 is compared against the study material's vocabulary.
  2. Statement 2 is compared against the study material's vocabulary.
  3. Statement 3 is compared against the study material's vocabulary.
  4. The two statements that are strongly supported by (share
     meaningful vocabulary with) the study material are identified as
     the TRUE statements.
  5. The one remaining statement - unrelated to the study material and
     to both TRUE statements - is identified as the LIE, and its
     zero-based index is set automatically.
  6. If the 3 statements don't cleanly split into exactly 2
     study-supported + 1 unrelated statement, the whole round is
     REJECTED and a new one is generated instead, up to
     MAX_GENERATION_ATTEMPTS attempts total. Only a round that resolves
     to exactly one auto-detected lie is ever shown to the user.
No manual/user-supplied lie index is ever used.
_generate_truth_trick(content)'s signature is unchanged.

SESSION STATE: st.session_state["practice_truth_trick"] holds:
    {
        "statements": [...],
        "lie": 0,
        "explanation": "...",
        "user_answer": None,
        "answered": False,
        "score_awarded": False,
    }
Once generated, the same question is kept across reruns.

NAVIGATION: this file does not create its own Practice-flow navigation
variable. If the project already tracks the active activity with a
variable such as practice_stage / current_activity / practice_step /
current_round, that variable is reused to advance from Truth & Trick to
the next activity. If no such variable is found in session state, this
file simply marks its own round as
st.session_state["practice_truth_trick"]["completed"] = True, which the
Practice hub can check instead.

Embedding: pass embedded=True (used by the Practice hub) to skip this
page's own header/topic bar, since the hub already shows those once
above all Practice activities. Called with no arguments (embedded=False),
this file behaves as a normal standalone page.

Imported in app.py with:
    from steps.step4_two_truths import show_two_truths
No other app.py changes are required for this feature.
"""

import difflib
import json
import re

import streamlit as st

from services.gemini_service import generate_content
from steps.step13_memory import log_answer
from steps.step19_ui import show_page_header, show_topic_bar

# How many times to re-ask Gemini for the ONE Truth & Trick round when the
# reply is missing, malformed, or fails validation.
MAX_GENERATION_ATTEMPTS = 3

# The study material is trimmed so the small model gets a short, simple
# prompt instead of being overwhelmed with a huge block of text.
MAX_CONTENT_CHARS = 1200

# Two statements (within the same round) are treated as a lazy near-copy
# of each other (not merely "about the same topic") when their full,
# normalized sentence text is at least this similar, character-for-character.
NEAR_DUPLICATE_TEXT_RATIO = 0.92

# Key under which this activity's state lives in st.session_state.
SESSION_KEY = "practice_truth_trick"

# Shown only after every generation attempt has failed. Never a raw
# exception or the old "Could not create a valid Truth & Trick question."
GENERATION_FAILURE_MESSAGE = (
    "⚠️ Could not generate a valid Truth & Trick round from the study material. "
    "Please try again."
)

# Candidate names for the Practice hub's existing sequential-navigation
# variable. The first one found in session state is reused; none is ever
# created here.
_NAV_KEY_CANDIDATES = ("practice_stage", "current_activity", "practice_step", "current_round")

# Mapping used only when the existing navigation variable is a string that
# identifies the current activity by name.
_NEXT_ACTIVITY_BY_NAME = {
    "truth_trick": "emoji_decode",
    "truth_and_trick": "emoji_decode",
    "two_truths": "emoji_decode",
}

# Candidate names for an existing running-score variable. The first one
# found in session state is incremented; none is ever created here.
_SCORE_KEY_CANDIDATES = ("score", "practice_score", "total_score")


# ---------------------------------------------------------------------------
# Prompt building — short and simple, exactly ONE round.
# ---------------------------------------------------------------------------

def _build_prompt(content):
    trimmed_content = (content or "")[:MAX_CONTENT_CHARS]

    return f"""Return ONLY JSON. No text before or after it, no Markdown.

Based ONLY on the study material below, write a "Two Truths and a Lie"
round with exactly 3 short statements:

- TWO statements must be TRUE and directly supported by the study material
  below. They must be two DIFFERENT facts - do not restate the same fact
  using different wording.
- ONE statement must be the LIE, and it must be COMPLETELY UNRELATED to the
  study material's topic:
    - Do NOT use a slightly changed fact from the topic.
    - Do NOT use a similar concept from the same chapter.
    - Do NOT use a partially-correct statement.
    - Do NOT use a confusing statement.
    - Do NOT use a common misconception about the topic.
    - Do NOT write anything that could reasonably be true according to
      the study material.
  Instead, pick the lie from a completely different, everyday domain -
  for example: sports, music, computers, transportation, geography, food,
  movies, or general everyday objects (choose naturally) - and make sure
  it is objectively false as written.

Example of the exact style required (study topic: Photosynthesis):
  TRUE: "Green plants use light energy to make food."
  TRUE: "Chlorophyll plays an important role in photosynthesis."
  LIE : "A cricket match consists of two teams."
This is GOOD because the lie is obviously unrelated (sports), not a
same-topic false statement.

Do NOT write a lie like "Plants perform photosynthesis at night" - that is
BAD because it is still about the same topic and could be mistaken for a
misconception instead of an obviously unrelated statement.

JSON:
{{
  "statements": ["statement 1", "statement 2", "statement 3"],
  "lie": 0,
  "explanation": "Short explanation, based only on the study material, of why the lie is false and unrelated to the topic."
}}

"lie" is the zero-based index of the false statement (0 = first, 1 = second,
2 = third). Vary the position of the lie across rounds - do not always put
it at index 0.

Study material:
{trimmed_content}
"""
# NOTE: the "lie" field above is requested only so the JSON shape stays
# familiar to the model; this file never reads data["lie"]. The lie index
# actually used is always computed independently in _auto_classify_lie().


# ---------------------------------------------------------------------------
# Cleaning + parsing (required steps, in this exact order)
# ---------------------------------------------------------------------------

def _clean_json_text(text):
    """
    1. Convert to string / strip.
    2. Remove ```json and ``` fences if present.
    3. Find the first "{".
    4. Find the last "}".
    5. Return only that section (or None if no {...} block exists).
    """
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "")
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return cleaned[start:end + 1]


def _parse_reply(text):
    """Clean the reply, then run json.loads(). Returns a dict or None."""
    json_text = _clean_json_text(text)
    if json_text is None:
        return None
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Validation (structure + duplicate check — the AI decides which
# statement is false)
# ---------------------------------------------------------------------------

def _normalize(text):
    cleaned = re.sub(r"[^\w\s]", " ", str(text or "").lower())
    return " ".join(cleaned.split())


def _has_near_duplicate_statement(statements):
    """
    True if any two statements are essentially the SAME sentence reworded
    (a lazy near-copy), not merely two statements that share topic
    vocabulary. Truth & Trick's whole design relies on the false statement
    closely resembling a true one (same topic, one changed fact) - e.g.
    "DNA replication occurs during S phase." vs "...during G1 phase." must
    NOT be rejected just because they share almost every other word.

    Compares full normalized sentences with difflib's character-level
    ratio, which tracks near-identical phrasing rather than shared
    subject-matter words, so a deliberately similar but factually
    different pair of statements is not penalized.
    """
    normalized = [_normalize(s) for s in statements]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a, b = normalized[i], normalized[j]
            if not a or not b:
                continue
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio >= NEAR_DUPLICATE_TEXT_RATIO:
                return True
    return False


# ---------------------------------------------------------------------------
# Domain / topic-relatedness validation (new in this revision)
# ---------------------------------------------------------------------------
#
# Everything below exists to enforce the "lie must be from a completely
# different domain" requirement in plain Python, as a backstop to the
# prompt instructions above. These are word-overlap heuristics over text
# that's already been generated - no extra Gemini calls, and they never
# run during answer submission or a plain rerun.

# Common short/filler words excluded when comparing the study material,
# the true statements, and the lie for shared vocabulary. Without this,
# ordinary connective words ("the", "is", "used", "process") that show up
# in almost any sentence would make ordinary text look "topic-related"
# regardless of actual subject matter.
_TOPIC_STOPWORDS = frozenset("""
a an the and or but of to in on at for with is are was were be been
being this that these those it its as by from into onto over under
about above below between among within without near far more most
less least very can could should would may might will shall do does
did each every any all both few many much other another such only
own same so than too also just now not no nor used use uses process
type kind system method way form part example statement true false
""".split())


def _tokenize(text):
    """
    Lowercase, alphabetic-only (3+ letter) word set, with common filler
    words removed. Used to compare the study material and the two TRUE
    statements against the LIE, so a lie that shares real, meaningful
    vocabulary with the topic can be caught and rejected.
    """
    return set(re.findall(r"[a-zA-Z]{3,}", (text or "").lower())) - _TOPIC_STOPWORDS


def _statement_grounded_in_content(statement_text, content_tokens, content):
    """
    True if this ONE statement is directly supported by the study
    material - a light-touch check (at least one shared meaningful word,
    or an exact substring match for very short statements) rather than a
    strict one, since paraphrasing is expected. Used per-statement by
    _auto_classify_lie() below; does not take the model's "lie" field
    into account at all.
    """
    statement_tokens = _tokenize(statement_text)

    if statement_tokens:
        return bool(statement_tokens & content_tokens)

    stripped = (statement_text or "").strip().lower()
    return bool(stripped) and stripped in (content or "").lower()


def _auto_classify_lie(statements, content):
    """
    Independently determine which of the 3 statements is the lie. The
    "lie" field Gemini returned is NEVER consulted here or anywhere else
    in this file - this function is the only source of truth for the
    lie index.

    Steps (matching the activity's requirements exactly):
      1. Compare statement 1 against the study material.
      2. Compare statement 2 against the study material.
      3. Compare statement 3 against the study material.
      4. Identify the two statements that are strongly supported by the
         study material (they share meaningful vocabulary with it, or
         match it verbatim for very short statements) - these are the
         TRUE statements.
      5. Identify the remaining statement as the unrelated LIE.
      6. Return its zero-based index.

    A round is only accepted when this cleanly produces EXACTLY 2
    study-supported statements and EXACTLY 1 unrelated statement. If all
    3 look supported, all 3 look unrelated, or 2 look unrelated, this
    returns None so the caller rejects the round and generates a fresh
    one instead of guessing.
    """
    content_tokens = _tokenize(content)

    grounded_flags = [
        _statement_grounded_in_content(statement, content_tokens, content)
        for statement in statements
    ]

    grounded_indices = [i for i, grounded in enumerate(grounded_flags) if grounded]
    unrelated_indices = [i for i, grounded in enumerate(grounded_flags) if not grounded]

    if len(grounded_indices) != 2 or len(unrelated_indices) != 1:
        return None  # doesn't cleanly split into 2 true + 1 lie

    lie_index = unrelated_indices[0]

    # Belt-and-braces: the auto-detected lie must also share no
    # meaningful vocabulary with either auto-detected TRUE statement
    # (not just with the raw study-material excerpt), so a lie that
    # happens to reuse a topic word missing from the trimmed content
    # chunk is still caught and the round rejected.
    true_tokens = set()
    for i in grounded_indices:
        true_tokens |= _tokenize(statements[i])

    lie_tokens = _tokenize(statements[lie_index])
    if lie_tokens and (lie_tokens & true_tokens):
        return None

    return lie_index


def _validate_and_clean(data, content):
    """
    A generated round is valid only when:
      - result is a dict
      - "statements" exists, is a list of exactly 3 non-empty strings
      - "explanation" exists and is a non-empty string
      - no two statements are identical or near-identical
      - the 3 statements cleanly auto-classify (see _auto_classify_lie)
        into exactly 2 study-supported TRUE statements + 1 unrelated LIE

    IMPORTANT: any "lie" field in `data` (i.e. whatever Gemini returned)
    is intentionally never read here. The lie index used is always the
    one independently computed by _auto_classify_lie() from the
    statement text itself, never a value the model (or a user) supplied.

    Returns a cleaned dict ({"statements": [...], "lie": int,
    "explanation": str}) on success, or None on failure.
    """
    if not isinstance(data, dict):
        return None

    raw_statements = data.get("statements")
    if not isinstance(raw_statements, list) or len(raw_statements) != 3:
        return None

    statements = []
    for item in raw_statements:
        if not isinstance(item, str) or not item.strip():
            return None
        statements.append(item.strip())

    normalized = [_normalize(s) for s in statements]
    if len(set(normalized)) != 3:
        return None  # exact duplicate
    if _has_near_duplicate_statement(statements):
        return None  # near-duplicate

    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        return None

    # ---- Auto-detect the lie. Gemini's own "lie" field is deliberately
    # never consulted - see _auto_classify_lie()'s docstring. ----
    lie = _auto_classify_lie(statements, content)
    if lie is None:
        return None  # doesn't cleanly split into 2 true + 1 unrelated lie

    return {"statements": statements, "lie": lie, "explanation": explanation.strip()}


# ---------------------------------------------------------------------------
# Generation — exactly ONE round, up to MAX_GENERATION_ATTEMPTS tries.
# ---------------------------------------------------------------------------

def _generate_truth_trick(content):
    for _attempt in range(MAX_GENERATION_ATTEMPTS):
        prompt = _build_prompt(content)
        try:
            raw_reply = generate_content(prompt)  # existing Gemini call
        except Exception:
            continue

        data = _parse_reply(raw_reply)
        if data is None:
            continue

        cleaned = _validate_and_clean(data, content)
        if cleaned is None:
            continue

        return {
            "statements": cleaned["statements"],
            "lie": cleaned["lie"],
            "explanation": cleaned["explanation"],
            "user_answer": None,
            "answered": False,
            "score_awarded": False,
        }

    return None


# ---------------------------------------------------------------------------
# Practice-flow integration helpers
# ---------------------------------------------------------------------------

def _find_existing_nav_key():
    for key in _NAV_KEY_CANDIDATES:
        if key in st.session_state:
            return key
    return None


def _advance_practice_flow():
    """
    Advance the existing sequential Practice flow from Truth & Trick to the
    next activity (Emoji Decode). Reuses whatever navigation variable the
    project already has; never introduces a second, competing one.
    """
    st.session_state[SESSION_KEY]["completed"] = True

    nav_key = _find_existing_nav_key()
    if nav_key is None:
        return

    current_value = st.session_state[nav_key]
    if isinstance(current_value, bool):
        return
    if isinstance(current_value, int):
        st.session_state[nav_key] = current_value + 1
    elif isinstance(current_value, str) and current_value in _NEXT_ACTIVITY_BY_NAME:
        st.session_state[nav_key] = _NEXT_ACTIVITY_BY_NAME[current_value]


def _award_point_if_needed():
    """Award exactly 1 point, at most once per round, to an existing score
    variable if the project already tracks one under a common name."""
    if st.session_state[SESSION_KEY]["score_awarded"]:
        return
    st.session_state[SESSION_KEY]["score_awarded"] = True

    for key in _SCORE_KEY_CANDIDATES:
        if key in st.session_state and isinstance(st.session_state[key], (int, float)):
            st.session_state[key] += 1
            return


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def show_two_truths(embedded=False):
    """
    Display the Truth & Trick activity: exactly ONE round, then hand
    control back to the sequential Practice flow.
    """
    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("No study material generated yet. Please generate study material from the Dashboard first.")
        return

    subject = study_material.get("subject")
    topic = study_material.get("topic")
    content = study_material.get("content")

    if not embedded:
        show_page_header("🎭", "Truth & Trick", "Two statements are true. One is a trick. Can you spot it?")
        show_topic_bar(subject, topic)
        st.markdown("---")

    # Generate only when this activity is active and no question yet
    # exists in session state. Never regenerates on a plain rerun.
    if SESSION_KEY not in st.session_state:
        with st.spinner("Creating your Truth & Trick round..."):
            round_data = _generate_truth_trick(content)

        if round_data is None:
            st.error(GENERATION_FAILURE_MESSAGE)
            return

        st.session_state[SESSION_KEY] = round_data

    data = st.session_state[SESSION_KEY]

    st.markdown("### 🎭 Truth & Trick")
    st.markdown("**Which statement is false?**")

    statement_labels = [f"{i + 1}. {data['statements'][i]}" for i in range(3)]

    default_index = data["user_answer"] if data["user_answer"] is not None else 0
    selected_label = st.radio(
        "Choose the statement you think is false:",
        statement_labels,
        index=default_index,
        disabled=data["answered"],
        key="truth_trick_radio",
    )
    selected_index = statement_labels.index(selected_label)
    if not data["answered"]:
        data["user_answer"] = selected_index

    st.markdown("")

    if not data["answered"]:
        if st.button("✅ Submit Answer", key="truth_trick_submit"):
            data["user_answer"] = selected_index
            data["answered"] = True

            is_correct = selected_index == data["lie"]
            if is_correct:
                _award_point_if_needed()

            if topic:
                log_answer(topic, is_correct, "Truth & Trick")

            st.rerun()
    else:
        is_correct = data["user_answer"] == data["lie"]

        if is_correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect — the false statement was: \"{data['statements'][data['lie']]}\"")

        st.markdown(f"**Explanation:**  \n{data['explanation']}")

        st.markdown("")

        if st.button("➡️ Next Round", type="primary", key="truth_trick_next"):
            _advance_practice_flow()
            st.rerun()