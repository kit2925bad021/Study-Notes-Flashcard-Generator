"""
Step 12 — Dynamic Questions (Dynamic Challenge)
------------------------------------------------
An adaptive multiple-choice question: the difficulty goes up after a
correct answer and down after a wrong one, and each new question is
generated with the previous question / answer / result in mind so the
student is not asked the same thing twice.

TWO WAYS THIS FILE IS USED
----------------------------
1. Standalone page (show_dynamic_questions(), used by the
   "🔄 Dynamic Questions" nav item in app.py, called with no arguments).

2. Practice hub (steps/step21_practice.py), stage 5 of 5
   ("Dynamic Challenge"): the hub imports and calls ONLY
   _generate_dynamic_question(...) and does its own render / submit /
   scoring. The hub retries the call itself, so this function is exactly
   ONE prompt/parse pass with no retry of its own.

STABLE INTERFACE (do not change without updating step21_practice.py):

    _generate_dynamic_question(subject, topic, content, difficulty,
                               previous_question=None,
                               previous_answer=None,
                               previous_result=None)

    Returns a dict, or None if the model's reply was unusable:
        {
            "question": "...",
            "options": ["...", "...", "...", "..."],   # list of 4 strings
            "correct_answer": 2,                       # INDEX into options
            "explanation": "...",
            "difficulty": "Easy" | "Medium" | "Hard",
        }

MODEL RULE: this file never hardcodes a model name or creates its own
Ollama client. Every generation call goes through
generate_content(prompt) from services/ollama_service.py.

SOURCE OF TRUTH: st.session_state["study_material"] (subject / topic /
content). No other content source is used.

GENERATION RULE (standalone page): Ollama is only called when the page
first opens with no question yet, or when the student clicks
"Next Question" / "Try Again". Never on a plain Streamlit rerun
(selecting an option, submitting an answer, etc.). A failed generation is
stored, so reruns do not call Ollama again until "Try Again" is clicked.

SESSION STATE: every key this page owns starts with "dynamic_" - none of
the Practice hub's "practice_*" keys are read or written here.

Imported in app.py with:
    from steps.step12_dynamic_questions import show_dynamic_questions
"""

import hashlib
import json
import random
import re

import streamlit as st

from services.ollama_service import generate_content
from steps.step13_memory import log_answer
from steps.step19_ui import show_page_header, show_topic_bar

# The study material is trimmed so the small local model gets a short,
# simple prompt.
MAX_CONTENT_CHARS = 1500

NUM_OPTIONS = 4
OPTION_LETTERS = ["A", "B", "C", "D"]

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
DEFAULT_DIFFICULTY = "Medium"

GENERATION_FAILURE_MESSAGE = "Could not create a valid Dynamic Question. Please try again."

_DIFFICULTY_GUIDE = {
    "Easy": "Ask a simple recall question about one basic idea from the material.",
    "Medium": "Ask a question that needs understanding of a concept, not just its name.",
    "Hard": "Ask a question that needs applying or comparing ideas from the material.",
}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _normalize_difficulty(difficulty):
    """Return one of DIFFICULTY_LEVELS (case-insensitive), else the default."""
    text = str(difficulty or "").strip().lower()
    for level in DIFFICULTY_LEVELS:
        if level.lower() == text:
            return level
    return DEFAULT_DIFFICULTY


def _build_prompt(subject, topic, content, difficulty,
                  previous_question=None, previous_answer=None, previous_result=None):
    trimmed_content = (content or "")[:MAX_CONTENT_CHARS]
    level = _normalize_difficulty(difficulty)

    history_block = ""
    if previous_question:
        history_block += f"The student's previous question was: {previous_question}\n"
        if previous_answer:
            history_block += f"The student answered: {previous_answer}\n"
        if previous_result is not None:
            outcome = "correct" if previous_result in (True, "correct", "Correct") else "wrong"
            history_block += f"That answer was {outcome}.\n"
        history_block += "Ask about a DIFFERENT idea from the material than the previous question.\n"

    return f"""Create exactly ONE multiple-choice question using ONLY the study material below.

Subject: {subject}
Topic: {topic}
Difficulty: {level}. {_DIFFICULTY_GUIDE[level]}
{history_block}
Return ONLY one valid JSON object with these fields:
question: the question text
options: a list of exactly 4 different answer choices (plain text, no A/B/C/D labels)
correct_answer: the letter (A, B, C or D) of the one correct option
explanation: one or two simple sentences explaining why that option is correct

Exactly ONE option must be correct. The wrong options must be plausible
but clearly wrong. Keep the language simple and beginner friendly.
Do not use Markdown. Do not use ```json. Do not write anything outside the JSON.

Study material:
{trimmed_content}
"""


# ---------------------------------------------------------------------------
# Cleaning + parsing
# ---------------------------------------------------------------------------

def _clean_json_text(text):
    """Strip code fences and return the {...} block, or None."""
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return cleaned[start:end + 1]


def _load_json_object(text):
    """Clean and json.loads() a reply. Returns a dict or None. Never uses eval()."""
    json_text = _clean_json_text(text)
    if json_text is None:
        return None
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", json_text)
    for candidate in (json_text, without_trailing_commas):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


_OPTION_PREFIX = re.compile(r"^\s*(?:option\s+)?\(?[A-Da-d1-4][\.\)\:\-]\s+")


def _clean_option(item):
    """Turn one option (string, or dict with a text-like key) into clean text."""
    if isinstance(item, dict):
        for key in ("text", "option", "value", "answer", "label", "content"):
            if item.get(key):
                item = item[key]
                break
        else:
            return ""
    text = str(item or "").strip()
    return _OPTION_PREFIX.sub("", text).strip()


def _normalize_options(raw_options):
    """Return a list of cleaned option strings, or None if not a usable list/dict."""
    if isinstance(raw_options, dict):
        raw_options = [raw_options[key] for key in sorted(raw_options, key=str)]
    if not isinstance(raw_options, list):
        return None
    return [_clean_option(item) for item in raw_options]


def _resolve_correct_index(raw_answer, options):
    """
    Work out which option the model says is correct. Accepts a letter
    ("B"), a 0-based or 1-based number, or the option text itself.
    Returns an index into `options`, or None if it cannot be resolved.
    """
    if raw_answer is None or isinstance(raw_answer, bool):
        return None

    if isinstance(raw_answer, int):
        if 0 <= raw_answer < len(options):
            return raw_answer
        return None

    text = str(raw_answer).strip()
    if not text:
        return None

    letter = text.strip(".:)( ").upper()
    if letter in OPTION_LETTERS[:len(options)]:
        return OPTION_LETTERS.index(letter)

    if text.isdigit() and 1 <= int(text) <= len(options):
        return int(text) - 1

    stripped = _OPTION_PREFIX.sub("", text).strip().lower()
    matches = [i for i, opt in enumerate(options) if opt.strip().lower() == stripped]
    if len(matches) == 1:
        return matches[0]
    return None


def _validate_question(data, difficulty):
    """
    Return the finished question dict, or None. Valid means: a question,
    exactly 4 distinct non-empty options, a correct answer that resolves
    to exactly one of them, and an explanation.
    """
    if not isinstance(data, dict):
        return None

    question = str(data.get("question") or "").strip()
    explanation = str(data.get("explanation") or "").strip()
    options = _normalize_options(data.get("options"))
    if not question or not explanation or options is None:
        return None
    if len(options) != NUM_OPTIONS or any(not opt for opt in options):
        return None
    if len({opt.lower() for opt in options}) != NUM_OPTIONS:
        return None

    raw_answer = data.get("correct_answer", data.get("answer"))
    correct_index = _resolve_correct_index(raw_answer, options)
    if correct_index is None:
        return None

    # Shuffle so the correct answer is not always in the same place.
    correct_text = options[correct_index]
    shuffled = list(options)
    random.shuffle(shuffled)

    return {
        "question": question,
        "options": shuffled,
        "correct_answer": shuffled.index(correct_text),
        "explanation": explanation,
        "difficulty": _normalize_difficulty(difficulty),
    }


# ---------------------------------------------------------------------------
# Generation — exactly ONE prompt/parse pass (the caller handles retries).
# ---------------------------------------------------------------------------

def _generate_dynamic_question(subject, topic, content, difficulty,
                               previous_question=None, previous_answer=None,
                               previous_result=None):
    """
    Ask Ollama for ONE question at `difficulty`, using the previous
    question / answer / result (if any) to steer variety.
    Returns the question dict described at the top of this file, or None
    if the reply could not be parsed or validated. A failure of the
    Ollama call itself is raised to the caller (the Practice hub catches
    it and retries).
    """
    prompt = _build_prompt(subject, topic, content, difficulty,
                           previous_question, previous_answer, previous_result)
    raw_reply = generate_content(prompt)  # the ONLY function used to call Ollama
    return _validate_question(_load_json_object(raw_reply), difficulty)


def _next_difficulty(current, was_correct):
    """One step harder after a correct answer, one step easier after a wrong one."""
    index = DIFFICULTY_LEVELS.index(_normalize_difficulty(current))
    index += 1 if was_correct else -1
    index = max(0, min(len(DIFFICULTY_LEVELS) - 1, index))
    return DIFFICULTY_LEVELS[index]


# ---------------------------------------------------------------------------
# Standalone page
# ---------------------------------------------------------------------------

_STATE_DEFAULTS = {
    "dynamic_question": None,       # the current question dict
    "dynamic_failed": False,        # last generation failed (stored so reruns don't re-call Ollama)
    "dynamic_difficulty": DEFAULT_DIFFICULTY,
    "dynamic_selected": None,       # chosen option index
    "dynamic_submitted": False,
    "dynamic_last_correct": None,
    "dynamic_score": 0,
    "dynamic_total": 0,
    "dynamic_source_key": None,     # fingerprint of the study material
}


def _init_session_state():
    for key, value in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _material_fingerprint(subject, topic, content):
    raw = f"{subject}\x1f{topic}\x1f{content}"
    return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()


def _reset_page_state():
    """Reset only this page's own keys (used when the study material changes)."""
    for key, value in _STATE_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("dynamic_radio", None)


def _load_new_question(subject, topic, content, previous=None):
    """
    Generate ONE new question at the current difficulty and store it (or
    store the failure). `previous` is (question, answer_text, was_correct)
    from the round that was just finished, or None.
    """
    prev_q, prev_a, prev_r = previous if previous else (None, None, None)

    with st.spinner("Creating your Dynamic Question..."):
        try:
            question = _generate_dynamic_question(
                subject, topic, content, st.session_state.dynamic_difficulty,
                previous_question=prev_q, previous_answer=prev_a, previous_result=prev_r,
            )
        except Exception:
            # Never show a traceback or a raw Ollama error to the student.
            question = None

    st.session_state.pop("dynamic_radio", None)
    st.session_state.dynamic_selected = None
    st.session_state.dynamic_submitted = False
    st.session_state.dynamic_last_correct = None
    st.session_state.dynamic_question = question
    st.session_state.dynamic_failed = question is None


def show_dynamic_questions(embedded=False):
    """Display the Dynamic Questions page: one adaptive question at a time."""
    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("No study material generated yet. Please generate study material from the Dashboard first.")
        return

    subject = study_material.get("subject")
    topic = study_material.get("topic")
    content = study_material.get("content")

    if not embedded:
        show_page_header("🔄", "Dynamic Questions", "Questions that adapt to how well you're doing.")
        show_topic_bar(subject, topic)
        st.markdown("---")

    _init_session_state()

    # A round stored for different study material is discarded.
    source_key = _material_fingerprint(subject, topic, content)
    if st.session_state.dynamic_source_key != source_key:
        _reset_page_state()
        st.session_state.dynamic_source_key = source_key

    # Generate only when there is neither a question nor a stored failure.
    if st.session_state.dynamic_question is None and not st.session_state.dynamic_failed:
        _load_new_question(subject, topic, content)

    if st.session_state.dynamic_failed:
        st.error(GENERATION_FAILURE_MESSAGE)
        if st.button("🔄 Try Again", key="dynamic_retry"):
            st.session_state.dynamic_failed = False
            st.rerun()
        return

    data = st.session_state.dynamic_question
    options = data["options"]
    submitted = st.session_state.dynamic_submitted

    st.caption(
        f"Difficulty: {data['difficulty']}  |  "
        f"Score: {st.session_state.dynamic_score}/{st.session_state.dynamic_total}"
    )
    st.markdown(f"### {data['question']}")

    selected = st.session_state.dynamic_selected
    default_index = selected if isinstance(selected, int) else None

    chosen = st.radio(
        "Choose an answer:",
        list(range(len(options))),
        index=default_index,
        format_func=lambda i: f"{OPTION_LETTERS[i]}. {options[i]}",
        disabled=submitted,
        key="dynamic_radio",
    )
    if chosen is not None and not submitted:
        st.session_state.dynamic_selected = chosen

    if not submitted:
        if st.button("✅ Submit Answer", disabled=st.session_state.dynamic_selected is None,
                     key="dynamic_submit"):
            is_correct = st.session_state.dynamic_selected == data["correct_answer"]
            st.session_state.dynamic_submitted = True
            st.session_state.dynamic_last_correct = is_correct
            st.session_state.dynamic_total += 1
            if is_correct:
                st.session_state.dynamic_score += 1
            if topic:
                log_answer(topic, is_correct, "Dynamic Challenge", data["difficulty"])
            st.rerun()
        return

    # ---- Result ----
    is_correct = st.session_state.dynamic_last_correct
    correct_index = data["correct_answer"]

    if is_correct:
        st.success("✅ Correct!")
    else:
        chosen_index = st.session_state.dynamic_selected
        st.error("❌ Incorrect")
        if isinstance(chosen_index, int) and 0 <= chosen_index < len(options):
            st.markdown(f"**Your answer:** {OPTION_LETTERS[chosen_index]}. {options[chosen_index]}")
        st.markdown(f"**Correct answer:** {OPTION_LETTERS[correct_index]}. {options[correct_index]}")
    st.markdown(f"**💡 Explanation:**  \n{data['explanation']}")

    st.markdown("")

    if st.button("➡️ Next Question", type="primary", key="dynamic_next"):
        chosen_index = st.session_state.dynamic_selected
        previous_answer = (
            options[chosen_index]
            if isinstance(chosen_index, int) and 0 <= chosen_index < len(options)
            else None
        )
        st.session_state.dynamic_difficulty = _next_difficulty(data["difficulty"], is_correct)
        _load_new_question(
            subject, topic, content,
            previous=(data["question"], previous_answer, is_correct),
        )
        st.rerun()