"""
Step 6 — Scenario Question
--------------------------------
A single-round activity inside the sequential Practice flow:

    Quiz -> Next Round -> Truth & Trick -> Next Round -> Emoji Decode ->
    Next Round -> Scenario Question -> Next Round -> Dynamic Challenge ->
    Final Score

Scenario Question generates EXACTLY ONE scenario: a short, realistic
situation related to the study material, a question that tests whether
the student can APPLY the concept (not just recall it), and exactly
three multiple-choice options (A/B/C) with exactly one correct answer.
There is no "Generate New Scenario" / "Try Another Scenario" button -
after the round is answered, the existing "➡️ Next Round" button only
advances the Practice flow to the next activity (Dynamic Challenge). It
never triggers another scenario generation.

MODEL RULE: this file never hardcodes a model name or creates its own
Ollama client. Every generation call goes through
generate_content(prompt) from services/ollama_service.py.

GENERATION RULE: generate_content() is called AT MOST during the first
render of this activity (when no scenario exists yet in session state),
and up to MAX_GENERATION_ATTEMPTS times if Ollama's reply is malformed.
It is never called again on a plain Streamlit rerun (selecting an
option, submitting an answer, etc.).

SOURCE OF TRUTH: st.session_state["study_material"]["content"].
No other content/source variable is introduced.

ROBUSTNESS: the raw Ollama reply is first coerced to a string (so a
non-string / None reply can never raise before we even try to parse
it), then cleaned (strip ```json / ``` fences, extract the {...}
block), parsed with json.loads(), then structurally validated (non-empty
scenario/question/explanation, exactly options A/B/C, all three
different, correct_answer in {A, B, C}). If any step fails, generation
is retried (max 3 attempts total). If it still fails, a clean,
non-technical message is shown.

FIX (this revision): distractor quality. Previously the two wrong
options only had to be non-blank, non-duplicate, and "plausible" -
nothing stopped the model from generating same-topic distractors
(closely related concepts, synonyms of the correct answer, or partially
correct options from the same chapter), which makes the scenario much
easier to guess by elimination and defeats the point of testing whether
the student can actually apply the concept. Two things changed to fix
this:
  1. The prompt (_build_prompt) now explicitly requires the two WRONG
     options to be simple, real concepts from a completely different,
     everyday domain (computers, music, sports, geography, food,
     transportation, etc.) that share nothing with the scenario's topic
     - with a concrete good/bad example - instead of just asking for
     "plausible" wrong answers, and asks the model to vary which letter
     (A/B/C) is correct.
  2. _validate_and_clean() (called via the new _validate_and_clean(data,
     content) signature) now also rejects a scenario if the correct
     option isn't grounded in the study material, or if either WRONG
     option shares a meaningful word with the scenario, the question, or
     the correct answer - see _tokenize() / _topic_tokens() /
     _option_is_topic_related() below. This is a plain-Python text
     check, not a model call, so it adds no extra Ollama requests and
     never runs during answer submission or a normal rerun.

FIX (this revision): the correct answer is now fully auto-detected.
Ollama may still return a "correct_answer" field in its JSON reply, but
that value is never read or trusted anywhere in this file anymore - see
_auto_classify_correct_answer(). Instead, after the scenario, question,
and 3 options are generated:
  1. Option A is compared against the study material.
  2. Option B is compared against the study material.
  3. Option C is compared against the study material.
  4. The ONE option that is grounded in (shares meaningful vocabulary
     with) the study material is identified as the CORRECT answer.
  5. The other TWO options - which must also be unrelated to the
     scenario/question/correct-answer wording, not just to the raw
     study-material excerpt - are confirmed as the WRONG options.
  6. If this doesn't cleanly resolve to exactly 1 grounded option + 2
     unrelated options (e.g. all three look grounded, none do, or two
     do), the whole scenario is REJECTED and a new one is generated
     instead, up to MAX_GENERATION_ATTEMPTS attempts total. Only a
     scenario that resolves to exactly one auto-detected correct answer
     is ever shown to the user.
No manual/user-supplied correct answer is ever used.
_generate_scenario(content)'s signature is unchanged.

SESSION STATE: st.session_state["practice_scenario"] holds:
    {
        "scenario": "...",
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "..."},
        "correct_answer": "A",
        "explanation": "...",
        "user_answer": None,
        "answered": False,
        "score_awarded": False,
    }
Once generated, the same scenario is kept across reruns.

NAVIGATION: this file does not create its own Practice-flow navigation
variable. If the project already tracks the active activity with a
variable such as practice_stage / current_activity / practice_step /
current_round, that variable is reused to advance from Scenario Question
to the next activity (Dynamic Challenge). If no such variable is found
in session state, this file simply marks its own round as
st.session_state["practice_scenario"]["completed"] = True, which the
Practice hub can check instead.

Embedding: pass embedded=True (used by the Practice hub) to skip this
page's own header/topic bar, since the hub already shows those once
above all Practice activities. Called with no arguments (embedded=False),
this file behaves as a normal standalone page.

Imported in app.py with:
    from steps.step6_scenarios import show_scenarios
No other app.py changes are required for this feature.
"""

import json
import re

import streamlit as st

from services.ollama_service import generate_content
from steps.step13_memory import log_answer
from steps.step19_ui import show_page_header, show_topic_bar

# How many times to re-ask Ollama for the ONE scenario round when the
# reply is missing, malformed, or fails validation.
MAX_GENERATION_ATTEMPTS = 3

# The study material is trimmed so the small model gets a short, simple
# prompt instead of being overwhelmed with a huge block of text.
MAX_CONTENT_CHARS = 1200

# Key under which this activity's state lives in st.session_state.
SESSION_KEY = "practice_scenario"

# Shown only after every generation attempt has failed. Never a raw
# exception or the old "Could not create a valid scenario question."
GENERATION_FAILURE_MESSAGE = (
    "⚠️ Could not generate a valid Scenario round from the study material. "
    "Please try again."
)

# Candidate names for the Practice hub's existing sequential-navigation
# variable. The first one found in session state is reused; none is ever
# created here.
_NAV_KEY_CANDIDATES = ("practice_stage", "current_activity", "practice_step", "current_round")

# Mapping used only when the existing navigation variable is a string that
# identifies the current activity by name.
_NEXT_ACTIVITY_BY_NAME = {
    "scenario": "dynamic_challenge",
    "scenario_question": "dynamic_challenge",
    "scenarios": "dynamic_challenge",
}

# Candidate names for an existing running-score variable. The first one
# found in session state is incremented; none is ever created here.
_SCORE_KEY_CANDIDATES = ("score", "practice_score", "total_score")


# ---------------------------------------------------------------------------
# Prompt building — short and simple, exactly ONE scenario.
# ---------------------------------------------------------------------------

def _build_prompt(content):
    trimmed_content = (content or "")[:MAX_CONTENT_CHARS]

    return f"""Return ONLY JSON. No text before or after it, no Markdown, no code fences.

Based ONLY on the study material below, create ONE short, realistic
scenario that tests whether a student can APPLY a concept from the
material (not just recall a definition). Then ask a question about that
scenario with exactly three multiple-choice options.

Rules:
- The scenario must be topic-specific, based only on information in the
  study material, and understandable for a college student without
  unnecessary complexity.
- The question must test the concept described in the scenario and have
  one clearly defensible answer.
- Exactly ONE option (the correct one) is directly supported by the study
  material above.
- The other TWO options must be COMPLETELY UNRELATED to the topic:
    - Do NOT create two plausible same-topic answers.
    - Do NOT create partially correct options.
    - Do NOT create closely related concepts as distractors.
    - Do NOT create synonyms of the correct answer.
    - Do NOT create confusing or ambiguous options.
  Instead, each wrong option should be a simple, real, sensible concept
  from a completely different everyday domain - for example: computers,
  music, sports, geography, food, transportation, etc. (pick whichever
  fits naturally).
- Do not always make A the correct answer - randomize which letter
  (A, B, or C) is correct.
- The explanation must clearly support the correct option, based only on
  the study material.

Example of the exact style required (study topic: Photosynthesis):
  Scenario: "Riya places a green plant near a window where it receives
  sunlight. After several hours, she observes that the plant is carrying
  out the process discussed in the chapter."
  Question: "Which process is being demonstrated?"
  A. Photosynthesis           <- CORRECT
  B. A computer operating system <- UNRELATED
  C. A musical instrument         <- UNRELATED
This is GOOD because the wrong options are unrelated, everyday concepts
from completely different domains.

Do NOT write wrong options like "Cellular respiration" or "Transpiration"
for that same scenario - that is BAD because all three options would be
related to biology, making them same-topic distractors instead of
unrelated options.

JSON:
{{
  "scenario": "A short realistic situation related to the study material.",
  "question": "What should the student identify, choose, or conclude?",
  "options": {{
    "A": "First possible answer",
    "B": "Second possible answer",
    "C": "Third possible answer"
  }},
  "correct_answer": "A",
  "explanation": "Short explanation of why the correct answer is correct."
}}

Study material:
{trimmed_content}
"""
# NOTE: the "correct_answer" field above is requested only so the JSON
# shape stays familiar to the model; this file never reads
# data["correct_answer"]. The correct letter actually used is always
# computed independently in _auto_classify_correct_answer().


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
    cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return cleaned[start:end + 1]


def _parse_reply(text):
    """Clean the reply, then run json.loads(). Returns a dict or None. Never uses eval()."""
    json_text = _clean_json_text(text)
    if json_text is None:
        return None
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Domain / topic-relatedness validation (new in this revision)
# ---------------------------------------------------------------------------
#
# Everything below exists to enforce the "wrong options must be from a
# completely different domain" requirement in plain Python, as a
# backstop to the prompt instructions above. These are word-overlap
# heuristics over text that's already been generated - no extra Ollama
# calls, and they never run during answer submission or a plain rerun.

# Common short/filler words excluded when comparing the scenario, the
# question, the correct answer, and each wrong option for shared
# vocabulary. Without this, ordinary connective words ("the", "is",
# "process", "used") that show up in almost any sentence would make
# nearly every option look "topic-related" no matter its actual subject.
_TOPIC_STOPWORDS = frozenset("""
a an the and or but of to in on at for with is are was were be been
being this that these those it its as by from into onto over under
about above below between among within without near far more most
less least very can could should would may might will shall do does
did each every any all both few many much other another such only
own same so than too also just now not no nor used use uses process
type kind system method way form part example situation student
scenario question observe observes carrying occurs occurred
""".split())


def _tokenize(text):
    """
    Lowercase, alphabetic-only (3+ letter) word set, with common filler
    words removed. Used to compare the scenario/question/correct-answer
    wording against each wrong option so a wrong option that shares a
    real, meaningful word with the topic can be caught and rejected.
    """
    return set(re.findall(r"[a-zA-Z]{3,}", (text or "").lower())) - _TOPIC_STOPWORDS


def _topic_tokens(scenario_text, question_text, correct_option_text):
    """
    The set of meaningful words that define "the topic" for this specific
    scenario: the scenario text, the question text, and the correct
    answer's own wording. Wrong options are checked against exactly this
    set (not the whole study material) so the unrelatedness check stays
    focused on this scenario's topic rather than incidentally flagging
    words that merely happen to also appear elsewhere in a long study
    document.
    """
    return (
        _tokenize(scenario_text)
        | _tokenize(question_text)
        | _tokenize(correct_option_text)
    )


def _option_is_topic_related(option_text, topic_tokens):
    """
    True if this option shares at least one meaningful word with the
    scenario/question/correct-answer topic - i.e. it is NOT the
    "completely unrelated, different-domain" distractor this activity
    requires. Used only for the two WRONG options; the correct option is
    expected (and required elsewhere) to overlap with the topic.
    """
    option_tokens = _tokenize(option_text)
    if not option_tokens or not topic_tokens:
        return False
    return bool(option_tokens & topic_tokens)


def _option_grounded_in_content(option_text, content_tokens, content):
    """
    True if this ONE option is directly supported by the study material -
    a light-touch check (at least one shared meaningful word, or an
    exact substring match for very short answers) rather than a strict
    one, since paraphrasing is expected. Used per-option by
    _auto_classify_correct_answer() below; does not take the model's
    "correct_answer" field into account at all.
    """
    option_tokens = _tokenize(option_text)

    if option_tokens:
        return bool(option_tokens & content_tokens)

    stripped = (option_text or "").strip().lower()
    return bool(stripped) and stripped in (content or "").lower()


def _auto_classify_correct_answer(scenario, question, options, content):
    """
    Independently determine which of options A/B/C is correct. The
    "correct_answer" field Ollama returned is NEVER consulted here or
    anywhere else in this file - this function is the only source of
    truth for the correct letter.

    Steps (matching the activity's requirements exactly):
      1. Compare option A against the study material.
      2. Compare option B against the study material.
      3. Compare option C against the study material.
      4. The option that is directly supported by the study material
         (shares meaningful vocabulary with it, or matches it verbatim
         for very short options) is the CORRECT answer.
      5. The other two options must ALSO be unrelated to the specific
         scenario/question/correct-answer wording (not just the raw
         study-material excerpt) to be confirmed as WRONG options.
      6. Return the correct letter ("A", "B", or "C").

    A scenario is only accepted when this cleanly produces EXACTLY 1
    study-supported option and EXACTLY 2 unrelated options. If all 3
    look grounded, none look grounded, or 2 look grounded, this returns
    None so the caller rejects the scenario and generates a fresh one
    instead of guessing.
    """
    content_tokens = _tokenize(content)

    grounded_flags = {
        letter: _option_grounded_in_content(options[letter], content_tokens, content)
        for letter in ["A", "B", "C"]
    }

    grounded_letters = [letter for letter in ["A", "B", "C"] if grounded_flags[letter]]
    wrong_letters = [letter for letter in ["A", "B", "C"] if not grounded_flags[letter]]

    if len(grounded_letters) != 1 or len(wrong_letters) != 2:
        return None  # doesn't cleanly split into 1 correct + 2 wrong

    correct_answer = grounded_letters[0]
    correct_option_text = options[correct_answer]

    # Belt-and-braces: each auto-detected WRONG option must also share no
    # meaningful vocabulary with THIS scenario's own topic (its scenario
    # text, question text, and correct-answer wording), not just the raw
    # study-material excerpt, so a wrong option that happens to reuse a
    # topic word missing from the trimmed content chunk is still caught
    # and the scenario rejected.
    topic_tokens = _topic_tokens(scenario, question, correct_option_text)
    for letter in wrong_letters:
        if _option_is_topic_related(options[letter], topic_tokens):
            return None

    return correct_answer


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_and_clean(data, content):
    """
    A generated scenario is valid only when:
      - result is a dict
      - "scenario" is a non-empty string
      - "question" is a non-empty string
      - "options" is a dict with exactly the keys A, B, C
      - every option is a non-empty string, and all three are different
      - "explanation" is a non-empty string
      - the 3 options cleanly auto-classify (see
        _auto_classify_correct_answer) into exactly 1 study-supported
        CORRECT option + 2 unrelated WRONG options

    IMPORTANT: any "correct_answer" field in `data` (i.e. whatever
    Ollama returned) is intentionally never read here. The correct
    letter used is always the one independently computed by
    _auto_classify_correct_answer() from the option text itself, never
    a value the model (or a user) supplied.

    Returns a cleaned dict on success, or None on failure.
    """
    if not isinstance(data, dict):
        return None

    scenario = data.get("scenario")
    question = data.get("question")
    options = data.get("options")
    explanation = data.get("explanation")

    if not isinstance(scenario, str) or not scenario.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None

    if not isinstance(options, dict) or set(options.keys()) != {"A", "B", "C"}:
        return None

    cleaned_options = {}
    normalized_values = []
    for letter in ["A", "B", "C"]:
        value = options.get(letter)
        if not isinstance(value, str) or not value.strip():
            return None
        cleaned_options[letter] = value.strip()
        normalized_values.append(value.strip().lower())

    # Reject duplicate / equivalent options.
    if len(set(normalized_values)) != 3:
        return None

    scenario = scenario.strip()
    question = question.strip()
    explanation = explanation.strip()

    # ---- Auto-detect the correct answer. Ollama's own "correct_answer"
    # field is deliberately never consulted - see
    # _auto_classify_correct_answer()'s docstring. ----
    correct_answer = _auto_classify_correct_answer(scenario, question, cleaned_options, content)
    if correct_answer is None:
        return None  # doesn't cleanly split into 1 correct + 2 unrelated wrong options

    return {
        "scenario": scenario,
        "question": question,
        "options": cleaned_options,
        "correct_answer": correct_answer,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Generation — exactly ONE scenario, up to MAX_GENERATION_ATTEMPTS tries.
# ---------------------------------------------------------------------------

def _generate_scenario(content):
    for _attempt in range(MAX_GENERATION_ATTEMPTS):
        prompt = _build_prompt(content)
        try:
            raw_reply = generate_content(prompt)  # existing Ollama call
        except Exception:
            continue

        data = _parse_reply(raw_reply)
        if data is None:
            continue

        cleaned = _validate_and_clean(data, content)
        if cleaned is None:
            continue

        return {
            "scenario": cleaned["scenario"],
            "question": cleaned["question"],
            "options": cleaned["options"],
            "correct_answer": cleaned["correct_answer"],
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
    Advance the existing sequential Practice flow from Scenario Question to
    the next activity (Dynamic Challenge). Reuses whatever navigation
    variable the project already has; never introduces a second, competing
    one.
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

def show_scenarios(embedded=False):
    """
    Display the Scenario Question activity: exactly ONE round, then hand
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
        show_page_header("🎯", "Scenario Question", "Use what you learned to solve a realistic situation.")
        show_topic_bar(subject, topic)
        st.markdown("---")

    # Generate only when this activity is active and no scenario yet
    # exists in session state. Never regenerates on a plain rerun.
    if SESSION_KEY not in st.session_state:
        with st.spinner("Creating your Scenario round..."):
            round_data = _generate_scenario(content)

        if round_data is None:
            st.error(GENERATION_FAILURE_MESSAGE)
            return

        st.session_state[SESSION_KEY] = round_data

    data = st.session_state[SESSION_KEY]

    st.markdown("### 🎯 Scenario Question")

    st.markdown("#### 📖 Scenario")
    st.write(data["scenario"])

    st.markdown("#### ❓ Question")
    st.write(data["question"])

    option_letters = ["A", "B", "C"]
    option_labels = [f"{letter}. {data['options'][letter]}" for letter in option_letters]

    default_index = 0
    if data["user_answer"] in option_letters:
        default_index = option_letters.index(data["user_answer"])

    selected_label = st.radio(
        "Choose your answer:",
        option_labels,
        index=default_index,
        disabled=data["answered"],
        key="scenario_radio",
    )
    selected_letter = selected_label.split(".")[0].strip()
    if not data["answered"]:
        data["user_answer"] = selected_letter

    st.markdown("")

    if not data["answered"]:
        if st.button("✅ Submit Answer", key="scenario_submit"):
            data["user_answer"] = selected_letter
            data["answered"] = True

            is_correct = selected_letter == data["correct_answer"]
            if is_correct:
                _award_point_if_needed()

            if topic:
                log_answer(topic, is_correct, "Scenario")

            st.rerun()
    else:
        is_correct = data["user_answer"] == data["correct_answer"]

        if is_correct:
            st.success("✅ Correct!")
        else:
            st.error("❌ Wrong")

        correct_letter = data["correct_answer"]
        st.markdown(f"**Correct Answer:** {correct_letter}. {data['options'][correct_letter]}")
        st.markdown(f"**Explanation:** {data['explanation']}")

        st.markdown("")

        if st.button("➡️ Next Round", type="primary", key="scenario_next"):
            _advance_practice_flow()
            st.rerun()