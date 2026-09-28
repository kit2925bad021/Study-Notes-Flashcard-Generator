"""
Step 21 — Interactive Practice (sequential hub)
-------------------------------------------------
Strictly sequential Practice flow. The student sees exactly ONE activity
at a time, with exactly ONE generated item for that activity:

    Practice 1/5 -> Quiz             -> Next Round
    Practice 2/5 -> Truth & Trick    -> Next Round
    Practice 3/5 -> Emoji Decode     -> Next Round
    Practice 4/5 -> Scenario Question -> Next Round
    Practice 5/5 -> Dynamic Challenge -> Next Round
    -> Final Score

"➡️ Next Round" NEVER regenerates another item of the same activity - it
only advances to the next activity in the list above (or to the Final
Score screen after Dynamic Challenge).

ANSWER-RESULT UI (all five activities, via the shared _render_result() /
_get_answer_texts() below): after Submit, the button under the result
depends on the outcome, never both at once:
  - Correct  -> "✅ Correct!", the explanation, one motivational line
                ("🌟 ..."), then "➡️ Next Round" (advances to the next
                activity, as above).
  - Incorrect -> "❌ Incorrect", the student's own answer, the correct
                answer, the explanation (no motivational line), then
                "🔄 Try Again".
"🔄 Try Again" (on a wrong answer) is NOT the same as "➡️ Next Round":
it stays on the CURRENT activity, resets only that activity's
question/answer state, and generates ONE new item for that same
activity via _start_stage() - the same function used for every normal
stage load - so the score, the other four activities, the study
material, and every other session key are untouched. (A same-named
"🔄 Try Again" button also exists on the separate Final Score screen -
that one restarts the whole Practice attempt from Quiz; the two are
unrelated.)

Each activity is worth exactly 1 point. Max score = 5.
Percentage = score / 5 * 100.
Score >= 4/5 unlocks Flashcards; below that, Flashcards stay locked.

GENERATION: this file does not create its own Ollama client, hardcode a
model, or duplicate any activity's JSON parsing/validation logic. It
calls straight into each activity's own existing, already-tested "generate
ONE item" function (each of which calls generate_content() internally,
with its own retries and validation):

    steps/step3_quiz.py          -> _generate_one_question(subject, topic,
                                     content, language_note, avoid_questions)
    steps/step4_two_truths.py    -> _generate_truth_trick(content)
    steps/step5_emoji_decoder.py -> _generate_emoji_challenge(content, seen)
    steps/step6_scenarios.py     -> _generate_scenario(content)
    steps/step12_dynamic_questions.py -> _generate_dynamic_question(subject,
                                     topic, content, difficulty,
                                     previous_question=None,
                                     previous_answer=None,
                                     previous_result=None)

NOTE ON SIGNATURES: these are called with exactly the argument counts
their current definitions accept (checked directly against the files in
this project as of this revision). If any of those functions' signatures
change later, only the corresponding branch in _generate_for_stage()
below needs to be updated - never guess a signature.

NOTE ON RETRIES: _generate_one_question, _generate_truth_trick,
_generate_emoji_challenge, and _generate_scenario all retry internally
(up to 3 attempts) if Ollama's reply is malformed. _generate_dynamic_question
does not - it is exactly one prompt/parse pass with no retry of its own.
To give Dynamic Challenge the same reliability, this file retries that one
call itself (see MAX_DYNAMIC_ATTEMPTS / the "dynamic" branch of
_generate_for_stage), without modifying step12_dynamic_questions.py.

Ollama is called ONLY when a stage starts: the very first time the
student reaches that activity, when they click "🔁 Retry" after a failed
generation, or when they click "🔄 Try Again" after a wrong answer (which
generates one new item for that same activity, per the ANSWER-RESULT UI
note above) - never on a plain Streamlit rerun (selecting an option,
clicking Submit, etc.).

SOURCE OF TRUTH: st.session_state["study_material"]["content"] (plus
subject/topic from the same dict). No uploaded_text or other source
variable is introduced.

This file does not modify steps/step3_quiz.py, step4_two_truths.py,
step5_emoji_decoder.py, step6_scenarios.py, step12_dynamic_questions.py,
services/ollama_service.py, or app.py - their standalone pages
(embedded=False) keep working exactly as before.

SESSION STATE: uses the flat practice_* keys already established for this
hub (practice_stage, practice_score, practice_current_item, etc.) rather
than introducing a second, competing state container.

Imported in app.py with:
    from steps.step21_practice import show_practice
and called with no arguments: show_practice()

FIX (this revision): Final Score's button is now labeled "🃏 Open
Flashcards" instead of "🔓 Unlock Flashcards" - unlocking already happens
automatically the moment the score is shown (flashcards_unlocked is set
above, before this button renders), and is now surfaced as a plain
st.success() line instead of something the student has to click. The
button's own behavior (navigate to "📝 Flashcards", no Ollama call) is
unchanged; step2_flashcards.py now auto-generates the deck on arrival, so
"Open Flashcards" really does open a ready deck, not an empty page
waiting for another click.
"""

import random

import streamlit as st

from steps.step13_memory import log_answer, record_topic
from steps.step18_settings import language_instruction
from steps.step19_ui import show_topic_bar

from steps.step3_quiz import _generate_one_question
from steps.step4_two_truths import _generate_truth_trick
from steps.step5_emoji_decoder import (
    _generate_emoji_challenge,
    _normalize_concept,
    _significant_words,
)

# Used only by the Emoji "Show Answer" reveal (max 3 words). Imported
# separately so that an older step5_emoji_decoder.py without it can never
# stop this whole page from loading; the fallback simply keeps 3 words.
try:
    from steps.step5_emoji_decoder import _shorten_answer
except ImportError:
    def _shorten_answer(answer, concept=""):
        words = " ".join(str(answer or "").split()).strip(" \t\"'.,;:!?").split()
        return " ".join(words[:3])
from steps.step6_scenarios import _generate_scenario
from steps.step12_dynamic_questions import _generate_dynamic_question

# ----------------------------------------------------------------------
# Stage configuration
# ----------------------------------------------------------------------

STAGE_ORDER = ["quiz", "truth_trick", "emoji", "scenario", "dynamic"]

STAGE_INFO = {
    "quiz": {"number": 1, "emoji": "📝", "label": "Quiz"},
    "truth_trick": {"number": 2, "emoji": "🎭", "label": "Truth & Trick"},
    "emoji": {"number": 3, "emoji": "🪄", "label": "Emoji Decode"},
    "scenario": {"number": 4, "emoji": "🎯", "label": "Scenario Question"},
    "dynamic": {"number": 5, "emoji": "🔀", "label": "Dynamic Challenge"},
}

TOTAL_ACTIVITIES = len(STAGE_ORDER)
PASS_SCORE = 4  # score >= 4/5 unlocks Flashcards

# _generate_dynamic_question() (step12_dynamic_questions.py) makes exactly
# ONE Ollama call and returns None on any parsing/shape failure - unlike
# every other stage's generator, it has no retry loop of its own. Retrying
# it here gives Dynamic Challenge the same reliability the other four
# stages already get internally, without touching step12_dynamic_questions.py.
MAX_DYNAMIC_ATTEMPTS = 3

# ----------------------------------------------------------------------
# Motivational feedback: one short "🌟 ..." line, shown only alongside a
# CORRECT answer (see _render_result). One line is chosen at random each
# time so the same message doesn't repeat every round. A wrong answer
# gets no motivational line - just the result block itself.
# ----------------------------------------------------------------------

CORRECT_LINES = [
    "Great job! You got it! 🎉",
    "Excellent! Keep it up! 🌟",
    "Well done! You're on the right track! 💜",
]

# Widget keys that must be cleared whenever a new item is generated (or a
# new Practice attempt starts), otherwise a leftover selection from a
# previous question could silently pre-select an option on the new one.
_WIDGET_KEYS = [
    "practice_quiz_radio",
    "practice_tt_radio",
    "practice_emoji_input",
    "practice_emoji_answer_shown",
    "practice_scenario_radio",
    "practice_dynamic_radio",
]


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

def _empty_history():
    return {
        "quiz_seen": [],
        "emoji_seen": [],
    }


def _init_practice_state():
    defaults = {
        "practice_stage": "quiz",
        "practice_score": 0,
        "practice_percentage": 0,
        "practice_round_loaded": False,
        "practice_current_item": None,
        "practice_selected": None,
        "practice_submitted": False,
        "practice_last_correct": None,
        "practice_feedback_line": None,
        "practice_error": None,
        "practice_history": _empty_history(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Flashcard unlock flag - False until a Practice attempt scores >= 4/5.
    # Reused as-is if some other part of the app already set it.
    if "flashcards_unlocked" not in st.session_state:
        st.session_state["flashcards_unlocked"] = False


def _clear_widget_keys():
    for key in _WIDGET_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def _reset_practice():
    """Start a brand-new Practice attempt from the very first activity."""
    st.session_state.practice_stage = "quiz"
    st.session_state.practice_score = 0
    st.session_state.practice_percentage = 0
    st.session_state.practice_round_loaded = False
    st.session_state.practice_current_item = None
    st.session_state.practice_selected = None
    st.session_state.practice_submitted = False
    st.session_state.practice_last_correct = None
    st.session_state.practice_feedback_line = None
    st.session_state.practice_error = None
    st.session_state.practice_history = _empty_history()
    # A fresh attempt starts locked again - the previous attempt's unlock
    # (if any) does not carry over.
    st.session_state["flashcards_unlocked"] = False
    if "practice_state" in st.session_state:
        st.session_state["practice_state"]["flashcards_unlocked"] = False
    _clear_widget_keys()


# ----------------------------------------------------------------------
# Generation (ONE item per stage, via each activity's own generator)
# ----------------------------------------------------------------------

def _generate_for_stage(stage, subject, topic, content):
    """
    Call the right activity's existing single-item generator. Returns
    {"kind": stage, "data": <item dict>} on success, or None if that
    activity's generator could not produce a valid item after its own
    internal retries.
    """
    history = st.session_state.practice_history

    if stage == "quiz":
        avoid = list(history["quiz_seen"])[-8:]
        try:
            question = _generate_one_question(subject, topic, content, language_instruction(), avoid)
        except Exception:
            question = None
        if question is None:
            return None
        history["quiz_seen"].append(question["question"])
        return {"kind": "quiz", "data": question}

    if stage == "truth_trick":
        try:
            round_data = _generate_truth_trick(content)
        except Exception:
            round_data = None
        if round_data is None:
            return None
        return {"kind": "truth_trick", "data": round_data}

    if stage == "emoji":
        try:
            challenge = _generate_emoji_challenge(content, history["emoji_seen"])
        except Exception:
            challenge = None
        if challenge is None:
            return None
        history["emoji_seen"].append(challenge["concept"])
        return {"kind": "emoji", "data": challenge}

    if stage == "scenario":
        try:
            scenario = _generate_scenario(content)
        except Exception:
            scenario = None
        if scenario is None:
            return None
        return {"kind": "scenario", "data": scenario}

    if stage == "dynamic":
        # See MAX_DYNAMIC_ATTEMPTS above: this function has no retry of its
        # own, so a single malformed reply from the model must not fail the
        # whole stage - retry a fresh call, exactly as the other stages'
        # own generators already do internally.
        for _attempt in range(MAX_DYNAMIC_ATTEMPTS):
            try:
                question = _generate_dynamic_question(
                    subject, topic, content, "Medium",
                    previous_question=None, previous_answer=None, previous_result=None,
                )
            except Exception:
                question = None
            if question is not None:
                return {"kind": "dynamic", "data": question}
        return None

    return None


def _start_stage(stage, subject, topic, content):
    """
    Generate the ONE item for `stage` and store it as the current item,
    resetting the answer state. Returns True on success. On failure,
    stores a retry-visible error message and returns False.
    """
    label = STAGE_INFO[stage]["label"]
    with st.spinner(f"Creating your {label} round..."):
        item = _generate_for_stage(stage, subject, topic, content)

    _clear_widget_keys()

    if item is None:
        st.session_state.practice_error = f"Could not create a valid {label} round. Please try again."
        st.session_state.practice_round_loaded = False
        return False

    st.session_state.practice_current_item = item
    st.session_state.practice_selected = None
    st.session_state.practice_submitted = False
    st.session_state.practice_last_correct = None
    st.session_state.practice_feedback_line = None
    st.session_state.practice_error = None
    st.session_state.practice_round_loaded = True
    return True


# ----------------------------------------------------------------------
# Shared submit / result helpers
# ----------------------------------------------------------------------

def _submit_answer(is_correct, topic, activity_label, difficulty=None):
    st.session_state.practice_submitted = True
    st.session_state.practice_last_correct = is_correct
    # Exactly one short motivational line, chosen at random so it varies
    # round to round. Only shown on a correct answer (see
    # _render_result) - a wrong answer never gets this line.
    st.session_state.practice_feedback_line = (
        random.choice(CORRECT_LINES) if is_correct else None
    )
    if is_correct:
        st.session_state.practice_score += 1
    if difficulty is not None:
        log_answer(topic, is_correct, activity_label, difficulty)
    else:
        log_answer(topic, is_correct, activity_label)
    st.rerun()


def _get_answer_texts(stage, data, selected):
    """
    Return (your_answer_text, correct_answer_text) for the wrong-answer
    result view, shaped per activity. Only called after submission, so
    the correct answer is never exposed before then.
    """
    if stage in ("quiz", "scenario"):
        options = data["options"]
        correct = data["correct_answer"]
        your_text = f"{selected}. {options[selected]}" if selected in options else "No answer"
        correct_text = f"{correct}. {options[correct]}"
        return your_text, correct_text
    if stage == "truth_trick":
        statements = data["statements"]
        lie = data["lie"]
        your_text = (
            statements[selected]
            if isinstance(selected, int) and 0 <= selected < len(statements)
            else "No answer"
        )
        correct_text = statements[lie]
        return your_text, correct_text
    if stage == "emoji":
        your_text = selected.strip() if isinstance(selected, str) and selected.strip() else "No answer"
        correct_text = data["answer"]
        return your_text, correct_text
    if stage == "dynamic":
        options = data["options"]
        correct = data["correct_answer"]
        your_text = (
            options[selected]
            if isinstance(selected, int) and 0 <= selected < len(options)
            else "No answer"
        )
        correct_text = (
            options[correct]
            if isinstance(correct, int) and 0 <= correct < len(options)
            else str(correct)
        )
        return your_text, correct_text
    return "No answer", ""


def _render_result(stage, data):
    """
    Render the full answer-result block for any of the five Practice
    activities, right below the question, once the current round has
    been submitted:
      - Correct: "✅ Correct!", then the explanation, then exactly one
        short motivational line ("🌟 ...").
      - Incorrect: "❌ Incorrect", the student's own answer, the
        correct answer, then the explanation - no motivational line.
    This is the single source of truth for that block so all five
    activities show it identically; only the "Next Round"/"Try Again"
    button below it (see show_practice()) differs by outcome.
    """
    is_correct = st.session_state.practice_last_correct
    explanation = data.get("explanation", "")

    if is_correct:
        st.success("✅ Correct!")
        st.markdown(f"**💡 Explanation:**  \n{explanation}")
        line = st.session_state.practice_feedback_line or random.choice(CORRECT_LINES)
        st.markdown(f"🌟 {line}")
    else:
        selected = st.session_state.practice_selected
        your_text, correct_text = _get_answer_texts(stage, data, selected)
        st.error("❌ Incorrect")
        st.markdown(f"**Your answer:** {your_text}")
        st.markdown(f"**Correct answer:** {correct_text}")
        st.markdown(f"**💡 Explanation:**  \n{explanation}")


def _answers_match(user_answer, correct_answer):
    """Lenient correctness check for the Emoji Decode free-text answer."""
    user_answer = (user_answer or "").strip()
    if not user_answer:
        return False
    if _normalize_concept(user_answer) == _normalize_concept(correct_answer):
        return True
    user_words = _significant_words(user_answer)
    correct_words = _significant_words(correct_answer)
    if not user_words or not correct_words:
        return False
    overlap = len(user_words & correct_words) / len(correct_words)
    return overlap >= 0.6


# ----------------------------------------------------------------------
# Per-activity rendering
# ----------------------------------------------------------------------

def _render_quiz(data, topic):
    st.markdown(f"### {data['question']}")
    option_keys = ["A", "B", "C", "D"]
    submitted = st.session_state.practice_submitted

    selected = st.session_state.practice_selected
    default_index = option_keys.index(selected) if selected in option_keys else None

    chosen = st.radio(
        "Choose your answer:",
        option_keys,
        index=default_index,
        format_func=lambda key: f"{key}. {data['options'][key]}",
        disabled=submitted,
        key="practice_quiz_radio",
    )
    if chosen is not None and not submitted:
        st.session_state.practice_selected = chosen

    if not submitted:
        if st.button("✅ Submit Answer", disabled=st.session_state.practice_selected is None, key="practice_submit"):
            _submit_answer(st.session_state.practice_selected == data["correct_answer"], topic, "Quiz")
    else:
        _render_result("quiz", data)


def _render_truth_trick(data, topic):
    st.markdown("**Which statement is false?**")
    labels = [f"{i + 1}. {data['statements'][i]}" for i in range(3)]
    submitted = st.session_state.practice_submitted

    selected = st.session_state.practice_selected  # 0-based index once chosen
    default_index = selected if isinstance(selected, int) else None

    chosen_label = st.radio(
        "Choose the statement you think is false:",
        labels,
        index=default_index,
        disabled=submitted,
        key="practice_tt_radio",
    )
    if chosen_label is not None and not submitted:
        st.session_state.practice_selected = labels.index(chosen_label)

    if not submitted:
        if st.button("✅ Submit Answer", disabled=st.session_state.practice_selected is None, key="practice_submit"):
            _submit_answer(st.session_state.practice_selected == data["lie"], topic, "Truth & Trick")
    else:
        _render_result("truth_trick", data)


def _render_emoji(data, topic):
    st.markdown(f"#### 💡 {data['concept']}")
    st.markdown(
        f"<div style='font-size:2rem; text-align:center; padding: 0.5rem 0;'>"
        f"{data['emoji_sequence']}</div>",
        unsafe_allow_html=True,
    )
    if data.get("hint"):
        st.caption(f"💡 Hint: {data['hint']}")

    submitted = st.session_state.practice_submitted

    if not submitted:
        # 👁️ Show Answer only REVEALS the short (max 3 words) answer - it
        # never submits, never marks anything correct, and never touches
        # the typed answer (the input keeps its own session-state key) or
        # the current item, so nothing is regenerated. The flag is cleared
        # with the other per-item keys (Try Again / Next Round / restart).
        if st.session_state.get("practice_emoji_answer_shown", False):
            short_answer = _shorten_answer(data["answer"], data["concept"])
            st.markdown(f"**💡 Answer:** `{short_answer}`")
        elif st.button("👁️ Show Answer", use_container_width=True, key="practice_emoji_show_answer"):
            st.session_state.practice_emoji_answer_shown = True
            st.rerun()

        # The input stays visible before AND after the reveal; the typed
        # answer is only evaluated when Submit Answer is clicked.
        answer_text = st.text_input("Your Answer:", key="practice_emoji_input")
        if st.button("✅ Submit Answer", disabled=not answer_text.strip(), key="practice_submit"):
            st.session_state.practice_selected = answer_text
            _submit_answer(_answers_match(answer_text, data["answer"]), topic, "Emoji Decode")
    elif st.session_state.practice_last_correct:
        st.success("✅ Correct!")
        st.markdown("Excellent! You understood the concept.")
    else:
        # ❌ Incorrect / Your answer / Correct answer / 💡 Explanation
        # (unchanged shared block); "🔄 Try Again" comes from show_practice().
        _render_result("emoji", data)


def _render_scenario(data, topic):
    st.markdown("#### 📖 Scenario")
    st.write(data["scenario"])
    st.markdown("#### ❓ Question")
    st.write(data["question"])

    option_letters = ["A", "B", "C"]
    submitted = st.session_state.practice_submitted

    selected = st.session_state.practice_selected
    default_index = option_letters.index(selected) if selected in option_letters else None

    chosen = st.radio(
        "Choose your answer:",
        option_letters,
        index=default_index,
        format_func=lambda letter: f"{letter}. {data['options'][letter]}",
        disabled=submitted,
        key="practice_scenario_radio",
    )
    if chosen is not None and not submitted:
        st.session_state.practice_selected = chosen

    if not submitted:
        if st.button("✅ Submit Answer", disabled=st.session_state.practice_selected is None, key="practice_submit"):
            _submit_answer(st.session_state.practice_selected == data["correct_answer"], topic, "Scenario")
    else:
        _render_result("scenario", data)


def _render_dynamic(data, topic):
    st.markdown(f"**{data['question']}**")
    if data.get("difficulty"):
        st.caption(f"Difficulty: {data['difficulty']}")

    options = data["options"]
    indices = list(range(len(options)))
    submitted = st.session_state.practice_submitted

    default_index = st.session_state.practice_selected if isinstance(st.session_state.practice_selected, int) else None

    chosen = st.radio(
        "Choose an answer:",
        indices,
        index=default_index,
        format_func=lambda i: options[i],
        disabled=submitted,
        key="practice_dynamic_radio",
    )
    if chosen is not None and not submitted:
        st.session_state.practice_selected = chosen

    if not submitted:
        if st.button("✅ Submit Answer", disabled=st.session_state.practice_selected is None, key="practice_submit"):
            is_correct = st.session_state.practice_selected == data["correct_answer"]
            _submit_answer(is_correct, topic, "Dynamic Challenge", data.get("difficulty"))
    else:
        _render_result("dynamic", data)


_RENDERERS = {
    "quiz": _render_quiz,
    "truth_trick": _render_truth_trick,
    "emoji": _render_emoji,
    "scenario": _render_scenario,
    "dynamic": _render_dynamic,
}


# ----------------------------------------------------------------------
# Stage transitions
# ----------------------------------------------------------------------

def _advance_stage(topic):
    current = st.session_state.practice_stage
    idx = STAGE_ORDER.index(current)

    if idx + 1 < len(STAGE_ORDER):
        st.session_state.practice_stage = STAGE_ORDER[idx + 1]
        st.session_state.practice_round_loaded = False
        st.session_state.practice_current_item = None
        st.session_state.practice_selected = None
        st.session_state.practice_submitted = False
        st.session_state.practice_last_correct = None
        st.session_state.practice_feedback_line = None
        st.session_state.practice_error = None
        _clear_widget_keys()
    else:
        st.session_state.practice_stage = "complete"
        st.session_state.practice_percentage = round(
            st.session_state.practice_score / TOTAL_ACTIVITIES * 100
        )
        record_topic(topic)


# ----------------------------------------------------------------------
# Final score screen
# ----------------------------------------------------------------------

def _show_final_score():
    score = st.session_state.practice_score
    percentage = st.session_state.practice_percentage
    passed = score >= PASS_SCORE

    st.markdown("## 🎉 Practice Completed!")
    st.markdown(f"**Score:** {score} / {TOTAL_ACTIVITIES}")
    st.markdown(f"**Percentage:** {percentage}%")

    # Passing the final score only marks Flashcards as eligible to unlock -
    # it does NOT open/navigate to Flashcards and it does NOT call Ollama.
    # The existing practice_state container (if already used elsewhere to
    # gate Flashcards) is kept in sync with the top-level flag instead of
    # introducing a second, competing source of truth.
    st.session_state["flashcards_unlocked"] = passed
    st.session_state.setdefault("practice_state", {})
    st.session_state["practice_state"]["flashcards_unlocked"] = passed

    if passed:
        # Unlocking itself already happened above (flashcards_unlocked was
        # just set to True) - "🎉 Flashcards Unlocked!" is shown here only
        # as that unlock notice, never as a second click the student has
        # to make. The button below is the "open it" step: on explicit click,
        # it navigates to the Flashcards page, where the deck is generated
        # automatically (see step2_flashcards.py) so it's already there
        # waiting - no separate "Generate Flashcards" click needed either.
        st.success("🎉 Flashcards Unlocked!")
        if st.button("🃏 Open Flashcards", type="primary", key="practice_open_flashcards"):
            # Only now - on explicit click - do we open Flashcards. This
            # reuses the app's existing page-navigation mechanism; no new
            # navigation system and no Ollama call happen here.
            st.session_state.current_page = "📝 Flashcards"
            st.rerun()
    else:
        st.markdown("### 🔒 Flashcards Locked")
        st.warning("Review the study material and try the practice again.")

        st.markdown("")
        if st.button("🔄 Try Practice Again", type="primary", key="practice_try_again"):
            # A full-session restart (unlike the per-round "🔄 Try Again" in
            # _render_result/show_practice, which stays on the current
            # activity): back to Quiz, score reset to 0, previous activity
            # data cleared. practice_round_loaded is left False by
            # _reset_practice(), so the normal stage-load path at the top
            # of show_practice() generates a fresh Quiz question on the
            # very next run - study material, flashcards, login and every
            # other part of the app are untouched.
            _reset_practice()
            st.rerun()

    st.markdown("")
    if st.button("🏠 Back to Dashboard", key="practice_back_to_dashboard"):
        # Same page-switch mechanism app.py's own nav uses (current_page).
        st.session_state.current_page = "🏠 Dashboard"
        st.rerun()


# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------

def show_practice():
    """Display the full, strictly-sequential Practice flow."""
    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("Please create Study Material first.")
        return

    subject = study_material["subject"]
    topic = study_material["topic"]
    content = study_material["content"]

    st.markdown("## 🎯 Interactive Practice")
    show_topic_bar(subject, topic)
    st.markdown("---")

    _init_practice_state()

    stage = st.session_state.practice_stage

    if stage == "complete":
        _show_final_score()
        return

    info = STAGE_INFO[stage]
    st.markdown(f"**Practice {info['number']}/{TOTAL_ACTIVITIES} — {info['label']}**")
    st.progress(info["number"] / TOTAL_ACTIVITIES)
    st.markdown("---")

    # ---- Generate the ONE item for this stage, only if it hasn't been
    # generated yet (never on a plain rerun). ----
    if not st.session_state.practice_round_loaded:
        if st.session_state.practice_error:
            st.error(f"⚠️ {st.session_state.practice_error}")
            if st.button("🔁 Retry", type="primary", key="practice_retry"):
                _start_stage(stage, subject, topic, content)
                st.rerun()
            return

        # Always rerun immediately after attempting generation, whether it
        # succeeded or failed, so the page is never left blank: on the
        # next run, practice_round_loaded is either True (item renders
        # below) or practice_error is set (error+Retry branch renders).
        _start_stage(stage, subject, topic, content)
        st.rerun()

    st.markdown(f"### {info['emoji']} {info['label']}")
    _RENDERERS[stage](st.session_state.practice_current_item["data"], topic)

    if st.session_state.practice_submitted:
        st.markdown("")
        if st.session_state.practice_last_correct:
            # Label names the next stage ("➡️ Continue to Truth & Trick", ...);
            # after the last stage it leads to the final screen.
            _next_idx = STAGE_ORDER.index(stage) + 1
            _next_label = (
                f"➡️ Continue to {STAGE_INFO[STAGE_ORDER[_next_idx]]['label']}"
                if _next_idx < len(STAGE_ORDER)
                else "➡️ Finish Practice"
            )
            if st.button(_next_label, type="primary", key="practice_next_round"):
                _advance_stage(topic)
                st.rerun()
        else:
            if st.button("🔄 Try Again", type="primary", key="practice_try_again_round"):
                _start_stage(stage, subject, topic, content)
                st.rerun()