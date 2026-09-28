"""
Step 10 — Advanced Flashcard Modes
-------------------------------------
Renders the "🎴 Advanced Flashcard Modes" page inside the Streamlit app.

This page lets the student study the SAME topic (from the study material
already generated on the Dashboard) using five different question styles:
Normal Q&A, Two Truths and a Lie, Emoji Decoder, Scenario Question, and
Reverse Question.

Wire this up in app.py the same way Steps 1, 2, and 9 are wired up, e.g.:

    from steps.step10_flashcard_modes import show_flashcard_modes
    ...
    elif st.session_state.current_page == "🎴 Advanced Flashcard Modes":
        show_flashcard_modes()

This file does NOT modify st.session_state.study_material, the existing
flashcards (Step 2), or any other step. It only reads study_material and
writes to its own session_state keys (all prefixed with "flashcard_mode").
"""

import json
import re

import streamlit as st

from services.ollama_service import generate_content
from steps.step19_ui import show_page_header, show_topic_bar


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

MODE_OPTIONS = [
    "🧠 Normal Q&A",
    "🎭 Two Truths and a Lie",
    "😀 Emoji Decoder",
    "🎯 Scenario Question",
    "🔄 Reverse Question",
]

NUM_QUESTIONS = 5


# ----------------------------------------------------------------------
# 1. PROMPT BUILDER
# ----------------------------------------------------------------------

def _build_mode_prompt(subject, topic, content, mode):
    """
    Build a prompt asking Ollama for exactly 5 items in the JSON shape
    that matches the selected mode. All content must come from the given
    study material only — no unrelated/invented facts.
    """

    # Shared instructions used in every mode's prompt.
    shared_intro = f"""
You are helping a student study using their existing study material.

Subject: {subject}
Topic: {topic}

Study material (this is the ONLY source of facts you may use — do not
invent unrelated facts):
{content}

Use simple, beginner-friendly English. Generate exactly {NUM_QUESTIONS}
items. Respond with ONLY valid JSON (no markdown code fences, no extra
commentary before or after it).
"""

    if mode == "🧠 Normal Q&A":
        schema = """
Respond in this exact JSON shape:
{
  "items": [
    {
      "question": "string",
      "answer": "string"
    }
  ]
}
"""

    elif mode == "🎭 Two Truths and a Lie":
        schema = """
For each item, write exactly 3 short statements about the topic: 2 must be
TRUE and 1 must be FALSE. "false_index" is the 0-based index (0, 1, or 2)
of the false statement in the "statements" list.

Respond in this exact JSON shape:
{
  "items": [
    {
      "statements": ["string", "string", "string"],
      "false_index": 0,
      "explanation": "string explaining why that statement is false"
    }
  ]
}
"""

    elif mode == "😀 Emoji Decoder":
        schema = """
For each item, choose a short sequence of emojis (2-5 emojis) that
represents a concept from the study material.

Respond in this exact JSON shape:
{
  "items": [
    {
      "emojis": "string of emojis",
      "answer": "the concept the emojis represent",
      "explanation": "short explanation connecting the emojis to the concept"
    }
  ]
}
"""

    elif mode == "🎯 Scenario Question":
        schema = """
For each item, describe a short, realistic college/everyday-life scenario
where a concept from the study material applies, then give a
multiple-choice question about which concept applies, with exactly 4
options where only one is correct.

Respond in this exact JSON shape:
{
  "items": [
    {
      "scenario": "short real-life scenario",
      "question": "which concept applies here?",
      "options": ["string", "string", "string", "string"],
      "correct_answer": "the exact text of the correct option",
      "explanation": "short explanation of why that option is correct"
    }
  ]
}
"""

    elif mode == "🔄 Reverse Question":
        schema = """
For each item, give a concept/answer FIRST, then the question that the
student should try to come up with that would lead to that answer.

Respond in this exact JSON shape:
{
  "items": [
    {
      "concept": "the answer/concept shown to the student first",
      "question": "the expected question that leads to this concept",
      "explanation": "short explanation connecting the concept and question"
    }
  ]
}
"""

    else:
        # Fallback — should not normally happen since the selectbox
        # restricts choices to the modes above.
        schema = """
Respond in this exact JSON shape:
{
  "items": [
    {
      "question": "string",
      "answer": "string"
    }
  ]
}
"""

    return shared_intro + schema


# ----------------------------------------------------------------------
# 2. GENERATION
# ----------------------------------------------------------------------

def _generate_mode_content(subject, topic, content, mode):
    """
    Build the prompt for the given mode, call Ollama, and return the raw
    text response. Any exception from generate_content() is left to the
    caller (show_flashcard_modes) to catch and display gracefully.
    """
    prompt = _build_mode_prompt(subject, topic, content, mode)
    raw_text = generate_content(prompt)
    return raw_text


# ----------------------------------------------------------------------
# 3. SAFE JSON PARSING
# ----------------------------------------------------------------------

def _safe_parse_json(text):
    """
    Try to safely parse Ollama's response as JSON.

    Handles the common case where Ollama wraps the JSON in ```json ... ```
    code fences even when asked not to. Returns the parsed "items" list on
    success, or None if parsing fails for any reason (never raises).
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```).
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    items = data.get("items")
    if not isinstance(items, list) or len(items) == 0:
        return None

    return items


# ----------------------------------------------------------------------
# 4. PAGE UI
# ----------------------------------------------------------------------

def show_flashcard_modes():
    """Render the full Advanced Flashcard Modes page (Step 10)."""

    show_page_header("🎴", "Advanced Flashcard Modes", "Study the same topic in different question styles.")

    # ------------------------------------------------------------
    # Require study material from the Dashboard first.
    # ------------------------------------------------------------
    study_material = st.session_state.get("study_material")

    if not study_material:
        st.warning("Please generate study material from the Dashboard first.")
        return

    subject = study_material.get("subject", "")
    topic = study_material.get("topic", "")
    content = study_material.get("content", "")

    show_topic_bar(subject, topic)

    st.markdown("---")

    # ------------------------------------------------------------
    # Mode selector + generate button.
    # ------------------------------------------------------------
    selected_mode = st.selectbox("Choose Learning Mode", MODE_OPTIONS)

    generate_clicked = st.button("✨ Generate Questions", type="primary")

    if generate_clicked:
        with st.spinner("Generating your questions..."):
            try:
                raw_text = _generate_mode_content(subject, topic, content, selected_mode)
                items = _safe_parse_json(raw_text)

                # Store everything needed to render this set of questions.
                st.session_state.flashcard_modes = {
                    "mode": selected_mode,
                    "items": items,          # None if parsing failed
                    "raw_text": raw_text,
                }

                # Reset navigation + any previously checked answers whenever
                # a fresh set of questions is generated.
                st.session_state.flashcard_mode_index = 0
                st.session_state.flashcard_mode_answers = {}

                if items:
                    st.success("✅ Questions generated!")
                else:
                    st.warning(
                        "Generated content, but couldn't parse it into "
                        "structured questions. Showing the raw result below."
                    )
            except Exception as e:
                # Handle Ollama / API errors gracefully — never crash the page.
                st.error(f"Sorry, we couldn't generate questions: {e}")
                return

    # ------------------------------------------------------------
    # Nothing generated yet.
    # ------------------------------------------------------------
    flashcard_modes = st.session_state.get("flashcard_modes")
    if not flashcard_modes:
        st.info("Choose a mode above and click **✨ Generate Questions** to begin.")
        return

    items = flashcard_modes.get("items")
    generated_mode = flashcard_modes.get("mode")
    raw_text = flashcard_modes.get("raw_text", "")

    # Safe fallback: if we couldn't parse JSON, just show the raw text
    # instead of crashing or rendering broken widgets.
    if not items:
        st.markdown("---")
        st.markdown(raw_text if raw_text else "_No content was returned._")
        return

    # ------------------------------------------------------------
    # Navigation state.
    # ------------------------------------------------------------
    if "flashcard_mode_index" not in st.session_state:
        st.session_state.flashcard_mode_index = 0
    if "flashcard_mode_answers" not in st.session_state:
        st.session_state.flashcard_mode_answers = {}

    total = len(items)
    # Keep index in range in case a regenerate returned fewer items.
    if st.session_state.flashcard_mode_index > total - 1:
        st.session_state.flashcard_mode_index = 0

    st.markdown("---")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.flashcard_mode_index > 0:
                st.session_state.flashcard_mode_index -= 1
    with nav_col2:
        current_num = st.session_state.flashcard_mode_index + 1
        st.markdown(
            f"<p style='text-align:center;'>Question {current_num} of {total}</p>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("➡️ Next", use_container_width=True):
            if st.session_state.flashcard_mode_index < total - 1:
                st.session_state.flashcard_mode_index += 1

    current_item = items[st.session_state.flashcard_mode_index]
    current_index = st.session_state.flashcard_mode_index

    # Render using the mode that was active when the questions were
    # generated (in case the student changes the selectbox afterwards
    # without regenerating).
    with st.container(border=True):
        if generated_mode == "🧠 Normal Q&A":
            _render_normal_qa(current_item)
        elif generated_mode == "🎭 Two Truths and a Lie":
            _render_two_truths_and_a_lie(current_item, current_index)
        elif generated_mode == "😀 Emoji Decoder":
            _render_emoji_decoder(current_item, current_index)
        elif generated_mode == "🎯 Scenario Question":
            _render_scenario_question(current_item, current_index)
        elif generated_mode == "🔄 Reverse Question":
            _render_reverse_question(current_item, current_index)
        else:
            st.write(current_item)


# ----------------------------------------------------------------------
# MODE-SPECIFIC RENDERERS
# ----------------------------------------------------------------------

def _render_normal_qa(item):
    """🧠 Normal Q&A: question, with the answer behind an expander."""
    question = item.get("question", "Question not available.")
    answer = item.get("answer", "Answer not available.")

    st.markdown(f"**Question:** {question}")
    with st.expander("👁️ Show Answer"):
        st.write(answer)


def _render_two_truths_and_a_lie(item, index):
    """🎭 Two Truths and a Lie: pick the false statement."""
    statements = item.get("statements", [])
    false_index = item.get("false_index")
    explanation = item.get("explanation", "")

    st.markdown("**Which statement is FALSE?**")

    answer_key = f"ttal_{index}"
    selected = st.radio(
        "Select the false statement:",
        options=list(range(len(statements))),
        format_func=lambda i: statements[i] if i < len(statements) else "",
        key=f"ttal_radio_{index}",
    )

    if st.button("Check Answer", key=f"ttal_check_{index}"):
        st.session_state.flashcard_mode_answers[answer_key] = selected

    if answer_key in st.session_state.flashcard_mode_answers:
        chosen = st.session_state.flashcard_mode_answers[answer_key]
        if chosen == false_index:
            st.success("✅ Correct! That statement is the lie.")
        else:
            st.error("❌ Not quite — that statement is actually true.")
        st.markdown(f"**Explanation:** {explanation}")


def _render_emoji_decoder(item, index):
    """😀 Emoji Decoder: guess the concept from emojis."""
    emojis = item.get("emojis", "")
    answer = item.get("answer", "")
    explanation = item.get("explanation", "")

    st.markdown(f"<h2 style='text-align:center;'>{emojis}</h2>", unsafe_allow_html=True)

    guess_key = f"emoji_guess_{index}"
    st.text_input("What concept do these emojis represent?", key=guess_key)

    check_key = f"emoji_checked_{index}"
    if st.button("Check Answer", key=f"emoji_check_{index}"):
        st.session_state.flashcard_mode_answers[check_key] = True

    if st.session_state.flashcard_mode_answers.get(check_key):
        st.markdown(f"**Correct Answer:** {answer}")
        st.markdown(f"**Explanation:** {explanation}")


def _render_scenario_question(item, index):
    """🎯 Scenario Question: multiple-choice based on a short scenario."""
    scenario = item.get("scenario", "")
    question = item.get("question", "")
    options = item.get("options", [])
    correct_answer = item.get("correct_answer", "")
    explanation = item.get("explanation", "")

    st.markdown(f"**Scenario:** {scenario}")
    st.markdown(f"**Question:** {question}")

    answer_key = f"scenario_{index}"
    selected = st.radio(
        "Choose the best answer:",
        options=options,
        key=f"scenario_radio_{index}",
    )

    if st.button("Check Answer", key=f"scenario_check_{index}"):
        st.session_state.flashcard_mode_answers[answer_key] = selected

    if answer_key in st.session_state.flashcard_mode_answers:
        chosen = st.session_state.flashcard_mode_answers[answer_key]
        if chosen == correct_answer:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Not quite. The correct answer is: {correct_answer}")
        st.markdown(f"**Explanation:** {explanation}")


def _render_reverse_question(item, index):
    """🔄 Reverse Question: given the concept, guess the question."""
    concept = item.get("concept", "")
    question = item.get("question", "")
    explanation = item.get("explanation", "")

    st.markdown(f"**Concept / Answer:** {concept}")
    st.caption("Think about what question would lead to this answer.")

    reveal_key = f"reverse_revealed_{index}"
    if st.button("Check Answer", key=f"reverse_check_{index}"):
        st.session_state.flashcard_mode_answers[reveal_key] = True

    if st.session_state.flashcard_mode_answers.get(reveal_key):
        st.markdown(f"**Expected Question:** {question}")
        st.markdown(f"**Explanation:** {explanation}")