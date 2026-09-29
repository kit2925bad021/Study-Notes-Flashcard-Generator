"""
Step 14 — Smart Revision
-------------------------------------
Renders the "📚 Smart Revision" page inside the Streamlit app.

This page helps the student decide what to revise next, based ONLY on
the learning performance data already collected in Learning Memory
(Step 13) — weak topics, accuracy, and recently studied topics. It does
NOT claim to know the student's long-term memory or understanding; it
only reflects what has happened in the current application session.

Wire this up in app.py the same way the other steps are wired up:

    from steps.step14_smart_revision import show_smart_revision
    ...
    elif st.session_state.current_page == "📚 Smart Revision":
        show_smart_revision()

This file does NOT modify study_material, flashcards, quiz questions,
or Step 13's own logic. It only READS from Learning Memory (via
step13_memory's functions) and writes to its own session_state key,
st.session_state.smart_revision_plan.
"""

import streamlit as st

from services.gemini_service import generate_content

# Reuse the Learning Memory helpers from Step 13 instead of duplicating
# any tracking logic here.
from steps.step13_memory import (
    initialize_learning_memory,
    get_memory_summary,
    get_weak_topics,
)
from steps.step19_ui import show_page_header


# ----------------------------------------------------------------------
# 1. PROMPT BUILDER
# ----------------------------------------------------------------------

def _build_revision_prompt(memory_summary):
    """
    Build a prompt asking Gemini for a short, personalized revision plan
    based ONLY on the given memory_summary (from Step 13). No topics or
    facts outside of memory_summary should be introduced.
    """

    weak_topics = memory_summary.get("weak_topics", [])
    recent_topics = memory_summary.get("recent_topics", [])
    total_questions = memory_summary.get("total_questions", 0)
    correct_answers = memory_summary.get("correct_answers", 0)
    incorrect_answers = memory_summary.get("incorrect_answers", 0)
    accuracy = memory_summary.get("accuracy", 0)

    # Format the lists as simple, readable text for the prompt.
    weak_topics_text = ", ".join(weak_topics) if weak_topics else "None recorded yet"
    recent_topics_text = ", ".join(recent_topics) if recent_topics else "None recorded yet"

    prompt = f"""
You are a study coach helping a college student decide what to revise
next. Base your plan ONLY on the learning data below, which comes from
the student's current study session. This data is NOT a measure of
their long-term memory or overall understanding — it only reflects
what has happened in this session so far. Do not invent topics, facts,
or statistics that are not included in this data.

LEARNING DATA:
- Total questions attempted: {total_questions}
- Correct answers: {correct_answers}
- Incorrect answers: {incorrect_answers}
- Overall accuracy: {accuracy:.0f}%
- Weak topics (below 60% accuracy): {weak_topics_text}
- Recently studied topics: {recent_topics_text}

TASK:
Create a short, practical revision plan for this student. Prioritize in
this order:
1. Weak topics
2. Topics with low accuracy
3. Recently studied topics that may need reinforcement
4. Topics with more incorrect/unanswered questions

Keep the language simple and practical for a college student. Use ONLY
the topics named in the learning data above — do not introduce new
topics.

Respond using EXACTLY this format, with no extra commentary before or
after it:

## 🎯 Today's Revision Focus
(1-2 sentences summarizing what to focus on today)

## 🔴 High Priority
(the weakest topics, with a short reason for each)

## 🟡 Practice More
(topics that could use extra practice, but aren't critical)

## 🟢 Quick Review
(topics that are in good shape — just a quick refresher)

## 📝 Revision Tips
(2-3 short, practical study tips based on this data)
"""
    return prompt


# ----------------------------------------------------------------------
# 2. GENERATION
# ----------------------------------------------------------------------

def _generate_revision_plan(memory_summary):
    """
    Build the prompt from the memory summary, call Gemini, and return
    the raw response text. Any exception from generate_content() is
    left to the caller (show_smart_revision) to catch and display
    gracefully.
    """
    prompt = _build_revision_prompt(memory_summary)
    result_text = generate_content(prompt)
    return result_text


# ----------------------------------------------------------------------
# 3. SHOWING THE PLAN
# ----------------------------------------------------------------------

def _split_plan(plan_text):
    """Split the plan into [(heading, body), ...] using its '## ' headings."""
    sections = []
    heading = None
    lines = []
    for line in plan_text.splitlines():
        if line.strip().startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(lines).strip()))
            heading = line.strip().lstrip("#").strip()
            lines = []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def _render_plan(plan_text):
    """
    Show each part of the plan in a box whose colour matches its meaning
    (red = urgent, yellow = practice, green = fine). The traffic-light
    colours let the student see the priorities at a glance.
    """
    sections = _split_plan(plan_text)

    # The model did not follow the format: show everything it wrote.
    if len(sections) < 2:
        with st.container(border=True):
            st.markdown(plan_text)
        return

    for heading, body in sections:
        text = f"**{heading}**\n\n{body}" if body else f"**{heading}**"
        if "🔴" in heading:
            st.error(text)
        elif "🟡" in heading:
            st.warning(text)
        elif "🟢" in heading:
            st.success(text)
        elif "🎯" in heading:
            st.info(text)
        else:
            with st.container(border=True):
                st.markdown(text)


# ----------------------------------------------------------------------
# 4. PAGE UI
# ----------------------------------------------------------------------

def show_smart_revision():
    """Render the full Smart Revision page (Step 14)."""

    show_page_header(
        "📚", "Smart Revision",
        "See which topics need the most attention, based on your quiz and question results.",
    )

    # ------------------------------------------------------------
    # Make sure Learning Memory exists, then pull the summary.
    # (This never overwrites existing memory — see Step 13.)
    # ------------------------------------------------------------
    initialize_learning_memory()
    memory_summary = get_memory_summary()

    # ------------------------------------------------------------
    # Top metrics, straight from Learning Memory — nothing invented.
    # ------------------------------------------------------------
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("📊 Questions Attempted", memory_summary["total_questions"])
    with metric_col2:
        st.metric("✅ Correct", memory_summary["correct_answers"])
    with metric_col3:
        st.metric("❌ Incorrect", memory_summary["incorrect_answers"])
    with metric_col4:
        st.metric("🎯 Accuracy", f"{memory_summary['accuracy']:.0f}%")

    if memory_summary["total_questions"] > 0:
        st.progress(min(memory_summary["accuracy"], 100) / 100)

    st.markdown("---")

    # ------------------------------------------------------------
    # Weak topics, straight from Step 13's get_weak_topics().
    # ------------------------------------------------------------
    st.markdown("### ⚠️ Topics Needing More Practice")

    memory = st.session_state.learning_memory
    weak_topics = get_weak_topics()
    if weak_topics:
        for topic in weak_topics:
            stats = memory["topics"].get(topic, {})
            attempts = stats.get("attempts", 0)
            accuracy = (stats.get("correct", 0) / attempts) * 100 if attempts else 0
            st.warning(f"**{topic}** — {accuracy:.0f}% accuracy ({attempts} attempts)")
    else:
        st.success("🎉 No weak topics detected yet. Keep practicing!")

    st.markdown("---")

    # ------------------------------------------------------------
    # No learning data yet -> don't call Gemini, just explain why.
    # ------------------------------------------------------------
    has_learning_data = memory_summary["total_questions"] > 0

    if not has_learning_data:
        st.info(
            "Complete some quizzes or dynamic questions first so Smart "
            "Revision has learning data to work with."
        )
        return

    # ------------------------------------------------------------
    # One button: "Create" first, "Regenerate" once a plan exists.
    # ------------------------------------------------------------
    has_plan = bool(st.session_state.get("smart_revision_plan"))
    generate_clicked = st.button(
        "🔄 Regenerate Revision Plan" if has_plan else "✨ Create My Revision Plan",
        type="secondary" if has_plan else "primary",
    )

    if generate_clicked:
        with st.spinner("Analyzing your learning progress..."):
            try:
                # Re-fetch the freshest summary right before generating,
                # in case more questions were answered since page load.
                latest_summary = get_memory_summary()
                plan = _generate_revision_plan(latest_summary)
                st.session_state.smart_revision_plan = plan
                st.success("✅ Revision plan created!")
            except Exception as e:
                # Handle Gemini / API errors gracefully — never crash the page.
                st.error(f"Sorry, we couldn't create a revision plan: {e}")
                return

    # ------------------------------------------------------------
    # Display the stored plan, if any.
    # ------------------------------------------------------------
    plan = st.session_state.get("smart_revision_plan")
    if plan:
        st.markdown("---")
        _render_plan(plan)

    st.markdown("---")

    # ------------------------------------------------------------
    # Recently studied topics.
    # ------------------------------------------------------------
    st.markdown("### 🕒 Recently Studied Topics")

    recent_topics = memory_summary.get("recent_topics", [])
    if recent_topics:
        for topic in recent_topics:
            st.write(f"- {topic}")
    else:
        st.info("No topics studied yet this session.")