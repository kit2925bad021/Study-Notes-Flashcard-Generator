"""
Step 13 — Learning Memory
-------------------------------------
A simple, session-only "Learning Memory" module that tracks how the
student is doing across topics during the current app session:
topics studied, questions attempted, correct/incorrect answers, which
topics need more practice, and which topics were studied most recently.

This is intentionally SQLite-free for now — everything lives in
st.session_state.learning_memory and disappears when the session ends.
Database persistence will be added in a later step.

Wire the UI page up in app.py the same way the other steps are:

    from steps.step13_memory import show_memory
    ...
    elif st.session_state.current_page == "🧠 Learning Memory":
        show_memory()

The tracking functions below (record_topic, record_answer,
get_weak_topics, get_memory_summary) are also meant to be imported and
called from OTHER step files (quizzes, dynamic questions, etc.) so they
can log results into this shared memory. For example, another step
could do:

    from steps.step13_memory import initialize_learning_memory, record_topic, record_answer

    initialize_learning_memory()
    record_topic(topic)
    record_answer(topic, is_correct)

Pages that ask questions should call log_answer(topic, is_correct, question_type)
instead: it does record_answer() AND saves the result in SQLite, so the
Progress page (step 16) sees it too.

This file does not touch st.session_state.study_material, flashcards,
or any other step's session state — it only reads/writes its own key,
st.session_state.learning_memory.
"""

import streamlit as st


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

MAX_RECENT_TOPICS = 10
WEAK_TOPIC_ACCURACY_THRESHOLD = 60  # percent


# ----------------------------------------------------------------------
# 1. INITIALIZATION
# ----------------------------------------------------------------------

def initialize_learning_memory():
    """
    Make sure st.session_state.learning_memory exists, with the correct
    starting shape. Safe to call as many times as you like — it will
    NEVER overwrite memory that already exists.
    """
    if "learning_memory" not in st.session_state:
        st.session_state.learning_memory = {
            "topics": {},          # {topic_name: {"attempts", "correct", "incorrect"}}
            "recent_topics": [],   # list of topic names, most recent first
            "total_questions": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
        }


# ----------------------------------------------------------------------
# 2. RECORD A STUDIED TOPIC
# ----------------------------------------------------------------------

def record_topic(topic):
    """
    Record that the student studied/started a topic. Creates a fresh
    tracking entry for the topic if it's new, and moves it to the front
    of "recent_topics" (without duplicating it), keeping only the most
    recent MAX_RECENT_TOPICS topics.
    """
    # Make sure memory exists before we try to use it.
    initialize_learning_memory()

    memory = st.session_state.learning_memory

    # Ignore empty/blank topic names — nothing useful to record.
    if not topic or not str(topic).strip():
        return

    topic = str(topic).strip()

    # Create a fresh tracking entry for this topic if we haven't seen it yet.
    if topic not in memory["topics"]:
        memory["topics"][topic] = {
            "attempts": 0,
            "correct": 0,
            "incorrect": 0,
        }

    # Update recent_topics: remove any existing occurrence of this topic
    # first, then add it to the front, so it doesn't appear twice and
    # always reflects the most recent study session.
    recent_topics = memory["recent_topics"]
    if topic in recent_topics:
        recent_topics.remove(topic)
    recent_topics.insert(0, topic)

    # Keep only the most recent MAX_RECENT_TOPICS topics.
    memory["recent_topics"] = recent_topics[:MAX_RECENT_TOPICS]


# ----------------------------------------------------------------------
# 3. RECORD AN ANSWER
# ----------------------------------------------------------------------

def record_answer(topic, is_correct):
    """
    Record the result of one answered question for the given topic.
    Updates both that topic's stats and the global totals.

    If the topic hasn't been recorded yet (record_topic wasn't called
    first), it's created automatically so this function never crashes.
    """
    initialize_learning_memory()

    memory = st.session_state.learning_memory

    if not topic or not str(topic).strip():
        return

    topic = str(topic).strip()

    # Make sure this topic has a tracking entry, in case record_answer()
    # is called without record_topic() being called first.
    if topic not in memory["topics"]:
        memory["topics"][topic] = {
            "attempts": 0,
            "correct": 0,
            "incorrect": 0,
        }

    topic_stats = memory["topics"][topic]

    # ---- Update this topic's stats ----
    topic_stats["attempts"] += 1
    if is_correct:
        topic_stats["correct"] += 1
    else:
        topic_stats["incorrect"] += 1

    # ---- Update global stats ----
    memory["total_questions"] += 1
    if is_correct:
        memory["correct_answers"] += 1
    else:
        memory["incorrect_answers"] += 1


def log_answer(topic, is_correct, question_type="Quiz", difficulty="Mixed"):
    """
    Record one answered question in BOTH places:
      * Learning Memory (this session)  -> Smart Revision / Learning Memory pages
      * SQLite question_results table   -> Progress page (kept between sessions)

    The database part is skipped silently when no student is signed in, and a
    database problem never stops the student from carrying on.
    """
    record_answer(topic, is_correct)

    student_id = st.session_state.get("student_id")
    if not student_id:
        return
    try:
        from database.database import save_question_result

        save_question_result(student_id, str(topic).strip(), question_type, is_correct, difficulty)
    except Exception as e:
        print(f"[memory] Could not save the question result: {e}")


# ----------------------------------------------------------------------
# 4. WEAK TOPICS
# ----------------------------------------------------------------------

def get_weak_topics():
    """
    Return a list of topic names where the student's accuracy is below
    WEAK_TOPIC_ACCURACY_THRESHOLD percent. Topics with zero attempts are
    not considered weak (there's no evidence yet either way).

    Returns an empty list if there are no weak topics.
    """
    initialize_learning_memory()

    memory = st.session_state.learning_memory
    weak_topics = []

    for topic_name, stats in memory["topics"].items():
        attempts = stats["attempts"]
        if attempts == 0:
            continue  # No data yet — don't flag it as weak.

        accuracy = (stats["correct"] / attempts) * 100
        if accuracy < WEAK_TOPIC_ACCURACY_THRESHOLD:
            weak_topics.append(topic_name)

    return weak_topics


# ----------------------------------------------------------------------
# 5. MEMORY SUMMARY
# ----------------------------------------------------------------------

def get_memory_summary():
    """
    Return a dictionary summarizing the student's overall learning
    memory:

        {
            "total_questions": int,
            "correct_answers": int,
            "incorrect_answers": int,
            "accuracy": float,       # percentage, 0-100, 0 if no questions yet
            "weak_topics": [...],
            "recent_topics": [...],
        }
    """
    initialize_learning_memory()

    memory = st.session_state.learning_memory
    total_questions = memory["total_questions"]
    correct_answers = memory["correct_answers"]
    incorrect_answers = memory["incorrect_answers"]

    # Calculate accuracy safely — avoid dividing by zero.
    if total_questions > 0:
        accuracy = (correct_answers / total_questions) * 100
    else:
        accuracy = 0

    return {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "accuracy": accuracy,
        "weak_topics": get_weak_topics(),
        "recent_topics": memory["recent_topics"],
    }


# ----------------------------------------------------------------------
# HELPER: CLEAR MEMORY
# ----------------------------------------------------------------------

def _clear_learning_memory():
    """
    Reset ONLY the learning memory back to its starting shape. Does not
    touch study_material, flashcards, or any other session-state key.
    """
    st.session_state.learning_memory = {
        "topics": {},
        "recent_topics": [],
        "total_questions": 0,
        "correct_answers": 0,
        "incorrect_answers": 0,
    }


# ----------------------------------------------------------------------
# 6. PAGE UI
# ----------------------------------------------------------------------

def show_memory():
    """Render the full Learning Memory page (Step 13)."""

    st.markdown("## 🧠 Learning Memory")
    st.caption(
        "A running record of how you're doing this session — topics "
        "studied, questions attempted, and where you might need more practice."
    )

    # Make sure memory exists before we read from it.
    initialize_learning_memory()

    summary = get_memory_summary()

    st.markdown("---")

    # ------------------------------------------------------------
    # Top metrics.
    # ------------------------------------------------------------
    memory = st.session_state.learning_memory
    topics_studied = len(memory["topics"])

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("📚 Topics Studied", topics_studied)
    with metric_col2:
        st.metric("❓ Questions Attempted", summary["total_questions"])
    with metric_col3:
        st.metric("✅ Correct Answers", summary["correct_answers"])
    with metric_col4:
        st.metric("🎯 Accuracy", f"{summary['accuracy']:.0f}%")

    st.markdown("---")

    # ------------------------------------------------------------
    # Weak topics.
    # ------------------------------------------------------------
    st.markdown("### ⚠️ Topics That Need More Practice")

    weak_topics = summary["weak_topics"]
    if weak_topics:
        for topic in weak_topics:
            stats = memory["topics"][topic]
            attempts = stats["attempts"]
            accuracy = (stats["correct"] / attempts) * 100 if attempts > 0 else 0
            st.warning(f"**{topic}** — {accuracy:.0f}% accuracy ({attempts} attempts)")
    else:
        st.success("🎉 No weak topics yet!")

    st.markdown("---")

    # ------------------------------------------------------------
    # Recently studied topics.
    # ------------------------------------------------------------
    st.markdown("### 🕒 Recently Studied")

    recent_topics = summary["recent_topics"]
    if recent_topics:
        for topic in recent_topics:
            st.write(f"- {topic}")
    else:
        st.info("No topics studied yet this session.")

    st.markdown("---")

    # ------------------------------------------------------------
    # Full per-topic performance table.
    # ------------------------------------------------------------
    st.markdown("### 📊 Topic Performance")

    topics = memory["topics"]
    if not topics:
        st.info("No topic performance data yet. Answer some questions to see stats here.")
    else:
        # Simple table-like layout using Streamlit columns, with a
        # header row followed by one row per topic.
        header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns(
            [3, 1, 1, 1, 1]
        )
        header_col1.markdown("**Topic**")
        header_col2.markdown("**Attempts**")
        header_col3.markdown("**Correct**")
        header_col4.markdown("**Incorrect**")
        header_col5.markdown("**Accuracy**")

        for topic_name, stats in topics.items():
            attempts = stats["attempts"]
            correct = stats["correct"]
            incorrect = stats["incorrect"]
            accuracy = (correct / attempts) * 100 if attempts > 0 else 0

            row_col1, row_col2, row_col3, row_col4, row_col5 = st.columns(
                [3, 1, 1, 1, 1]
            )
            row_col1.write(topic_name)
            row_col2.write(attempts)
            row_col3.write(correct)
            row_col4.write(incorrect)
            row_col5.write(f"{accuracy:.0f}%")

    st.markdown("---")

    # ------------------------------------------------------------
    # Clear memory button.
    # ------------------------------------------------------------
    if st.button("🗑️ Clear Learning Memory"):
        _clear_learning_memory()
        st.success("Learning memory cleared.")
        st.rerun()