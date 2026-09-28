"""
steps/step16_progress.py

Step 16: Real Progress Dashboard for the Study Notes & Flashcard Generator.

This dashboard reads REAL data from the existing SQLite database
(database/database.py). It does not duplicate any database logic and it
does not create or connect a student profile system — it simply displays
whatever is already stored for the student currently in
st.session_state.student_id (if any).

Built with Streamlit + Plotly only.
"""

import streamlit as st
import plotly.graph_objects as go

# Reuse the existing database layer. We do NOT redefine or duplicate any
# database logic here — everything comes from database/database.py.
from database.database import (
    initialize_database,
    get_question_results,
    get_study_sessions,
    get_topic_statistics,
)
from steps.step19_ui import show_empty_state, show_page_header

# Topics below this accuracy (%) are flagged as "Areas to Improve".
WEAK_ACCURACY = 60

# Colours: one meaning everywhere (green = good, amber = okay, red = weak).
COLOR_GOOD = "#10b981"
COLOR_OK = "#f59e0b"
COLOR_WEAK = "#ef4444"
COLOR_NEUTRAL = "#4f46e5"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_progress_data(student_id):
    """
    Pull together everything we need for the dashboard in one place.

    Returns a dictionary with:
        - "results": list of question result rows
        - "sessions": list of study session rows
        - "topic_stats": list of per-topic statistics rows

    If the database can't be reached for some reason, we fail safely and
    return empty lists instead of crashing the app.
    """
    try:
        results = get_question_results(student_id)
    except Exception:
        results = []

    try:
        sessions = get_study_sessions(student_id)
    except Exception:
        sessions = []

    try:
        topic_stats = get_topic_statistics(student_id)
    except Exception:
        topic_stats = []

    return {
        "results": results,
        "sessions": sessions,
        "topic_stats": topic_stats,
    }


def calculate_accuracy(results):
    """
    Calculate overall accuracy as a percentage:

        accuracy = correct answers / total answers * 100

    `results` is expected to be a list of dict-like rows, each with an
    "is_correct" field (1 or 0). If there are no results, returns 0.
    """
    if not results:
        return 0

    total = len(results)
    correct = sum(1 for row in results if row["is_correct"] == 1)

    accuracy = (correct / total) * 100
    return round(accuracy, 2)


# ---------------------------------------------------------------------------
# Small chart-building helpers (keeps show_progress() easier to read)
# ---------------------------------------------------------------------------

def _accuracy_color(accuracy):
    """Traffic-light colour for an accuracy percentage."""
    if accuracy < WEAK_ACCURACY:
        return COLOR_WEAK
    if accuracy < 80:
        return COLOR_OK
    return COLOR_GOOD


def _build_correct_vs_incorrect_chart(results):
    """Build a donut chart of correct vs incorrect answers."""
    correct_count = sum(1 for row in results if row["is_correct"] == 1)
    incorrect_count = sum(1 for row in results if row["is_correct"] == 0)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Correct", "Incorrect"],
                values=[correct_count, incorrect_count],
                hole=0.6,
                marker=dict(colors=[COLOR_GOOD, COLOR_WEAK]),
                textinfo="value+percent",
            )
        ]
    )
    fig.update_layout(
        title="Question Performance",
        showlegend=True,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


def _build_topic_accuracy_chart(topic_stats):
    """Build a bar chart of accuracy (%) per topic, coloured by how well it is going."""
    topics = [row["topic"] for row in topic_stats]
    accuracies = [row["accuracy"] for row in topic_stats]

    fig = go.Figure(
        data=[
            go.Bar(
                x=topics,
                y=accuracies,
                marker_color=[_accuracy_color(value) for value in accuracies],
            )
        ]
    )
    fig.add_hline(
        y=WEAK_ACCURACY,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text=f"{WEAK_ACCURACY}% target",
    )
    fig.update_layout(
        title="Accuracy by Topic",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Main Streamlit page
# ---------------------------------------------------------------------------

def show_progress():
    """
    Render the Progress Dashboard page.

    This function is safe to call even if:
    - no student is set in session state
    - the database has no data yet for the student
    - the database is empty or unreachable
    """
    # Make sure the database and its tables exist. This is safe to call
    # multiple times (it only creates tables if they're missing).
    initialize_database()

    student_id = st.session_state.get("student_id", None)
    student_name = st.session_state.get("student_name", "")

    show_page_header(
        "📊", "Your Learning Progress",
        f"Hi {student_name}! Here is how your studying is going." if student_name
        else "See how your studying is going.",
    )

    # ------------------------------------------------------------------
    # Guard: no student profile connected yet
    # ------------------------------------------------------------------
    if not student_id:
        show_empty_state(
            "No student profile yet",
            "Save your Student Profile first, then your progress will appear here.",
            "👤",
        )
        return  # Stop here — nothing else to show without a student.

    # ------------------------------------------------------------------
    # Load real data from the database
    # ------------------------------------------------------------------
    data = get_progress_data(student_id)
    results = data["results"]
    sessions = data["sessions"]
    topic_stats = data["topic_stats"]

    # ------------------------------------------------------------------
    # Top-level metrics
    # ------------------------------------------------------------------
    topics_studied = len(topic_stats)
    questions_answered = len(results)
    correct_answers = sum(1 for row in results if row["is_correct"] == 1)
    accuracy = calculate_accuracy(results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Topics Studied", topics_studied)
    col2.metric("❓ Questions Answered", questions_answered)
    col3.metric("✅ Correct Answers", correct_answers)
    col4.metric("🎯 Accuracy", f"{accuracy}%")

    # ------------------------------------------------------------------
    # Charts side by side: overall performance + accuracy per topic
    # ------------------------------------------------------------------
    st.subheader("📈 Performance Overview")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        if results:
            st.plotly_chart(_build_correct_vs_incorrect_chart(results), use_container_width=True)
        else:
            st.info("No question activity recorded yet.")
    with chart_right:
        if topic_stats:
            st.plotly_chart(_build_topic_accuracy_chart(topic_stats), use_container_width=True)
        else:
            st.info("No topic statistics available yet.")

    # ------------------------------------------------------------------
    # Topic performance breakdown (a real table instead of loose text lines)
    # ------------------------------------------------------------------
    st.subheader("📚 Topic Performance")

    if topic_stats:
        st.dataframe(
            [
                {
                    "Topic": row["topic"],
                    "Attempted": row["total_questions"],
                    "Correct": row["correct_answers"],
                    "Incorrect": row["incorrect_answers"],
                    "Accuracy (%)": row["accuracy"],
                }
                for row in topic_stats
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No topic statistics available yet.")

    # ------------------------------------------------------------------
    # Recent study sessions
    # ------------------------------------------------------------------
    st.subheader("🕒 Recent Study Sessions")

    if sessions:
        # Show the most recent sessions first (already ordered by the DB layer)
        st.dataframe(
            [
                {
                    "Subject": session["subject"] or "Unknown subject",
                    "Topic": session["topic"] or "Unknown topic",
                    "Started": session["started_at"] or "Unknown time",
                }
                for session in sessions[:10]  # limit to 10 most recent for readability
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No study activity recorded yet.")

    # ------------------------------------------------------------------
    # Weak topics (areas to improve)
    # ------------------------------------------------------------------
    st.subheader("🎯 Areas to Improve")

    weak_topics = [row for row in topic_stats if row["accuracy"] < WEAK_ACCURACY]

    if weak_topics:
        for row in weak_topics:
            st.warning(f"**{row['topic']}** — Accuracy: {row['accuracy']}%")
    else:
        st.success("🎉 No weak topics detected yet.")


# ---------------------------------------------------------------------------
# Allow running this page standalone for quick testing:
#   streamlit run steps/step16_progress.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show_progress()