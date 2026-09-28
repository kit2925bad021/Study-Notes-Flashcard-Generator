"""
steps/step17_profile.py

Step 17: Student Profile page for the Study Notes & Flashcard Generator.

This page lets a student enter basic profile info (name, college, course,
year of study, preferred language) and saves it so other steps (like the
Step 16 Progress Dashboard) can use st.session_state.student_id.

IMPORTANT ABOUT THE DATABASE:
The existing `students` table only stores `name` and `created_at` — we do
NOT modify the schema. Only the student's name is saved to SQLite via
create_student(name). The rest of the profile fields (college, course,
year of study, preferred language) are kept in Streamlit session state
only, for now.

This file does not create logins, passwords, or any authentication.
"""

from html import escape

import streamlit as st

# Reuse the existing database layer only — we do not duplicate or modify
# any database logic here.
from database.database import create_student, update_student_name
from steps.step19_ui import show_empty_state, show_page_header


# ---------------------------------------------------------------------------
# Options used by the form
# ---------------------------------------------------------------------------

YEAR_OPTIONS = [
    "1st Year",
    "2nd Year",
    "3rd Year",
    "4th Year",
    "Other",
]

LANGUAGE_OPTIONS = [
    "English",
    "Tamil",
    "Hindi",
    "Spanish",
]


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

def initialize_profile():
    """
    Make sure all the profile-related session-state keys exist.

    This never overwrites existing values — it only fills in defaults
    the first time the page is loaded, so refreshing the page (or moving
    between steps) doesn't wipe out what's already there.
    """
    defaults = {
        "student_id": None,
        "student_name": "",
        "college": "",
        "course": "",
        "year_of_study": YEAR_OPTIONS[0],
        "preferred_language": LANGUAGE_OPTIONS[0],
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ---------------------------------------------------------------------------
# Saving the profile
# ---------------------------------------------------------------------------

def save_profile(name, college, course, year_of_study, language):
    """
    Save the student's profile.

    - If no student_id exists yet, create a new student row in SQLite
      (this is the only field the database actually stores) and remember
      the new ID in session state.
    - If a student_id already exists, we just update the session-state
      values — we do NOT create a second student record.

    The other profile fields (college, course, year, language) always
    live in session state only, since the database schema doesn't have
    columns for them yet.
    """
    try:
        if st.session_state.get("student_id") is None:
            # No student saved yet — create one in the database.
            new_id = create_student(name)
            if new_id is None:
                # create_student() failed internally; don't crash the app.
                st.error(
                    "Could not save your profile: the database did not create the "
                    "student record. The exact database error is printed in the "
                    "terminal where Streamlit is running."
                )
                return False
            st.session_state.student_id = new_id
        else:
            # Student already exists (for example linked at login): keep the
            # name stored in the database in step with the profile.
            update_student_name(st.session_state.student_id, name)

        st.session_state.student_name = name
        st.session_state.college = college
        st.session_state.course = course
        st.session_state.year_of_study = year_of_study
        st.session_state.preferred_language = language

        # Keep the Settings page in step: the language picked here becomes the
        # language used for generated study material and quizzes.
        if "settings" in st.session_state:
            st.session_state.settings["language"] = language
        return True

    except Exception as e:
        # Show the real error so it is easy to see what failed
        # (no secrets are involved here - only a database/Python error).
        st.error(f"Saving the profile failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Profile summary card
# ---------------------------------------------------------------------------

def _render_summary():
    """Show the saved profile as a card (or a friendly hint if nothing is saved yet)."""
    student_id = st.session_state.get("student_id")
    if student_id is None:
        show_empty_state(
            "No profile saved yet",
            "Fill in the form and press Save Profile to create yours.",
            "👤",
        )
        return

    name = st.session_state.student_name or "Student"
    initial = escape(name.strip()[:1].upper() or "S")

    chips = []
    for icon, value in (
        ("🏫", st.session_state.college),
        ("🎓", st.session_state.course),
        ("📅", st.session_state.year_of_study),
        ("🌐", st.session_state.preferred_language),
    ):
        if value:
            chips.append(f'<span class="sf-chip">{icon} {escape(str(value))}</span>')

    st.markdown(
        '<div class="app-card">'
        '<div class="sf-profile-top">'
        f'<div class="sf-avatar">{initial}</div>'
        f'<div><div class="sf-profile-name">{escape(name)}</div>'
        f'<div class="sf-profile-sub">Student ID: {escape(str(student_id))}</div></div>'
        "</div>"
        f'<div class="sf-chip-row">{"".join(chips)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Streamlit page
# ---------------------------------------------------------------------------

def show_profile():
    """Render the Student Profile page."""
    # Make sure session state has all the keys we need before we read them.
    initialize_profile()

    show_page_header(
        "👤", "Student Profile",
        "Set up your study profile to personalize your learning experience.",
    )

    form_col, summary_col = st.columns([3, 2])

    # ------------------------------------------------------------------
    # Profile form (left)
    # ------------------------------------------------------------------
    with form_col:
        with st.form("student_profile_form"):
            name = st.text_input(
                "👤 Name",
                value=st.session_state.student_name,
            )

            college = st.text_input(
                "🏫 College / Institution",
                value=st.session_state.college,
            )

            course = st.text_input(
                "🎓 Course",
                value=st.session_state.course,
            )

            year_of_study = st.selectbox(
                "📅 Year of Study",
                options=YEAR_OPTIONS,
                index=YEAR_OPTIONS.index(st.session_state.year_of_study)
                if st.session_state.year_of_study in YEAR_OPTIONS
                else 0,
            )

            preferred_language = st.selectbox(
                "🌐 Preferred Language",
                options=LANGUAGE_OPTIONS,
                index=LANGUAGE_OPTIONS.index(st.session_state.preferred_language)
                if st.session_state.preferred_language in LANGUAGE_OPTIONS
                else 0,
            )

            submitted = st.form_submit_button("💾 Save Profile", type="primary")

            if submitted:
                # Basic validation — name is required.
                if not name.strip():
                    st.error("Name cannot be empty. Please enter your name.")
                else:
                    success = save_profile(
                        name.strip(),
                        college.strip(),
                        course.strip(),
                        year_of_study,
                        preferred_language,
                    )
                    if success:
                        st.success("✅ Profile saved successfully!")

        st.caption(
            "Your name is stored in the local database. College, course, year and "
            "language stay in this session only."
        )

    # ------------------------------------------------------------------
    # Profile summary (right) - drawn after the form, so it is always up to date
    # ------------------------------------------------------------------
    with summary_col:
        _render_summary()


# ---------------------------------------------------------------------------
# Allow running this page standalone for quick testing:
#   streamlit run steps/step17_profile.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show_profile()