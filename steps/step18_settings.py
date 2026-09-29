"""
steps/step18_settings.py

Step 18: Settings page for the Study Notes & Flashcard Generator.

All settings live in st.session_state only - nothing here touches the
SQLite database, study material, flashcards, quiz results, or learning
memory from earlier steps.

Other steps READ the settings with the helpers below:

    from steps.step18_settings import get_setting, language_instruction

    get_setting("questions_per_quiz")   # -> 5
    language_instruction()              # -> "" for English, or a sentence
                                        #    to add to an Gemini prompt
"""

import streamlit as st

from steps.step19_ui import show_page_header


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "language": "English",
    "questions_per_quiz": 5,
    "show_explanations": True,
    "auto_generate": False,
}

LANGUAGE_OPTIONS = [
    "English",
    "Tamil",
    "Hindi",
    "Spanish",
]


# ---------------------------------------------------------------------------
# Session state + helpers other steps can import
# ---------------------------------------------------------------------------

def initialize_settings():
    """
    Make sure st.session_state.settings exists.

    If it already exists we leave it alone - we never overwrite settings the
    student already changed. The first time, the language starts from the
    language chosen on the Student Profile page (if there is one).
    """
    if "settings" not in st.session_state:
        settings = dict(DEFAULT_SETTINGS)
        profile_language = st.session_state.get("preferred_language")
        if profile_language in LANGUAGE_OPTIONS:
            settings["language"] = profile_language
        st.session_state.settings = settings


def get_setting(key):
    """Safely read one setting (falls back to the default value)."""
    initialize_settings()
    return st.session_state.settings.get(key, DEFAULT_SETTINGS.get(key))


def language_instruction():
    """
    A sentence to add to an Gemini prompt so the answer follows the chosen
    language. Returns "" for English.

    Headings and format keywords stay in English on purpose: the parsers in
    the other steps look for words like "## Summary", "Q1:" or "ANSWER:".
    """
    language = get_setting("language")
    if language == "English":
        return ""
    return (
        f"Write all explanations and content in {language}. Keep every heading, "
        "label and format keyword (for example '## Summary', 'Q1:', 'A:', "
        "'ANSWER:', 'EXPLANATION:') in English exactly as shown, and keep "
        "technical terms in English where that is clearer."
    )


def _reset_settings():
    """Button callback: runs BEFORE the page redraws, so widgets show the defaults."""
    st.session_state.settings = dict(DEFAULT_SETTINGS)
    st.session_state.settings_reset_notice = True


# ---------------------------------------------------------------------------
# Main Streamlit page
# ---------------------------------------------------------------------------

def show_settings():
    """Render the Settings page."""
    initialize_settings()

    show_page_header("⚙️", "Settings", "Customize your study experience. Changes apply instantly.")

    if st.session_state.pop("settings_reset_notice", False):
        st.success("✅ Settings reset to default.")

    settings = st.session_state.settings
    left, right = st.columns(2)

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    with left:
        with st.container(border=True):
            st.markdown("#### 🌐 Language")

            current_language = settings.get("language", DEFAULT_SETTINGS["language"])
            language_index = (
                LANGUAGE_OPTIONS.index(current_language)
                if current_language in LANGUAGE_OPTIONS
                else 0
            )
            selected_language = st.selectbox(
                "Preferred Language",
                options=LANGUAGE_OPTIONS,
                index=language_index,
            )
            st.session_state.settings["language"] = selected_language
            # Keep the Student Profile page in step with this choice.
            if "preferred_language" in st.session_state:
                st.session_state.preferred_language = selected_language

            st.caption(
                "New study material and quizzes are written in this language. "
                "Quality can vary between languages, so check the output."
            )

    # ------------------------------------------------------------------
    # Quiz settings
    # ------------------------------------------------------------------
    with right:
        with st.container(border=True):
            st.markdown("#### ❓ Quiz Settings")

            questions_per_quiz = st.number_input(
                "Questions per Quiz",
                min_value=3,
                max_value=20,
                value=int(settings.get("questions_per_quiz", DEFAULT_SETTINGS["questions_per_quiz"])),
                step=1,
            )
            st.session_state.settings["questions_per_quiz"] = int(questions_per_quiz)

            show_explanations = st.toggle(
                "Show Explanations",
                value=bool(settings.get("show_explanations", DEFAULT_SETTINGS["show_explanations"])),
            )
            st.session_state.settings["show_explanations"] = show_explanations

    # ------------------------------------------------------------------
    # Learning preferences
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 💡 Learning Preferences")
        auto_generate = st.toggle(
            "Auto Generate Study Material",
            value=bool(settings.get("auto_generate", DEFAULT_SETTINGS["auto_generate"])),
        )
        st.session_state.settings["auto_generate"] = auto_generate

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    st.caption("✅ Every change is saved automatically for this session - there is no Save button to press.")
    st.button("🔄 Reset to Default", on_click=_reset_settings)


# ---------------------------------------------------------------------------
# Allow running this page standalone for quick testing:
#   streamlit run steps/step18_settings.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show_settings()