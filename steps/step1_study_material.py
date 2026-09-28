"""
Step 1 — Study Material Page
------------------------------
Contains the helper functions that parse Gemini's markdown response
into sections, plus the show_study_material() function that renders
the "📖 Study Material" page.

Imported in app.py with:
    from steps.step1_study_material import show_study_material
"""

import streamlit as st


def split_study_material_sections(content):
    """
    Split Gemini's markdown output (which uses "## " headings) into a
    dictionary of {heading_text: section_body}. This lets us show each
    part (explanation, flashcards, quiz, etc.) in its own section below.
    """
    sections = {}
    current_heading = None
    current_lines = []

    for line in content.splitlines():
        if line.strip().startswith("## "):
            # Save the previous section before starting a new one
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line.strip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last section
    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def find_section(sections, keyword):
    """Find a section whose heading contains the given keyword (case-insensitive)."""
    keyword = keyword.lower()
    for heading, body in sections.items():
        if keyword in heading.lower():
            return body
    return None


def show_study_material():
    """Display the full Study Material page using data from session state."""
    st.subheader("📖 Study Material")

    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("No study material generated yet. Go to Dashboard and generate study material first.")
        return

    subject = study_material["subject"]
    topic = study_material["topic"]
    content = study_material["content"]

    # ---- Subject and topic display ----
    info_col1, info_col2 = st.columns(2)
    info_col1.markdown(f"**📘 Subject:** {subject}")
    info_col2.markdown(f"**📌 Topic/Chapter:** {topic}")

    st.markdown("---")

    # Break the Gemini markdown output into labelled sections
    sections = split_study_material_sections(content)

    # ---- Simple Explanation ----
    with st.container(border=True):
        st.markdown("### 📖 Simple Explanation")
        body = find_section(sections, "explanation")
        st.markdown(body if body else "_No explanation generated._")

    st.write("")

    # ---- Important Points ----
    with st.container(border=True):
        st.markdown("### ⭐ Important Points")
        body = find_section(sections, "important points")
        st.markdown(body if body else "_No key points generated._")

    st.write("")

    # ---- Flashcards ----
    with st.container(border=True):
        st.markdown("### 🧠 Flashcards")
        body = find_section(sections, "flashcards")
        st.markdown(body if body else "_No flashcards generated._")

    st.write("")

    # ---- Quiz Questions ----
    with st.container(border=True):
        st.markdown("### ❓ Quiz Questions")
        body = find_section(sections, "quiz")
        st.markdown(body if body else "_No quiz questions generated._")

    st.write("")

    # ---- Revision Summary ----
    with st.container(border=True):
        st.markdown("### 📝 Revision Summary")
        body = find_section(sections, "revision summary")
        st.markdown(body if body else "_No revision summary generated._")