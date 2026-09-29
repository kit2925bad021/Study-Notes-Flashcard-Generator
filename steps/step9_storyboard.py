"""
Step 9 — Storyboard Learning Mode
-----------------------------------
Renders the "🎞️ Storyboard" page inside the Streamlit app.

This page turns the study material already generated on the Dashboard
into a 6-scene visual storyboard using Gemini. It is TEXT ONLY:
no images, no videos, no image-generation API calls.

Wire this up in app.py the same way Steps 1 and 2 are wired up, e.g.:

    from steps.step9_storyboard import show_storyboard
    ...
    elif st.session_state.current_page == "🎞️ Storyboard":
        show_storyboard()

This file does not touch st.session_state.study_material or any other
step's session state — it only reads study_material / cinematic_explanation
(if present) and writes to its own keys (storyboard, storyboard_scene_index).
"""

import re

import streamlit as st

from services.gemini_service import generate_content
from steps.step19_ui import require_study_material, show_page_header, show_topic_bar


# ----------------------------------------------------------------------
# 1. PROMPT BUILDER
# ----------------------------------------------------------------------

def _build_storyboard_prompt(subject, topic, content, cinematic_explanation):
    """
    Build the prompt sent to Gemini asking for exactly 6 storyboard scenes,
    grounded strictly in the given study material (no invented facts).
    """

    # cinematic_explanation is optional — only include it if we have it,
    # so the prompt stays clean when it's not available.
    extra_context = ""
    if cinematic_explanation:
        extra_context = f"""
Additional cinematic-style explanation of the topic (use this for tone/flavor
only, not as a source of new facts):
{cinematic_explanation}
"""

    prompt = f"""
You are an educational storyboard writer helping a student understand a
topic through a simple sequence of visual learning scenes.

Subject: {subject}
Topic: {topic}

Study material (this is the ONLY source of facts you may use):
{content}
{extra_context}

TASK:
Create exactly 6 storyboard scenes that together explain this topic in a
clear, logical learning sequence — like a simple explainer video broken
into scenes.

Rules:
- Use simple, beginner-friendly English.
- Be educational, not fictional — do not invent characters, events, or
  facts that are not supported by the study material above.
- The "Visual" description should describe something that could later be
  turned into a single illustration or image (describe the scene, not
  camera direction or special effects).
- Keep each field concise (1-3 sentences).
- The 6 scenes should build on each other in a logical order, moving from
  introducing the topic to a final takeaway/summary scene.

Respond using EXACTLY this format, with no extra commentary before or
after it:

## Scene 1
**Title:** ...
**Visual:** ...
**Narration:** ...
**Learning Point:** ...

## Scene 2
**Title:** ...
**Visual:** ...
**Narration:** ...
**Learning Point:** ...

## Scene 3
**Title:** ...
**Visual:** ...
**Narration:** ...
**Learning Point:** ...

## Scene 4
**Title:** ...
**Visual:** ...
**Narration:** ...
**Learning Point:** ...

## Scene 5
**Title:** ...
**Visual:** ...
**Narration:** ...
**Learning Point:** ...

## Scene 6
**Title:** ...
**Visual:** ...
**Narration:** ...
**Learning Point:** ...
"""
    return prompt


# ----------------------------------------------------------------------
# 2. PARSER
# ----------------------------------------------------------------------

def _parse_storyboard(text):
    """
    Parse Gemini's raw text response into a list of scene dicts:

        {
            "number": int,
            "title": str,
            "visual": str,
            "narration": str,
            "learning_point": str,
        }

    This parser is intentionally forgiving. If it can't find clean scene
    blocks (e.g. Gemini didn't follow the format exactly), it returns an
    empty list rather than raising an error — the caller is expected to
    fall back to showing the raw text in that case.
    """

    if not text:
        return []

    scenes = []

    # Split the text on "## Scene <number>" headings.
    # re.split with a capturing group returns:
    # [before_first_match, number1, block1, number2, block2, ...]
    # so we walk it in pairs of (number, block).
    scene_blocks = re.split(r"##\s*Scene\s*(\d+)", text)

    for i in range(1, len(scene_blocks) - 1, 2):
        number_str = scene_blocks[i]
        block = scene_blocks[i + 1]

        try:
            number = int(number_str)
        except ValueError:
            continue

        title = _extract_field(block, "Title")
        visual = _extract_field(block, "Visual")
        narration = _extract_field(block, "Narration")
        learning_point = _extract_field(block, "Learning Point")

        # If we couldn't find at least a title, treat this scene as
        # unparseable rather than showing an empty card.
        if not title:
            continue

        scenes.append(
            {
                "number": number,
                "title": title,
                "visual": visual or "Not provided.",
                "narration": narration or "Not provided.",
                "learning_point": learning_point or "Not provided.",
            }
        )

    return scenes


def _extract_field(block, field_name):
    """
    Pull the text after '**<field_name>:**' up to the next '**' marker or
    the end of the block. Returns an empty string if the field isn't found.
    """
    pattern = rf"\*\*{re.escape(field_name)}:\*\*\s*(.*?)(?=\n\*\*|\Z)"
    match = re.search(pattern, block, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


# ----------------------------------------------------------------------
# 3. GENERATION
# ----------------------------------------------------------------------

def _generate_storyboard(subject, topic, content, cinematic_explanation):
    """
    Build the prompt, call Gemini, parse the response, and return a dict:

        {
            "scenes": [...],       # list of parsed scene dicts (may be empty)
            "raw_text": "...",     # the raw Gemini response, for fallback display
        }

    Raises whatever exception generate_content() raises — the caller
    (show_storyboard) is responsible for catching it and showing a
    friendly error message.
    """
    prompt = _build_storyboard_prompt(subject, topic, content, cinematic_explanation)
    raw_text = generate_content(prompt)
    scenes = _parse_storyboard(raw_text)

    return {
        "scenes": scenes,
        "raw_text": raw_text,
    }


# ----------------------------------------------------------------------
# 4. NAVIGATION CALLBACKS
# ----------------------------------------------------------------------
# Streamlit runs a button's on_click callback BEFORE it redraws the page.
# That way "Scene 3 of 6" is always correct. (When the buttons changed the
# index while the page was already being drawn, the label showed the OLD
# scene number after pressing Next.)

def _previous_scene():
    if st.session_state.storyboard_scene_index > 0:
        st.session_state.storyboard_scene_index -= 1


def _next_scene(max_index):
    if st.session_state.storyboard_scene_index < max_index:
        st.session_state.storyboard_scene_index += 1


# ----------------------------------------------------------------------
# 5. PAGE UI
# ----------------------------------------------------------------------

def show_storyboard():
    """Render the full Storyboard page (Step 9)."""

    show_page_header(
        "🎞️", "Storyboard",
        "Your study material as six easy-to-follow visual learning scenes.",
    )

    # ------------------------------------------------------------
    # Require study material from the Dashboard first.
    # ------------------------------------------------------------
    study_material = require_study_material()
    if study_material is None:
        return

    subject = study_material.get("subject", "")
    topic = study_material.get("topic", "")
    content = study_material.get("content", "")

    # Optional extra context from an earlier step, if it exists.
    cinematic_explanation = st.session_state.get("cinematic_explanation")

    show_topic_bar(subject, topic)

    # ------------------------------------------------------------
    # One button: "Generate" first, "Regenerate" once a storyboard exists.
    # ------------------------------------------------------------
    existing = st.session_state.get("storyboard")
    has_storyboard = bool(existing and existing.get("scenes"))
    generate_clicked = st.button(
        "🔄 Regenerate Storyboard" if has_storyboard else "🎞️ Generate Storyboard",
        type="secondary" if has_storyboard else "primary",
    )

    if generate_clicked:
        with st.spinner("Creating your visual storyboard..."):
            try:
                result = _generate_storyboard(
                    subject, topic, content, cinematic_explanation
                )
                st.session_state.storyboard = result
                st.session_state.storyboard_scene_index = 0  # always start at Scene 1
                st.success("✅ Storyboard generated!")
            except Exception as e:
                # Handle Gemini / API errors gracefully — don't crash the page.
                st.error(f"Sorry, we couldn't generate the storyboard: {e}")
                return

    # ------------------------------------------------------------
    # Display the stored storyboard, if any.
    # ------------------------------------------------------------
    storyboard = st.session_state.get("storyboard")

    if not storyboard:
        st.info("Click **🎞️ Generate Storyboard** to get started.")
        return

    scenes = storyboard.get("scenes", [])
    raw_text = storyboard.get("raw_text", "")

    if not scenes:
        # Safe fallback: parsing failed, so just show the raw response
        # instead of crashing or showing nothing.
        # Never show the raw model reply to the student.
        st.warning(
            "Unable to build the storyboard scenes right now. "
            "Please click Regenerate Storyboard to try again."
        )
        return

    st.markdown("---")

    # ------------------------------------------------------------
    # Sequential scene viewer (view one scene at a time).
    # ------------------------------------------------------------
    if "storyboard_scene_index" not in st.session_state:
        st.session_state.storyboard_scene_index = 0

    # Keep the index in range in case a regenerate returned fewer scenes.
    max_index = len(scenes) - 1
    if st.session_state.storyboard_scene_index > max_index:
        st.session_state.storyboard_scene_index = 0

    current = st.session_state.storyboard_scene_index

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        st.button(
            "⬅️ Previous",
            use_container_width=True,
            disabled=current == 0,
            on_click=_previous_scene,
        )
    with nav_col2:
        st.markdown(
            f"<p style='text-align:center;'>Scene {current + 1} of {len(scenes)}</p>",
            unsafe_allow_html=True,
        )
        st.progress((current + 1) / len(scenes))
    with nav_col3:
        st.button(
            "Next ➡️",
            use_container_width=True,
            type="primary",
            disabled=current == max_index,
            on_click=_next_scene,
            args=(max_index,),
        )

    _render_scene(scenes[current])

    # ------------------------------------------------------------
    # Full storyboard, all scenes listed in order.
    # ------------------------------------------------------------
    with st.expander("📜 View all scenes"):
        for scene in scenes:
            _render_scene(scene)


def _render_scene(scene):
    """Render a single scene: visual and narration side by side, takeaway underneath."""
    with st.container(border=True):
        st.markdown(f"### 🎬 Scene {scene['number']}: {scene['title']}")

        visual_col, narration_col = st.columns(2)
        with visual_col:
            st.markdown("**👁️ Visual**")
            st.write(scene["visual"])
        with narration_col:
            st.markdown("**🎙️ Narration**")
            st.write(scene["narration"])

        st.success(f"🧠 **Learning point:** {scene['learning_point']}")