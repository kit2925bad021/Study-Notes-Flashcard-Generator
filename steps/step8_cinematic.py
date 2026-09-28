"""
Step 8 — Cinematic Explanation
----------------------------------
Turns the study material already generated on the Dashboard into an
engaging, story-style "cinematic" explanation using Ollama.

Imported in app.py with:
    from steps.step8_cinematic import show_cinematic

FIX (this file only): the cinematic explanation was coming out almost
identical regardless of which study material was loaded. The small local
model (qwen2.5:0.5b) tends to fall back to a generic "movie narration"
template unless it is forced to actually use the material's own words.
To fix that, this version:

  1. Pulls the exact content out of the existing session state
     (st.session_state.study_material["content"]) — never a filename or
     topic title alone.
  2. Extracts the material's own recurring key terms and hands them to
     the model as a checklist it must work into the explanation.
  3. After generating, checks whether those key terms actually show up
     in the output. If they don't (i.e. the model went generic), it
     regenerates once with a stronger reminder.
  4. Detects when the underlying study material has changed (new
     upload / new generation) and clears any previous cinematic result
     so the old explanation is never shown next to new material.

Ollama is still only ever called through the existing generate_content()
from services.ollama_service, and only from inside the button handlers
(never on a plain rerun).
"""

import hashlib
import re
from collections import Counter

import streamlit as st

from services.ollama_service import generate_content
from steps.step19_ui import show_page_header, show_topic_bar

# ---------------------------------------------------------------------------
# Key-term extraction
# ---------------------------------------------------------------------------
# Small, generic stopword list so we don't have to pull in nltk/spacy just
# to find the handful of important, recurring words in the material.
_STOPWORDS = set(
    """
    a an the of and or but if then else for to in on at by with from as is
    are was were be been being this that these those it its into your you
    student college notes study material subject topic chapter which who
    whom what when where why how not no yes can will would should could
    also each every some any all many much more most other another such
    only own same so than too very just about above below over under
    between within without across during before after again further once
    here there their there's they them theirs he she his her we our us
    """.split()
)

MAX_CONTENT_CHARS = 6000  # keep the prompt focused and inside model context
MAX_KEY_TERMS = 14


def _extract_key_terms(content, max_terms=MAX_KEY_TERMS):
    """
    Pull out the material's own recurring, meaningful words — a cheap proxy
    for "the concepts this material is actually about". No external NLP
    libraries required; just frequency + stopword filtering.
    """
    if not content:
        return []

    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", content)
    if not words:
        return []

    counts = Counter()
    first_index = {}
    original_case = {}
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        counts[lw] += 1
        if lw not in first_index:
            first_index[lw] = i
            original_case[lw] = w

    if not counts:
        return []

    # Rank by how often the term recurs in the material (repeated terms are
    # a good signal of "this is actually what the material is about"),
    # falling back to first-appearance order to break ties.
    ranked = sorted(counts.keys(), key=lambda lw: (-counts[lw], first_index[lw]))
    return [original_case[lw] for lw in ranked[:max_terms]]


def _count_terms_present(text, key_terms):
    if not key_terms or not text:
        return 0
    text_lower = text.lower()
    return sum(1 for term in key_terms if term.lower() in text_lower)


def _is_sufficiently_grounded(text, key_terms):
    """True if the generated explanation actually used enough of the
    material's own key terms to be considered 'about' this material."""
    if not key_terms:
        # No extractable terms (e.g. very short material) — nothing to
        # check against, so don't force a pointless retry.
        return True
    required = max(3, len(key_terms) // 3)
    required = min(required, len(key_terms))
    return _count_terms_present(text, key_terms) >= required


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_cinematic_prompt(subject, topic, content, key_terms):
    """Build the prompt sent to Ollama, strongly grounded in the actual
    uploaded/generated study material."""
    trimmed_content = (content or "").strip()
    if len(trimmed_content) > MAX_CONTENT_CHARS:
        trimmed_content = trimmed_content[:MAX_CONTENT_CHARS] + "\n...(truncated)"

    terms_block = ", ".join(key_terms) if key_terms else "(none detected — rely fully on the material below)"

    return f"""You are creating a cinematic explanation of the student's uploaded study material.

Use the supplied study material as the source of truth. Every scene must be
based on concepts, terms, processes, examples, or relationships found in the
supplied material. Do not introduce a different subject. Do not use generic
filler. Do not invent information that is not supported by the study
material. Preserve the terminology and meaning of the study material.

Subject: {subject}
Topic/Chapter: {topic}

Study material (this is the ONLY source of truth — do not go beyond it):
\"\"\"
{trimmed_content}
\"\"\"

Key terms from THIS material that must appear, worked in naturally (do not
swap these out for a different subject's vocabulary): {terms_block}

Write in simple, beginner-friendly English, concise enough for a college
student to read in a few minutes. Structure your response using markdown
headings, in this exact order:

## 🎬 Title
A short, catchy title built from the actual topic and material above.

## 📖 Introduction
One short paragraph that sets up what THIS specific material is about.

## 🎥 Scenes
Break the material into 2 to 5 scenes — one scene per major concept, step,
or process that is actually described in the material above. Use as many
scenes as the material naturally supports (do not pad with filler scenes).
For each scene, use a sub-heading naming the concept it covers, e.g.
"### Scene 1: <concept from the material>", then explain that concept using
the material's own terminology, with a short real-life analogy if it helps.

## 🧠 Key Takeaway
2 to 4 bullet points with the single most important facts from the material.

## 🎯 Conclusion
A short, punchy wrap-up that ties the scenes back together.
"""


def generate_cinematic_explanation(subject, topic, content):
    """Build the prompt, call Ollama, validate it's actually grounded in
    this material, and return the raw response text."""
    key_terms = _extract_key_terms(content)
    prompt = _build_cinematic_prompt(subject, topic, content, key_terms)

    result_text = generate_content(prompt)

    if not _is_sufficiently_grounded(result_text, key_terms):
        # The model likely fell back to a generic template. Regenerate once
        # with an explicit reminder of the terms it needs to use.
        reminder = (
            "\n\nIMPORTANT REMINDER: your previous attempt did not clearly "
            "use the study material above — it read as generic. Rewrite the "
            "cinematic explanation so it is unmistakably about THIS "
            f"material and naturally includes these terms: {', '.join(key_terms)}. "
            "Do not describe any other subject or topic."
        )
        result_text = generate_content(prompt + reminder)

    return result_text


# ---------------------------------------------------------------------------
# Material-change detection
# ---------------------------------------------------------------------------

def _material_signature(study_material):
    """A stable fingerprint of the current study material, so we can tell
    when the student has uploaded/generated something new."""
    subject = str(study_material.get("subject", ""))
    topic = str(study_material.get("topic", ""))
    content = str(study_material.get("content", ""))
    raw = f"{subject}\x00{topic}\x00{content}".encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()


def show_cinematic():
    """Display the full Cinematic Explanation page."""
    show_page_header("🎬", "Cinematic Explanation", "Your study notes retold as a short story.")

    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("Please generate study material from the Dashboard first.")
        return

    subject = study_material["subject"]
    topic = study_material["topic"]
    content = study_material["content"]

    # ---- Subject and topic display ----
    show_topic_bar(subject, topic)

    st.markdown("---")

    # ---- Session state for this page ----
    if "cinematic_explanation" not in st.session_state:
        st.session_state.cinematic_explanation = None
    if "cinematic_material_signature" not in st.session_state:
        st.session_state.cinematic_material_signature = None

    # If the underlying study material has changed since the last cinematic
    # explanation was generated (new upload, new topic, regenerated notes,
    # etc.), the old explanation no longer matches it — drop it so the
    # student is never shown a story about the previous material.
    current_signature = _material_signature(study_material)
    if current_signature != st.session_state.cinematic_material_signature:
        st.session_state.cinematic_explanation = None
        st.session_state.cinematic_material_signature = current_signature

    # ---- Buttons to create / regenerate the explanation ----
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        create_clicked = st.button("🎬 Create Cinematic Explanation", type="primary")

    with button_col2:
        regenerate_clicked = st.button("🔄 Regenerate")

    if create_clicked or regenerate_clicked:
        with st.spinner("Creating your cinematic explanation..."):
            try:
                result_text = generate_cinematic_explanation(subject, topic, content)
                st.session_state.cinematic_explanation = result_text
                st.session_state.cinematic_material_signature = current_signature
                st.success("✅ Cinematic explanation created!")
            except Exception as e:
                st.error(f"⚠️ Something went wrong while creating the cinematic explanation. Details: {e}")

    st.markdown("---")

    # ---- Display the explanation, if one exists ----
    if st.session_state.cinematic_explanation:
        with st.container(border=True):
            st.markdown(st.session_state.cinematic_explanation)
    else:
        st.info("Click '🎬 Create Cinematic Explanation' to turn your study material into a story.")