"""
steps/step20_study_flow.py

The Study Material flow:

    Dashboard (generate form)  ->  Study Material page  ->  Take Quiz

Two entry points used by app.py:

  show_generate_form(subject_list, go_to_page)
      The form (file, subject, topic) shown at the top of the DASHBOARD.
      On "Generate Study Material" it makes SIX small Ollama requests (one
      per section - see "Generation" below), saves the combined result in
      st.session_state.study_material (and in SQLite when a student
      profile is saved), then sends the student to the Study Material
      page. Generated text is NOT shown on the Dashboard; only a progress
      bar is.

  show_study_flow(subject_list, go_to_page)
      The dedicated Study Material page: shows the COMPLETE material in
      cards / expandable sections plus a single "Take Quiz" action. That
      button only navigates - the Quiz page generates its own questions,
      and only when the student asks there.

st.session_state.study_material keeps the exact shape the rest of the app
already uses: {"subject": ..., "topic": ..., "content": <markdown text>}.

WHY SIX REQUESTS INSTEAD OF ONE
--------------------------------
Asking a tiny local model (qwen2.5:0.5b) to write six differently-purposed
sections in a single answer reliably collapsed into six versions of the
same paragraph - the model doesn't have the capacity to track "don't
repeat what you already wrote three sections ago" over one long
generation. Generating one section per request fixes that structurally:
each request only has ONE job, and every request from Key Points onward
is shown the sections written just before it and told explicitly not to
restate them. The six pieces are still combined into the exact same
`content` markdown string (with fixed "## " headings, in a fixed order)
that the rest of the app already expects, so parsing, display, download,
and database storage below are unchanged in shape.

Only "Generate Study Material" triggers these six requests, and only
once per click (not on Streamlit reruns) - see show_generate_form() and
_generate() below.

WHY SECTIONS COULD STILL REPEAT DESPITE THE ABOVE
----------------------------------------------------
Two extra problems were found on top of the "one job per request" split:

1. Context-window overflow. By the time a later prompt (Detailed Notes,
   Examples) includes the uploaded source text AND everything written
   so far, it can be longer than Ollama's default 2048-token context
   window. When that happens, Ollama silently drops the oldest part of
   the prompt - often the "do not repeat this" instructions - and the
   model falls back to generic, repetitive text. Every request below
   now explicitly sets a large-enough num_ctx (see GEN_OPTIONS /
   DETAILED_NOTES_OPTIONS) so nothing gets truncated.
2. No real validation. The previous version only warned the student
   after the fact (_sections_look_duplicated). Key Points and Important
   Questions can still contain near-duplicate bullets/questions (same
   fact, reworded) even when the six sections overall look distinct, and
   nothing caught a whole section that quietly restated an earlier one.
   _run_section() now retries a section once (with a stronger
   instruction) when it overlaps too much with what was already written,
   and _dedupe_bullets() / _dedupe_questions() drop exact and
   near-duplicate lines using the normalize -> compare -> keep-unique
   rule, capping to the requested count.
"""

import difflib
import io
import re
from html import escape

import streamlit as st

from services.ollama_service import generate_content
from steps.step13_memory import record_topic
from steps.step18_settings import language_instruction
from steps.step19_ui import show_info_card

# Keep prompts small: a local model reads (and answers) faster.
MAX_SOURCE_CHARS = 6000

# (key, label shown to the student, words that identify the heading)
# Order matches the six sections the student sees, and the order they are
# generated in (see _generate below).
SECTIONS = [
    ("summary", "📝 Summary", ("summary",)),
    ("key_points", "🔑 Key Points", ("key point", "key points")),
    ("important_concepts", "💡 Important Concepts", ("important concept", "concept")),
    ("detailed_notes", "📖 Detailed Notes", ("detailed note", "notes")),
    ("examples", "📌 Examples", ("example",)),
    ("important_questions", "❓ Important Questions", ("important question", "question")),
]
OPEN_BY_DEFAULT = {"summary", "key_points"}

# Only a level-2 heading ("## Heading") starts a new section. A deeper
# heading such as "### Process" (used inside Detailed Notes) is left
# alone so it still renders as a real sub-heading instead of being
# swallowed.
_HEADING_RE = re.compile(r"^\s{0,3}##\s+(.+?)\s*#*\s*$")

# Session-state keys created by the pages that work on ONE topic (flashcards,
# quiz, scenarios, games, storyboard, ...). When a NEW topic is generated they
# are removed, so no page shows content from the previous topic. Running scores
# of the three simple games (scenarios, emoji, two truths) are kept as session totals.
_TOPIC_STATE_PREFIXES = (
    "flashcard", "quiz_", "scenario_", "emoji_", "two_truths_",
    "cinematic_", "storyboard", "pop_culture", "dynamic_",
)
_KEEP_KEYS = {
    "scenario_score", "scenario_attempts",
    "emoji_score", "emoji_attempts",
    "two_truths_score", "two_truths_attempts",
}


# ----------------------------------------------------------------------
# File reading
# ----------------------------------------------------------------------
def extract_text(uploaded_file):
    """Return the text inside an uploaded TXT / PDF / DOCX file (or raise RuntimeError)."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        # Step 7 already uses PyPDF2, so try that first, then the newer pypdf.
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            try:
                from pypdf import PdfReader
            except ImportError:
                raise RuntimeError("Reading PDF files needs PyPDF2. Run: pip install PyPDF2")
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        try:
            import docx  # python-docx (also used by step 7)
        except ImportError:
            raise RuntimeError("Reading DOCX files needs python-docx. Run: pip install python-docx")
        document = docx.Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    raise RuntimeError(f"Unsupported file type: {uploaded_file.name}")


# ----------------------------------------------------------------------
# Prompts - one small, focused prompt per section (plain Python, no
# Streamlit calls). Every prompt after the first is shown what was
# already written so it can't just restate it.
# ----------------------------------------------------------------------
def _preamble(subject, topic, source_text, language_note):
    """Shared opening for every section prompt: subject/topic, the source
    of truth (or a note that there isn't one), and the language style."""
    if source_text.strip():
        source_block = (
            "Base your answer only on the following uploaded study material - "
            "do not add facts, definitions, or examples it does not support:\n"
            "\"\"\"\n"
            f"{source_text.strip()[:MAX_SOURCE_CHARS]}\n\"\"\"\n"
        )
    else:
        source_block = (
            "No file was uploaded for this topic, so base your answer on "
            "accurate, well-established knowledge of this subject and "
            "topic instead of guessing.\n"
        )
    style_line = language_note or "Use simple English."
    return (
        "You are a careful study assistant for college students.\n"
        f"Subject: {subject}\nTopic/Chapter: {topic}\n{source_block}{style_line}\n\n"
    )


def _already_written_block(**sections):
    """Show a later prompt the sections written just before it, so it can
    avoid repeating them. Only pass the sections that already exist."""
    parts = []
    for label, text in sections.items():
        if text:
            parts.append(f"--- {label} (already written) ---\n{text.strip()}")
    if not parts:
        return ""
    return (
        "\nFor reference, here is what has already been written about this "
        "topic. Do NOT repeat these sentences or just reformat them - add "
        "content that is not already covered above:\n"
        + "\n\n".join(parts)
        + "\n\n"
    )


def prompt_summary(subject, topic, source_text, language_note):
    return _preamble(subject, topic, source_text, language_note) + """Write ONLY the "Summary" - 1-2 connected paragraphs of normal prose.
Do NOT use bullet points, numbers, or headings. Explain the topic from
beginning to end at a high level: the main idea, and the purpose,
process, and conclusion when they are present. Connect the ideas
naturally in simple, student-friendly language, keeping the source's own
technical terms, but leave out small details. Think: "If a student only
had 30 seconds to understand this chapter, what should they know?"

Style example only (do NOT copy its wording or subject matter - write
about the actual topic above instead): "Fertilization is a process
involved in sexual reproduction where male and female gametes combine to
form a new organism. It contributes to genetic variation and begins a
sequence of cell division and development."

Write only the summary paragraphs. No heading, no label, nothing else."""


def prompt_key_points(subject, topic, source_text, language_note, summary_text):
    return (
        _preamble(subject, topic, source_text, language_note)
        + _already_written_block(Summary=summary_text)
        + """Write ONLY the "Key Points" - 5 to 8 VERY SHORT bullet points for
last-minute revision, one per line, each starting with "- ". Each bullet
is a single fact, definition, formula, term, or short statement - never a
full sentence explanation and never a paragraph. Prefer, when the
material supports them: definitions, formulas/equations, important
facts, steps in a process, classifications, differences/comparisons, and
key terminology.

Do NOT write connected paragraphs. Do NOT turn the Summary above into
bullets - pick different, more specific facts than the Summary mentions.
Do not add anything the topic/material does not support.

Style example only (do NOT copy its wording or subject matter): "Fusion
of male and female gametes" / "Produces a zygote" / "Creates genetic
variation".

Write only the bullet list. No heading, no label, nothing else."""
    )


def prompt_important_concepts(subject, topic, source_text, language_note, summary_text, key_points_text):
    return (
        _preamble(subject, topic, source_text, language_note)
        + _already_written_block(Summary=summary_text, **{"Key Points": key_points_text})
        + """Write ONLY the "Important Concepts" - the core IDEAS a student
must UNDERSTAND, not facts to memorise. Write FEWER THAN 5 concepts:
prefer 3-4, and write only 1-2 if that is genuinely all the material
supports. Never pad this list to reach a target number.

Use exactly this format, one bullet per concept:

- <Concept Name> - <one or two sentences explaining the IDEA: what it
  means, how it works, or how it relates to another concept>

A concept is NOT a short fact restated with more words. Compare:

Key Point (already written above): "Uses inheritance"
WRONG Important Concept (just restates the same fact): "Uses
inheritance - the class uses inheritance."
RIGHT Important Concept (explains an idea): "Method overriding - a
derived class can provide its own implementation of a method already
defined in its base class, which is what lets inheritance produce
different behaviour in different subclasses."

Every concept must explain something the Key Points above only named,
never repeat a Key Point in slightly different words, and must not
just restate the Summary. Do not invent a concept the material does
not support.

Write only the bullet list in that format. No extra heading, nothing
else."""
    )


def prompt_detailed_notes(subject, topic, source_text, language_note, summary_text, key_points_text, concepts_text):
    return (
        _preamble(subject, topic, source_text, language_note)
        + _already_written_block(
            Summary=summary_text,
            **{"Key Points": key_points_text, "Important Concepts": concepts_text},
        )
        + """Write ONLY the "Detailed Notes" - the LONGEST section, and the
complete explanation a student needs to actually study the topic. Use
short "### Sub-heading" headings such as Definition, Process, Components
Involved, Importance, and Result when they fit the material, explain any
process step by step, include definitions, terminology, and
relationships between ideas, and organize everything logically.

This section must add NEW detail that is not already above - do not
repeat the Summary or Key Points, and do not just restate the Important
Concepts explanations word for word. Answer: "How does this topic
actually work?"

Write only the detailed notes (with their ### sub-headings). No top-level
heading, nothing else."""
    )


def prompt_examples(subject, topic, source_text, language_note, concepts_text, detailed_notes_text):
    return (
        _preamble(subject, topic, source_text, language_note)
        + _already_written_block(
            **{"Important Concepts": concepts_text, "Detailed Notes": detailed_notes_text}
        )
        + """Write ONLY the "Examples" - 2 to 4 short examples that show a
concept above in a simple, concrete situation. Do not force 4 examples
if the material only supports 2 or 3 good ones. Use exactly this format:

Example 1:
<the example itself, about two lines - a concrete situation, not a
definition>

Example 2:
<...>

If the uploaded material already contains an example, use that one and
add "(from the material)" at the end. If it does not, you may write one
simple illustrative example ONLY if it can be worked out directly from
the material, and add "(illustrative example)" at the end. Never just
restate a concept's definition or a Key Point as if it were an example,
and never repeat a sentence from the Detailed Notes above - show the
idea in action instead of describing it again.

Write only the example list in that format. No extra heading, no title
line, no "what it demonstrates" line, nothing else."""
    )


def prompt_important_questions(subject, topic, source_text, language_note):
    return (
        _preamble(subject, topic, source_text, language_note)
        + """Write ONLY the "Important Questions" - exactly 5 study/revision
questions for self-testing, based only on the topic/material above.
Cover 5 DIFFERENT parts of the topic - never write two questions that
test the same fact in different words. Vary the question type across
the 5 (for example: definition, concept understanding, process,
cause/effect, comparison, explanation, application) when the material
supports that variety. Use exactly this format:

1. <question text>?
   Answer: <a clear, complete answer using only information from
   above - not just one word>

2. <question text>?
   Answer: <...>

(continue through 5)

These are open study questions with written answers, not a
multiple-choice quiz, so never write answer options (A/B/C/D).

Write only the numbered list in that format. No extra heading, nothing
else."""
    )


_WORD_RE = re.compile(r"[a-z']+")
# Demote any level-2 heading the model invents inside its own answer
# (e.g. if it echoes "## Summary") to level-3, so combining the six
# independently-generated pieces below can never accidentally create a
# stray extra heading.
_STRAY_HEADING_RE = re.compile(r"(?m)^(\s{0,3})##(?!#)\s+")


def _clean_section_text(text):
    text = (text or "").strip()
    return _STRAY_HEADING_RE.sub(r"\1### ", text)


# ----------------------------------------------------------------------
# Ollama request options for the six section calls.
#
# num_ctx is set explicitly (rather than left at Ollama's 2048-token
# default) because a later prompt = source text + everything written so
# far + instructions, which can otherwise overflow the default window
# and get silently truncated - see the module docstring. Detailed Notes
# gets a bigger window and a bigger num_predict because it is meant to
# be the longest section. Temperature is nudged up slightly from the
# service default (0.2) so the six sections don't fall into near-
# identical phrasing purely because a tiny model is very deterministic.
# ----------------------------------------------------------------------
GEN_OPTIONS = {"num_ctx": 4096, "temperature": 0.35}
DETAILED_NOTES_OPTIONS = {"num_ctx": 6144, "temperature": 0.35, "num_predict": 1536}

_RETRY_NOTE = (
    "\n\nIMPORTANT: your previous attempt at this section repeated "
    "sentences, facts, or explanations that were already written above. "
    "Write it again from scratch using different sentences, a different "
    "structure, and details that are not already covered above."
)


def _word_overlap_ratio(text_a, text_b):
    """Jaccard overlap of the words in two sections. Short sections (a
    handful of words) are treated as having no meaningful overlap."""
    words_a = set(_WORD_RE.findall((text_a or "").lower()))
    words_b = set(_WORD_RE.findall((text_b or "").lower()))
    if len(words_a) < 12 or len(words_b) < 12:
        return 0.0
    return len(words_a & words_b) / max(1, len(words_a | words_b))


def _too_similar_to_existing(new_text, existing_texts, threshold=0.75):
    """True if new_text is mostly the same vocabulary as any section that
    was already written (whole-section duplicate check)."""
    return any(
        _word_overlap_ratio(new_text, existing) > threshold
        for existing in existing_texts
        if existing
    )


def _run_section(prompt, existing_texts, options=None):
    """
    Run one Ollama request for a section. If the result overlaps too
    much with a section written earlier in this same generation, ask
    once more with a stronger anti-repeat instruction and use that
    instead - this is the "rewrite the shorter/later section" rule from
    the content spec, applied automatically instead of only warning the
    student after the fact.
    """
    text = _clean_section_text(generate_content(prompt, options=options))
    if existing_texts and _too_similar_to_existing(text, existing_texts):
        retry_text = _clean_section_text(
            generate_content(prompt + _RETRY_NOTE, options=options)
        )
        if retry_text:
            text = retry_text
    return text


def _light_stem(word):
    """Crude suffix strip (ing/edly/ed/es/s) so 'improves' and 'improved'
    compare as the same word. Good enough for near-duplicate detection,
    not meant to be a real stemmer."""
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _normalize_line(text):
    """
    Lowercase -> strip punctuation -> light-stem each word -> sort the
    words. Sorting makes the comparison order-independent, which is what
    catches reworded duplicates like "Polymorphism improves code reuse"
    vs "Code reuse is improved through polymorphism" - same words, same
    fact, different order/tense, and a plain left-to-right string
    comparison would miss that.
    """
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [_light_stem(w) for w in text.split() if w]
    return " ".join(sorted(words))


def _is_near_duplicate(norm_a, norm_b, threshold=0.58):
    """True if two ALREADY-NORMALIZED (see _normalize_line) lines are the
    same fact reworded."""
    if not norm_a or not norm_b:
        return False
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold


def _is_containment_duplicate(short_norm, long_norm, threshold=0.85):
    """True if nearly all of a SHORT normalized line's words also appear
    in a LONGER normalized line - catches a short fact (e.g. a Key
    Point) that was simply padded into a longer sentence rather than
    the longer line genuinely explaining a different idea. Skipped for
    very short lines (1 word) where this would be unreliable, and only
    meaningful when the second line is actually longer."""
    if not short_norm or not long_norm:
        return False
    short_words = short_norm.split()
    long_words_set = set(long_norm.split())
    if len(short_words) < 2 or len(long_words_set) <= len(short_words):
        return False
    found = sum(1 for w in short_words if w in long_words_set)
    return (found / len(short_words)) >= threshold


_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*)$")


def _extract_normalized_bullets(text):
    """Parse bullet lines out of text and return their normalized forms
    (see _normalize_line), used to check one section's bullets against
    another section's bullets (e.g. concepts vs key points)."""
    norms = []
    for line in (text or "").splitlines():
        match = _BULLET_RE.match(line)
        if not match:
            continue
        norm = _normalize_line(match.group(1).strip())
        if norm:
            norms.append(norm)
    return norms


def _dedupe_bullets(text, max_count=8, avoid_norms=None, bullet_prefix="- "):
    """
    DUPLICATE VALIDATION: normalize every bullet, reject exact and near
    duplicates against each other, then keep only the first max_count
    unique ones (the model is asked for the most important points
    first). avoid_norms is an optional list of normalized lines from a
    DIFFERENT section (e.g. Key Points) that a bullet here must not
    duplicate either - this is what stops "Important Concept" bullets
    from just restating a "Key Point" in other words.
    """
    avoid_norms = avoid_norms or []
    kept_raw, kept_norm = [], []
    for line in (text or "").splitlines():
        match = _BULLET_RE.match(line)
        if not match:
            continue
        bullet = match.group(1).strip()
        if not bullet:
            continue
        norm = _normalize_line(bullet)
        if not norm:
            continue
        if any(norm == existing or _is_near_duplicate(norm, existing) for existing in kept_norm):
            continue
        if any(
            norm == existing
            or _is_near_duplicate(norm, existing)
            or _is_containment_duplicate(existing, norm)
            for existing in avoid_norms
        ):
            continue
        kept_raw.append(bullet)
        kept_norm.append(norm)
        if len(kept_raw) >= max_count:
            break
    return "\n".join(f"{bullet_prefix}{b}" for b in kept_raw)


_QUESTION_START_RE = re.compile(r"(?m)^\s*\d+\.\s+.*\?\s*$")
_QLINE_RE = re.compile(r"^\s*\d+\.\s*(.*)$")
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+\.\s*")


def _split_numbered_blocks(text, start_re):
    """Split text into one chunk per numbered item, using start_re to
    find where each item begins."""
    text = (text or "").strip()
    if not text:
        return []
    starts = [m.start() for m in start_re.finditer(text)]
    if not starts:
        return [text]
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]].strip() for i in range(len(starts) - 1)]


def _dedupe_questions(text, max_count=5):
    """
    QUESTION VALIDATION: normalize each question, reject exact and near
    duplicates (same fact asked a different way), keep at most
    max_count, and renumber what's left. If duplicates have to be
    dropped, the result can end up with fewer than max_count questions -
    that's intentional: a repeated question is worse than a short list
    (see "do not fill sections with repeated content just to meet the
    required number").
    """
    blocks = _split_numbered_blocks(text, _QUESTION_START_RE)
    kept_blocks, kept_norm = [], []
    for block in blocks:
        first_line = block.splitlines()[0] if block else ""
        match = _QLINE_RE.match(first_line)
        question_text = match.group(1).strip() if match else first_line.strip()
        norm = _normalize_line(question_text)
        if not norm or any(
            norm == existing or _is_near_duplicate(norm, existing) for existing in kept_norm
        ):
            continue
        kept_blocks.append(block)
        kept_norm.append(norm)
        if len(kept_blocks) >= max_count:
            break
    renumbered = []
    for i, block in enumerate(kept_blocks, start=1):
        renumbered.append(_LEADING_NUMBER_RE.sub(f"{i}. ", block, count=1))
    return "\n\n".join(renumbered)


_EXAMPLE_START_RE = re.compile(r"(?m)^\s*Example\s*\d+\s*:")


def _cap_blocks(text, start_re, max_count):
    """Keep only the first max_count numbered blocks (used for Examples,
    which are told 2-4 but occasionally overshoot). Trims from the tail
    only, so existing numbering stays correct without renumbering."""
    blocks = _split_numbered_blocks(text, start_re)
    if len(blocks) <= max_count:
        return text
    return "\n\n".join(blocks[:max_count])


def _section_key(heading_text):
    """Map a heading such as '🔑 Key Points' to one of our section keys (or None)."""
    lowered = heading_text.lower()
    for key, _label, words in SECTIONS:
        if any(word in lowered for word in words):
            return key
    return None


def parse_sections(content):
    """
    Split the combined markdown back into {section_key: text}. Since the
    six headings below are always written by us (not the model), this is
    mostly a formality that also protects older material saved by a
    previous version of this file, or content edited by hand.

    Only a level-2 heading ("## ...") is a section boundary; a deeper
    heading (e.g. "### Process" inside Detailed Notes) is left exactly as
    written, so it still renders as a real sub-heading.
    """
    found = {"_intro": []}  # text before the first recognised heading
    current = "_intro"
    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            key = _section_key(match.group(1))
            if key and key not in found:
                current = key
                found[current] = []
                continue
            found[current].append(f"**{match.group(1)}**")
            continue
        found[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in found.items() if "\n".join(lines).strip()}


def _sections_look_duplicated(sections):
    """
    Rough heuristic only - not a factual check. Flags when two of the six
    sections still share most of their vocabulary despite the separate,
    context-aware requests, the num_ctx fix, and the per-section retry
    in _run_section(). Kept as a final display-time safety net for the
    student, on top of (not instead of) the generation-time checks above.
    """
    known_keys = [key for key, _label, _words in SECTIONS]
    texts = {k: v for k, v in sections.items() if k in known_keys and v}
    keys = list(texts.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _word_overlap_ratio(texts[keys[i]], texts[keys[j]]) > 0.75:
                return True
    return False


def _friendly_error(error):
    """RuntimeErrors from ollama_service are already readable; show others with their type."""
    if isinstance(error, RuntimeError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _find_student_id():
    """Best effort: the Student Profile page stores the id in session_state."""
    for key in ("student_id", "current_student_id"):
        value = st.session_state.get(key)
        if value:
            return value
    return None


def _save_to_database(subject, topic, content):
    """Save the material to SQLite when a student is known. Never blocks the page."""
    student_id = _find_student_id()
    if not student_id:
        return
    try:
        from database.database import save_study_material, save_study_session

        save_study_material(student_id, subject, topic, content)
        save_study_session(student_id, subject, topic)
    except Exception as e:  # database problems must not stop the student
        st.warning(f"Study material was generated but could not be saved to the database: {e}")


def reset_generated_content():
    """
    Remove everything the other pages generated for the PREVIOUS topic
    (flashcards, quiz, scenarios, storyboard, ...). Without this, a student
    who generates a new topic would still see the old flashcards and quiz
    under the new topic's name.
    """
    for key in list(st.session_state.keys()):
        if (
            isinstance(key, str)
            and key.startswith(_TOPIC_STATE_PREFIXES)
            and key not in _KEEP_KEYS
        ):
            del st.session_state[key]


# ----------------------------------------------------------------------
# Generation (SIX small Ollama requests, one per section - see module
# docstring for why). Still only ever triggered by the button below, and
# still only once per click, not on reruns.
# ----------------------------------------------------------------------
def _generate(subject, topic, uploaded_file, go_to_page):
    """Run six Ollama requests (one per section), store the combined result,
    open the Study Material page."""
    source_text = ""
    if uploaded_file is not None:
        try:
            source_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Study Material generation failed (reading the uploaded file): {_friendly_error(e)}")
            return
        if not source_text.strip():
            st.error(
                "Study Material generation failed: no text could be read from "
                f"'{uploaded_file.name}' (it may be a scanned/image-only file)."
            )
            return

    was_truncated = len(source_text.strip()) > MAX_SOURCE_CHARS
    language_note = language_instruction()

    progress = st.progress(0.0, text="Starting Ollama... (the first request can take a moment while the model loads)")
    texts = {}
    try:
        progress.progress(0 / 6, text="Writing Summary... (1/6)")
        texts["summary"] = _run_section(
            prompt_summary(subject, topic, source_text, language_note),
            existing_texts=[],
            options=GEN_OPTIONS,
        )

        progress.progress(1 / 6, text="Writing Key Points... (2/6)")
        texts["key_points"] = _dedupe_bullets(
            _run_section(
                prompt_key_points(subject, topic, source_text, language_note, texts["summary"]),
                existing_texts=[texts["summary"]],
                options=GEN_OPTIONS,
            ),
            max_count=8,
        )
        key_point_norms = _extract_normalized_bullets(texts["key_points"])

        progress.progress(2 / 6, text="Writing Important Concepts... (3/6)")
        texts["important_concepts"] = _dedupe_bullets(
            _run_section(
                prompt_important_concepts(
                    subject, topic, source_text, language_note, texts["summary"], texts["key_points"]
                ),
                existing_texts=[texts["summary"], texts["key_points"]],
                options=GEN_OPTIONS,
            ),
            max_count=4,
            avoid_norms=key_point_norms,
        )

        progress.progress(3 / 6, text="Writing Detailed Notes... (4/6)")
        texts["detailed_notes"] = _run_section(
            prompt_detailed_notes(
                subject, topic, source_text, language_note,
                texts["summary"], texts["key_points"], texts["important_concepts"],
            ),
            existing_texts=[texts["summary"], texts["key_points"], texts["important_concepts"]],
            options=DETAILED_NOTES_OPTIONS,
        )

        progress.progress(4 / 6, text="Writing Examples... (5/6)")
        texts["examples"] = _cap_blocks(
            _run_section(
                prompt_examples(
                    subject, topic, source_text, language_note,
                    texts["important_concepts"], texts["detailed_notes"],
                ),
                existing_texts=[texts["important_concepts"], texts["detailed_notes"]],
                options=GEN_OPTIONS,
            ),
            _EXAMPLE_START_RE,
            max_count=4,
        )

        progress.progress(5 / 6, text="Writing Important Questions... (6/6)")
        texts["important_questions"] = _dedupe_questions(
            _run_section(
                prompt_important_questions(subject, topic, source_text, language_note),
                existing_texts=[],
                options=GEN_OPTIONS,
            ),
            max_count=5,
        )
    except Exception as e:
        progress.empty()
        st.error(f"Study Material generation failed: {_friendly_error(e)}")
        return
    progress.progress(1.0, text="Done!")
    progress.empty()

    if not any(texts.values()):
        st.error("Study Material generation failed: Ollama returned an empty response.")
        return

    # Combine the six independently-generated pieces into the same
    # markdown shape the rest of the app already expects.
    result_text = "\n\n".join(
        f"## {label}\n{texts[key]}" for key, label, _words in SECTIONS if texts.get(key)
    )

    # New topic: drop the old flashcards / quiz / etc., then store the new material.
    reset_generated_content()
    st.session_state.study_material = {
        "subject": subject,
        "topic": topic,
        "content": result_text,
    }
    _save_to_database(subject, topic, result_text)
    record_topic(topic)  # shows up in Learning Memory -> "Recently studied"

    message = "✅ Study material generated!"
    if was_truncated:
        message += (
            f" Note: only the first {MAX_SOURCE_CHARS:,} characters of your file were used. "
            "Upload a shorter section to cover the rest."
        )
    st.session_state.flash_message = message
    go_to_page("📖 Study Material")
    st.rerun()


def show_generate_form(subject_list, go_to_page):
    """The 'Generate Study Material' card at the top of the Dashboard."""
    with st.container(border=True):
        st.markdown(
            '<div class="sf-card-title">✨ Generate Study Material</div>'
            '<p class="sf-section-sub">Choose a subject, enter a topic (or upload a chapter) '
            "and we will build your notes on the next page.</p>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader(
                "📤 Upload Chapter (PDF, DOCX, or TXT) - optional",
                type=["pdf", "docx", "txt"],
                key="sm_upload",
                help=f"Only the first {MAX_SOURCE_CHARS:,} characters are used, to keep generation fast.",
            )
        with col2:
            subject = st.selectbox("📘 Select Subject", subject_list, key="sm_subject")
            topic = st.text_input(
                "✏️ Enter Topic / Chapter Name",
                placeholder="e.g. Chapter 3 - Photosynthesis",
                key="sm_topic",
            )
        clicked = st.button("✨ Generate Study Material", type="primary", key="sm_generate")

    if clicked:
        if not topic.strip():
            st.warning("Please enter a topic or chapter name first.")
        else:
            _generate(subject, topic.strip(), uploaded_file, go_to_page)


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------
def _render_material(material):
    topic = escape(str(material.get("topic", "")))
    subject = escape(str(material.get("subject", "")))
    st.markdown(
        f'<div class="sf-welcome-title">📚 {topic}</div>'
        f'<p class="sf-welcome-sub">{subject}</p>',
        unsafe_allow_html=True,
    )

    content = str(material.get("content", ""))
    sections = parse_sections(content)
    intro = sections.pop("_intro", "")

    # Expect all six; if fewer than four were recognised, fall back to
    # showing everything as-is rather than a half-empty set of expanders.
    if len(sections) >= 4:
        word_count = len(content.split())
        read_minutes = max(1, round(word_count / 200))
        card1, card2, card3 = st.columns(3)
        with card1:
            show_info_card("Sections", f"{len(sections)}/6", "🗂️")
        with card2:
            show_info_card("Words", f"{word_count:,}", "🔤")
        with card3:
            show_info_card("Reading time", f"~{read_minutes} min", "⏱️")

        if _sections_look_duplicated(sections):
            st.warning(
                "Two or more sections still look very similar to each other. "
                "You can regenerate the study material for more distinct sections."
            )

        expand_all = st.toggle("Expand all sections", key="sm_expand_all")

        if intro:
            with st.container(border=True):
                st.markdown(intro)
        for key, label, _words in SECTIONS:
            text = sections.get(key)
            if text:
                with st.expander(label, expanded=expand_all or key in OPEN_BY_DEFAULT):
                    st.markdown(text)
    else:
        # Something went wrong upstream: still show everything generated.
        with st.container(border=True):
            st.markdown(content)

    file_stem = re.sub(r"[^\w\-]+", "_", str(material.get("topic", ""))).strip("_") or "study_material"
    st.download_button(
        "⬇️ Download as Markdown",
        data=f"# {material.get('topic', '')} ({material.get('subject', '')})\n\n{content}",
        file_name=f"{file_stem}.md",
        mime="text/markdown",
        key="sm_download",
    )


def _start_quiz(go_to_page):
    """
    "Take Quiz" button callback: flag that a fresh quiz must be generated
    automatically as soon as the Quiz page loads, then navigate there. This
    is the ONLY way quiz_autogenerate gets set - see steps/step3_quiz.py,
    which reads and clears this flag once at the top of show_quiz().
    """
    st.session_state["quiz_autogenerate"] = True
    go_to_page("❓ Quiz")


def _render_take_quiz(go_to_page):
    """
    Single, clear next step after reading the material: take the quiz.
    Clicking this button both navigates to the Quiz page AND starts quiz
    generation automatically - there is no separate "Generate Quiz" button.
    """
    st.markdown("---")
    st.markdown(
        '<div class="sf-section-title">📝 Ready to test what you learned?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sf-section-sub">Take a 5-question quiz on this material. '
        "Score 75% or higher to unlock Flashcards.</p>",
        unsafe_allow_html=True,
    )
    st.button(
        "📝 Take Quiz",
        key="sm_take_quiz",
        type="primary",
        on_click=_start_quiz,
        args=(go_to_page,),
        use_container_width=True,
    )


def show_study_flow(subject_list, go_to_page):
    """Entry point used by app.py for the 'Study Material' page."""
    flash = st.session_state.pop("flash_message", None)
    if flash:
        st.success(flash)

    material = st.session_state.get("study_material")
    if isinstance(material, dict) and material.get("content"):
        _render_material(material)
        _render_take_quiz(go_to_page)
        return

    st.markdown(
        '<div class="sf-empty">No study material yet. '
        "Generate it from the Dashboard first.</div>",
        unsafe_allow_html=True,
    )
    st.button("Go to Dashboard", key="sm_empty_dashboard", type="primary",
              on_click=go_to_page, args=("🏠 Dashboard",))