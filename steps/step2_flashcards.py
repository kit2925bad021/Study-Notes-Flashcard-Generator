"""
Step 2 — Flashcards Page
--------------------------
Generates flashcards (via Ollama) from the study material that was
already created on the Dashboard, and lets the student review them
one card at a time.

VISUAL THEME (UI/UX only — no generation logic below is touched):
The whole page uses Mentora AI's Soft Lavender theme (light lavender
background, white content areas, violet accents), and the Flashcards
themselves get a special large, colorful, click-to-flip card design.
See _inject_theme_css() and _render_flashcard() for all of the styling.

FLOW (this revision): Flashcards now "automatically open" the moment the
student arrives here unlocked with no deck yet - the very first
generation happens on page load (once per unlock, via
flashcard_auto_generated), instead of waiting for a "Generate Flashcards"
click. The manual Generate/Regenerate button is still there for a fresh
deck afterwards. Once every card has been marked "✓ I Know It" or
"× Need Review" (see _advance_or_complete), the page switches to a
completion screen - a "🎓 Flashcards Complete!" card with one random line
from COMPLETION_QUOTES, a known/review count, and a single "⬅ Back to
Dashboard" button that navigates to "🏠 Dashboard". Nothing about
flashcard generation, deduplication, or the flip-card itself changed.

Imported in app.py with:
    from steps.step2_flashcards import show_flashcards
"""

import html
import random
import re

import streamlit as st

from services.ollama_service import generate_content
from steps.step13_memory import record_topic
from steps.step19_ui import show_topic_bar

# ----------------------------------------------------------------------
# Duplicate-question detection
# ----------------------------------------------------------------------
# How many flashcards we ultimately want on screen. Kept as one constant so
# the initial prompt and the "top-up" logic below always agree on the number.
TARGET_CARD_COUNT = 10

# How many times we're willing to go back to Ollama for a single click of
# Generate/Regenerate: 1 initial request + up to 2 "top-up" requests for
# whatever is still missing after duplicates are removed. This keeps
# requirement (12) — Ollama only runs on an explicit click, never on a
# plain Streamlit rerun — while still giving duplicates a couple of chances
# to be replaced.
MAX_GENERATION_ATTEMPTS = 3

# One short line shown at random on the "you finished all the flashcards"
# completion screen - never the same message every time, and never tied
# to a specific subject/topic (these are generic, on purpose).
COMPLETION_QUOTES = [
    "Every card you flip is a little more locked into memory. Nicely done!",
    "That's a full deck reviewed - your recall just got a real workout.",
    "Consistency beats cramming, and you just proved it. Great session!",
    "You made it through every card - that's how mastery actually happens.",
    "Small, steady reps like this are exactly what turns study into memory.",
]

# Small, generic words that don't carry the "concept" of a question, so two
# questions that only differ in phrasing (not in the fact being tested)
# still end up with the same keyword set. This is intentionally simple
# (no external NLP library) but is enough to catch things like
# "What is photosynthesis?" vs "Can you explain what photosynthesis is?".
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "explain", "describe", "define", "state", "give", "tell", "name",
    "you", "me", "please", "briefly", "in", "on", "at", "for", "to", "of",
    "and", "or", "but", "with", "about", "it", "its", "this", "that",
    "these", "those", "does", "means", "meaning", "mean", "term", "concept",
}


def _normalize_question(text):
    """
    Lowercase, strip punctuation, and collapse whitespace so two questions
    that differ only in casing/punctuation/spacing compare equal.
    """
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)  # drop punctuation like ? , . '
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _question_keywords(question):
    """
    Reduce a question to the set of "meaningful" words it's actually
    testing, ignoring question words and filler. Two differently-worded
    questions about the same fact end up with the same (or near-identical)
    keyword set.
    """
    normalized = _normalize_question(question)
    return {word for word in normalized.split() if word not in _STOPWORDS and len(word) > 2}


def _is_near_duplicate(keywords_a, keywords_b, threshold=0.6):
    """
    True if two keyword sets are similar enough to be considered the same
    underlying question (Jaccard similarity), not just an exact string
    match. This is what catches reworded duplicates.
    """
    if not keywords_a or not keywords_b:
        return keywords_a == keywords_b
    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return (len(intersection) / len(union)) >= threshold


def _is_valid_card(question, answer):
    """
    Basic sanity check that the answer actually answers the question,
    rather than being empty, a copy of the question, or a one-word stub.
    """
    if not question or not answer:
        return False
    if len(answer.split()) < 2:
        return False
    # An answer that's just the question restated isn't a real answer.
    if _normalize_question(answer) == _normalize_question(question):
        return False
    return True


def _dedupe_flashcards(flashcards):
    """
    Walk the flashcards in order and keep only the first occurrence of each
    question, dropping exact duplicates AND near-identical reworded
    duplicates. Also drops any card that fails the answer sanity check.
    """
    unique_cards = []
    seen_keyword_sets = []

    for card in flashcards:
        question = (card.get("question") or "").strip()
        answer = (card.get("answer") or "").strip()

        if not _is_valid_card(question, answer):
            continue

        keywords = _question_keywords(question)
        if any(_is_near_duplicate(keywords, seen) for seen in seen_keyword_sets):
            continue  # duplicate (exact or reworded) — keep the first one only

        unique_cards.append({"question": question, "answer": answer})
        seen_keyword_sets.append(keywords)

    return unique_cards


def _build_flashcard_prompt(subject, topic, content, count=TARGET_CARD_COUNT):
    """Build the prompt sent to Ollama, based on the existing study material."""
    numbered_format = "\n".join(f"Q{i}: <question>\nA{i}: <answer>" for i in range(1, count + 1))
    return f"""
You are a helpful study assistant for college students.

Here is study material that was already generated for this student:

Subject: {subject}
Topic/Chapter: {topic}

Study material:
{content}

Based ONLY on the study material above, create exactly {count} flashcards
to help the student revise. Keep the language simple and beginner
friendly.

IMPORTANT: every flashcard must test a DIFFERENT fact or concept from the
study material. Do not create two flashcards that ask about the same idea
using different wording — each question must be unique in what it tests,
and each answer must directly and correctly answer its own question.

Use this EXACT format for every flashcard (no extra text before or after):

{numbered_format}
"""


def _parse_flashcards(text):
    """
    Turn Ollama's "Q1: ... A1: ..." style output into a list of
    {"question": ..., "answer": ...} dictionaries. Tolerant of extra
    spacing, bold markers, or slightly different punctuation.
    """
    # Remove markdown bold markers so matching is simpler
    cleaned_text = text.replace("*", "")

    pattern = re.compile(
        r"Q\s*\d+\s*[:.\-]?\s*(.*?)\s*A\s*\d+\s*[:.\-]?\s*(.*?)(?=(?:Q\s*\d+\s*[:.\-]?)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    flashcards = []
    for question, answer in pattern.findall(cleaned_text):
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            flashcards.append({"question": question, "answer": answer})

    return flashcards


def _build_topup_prompt(subject, topic, content, existing_questions, needed_count):
    """
    Prompt used to replace duplicates that got removed. Lists the questions
    already kept so Ollama is steered toward NEW concepts instead of
    rewording facts that are already covered.
    """
    already_covered = "\n".join(f"- {q}" for q in existing_questions) or "(none yet)"
    numbered_format = "\n".join(f"Q{i}: <question>\nA{i}: <answer>" for i in range(1, needed_count + 1))
    return f"""
You are a helpful study assistant for college students.

Here is study material that was already generated for this student:

Subject: {subject}
Topic/Chapter: {topic}

Study material:
{content}

The student already has flashcards for these questions. Do NOT repeat any
of them, and do NOT ask a differently-worded question about the same fact:

{already_covered}

Create exactly {needed_count} NEW flashcards that each test a DIFFERENT
fact or concept from the study material that is not already covered above.
Keep the language simple and beginner friendly, and make sure each answer
directly and correctly answers its own question. Use this EXACT format for
every flashcard (no extra text before or after):

{numbered_format}
"""


def _generate_flashcards(subject, topic, content, target_count=TARGET_CARD_COUNT):
    """
    Call Ollama, then deduplicate (exact and reworded-duplicate questions).
    If duplicates left us short of target_count, ask Ollama for just the
    missing cards — up to MAX_GENERATION_ATTEMPTS total requests — instead
    of regenerating the whole set. Raises on failure of the first request;
    failures on top-up requests are logged and simply leave the set a bit
    smaller rather than breaking the page.
    """
    prompt = _build_flashcard_prompt(subject, topic, content, count=target_count)
    result_text = generate_content(prompt)
    flashcards = _dedupe_flashcards(_parse_flashcards(result_text))

    attempts = 1
    while len(flashcards) < target_count and attempts < MAX_GENERATION_ATTEMPTS:
        attempts += 1
        needed = target_count - len(flashcards)
        existing_questions = [card["question"] for card in flashcards]

        try:
            topup_prompt = _build_topup_prompt(subject, topic, content, existing_questions, needed)
            topup_text = generate_content(topup_prompt)
        except Exception as e:
            print(f"[flashcards] Top-up attempt {attempts} failed: {e}")
            break

        topup_cards = _parse_flashcards(topup_text)
        # Re-dedupe the combined list so the top-up cards are also checked
        # against each other and against everything kept so far.
        flashcards = _dedupe_flashcards(flashcards + topup_cards)

    return flashcards[:target_count]


def _save_flashcards_to_database(topic, flashcards):
    """Keep the generated cards in SQLite when a student is signed in (never blocks the page)."""
    student_id = st.session_state.get("student_id")
    if not student_id:
        return
    try:
        from database.database import save_flashcard

        for card in flashcards:
            save_flashcard(student_id, topic, card["question"], card["answer"])
    except Exception as e:
        print(f"[flashcards] Could not save flashcards: {e}")


# ----------------------------------------------------------------------
# Soft Lavender theme (UI only)
# ----------------------------------------------------------------------
# Applied while the Flashcards page is open. Keeps the overall app clean
# and professional (light lavender background, white panels, violet
# accents) while leaving the special colorful treatment to the flashcard
# itself, rendered separately in _render_flashcard() below.

_THEME_CSS = """
<style>
:root {
    --mentora-bg: #F3EEFF;
    --mentora-violet: #8B5CF6;
    --mentora-lavender: #A78BFA;
    --mentora-white: #FFFFFF;
    --mentora-ink: #2D2342;
    --mentora-muted: #6B5B95;
}

.stApp {
    background: var(--mentora-bg);
}

section[data-testid="stSidebar"] {
    background: var(--mentora-white);
    border-right: 1px solid rgba(139, 92, 246, 0.12);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--mentora-ink);
}

.fc-page-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--mentora-ink);
    margin: 0 0 0.15rem 0;
}

.fc-page-subtitle {
    font-size: 1.02rem;
    color: var(--mentora-muted);
    margin: 0 0 1.1rem 0;
}

div[data-testid="stButton"] > button,
.stButton > button {
    background: var(--mentora-white);
    color: var(--mentora-ink);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 12px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(139, 92, 246, 0.10);
    transition: all 0.18s ease;
}

div[data-testid="stButton"] > button:hover,
.stButton > button:hover {
    background: var(--mentora-violet);
    color: var(--mentora-white);
    border-color: var(--mentora-violet);
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.28);
}

div[data-testid="stButton"] > button:disabled,
.stButton > button:disabled {
    opacity: 0.45;
}

div[data-testid="stButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--mentora-violet), var(--mentora-lavender));
    color: var(--mentora-white);
    border: none;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stProgress"] div[role="progressbar"] > div {
    background: linear-gradient(90deg, var(--mentora-violet), var(--mentora-lavender));
}

.fc-locked {
    background: var(--mentora-white);
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.12);
    padding: 48px 32px;
    text-align: center;
    max-width: 560px;
    margin: 32px auto;
}

.fc-locked-icon {
    font-size: 2.6rem;
    margin-bottom: 10px;
}

.fc-locked-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--mentora-ink);
    margin-bottom: 8px;
}

.fc-locked-text {
    font-size: 1rem;
    color: var(--mentora-muted);
}

.fc-complete {
    background: var(--mentora-white);
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.12);
    padding: 48px 32px;
    text-align: center;
    max-width: 560px;
    margin: 32px auto;
}

.fc-complete-icon {
    font-size: 2.6rem;
    margin-bottom: 10px;
}

.fc-complete-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--mentora-ink);
    margin-bottom: 8px;
}

.fc-complete-quote {
    font-size: 1.05rem;
    font-style: italic;
    color: var(--mentora-muted);
    margin-bottom: 14px;
}

.fc-complete-stats {
    font-size: 0.95rem;
    color: var(--mentora-muted);
    margin-bottom: 4px;
}

.fc-complete-text {
    font-size: 1.05rem;
    color: var(--mentora-muted);
    margin-bottom: 4px;
}

.fc-progress-label {
    font-weight: 700;
    color: var(--mentora-ink);
    margin-bottom: 0.35rem;
}
</style>
"""


def _inject_theme_css():
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# The flashcard itself: large, colorful, click-to-flip
# ----------------------------------------------------------------------
# Rendered inside a self-contained HTML/CSS/JS document (via st.iframe,
# which embeds the HTML string directly via srcdoc - the same underlying
# mechanism the now-deprecated streamlit.components.v1.html used) so the
# 3D flip can run smoothly, purely client-side, without needing a
# Streamlit rerun to change the flip state.
# The question/answer text is HTML-escaped before insertion — it is never
# regenerated or modified here, only displayed exactly as already stored
# in st.session_state.flashcards.

_FLASHCARD_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
    html, body {
        margin: 0;
        padding: 0;
        background: transparent;
        overflow: hidden;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .fc-scene {
        position: relative;
        width: 100%;
        min-height: 520px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
        padding: 24px 10px 10px 10px;
    }
    .fc-decor {
        position: absolute;
        pointer-events: none;
        z-index: 0;
    }
    .fc-blob {
        border-radius: 50%;
        filter: blur(1px);
    }
    .fc-card {
        position: relative;
        z-index: 1;
        width: min(720px, 92vw);
        height: 400px;
        perspective: 1600px;
        cursor: pointer;
        outline: none;
    }
    .fc-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.7s cubic-bezier(0.4, 0.2, 0.2, 1);
        transform-style: preserve-3d;
    }
    .fc-card.flipped .fc-card-inner {
        transform: rotateY(180deg);
    }
    .fc-face {
        position: absolute;
        inset: 0;
        backface-visibility: hidden;
        -webkit-backface-visibility: hidden;
        border-radius: 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 36px 44px;
        box-sizing: border-box;
        box-shadow: 0 20px 45px rgba(139, 92, 246, 0.22), 0 2px 8px rgba(45, 35, 66, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.18);
    }
    .fc-front {
        background: linear-gradient(135deg, #FFFFFF 0%, #F3EEFF 45%, #E9D5FF 100%);
    }
    .fc-back {
        background: linear-gradient(135deg, #F5F0FF 0%, #E9D5FF 50%, #FBCFE8 100%);
        transform: rotateY(180deg);
    }
    .fc-label {
        font-size: 0.85rem;
        letter-spacing: 0.14em;
        font-weight: 700;
        color: #8B5CF6;
        margin-bottom: 16px;
        text-transform: uppercase;
    }
    .fc-content {
        font-size: clamp(1.05rem, 1.4vw + 0.6rem, 1.5rem);
        font-weight: 600;
        color: #2D2342;
        line-height: 1.55;
        max-height: 250px;
        overflow-y: auto;
    }
    .fc-hint {
        position: absolute;
        bottom: 16px;
        font-size: 0.75rem;
        color: #7C6FA6;
        opacity: 0.85;
    }
    @media (max-width: 520px) {
        .fc-card { height: 330px; }
        .fc-face { padding: 24px 20px; border-radius: 20px; }
        .fc-label { font-size: 0.72rem; margin-bottom: 10px; }
        .fc-hint { font-size: 0.68rem; bottom: 10px; }
    }
</style>
</head>
<body>
    <div class="fc-scene">
        <div class="fc-decor" style="top:6%; left:4%; font-size:1.6rem; opacity:0.7;">&#10024;</div>
        <div class="fc-decor" style="top:12%; right:6%; font-size:1.15rem; opacity:0.6;">&#10024;</div>
        <div class="fc-decor" style="bottom:12%; left:9%; font-size:1.3rem; opacity:0.6;">&#10024;</div>
        <div class="fc-decor" style="bottom:8%; right:12%; font-size:1.1rem; opacity:0.55;">&#11088;</div>
        <div class="fc-decor fc-blob" style="width:140px; height:140px; top:-30px; left:-40px; background: radial-gradient(circle, rgba(167,139,250,0.35), transparent 70%);"></div>
        <div class="fc-decor fc-blob" style="width:170px; height:170px; bottom:-40px; right:-50px; background: radial-gradient(circle, rgba(249,168,212,0.30), transparent 70%);"></div>
        <div class="fc-decor" style="top:42%; left:1%; width:10px; height:10px; border-radius:50%; background:#8B5CF6; opacity:0.45;"></div>
        <div class="fc-decor" style="top:55%; right:2%; width:14px; height:14px; border-radius:50%; background:#F9A8D4; opacity:0.45;"></div>

        <div class="fc-card" id="fcCard" tabindex="0" role="button"
             aria-label="Flashcard. Click or press Enter to flip."
             onclick="this.classList.toggle('flipped')"
             onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault(); this.classList.toggle('flipped');}">
            <div class="fc-card-inner">
                <div class="fc-face fc-front">
                    <div class="fc-label">Question</div>
                    <div class="fc-content">__QUESTION__</div>
                    <div class="fc-hint">Click to reveal the answer</div>
                </div>
                <div class="fc-face fc-back">
                    <div class="fc-label">Answer</div>
                    <div class="fc-content">__ANSWER__</div>
                    <div class="fc-hint">Click to flip back</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


def _render_flashcard(question, answer):
    """
    Render the current card's existing question/answer (never regenerated
    or modified here) as one large, click-to-flip 3D card.
    """
    rendered_html = (
        _FLASHCARD_HTML_TEMPLATE
        .replace("__QUESTION__", html.escape(question))
        .replace("__ANSWER__", html.escape(answer))
    )
    # st.iframe replaces the deprecated components.html(...) call above.
    # rendered_html always contains "<" (it starts with "<!doctype html>"),
    # so st.iframe treats it as a raw HTML string and embeds it via srcdoc -
    # exactly like components.html did - rather than as a URL or file path.
    # height=580 is passed through unchanged (a fixed pixel height, same as
    # before); width is left at st.iframe's default ("stretch"), which
    # matches components.html's old default of the app's element width.
    # st.iframe has no "scrolling" parameter (it always allows a scrollbar
    # if content overflows, instead of the old scrolling=False/"never show
    # one"), but the template's own <html>/<body> already set
    # `overflow: hidden` and every face fits inside the fixed 400px card /
    # 520px scene, so no scrollbar was or is ever actually triggered here.
    st.iframe(rendered_html, height=580)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

def _advance_or_complete(current_index, total_cards):
    """
    After marking the current card "known" or "review": if every card
    (not just this one) now has a mark, the deck is done - switch to the
    completion screen (picking this deck's one completion quote now, so
    it stays fixed across reruns) instead of moving to a next card that
    may not exist. Otherwise, advance to the next card as before.
    """
    if len(st.session_state.flashcard_status) >= total_cards:
        st.session_state.flashcard_completed = True
        st.session_state.flashcard_quote = random.choice(COMPLETION_QUOTES)
    elif current_index < total_cards - 1:
        st.session_state.flashcard_index += 1


def show_flashcards():
    """Display the full Flashcards page."""
    _inject_theme_css()

    st.markdown(
        "<div class='fc-page-title'>🃏 Flashcards</div>"
        "<div class='fc-page-subtitle'>Flip, recall, and strengthen your memory.</div>",
        unsafe_allow_html=True,
    )

    # ---- Access control: Flashcards stay locked until Practice has been
    # completed with a score of at least 4/5. This reuses the unlock flag
    # Practice itself sets (st.session_state["flashcards_unlocked"]) —
    # no second unlock mechanism is introduced here. Nothing below this
    # is generated or displayed, and no Ollama call happens, until
    # unlocked. ----
    if not st.session_state.get("flashcards_unlocked", False):
        st.markdown(
            """
            <div class="fc-locked">
                <div class="fc-locked-icon">🔒</div>
                <div class="fc-locked-title">Flashcards Locked</div>
                <div class="fc-locked-text">Complete Practice with at least 4/5 to unlock your flashcards.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    study_material = st.session_state.get("study_material")

    if study_material is None:
        st.info("No study material generated yet. Please generate study material from the Dashboard first.")
        return

    subject = study_material["subject"]
    topic = study_material["topic"]
    content = study_material["content"]

    # ---- Subject and topic display ----
    show_topic_bar(subject, topic)

    st.markdown("---")

    # ---- Session state for this page ----
    # flashcards: the list of {"question", "answer"} dicts
    # flashcard_index: which card is currently shown
    # flashcard_status: optional, ephemeral "known"/"review" marks per card
    # index (UI-only — no new database or persistent progress system)
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = None
    if "flashcard_index" not in st.session_state:
        st.session_state.flashcard_index = 0
    if "flashcard_status" not in st.session_state:
        st.session_state.flashcard_status = {}
    # flashcard_completed: True once every card has been marked "known" or
    # "review" - this is what switches the page over to the completion
    # screen (see below). flashcard_quote: the one completion line chosen
    # for THIS completed deck - picked once, when flashcard_completed
    # first becomes True, so it doesn't change on every rerun.
    if "flashcard_completed" not in st.session_state:
        st.session_state.flashcard_completed = False
    if "flashcard_quote" not in st.session_state:
        st.session_state.flashcard_quote = None
    # flashcard_auto_generated: this page auto-generates the deck the
    # very first time it's opened unlocked with nothing loaded yet - no
    # click required, so Flashcards are "automatically open" right after
    # Practice unlocks them. This flag makes sure that only ever happens
    # ONCE per unlock; if it fails, the student uses the manual
    # Generate/Regenerate button below instead of retrying on every rerun.
    if "flashcard_auto_generated" not in st.session_state:
        st.session_state.flashcard_auto_generated = False

    if st.session_state.flashcards is None and not st.session_state.flashcard_auto_generated:
        st.session_state.flashcard_auto_generated = True
        with st.spinner("Creating your flashcards from the study material..."):
            try:
                auto_flashcards = _generate_flashcards(subject, topic, content, target_count=TARGET_CARD_COUNT)
            except Exception as e:
                auto_flashcards = None
                st.session_state.flashcard_auto_error = str(e)
            if auto_flashcards:
                st.session_state.flashcards = auto_flashcards
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_status = {}
                st.session_state.flashcard_completed = False
                st.session_state.flashcard_quote = None
                _save_flashcards_to_database(topic, auto_flashcards)
                record_topic(topic)
        st.rerun()

    # ---- Buttons to generate / regenerate flashcards ----
    # One button: "Generate" first (only reachable if the automatic
    # generation above failed), "Regenerate" once cards exist.
    has_cards = bool(st.session_state.flashcards)
    generate_clicked = st.button(
        "🔄 Regenerate Flashcards" if has_cards else "✨ Generate Flashcards",
        type="secondary" if has_cards else "primary",
    )

    if generate_clicked:
        with st.spinner("Creating flashcards from your study material..."):
            try:
                new_flashcards = _generate_flashcards(subject, topic, content, target_count=TARGET_CARD_COUNT)
                if not new_flashcards:
                    st.error("⚠️ Ollama did not return any flashcards. Please try again.")
                else:
                    st.session_state.flashcards = new_flashcards
                    st.session_state.flashcard_index = 0
                    st.session_state.flashcard_status = {}
                    st.session_state.flashcard_completed = False
                    st.session_state.flashcard_quote = None
                    _save_flashcards_to_database(topic, new_flashcards)
                    record_topic(topic)
                    if len(new_flashcards) < TARGET_CARD_COUNT:
                        st.success(
                            f"✅ Flashcards generated! ({len(new_flashcards)} unique cards after "
                            "removing duplicate/reworded questions.)"
                        )
                    else:
                        st.success("✅ Flashcards generated!")
            except Exception as e:
                st.error(f"⚠️ Flashcard generation failed: {e}")

    st.markdown("---")

    # ---- Display the flashcards, if any exist ----
    flashcards = st.session_state.flashcards

    if not flashcards:
        error_detail = st.session_state.get("flashcard_auto_error")
        if error_detail:
            st.error(f"⚠️ Flashcard generation failed: {error_detail}")
        else:
            st.info("Click '✨ Generate Flashcards' to create flashcards from your study material.")
        return

    # ---- Completion screen: shown once every card has been marked
    # "known" or "review" (see the I Know It / Need Review handlers
    # below), instead of the card/progress UI. ----
    if st.session_state.flashcard_completed:
        st.markdown(
            """
            <div class="fc-complete">
                <div class="fc-complete-icon">🎉</div>
                <div class="fc-complete-title">Flashcards Completed!</div>
                <div class="fc-complete-text">You completed all the flashcards for this study session.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Review: leave the completion screen and go back through the SAME
        # deck from card 1. Only the temporary completion/progress UI state
        # is cleared (so marking cards can complete the deck again) - the
        # deck itself is kept, nothing is regenerated and Ollama is not
        # called.
        if st.button("Review", key="fc_review_again"):
            st.session_state.flashcard_completed = False
            st.session_state.flashcard_quote = None
            st.session_state.flashcard_status = {}
            st.session_state.flashcard_index = 0
            st.rerun()

        if st.button("🏠 Back to Dashboard", type="primary", key="fc_back_to_dashboard"):
            # Reuses the app's existing page-navigation mechanism - the same
            # current_page + st.rerun() pattern Practice's "Open Flashcards"
            # button already uses (see step21_practice.py). No second
            # navigation system, no intermediate screen, and no extra click:
            # the Dashboard renders on the very next run.
            #
            # Only the TEMPORARY flashcard-completion UI state is cleared
            # below, so a later visit to Flashcards starts this deck over
            # from card 1 instead of reopening on the completion screen.
            # Nothing else is touched: st.session_state.flashcards (the
            # actual deck) and flashcard_auto_generated are left as-is, so
            # no new deck is generated and Ollama is not called; study
            # material, saved flashcard history in the database, and
            # authentication/session/login state are completely untouched.
            st.session_state.flashcard_completed = False
            st.session_state.flashcard_quote = None
            st.session_state.flashcard_status = {}
            st.session_state.flashcard_index = 0
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
        return

    total_cards = len(flashcards)

    # Keep the current index within a valid range
    if st.session_state.flashcard_index >= total_cards:
        st.session_state.flashcard_index = 0
    if st.session_state.flashcard_index < 0:
        st.session_state.flashcard_index = 0

    current_index = st.session_state.flashcard_index
    current_card = flashcards[current_index]

    # ---- Progress indicator ----
    st.markdown(
        f"<div class='fc-progress-label'>Card {current_index + 1} of {total_cards}</div>",
        unsafe_allow_html=True,
    )
    st.progress((current_index + 1) / total_cards)

    # ---- The flashcard itself: large, colorful, click-to-flip ----
    _render_flashcard(current_card["question"], current_card["answer"])

    st.markdown("")

    # ---- Controls: I Know It / Need Review, then Previous / Next ----
    know_col, review_col = st.columns(2)

    with know_col:
        if st.button("✓ I Know It", use_container_width=True):
            st.session_state.flashcard_status[current_index] = "known"
            _advance_or_complete(current_index, total_cards)
            st.rerun()

    with review_col:
        if st.button("× Need Review", use_container_width=True):
            st.session_state.flashcard_status[current_index] = "review"
            _advance_or_complete(current_index, total_cards)
            st.rerun()

    st.markdown("")

    prev_col, next_col = st.columns(2)

    with prev_col:
        if st.button("← Previous", disabled=(current_index == 0), use_container_width=True):
            st.session_state.flashcard_index -= 1
            st.rerun()

    with next_col:
        if st.button("Next →", type="primary", use_container_width=True):
            if current_index >= total_cards - 1:
                # Past the final flashcard: show the completion screen.
                st.session_state.flashcard_completed = True
                st.session_state.flashcard_quote = random.choice(COMPLETION_QUOTES)
            else:
                st.session_state.flashcard_index += 1
            st.rerun()