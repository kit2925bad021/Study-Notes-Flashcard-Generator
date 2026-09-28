"""
steps/step19_ui.py

Step 19: Reusable UI styling and helper functions for the
Study Notes & Flashcard Generator (Mentora AI).

This module contains ONLY presentation helpers - no database calls, no AI
calls, no page/routing logic. (`require_study_material` only READS
st.session_state.study_material; it never changes it.)

Design idea: Mentora AI's "Soft Lavender" theme - a calm, light-lavender
background, white content cards, and violet accents, aimed at a clean,
modern, professional look for a college study app on a laptop screen.
No external fonts or network resources are loaded; only a local/system
font stack is used.

Soft Sky Blue theme (after login): the Soft Lavender CSS below is still
what the Login page (and any logged-out screen) uses, completely
unchanged. Once the user is logged in, apply_custom_css() injects one
extra, small "Soft Sky Blue" block AFTER it. That block does not create a
second theme system - it re-points the SAME --sf-* variables the existing
CSS already uses (so every page, card, chip, button, metric and progress
bar follows automatically) and only overrides the few values that were
hard-coded to lavender (button text/shadows, hero gradient, card shadows,
avatar text, progress fill) plus best-effort selectors for the sidebar
menu and Streamlit's native selected states. Logging out returns to the
lavender look. See _LOGIN_STATE_KEYS for how "logged in" is detected.
"""

from html import escape

import streamlit as st


# ---------------------------------------------------------------------------
# 1. Custom CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
    :root {
        --sf-bg: #F3EEFF;
        --sf-primary: #8B5CF6;
        --sf-primary-dark: #7C3AED;
        --sf-secondary: #A78BFA;
        --sf-ink: #2D2342;
        --sf-muted: rgba(45, 35, 66, 0.65);
        --sf-white: #FFFFFF;
        --sf-soft: #EDE9FE;
        --sf-soft-hover: #E4D9FC;
        --sf-border: #E9DDFB;
        --sf-success-bg: #DCFCE7;
        --sf-warning-bg: #FFF4E5;
        --sf-error-bg: #FEE2E2;
        --sf-secondary-text: #6D28D9;
        --sf-font: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }

    /* ---------- General layout ---------- */
    .stApp {
        background-color: var(--sf-bg);
        color: var(--sf-ink);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* ---------- Headings (no external font - local/system stack only) ---------- */
    h1, h2, h3,
    .sf-page-title, .sf-welcome-title, .sf-card-title, .sf-section-title,
    .sf-hero h1, .sf-profile-name {
        font-family: var(--sf-font);
        letter-spacing: -0.01em;
        color: var(--sf-ink);
    }
    h1, h2, h3 { font-weight: 700; }
    h1 { margin-bottom: 0.2rem; }

    /* ---------- Buttons ---------- */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-family: var(--sf-font);
        border: none;
        transition: background-color 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
    }

    /* Primary buttons: violet fill, white text */
    .stButton button[kind="primary"],
    .stButton button[data-testid="stBaseButton-primary"],
    .stFormSubmitButton button[kind="primary"],
    .stFormSubmitButton button[data-testid="stBaseButton-primary"] {
        background-color: var(--sf-primary);
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.25);
    }
    .stButton button[kind="primary"]:hover,
    .stButton button[data-testid="stBaseButton-primary"]:hover,
    .stFormSubmitButton button[kind="primary"]:hover,
    .stFormSubmitButton button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--sf-primary-dark);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.32);
    }

    /* Secondary buttons and the sidebar menu are styled in app.py.
       Here only a soft-lavender look for secondary buttons in the MAIN area
       is added, so the primary action stands out. The sidebar is excluded
       on purpose. */
    .stButton button[kind="secondary"]:not([data-testid="stSidebar"] *),
    .stButton button[data-testid="stBaseButton-secondary"]:not([data-testid="stSidebar"] *),
    .stDownloadButton button:not([data-testid="stSidebar"] *) {
        background-color: var(--sf-soft);
        color: var(--sf-secondary-text) !important;
        border: 1px solid var(--sf-border);
    }
    .stButton button[kind="secondary"]:not([data-testid="stSidebar"] *):hover,
    .stButton button[data-testid="stBaseButton-secondary"]:not([data-testid="stSidebar"] *):hover,
    .stDownloadButton button:not([data-testid="stSidebar"] *):hover {
        background-color: var(--sf-soft-hover);
        border-color: var(--sf-primary);
    }
    .stButton button:disabled {
        opacity: 0.5;
    }

    /* ---------- Progress bars, expanders, metrics ---------- */
    div[data-testid="stProgress"] div[role="progressbar"] > div,
    .stProgress > div > div > div > div {
        background-color: var(--sf-primary);
    }
    div[data-testid="stExpander"] {
        background-color: var(--sf-white);
        border: 1px solid var(--sf-border);
        border-radius: 12px;
    }
    div[data-testid="stMetric"] {
        background-color: var(--sf-white);
        border: 1px solid var(--sf-border);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.06);
    }

    /* ---------- Alerts (success / warning / error / info) ---------- */
    /* Streamlit's exact alert DOM varies by version, so a few likely
       selectors are covered here; the outer box always gets the calm,
       rounded Soft Lavender treatment. */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--sf-border);
    }
    div[data-testid="stAlert"][kind="success"],
    div[data-baseweb="notification"][kind="positive"] {
        background-color: var(--sf-success-bg) !important;
        border-color: transparent;
    }
    div[data-testid="stAlert"][kind="warning"],
    div[data-baseweb="notification"][kind="warning"] {
        background-color: var(--sf-warning-bg) !important;
        border-color: transparent;
    }
    div[data-testid="stAlert"][kind="error"],
    div[data-baseweb="notification"][kind="negative"] {
        background-color: var(--sf-error-bg) !important;
        border-color: transparent;
    }
    div[data-testid="stAlert"][kind="info"],
    div[data-baseweb="notification"][kind="info"] {
        background-color: var(--sf-soft) !important;
        border-color: transparent;
    }

    /* ---------- Header: clean white/lavender premium card ---------- */
    .sf-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F5FF 55%, #EDE9FE 100%);
        border: 1px solid var(--sf-border);
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(139, 92, 246, 0.10);
        padding: 1.3rem 1.6rem 1.4rem;
        margin-bottom: 1.3rem;
    }
    .sf-hero h1 {
        color: var(--sf-ink) !important;
        font-size: 1.85rem;
        margin: 0;
        padding: 0;
    }
    .sf-hero p {
        color: var(--sf-muted) !important;
        margin: 0.3rem 0 0;
        font-size: 1.02rem;
    }

    /* ---------- Page header (icon tile + title + subtitle) ---------- */
    .sf-page-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin: 0.2rem 0 1rem;
    }
    .sf-page-icon {
        width: 3rem;
        height: 3rem;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        border-radius: 10px;
        background-color: var(--sf-soft);
    }
    .sf-page-title { font-size: 1.6rem; font-weight: 700; line-height: 1.2; color: var(--sf-ink); }
    .sf-page-sub { color: var(--sf-muted); font-size: 0.95rem; }

    /* ---------- Subject / topic labels: soft lavender chips ---------- */
    .sf-chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.9rem; }
    .sf-chip {
        display: inline-block;
        padding: 0.12rem 0.6rem;
        border-radius: 6px;
        background-color: var(--sf-soft);
        border: 1px solid var(--sf-border);
        color: var(--sf-ink);
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* ---------- Cards ---------- */
    .app-card {
        background-color: var(--sf-white);
        border: 1px solid var(--sf-border);
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.08);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .info-card {
        background-color: var(--sf-white);
        border: 1px solid var(--sf-border);
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.08);
        padding: 0.9rem 1.1rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .info-card .info-icon { font-size: 1.5rem; margin-bottom: 0.1rem; }
    .info-card .info-value { font-size: 1.6rem; font-weight: 700; color: var(--sf-primary); }
    .info-card .info-title { font-size: 0.85rem; color: var(--sf-muted); margin-top: 0.1rem; }

    /* ---------- Section headers ---------- */
    .section-header { margin-top: 1.5rem; margin-bottom: 0.75rem; }
    .section-header h3 { margin-bottom: 0.1rem; }
    .section-header p { color: var(--sf-muted); margin-top: 0; }

    /* ---------- Classes used by the Study Material flow (step 20) ---------- */
    .sf-card-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.15rem; color: var(--sf-ink); }
    .sf-section-title { font-size: 1.25rem; font-weight: 700; margin: 1.6rem 0 0.1rem; color: var(--sf-ink); }
    .sf-section-sub { color: var(--sf-muted); margin: 0 0 0.8rem; }
    .sf-welcome-title { font-size: 1.8rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.1rem; color: var(--sf-ink); }
    .sf-welcome-sub { color: var(--sf-muted); margin-top: 0; }
    .sf-empty {
        text-align: center;
        padding: 1.6rem 1rem;
        background-color: var(--sf-white);
        border: 1px dashed var(--sf-border);
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* ---------- Profile card ---------- */
    .sf-profile-top { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.9rem; }
    .sf-avatar {
        width: 3.6rem;
        height: 3.6rem;
        flex-shrink: 0;
        border-radius: 50%;
        background-color: var(--sf-primary);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--sf-font);
        font-size: 1.5rem;
        font-weight: 700;
    }
    .sf-profile-name { font-size: 1.2rem; font-weight: 700; line-height: 1.2; color: var(--sf-ink); }
    .sf-profile-sub { color: var(--sf-muted); font-size: 0.9rem; }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 2rem 1rem;
        background-color: var(--sf-white);
        border: 1px dashed var(--sf-border);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .empty-state .empty-icon { font-size: 2.2rem; margin-bottom: 0.4rem; }
    .empty-state .empty-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.2rem; color: var(--sf-ink); }
    .empty-state .empty-message { color: var(--sf-muted); font-size: 0.9rem; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { border-right: 1px solid var(--sf-border); }

    /* ---------- Small screens ---------- */
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .sf-hero h1 { font-size: 1.45rem; }
        .sf-welcome-title { font-size: 1.45rem; }
    }
</style>
"""


# ---------------------------------------------------------------------------
# 1b. Soft Sky Blue theme - applied ONLY after a successful login
# ---------------------------------------------------------------------------
#
# Palette:  primary #A9D6E5 | secondary #89C2D9 | background #F0FAFD |
#           card #FFFFFF | text #243B53 | secondary text #526777 |
#           border #D6EEF5 | hover/active #61AFC8
# (#2B7A99 is a darker shade of the same blue used ONLY for small accent
# text on white, because #A9D6E5 / #61AFC8 are too light to read as text.)
#
# Design rule: mostly light background, white cards, soft-blue accents -
# never a strongly blue screen. Primary buttons are soft sky blue with DARK
# text (white on #A9D6E5 would be unreadable).

_SKY_BLUE_CSS = """
<style>
    /* Re-point the existing theme variables; everything that already uses
       var(--sf-...) switches automatically. Success/warning/error colours
       are intentionally NOT redefined, so alerts keep their meaning. */
    :root {
        --sf-bg: #F0FAFD;
        --sf-primary: #A9D6E5;
        --sf-primary-dark: #61AFC8;
        --sf-secondary: #89C2D9;
        --sf-ink: #243B53;
        --sf-muted: #526777;
        --sf-white: #FFFFFF;
        --sf-soft: #E4F4F9;
        --sf-soft-hover: #D6EEF5;
        --sf-border: #D6EEF5;
        --sf-secondary-text: #243B53;
        --sf-accent-text: #2B7A99;
    }

    /* ---------- Primary buttons: soft sky blue, dark readable text ---------- */
    .stButton button[kind="primary"],
    .stButton button[data-testid="stBaseButton-primary"],
    .stFormSubmitButton button[kind="primary"],
    .stFormSubmitButton button[data-testid="stBaseButton-primary"] {
        background-color: var(--sf-primary);
        color: var(--sf-ink) !important;
        box-shadow: 0 2px 8px rgba(97, 175, 200, 0.28);
    }
    .stButton button[kind="primary"] p,
    .stButton button[data-testid="stBaseButton-primary"] p,
    .stFormSubmitButton button[kind="primary"] p,
    .stFormSubmitButton button[data-testid="stBaseButton-primary"] p {
        color: var(--sf-ink) !important;
    }
    .stButton button[kind="primary"]:hover,
    .stButton button[data-testid="stBaseButton-primary"]:hover,
    .stFormSubmitButton button[kind="primary"]:hover,
    .stFormSubmitButton button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--sf-primary-dark);
        box-shadow: 0 4px 12px rgba(97, 175, 200, 0.38);
    }

    /* ---------- Progress indicators ---------- */
    div[data-testid="stProgress"] div[role="progressbar"] {
        background-color: var(--sf-border);
    }
    div[data-testid="stProgress"] div[role="progressbar"] > div,
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--sf-primary) 0%, var(--sf-primary-dark) 100%);
    }

    /* ---------- Hard-coded lavender values in the base CSS ---------- */
    div[data-testid="stMetric"] { box-shadow: 0 2px 8px rgba(97, 175, 200, 0.10); }
    .sf-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FAFD 55%, #D6EEF5 100%);
        border-top: 3px solid var(--sf-primary);
        box-shadow: 0 4px 18px rgba(97, 175, 200, 0.14);
    }
    .app-card,
    .info-card { box-shadow: 0 2px 10px rgba(97, 175, 200, 0.12); }
    .info-card .info-value { color: var(--sf-accent-text); }
    .sf-avatar { color: var(--sf-ink); }

    /* ---------- Links ---------- */
    .stMarkdown a { color: var(--sf-accent-text); }

    /* ---------- Sidebar (menu itself is styled in app.py, so these use
       !important and a specific prefix; best-effort for radio OR button menus) ---------- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: var(--sf-white) !important;
    }
    section[data-testid="stSidebar"] { border-right: 1px solid var(--sf-border); }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: var(--sf-soft-hover) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: var(--sf-primary) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
        color: var(--sf-ink) !important;
    }

    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {
        background-color: var(--sf-primary) !important;
        color: var(--sf-ink) !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
    section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--sf-primary-dark) !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"]:hover {
        background-color: var(--sf-soft-hover) !important;
        color: var(--sf-ink) !important;
    }

    /* ---------- Streamlit's native selected / focus states ----------
       (best-effort: these follow .streamlit/config.toml's primaryColor by
       default, which is global and would also recolour the Login page) */
    label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background-color: var(--sf-primary-dark) !important;
        border-color: var(--sf-primary-dark) !important;
    }
    label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {
        background-color: var(--sf-primary-dark) !important;
        border-color: var(--sf-primary-dark) !important;
    }
    div[data-baseweb="tab-highlight"] { background-color: var(--sf-primary-dark) !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--sf-ink) !important; }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--sf-primary-dark) !important;
        box-shadow: 0 0 0 1px var(--sf-primary-dark) !important;
    }
</style>
"""


# Session-state keys that mean "the user has logged in". The theme switches
# to Soft Sky Blue as soon as ANY of them holds a truthy value; while none
# does (the Login / Register screens), the original lavender look is kept.
# Widget-style keys such as "username" / "email" / "password" are left out
# on purpose: they are truthy while someone is still typing on the Login
# page. If your login stores its flag under a different name, add it here
# (or pass logged_in=... to apply_custom_css()).
_LOGIN_STATE_KEYS = (
    "logged_in",
    "is_logged_in",
    "authenticated",
    "is_authenticated",
    "user_id",
    "user",
    "current_user",
    "user_email",
)


def is_user_logged_in():
    """True once the login flow has stored a logged-in user in session state.
    Read-only - it never changes session state or the login logic."""
    return any(bool(st.session_state.get(key)) for key in _LOGIN_STATE_KEYS)


def apply_custom_css(logged_in=None):
    """
    Inject the app-wide CSS. Call once near the top of app.py (unchanged).

    Always injects the Soft Lavender CSS (the Login page look). When the
    user is logged in, the Soft Sky Blue block is injected right after it
    and overrides it. `logged_in` normally stays None (auto-detected via
    is_user_logged_in()); pass True/False only to force one theme.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    if logged_in is None:
        logged_in = is_user_logged_in()
    if logged_in:
        st.markdown(_SKY_BLUE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. App header
# ---------------------------------------------------------------------------

def show_app_header():
    """Display the app's main title and subtitle as the premium lavender header."""
    st.markdown(
        '<div class="sf-hero">'
        "<h1>📚 Study Notes &amp; Flashcard Generator</h1>"
        "<p>Turn your chapters into smarter study material</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 3. Page header, topic bar, section header
# ---------------------------------------------------------------------------

def show_page_header(icon, title, subtitle=None):
    """
    Title block used at the top of every page, so all pages look the same:

        show_page_header("❓", "Quiz", "Test yourself on this topic.")
    """
    subtitle_html = f'<div class="sf-page-sub">{escape(str(subtitle))}</div>' if subtitle else ""
    st.markdown(
        '<div class="sf-page-header">'
        f'<div class="sf-page-icon">{escape(str(icon))}</div>'
        f'<div><div class="sf-page-title">{escape(str(title))}</div>{subtitle_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def show_topic_bar(subject, topic):
    """Show the current subject and topic as two soft-lavender chip labels."""
    st.markdown(
        '<div class="sf-chip-row">'
        f'<span class="sf-chip">📘 {escape(str(subject))}</span>'
        f'<span class="sf-chip">📌 {escape(str(topic))}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def show_section_header(icon, title, description=None):
    """
    Reusable section header, e.g.:

        show_section_header("🧠", "Flashcards", "Review important concepts quickly.")
    """
    description_html = f"<p>{escape(str(description))}</p>" if description else ""
    st.markdown(
        '<div class="section-header">'
        f"<h3>{escape(str(icon))} {escape(str(title))}</h3>{description_html}"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 4. Info / metric card
# ---------------------------------------------------------------------------

def show_info_card(title, value, icon):
    """
    Reusable metric/info card, e.g.:

        show_info_card("Questions Answered", 24, "❓")
    """
    st.markdown(
        '<div class="info-card">'
        f'<div class="info-icon">{escape(str(icon))}</div>'
        f'<div class="info-value">{escape(str(value))}</div>'
        f'<div class="info-title">{escape(str(title))}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 5. Empty state
# ---------------------------------------------------------------------------

def show_empty_state(title, message, icon="📚"):
    """
    Friendly placeholder shown when there's no data yet, e.g.:

        show_empty_state("No flashcards yet", "Generate some flashcards to see them here.")
    """
    st.markdown(
        '<div class="empty-state">'
        f'<div class="empty-icon">{escape(str(icon))}</div>'
        f'<div class="empty-title">{escape(str(title))}</div>'
        f'<div class="empty-message">{escape(str(message))}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def require_study_material():
    """
    Return st.session_state.study_material if it exists, otherwise show a
    friendly empty state and return None. Pages use it like this:

        material = require_study_material()
        if material is None:
            return
    """
    material = st.session_state.get("study_material")
    if not isinstance(material, dict) or not material.get("content"):
        show_empty_state(
            "No study material yet",
            "Generate study material from the Dashboard first, then come back here.",
            "📖",
        )
        return None
    return material


# ---------------------------------------------------------------------------
# 6. Success / error messages
# ---------------------------------------------------------------------------

def show_success_message(message):
    """Consistent success message styling (wraps st.success)."""
    st.success(message)


def show_error_message(message):
    """Consistent error message styling (wraps st.error)."""
    st.error(message)