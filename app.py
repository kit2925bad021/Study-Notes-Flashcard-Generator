"""
Mentora (Study Notes & Flashcard Generator)
----------------------------------------------
Main Streamlit controller.

This file only handles: page setup, styling, the Login / Register pages
(accounts are stored in SQLite),
sidebar navigation, session state, the Dashboard (including Gemini
generation is now on its own Study Material page), and routing to every step page. Each
step's actual UI/logic lives in its own module under steps/ and is imported below.

Run with: python -m streamlit run app.py
"""

import os  # used to check the login illustration file exists before displaying it
import hashlib  # password hashing for the login
import hmac  # constant-time password comparison
import re  # name and email validation
import secrets  # random salt for each password
from html import escape  # used to safely show user text inside HTML cards
from pathlib import Path  # only used to locate the login page's illustration

import streamlit as st

# Gemini integration. app.py itself makes NO Gemini calls any more:
# Study Material is generated from the Dashboard form (steps/step20_study_flow.py,
# one streamed request) and shown on its own page and every other step module makes its own call
# only when the student opens that page and clicks its button.
from services.gemini_service import generate_content  # noqa: F401 (kept for compatibility)

# ----------------------------------------------------------------------
# SAFE STEP IMPORTS
# Each step is imported inside its own try/except so that one missing or
# broken module doesn't take down the rest of the app. If a module can't
# be imported, its page shows a friendly error instead of crashing.
# ----------------------------------------------------------------------

try:
    from steps.step1_study_material import show_study_material
    STEP1_AVAILABLE = True
except ImportError as e:
    STEP1_AVAILABLE = False
    print(f"Step 1 import error: {e}")

# New dedicated Study Material page (generation + sectioned display + next steps)
try:
    from steps.step20_study_flow import show_generate_form, show_study_flow
    STEP20_AVAILABLE = True
except ImportError as e:
    STEP20_AVAILABLE = False
    print(f"Step 20 import error: {e}")

try:
    from steps.step2_flashcards import show_flashcards
    STEP2_AVAILABLE = True
except ImportError as e:
    STEP2_AVAILABLE = False
    print(f"Step 2 import error: {e}")

try:
    from steps.step3_quiz import show_quiz
    STEP3_AVAILABLE = True
except ImportError as e:
    STEP3_AVAILABLE = False
    print(f"Step 3 import error: {e}")

try:
    from steps.step4_two_truths import show_two_truths
    STEP4_AVAILABLE = True
except ImportError as e:
    STEP4_AVAILABLE = False
    print(f"Step 4 import error: {e}")

try:
    from steps.step5_emoji_decoder import show_emoji_decoder
    STEP5_AVAILABLE = True
except ImportError as e:
    STEP5_AVAILABLE = False
    print(f"Step 5 import error: {e}")

try:
    from steps.step6_scenarios import show_scenarios
    STEP6_AVAILABLE = True
except ImportError as e:
    STEP6_AVAILABLE = False
    print(f"Step 6 import error: {e}")

try:
    from steps.step7_file_upload import show_file_upload
    STEP7_AVAILABLE = True
except ImportError as e:
    STEP7_AVAILABLE = False
    print(f"Step 7 import error: {e}")

try:
    from steps.step8_cinematic import show_cinematic
    STEP8_AVAILABLE = True
except ImportError as e:
    STEP8_AVAILABLE = False
    print(f"Step 8 import error: {e}")

try:
    from steps.step9_storyboard import show_storyboard
    STEP9_AVAILABLE = True
except ImportError as e:
    STEP9_AVAILABLE = False
    print(f"Step 9 import error: {e}")

try:
    from steps.step11_pop_culture import show_pop_culture
    STEP11_AVAILABLE = True
except ImportError as e:
    STEP11_AVAILABLE = False
    print(f"Step 11 import error: {e}")

try:
    from steps.step12_dynamic_questions import show_dynamic_questions
    STEP12_AVAILABLE = True
except ImportError as e:
    STEP12_AVAILABLE = False
    print(f"Step 12 import error: {e}")

try:
    from steps.step13_memory import show_memory
    STEP13_AVAILABLE = True
except ImportError as e:
    STEP13_AVAILABLE = False
    print(f"Step 13 import error: {e}")

try:
    from steps.step14_smart_revision import show_smart_revision
    STEP14_AVAILABLE = True
except ImportError as e:
    STEP14_AVAILABLE = False
    print(f"Step 14 import error: {e}")

try:
    from steps.step16_progress import show_progress
    STEP16_AVAILABLE = True
except ImportError as e:
    STEP16_AVAILABLE = False
    print(f"Step 16 import error: {e}")

try:
    from steps.step17_profile import show_profile
    STEP17_AVAILABLE = True
except ImportError as e:
    STEP17_AVAILABLE = False
    print(f"Step 17 import error: {e}")

try:
    from steps.step18_settings import show_settings
    STEP18_AVAILABLE = True
except ImportError as e:
    STEP18_AVAILABLE = False
    print(f"Step 18 import error: {e}")

try:
    from steps.step19_ui import apply_custom_css
    STEP19_AVAILABLE = True
except ImportError as e:
    STEP19_AVAILABLE = False
    print(f"Step 19 import error: {e}")

# Practice / Interactive Learning hub — hosts Quiz, Two Truths & A Lie,
# Scenario, Dynamic Question and Emoji Decoder as tabs on one page. This
# is what now renders for the "❓ Quiz" page (the page "📝 Take Quiz" on
# the Study Material page already navigates to), so Take Quiz lands on
# the hub instead of the old quiz-only page.
try:
    from steps.step21_practice import show_practice
    STEP21_AVAILABLE = True
except ImportError as e:
    STEP21_AVAILABLE = False
    print(f"Step 21 import error: {e}")

# Step 15 — SQLite database layer (not a UI step; used for initialization
# and by step16/step17 for real progress + profile data).
try:
    from database.database import (
        initialize_database,
        get_or_create_student,
        create_user,  # Login/Register accounts (users table)
        get_user_by_email,
        EmailAlreadyExistsError,
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# This must be the first Streamlit command in the script.
# (Only the page title was changed to "Mentora".)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Mentora",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# DATABASE INITIALIZATION (Step 15)
# Safe to call on every run — it only creates tables if they don't
# already exist, and never inserts any fake/sample data.
# ----------------------------------------------------------------------
@st.cache_resource
def _initialize_database_once():
    """Create the tables once per server process instead of on every rerun."""
    initialize_database()
    return True


if DATABASE_AVAILABLE:
    try:
        _initialize_database_once()
    except Exception as e:
        st.error(f"⚠️ Could not initialize the database: {e}")
else:
    st.warning("⚠️ Database module not available. Progress and Profile features will not work.")

# ----------------------------------------------------------------------
# STEP 19 UI HELPERS — supplementary CSS
# Adds extra styling (cards, empty states, etc.) used by step16/step17
# without touching or overriding the existing look below.
# ----------------------------------------------------------------------
if STEP19_AVAILABLE:
    apply_custom_css()

# ----------------------------------------------------------------------
# SIMPLE CUSTOM STYLING (existing)
# These classes are kept exactly as they were, because other pages
# may still use them.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }
    .mode-card {
        background-color: #f8f9fb;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        height: 130px;
    }
    .mode-card h4 {
        margin-bottom: 5px;
    }
    .section-space {
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# MENTORA DASHBOARD STYLING (new)
# Light, clean student-dashboard look: white cards, rounded corners,
# subtle shadows, and a polished sidebar. Only CSS — no extra libraries.
# Classes that start with "sf-" are used by the new dashboard HTML below.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Page background and text ---------- */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #F5F7FF;
    }
    [data-testid="stHeader"] {
        background: #F5F7FF;
    }
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #1E1B4B;
    }
    .stApp h1 {
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #475569;
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 1200px;
    }

    /* =====================================================================
       COMPACT SIDEBAR — SINGLE SPACING SYSTEM
       Clean, minimal, professional: white background, small violet accents.
       Everything below is scoped to [data-testid="stSidebar"] and the
       "sf-sidebar-*" / "sf-nav-*" classes, so it cannot affect the
       Dashboard, any other page, or the Login/Register screen.

       ---- Why spacing kept drifting before ----
       Streamlit wraps every st.sidebar.* call in its own element
       container (and buttons in an extra .stButton div, and expanders
       in a <details>), each of which can carry its own default margin
       or padding. Earlier passes only zeroed some of those wrappers, so
       a hidden margin from one layer would silently stack on top of the
       margin we intentionally set, making the visible gap larger than
       the CSS "said" it was — and any later tweak that assumed a clean
       baseline would drift again.

       This block fixes that two ways:
         1. Every wrapper that is not one of our own content elements
            (stElementContainer, stVerticalBlockBorderWrapper, .stButton,
            stMarkdownContainer, <details>) is force-zeroed for margin
            AND padding, so it can never add hidden space.
         2. All real spacing is driven by ONE set of CSS variables below
            — every gap in the sidebar traces back to exactly one
            variable, so there is exactly one place to change any
            distance, and no other rule anywhere adds to it.

       Colors: background #FFFFFF · main text #27304A · secondary text
       #64748B · violet #7C3AED · light violet (active highlight) #DCE3FF ·
       pink accent (logout) #EC4899.
       ===================================================================== */
    [data-testid="stSidebar"] {
        /* ---- The single spacing scale. Every margin/height below reads
           from one of these — change a distance here, nowhere else. ---- */
        --sf-gap-brand-tagline: 2px;     /* Branding -> tagline: 0-2px */
        --sf-gap-tagline-welcome: 14px;  /* Tagline -> Welcome: 12-16px */
        --sf-gap-welcome-home: 20px;     /* Welcome -> HOME: 18-22px */
        --sf-gap-label-item: 7px;        /* HOME/MATERIALS -> first item: 6-8px */
        --sf-gap-item: 9px;              /* every nav item -> next item: 8-10px */
        --sf-gap-group: 24px;            /* Learn & Remember -> MATERIALS,
                                             Insights -> Settings: 22-26px */
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"][aria-expanded="true"] {
        background: #EEF2FF;
        border-right: 1px solid #E2E8F0;
        width: 232px !important;
        min-width: 232px !important;
        max-width: 232px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 232px;
        padding: 14px 12px;
    }

    /* ---- Neutralize every Streamlit wrapper layer so ONLY the rules
       below this point can ever contribute spacing. No wrapper layer
       here is allowed its own margin or padding. ---- */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] .stButton {
        width: 100%;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] > details {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ---- Compact brand header ---- */
    .sf-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 19px;
        font-weight: 700;
        color: #27304A;
        padding: 0 2px;
        margin: 0 0 var(--sf-gap-brand-tagline) 0;
    }
    .sf-sidebar-brand-icon {
        font-size: 18px;
        line-height: 1;
    }
    /* Tagline may wrap onto a second line ("Learn. Create. Practice." /
       "Master.") on the compact sidebar width — that's expected. Its
       own bottom margin is the ONLY thing separating it from Welcome. */
    .sf-sidebar-tagline {
        color: #64748B !important;
        font-size: 11.5px;
        font-weight: 500;
        line-height: 1.4;
        padding: 0 2px;
        margin: 0 0 var(--sf-gap-tagline-welcome) 0;
    }
    /* Welcome row: its own bottom margin is the ONLY thing separating it
       from HOME — nothing below it adds a top margin on top of this. */
    .sf-sidebar-welcome {
        color: #27304A !important;
        font-size: 13px;
        font-weight: 500;
        padding: 0 2px;
        margin: 0 0 var(--sf-gap-welcome-home) 0;
    }
    /* Section labels — HOME / MATERIALS. No top margin: the gap above
       each is already fully provided by whatever precedes it (Welcome,
       or the .sf-nav-gap spacer below). The bottom margin here is the
       ONLY source of the label-to-first-item gap. */
    .sf-nav-group-label {
        color: #7C3AED;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        padding: 0 2px;
        margin: 0 0 var(--sf-gap-label-item) 0;
    }
    /* Plain whitespace spacer between nav groups (no divider line, no
       border). Every top-level nav item already carries a bottom margin
       of --sf-gap-item; this spacer adds exactly the remainder needed to
       reach --sf-gap-group in total — the formula is written out with
       calc() so the total is never a mystery or a second competing
       margin hiding somewhere else. */
    .sf-nav-gap {
        height: calc(var(--sf-gap-group) - var(--sf-gap-item));
        margin: 0;
        padding: 0;
    }


    /* ---------- Buttons (main area) ---------- */
    .stButton {
        width: 100%;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        border: 1px solid transparent;
        color: #ffffff;
        box-shadow: 0 1px 2px rgba(79, 70, 229, 0.25);
        transition: box-shadow 0.18s ease, transform 0.18s ease;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        border-color: transparent;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.20);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="stBaseButton-primary"] p {
        color: #ffffff !important;
    }

    /* ---------- Sidebar navigation buttons (compact, flat rows) ----------
       Every top-level nav row below — Dashboard, Study Material, Create,
       Practice, Learn & Remember, File Upload, Settings, Logout — shares
       this ONE bottom-margin rule (--sf-gap-item) for normal item-to-item
       spacing. There is nowhere else that spacing could be coming from. */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #27304A;
        font-weight: 500;
        font-size: 13.5px;
        min-height: 32px;
        line-height: 1.15;
        padding: 0.35rem 0.5rem;
        border-radius: 6px;
        margin: 0 0 var(--sf-gap-item) 0;
        transition: background 0.12s ease, color 0.12s ease;
    }
    [data-testid="stSidebar"] .stButton > button p {
        color: #27304A !important;
        font-weight: 500;
        font-size: 13.5px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #E4E9FF !important;
    }
    /* Remove Streamlit's default focus ring so only our own subtle
       light-violet active highlight (below) is ever visible — no thick
       outline, no pink/red border. */
    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:focus:not(:active),
    [data-testid="stSidebar"] .stButton > button:focus-visible,
    [data-testid="stSidebar"] .stButton > button:active {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    /* ONLY the active page (Dashboard) ever gets the lavender fill. */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: #DCE3FF !important;
        border-radius: 7px !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p {
        color: #6D28D9 !important;
        font-weight: 600;
    }
    /* Logout: subtle pink text, still a plain compact row (no big button) */
    [data-testid="stSidebar"] div[class*="st-key-nav_logout"] .stButton > button p {
        color: #EC4899 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] div[class*="st-key-nav_logout"] .stButton > button:hover {
        background: #FDF2F8 !important;
    }

    /* ---- Collapsible groups (Create / Practice / Learn & Remember /
       Insights): strip every bit of the default expander "card" look —
       no border, no shadow, no background panel, no oversized padding —
       so each group is one compact flat row with a small chevron, and its
       children are a plain indented list, never a bordered box. The
       expander shares the SAME --sf-gap-item bottom margin as every other
       top-level nav row above, for the same 8-10px spacing. ---- */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        margin: 0 0 var(--sf-gap-item) 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"] {
        padding: 0.35rem 0.5rem !important;
        margin: 0 !important;
        border-radius: 6px;
        font-weight: 500;
        font-size: 13.5px;
        color: #27304A;
        background: transparent !important;
        min-height: 32px !important;
        display: flex;
        align-items: center;
        box-sizing: border-box;
    }
    /* Only Dashboard should ever show the active lavender fill. Streamlit
       (and some browsers) give an expanded/open <summary> its own default
       background, which could make a collapsed-but-previously-active
       group like "Practice" appear highlighted — these rules force every
       state of a group header back to transparent except the deliberate
       :hover below, so Create/Practice/Learn & Remember can never show
       an active-looking background. */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary[aria-expanded="true"],
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"][aria-expanded="true"],
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-within,
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"]:focus-within {
        background: transparent !important;
    }
    /* Force the label text Streamlit nests inside <summary> to inherit our
       compact, mixed-case style instead of any larger default — this is
       what keeps "🎨 Create" reading as a plain nav row, not a heading. */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"] p,
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"] span {
        font-size: inherit !important;
        font-weight: inherit !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        color: inherit !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"] svg {
        width: 12px;
        height: 12px;
        color: #9CA3AF;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    [data-testid="stSidebar"] [data-testid="stExpanderHeader"]:hover {
        background: #E4E9FF !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stSidebar"] [data-testid="stExpander"] > div:last-child {
        padding: 1px 0 1px 10px !important;
        margin: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Children inside a group use a smaller font but the SAME single
       --sf-gap-item margin rule above (inherited from the general
       .stButton > button rule — not redeclared here), for consistent
       spacing between child rows too. */
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {
        font-size: 13px;
        font-weight: 500;
        min-height: 30px;
        padding: 0.3rem 0.5rem;
        color: #4B5563;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button p {
        color: #4B5563 !important;
        font-size: 13px;
        font-weight: 500;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: #DCE3FF !important;
        border-radius: 7px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button[data-testid="stBaseButton-primary"] p {
        color: #6D28D9 !important;
        font-weight: 600;
    }

    /* ---------- Cards ---------- */
    .sf-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        min-height: 110px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.06);
        margin-bottom: 10px;
    }
    .sf-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E1B4B;
        margin-bottom: 6px;
    }
    .sf-card-text {
        color: #6b7280 !important;
        font-size: 0.93rem;
        line-height: 1.45;
        margin: 0;
    }

    /* ---------- Headings inside the dashboard ---------- */
    .sf-welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E1B4B;
        margin: 0 0 4px 0;
    }
    .sf-welcome-sub {
        color: #6b7280 !important;
        margin: 0 0 22px 0;
    }
    .sf-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E1B4B;
        margin: 30px 0 4px 0;
    }
    .sf-section-sub {
        color: #6b7280 !important;
        font-size: 0.95rem;
        margin: 0 0 12px 0;
    }

    /* ---------- Empty states ---------- */
    .sf-empty {
        background: #ffffff;
        border: 1px dashed #d1d5db;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #6b7280;
    }

    /* ---------- Inputs ---------- */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 1px dashed #c7d2fe;
        background: #fafbff;
    }
    [data-baseweb="input"],
    [data-baseweb="select"] > div {
        border-radius: 10px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    /* =====================================================================
       MENTORA LOGIN / REGISTER THEME — soft blue-to-purple, glass card
       Everything below is namespaced under "sf-auth-*" classes (or scoped
       to the "sf_auth_card" container and the login-only button keys) so
       it can never affect the sidebar, the Dashboard, or any other page.
       Palette: Primary #4F46E5 · Secondary #7C3AED · Accent #8B5CF6 ·
       Background #F5F3FF · Card #FFFFFF · Text #1E1B4B · Muted #6B7280.
       ===================================================================== */
    .sf-auth-block [data-testid="stAppViewBlockContainer"],
    .sf-auth-page .block-container {
        padding-top: 1.5rem;
    }

    /* ---------- Animations (all slow and subtle) ---------- */
    @keyframes sfFloat {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50%      { transform: translateY(-14px) rotate(6deg); }
    }
    @keyframes sfFloatSoft {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-8px); }
    }
    @keyframes sfTwinkle {
        0%, 100% { opacity: 0.35; transform: translateY(0) scale(0.9); }
        50%      { opacity: 0.9;  transform: translateY(-12px) scale(1.1); }
    }
    @keyframes sfGlowPulse {
        0%, 100% { opacity: 0.30; transform: scale(1); }
        50%      { opacity: 0.45; transform: scale(1.08); }
    }
    @keyframes sfCardGlow {
        0%, 100% { opacity: 0.45; }
        50%      { opacity: 0.75; }
    }
    @keyframes sfCloudDrift {
        0%, 100% { transform: translateX(0); }
        50%      { transform: translateX(18px); }
    }

    /* ---------- Background glows + decorations (purely visual) ---------- */
    .sf-auth-glow {
        position: fixed;
        border-radius: 50%;
        filter: blur(90px);
        z-index: 0;
        pointer-events: none;
        animation: sfGlowPulse 10s ease-in-out infinite;
    }
    .sf-auth-glow-1 {
        width: 420px; height: 420px; top: -140px; left: -120px;
        background: radial-gradient(circle, #4F46E5 0%, transparent 70%);
    }
    .sf-auth-glow-2 {
        width: 460px; height: 460px; bottom: -170px; right: -130px;
        background: radial-gradient(circle, #7C3AED 0%, transparent 70%);
        animation-delay: -4s;
    }
    .sf-auth-glow-3 {
        width: 320px; height: 320px; top: 40%; left: 42%;
        background: radial-gradient(circle, #8B5CF6 0%, transparent 70%);
        animation-delay: -7s;
    }
    .sf-deco {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        user-select: none;
        line-height: 1;
    }
    .sf-spark {
        animation: sfTwinkle 8s ease-in-out infinite;
    }
    .sf-spark-1 { top: 9%;  left: 12%; font-size: 1.5rem; animation-delay: 0s; }
    .sf-spark-2 { top: 16%; right: 14%; font-size: 1.1rem; animation-delay: -2s; }
    .sf-spark-3 { top: 46%; left: 6%;  font-size: 1.0rem; animation-delay: -4s; }
    .sf-spark-4 { top: 60%; right: 8%; font-size: 1.4rem; animation-delay: -1s; }
    .sf-spark-5 { top: 82%; left: 26%; font-size: 1.0rem; animation-delay: -5s; }
    .sf-spark-6 { top: 30%; right: 30%; font-size: 0.9rem; animation-delay: -3s; }
    .sf-icon-float {
        font-size: 2.3rem;
        opacity: 0.85;
        filter: drop-shadow(0 6px 10px rgba(79, 70, 229, 0.25));
        animation: sfFloat 9s ease-in-out infinite;
    }
    .sf-icon-book { top: 22%; left: 16%; }
    .sf-icon-cap  { top: 24%; right: 16%; animation-delay: -4s; }

    /* Soft floating clouds near the bottom corners (pure CSS shapes) */
    .sf-cloud {
        position: fixed;
        bottom: 4%;
        width: 190px;
        height: 54px;
        border-radius: 60px;
        background: rgba(255, 255, 255, 0.75);
        box-shadow: 0 10px 30px -8px rgba(124, 58, 237, 0.25);
        z-index: 0;
        pointer-events: none;
        animation: sfCloudDrift 16s ease-in-out infinite;
    }
    .sf-cloud::before, .sf-cloud::after {
        content: "";
        position: absolute;
        background: rgba(255, 255, 255, 0.75);
        border-radius: 50%;
    }
    .sf-cloud::before { width: 80px; height: 80px; top: -38px; left: 26px; }
    .sf-cloud::after  { width: 60px; height: 60px; top: -26px; left: 92px; }
    .sf-cloud-left  { left: 3%; }
    .sf-cloud-right { right: 3%; transform: scale(0.85); animation-delay: -8s; }

    /* ---------- Glass auth card (centered) ---------- */
    .st-key-sf_auth_card {
        position: relative;
        z-index: 2;
        isolation: isolate;
        width: 100%;
        max-width: 440px;
        margin: 3vh auto 2rem auto;
        background: rgba(255, 255, 255, 0.90);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 26px;
        padding: 2.2rem 2.1rem 1.6rem 2.1rem;
        box-shadow: 0 25px 60px -25px rgba(79, 70, 229, 0.35),
            0 2px 8px rgba(30, 27, 75, 0.05);
    }
    /* Subtle glowing gradient border behind the card edge */
    .st-key-sf_auth_card::before {
        content: "";
        position: absolute;
        inset: -2px;
        z-index: -1;
        border-radius: 28px;
        background: linear-gradient(135deg, #7C3AED, #4F46E5, #8B5CF6);
        filter: blur(6px);
        animation: sfCardGlow 6s ease-in-out infinite;
    }

    /* Brand lockup inside the card */
    .sf-auth-logo-wrap { text-align: center; }
    .sf-auth-logo {
        width: 68px;
        height: 68px;
        margin: 0 auto 10px auto;
        border-radius: 20px;
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 12px 24px -10px rgba(79, 70, 229, 0.65);
        animation: sfFloatSoft 7s ease-in-out infinite;
    }
    .sf-auth-brand {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        width: fit-content;
        margin: 0 auto;
    }
    .sf-auth-tagline {
        color: #6B7280 !important;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-align: center;
        margin: 4px 0 18px 0;
    }
    .sf-auth-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E1B4B;
        text-align: center;
        margin-bottom: 6px;
    }
    .sf-auth-switch-note {
        text-align: center;
        color: #6B7280 !important;
        margin: 14px 0 0 0;
        font-size: 0.92rem;
    }

    /* Form fields */
    .st-key-sf_auth_card [data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
        box-shadow: none;
    }
    .st-key-sf_auth_card [data-testid="stForm"] label p {
        font-weight: 600;
        color: #1E1B4B !important;
        font-size: 0.9rem;
    }
    .st-key-sf_auth_card [data-baseweb="input"] {
        border-radius: 12px;
        background: #ffffff;
    }
    .st-key-sf_auth_card [data-baseweb="input"]:focus-within {
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.30);
        border-color: #8B5CF6;
    }
    .st-key-sf_auth_card [data-baseweb="input"] button {
        color: #6B7280;   /* show / hide password eye icon */
    }

    /* Large gradient Login / Register button */
    .st-key-sf_auth_card [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        background-size: 140% 140%;
        background-position: 0% 50%;
        border: none;
        color: #ffffff;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.02rem;
        padding: 0.75rem 1.1rem;
        margin-top: 0.5rem;
        box-shadow: 0 12px 24px -10px rgba(79, 70, 229, 0.7);
        transition: background-position 0.4s ease, transform 0.18s ease,
            box-shadow 0.18s ease;
    }
    .st-key-sf_auth_card [data-testid="stFormSubmitButton"] > button:hover {
        background-position: 100% 50%;
        transform: translateY(-2px);
        box-shadow: 0 16px 28px -10px rgba(67, 56, 202, 0.75);
        color: #ffffff;
    }
    .st-key-sf_auth_card [data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0);
    }
    .st-key-sf_auth_card [data-testid="stFormSubmitButton"] > button p {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* "Forgot Password?" and "Sign up" / "Login" switch buttons are shown
       as plain text links (each has its own widget key). */
    .st-key-forgot_pw_btn,
    .st-key-go_register_btn,
    .st-key-go_login_btn {
        display: flex;
        justify-content: center;
    }
    .st-key-forgot_pw_btn button,
    .st-key-go_register_btn button,
    .st-key-go_login_btn button {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0.1rem 0.4rem;
        min-height: 0;
        color: #4F46E5;
        font-weight: 600;
    }
    .st-key-forgot_pw_btn button p,
    .st-key-go_register_btn button p,
    .st-key-go_login_btn button p {
        color: #4F46E5 !important;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .st-key-forgot_pw_btn button:hover p,
    .st-key-go_register_btn button:hover p,
    .st-key-go_login_btn button:hover p {
        color: #7C3AED !important;
        text-decoration: underline;
    }
    .st-key-forgot_pw_btn button:hover,
    .st-key-go_register_btn button:hover,
    .st-key-go_login_btn button:hover {
        background: transparent;
        border: none;
    }

    @media (max-width: 900px) {
        .sf-icon-book { left: 6%; }
        .sf-icon-cap  { right: 6%; }
        .sf-cloud { width: 150px; }
    }
    @media (max-width: 640px) {
        .st-key-sf_auth_card {
            max-width: 100%;
            margin-top: 1rem;
            padding: 1.6rem 1.2rem 1.2rem 1.2rem;
            border-radius: 22px;
        }
        .sf-auth-brand { font-size: 1.75rem; }
        .sf-icon-float { font-size: 1.6rem; opacity: 0.6; }
        .sf-icon-book { top: 3%; left: 5%; }
        .sf-icon-cap  { top: 3%; right: 5%; }
        .sf-spark-3, .sf-spark-6 { display: none; }
        .sf-cloud { width: 110px; height: 40px; opacity: 0.8; }
        .sf-cloud::before { width: 56px; height: 56px; top: -26px; left: 18px; }
        .sf-cloud::after  { width: 42px; height: 42px; top: -18px; left: 62px; }
    }
    @media (prefers-reduced-motion: reduce) {
        .sf-auth-glow, .sf-deco, .sf-cloud, .sf-auth-logo,
        .st-key-sf_auth_card::before {
            animation: none !important;
        }
    }

    /* ---------- Dashboard statistics and lists ---------- */
    .sf-stat {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.06);
        margin-bottom: 8px;
    }
    .sf-stat-label {
        color: #6b7280;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .sf-stat-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1E1B4B;
        line-height: 1.1;
    }
    .sf-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.06);
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .sf-item-title {
        font-weight: 600;
        color: #1E1B4B;
    }
    .sf-item-sub {
        color: #6b7280;
        font-size: 0.85rem;
    }
    .sf-user {
        color: #6b7280;
        font-size: 0.85rem;
        padding: 0 8px 12px 8px;
    }
    [data-testid="stExpander"] {
        background: #ffffff;
        border-radius: 14px;
    }

    /* ---------- Small screens ---------- */
    @media (max-width: 768px) {
        .sf-card {
            min-height: auto;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# SAMPLE / STATIC DATA
# ----------------------------------------------------------------------

# Subjects available in the dropdown
subject_list = [
    "Artificial Intelligence",
    "Machine learning",
    "Cybersecurity",
    "Foundations of Data Science",
    "Computer Science",
    "Data Structure and Algorithms",
    "English",
]

# ----------------------------------------------------------------------
# SESSION STATE
# NOTE: every key below is only set if it doesn't already exist, so
# navigating between pages never resets progress made on any step
# (study_material, flashcards, cinematic_explanation, storyboard,
# pop_culture_connections, the dynamic-quiz state,
# etc. are all owned and initialized by their own step modules).
# ----------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# Holds the last Gemini-generated study material:
# {"subject": ..., "topic": ..., "content": ...}
if "study_material" not in st.session_state:
    st.session_state.study_material = None

# ---- Login state ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
# Kept because existing pages read it: {"name": ..., "username": <email>}
if "current_user" not in st.session_state:
    st.session_state.current_user = None
# Which login-screen view to show: "login" or "register"
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"

# ---- Dashboard statistics (new) ----
# Simple lists that record what happened in this session.
if "material_history" not in st.session_state:
    st.session_state.material_history = []  # [{"subject": ..., "topic": ...}, ...]
if "seen_material_keys" not in st.session_state:
    st.session_state.seen_material_keys = set()
if "completed_quizzes" not in st.session_state:
    st.session_state.completed_quizzes = set()
# True when the "Create Study Material" section should be open
if "show_create_form" not in st.session_state:
    st.session_state.show_create_form = False

# ---- Quiz -> Flashcards unlock state ----
# Flashcards stay locked until the student passes the Quiz (>= 75%) for the
# CURRENT study material. Reset automatically whenever a new/different
# study material is generated (see record_session_activity() below).
if "quiz_passed" not in st.session_state:
    st.session_state.quiz_passed = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None
if "quiz_percentage" not in st.session_state:
    st.session_state.quiz_percentage = None
# Identifies which study material the current quiz_passed state belongs to.
if "active_material_key" not in st.session_state:
    st.session_state.active_material_key = None

# ----------------------------------------------------------------------
# AUTHENTICATION: LOGIN + REGISTER
# Accounts live in the "users" table of the existing SQLite database
# (database/study.db, see database/database.py). A password is NEVER stored:
# only a random salt and a PBKDF2-HMAC-SHA256 hash of it are saved.
# Nothing in this section calls Gemini, so logging in, registering and
# logging out never generate anything.
# ----------------------------------------------------------------------

# ---- Validation rules and messages ----
NAME_PATTERN = re.compile(r"^[A-Za-z]+(?:\s[A-Za-z]+)*$", re.ASCII)
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*"  # part before @ (dots only between characters)
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"  # domain with a dot
)

NAME_ERROR = "Name must contain only letters and spaces."
NAME_REQUIRED_ERROR = "Full name is required."
EMAIL_ERROR = "Please enter a valid email address."
PASSWORD_ERROR = (
    "Password must contain at least 8 characters, including uppercase, "
    "lowercase, and a number."
)
MISMATCH_ERROR = "Passwords do not match."
EMAIL_TAKEN_ERROR = "Email already registered."
LOGIN_ERROR = "Invalid email or password."
GENERIC_ERROR = "Something went wrong. Please try again."
REGISTER_SUCCESS = "Registration successful. Please log in."

# ---- Password hashing settings ----
HASH_ALGORITHM = "pbkdf2_sha256"  # PBKDF2-HMAC with SHA-256 (built into Python)
PASSWORD_ITERATIONS = 600_000  # slows down password guessing
SALT_BYTES = 16  # a new random salt for every account
# Used to spend the same time on unknown emails, so response time doesn't
# reveal whether an email is registered.
_DUMMY_SALT = secrets.token_bytes(SALT_BYTES)


# ---- Cleaning and validating input (no Streamlit calls) ----
def normalize_name(full_name):
    """Trim the name and turn runs of spaces into a single space."""
    return " ".join((full_name or "").split())


def normalize_email(email):
    """Trim and lower-case an email, so ANISHA@GMAIL.COM -> anisha@gmail.com."""
    return (email or "").strip().lower()


def validate_full_name(full_name):
    """Return an error message, or None if the (normalised) name is fine."""
    if not full_name:
        return NAME_REQUIRED_ERROR
    if not NAME_PATTERN.fullmatch(full_name):
        return NAME_ERROR
    return None


def validate_email(email):
    """Return an error message, or None if the (normalised) email is fine."""
    if not email or len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        return EMAIL_ERROR
    return None


def validate_password(password):
    """At least 8 characters with an uppercase letter, a lowercase letter and a number."""
    password = password or ""
    if (
        len(password) < 8
        or not any(ch.isupper() for ch in password)
        or not any(ch.islower() for ch in password)
        or not any(ch in "0123456789" for ch in password)
    ):
        return PASSWORD_ERROR
    return None


# ---- Password hashing (no Streamlit calls) ----
def _derive_key(password, salt, iterations):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def hash_new_password(password):
    """Hash a password with a fresh random salt. Returns the values to store."""
    salt = secrets.token_bytes(SALT_BYTES)
    return {
        "password_hash": _derive_key(password, salt, PASSWORD_ITERATIONS).hex(),
        "salt": salt.hex(),
        "iterations": PASSWORD_ITERATIONS,
        "hash_algorithm": HASH_ALGORITHM,
    }


def verify_password(password, user):
    """Check a password against a saved user row (constant-time comparison)."""
    if user.get("hash_algorithm") != HASH_ALGORITHM:
        return False
    try:
        salt = bytes.fromhex(user["salt"])
        stored_hash = bytes.fromhex(user["password_hash"])
        candidate = _derive_key(password, salt, int(user["iterations"]))
    except (KeyError, TypeError, ValueError):
        return False
    return hmac.compare_digest(stored_hash, candidate)


# ---- Register / log in (no Streamlit calls) ----
def register_user(full_name, email, password, confirm_password):
    """
    Validate the form and save a new account.
    Returns a list of error messages (an empty list means it worked).
    Unexpected database errors are raised, so the page can show a generic message.
    """
    full_name = normalize_name(full_name)
    email = normalize_email(email)

    errors = []
    for error in (validate_full_name(full_name), validate_email(email), validate_password(password)):
        if error:
            errors.append(error)
    if password != confirm_password:
        errors.append(MISMATCH_ERROR)
    if errors:
        return errors

    # Check SQLite first for a friendly message (the UNIQUE constraint still
    # protects against two registrations arriving at the same moment).
    if get_user_by_email(email) is not None:
        return [EMAIL_TAKEN_ERROR]

    credentials = hash_new_password(password)
    try:
        create_user(
            full_name,
            email,
            credentials["password_hash"],
            credentials["salt"],
            credentials["iterations"],
            credentials["hash_algorithm"],
        )
    except EmailAlreadyExistsError:
        return [EMAIL_TAKEN_ERROR]
    return []


def authenticate(email, password):
    """
    Check an email + password. Returns the user's row, or None if they are wrong.
    The same result (and about the same time) is used whether the email is
    unknown or the password is wrong. Unexpected database errors are raised.
    """
    user = get_user_by_email(normalize_email(email))
    if user is None:
        _derive_key(password, _DUMMY_SALT, PASSWORD_ITERATIONS)  # keep timing similar
        return None
    return user if verify_password(password, user) else None


# ---- Session helpers ----
def is_authenticated():
    """True only while a user is logged in."""
    return bool(st.session_state.get("logged_in", False)) and st.session_state.get("user_id") is not None


def current_user_name():
    """The logged-in user's name for greetings."""
    return str(
        st.session_state.get("user_name")
        or (st.session_state.get("current_user") or {}).get("name")
        or "Student"
    )


def start_session(user):
    """Mark the user as logged in (called after a successful login)."""
    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.user_name = user["full_name"]
    st.session_state.user_email = user["email"]
    # Existing pages read current_user, so keep it filled in.
    st.session_state.current_user = {"name": user["full_name"], "username": user["email"]}

    # Link this login to ONE permanent student row in SQLite, so progress, study
    # material and results are kept every time this person logs in.
    if DATABASE_AVAILABLE:
        try:
            student_id = get_or_create_student(user["email"], user["full_name"])
            if student_id:
                st.session_state.student_id = student_id
                st.session_state.student_name = user["full_name"]
        except Exception as e:
            print(f"Could not link the login to a student record: {e}")
    st.session_state.current_page = "🏠 Dashboard"


def set_auth_view(view_name):
    """Button callback: switch between the Login and Register views."""
    st.session_state.auth_view = view_name


def logout():
    """
    Button callback: log out and go back to the Login page.

    The whole session is cleared, so the next person to log in on this browser
    tab never sees the previous user's in-session material. Everything SAVED in
    SQLite (study material, flashcards, results, accounts) is not touched.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.user_email = None
    st.session_state.current_user = None
    st.session_state.auth_view = "login"


# ---- Login page illustration ----
# The image lives next to app.py at assets/login_illustration.png. It is
# resolved by absolute path (not the current working directory) so it is
# found no matter where "streamlit run" is launched from.
LOGIN_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "login_illustration.png"


# ---- The Login / Register pages (Mentora blue + purple theme) ----
# NOTE: this function only changes visual presentation. Every authentication
# call below (register_user / authenticate / start_session / set_auth_view)
# is the exact same function used before this redesign — nothing about
# validation, hashing, SQLite, session_state or Gemini was touched.
def show_forgot_notice():
    """Button callback for the "Forgot Password?" link (display-only message)."""
    st.session_state.auth_info = (
        "Password reset is not available yet. Please contact your administrator."
    )


def show_login_page():
    """Draw the Login or Register screen with the Mentora blue + purple theme."""
    view = st.session_state.get("auth_view", "login")

    # This <style> tag is only ever emitted while show_login_page() itself is
    # running (i.e. before the user is logged in). Streamlit reruns the whole
    # script on every navigation, so once someone logs in this function is
    # never called again and the Dashboard goes back to its own #F5F7FF
    # background — none of the two-column login layout can leak past login.
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background: #F5F3FF !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }

        /* ---- Right half: form column (login card centred) ---- */
        .block-container {
            position: relative;
            z-index: 1;
            box-sizing: border-box;
            width: 50vw;
            max-width: none !important;
            min-height: 100vh;
            margin-left: 50vw;
            padding: 2rem 3rem !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        /* ---- Left half: branding panel ---- */
        @keyframes sfHeroFloat {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50%      { transform: translateY(-14px) rotate(3deg); }
        }
        @keyframes sfHeroGlow {
            0%, 100% { opacity: 0.55; transform: scale(1); }
            50%      { opacity: 0.85; transform: scale(1.08); }
        }
        @keyframes sfHeroTwinkle {
            0%, 100% { opacity: 0.25; transform: scale(0.9); }
            50%      { opacity: 0.85; transform: scale(1.15); }
        }
        .sf-hero {
            position: fixed;
            top: 0; left: 0;
            width: 50vw; height: 100vh;
            z-index: 0;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 55%, #6366F1 100%);
        }
        .sf-hero-circle {
            position: absolute;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0) 70%);
            animation: sfHeroGlow 9s ease-in-out infinite;
        }
        .sf-hero-c1 { width: 460px; height: 460px; top: -140px; left: -120px; }
        .sf-hero-c2 { width: 380px; height: 380px; bottom: -120px; right: -100px; animation-delay: -3s; }
        .sf-hero-c3 { width: 220px; height: 220px; top: 55%; left: 8%; animation-delay: -6s; opacity: 0.5; }
        .sf-hero-emoji, .sf-hero-spark {
            position: absolute;
            line-height: 1;
            user-select: none;
            pointer-events: none;
        }
        .sf-hero-emoji { animation: sfHeroFloat 9s ease-in-out infinite; }
        .sf-hero-book { font-size: 5.5rem; bottom: 12%; left: 12%; opacity: 0.22; }
        .sf-hero-cap  { font-size: 3.8rem; top: 13%; right: 14%; opacity: 0.26; animation-delay: -4s; }
        .sf-hero-spark { font-size: 1.3rem; animation: sfHeroTwinkle 7s ease-in-out infinite; }
        .sf-hero-s1 { top: 22%; left: 16%; }
        .sf-hero-s2 { top: 68%; right: 16%; animation-delay: -3s; }
        .sf-hero-s3 { top: 40%; right: 10%; font-size: 1rem; animation-delay: -5s; }
        .sf-hero-inner {
            position: relative;
            z-index: 1;
            text-align: center;
            padding: 0 3rem;
        }
        .sf-hero-logo {
            width: 92px; height: 92px;
            margin: 0 auto 22px auto;
            border-radius: 26px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.35);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 18px 40px -14px rgba(30, 27, 75, 0.55);
        }
        .sf-hero-brand {
            color: #FFFFFF !important;
            font-size: 4.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.05;
        }
        .sf-hero-tag {
            color: #FFFFFF !important;
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-top: 14px;
        }
        .sf-hero-sub {
            color: #E0E7FF !important;
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: 10px;
        }

        /* ---- Login card: large white card ---- */
        .st-key-sf_auth_card {
            max-width: 480px;
            margin: 0 auto;
            background: #FFFFFF;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
            border: 1px solid #EDE9FE;
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(30, 27, 75, 0.12);
        }
        .st-key-sf_auth_card::before { display: none; }
        .sf-auth-title {
            text-align: left;
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #1E1B4B;
            margin-bottom: 4px;
        }
        .sf-auth-subtitle {
            color: #6B7280 !important;
            font-size: 1rem;
            margin-bottom: 1.4rem;
        }
        .st-key-sf_auth_card [data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            background-size: 100% 100%;
            background-position: 0 0;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.20);
            transform: none;
        }
        .st-key-sf_auth_card [data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            box-shadow: 0 8px 26px rgba(79, 70, 229, 0.30);
            transform: translateY(-1px);
            filter: brightness(1.06);
        }

        /* ---- Smaller screens: stack the two columns ---- */
        @media (max-width: 900px) {
            .sf-hero {
                position: relative;
                width: 100%; height: auto;
                min-height: 230px;
                border-radius: 20px;
                margin-bottom: 1rem;
                padding: 2rem 0;
            }
            .sf-hero-brand { font-size: 2.6rem; }
            .sf-hero-tag { font-size: 1.1rem; }
            .sf-hero-logo { width: 68px; height: 68px; margin-bottom: 14px; }
            .block-container {
                width: 100%;
                margin-left: 0;
                min-height: auto;
                display: block;
                padding: 1.25rem 1rem !important;
            }
            .st-key-sf_auth_card { padding: 28px 22px; }
        }
        @media (prefers-reduced-motion: reduce) {
            .sf-hero-circle, .sf-hero-emoji, .sf-hero-spark { animation: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Left-hand branding panel (purely visual): gradient, glowing circles,
    # a subtle book / cap accent and a few sparkles behind the brand text.
    st.markdown(
        '<div class="sf-hero">'
        '<div class="sf-hero-circle sf-hero-c1"></div>'
        '<div class="sf-hero-circle sf-hero-c2"></div>'
        '<div class="sf-hero-circle sf-hero-c3"></div>'
        '<span class="sf-hero-emoji sf-hero-book">📚</span>'
        '<span class="sf-hero-emoji sf-hero-cap">🎓</span>'
        '<span class="sf-hero-spark sf-hero-s1">✨</span>'
        '<span class="sf-hero-spark sf-hero-s2">✨</span>'
        '<span class="sf-hero-spark sf-hero-s3">✨</span>'
        '<div class="sf-hero-inner">'
        '<div class="sf-hero-logo">'
        '<svg viewBox="0 0 64 64" width="56" height="56" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-label="Mentora logo">'
        '<polyline points="16,28 16,10 26,21 32,14 38,21 48,10 48,28" '
        'stroke="#ffffff" stroke-width="4.5" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        '<path d="M8 37 Q20 32 32 37 Q44 32 56 37 V53 Q44 48 32 53 '
        'Q20 48 8 53 Z" fill="#ffffff" fill-opacity="0.95"/>'
        '<path d="M32 37 V53" stroke="#7C3AED" stroke-width="2"/>'
        '</svg></div>'
        '<div class="sf-hero-brand">Mentora</div>'
        '<div class="sf-hero-tag">Learn • Grow • Succeed</div>'
        '<div class="sf-hero-sub">Your AI-powered learning companion</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ---------------- Centered glass card ----------------
    with st.container(key="sf_auth_card"):
        # Heading + subtitle (the logo / brand now live in the left panel)
        if view == "register":
            st.markdown(
                '<div class="sf-auth-title">Create Your Account</div>'
                '<div class="sf-auth-subtitle">Start your learning journey</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sf-auth-title">Welcome Back!</div>'
                '<div class="sf-auth-subtitle">Continue your learning journey</div>',
                unsafe_allow_html=True,
            )

        # One-time messages (for example after registering)
        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.success(notice)
        info = st.session_state.pop("auth_info", None)
        if info:
            st.info(info)

        if not DATABASE_AVAILABLE:
            st.error("Login is unavailable because the database could not be loaded.")
            return

        # ---------------- Register view ----------------
        if view == "register":
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", max_chars=100)
                reg_email = st.text_input("Email", placeholder="you@example.com", max_chars=254)
                reg_password = st.text_input("Password", type="password")
                reg_confirm = st.text_input("Confirm Password", type="password")
                st.caption("Password: at least 8 characters, with uppercase, lowercase and a number.")
                register_clicked = st.form_submit_button("Register", type="primary")

            if register_clicked:
                try:
                    errors = register_user(reg_name, reg_email, reg_password, reg_confirm)
                except Exception as e:
                    print(f"[auth] Registration failed unexpectedly: {e}")
                    errors = [GENERIC_ERROR]

                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state.auth_notice = REGISTER_SUCCESS
                    st.session_state.auth_view = "login"
                    st.rerun()

            st.markdown('<p class="sf-auth-switch-note">Already have an account?</p>', unsafe_allow_html=True)
            st.button("Login", key="go_login_btn", on_click=set_auth_view, args=("login",))

        # ---------------- Login view (default) ----------------
        else:
            with st.form("login_form"):
                # The password box has Streamlit's built-in show / hide (eye) icon.
                login_email = st.text_input("Email", placeholder="you@example.com", max_chars=254)
                login_password = st.text_input("Password", type="password")
                login_clicked = st.form_submit_button("Login →", type="primary")

            if login_clicked:
                if not login_email.strip() or not login_password:
                    st.error("Please enter your email and password.")
                else:
                    user = None
                    failed = False
                    try:
                        user = authenticate(login_email, login_password)
                    except Exception as e:
                        # Details go to the terminal only, never to the page.
                        print(f"[auth] Login failed unexpectedly: {e}")
                        failed = True

                    if failed:
                        st.error(GENERIC_ERROR)
                    elif user is None:
                        st.error(LOGIN_ERROR)
                    else:
                        start_session(user)
                        st.rerun()

            st.button("Forgot Password?", key="forgot_pw_btn", on_click=show_forgot_notice)

            st.markdown('<p class="sf-auth-switch-note">Don\'t have an account?</p>', unsafe_allow_html=True)
            st.button("Sign up", key="go_register_btn", on_click=set_auth_view, args=("register",))


def require_login():
    """Show the Login page and stop the script unless the user is logged in."""
    if not is_authenticated():
        show_login_page()
        st.stop()


# ----------------------------------------------------------------------
# LOGIN GATE
# Until the student logs in, only the Login / Register page is shown.
# st.stop() ends the script here, so nothing below (sidebar, dashboard,
# steps) is displayed or run.
# ----------------------------------------------------------------------
require_login()


# ----------------------------------------------------------------------
# DASHBOARD STATISTICS HELPER
# Looks at the session data your existing steps already create
# (study_material, quiz_questions, quiz_index) and records it, so the
# dashboard can show real numbers. Nothing here is made up.
# ----------------------------------------------------------------------
def record_session_activity():
    """Remember new study materials and finished quizzes for the dashboard."""
    material = st.session_state.get("study_material")
    if isinstance(material, dict) and material.get("topic"):
        material_key = (
            str(material.get("subject", "")),
            str(material.get("topic", "")),
            hash(str(material.get("content", ""))),
        )

        # A new/different study material means any quiz result and the
        # Flashcards unlock earned for the PREVIOUS material no longer apply
        # (passing the quiz for Chapter A must not unlock Flashcards for
        # Chapter B). This does not touch the saved study history.
        if st.session_state.active_material_key != material_key:
            st.session_state.active_material_key = material_key
            st.session_state.quiz_passed = False
            st.session_state.quiz_score = None
            st.session_state.quiz_percentage = None
            st.session_state.quiz_questions = None
            st.session_state.quiz_index = 0
            st.session_state.quiz_selected = {}
            st.session_state.quiz_submitted = {}
            st.session_state.quiz_correct = {}
            st.session_state.quiz_celebrated = False
            st.session_state.quiz_autogenerate = False

        if material_key not in st.session_state.seen_material_keys:
            st.session_state.seen_material_keys.add(material_key)
            st.session_state.material_history.insert(
                0,
                {
                    "subject": str(material.get("subject", "")),
                    "topic": str(material.get("topic", "")),
                },
            )

    quiz_questions = st.session_state.get("quiz_questions")
    if isinstance(quiz_questions, list) and quiz_questions:
        if st.session_state.get("quiz_index", 0) >= len(quiz_questions):
            quiz_key = hash(
                tuple(str(q.get("question", "")) for q in quiz_questions if isinstance(q, dict))
            )
            st.session_state.completed_quizzes.add(quiz_key)


record_session_activity()

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# The internal page names (the first item in each pair below) are the
# same names the page routing at the bottom of this file uses, so every
# existing page keeps working. Only the label shown in the sidebar
# changes for a few of them.
# ----------------------------------------------------------------------

# Every page in the app (used to route pages and to validate current_page)
sidebar_options = [
    "🏠 Dashboard",
    "📖 Study Material",
    "📝 Flashcards",
    "❓ Quiz",
    "🎭 Two Truths & Lie",
    "😀 Emoji Decoder",
    "🎯 Scenario Questions",
    "📤 File Upload",
    "🎬 Cinematic Explanation",
    "🎞️ Storyboard",
    "🎬 Pop Culture",
    "🔄 Dynamic Questions",
    "🧠 Learning Memory",
    "📚 Smart Revision",
    "📊 Progress",
    "👤 Student Profile",
    "⚙️ Settings",
]

# Safety check: if the saved page name is unknown, go back to the Dashboard
if st.session_state.current_page not in sidebar_options:
    st.session_state.current_page = "🏠 Dashboard"


# Short page names -> the page names used everywhere in this file.
# go_to_page() accepts either, e.g. go_to_page("quiz") or go_to_page("❓ Quiz").
PAGE_SLUGS = {
    "dashboard": "🏠 Dashboard",
    "study_material": "📖 Study Material",
    "flashcards": "📝 Flashcards",
    "quiz": "❓ Quiz",
    "two_truths": "🎭 Two Truths & Lie",
    "emoji_decoder": "😀 Emoji Decoder",
    "scenarios": "🎯 Scenario Questions",
    "smart_revision": "📚 Smart Revision",
    "progress": "📊 Progress",
    "profile": "👤 Student Profile",
    "settings": "⚙️ Settings",
}


def go_to_page(page_name):
    """Button callback: switch to another page (Streamlit reruns automatically after a callback)."""
    st.session_state.current_page = PAGE_SLUGS.get(page_name, page_name)


def open_dashboard():
    """Button callback: open the Dashboard (where study material is generated first)."""
    st.session_state.current_page = "🏠 Dashboard"


def _nav_button(label, page_name, key):
    """One top-level sidebar nav button, highlighted when it's the active page."""
    st.sidebar.button(
        label,
        key=key,
        type="primary" if st.session_state.current_page == page_name else "secondary",
        on_click=go_to_page,
        args=(page_name,),
    )


def _nav_child_button(label, page_name, key):
    """Same as _nav_button, but meant to be called inside a group's expander."""
    st.button(
        label,
        key=key,
        type="primary" if st.session_state.current_page == page_name else "secondary",
        on_click=go_to_page,
        args=(page_name,),
    )


# ---- Compact brand header: brand, tagline, signed-in student ----
signed_in_name = escape(current_user_name())
st.sidebar.markdown(
    '<div class="sf-sidebar-brand">'
    '<span class="sf-sidebar-brand-icon">🤖</span> Mentora AI</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p class="sf-sidebar-tagline">Learn. Create. Practice. Master.</p>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f'<div class="sf-sidebar-welcome">👋 Welcome, {signed_in_name}</div>',
    unsafe_allow_html=True,
)
# ---- HOME: Dashboard, Study Material — always visible, no collapsing ----
st.sidebar.markdown('<p class="sf-nav-group-label">Home</p>', unsafe_allow_html=True)
_nav_button("📊 Dashboard", "🏠 Dashboard", "nav_dashboard")
_nav_button("📖 Study Material", "📖 Study Material", "nav_study_material")

# ---- Create (collapsible): Pop Culture Lens, Cinematic Learning, Storyboard Lab ----
studio_children = ["🎬 Pop Culture", "🎞️ Storyboard", "🎬 Cinematic Explanation"]
with st.sidebar.expander("🎨 Create", expanded=st.session_state.current_page in studio_children):
    _nav_child_button("🎬 Pop Culture Lens", "🎬 Pop Culture", "nav_pop_culture")
    _nav_child_button("🎥 Cinematic Learning", "🎬 Cinematic Explanation", "nav_cinematic")
    _nav_child_button("🖼️ Storyboard Lab", "🎞️ Storyboard", "nav_storyboard")

# ---- Practice (collapsible): Quiz, Truth & Trick, Emoji Decode, Scenario
# Quest, Dynamic Challenge ----
practice_children = [
    "❓ Quiz", "🎭 Two Truths & Lie", "😀 Emoji Decoder",
    "🎯 Scenario Questions", "🔄 Dynamic Questions",
]
with st.sidebar.expander("🎯 Practice", expanded=st.session_state.current_page in practice_children):
    _nav_child_button("❓ Quiz", "❓ Quiz", "nav_quiz_classic")
    _nav_child_button("🎭 Truth & Trick", "🎭 Two Truths & Lie", "nav_two_truths")
    _nav_child_button("🪄 Emoji Decode", "😀 Emoji Decoder", "nav_emoji_decoder")
    _nav_child_button("🎯 Scenario Quest", "🎯 Scenario Questions", "nav_scenarios")
    _nav_child_button("🔀 Dynamic Challenge", "🔄 Dynamic Questions", "nav_dynamic_questions")

# ---- Learn & Remember (collapsible): Flashcards, Smart Revision,
# Learning Memory ----
learn_children = [
    "📝 Flashcards", "📚 Smart Revision", "🧠 Learning Memory",
]
with st.sidebar.expander("🧠 Learn & Remember", expanded=st.session_state.current_page in learn_children):
    _nav_child_button("🃏 Flashcards", "📝 Flashcards", "nav_flashcards")
    _nav_child_button("🧠 Smart Revision", "📚 Smart Revision", "nav_smart_revision")
    _nav_child_button("💾 Learning Memory", "🧠 Learning Memory", "nav_learning_memory")

# ---- Plain whitespace gap (no line) before the "Materials" heading, on
# top of Learn & Remember's own bottom margin, so this group boundary
# reads clearly wider than a normal item-to-item gap ----
st.sidebar.markdown('<div class="sf-nav-gap"></div>', unsafe_allow_html=True)

# ---- MATERIALS: File Upload — always visible, no collapsing ----
st.sidebar.markdown('<p class="sf-nav-group-label">Materials</p>', unsafe_allow_html=True)
_nav_button("📤 File Upload", "📤 File Upload", "nav_file_upload")

# ---- Insights (collapsible): Learning Progress, My Profile ----
insights_children = ["📊 Progress", "👤 Student Profile"]
with st.sidebar.expander("📊 Insights", expanded=st.session_state.current_page in insights_children):
    _nav_child_button("📈 Learning Progress", "📊 Progress", "nav_progress")
    _nav_child_button("👤 My Profile", "👤 Student Profile", "nav_profile")

# ---- Plain whitespace gap (no line) before Settings/Logout, on top of
# Insights' own bottom margin, so this group boundary matches the
# Learn & Remember → Materials gap above ----
st.sidebar.markdown('<div class="sf-nav-gap"></div>', unsafe_allow_html=True)

# ---- Settings, then Logout — compact rows, no card/button styling ----
_nav_button("⚙️ Settings", "⚙️ Settings", "nav_settings")
st.sidebar.button("🚪 Logout", key="nav_logout", on_click=logout)

# ----------------------------------------------------------------------
# HEADER (shown on every page)
# ----------------------------------------------------------------------
st.title("Mentora")
st.markdown(
    '<p class="main-subtitle">Turn your study material into smarter learning.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# AUTHENTICATION CHECK (second lock)
# Every protected page below (Dashboard, Study Material, Flashcards, Quiz,
# Two Truths, Emoji Decoder, Scenarios, Smart Revision, Progress, Profile,
# Settings and the extra study tools) is only reached by passing through this
# line, so none of them can show unless st.session_state.logged_in is True.
# ----------------------------------------------------------------------
require_login()

# ----------------------------------------------------------------------
# STUDY MATERIAL GATE
# These pages all build on the generated study material. If there is none
# yet, show one clear message with a button instead of an empty page.
# (Nothing is generated here - the button only opens the Study Material page.)
# ----------------------------------------------------------------------
PAGES_NEEDING_MATERIAL = {
    "📝 Flashcards",
    "❓ Quiz",
    "🎭 Two Truths & Lie",
    "😀 Emoji Decoder",
    "🎯 Scenario Questions",
    "🎬 Cinematic Explanation",
    "🎞️ Storyboard",
    "🎬 Pop Culture",
    "🔄 Dynamic Questions",
}
_material = st.session_state.get("study_material")
if st.session_state.current_page in PAGES_NEEDING_MATERIAL and not (
    isinstance(_material, dict) and _material.get("content")
):
    st.markdown(
        '<div class="sf-empty">📖 <b>No study material yet.</b><br>'
        "Generate study material on the Dashboard first, then come back to "
        f"{escape(st.session_state.current_page)}.</div>",
        unsafe_allow_html=True,
    )
    st.button("Go to Dashboard", key="gate_go_dashboard", type="primary",
              on_click=open_dashboard)
    st.stop()

# ----------------------------------------------------------------------
# PAGE: DASHBOARD
# Visual redesign only — same session-state values, same callbacks, same
# target pages as before. Nothing on this page calls Gemini; the one
# existing exception (the "Create Study Material" form below) already
# only calls it when the student explicitly submits that form, exactly
# as it did before this redesign.
# ----------------------------------------------------------------------
if st.session_state.current_page == "🏠 Dashboard":

    # ---- Dashboard-only styling (scoped to "sf-dash-" classes, so it
    # cannot affect any other page) ----
    st.markdown(
        """
        <style>
        .sf-dash-hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.20);
        }
        .sf-dash-hero-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 4px 0;
        }
        .sf-dash-hero-sub {
            color: #ede9fe !important;
            font-size: 1.05rem;
            font-weight: 500;
            margin: 0 0 6px 0;
        }
        .sf-dash-hero-tag {
            color: #ddd6fe !important;
            font-size: 0.9rem;
            margin: 0;
        }
        .sf-dash-stat {
            background: #ffffff;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 18px 18px;
            box-shadow: 0 8px 25px rgba(79, 70, 229, 0.06);
            height: 100%;
        }
        .sf-dash-stat-icon {
            font-size: 1.5rem;
            margin-bottom: 6px;
        }
        .sf-dash-stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #1E1B4B;
            line-height: 1.1;
        }
        .sf-dash-stat-label {
            color: #6b7280;
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 2px;
        }
        .sf-dash-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1E1B4B;
            margin: 28px 0 12px 0;
        }
        .sf-dash-continue-card {
            background: #ffffff;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 8px 25px rgba(79, 70, 229, 0.06);
        }
        .sf-dash-continue-eyebrow {
            color: #6d28d9 !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 0 0 6px 0;
        }
        .sf-dash-continue-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1E1B4B;
            margin: 0 0 6px 0;
        }
        .sf-dash-continue-text {
            color: #6b7280 !important;
            font-size: 0.92rem;
            margin: 0;
        }
        .sf-dash-quick-icon {
            font-size: 1.4rem;
            text-align: center;
            margin-bottom: 2px;
        }
        .sf-dash-activity-icon {
            font-size: 1.1rem;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Numbers for the statistics cards (real session data — unchanged) ----
    materials_count = len(st.session_state.material_history)
    saved_flashcards = st.session_state.get("flashcards")
    flashcards_count = len(saved_flashcards) if isinstance(saved_flashcards, list) else 0
    quizzes_count = len(st.session_state.completed_quizzes)
    # Progress = how many of the 3 main steps are done (material, flashcards, quiz)
    steps_done = sum([materials_count > 0, flashcards_count > 0, quizzes_count > 0])
    progress_percent = round(steps_done / 3 * 100)

    # ---- 1. HEADER ----
    student_name = escape(current_user_name())
    st.markdown(
        f'<div class="sf-dash-hero">'
        f'<div class="sf-dash-hero-title">👋 Welcome back, {student_name}!</div>'
        '<p class="sf-dash-hero-sub">Ready to continue your learning?</p>'
        '<p class="sf-dash-hero-tag">Consistent practice beats last-minute cramming — let\'s make today count.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- STEP 1 of the flow: generate study material ----
    # Same existing form (file, subject, topic) as before. On success the
    # student is taken to the Study Material page. Nothing generated is
    # displayed here, and Gemini is only called if the student submits it.
    with st.expander("➕ Create New Study Material", expanded=st.session_state.get("show_create_form", False)):
        if STEP20_AVAILABLE:
            show_generate_form(subject_list, go_to_page)
        else:
            st.error("Step 20 module (steps/step20_study_flow.py) is not available.")

    # ---- 2. STATISTICS CARDS (same real session values as before) ----
    st.markdown('<div class="sf-dash-section-title">📊 Your Learning Snapshot</div>', unsafe_allow_html=True)
    stats = [
        ("📚", "Chapters Studied", str(materials_count)),
        ("🧠", "Flashcards", str(flashcards_count)),
        ("🎯", "Quizzes Completed", str(quizzes_count)),
        ("📈", "Learning Progress", f"{progress_percent}%"),
    ]
    stat_columns = st.columns(4)
    for i, (stat_icon, stat_label, stat_value) in enumerate(stats):
        with stat_columns[i]:
            st.markdown(
                f'<div class="sf-dash-stat">'
                f'<div class="sf-dash-stat-icon">{stat_icon}</div>'
                f'<div class="sf-dash-stat-value">{stat_value}</div>'
                f'<div class="sf-dash-stat-label">{stat_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.progress(progress_percent / 100)
    st.caption(
        "Numbers show your activity in this session. Progress counts three steps: "
        "study material, flashcards, and a finished quiz."
    )

    # ---- 3. CONTINUE LEARNING ----
    st.markdown('<div class="sf-dash-section-title">📖 Continue Learning</div>', unsafe_allow_html=True)

    current_material = st.session_state.study_material
    if isinstance(current_material, dict) and current_material.get("topic"):
        continue_topic = escape(str(current_material.get("topic", "")))
        continue_subject = escape(str(current_material.get("subject", "")))
        st.markdown(
            f'<div class="sf-dash-continue-card">'
            f'<p class="sf-dash-continue-eyebrow">Pick up where you left off</p>'
            f'<div class="sf-dash-continue-title">{continue_topic} <span style="color:#9ca3af;font-weight:500;">'
            f'({continue_subject})</span></div>'
            f'<p class="sf-dash-continue-text">Continue studying the material you recently uploaded.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.button("▶️ Continue Studying", key="continue_btn", type="primary",
                  on_click=go_to_page, args=("📖 Study Material",))
    else:
        st.markdown(
            '<div class="sf-empty">📖 Nothing to continue yet. '
            'Generate study material above to get started.</div>',
            unsafe_allow_html=True,
        )

    # ---- 4. RECENT ACTIVITY ----
    # (Quick Actions stays removed. Take Quiz now lives on the Study
    # Material page instead of here, so this is back to plain full width.)
    st.markdown('<div class="sf-dash-section-title">🕒 Recent Activity</div>', unsafe_allow_html=True)

    if st.session_state.material_history:
        for item in st.session_state.material_history[:5]:
            st.markdown(
                f'<div class="sf-item">'
                f'<div class="sf-item-title"><span class="sf-dash-activity-icon">📄</span>'
                f'{escape(item["topic"])}</div>'
                f'<div class="sf-item-sub">Study material generated · {escape(item["subject"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="sf-empty">No activity yet. Your recent study materials, '
            'flashcards, and quizzes will show up here.</div>',
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------
# PAGE: STUDY MATERIAL
# Implemented in steps/step20_study_flow.py: the complete sectioned display
# of the generated material and the "What would you like to do next?"
# buttons. (The generate form and the ONE Gemini request live on the
# Dashboard, via show_generate_form().) (steps/step1_study_material.py is kept and still imported, but
# it only understands the old flashcards+quiz-in-one-answer format, so it
# is no longer routed to.)
# ----------------------------------------------------------------------
elif st.session_state.current_page == "📖 Study Material":
    if STEP20_AVAILABLE:
        show_study_flow(subject_list, go_to_page)
    else:
        st.subheader("📖 Study Material")
        st.error("Step 20 module (steps/step20_study_flow.py) is not available.")

# ----------------------------------------------------------------------
# PAGE: FLASHCARDS
# Implemented in steps/step2_flashcards.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "📝 Flashcards":
    if STEP2_AVAILABLE:
        show_flashcards()
    else:
        st.subheader("📝 Flashcards")
        st.error("Step 2 module is not available.")

# ----------------------------------------------------------------------
# PAGE: PRACTICE (formerly the standalone Quiz page)
# Implemented in steps/step21_practice.py, which hosts Quiz, Two Truths &
# A Lie, Scenario, Dynamic Question and Emoji Decoder together as tabs.
# "📝 Take Quiz" on the Study Material page navigates to this same
# "❓ Quiz" page name, so that button now lands here with no separate
# Generate Quiz button. steps/step3_quiz.py is still used directly below
# (it's what step21_practice.py's Quiz tab calls), so it stays imported.
# ----------------------------------------------------------------------
elif st.session_state.current_page == "❓ Quiz":
    if STEP21_AVAILABLE:
        show_practice()
    else:
        st.subheader("❓ Quiz")
        st.error("Step 21 module (steps/step21_practice.py) is not available.")

# ----------------------------------------------------------------------
# PAGE: TWO TRUTHS & A LIE
# Implemented in steps/step4_two_truths.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🎭 Two Truths & Lie":
    if STEP4_AVAILABLE:
        show_two_truths()
    else:
        st.subheader("🎭 Two Truths & Lie")
        st.error("Step 4 module is not available.")

# ----------------------------------------------------------------------
# PAGE: EMOJI DECODER
# Implemented in steps/step5_emoji_decoder.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "😀 Emoji Decoder":
    if STEP5_AVAILABLE:
        show_emoji_decoder()
    else:
        st.subheader("😀 Emoji Decoder")
        st.error("Step 5 module is not available.")

# ----------------------------------------------------------------------
# PAGE: SCENARIO QUESTIONS
# Implemented in steps/step6_scenarios.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🎯 Scenario Questions":
    if STEP6_AVAILABLE:
        show_scenarios()
    else:
        st.subheader("🎯 Scenario Questions")
        st.error("Step 6 module is not available.")

# ----------------------------------------------------------------------
# PAGE: FILE UPLOAD
# Implemented in steps/step7_file_upload.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "📤 File Upload":
    if STEP7_AVAILABLE:
        show_file_upload()
    else:
        st.subheader("📤 File Upload")
        st.error("Step 7 module is not available.")

# ----------------------------------------------------------------------
# PAGE: CINEMATIC EXPLANATION
# Implemented in steps/step8_cinematic.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🎬 Cinematic Explanation":
    if STEP8_AVAILABLE:
        show_cinematic()
    else:
        st.subheader("🎬 Cinematic Explanation")
        st.error("Step 8 module is not available.")

# ----------------------------------------------------------------------
# PAGE: STORYBOARD
# Implemented in steps/step9_storyboard.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🎞️ Storyboard":
    if STEP9_AVAILABLE:
        show_storyboard()
    else:
        st.subheader("🎞️ Storyboard")
        st.error("Step 9 module is not available.")

# ----------------------------------------------------------------------
# PAGE: POP CULTURE
# Implemented in steps/step11_pop_culture.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🎬 Pop Culture":
    if STEP11_AVAILABLE:
        show_pop_culture()
    else:
        st.subheader("🎬 Pop Culture")
        st.error("Step 11 module is not available.")

# ----------------------------------------------------------------------
# PAGE: DYNAMIC QUESTIONS
# Implemented in steps/step12_dynamic_questions.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🔄 Dynamic Questions":
    if STEP12_AVAILABLE:
        show_dynamic_questions()
    else:
        st.subheader("🔄 Dynamic Questions")
        st.error("Step 12 module is not available.")

# ----------------------------------------------------------------------
# PAGE: LEARNING MEMORY
# Implemented in steps/step13_memory.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "🧠 Learning Memory":
    if STEP13_AVAILABLE:
        show_memory()
    else:
        st.subheader("🧠 Learning Memory")
        st.error("Step 13 module is not available.")

# ----------------------------------------------------------------------
# PAGE: SMART REVISION
# Implemented in steps/step14_smart_revision.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "📚 Smart Revision":
    if STEP14_AVAILABLE:
        show_smart_revision()
    else:
        st.subheader("📚 Smart Revision")
        st.error("Step 14 module is not available.")

# ----------------------------------------------------------------------
# PAGE: PROGRESS
# Implemented in steps/step16_progress.py — REAL data from SQLite via
# database/database.py (student_id comes from the Student Profile page).
# ----------------------------------------------------------------------
elif st.session_state.current_page == "📊 Progress":
    if STEP16_AVAILABLE:
        show_progress()
    else:
        st.subheader("📊 Progress")
        st.error("Step 16 module is not available.")

# ----------------------------------------------------------------------
# PAGE: STUDENT PROFILE
# Implemented in steps/step17_profile.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "👤 Student Profile":
    if STEP17_AVAILABLE:
        show_profile()
    else:
        st.subheader("👤 Student Profile")
        st.error("Step 17 module is not available.")

# ----------------------------------------------------------------------
# PAGE: SETTINGS
# Implemented in steps/step18_settings.py
# ----------------------------------------------------------------------
elif st.session_state.current_page == "⚙️ Settings":
    if STEP18_AVAILABLE:
        show_settings()
    else:
        st.subheader("⚙️ Settings")
        st.error("Step 18 module is not available.")