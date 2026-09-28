"""
Step 3 — Quiz Page
---------------------
Generates ONE multiple-choice quiz question at a time (via Ollama) from the
study material already created on the Dashboard, and lets the student
answer it with instant feedback.

FIX (this revision): the question generation pipeline was too strict and
was rejecting perfectly good questions from the small local model, which
showed up to the student as:
    "Could not create a valid question from the study material."

Two things were causing that:
  1. Option keys/shape were required to match an exact dict with exact
     "A"/"B"/"C"/"D" keys. If Ollama sent lowercase keys ("a"/"b"/"c"/"d"),
     or sent options as a plain list of 4 strings instead of a dict, the
     whole question was thrown away even though it had everything needed.
  2. A "near-duplicate option" check compared option TEXT with a fuzzy
     text-similarity ratio (difflib) that isn't word-aware. Two options
     that just happen to share a lot of characters/words (e.g. two
     explanations about the same process, worded differently) could score
     "too similar" and get the whole question rejected, even though they
     were two genuinely different answers.

Both are fixed below: option keys are matched case-insensitively and a
plain 4-item list is also accepted, and duplicate options are now only
rejected when they are the SAME text (ignoring case/spacing) - not merely
"similar-sounding". Everything else (one Ollama call per question, no
generation on reruns, session-state driven flow) is unchanged.

FIX (this revision): the Practice hub was still occasionally showing
"Could not create a valid Quiz question. Please try again." even with the
fixes above. Two more gaps, both specific to this tiny 0.5B model:

  1. correct_answer extraction only accepted a letter at the very start
     of the field. A model this small doesn't always put the letter
     first even when told to ("Answer: A", "The correct option is C.") -
     and sometimes echoes the full option text instead of a letter at
     all. _read_option_letter() now also finds a standalone letter
     further into the text, and a new _resolve_correct_answer() falls
     back to matching the raw text against the option TEXTS themselves
     (only when that match is unambiguous - one and only one option) so
     a genuinely good question is no longer thrown away just because the
     model didn't format its answer exactly as asked.
  2. generate_content(prompt) inside _generate_one_question() was not
     wrapped in a try/except, unlike every other generator in this app
     (step5_emoji_decoder.py's _generate_emoji_challenge, for example).
     A transient Ollama failure (a connection error, or an HTTP 500 that
     survived ollama_service.py's own internal retries) would raise
     straight out of this function - and since the Practice hub's "quiz"
     branch does not wrap this call in its own try/except either, that
     would crash the whole Practice page instead of showing the normal
     "Could not create..." + Retry flow. It's now caught and treated as
     one failed attempt, same as a bad reply.

FIX (this revision): distractor quality. Previously the three "wrong"
options only had to be non-blank, non-duplicate text - nothing stopped
the model from generating same-topic distractors (alternative but wrong
answers from the same subject, partially-correct options, or synonyms of
the correct answer), which makes a question much easier to guess by
elimination and defeats the point of testing recall of the study
material. Two things changed to fix this:
  1. The prompt (_build_question_prompt) now explicitly requires the
     three wrong options to be simple, real concepts from a completely
     different, everyday domain (computers, music, sports, geography,
     food, transportation, etc.) that share nothing with the question's
     topic - with a concrete good/bad example - instead of just asking
     for "plausible" wrong answers.
  2. validate_quiz_question() now also rejects a question if any wrong
     option shares a meaningful (non-trivial) word with the question or
     the correct answer, or if the correct answer itself doesn't share
     anything with the study material - see section 3 below for the new
     _tokenize()/_topic_tokens()/_option_is_topic_related() helpers and
     _correct_answer_grounded_in_content(). This is a plain-Python text
     check, not a model call, so it adds no extra Ollama requests and
     never runs during answer submission or a normal rerun.

FIX (this revision): answer-key reliability. The model could previously
pick a `correct_answer` that "sounded right" without that answer actually
being traceable to a specific sentence in the study material - so a
student answering from an outside, correct reference (e.g. their own
notes) could be marked wrong by a `correct_answer` that was itself wrong.
The prompt now REQUIRES a new `evidence` field: a short VERBATIM excerpt
copied from the study material that the correct option must be a close
restatement of. Two new, purely deterministic (no extra Ollama calls)
Python checks enforce this before a question is ever shown:
  - _validate_evidence_support() confirms `evidence` is non-empty and
    actually appears in the study material, after tolerant normalization
    (lowercase, whitespace, and punctuation differences are ignored - but
    no wording may be invented).
  - _validate_correct_answer_against_evidence() confirms the correct
    option shares substantial meaningful-word overlap with that evidence
    (a conservative, non-AI lexical check - not a full semantic model),
    so a correct option that has nothing to do with its own claimed
    evidence is rejected rather than shown.
A failing question is rejected and regenerated, exactly like any other
validation failure - never shown to the student. The correct answer's
letter position (A/B/C/D) is also now randomized deterministically in
Python after generation, so it isn't always "A".

FIX (this revision): the "evidence" requirement above was the main cause
of the student-facing error "Could not create a valid Quiz round. Please
try again." - it demanded a VERBATIM excerpt copied word-for-word from
the study material, which this project's small local model frequently
could not produce even when the question and answer were otherwise
perfectly good, so most attempts were being discarded for a formatting
reason rather than a correctness one. Two things changed to fix this:
  1. The prompt (_build_question_prompt) no longer asks for or requires
     an "evidence" field. It now asks for exactly the simpler shape
     {"question", "options", "correct_answer", "explanation"} (still
     with a required "options" dict of A/B/C/D, and the three wrong
     options still required to come from a completely different,
     everyday domain than the question's topic).
  2. _validate_evidence_support() and _validate_correct_answer_against_evidence()
     are removed - they no longer gate anything. In their place,
     validate_quiz_question() now calls a new _auto_classify_correct_answer()
     that is the ONLY source of truth for which option is correct:
       - Option A is compared against the study material.
       - Option B is compared against the study material.
       - Option C is compared against the study material.
       - Option D is compared against the study material.
       - The ONE option that is grounded in the study material (reusing
         the existing _correct_answer_grounded_in_content() per option)
         is the CORRECT answer; the other THREE must also be unrelated
         to this question's specific topic (reusing the existing
         _topic_tokens() / _option_is_topic_related()) to be confirmed
         WRONG.
     Whatever Ollama itself put in "correct_answer" is NEVER read or
     trusted anywhere in this file any more - if it's right, wrong, or
     missing makes no difference, since validate_quiz_question()
     overwrites question["correct_answer"] with the auto-detected letter
     on success. If the 4 options don't cleanly split into exactly 1
     study-supported option + 3 unrelated ones, the question is rejected
     and a new one is generated instead, up to MAX_GENERATION_ATTEMPTS
     attempts total - the same "reject and retry" behavior as before,
     just driven by an independent check instead of a fragile verbatim
     quote. This is still ONE question per Ollama call, never a batch of
     several questions validated together.

FIX (this revision): root-cause investigation + temporary debugging.
This file was re-inspected top to bottom against a checklist of every
place that could produce "Could not create a valid Quiz round": every
generator, JSON parser, validator, retry loop, session-state key, and
the exact _generate_one_question(...) interface step21_practice.py
calls. Findings:
  - There is exactly ONE generation/validation path - _generate_one_question()
    - and exactly ONE question is ever requested, parsed, or validated
    per call. Nothing in this file asks Ollama for 5 questions, expects
    a "questions" array/list of 5, or uses any "N of 5 valid questions"
    logic - quiz_questions in session state is always a single-item list
    ([question]), never a longer batch.
  - The remaining, most likely source of an intermittent failure is
    this tiny local model occasionally producing a reply that fails to
    parse, or options that don't cleanly resolve to one correct answer -
    both already retried up to MAX_GENERATION_ATTEMPTS times, but with
    no visibility into WHICH stage failed on a given attempt.
Two changes address this directly:
  1. Study-source guard (Step 3): _generate_one_question() now checks
     `content` FIRST, before doing anything else. If it's missing or
     blank, Ollama is never called at all - it returns None immediately
     with the internal reason "Study material is missing." (only ever
     printed to the terminal via _print_quiz_debug(), never shown in the
     UI). This file still only ever reads the study material that its
     caller (show_quiz() / step21_practice.py) passes in as `content` -
     i.e. st.session_state["study_material"]["content"] - never
     `uploaded_text` or any other source, and never a hardcoded string.
  2. Temporary internal debugging (Step 2): a new _print_quiz_debug()
     helper prints one structured "QUIZ DEBUG" block per attempt - the
     attempt number, the first 500 characters of the raw Ollama reply,
     whether JSON parsing succeeded, the parsed dict (if any), whether
     validation succeeded, and the EXACT validation reason (via a new
     _validate_quiz_question_with_reason(), which validate_quiz_question()
     now wraps) - straight to the terminal running `streamlit run`, using
     plain print() so it always shows up regardless of logging
     configuration. This makes it possible to see, attempt by attempt,
     whether a failure was (1) Ollama itself / a connection error,
     (2) invalid JSON, (3) a missing required field, (4) missing/malformed
     options, or (5) the auto-classify-correct-answer check rejecting the
     4 options. None of this is ever shown in the Streamlit UI or exposes
     raw model output to the student - it is developer-only, read
     straight off the server console.

FIX (this revision): the actual "function argument error". The file was
re-checked for exactly what was asked: every definition of
_generate_one_question, confirming there is only ONE active definition,
and its signature already was (and remains) exactly
`(subject, topic, content, language_note, avoid_questions)` - no
duplicate/obsolete version existed to remove. The real bug was one level
deeper: `avoid_questions` is documented as optional ("previously used
questions to avoid repeating"), and the Practice hub may reasonably pass
None for it (e.g. on the very first question, before any history
exists) - but _build_question_prompt() did
`"\n".join(f"- {q}" for q in avoid_questions)` directly, which raises
`TypeError: 'NoneType' object is not iterable` when avoid_questions is
None. That happens while BUILDING the prompt, before the try/except
around generate_content() even runs, so the exception escaped
_generate_one_question() entirely uncaught - and since the Practice hub
calls this function directly with no try/except of its own (only
_generate_and_store_question(), used by the standalone page, wraps it),
that crash could take down the whole Practice page. Fixed with a single
normalization, `avoid_questions = avoid_questions or []`, applied both
at the top of _generate_one_question() (so every attempt, and anything
added to this function later, only ever sees a real list) and again
defensively inside _build_question_prompt() itself. avoid_questions
being None or an empty list are now both treated identically as "no
history yet" and never cause a failure.

FIX (this revision): root cause of "options do not cleanly resolve to
exactly one study-supported correct answer" confirmed via the QUIZ DEBUG
terminal output - Ollama, JSON parsing, and shaping were all already
succeeding on every attempt; only this one validation rule was rejecting
otherwise-good questions. The problem was
_auto_classify_correct_answer(): it required the three WRONG options to
share ZERO meaningful words with the question/correct-answer topic
(_topic_tokens() / _option_is_topic_related()), and that plain-word
overlap check has no idea which shared words are actually meaningful vs.
just ordinary vocabulary (e.g. "energy", "data", "construct" are not in
_TOPIC_STOPWORDS) - so a perfectly unrelated distractor like "A system
used to convert light energy into chemical energy" could get flagged
purely for reusing a generic word that also happened to appear in the
question or study material, even though it was clearly about a different
topic. _auto_classify_correct_answer(), _topic_tokens(), and
_option_is_topic_related() are removed - no keyword/topic-overlap check
is run against the wrong options any more, and no wrong option is ever
rejected just because it contains a common word. Validation now:
  1. trusts the model's own declared correct_answer letter (already
     normalized to A/B/C/D by _resolve_correct_answer/_read_option_letter
     - "A", "(A)", "Answer: A", "Option A", or the option's own text
     echoed back all resolve the same way as before; no second Ollama
     call is ever made to "verify" the answer),
  2. requires that declared correct option to be reasonably supported by
     the study material - reusing the existing light-touch
     _correct_answer_grounded_in_content() (shared meaningful word, or
     exact substring for very short answers) - checked ONLY for the one
     declared correct option, not for all four options any more, and
  3. requires the three wrong options not to be near-duplicate
     restatements of the correct option - a new _option_is_near_duplicate()
     using a plain character-level similarity ratio (difflib.SequenceMatcher)
     on the normalized option text, NOT a keyword/topic comparison - this
     only catches a distractor that is basically the correct answer
     paraphrased (an "obvious answer in disguise"), never one that merely
     shares an ordinary word with the topic or the study material.
Everything else - one Ollama call per attempt, the JSON parser, the
option-shape normalization, the exact-duplicate check, MAX_GENERATION_ATTEMPTS
= 3, one question per call, the QUIZ DEBUG terminal logging - is
unchanged.

MODEL RULE: this file never hardcodes a model name. Every generation call
goes through generate_content(prompt) from services/ollama_service.py, and
the model used is controlled ONLY by MODEL_NAME inside that file. This
file never creates its own Ollama client.

Embedding: pass embedded=True (used by the Practice hub,
steps/step21_practice.py) to skip this page's own header/topic bar, since
the Practice hub already shows those once above all five activity tabs.
Called with no arguments (embedded=False), this file behaves as a normal
standalone page, exactly as it did before.

INTEGRATION NOTE: steps/step21_practice.py (the Practice hub) imports and
calls _generate_one_question(subject, topic, content, language_note,
avoid_questions) directly - this exact function name and 5-argument
signature is a stable internal interface other modules depend on and must
not be changed without also updating step21_practice.py.

Imported in app.py with:
    from steps.step3_quiz import show_quiz
"""

import difflib
import logging
import json
import random
import re

import streamlit as st

from services.ollama_service import generate_content
from steps.step13_memory import log_answer, record_topic
from steps.step18_settings import get_setting, language_instruction
from steps.step19_ui import (
    require_study_material,
    show_page_header,
    show_topic_bar,
)

# Development-only logging. Nothing here is shown to the student; turn it on
# with  logging.basicConfig(level=logging.DEBUG)  in your terminal if needed.
logger = logging.getLogger(__name__)

# Stable option keys. Each key always maps to exactly one option text.
OPTION_KEYS = ["A", "B", "C", "D"]

# How many times we ask Ollama for ONE question before giving up on that
# request (a fresh single-question request each try). Requirement: max 3.
MAX_GENERATION_ATTEMPTS = 3

# How many of the most recently asked questions (this session) are given
# to Ollama as "please avoid repeating these" context, so back-to-back
# "New Question" clicks don't just re-serve the same fact. This is only a
# hint inside the PROMPT - it is never used to reject a generated question,
# so it can never be the reason generation "fails".
RECENT_QUESTIONS_TO_AVOID = 8


# --------------------------------------------------------------------------
# 1. PROMPT BUILDING - strict JSON in, strict JSON out (ONE question/call)
# --------------------------------------------------------------------------

def _build_question_prompt(subject, topic, content, language_note, avoid_questions, strict=False):
    """
    Build the prompt sent to Ollama for exactly ONE quiz question.

    strict=False is used on the first attempt. strict=True is used on
    retries (attempts 2 and 3) and repeats the JSON-only rules (plus the
    distractor-unrelatedness rule) in a short, impossible-to-miss
    reminder block, since a small local model is more likely to follow
    strict formatting the second time it's told plainly that its last
    answer didn't come back as clean JSON or still used same-topic wrong
    options.

    No "evidence" field is requested any more (see the module docstring's
    "FIX (this revision)" note) - the correct answer is determined
    independently in Python by _auto_classify_correct_answer(), not by
    asking the model to prove its own work with a verbatim quote.

    `avoid_questions` may be None (as well as an empty list) - both mean
    "no history to avoid yet" and are treated identically. This function
    never fails just because that argument is empty/None (see the module
    docstring's "FIX (this revision)" note on the argument crash).
    """
    language_line = f"\n{language_note}\n" if language_note else ""
    avoid_questions = avoid_questions or []
    avoid_lines = "\n".join(f"- {q}" for q in avoid_questions) or "- (none)"

    base_prompt = f"""You are a helpful study assistant for college students.
{language_line}
Subject: {subject}
Topic/Chapter: {topic}

Study material:
{content}

Based ONLY on the study material above, create exactly ONE multiple-choice
quiz question. The question and its correct answer must come only from the
study material - do not invent information that isn't there. Keep the
language simple and beginner friendly. Do not copy a sentence directly out
of the study material - write a new question about the same fact, ideally
one that asks the student to apply or reason about it rather than just
recall it word-for-word.

Try not to reuse any of these already-asked questions or the facts they
test (this is a soft preference, not a strict requirement):
{avoid_lines}

Rules:
- Create exactly four options: A, B, C, and D.
- Exactly ONE option (the correct one) is directly supported by the study
  material above, and it must be the only option that actually answers the
  question.
- The other THREE options must be COMPLETELY UNRELATED to the question's
  topic:
    - Do NOT use same-topic distractors (other facts/answers from the
      same subject or chapter).
    - Do NOT use closely related concepts as wrong options.
    - Do NOT use partially correct options.
    - Do NOT use synonyms or reworded versions of the correct answer.
    - Do NOT use any concept that appears in the study material above.
  Instead, each wrong option should be a simple, real, sensible concept
  from a completely different everyday domain - for example: computers,
  music, sports, geography, food, transportation, etc. (pick whichever
  fits naturally). Never let an unrelated option accidentally also answer
  the question.
- The four option texts must be four DIFFERENT strings - never repeat the
  same answer twice, never leave an option blank.
- correct_answer must be ONLY the letter A, B, C, or D (never the option
  text). Do not always make A the correct answer - vary the correct
  option's position across questions.
- explanation is one short sentence, based only on the study material,
  explaining why the correct option is correct.

Example of the exact style required (study material contains: "Green
plants convert light energy into chemical energy through photosynthesis"):
  Correct option : "Light energy into chemical energy"
  Wrong option   : "A system used to transmit computer data"
  Wrong option   : "A type of musical composition"
  Wrong option   : "A method used to construct bridges"
This is GOOD because the wrong options are unrelated, everyday, real
concepts from completely different domains (computer networking, music,
construction).
Do NOT write wrong options like "Absorption of water by roots" or
"Cellular respiration" for that same question - those are BAD because they
are still biology / the same topic, even though they are wrong answers.

Before you answer, internally check (do not show this reasoning, only
return the final JSON):
- Correct option: is it directly supported by the study material, and is
  it the only option that answers the question?
- Each wrong option: is it unrelated to the topic, definitely incorrect,
  not partially correct, and not a synonym or alternative form of the
  correct answer?
- Explanation: does it explain the correct answer using the study
  material?

Return ONLY valid JSON. Do not use Markdown. Do not use ```json code
fences. Do not add any explanation or commentary outside the JSON. Use
exactly this shape:

{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_answer": "A",
  "explanation": "..."
}}
"""

    if not strict:
        return base_prompt

    # Extra, very blunt reminder used on retries only.
    return base_prompt + """
IMPORTANT REMINDER - your previous reply could not be used, either because
it was not clean, valid JSON in the exact shape above, or because a wrong
option was still related to the question's topic. This time:
- Output ONLY the JSON object. Nothing before it, nothing after it.
- No ```json or ``` fences.
- No headings, no bullet points, no extra commentary.
- Every one of "question", "options" (with A, B, C, D), "correct_answer",
  and "explanation" must be present and non-empty.
- The three wrong options must come from a completely different, everyday
  domain (e.g. computers, music, sports, geography, food, transportation)
  and must share NOTHING with the question's topic or the study material -
  no same-topic distractors, no partially-correct options, no synonyms or
  reworded versions of the correct answer.
"""


# --------------------------------------------------------------------------
# 2. JSON EXTRACTION - tolerant of code fences, stray text, trailing commas
# --------------------------------------------------------------------------

def _strip_code_fences(text):
    """Remove ```json ... ``` (or plain ``` ... ```) fences around a reply."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text):
    """
    Pull the JSON object out of a raw model reply:
      1. strip ```json / ``` fences
      2. find the first "{"
      3. find the last "}"
      4. return only the text between them (inclusive)
    Returns None if no {...} block is present at all.
    """
    cleaned = _strip_code_fences(text or "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return cleaned[start:end + 1]


def _load_json_object(text):
    """
    Turn a raw model reply into a Python dict, tolerating markdown fences,
    stray text around the JSON, and a trailing comma before a closing
    brace/bracket. Uses ONLY json.loads() - never eval(). Returns None if
    nothing usable could be parsed.
    """
    json_text = _extract_json_object(text)
    if json_text is None:
        return None

    # Small local models occasionally leave a trailing comma before } or ]
    # - fix that before giving up on an otherwise-valid object.
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", json_text)

    for candidate in (json_text, without_trailing_commas):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _read_option_letter(value):
    """
    Pull a single A/B/C/D letter out of correct_answer, however it was
    written:
      - a leading letter, optionally in parens ("A", "(A)")
      - a letter that appears as its own token further into the text
        ("Answer: A", "The correct option is C.") - a very small local
        model doesn't always put the letter first even when told to.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    match = re.match(r"^\s*\(?\s*([A-Da-d])\s*\)?", text)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-Da-d])\b", text)
    return match.group(1).upper() if match else None


def _normalize_options(options_raw):
    """
    Turn whatever Ollama sent for "options" into a clean {"A": ..., "B":
    ..., "C": ..., "D": ...} dict, or None if it can't be made sense of.

    Accepts two shapes, since small local models don't always follow the
    dict-with-exact-keys shape we ask for:
      - a dict, with keys matched CASE-INSENSITIVELY ("a"/"A" both work)
      - a plain list/tuple of exactly 4 strings, assigned in order to
        A, B, C, D
    """
    options = {}

    if isinstance(options_raw, dict):
        # Match keys case-insensitively (Ollama sometimes replies with
        # lowercase "a", "b", "c", "d" instead of the requested uppercase).
        by_upper_key = {}
        for raw_key, raw_value in options_raw.items():
            if not isinstance(raw_key, str):
                continue
            by_upper_key[raw_key.strip().upper()] = raw_value
        for key in OPTION_KEYS:
            value = by_upper_key.get(key)
            if isinstance(value, str) and value.strip():
                options[key] = value.strip()

    elif isinstance(options_raw, (list, tuple)) and len(options_raw) == 4:
        for key, value in zip(OPTION_KEYS, options_raw):
            if isinstance(value, str) and value.strip():
                options[key] = value.strip()

    return options if len(options) == 4 else None


def _resolve_correct_answer(raw_value, options):
    """
    Work out which option key (A/B/C/D) the model intended as correct,
    even when it didn't reply with a bare letter.

    Tries, in order:
      1. A letter read directly out of the text (see _read_option_letter) -
         covers "A", "(A)", "Answer: A", etc.
      2. The raw text matched against the option TEXTS themselves, in case
         the model echoed the full answer instead of its letter. Only used
         if it matches exactly one option (ignoring case/spacing) - an
         ambiguous or no match is left for validation to reject, rather
         than guessing.

    Returns the resolved letter, or None if neither approach finds an
    answer.
    """
    letter = _read_option_letter(raw_value)
    if letter:
        return letter

    if isinstance(raw_value, str) and options:
        normalized_value = normalize_option_text(raw_value)
        if normalized_value:
            matches = [
                key for key, text in options.items()
                if normalize_option_text(text) == normalized_value
            ]
            if len(matches) == 1:
                return matches[0]

    return None


def _normalize_question(raw):
    """
    Turn one RAW parsed dict into the internal shape used everywhere else
    in this file, or return None if the shape is unusable. This only reads
    the data - whether it's good enough to show is decided separately by
    validate_quiz_question().

    Internal shape:
    {
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "correct_answer": "A",
        "evidence": "...",
        "explanation": "...",
    }

    NOTE: "correct_answer" here is only Ollama's own best guess, resolved
    via _resolve_correct_answer() purely so the dict has a well-formed
    shape while a question is in flight. It is never trusted as the truth
    - validate_quiz_question() independently determines and overwrites it
    (see _auto_classify_correct_answer()) before a question is ever shown.
    "evidence" (if the model still supplies it out of habit) is likewise
    kept only for backward-compatible shape; nothing in this file requires
    or validates it any more.
    """
    if not isinstance(raw, dict):
        return None

    question_text = raw.get("question")
    if not isinstance(question_text, str) or not question_text.strip():
        return None

    options = _normalize_options(raw.get("options"))
    if options is None:
        return None

    correct_answer = _resolve_correct_answer(
        raw.get("correct_answer") or raw.get("correct") or raw.get("answer"),
        options,
    )

    evidence = raw.get("evidence", "")
    if not isinstance(evidence, str):
        evidence = ""

    explanation = raw.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = ""

    return {
        "question": question_text.strip(),
        "options": options,
        "correct_answer": correct_answer,
        "evidence": evidence.strip(),
        "explanation": explanation.strip(),
    }


# --------------------------------------------------------------------------
# 3. VALIDATION - exactly the rules a question must satisfy to be shown
# --------------------------------------------------------------------------

def normalize_option_text(text):
    """Trim spaces, ignore upper/lower case, collapse repeated inner spaces."""
    if text is None:
        return ""
    return " ".join(str(text).split()).lower()


# Common short/filler words excluded when comparing the question, the
# correct answer, and each wrong option for shared vocabulary. Without
# this, ordinary connective words ("the", "is", "process", "used") that
# show up in almost any sentence would make nearly every option look
# "topic-related" no matter its actual subject, so the unrelatedness
# check below would reject good questions for the wrong reason.
_TOPIC_STOPWORDS = frozenset("""
a an the and or but of to in on at for with is are was were be been
being this that these those it its as by from into onto over under
about above below between among within without near far more most
less least very can could should would may might will shall do does
did each every any all both few many much other another such only
own same so than too also just now not no nor used use uses process
type kind system method way form part example
""".split())


def _tokenize(text):
    """
    Lowercase, alphabetic-only (3+ letter) word set, with common filler
    words removed. Used to compare the question/correct-answer wording
    against each wrong option so a wrong option that shares a real,
    meaningful word with the topic can be caught and rejected.
    """
    return set(re.findall(r"[a-zA-Z]{3,}", (text or "").lower())) - _TOPIC_STOPWORDS


def _option_is_near_duplicate(option_text_normalized, correct_text_normalized, threshold=0.82):
    """
    True when a WRONG option is such a close restatement of the correct
    option's own wording that it would let a student "guess" the correct
    answer by elimination - a paraphrase/near-duplicate, not merely an
    option that happens to share an ordinary word (e.g. "system",
    "energy", "process") with the correct answer or the study material.

    Uses a plain character-level similarity ratio (difflib.SequenceMatcher)
    on already-normalized (lowercased, whitespace-collapsed) option text -
    NOT a keyword/topic-overlap comparison - so this can never reject an
    option just because it contains a common word. A 0.82 ratio requires
    the two strings to be almost entirely the same characters in almost
    the same order (true paraphrases / minor rewordings), which real
    "completely unrelated, different-domain" distractors won't come
    close to.
    """
    if not option_text_normalized or not correct_text_normalized:
        return False
    ratio = difflib.SequenceMatcher(None, option_text_normalized, correct_text_normalized).ratio()
    return ratio >= threshold


def _correct_answer_grounded_in_content(correct_option_text, content):
    """
    The correct option must be directly supported by the study material,
    not invented by the model. This is a light-touch check (at least one
    shared meaningful word, or an exact substring match for very short
    answers) rather than a strict one, since paraphrasing is expected -
    but a "correct" answer that shares nothing at all with the source
    material is a strong sign the model drifted outside it.
    """
    content_tokens = _tokenize(content)
    option_tokens = _tokenize(correct_option_text)

    if option_tokens:
        return bool(option_tokens & content_tokens)

    stripped = (correct_option_text or "").strip().lower()
    return bool(stripped) and stripped in (content or "").lower()


def _randomize_option_order(question):
    """
    Reassign which letter (A/B/C/D) each option TEXT is attached to, at
    random, so the correct answer isn't always "A" (requirement 8). Only
    the letters move - no option text, evidence, or explanation is ever
    changed or invented. Position is tracked by index (not by matching
    text) so this stays correct even if two option texts were ever
    identical.
    """
    options = question.get("options")
    correct_answer = question.get("correct_answer")
    if not isinstance(options, dict) or correct_answer not in options:
        return question

    texts = [options[key] for key in OPTION_KEYS]
    correct_index = OPTION_KEYS.index(correct_answer)

    order = list(range(len(OPTION_KEYS)))
    random.shuffle(order)

    shuffled_texts = [texts[i] for i in order]
    new_correct_index = order.index(correct_index)

    randomized = dict(question)
    randomized["options"] = dict(zip(OPTION_KEYS, shuffled_texts))
    randomized["correct_answer"] = OPTION_KEYS[new_correct_index]
    return randomized


def _validate_quiz_question_with_reason(question, content):
    """
    Same checks as validate_quiz_question() (see that function for the
    full rule list), but returns (bool, reason) instead of just a bool -
    `reason` is a short human-readable string explaining exactly why a
    question was rejected (or "" on success). This exists only so the
    temporary QUIZ DEBUG output in _generate_one_question can report the
    exact validation reason for each attempt; validate_quiz_question()
    itself is kept as a plain bool-returning wrapper around this for
    backward compatibility with any other caller.

    Ollama's own declared "correct_answer" IS trusted here (already
    normalized to a plain A/B/C/D letter by _resolve_correct_answer /
    _read_option_letter inside _normalize_question) - it is checked, not
    replaced, and no second Ollama call is ever made to re-verify it.
    """
    if not isinstance(question, dict):
        return False, "parsed question is not a dict"

    question_text = question.get("question")
    if not isinstance(question_text, str) or not question_text.strip():
        return False, "question text is empty or missing"

    options = question.get("options")
    if not isinstance(options, dict) or set(options.keys()) != set(OPTION_KEYS):
        return False, "options must be a dict with exactly the keys A, B, C, D"

    normalized = {key: normalize_option_text(options[key]) for key in OPTION_KEYS}
    if any(not text for text in normalized.values()):
        return False, "one or more options is blank"

    # Exact-duplicate check only (case/whitespace-insensitive) - two
    # different-sounding-but-textually-distinct options are NOT rejected.
    if len(set(normalized.values())) != len(OPTION_KEYS):
        return False, "two options have the exact same text"

    explanation = question.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        return False, "explanation is empty or missing"

    # ---- correct_answer must already resolve to one of A/B/C/D (see
    # _resolve_correct_answer / _read_option_letter, which already
    # handle "A", "(A)", "Answer: A", "Option A", and the option's own
    # text echoed back). ----
    correct_answer = question.get("correct_answer")
    if correct_answer not in OPTION_KEYS:
        return False, "correct_answer could not be resolved to one of A/B/C/D"

    # ---- Requirement 7: the DECLARED correct option must be reasonably
    # supported by the study material - a light-touch check (at least one
    # shared meaningful word, or an exact substring match for very short
    # answers), checked ONLY for this one option, never all four. ----
    if not _correct_answer_grounded_in_content(options[correct_answer], content):
        return False, "declared correct answer is not supported by the study material"

    # ---- Requirement 8: the three distractors must not ALSO be obvious
    # answers to the question - i.e. a near-duplicate restatement of the
    # correct option (see _option_is_near_duplicate). This is a
    # character-level similarity check, NOT a keyword/topic-overlap
    # comparison, so a distractor is never rejected merely for sharing a
    # common word (e.g. "system", "energy", "process") with the correct
    # answer or the study material. ----
    for key in OPTION_KEYS:
        if key == correct_answer:
            continue
        if _option_is_near_duplicate(normalized[key], normalized[correct_answer]):
            return False, f"option {key} is a near-duplicate/obvious restatement of the correct answer"

    return True, ""


def validate_quiz_question(question, content):
    """
    True if `question` (already normalized) is safe to show to a student.
    A question is valid ONLY if:
      1. question text is a non-empty string.
      2. options is a dict with exactly the keys A, B, C, D.
      3. all four option values are non-empty strings.
      4. the four options are not duplicates of each other (same text,
         ignoring case/spacing - NOT "similar-sounding", just literally
         the same answer twice).
      5. correct_answer resolves to one of A/B/C/D.
      6. explanation is a non-empty string.
      7. the declared correct answer is reasonably supported by the study
         material (light-touch: shared meaningful word, or exact
         substring for very short answers).
      8. the three distractors are not near-duplicate/obvious
         restatements of the correct answer.

    correct_answer is Ollama's OWN declared letter (already normalized -
    see _resolve_correct_answer / _read_option_letter) - it is checked
    against the rules above, never independently recomputed or
    overwritten, and no second Ollama call is ever made to "verify" it.

    Rules 7 and 8 are deliberately NOT a keyword/topic-overlap comparison
    against every option: that used to reject a perfectly good distractor
    just because it shared an ordinary word (e.g. "system", "energy",
    "process") with the question or the study material. Rule 7 only
    checks the ONE declared-correct option; rule 8 only catches a
    distractor that is essentially the correct answer paraphrased, via
    character-level similarity, not shared vocabulary.

    A short explanation is still a VALID explanation - length is never
    checked here, only that it isn't empty.

    This is a thin wrapper around _validate_quiz_question_with_reason();
    see that function if you need the specific rejection reason (used
    internally for the temporary QUIZ DEBUG output).
    """
    is_valid, _reason = _validate_quiz_question_with_reason(question, content)
    return is_valid


# --------------------------------------------------------------------------
# 4. GENERATION + RETRY - one Ollama call per attempt, ONE question total
# --------------------------------------------------------------------------

def _print_quiz_debug(attempt_number, raw_reply, parse_status, parsed_data, validation_status, validation_reason):
    """
    TEMPORARY internal debugging (Step 2 of the root-cause investigation
    requested for this revision). Prints ONE structured block per
    generation attempt straight to the terminal/server console running
    `streamlit run` - never to the Streamlit UI, and never anything the
    student sees. This is what lets a developer tell, attempt by attempt,
    whether a failure was Ollama itself, JSON parsing, or validation -
    and exactly which validation rule tripped.

    Uses print() (not logger.debug) on purpose: it always reaches stdout
    regardless of how logging is configured, so it shows up unconditionally
    on the terminal, exactly where this is meant to be read from.

    `raw_reply` is truncated to its first 500 characters here, before
    printing - the point is a quick diagnostic, not dumping the entire
    model response, and this is still never shown to the student either
    way (see the module docstring / STEP 2 in the task this fulfils).
    """
    print("QUIZ DEBUG")
    print("----------")
    print(f"Attempt: {attempt_number}")
    print("Raw Ollama response:")
    print(raw_reply[:500] if isinstance(raw_reply, str) else "N/A")
    print()
    print("Parsing:")
    print(parse_status)
    print()
    print("Parsed data:")
    print(parsed_data if parsed_data is not None else "N/A")
    print()
    print("Validation:")
    print(validation_status)
    print()
    print("Validation reason:")
    print(validation_reason or "(none)")
    print()


def _generate_one_question(subject, topic, content, language_note, avoid_questions):
    """
    Ask Ollama for ONE new question, retrying up to MAX_GENERATION_ATTEMPTS
    times (a fresh call every time) if the reply can't be parsed or fails
    validation. The 2nd and 3rd attempts use a "strict" prompt with an
    extra, blunter reminder of the JSON-only and distractor-unrelatedness
    rules.

    Returns a valid, normalized question dict, or None if every attempt
    failed (or if `content` is empty/missing - see below).

    STABLE INTERFACE: steps/step21_practice.py (the Practice hub) calls
    this function directly with exactly this 5-argument signature. Do not
    change the name, argument order, or count without also updating that
    caller.

    STEP 3 (study source): `content` must be exactly
    st.session_state["study_material"]["content"], passed in by the
    caller (show_quiz() / step21_practice.py) - this function never reads
    session state or any other source itself. If it's missing or blank,
    Ollama is never called at all - this is checked first, below.

    STEP 2 (debugging): every attempt - including the "no study material"
    short-circuit - prints one QUIZ DEBUG block via _print_quiz_debug()
    to the terminal only. Nothing here is ever shown in the Streamlit UI.

    ARGUMENT SAFETY: `avoid_questions` may legitimately be None (e.g. the
    very first call, before any question has been asked yet) as well as
    an empty list - both are treated identically as "no history to avoid
    yet" and normalized to [] immediately below. This function never
    raises just because that argument is empty/None/falsy.
    """
    if not isinstance(content, str) or not content.strip():
        _print_quiz_debug(
            attempt_number=0,
            raw_reply=None,
            parse_status="N/A",
            parsed_data=None,
            validation_status="FAILED",
            validation_reason="Study material is missing.",
        )
        logger.debug("Study material is missing - Ollama was never called")
        return None

    # Normalize avoid_questions ONCE here (rather than relying solely on
    # _build_question_prompt's own guard) so every attempt in the loop
    # below, and any future code added to this function, sees a plain
    # list - never None - without needing to know this argument is
    # optional. This is the fix for the argument crash: previously,
    # avoid_questions=None reached "\n".join(... for q in avoid_questions)
    # inside _build_question_prompt and raised an uncaught
    # TypeError('NoneType' object is not iterable) - BEFORE the
    # try/except around generate_content() even ran, so it escaped this
    # function entirely. The Practice hub calls this function directly
    # with no try/except of its own, so that TypeError would crash the
    # whole Practice page instead of being treated as a normal failed
    # attempt.
    avoid_questions = avoid_questions or []

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        strict = attempt > 0  # attempts 2 and 3 (index 1, 2) are "strict"
        prompt = _build_question_prompt(
            subject, topic, content, language_note, avoid_questions, strict=strict
        )

        reply = None
        try:
            reply = generate_content(prompt)  # <- the ONLY function used to call Ollama
        except Exception as error:
            # A transient Ollama failure (e.g. a connection error, or an
            # HTTP 500 that survived ollama_service.py's own retries)
            # counts as a failed attempt here too - it is retried like any
            # other bad reply, never left to raise out of this function.
            # The Practice hub's "quiz" branch does not wrap this call in
            # its own try/except, so letting an exception escape here
            # would crash the whole Practice page instead of showing the
            # normal "Could not create a valid Quiz question" + Retry flow.
            _print_quiz_debug(
                attempt_number=attempt + 1,
                raw_reply=None,
                parse_status="N/A",
                parsed_data=None,
                validation_status="FAILED",
                validation_reason=f"generate_content raised an exception: {error}",
            )
            logger.debug("Attempt %d: generate_content failed (%s) - retrying", attempt + 1, error)
            continue

        raw = _load_json_object(reply)
        if raw is None:
            _print_quiz_debug(
                attempt_number=attempt + 1,
                raw_reply=reply,
                parse_status="FAILED",
                parsed_data=None,
                validation_status="N/A",
                validation_reason="reply could not be parsed as a JSON object",
            )
            logger.debug("Attempt %d: reply was not valid JSON - retrying", attempt + 1)
            continue

        candidate = _normalize_question(raw)
        if candidate is None:
            _print_quiz_debug(
                attempt_number=attempt + 1,
                raw_reply=reply,
                parse_status="SUCCESS",
                parsed_data=raw,
                validation_status="FAILED",
                validation_reason="JSON parsed, but missing a required field (question / 4 options)",
            )
            logger.debug("Attempt %d: JSON didn't have a usable shape - retrying", attempt + 1)
            continue

        # Validate FIRST - this checks (and trusts) the model's own
        # declared candidate["correct_answer"] letter, already normalized
        # by _resolve_correct_answer (see validate_quiz_question's
        # docstring). Only once that's known to be valid is it meaningful
        # to randomize which letter (A/B/C/D) it sits at (requirement:
        # correct answer isn't always A).
        is_valid, reason = _validate_quiz_question_with_reason(candidate, content)
        _print_quiz_debug(
            attempt_number=attempt + 1,
            raw_reply=reply,
            parse_status="SUCCESS",
            parsed_data=raw,
            validation_status="SUCCESS" if is_valid else "FAILED",
            validation_reason=reason,
        )
        if not is_valid:
            logger.debug("Attempt %d: question failed validation (%s) - retrying", attempt + 1, reason)
            continue

        candidate = _randomize_option_order(candidate)

        return candidate

    return None


def _generate_and_store_question(subject, topic, content):
    """
    Generate ONE fresh question and store it in session state, for the
    STANDALONE quiz page only (embedded=False). This is the entry point
    show_quiz() itself uses; the Practice hub does not call this - it
    calls _generate_one_question() directly and manages its own
    session-state keys.

    Shows its own st.error on failure. Returns True on success, False
    otherwise.
    """
    language_note = language_instruction()
    avoid_questions = list(st.session_state.get("quiz_seen_questions", []))[-RECENT_QUESTIONS_TO_AVOID:]

    try:
        question = _generate_one_question(subject, topic, content, language_note, avoid_questions)
    except Exception as e:
        st.error(f"⚠️ Something went wrong while generating the question. Details: {e}")
        return False

    if question is None:
        st.error("⚠️ Could not create a valid quiz question after 3 attempts. Please try again.")
        return False

    # ---- Store in session state (this is what makes the question
    # "sticky": selecting an option or a plain Streamlit rerun reads this
    # back instead of generating anything new). ----
    st.session_state.quiz_questions = [question]
    st.session_state.quiz_index = 0
    st.session_state.quiz_selected = {}
    st.session_state.quiz_submitted = {}
    st.session_state.quiz_correct = {}
    st.session_state.quiz_run = st.session_state.get("quiz_run", 0) + 1
    st.session_state.setdefault("quiz_seen_questions", [])
    st.session_state.quiz_seen_questions.append(question["question"])
    record_topic(topic)
    return True


# --------------------------------------------------------------------------
# 5. ANSWER CHECKING - always by option KEY, never by displayed text
# --------------------------------------------------------------------------

def _format_option(question, key):
    """Return the ORIGINAL option as shown to the student, e.g. 'A. Python'."""
    return f"{key}. {question['options'][key]}"


def _check_answer(question, selected_key):
    """True when the selected option key equals the stored correct_answer key."""
    return selected_key == question["correct_answer"]


# --------------------------------------------------------------------------
# 6. SESSION STATE
# --------------------------------------------------------------------------

def _init_session_state():
    """
    Set up every session-state key this page needs, but only the FIRST
    time (each `if key not in st.session_state` is a no-op on every later
    rerun) - so a plain rerun (selecting a radio option, Streamlit
    re-executing the script) never resets a question that's already
    loaded or already answered.
    """
    defaults = {
        "quiz_questions": None,
        "quiz_index": 0,
        "quiz_selected": {},
        "quiz_submitted": {},
        "quiz_correct": {},
        "quiz_run": 0,
        "quiz_seen_questions": [],
        "quiz_passed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# 7. PAGE (standalone use only - the Practice hub renders its own UI and
# never calls this function; see steps/step21_practice.py)
# --------------------------------------------------------------------------
def show_quiz(embedded=False):
    """
    Display the Quiz activity: ONE question, instant feedback, Pass/Fail.

    embedded=True (used by the Practice hub) skips this page's own header
    and topic bar, since the hub already shows those once above all five
    activity tabs.
    """
    material = require_study_material()
    if material is None:
        return

    subject = material["subject"]
    topic = material["topic"]
    content = material["content"]

    if not embedded:
        show_page_header("❓", "Quiz", "One multiple-choice question at a time, with instant feedback.")
        show_topic_bar(subject, topic)
        st.markdown("---")

    _init_session_state()

    # ---- Auto-generate: the very first time this is opened with nothing
    # loaded yet, ONE question is generated automatically - no "Generate
    # Quiz" button. quiz_questions stays set after this, so a plain rerun
    # (or picking an answer) never re-enters this block. ----
    if not st.session_state.quiz_questions:
        with st.spinner("Creating your quiz question from the study material..."):
            success = _generate_and_store_question(subject, topic, content)
        if not success:
            if st.button("🔁 Try Again", type="primary", key="quiz_retry_initial"):
                st.rerun()
            return
        st.rerun()

    question = st.session_state.quiz_questions[0]
    already_submitted = st.session_state.quiz_submitted.get(0, False)

    with st.container(border=True):
        st.markdown(f"### {question['question']}")

        # The radio's values are the option KEYS (A/B/C/D); format_func only
        # controls the text the student sees ("A. Python"). The widget key
        # includes quiz_run so a NEW question always gets a fresh,
        # unselected radio instead of inheriting the previous pick.
        selected_key = st.radio(
            "Choose your answer:",
            OPTION_KEYS,
            index=None,
            format_func=lambda key: _format_option(question, key),
            key=f"quiz_radio_run{st.session_state.quiz_run}",
            disabled=already_submitted,
        )

        # ---- Submit (only usable once per question, and only after
        # choosing). No Ollama call happens here - this is a pure
        # session-state update + a rerun to redraw the feedback below. ----
        if not already_submitted:
            if st.button("✅ Submit Answer", type="primary", disabled=selected_key is None, key="quiz_submit"):
                is_correct = _check_answer(question, selected_key)
                st.session_state.quiz_selected[0] = selected_key
                st.session_state.quiz_submitted[0] = True
                st.session_state.quiz_correct[0] = is_correct
                st.session_state.quiz_index = 1
                st.session_state.quiz_passed = is_correct or st.session_state.get("quiz_passed", False)
                log_answer(topic, is_correct, "Quiz")
                st.rerun()
        else:
            picked_key = st.session_state.quiz_selected.get(0)
            is_correct = st.session_state.quiz_correct.get(0, False)

            picked_text = (
                _format_option(question, picked_key)
                if picked_key in OPTION_KEYS
                else "No answer"
            )
            st.markdown(f"**Your Answer:**  \n{picked_text}")
            st.markdown(
                f"**Correct Answer:**  \n{_format_option(question, question['correct_answer'])}"
            )

            if is_correct:
                st.success("✅ Quiz Passed! Your flashcards are unlocked.")
            else:
                st.error("❌ Quiz Failed. Flashcards stay locked - give it another try.")

            if get_setting("show_explanations") and question["explanation"]:
                st.markdown(f"**Explanation:** {question['explanation']}")

    st.markdown("")

    # ---- Next-step controls (only shown after submitting; only these
    # buttons ever trigger a new Ollama call, and only one question at a
    # time, exactly as before). ----
    if already_submitted:
        is_correct = st.session_state.quiz_correct.get(0, False)

        if is_correct:
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("🃏 Open Flashcards", type="primary", use_container_width=True, key="quiz_open_flashcards"):
                    st.session_state.current_page = "📝 Flashcards"
                    st.rerun()
            with action_col2:
                if st.button("🔄 New Quiz Question", use_container_width=True, key="quiz_new_after_pass"):
                    with st.spinner("Creating a new question..."):
                        if _generate_and_store_question(subject, topic, content):
                            st.rerun()
        else:
            if st.button("🔄 Try Another Question", type="primary", key="quiz_retry_after_fail"):
                with st.spinner("Creating a new question..."):
                    if _generate_and_store_question(subject, topic, content):
                        st.rerun()