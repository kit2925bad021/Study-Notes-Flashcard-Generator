"""
diagnose_study_material.py - finds WHERE Study Material fails.

Run from the project folder (the one that contains app.py):
    python diagnose_study_material.py

It repeats the exact Gemini calls Study Material makes, one variable at a
time, and prints the REAL exception for the first one that breaks.
The API key is never printed.

  STAGE 1  same call as test_gemini.py (prompt only, no config)
  STAGE 2  RAW SDK call + temperature config (no gemini_service wrapper)
  STAGE 3  gemini_service.generate_content() + temperature config
  STAGE 4  the REAL Summary prompt from step20_study_flow (tiny topic)
  STAGE 5  the REAL Detailed Notes prompt with a ~6000-character source
           (the biggest prompt Study Material can send)
"""

import time
import traceback

from services import gemini_service
from services.gemini_service import _redact, generate_content

results = []


def stage(name, func):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    started = time.monotonic()
    try:
        text = func()
        seconds = time.monotonic() - started
        print(f"PASS in {seconds:.1f}s - {len(text)} characters returned")
        print("First 150 chars:", text[:150].replace("\n", " "))
        results.append((name, "PASS", ""))
    except Exception as error:
        seconds = time.monotonic() - started
        print(f"FAIL after {seconds:.1f}s")
        print("Exception type   :", type(error).__module__ + "." + type(error).__name__)
        print("Exception message:", _redact(error))
        for attr in ("code", "status", "message", "details"):
            value = getattr(error, attr, None)
            if value:
                print(f"  {attr:<8}: {_redact(value)}")
        print(_redact(traceback.format_exc()))
        results.append((name, "FAIL", type(error).__name__))


print("Model in use :", gemini_service.MODEL_NAME)
print("Timeout (ms) :", gemini_service.REQUEST_TIMEOUT_MS)
print("Key present  :", bool((__import__("os").getenv("GEMINI_API_KEY") or "").strip()))

SIMPLE_PROMPT = "Explain photosynthesis in two simple sentences."


# STAGE 1 ---------------------------------------------------------------
stage("STAGE 1: same as test_gemini.py (no config)",
      lambda: generate_content(SIMPLE_PROMPT))


# STAGE 2 ---------------------------------------------------------------
def raw_call_with_temperature():
    from google.genai import types
    client = gemini_service._get_client()
    response = client.models.generate_content(
        model=gemini_service.MODEL_NAME,
        contents=SIMPLE_PROMPT,
        config=types.GenerateContentConfig(temperature=0.35),
    )
    return (response.text or "").strip() or "(empty)"


stage("STAGE 2: RAW SDK call + temperature=0.35", raw_call_with_temperature)


# STAGE 3 ---------------------------------------------------------------
stage("STAGE 3: gemini_service + temperature=0.35 (what Study Material passes)",
      lambda: generate_content(SIMPLE_PROMPT, options={"num_ctx": 4096, "temperature": 0.35}))


# STAGES 4 and 5 use the real prompts from step20 ------------------------
try:
    from steps import step20_study_flow as s20
    have_step20 = True
except Exception as error:
    have_step20 = False
    print("\nCould not import steps.step20_study_flow:", type(error).__name__, _redact(error))

if have_step20:
    subject, topic = "Computer Science", "Photosynthesis basics"

    stage("STAGE 4: real Summary prompt (step20) + GEN_OPTIONS",
          lambda: generate_content(
              s20.prompt_summary(subject, topic, "", "Use simple English."),
              options=s20.GEN_OPTIONS))

    big_source = ("Photosynthesis converts light energy into chemical energy in plants. "
                  "Chlorophyll in the chloroplast absorbs light. Water and carbon dioxide "
                  "become glucose and oxygen. ") * 60
    big_source = big_source[: s20.MAX_SOURCE_CHARS]
    fake_summary = "Photosynthesis is how plants make food from light. " * 12
    fake_points = "\n".join(f"- Fact number {i} about photosynthesis" for i in range(8))
    fake_concepts = "\n".join(f"- Concept {i} - an idea about photosynthesis" for i in range(4))

    stage("STAGE 5: real Detailed Notes prompt (~6000-char source) + DETAILED_NOTES_OPTIONS",
          lambda: generate_content(
              s20.prompt_detailed_notes(subject, topic, big_source, "Use simple English.",
                                        fake_summary, fake_points, fake_concepts),
              options=s20.DETAILED_NOTES_OPTIONS))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, outcome, error_name in results:
    print(f"{outcome:<5} {name}  {error_name}")
print("\nThe FIRST stage marked FAIL is where Study Material breaks.")
print("Copy the 'Exception type' and 'Exception message' lines of that stage.")