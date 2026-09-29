"""
test_gemini.py - quick check that Gemini works the same way the app uses it.

Run from the project folder (the one that contains app.py):
    python test_gemini.py

It reads GEMINI_API_KEY from the .env file (never hard-code the key here)
and sends one small prompt through services/gemini_service.py.
"""

from services import gemini_service

print("Model  :", gemini_service.MODEL_NAME)
print("Sending request to Gemini...")

try:
    answer = gemini_service.generate_content(
        "Explain photosynthesis in two simple sentences."
    )
    print("\nGemini response:")
    print(answer)
    print("\nRESULT: PASS - Gemini is working.")
except Exception as error:
    print("\nRESULT: FAIL - Gemini request failed.")
    print(str(error))