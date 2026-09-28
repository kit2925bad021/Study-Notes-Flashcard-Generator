from services.gemini_service import generate_content


prompt = """
Explain photosynthesis in exactly 3 simple sentences
for a college student.
"""


print("Sending request to Gemini...")

result = generate_content(prompt)

print("\n--- Gemini Response ---\n")
print(result)