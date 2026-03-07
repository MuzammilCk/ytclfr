import os
import json
from google import genai
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.intelligence.llm_brain import _SYSTEM_PROMPT

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=_SYSTEM_PROMPT + "\n\nVIDEO TITLE: moonboy\nTAGS: (none)\n\n=== FRAME OCR TEXT (PRIMARY SOURCE) ===\n[00:10] JVKE - moonboy\n\n=== AUDIO TRANSCRIPT (SUPPLEMENTARY) ===\n(silent video — no spoken content)\n\n=== VIDEO DESCRIPTION (CONTEXT) ===\n(no description)\n\nAnalyze the above and return the JSON result.",
        config={"response_mime_type": "application/json", "temperature": 0.1}
    )
    print("RAW RESPONSE START")
    print(repr(res.text))
    print("RAW RESPONSE END")
    try:
        data = json.loads(res.text)
        print("JSON PARSE SUCCESS. Keys:", data.keys())
    except Exception as e:
        print("JSON PARSE FAILED:", e)
except Exception as e:
    print(f"Error: {e}")
