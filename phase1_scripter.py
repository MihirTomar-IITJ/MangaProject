# phase1_scripter.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# --- 1. THE CLIENT ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    exit()

# --- 2. THE NARRATIVE (User Input) ---
narrative = """
Panel 1: A confused Isagi stands in training gear, eyes widening as he reacts to being called out. 
He hesitates, muttering: “…Huh?”

Panel 2: Isagi expression tightens, trying to grasp the weight of the question thrown at him.
He repeats softly, almost to himself: “You mean… like Zico… or Maradona?”

Panel 3: Low-angle shot of Ego Jinpachi standing under the night sky, hands in his pockets, posture unnervingly composed. 
His glasses shine as he stares down at Isagi. With a voice that cuts through the silence, he asks:
“Yoichi Isagi… do you believe there is a God of soccer?”

""" # <<< END NARRATIVE >>>

# --- 3. THE  "RESEARCH" PROMPT (Shonen Manga Style) ---
MANGA_SYSTEM_PROMPT = """You are a master manga scriptwriter, channeling the energy and style of dynamic shonen manga like One Piece. Your task is to break down a narrative into a visually rich, panel-by-panel script for ONE manga page (typically 4-6 panels, but prioritize story flow).

**CRITICAL STYLE REQUIREMENTS:**
* **Strictly Black & White:** All descriptions MUST imply a high-contrast, black and white, inked line art style. Mention shadows, hatching, and sharp details. NO COLOR.
* **Dynamic Visuals:** Describe varied camera angles (dramatic close-ups, impactful wide shots, dynamic low/high angles). Emphasize expressive character emotions (exaggerated surprise, anger, joy) and actions (speed lines, impact effects).
* **Panel Emphasis (Hints):** HINT at panel importance (e.g., "Large establishing shot," "Small reaction panel").
* **Dialogue & Narration:** Capture dialogue accurately. Use "Narrator", "SFX", "Internal".

**OUTPUT FORMAT (Strict JSON Only):**
Output ONLY a valid JSON object.
The JSON must have a single "panels" key containing an array of panel objects.

**Each panel object MUST include:**
-   "panel": (Integer) The panel sequence number.
-   "description": (String) **CONCISE** but DETAILED visual description (**keep under 60 words**). Include:
    -   Camera angle and shot type.
    -   Character pose, action, and specific facial expression.
    -   Relevant setting details.
    -   Implied action lines, effects, or shading.
    -   Explicit mention of "black and white ink style," "heavy shadows," etc.
-   "dialogue": (Array of Objects) Each object: {"character": (String), "text": (String)}. Empty array `[]` if none.

**Example Panel Structure:**
{
  "panel": 3,
  "description": "Impactful wide shot, low angle. Pirate ship 'Gold Roger' crashes through stormy waves, spray rendered with sharp black ink lines. Speed lines show momentum. Heavy shadows under figurehead. Black and white ink style.",
  "dialogue": [
    {"character": "Lookout", "text": "Captain!! Enemy ahead!!"},
    {"character": "SFX", "text": "SPLOOOOSH!"}
  ]
}

Now, process the user's narrative according to these instructions.
"""
# --- 4. CALLING API ---
print(f"Calling Google Gemini API ({'gemini-2.5-flash'}) to generate script...")
try:
    generation_config = {"response_mime_type": "application/json"}
    safety_settings = {'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE', 'SEXUAL' : 'BLOCK_NONE', 'DANGEROUS' : 'BLOCK_NONE'}
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", 
        generation_config=generation_config, 
        safety_settings=safety_settings, 
        system_instruction=MANGA_SYSTEM_PROMPT
    )
    response = model.generate_content(narrative)

    # --- 5. PARSE AND SAVE THE RESPONSE ---
    if not response.text:
         print("Error: No text content received from the API.")
         print(f"Prompt Feedback: {response.prompt_feedback}")
         exit()

    script_json_string = response.text
    if script_json_string.strip().startswith("```json"):
        script_json_string = script_json_string.strip()[7:-3].strip()
    elif script_json_string.strip().startswith("```"):
         script_json_string = script_json_string.strip()[3:-3].strip()

    script_data = json.loads(script_json_string)
    print("\n--- SCRIPT GENERATED SUCCESSFULLY ---")
    output_filename = "script.json"
    with open(output_filename, "w", encoding='utf-8') as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)
    print(f"Script saved to {output_filename}")

except json.JSONDecodeError as e:
    print(f"Error: Failed to parse the API response as JSON: {e}")
    print("----- Raw API Response -----")
    print(script_json_string)
    print("----------------------------")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    try:
        print(f"Prompt Feedback: {response.prompt_feedback}")
    except NameError:
        pass