# phase1_scripter.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# --- 1. SETUP THE CLIENT ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    exit()

# --- 2. DEFINE THE NARRATIVE (User Input) ---
narrative = """
Panel 1: Close up on Tritoma, future empress of Amazon Lily, smiling brightly as she holds a box. Text above her exclaims "THERE ARE A HUNDRED TIMES MORE!!". Dialogue bubble points to her: "OH, THOSE ARE FOR LADY SHAKKY!!".
Panel 2: Medium shot over Tritoma's shoulder. Empress Gloriosa looks surprised, holding fan letters. Tritoma asks, "WHAT ABOUT ALL THOSE BACK THERE?". Gloriosa replies, "TRITOMA... EMPRESS GLORIOSA, HERE ARE YOUR FAN LETTERS!!".
Panel 3: Narration box explains Gloriosa was second only to Shakuyaku. Below, a wide, dramatic low-angle shot shows the massive pirate ship 'Gold Roger' crashing through stormy waves, kicking up spray. SFX indicates water sounds.
Panel 4: Small inset panel showing a tiny figure shouting from the Roger Pirates' ship. Dialogue: "IT'S ME!! ROGER SPEAKING!!".
Panel 5: Medium shot of a young Gloriosa on her ship's deck, looking flustered but excited towards the Roger Pirates' ship. Dialogue bubbles show her thoughts/exclamations: "O-OH... ♡ THAT'S... ♡", "CAPTAIN!! ENEMY AHEAD!!", "HE-HELLO THERE!!! AHEM!!". Another pirate shouts from below deck: "IT'S THE ROGER PIRATES!!". Another shouts: "HAND HER OVER, KUJA PIRATES!!".
""" # <<< END NARRATIVE >>>

# --- 3. THE UPDATED "RESEARCH" PROMPT (Shonen Manga Style) ---
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
# --- 4. CALL THE API ---
print(f"Calling Google Gemini API ({'gemini-1.5-flash'}) to generate script...")
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


#_______________________________________________________________________________________________________________---
# generate_reference_local_onepiece.py
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os

print("Starting One Piece Character Reference Image Generation...")

# --- 1. SETUP THE PIPELINE (Load Base Model) ---
print("Loading Stable Diffusion XL base model...")
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True # Corrected typo from previous script
    ).to("cuda")
    print("Base model loaded.")
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

# --- 2. DEFINE CHARACTER PROMPTS & FILENAMES ---

characters_to_generate = [
    # {
    #     "filename": "tritoma_reference.png",
    #     "prompt": "Close-up portrait of Tritoma from One Piece (future empress of Amazon Lily), young girl, dark short hair with bangs and headband, smiling brightly, facing forward, manga style by Eiichiro Oda, anime, monochrome, black and white line art, plain white background, high quality, sharp focus",
    #     "seed": 101 # Use different seeds for variation
    # }
    {
        "filename": "gloriosa_young_reference.png",
        "prompt": "Close-up portrait of young Gloriosa from One Piece (Amazon Lily empress), young woman, wavy blonde curly hair, slightly surprised/flustered expression, facing forward, manga style by Eiichiro Oda, anime, monochrome, black and white line art, plain white background, high quality, sharp focus",
        "seed": 102 # Use different seeds for variation
    }
]

negative_prompt = (
    "color, photograph, realistic, 3d render, blurry, ugly, "
    "low quality, worst quality, noisy, deformed, multiple characters, "
    "full body shot, holding objects, complex background, text, watermark, signature, words"
)

# --- 3. GENERATE IMAGES LOOP ---

# Enable slicing once
pipe.enable_attention_slicing()

for char_info in characters_to_generate:
    filename = char_info["filename"]
    prompt = char_info["prompt"]
    seed = char_info["seed"]

    print(f"\nGenerating reference image for: {filename}...")
    print(f"Prompt: '{prompt[:100]}...'")

    try:
        generator = torch.Generator("cuda").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35, 
            guidance_scale=8.0,    
            generator=generator
        ).images[0]

        # --- 4. SAVE THE IMAGE ---
        save_path = os.path.join("Reference Images", filename)
        image.save(save_path)
        print(f"Reference image saved successfully as '{save_path}'!")

    except Exception as e:
        print(f"\n--- ERROR generating {filename} ---")
        print(f"Error: {e}")
        if "CUDA out of memory" in str(e):
            print("CUDA Out of Memory! Try closing other applications.")
        # Continue to the next character

# pipe.disable_attention_slicing() # Can disable at the end if needed elsewhere

print("\n--- Reference Generation Complete ---")

#_______________________________________________________________________________________________________________---

# phase2_artist_local.py
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import json
import os
import glob # Import glob to find files
from ip_adapter import IPAdapterXL # Using NON-PLUS based on last successful load attempts
import gc # Garbage collector

print("Starting Phase 2: Local Image Generation...")

# --- 1. SETUP THE PIPELINE (Load Base Model, prepare for offloading) ---
# ...(Same as before)...
print("Loading Stable Diffusion XL base model...")
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    print("Base model partially loaded (CPU).")
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

# --- ENABLE MEMORY SAVING OPTIONS ---
print("Enabling memory optimizations (CPU offloading and attention slicing)...")
try:
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    print("Optimizations enabled.")
except Exception as e:
     print(f"Could not enable all optimizations: {e}")
     print("Attempting to move base pipe to GPU directly...")
     try: pipe.to("cuda"); print("Base pipe moved to GPU.")
     except Exception as cuda_e: print(f"FATAL: Could not move base pipe to GPU: {cuda_e}."); exit()

# --- 2. LOAD THE IP-ADAPTER ---
ip_model_path = "./ip-adapter_sdxl_vit-h.safetensors" # Using NON-PLUS weights file
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

if not os.path.exists(ip_model_path):
    print(f"Error: IP-Adapter weights '{os.path.basename(ip_model_path)}' not found.")
    exit()

print(f"Loading NON-PLUS IP-Adapter weights ({os.path.basename(ip_model_path)})...")
try:
    # Initialize adapter without explicit device when using offload
    ip_pipe = IPAdapterXL(pipe, image_encoder_path, ip_model_path)
    print("NON-PLUS IP-Adapter loaded.")
except Exception as e:
    print(f"Error loading IP-Adapter: {e}")
    exit()

# --- 3. AUTOMATICALLY FIND AND LOAD REFERENCE IMAGES ---
reference_images = {}
print("\nScanning for reference images (*_reference.png)...")
reference_files = glob.glob(os.path.join("Reference Images", "*_reference.png")) # Find all matching files

if not reference_files:
    print("Warning: No reference images found matching '*_reference.png'. Character consistency will be disabled.")
    # Proceeding without references might still be desired for some use cases
else:
    for ref_path in reference_files:
        try:
            # Extract character name from filename (e.g., "tritoma" from "tritoma_reference.png")
            char_name_key = os.path.basename(ref_path).replace("_reference.png", "")
            if char_name_key: # Ensure we got a name
                reference_images[char_name_key] = Image.open(ref_path).convert("RGB")
                print(f"Loaded reference image: {ref_path} for character key '{char_name_key}'")
            else:
                print(f"Warning: Could not extract character name from {ref_path}")
        except Exception as e:
            print(f"Error opening reference image {ref_path}: {e}")

    if not reference_images:
         print("Warning: Failed to load any valid reference images. Character consistency will be disabled.")

# --- 4. LOAD THE SCRIPT FROM PHASE 1 ---
script_path = "script.json"
# ...(Same script loading code as before)...
if not os.path.exists(script_path): print(f"Error: Script file '{script_path}' not found."); exit()
try:
    with open(script_path, "r", encoding='utf-8') as f: script_data = json.load(f)
    print(f"Loaded script data from: {script_path}")
except Exception as e: print(f"Error reading script.json: {e}"); exit()

# --- 5. THE MAIN GENERATION LOOP ---
panel_image_files = []
IP_ADAPTER_STRENGTH = 0.65
GENERATOR_SEED = 12345

print(f"\nStarting panel generation locally with IP-Adapter strength: {IP_ADAPTER_STRENGTH}")

for panel in script_data.get("panels", []):
    panel_num = panel.get("panel", "unknown")
    panel_prompt_desc = panel.get("description", "A manga panel.")

    print(f"\nGenerating panel {panel_num}...")
    print(f"Description: {panel_prompt_desc[:150]}...")

    # --- Determine which reference image to use (AUTOMATED) ---
    chosen_ref_image = None
    used_adapter = False
    description_lower = panel_prompt_desc.lower()

    # Iterate through the *discovered* character names (keys in the dictionary)
    if reference_images: # Only check if reference images were loaded
        for char_key, img in reference_images.items():
            # Check if any part of the key (split by '_') is in the description
            # This helps match "gloriosa" even if the key is "gloriosa_young"
            if any(name_part in description_lower for name_part in char_key.split('_')):
                chosen_ref_image = img
                print(f"Using reference for '{char_key}'.")
                used_adapter = True
                break # Use the first character match found

    if not used_adapter:
        print("No character keyword matched in description, or no references loaded. Generating without IP-Adapter.")
        chosen_ref_image = None

    try:
        # --- Prompt Setup (Same as before) ---
        full_prompt = f"manga art style by Eiichiro Oda, professional manga panel, (monochrome:1.5), black and white, ink drawing, detailed line art, high contrast shading, (no color:1.4), {panel_prompt_desc}"
        negative_prompt = "(color:1.5), colorful, vibrant, photograph, photo, realistic, 3d render, blurry, low quality, worst quality, noisy, deformed, ugly, disfigured, watermark, text, signature, words, letters"
        generator = torch.Generator("cpu").manual_seed(GENERATOR_SEED + panel_num) if GENERATOR_SEED is not None else None

        # --- Generate Image ---
        if used_adapter and chosen_ref_image:
             print(f"Generating with IP-Adapter...")
             image = ip_pipe.generate(
                pil_image=chosen_ref_image, prompt=full_prompt, negative_prompt=negative_prompt,
                scale=IP_ADAPTER_STRENGTH, num_samples=1, num_inference_steps=35,
                guidance_scale=7.5, generator=generator
            )[0]
        else:
            print("Generating with base pipeline...")
            image = pipe( # Use the base pipe (already setup for offloading)
                 prompt=full_prompt, negative_prompt=negative_prompt,
                 num_inference_steps=35, guidance_scale=7.5,
                 generator=generator
            ).images[0]

        # --- Save Image (Same as before) ---
        panel_filename = f"panel_{panel_num}.png"
        output_path = os.path.join("Generated Panels", panel_filename)
        image.save(output_path)
        panel_image_files.append(output_path)
        print(f"Successfully saved {output_path}")

        # --- Memory Cleanup (Same as before) ---
        del image, generator
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"--- Error generating panel {panel_num}: {e} ---")
        if "CUDA out of memory" in str(e): print("CUDA Out of Memory!")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

# --- END LOOP ---

print(f"\n--- Generation Complete ---")
if panel_image_files:
    print(f"Generated images: {', '.join(panel_image_files)}")
else:
    print("No panel images were generated successfully.")

#______________________________________________________________________________________________________________

# phase3_composer.py
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import json 

print("Starting Phase 3: Page Composition...")

# --- 1. Define Layout & Styling ---
PAGE_WIDTH, PAGE_HEIGHT = 850, 1200 
GUTTER = 20 
PANEL_BORDER_THICKNESS = 2
TEXT_PADDING = 10 
BUBBLE_CORNER_RADIUS = 12
BUBBLE_OUTLINE_THICKNESS = 2

# --- Font setup ---
# <<< Place arial.ttf or another .ttf font in your project folder >>>
try:
    font_path = "arial.ttf" # Or Comic Sans MS, etc.
    if not os.path.exists(font_path):
        font_path = ImageFont.load_default().path
        print(f"Warning: Font '{os.path.basename(font_path)}' not found in {os.getcwd()}, using Pillow default.")
    
    FONT_SIZE = 20 # Adjusted font size
    font = ImageFont.truetype(font_path, FONT_SIZE)
    
except Exception as e:
    print(f"Error loading font: {e}. Using default.")
    font = ImageFont.load_default()
    
# --- 2. Load Script for Dialogue ---
script_path = "script.json"
script_data = {}
if os.path.exists(script_path):
    try:
        with open(script_path, "r", encoding='utf-8') as f:
            script_data = json.load(f)
        print(f"Loaded script data from: {script_path}")
    except Exception as e:
        print(f"Error reading script.json: {e}. Proceeding without dialogue.")
else:
    print(f"Warning: Script file '{script_path}' not found. Proceeding without dialogue.")

panels_in_script = script_data.get("panels", [])
num_panels_expected = len(panels_in_script)

# --- 3. Determine Layout Grid (Simple Auto-Grid based on panel count) ---
if num_panels_expected <= 0:
    print("Error: No panels found in script.json. Cannot compose page.")
    exit()
elif num_panels_expected <= 3:
    PANELS_WIDE = 1
    PANELS_HIGH = num_panels_expected
elif num_panels_expected <= 4:
     PANELS_WIDE = 2
     PANELS_HIGH = 2
elif num_panels_expected <= 6:
     PANELS_WIDE = 2
     PANELS_HIGH = 3
else: # Max 8 panels in 2x4 grid
    PANELS_WIDE = 2
    PANELS_HIGH = 4
    print(f"Warning: More than 6 panels ({num_panels_expected}). Using a 2x{PANELS_HIGH} grid. Page might look crowded.")
    num_panels_expected = PANELS_WIDE * PANELS_HIGH # Limit processing


panel_width = (PAGE_WIDTH - (PANELS_WIDE + 1) * GUTTER) // PANELS_WIDE
panel_height = (PAGE_HEIGHT - (PANELS_HIGH + 1) * GUTTER) // PANELS_HIGH

print(f"Using layout grid: {PANELS_WIDE} wide x {PANELS_HIGH} high")
print(f"Calculated panel size: {panel_width}x{panel_height}")

# --- 4. Create Blank Page ---
page = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), 'white')
draw = ImageDraw.Draw(page)

# --- 5. Find Panel Image Files ---
panel_files = sorted(
    glob.glob(os.path.join("Generated Panels", "panel_*.png")), 
    key=lambda name: int(os.path.basename(name).split('_')[1].split('.')[0]) 
)

if not panel_files:
    print("Error: No panel image files (panel_*.png) found. Cannot compose page.")
    exit()

print(f"Found panel image files: {panel_files}")


# --- Helper Function for Wrapping Text ---
def wrap_text(text, font, max_width):
    lines = []
    if not text: return lines
    # Simple split first
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        words = paragraph.split(' ')
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Use textbbox for more accurate size in newer Pillow versions
            try: 
                 line_bbox = font.getbbox(test_line) 
                 line_width = line_bbox[2] - line_bbox[0]
            except AttributeError: # Fallback for older Pillow
                 line_width = font.getsize(test_line)[0]

            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line: # Append the line if it has words
                    lines.append(' '.join(current_line))
                current_line = [word] # Start new line with the current word
                # Handle very long words that exceed max_width (optional: could add hyphenation)
                try: 
                    word_bbox = font.getbbox(word)
                    word_width = word_bbox[2] - word_bbox[0]
                except AttributeError:
                    word_width = font.getsize(word)[0]

                if word_width > max_width:
                    lines.append(word) # Add the long word on its own line
                    current_line = [] # Reset line
                    
        if current_line: # Add the last line
           lines.append(' '.join(current_line))
           
    return [line for line in lines if line] # Filter out potential empty lines

# --- 6. Paste Panels and Add Dialogue ---
print("Composing manga page and adding dialogue...")
panel_index = 0
for row in range(PANELS_HIGH):
    for col in range(PANELS_WIDE):
        if panel_index >= len(panel_files) or panel_index >= num_panels_expected:
            break # Stop if we run out of images or exceed expected panels

        file_path = panel_files[panel_index]
        panel_info = panels_in_script[panel_index] if panel_index < len(panels_in_script) else {}
        panel_number_actual = panel_info.get("panel", panel_index + 1) 

        try:
            panel_img = Image.open(file_path).convert('RGB') 
            
            # Resize panel
            panel_img = panel_img.resize((panel_width, panel_height), Image.Resampling.LANCZOS) 
            
            # Calculate position based on grid row/col
            pos_x = GUTTER + col * (panel_width + GUTTER)
            pos_y = GUTTER + row * (panel_height + GUTTER)
            pos = (pos_x, pos_y)
            
            # Paste panel
            page.paste(panel_img, pos)
            
            # Draw border
            draw.rectangle(
                (pos[0]-PANEL_BORDER_THICKNESS, pos[1]-PANEL_BORDER_THICKNESS, 
                 pos[0] + panel_width+PANEL_BORDER_THICKNESS, pos[1] + panel_height+PANEL_BORDER_THICKNESS), 
                outline='black', width=PANEL_BORDER_THICKNESS
            )
            print(f"Pasted {os.path.basename(file_path)} at grid [{row},{col}] (Panel {panel_number_actual}).")

            # --- Add Dialogue Bubbles ---
            dialogues = panel_info.get("dialogue", [])
            if dialogues:
                dialogue_max_text_width = int(panel_width * 0.6) # Max width relative to panel
                bubble_margin = 15 # Margin from panel edges
                
                # Simple placement logic: alternate top-left, top-right, bottom-left, etc.
                bubble_base_x = pos[0] + bubble_margin
                bubble_base_y = pos[1] + bubble_margin
                if len(dialogues) > 1 and panel_index % 2 != 0: # Place second bubble top-rightish
                     bubble_base_x = pos[0] + panel_width - dialogue_max_text_width - bubble_margin * 2
                if len(dialogues) > 2 and panel_index % 2 == 0: # Third bubble bottom-leftish
                     bubble_base_y = pos[1] + panel_height - bubble_margin # We calculate height later
                if len(dialogues) > 3 and panel_index % 2 != 0: # Fourth bubble bottom-rightish
                     bubble_base_x = pos[0] + panel_width - dialogue_max_text_width - bubble_margin * 2
                     bubble_base_y = pos[1] + panel_height - bubble_margin

                current_bubble_y = bubble_base_y

                for d_idx, dialogue_entry in enumerate(dialogues):
                    character = dialogue_entry.get("character", "")
                    text = dialogue_entry.get("text", "").strip()
                    
                    if not text: continue

                    is_narrator_or_sfx = character.lower() in ["narrator", "sfx", "internal"]
                    
                    # Add character name if not Narrator/SFX/Internal
                    display_text = f"{character.upper()}:\n{text}" if not is_narrator_or_sfx else text
                    
                    wrapped_lines = wrap_text(display_text, font, dialogue_max_text_width)
                    
                    if not wrapped_lines: continue

                    # Calculate text block dimensions
                    text_height = 0
                    max_line_width = 0
                    line_heights = []
                    for line in wrapped_lines:
                        try: # Use textbbox if available
                             bbox = font.getbbox(line) 
                             line_width = bbox[2] - bbox[0]
                             l_h = bbox[3] - bbox[1] + 5 # Add line spacing
                        except AttributeError: # Fallback
                             size = font.getsize(line)
                             line_width = size[0]
                             l_h = size[1] + 5
                        line_heights.append(l_h)
                        text_height += l_h
                        if line_width > max_line_width: max_line_width = line_width
                    
                    text_height -= 5 # Remove spacing after last line

                    # Bubble dimensions
                    bubble_width = max_line_width + 2 * TEXT_PADDING
                    bubble_height = text_height + 2 * TEXT_PADDING

                    # Adjust position for bottom-anchored bubbles
                    if current_bubble_y > pos[1] + panel_height / 2: # Simple check if it's bottom half
                        bubble_y1 = current_bubble_y - bubble_height
                    else:
                        bubble_y1 = current_bubble_y

                    bubble_x1 = bubble_base_x
                    bubble_x2 = bubble_x1 + bubble_width
                    bubble_y2 = bubble_y1 + bubble_height

                    # Ensure bubble stays roughly within panel bounds (optional refinement)
                    bubble_x1 = max(pos[0] + 5, bubble_x1)
                    bubble_y1 = max(pos[1] + 5, bubble_y1)
                    bubble_x2 = min(pos[0] + panel_width - 5, bubble_x2)
                    bubble_y2 = min(pos[1] + panel_height - 5, bubble_y2)
                    # Recalculate width/height if bounds changed it significantly (optional)

                    # Draw bubble (Rounded for dialogue, Square for Narrator/SFX)
                    if is_narrator_or_sfx:
                        draw.rectangle( (bubble_x1, bubble_y1, bubble_x2, bubble_y2), 
                                         fill='white', outline='black', width=BUBBLE_OUTLINE_THICKNESS)
                    else:
                        draw.rounded_rectangle( (bubble_x1, bubble_y1, bubble_x2, bubble_y2), 
                                                radius=BUBBLE_CORNER_RADIUS, 
                                                fill='white', outline='black', width=BUBBLE_OUTLINE_THICKNESS)
                    
                    # Draw text inside
                    text_x = bubble_x1 + TEXT_PADDING
                    text_y = bubble_y1 + TEXT_PADDING
                    
                    for line_idx, line in enumerate(wrapped_lines):
                        draw.text((text_x, text_y), line, font=font, fill='black')
                        text_y += line_heights[line_idx] # Move by calculated line height + spacing

                    # Update placement for next potential bubble in same panel (simple vertical stack)
                    if current_bubble_y <= pos[1] + panel_height / 2:
                        current_bubble_y = bubble_y2 + 10 # Stack downwards
                    # (Add logic for horizontal placement if needed)

        except FileNotFoundError:
            print(f"Error: Could not find panel image file {file_path}.")
        except Exception as e:
            print(f"Error composing panel {file_path} (Panel {panel_number_actual}): {e}")

        panel_index += 1 # Move to the next panel image

# --- 7. Save Final Page ---
# --- 7. Save Final Page ---
final_page_path = os.path.join("Generated Page", "My_Manga_Page_OnePieceStyle.png")
try:
    page.save(final_page_path)
    print(f"\n--- SUCCESS! ---")
    print(f"Final manga page saved as: {final_page_path}")
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to save the final page: {e}")