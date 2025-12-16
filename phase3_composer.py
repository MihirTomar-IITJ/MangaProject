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

# --- 3. Determine Layout Grid  ---
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
                        try: 
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

                    # Ensure bubble stays roughly within panel bounds 
                    bubble_x1 = max(pos[0] + 5, bubble_x1)
                    bubble_y1 = max(pos[1] + 5, bubble_y1)
                    bubble_x2 = min(pos[0] + panel_width - 5, bubble_x2)
                    bubble_y2 = min(pos[1] + panel_height - 5, bubble_y2)
                    # Recalculate width/height if bounds changed it significantly 

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