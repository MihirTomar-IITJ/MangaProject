# phase2_artist_local.py
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import json
import os
import glob 
from ip_adapter import IPAdapterXL 
import gc 

print("Starting Phase 2: Local Image Generation...")

# --- 1. SETUP THE PIPELINE (Load Base Model, prepare for offloading) ---
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
ip_model_path = "./ip-adapter_sdxl_vit-h.safetensors"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

if not os.path.exists(ip_model_path):
    print(f"Error: IP-Adapter weights '{os.path.basename(ip_model_path)}' not found.")
    exit()

print(f"Loading NON-PLUS IP-Adapter weights ({os.path.basename(ip_model_path)})...")
try:
    ip_pipe = IPAdapterXL(pipe, image_encoder_path, ip_model_path,device="cuda")
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
    
else:
    for ref_path in reference_files:
        try:
            # Extract character name from filename (e.g., "tritoma" from "tritoma_reference.png")
            char_name_key = os.path.basename(ref_path).replace("_reference.png", "")
            if char_name_key:
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

    if reference_images: # Only check if reference images were loaded
        for char_key, img in reference_images.items():
            if any(name_part in description_lower for name_part in char_key.split('_')):
                chosen_ref_image = img
                print(f"Using reference for '{char_key}'.")
                used_adapter = True
                break 

    if not used_adapter:
        print("No character keyword matched in description, or no references loaded. Generating without IP-Adapter.")
        chosen_ref_image = None

    try:
        full_prompt = f"manga art style by Eiichiro Oda, professional manga panel, (monochrome:1.5), black and white, ink drawing, detailed line art, high contrast shading, (no color:1.4), {panel_prompt_desc}"
        negative_prompt = "(color:1.5), colorful, vibrant, photograph, photo, realistic, 3d render, blurry, low quality, worst quality, noisy, deformed, ugly, disfigured, watermark, text, signature, words, letters"
        generator = torch.Generator("cpu").manual_seed(GENERATOR_SEED + panel_num) if GENERATOR_SEED is not None else None

        # --- Generate Image ---
        if used_adapter and chosen_ref_image:
             print(f"Generating with IP-Adapter...")
             image = ip_pipe.generate(
                pil_image=chosen_ref_image, prompt=full_prompt, negative_prompt=negative_prompt,
                scale=IP_ADAPTER_STRENGTH, num_samples=1, num_inference_steps=35,
                guidance_scale=7.5, 
            )[0]
        else:
            print("Generating with base pipeline...")
            image = pipe( # Use the base pipe (already setup for offloading)
                 prompt=full_prompt, negative_prompt=negative_prompt,
                 num_inference_steps=35, guidance_scale=7.5,
                 
            ).images[0]

        # --- Save Image ---
        panel_filename = f"panel_{panel_num}.png"
        output_path = os.path.join("Generated Panels", panel_filename)
        image.save(output_path)
        panel_image_files.append(output_path)
        print(f"Successfully saved {output_path}")

        # --- Memory Cleanup  ---
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