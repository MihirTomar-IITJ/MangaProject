# generate_reference_local_onepiece.py
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os

print("Starting One Piece Character Reference Image Generation...")

# --- 1. Base Model ---
print("Loading Stable Diffusion XL base model...")
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True 
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
    #     "seed": 101
    # }
    {
        "filename": "isagi_reference.png",
"prompt": "Close-up portrait of Yoichi Isagi from Blue Lock, teenage boy with short dark messy hair and sharp eyes, determined yet slightly confused expression, facing forward, manga style by Muneyuki Kaneshiro & Yusuke Nomura, black and white line art, monochrome, plain white background, high quality, sharp focus",
"seed": 102
 }
]

negative_prompt = (
    "color, photograph, realistic, 3d render, blurry, ugly, "
    "low quality, worst quality, noisy, deformed, multiple characters, "
    "full body shot, holding objects, complex background, text, watermark, signature, words"
)

# --- 3. GENERATE IMAGES ---

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