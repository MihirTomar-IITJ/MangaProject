# MangaProject: AI-Powered Manga Generation Pipeline

MangaProject is an automated pipeline designed to transform text narratives into fully composed manga pages using Generative AI. It leverages Google's Gemini for scriptwriting and Stability AI's Stable Diffusion XL (with IP-Adapter) for consistent character generation.

## 🚀 Features

-   **Phase 1: Script Generation (`phase1_scripter.py`)**
    -   Converts a text narrative into a structural manga script (JSON format).
    -   Uses **Google Gemini 1.5 Flash** to analyze the story and breakdown it into panels with descriptions, dialogue, and camera angles.
    -   Ensures "One Piece" shonen style descriptions.

-   **Phase 2: Panel Generation (`phase2_artist.py`)**
    -   Generates high-quality manga panels based on the script.
    -   Uses **Stable Diffusion XL (SDXL)** for image synthesis.
    -   Integrates **IP-Adapter** to maintain character consistency across panels using reference images.
    -   Optimized for local execution with memory offloading.

-   **Phase 3: Page Composition (`phase3_composer.py`)**
    -   Assembles generated panels into a final manga page.
    -   Automatically calculates layout grids based on panel count.
    -   Adds dialogue bubbles and places text intelligently.
    -   Outputs the final image in a "One Piece" style format.

## 📋 Requirements

-   **Python 3.10+**
-   **CUDA-capable GPU** (Recommended for SDXL generation)
-   **Google Gemini API Key**

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MangaProject.git
    cd MangaProject
    ```

2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `torch` separately to ensure CUDA support.*

3.  **Setup Environment Variables:**
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

4.  **Download Models:**
    -   The script automatically attempts to download SDXL models.
    -   **IP-Adapter:** You need to manually place `ip-adapter_sdxl_vit-h.safetensors` in the root directory (or update the path in `phase2_artist.py`).

## 📖 Usage

 The pipeline is divided into three sequential phases. Run them in order:

### 1. Generate Script
Edit the `narrative` variable in `phase1_scripter.py` with your story, then run:
```bash
python phase1_scripter.py
```
*Output: `script.json`*

### 2. Generate Panels
Ensure you have reference images name as `{character}_reference.png` (e.g., `isagi_reference.png`) for character consistency. Then run:
```bash
python phase2_artist.py
```
*Output: `panel_1.png`, `panel_2.png`, etc.*

### 3. Compose Page
Run the composer to stitch everything together:
```bash
python phase3_composer.py
```
*Output: `My_Manga_Page_OnePieceStyle.png`*

## 📁 Project Structure

```
MangaProject/
├── main.py                  # Entry point (optional)
├── phase1_scripter.py       # Narrative -> JSON Script
├── phase2_artist.py         # JSON Script -> Panel Images
├── phase3_composer.py       # Panel Images -> Final Page
├── generate_reference.py    # Helper to generate character ref sheets
├── script.json              # Intermediate script file
├── *.png                    # Generated images and references
└── .env                     # API Keys
```

## ⚠️ Notes

-   **Character Consistency:** Uses IP-Adapter. Make sure to have clear reference images for best results.
-   **Fonts:** The composer looks for `arial.ttf`. If not found, it falls back to the default Pillow font.

## 📄 License

[MIT License](LICENSE)
