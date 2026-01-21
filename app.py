import streamlit as st
import os
import subprocess

st.set_page_config(page_title="Power Fair Value Pipeline", layout="centered")
st.title("European Power Fair Value: End-to-End Pipeline")

RESULTS_DIR = "results"
DATA_DIR = "data"
ENV_FILE = ".env"

# --- 1. Gemini API Key Setup ---
st.header("1. Configure Gemini API Key")
gemini_key = st.text_input("Enter your Gemini API Key:", type="password")
if st.button("Save API Key"):
    with open(ENV_FILE, "w") as f:
        f.write(f"GOOGLE_API_KEY={gemini_key}\n")
    st.success("API key saved to .env file.")

# --- 2. Upload Data ZIPs ---
st.header("2. Upload Raw Data ZIP Files")
zip_files = st.file_uploader(
    "Upload 4 ZIP files (generation.zip, load.zip, market.zip, prices.zip)",
    type="zip", accept_multiple_files=True)

uploaded_names = [f.name for f in zip_files]
expected = ["generation.zip", "load.zip", "market.zip", "prices.zip"]

if len(zip_files) == 4 and all(e in uploaded_names for e in expected):
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in zip_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.read())
    st.success("All ZIP files uploaded and saved.")
    ready_for_pipeline = True
else:
    st.info("Please upload all 4 required ZIP files with correct names.")
    ready_for_pipeline = False

# --- 3. Run Pipeline ---
st.header("3. Run Full Pipeline")
run_pipeline = st.button("Run Pipeline (Preprocess → Forecast → Trading → LLM)")

if run_pipeline and ready_for_pipeline:
    st.info("Running part_1.py (data preprocessing)...")
    result1 = subprocess.run(["python", "part_1.py"], capture_output=True, text=True)
    st.text(result1.stdout)
    if result1.returncode != 0:
        st.error("part_1.py failed. Check logs.")
        st.stop()

    st.info("Running part_2.py (forecasting)...")
    result2 = subprocess.run(["python", "part_2.py"], capture_output=True, text=True)
    st.text(result2.stdout)
    if result2.returncode != 0:
        st.error("part_2.py failed. Check logs.")
        st.stop()

    st.info("Running part_3.py (trading signals)...")
    result3 = subprocess.run(["python", "part_3.py"], capture_output=True, text=True)
    st.text(result3.stdout)
    if result3.returncode != 0:
        st.error("part_3.py failed. Check logs.")
        st.stop()

    st.info("Running part_4.py (LLM commentary)...")
    result4 = subprocess.run(["python", "part_4.py"], capture_output=True, text=True)
    st.text(result4.stdout)
    if result4.returncode != 0:
        st.error("part_4.py failed. Check logs.")
        st.stop()

    st.success("Pipeline complete! Results saved in 'results/' directory.")

    # --- 4. Show LLM Output ---
    commentary_path = os.path.join(RESULTS_DIR, "part4_trader_commentary.txt")
    if os.path.exists(commentary_path):
        st.header("4. LLM Trader Commentary Output")
        with open(commentary_path, "r", encoding="utf-8") as f:
            commentary = f.read()
        st.text_area("Trader Commentary", commentary, height=250)
    else:
        st.warning("LLM commentary file not found.")

    # --- 5. Download Results ---
    st.header("5. Download Output Files")
    for fname in os.listdir(RESULTS_DIR):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                st.download_button(f"Download {fname}", f, file_name=fname)
else:
    st.info("Upload all required ZIPs and save API key to enable pipeline run.")
