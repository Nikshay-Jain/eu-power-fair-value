"""
Part 4: AI-Accelerated Workflow — Automated "Drivers" Commentary
This script programmatically generates a daily trader briefing by analyzing 
feature importance from Part 2 and trading signals from Part 3.

Requirements:
- pip install langchain-google-genai pandas python-dotenv
- A valid GOOGLE_API_KEY in your .env file
"""

import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ============================================================================
# Configuration & Logging 
# ============================================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OUTPUT_DIR = "results"
LOG_FILE = os.path.join(OUTPUT_DIR, "part4_ai_log.txt")
COMMENTARY_FILE = os.path.join(OUTPUT_DIR, "part4_trader_commentary.txt")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# Data Loading [cite: 10]
# ============================================================================
def load_model_context():
    """Extracts the specific metrics required for grounded LLM commentary."""
    try:
        # Load feature importance (Top 5) from Part 2 [cite: 3]
        importance_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part2_feature_importance.csv"))
        top_drivers = importance_df.head(5).to_dict(orient='records')

        # Load latest trading signal from Part 3 [cite: 3]
        signals_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part3_trading_signals.csv"))
        latest_signal = signals_df.iloc[0].to_dict()

        return {
            "top_drivers": top_drivers,
            "signal": latest_signal,
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        }
    except Exception as e:
        logging.error(f"Data loading failure: {e}")
        raise

# ============================================================================
# AI Narrative Generation [cite: 8, 12]
# ============================================================================
def generate_trader_commentary(context):
    """Programmatically generates grounded commentary using LangChain."""
    
    if not GOOGLE_API_KEY:
        error_msg = "GOOGLE_API_KEY missing. Ensure it is set in environment variables."
        logging.error(error_msg)
        return error_msg

    # Initialize Gemini Pro 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    # Prompt Template designed to prevent "hallucination" 
    template = """
    Role: Senior European Power Trader Analyst.
    Task: Write a 3-paragraph daily briefing for the desk based on model outputs.

    DATA CONTEXT (STRICT ADHERENCE):
    - Market: DE-LU (Germany-Luxembourg)
    - Date: {timestamp}
    - Signal: {signal_type}
    - Fair Value (P50): {p50_price} EUR/MWh
    - Forward Price: {fwd_price} EUR/MWh
    - Edge: {edge} EUR/MWh
    - Win Probability: {prob}%
    - Top 5 Model Drivers (by Gain): {drivers}

    INSTRUCTIONS:
    1. Paragraph 1: State the current signal and the magnitude of the "Fair Value" edge vs the forward curve.
    2. Paragraph 2: Explain the prediction using the Top 5 Drivers. Link the market fundamentals to the signal.
    3. Paragraph 3: Mention the risk/confidence level (Win Prob) and cite the underlying tables: 
       'part2_feature_importance.csv' and 'part3_trading_signals.csv'.
    
    CRITICAL: Do not invent any numbers. If a driver is listed, assume its influence is significant.
    """

    prompt = PromptTemplate(
        input_variables=["timestamp", "signal_type", "p50_price", "fwd_price", "edge", "prob", "drivers"],
        template=template
    )

    # Prepare inputs
    chain_input = {
        "timestamp": context["timestamp"],
        "signal_type": context["signal"]["signal"],
        "p50_price": round(context["signal"]["p50"], 2),
        "fwd_price": round(context["signal"]["forward_price"], 2),
        "edge": round(context["signal"]["edge"], 2),
        "prob": round(context["signal"]["probability"] * 100, 1),
        "drivers": ", ".join([d['feature'] for d in context["top_drivers"]])
    }

    try:
        logging.info(f"Invoking LLM with context: {chain_input}")
        response = llm.invoke(prompt.format(**chain_input))
        
        commentary = response.content
        logging.info("Commentary successfully generated.")
        
        return commentary

    except Exception as e:
        failure_msg = f"LLM Generation Failed: {e}"
        logging.error(failure_msg)
        return failure_msg

# ============================================================================
# Main Loop
# ============================================================================
def main():
    print("Starting Part 4: Automated Driver Commentary...")
    
    # 1. Load context
    context = load_model_context()
    
    # 2. Generate AI narrative
    briefing = generate_trader_commentary(context)
    
    # 3. Save output to file [cite: 14]
    with open(COMMENTARY_FILE, "w", encoding="utf-8") as f:
        f.write(briefing)
    
    print(f"✓ Commentary generated and saved to {COMMENTARY_FILE}")
    print(f"✓ AI audit logs written to {LOG_FILE}")
    print("\n--- PREVIEW OF DAILY BRIEFING ---")
    print(briefing)

if __name__ == "__main__":
    main()