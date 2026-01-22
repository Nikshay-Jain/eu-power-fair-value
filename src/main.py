import logging
from part_1 import main as run_part_1
from part_2 import main as run_part_2
from part_3 import main as run_part_3
from part_4 import main as run_part_4

# Centralized Logging for User Tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def run_pipeline():
    logging.info("STARTING EUROPEAN POWER FAIR VALUE PIPELINE")
    
    try:
        # Part 1: Data Ingestion & QA
        logging.info("--- PART 1: INGESTION & QUALITY ASSURANCE ---")
        run_part_1()
        
        # Part 2: Forecasting
        logging.info("--- PART 2: ML FORECASTING & VALIDATION ---")
        run_part_2()
        
        # Part 3: Trading Strategy
        logging.info("--- PART 3: PROMPT CURVE SIGNALS ---")
        run_part_3()
        
        # Part 4: AI Workflow
        logging.info("--- PART 4: AI TRADER COMMENTARY ---")
        run_part_4()
        
        logging.info("✨ PIPELINE COMPLETE. View results in the /results folder.")

    except Exception as e:
        logging.error(f"❌ PIPELINE FAILED: {str(e)}")

if __name__ == "__main__":
    run_pipeline()