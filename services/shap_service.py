import json
import os
from datetime import datetime, timezone
from utils.logger import logger

class ShapService:
    ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
    METADATA_PATH = os.path.join(ASSETS_DIR, "shap_metadata.json")

    @classmethod
    def get_explainability(cls) -> dict:
        logger.info("Serving Explainability data...")
        if os.path.exists(cls.METADATA_PATH):
            with open(cls.METADATA_PATH, "r") as f:
                base_data = json.load(f)
                
            # Injecting dynamic status and production wrappers requested by the user
            base_data["status"] = "success"
            base_data["prediction_model"] = "Vanilla LSTM"
            base_data["forecast_window"] = "Next 2 Hours"
            base_data["generated_at"] = datetime.now(timezone.utc).isoformat()
            
            return base_data
        else:
            logger.error("SHAP metadata file is missing.")
            raise FileNotFoundError("SHAP metadata not found. Please run SHAP precomputation.")
