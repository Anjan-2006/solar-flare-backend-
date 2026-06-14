import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timezone
from utils.logger import logger
from .model_loader import ModelLoader
from .preprocessing import Preprocessor
from .feature_engineering import FeatureEngineer
from .sequence_generator import SequenceGenerator

class PredictionService:
    THRESHOLD = 0.48
    MODEL_NAME = "Vanilla LSTM"

    @classmethod
    def get_confidence(cls, probability: float) -> str:
        if probability >= 0.80:
            return "High"
        elif probability >= 0.60:
            return "Moderate"
        else:
            return "Low"

    @classmethod
    def process_csv(cls, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Processing generic CSV for Prediction...")
        cleaned_df = Preprocessor.clean_data(df)
        featured_df = FeatureEngineer.generate_features(cleaned_df)
        scaled_data = Preprocessor.scale_features(featured_df, FeatureEngineer.FINAL_FEATURES)
        X_seq = SequenceGenerator.generate(scaled_data)
        
        model = ModelLoader.get_model()
        probabilities = model.predict(X_seq, batch_size=32)
        
        final_prob = float(probabilities[-1][0])
        prediction_text = "Flare Likely" if final_prob >= cls.THRESHOLD else "No Flare"
        logger.info(f"CSV Prediction generated: {prediction_text} (Prob: {final_prob:.4f})")
        
        return {
            "status": "success",
            "probability": round(final_prob, 4),
            "prediction": prediction_text,
            "confidence": cls.get_confidence(final_prob),
            "threshold": cls.THRESHOLD,
            "forecast_window": "Next 2 Hours",
            "prediction_model": cls.MODEL_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_sequences": len(X_seq)
        }

    @classmethod
    def predict_manual(cls, features: List[float]) -> Dict[str, Any]:
        logger.info("Running manual manual dummy prediction...")
        model = ModelLoader.get_model()
        X_test = np.array(features, dtype=np.float32).reshape(1, 1, 25)
        X_test_seq = np.tile(X_test, (1, 120, 1))

        probability = float(model.predict(X_test_seq, verbose=0)[0][0])
        prediction = "Flare Likely" if probability >= cls.THRESHOLD else "No Flare"
        logger.info(f"Manual Prediction generated: {prediction} (Prob: {probability:.4f})")
        
        return {
            "status": "success",
            "probability": round(probability, 4),
            "prediction": prediction,
            "confidence": cls.get_confidence(probability),
            "threshold": cls.THRESHOLD,
            "forecast_window": "Next 2 Hours",
            "prediction_model": cls.MODEL_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
