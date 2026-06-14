from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from utils.logger import logger
from services.live_data_service import LiveDataService
from services.prediction_service import PredictionService
from services.preprocessing import Preprocessor
from services.feature_engineering import FeatureEngineer
from services.sequence_generator import SequenceGenerator
from services.model_loader import ModelLoader

router = APIRouter()

@router.get("")
async def predict_live():
    logger.info("Initiating Live NOAA Prediction task...")
    try:
        try:
            df = LiveDataService.fetch_live_data()
            logger.info("Successfully fetched NOAA metrics.")
        except Exception as e:
            logger.error(f"NOAA Extractor failure: {e}")
            return JSONResponse(status_code=503, content={"status": "error", "message": f"NOAA Data Fetch Error: {str(e)}"})
        
        cleaned_df = Preprocessor.clean_data(df)
        featured_df = FeatureEngineer.generate_features(cleaned_df)
        
        # Grab latest raw conditions for the polished response before scaling wipes values
        latest_series = cleaned_df.iloc[-1]
        latest_conditions = {
            "xrsb_flux": float(latest_series.get('xrsb_flux', 0.0)),
            "Bz": float(latest_series.get('Bz', 0.0)),
            "flow_speed": float(latest_series.get('flow_speed', 0.0))
        }

        scaled_data = Preprocessor.scale_features(featured_df, FeatureEngineer.FINAL_FEATURES)
        X_seq = SequenceGenerator.generate(scaled_data)
        latest_seq = X_seq[-1:]
        
        model = ModelLoader.get_model()
        probability = float(model.predict(latest_seq, verbose=0)[0][0])
        
        threshold = PredictionService.THRESHOLD
        prediction_text = "Flare Likely" if probability >= threshold else "No Flare"
        confidence = PredictionService.get_confidence(probability)
        
        logger.info(f"Live Prediction Result: {prediction_text} ({probability:.4f})")
        
        return {
            "status": "success",
            "probability": round(probability, 4),
            "prediction": prediction_text,
            "confidence": confidence,
            "threshold": threshold,
            "forecast_window": "Next 2 Hours",
            "prediction_model": PredictionService.MODEL_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "NOAA Live Data",
            "latest_conditions": latest_conditions
        }
    except Exception as e:
        logger.exception("Unexpected error in live prediction pipeline.")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Prediction Pipeline Error: {str(e)}"})
