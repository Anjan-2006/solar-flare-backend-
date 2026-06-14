from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class PredictionResponse(BaseModel):
    status: str
    probability: float
    prediction: str
    confidence: str
    threshold: float
    forecast_window: str
    prediction_model: str
    generated_at: str
    num_sequences: Optional[int] = None
    source: Optional[str] = None
    latest_conditions: Optional[Dict[str, float]] = None
    message: Optional[str] = None
