from fastapi import APIRouter
from services.model_loader import ModelLoader

router = APIRouter()

@router.get("/health")
def read_health():
    return {
        "status": "running"
    }

@router.get("/model-status")
def read_model_status():
    try:
        model = ModelLoader.get_model()
        is_loaded = model is not None
    except Exception:
        is_loaded = False
        
    return {
        "model_loaded": is_loaded,
        "model_name": "final_full_vanilla_lstm_resumed.keras" if is_loaded else None
    }
