from fastapi import APIRouter, HTTPException
from services.shap_service import ShapService
from services.llm_explainer import generate_ai_explanation

router = APIRouter()

@router.get("")
async def get_explainability():
    try:
        # 1. Load SHAP metadata
        explainability_data = ShapService.get_explainability()
        
        # 2. Extract top features
        top_features = explainability_data.get("top_features", [])
        
        # Mocking the latest prediction context for the explanation since it's static in this endpoint right now
        prediction_status = "Flare Likely"
        prob = 0.84
        
        # 3. Call generate_ai_explanation
        ai_exp = generate_ai_explanation(
            prediction=prediction_status, 
            probability=prob, 
            top_features=top_features
        )
        
        # 4. Return AI explanation in API response
        explainability_data["ai_explanation"] = ai_exp
        
        return explainability_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
