from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
from utils.logger import logger
from schemas.prediction_schema import ManualInputSchema
from schemas.response_schema import PredictionResponse
from services.prediction_service import PredictionService

router = APIRouter()

@router.post("/manual", response_model=PredictionResponse)
async def predict_manual(request: ManualInputSchema):
    try:
        result = PredictionService.predict_manual(request.dummy_features)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Manual prediction failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.post("/csv", response_model=PredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    logger.info(f"Received CSV Upload: {file.filename}")
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Only CSV files are allowed."})
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        result = PredictionService.process_csv(df)
        return PredictionResponse(**result)
        
    except ValueError as val_e:
        logger.warning(f"Validation error on CSV: {val_e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": str(val_e)})
    except Exception as e:
        logger.exception("CSV Processing failed.")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Error processing CSV: {str(e)}"})
