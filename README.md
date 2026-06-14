# Solar Flare Prediction System - Backend Architecture

## Overview
This is a production-ready, highly optimized FastAPI backend serving an AI-powered Solar Flare Prediction System. It is designed to run efficiently on an NVIDIA RTX 3050 (4GB VRAM) via TensorFlow memory growth limitation. 

The system performs Real-Time Data Fetching (NOAA Space Weather API), Dynamic Feature Engineering (Rolling Means, Lags, Differencing), Deep Learning Inference (LSTM), and XAI Interpretability (SHAP + LLM Integration via Groq).

## Core Architecture
- **Framework:** FastAPI
- **ML Framework:** TensorFlow/Keras (Singleton Model Loader)
- **Interpretability:** SHAP (Images) + Groq SDK (`llama-3.1-8b-instant`)
- **Data Pipeline:** Pandas / NumPy
- **Static Assets:** Hosted via `fastapi.staticfiles`

---

## Directory Structure
```
app/backend/
├── main.py                     # FastAPI root and GPU initialization
├── requirements.txt            
├── .env                        # Groq API keys
├── assets/                     # Static files (Real SHAP plots from Colab)
│   ├── shap_feature_importance.png
│   ├── shap_summary_plot.png
│   └── shap_metadata.json      
├── models/
│   └── final_full_vanilla_lstm_resumed.keras
├── routes/
│   ├── health.py               # Health checks
│   ├── live.py                 # Real-time NOAA inference
│   ├── explainability.py       # SHAP + LLM reasoning
│   └── predict.py              # CSV/JSON dataset endpoint
├── schemas/                    # Pydantic validation models
│   ├── prediction_schema.py
│   └── response_schema.py
├── services/
│   ├── model_loader.py         # Singleton instance with GPU logic
│   ├── feature_engineering.py  # Calculates Lags/Rolling windows
│   ├── live_data_service.py    # Fetches xrays/plasma/mag JSONs from NOAA
│   ├── sequence_generator.py   # Reshapes to (120, 25)
│   ├── shap_service.py         # Loads precomputed SHAP metadata
│   └── llm_explainer.py        # Generates LLM insights via Groq Llama 3.1
└── utils/
    ├── constants.py            
    └── logger.py               
```

---

## API Endpoints & Response Formats

### 1. `GET /health`
Verifies backend heartbeat and GPU allocation status.
**Response:**
```json
{
  "status": "healthy",
  "gpu_enabled": true,
  "timestamp": "2026-05-19T18:15:00Z"
}
```

### 2. `GET /live`
Fetches the last 7 days of NOAA data, joins X-ray, Plasma, and Magnetometer data in UTC, engineers 25 features, creates a `(1, 120, 25)` tensor, and generates a live prediction.

**Response:**
```json
{
  "status": "success",
  "timestamp": "2026-05-19T18:15:00.123456+00:00",
  "prediction": "Flare Likely",
  "flare_risk_level": "High",
  "probability": 0.84,
  "latest_conditions": {
    "xrsa_flux": 1.5e-07,
    "xrsb_flux": 2.2e-06,
    "Bz": -5.2,
    "proton_density": "..."
  }
}
```

### 3. `GET /explainability`
Returns the static SHAP plots (created during Colab training), top features, and an **AI-generated explanation** detailing the scientific reasoning for the prediction via the Groq API.
**Response:**
```json
{
  "status": "success",
  "prediction_model": "Vanilla LSTM",
  "forecast_window": "Next 2 Hours",
  "total_features": 25,
  "top_features": [
    {"feature": "Bz", "importance": 0.0014},
    {"feature": "log_xrsb", "importance": 0.0010},
    {"feature": "By", "importance": 0.0009},
    {"feature": "xrsb_lag_1", "importance": 0.0008},
    {"feature": "xrsb_roll_mean_5", "importance": 0.0007}
  ],
  "summary_plot": "/assets/shap_summary_plot.png",
  "importance_plot": "/assets/shap_feature_importance.png",
  "ai_explanation": "The prediction was primarily influenced by strong southward magnetic field activity (Bz). This, combined with elevated baseline X-ray flux intensity (log_xrsb), creates magnetic instability indicative of an impending solar flare.",
  "generated_at": "2026-05-19T18:15:00+00:00"
}
```

### 4. Static Asset Rendering
The frontend can directly call images rendered by the FastAPI static mount:
- `http://127.0.0.1:8000/assets/shap_feature_importance.png`
- `http://127.0.0.1:8000/assets/shap_summary_plot.png`
