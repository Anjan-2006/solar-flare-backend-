from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from routes import health, predict, live, explainability
from services.model_loader import ModelLoader

# Robust path solution
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

# Ensure assets dir exists
ASSETS_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Solar Flare Prediction Backend...")
    try:
        ModelLoader.get_model()
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Solar Flare Backend",
    description="Local FastAPI setup for basic testing and actual model inference.",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount Routes
app.include_router(health.router)
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(live.router, prefix="/predict/live", tags=["Live Prediction"])
app.include_router(explainability.router, prefix="/explainability", tags=["Explainability"])
