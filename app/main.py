"""
PhishLens - High-Precision Scam & Phishing Detection Application
FastAPI entry point serving REST API and Neo-Brutalist web interface.
"""

from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.config import settings
from app.ml.classifier import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("phishlens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on application startup."""
    logger.info("Initializing PhishLens application...")
    success = model_manager.load_artifacts(
        classifier_path=settings.model_path,
        vectorizer_path=settings.vectorizer_path,
        metadata_path=settings.metadata_path
    )
    if not success:
        logger.warning(
            "Model artifacts could not be loaded on startup. "
            "Ensure training has been executed via: python training/train.py"
        )
    yield
    logger.info("PhishLens application shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Precision-Oriented Scam & Smishing Detection System with OCR & ML Inference",
    lifespan=lifespan
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, tags=["Scam Detection"])

# Static UI directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "PhishLens API is running. Access /docs for Swagger UI."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
