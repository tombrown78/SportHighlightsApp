"""
Sports Highlights App - FastAPI Backend
Basketball video analysis with AI-powered player tracking and action recognition
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import videos, players, clips
from app.core.config import settings
from app.core.database import engine, Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Sports Highlights API...")
    logger.info(f"Processing mode: {settings.PROCESSING_MODE}")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created/verified")
    
    # Pre-load AI models for faster first inference
    from app.services.detector import PlayerDetector
    try:
        detector = PlayerDetector()
        logger.info("AI models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load AI models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sports Highlights API...")


# Create FastAPI app
app = FastAPI(
    title="Sports Highlights API",
    description="AI-powered basketball video analysis - player tracking, action recognition, and highlight generation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(videos.router, prefix="/api/videos", tags=["Videos"])
app.include_router(players.router, prefix="/api/players", tags=["Players"])
app.include_router(clips.router, prefix="/api/clips", tags=["Clips"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Sports Highlights API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "processing_mode": settings.PROCESSING_MODE
    }
