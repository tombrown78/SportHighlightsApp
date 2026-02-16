"""Application configuration"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database
    DATABASE_URL: str = "postgresql://sports:sports123@localhost:5432/sports_highlights"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Processing mode: "local" or "cloud"
    PROCESSING_MODE: str = "local"
    
    # Cloud provider settings (only used if PROCESSING_MODE=cloud)
    CLOUD_PROVIDER: Optional[str] = None
    REPLICATE_API_TOKEN: Optional[str] = None
    ROBOFLOW_API_KEY: Optional[str] = None
    RUNPOD_API_KEY: Optional[str] = None
    
    # File paths
    VIDEOS_DIR: str = "/app/videos"
    CLIPS_DIR: str = "/app/clips"
    MODELS_DIR: str = "/app/models"
    
    # YouTube Authentication (optional - for private/age-restricted videos)
    YOUTUBE_USE_OAUTH: bool = False  # Set to True to use OAuth2
    YOUTUBE_USERNAME: Optional[str] = None  # Legacy: email (not recommended)
    YOUTUBE_PASSWORD: Optional[str] = None  # Legacy: password (not recommended)
    # Alternative: Place a cookies.txt file at /app/videos/youtube_cookies.txt
    
    # Video processing settings
    MAX_VIDEO_SIZE_MB: int = 2000  # 2GB max upload
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv"]
    
    # Detection settings (tuned for amateur footage)
    DETECTION_CONFIDENCE: float = 0.3  # Lower threshold for variable lighting
    TRACKING_CONFIDENCE: float = 0.4
    OCR_CONFIDENCE: float = 0.5
    
    # Temporal smoothing for jersey OCR
    JERSEY_VOTE_WINDOW: int = 30  # frames to consider for voting
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
