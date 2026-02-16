"""Video API endpoints"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
import os
import uuid
import aiofiles
import logging

from app.core.database import get_db
from app.core.config import settings
from app.models.database import Video, Player, Action
from app.models.schemas import VideoUploadResponse, VideoResponse, VideoProcessingStatus
from app.services.video_processor import download_youtube_video, get_video_info
from app.workers.tasks import process_video_task

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a video file or provide a YouTube URL for analysis.
    The video will be queued for processing.
    """
    if not file and not youtube_url:
        raise HTTPException(status_code=400, detail="Either file or youtube_url must be provided")
    
    video_id = uuid.uuid4()
    
    try:
        if youtube_url:
            # Download from YouTube
            logger.info(f"Downloading YouTube video: {youtube_url}")
            file_path, filename = await download_youtube_video(youtube_url, video_id)
            original_url = youtube_url
        else:
            # Handle file upload
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in settings.SUPPORTED_VIDEO_FORMATS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported format. Supported: {settings.SUPPORTED_VIDEO_FORMATS}"
                )
            
            # Save uploaded file
            file_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}{ext}")
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            original_url = None
        
        # Get video metadata
        video_info = await get_video_info(file_path)
        
        # Create database record
        video = Video(
            id=video_id,
            filename=filename,
            original_url=original_url,
            file_path=file_path,
            duration_seconds=video_info.get("duration"),
            width=video_info.get("width"),
            height=video_info.get("height"),
            fps=video_info.get("fps"),
            status="queued"
        )
        
        db.add(video)
        await db.commit()
        
        # Queue for processing
        process_video_task.delay(str(video_id))
        
        return VideoUploadResponse(
            id=video_id,
            filename=filename,
            status="queued",
            message="Video uploaded successfully and queued for processing"
        )
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[VideoResponse])
async def list_videos(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all videos with optional status filter"""
    query = select(Video).order_by(Video.created_at.desc())
    
    if status:
        query = query.where(Video.status == status)
    
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    videos = result.scalars().all()
    
    # Add counts
    response = []
    for video in videos:
        # Get player count
        player_count = await db.execute(
            select(func.count(Player.id)).where(Player.video_id == video.id)
        )
        # Get action count
        action_count = await db.execute(
            select(func.count(Action.id)).where(Action.video_id == video.id)
        )
        
        video_dict = {
            "id": video.id,
            "filename": video.filename,
            "original_url": video.original_url,
            "duration_seconds": video.duration_seconds,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "status": video.status,
            "error_message": video.error_message,
            "created_at": video.created_at,
            "processed_at": video.processed_at,
            "player_count": player_count.scalar(),
            "action_count": action_count.scalar()
        }
        response.append(VideoResponse(**video_dict))
    
    return response


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get video details by ID"""
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get counts
    player_count = await db.execute(
        select(func.count(Player.id)).where(Player.video_id == video.id)
    )
    action_count = await db.execute(
        select(func.count(Action.id)).where(Action.video_id == video.id)
    )
    
    return VideoResponse(
        id=video.id,
        filename=video.filename,
        original_url=video.original_url,
        duration_seconds=video.duration_seconds,
        width=video.width,
        height=video.height,
        fps=video.fps,
        status=video.status,
        error_message=video.error_message,
        created_at=video.created_at,
        processed_at=video.processed_at,
        player_count=player_count.scalar(),
        action_count=action_count.scalar()
    )


@router.get("/{video_id}/status", response_model=VideoProcessingStatus)
async def get_video_status(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get video processing status"""
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # TODO: Get progress from Redis/Celery
    progress = None
    current_step = None
    
    if video.status == "processing":
        current_step = "Analyzing video..."
    
    return VideoProcessingStatus(
        id=video.id,
        status=video.status,
        progress=progress,
        current_step=current_step,
        error_message=video.error_message
    )


@router.get("/{video_id}/stream")
async def stream_video(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Stream video file"""
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video.file_path,
        media_type="video/mp4",
        filename=video.filename
    )


@router.delete("/{video_id}")
async def delete_video(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete a video and all associated data"""
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete file
    if os.path.exists(video.file_path):
        os.remove(video.file_path)
    
    # Delete from database (cascades to related records)
    await db.delete(video)
    await db.commit()
    
    return {"message": "Video deleted successfully"}


@router.post("/youtube/setup-auth")
async def setup_youtube_auth():
    """
    Initialize YouTube OAuth authentication.
    This will open a browser window for you to log in to YouTube.
    Run this once to enable access to private/age-restricted videos.
    """
    from app.services.video_processor import setup_youtube_oauth
    
    try:
        success = await setup_youtube_oauth()
        if success:
            return {"message": "YouTube authentication successful", "status": "success"}
        else:
            raise HTTPException(
                status_code=500, 
                detail="OAuth setup failed. Check server logs for details."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/youtube/auth-status")
async def check_youtube_auth():
    """Check if YouTube authentication is configured"""
    import os
    from app.core.config import settings
    
    cookies_path = os.path.join(settings.VIDEOS_DIR, "youtube_cookies.txt")
    
    auth_methods = {
        "cookies_file": os.path.exists(cookies_path),
        "oauth_enabled": settings.YOUTUBE_USE_OAUTH,
        "credentials_set": bool(settings.YOUTUBE_USERNAME and settings.YOUTUBE_PASSWORD)
    }
    
    is_authenticated = any(auth_methods.values())
    
    return {
        "authenticated": is_authenticated,
        "methods": auth_methods,
        "message": "YouTube authentication is configured" if is_authenticated else "No YouTube authentication configured. Private videos will not be accessible."
    }
