"""Video API endpoints"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
import os
import uuid
import aiofiles
import asyncio
import json
import logging
import redis.asyncio as aioredis

from app.core.database import get_db
from app.core.config import settings
from app.models.database import Video, Player, Action
from app.models.schemas import VideoUploadResponse, VideoResponse, VideoProcessingStatus
from app.services.video_processor import download_youtube_video, get_video_info
from app.workers.tasks import process_video_task

router = APIRouter()
logger = logging.getLogger(__)


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    analysis_mode: Optional[str] = Form("full"),
    target_jersey: Optional[str] = Form(None),
    home_team: Optional[str] = Form(None),
    away_team: Optional[str] = Form(None),
    home_color: Optional[str] = Form(None),
    away_color: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a video file or provide a YouTube URL for analysis.
    
    Analysis modes:
    - "full": Analyze all players (default)
    - "targeted": Focus on a specific jersey number (faster)
    
    Optional team info helps identify which team each player belongs to.
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
        
        # Build processing options
        processing_options = {
            "analysis_mode": analysis_mode,
            "target_jersey": target_jersey,
            "home_team": home_team,
            "away_team": away_team,
            "home_color": home_color,
            "away_color": away_color,
        }
        
        # Queue for processing with options and store task ID
        task = process_video_task.delay(str(video_id), processing_options)
        video.celery_task_id = task.id
        await db.commit()
        
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
    
    # If processing, cancel first
    if video.status in ["processing", "queued"] and video.celery_task_id:
        try:
            redis_client = aioredis.from_url(settings.REDIS_URL)
            await redis_client.set(f"video_cancel:{video_id}", "1", ex=3600)
            await redis_client.close()
            
            from app.workers.tasks import celery_app
            celery_app.control.revoke(video.celery_task_id, terminate=True)
        except Exception as e:
            logger.warning(f"Error cancelling task during delete: {e}")
    
    # Delete file
    if os.path.exists(video.file_path):
        os.remove(video.file_path)
    
    # Delete from database (cascades to related records)
    await db.delete(video)
    await db.commit()
    
    return {"message": "Video deleted successfully"}


@router.post("/{video_id}/cancel")
async def cancel_video_processing(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Cancel a video that is currently processing or queued.
    
    This will:
    1. Set a cancellation flag in Redis
    2. Revoke the Celery task
    3. Update the video status to 'cancelling'
    
    The task will check the flag and clean up gracefully.
    """
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status not in ["processing", "queued"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel video with status '{video.status}'. Only 'processing' or 'queued' videos can be cancelled."
        )
    
    try:
        # Set cancellation flag in Redis
        redis_client = aioredis.from_url(settings.REDIS_URL)
        await redis_client.set(f"video_cancel:{video_id}", "1", ex=3600)  # Expires in 1 hour
        
        # Publish cancellation event for SSE
        await redis_client.publish(
            f"video_progress:{video_id}",
            json.dumps({"event": "cancelling", "message": "Cancellation requested..."})
        )
        
        await redis_client.close()
        
        # Revoke Celery task if we have the task ID
        if video.celery_task_id:
            from app.workers.tasks import celery_app
            celery_app.control.revoke(video.celery_task_id, terminate=True)
            logger.info(f"Revoked Celery task {video.celery_task_id}")
        
        # Update video status
        video.status = "cancelling"
        await db.commit()
        
        return {
            "message": "Cancellation requested",
            "video_id": str(video_id),
            "status": "cancelling"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel: {str(e)}")


@router.post("/{video_id}/retry")
async def retry_video_processing(
    video_id: uuid.UUID,
    analysis_mode: Optional[str] = Form(None),
    target_jersey: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Retry processing for a failed or cancelled video.
    
    Optionally update analysis options for the retry.
    """
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot retry video with status '{video.status}'. Only 'failed' or 'cancelled' videos can be retried."
        )
    
    # Verify the video file still exists
    if not os.path.exists(video.file_path):
        raise HTTPException(
            status_code=400,
            detail="Video file no longer exists. Please upload the video again."
        )
    
    try:
        # Clear any cancellation flag
        redis_client = aioredis.from_url(settings.REDIS_URL)
        await redis_client.delete(f"video_cancel:{video_id}")
        await redis_client.close()
        
        # Build processing options
        processing_options = {
            "analysis_mode": analysis_mode or "full",
            "target_jersey": target_jersey,
        }
        
        # Reset video status and clear error
        video.status = "queued"
        video.error_message = None
        video.processed_at = None
        
        # Queue for processing
        task = process_video_task.delay(str(video_id), processing_options)
        video.celery_task_id = task.id
        
        await db.commit()
        
        return {
            "message": "Video queued for reprocessing",
            "video_id": str(video_id),
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Error retrying video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry: {str(e)}")


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


@router.get("/{video_id}/progress")
async def stream_progress(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Server-Sent Events endpoint for real-time processing progress.
    
    Events:
    - progress: {percent, frame, total_frames, stage}
    - detection: {type, track_id, jersey_number, confidence, frame}
    - action: {type, player_track_id, confidence, frame}
    - complete: {players_count, actions_count}
    - error: {message}
    """
    # Verify video exists
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    async def event_generator():
        """Generate SSE events from Redis pub/sub"""
        redis_client = None
        pubsub = None
        
        try:
            redis_client = aioredis.from_url(settings.REDIS_URL)
            pubsub = redis_client.pubsub()
            channel = f"video_progress:{video_id}"
            
            await pubsub.subscribe(channel)
            logger.info(f"SSE client subscribed to {channel}")
            
            # Send initial connection event
            yield f"data: {json.dumps({'event': 'connected', 'video_id': str(video_id)})}\n\n"
            
            # Check current status
            result = await db.execute(select(Video).where(Video.id == video_id))
            video = result.scalar_one_or_none()
            
            if video and video.status == "completed":
                yield f"data: {json.dumps({'event': 'complete', 'status': 'completed'})}\n\n"
                return
            elif video and video.status == "failed":
                yield f"data: {json.dumps({'event': 'error', 'message': video.error_message or 'Processing failed'})}\n\n"
                return
            
            # Listen for events with timeout
            timeout_count = 0
            max_timeouts = 600  # 10 minutes max (600 * 1 second)
            
            while timeout_count < max_timeouts:
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0
                    )
                    
                    if message and message['type'] == 'message':
                        data = message['data']
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        
                        event_data = json.loads(data)
                        yield f"data: {json.dumps(event_data)}\n\n"
                        
                        # Check for terminal events
                        if event_data.get('event') in ['complete', 'error']:
                            break
                        
                        timeout_count = 0  # Reset timeout on activity
                    else:
                        timeout_count += 1
                        # Send keepalive every 15 seconds
                        if timeout_count % 15 == 0:
                            yield f": keepalive\n\n"
                            
                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count % 15 == 0:
                        yield f": keepalive\n\n"
                        
                        # Check if video status changed
                        await db.refresh(video)
                        if video.status == "completed":
                            yield f"data: {json.dumps({'event': 'complete', 'status': 'completed'})}\n\n"
                            break
                        elif video.status == "failed":
                            yield f"data: {json.dumps({'event': 'error', 'message': video.error_message})}\n\n"
                            break
                    
        except Exception as e:
            logger.error(f"SSE error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        finally:
            if pubsub:
                await pubsub.unsubscribe(f"video_progress:{video_id}")
                await pubsub.close()
            if redis_client:
                await redis_client.close()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
