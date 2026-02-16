"""Clip API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid
import os

from app.core.database import get_db
from app.core.config import settings
from app.models.database import Clip, Video, Player, PlayerSegment
from app.models.schemas import ClipCreateRequest, ClipResponse
from app.services.clip_generator import generate_clip

router = APIRouter()


@router.post("/", response_model=ClipResponse)
async def create_clip(
    request: ClipCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a video clip from specified time range"""
    # Verify video exists
    video_result = await db.execute(select(Video).where(Video.id == request.video_id))
    video = video_result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Validate time range
    if request.start_time < 0:
        raise HTTPException(status_code=400, detail="Start time cannot be negative")
    
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="End time must be greater than start time")
    
    if video.duration_seconds and request.end_time > video.duration_seconds:
        raise HTTPException(status_code=400, detail="End time exceeds video duration")
    
    # Generate clip
    clip_id = uuid.uuid4()
    clip_filename = f"{clip_id}.mp4"
    clip_path = os.path.join(settings.CLIPS_DIR, clip_filename)
    
    try:
        await generate_clip(
            input_path=video.file_path,
            output_path=clip_path,
            start_time=request.start_time,
            end_time=request.end_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate clip: {str(e)}")
    
    # Create database record
    title = request.title
    if not title and request.player_id:
        # Auto-generate title from player
        player_result = await db.execute(select(Player).where(Player.id == request.player_id))
        player = player_result.scalar_one_or_none()
        if player:
            title = f"Player #{player.jersey_number} highlight"
    
    clip = Clip(
        id=clip_id,
        video_id=request.video_id,
        player_id=request.player_id,
        title=title,
        file_path=clip_path,
        start_time=request.start_time,
        end_time=request.end_time,
        duration_seconds=request.end_time - request.start_time
    )
    
    db.add(clip)
    await db.commit()
    await db.refresh(clip)
    
    return ClipResponse(
        id=clip.id,
        video_id=clip.video_id,
        player_id=clip.player_id,
        title=clip.title,
        file_path=clip.file_path,
        start_time=clip.start_time,
        end_time=clip.end_time,
        duration_seconds=clip.duration_seconds,
        download_url=f"/api/clips/{clip.id}/download",
        created_at=clip.created_at
    )


@router.post("/player/{player_id}/highlights", response_model=List[ClipResponse])
async def create_player_highlights(
    player_id: uuid.UUID,
    merge_threshold: float = 3.0,  # Merge segments within 3 seconds
    db: AsyncSession = Depends(get_db)
):
    """Generate highlight clips for all segments where a player is active"""
    # Get player
    player_result = await db.execute(select(Player).where(Player.id == player_id))
    player = player_result.scalar_one_or_none()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get video
    video_result = await db.execute(select(Video).where(Video.id == player.video_id))
    video = video_result.scalar_one_or_none()
    
    # Get segments
    segments_result = await db.execute(
        select(PlayerSegment)
        .where(PlayerSegment.player_id == player_id)
        .order_by(PlayerSegment.start_time)
    )
    segments = segments_result.scalars().all()
    
    if not segments:
        raise HTTPException(status_code=404, detail="No segments found for player")
    
    # Merge nearby segments
    merged_segments = []
    current_start = segments[0].start_time
    current_end = segments[0].end_time
    
    for segment in segments[1:]:
        if segment.start_time - current_end <= merge_threshold:
            # Merge with current segment
            current_end = max(current_end, segment.end_time)
        else:
            # Save current and start new
            merged_segments.append((current_start, current_end))
            current_start = segment.start_time
            current_end = segment.end_time
    
    merged_segments.append((current_start, current_end))
    
    # Generate clips for each merged segment
    clips = []
    for i, (start_time, end_time) in enumerate(merged_segments):
        clip_id = uuid.uuid4()
        clip_filename = f"{clip_id}.mp4"
        clip_path = os.path.join(settings.CLIPS_DIR, clip_filename)
        
        # Add padding
        padded_start = max(0, start_time - 1.0)
        padded_end = end_time + 1.0
        if video.duration_seconds:
            padded_end = min(padded_end, video.duration_seconds)
        
        try:
            await generate_clip(
                input_path=video.file_path,
                output_path=clip_path,
                start_time=padded_start,
                end_time=padded_end
            )
        except Exception as e:
            continue  # Skip failed clips
        
        clip = Clip(
            id=clip_id,
            video_id=player.video_id,
            player_id=player_id,
            title=f"#{player.jersey_number} Highlight {i + 1}",
            file_path=clip_path,
            start_time=padded_start,
            end_time=padded_end,
            duration_seconds=padded_end - padded_start
        )
        
        db.add(clip)
        clips.append(clip)
    
    await db.commit()
    
    return [ClipResponse(
        id=c.id,
        video_id=c.video_id,
        player_id=c.player_id,
        title=c.title,
        file_path=c.file_path,
        start_time=c.start_time,
        end_time=c.end_time,
        duration_seconds=c.duration_seconds,
        download_url=f"/api/clips/{c.id}/download",
        created_at=c.created_at
    ) for c in clips]


@router.get("/video/{video_id}", response_model=List[ClipResponse])
async def get_clips_by_video(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get all clips for a video"""
    result = await db.execute(
        select(Clip)
        .where(Clip.video_id == video_id)
        .order_by(Clip.start_time)
    )
    clips = result.scalars().all()
    
    return [ClipResponse(
        id=c.id,
        video_id=c.video_id,
        player_id=c.player_id,
        title=c.title,
        file_path=c.file_path,
        start_time=c.start_time,
        end_time=c.end_time,
        duration_seconds=c.duration_seconds,
        download_url=f"/api/clips/{c.id}/download",
        created_at=c.created_at
    ) for c in clips]


@router.get("/{clip_id}", response_model=ClipResponse)
async def get_clip(clip_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get clip details"""
    result = await db.execute(select(Clip).where(Clip.id == clip_id))
    clip = result.scalar_one_or_none()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    return ClipResponse(
        id=clip.id,
        video_id=clip.video_id,
        player_id=clip.player_id,
        title=clip.title,
        file_path=clip.file_path,
        start_time=clip.start_time,
        end_time=clip.end_time,
        duration_seconds=clip.duration_seconds,
        download_url=f"/api/clips/{clip.id}/download",
        created_at=clip.created_at
    )


@router.get("/{clip_id}/download")
async def download_clip(clip_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Download a clip file"""
    result = await db.execute(select(Clip).where(Clip.id == clip_id))
    clip = result.scalar_one_or_none()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    if not os.path.exists(clip.file_path):
        raise HTTPException(status_code=404, detail="Clip file not found")
    
    filename = clip.title.replace(" ", "_") + ".mp4" if clip.title else f"clip_{clip_id}.mp4"
    
    return FileResponse(
        clip.file_path,
        media_type="video/mp4",
        filename=filename
    )


@router.delete("/{clip_id}")
async def delete_clip(clip_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete a clip"""
    result = await db.execute(select(Clip).where(Clip.id == clip_id))
    clip = result.scalar_one_or_none()
    
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    # Delete file
    if os.path.exists(clip.file_path):
        os.remove(clip.file_path)
    
    await db.delete(clip)
    await db.commit()
    
    return {"message": "Clip deleted successfully"}
