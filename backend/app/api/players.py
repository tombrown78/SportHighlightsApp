"""Player API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
import uuid

from app.core.database import get_db
from app.models.database import Player, PlayerSegment, Action, Video
from app.models.schemas import (
    PlayerResponse, PlayerDetailResponse, PlayerSegmentResponse,
    ActionResponse, TimelineMarker, PlayerTimeline, PlayerStats, ActionSummary
)

router = APIRouter()


@router.get("/video/{video_id}", response_model=List[PlayerResponse])
async def get_players_by_video(
    video_id: uuid.UUID,
    jersey_number: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get all players detected in a video"""
    # Verify video exists
    video_result = await db.execute(select(Video).where(Video.id == video_id))
    if not video_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Video not found")
    
    query = select(Player).where(Player.video_id == video_id)
    
    if jersey_number:
        query = query.where(Player.jersey_number == jersey_number)
    
    query = query.order_by(Player.jersey_number)
    result = await db.execute(query)
    players = result.scalars().all()
    
    response = []
    for player in players:
        # Get segment count
        segment_count = await db.execute(
            select(func.count(PlayerSegment.id)).where(PlayerSegment.player_id == player.id)
        )
        # Get action count
        action_count = await db.execute(
            select(func.count(Action.id)).where(Action.player_id == player.id)
        )
        
        response.append(PlayerResponse(
            id=player.id,
            video_id=player.video_id,
            jersey_number=player.jersey_number,
            team=player.team,
            team_color=player.team_color,
            track_id=player.track_id,
            confidence=player.confidence,
            first_seen_frame=player.first_seen_frame,
            last_seen_frame=player.last_seen_frame,
            segment_count=segment_count.scalar(),
            action_count=action_count.scalar()
        ))
    
    return response


@router.get("/{player_id}", response_model=PlayerDetailResponse)
async def get_player(player_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get detailed player information including segments and actions"""
    result = await db.execute(select(Player).where(Player.id == player_id))
    player = result.scalar_one_or_none()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get segments
    segments_result = await db.execute(
        select(PlayerSegment)
        .where(PlayerSegment.player_id == player_id)
        .order_by(PlayerSegment.start_time)
    )
    segments = segments_result.scalars().all()
    
    # Get actions
    actions_result = await db.execute(
        select(Action)
        .where(Action.player_id == player_id)
        .order_by(Action.timestamp)
    )
    actions = actions_result.scalars().all()
    
    # Get counts
    segment_count = len(segments)
    action_count = len(actions)
    
    return PlayerDetailResponse(
        id=player.id,
        video_id=player.video_id,
        jersey_number=player.jersey_number,
        team=player.team,
        track_id=player.track_id,
        confidence=player.confidence,
        first_seen_frame=player.first_seen_frame,
        last_seen_frame=player.last_seen_frame,
        segment_count=segment_count,
        action_count=action_count,
        segments=[PlayerSegmentResponse(
            id=s.id,
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            start_time=s.start_time,
            end_time=s.end_time,
            is_active=s.is_active
        ) for s in segments],
        actions=[ActionResponse(
            id=a.id,
            video_id=a.video_id,
            player_id=a.player_id,
            action_type=a.action_type,
            frame=a.frame,
            timestamp=a.timestamp,
            confidence=a.confidence,
            action_data=a.action_data
        ) for a in actions]
    )


@router.get("/{player_id}/timeline", response_model=PlayerTimeline)
async def get_player_timeline(player_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get timeline data for a player (for visualization)"""
    result = await db.execute(select(Player).where(Player.id == player_id))
    player = result.scalar_one_or_none()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get segments
    segments_result = await db.execute(
        select(PlayerSegment)
        .where(PlayerSegment.player_id == player_id)
        .order_by(PlayerSegment.start_time)
    )
    segments = segments_result.scalars().all()
    
    # Get actions
    actions_result = await db.execute(
        select(Action)
        .where(Action.player_id == player_id)
        .order_by(Action.timestamp)
    )
    actions = actions_result.scalars().all()
    
    # Build timeline markers
    markers = []
    
    for segment in segments:
        markers.append(TimelineMarker(
            time=segment.start_time,
            frame=segment.start_frame,
            type="segment_start",
            player_id=player.id,
            player_jersey=player.jersey_number,
            label=f"#{player.jersey_number} appears"
        ))
        markers.append(TimelineMarker(
            time=segment.end_time,
            frame=segment.end_frame,
            type="segment_end",
            player_id=player.id,
            player_jersey=player.jersey_number,
            label=f"#{player.jersey_number} exits"
        ))
    
    for action in actions:
        markers.append(TimelineMarker(
            time=action.timestamp,
            frame=action.frame,
            type="action",
            player_id=player.id,
            player_jersey=player.jersey_number,
            action_type=action.action_type,
            label=f"#{player.jersey_number} {action.action_type}"
        ))
    
    # Sort markers by time
    markers.sort(key=lambda m: m.time)
    
    return PlayerTimeline(
        player=PlayerResponse(
            id=player.id,
            video_id=player.video_id,
            jersey_number=player.jersey_number,
            team=player.team,
            team_color=player.team_color,
            track_id=player.track_id,
            confidence=player.confidence,
            first_seen_frame=player.first_seen_frame,
            last_seen_frame=player.last_seen_frame,
            segment_count=len(segments),
            action_count=len(actions)
        ),
        segments=[PlayerSegmentResponse(
            id=s.id,
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            start_time=s.start_time,
            end_time=s.end_time,
            is_active=s.is_active
        ) for s in segments],
        actions=[ActionResponse(
            id=a.id,
            video_id=a.video_id,
            player_id=a.player_id,
            action_type=a.action_type,
            frame=a.frame,
            timestamp=a.timestamp,
            confidence=a.confidence,
            action_data=a.action_data
        ) for a in actions],
        markers=markers
    )


@router.get("/{player_id}/stats", response_model=PlayerStats)
async def get_player_stats(player_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get statistics for a player"""
    result = await db.execute(select(Player).where(Player.id == player_id))
    player = result.scalar_one_or_none()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get segments for total time
    segments_result = await db.execute(
        select(PlayerSegment).where(PlayerSegment.player_id == player_id)
    )
    segments = segments_result.scalars().all()
    total_time = sum(s.end_time - s.start_time for s in segments)
    
    # Get action counts by type
    action_counts_result = await db.execute(
        select(Action.action_type, func.count(Action.id), func.avg(Action.confidence))
        .where(Action.player_id == player_id)
        .group_by(Action.action_type)
    )
    action_counts = action_counts_result.all()
    
    action_dict = {row[0]: row[1] for row in action_counts}
    action_summary = [
        ActionSummary(
            action_type=row[0],
            count=row[1],
            avg_confidence=float(row[2]) if row[2] else None
        )
        for row in action_counts
    ]
    
    return PlayerStats(
        player=PlayerResponse(
            id=player.id,
            video_id=player.video_id,
            jersey_number=player.jersey_number,
            team=player.team,
            team_color=player.team_color,
            track_id=player.track_id,
            confidence=player.confidence,
            first_seen_frame=player.first_seen_frame,
            last_seen_frame=player.last_seen_frame,
            segment_count=len(segments),
            action_count=sum(action_dict.values())
        ),
        total_time_visible=total_time,
        action_counts=action_dict,
        actions=action_summary
    )
