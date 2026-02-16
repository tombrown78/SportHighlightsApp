"""Pydantic schemas for API request/response models"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


# ============ Video Schemas ============

class VideoUploadResponse(BaseModel):
    """Response after uploading a video"""
    id: UUID
    filename: str
    status: str
    message: str


class VideoBase(BaseModel):
    """Base video schema"""
    filename: str
    original_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None


class VideoResponse(VideoBase):
    """Video response schema"""
    id: UUID
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    player_count: Optional[int] = None
    action_count: Optional[int] = None
    
    class Config:
        from_attributes = True


class VideoProcessingStatus(BaseModel):
    """Video processing status"""
    id: UUID
    status: str
    progress: Optional[float] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None


# ============ Player Schemas ============

class PlayerBase(BaseModel):
    """Base player schema"""
    jersey_number: Optional[str] = None
    team: Optional[str] = None
    confidence: Optional[float] = None


class PlayerResponse(PlayerBase):
    """Player response schema"""
    id: UUID
    video_id: UUID
    track_id: Optional[int] = None
    first_seen_frame: Optional[int] = None
    last_seen_frame: Optional[int] = None
    segment_count: Optional[int] = None
    action_count: Optional[int] = None
    
    class Config:
        from_attributes = True


class PlayerSegmentResponse(BaseModel):
    """Player segment response"""
    id: UUID
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    is_active: bool
    
    class Config:
        from_attributes = True


class PlayerDetailResponse(PlayerResponse):
    """Detailed player response with segments and actions"""
    segments: List[PlayerSegmentResponse] = []
    actions: List["ActionResponse"] = []


# ============ Action Schemas ============

class ActionBase(BaseModel):
    """Base action schema"""
    action_type: str
    frame: int
    timestamp: float
    confidence: Optional[float] = None


class ActionResponse(ActionBase):
    """Action response schema"""
    id: UUID
    video_id: UUID
    player_id: Optional[UUID] = None
    player_jersey: Optional[str] = None
    action_data: Optional[dict] = None
    
    class Config:
        from_attributes = True


class ActionSummary(BaseModel):
    """Summary of actions for a player or video"""
    action_type: str
    count: int
    avg_confidence: Optional[float] = None


# ============ Clip Schemas ============

class ClipCreateRequest(BaseModel):
    """Request to create a clip"""
    video_id: UUID
    player_id: Optional[UUID] = None
    start_time: float
    end_time: float
    title: Optional[str] = None


class ClipResponse(BaseModel):
    """Clip response schema"""
    id: UUID
    video_id: UUID
    player_id: Optional[UUID] = None
    title: Optional[str] = None
    file_path: str
    start_time: float
    end_time: float
    duration_seconds: Optional[float] = None
    download_url: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Timeline Schemas ============

class TimelineMarker(BaseModel):
    """A marker on the video timeline"""
    time: float
    frame: int
    type: str  # "segment_start", "segment_end", "action"
    player_id: Optional[UUID] = None
    player_jersey: Optional[str] = None
    action_type: Optional[str] = None
    label: Optional[str] = None


class PlayerTimeline(BaseModel):
    """Timeline data for a specific player"""
    player: PlayerResponse
    segments: List[PlayerSegmentResponse]
    actions: List[ActionResponse]
    markers: List[TimelineMarker]


# ============ Stats Schemas ============

class PlayerStats(BaseModel):
    """Statistics for a player"""
    player: PlayerResponse
    total_time_visible: float  # seconds
    action_counts: dict  # action_type -> count
    actions: List[ActionSummary]


class VideoStats(BaseModel):
    """Statistics for a video"""
    video: VideoResponse
    total_players: int
    total_actions: int
    action_summary: List[ActionSummary]
    player_stats: List[PlayerStats]


# Update forward references
PlayerDetailResponse.model_rebuild()
