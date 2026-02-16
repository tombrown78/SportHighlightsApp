"""SQLAlchemy database models"""

from sqlalchemy import Column, String, Float, Integer, Boolean, ForeignKey, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class Video(Base):
    """Video model"""
    __tablename__ = "videos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_url = Column(String(1024))
    file_path = Column(String(512), nullable=False)
    duration_seconds = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    fps = Column(Float)
    status = Column(String(50), default="pending")
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # Relationships
    players = relationship("Player", back_populates="video", cascade="all, delete-orphan")
    actions = relationship("Action", back_populates="video", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="video", cascade="all, delete-orphan")


class Player(Base):
    """Player detected in a video"""
    __tablename__ = "players"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    jersey_number = Column(String(10))
    team = Column(String(50))
    track_id = Column(Integer)
    confidence = Column(Float)
    first_seen_frame = Column(Integer)
    last_seen_frame = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    video = relationship("Video", back_populates="players")
    segments = relationship("PlayerSegment", back_populates="player", cascade="all, delete-orphan")
    actions = relationship("Action", back_populates="player")
    clips = relationship("Clip", back_populates="player")


class PlayerSegment(Base):
    """Time segment where a player is visible/active"""
    __tablename__ = "player_segments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="segments")


class Action(Base):
    """Detected basketball action"""
    __tablename__ = "actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id", ondelete="SET NULL"))
    action_type = Column(String(50), nullable=False)
    frame = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    confidence = Column(Float)
    action_data = Column(JSONB)  # Renamed from 'metadata' which is reserved in SQLAlchemy
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    video = relationship("Video", back_populates="actions")
    player = relationship("Player", back_populates="actions")


class Clip(Base):
    """Generated video clip"""
    __tablename__ = "clips"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id", ondelete="SET NULL"))
    title = Column(String(255))
    file_path = Column(String(512), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration_seconds = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    video = relationship("Video", back_populates="clips")
    player = relationship("Player", back_populates="clips")
