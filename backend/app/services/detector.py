"""
Player Detection and Tracking Service
Uses YOLOv8 for detection and ByteTrack for multi-object tracking
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import supervision as sv

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result"""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    frame_number: int


@dataclass
class PlayerTrack:
    """Tracked player across frames"""
    track_id: int
    detections: List[Detection]
    jersey_numbers: List[str]  # All detected jersey numbers for voting
    first_frame: int
    last_frame: int
    # Appearance-based re-identification fields
    appearance_embedding: Optional[np.ndarray] = None  # 512-dim visual embedding
    appearance_cluster_id: Optional[int] = None  # Cluster ID after appearance-based grouping
    appearance_features: Optional[Dict] = None  # Human-readable features (colors, etc.)
    merged_track_ids: Optional[List[int]] = None  # Track IDs merged into this one
    
    @property
    def jersey_number(self) -> Optional[str]:
        """Get most common jersey number (temporal voting)"""
        if not self.jersey_numbers:
            return None
        from collections import Counter
        counts = Counter(self.jersey_numbers)
        most_common = counts.most_common(1)
        if most_common and most_common[0][1] >= 3:  # Require at least 3 detections
            return most_common[0][0]
        return None
    
    @property
    def confidence(self) -> float:
        """Average confidence across detections"""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


class PlayerDetector:
    """
    YOLOv8 + ByteTrack player detection and tracking pipeline
    Optimized for basketball footage
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing PlayerDetector on {self.device}")
        
        # Load YOLOv8 model
        # Using yolov8x (extra large) for best accuracy - requires 8GB+ VRAM
        # For GPUs with <8GB VRAM, use yolov8m (medium) or yolov8s (small)
        self.model = YOLO("yolov8x.pt")
        self.model.to(self.device)
        
        # Initialize ByteTrack tracker with optimized parameters
        # - track_buffer=90: Keep lost tracks longer (3 seconds at 30fps) to handle occlusions
        # - match_thresh=0.65: More aggressive matching to reduce track fragmentation
        self.tracker = sv.ByteTrack(
            track_thresh=settings.TRACKING_CONFIDENCE,
            track_buffer=90,
            match_thresh=0.65,
            frame_rate=30
        )
        
        # Class IDs for person detection (COCO dataset)
        self.person_class_id = 0
        self.ball_class_id = 32  # Sports ball
        
        logger.info("PlayerDetector initialized successfully")
    
    def detect_frame(self, frame: np.ndarray, frame_number: int) -> List[Detection]:
        """
        Detect players and ball in a single frame
        
        Args:
            frame: BGR image as numpy array
            frame_number: Current frame number
            
        Returns:
            List of Detection objects
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=settings.DETECTION_CONFIDENCE,
            classes=[self.person_class_id, self.ball_class_id],
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Convert to supervision Detections for tracking
            sv_detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
            
            # Update tracker
            tracked = self.tracker.update_with_detections(sv_detections)
            
            # Create Detection objects
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i]
                conf = tracked.confidence[i] if tracked.confidence is not None else 0.5
                class_id = tracked.class_id[i] if tracked.class_id is not None else 0
                track_id = tracked.tracker_id[i] if tracked.tracker_id is not None else -1
                
                class_name = "person" if class_id == self.person_class_id else "ball"
                
                detections.append(Detection(
                    track_id=int(track_id),
                    bbox=tuple(bbox),
                    confidence=float(conf),
                    class_name=class_name,
                    frame_number=frame_number
                ))
        
        return detections
    
    def process_video(
        self,
        video_path: str,
        progress_callback=None,
        detection_callback=None
    ) -> Tuple[Dict[int, PlayerTrack], List[Detection]]:
        """
        Process entire video and return tracked players
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback(frame_num, total_frames)
            detection_callback: Optional callback(track_id, class_name, confidence, frame, bbox)
            
        Returns:
            Tuple of (player_tracks dict, all_detections list)
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {total_frames} frames at {fps} FPS")
        
        # Reset tracker for new video with optimized parameters
        self.tracker = sv.ByteTrack(
            track_thresh=settings.TRACKING_CONFIDENCE,
            track_buffer=90,
            match_thresh=0.65,
            frame_rate=int(fps) if fps > 0 else 30
        )
        
        player_tracks: Dict[int, PlayerTrack] = {}
        all_detections: List[Detection] = []
        ball_detections: List[Detection] = []
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect in this frame
            detections = self.detect_frame(frame, frame_number)
            
            for det in detections:
                all_detections.append(det)
                
                # Call detection callback if provided
                if detection_callback:
                    try:
                        detection_callback(
                            det.track_id,
                            det.class_name,
                            det.confidence,
                            det.frame_number,
                            list(det.bbox)
                        )
                    except Exception as e:
                        logger.warning(f"Detection callback error: {e}")
                
                if det.class_name == "person" and det.track_id >= 0:
                    # Update or create player track
                    if det.track_id not in player_tracks:
                        player_tracks[det.track_id] = PlayerTrack(
                            track_id=det.track_id,
                            detections=[],
                            jersey_numbers=[],
                            first_frame=frame_number,
                            last_frame=frame_number
                        )
                    
                    track = player_tracks[det.track_id]
                    track.detections.append(det)
                    track.last_frame = frame_number
                
                elif det.class_name == "ball":
                    ball_detections.append(det)
            
            frame_number += 1
            
            # Progress callback
            if progress_callback and frame_number % 30 == 0:
                progress_callback(frame_number, total_frames)
        
        cap.release()
        
        logger.info(f"Processed {frame_number} frames, found {len(player_tracks)} player tracks")
        
        return player_tracks, all_detections
    
    def get_player_crop(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract player crop from frame for OCR"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return frame[y1:y2, x1:x2]


def filter_valid_tracks(
    player_tracks: Dict[int, PlayerTrack],
    fps: float,
    frame_width: int,
    frame_height: int
) -> Dict[int, PlayerTrack]:
    """
    Filter out likely false positive tracks based on multiple criteria.
    
    This reduces the player count from hundreds to a realistic number by removing:
    - Brief appearances (crowd, people walking by)
    - Low confidence detections
    - Very small bounding boxes (distant people)
    - Detections mostly at frame edges (partial views)
    
    Args:
        player_tracks: Dictionary of track_id -> PlayerTrack
        fps: Video frames per second
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels
        
    Returns:
        Filtered dictionary of valid player tracks
    """
    frame_area = frame_width * frame_height
    filtered_tracks = {}
    
    stats = {
        "total": len(player_tracks),
        "filtered_duration": 0,
        "filtered_detections": 0,
        "filtered_confidence": 0,
        "filtered_bbox_size": 0,
        "filtered_edge": 0,
        "kept": 0
    }
    
    for track_id, track in player_tracks.items():
        # Calculate track duration in seconds
        duration_frames = track.last_frame - track.first_frame
        duration_seconds = duration_frames / fps if fps > 0 else 0
        
        # Filter 1: Minimum duration
        if duration_seconds < settings.MIN_TRACK_DURATION_SECONDS:
            stats["filtered_duration"] += 1
            continue
        
        # Filter 2: Minimum detection count
        if len(track.detections) < settings.MIN_TRACK_DETECTIONS:
            stats["filtered_detections"] += 1
            continue
        
        # Filter 3: Minimum average confidence
        if track.confidence < settings.MIN_TRACK_CONFIDENCE:
            stats["filtered_confidence"] += 1
            continue
        
        # Filter 4: Minimum bounding box size (filter distant crowd)
        avg_bbox_area = _calculate_avg_bbox_area(track.detections)
        bbox_area_ratio = avg_bbox_area / frame_area
        if bbox_area_ratio < settings.MIN_BBOX_AREA_RATIO:
            stats["filtered_bbox_size"] += 1
            continue
        
        # Filter 5: Edge detection ratio (filter partial views)
        edge_ratio = _calculate_edge_ratio(track.detections, frame_width, frame_height)
        if edge_ratio > settings.MAX_EDGE_RATIO:
            stats["filtered_edge"] += 1
            continue
        
        # Track passed all filters
        filtered_tracks[track_id] = track
        stats["kept"] += 1
    
    logger.info(f"Track filtering results: {stats}")
    logger.info(f"Reduced from {stats['total']} to {stats['kept']} tracks")
    
    return filtered_tracks


def _calculate_avg_bbox_area(detections: List[Detection]) -> float:
    """Calculate average bounding box area across detections"""
    if not detections:
        return 0.0
    
    total_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        area = (x2 - x1) * (y2 - y1)
        total_area += area
    
    return total_area / len(detections)


def _calculate_edge_ratio(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
    edge_threshold: int = 10
) -> float:
    """
    Calculate what ratio of detections are at frame edges.
    High ratio suggests partial/entering/exiting detections.
    """
    if not detections:
        return 0.0
    
    edge_count = 0
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Check if bbox touches any edge
        at_left = x1 < edge_threshold
        at_right = x2 > frame_width - edge_threshold
        at_top = y1 < edge_threshold
        at_bottom = y2 > frame_height - edge_threshold
        
        if at_left or at_right or at_top or at_bottom:
            edge_count += 1
    
    return edge_count / len(detections)


def deduplicate_tracks_by_jersey(
    player_tracks: Dict[int, PlayerTrack]
) -> Dict[int, PlayerTrack]:
    """
    Merge tracks that have the same jersey number.
    
    This handles track fragmentation where the same player gets multiple
    track IDs due to occlusion or tracking failures.
    
    Strategy:
    - Group tracks by jersey number
    - For each jersey number, keep the track with most detections as primary
    - Merge detections from other tracks into the primary
    - Tracks without jersey numbers are kept as-is
    
    Args:
        player_tracks: Dictionary of track_id -> PlayerTrack
        
    Returns:
        Deduplicated dictionary of player tracks
    """
    from collections import defaultdict
    
    # Group tracks by jersey number
    jersey_groups: Dict[str, List[PlayerTrack]] = defaultdict(list)
    no_jersey_tracks: Dict[int, PlayerTrack] = {}
    
    for track_id, track in player_tracks.items():
        jersey = track.jersey_number
        if jersey:
            jersey_groups[jersey].append(track)
        else:
            no_jersey_tracks[track_id] = track
    
    # Merge tracks with same jersey number
    merged_tracks: Dict[int, PlayerTrack] = {}
    merge_count = 0
    
    for jersey, tracks in jersey_groups.items():
        if len(tracks) == 1:
            # Only one track with this jersey, keep as-is
            merged_tracks[tracks[0].track_id] = tracks[0]
        else:
            # Multiple tracks with same jersey - merge them
            merge_count += len(tracks) - 1
            
            # Sort by detection count, keep the one with most detections
            tracks.sort(key=lambda t: len(t.detections), reverse=True)
            primary = tracks[0]
            
            # Merge detections from other tracks
            for secondary in tracks[1:]:
                primary.detections.extend(secondary.detections)
                primary.jersey_numbers.extend(secondary.jersey_numbers)
                primary.first_frame = min(primary.first_frame, secondary.first_frame)
                primary.last_frame = max(primary.last_frame, secondary.last_frame)
            
            # Sort detections by frame number
            primary.detections.sort(key=lambda d: d.frame_number)
            
            merged_tracks[primary.track_id] = primary
    
    # Add tracks without jersey numbers
    merged_tracks.update(no_jersey_tracks)
    
    if merge_count > 0:
        logger.info(f"Merged {merge_count} duplicate tracks by jersey number")
    
    logger.info(f"After deduplication: {len(merged_tracks)} unique players")
    
    return merged_tracks
