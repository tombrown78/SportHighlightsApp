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
        
        # Initialize ByteTrack tracker
        # Note: supervision's ByteTrack API varies by version
        self.tracker = sv.ByteTrack(
            track_thresh=settings.TRACKING_CONFIDENCE,
            track_buffer=30,
            match_thresh=0.8,
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
        progress_callback=None
    ) -> Tuple[Dict[int, PlayerTrack], List[Detection]]:
        """
        Process entire video and return tracked players
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback(frame_num, total_frames)
            
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
        
        # Reset tracker for new video
        self.tracker = sv.ByteTrack(
            track_thresh=settings.TRACKING_CONFIDENCE,
            track_buffer=30,
            match_thresh=0.8,
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
