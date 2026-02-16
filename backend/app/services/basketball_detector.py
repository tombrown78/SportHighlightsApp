"""
Basketball-Specific Detection Service
Uses pre-trained models for hoop/rim detection and basketball tracking
"""

import logging
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HoopDetection:
    """Detected basketball hoop/rim"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    frame_number: int
    center: Tuple[float, float]  # center x, y


@dataclass
class BasketballDetection:
    """Detected basketball"""
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_number: int
    center: Tuple[float, float]


@dataclass
class ShotEvent:
    """Detected shot attempt with made/missed classification"""
    frame_start: int
    frame_end: int
    timestamp_start: float
    timestamp_end: float
    shot_made: bool
    confidence: float
    player_track_id: Optional[int] = None
    shot_location: Optional[Tuple[float, float]] = None  # For shot charts


class BasketballDetector:
    """
    Basketball-specific detection using pre-trained models
    
    Uses Roboflow pre-trained YOLOv8 model for:
    - Basketball detection (more accurate than COCO sports ball)
    - Hoop/rim detection
    
    Falls back to COCO model if basketball model not available.
    """
    
    # Roboflow model info - can be downloaded or used via API
    ROBOFLOW_WORKSPACE = "rohit-krishnan-xr6xf"
    ROBOFLOW_PROJECT = "basketball_and_hoops"
    ROBOFLOW_VERSION = 3
    
    def __init__(self, use_roboflow_api: bool = False):
        """
        Initialize basketball detector
        
        Args:
            use_roboflow_api: If True, use Roboflow hosted inference API
                             If False, use local YOLOv8 model (download weights)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_roboflow_api = use_roboflow_api
        self.model = None
        self.roboflow_model = None
        
        # Model paths
        self.models_dir = Path(settings.MODELS_DIR)
        self.basketball_model_path = self.models_dir / "basketball_and_hoops.pt"
        
        # Class mappings for the Roboflow model
        # These may vary based on the specific model version
        self.class_names = {
            0: "basketball",
            1: "hoop"
        }
        
        # Hoop tracking for stability
        self.hoop_history: List[HoopDetection] = []
        self.stable_hoop_position: Optional[Tuple[float, float, float, float]] = None
        
        logger.info(f"Initializing BasketballDetector on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the basketball detection model"""
        if self.use_roboflow_api:
            self._load_roboflow_api()
        else:
            self._load_local_model()
    
    def _load_roboflow_api(self):
        """Load model via Roboflow API (requires API key)"""
        try:
            from roboflow import Roboflow
            
            api_key = settings.ROBOFLOW_API_KEY
            if not api_key:
                logger.warning("ROBOFLOW_API_KEY not set, falling back to COCO model")
                self._load_fallback_model()
                return
            
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(self.ROBOFLOW_WORKSPACE).project(self.ROBOFLOW_PROJECT)
            self.roboflow_model = project.version(self.ROBOFLOW_VERSION).model
            logger.info("Loaded Roboflow basketball model via API")
            
        except Exception as e:
            logger.warning(f"Failed to load Roboflow API model: {e}")
            self._load_fallback_model()
    
    def _load_local_model(self):
        """Load local YOLOv8 model weights"""
        if self.basketball_model_path.exists():
            try:
                self.model = YOLO(str(self.basketball_model_path))
                self.model.to(self.device)
                logger.info(f"Loaded basketball model from {self.basketball_model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load basketball model: {e}")
        
        # Try to download from Roboflow if API key available
        if settings.ROBOFLOW_API_KEY:
            try:
                self._download_roboflow_weights()
                if self.basketball_model_path.exists():
                    self.model = YOLO(str(self.basketball_model_path))
                    self.model.to(self.device)
                    logger.info("Downloaded and loaded basketball model from Roboflow")
                    return
            except Exception as e:
                logger.warning(f"Failed to download Roboflow weights: {e}")
        
        # Fall back to COCO model
        self._load_fallback_model()
    
    def _download_roboflow_weights(self):
        """Download YOLOv8 weights from Roboflow"""
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
            project = rf.workspace(self.ROBOFLOW_WORKSPACE).project(self.ROBOFLOW_PROJECT)
            version = project.version(self.ROBOFLOW_VERSION)
            
            # Download in YOLOv8 format
            self.models_dir.mkdir(parents=True, exist_ok=True)
            dataset = version.download("yolov8", location=str(self.models_dir / "basketball_dataset"))
            
            # Find the weights file
            weights_path = self.models_dir / "basketball_dataset" / "weights" / "best.pt"
            if weights_path.exists():
                import shutil
                shutil.copy(weights_path, self.basketball_model_path)
                logger.info(f"Downloaded basketball model weights to {self.basketball_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download Roboflow weights: {e}")
            raise
    
    def _load_fallback_model(self):
        """Load standard YOLOv8 as fallback (uses COCO classes)"""
        logger.info("Using YOLOv8 with COCO classes as fallback")
        self.model = YOLO("yolov8x.pt")
        self.model.to(self.device)
        
        # Update class mappings for COCO
        self.class_names = {
            32: "basketball",  # COCO sports ball
            # No hoop class in COCO
        }
        self.using_fallback = True
    
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> Tuple[List[BasketballDetection], List[HoopDetection]]:
        """
        Detect basketballs and hoops in a single frame
        
        Args:
            frame: BGR image as numpy array
            frame_number: Current frame number
            
        Returns:
            Tuple of (basketball_detections, hoop_detections)
        """
        basketballs = []
        hoops = []
        
        if self.roboflow_model:
            # Use Roboflow API
            result = self.roboflow_model.predict(frame, confidence=40).json()
            
            for pred in result.get("predictions", []):
                x = pred["x"]
                y = pred["y"]
                w = pred["width"]
                h = pred["height"]
                
                bbox = (x - w/2, y - h/2, x + w/2, y + h/2)
                center = (x, y)
                confidence = pred["confidence"]
                class_name = pred["class"].lower()
                
                if "ball" in class_name or "basketball" in class_name:
                    basketballs.append(BasketballDetection(
                        bbox=bbox,
                        confidence=confidence,
                        frame_number=frame_number,
                        center=center
                    ))
                elif "hoop" in class_name or "rim" in class_name:
                    hoops.append(HoopDetection(
                        bbox=bbox,
                        confidence=confidence,
                        frame_number=frame_number,
                        center=center
                    ))
        
        elif self.model:
            # Use local YOLOv8 model
            if hasattr(self, 'using_fallback') and self.using_fallback:
                # COCO model - only detect sports ball (class 32)
                results = self.model(
                    frame,
                    conf=settings.DETECTION_CONFIDENCE,
                    classes=[32],  # Sports ball
                    verbose=False
                )[0]
            else:
                # Basketball-specific model
                results = self.model(
                    frame,
                    conf=settings.DETECTION_CONFIDENCE,
                    verbose=False
                )[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    bbox = tuple(boxes[i])
                    conf = float(confidences[i])
                    class_id = int(class_ids[i])
                    
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    center = (cx, cy)
                    
                    class_name = self.class_names.get(class_id, "unknown")
                    
                    if "ball" in class_name or "basketball" in class_name:
                        basketballs.append(BasketballDetection(
                            bbox=bbox,
                            confidence=conf,
                            frame_number=frame_number,
                            center=center
                        ))
                    elif "hoop" in class_name or "rim" in class_name:
                        hoops.append(HoopDetection(
                            bbox=bbox,
                            confidence=conf,
                            frame_number=frame_number,
                            center=center
                        ))
        
        # Update hoop tracking
        if hoops:
            self._update_hoop_tracking(hoops)
        
        return basketballs, hoops
    
    def _update_hoop_tracking(self, hoops: List[HoopDetection]):
        """
        Track hoop position over time for stability
        Hoops don't move, so we can average detections for accuracy
        """
        # Add to history
        self.hoop_history.extend(hoops)
        
        # Keep last 100 detections
        if len(self.hoop_history) > 100:
            self.hoop_history = self.hoop_history[-100:]
        
        # Calculate stable position (average of recent detections)
        if len(self.hoop_history) >= 10:
            x1s = [h.bbox[0] for h in self.hoop_history]
            y1s = [h.bbox[1] for h in self.hoop_history]
            x2s = [h.bbox[2] for h in self.hoop_history]
            y2s = [h.bbox[3] for h in self.hoop_history]
            
            self.stable_hoop_position = (
                np.median(x1s),
                np.median(y1s),
                np.median(x2s),
                np.median(y2s)
            )
    
    def get_stable_hoop_position(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the stable (averaged) hoop position"""
        return self.stable_hoop_position
    
    def classify_shot(
        self,
        ball_trajectory: List[BasketballDetection],
        hoop_position: Tuple[float, float, float, float],
        fps: float
    ) -> Optional[ShotEvent]:
        """
        Classify a shot as made or missed based on ball trajectory
        
        A shot is made if:
        1. Ball enters hoop region from above
        2. Ball exits hoop region from below
        3. Ball passes through the center of the hoop
        
        Args:
            ball_trajectory: List of ball detections during shot
            hoop_position: Stable hoop bounding box
            fps: Video frame rate
            
        Returns:
            ShotEvent if shot detected, None otherwise
        """
        if len(ball_trajectory) < 5:
            return None
        
        hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop_position
        hoop_cx = (hoop_x1 + hoop_x2) / 2
        hoop_cy = (hoop_y1 + hoop_y2) / 2
        hoop_width = hoop_x2 - hoop_x1
        hoop_height = hoop_y2 - hoop_y1
        
        # Track ball position relative to hoop
        above_hoop = []
        in_hoop = []
        below_hoop = []
        
        for ball in ball_trajectory:
            bx, by = ball.center
            
            # Check horizontal alignment with hoop
            horizontally_aligned = hoop_x1 - hoop_width < bx < hoop_x2 + hoop_width
            
            if not horizontally_aligned:
                continue
            
            if by < hoop_y1:
                above_hoop.append(ball)
            elif hoop_y1 <= by <= hoop_y2:
                in_hoop.append(ball)
            else:
                below_hoop.append(ball)
        
        # Shot made criteria:
        # 1. Ball was above hoop
        # 2. Ball passed through hoop region
        # 3. Ball went below hoop
        # 4. Ball passed near center of hoop
        
        shot_made = False
        confidence = 0.5
        
        if above_hoop and in_hoop and below_hoop:
            # Check if ball passed through center
            for ball in in_hoop:
                dist_to_center = abs(ball.center[0] - hoop_cx)
                if dist_to_center < hoop_width * 0.4:  # Within 40% of center
                    shot_made = True
                    confidence = 0.85
                    break
        
        elif above_hoop and in_hoop and not below_hoop:
            # Ball entered but didn't exit below - likely rim out
            shot_made = False
            confidence = 0.7
        
        elif above_hoop and not in_hoop:
            # Ball never entered hoop region - air ball or off backboard
            shot_made = False
            confidence = 0.8
        
        else:
            # Unclear trajectory
            return None
        
        # Create shot event
        all_balls = ball_trajectory
        frame_start = all_balls[0].frame_number
        frame_end = all_balls[-1].frame_number
        
        return ShotEvent(
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_start=frame_start / fps,
            timestamp_end=frame_end / fps,
            shot_made=shot_made,
            confidence=confidence
        )


class ShotTracker:
    """
    Tracks shots throughout a video and classifies made/missed
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.shot_events: List[ShotEvent] = []
        
        # Shot detection parameters
        self.min_shot_frames = 15  # Minimum frames for a shot arc
        self.max_shot_frames = 90  # Maximum frames (3 seconds)
        self.shot_velocity_threshold = 30  # Minimum upward velocity
        
    def detect_shot_attempts(
        self,
        ball_detections: List[BasketballDetection],
        hoop_position: Optional[Tuple[float, float, float, float]]
    ) -> List[ShotEvent]:
        """
        Detect shot attempts from ball trajectory
        
        Args:
            ball_detections: All ball detections from video
            hoop_position: Stable hoop position (if detected)
            
        Returns:
            List of detected shot events
        """
        if len(ball_detections) < self.min_shot_frames:
            return []
        
        # Sort by frame
        ball_detections = sorted(ball_detections, key=lambda b: b.frame_number)
        
        shots = []
        i = 0
        
        while i < len(ball_detections) - self.min_shot_frames:
            # Look for upward ball movement (shot release)
            window = ball_detections[i:i + 10]
            
            if len(window) < 5:
                i += 1
                continue
            
            # Calculate vertical velocity
            y_start = window[0].center[1]
            y_end = window[-1].center[1]
            frames = window[-1].frame_number - window[0].frame_number
            
            if frames == 0:
                i += 1
                continue
            
            vy = (y_end - y_start) / frames  # Negative = upward
            
            # Detect shot release (strong upward movement)
            if vy < -self.shot_velocity_threshold:
                # Found potential shot - track until ball stops rising
                shot_start = i
                shot_end = i + self.min_shot_frames
                
                # Extend shot window until ball starts falling consistently
                while shot_end < min(len(ball_detections), i + self.max_shot_frames):
                    if shot_end + 5 >= len(ball_detections):
                        break
                    
                    future_window = ball_detections[shot_end:shot_end + 5]
                    if len(future_window) < 2:
                        break
                    
                    future_vy = (future_window[-1].center[1] - future_window[0].center[1])
                    future_frames = future_window[-1].frame_number - future_window[0].frame_number
                    
                    if future_frames > 0:
                        future_vy /= future_frames
                    
                    # Ball is falling and slowing down - shot complete
                    if future_vy > 20:  # Falling
                        shot_end += 5
                        break
                    
                    shot_end += 1
                
                # Extract shot trajectory
                shot_trajectory = ball_detections[shot_start:shot_end]
                
                # Classify shot if hoop detected
                if hoop_position and len(shot_trajectory) >= self.min_shot_frames:
                    detector = BasketballDetector.__new__(BasketballDetector)
                    shot_event = detector.classify_shot(
                        shot_trajectory,
                        hoop_position,
                        self.fps
                    )
                    
                    if shot_event:
                        shots.append(shot_event)
                else:
                    # No hoop - create shot attempt without made/missed
                    shots.append(ShotEvent(
                        frame_start=shot_trajectory[0].frame_number,
                        frame_end=shot_trajectory[-1].frame_number,
                        timestamp_start=shot_trajectory[0].frame_number / self.fps,
                        timestamp_end=shot_trajectory[-1].frame_number / self.fps,
                        shot_made=False,  # Unknown
                        confidence=0.5
                    ))
                
                # Skip past this shot
                i = shot_end
            else:
                i += 1
        
        self.shot_events = shots
        logger.info(f"Detected {len(shots)} shot attempts")
        
        return shots
    
    def get_shot_stats(self) -> Dict:
        """Get shot statistics"""
        total = len(self.shot_events)
        made = sum(1 for s in self.shot_events if s.shot_made)
        missed = total - made
        
        return {
            "total_shots": total,
            "shots_made": made,
            "shots_missed": missed,
            "shooting_percentage": (made / total * 100) if total > 0 else 0
        }
