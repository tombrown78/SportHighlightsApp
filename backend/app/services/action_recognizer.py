"""
Basketball Action Recognition Service
Detects shots, rebounds, assists, and other basketball actions
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Basketball action types"""
    SHOT_ATTEMPT = "shot_attempt"
    SHOT_MADE = "shot_made"
    SHOT_MISSED = "shot_missed"
    THREE_POINT_ATTEMPT = "three_point_attempt"
    REBOUND = "rebound"
    ASSIST = "assist"
    STEAL = "steal"
    TURNOVER = "turnover"
    BLOCK = "block"
    DRIBBLE = "dribble"
    PASS = "pass"


@dataclass
class ActionDetection:
    """Detected basketball action"""
    action_type: ActionType
    frame: int
    timestamp: float
    confidence: float
    player_track_id: Optional[int] = None
    metadata: Optional[dict] = None


class ActionRecognizer:
    """
    Basketball action recognition using heuristics and ball tracking
    
    This is a simplified implementation that uses:
    - Ball trajectory analysis for shot detection
    - Player-ball proximity for possession
    - Position changes for rebounds
    
    For production, consider using a trained action recognition model
    like R(2+1)D or SlowFast trained on basketball data.
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        
        # Court dimensions (approximate, in pixels - will be calibrated)
        self.three_point_distance = 300  # pixels from basket
        
        # Thresholds
        self.shot_velocity_threshold = 50  # pixels per frame
        self.possession_distance = 100  # pixels
        self.rebound_window = 60  # frames after shot
        
    def analyze_ball_trajectory(
        self,
        ball_positions: List[Tuple[int, Tuple[float, float, float, float]]]
    ) -> List[ActionDetection]:
        """
        Analyze ball trajectory to detect shots
        
        Args:
            ball_positions: List of (frame, bbox) for ball detections
            
        Returns:
            List of detected shot actions
        """
        actions = []
        
        if len(ball_positions) < 10:
            return actions
        
        # Calculate ball centers and velocities
        centers = []
        for frame, bbox in ball_positions:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((frame, cx, cy))
        
        # Detect upward ball movement (potential shots)
        for i in range(5, len(centers) - 5):
            frame, cx, cy = centers[i]
            
            # Look at velocity over window
            prev_frame, prev_cx, prev_cy = centers[i - 5]
            next_frame, next_cx, next_cy = centers[i + 5]
            
            # Upward velocity (negative y in image coordinates)
            vy = (cy - prev_cy) / max(1, frame - prev_frame)
            
            # Check for upward movement followed by downward (arc)
            vy_after = (next_cy - cy) / max(1, next_frame - frame)
            
            if vy < -self.shot_velocity_threshold and vy_after > 0:
                # Potential shot detected
                timestamp = frame / self.fps
                
                actions.append(ActionDetection(
                    action_type=ActionType.SHOT_ATTEMPT,
                    frame=frame,
                    timestamp=timestamp,
                    confidence=0.7,
                    metadata={"ball_velocity": float(vy)}
                ))
        
        return actions
    
    def detect_possession_changes(
        self,
        player_positions: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
        ball_positions: List[Tuple[int, Tuple[float, float, float, float]]]
    ) -> List[Tuple[int, int]]:
        """
        Detect when ball possession changes between players
        
        Returns:
            List of (frame, player_track_id) for possession changes
        """
        possession_changes = []
        current_possessor = None
        
        for frame, ball_bbox in ball_positions:
            ball_cx = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_cy = (ball_bbox[1] + ball_bbox[3]) / 2
            
            # Find closest player
            closest_player = None
            closest_distance = float('inf')
            
            for track_id, positions in player_positions.items():
                # Find player position at this frame
                for pframe, pbbox in positions:
                    if pframe == frame:
                        pcx = (pbbox[0] + pbbox[2]) / 2
                        pcy = (pbbox[1] + pbbox[3]) / 2
                        
                        distance = np.sqrt((ball_cx - pcx)**2 + (ball_cy - pcy)**2)
                        
                        if distance < closest_distance and distance < self.possession_distance:
                            closest_distance = distance
                            closest_player = track_id
                        break
            
            if closest_player is not None and closest_player != current_possessor:
                possession_changes.append((frame, closest_player))
                current_possessor = closest_player
        
        return possession_changes
    
    def detect_rebounds(
        self,
        shot_frames: List[int],
        possession_changes: List[Tuple[int, int]]
    ) -> List[ActionDetection]:
        """
        Detect rebounds based on possession changes after shots
        """
        rebounds = []
        
        for shot_frame in shot_frames:
            # Look for possession change within rebound window
            for poss_frame, player_id in possession_changes:
                if shot_frame < poss_frame <= shot_frame + self.rebound_window:
                    rebounds.append(ActionDetection(
                        action_type=ActionType.REBOUND,
                        frame=poss_frame,
                        timestamp=poss_frame / self.fps,
                        confidence=0.6,
                        player_track_id=player_id,
                        metadata={"shot_frame": shot_frame}
                    ))
                    break  # Only one rebound per shot
        
        return rebounds
    
    def analyze_video(
        self,
        player_detections: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
        ball_detections: List[Tuple[int, Tuple[float, float, float, float]]],
        fps: float
    ) -> List[ActionDetection]:
        """
        Analyze full video and detect all actions
        
        Args:
            player_detections: Dict mapping track_id to list of (frame, bbox)
            ball_detections: List of (frame, bbox) for ball
            fps: Video frame rate
            
        Returns:
            List of all detected actions
        """
        self.fps = fps
        all_actions = []
        
        # Detect shots from ball trajectory
        shots = self.analyze_ball_trajectory(ball_detections)
        all_actions.extend(shots)
        
        # Detect possession changes
        possession_changes = self.detect_possession_changes(
            player_detections, ball_detections
        )
        
        # Detect rebounds
        shot_frames = [a.frame for a in shots if a.action_type == ActionType.SHOT_ATTEMPT]
        rebounds = self.detect_rebounds(shot_frames, possession_changes)
        all_actions.extend(rebounds)
        
        # Assign shots to players based on possession before shot
        for action in all_actions:
            if action.action_type == ActionType.SHOT_ATTEMPT and action.player_track_id is None:
                # Find who had possession before the shot
                for poss_frame, player_id in reversed(possession_changes):
                    if poss_frame < action.frame:
                        action.player_track_id = player_id
                        break
        
        # Sort by frame
        all_actions.sort(key=lambda a: a.frame)
        
        logger.info(f"Detected {len(all_actions)} actions: "
                   f"{len(shots)} shots, {len(rebounds)} rebounds")
        
        return all_actions
