"""
Pose Estimation Service
Uses MediaPipe for player pose detection and shot form analysis
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# MediaPipe imports - lazy loaded to avoid import errors if not installed
mp = None
mp_pose = None
mp_drawing = None


def _init_mediapipe():
    """Lazy initialize MediaPipe"""
    global mp, mp_pose, mp_drawing
    if mp is None:
        try:
            import mediapipe as _mp
            mp = _mp
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe initialized successfully")
        except ImportError:
            logger.warning("MediaPipe not installed. Pose estimation disabled.")
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")


class PoseLandmark(Enum):
    """Key pose landmarks for basketball analysis"""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass
class PoseKeypoint:
    """Single pose keypoint"""
    landmark: PoseLandmark
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    z: float  # Depth (relative)
    visibility: float  # Confidence 0-1


@dataclass
class PlayerPose:
    """Full pose for a player"""
    track_id: int
    frame_number: int
    keypoints: Dict[PoseLandmark, PoseKeypoint]
    
    def get_keypoint(self, landmark: PoseLandmark) -> Optional[PoseKeypoint]:
        """Get a specific keypoint"""
        return self.keypoints.get(landmark)
    
    def get_shooting_arm_angle(self) -> Optional[float]:
        """
        Calculate shooting arm angle (elbow angle)
        Returns angle in degrees, or None if keypoints not visible
        """
        # Try right arm first (most common shooting arm)
        shoulder = self.get_keypoint(PoseLandmark.RIGHT_SHOULDER)
        elbow = self.get_keypoint(PoseLandmark.RIGHT_ELBOW)
        wrist = self.get_keypoint(PoseLandmark.RIGHT_WRIST)
        
        if not all([shoulder, elbow, wrist]):
            # Try left arm
            shoulder = self.get_keypoint(PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_keypoint(PoseLandmark.LEFT_ELBOW)
            wrist = self.get_keypoint(PoseLandmark.LEFT_WRIST)
        
        if not all([shoulder, elbow, wrist]):
            return None
        
        # Check visibility
        if min(shoulder.visibility, elbow.visibility, wrist.visibility) < 0.5:
            return None
        
        # Calculate angle at elbow
        return self._calculate_angle(
            (shoulder.x, shoulder.y),
            (elbow.x, elbow.y),
            (wrist.x, wrist.y)
        )
    
    def get_knee_bend_angle(self) -> Optional[float]:
        """
        Calculate knee bend angle for shot form analysis
        Returns average of both knees, or single knee if only one visible
        """
        angles = []
        
        for side in ['LEFT', 'RIGHT']:
            hip = self.get_keypoint(PoseLandmark[f'{side}_HIP'])
            knee = self.get_keypoint(PoseLandmark[f'{side}_KNEE'])
            ankle = self.get_keypoint(PoseLandmark[f'{side}_ANKLE'])
            
            if all([hip, knee, ankle]):
                if min(hip.visibility, knee.visibility, ankle.visibility) >= 0.5:
                    angle = self._calculate_angle(
                        (hip.x, hip.y),
                        (knee.x, knee.y),
                        (ankle.x, ankle.y)
                    )
                    angles.append(angle)
        
        if angles:
            return np.mean(angles)
        return None
    
    def is_shooting_pose(self) -> Tuple[bool, float]:
        """
        Detect if player is in a shooting pose
        
        Returns:
            Tuple of (is_shooting, confidence)
        """
        # Get arm positions
        r_shoulder = self.get_keypoint(PoseLandmark.RIGHT_SHOULDER)
        r_elbow = self.get_keypoint(PoseLandmark.RIGHT_ELBOW)
        r_wrist = self.get_keypoint(PoseLandmark.RIGHT_WRIST)
        
        l_shoulder = self.get_keypoint(PoseLandmark.LEFT_SHOULDER)
        l_elbow = self.get_keypoint(PoseLandmark.LEFT_ELBOW)
        l_wrist = self.get_keypoint(PoseLandmark.LEFT_WRIST)
        
        confidence = 0.0
        is_shooting = False
        
        # Check for raised arm (shooting motion)
        for shoulder, elbow, wrist in [
            (r_shoulder, r_elbow, r_wrist),
            (l_shoulder, l_elbow, l_wrist)
        ]:
            if not all([shoulder, elbow, wrist]):
                continue
            
            if min(shoulder.visibility, elbow.visibility, wrist.visibility) < 0.5:
                continue
            
            # Shooting indicators:
            # 1. Wrist above shoulder (arm raised)
            # 2. Elbow roughly at shoulder height or above
            # 3. Arm extended upward
            
            wrist_above_shoulder = wrist.y < shoulder.y
            elbow_raised = elbow.y < shoulder.y + 0.1
            arm_extended = wrist.y < elbow.y
            
            if wrist_above_shoulder and elbow_raised and arm_extended:
                is_shooting = True
                
                # Calculate confidence based on how "textbook" the form is
                elbow_angle = self._calculate_angle(
                    (shoulder.x, shoulder.y),
                    (elbow.x, elbow.y),
                    (wrist.x, wrist.y)
                )
                
                # Ideal shooting elbow angle is around 90-110 degrees at release
                if 80 <= elbow_angle <= 120:
                    confidence = 0.9
                elif 60 <= elbow_angle <= 140:
                    confidence = 0.7
                else:
                    confidence = 0.5
                
                break
        
        return is_shooting, confidence
    
    @staticmethod
    def _calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)


@dataclass
class ShotFormAnalysis:
    """Analysis of a player's shooting form"""
    track_id: int
    frame_range: Tuple[int, int]
    
    # Form metrics
    avg_elbow_angle: Optional[float] = None
    avg_knee_bend: Optional[float] = None
    release_height: Optional[float] = None  # Relative to body height
    follow_through: bool = False
    
    # Quality scores (0-100)
    form_score: float = 0.0
    consistency_score: float = 0.0
    
    def get_feedback(self) -> List[str]:
        """Generate feedback based on form analysis"""
        feedback = []
        
        if self.avg_elbow_angle is not None:
            if self.avg_elbow_angle < 80:
                feedback.append("Elbow angle too tight - try extending more at release")
            elif self.avg_elbow_angle > 120:
                feedback.append("Elbow angle too wide - keep elbow closer to 90 degrees")
            else:
                feedback.append("Good elbow angle at release")
        
        if self.avg_knee_bend is not None:
            if self.avg_knee_bend > 160:
                feedback.append("Not enough knee bend - use your legs more")
            elif self.avg_knee_bend < 120:
                feedback.append("Too much knee bend - you may be losing power")
            else:
                feedback.append("Good knee bend for power generation")
        
        if not self.follow_through:
            feedback.append("Work on your follow-through - keep wrist snapped after release")
        
        return feedback


class PoseEstimator:
    """
    MediaPipe-based pose estimation for basketball players
    """
    
    def __init__(self, model_complexity: int = 1):
        """
        Initialize pose estimator
        
        Args:
            model_complexity: 0=lite, 1=full, 2=heavy (more accurate but slower)
        """
        _init_mediapipe()
        
        self.model_complexity = model_complexity
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info(f"PoseEstimator initialized with complexity={model_complexity}")
    
    def estimate_pose(
        self,
        frame: np.ndarray,
        player_bbox: Optional[Tuple[float, float, float, float]] = None,
        track_id: int = -1,
        frame_number: int = 0
    ) -> Optional[PlayerPose]:
        """
        Estimate pose for a player in frame
        
        Args:
            frame: BGR image
            player_bbox: Optional bounding box to crop to player
            track_id: Player track ID
            frame_number: Current frame number
            
        Returns:
            PlayerPose if detected, None otherwise
        """
        # Crop to player if bbox provided
        if player_bbox is not None:
            x1, y1, x2, y2 = map(int, player_bbox)
            h, w = frame.shape[:2]
            
            # Add padding
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            player_frame = frame[y1:y2, x1:x2]
            
            if player_frame.size == 0:
                return None
        else:
            player_frame = frame
            x1, y1 = 0, 0
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(player_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        keypoints = {}
        ph, pw = player_frame.shape[:2]
        
        for landmark_enum in PoseLandmark:
            idx = landmark_enum.value
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                
                # Convert to frame coordinates if cropped
                if player_bbox is not None:
                    # Normalize to full frame
                    abs_x = (lm.x * pw + x1) / frame.shape[1]
                    abs_y = (lm.y * ph + y1) / frame.shape[0]
                else:
                    abs_x = lm.x
                    abs_y = lm.y
                
                keypoints[landmark_enum] = PoseKeypoint(
                    landmark=landmark_enum,
                    x=abs_x,
                    y=abs_y,
                    z=lm.z,
                    visibility=lm.visibility
                )
        
        return PlayerPose(
            track_id=track_id,
            frame_number=frame_number,
            keypoints=keypoints
        )
    
    def analyze_shot_form(
        self,
        poses: List[PlayerPose],
        track_id: int
    ) -> Optional[ShotFormAnalysis]:
        """
        Analyze shooting form from a sequence of poses
        
        Args:
            poses: List of poses during shot
            track_id: Player track ID
            
        Returns:
            ShotFormAnalysis if enough data, None otherwise
        """
        if len(poses) < 5:
            return None
        
        # Filter to poses for this player
        player_poses = [p for p in poses if p.track_id == track_id]
        
        if len(player_poses) < 5:
            return None
        
        # Collect metrics
        elbow_angles = []
        knee_bends = []
        shooting_frames = []
        
        for pose in player_poses:
            is_shooting, conf = pose.is_shooting_pose()
            
            if is_shooting and conf > 0.5:
                shooting_frames.append(pose.frame_number)
                
                elbow = pose.get_shooting_arm_angle()
                if elbow is not None:
                    elbow_angles.append(elbow)
                
                knee = pose.get_knee_bend_angle()
                if knee is not None:
                    knee_bends.append(knee)
        
        if not shooting_frames:
            return None
        
        # Calculate averages
        avg_elbow = np.mean(elbow_angles) if elbow_angles else None
        avg_knee = np.mean(knee_bends) if knee_bends else None
        
        # Check for follow-through (arm stays extended after release)
        follow_through = False
        if len(shooting_frames) >= 3:
            # Check if shooting pose maintained for multiple frames
            frame_diffs = np.diff(shooting_frames)
            if np.all(frame_diffs <= 5):  # Consecutive frames
                follow_through = True
        
        # Calculate form score
        form_score = 50.0  # Base score
        
        if avg_elbow is not None:
            if 85 <= avg_elbow <= 105:
                form_score += 20
            elif 75 <= avg_elbow <= 115:
                form_score += 10
        
        if avg_knee is not None:
            if 130 <= avg_knee <= 150:
                form_score += 15
            elif 120 <= avg_knee <= 160:
                form_score += 8
        
        if follow_through:
            form_score += 15
        
        # Consistency score based on variance
        consistency_score = 50.0
        if len(elbow_angles) >= 3:
            elbow_std = np.std(elbow_angles)
            if elbow_std < 10:
                consistency_score += 25
            elif elbow_std < 20:
                consistency_score += 15
        
        return ShotFormAnalysis(
            track_id=track_id,
            frame_range=(min(shooting_frames), max(shooting_frames)),
            avg_elbow_angle=avg_elbow,
            avg_knee_bend=avg_knee,
            follow_through=follow_through,
            form_score=min(100, form_score),
            consistency_score=min(100, consistency_score)
        )
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose: PlayerPose,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw pose skeleton on frame
        
        Args:
            frame: BGR image
            pose: PlayerPose to draw
            color: BGR color for skeleton
            
        Returns:
            Frame with pose drawn
        """
        h, w = frame.shape[:2]
        
        # Draw keypoints
        for kp in pose.keypoints.values():
            if kp.visibility > 0.5:
                x = int(kp.x * w)
                y = int(kp.y * h)
                cv2.circle(frame, (x, y), 5, color, -1)
        
        # Draw connections
        connections = [
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
            (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
            (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
            (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
            (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
            (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
            (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
            (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
        ]
        
        for start, end in connections:
            kp1 = pose.keypoints.get(start)
            kp2 = pose.keypoints.get(end)
            
            if kp1 and kp2 and kp1.visibility > 0.5 and kp2.visibility > 0.5:
                x1, y1 = int(kp1.x * w), int(kp1.y * h)
                x2, y2 = int(kp2.x * w), int(kp2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        return frame
    
    def close(self):
        """Release resources"""
        if self.pose:
            self.pose.close()
