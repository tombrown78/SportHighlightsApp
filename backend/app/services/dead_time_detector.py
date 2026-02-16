"""
Dead Time Detection Service
Identifies periods of inactivity in basketball videos for automatic removal
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SegmentType(str, Enum):
    """Type of video segment"""
    ACTIVE_PLAY = "active_play"
    DEAD_TIME = "dead_time"
    TIMEOUT = "timeout"
    FREE_THROW_SETUP = "free_throw_setup"
    OUT_OF_BOUNDS = "out_of_bounds"
    CELEBRATION = "celebration"
    UNKNOWN = "unknown"


@dataclass
class VideoSegment:
    """A segment of video with activity classification"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    segment_type: SegmentType
    confidence: float
    motion_score: float  # Average motion in segment
    ball_visible_ratio: float  # Ratio of frames with ball visible
    player_spread: float  # How spread out players are (0-1)


@dataclass
class CondensedVideo:
    """Result of video condensation"""
    active_segments: List[VideoSegment]
    dead_segments: List[VideoSegment]
    total_duration: float
    active_duration: float
    compression_ratio: float  # active_duration / total_duration


class DeadTimeDetector:
    """
    Detects dead time in basketball videos using multiple signals:
    1. Motion analysis (optical flow)
    2. Ball visibility
    3. Player positioning (spread vs clustered)
    4. Scene changes
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize dead time detector
        
        Args:
            fps: Video frame rate
        """
        self.fps = fps
        
        # Detection parameters
        self.motion_threshold = 5.0  # Minimum motion to be "active"
        self.min_segment_duration = 2.0  # Minimum segment length in seconds
        self.ball_visibility_threshold = 0.3  # Min ratio of frames with ball
        self.player_spread_threshold = 0.4  # Min spread to be "active"
        
        # Smoothing window
        self.smoothing_window = int(fps * 1.0)  # 1 second window
        
    def calculate_motion_score(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> float:
        """
        Calculate motion score between two frames using optical flow
        
        Args:
            prev_frame: Previous frame (BGR)
            curr_frame: Current frame (BGR)
            
        Returns:
            Motion score (higher = more motion)
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Calculate magnitude of flow
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Return mean magnitude (average motion)
        return float(np.mean(magnitude))
    
    def calculate_player_spread(
        self,
        player_bboxes: List[Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Calculate how spread out players are on the court
        
        Args:
            player_bboxes: List of player bounding boxes
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Spread score 0-1 (1 = very spread out)
        """
        if len(player_bboxes) < 2:
            return 0.5  # Unknown
        
        # Get player centers
        centers = []
        for bbox in player_bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))
        
        centers = np.array(centers)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt(
                    (centers[i, 0] - centers[j, 0])**2 +
                    (centers[i, 1] - centers[j, 1])**2
                )
                distances.append(dist)
        
        if not distances:
            return 0.5
        
        # Normalize by frame diagonal
        diagonal = np.sqrt(frame_width**2 + frame_height**2)
        avg_distance = np.mean(distances) / diagonal
        
        # Scale to 0-1 (typical spread is 0.1-0.4 of diagonal)
        spread = min(1.0, avg_distance / 0.3)
        
        return spread
    
    def detect_scene_change(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        threshold: float = 30.0
    ) -> bool:
        """
        Detect if there's a scene change (camera cut)
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            threshold: Difference threshold
            
        Returns:
            True if scene change detected
        """
        # Calculate histogram difference
        prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        curr_hist = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(curr_hist, curr_hist)
        
        # Compare histograms
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
        
        return diff > threshold
    
    def analyze_video(
        self,
        video_path: str,
        ball_detections: Optional[List[Tuple[int, Tuple[float, float, float, float]]]] = None,
        player_detections: Optional[Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]] = None,
        progress_callback=None
    ) -> CondensedVideo:
        """
        Analyze video and identify active vs dead time segments
        
        Args:
            video_path: Path to video file
            ball_detections: Optional list of (frame, bbox) for ball
            player_detections: Optional dict of track_id -> [(frame, bbox)]
            progress_callback: Optional callback(frame, total)
            
        Returns:
            CondensedVideo with segment classifications
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.fps = fps
        
        # Build ball visibility map
        ball_frames = set()
        if ball_detections:
            ball_frames = {frame for frame, _ in ball_detections}
        
        # Build player positions map
        player_positions_by_frame: Dict[int, List[Tuple[float, float, float, float]]] = {}
        if player_detections:
            for track_id, detections in player_detections.items():
                for frame, bbox in detections:
                    if frame not in player_positions_by_frame:
                        player_positions_by_frame[frame] = []
                    player_positions_by_frame[frame].append(bbox)
        
        # Analyze frames
        frame_scores = []
        prev_frame = None
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for faster processing
            small_frame = cv2.resize(frame, (640, 360))
            
            # Calculate motion
            motion_score = 0.0
            if prev_frame is not None:
                motion_score = self.calculate_motion_score(prev_frame, small_frame)
            
            # Check ball visibility
            ball_visible = frame_number in ball_frames
            
            # Calculate player spread
            player_spread = 0.5
            if frame_number in player_positions_by_frame:
                player_spread = self.calculate_player_spread(
                    player_positions_by_frame[frame_number],
                    width, height
                )
            
            # Detect scene change
            scene_change = False
            if prev_frame is not None:
                scene_change = self.detect_scene_change(prev_frame, small_frame)
            
            frame_scores.append({
                'frame': frame_number,
                'motion': motion_score,
                'ball_visible': ball_visible,
                'player_spread': player_spread,
                'scene_change': scene_change
            })
            
            prev_frame = small_frame.copy()
            frame_number += 1
            
            if progress_callback and frame_number % 100 == 0:
                progress_callback(frame_number, total_frames)
        
        cap.release()
        
        # Smooth scores
        smoothed_scores = self._smooth_scores(frame_scores)
        
        # Classify segments
        segments = self._classify_segments(smoothed_scores, fps)
        
        # Separate active and dead segments
        active_segments = [s for s in segments if s.segment_type == SegmentType.ACTIVE_PLAY]
        dead_segments = [s for s in segments if s.segment_type != SegmentType.ACTIVE_PLAY]
        
        # Calculate stats
        total_duration = total_frames / fps
        active_duration = sum(s.end_time - s.start_time for s in active_segments)
        compression_ratio = active_duration / total_duration if total_duration > 0 else 1.0
        
        logger.info(f"Video analysis complete: {len(active_segments)} active segments, "
                   f"{len(dead_segments)} dead segments")
        logger.info(f"Compression: {total_duration:.1f}s -> {active_duration:.1f}s "
                   f"({compression_ratio*100:.1f}%)")
        
        return CondensedVideo(
            active_segments=active_segments,
            dead_segments=dead_segments,
            total_duration=total_duration,
            active_duration=active_duration,
            compression_ratio=compression_ratio
        )
    
    def _smooth_scores(self, frame_scores: List[Dict]) -> List[Dict]:
        """Apply smoothing to frame scores"""
        if len(frame_scores) < self.smoothing_window:
            return frame_scores
        
        smoothed = []
        
        for i in range(len(frame_scores)):
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(frame_scores), i + self.smoothing_window // 2)
            window = frame_scores[start:end]
            
            smoothed.append({
                'frame': frame_scores[i]['frame'],
                'motion': np.mean([f['motion'] for f in window]),
                'ball_visible': np.mean([f['ball_visible'] for f in window]),
                'player_spread': np.mean([f['player_spread'] for f in window]),
                'scene_change': frame_scores[i]['scene_change']
            })
        
        return smoothed
    
    def _classify_segments(
        self,
        frame_scores: List[Dict],
        fps: float
    ) -> List[VideoSegment]:
        """Classify frames into segments"""
        if not frame_scores:
            return []
        
        segments = []
        current_type = None
        segment_start = 0
        segment_scores = []
        
        for i, score in enumerate(frame_scores):
            # Determine frame type
            is_active = (
                score['motion'] > self.motion_threshold or
                (score['ball_visible'] > self.ball_visibility_threshold and
                 score['player_spread'] > self.player_spread_threshold)
            )
            
            frame_type = SegmentType.ACTIVE_PLAY if is_active else SegmentType.DEAD_TIME
            
            # Check for segment boundary
            if current_type is None:
                current_type = frame_type
                segment_start = i
                segment_scores = [score]
            elif frame_type != current_type or score['scene_change']:
                # End current segment
                if len(segment_scores) >= self.min_segment_duration * fps:
                    segments.append(self._create_segment(
                        segment_start, i - 1, segment_scores, current_type, fps
                    ))
                
                # Start new segment
                current_type = frame_type
                segment_start = i
                segment_scores = [score]
            else:
                segment_scores.append(score)
        
        # Add final segment
        if segment_scores and len(segment_scores) >= self.min_segment_duration * fps:
            segments.append(self._create_segment(
                segment_start, len(frame_scores) - 1, segment_scores, current_type, fps
            ))
        
        # Merge short segments
        segments = self._merge_short_segments(segments, fps)
        
        return segments
    
    def _create_segment(
        self,
        start_frame: int,
        end_frame: int,
        scores: List[Dict],
        segment_type: SegmentType,
        fps: float
    ) -> VideoSegment:
        """Create a VideoSegment from frame scores"""
        motion_scores = [s['motion'] for s in scores]
        ball_visible = [s['ball_visible'] for s in scores]
        player_spreads = [s['player_spread'] for s in scores]
        
        return VideoSegment(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_frame / fps,
            end_time=end_frame / fps,
            segment_type=segment_type,
            confidence=0.8,  # Could be refined based on score consistency
            motion_score=np.mean(motion_scores),
            ball_visible_ratio=np.mean(ball_visible),
            player_spread=np.mean(player_spreads)
        )
    
    def _merge_short_segments(
        self,
        segments: List[VideoSegment],
        fps: float,
        min_duration: float = 3.0
    ) -> List[VideoSegment]:
        """Merge segments shorter than min_duration with neighbors"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            segment = segments[i]
            duration = segment.end_time - segment.start_time
            
            if duration < min_duration and i > 0:
                # Merge with previous segment
                prev = merged[-1]
                merged[-1] = VideoSegment(
                    start_frame=prev.start_frame,
                    end_frame=segment.end_frame,
                    start_time=prev.start_time,
                    end_time=segment.end_time,
                    segment_type=prev.segment_type,  # Keep previous type
                    confidence=(prev.confidence + segment.confidence) / 2,
                    motion_score=(prev.motion_score + segment.motion_score) / 2,
                    ball_visible_ratio=(prev.ball_visible_ratio + segment.ball_visible_ratio) / 2,
                    player_spread=(prev.player_spread + segment.player_spread) / 2
                )
            else:
                merged.append(segment)
            
            i += 1
        
        return merged
    
    def get_condensed_timestamps(
        self,
        condensed: CondensedVideo
    ) -> List[Tuple[float, float]]:
        """
        Get list of (start, end) timestamps for active segments
        
        Args:
            condensed: CondensedVideo result
            
        Returns:
            List of (start_time, end_time) tuples
        """
        return [(s.start_time, s.end_time) for s in condensed.active_segments]
    
    def generate_condensed_video(
        self,
        video_path: str,
        output_path: str,
        condensed: CondensedVideo,
        add_transitions: bool = True
    ) -> str:
        """
        Generate a condensed video with only active segments
        
        Args:
            video_path: Input video path
            output_path: Output video path
            condensed: CondensedVideo with segment info
            add_transitions: Add fade transitions between segments
            
        Returns:
            Path to output video
        """
        import subprocess
        
        if not condensed.active_segments:
            logger.warning("No active segments to export")
            return video_path
        
        # Build FFmpeg filter for concatenating segments
        filter_parts = []
        inputs = []
        
        for i, segment in enumerate(condensed.active_segments):
            # Trim each segment
            start = segment.start_time
            duration = segment.end_time - segment.start_time
            
            filter_parts.append(
                f"[0:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS[v{i}];"
                f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[a{i}]"
            )
            inputs.append(f"[v{i}][a{i}]")
        
        # Concatenate
        concat_filter = f"{''.join(inputs)}concat=n={len(condensed.active_segments)}:v=1:a=1[outv][outa]"
        
        full_filter = ";".join(filter_parts) + ";" + concat_filter
        
        # Run FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", full_filter,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Generated condensed video: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
