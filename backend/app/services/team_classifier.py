"""
Team Classification Service
Uses K-Means clustering on jersey colors to assign players to teams
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

logger = logging.getLogger(__name__)


class TeamLabel(str, Enum):
    """Team labels"""
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass
class TeamAssignment:
    """Team assignment for a player"""
    track_id: int
    team: TeamLabel
    dominant_color: Tuple[int, int, int]  # RGB
    confidence: float


class TeamClassifier:
    """
    Classifies players into teams based on jersey color
    
    Uses K-Means clustering in LAB color space for perceptual uniformity.
    Can optionally detect referees (typically black/white striped).
    """
    
    def __init__(self, detect_referees: bool = True):
        """
        Initialize team classifier
        
        Args:
            detect_referees: If True, attempt to identify referees separately
        """
        self.detect_referees = detect_referees
        self.team_colors: Dict[TeamLabel, Tuple[int, int, int]] = {}
        self.assignments: Dict[int, TeamAssignment] = {}
        
        # Referee detection thresholds
        self.referee_saturation_threshold = 30  # Low saturation = grayscale
        self.referee_pattern_threshold = 0.3    # High variance = stripes
        
    def extract_jersey_color(
        self,
        player_crop: np.ndarray,
        use_torso_only: bool = True
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Extract dominant jersey color from player crop
        
        Args:
            player_crop: BGR image of player
            use_torso_only: If True, focus on torso region (more reliable)
            
        Returns:
            Tuple of (LAB color array, RGB color tuple)
        """
        if player_crop is None or player_crop.size == 0:
            return np.array([128, 128, 128]), (128, 128, 128)
        
        h, w = player_crop.shape[:2]
        
        if use_torso_only and h > 20 and w > 10:
            # Extract torso region (upper 60%, middle 80% width)
            y1 = int(h * 0.15)
            y2 = int(h * 0.55)
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)
            
            torso = player_crop[y1:y2, x1:x2]
            
            if torso.size > 0:
                player_crop = torso
        
        # Convert to LAB color space (perceptually uniform)
        try:
            lab = cv2.cvtColor(player_crop, cv2.COLOR_BGR2LAB)
        except cv2.error:
            return np.array([128, 128, 128]), (128, 128, 128)
        
        # Reshape to list of pixels
        pixels = lab.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (shadows/highlights)
        l_channel = pixels[:, 0]
        mask = (l_channel > 30) & (l_channel < 220)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 10:
            filtered_pixels = pixels
        
        # Calculate dominant color (mean of filtered pixels)
        dominant_lab = filtered_pixels.mean(axis=0)
        
        # Convert back to RGB for display
        dominant_lab_img = np.uint8([[dominant_lab]])
        dominant_rgb_img = cv2.cvtColor(dominant_lab_img, cv2.COLOR_LAB2BGR)
        dominant_rgb = tuple(int(c) for c in dominant_rgb_img[0, 0][::-1])  # BGR to RGB
        
        return dominant_lab, dominant_rgb
    
    def is_referee(self, player_crop: np.ndarray) -> bool:
        """
        Check if player crop looks like a referee (grayscale/striped)
        
        Args:
            player_crop: BGR image of player
            
        Returns:
            True if likely a referee
        """
        if not self.detect_referees:
            return False
        
        if player_crop is None or player_crop.size == 0:
            return False
        
        h, w = player_crop.shape[:2]
        if h < 20 or w < 10:
            return False
        
        # Extract torso region
        y1 = int(h * 0.15)
        y2 = int(h * 0.55)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        torso = player_crop[y1:y2, x1:x2]
        
        if torso.size == 0:
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        # Check saturation (referees typically have low saturation - black/white)
        avg_saturation = hsv[:, :, 1].mean()
        
        if avg_saturation < self.referee_saturation_threshold:
            # Low saturation - could be referee
            # Check for stripe pattern (high horizontal variance)
            gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
            
            # Calculate row-wise variance
            row_means = gray.mean(axis=1)
            row_variance = np.var(row_means)
            
            # Normalize by image brightness
            normalized_variance = row_variance / (gray.mean() + 1)
            
            if normalized_variance > self.referee_pattern_threshold:
                return True
        
        return False
    
    def classify_teams(
        self,
        player_crops: Dict[int, List[np.ndarray]],
        n_teams: int = 2
    ) -> Dict[int, TeamAssignment]:
        """
        Classify all players into teams using K-Means clustering
        
        Args:
            player_crops: Dict mapping track_id to list of player crop images
            n_teams: Number of teams (default 2)
            
        Returns:
            Dict mapping track_id to TeamAssignment
        """
        if not player_crops:
            return {}
        
        # Extract colors for each player
        player_colors: Dict[int, List[np.ndarray]] = {}
        player_rgb_colors: Dict[int, List[Tuple[int, int, int]]] = {}
        referee_candidates: List[int] = []
        
        for track_id, crops in player_crops.items():
            colors_lab = []
            colors_rgb = []
            referee_votes = 0
            
            for crop in crops[:20]:  # Sample up to 20 crops per player
                if crop is None or crop.size == 0:
                    continue
                
                # Check for referee
                if self.is_referee(crop):
                    referee_votes += 1
                
                lab, rgb = self.extract_jersey_color(crop)
                colors_lab.append(lab)
                colors_rgb.append(rgb)
            
            if colors_lab:
                player_colors[track_id] = colors_lab
                player_rgb_colors[track_id] = colors_rgb
                
                # If majority of samples look like referee
                if referee_votes > len(crops[:20]) * 0.5:
                    referee_candidates.append(track_id)
        
        if not player_colors:
            return {}
        
        # Calculate average color per player
        player_avg_colors = {}
        player_avg_rgb = {}
        
        for track_id, colors in player_colors.items():
            avg_lab = np.mean(colors, axis=0)
            player_avg_colors[track_id] = avg_lab
            
            # Average RGB
            rgb_colors = player_rgb_colors[track_id]
            avg_rgb = tuple(int(np.mean([c[i] for c in rgb_colors])) for i in range(3))
            player_avg_rgb[track_id] = avg_rgb
        
        # Remove referees from clustering
        non_referee_ids = [tid for tid in player_avg_colors.keys() 
                          if tid not in referee_candidates]
        
        if len(non_referee_ids) < n_teams:
            logger.warning(f"Not enough players ({len(non_referee_ids)}) for {n_teams} teams")
            # Assign all to unknown
            for track_id in player_avg_colors.keys():
                self.assignments[track_id] = TeamAssignment(
                    track_id=track_id,
                    team=TeamLabel.UNKNOWN,
                    dominant_color=player_avg_rgb.get(track_id, (128, 128, 128)),
                    confidence=0.0
                )
            return self.assignments
        
        # Prepare data for K-Means
        X = np.array([player_avg_colors[tid] for tid in non_referee_ids])
        
        # Run K-Means clustering
        kmeans = KMeans(n_clusters=n_teams, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate cluster centers in RGB
        cluster_colors = {}
        for i in range(n_teams):
            cluster_mask = labels == i
            cluster_ids = [non_referee_ids[j] for j in range(len(non_referee_ids)) if cluster_mask[j]]
            
            if cluster_ids:
                cluster_rgb = [player_avg_rgb[tid] for tid in cluster_ids]
                avg_rgb = tuple(int(np.mean([c[i] for c in cluster_rgb])) for i in range(3))
                cluster_colors[i] = avg_rgb
        
        # Store team colors
        team_labels = [TeamLabel.TEAM_A, TeamLabel.TEAM_B]
        for i, label in enumerate(team_labels[:n_teams]):
            if i in cluster_colors:
                self.team_colors[label] = cluster_colors[i]
        
        # Calculate confidence based on distance to cluster center
        distances = kmeans.transform(X)
        
        # Assign teams
        for i, track_id in enumerate(non_referee_ids):
            cluster = labels[i]
            team = team_labels[cluster] if cluster < len(team_labels) else TeamLabel.UNKNOWN
            
            # Confidence based on relative distance to clusters
            dist_to_own = distances[i, cluster]
            dist_to_other = distances[i, 1 - cluster] if n_teams == 2 else np.min(
                [distances[i, j] for j in range(n_teams) if j != cluster]
            )
            
            # Higher confidence if much closer to own cluster
            confidence = dist_to_other / (dist_to_own + dist_to_other + 1e-6)
            confidence = min(1.0, max(0.0, confidence))
            
            self.assignments[track_id] = TeamAssignment(
                track_id=track_id,
                team=team,
                dominant_color=player_avg_rgb[track_id],
                confidence=confidence
            )
        
        # Assign referees
        for track_id in referee_candidates:
            self.assignments[track_id] = TeamAssignment(
                track_id=track_id,
                team=TeamLabel.REFEREE,
                dominant_color=player_avg_rgb.get(track_id, (128, 128, 128)),
                confidence=0.8
            )
        
        # Log results
        team_counts = Counter(a.team for a in self.assignments.values())
        logger.info(f"Team classification: {dict(team_counts)}")
        logger.info(f"Team colors: {self.team_colors}")
        
        return self.assignments
    
    def get_team_for_player(self, track_id: int) -> Optional[TeamAssignment]:
        """Get team assignment for a specific player"""
        return self.assignments.get(track_id)
    
    def get_team_color(self, team: TeamLabel) -> Optional[Tuple[int, int, int]]:
        """Get the dominant color for a team"""
        return self.team_colors.get(team)
    
    def get_players_by_team(self, team: TeamLabel) -> List[int]:
        """Get all player track IDs for a team"""
        return [
            track_id for track_id, assignment in self.assignments.items()
            if assignment.team == team
        ]


def classify_teams_from_video(
    video_path: str,
    player_tracks: Dict[int, 'PlayerTrack'],
    sample_interval: int = 30
) -> Dict[int, TeamAssignment]:
    """
    Convenience function to classify teams from a video
    
    Args:
        video_path: Path to video file
        player_tracks: Dict of player tracks from detector
        sample_interval: Sample every N frames for efficiency
        
    Returns:
        Dict mapping track_id to TeamAssignment
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return {}
    
    # Collect player crops
    player_crops: Dict[int, List[np.ndarray]] = {tid: [] for tid in player_tracks.keys()}
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number % sample_interval == 0:
            for track_id, track in player_tracks.items():
                # Find detection at this frame
                for det in track.detections:
                    if det.frame_number == frame_number:
                        x1, y1, x2, y2 = map(int, det.bbox)
                        
                        # Ensure valid crop
                        h, w = frame.shape[:2]
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2].copy()
                            player_crops[track_id].append(crop)
                        break
        
        frame_number += 1
    
    cap.release()
    
    # Classify teams
    classifier = TeamClassifier(detect_referees=True)
    return classifier.classify_teams(player_crops)
