"""
Player Re-Identification Service
Uses appearance-based embeddings to identify and cluster players
even when jersey OCR fails.

Uses OSNet model via timm for person re-identification embeddings.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

# Lazy load model to avoid import issues
_model_instance = None
_transform_instance = None


def get_reid_model():
    """Get or create the ReID model instance using timm"""
    global _model_instance, _transform_instance
    
    if _model_instance is None:
        try:
            import torch
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing OSNet feature extractor on {device}")
            
            # Use osnet_x1_0 from timm - a lightweight ReID model
            # This model outputs 512-dim features suitable for person re-identification
            _model_instance = timm.create_model(
                'osnet_x1_0',
                pretrained=True,
                num_classes=0  # Remove classifier, get features only
            )
            _model_instance = _model_instance.to(device)
            _model_instance.eval()
            
            # Get the transform for this model
            config = resolve_data_config({}, model=_model_instance)
            _transform_instance = create_transform(**config)
            
            logger.info("OSNet feature extractor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {e}")
            raise
    
    return _model_instance, _transform_instance


@dataclass
class PlayerAppearance:
    """Appearance information for a player"""
    track_id: int
    embedding: np.ndarray  # 512-dim appearance embedding
    cluster_id: Optional[int] = None  # Assigned cluster after clustering
    appearance_features: Optional[Dict] = None  # Human-readable features


@dataclass
class AppearanceCluster:
    """A cluster of tracks that likely belong to the same player"""
    cluster_id: int
    track_ids: List[int]
    centroid_embedding: np.ndarray
    confidence: float  # How tight the cluster is


class PlayerReID:
    """
    Person Re-Identification for basketball players.
    
    Uses OSNet pre-trained model to generate visual embeddings
    that capture appearance features like:
    - Jersey color and pattern
    - Shoe color
    - Body shape
    - Skin tone
    - Hair style
    - Accessories (headbands, arm sleeves, etc.)
    
    These embeddings can be used to:
    1. Merge fragmented tracks of the same player
    2. Identify players even when jersey OCR fails
    3. Reduce false player count from hundreds to actual unique individuals
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize PlayerReID service.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider
                                  two embeddings as the same person (0-1)
        """
        self.model = None
        self.transform = None
        self.device = None
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = 512  # OSNet output dimension
        
    def _ensure_model(self):
        """Ensure model is loaded"""
        if self.model is None:
            import torch
            self.model, self.transform = get_reid_model()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def preprocess_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess player crop for ReID model.
        
        Args:
            crop: BGR image of player
            
        Returns:
            Preprocessed RGB image or None if invalid
        """
        if crop is None or crop.size == 0:
            return None
        
        h, w = crop.shape[:2]
        
        # Skip very small crops
        if h < 30 or w < 15:
            return None
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        return rgb
    
    def get_embedding(self, player_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-dimensional appearance embedding from player crop.
        
        Args:
            player_crop: BGR image of player
            
        Returns:
            512-dim numpy array or None if extraction failed
        """
        import torch
        from PIL import Image
        
        self._ensure_model()
        
        processed = self.preprocess_crop(player_crop)
        if processed is None:
            return None
        
        try:
            # Convert to PIL Image for timm transform
            pil_image = Image.fromarray(processed)
            
            # Apply transform and add batch dimension
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
            
            # Convert to numpy and normalize
            embedding = features[0].cpu().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None
    
    def get_embeddings_batch(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple crops efficiently.
        
        Args:
            crops: List of BGR player crop images
            
        Returns:
            List of embeddings (None for failed extractions)
        """
        import torch
        from PIL import Image
        
        self._ensure_model()
        
        # Preprocess all crops
        processed = []
        valid_indices = []
        
        for i, crop in enumerate(crops):
            p = self.preprocess_crop(crop)
            if p is not None:
                processed.append(p)
                valid_indices.append(i)
        
        if not processed:
            return [None] * len(crops)
        
        try:
            # Convert to tensors
            tensors = []
            for img in processed:
                pil_image = Image.fromarray(img)
                tensor = self.transform(pil_image)
                tensors.append(tensor)
            
            # Stack into batch
            batch = torch.stack(tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(batch)
            
            # Map back to original indices
            results = [None] * len(crops)
            for i, idx in enumerate(valid_indices):
                emb = features[i].cpu().numpy()
                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                results[idx] = emb
            
            return results
            
        except Exception as e:
            logger.warning(f"Batch embedding extraction failed: {e}")
            return [None] * len(crops)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding (512-dim)
            emb2: Second embedding (512-dim)
            
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity (1 - cosine distance)
        return 1.0 - cosine(emb1, emb2)
    
    def compute_track_embedding(
        self,
        track_crops: List[np.ndarray],
        sample_interval: int = 10,
        max_samples: int = 20
    ) -> Optional[np.ndarray]:
        """
        Compute average embedding for a track from multiple crops.
        
        Sampling multiple frames and averaging provides a more robust
        representation than a single frame.
        
        Args:
            track_crops: List of player crops from the track
            sample_interval: Sample every N crops
            max_samples: Maximum number of samples to use
            
        Returns:
            Average 512-dim embedding or None
        """
        if not track_crops:
            return None
        
        # Sample crops
        sample_indices = list(range(0, len(track_crops), sample_interval))[:max_samples]
        sampled_crops = [track_crops[i] for i in sample_indices]
        
        # Get embeddings
        embeddings = self.get_embeddings_batch(sampled_crops)
        
        # Filter valid embeddings
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if not valid_embeddings:
            return None
        
        # Average embeddings
        avg_embedding = np.mean(valid_embeddings, axis=0)
        
        # Re-normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    def cluster_players(
        self,
        track_embeddings: Dict[int, np.ndarray],
        distance_threshold: float = None
    ) -> Dict[int, int]:
        """
        Cluster track embeddings to identify unique players.
        
        Uses Agglomerative Clustering with cosine distance.
        Tracks in the same cluster are likely the same person.
        
        Args:
            track_embeddings: Dict mapping track_id to embedding
            distance_threshold: Max distance to merge clusters (default: 1 - similarity_threshold)
            
        Returns:
            Dict mapping track_id to cluster_id
        """
        if not track_embeddings:
            return {}
        
        if distance_threshold is None:
            distance_threshold = 1.0 - self.similarity_threshold
        
        track_ids = list(track_embeddings.keys())
        embeddings = np.array([track_embeddings[tid] for tid in track_ids])
        
        if len(embeddings) < 2:
            # Only one track, assign to cluster 0
            return {track_ids[0]: 0}
        
        try:
            # Agglomerative clustering with cosine distance
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average'
            )
            
            labels = clustering.fit_predict(embeddings)
            
            # Map track_ids to cluster labels
            track_to_cluster = {tid: int(label) for tid, label in zip(track_ids, labels)}
            
            # Log clustering results
            n_clusters = len(set(labels))
            logger.info(f"Clustered {len(track_ids)} tracks into {n_clusters} unique players")
            
            return track_to_cluster
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback: each track is its own cluster
            return {tid: i for i, tid in enumerate(track_ids)}
    
    def merge_tracks_by_appearance(
        self,
        player_tracks: Dict[int, 'PlayerTrack'],
        track_embeddings: Dict[int, np.ndarray]
    ) -> Dict[int, 'PlayerTrack']:
        """
        Merge tracks that belong to the same player based on appearance.
        
        This is similar to jersey-based deduplication but works even when
        jersey numbers are not detected.
        
        Args:
            player_tracks: Dict of track_id -> PlayerTrack
            track_embeddings: Dict of track_id -> embedding
            
        Returns:
            Merged player tracks dict
        """
        if not player_tracks or not track_embeddings:
            return player_tracks
        
        # Cluster tracks by appearance
        track_to_cluster = self.cluster_players(track_embeddings)
        
        # Group tracks by cluster
        from collections import defaultdict
        cluster_tracks: Dict[int, List[int]] = defaultdict(list)
        
        for track_id, cluster_id in track_to_cluster.items():
            if track_id in player_tracks:
                cluster_tracks[cluster_id].append(track_id)
        
        # Merge tracks in each cluster
        merged_tracks: Dict[int, 'PlayerTrack'] = {}
        merge_count = 0
        
        for cluster_id, track_ids in cluster_tracks.items():
            if len(track_ids) == 1:
                # Single track in cluster, keep as-is
                tid = track_ids[0]
                track = player_tracks[tid]
                track.appearance_cluster_id = cluster_id
                merged_tracks[tid] = track
            else:
                # Multiple tracks - merge them
                merge_count += len(track_ids) - 1
                
                # Sort by detection count, keep the one with most detections as primary
                tracks = [player_tracks[tid] for tid in track_ids]
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
                
                # Store cluster info
                primary.appearance_cluster_id = cluster_id
                primary.merged_track_ids = track_ids
                
                merged_tracks[primary.track_id] = primary
        
        if merge_count > 0:
            logger.info(f"Merged {merge_count} tracks by appearance similarity")
        
        logger.info(f"After appearance-based merging: {len(merged_tracks)} unique players")
        
        return merged_tracks
    
    def extract_appearance_features(self, crop: np.ndarray) -> Dict:
        """
        Extract human-readable appearance features from a player crop.
        
        This provides interpretable features like dominant colors
        that can be displayed in the UI.
        
        Args:
            crop: BGR image of player
            
        Returns:
            Dict with appearance features
        """
        if crop is None or crop.size == 0:
            return {}
        
        features = {}
        
        try:
            h, w = crop.shape[:2]
            
            # Extract jersey region (upper body)
            jersey_y1 = int(h * 0.15)
            jersey_y2 = int(h * 0.55)
            jersey_x1 = int(w * 0.15)
            jersey_x2 = int(w * 0.85)
            jersey_region = crop[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
            
            if jersey_region.size > 0:
                # Get dominant jersey color
                jersey_color = self._get_dominant_color(jersey_region)
                features['jersey_color'] = jersey_color
                features['jersey_color_name'] = self._color_to_name(jersey_color)
            
            # Extract shorts region (lower body)
            shorts_y1 = int(h * 0.55)
            shorts_y2 = int(h * 0.75)
            shorts_region = crop[shorts_y1:shorts_y2, jersey_x1:jersey_x2]
            
            if shorts_region.size > 0:
                shorts_color = self._get_dominant_color(shorts_region)
                features['shorts_color'] = shorts_color
                features['shorts_color_name'] = self._color_to_name(shorts_color)
            
            # Extract shoe region (feet)
            shoe_y1 = int(h * 0.85)
            shoe_region = crop[shoe_y1:, :]
            
            if shoe_region.size > 0:
                shoe_color = self._get_dominant_color(shoe_region)
                features['shoe_color'] = shoe_color
                features['shoe_color_name'] = self._color_to_name(shoe_color)
            
        except Exception as e:
            logger.warning(f"Failed to extract appearance features: {e}")
        
        return features
    
    def _get_dominant_color(self, region: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color from a region as RGB tuple"""
        if region is None or region.size == 0:
            return (128, 128, 128)
        
        # Convert to LAB for better color clustering
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)
        
        # Filter out very dark/bright pixels
        l_channel = pixels[:, 0]
        mask = (l_channel > 30) & (l_channel < 220)
        filtered = pixels[mask]
        
        if len(filtered) < 10:
            filtered = pixels
        
        # Get mean color
        mean_lab = filtered.mean(axis=0)
        
        # Convert back to RGB
        mean_lab_img = np.uint8([[mean_lab]])
        mean_bgr = cv2.cvtColor(mean_lab_img, cv2.COLOR_LAB2BGR)
        
        # BGR to RGB
        return tuple(int(c) for c in mean_bgr[0, 0][::-1])
    
    def _color_to_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB color to approximate color name"""
        r, g, b = rgb
        
        # Simple color classification
        if max(r, g, b) < 50:
            return "black"
        if min(r, g, b) > 200:
            return "white"
        
        # Check for grayscale
        if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            if max(r, g, b) > 150:
                return "light gray"
            return "gray"
        
        # Determine dominant channel
        if r > g and r > b:
            if r > 200 and g < 100 and b < 100:
                return "red"
            if r > 200 and g > 150:
                return "orange" if g < 200 else "yellow"
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            if b > 150 and r < 100:
                return "blue"
            if r > 100 and b > 150:
                return "purple"
            return "blue"
        
        return "mixed"


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize embedding to bytes for database storage"""
    if embedding is None:
        return None
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(data: bytes) -> Optional[np.ndarray]:
    """Deserialize embedding from bytes"""
    if data is None:
        return None
    return np.frombuffer(data, dtype=np.float32)
