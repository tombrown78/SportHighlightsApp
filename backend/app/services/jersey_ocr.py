"""
Jersey Number OCR Service
Uses PaddleOCR for reading jersey numbers from player crops
"""

import logging
from typing import Optional, List, Tuple
import numpy as np
import cv2
import re

logger = logging.getLogger(__name__)

# Lazy load PaddleOCR to avoid import issues
_ocr_instance = None


def get_ocr():
    """Get or create PaddleOCR instance"""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,
            show_log=False
        )
        logger.info("PaddleOCR initialized")
    return _ocr_instance


class JerseyOCR:
    """
    Jersey number recognition using OCR
    Optimized for sports jerseys with temporal voting
    """
    
    def __init__(self):
        self.ocr = None  # Lazy load
        self.jersey_pattern = re.compile(r'^[0-9]{1,2}$')  # 1-2 digit numbers
        
    def _ensure_ocr(self):
        """Ensure OCR is loaded"""
        if self.ocr is None:
            self.ocr = get_ocr()
    
    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess player crop for better OCR results
        
        Args:
            crop: BGR image of player
            
        Returns:
            Preprocessed image
        """
        if crop is None or crop.size == 0:
            return None
        
        # Resize if too small
        h, w = crop.shape[:2]
        if h < 50 or w < 30:
            scale = max(50 / h, 30 / w)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Convert back to BGR for PaddleOCR
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def extract_jersey_region(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract the jersey/torso region from player crop
        Jersey numbers are typically in the upper-middle portion
        """
        if crop is None or crop.size == 0:
            return crop
        
        h, w = crop.shape[:2]
        
        # Focus on upper 60% (torso area) and middle 80% width
        y_start = int(h * 0.15)
        y_end = int(h * 0.65)
        x_start = int(w * 0.1)
        x_end = int(w * 0.9)
        
        return crop[y_start:y_end, x_start:x_end]
    
    def read_jersey_number(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Read jersey number from player crop
        
        Args:
            crop: BGR image of player
            
        Returns:
            Tuple of (jersey_number or None, confidence)
        """
        self._ensure_ocr()
        
        if crop is None or crop.size == 0:
            return None, 0.0
        
        try:
            # Extract jersey region
            jersey_region = self.extract_jersey_region(crop)
            if jersey_region is None or jersey_region.size == 0:
                return None, 0.0
            
            # Preprocess
            processed = self.preprocess_crop(jersey_region)
            if processed is None:
                return None, 0.0
            
            # Run OCR
            results = self.ocr.ocr(processed, cls=True)
            
            if not results or not results[0]:
                return None, 0.0
            
            # Find jersey numbers in results
            best_number = None
            best_confidence = 0.0
            
            for line in results[0]:
                if line and len(line) >= 2:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    
                    # Check if it looks like a jersey number
                    if self.jersey_pattern.match(text):
                        # Prefer higher confidence
                        if confidence > best_confidence:
                            best_number = text
                            best_confidence = confidence
            
            return best_number, best_confidence
            
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return None, 0.0
    
    def read_jersey_numbers_batch(
        self,
        crops: List[np.ndarray]
    ) -> List[Tuple[Optional[str], float]]:
        """
        Read jersey numbers from multiple crops
        
        Args:
            crops: List of BGR images
            
        Returns:
            List of (jersey_number, confidence) tuples
        """
        results = []
        for crop in crops:
            number, conf = self.read_jersey_number(crop)
            results.append((number, conf))
        return results


def temporal_vote_jersey(
    jersey_readings: List[Tuple[Optional[str], float]],
    min_votes: int = 3,
    min_confidence: float = 0.5
) -> Optional[str]:
    """
    Apply temporal voting to determine most likely jersey number
    
    Args:
        jersey_readings: List of (number, confidence) from multiple frames
        min_votes: Minimum number of consistent readings required
        min_confidence: Minimum average confidence required
        
    Returns:
        Most likely jersey number or None
    """
    from collections import defaultdict
    
    # Count votes weighted by confidence
    votes = defaultdict(lambda: {"count": 0, "total_conf": 0.0})
    
    for number, conf in jersey_readings:
        if number is not None and conf >= min_confidence:
            votes[number]["count"] += 1
            votes[number]["total_conf"] += conf
    
    if not votes:
        return None
    
    # Find best candidate
    best_number = None
    best_score = 0
    
    for number, data in votes.items():
        if data["count"] >= min_votes:
            # Score = count * average confidence
            avg_conf = data["total_conf"] / data["count"]
            score = data["count"] * avg_conf
            
            if score > best_score:
                best_score = score
                best_number = number
    
    return best_number
