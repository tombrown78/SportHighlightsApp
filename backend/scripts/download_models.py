#!/usr/bin/env python3
"""
Pre-download AI models for the Sports Highlights App.
Run this script once after installation to avoid runtime downloads.

Usage:
    python scripts/download_models.py
"""

import os
import sys

def download_yolo_model():
    """Download YOLOv8 model"""
    print("Downloading YOLOv8x model...")
    from ultralytics import YOLO
    
    # This will download the model if not present
    model = YOLO("yolov8x.pt")
    print(f"YOLOv8x model ready at: {model.ckpt_path}")
    return model.ckpt_path

def download_easyocr_models():
    """Download EasyOCR models"""
    print("Downloading EasyOCR models...")
    import easyocr
    
    # This will download English models
    reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for download
    print("EasyOCR models ready")

def main():
    print("=" * 50)
    print("Sports Highlights - Model Downloader")
    print("=" * 50)
    print()
    
    try:
        yolo_path = download_yolo_model()
        print(f"  YOLO model: {yolo_path}")
    except Exception as e:
        print(f"  Error downloading YOLO: {e}")
        sys.exit(1)
    
    print()
    
    try:
        download_easyocr_models()
    except Exception as e:
        print(f"  Error downloading EasyOCR models: {e}")
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("All models downloaded successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
