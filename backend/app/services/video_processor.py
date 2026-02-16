"""
Video Processing Service
Handles video downloads, metadata extraction, and processing
"""

import logging
import os
import asyncio
from typing import Tuple, Dict, Optional
import subprocess
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


async def download_youtube_video(url: str, video_id) -> Tuple[str, str]:
    """
    Download video from YouTube using yt-dlp
    
    Args:
        url: YouTube video URL
        video_id: UUID for the video
        
    Returns:
        Tuple of (file_path, original_filename)
    """
    output_template = os.path.join(settings.VIDEOS_DIR, f"{video_id}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--print", "filename",
        url
    ]
    
    logger.info(f"Downloading YouTube video: {url}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        logger.error(f"yt-dlp error: {error_msg}")
        raise Exception(f"Failed to download video: {error_msg}")
    
    # Get the output filename
    output_file = stdout.decode().strip().split('\n')[-1]
    
    if not os.path.exists(output_file):
        # Try with .mp4 extension
        output_file = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
    
    if not os.path.exists(output_file):
        raise Exception("Downloaded file not found")
    
    # Get original title for filename
    title_cmd = ["yt-dlp", "--get-title", url]
    title_process = await asyncio.create_subprocess_exec(
        *title_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    title_stdout, _ = await title_process.communicate()
    original_filename = title_stdout.decode().strip() + ".mp4" if title_stdout else f"{video_id}.mp4"
    
    logger.info(f"Downloaded: {output_file}")
    
    return output_file, original_filename


async def get_video_info(file_path: str) -> Dict:
    """
    Get video metadata using ffprobe
    
    Args:
        file_path: Path to video file
        
    Returns:
        Dict with duration, width, height, fps
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        logger.warning(f"ffprobe error: {stderr.decode() if stderr else 'Unknown'}")
        return {}
    
    try:
        data = json.loads(stdout.decode())
        
        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            return {}
        
        # Parse frame rate
        fps = 30.0
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        
        # Get duration
        duration = float(data.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))
        
        return {
            "duration": duration,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps
        }
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Error parsing video info: {e}")
        return {}


def get_video_info_sync(file_path: str) -> Dict:
    """Synchronous version of get_video_info for use in Celery tasks"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {}
    
    try:
        data = json.loads(result.stdout)
        
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            return {}
        
        fps = 30.0
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        
        duration = float(data.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))
        
        return {
            "duration": duration,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps
        }
        
    except (json.JSONDecodeError, KeyError, ValueError):
        return {}
