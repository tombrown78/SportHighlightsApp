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


def get_youtube_auth_args() -> list:
    """
    Get authentication arguments for yt-dlp based on available credentials.
    
    Supports multiple authentication methods:
    1. Cookies file (most reliable for private videos)
    2. OAuth token (if configured)
    3. Username/password (legacy, not recommended)
    """
    auth_args = []
    
    # Method 1: Cookies file (recommended)
    cookies_path = os.path.join(settings.VIDEOS_DIR, "youtube_cookies.txt")
    if os.path.exists(cookies_path):
        auth_args.extend(["--cookies", cookies_path])
        logger.info("Using YouTube cookies file for authentication")
    
    # Method 2: OAuth2 (yt-dlp built-in)
    elif settings.YOUTUBE_USE_OAUTH:
        auth_args.extend(["--username", "oauth2", "--password", ""])
        logger.info("Using YouTube OAuth2 authentication")
    
    # Method 3: Username/password (legacy - not recommended due to 2FA)
    elif settings.YOUTUBE_USERNAME and settings.YOUTUBE_PASSWORD:
        auth_args.extend([
            "--username", settings.YOUTUBE_USERNAME,
            "--password", settings.YOUTUBE_PASSWORD
        ])
        logger.info("Using YouTube username/password authentication")
    
    return auth_args


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
    
    # Base command
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--print", "filename",
    ]
    
    # Add authentication if available
    auth_args = get_youtube_auth_args()
    cmd.extend(auth_args)
    
    # Add the URL
    cmd.append(url)
    
    logger.info(f"Downloading YouTube video: {url}")
    if auth_args:
        logger.info("Authentication enabled for this download")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        logger.error(f"yt-dlp error: {error_msg}")
        
        # Check for common auth errors
        if "Private video" in error_msg:
            raise Exception("This is a private video. Please set up YouTube authentication in settings.")
        elif "Sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted video. Please set up YouTube authentication in settings.")
        elif "requires authentication" in error_msg.lower():
            raise Exception("This video requires authentication. Please set up YouTube credentials.")
        
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
    title_cmd.extend(auth_args)  # Use same auth for title fetch
    
    title_process = await asyncio.create_subprocess_exec(
        *title_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    title_stdout, _ = await title_process.communicate()
    original_filename = title_stdout.decode().strip() + ".mp4" if title_stdout else f"{video_id}.mp4"
    
    logger.info(f"Downloaded: {output_file}")
    
    return output_file, original_filename


async def setup_youtube_oauth():
    """
    Interactive OAuth setup for YouTube.
    This should be run once to authenticate with YouTube.
    
    Returns:
        bool: True if authentication was successful
    """
    logger.info("Starting YouTube OAuth setup...")
    
    # yt-dlp will open a browser for OAuth
    cmd = [
        "yt-dlp",
        "--username", "oauth2",
        "--password", "",
        "--skip-download",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Any video to trigger auth
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        logger.info("YouTube OAuth setup completed successfully")
        return True
    else:
        logger.error(f"OAuth setup failed: {stderr.decode()}")
        return False


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
