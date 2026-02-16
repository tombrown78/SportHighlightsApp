"""
Clip Generation Service
Uses FFmpeg to extract video clips
"""

import logging
import asyncio
import subprocess
import os

logger = logging.getLogger(__name__)


async def generate_clip(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    reencode: bool = False
) -> str:
    """
    Generate a video clip from input video
    
    Args:
        input_path: Path to source video
        output_path: Path for output clip
        start_time: Start time in seconds
        end_time: End time in seconds
        reencode: If True, re-encode video (slower but more accurate cuts)
        
    Returns:
        Path to generated clip
    """
    duration = end_time - start_time
    
    if reencode:
        # Re-encode for frame-accurate cuts
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]
    else:
        # Stream copy for fast extraction (may have slight inaccuracy at start)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path
        ]
    
    logger.info(f"Generating clip: {start_time:.2f}s - {end_time:.2f}s")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        logger.error(f"FFmpeg error: {error_msg}")
        raise Exception(f"Failed to generate clip: {error_msg}")
    
    if not os.path.exists(output_path):
        raise Exception("Clip file was not created")
    
    logger.info(f"Clip generated: {output_path}")
    
    return output_path


def generate_clip_sync(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    reencode: bool = False
) -> str:
    """Synchronous version for use in Celery tasks"""
    duration = end_time - start_time
    
    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to generate clip: {result.stderr}")
    
    return output_path


async def generate_thumbnail(
    input_path: str,
    output_path: str,
    timestamp: float = 0,
    width: int = 320
) -> str:
    """
    Generate a thumbnail image from video
    
    Args:
        input_path: Path to source video
        output_path: Path for output image (jpg/png)
        timestamp: Time in seconds to capture
        width: Output width (height auto-calculated)
        
    Returns:
        Path to generated thumbnail
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", input_path,
        "-vframes", "1",
        "-vf", f"scale={width}:-1",
        output_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    await process.communicate()
    
    if process.returncode != 0 or not os.path.exists(output_path):
        raise Exception("Failed to generate thumbnail")
    
    return output_path
