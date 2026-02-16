"""
Celery Background Tasks
Handles video processing in the background
"""

import logging
from celery import Celery
from datetime import datetime
import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "sports_highlights",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time (GPU memory)
)


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, video_id: str, options: dict = None):
    """
    Main video processing task
    
    Options:
    - analysis_mode: "full" or "targeted"
    - target_jersey: Jersey number to focus on (if targeted mode)
    - home_team, away_team: Team names
    - home_color, away_color: Jersey colors
    
    Steps:
    1. Load video
    2. Run player detection and tracking
    3. Run jersey OCR
    4. Run action recognition
    5. Save results to database
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.database import Video, Player, PlayerSegment, Action
    from app.services.detector import PlayerDetector
    from app.services.jersey_ocr import JerseyOCR, temporal_vote_jersey
    from app.services.action_recognizer import ActionRecognizer
    from app.services.video_processor import get_video_info_sync
    
    # Parse options
    options = options or {}
    analysis_mode = options.get("analysis_mode", "full")
    target_jersey = options.get("target_jersey")
    home_team = options.get("home_team")
    away_team = options.get("away_team")
    home_color = options.get("home_color")
    away_color = options.get("away_color")
    
    logger.info(f"Starting video processing: {video_id}")
    logger.info(f"Analysis mode: {analysis_mode}, Target: {target_jersey}")
    
    # Create sync database session
    engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get video from database
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video not found: {video_id}")
            return {"error": "Video not found"}
        
        # Update status
        video.status = "processing"
        session.commit()
        
        # Get video info
        video_info = get_video_info_sync(video.file_path)
        fps = video_info.get("fps", 30.0)
        
        # Initialize services
        detector = PlayerDetector()
        ocr = JerseyOCR()
        action_recognizer = ActionRecognizer(fps=fps)
        
        def progress_callback(frame, total):
            progress = frame / total * 100
            self.update_state(
                state="PROGRESS",
                meta={"progress": progress, "frame": frame, "total": total}
            )
            logger.info(f"Processing: {progress:.1f}% ({frame}/{total})")
        
        # Run detection and tracking
        logger.info("Running player detection and tracking...")
        player_tracks, all_detections = detector.process_video(
            video.file_path,
            progress_callback=progress_callback
        )
        
        # Open video for OCR crops
        logger.info("Running jersey number OCR...")
        cap = cv2.VideoCapture(video.file_path)
        
        # Process each player track for jersey numbers
        for track_id, track in player_tracks.items():
            jersey_readings = []
            
            # Sample frames for OCR (every 10th detection to save time)
            sample_indices = list(range(0, len(track.detections), 10))[:30]  # Max 30 samples
            
            for idx in sample_indices:
                det = track.detections[idx]
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Get player crop
                    crop = detector.get_player_crop(frame, det.bbox)
                    
                    # Run OCR
                    number, conf = ocr.read_jersey_number(crop)
                    if number:
                        jersey_readings.append((number, conf))
                        track.jersey_numbers.append(number)
            
            # Apply temporal voting
            final_jersey = temporal_vote_jersey(jersey_readings)
            
            logger.info(f"Track {track_id}: Jersey #{final_jersey} "
                       f"(from {len(jersey_readings)} readings)")
        
        cap.release()
        
        # Run action recognition
        logger.info("Running action recognition...")
        
        # Prepare data for action recognizer
        player_positions = {}
        ball_positions = []
        
        for det in all_detections:
            if det.class_name == "person" and det.track_id >= 0:
                if det.track_id not in player_positions:
                    player_positions[det.track_id] = []
                player_positions[det.track_id].append((det.frame_number, det.bbox))
            elif det.class_name == "ball":
                ball_positions.append((det.frame_number, det.bbox))
        
        actions = action_recognizer.analyze_video(
            player_positions, ball_positions, fps
        )
        
        # Save results to database
        logger.info("Saving results to database...")
        
        # Create player records
        player_id_map = {}  # track_id -> player.id
        
        for track_id, track in player_tracks.items():
            # In targeted mode, skip players that don't match target jersey
            if analysis_mode == "targeted" and target_jersey:
                if track.jersey_number != target_jersey:
                    continue
            
            # Determine team based on jersey color (if provided)
            team = None
            if home_team and away_team:
                # This is a simplified assignment - could be enhanced with color detection
                team = home_team  # Default, would need color analysis to improve
            
            player = Player(
                video_id=video.id,
                jersey_number=track.jersey_number,
                team=team,
                track_id=track_id,
                confidence=track.confidence,
                first_seen_frame=track.first_frame,
                last_seen_frame=track.last_frame
            )
            session.add(player)
            session.flush()  # Get the ID
            player_id_map[track_id] = player.id
            
            # Create segments (merge consecutive detections)
            segments = merge_detections_to_segments(track.detections, fps, gap_threshold=1.0)
            
            for start_frame, end_frame in segments:
                segment = PlayerSegment(
                    player_id=player.id,
                    video_id=video.id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    is_active=True
                )
                session.add(segment)
        
        # Create action records
        for action in actions:
            player_id = None
            if action.player_track_id is not None:
                player_id = player_id_map.get(action.player_track_id)
            
            action_record = Action(
                video_id=video.id,
                player_id=player_id,
                action_type=action.action_type.value,
                frame=action.frame,
                timestamp=action.timestamp,
                confidence=action.confidence,
                metadata=action.metadata
            )
            session.add(action_record)
        
        # Update video status
        video.status = "completed"
        video.processed_at = datetime.utcnow()
        session.commit()
        
        logger.info(f"Video processing complete: {video_id}")
        logger.info(f"  - Players: {len(player_tracks)}")
        logger.info(f"  - Actions: {len(actions)}")
        
        return {
            "status": "completed",
            "players": len(player_tracks),
            "actions": len(actions)
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        
        # Update video status
        video.status = "failed"
        video.error_message = str(e)
        session.commit()
        
        return {"error": str(e)}
        
    finally:
        session.close()


def merge_detections_to_segments(detections, fps, gap_threshold=1.0):
    """
    Merge consecutive detections into segments
    
    Args:
        detections: List of Detection objects
        fps: Video frame rate
        gap_threshold: Max gap in seconds to merge
        
    Returns:
        List of (start_frame, end_frame) tuples
    """
    if not detections:
        return []
    
    # Sort by frame
    sorted_dets = sorted(detections, key=lambda d: d.frame_number)
    
    segments = []
    current_start = sorted_dets[0].frame_number
    current_end = sorted_dets[0].frame_number
    
    gap_frames = int(gap_threshold * fps)
    
    for det in sorted_dets[1:]:
        if det.frame_number - current_end <= gap_frames:
            # Extend current segment
            current_end = det.frame_number
        else:
            # Save current segment and start new one
            segments.append((current_start, current_end))
            current_start = det.frame_number
            current_end = det.frame_number
    
    # Don't forget the last segment
    segments.append((current_start, current_end))
    
    return segments
