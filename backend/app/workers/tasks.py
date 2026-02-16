"""
Celery Background Tasks
Handles video processing in the background
"""

import logging
import json
from celery import Celery
from datetime import datetime
import cv2
import redis

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


class ProgressPublisher:
    """Publishes processing progress events to Redis for SSE streaming"""
    
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.channel = f"video_progress:{video_id}"
        self.cancel_key = f"video_cancel:{video_id}"
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.detection_count = 0
        self.action_count = 0
        self.players_found = set()
    
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested"""
        try:
            return self.redis_client.exists(self.cancel_key) > 0
        except Exception as e:
            logger.warning(f"Error checking cancellation: {e}")
            return False
    
    def clear_cancel_flag(self):
        """Clear the cancellation flag"""
        try:
            self.redis_client.delete(self.cancel_key)
        except Exception as e:
            logger.warning(f"Error clearing cancel flag: {e}")
        
    def publish(self, event_type: str, data: dict):
        """Publish an event to Redis"""
        event = {"event": event_type, **data}
        try:
            self.redis_client.publish(self.channel, json.dumps(event))
        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")
    
    def progress(self, frame: int, total_frames: int, stage: str):
        """Publish progress update"""
        percent = (frame / total_frames * 100) if total_frames > 0 else 0
        self.publish("progress", {
            "percent": round(percent, 1),
            "frame": frame,
            "total_frames": total_frames,
            "stage": stage
        })
    
    def detection(self, track_id: int, class_name: str, confidence: float, frame: int, bbox: list = None):
        """Publish detection event"""
        self.detection_count += 1
        # Only publish every 10th detection to avoid flooding
        if self.detection_count % 10 == 0:
            self.publish("detection", {
                "track_id": track_id,
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "frame": frame,
                "count": self.detection_count
            })
    
    def player_found(self, track_id: int, jersey_number: str, confidence: float):
        """Publish when a new player is identified"""
        if track_id not in self.players_found:
            self.players_found.add(track_id)
            self.publish("player", {
                "track_id": track_id,
                "jersey_number": jersey_number,
                "confidence": round(confidence, 2),
                "total_players": len(self.players_found)
            })
    
    def action_detected(self, action_type: str, player_track_id: int, confidence: float, frame: int, timestamp: float):
        """Publish action detection event"""
        self.action_count += 1
        self.publish("action", {
            "action_type": action_type,
            "player_track_id": player_track_id,
            "confidence": round(confidence, 2),
            "frame": frame,
            "timestamp": round(timestamp, 2),
            "count": self.action_count
        })
    
    def stage_change(self, stage: str, message: str):
        """Publish stage change event"""
        self.publish("stage", {
            "stage": stage,
            "message": message
        })
    
    def complete(self, players_count: int, actions_count: int):
        """Publish completion event"""
        self.publish("complete", {
            "players_count": players_count,
            "actions_count": actions_count,
            "status": "completed"
        })
    
    def error(self, message: str):
        """Publish error event"""
        self.publish("error", {
            "message": message
        })
    
    def cancelled(self):
        """Publish cancellation event"""
        self.publish("cancelled", {
            "message": "Processing was cancelled",
            "status": "cancelled"
        })
    
    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
        except:
            pass


class CancelledException(Exception):
    """Raised when video processing is cancelled"""
    pass


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
    from app.services.detector import PlayerDetector, filter_valid_tracks, deduplicate_tracks_by_jersey
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
    
    # Initialize progress publisher
    progress_pub = ProgressPublisher(video_id)
    
    # Create sync database session
    engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get video from database
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"Video not found: {video_id}")
            progress_pub.error("Video not found")
            return {"error": "Video not found"}
        
        # Check for cancellation before starting
        if progress_pub.is_cancelled():
            raise CancelledException("Cancelled before processing started")
        
        # Update status
        video.status = "processing"
        session.commit()
        
        progress_pub.stage_change("init", "Initializing video analysis...")
        
        # Get video info
        video_info = get_video_info_sync(video.file_path)
        fps = video_info.get("fps", 30.0)
        total_frames = int(video_info.get("duration", 0) * fps)
        
        # Check for cancellation
        if progress_pub.is_cancelled():
            raise CancelledException("Cancelled during initialization")
        
        # Initialize services
        progress_pub.stage_change("loading", "Loading AI models...")
        detector = PlayerDetector()
        ocr = JerseyOCR()
        action_recognizer = ActionRecognizer(fps=fps)
        
        # Check for cancellation after model loading
        if progress_pub.is_cancelled():
            raise CancelledException("Cancelled after model loading")
        
        def progress_callback(frame, total):
            # Check for cancellation during detection
            if progress_pub.is_cancelled():
                raise CancelledException("Cancelled during detection")
            
            progress = frame / total * 100
            self.update_state(
                state="PROGRESS",
                meta={"progress": progress, "frame": frame, "total": total}
            )
            progress_pub.progress(frame, total, "detection")
            logger.info(f"Processing: {progress:.1f}% ({frame}/{total})")
        
        def detection_callback(track_id, class_name, confidence, frame, bbox):
            """Called for each detection"""
            progress_pub.detection(track_id, class_name, confidence, frame, bbox)
        
        # Run detection and tracking
        progress_pub.stage_change("detection", "Detecting and tracking players...")
        logger.info("Running player detection and tracking...")
        player_tracks, all_detections = detector.process_video(
            video.file_path,
            progress_callback=progress_callback,
            detection_callback=detection_callback
        )
        
        # Get video dimensions for filtering
        cap_temp = cv2.VideoCapture(video.file_path)
        frame_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_temp.release()
        
        # Filter tracks to remove false positives (crowd, brief appearances, etc.)
        progress_pub.stage_change("filtering", "Filtering valid player tracks...")
        logger.info(f"Filtering {len(player_tracks)} raw tracks...")
        player_tracks = filter_valid_tracks(
            player_tracks,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height
        )
        logger.info(f"After filtering: {len(player_tracks)} tracks remain")
        
        # Open video for OCR crops
        progress_pub.stage_change("ocr", "Reading jersey numbers...")
        logger.info("Running jersey number OCR...")
        cap = cv2.VideoCapture(video.file_path)
        
        # Check for cancellation before OCR
        if progress_pub.is_cancelled():
            cap.release()
            raise CancelledException("Cancelled before OCR")
        
        # Process each player track for jersey numbers
        track_count = 0
        total_tracks = len(player_tracks)
        
        for track_id, track in player_tracks.items():
            # Check for cancellation every few tracks
            if track_count % 5 == 0 and progress_pub.is_cancelled():
                cap.release()
                raise CancelledException("Cancelled during OCR")
            
            track_count += 1
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
            
            # Publish player found event
            avg_conf = sum(r[1] for r in jersey_readings) / len(jersey_readings) if jersey_readings else 0
            progress_pub.player_found(track_id, final_jersey or "?", avg_conf)
            
            # Update OCR progress
            progress_pub.progress(track_count, total_tracks, "ocr")
            
            logger.info(f"Track {track_id}: Jersey #{final_jersey} "
                       f"(from {len(jersey_readings)} readings)")
        
        cap.release()
        
        # Deduplicate tracks by jersey number (merge fragmented tracks)
        progress_pub.stage_change("dedup", "Merging duplicate player tracks...")
        logger.info(f"Deduplicating {len(player_tracks)} tracks by jersey number...")
        player_tracks = deduplicate_tracks_by_jersey(player_tracks)
        logger.info(f"After deduplication: {len(player_tracks)} unique players")
        
        # Check for cancellation before team classification
        if progress_pub.is_cancelled():
            raise CancelledException("Cancelled before team classification")
        
        # Run team classification using jersey colors
        progress_pub.stage_change("teams", "Classifying teams by jersey color...")
        logger.info("Running team classification...")
        
        team_assignments = {}
        try:
            from app.services.team_classifier import classify_teams_from_video
            team_assignments = classify_teams_from_video(
                video.file_path,
                player_tracks,
                sample_interval=30
            )
            logger.info(f"Team classification complete: {len(team_assignments)} players assigned")
        except Exception as e:
            logger.warning(f"Team classification failed (non-fatal): {e}")
        
        # Check for cancellation before action recognition
        if progress_pub.is_cancelled():
            raise CancelledException("Cancelled before action recognition")
        
        # Run action recognition
        progress_pub.stage_change("actions", "Recognizing basketball actions...")
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
        
        # Publish action events
        for action in actions:
            progress_pub.action_detected(
                action.action_type.value,
                action.player_track_id,
                action.confidence,
                action.frame,
                action.timestamp
            )
        
        # Check for cancellation before saving
        if progress_pub.is_cancelled():
            raise CancelledException("Cancelled before saving results")
        
        # Save results to database
        progress_pub.stage_change("saving", "Saving analysis results...")
        logger.info("Saving results to database...")
        
        # Create player records
        player_id_map = {}  # track_id -> player.id
        
        for track_id, track in player_tracks.items():
            # In targeted mode, skip players that don't match target jersey
            if analysis_mode == "targeted" and target_jersey:
                if track.jersey_number != target_jersey:
                    continue
            
            # Determine team from classification or fallback to provided names
            team = None
            team_color = None
            
            if track_id in team_assignments:
                assignment = team_assignments[track_id]
                if assignment.team.value == "team_a":
                    team = home_team or "Team A"
                elif assignment.team.value == "team_b":
                    team = away_team or "Team B"
                elif assignment.team.value == "referee":
                    team = "Referee"
                team_color = assignment.dominant_color  # RGB tuple
            elif home_team:
                team = home_team  # Fallback to provided team name
            
            # Convert RGB tuple to hex color string
            team_color_hex = None
            if team_color:
                team_color_hex = "#{:02x}{:02x}{:02x}".format(*team_color)
            
            player = Player(
                video_id=video.id,
                jersey_number=track.jersey_number,
                team=team,
                team_color=team_color_hex,
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
                action_data=action.metadata
            )
            session.add(action_record)
        
        # Update video status
        video.status = "completed"
        video.processed_at = datetime.utcnow()
        session.commit()
        
        # Publish completion
        progress_pub.complete(len(player_tracks), len(actions))
        
        logger.info(f"Video processing complete: {video_id}")
        logger.info(f"  - Players: {len(player_tracks)}")
        logger.info(f"  - Actions: {len(actions)}")
        
        return {
            "status": "completed",
            "players": len(player_tracks),
            "actions": len(actions)
        }
    
    except CancelledException as e:
        logger.info(f"Video processing cancelled: {video_id} - {e}")
        
        # Update video status
        video.status = "cancelled"
        video.error_message = None
        session.commit()
        
        # Publish cancellation event
        progress_pub.cancelled()
        
        # Clear the cancel flag
        progress_pub.clear_cancel_flag()
        
        return {"status": "cancelled", "message": str(e)}
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        
        # Update video status
        video.status = "failed"
        video.error_message = str(e)
        session.commit()
        
        # Publish error
        progress_pub.error(str(e))
        
        return {"error": str(e)}
        
    finally:
        progress_pub.close()
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
