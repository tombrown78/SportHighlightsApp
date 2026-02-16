# Sports Highlights App - Basketball Video Analysis

AI-powered basketball video analysis application that identifies players, tracks their movements, detects game actions, and generates highlight clips. Inspired by apps like [Hooper](https://www.hooper.gg/).

## Features

### Core Features
- **Player Detection & Tracking**: YOLOv8 + ByteTrack for accurate multi-player tracking
- **Jersey Number Recognition**: EasyOCR with temporal voting for reliable jersey identification
- **Team Classification**: Automatic team assignment using K-Means clustering on jersey colors
- **Action Recognition**: Detects shots, rebounds, assists, steals, and other basketball actions

### Advanced Features (New)
- **Hoop/Rim Detection**: Basketball-specific detection using Roboflow pre-trained models
- **Shot Made/Missed Classification**: Analyzes ball trajectory relative to hoop
- **Pose Estimation**: MediaPipe-based player pose analysis for shot form feedback
- **Dead Time Detection**: Automatically identifies inactive periods for game condensation
- **Team Color Display**: Visual team indicators in the UI

### Output Features
- **Timeline Generation**: Interactive timeline showing player involvement
- **Clip Export**: Generate highlight clips for specific players
- **Play-by-Play**: Chronological list of all detected actions
- **Shot Charts**: Visual representation of shot locations (coming soon)

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (RTX 4080 with 16GB VRAM recommended)
- Minimum 8GB VRAM for full pipeline
- ~15GB disk space for models and dependencies

### Software
- Docker Desktop with WSL2 backend (Windows) or Docker Engine (Linux)
- NVIDIA Container Toolkit installed
- Git

## Quick Start

### 1. Clone and Configure

```powershell
# Clone the repository
git clone https://github.com/yourusername/SportHighlightsApp.git
cd SportHighlightsApp

# Copy environment file
cp .env.example .env

# Edit .env with your settings (see Configuration section below)
```

### 2. Start Services

```powershell
# Start all services (first run downloads AI models ~2-3GB)
docker compose up -d

# View logs to monitor startup
docker compose logs -f backend

# Wait for "Application startup complete" message
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Configuration

### Environment Variables (.env)

```bash
# Database (default values work for local development)
POSTGRES_USER=sports
POSTGRES_PASSWORD=sports123
POSTGRES_DB=sports_highlights

# Processing Mode
PROCESSING_MODE=local  # Use local GPU

# ============ OPTIONAL: Enhanced Detection ============

# Roboflow API Key (for basketball/hoop detection)
# Get a free API key at: https://roboflow.com/
ROBOFLOW_API_KEY=your_api_key_here

# ============ OPTIONAL: YouTube Authentication ============

# For private/age-restricted videos
YOUTUBE_USE_OAUTH=false
```

### Setting Up Roboflow (Optional but Recommended)

Roboflow provides pre-trained models for basketball and hoop detection, which significantly improves shot detection accuracy.

1. **Create a free account** at [roboflow.com](https://roboflow.com/)

2. **Get your API key**:
   - Go to Settings → API Keys
   - Copy your Private API Key

3. **Add to .env file**:
   ```bash
   ROBOFLOW_API_KEY=your_api_key_here
   ```

4. **Restart the backend**:
   ```powershell
   docker compose restart backend worker
   ```

**Without Roboflow**: The app falls back to COCO's "sports ball" class for ball detection. Hoop detection and shot made/missed classification will not be available.

**With Roboflow**: You get:
- Basketball-specific ball detection (more accurate)
- Hoop/rim detection
- Shot made/missed classification (95%+ accuracy)

## Database Setup

### New Installation

The database schema is created automatically on first startup. No manual SQL commands needed.

### Existing Installation (Adding New Features)

If you're upgrading from a previous version, run this SQL command to add the new `team_color` column:

```powershell
# Connect to the database
docker exec -it sports-postgres psql -U sports -d sports_highlights

# Run the migration
ALTER TABLE players ADD COLUMN IF NOT EXISTS team_color VARCHAR(7);

# Exit
\q
```

Or as a one-liner:
```powershell
docker exec sports-postgres psql -U sports -d sports_highlights -c "ALTER TABLE players ADD COLUMN IF NOT EXISTS team_color VARCHAR(7);"
```

### Verify Database Schema

```powershell
# List all tables
docker exec sports-postgres psql -U sports -d sports_highlights -c "\dt"

# Check players table structure
docker exec sports-postgres psql -U sports -d sports_highlights -c "\d players"

# Check row counts
docker exec sports-postgres psql -U sports -d sports_highlights -c "SELECT 'videos' as table_name, COUNT(*) FROM videos UNION ALL SELECT 'players', COUNT(*) FROM players UNION ALL SELECT 'actions', COUNT(*) FROM actions UNION ALL SELECT 'clips', COUNT(*) FROM clips;"
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend       │
│   (Next.js)     │     │   (FastAPI)     │
│   Port 3000     │     │   Port 8000     │
└─────────────────┘     └────────┬────────┘
                                │
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │PostgreSQL│ │  Redis   │ │  Worker  │
             │  :5432   │ │  :6379   │ │ (Celery) │
             └──────────┘ └──────────┘ └──────────┘
```

### Processing Pipeline

```
Video Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Player Detection (YOLOv8x)                           │
│    - Detects players and ball in each frame             │
│    - Optional: Roboflow model for hoop detection        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Multi-Object Tracking (ByteTrack)                    │
│    - Maintains consistent player IDs across frames      │
│    - 90-frame track buffer for occlusion handling       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Jersey Number OCR (EasyOCR)                          │
│    - Reads jersey numbers from player crops             │
│    - Temporal voting for accuracy                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Team Classification (K-Means)                        │
│    - Clusters players by jersey color                   │
│    - Identifies referees                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Action Recognition                                   │
│    - Ball trajectory analysis for shots                 │
│    - Player-ball proximity for possession               │
│    - Optional: Pose estimation for shot form            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Results stored in PostgreSQL
```

## Supported Video Formats

- **File uploads**: MP4, MOV, AVI, MKV, WEBM, WMV (up to 2GB)
- **YouTube**: Any public or unlisted video URL
- **Resolution**: 480p to 4K (higher resolution = better jersey detection)

## Development

### Running Locally

```powershell
# Start with hot-reload
docker compose up

# Rebuild after dependency changes
docker compose build backend

# Check GPU is accessible
docker compose exec backend nvidia-smi

# Run tests
docker compose exec backend pytest
```

### Backend Development

```powershell
# Enter backend container
docker compose exec backend bash

# Install new dependencies
pip install package_name
pip freeze > requirements.txt

# Run specific tests
pytest tests/test_detector.py -v
```

### Frontend Development

```powershell
# Enter frontend container
docker compose exec frontend sh

# Install new dependencies
npm install package_name
```

### Useful Commands

```powershell
# View real-time logs
docker compose logs -f backend worker

# Restart specific service
docker compose restart backend

# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes database)
docker compose down -v

# Check service status
docker compose ps
```

## API Endpoints

### Videos
- `POST /api/videos/upload` - Upload a video file
- `POST /api/videos/youtube` - Analyze YouTube video
- `GET /api/videos/{id}` - Get video details
- `GET /api/videos/{id}/stream` - Stream video file
- `DELETE /api/videos/{id}` - Delete video and analysis

### Players
- `GET /api/players/video/{video_id}` - Get all players in video
- `GET /api/players/{id}` - Get player details
- `GET /api/players/{id}/timeline` - Get player timeline
- `GET /api/players/{id}/stats` - Get player statistics

### Actions
- `GET /api/videos/{id}/actions` - Get all actions in video

### Clips
- `GET /api/clips/video/{video_id}` - Get all clips for video
- `POST /api/clips/player/{player_id}/highlights` - Generate highlights
- `DELETE /api/clips/{id}` - Delete a clip

## Troubleshooting

### GPU Not Detected

```powershell
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Ensure NVIDIA Container Toolkit is installed
# Windows: Included with Docker Desktop
# Linux: sudo apt install nvidia-container-toolkit
```

### Database Connection Issues

```powershell
# Check if PostgreSQL is running
docker compose ps postgres

# View PostgreSQL logs
docker compose logs postgres

# Reset database (WARNING: deletes all data)
docker compose down -v
docker compose up -d
```

### Video Processing Stuck

```powershell
# Check worker logs
docker compose logs -f worker

# Restart worker
docker compose restart worker

# Check Redis connection
docker compose exec backend redis-cli -h redis ping
```

### Out of Memory (OOM)

If you see CUDA out of memory errors:

1. Reduce batch size in `backend/app/core/config.py`
2. Use a smaller YOLO model (yolov8m instead of yolov8x)
3. Process lower resolution videos

## YouTube Authentication (Optional)

For private or age-restricted videos:

### Method 1: Cookies File (Recommended)

1. Install browser extension "Get cookies.txt" (Chrome/Firefox)
2. Go to YouTube and log in
3. Export cookies for youtube.com
4. Save as `youtube_cookies.txt` in project root
5. Mount in Docker (already configured in docker-compose.yml)

### Method 2: OAuth2

1. Set `YOUTUBE_USE_OAUTH=true` in `.env`
2. Check backend logs for authentication URL
3. Open URL in browser and log in
4. Token is cached for future use

### Check Auth Status

```bash
curl http://localhost:8000/api/videos/youtube/auth-status
```

## Performance Expectations

| Video Length | Processing Time | Notes |
|--------------|-----------------|-------|
| 1 minute     | ~2-3 minutes    | Full analysis |
| 10 minutes   | ~15-20 minutes  | Full analysis |
| 1 hour       | ~60-90 minutes  | Full analysis |

Processing speed depends on:
- GPU performance (RTX 4080 recommended)
- Video resolution
- Number of players on screen
- Analysis mode (full vs targeted)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `docker compose exec backend pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [Supervision](https://github.com/roboflow/supervision) - ByteTrack implementation
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Jersey number recognition
- [MediaPipe](https://github.com/google/mediapipe) - Pose estimation
- [Roboflow](https://roboflow.com/) - Basketball detection models
- [Hooper](https://www.hooper.gg/) - Inspiration for features
