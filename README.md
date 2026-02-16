# Sports Highlights App - Basketball Video Analysis

AI-powered basketball video analysis application that identifies players, tracks their involvement in plays, and detects game actions.

## Features

- **Player Detection & Tracking**: Uses YOLOv8 + ByteTrack to detect and track players throughout the video
- **Jersey Number Recognition**: OCR-based identification of players by jersey number
- **Action Recognition**: Detects shots, rebounds, assists, steals, and other basketball actions
- **Timeline Generation**: Shows when specific players are involved in the action
- **Clip Export**: Generate highlight clips for specific players
- **YouTube Support**: Analyze videos directly from YouTube URLs

## Requirements

- Docker Desktop with WSL2 backend
- NVIDIA GPU with CUDA support (RTX 4080 with 16GB VRAM recommended)
- NVIDIA Container Toolkit installed
- ~10GB disk space for models and dependencies

## Quick Start

```powershell
# 1. Copy environment file
cp .env.example .env

# 2. Start all services (first run downloads AI models ~2-3GB)
docker compose up -d

# 3. View logs
docker compose logs -f backend

# 4. Access the app
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
```

## Development

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
              │ PostgreSQL│ │  Redis   │ │  Worker  │
              │   :5432   │ │  :6379   │ │ (Celery) │
              └──────────┘ └──────────┘ └──────────┘
```

## GPU Processing Pipeline

1. **YOLOv8**: Detects players and ball in each frame
2. **ByteTrack**: Maintains consistent player IDs across frames
3. **PaddleOCR**: Reads jersey numbers from player crops
4. **Action Recognition**: Classifies basketball actions

## Supported Video Formats

- **File uploads**: MP4, MOV, AVI, MKV, WEBM, WMV
- **YouTube**: Paste any public/unlisted video URL
- **Resolution**: 480p to 4K (higher = better jersey detection)

## License

MIT
