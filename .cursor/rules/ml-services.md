# ML/CV Services Rules

## Available Pre-trained Models

No training required - use these directly:

| Model | Package | Load Command |
|-------|---------|--------------|
| YOLOv8 | ultralytics | `YOLO("yolov8x.pt")` |
| ByteTrack | supervision | `sv.ByteTrack()` |
| EasyOCR | easyocr | `easyocr.Reader(['en'])` |
| MediaPipe Pose | mediapipe | `mp.solutions.pose.Pose()` |
| Roboflow Models | roboflow | Via API or download weights |

## Adding New ML Services

### 1. Check for Pre-trained Models First

Before implementing custom solutions:
- Search Roboflow Universe for pre-trained models
- Check HuggingFace for relevant models
- Look for PyTorch Hub models

### 2. Service Structure

```python
class MLService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load with fallback"""
        try:
            # Try primary model (e.g., Roboflow)
            if settings.ROBOFLOW_API_KEY:
                self._load_roboflow_model()
            else:
                self._load_fallback_model()
        except Exception as e:
            logger.warning(f"Using fallback: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Always have a fallback that works without API keys"""
        self.model = YOLO("yolov8x.pt")  # COCO pretrained
```

### 3. GPU Memory Management

```python
# Check available memory before loading large models
if torch.cuda.is_available():
    free_memory = torch.cuda.get_device_properties(0).total_memory
    if free_memory < 8 * 1024**3:  # Less than 8GB
        logger.warning("Low VRAM, using smaller model")
        model_name = "yolov8m.pt"  # Medium instead of extra-large
```

## Roboflow Integration

### Using Roboflow API

```python
from roboflow import Roboflow

# Check for API key
if settings.ROBOFLOW_API_KEY:
    rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
    project = rf.workspace("workspace").project("project")
    model = project.version(1).model
    
    # Inference
    result = model.predict("image.jpg", confidence=40).json()
```

### Downloading Weights for Local Inference

```python
# Download YOLOv8 format weights
version = project.version(1)
dataset = version.download("yolov8", location="./models/")

# Use locally
model = YOLO("./models/weights/best.pt")
```

## MediaPipe Integration

```python
import mediapipe as mp

# Lazy initialization
mp_pose = None

def init_mediapipe():
    global mp_pose
    if mp_pose is None:
        mp_pose = mp.solutions.pose
    return mp_pose

# Usage
pose = init_mediapipe().Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=lite, 1=full, 2=heavy
    min_detection_confidence=0.5
)
```

## Processing Pipeline Integration

To add a new service to the video processing pipeline:

1. Create service in `backend/app/services/`
2. Import in `backend/app/workers/tasks.py`
3. Add processing step in `process_video_task()`
4. Update progress publisher with new stage
5. Save results to database

```python
# In tasks.py
from app.services.new_service import NewService

# In process_video_task()
progress_pub.stage_change("new_stage", "Running new analysis...")
new_service = NewService()
results = new_service.process(video_path)

# Save to database
for result in results:
    record = NewModel(video_id=video.id, **result)
    session.add(record)
```

## Performance Considerations

- Process frames in batches when possible
- Use GPU for inference (`model.to("cuda")`)
- Sample frames for expensive operations (e.g., OCR every 10th frame)
- Cache model instances (don't reload per video)
- Use appropriate model size for available VRAM
