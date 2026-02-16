# Backend Development Rules

## Project Structure

```
backend/
├── app/
│   ├── api/           # FastAPI route handlers
│   ├── core/          # Config, database, utilities
│   ├── models/        # SQLAlchemy + Pydantic models
│   ├── services/      # ML/CV processing services
│   └── workers/       # Celery background tasks
├── tests/             # Pytest tests
└── requirements.txt   # Python dependencies
```

## Adding New Services

### Service Template

```python
"""
Service Name - Brief description
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ResultClass:
    """Dataclass for results"""
    field: str
    confidence: float

class ServiceClass:
    """
    Main service class
    
    Follows pattern:
    - Lazy model loading with fallback
    - Configurable via settings
    - Proper logging
    """
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model with fallback handling"""
        try:
            # Primary model loading
            logger.info("Loading primary model...")
        except Exception as e:
            logger.warning(f"Primary model failed, using fallback: {e}")
            # Fallback model
    
    def process(self, input_data) -> List[ResultClass]:
        """Main processing method"""
        results = []
        # Processing logic
        return results
```

## Database Models

### SQLAlchemy Model Pattern

```python
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

class NewModel(Base):
    __tablename__ = "table_name"
    
    # Always include these
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign keys
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"))
    
    # Relationships
    video = relationship("Video", back_populates="new_models")
```

### Pydantic Schema Pattern

```python
from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class NewModelResponse(BaseModel):
    id: UUID
    field: str
    optional_field: Optional[str] = None
    
    class Config:
        from_attributes = True
```

## API Endpoints

### Route Handler Pattern

```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter()

@router.get("/{id}", response_model=ResponseSchema)
async def get_item(id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Endpoint description"""
    result = await db.execute(select(Model).where(Model.id == id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return item
```

## Configuration

Add new settings to `backend/app/core/config.py`:

```python
class Settings(BaseSettings):
    # New setting with default
    NEW_SETTING: str = "default_value"
    NEW_OPTIONAL: Optional[str] = None
```

## Pre-trained Models

Available models (no training required):
- **YOLOv8**: `from ultralytics import YOLO; model = YOLO("yolov8x.pt")`
- **MediaPipe**: `import mediapipe as mp; mp.solutions.pose`
- **EasyOCR**: `import easyocr; reader = easyocr.Reader(['en'])`
- **Roboflow**: Requires API key, use hosted inference or download weights

## Testing

```python
# tests/test_service.py
import pytest
from app.services.my_service import MyService

def test_service_initialization():
    service = MyService()
    assert service.model is not None

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```
