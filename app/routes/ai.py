from PIL import Image
from fastapi import APIRouter, UploadFile, Request

from app.services.ai import AIService
from app.config import settings_factory
from app.limiter import limiter

settings = settings_factory()

ai_router = APIRouter()

@ai_router.post('/predict')
@limiter.limit(settings.API_RATE_LIMIT)
def predict_space(request: Request, file: UploadFile):
    image = Image.open(file.file).convert('RGB')
    return AIService.predict(image)
