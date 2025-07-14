from PIL import Image
from fastapi import APIRouter, UploadFile

from app.services.ai import AIService

ai_router = APIRouter()

@ai_router.post('/predict')
def predict_space(file: UploadFile):
    image = Image.open(file.file).convert('RGB')
    return AIService.predict(image)
