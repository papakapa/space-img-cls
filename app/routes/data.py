from fastapi import APIRouter

from models.predict import SPACE_CLASSES

data_router = APIRouter()

@data_router.get('/classes')
def get_classes():
    return { 'classes': SPACE_CLASSES}