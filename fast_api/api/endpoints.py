from fastapi import APIRouter
from .routes.agriculture_recognition import router as agriculture_recognition_router


router = APIRouter(prefix='/api/v1/agriculture-recognition',responses={'404':{'description':'Not found test'}})
router.include_router(agriculture_recognition_router)