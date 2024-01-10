from fastapi import APIRouter, Depends, Query, UploadFile, BackgroundTasks, File, status
from fastapi.responses import JSONResponse
from typing import Union
from api.controllers.controller import agricultureRecognitionCtrl
from core.config import settings
from api.tasks.worker import task_train, add
from api.models.agriculture_recognition_info import AgricultureRecognitionInfo
from api.models.options import TrainOptions
from PIL import Image
from utils.pagination import pagination_info
import numpy as np
import io
import cv2
import base64
import time

router = APIRouter(
    prefix='/agriculture-recognition',
    tags=['Agriculture recognition'],
    responses={404: {'description': 'Not Found'}}
)

@router.post('/recognition')
async def recognition(file: bytes = File()):
    start = time.time()
    img = Image.open(io.BytesIO(file))
    img_resize = img.resize((180, 180))

    result = await agricultureRecognitionCtrl.recognition(img_resize)
    end = time.time()
    return {'result': result, 'time': float(end-start)}

@router.post('/multi-recognition')
async def multi_recognition(leaf: bytes = File(default=None), fruit: bytes = File(default=None), flower: bytes = File(default=None), bark: bytes = File(default=None)):
    start = time.time()
    images = []

    if(leaf):
        img = Image.open(io.BytesIO(leaf))
        img_resize = img.resize((180, 180))
        images.append({'file': img_resize, 'key': 'leaf'})
    if(fruit):
        img = Image.open(io.BytesIO(fruit))
        img_resize = img.resize((180, 180))
        images.append({'file': img_resize, 'key': 'fruit'})
    if(flower):
        img = Image.open(io.BytesIO(flower))
        img_resize = img.resize((180, 180))
        images.append({'file': img_resize, 'key': 'flower'})
    if(bark):
        img = Image.open(io.BytesIO(bark))
        img_resize = img.resize((180, 180))
        images.append({'file': img_resize, 'key': 'bark'})

    result = await agricultureRecognitionCtrl.multi_recognition(images)
    end = time.time()
    return {'results': result, 'time': float(end-start)}

