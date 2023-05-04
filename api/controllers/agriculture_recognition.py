from utils.singleton import SingletonMeta
from api.services.crud import InfoCrud
from api.ai_model.agriculture_recognition import AgricultureRecognition
from core.config import settings
from api.models.agriculture_recognition_info import AgricultureRecognitionInfo
from utils.string2digit import hash_string2digit
from utils.pyobjectid import PyObjectId

import cloudinary.uploader as cloud_uploader
import time

# from db.database import


class AgricultureRecognitionController(metaclass=SingletonMeta):
    def __init__(self):
        self.infoCrud = InfoCrud()
        self.agricultureRecognition = AgricultureRecognition()

    async def recognition(self, image):
        objs = self.agricultureRecognition.predict(image)
        return objs
