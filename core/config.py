import os

from dotenv import read_dotenv
from pydantic import BaseSettings

read_dotenv(dotenv='../../env/agriculture-product-identities-app.env')


class CommonSettings(BaseSettings):
    DEBUG_MODE: bool = True


class ServerSettings(BaseSettings):
    HOST: str = 'localhost'
    PORT: int = 8080

class ConstValue(BaseSettings):
    MAX_ELEMENTS = 100000

class DatabaseSettings(BaseSettings):
    DB_URL: str = os.environ.get(
        'MONGODB_URL', 'mongodb+srv://admin:admin@training.lhxn7pm.mongodb.net/training?retryWrites=true&w=majority')
    DB_NAME: str = os.environ.get('MONGODB_NAME', 'training_agriculture')


class CloudinarySettings(BaseSettings):
    CLOUD_NAME: str = os.environ.get('CLOUD_NAME', 'devdaz')
    API_KEY: str = os.environ.get('API_KEY', '144841165449925')
    API_SECRET: str = os.environ.get(
        'API_SECRET', '4rblDOdRmkYaXmopzRlJ3dXJ0NQ')
    STORE: str = 'images'


class RedisSettings(BaseSettings):
    REDIS_URL: str = os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379')


class Settings(
    CommonSettings,
    ServerSettings,
    DatabaseSettings,
    CloudinarySettings,
    RedisSettings,
    ConstValue
):
    pass


settings = Settings()
