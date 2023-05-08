FROM python:3.9.13-slim-buster

WORKDIR /fast_api

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY /fast_api .

CMD [ "python", "main.py"]

EXPOSE 8080