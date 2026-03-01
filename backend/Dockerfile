FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg tesseract-ocr libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
ENV PYTHONPATH=/app
