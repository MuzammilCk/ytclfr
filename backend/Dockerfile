FROM python:3.11-slim

# Install system dependencies (ffmpeg for whisper, tesseract for OCR, libgl1 for OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . /app/

# Expose FastAPI port
EXPOSE 8000

# Default command (overridden in compose)
CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
