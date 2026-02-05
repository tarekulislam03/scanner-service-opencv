FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
# libgl1-mesa-glx is not available in Debian Bookworm (python:3.10-slim), and not needed for headless.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port 8000, can be overridden by env variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
