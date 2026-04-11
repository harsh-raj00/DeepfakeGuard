# ====================================================================
# FaceAuth Guard - Dockerfile for Render Cloud
# ====================================================================

# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies required by OpenCV and dlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files first (to optimize Docker build caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the startup script is executable
RUN chmod +x start.sh

# Render binds to a dynamic PORT, default to 5000 if not set
ENV PORT=5000

# Execute the startup script
CMD ["./start.sh"]
