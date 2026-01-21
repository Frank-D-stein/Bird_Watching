FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY config.py .
COPY bird_detector.py .
COPY species_identifier.py .
COPY weather_monitor.py .
COPY data_logger.py .

# Create data directory
RUN mkdir -p /app/data/logs /app/data/images

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Run the application
CMD ["python", "app.py"]
