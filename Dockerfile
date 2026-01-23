FROM python:3.11-slim

# Install system dependencies for OpenCV
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && apt-get clean \
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
COPY alerts.py .
COPY audio_analyzer.py .
COPY migration_tracker.py .
COPY dashboard_server.py .
COPY object_detector.py .
COPY templates ./templates
COPY setup_model.py .
COPY data/models/bird_labels.txt ./data/models/

# Create data directories
RUN mkdir -p /app/data/logs /app/data/images /app/data/audio /app/data/models

# Generate the bird classifier model (if not mounted from host)
RUN python setup_model.py --output-dir /app/data/models --model-name bird_classifier.onnx || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Run the application
EXPOSE 5000
CMD ["python", "app.py"]
