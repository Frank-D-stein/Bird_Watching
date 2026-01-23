# Bird Watching Application 🦅

A Python-based containerized application for monitoring birds at a bird feeder using computer vision. The application captures bird sightings, logs observations, and optionally monitors weather conditions.

## Features

- **Bird Detection**: Uses computer vision and motion detection to identify when birds visit the feeder
- **Image Capture**: Automatically captures and saves images of detected birds
- **Data Logging**: Records sightings with timestamps, species information, and environmental conditions
- **Weather Monitoring**: Optional integration with weather APIs to log conditions during bird visits
- **Containerized**: Runs in Docker for easy deployment on any device with a webcam
- **Species Identification (Neural Net)**: Optional ONNX model for accurate bird species identification
- **Object & Human Detection**: Detects and categorizes humans, animals, vehicles, and general objects with color-coded overlays
- **Web Dashboard**: Real-time monitoring dashboard with stats, recent sightings, and live detection counts
- **Alerts**: Notifications when rare species are detected
- **Audio Analysis**: Optional bird song recognition via audio model
- **Migration Tracking**: Seasonal and monthly trend analysis from logged sightings
- **Multi-Camera Support**: Monitor multiple feeding stations at once

## Architecture

The application consists of several modular components:

- `app.py` - Main application entry point and orchestration
- `bird_detector.py` - Computer vision module for detecting bird motion
- `species_identifier.py` - Species identification (placeholder for future ML model)
- `weather_monitor.py` - Weather data collection from external APIs
- `data_logger.py` - Logging and data persistence
- `alerts.py` - Rare species alerts
- `audio_analyzer.py` - Bird song recognition
- `migration_tracker.py` - Seasonal and monthly trend analysis
- `dashboard_server.py` - Web dashboard API
- `object_detector.py` - General object/human detection
- `config.py` - Configuration management

## Requirements

- Docker and Docker Compose
- Webcam connected to the device (e.g., `/dev/video0` on Linux)
- (Optional) OpenWeatherMap API key for weather monitoring

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Frank-D-stein/Bird_Watching.git
cd Bird_Watching
```

### 2. Configure Environment Variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```bash
# Camera Settings
CAMERA_INDEX=0              # Camera device index
CAPTURE_INTERVAL=60         # Seconds between captures when bird detected
IMAGE_WIDTH=1280            # Camera resolution width
IMAGE_HEIGHT=720            # Camera resolution height

# Detection Settings
MOTION_THRESHOLD=25         # Motion sensitivity (lower = more sensitive)
MIN_CONTOUR_AREA=500       # Minimum size to consider as a bird

# Weather API (Optional)
WEATHER_API_KEY=your_key_here
LOCATION_LAT=40.7128
LOCATION_LON=-74.0060
```

### 3. Build and Run with Docker

```bash
docker-compose up -d
```

This will:
- Build the Docker image
- Start the container in detached mode
- Mount the `./data` directory for persistent storage
- Access the camera device

### 4. View Logs

```bash
docker-compose logs -f bird-watcher
```

## Usage

### Running the Application

Once started, the application will:

1. Initialize the camera
2. Start monitoring for bird motion
3. When a bird is detected:
   - Capture an image
   - Identify the species (placeholder)
   - Fetch current weather conditions
   - Log all data to CSV and JSON files
   - Save the annotated image

### Accessing Data

All captured data is stored in the `./data` directory:

```
data/
├── logs/
│   ├── bird_sightings.csv          # CSV log of all sightings
│   ├── sighting_20260121_120000.json  # Individual sighting records
│   └── ...
├── images/
│   ├── bird_20260121_120000.jpg    # Captured bird images
│   └── ...
├── audio/
│   ├── song_20260121_120000.wav    # Captured bird audio
│   └── ...
└── app.log                          # Application logs
```

### Viewing Results

#### CSV Log Format

The `bird_sightings.csv` file contains:
- Timestamp
- Species name
- Confidence score
- Bird size estimate
- Temperature
- Weather conditions
- Image path
- Notes

#### JSON Records

Each sighting also has a detailed JSON record with complete detection metadata.

### Web Dashboard

The dashboard provides live stats, recent sightings, and migration insights:

```
http://localhost:5000
```

The API endpoints are:
- `/api/stats` for summary stats, recent sightings, and migration summaries
- `/api/sightings` for recent sighting records

## Configuration

### Camera Settings

- `CAMERA_INDEX`: Which camera to use (0 for default, 1 for secondary, etc.)
- `IMAGE_WIDTH`, `IMAGE_HEIGHT`: Resolution of captured images
- `CAPTURE_INTERVAL`: Cooldown period between captures (prevents duplicate logs)

### Detection Settings

- `MOTION_THRESHOLD`: Sensitivity for motion detection (0-255, lower = more sensitive)
- `MIN_CONTOUR_AREA`: Minimum pixel area to consider as a bird (filters noise)

### Weather API

To enable weather monitoring:

1. Sign up for a free API key at [OpenWeatherMap](https://openweathermap.org/api)
2. Add your API key, latitude, and longitude to `.env`

### ML Species Model (Southeastern US Birds)

The application includes a bird species classifier trained to identify 51 species commonly found in the southeastern United States, including:

- Northern Cardinal, Blue Jay, American Robin
- Carolina Chickadee, Carolina Wren, Tufted Titmouse
- Eastern Bluebird, Mourning Dove, Ruby-throated Hummingbird
- Woodpeckers (Red-bellied, Downy, Pileated, Red-headed)
- Warblers (Pine, Yellow-rumped, Prothonotary)
- And many more...

**Quick Setup:**

```bash
# Install dependencies and create the model
pip install onnx
python setup_model.py

# The model is automatically enabled in docker-compose.yml
docker-compose up -d --build
```

**Manual Setup:**

1. Run the model setup script:
   ```bash
   python setup_model.py --output-dir ./data/models
   ```

2. The script creates:
   - `data/models/bird_classifier.onnx` - The classification model
   - `data/models/bird_labels.txt` - Species labels (51 SE US birds)

3. These paths are pre-configured in `docker-compose.yml`:
   ```yaml
   - ML_MODEL_PATH=/app/data/models/bird_classifier.onnx
   - ML_LABELS_PATH=/app/data/models/bird_labels.txt
   ```

**Improving Classification Accuracy:**

The default model is a placeholder. To get better accuracy:

1. **Download a pre-trained model** (recommended):
   ```bash
   # Install PyTorch first
   pip install torch torchvision onnx
   
   # Download MobileNetV2 (fast, ~14MB)
   python download_model.py --model mobilenet
   
   # Or EfficientNet-B0 (more accurate, ~21MB)
   python download_model.py --model efficientnet
   ```

2. **Fine-tune on your own bird images** (best accuracy):
   ```bash
   # Create training script
   python download_model.py --model mobilenet --create-training-script
   
   # Organize your images:
   # data/training/Northern Cardinal/img1.jpg, img2.jpg, ...
   # data/training/Blue Jay/img1.jpg, img2.jpg, ...
   
   # Train the model
   python data/models/train_bird_model.py --data-dir ./data/training --epochs 10
   ```

3. **Use a pre-trained bird model** from Hugging Face or similar sources

**Using Your Own Model:**

To use a custom-trained model:

1. Export your model to ONNX format (input: NCHW float32, output: class logits)
2. Create a labels file with one species per line
3. Update the paths in `docker-compose.yml` or `.env`

```
ML_MODEL_PATH=/app/data/models/your_model.onnx
ML_LABELS_PATH=/app/data/models/your_labels.txt
ML_MIN_CONFIDENCE=0.3
```

### Audio Analysis

To enable bird song recognition:

```
AUDIO_ENABLED=1
AUDIO_MODEL_PATH=/app/data/models/audio_model.onnx
AUDIO_LABELS_PATH=/app/data/models/audio_labels.txt
# Optional for testing without live capture
AUDIO_SAMPLE_PATH=/app/data/audio/sample.wav
```

For live microphone capture, install `sounddevice` on the host or in the container.

### Object & Human Detection

The application can detect and categorize humans, animals, vehicles, and general objects in the camera feed. Detections are shown on the live video feed with color-coded bounding boxes:

| Category | Color | Examples |
|----------|-------|----------|
| Human | Red | Person |
| Bird | Green | Bird |
| Animal | Orange | Cat, Dog, Horse |
| Vehicle | Magenta | Car, Bicycle, Truck |
| Object | Cyan | Backpack, Chair, Bottle |

**Configuration:**

```
# Enable object detection (default: enabled)
OBJECT_DETECTION_ENABLED=1

# Optional: Use a YOLO ONNX model for better accuracy
OBJECT_MODEL_PATH=/app/data/models/yolov8n.onnx

# Detection thresholds
OBJECT_CONFIDENCE_THRESHOLD=0.4
OBJECT_NMS_THRESHOLD=0.5
```

Without an ONNX model, the system uses motion-based detection with size heuristics to categorize detections.

**Using a YOLO Model:**

1. Download a YOLOv8 ONNX model (e.g., `yolov8n.onnx`)
2. Place it in `data/models/`
3. Set `OBJECT_MODEL_PATH=/app/data/models/yolov8n.onnx`

The dashboard displays a "Live Detections" panel showing real-time counts for each category.

### Alerts

Configure a webhook (Slack/Discord/etc.) for rare species alerts:

```
RARE_SPECIES=Snowy Owl,Golden Eagle
ALERT_WEBHOOK_URL=https://example.com/webhook
ALERT_MIN_CONFIDENCE=0.7
ALERT_COOLDOWN_SECONDS=300
```

### Dashboard

```
DASHBOARD_ENABLED=1
DASHBOARD_PORT=5000
```

### Multi-Camera

To monitor multiple feeding stations:

```
CAMERA_INDEXES=0,1
```

Update `docker-compose.yml` to map each device (`/dev/video1`, etc.).

## Development

### Running Without Docker

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

### Project Structure

```
Bird_Watching/
├── app.py                      # Main application
├── bird_detector.py           # Bird detection module
├── species_identifier.py      # Species identification
├── weather_monitor.py         # Weather monitoring
├── data_logger.py            # Data logging
├── config.py                 # Configuration
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Additional Ideas

- Habitat correlation analysis (link sightings to vegetation and feeder types)
- Automated time-lapse summary exports

## Troubleshooting

### Camera Not Detected

If the camera isn't working:

1. Check camera permissions: `ls -l /dev/video0`
2. Verify camera index in `.env`
3. Try running with `--privileged` flag in docker-compose.yml
4. Test camera outside Docker: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### No Birds Detected

If birds aren't being detected:

1. Lower `MOTION_THRESHOLD` to increase sensitivity
2. Decrease `MIN_CONTOUR_AREA` to detect smaller birds
3. Check camera view - ensure it can see the feeder
4. Review `app.log` for debugging information

### Weather Data Not Working

1. Verify API key is correct
2. Check latitude/longitude coordinates
3. Ensure container has internet access
4. Weather monitoring is optional - app works without it

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- OpenCV for computer vision capabilities
- OpenWeatherMap for weather data API
- All bird enthusiasts and contributors
