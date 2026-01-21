# Bird Watching Application 🦅

A Python-based containerized application for monitoring birds at a bird feeder using computer vision. The application captures bird sightings, logs observations, and optionally monitors weather conditions.

## Features

- **Bird Detection**: Uses computer vision and motion detection to identify when birds visit the feeder
- **Image Capture**: Automatically captures and saves images of detected birds
- **Data Logging**: Records sightings with timestamps, species information, and environmental conditions
- **Weather Monitoring**: Optional integration with weather APIs to log conditions during bird visits
- **Containerized**: Runs in Docker for easy deployment on any device with a webcam
- **Species Identification**: Placeholder for future ML model integration for bird species recognition

## Architecture

The application consists of several modular components:

- `app.py` - Main application entry point and orchestration
- `bird_detector.py` - Computer vision module for detecting bird motion
- `species_identifier.py` - Species identification (placeholder for future ML model)
- `weather_monitor.py` - Weather data collection from external APIs
- `data_logger.py` - Logging and data persistence
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

## Future Enhancements

- **Machine Learning Model**: Integrate a trained neural network for accurate bird species identification
- **Web Dashboard**: Real-time monitoring dashboard with statistics and recent sightings
- **Alerts**: Notifications when rare species are detected
- **Audio Analysis**: Incorporate bird song recognition
- **Migration Tracking**: Analyze patterns and seasonal variations
- **Multi-Camera Support**: Monitor multiple feeding stations

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
