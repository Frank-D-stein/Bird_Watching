# Quick Start Guide 🚀

Get your bird watching system up and running in minutes!

## Prerequisites

- Device with Docker installed (Raspberry Pi, Linux PC, etc.)
- Webcam connected to the device
- (Optional) OpenWeatherMap API key

## 5-Minute Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Frank-D-stein/Bird_Watching.git
cd Bird_Watching
```

### 2. Configure (Optional)

For basic setup, the defaults work fine. For custom configuration:

```bash
cp .env.example .env
nano .env  # Edit settings
```

### 3. Start the Application

```bash
docker-compose up -d
```

That's it! The application is now running and monitoring your bird feeder.

## Viewing Results

### Check Logs

```bash
# Live logs
docker-compose logs -f

# Recent logs
docker-compose logs --tail=50
```

### Access Data

All captured data is in the `./data` directory:

```bash
# View all sightings
cat data/logs/bird_sightings.csv

# View captured images
ls -lh data/images/
```

### See Statistics

```bash
# View application logs for session statistics
docker-compose logs | grep "STATISTICS" -A 10
```

## Common Commands

```bash
# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# View resource usage
docker stats bird-watching-app

# Update the application
git pull
docker-compose up -d --build
```

## Troubleshooting

### Camera Not Found?

1. Check camera device:
   ```bash
   ls -l /dev/video*
   ```

2. Update device in `docker-compose.yml` if needed:
   ```yaml
   devices:
     - /dev/video1:/dev/video0  # If your camera is video1
   ```

### No Birds Detected?

1. Lower the detection threshold in `.env`:
   ```
   MOTION_THRESHOLD=15
   MIN_CONTOUR_AREA=300
   ```

2. Ensure the camera has a clear view of the feeder

### View More Logs?

```bash
docker-compose logs -f bird-watcher
```

## Next Steps

- **Add Weather Data**: Get an API key from [OpenWeatherMap](https://openweathermap.org/api) and add to `.env`
- **Adjust Detection**: Tune `MOTION_THRESHOLD` and `MIN_CONTOUR_AREA` for your environment
- **Analyze Data**: Use the CSV logs in `data/logs/bird_sightings.csv` for analysis
- **Future Enhancement**: Train/integrate an ML model for species identification

## Support

For issues or questions:
- Check the main [README.md](README.md) for detailed documentation
- Review logs: `docker-compose logs`
- Open an issue on GitHub

Happy Bird Watching! 🐦
