# SirenLedger 2.0 – Emergency Vehicle Siren Detection System

A comprehensive Python-based system for detecting, logging, and analyzing emergency vehicle sirens using machine learning. SirenLedger 2.0 features a complete rewrite with modern architecture, web dashboard, REST API, and containerized deployment.

## 🚨 Features

### Core Detection
- **Real-time Audio Processing**: Continuous monitoring using Google's YAMNet model
- **Multiple Siren Types**: Detects police cars, ambulances, fire trucks, and civil defense sirens
- **High Accuracy**: Configurable confidence thresholds with robust detection algorithms
- **Event Aggregation**: Groups related detections into coherent siren events

### Data Storage & Analytics
- **SQLite Database**: Structured storage for events, detections, and daily reports
- **Detailed Metrics**: Duration tracking, confidence scores, and hourly distributions
- **Data Migration**: Import existing CSV data from SirenLedger 1.0
- **Automatic Reports**: Daily, weekly, and monthly activity summaries

### Web Interface
- **Interactive Dashboard**: Real-time visualizations with Plotly
- **REST API**: Complete API for data access and integration
- **WebSocket Support**: Live updates for real-time detection alerts
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices
- **Real-time Updates**: Live charts and statistics with WebSocket push notifications

### Deployment Options
- **Raspberry Pi Native**: Direct installation for Pi OS
- **Docker Containers**: Containerized deployment with docker-compose
- **Production Ready**: Nginx reverse proxy, health checks, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │    │   YAMNet Model   │    │    Database     │
│  (Microphone)   │───▶│  (Classification)│───▶│   (SQLite)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Audio Processor │    │ Siren Detector   │    │ Event Aggregator│
│                 │    │    Service       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │   Web Dashboard  │
                       │   & REST API     │
                       └──────────────────┘
```

### Components

- **Audio Processing**: Real-time capture and windowing for ML inference
- **YAMNet Classifier**: TensorFlow Lite model for audio event classification  
- **Event Aggregation**: Groups nearby detections into meaningful events
- **Database Layer**: SQLAlchemy ORM with migration support
- **Web Interface**: Flask application with Plotly visualizations
- **Configuration**: Pydantic models with file/environment variable support

## 🚀 Quick Start

### Prerequisites

- **Raspberry Pi 3B+ or 4** (1GB+ RAM)
- **Microphone** (USB, HAT, or I2S)
- **Raspberry Pi OS** (Bookworm recommended)
- **Python 3.11+**

### Installation

1. **Install system dependencies**:
```bash
sudo apt update
sudo apt install -y python3-venv libportaudio2 portaudio19-dev wget git curl
```

2. **Install uv (modern Python package manager)**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # Add uv to PATH
```

3. **Clone and setup**:
```bash
git clone <repository-url>
cd SirenLedger
uv sync  # Creates virtual environment and installs dependencies
source .venv/bin/activate  # Activate the environment created by uv
```

4. **Download YAMNet model**:
```bash
wget -O yamnet.tflite "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
wget -O yamnet_label_list.txt "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt"
```

5. **Test audio setup**:
```bash
siren-counter test-audio
```

6. **Run detection service**:
```bash
siren-counter run
```

7. **Start web dashboard** (in another terminal):
```bash
siren-web
```

Visit http://localhost:5555 for the dashboard.

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Create data directory
mkdir -p data

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f siren-detector

# Access dashboard
open http://localhost:5555
```

### Production Deployment

```bash
# Production stack with Nginx
docker-compose --profile production up -d

# Access via Nginx proxy
open http://localhost
```

## ⚙️ Configuration

### Configuration File

Create `siren_config.yaml`:

```yaml
# Audio settings
audio:
  sample_rate: 48000  # Will be resampled to 16kHz for YAMNet
  device_id: null    # null for default device
  channels: 2        # Stereo input, converted to mono

# Detection settings
confidence_threshold: 0.20
siren_keywords:
  - "siren"
  - "police car"
  - "ambulance"
  - "fire engine"
  - "fire truck"

# Database
database:
  url: "sqlite:///siren_ledger.db"
  echo: false

# Paths
model_path: "yamnet.tflite"
labels_path: "yamnet_label_list.txt"

# Logging
log_level: "INFO"
```

### Environment Variables

All settings can be overridden with environment variables using the `SIREN_` prefix:

```bash
export SIREN_CONFIDENCE_THRESHOLD=0.25
export SIREN_AUDIO__DEVICE_ID=1
export SIREN_DATABASE__URL="sqlite:///data/siren.db"
```

## 📊 API Usage

### REST API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/reports/daily` - Daily activity reports
- `GET /api/v1/reports/daily/{date}` - Report for specific date
- `POST /api/v1/reports/daily/{date}/generate` - Manually generate report
- `GET /api/v1/events` - Siren events with filtering
- `GET /api/v1/events/{id}/detections` - Individual detections for an event
- `GET /api/v1/stats/summary` - Summary statistics
- `GET /api/v1/stats/hourly` - Hourly distribution
- `GET /api/v1/config` - Current configuration (sanitized)

### Example API Usage

```bash
# Get last 7 days of reports
curl http://localhost:5555/api/v1/reports/daily?limit=7

# Get events from specific date range
curl "http://localhost:5555/api/v1/events?start_date=2024-01-01&end_date=2024-01-07"

# Get summary statistics
curl http://localhost:5555/api/v1/stats/summary
```

## 📈 Dashboard Features

### Visualizations

1. **Daily Counts Chart**: Bar chart showing siren events per day
2. **Duration Analysis**: Line chart showing total daily siren minutes
3. **Hourly Distribution**: Heatmap of siren activity by hour of day
4. **Confidence Timeline**: Scatter plot showing detection confidence over time
5. **Summary Statistics**: Key metrics and totals

### Interactive Features

- **Date Range Selection**: Filter data by custom date ranges
- **Real-time Updates**: WebSocket-powered live updates for instant detection alerts
- **Connection Status**: Visual indicator for WebSocket connection health
- **Responsive Design**: Works on all device sizes
- **Export Options**: Download charts and data in various formats

## 🔧 CLI Commands

### Basic Commands

```bash
# Run detection service
siren-counter run

# Show system status
siren-counter status

# Test audio device
siren-counter test-audio --device-id 1

# Generate configuration file
siren-counter init-config

# Migrate legacy CSV data
siren-counter migrate-csv --csv-file siren_daily_counts.csv

# Generate reports
siren-counter report --start-date 2024-01-01 --format json

# Cleanup old data
siren-counter cleanup --days 365
```

### Configuration Options

```bash
# Use custom config file
siren-counter -c my_config.yaml run

# Set log level
siren-counter --log-level DEBUG run
```

## 🔄 Data Migration

### From SirenLedger 1.0

Migrate existing CSV data:

```bash
siren-counter migrate-csv --csv-file siren_daily_counts.csv
```

This imports daily counts while preserving historical data. Note that detailed detection data is not available from the CSV format.

## 📝 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/estcarisimo/SirenLedger
cd SirenLedger

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with development dependencies
uv sync --extra dev

# Activate environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy siren_ledger/

# Format code
black siren_ledger/
```

### Project Structure

```
siren_ledger/
├── audio/              # Audio processing and YAMNet classifier
├── cli/                # Command-line interface
├── config/             # Configuration management
├── models/             # Pydantic data models
├── service/            # Detection service and event aggregation
├── storage/            # Database models and operations
└── web/                # Flask web app and REST API
```

## 🐛 Troubleshooting

### Common Issues

**Audio device not found**:
```bash
# List available devices
siren-counter test-audio

# Test specific device
siren-counter test-audio --device-id 1
```

**Permission errors**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
```

**Model files missing**:
```bash
# Re-download models
wget -O yamnet.tflite "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
wget -O yamnet_label_list.txt "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt"
```

**Docker audio issues**:
```bash
# Ensure audio devices are accessible
ls -la /dev/snd/

# Check container audio access
docker exec -it siren-detector ls -la /dev/snd/
```

### Performance Tuning

**Raspberry Pi Optimization**:
- Use Class 10 SD card or SSD
- Ensure adequate power supply (3A+)
- Consider overclocking for Pi 3B+
- Use lite OS image without desktop

**Detection Tuning**:
- Adjust `confidence_threshold` (0.15-0.30)
- Modify `siren_keywords` for specific needs
- Check microphone positioning and quality

## 📋 System Requirements

### Minimum Requirements
- Raspberry Pi 3B+ (1GB RAM)
- 8GB SD card (Class 10)
- USB microphone
- Internet connection (for model download)

### Recommended Requirements  
- Raspberry Pi 4 (2GB+ RAM)
- 32GB SD card or SSD
- Quality USB microphone or HAT
- Ethernet connection
- Case with cooling

## 🔒 Security Considerations

- **Network Access**: API endpoints are rate-limited
- **Container Security**: Non-root user, minimal attack surface
- **Data Privacy**: All processing is local, no external data transmission
- **Authentication**: Consider adding authentication for production deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings in NumPy style
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google YAMNet**: Audio event classification model
- **TensorFlow**: Machine learning framework
- **Flask & Plotly**: Web interface and visualizations
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation and serialization

---

**Happy monitoring! May your Pi detect sirens accurately and efficiently.** 🚨📊