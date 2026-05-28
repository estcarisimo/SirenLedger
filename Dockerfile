# Multi-stage Dockerfile for SirenLedger
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Audio libraries for sounddevice
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    # System utilities
    wget \
    curl \
    # Build tools (needed for some Python packages)
    gcc \
    g++ \
    # Python development headers
    python3-dev \
    # Additional build dependencies
    pkg-config \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 siren && \
    useradd --uid 1000 --gid siren --shell /bin/bash --create-home siren

# Set working directory
WORKDIR /app

# Use pip for Docker (more reliable than uv in containers)
# Install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Production stage
FROM base as production

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY siren_ledger/ ./siren_ledger/

# Install dependencies directly (not editable for production)
RUN pip install --no-cache-dir .

# Create directories for data and models with proper permissions
RUN mkdir -p /app/data /app/models && \
    chmod -R 777 /app/data && \
    chown -R siren:siren /app

# Download YAMNet model and labels (if not provided via volume)
USER siren
RUN cd /app/models && \
    wget -O yamnet.tflite \
        "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" && \
    wget -O yamnet_label_list.txt \
        "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt"

# Set environment variables
ENV PYTHONPATH=/app
ENV SIREN_DATABASE__URL=sqlite:///data/siren_ledger.db
ENV SIREN_MODEL_PATH=/app/models/yamnet.tflite
ENV SIREN_LABELS_PATH=/app/models/yamnet_label_list.txt

# Expose ports
EXPOSE 5555

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5555/api/v1/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "siren_ledger.cli", "run"]

# Development stage
FROM base as development

# Copy essential files first
COPY pyproject.toml README.md LICENSE ./
COPY siren_ledger/ ./siren_ledger/

# Install package with development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the files for development
COPY . .

# Create directories for data and models with proper permissions (same as production)
RUN mkdir -p /app/data /app/models && \
    chmod -R 777 /app/data && \
    chown -R siren:siren /app

USER siren

# Expose ports for development (Flask debug + any additional services)
EXPOSE 5555 8000

# Development command
CMD ["python", "-m", "siren_ledger.web.app"]