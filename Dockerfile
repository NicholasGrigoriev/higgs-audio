FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask for API server
RUN pip install flask

# Copy HiggsAudio source code
COPY . .

# Install HiggsAudio package in development mode
RUN pip install -e .

# Create output directory
RUN mkdir -p /app/output

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run API server
CMD ["python3", "api_server.py"]