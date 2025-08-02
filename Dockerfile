FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /app

# Install additional system dependencies (PyTorch and Python already included)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install  -r requirements.txt

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