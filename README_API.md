# HiggsAudio API Server

Docker-based API service for HiggsAudio voice cloning with REST endpoints.

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Set output directory (optional, defaults to ./output)
export TTS_OUTPUT_DIR=/path/to/your/audio/output

# Build and start the service
docker-compose up --build -d

# Check service health
curl http://localhost:8000/health
```

### 2. Alternative: Run with Docker

```bash
# Build the image
docker build -t higgs-audio-api .

# Run the container
docker run -d \
  --name higgs-audio-api \
  -p 8000:8000 \
  -v /path/to/output:/app/output \
  higgs-audio-api
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Generate TTS (Async)
```bash
POST /tts/generate
Content-Type: application/json

{
  "text": "Hello, this is a test message for voice cloning.",
  "ref_audio": "broom_salesman",
  "seed": 12345,
  "request_id": "test_001"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "request_id": "test_001",
  "estimated_time_minutes": 2,
  "message": "TTS generation started"
}
```

### Check Job Status
```bash
GET /tts/status/{job_id}
```

**Response (Processing):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "started_at": "2024-07-31T10:30:00Z",
  "text": "Hello, this is a test...",
  "ref_audio": "broom_salesman"
}
```

**Response (Completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "completed_at": "2024-07-31T10:32:30Z",
  "processing_time_seconds": 150.5,
  "output_filename": "20240731_103000_test_001_550e8400.wav",
  "file_size_bytes": 1024000,
  "audio_url": "/tts/download/550e8400-e29b-41d4-a716-446655440000"
}
```

### Download Audio File
```bash
GET /tts/download/{job_id}
```

### List All Jobs (Debug)
```bash
GET /tts/jobs
```

### Service Information
```bash
GET /info
```

## Job Status States

- **pending**: Job created, waiting to start
- **processing**: HiggsAudio generation in progress
- **completed**: Generation successful, audio file ready
- **failed**: Generation failed (check error field)
- **timeout**: Processing took longer than 10 minutes

## Configuration

### Environment Variables

- `OUTPUT_DIR`: Directory for generated audio files (default: `/app/output`)
- `MAX_PROCESSING_TIME`: Timeout in seconds (default: 600)

### Volume Mounts

- `/app/output`: Audio output directory (mount to host for persistence)
- `/app/models`: Model/reference audio directory (optional, read-only)

## Usage Examples

### Python Client Example

```python
import requests
import time
import json

# Start TTS generation
response = requests.post('http://localhost:8000/tts/generate', json={
    "text": "Welcome to HiggsAudio voice cloning!",
    "ref_audio": "broom_salesman",
    "request_id": "welcome_001"
})

job_data = response.json()
job_id = job_data['job_id']
print(f"Started job: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f'http://localhost:8000/tts/status/{job_id}')
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']}")
    
    if status_data['status'] == 'completed':
        # Download the audio file
        audio_response = requests.get(f'http://localhost:8000/tts/download/{job_id}')
        
        with open(f"{job_data['request_id']}.wav", 'wb') as f:
            f.write(audio_response.content)
        
        print(f"Audio saved! Processing time: {status_data['processing_time_seconds']:.1f}s")
        break
    
    elif status_data['status'] in ['failed', 'timeout']:
        print(f"Job failed: {status_data.get('error', 'Unknown error')}")
        break
    
    time.sleep(5)  # Check every 5 seconds
```

### cURL Example

```bash
# Generate TTS
JOB_ID=$(curl -s -X POST http://localhost:8000/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from cURL!","ref_audio":"broom_salesman","request_id":"curl_test"}' \
  | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Wait and check status
while true; do
  STATUS=$(curl -s http://localhost:8000/tts/status/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "completed" ]; then
    # Download audio
    curl -o "output.wav" http://localhost:8000/tts/download/$JOB_ID
    echo "Audio downloaded!"
    break
  elif [ "$STATUS" = "failed" ] || [ "$STATUS" = "timeout" ]; then
    echo "Job failed!"
    break
  fi
  
  sleep 5
done
```

## Docker Logs

```bash
# View logs
docker-compose logs -f higgs-audio-api

# Or with Docker
docker logs -f higgs-audio-api
```

## Troubleshooting

### Check Service Health
```bash
curl http://localhost:8000/health
```

### Verify HiggsAudio Installation
```bash
docker exec higgs-audio-api python3 -c "import boson_multimodal; print('HiggsAudio OK')"
```

### Check Output Directory
```bash
docker exec higgs-audio-api ls -la /app/output/
```

### Monitor Processing
```bash
docker exec higgs-audio-api ps aux | grep python
```