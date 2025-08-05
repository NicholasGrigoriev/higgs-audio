# Google Drive Integration for HiggsAudio

This document describes the Google Drive upload integration for HiggsAudio generated files.

## Overview

The integration adds automatic Google Drive upload capability to the HiggsAudio API server. When audio generation completes, the system:

1. Saves the audio file to the shared volume
2. Sends a message to the generation complete queue (optional)
3. Sends a message to the Google Drive upload queue
4. QArchistrator processes the upload request and uploads to Google Drive
5. Sends a completion message with Google Drive links

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────┐
│  HiggsAudio API │ ──SQS──▶│  QArchistrator   │ ──API─▶│ Google Drive │
│    Container    │         │    Container     │         │              │
└────────┬────────┘         └────────┬─────────┘         └──────────────┘
         │                           │
         └──── Shared Volume ────────┘
              (/app/output)
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# SQS Queue URLs
GENERATION_COMPLETE_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/generation-complete
GOOGLE_DRIVE_UPLOAD_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/google-drive-upload
UPLOAD_COMPLETE_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/upload-complete

# Google Drive Configuration
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id
```

### Google Service Account

1. Create a Google Cloud Project
2. Enable the Google Drive API
3. Create a service account with Drive API access
4. Download the service account JSON key
5. Save it as `queeArchistrator/data/google_service_account.json`
6. Share your Google Drive folder with the service account email

### AWS SQS Queues

Create three SQS queues:

1. **generation-complete**: Notifies when audio generation is done
2. **google-drive-upload**: Triggers Google Drive upload
3. **upload-complete**: Notifies when upload is complete

## Setup Instructions

### 1. Update Docker Compose

Both docker-compose files have been updated with the necessary environment variables and shared volume configuration.

### 2. Configure QArchistrator

In the QArchistrator web UI:

1. Add the Google Drive upload queue:
   - Name: `Google Drive Upload`
   - Queue URL: Your `GOOGLE_DRIVE_UPLOAD_QUEUE_URL`
   - Region: Your AWS region
   - Status: Active

2. Create an action for the queue:
   - Name: `Upload to Google Drive`
   - Script Type: `Script File`
   - Script Content: `google_drive_upload.py`
   - Status: Active

### 3. Start Services

```bash
# From queeArchistrator directory
docker-compose up -d

# Or from higgs-audio directory (standalone)
docker-compose up -d
```

## Message Format

### Google Drive Upload Queue Message

```json
{
    "job_id": "uuid",
    "request_id": "user_request_id",
    "file_path": "/app/output/filename.wav",
    "file_name": "filename.wav",
    "file_size_bytes": 12345,
    "text": "Original text",
    "ref_audio": "voice_reference",
    "created_at": "2024-01-20T10:00:00Z",
    "completed_at": "2024-01-20T10:01:00Z",
    "source_system": "higgs-audio"
}
```

### Upload Complete Queue Message

```json
{
    "job_id": "uuid",
    "request_id": "user_request_id",
    "status": "uploaded",
    "google_drive_id": "drive_file_id",
    "google_drive_link": "https://drive.google.com/file/...",
    "google_drive_download_link": "https://drive.google.com/uc?id=...",
    "file_name": "filename.wav",
    "file_size": "12345",
    "uploaded_at": "2024-01-20T10:02:00Z"
}
```

## Testing

1. Send a TTS generation request:
```bash
curl -X POST http://localhost:8000/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test",
    "ref_audio": "belinda",
    "request_id": "test-123"
  }'
```

2. Check job status:
```bash
curl http://localhost:8000/tts/status/{job_id}
```

3. Monitor QArchistrator executions in the web UI at http://localhost:8888

4. Verify file appears in Google Drive folder

## Troubleshooting

### Common Issues

1. **SQS messages not being sent**
   - Check AWS credentials are set correctly
   - Verify queue URLs are valid
   - Check HiggsAudio logs: `docker logs higgs-audio-api`

2. **Google Drive upload fails**
   - Verify service account JSON exists in `data/google_service_account.json`
   - Check service account has access to the Drive folder
   - Verify Google Drive API is enabled in your project

3. **File not found errors**
   - Ensure the shared volume is mounted correctly
   - Check file paths are being translated correctly
   - Verify files exist in the output directory

### Debugging

View logs:
```bash
# HiggsAudio API logs
docker logs higgs-audio-api

# QArchistrator logs
docker logs queuearchitector-queuearchitector-1

# Check shared volume
docker exec higgs-audio-api ls -la /app/output
```

## Security Notes

- Store AWS credentials securely
- Keep Google service account key private
- Use IAM roles with minimal permissions for SQS
- Regularly rotate credentials
- Monitor queue activity for anomalies