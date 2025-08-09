#!/usr/bin/env python3
"""
HiggsAudio API Server
Provides REST API interface for HiggsAudio voice cloning
"""

import os
import json
import uuid
import tempfile
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom logging filter to reduce health check noise  
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        # Filter out health check requests from werkzeug logs
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            if 'GET /health' in message and record.name == 'werkzeug':
                return False
        return True

# Apply filter to werkzeug logger (Flask's request logger)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(HealthCheckFilter())

app = Flask(__name__)

# Configuration
OUTPUT_DIR = "/app/output"  # Docker volume mount point
GENERATION_SCRIPT = "/app/examples/generation.py"
MAX_PROCESSING_TIME = 600  # 10 minutes timeout


# In-memory job tracking
jobs = {}
job_lock = threading.Lock()

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

def create_job_id():
    """Generate unique job ID"""
    return str(uuid.uuid4())

def save_job(job_id, status, **kwargs):
    """Thread-safe job saving"""
    with job_lock:
        if job_id not in jobs:
            jobs[job_id] = {}
        jobs[job_id].update({
            'status': status,
            'updated_at': datetime.now().isoformat(),
            **kwargs
        })


def get_job(job_id):
    """Thread-safe job retrieval"""
    with job_lock:
        return jobs.get(job_id, {})

def process_tts_job(job_id, text, ref_audio, seed, temperature, output_filename, tg_chat_id=None, use_system_message=False, voice_description=None):
    """Process TTS job in background thread"""
    try:
        logger.info(f"Starting TTS processing for job {job_id}")
        logger.info(f"Job details - Text length: {len(text)} chars, Ref audio: {ref_audio}, Seed: {seed}")
        save_job(job_id, JobStatus.PROCESSING, 
                 started_at=datetime.now().isoformat())
        
        # Create temporary transcript file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as transcript_file:
            transcript_file.write(text)
            transcript_path = transcript_file.name
        
        logger.info(f"Created transcript file: {transcript_path}")
        
        # Handle custom voice descriptions by temporarily updating the profile.yaml
        original_profile_content = None
        profile_path = "/app/examples/voice_prompts/profile.yaml"
        
        if voice_description and ref_audio.startswith('profile:'):
            # Extract the profile name (everything after "profile:")
            profile_name = ref_audio[len('profile:'):]
            
            # Back up and update the existing profile.yaml
            import yaml
            try:
                # Read the original profile
                with open(profile_path, 'r', encoding='utf-8') as f:
                    original_profile_content = f.read()
                    profile_data = yaml.safe_load(original_profile_content)
                
                # Add/update the custom voice description
                profile_data['profiles'][profile_name] = voice_description
                
                # Write the updated profile
                with open(profile_path, 'w', encoding='utf-8') as f:
                    yaml.dump(profile_data, f)
                
                logger.info(f"Updated profile.yaml with custom voice: {profile_name}")
                logger.info(f"Custom voice description: {voice_description[:100]}...")
            except Exception as e:
                logger.error(f"Failed to update profile.yaml: {e}")
                original_profile_content = None
        
        # Full output path
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Output will be saved to: {output_path}")
        
        # Build HiggsAudio CLI command
        cmd = [
            'python3', GENERATION_SCRIPT,
            '--transcript', transcript_path,
            '--ref_audio', ref_audio,
            '--seed', str(seed),
            '--out_path', output_path
        ]
        
        # Add flag for voice descriptions (profile: refs)
        if ref_audio.startswith('profile:') and (use_system_message or voice_description):
            cmd.append('--ref_audio_in_system_message')
        
        logger.info(f"Executing HiggsAudio command: {' '.join(cmd)}")
        logger.info(f"Working directory: /app")
        
        # Check if generation script exists
        if not os.path.exists(GENERATION_SCRIPT):
            raise Exception(f"Generation script not found: {GENERATION_SCRIPT}")
        
        # Run HiggsAudio generation with timeout
        start_time = time.time()
        logger.info(f"Starting subprocess at {datetime.now().isoformat()}")
        
        # Use Popen for real-time output monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/app',
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Thread-safe lists to store output
        stdout_lines = []
        stderr_lines = []
        
        def read_stdout():
            """Read stdout and log each line"""
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.rstrip('\n\r')
                    logger.info(f"[Job {job_id}] STDOUT: {line}")
                    stdout_lines.append(line)
            process.stdout.close()
        
        def read_stderr():
            """Read stderr and log each line"""
            for line in iter(process.stderr.readline, ''):
                if line:
                    line = line.rstrip('\n\r')
                    logger.warning(f"[Job {job_id}] STDERR: {line}")
                    stderr_lines.append(line)
            process.stderr.close()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # Monitor process with periodic logging
        last_log_time = time.time()
        while process.poll() is None:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Log progress every 30 seconds
            if current_time - last_log_time >= 30:
                logger.info(f"Job {job_id} still processing... Elapsed: {elapsed:.1f}s")
                last_log_time = current_time
                
                # Update job with current progress
                save_job(job_id, JobStatus.PROCESSING,
                        started_at=datetime.now().isoformat(),
                        elapsed_seconds=elapsed)
            
            # Check for timeout
            if elapsed > MAX_PROCESSING_TIME:
                logger.warning(f"Job {job_id} timeout reached, terminating process")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                raise subprocess.TimeoutExpired(cmd, MAX_PROCESSING_TIME)
                
            time.sleep(1)  # Check every second
        
        # Wait for threads to finish reading any remaining output
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        # Get return code
        return_code = process.returncode
        
        # Join output lines for final logging
        stdout = '\n'.join(stdout_lines)
        stderr = '\n'.join(stderr_lines)
        
        processing_time = time.time() - start_time
        logger.info(f"HiggsAudio process completed with return code: {return_code}")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        
        if stdout:
            logger.info(f"HiggsAudio stdout: {stdout[:500]}...")  # Truncate long output
        if stderr:
            logger.info(f"HiggsAudio stderr: {stderr[:500]}...")  # Truncate long output
        
        # Clean up temporary files
        try:
            os.unlink(transcript_path)
            logger.info(f"Cleaned up transcript file: {transcript_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up transcript file: {e}")
        
        # Restore original profile.yaml if it was modified
        if original_profile_content is not None:
            try:
                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(original_profile_content)
                logger.info(f"Restored original profile.yaml")
            except Exception as e:
                logger.warning(f"Failed to restore profile.yaml: {e}")
        
        if return_code == 0:
            # Get audio file size and check if file exists
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                
                save_job(job_id, JobStatus.COMPLETED,
                        completed_at=datetime.now().isoformat(),
                        processing_time_seconds=processing_time,
                        output_file=output_filename,
                        file_size_bytes=file_size,
                        stdout=stdout,
                        stderr=stderr)
                
                logger.info(f"Job {job_id} completed successfully in {processing_time:.1f}s")
            else:
                raise Exception("Output file was not created")
        else:
            raise Exception(f"HiggsAudio failed with return code {return_code}: {stderr}")
    
    except subprocess.TimeoutExpired:
        logger.error(f"Job {job_id} timed out after {MAX_PROCESSING_TIME}s")
        
        # Restore original profile.yaml if it was modified
        if 'original_profile_content' in locals() and original_profile_content is not None:
            try:
                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(original_profile_content)
                logger.info(f"Restored original profile.yaml after timeout")
            except Exception as restore_error:
                logger.warning(f"Failed to restore profile.yaml after timeout: {restore_error}")
        
        save_job(job_id, JobStatus.TIMEOUT,
                failed_at=datetime.now().isoformat(),
                error="Processing timed out after 10 minutes")
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        
        # Restore original profile.yaml if it was modified
        if 'original_profile_content' in locals() and original_profile_content is not None:
            try:
                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(original_profile_content)
                logger.info(f"Restored original profile.yaml after error")
            except Exception as restore_error:
                logger.warning(f"Failed to restore profile.yaml after error: {restore_error}")
        
        save_job(job_id, JobStatus.FAILED,
                failed_at=datetime.now().isoformat(),
                error=str(e),
                stdout='\n'.join(stdout_lines) if 'stdout_lines' in locals() else '',
                stderr='\n'.join(stderr_lines) if 'stderr_lines' in locals() else '')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Don't log health checks to reduce noise
    return jsonify({
        'status': 'healthy',
        'service': 'higgs-audio-api',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/tts/generate', methods=['POST'])
def generate_tts():
    """Generate text-to-speech using HiggsAudio"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        
        # Create job ID first
        job_id = create_job_id()
        
        # Support both ref_audio (voice cloning) and voice_description
        ref_audio = data.get('ref_audio')
        voice_description = data.get('voice_description')
        
        # Determine which voice mode to use
        if voice_description:
            # Use voice description - generate a unique profile name for this custom description
            ref_audio = f"profile:custom_voice_{job_id[:8]}"
        elif not ref_audio:
            # Default to a voice if neither is specified
            ref_audio = 'broom_salesman'
        
        seed = data.get('seed', 12345)
        temperature = data.get('temperature', 0.3)  # For future use
        request_id = data.get('request_id', str(uuid.uuid4()))
        tg_chat_id = data.get('tg_chat_id')  # Get tg_chat_id from request
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sanitize request_id for Windows-compatible filename
        # Replace colons, slashes, and other invalid characters
        safe_request_id = request_id.replace(':', '-').replace('/', '-').replace('\\', '-').replace('?', '-').replace('*', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
        
        output_filename = f"{timestamp}_{safe_request_id}_{job_id}.wav"
        
        # Save initial job info
        save_job(job_id, JobStatus.PENDING,
                created_at=datetime.now().isoformat(),
                text=text,
                ref_audio=ref_audio,
                seed=seed,
                temperature=temperature,
                request_id=request_id,
                tg_chat_id=tg_chat_id,  # Save tg_chat_id
                output_filename=output_filename)
        
        # Start background processing
        # Use system message flag for voice descriptions
        use_system_message = voice_description is not None
        thread = threading.Thread(
            target=process_tts_job,
            args=(job_id, text, ref_audio, seed, temperature, output_filename, tg_chat_id, use_system_message, voice_description)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started job {job_id} for request {request_id}")
        
        return jsonify({
            'job_id': job_id,
            'status': JobStatus.PENDING,
            'request_id': request_id,
            'estimated_time_minutes': min(len(text) / 100, 10),  # Rough estimate
            'message': 'TTS generation started'
        }), 202
    
    except Exception as e:
        logger.error(f"Error starting TTS generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status and results"""
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'updated_at': job['updated_at']
    }
    
    # Add job-specific fields
    for field in ['created_at', 'started_at', 'completed_at', 'failed_at', 
                  'text', 'ref_audio', 'seed', 'temperature', 'request_id', 'tg_chat_id',
                  'processing_time_seconds', 'output_filename', 'file_size_bytes',
                  'error', 'stdout', 'stderr']:
        if field in job:
            response[field] = job[field]
    
    # For completed jobs, add file info
    if job['status'] == JobStatus.COMPLETED and 'output_filename' in job:
        response['audio_url'] = f"/tts/download/{job_id}"
        response['file_path'] = f"/app/output/{job['output_filename']}"
    
    return jsonify(response)

@app.route('/tts/download/<job_id>', methods=['GET'])
def download_audio(job_id):
    """Download generated audio file"""
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] != JobStatus.COMPLETED:
        return jsonify({'error': 'Job not completed'}), 400
    
    if 'output_filename' not in job:
        return jsonify({'error': 'No output file available'}), 404
    
    file_path = os.path.join(OUTPUT_DIR, job['output_filename'])
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Output file not found'}), 404
    
    from flask import send_file
    return send_file(file_path, as_attachment=True, 
                    download_name=f"{job.get('request_id', job_id)}.wav")

@app.route('/tts/jobs', methods=['GET'])
def list_jobs():
    """List all jobs (for debugging)"""
    with job_lock:
        return jsonify({
            'total_jobs': len(jobs),
            'jobs': dict(jobs)
        })

@app.route('/info', methods=['GET'])
def get_info():
    """Get service information"""
    with job_lock:
        active_jobs = sum(1 for job in jobs.values() if job.get('status') in ['pending', 'processing'])
        completed_jobs = sum(1 for job in jobs.values() if job.get('status') == 'completed')
        failed_jobs = sum(1 for job in jobs.values() if job.get('status') in ['failed', 'timeout'])
    
    return jsonify({
        'service': 'HiggsAudio API Server',
        'version': '1.0',
        'generation_script': GENERATION_SCRIPT,
        'output_directory': OUTPUT_DIR,
        'max_processing_time_seconds': MAX_PROCESSING_TIME,
        'job_stats': {
            'active_jobs': active_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'total_jobs': len(jobs)
        },
        'available_endpoints': [
            'GET /health',
            'POST /tts/generate',
            'GET /tts/status/<job_id>',
            'GET /tts/download/<job_id>',
            'GET /tts/jobs',
            'GET /info'
        ]
    })

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting HiggsAudio API Server")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Generation script: {GENERATION_SCRIPT}")
    logger.info(f"Max processing time: {MAX_PROCESSING_TIME} seconds")
    
    # Check if generation script exists
    if os.path.exists(GENERATION_SCRIPT):
        logger.info(f"✓ Generation script found: {GENERATION_SCRIPT}")
    else:
        logger.error(f"✗ Generation script NOT FOUND: {GENERATION_SCRIPT}")
        logger.error("Service will fail to process jobs until this is fixed!")
    
    # Check for GPU availability
    cuda_available = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
    if cuda_available:
        logger.info("✓ CUDA environment detected - GPU acceleration enabled")
    else:
        logger.info("○ CPU mode - GPU acceleration disabled")
    
    logger.info("Health check endpoint configured (logs filtered)")
    logger.info("Ready to accept TTS generation requests!")
    logger.info("=" * 60)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
