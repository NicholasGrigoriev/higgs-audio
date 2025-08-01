# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

### Prerequisites
- NVIDIA GPU with at least 24GB memory (recommended for optimal performance)
- CUDA-compatible environment (recommended: NVIDIA Deep Learning Container)
- Python 3.10

### Installation Commands
```bash
# Clone repository (original upstream)
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

### Docker Installation
```bash
# Build and run with Docker
docker build -t higgs-audio .
docker run --gpus all -p 5000:5000 higgs-audio

# Or use docker-compose
docker-compose up --build
```

### Alternative Installation Methods
- **venv**: `python3 -m venv higgs_audio_env && source higgs_audio_env/bin/activate`
- **conda**: `conda create -y --prefix ./conda_env --channel "conda-forge" "python==3.10.*"`
- **uv**: `uv venv --python 3.10 && source .venv/bin/activate`

## Core Commands

### Code Quality
```bash
# Format and lint code
ruff format .
ruff check .
```

### Audio Generation Examples
```bash
# Basic generation with voice cloning
python3 examples/generation.py \
--transcript "Your text here" \
--ref_audio belinda \
--temperature 0.3 \
--out_path output.wav

# Smart voice (auto voice selection)
python3 examples/generation.py \
--transcript "Your text here" \
--temperature 0.3 \
--out_path output.wav

# Multi-speaker dialog
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path output.wav
```

### vLLM Server (High Throughput)
```bash
# Start vLLM server
docker run --gpus all --ipc=host --shm-size=20gb --network=host \
bosonai/higgs-audio-vllm:latest \
--served-model-name "higgs-audio-v2-generation-3B-base" \
--model "bosonai/higgs-audio-v2-generation-3B-base" \
--audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
--limit-mm-per-prompt audio=50 \
--max-model-len 8192 \
--port 8000

# Test with curl
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "higgs-audio-v2-generation-3B-base", "voice": "en_woman", "input": "Hello world!", "response_format": "pcm"}' \
  --output - | ffmpeg -f s16le -ar 24000 -ac 1 -i - speech.wav
```

## Architecture Overview

### Core Model Components
- **HiggsAudioModel**: Main multimodal model combining LLM with audio generation (`boson_multimodal/model/higgs_audio/modeling_higgs_audio.py`)
- **HiggsAudioServeEngine**: High-level serving interface (`boson_multimodal/serve/serve_engine.py`)
- **Audio Tokenizer**: Unified tokenizer for semantic and acoustic features (`boson_multimodal/audio_processing/higgs_audio_tokenizer.py`)
- **DualFFN Architecture**: Audio-specific expert module enhancing LLM capabilities with minimal overhead

### Data Flow
1. **Input Processing**: ChatML format with text/audio content → tokenization
2. **Model Architecture**: Llama-3.2-3B base + DualFFN audio adapter + delay pattern for RVQ codebooks
3. **Audio Generation**: Residual Vector-Quantization (RVQ) with multiple codebooks → audio output

### Key Directories
- `boson_multimodal/model/higgs_audio/`: Core model implementation
- `boson_multimodal/serve/`: Serving engine and utilities
- `boson_multimodal/audio_processing/`: Audio tokenization and processing
- `examples/`: Usage examples and demos
- `examples/voice_prompts/`: Reference voices for cloning
- `examples/transcript/`: Sample transcripts for testing

### Model Capabilities
- Zero-shot voice cloning from reference audio
- Multi-speaker dialog generation
- Smart voice selection (automatic voice assignment)
- Streaming audio generation
- Multi-language support (English, Chinese, etc.)
- Background music generation alongside speech
- Melodic humming with cloned voices

### Performance Notes
- **A100 40GB**: ~1500 tokens/s (60 seconds audio/second)
- **RTX 4090 24GB**: ~600 tokens/s (24 seconds audio/second)
- Uses delay pattern for streaming-compatible generation
- 91% of original LLM training speed preserved with DualFFN

## Development Notes

### Code Style
- Uses Ruff for formatting and linting (configured in `pyproject.toml`)
- Line length: 119 characters
- Python 3.10 target version
- Double quotes for strings, space indentation

### Model Configuration
- Base models: `bosonai/higgs-audio-v2-generation-3B-base` (3.6B LLM + 2.2B audio adapter)
- Audio tokenizer: `bosonai/higgs-audio-v2-tokenizer`
- Default generation parameters: temperature=0.3, top_p=0.95, top_k=50

### API Server
```bash
# Start Flask API server
python3 api_server.py

# Server runs on http://localhost:5000 with endpoints:
# GET /health - Health check
# POST /generate - Audio generation
```

### Testing
This project currently uses example scripts for validation rather than formal unit tests:
```bash
# Test basic generation
python3 examples/generation.py --transcript "Hello world" --out_path test_output.wav

# Test voice cloning
python3 examples/generation.py --transcript "Hello world" --ref_audio belinda --out_path test_clone.wav

# Test multi-speaker dialogue
python3 examples/generation.py --transcript examples/transcript/multi_speaker/en_argument.txt --out_path test_dialog.wav
```