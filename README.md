<h1 align="center">Higgs Audio V2: Redefining Expressiveness in Audio Generation</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://boson.ai/blog/higgs-audio-v2"><img src='https://img.shields.io/badge/🚀-Launch Blogpost-228B22' style="margin-right: 5px;"></a>
  <a href="https://boson.ai/demo/tts"><img src="https://img.shields.io/badge/🕹️-Boson%20AI%20Playground-9C276A" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/spaces/smola/higgs_audio_v2"><img src="https://img.shields.io/badge/🎮-HF%20Space%20Playground-8A2BE2" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base"><img src="https://img.shields.io/badge/🤗-Checkpoints (3.6B LLM + 2.2B audio adapter)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>


We are open-sourcing Higgs Audio v2, a powerful audio foundation model pretrained on over 10 million hours of audio data and a diverse set of text data. Despite having no post-training or fine-tuning, Higgs Audio v2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

On [EmergentTTS-Eval](https://github.com/boson-ai/emergenttts-eval-public), it achieves win rates of **75.7%** and **55.7%** over "gpt-4o-mini-tts" on the "Emotions" and "Questions" categories, respectively. It also obtains state-of-the-art performance on traditional TTS benchmarks like Seed-TTS Eval and Emotional Speech Dataset (ESD). Moreover, the model demonstrates capabilities rarely seen in previous systems, including generating natural multi-speaker dialogues in multiple languages, automatic prosody adaptation during narration, melodic humming with the cloned voice, and simultaneous generation of speech and background music.

<p align="center">
    <img src="figures/emergent-tts-emotions-win-rate.png" width=900>
</p>

Here's the demo video that shows some of its emergent capabilities (remember to unmute):

<video src="https://github.com/user-attachments/assets/0fd73fad-097f-48a9-9f3f-bc2a63b3818d" type="video/mp4" width="80%" controls>
</video>

Here's another demo video that show-cases the model's multilingual capability and how it enabled live translation (remember to unmute):

<video src="https://github.com/user-attachments/assets/2b9b01ff-67fc-4bd9-9714-7c7df09e38d6" type="video/mp4" width="80%" controls>
</video>

## Installation

We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. Following are two docker images that we have verified:
- nvcr.io/nvidia/pytorch:25.02-py3
- nvcr.io/nvidia/pytorch:25.01-py3

Here's an example command for launching a docker container environment. Please also check the [official NVIDIA documentations](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash
```

### Option 1: Direct installation


```bash
git clone https://github.com/NicholasGrigoriev/higgs-audio.git
cd higgs-audio

pip install -r requirements.txt
pip install -e .
```

### Option 2: Using venv

```bash
git clone https://github.com/NicholasGrigoriev/higgs-audio.git
cd higgs-audio

python3 -m venv higgs_audio_env
source higgs_audio_env/bin/activate
pip install -r requirements.txt
pip install -e .
```


### Option 3: Using conda
```bash
git clone https://github.com/NicholasGrigoriev/higgs-audio.git
cd higgs-audio

conda create -y --prefix ./conda_env --override-channels --strict-channel-priority --channel "conda-forge" "python==3.10.*"
conda activate ./conda_env
pip install -r requirements.txt
pip install -e .

# Uninstalling environment:
conda deactivate
conda remove -y --prefix ./conda_env --all
```

### Option 4: Using uv
```bash
git clone https://github.com/NicholasGrigoriev/higgs-audio.git
cd higgs-audio

uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Option 5: Using vllm

For advanced usage with higher throughput, we also built OpenAI compatible API server backed by vLLM engine for you to use.
Please refer to [examples/vllm](./examples/vllm) for more details.

### Option 6: Using Docker with API Server

For production use, we provide a REST API server with Docker support:

```bash
# Build and run with Docker
docker build -t higgs-audio .
docker-compose up -d

# API will be available at http://localhost:8000
```

The API server includes:
- Asynchronous job processing
- n8n webhook integration for workflow automation
- AWS SQS integration for message queue processing
- Health monitoring and job status tracking


## Usage

> [!TIP]
> For optimal performance, run the generation examples on a machine equipped with GPU with at least 24GB memory!

### Get Started

Here's a basic python snippet to help you get started.

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

We also provide a list of examples under [examples](./examples). In the following we highlight a few examples to help you use Higgs Audio v2.

### REST API Usage

The Docker setup includes a REST API server for easy integration:

```bash
# Generate audio
curl -X POST http://localhost:8000/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test",
    "ref_audio": "belinda",
    "request_id": "test-123"
  }'

# Check job status
curl http://localhost:8000/tts/status/{job_id}

# Download generated audio
curl http://localhost:8000/tts/download/{job_id} -o output.wav
```

API Endpoints:
- `POST /tts/generate` - Generate audio from text
- `GET /tts/status/{job_id}` - Get job status
- `GET /tts/download/{job_id}` - Download generated audio
- `GET /health` - Health check
- `GET /info` - Service information

### Zero-Shot Voice Cloning
Generate audio that sounds similar as the provided [reference audio](./examples/voice_prompts/belinda.wav).

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation.wav
```

The generation script will automatically use `cuda:0` if it founds cuda is available. To change the device id, specify `--device_id`:

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--device_id 0 \
--out_path generation.wav
```

You can also try other voices. Check more example voices in [examples/voice_prompts](./examples/voice_prompts). You can also add your own voice to the folder.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio broom_salesman \
--temperature 0.3 \
--out_path generation.wav
```

### Single-speaker Generation with Smart Voice
If you do not specify reference voice, the model will decide the voice based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--temperature 0.3 \
--out_path generation.wav
```


### Multi-speaker Dialog with Smart Voice
Generate multi-speaker dialog. The model will decide the voices based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-speaker Dialog with Voice Clone

Generate multi-speaker dialog with the voices you picked.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```


## Technical Details
<img src="figures/higgs_audio_v2_architecture_combined.png" width=900>


Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:
- We developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as **AudioVerse**. The in-house understanding model is finetuned on top of [Higgs Audio v1 Understanding](https://www.boson.ai/blog/higgs-audio), which adopts the "understanding variant" shown in the architecture figure.
- We trained a unified audio tokenizer from scratch that captures both semantic and acoustic features. We also open-sourced our evaluation set on [HuggingFace](https://huggingface.co/datasets/bosonai/AudioTokenBench). Learn more in the [tokenizer blog](./tech_blogs/TOKENIZER_BLOG.md).
- We proposed the DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead. See the [architecture blog](./tech_blogs/ARCHITECTURE_BLOG.md).

## Evaluation

Here's the performance of Higgs Audio v2 on four benchmarks,  [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval), [Emotional Speech Dataset (ESD)](https://paperswithcode.com/dataset/esd), [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), and Multi-speaker Eval:

#### Seed-TTS Eval & ESD

We prompt Higgs Audio v2 with the reference text, reference audio, and target text for zero-shot TTS. We use the standard evaluation metrics from Seed-TTS Eval and ESD.

|                              | SeedTTS-Eval| | ESD   |                 |
|------------------------------|--------|--------|---------|-------------------|
|                              | WER ↓ | SIM ↑ | WER ↓ | SIM (emo2vec) ↑ |
| Cosyvoice2                   | 2.28   | 65.49  | 2.71    | 80.48             |
| Qwen2.5-omni†                | 2.33   | 64.10  | -       | -                 |
| ElevenLabs Multilingual V2   | **1.43**   | 50.00  | 1.66    | 65.87             |
| Higgs Audio v1                | 2.18   | 66.27  | **1.49**    | 82.84             |
| Higgs Audio v2 (base)         | 2.44   | **67.70**  | 1.78    | **86.13**         |


#### EmergentTTS-Eval ("Emotions" and "Questions")

Following the [EmergentTTS-Eval Paper](https://arxiv.org/abs/2505.23009), we report the win-rate over "gpt-4o-mini-tts" with the "alloy" voice. The judge model is Gemini 2.5 Pro.

| Model                              | Emotions (%) ↑ | Questions (%) ↑ |
|------------------------------------|--------------|----------------|
| Higgs Audio v2 (base)               | **75.71%**   | **55.71%**         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)       | 61.64%       | 47.85%         |
| [Hume.AI](https://www.hume.ai/research)                            | 61.60%       | 43.21%         |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)  | 50.00%       | 50.00%         |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)      | 41.60%       | 51.78%         |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)               | 40.86%        | 47.32%         |
| [ElevenLabs Multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)         | 30.35%       | 39.46%         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)                    | 29.28%       | 48.21%         |
| [Sesame csm-1B](https://github.com/SesameAILabs/csm)                      | 15.96%       | 31.78%         |

<sup><sub>'†' means using the strong-prompting method described in the paper.</sub></sup>


#### Multi-speaker Eval

We also designed a multi-speaker evaluation benchmark to evaluate the capability of Higgs Audio v2 for multi-speaker dialog generation. The benchmark contains three subsets

- `two-speaker-conversation`: 1000 synthetic dialogues involving two speakers. We fix two reference audio clips to evaluate the model's ability in double voice cloning for utterances ranging from 4 to 10 dialogues between two randomly chosen persona.
- `small talk (no ref)`: 250 synthetic dialogues curated in the same way as above, but are characterized by short utterances and a limited number of turns (4–6), we do not fix reference audios in this case and this set is designed to evaluate the model's ability to automatically assign appropriate voices to speakers.
- `small talk (ref)`: 250 synthetic dialogues similar to above, but contains even shorter utterances as this set is meant to include reference clips in it's context, similar to `two-speaker-conversation`.


We report the word-error-rate (WER) and the geometric mean between intra-speaker similarity and inter-speaker dis-similarity on these three subsets. Other than Higgs Audio v2, we also evaluated [MoonCast](https://github.com/jzq2000/MoonCast) and [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626), two of the most popular open-source models capable of multi-speaker dialog generation. Results are summarized in the following table. We are not able to run [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) on our "two-speaker-conversation" subset due to its strict limitation on the length of the utterances and output audio.

|                                                | two-speaker-conversation |                |small talk |                | small talk (no ref) |                |
| ---------------------------------------------- | -------------- | ------------------ | ---------- | -------------- | ------------------- | -------------- |
|                                                | WER ↓                      | Mean Sim & Dis-sim ↑ | WER ↓       |  Mean Sim & Dis-sim ↑ | WER ↓               | Mean Sim & Dis-sim ↑ |
| [MoonCast](https://github.com/jzq2000/MoonCast) | 38.77                    | 46.02         | **8.33**       | 63.68          | 24.65               | 53.94 |
| [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626)         | \-                       | \-             | 17.62      | 63.15          | 19.46               | **61.14**          |
| Higgs Audio v2 (base)     | **18.88**                    | **51.95**          | 11.89      | **67.92**              | **14.65**               | 55.28              |


## API Usage Examples

### Multi-Speaker Dialogue Generation

The API server provides a `/tts/dialogue` endpoint for generating multi-speaker dialogues. This endpoint accepts an array of dialogue turns with speaker assignments and voice specifications.

#### Dialogue Request Format

```bash
curl -X POST http://localhost:8000/tts/dialogue \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": [
      {
        "speaker": "Alice",
        "text": "Hey Bob, have you seen the new AI model?",
        "voice": "belinda"
      },
      {
        "speaker": "Bob",
        "text": "Yes! It is incredible how far we have come with voice synthesis.",
        "voice": "profile:male_en_british"
      },
      {
        "speaker": "Alice",
        "text": "I know, right? The quality is almost indistinguishable from human speech."
      },
      {
        "speaker": "Bob",
        "text": "Absolutely. What excites me most is the potential for creative applications."
      }
    ],
    "options": {
      "seed": 12345,
      "temperature": 0.3,
      "scene_prompt": "Two colleagues having an enthusiastic conversation about AI technology in a quiet office"
    },
    "request_id": "dialogue-example-001"
  }'
```

#### Voice Specification Options

1. **Voice Cloning**: Use pre-loaded voice names
   - `"voice": "belinda"` - Female voice
   - `"voice": "broom_salesman"` - Male voice  
   - `"voice": "en_man"`, `"voice": "en_woman"` - Alternative voices

2. **Voice Descriptions**: Use profile-based descriptions
   - `"voice": "profile:male_en_british"` - British male voice
   - `"voice": "profile:female_en_story"` - Female storyteller voice
   - `"voice": "profile:Young enthusiastic female with slight valley girl accent"`
   - `"voice": "profile:Deep male voice, slow and mysterious, with a slight rasp"`

3. **Automatic Voice Assignment**: Omit voice field
   - The system will automatically alternate between appropriate voices

#### Response Format

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "request_id": "dialogue-example-001",
  "speakers": ["Alice", "Bob"],
  "dialogue_turns": 4,
  "estimated_time_minutes": 2,
  "message": "Dialogue generation started"
}
```

#### Checking Job Status

```bash
curl http://localhost:8000/tts/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

#### Python Example

```python
import requests
import time

# Submit dialogue generation request
response = requests.post(
    "http://localhost:8000/tts/dialogue",
    json={
        "dialogue": [
            {"speaker": "Narrator", "text": "Once upon a time, in a distant kingdom...", 
             "voice": "profile:Elderly female storyteller with warm, crackling voice"},
            {"speaker": "Knight", "text": "I shall defend this realm with my life!", 
             "voice": "profile:Young brave male knight, heroic and determined"},
            {"speaker": "Dragon", "text": "You dare challenge me, mortal?", 
             "voice": "profile:Ancient dragon, deep rumbling voice with echo"}
        ],
        "options": {"temperature": 0.4, "scene_prompt": "Epic fantasy scene"},
        "request_id": "fantasy-dialogue"
    }
)

job_data = response.json()
job_id = job_data["job_id"]

# Poll for completion
while True:
    status_response = requests.get(f"http://localhost:8000/tts/status/{job_id}")
    status = status_response.json()
    
    if status["status"] == "completed":
        print(f"Dialogue ready: {status['output_filename']}")
        # Download the audio file
        audio_response = requests.get(f"http://localhost:8000/tts/download/{job_id}")
        with open("dialogue.wav", "wb") as f:
            f.write(audio_response.content)
        break
    elif status["status"] == "failed":
        print(f"Generation failed: {status.get('error')}")
        break
    
    time.sleep(5)  # Wait 5 seconds before checking again
```

## Citation

If you feel the repository is helpful, please kindly cite as:

```
@misc{higgsaudio2025,
  author       = {{Boson AI}},
  title        = {{Higgs Audio V2: Redefining Expressiveness in Audio Generation}},
  year         = {2025},
  howpublished = {\url{https://github.com/boson-ai/higgs-audio}},
  note         = {GitHub repository. Release blog available at \url{https://www.boson.ai/blog/higgs-audio-v2}},
}
```

## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.
