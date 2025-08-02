#!/usr/bin/env python3
"""
Script to pre-download and cache Higgs Audio models for faster container startup.
Run this during Docker build or as an initialization step.
"""

import os
import sys
import time
from pathlib import Path

def download_models():
    """Download all required models for Higgs Audio."""
    
    print("🚀 Starting Higgs Audio model pre-download...")
    start_time = time.time()
    
    try:
        # Import required modules
        print("📦 Importing dependencies...")
        from transformers import AutoTokenizer, AutoConfig
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        from boson_multimodal.model.higgs_audio import HiggsAudioModel
        import torch
        
        # Set cache directories
        cache_dir = os.environ.get('HF_HOME', '/root/.cache/huggingface')
        print(f"📁 Using cache directory: {cache_dir}")
        
        # Download main generation model
        print("🧠 Downloading main generation model (bosonai/higgs-audio-v2-generation-3B-base)...")
        model_start = time.time()
        model = HiggsAudioModel.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        )
        model_time = time.time() - model_start
        print(f"✅ Main model downloaded in {model_time:.1f}s")
        
        # Download tokenizer
        print("🔤 Downloading text tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            cache_dir=cache_dir
        )
        tokenizer_time = time.time() - tokenizer_start
        print(f"✅ Text tokenizer downloaded in {tokenizer_time:.1f}s")
        
        # Download audio tokenizer
        print("🎵 Downloading audio tokenizer (bosonai/higgs-audio-v2-tokenizer)...")
        audio_tokenizer_start = time.time()
        audio_tokenizer = load_higgs_audio_tokenizer(
            "bosonai/higgs-audio-v2-tokenizer",
            device="cpu"
        )
        audio_tokenizer_time = time.time() - audio_tokenizer_start
        print(f"✅ Audio tokenizer downloaded in {audio_tokenizer_time:.1f}s")
        
        # Check cache size
        cache_path = Path(cache_dir)
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            print(f"📊 Total cache size: {size_gb:.2f} GB")
        
        total_time = time.time() - start_time
        print(f"\n🎉 All models downloaded successfully in {total_time:.1f}s!")
        print("⚡ Future container startups will be much faster!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_models():
    """Verify that all models are properly cached."""
    
    print("\n🔍 Verifying model cache...")
    
    cache_dir = os.environ.get('HF_HOME', '/root/.cache/huggingface')
    hub_cache = Path(cache_dir) / "hub"
    
    expected_models = [
        "models--bosonai--higgs-audio-v2-generation-3B-base",
        "models--bosonai--higgs-audio-v2-tokenizer",
        "models--bosonai--hubert_base"  # This might be downloaded as dependency
    ]
    
    found_models = []
    missing_models = []
    
    for model in expected_models:
        model_path = hub_cache / model
        if model_path.exists():
            found_models.append(model)
            # Check size
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024**2)
            print(f"✅ {model}: {size_mb:.1f} MB")
        else:
            missing_models.append(model)
            print(f"❌ {model}: Not found")
    
    print(f"\n📈 Models found: {len(found_models)}")
    if missing_models:
        print(f"⚠️  Models missing: {missing_models}")
        return False
    else:
        print("🎯 All expected models are cached!")
        return True

if __name__ == "__main__":
    print("🔧 Higgs Audio Startup Optimizer")
    print("=" * 50)
    
    # Set environment variables
    os.environ.setdefault('HF_HOME', '/root/.cache/huggingface')
    os.environ.setdefault('TRANSFORMERS_CACHE', '/root/.cache/huggingface/transformers')
    os.environ.setdefault('HF_HUB_CACHE', '/root/.cache/huggingface/hub')
    
    success = download_models()
    
    if success:
        verify_models()
        print("\n🚀 Optimization complete! Container startup should now be faster.")
        sys.exit(0)
    else:
        print("\n💥 Optimization failed. Check the errors above.")
        sys.exit(1)