#!/usr/bin/env python3
"""Test script for direct voice descriptions"""

import subprocess
import sys

# Test cases for direct voice descriptions
test_cases = [
    {
        "name": "Ship AI Test",
        "text": "Welcome aboard. I have been maintaining this vessel for centuries. Your presence is... noted.",
        "description": "Ancient artificial intelligence, elderly robotic voice with electronic distortion, cold and calculating with barely hidden contempt, passive-aggressive undertones",
    },
    {
        "name": "Storyteller Test",
        "text": "Once upon a time, in a land far, far away, there lived a curious little dragon.",
        "description": "Warm elderly female storyteller, slight crackle in voice, slow dramatic pacing, mysterious and magical tone",
    },
    {
        "name": "Sports Commentator Test",
        "text": "And he shoots! What an incredible play! The crowd goes absolutely wild!",
        "description": "Energetic sports commentator, fast-paced delivery, dynamic pitch variations, building excitement, enthusiastic",
    }
]

def test_voice_description(test_case):
    """Test a single voice description"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_case['name']}")
    print(f"Description: {test_case['description'][:50]}...")
    print(f"{'='*60}")
    
    cmd = [
        'python3', 'examples/generation.py',
        '--transcript', f'"{test_case["text"]}"',
        '--ref_audio', f'profile:{test_case["description"]}',
        '--ref_audio_in_system_message',
        '--seed', '12345',
        '--temperature', '0.7',
        '--out_path', f'test_{test_case["name"].replace(" ", "_").lower()}.wav'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(
            ' '.join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            cwd='/Users/nicholasgrigoriev/Documents/personal/voice_cloning/higgs-audio'
        )
        
        if result.returncode == 0:
            print(f"✓ Success! Audio generated.")
        else:
            print(f"✗ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

if __name__ == "__main__":
    print("Testing Direct Voice Descriptions")
    print("This verifies that voice descriptions can be passed directly without YAML")
    
    success_count = 0
    for test_case in test_cases:
        if test_voice_description(test_case):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(test_cases)} tests passed")
    print(f"{'='*60}")
    
    if success_count == len(test_cases):
        print("✓ All tests passed! Direct voice descriptions are working.")
    else:
        print("✗ Some tests failed. Check the implementation.")
        sys.exit(1)