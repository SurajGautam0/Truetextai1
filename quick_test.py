#!/usr/bin/env python3
"""
Quick test script for detector endpoints.
"""

import requests
import json
import time

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    print("🔍 Testing detector endpoints...")
    
    # Test 1: Home endpoint
    try:
        response = requests.get(f"{base_url}/")
        data = response.json()
        print(f"✅ Home: {data['message']}")
        print(f"📋 Features: {list(data['features'].keys())}")
    except Exception as e:
        print(f"❌ Home failed: {e}")
        return
    
    # Test 2: Get available models
    try:
        response = requests.get(f"{base_url}/detect/models")
        data = response.json()
        if data.get('success'):
            print(f"✅ Models: {len(data['available_models'])} available")
            print(f"📋 First 3 models: {data['available_models'][:3]}")
        else:
            print(f"❌ Models failed: {data.get('error')}")
    except Exception as e:
        print(f"❌ Models failed: {e}")
    
    # Test 3: Basic detection
    test_text = "This is a test text to check AI detection capabilities."
    payload = {
        "text": test_text,
        "use_ensemble": True
    }
    
    try:
        response = requests.post(f"{base_url}/detect", json=payload)
        data = response.json()
        if data.get('success'):
            print("✅ Detection: Success!")
            result = data.get('detection_result', {})
            if 'ensemble_score' in result:
                print(f"🎯 Ensemble score: {result['ensemble_score']:.3f}")
        else:
            print(f"❌ Detection failed: {data.get('error')}")
    except Exception as e:
        print(f"❌ Detection failed: {e}")

if __name__ == "__main__":
    # Wait a bit for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    test_endpoints() 