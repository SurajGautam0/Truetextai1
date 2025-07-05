#!/usr/bin/env python3
"""
Test script for detector integration with the main application.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_home_endpoint():
    """Test the home endpoint to verify detector is included."""
    print("🏠 Testing home endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        data = response.json()
        print(f"✅ Home endpoint: {data['message']}")
        print(f"📋 Features: {data['features']}")
        return data['features'].get('ai_detection', False)
    except Exception as e:
        print(f"❌ Home endpoint failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint."""
    print("\n💚 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"✅ Health check: {data['status']}")
        print(f"🔧 Version: {data['version']}")
        return data['features'].get('ai_detection', False)
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_get_models():
    """Test getting available detection models."""
    print("\n🔍 Testing get models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/detect/models")
        data = response.json()
        if data.get('success'):
            print(f"✅ Available models: {len(data['available_models'])}")
            print(f"📋 Models: {data['available_models'][:3]}...")  # Show first 3
            return True
        else:
            print(f"❌ Get models failed: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Get models endpoint failed: {e}")
        return False

def test_detection_endpoint():
    """Test the main detection endpoint."""
    print("\n🔍 Testing detection endpoint...")
    
    test_text = """
    Artificial intelligence has revolutionized the way we approach problem-solving in modern technology. 
    Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions 
    with remarkable accuracy. This advancement has opened up new possibilities in fields ranging from healthcare 
    to autonomous vehicles, demonstrating the transformative potential of AI in our daily lives.
    """
    
    payload = {
        "text": test_text,
        "use_ensemble": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/detect", json=payload)
        data = response.json()
        
        if data.get('success'):
            print("✅ Detection successful!")
            result = data.get('detection_result', {})
            if 'ensemble_score' in result:
                print(f"🎯 Ensemble score: {result['ensemble_score']:.3f}")
            if 'model_scores' in result:
                print(f"📊 Model scores: {len(result['model_scores'])} models used")
            return True
        else:
            print(f"❌ Detection failed: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Detection endpoint failed: {e}")
        return False

def test_single_model_detection():
    """Test single model detection."""
    print("\n🎯 Testing single model detection...")
    
    test_text = "This is a simple test text to check AI detection capabilities."
    
    payload = {
        "text": test_text,
        "model": "roberta-base-openai-detector"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/detect/single", json=payload)
        data = response.json()
        
        if data.get('success'):
            print("✅ Single model detection successful!")
            result = data.get('detection_result', {})
            if 'ai_probability' in result:
                print(f"🤖 AI probability: {result['ai_probability']:.3f}")
            return True
        else:
            print(f"❌ Single model detection failed: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Single model detection failed: {e}")
        return False

def test_segment_analysis():
    """Test text segment analysis."""
    print("\n📊 Testing segment analysis...")
    
    test_text = """
    Artificial intelligence has revolutionized the way we approach problem-solving in modern technology. 
    Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions 
    with remarkable accuracy. This advancement has opened up new possibilities in fields ranging from healthcare 
    to autonomous vehicles, demonstrating the transformative potential of AI in our daily lives.
    """
    
    payload = {
        "text": test_text,
        "segment_length": 100
    }
    
    try:
        response = requests.post(f"{BASE_URL}/detect/segments", json=payload)
        data = response.json()
        
        if data.get('success'):
            print("✅ Segment analysis successful!")
            result = data.get('segment_analysis', {})
            if 'segments' in result:
                print(f"📝 Analyzed {len(result['segments'])} segments")
            return True
        else:
            print(f"❌ Segment analysis failed: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Segment analysis failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting detector integration tests...\n")
    
    tests = [
        ("Home Endpoint", test_home_endpoint),
        ("Health Endpoint", test_health_endpoint),
        ("Get Models", test_get_models),
        ("Detection", test_detection_endpoint),
        ("Single Model Detection", test_single_model_detection),
        ("Segment Analysis", test_segment_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Detector integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the server logs for details.")

if __name__ == "__main__":
    main() 