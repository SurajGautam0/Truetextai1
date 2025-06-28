#!/usr/bin/env python3
"""
Test script for the backend rewriter integration
"""

import requests
import json
import time

def test_rewriter_endpoint():
    """Test the rewrite_only endpoint"""
    
    # Test data
    test_text = "This is a sample text that needs to be rewritten to make it more human-like and engaging for readers."
    
    # API endpoint
    url = "http://localhost:5000/rewrite_only"
    
    # Test payload
    payload = {
        "text": test_text,
        "enhanced": True
    }
    
    print("Testing rewriter endpoint...")
    print(f"Input text: {test_text}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)
    
    try:
        # Make the request
        start_time = time.time()
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        end_time = time.time()
        
        print(f"Response status: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Rewritten text: {result.get('rewritten_text', 'No text returned')}")
            print(f"Statistics: {json.dumps(result.get('statistics', {}), indent=2)}")
        else:
            print("❌ Error!")
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the backend server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_health_endpoint():
    """Test the health check endpoint"""
    
    url = "http://localhost:5000/health"
    
    print("\nTesting health endpoint...")
    print(f"URL: {url}")
    print("-" * 30)
    
    try:
        response = requests.get(url)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed!")
            print(f"Status: {result.get('status')}")
            print(f"Features: {json.dumps(result.get('features', {}), indent=2)}")
        else:
            print("❌ Health check failed!")
            print(f"Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the backend server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing Backend Rewriter Integration")
    print("=" * 50)
    
    # Test health endpoint first
    test_health_endpoint()
    
    # Test rewriter endpoint
    test_rewriter_endpoint()
    
    print("\n" + "=" * 50)
    print("Test completed!") 