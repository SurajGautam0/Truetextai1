#!/usr/bin/env python3
"""
Quick test script for rewriter endpoints.
"""

import requests
import json
import time

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    print("✍️ Testing rewriter endpoints...")
    
    # Test 1: Home endpoint
    try:
        response = requests.get(f"{base_url}/")
        data = response.json()
        print(f"✅ Home: {data['message']}")
        print(f"📋 Features: {list(data['features'].keys())}")
    except Exception as e:
        print(f"❌ Home failed: {e}")
        return
    
    # Test 2: Health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        data = response.json()
        if data.get('status') == 'OK':
            print(f"✅ Health: Server is running (version {data.get('version', 'unknown')})")
            print(f"📋 Features: {list(data['features'].keys())}")
        else:
            print(f"❌ Health failed: {data}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
    
    # Test 3: Text rewriting
    test_text = "This is a simple test text that needs to be rewritten for better clarity."
    payload = {
        "text": test_text,
        "enhanced": True
    }
    
    try:
        response = requests.post(f"{base_url}/rewrite", json=payload)
        data = response.json()
        if data.get('success'):
            print("✅ Rewrite: Success!")
            print(f"📝 Original: {test_text}")
            print(f"✍️ Rewritten: {data.get('rewritten_text', '')[:100]}...")
            stats = data.get('stats', {})
            if stats:
                print(f"📊 Length change: {stats.get('length_diff', 0)} characters")
        else:
            print(f"❌ Rewrite failed: {data.get('error')}")
    except Exception as e:
        print(f"❌ Rewrite failed: {e}")
    
    # Test 4: Synonym endpoint
    test_word = "beautiful"
    payload = {"word": test_word}
    
    try:
        response = requests.post(f"{base_url}/synonym", json=payload)
        data = response.json()
        if data.get('success'):
            print(f"✅ Synonym: '{test_word}' -> '{data.get('synonym', '')}'")
        else:
            print(f"❌ Synonym failed: {data.get('error')}")
    except Exception as e:
        print(f"❌ Synonym failed: {e}")
    
    # Test 5: Text refinement
    test_text_refine = "this text has some grammar issue and need refinement"
    payload = {"text": test_text_refine}
    
    try:
        response = requests.post(f"{base_url}/refine", json=payload)
        data = response.json()
        if data.get('success'):
            print("✅ Refine: Success!")
            print(f"📝 Original: {test_text_refine}")
            print(f"🔧 Refined: {data.get('refined_text', '')}")
        else:
            print(f"❌ Refine failed: {data.get('error')}")
    except Exception as e:
        print(f"❌ Refine failed: {e}")
    
    # Test 6: Academic rewriting
    test_text_academic = "This is a simple test sentence that should be rewritten in academic style."
    payload = {"text": test_text_academic}
    
    try:
        response = requests.post(f"{base_url}/rewrite_academic", json=payload)
        data = response.json()
        if data.get('success'):
            print("✅ Academic Rewrite: Success!")
            print(f"📝 Original: {test_text_academic}")
            print(f"🎓 Academic: {data.get('rewritten_text', '')}")
        else:
            print(f"❌ Academic rewrite failed: {data.get('error')}")
    except Exception as e:
        print(f"❌ Academic rewrite failed: {e}")

if __name__ == "__main__":
    # Wait a bit for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    test_endpoints()