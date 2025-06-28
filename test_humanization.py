#!/usr/bin/env python3
"""
Test script for rewriter functionality
"""

import requests
import json
import time

def test_rewriter_modes():
    """Test different rewriter modes"""
    
    # Test text that's likely to be detected as AI
    test_text = "Artificial intelligence has revolutionized the way we approach problem-solving in modern technology. Machine learning algorithms can process vast amounts of data efficiently and provide accurate predictions. This technological advancement has significant implications for various industries including healthcare, finance, and transportation."
    
    # API endpoint
    url = "http://localhost:5000/rewrite_only"
    
    print("🧪 Testing Rewriter Modes")
    print("=" * 60)
    print(f"Original text: {test_text}")
    print("-" * 60)
    
    # Test different modes
    modes = [
        {"name": "Basic Mode", "enhanced": False},
        {"name": "Enhanced Mode", "enhanced": True}
    ]
    
    for mode in modes:
        print(f"\n📝 Testing {mode['name']}")
        print("-" * 40)
        
        payload = {
            "text": test_text,
            "enhanced": mode["enhanced"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Processing time: {end_time - start_time:.2f}s")
                print(f"Enhanced mode: {mode['enhanced']}")
                print(f"Rewritten text: {result.get('rewritten_text', 'No text returned')}")
                print(f"Original length: {len(test_text)} characters")
                print(f"New length: {len(result.get('rewritten_text', ''))} characters")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the backend server is running on port 5000")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

def test_synonym_functionality():
    """Test synonym functionality"""
    
    test_words = ["important", "significant", "technology", "innovation"]
    url = "http://localhost:5000/synonym"
    
    print("\n🔍 Testing Synonym Functionality")
    print("=" * 50)
    
    for word in test_words:
        print(f"\n📝 Testing synonym for: {word}")
        print("-" * 30)
        
        payload = {"word": word}
        
        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                result = response.json()
                synonym = result.get('synonym', 'No synonym found')
                print(f"✅ Synonym: {synonym}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the backend server is running on port 5000")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

def test_refine_functionality():
    """Test text refinement functionality"""
    
    test_text = "this is a test sentence that needs refinement. it has some grammar issues and could be improved."
    url = "http://localhost:5000/refine"
    
    print("\n🔍 Testing Text Refinement")
    print("=" * 50)
    print(f"Original text: {test_text}")
    print("-" * 50)
    
    payload = {"text": test_text}
    
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            refined_text = result.get('refined_text', 'No refined text returned')
            print(f"✅ Refined text: {refined_text}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the backend server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Testing Rewriter System")
    print("=" * 60)
    
    # Test different modes
    test_rewriter_modes()
    
    # Test synonym functionality
    test_synonym_functionality()
    
    # Test refine functionality
    test_refine_functionality()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("\n💡 Tips for better rewriting:")
    print("- Use enhanced mode for more sophisticated text rewriting")
    print("- The system applies multiple passes for maximum improvement")
    print("- Synonym replacement helps avoid repetitive language")
    print("- Text refinement improves grammar and readability") 