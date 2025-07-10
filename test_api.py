import requests
import json

def test_rewrite_api():
    """Tests the /rewrite endpoint of the API."""
    url = "http://127.0.0.1:5000/rewrite"
    data = {
        "text": "This is a test sentence.",
        "enhanced": True
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:", response.json())
        else:
            print("Error:", response.text)
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_rewrite_api()
