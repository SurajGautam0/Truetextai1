import subprocess
import sys

def download_spacy_model():
    """Download the spaCy model required for the rewriter"""
    print("Downloading spaCy model for text processing...")
    try:
        # Download the spaCy model
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("✓ spaCy model 'en_core_web_sm' downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download spaCy model: {e}")
    except Exception as e:
        print(f"✗ Unexpected error downloading spaCy model: {e}")

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            'punkt',
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for data_package in nltk_data:
            print(f"Downloading {data_package}...")
            try:
                nltk.download(data_package, quiet=True)
                print(f"✓ {data_package} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {data_package}: {e}")
                
    except ImportError:
        print("✗ NLTK not installed. Please install it with: pip install nltk")
    except Exception as e:
        print(f"✗ Unexpected error downloading NLTK data: {e}")

if __name__ == "__main__":
    print("🚀 Downloading models for Text Rewriter")
    print("=" * 50)
    
    # Download spaCy model
    download_spacy_model()
    
    # Download NLTK data
    download_nltk_data()
    
    print("\n" + "=" * 50)
    print("Model download completed!")
    print("\n💡 Make sure you have the following packages installed:")
    print("- flask")
    print("- flask_cors") 
    print("- nltk")
    print("- spacy")
    print("- textblob")
