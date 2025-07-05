# Text Rewriter & AI Detector Backend

A Flask-based backend service for text rewriting, refinement, and AI-generated text detection using local NLP tools.

## Features

- **Text Rewriting**: Advanced text rewriting with enhanced mode
- **Synonym Replacement**: Intelligent synonym suggestions using WordNet
- **Text Refinement**: Grammar correction and text improvement using spaCy and TextBlob
- **AI Text Detection**: Multi-model ensemble detection of AI-generated content
- **Local Processing**: All processing done locally without external API calls

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download required models:

```bash
python download_models.py
```

## Usage

Start the server:

```bash
python main.py
```

The server will run on `http://localhost:5000`

## API Endpoints

### Text Processing

- `GET /` - Health check
- `GET /health` - Detailed health information
- `POST /rewrite` - Main rewrite endpoint with cleaning
- `POST /rewrite_only` - Rewrite without additional cleaning
- `POST /synonym` - Get synonym for a word
- `POST /refine` - Refine text using local NLP tools

### AI Detection

- `GET /detect/models` - Get list of available detection models
- `POST /detect` - Main AI detection endpoint (ensemble)
- `POST /detect/single` - Single model detection
- `POST /detect/segments` - Analyze text segments for AI detection

## Testing

Run the test suite:

```bash
python test_rewriter.py
python test_humanization.py
python test_detector_integration.py
```

## AI Detection Usage Examples

### Basic Detection
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "use_ensemble": true}'
```

### Single Model Detection
```bash
curl -X POST http://localhost:5000/detect/single \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "model": "roberta-base-openai-detector"}'
```

### Get Available Models
```bash
curl http://localhost:5000/detect/models
```

## Dependencies

- Flask
- Flask-CORS
- NLTK
- spaCy
- TextBlob
- PyTorch
- Transformers
- NumPy
- SciPy
- Scikit-learn
- Joblib
- Requests
