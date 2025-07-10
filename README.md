# Text Rewriter Backend

A Flask-based backend service for text rewriting, refinement, and synonym replacement using local NLP tools.

## Features

- **Text Rewriting**: Advanced text rewriting with enhanced mode
- **Synonym Replacement**: Intelligent synonym suggestions using WordNet
- **Text Refinement**: Grammar correction and text improvement using spaCy and TextBlob
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

## Testing

Run the test suite:

```bash
python test_rewriter.py
python test_humanization.py
```

## Usage Examples

### Text Rewriting

```bash
curl -X POST http://localhost:5000/rewrite \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "enhanced": true}'
```

### Synonym Replacement

```bash
curl -X POST http://localhost:5000/synonym \
  -H "Content-Type: application/json" \
  -d '{"word": "example"}'
```

### Text Refinement

```bash
curl -X POST http://localhost:5000/refine \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Dependencies

- Flask
- Flask-CORS
- NLTK
- spaCy
- TextBlob
- NumPy
- Requests
