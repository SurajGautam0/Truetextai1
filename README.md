# Text Rewriter Backend

A Flask-based backend service for text rewriting and refinement using local NLP tools.

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

## Dependencies

- Flask
- Flask-CORS
- NLTK
- spaCy
- TextBlob
