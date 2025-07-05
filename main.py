import os
import json
import time
import logging
import re
from typing import Dict, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS

# Importing core functions
from rewriter import rewrite_text, get_synonym, refine_text
from detector import AITextDetector

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app, origins="*")

def clean_final_text(text: str) -> str:
    """Post-process the rewritten text for formatting corrections."""
    if not text:
        return text
    text = text.replace("—", ", ")
    text = re.sub(r' +([,.])', r'\1', text)
    return text

class RewriterService:
    def __init__(self):
        logger.info("✅ RewriterService has been initialized")

    def rewrite_text_with_cleaning(self, text: str, enhanced: bool = True) -> Tuple[str, Dict]:
        stats = {
            "original_length": len(text),
            "enhanced_mode": enhanced,
            "processing_steps": []
        }
        try:
            logger.info("➡️ Starting rewriting process")
            result_text, err = rewrite_text(text, enhanced=enhanced)
            if err:
                logger.warning(f"⚠️ Rewriting failed: {err}")
                result_text = text
                stats["processing_steps"].append("rewrite_failed")
            else:
                stats["processing_steps"].append("rewritten")

            logger.info("🧹 Cleaning up final output text")
            result_text = clean_final_text(result_text)
            stats["processing_steps"].append("text_cleaned")

            stats["final_length"] = len(result_text)
            stats["length_diff"] = stats["final_length"] - stats["original_length"]

            return result_text, stats

        except Exception as e:
            logger.error(f"❌ Error during rewriting: {str(e)}")
            return text, {
                **stats,
                "error": str(e),
                "processing_steps": stats["processing_steps"] + ["error_occurred"]
            }

rewriter_service = RewriterService()

class DetectorService:
    def __init__(self):
        logger.info("🔍 Initializing DetectorService...")
        self.detector = AITextDetector()
        logger.info("✅ DetectorService has been initialized")

    def detect_ai_text(self, text: str, models: list = None, use_ensemble: bool = True) -> Dict:
        """
        Detect AI-generated text using the detector.
        
        Args:
            text: Input text to analyze
            models: List of specific models to use (optional)
            use_ensemble: Whether to use ensemble detection
            
        Returns:
            Dict with detection results
        """
        try:
            logger.info(f"🔍 AI detection request received (ensemble={use_ensemble})")
            
            if use_ensemble:
                if models:
                    result = self.detector.detect_selected_models(text, models)
                else:
                    result = self.detector.detect_ensemble(text)
            else:
                if models and len(models) == 1:
                    result = self.detector.detect_single_model(text, models[0])
                else:
                    result = self.detector.detect_all_models(text)
            
            return {
                "success": True,
                "detection_result": result,
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"❌ Error during AI detection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text_length": len(text)
            }

    def get_available_models(self) -> Dict:
        """Get list of available detection models."""
        try:
            models = self.detector.get_available_models()
            return {
                "success": True,
                "available_models": models,
                "total_models": len(models)
            }
        except Exception as e:
            logger.error(f"❌ Error getting available models: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_text_segments(self, text: str, segment_length: int = 200) -> Dict:
        """Analyze text segments for AI detection."""
        try:
            result = self.detector.analyze_text_segments(text, segment_length)
            return {
                "success": True,
                "segment_analysis": result,
                "segment_length": segment_length
            }
        except Exception as e:
            logger.error(f"❌ Error during segment analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

detector_service = DetectorService()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "🚀 Welcome to the Text Rewriter & AI Detector Service!",
        "features": {
            "rewrite": True,
            "synonym_lookup": True,
            "text_refinement": True,
            "ai_detection": True,
            "offline_mode": True
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "OK",
        "server_time": time.time(),
        "features": {
            "rewriting": True,
            "synonym_search": True,
            "text_refining": True,
            "ai_detection": True
        },
        "version": "v3.1.0"
    })

@app.route('/rewrite', methods=['GET', 'POST'])
def rewrite_handler():
    if request.method == 'GET':
        return jsonify({
            "message": "📬 Send a POST request with JSON body: { 'text': ..., 'enhanced': true }"
        })

    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        enhanced = data.get('enhanced', True)

        if not text:
            return jsonify({"error": "Missing 'text' field in request."}), 400

        logger.info(f"📨 Rewrite request received (enhanced={enhanced})")
        rewritten, stats = rewriter_service.rewrite_text_with_cleaning(text, enhanced)

        return jsonify({
            "rewritten_text": rewritten,
            "stats": stats,
            "success": True
        })

    except Exception as e:
        logger.error(f"❌ Exception in /rewrite: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/synonym', methods=['POST'])
def synonym_handler():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        word = data.get('word', '').strip()

        if not word:
            return jsonify({"error": "Missing 'word' field."}), 400

        logger.info(f"🔍 Synonym request for: {word}")
        synonym, err = get_synonym(word)

        if err:
            return jsonify({"error": err, "success": False}), 500

        return jsonify({
            "word": word,
            "synonym": synonym,
            "success": True
        })

    except Exception as e:
        logger.error(f"❌ Exception in /synonym: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/refine', methods=['POST'])
def refine_handler():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({"error": "Missing 'text' field."}), 400

        logger.info("🔧 Refining input text")
        refined, err = refine_text(text)

        if err:
            return jsonify({"error": err, "success": False}), 500

        return jsonify({
            "original_text": text,
            "refined_text": refined,
            "success": True
        })

    except Exception as e:
        logger.error(f"❌ Exception in /refine: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/rewrite_only', methods=['POST'])
def rewrite_only_handler():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        enhanced = data.get('enhanced', True)

        if not text:
            return jsonify({"error": "Missing 'text' field."}), 400

        logger.info(f"⚙️ Direct rewrite (enhanced={enhanced})")
        rewritten, err = rewrite_text(text, enhanced)

        if err:
            return jsonify({"error": err, "success": False}), 500

        return jsonify({
            "original_text": text,
            "rewritten_text": rewritten,
            "enhanced": enhanced,
            "success": True
        })

    except Exception as e:
        logger.error(f"❌ Exception in /rewrite_only: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

# AI Detection Endpoints
@app.route('/detect', methods=['GET', 'POST'])
def detect_handler():
    if request.method == 'GET':
        return jsonify({
            "message": "📬 Send a POST request with JSON body: { 'text': ..., 'models': [...], 'use_ensemble': true }"
        })

    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        models = data.get('models', None)
        use_ensemble = data.get('use_ensemble', True)

        if not text:
            return jsonify({"error": "Missing 'text' field in request."}), 400

        logger.info(f"🔍 AI detection request received (ensemble={use_ensemble})")
        result = detector_service.detect_ai_text(text, models, use_ensemble)

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Exception in /detect: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/detect/models', methods=['GET'])
def get_models_handler():
    """Get list of available detection models."""
    try:
        result = detector_service.get_available_models()
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Exception in /detect/models: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/detect/segments', methods=['POST'])
def detect_segments_handler():
    """Analyze text segments for AI detection."""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        segment_length = data.get('segment_length', 200)

        if not text:
            return jsonify({"error": "Missing 'text' field in request."}), 400

        logger.info(f"🔍 Segment analysis request received (length={segment_length})")
        result = detector_service.analyze_text_segments(text, segment_length)

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Exception in /detect/segments: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/detect/single', methods=['POST'])
def detect_single_model_handler():
    """Detect using a single specific model."""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        model_name = data.get('model', 'roberta-base-openai-detector')

        if not text:
            return jsonify({"error": "Missing 'text' field in request."}), 400

        logger.info(f"🔍 Single model detection request for: {model_name}")
        result = detector_service.detector.detect_single_model(text, model_name)

        return jsonify({
            "success": True,
            "model": model_name,
            "detection_result": result,
            "text_length": len(text)
        })

    except Exception as e:
        logger.error(f"❌ Exception in /detect/single: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    logger.info("🚀 Launching Text Rewriter & AI Detector API...")
    app.run(host='0.0.0.0', port=5000, debug=True)
