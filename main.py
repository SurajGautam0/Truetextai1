import os
import json
import time
import logging
import re
from typing import Dict, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS

# Importing core functions
from rewriter import rewrite_text, get_synonym, refine_text, rewrite_text_academic, rewrite_text_academic

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

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "🚀 Welcome to the Academic Text Rewriter Service!",
        "features": {
            "rewrite": True,
            "academic_rewrite": True,
            "synonym_lookup": True,
            "text_refinement": True,
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
            "academic_rewriting": True,
            "synonym_search": True,
            "text_refining": True
        },
        "version": "v3.1.0"
    })

@app.route('/rewrite', methods=['GET', 'POST'])
def rewrite_handler():
    if request.method == 'GET':
        return jsonify({
            "message": "Send a POST request with JSON body: { 'text': ..., 'enhanced': false }"
        })

    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        enhanced = data.get('enhanced', False)

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
        enhanced = data.get('enhanced', False)

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

@app.route('/rewrite_academic', methods=['POST'])
def rewrite_academic_handler():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. JSON required."}), 400

        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({"error": "Missing 'text' field."}), 400

        logger.info("🎓 Academic rewrite request")
        rewritten, err = rewrite_text_academic(text)

        if err:
            return jsonify({"error": err, "success": False}), 500

        return jsonify({
            "original_text": text,
            "rewritten_text": rewritten,
            "style": "academic",
            "success": True
        })

    except Exception as e:
        logger.error(f"❌ Exception in /rewrite_academic: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    logger.info("🚀 Launching Text Rewriter API...")
    app.run(host='0.0.0.0', port=5000, debug=True)

