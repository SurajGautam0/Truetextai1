import os
import json
import time
from typing import Dict, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import re

# Import only rewriter functionality
from rewriter import rewrite_text, get_synonym, refine_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

def clean_final_text(text: str) -> str:
    """
    Clean the final text by:
    1. Replacing every "—" with ", "
    2. Removing spaces that appear before "," or "."
    """
    if not text:
        return text
    
    # Step 1: Replace em dashes with commas
    cleaned_text = text.replace("—", ", ")
    
    # Step 2: Remove spaces before commas and periods
    # This regex finds spaces that are followed by comma or period
    cleaned_text = re.sub(r' +([,.])', r'\1', cleaned_text)
    
    return cleaned_text

class RewriterService:
    """Main service for text rewriting functionality"""
    
    def __init__(self):
        logger.info("RewriterService initialized")
    
    def rewrite_text_with_cleaning(
        self, 
        text: str, 
        enhanced: bool = True
    ) -> Tuple[str, Dict]:
        """
        Complete text rewriting pipeline:
        1. Rewrite the text
        2. Clean the final text
        """
        
        stats = {
            "original_length": len(text),
            "enhanced_rewriting_used": enhanced,
            "processing_steps": []
        }
        
        try:
            # Step 1: Rewriting
            logger.info("Starting rewriting step")
            final_text, err = rewrite_text(text, enhanced=enhanced)
            
            if err:
                logger.warning(f"Rewriting failed: {err}")
                final_text = text
                stats["processing_steps"].append("rewriting_failed")
            else:
                stats["processing_steps"].append("rewriting")
            
            # Step 2: Clean the final text
            logger.info("Cleaning final text")
            final_text = clean_final_text(final_text)
            stats["processing_steps"].append("text_cleaning")
            
            stats["final_length"] = len(final_text)
            stats["length_change"] = stats["final_length"] - stats["original_length"]
            
            return final_text, stats
            
        except Exception as e:
            logger.error(f"Error in rewriting pipeline: {str(e)}")
            return text, {
                **stats,
                "error": str(e),
                "processing_steps": stats["processing_steps"] + ["error"]
            }

# Initialize service
rewriter_service = RewriterService()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "🚀 Text Rewriter Server is running!",
        "features": {
            "text_rewriting": True,
            "synonym_support": True,
            "text_refinement": True,
            "local_processing": True
        }
    })

@app.route('/health', methods=['GET'])
def detailed_health():
    """Detailed health check with system information"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "features": {
            "text_rewriting_available": True,
            "local_processing": True
        },
        "version": "3.0.0"
    })

@app.route('/rewrite', methods=['POST'])
def rewrite_handler():
    """Main rewrite endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        text = data.get('text', '').strip()
        enhanced = data.get('enhanced', True)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        logger.info(f"Processing rewrite request (enhanced: {enhanced})")
        
        final_text, stats = rewriter_service.rewrite_text_with_cleaning(text, enhanced)
        
        return jsonify({
            "rewritten_text": final_text,
            "stats": stats,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in rewrite handler: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/synonym', methods=['POST'])
def synonym_handler():
    """Get synonym for a word"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({"error": "No word provided"}), 400
        
        logger.info(f"Processing synonym request for: {word}")
        
        synonym, error = get_synonym(word)
        
        if error:
            return jsonify({
                "error": error,
                "success": False
            }), 500
        
        return jsonify({
            "original_word": word,
            "synonym": synonym,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in synonym handler: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/refine', methods=['POST'])
def refine_handler():
    """Refine text using local NLP tools"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        logger.info("Processing refine request")
        
        refined_text, error = refine_text(text)
        
        if error:
            return jsonify({
                "error": error,
                "success": False
            }), 500
        
        return jsonify({
            "original_text": text,
            "refined_text": refined_text,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in refine handler: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/rewrite_only', methods=['POST'])
def rewrite_only_handler():
    """Rewrite text without additional cleaning"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        text = data.get('text', '').strip()
        enhanced = data.get('enhanced', True)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        logger.info(f"Processing rewrite_only request (enhanced: {enhanced})")
        
        rewritten_text, error = rewrite_text(text, enhanced=enhanced)
        
        if error:
            return jsonify({
                "error": error,
                "success": False
            }), 500
        
        return jsonify({
            "original_text": text,
            "rewritten_text": rewritten_text,
            "enhanced": enhanced,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in rewrite_only handler: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    logger.info("Starting Text Rewriter Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)