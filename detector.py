import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import logging
import requests
import json
import re
import os
import hashlib
from collections import Counter, defaultdict
import scipy.stats as stats
from datetime import datetime, timedelta
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import functools
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Create a cache directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Simple caching decorator for detection results
def cached_detection(ttl_seconds=3600):
    """
    Decorator to cache detection results.
    
    Args:
        ttl_seconds: Time to live for cache in seconds (default: 1 hour)
    """
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(text, *args, **kwargs):
            # Create a cache key based on text content and function arguments
            text_hash = hashlib.md5(text.encode()).hexdigest()
            args_str = str(args) + str(sorted(kwargs.items()))
            args_hash = hashlib.md5(args_str.encode()).hexdigest()
            cache_key = f"{func.__name__}_{text_hash}_{args_hash}"
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.joblib")
            
            # Check if result is in cache and not expired
            if os.path.exists(cache_path):
                mtime = os.path.getmtime(cache_path)
                if time.time() - mtime < ttl_seconds:
                    try:
                        return joblib.load(cache_path)
                    except Exception:
                        # If loading fails, recompute
                        pass
            
            # Compute the result
            result = func(text, *args, **kwargs)
            
            # Store in cache
            try:
                joblib.dump(result, cache_path)
            except Exception as e:
                logging.warning(f"Failed to cache result: {e}")
                
            return result
            
        return wrapper
    
    return decorator

class LinguisticFeatureExtractor:
    """
    Extractor for advanced linguistic features and AI pattern detection.
    """
    
    def __init__(self):
        # Initialize any required resources or models
        pass
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with extracted feature values
        """
        features = {}
        
        try:
            # Basic features
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            num_words = len(words)
            num_sentences = len(sentences)
            
            features['num_words'] = num_words
            features['num_sentences'] = num_sentences
            features['avg_word_length'] = sum(len(w) for w in words) / num_words if num_words > 0 else 0
            features['avg_sentence_length'] = num_words / num_sentences if num_sentences > 0 else 0
            
            # Sentiment features
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            features['sentiment_pos'] = sentiment['pos']
            features['sentiment_neg'] = sentiment['neg']
            features['sentiment_neutral'] = sentiment['neu']
            features['sentiment_compound'] = sentiment['compound']
            
            # Lexical diversity
            features['lexical_diversity'] = len(set(words)) / num_words if num_words > 0 else 0
            
        except Exception as e:
            logging.warning(f"Feature extraction error: {e}")
        
        return features
    
    def analyze_ai_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze text for patterns indicative of AI-generated content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with AI pattern indicators
        """
        patterns = {}
        
        try:
            # Basic patterns
            words = word_tokenize(text)
            num_words = len(words)
            num_sentences = len(sent_tokenize(text))
            
            # Repetitive phrases (bigram repetition)
            bigrams = list(ngrams(words, 2))
            bigram_freq = FreqDist(bigrams)
            repetitive_phrases = [phrase for phrase, count in bigram_freq.items() if count > 1]
            patterns['bigram_repetition'] = len(repetitive_phrases) / (num_words - 1) if num_words > 1 else 0
            
            # Cliché density (simple heuristic: overused bigrams)
            common_bigrams = set([
                ('to', 'be'), ('of', 'the'), ('in', 'the'), ('at', 'the'), ('on', 'the'), 
                ('and', 'the'), ('for', 'the'), ('with', 'the'), ('as', 'a'), ('by', 'the')
            ])
            cliche_count = sum(1 for bg in repetitive_phrases if bg in common_bigrams)
            patterns['cliche_density'] = cliche_count / (num_words - 1) if num_words > 1 else 0
            
            # Sentence length uniformity (AI texts often have similar sentence lengths)
            sentence_lengths = [len(s.split()) for s in sent_tokenize(text)]
            if len(sentence_lengths) > 1:
                mean_length = np.mean(sentence_lengths)
                std_length = np.std(sentence_lengths)
                patterns['sentence_length_uniformity'] = 1.0 - min(1.0, std_length / mean_length)
            else:
                patterns['sentence_length_uniformity'] = 1.0
            
            # Sentiment consistency (AI texts often have consistent sentiment)
            if num_sentences > 1:
                sentiment_changes = sum(1 for i in range(1, len(sentence_lengths)) 
                                        if (sentence_lengths[i] - sentence_lengths[i-1]) * (sentence_lengths[i-1] - mean_length) < 0)
                patterns['sentiment_consistency'] = 1.0 - min(1.0, sentiment_changes / (num_sentences - 1))
            else:
                patterns['sentiment_consistency'] = 1.0
            
        except Exception as e:
            logging.warning(f"AI pattern analysis error: {e}")
        
        return patterns

class AITextDetector:
    """
    A utility class for detecting AI-generated text using multiple open source models.
    Enhanced with advanced linguistic analysis and pattern detection.
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logger()
        self.feature_extractor = LinguisticFeatureExtractor()
        self.model_weights = self._get_default_model_weights()
        self.perplexity_thresholds = self._get_default_perplexity_thresholds()
        
    def _get_default_model_weights(self) -> Dict[str, float]:
        """Get default weights for each model for weighted ensemble"""
        return {
            "roberta-base-openai-detector": 0.8,
            "roberta-large-openai-detector": 0.9, 
            "chatgpt-detector": 0.85,
            "mixed-detector": 0.95,
            "multilingual-detector": 0.7,
            "distilbert-detector": 0.6,
            "bert-detector": 0.65,
            "bart-detector": 0.5
        }
    
    def _get_default_perplexity_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Get default perplexity thresholds for different models (low, high)"""
        return {
            "roberta-base-openai-detector": (2.0, 8.0),
            "roberta-large-openai-detector": (1.8, 7.5),
            "chatgpt-detector": (2.2, 8.5),
            "mixed-detector": (2.0, 8.0),
            "bart-detector": (3.0, 10.0)
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the detector."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def load_model(self, model_name: str = "roberta-base-openai-detector") -> bool:
        """
        Load a specific AI detection model.
        
        Args:
            model_name: Name of the model to load. Options:
                - "roberta-base-openai-detector": OpenAI's RoBERTa detector
                - "roberta-large-openai-detector": OpenAI's RoBERTa large detector
                - "hello-simpleai/chatgpt-detector-roberta": ChatGPT detector
                - "andreas122001/roberta-mixed-detector": Mixed AI detector
                - "AI4Bharat/IndicBERTv2-MLM-only": Good for multilingual detection
                - "microsoft/DialoGPT-medium": Dialog-specific detection
                - "unitary/toxic-bert": Can help identify AI patterns
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_map = {
                "roberta-base-openai-detector": "roberta-base-openai-detector",
                "roberta-large-openai-detector": "roberta-large-openai-detector", 
                "chatgpt-detector": "hello-simpleai/chatgpt-detector-roberta",
                "mixed-detector": "andreas122001/roberta-mixed-detector",
                "multilingual-detector": "papluca/xlm-roberta-base-language-detection",
                "distilbert-detector": "distilbert-base-uncased-finetuned-sst-2-english",
                "bert-detector": "textattack/bert-base-uncased-ag-news",
                "bart-detector": "facebook/bart-base"  # Note: Not ideal for classification
            }
            
            hf_model_name = model_map.get(model_name, model_name)
            
            self.logger.info(f"Loading model: {hf_model_name}")
            
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(hf_model_name)
            self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
            self.models[model_name].to(self.device)
            self.models[model_name].eval()
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def detect_single_model(self, text: str, model_name: str) -> Dict[str, float]:
        """
        Detect AI-generated text using a single model.
        
        Args:
            text: Input text to analyze
            model_name: Name of the model to use
            
        Returns:
            Dict with 'ai_probability' and 'human_probability'
        """
        if model_name not in self.models:
            if not self.load_model(model_name):
                raise ValueError(f"Failed to load model: {model_name}")
        
        try:
            # Special handling for BART model (seq2seq, not classification)
            if model_name == "bart-detector":
                return self._detect_with_bart(text, model_name)
            
            # Regular classification model handling
            # Tokenize the input text
            inputs = self.tokenizers[model_name](
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.models[model_name](**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
            # Convert to numpy for easier handling
            probs = probabilities.cpu().numpy()[0]
            
            # Most models output [human, ai] probabilities
            if len(probs) == 2:
                human_prob = float(probs[0])
                ai_prob = float(probs[1])
            else:
                # Handle edge cases
                ai_prob = float(probs[0]) if len(probs) == 1 else 0.5
                human_prob = 1.0 - ai_prob
            
            # Calculate perplexity for more information
            with torch.no_grad():
                perplexity = None
                try:
                    # Only calculate if model supports it
                    if hasattr(self.models[model_name], "compute_transition_scores"):
                        # Calculate approximated perplexity
                        input_ids = inputs["input_ids"]
                        outputs = self.models[model_name].generate(
                            input_ids, 
                            max_length=input_ids.shape[1] + 5,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        transition_scores = self.models[model_name].compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )
                        perplexity = float(-torch.mean(transition_scores).exp())
                except Exception as e:
                    self.logger.debug(f"Perplexity calculation error for {model_name}: {e}")
            
            result = {
                'ai_probability': ai_prob,
                'human_probability': human_prob,
                'model_used': model_name
            }
            
            if perplexity is not None:
                result['perplexity'] = perplexity
                
                # Adjust probability based on perplexity thresholds
                if model_name in self.perplexity_thresholds:
                    low_thresh, high_thresh = self.perplexity_thresholds[model_name]
                    if perplexity < low_thresh:
                        # Lower perplexity suggests more AI-like text
                        result['ai_probability'] = min(0.95, result['ai_probability'] * 1.2)
                        result['human_probability'] = 1.0 - result['ai_probability']
                    elif perplexity > high_thresh:
                        # Higher perplexity suggests more human-like text
                        result['human_probability'] = min(0.95, result['human_probability'] * 1.2)
                        result['ai_probability'] = 1.0 - result['human_probability']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during detection with {model_name}: {str(e)}")
            return {
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'model_used': model_name,
                'error': str(e)
            }
    
    def detect_ensemble(self, text: str, models: Optional[List[str]] = None, use_weighted: bool = True) -> Dict:
        """
        Detect AI-generated text using multiple models and ensemble their results.
        Enhanced with weighted voting and linguistic feature analysis.
        
        Args:
            text: Input text to analyze
            models: List of model names to use. If None, uses default models.
            use_weighted: Whether to use weighted voting for ensemble
            
        Returns:
            Dict with ensemble results and individual model results
        """
        if models is None:
            models = [
                "chatgpt-detector",
                "mixed-detector",
                "roberta-base-openai-detector"
            ]
        
        results = {}
        ai_probs = []
        human_probs = []
        model_weights = []
        
        for model_name in models:
            try:
                result = self.detect_single_model(text, model_name)
                results[model_name] = result
                
                if 'error' not in result:
                    ai_probs.append(result['ai_probability'])
                    human_probs.append(result['human_probability'])
                    # Get model weight (or default to 1.0 if not specified)
                    weight = self.model_weights.get(model_name, 1.0)
                    model_weights.append(weight)
                    
            except Exception as e:
                self.logger.error(f"Error with model {model_name}: {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'ai_probability': 0.5,
                    'human_probability': 0.5
                }
        
        # Calculate ensemble results
        ensemble_result = {}
        
        if ai_probs:
            if use_weighted and model_weights:
                # Weighted average
                weights_sum = sum(model_weights)
                if weights_sum > 0:
                    ensemble_ai_prob = sum(p * w for p, w in zip(ai_probs, model_weights)) / weights_sum
                    ensemble_human_prob = sum(p * w for p, w in zip(human_probs, model_weights)) / weights_sum
                else:
                    ensemble_ai_prob = np.mean(ai_probs)
                    ensemble_human_prob = np.mean(human_probs)
            else:
                # Simple average
                ensemble_ai_prob = np.mean(ai_probs)
                ensemble_human_prob = np.mean(human_probs)
            
            # Calculate confidence
            confidence = 1.0 - np.std(ai_probs)  # Higher std = lower confidence
            
            # Extract linguistic features
            linguistic_features = self.feature_extractor.extract_features(text)
            ai_patterns = self.feature_extractor.analyze_ai_patterns(text)
            
            # Adjust ensemble probability based on linguistic features and AI patterns
            if linguistic_features and ai_patterns:
                # Features that suggest AI text when high
                ai_indicators = [
                    ai_patterns.get('sentence_length_uniformity', 0.5),
                    ai_patterns.get('sentence_structure_uniformity', 0.5),
                    ai_patterns.get('sentiment_consistency', 0.5),
                    ai_patterns.get('cliche_density', 0) * 5.0,  # Scale up this feature
                    1.0 - linguistic_features.get('lexical_diversity', 0.5),  # Lower diversity = more AI-like
                    ai_patterns.get('bigram_repetition', 0) * 3.0  # Scale up this feature
                ]
                
                # Features that suggest human text when high
                human_indicators = [
                    1.0 - ai_patterns.get('sentence_start_diversity', 0.5),
                    linguistic_features.get('sentence_length_variance', 0) / 10.0,  # Normalize
                    linguistic_features.get('sentence_length_std', 0) / 10.0,  # Normalize
                ]
                
                # Calculate average indicators
                avg_ai_indicator = sum(ai_indicators) / len(ai_indicators) if ai_indicators else 0.5
                avg_human_indicator = sum(human_indicators) / len(human_indicators) if human_indicators else 0.5
                
                # Adjust probabilities with a small weight to the linguistic features (20%)
                linguistic_weight = 0.2
                ensemble_ai_prob = (ensemble_ai_prob * (1.0 - linguistic_weight)) + (avg_ai_indicator * linguistic_weight)
                ensemble_human_prob = (ensemble_human_prob * (1.0 - linguistic_weight)) + (avg_human_indicator * linguistic_weight)
                
                # Normalize to ensure they sum to 1.0
                total_prob = ensemble_ai_prob + ensemble_human_prob
                if total_prob > 0:
                    ensemble_ai_prob /= total_prob
                    ensemble_human_prob /= total_prob
        else:
            ensemble_ai_prob = 0.5
            ensemble_human_prob = 0.5
            confidence = 0.0
            linguistic_features = {}
            ai_patterns = {}
        
        # Create enhanced ensemble result
        ensemble_result = {
            'ensemble_ai_probability': float(ensemble_ai_prob),
            'ensemble_human_probability': float(ensemble_human_prob),
            'confidence': float(max(0.0, confidence)),
            'prediction': 'AI-generated' if ensemble_ai_prob > 0.5 else 'Human-written',
            'prediction_strength': 'Strong' if abs(ensemble_ai_prob - 0.5) > 0.3 else 
                                   'Moderate' if abs(ensemble_ai_prob - 0.5) > 0.15 else 'Weak',
            'individual_results': results,
            'models_used': models,
            'weighted_ensemble': use_weighted
        }
        
        # Add feature analysis summaries if available
        if linguistic_features:
            ensemble_result['feature_analysis'] = {
                'lexical_diversity': linguistic_features.get('lexical_diversity', 0),
                'avg_sentence_length': linguistic_features.get('avg_sentence_length', 0),
                'sentence_length_variance': linguistic_features.get('sentence_length_variance', 0),
                'sentiment_stats': {
                    'positive': linguistic_features.get('sentiment_pos', 0),
                    'negative': linguistic_features.get('sentiment_neg', 0),
                    'neutral': linguistic_features.get('sentiment_neutral', 0),
                    'compound': linguistic_features.get('sentiment_compound', 0),
                }
            }
        
        if ai_patterns:
            ensemble_result['ai_pattern_analysis'] = {
                'repetitive_phrases': ai_patterns.get('bigram_repetition', 0),
                'sentence_uniformity': ai_patterns.get('sentence_length_uniformity', 0),
                'sentiment_consistency': ai_patterns.get('sentiment_consistency', 0),
                'cliche_density': ai_patterns.get('cliche_density', 0)
            }
        
        return ensemble_result
    
    def analyze_text_segments(self, text: str, segment_length: int = 200) -> Dict:
        """
        Analyze text by breaking it into segments for more detailed analysis.
        Enhanced with advanced segment consistency analysis.
        
        Args:
            text: Input text to analyze
            segment_length: Length of each segment in characters
            
        Returns:
            Dict with segment-wise analysis and overall results
        """
        # Split text into segments
        segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
        segment_results = []
        
        for i, segment in enumerate(segments):
            if len(segment.strip()) < 50:  # Skip very short segments
                continue
                
            result = self.detect_ensemble(segment)
            result['segment_index'] = i
            result['segment_text'] = segment[:100] + "..." if len(segment) > 100 else segment
            
            # Extract linguistic features for this segment
            result['linguistic_features'] = self.feature_extractor.extract_features(segment)
            
            segment_results.append(result)
        
        # Calculate overall statistics
        if segment_results:
            # Basic statistics
            overall_ai_prob = np.mean([r['ensemble_ai_probability'] for r in segment_results])
            overall_confidence = np.mean([r['confidence'] for r in segment_results])
            
            # Calculate consistency (how similar are the predictions across segments)
            ai_probs = [r['ensemble_ai_probability'] for r in segment_results]
            consistency = 1.0 - np.std(ai_probs) if len(ai_probs) > 1 else 1.0
            
            # Analyze stylistic consistency across segments
            if len(segment_results) > 1:
                # Extract key features for consistency analysis
                sent_lengths = [r['linguistic_features'].get('avg_sentence_length', 0) for r in segment_results 
                               if 'linguistic_features' in r and 'avg_sentence_length' in r['linguistic_features']]
                
                lex_diversity = [r['linguistic_features'].get('lexical_diversity', 0) for r in segment_results 
                                if 'linguistic_features' in r and 'lexical_diversity' in r['linguistic_features']]
                
                sentiment = [r['linguistic_features'].get('sentiment_compound', 0) for r in segment_results 
                            if 'linguistic_features' in r and 'sentiment_compound' in r['linguistic_features']]
                
                # Calculate consistency metrics (lower std = higher consistency)
                style_consistency_metrics = {}
                
                if sent_lengths:
                    style_consistency_metrics['sentence_length_consistency'] = 1.0 - min(1.0, np.std(sent_lengths) / (np.mean(sent_lengths) + 0.01))
                
                if lex_diversity:
                    style_consistency_metrics['lexical_diversity_consistency'] = 1.0 - min(1.0, np.std(lex_diversity) / (np.mean(lex_diversity) + 0.01))
                
                if sentiment:
                    style_consistency_metrics['sentiment_consistency'] = 1.0 - min(1.0, np.std(sentiment) / 0.5)  # Normalize to 0-1
                
                # Overall style consistency (average of metrics)
                if style_consistency_metrics:
                    style_consistency = sum(style_consistency_metrics.values()) / len(style_consistency_metrics)
                else:
                    style_consistency = 0.5  # Default
            else:
                style_consistency = 0.5
                style_consistency_metrics = {}
                
            # Enhanced stylistic transition analysis (potential indicator of mixed human/AI text)
            if len(segment_results) > 1:
                # Calculate the absolute differences between adjacent segments
                ai_prob_diffs = [abs(ai_probs[i] - ai_probs[i-1]) for i in range(1, len(ai_probs))]
                max_ai_prob_diff = max(ai_prob_diffs) if ai_prob_diffs else 0
                avg_ai_prob_diff = np.mean(ai_prob_diffs) if ai_prob_diffs else 0
                
                # Detect any abrupt style transitions
                abrupt_transitions = sum(1 for diff in ai_prob_diffs if diff > 0.3)
                
                # If we have strong transitions, this could indicate mixed source text
                transition_analysis = {
                    'max_transition': float(max_ai_prob_diff),
                    'avg_transition': float(avg_ai_prob_diff),
                    'abrupt_transitions': abrupt_transitions,
                    'segments_with_abrupt_transitions': [i+1 for i in range(len(ai_prob_diffs)) if ai_prob_diffs[i] > 0.3],
                    'possible_mixed_source': abrupt_transitions > 0 and max_ai_prob_diff > 0.4
                }
            else:
                transition_analysis = {
                    'max_transition': 0.0,
                    'avg_transition': 0.0,
                    'abrupt_transitions': 0,
                    'possible_mixed_source': False
                }
        else:
            overall_ai_prob = 0.5
            overall_confidence = 0.0
            consistency = 0.0
            style_consistency = 0.5
            style_consistency_metrics = {}
            transition_analysis = {'possible_mixed_source': False}
        
        # Prepare final result
        result = {
            'overall_ai_probability': float(overall_ai_prob),
            'overall_prediction': 'AI-generated' if overall_ai_prob > 0.5 else 'Human-written',
            'confidence': float(overall_confidence),
            'consistency': float(max(0.0, consistency)),
            'style_consistency': float(style_consistency),
            'style_consistency_metrics': style_consistency_metrics,
            'transition_analysis': transition_analysis,
            'segment_results': segment_results,
            'total_segments': len(segment_results),
            'text_length': len(text),
            'segment_length_used': segment_length
        }
        
        return result
    
    def detect_ai_lines(self, text: str, threshold: float = 0.6, min_line_length: int = 20) -> Dict:
        """
        Detect which specific lines in the text are likely AI-generated.
        Enhanced with advanced pattern matching and context-aware analysis.
        
        Args:
            text: Input text to analyze
            threshold: Threshold for considering a line AI-generated (0.0 to 1.0)
            min_line_length: Minimum line length to analyze (characters)
            
        Returns:
            Dict with line-by-line analysis results
        """
        lines = text.split('\n')
        line_results = []
        ai_detected_lines = []
        human_lines = []
        
        # First pass: analyze each line individually
        for i, line in enumerate(lines):
            # Skip empty or very short lines
            if len(line.strip()) < min_line_length:
                continue
            
            try:
                # Analyze each line
                result = self.detect_ensemble(line.strip())
                ai_prob = result['ensemble_ai_probability']
                
                # Extract linguistic features
                features = self.feature_extractor.extract_features(line.strip())
                ai_patterns = self.feature_extractor.analyze_ai_patterns(line.strip())
                
                line_analysis = {
                    'line_number': i + 1,
                    'line_text': line.strip(),
                    'ai_probability': ai_prob,
                    'is_ai_generated': ai_prob > threshold,
                    'confidence': result['confidence'],
                    'linguistic_features': {k: v for k, v in features.items() if k in [
                        'lexical_diversity', 'avg_sentence_length', 'sentiment_compound'
                    ]} if features else {},
                    'ai_patterns': {k: v for k, v in ai_patterns.items() if k in [
                        'bigram_repetition', 'cliche_density', 'sentence_length_uniformity'
                    ]} if ai_patterns else {}
                }
                
                line_results.append(line_analysis)
                
                if ai_prob > threshold:
                    ai_detected_lines.append({
                        'line_number': i + 1,
                        'text': line.strip(),
                        'ai_probability': ai_prob
                    })
                else:
                    human_lines.append({
                        'line_number': i + 1,
                        'text': line.strip(),
                        'ai_probability': ai_prob
                    })
                    
            except Exception as e:
                self.logger.error(f"Error analyzing line {i+1}: {str(e)}")
                continue
        
        # Second pass: Context-aware analysis
        # Look at surrounding lines for context
        context_aware_results = []
        
        for i, line_analysis in enumerate(line_results):
            # Get surrounding line analysis
            prev_line = line_results[i-1] if i > 0 else None
            next_line = line_results[i+1] if i < len(line_results)-1 else None
            
            # Current classification
            current_ai_prob = line_analysis['ai_probability']
            
            # Adjust based on surrounding context
            if prev_line and next_line:
                # If both surrounding lines are strongly classified the same way
                # but current line is weakly classified differently, adjust it
                prev_ai = prev_line['ai_probability'] > 0.7
                next_ai = next_line['ai_probability'] > 0.7
                current_weak = 0.4 < current_ai_prob < 0.6
                
                if prev_ai and next_ai and current_weak:
                    # Surrounded by AI text but weakly human - adjust up
                    adjusted_prob = min(0.75, current_ai_prob * 1.3)
                elif not prev_ai and not next_ai and current_weak:
                    # Surrounded by human text but weakly AI - adjust down
                    adjusted_prob = max(0.25, current_ai_prob * 0.7)
                else:
                    adjusted_prob = current_ai_prob
            else:
                adjusted_prob = current_ai_prob
            
            # Update with context-aware probability
            updated_analysis = line_analysis.copy()
            if abs(adjusted_prob - current_ai_prob) > 0.05:
                updated_analysis['ai_probability'] = float(adjusted_prob)
                updated_analysis['is_ai_generated'] = adjusted_prob > threshold
                updated_analysis['context_adjusted'] = True
            
            context_aware_results.append(updated_analysis)
        
        # Replace original analysis with context-aware analysis
        line_results = context_aware_results
        
        # Recalculate AI/human lines based on context-aware analysis
        ai_detected_lines = []
        human_lines = []
        for result in line_results:
            if result['is_ai_generated']:
                ai_detected_lines.append({
                    'line_number': result['line_number'],
                    'text': result['line_text'],
                    'ai_probability': result['ai_probability'],
                    'context_adjusted': result.get('context_adjusted', False)
                })
            else:
                human_lines.append({
                    'line_number': result['line_number'],
                    'text': result['line_text'],
                    'ai_probability': result['ai_probability'],
                    'context_adjusted': result.get('context_adjusted', False)
                })
        
        # Calculate overall statistics
        if line_results:
            total_lines = len(line_results)
            ai_lines_count = len(ai_detected_lines)
            human_lines_count = len(human_lines)
            
            overall_ai_percentage = (ai_lines_count / total_lines) * 100
            avg_ai_probability = np.mean([r['ai_probability'] for r in line_results])
            
        else:
            total_lines = ai_lines_count = human_lines_count = 0
            overall_ai_percentage = avg_ai_probability = 0.0
        
        return {
            'ai_detected_lines': ai_detected_lines,
            'human_lines': human_lines,
            'line_analysis': line_results,
            'statistics': {
                'total_lines_analyzed': total_lines,
                'ai_generated_lines': ai_lines_count,
                'human_written_lines': human_lines_count,
                'ai_percentage': float(overall_ai_percentage),
                'average_ai_probability': float(avg_ai_probability)
            },
            'threshold_used': threshold
        }

    def detect_ai_sentences(self, text: str, threshold: float = 0.6) -> Dict:
        """
        Detect which specific sentences in the text are likely AI-generated.
        Enhanced with advanced pattern matching and context-aware analysis.
        
        Args:
            text: Input text to analyze
            threshold: Threshold for considering a sentence AI-generated
            
        Returns:
            Dict with sentence-by-sentence analysis results
        """
        # Split text into sentences using NLTK for better accuracy
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple regex if NLTK fails
            sentences = re.split(r'[.!?]+', text)
            
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        sentence_results = []
        ai_detected_sentences = []
        human_sentences = []
        
        # First pass: analyze each sentence individually
        for i, sentence in enumerate(sentences):
            try:
                # Basic model-based detection
                result = self.detect_ensemble(sentence)
                ai_prob = result['ensemble_ai_probability']
                
                # Extract advanced linguistic features
                features = self.feature_extractor.extract_features(sentence)
                ai_patterns = self.feature_extractor.analyze_ai_patterns(sentence)
                
                # Adjust probability based on linguistic features (subtle adjustment)
                if features and ai_patterns:
                    # AI indicators that might increase probability
                    cliche_density = ai_patterns.get('cliche_density', 0)
                    sentiment_consistency = ai_patterns.get('sentiment_consistency', 0.5)
                    repetition = ai_patterns.get('bigram_repetition', 0)
                    
                    # Calculate adjustment (small effect, max ±0.1)
                    adjustment = ((cliche_density * 0.3) + 
                                 (sentiment_consistency * 0.2) + 
                                 (repetition * 0.5)) / 10.0
                    
                    # Apply adjustment
                    adjusted_ai_prob = min(0.95, max(0.05, ai_prob + adjustment))
                else:
                    adjusted_ai_prob = ai_prob
                
                sentence_analysis = {
                    'sentence_number': i + 1,
                    'sentence_text': sentence,
                    'ai_probability': adjusted_ai_prob,
                    'original_ai_probability': ai_prob,
                    'is_ai_generated': adjusted_ai_prob > threshold,
                    'confidence': result['confidence'],
                    'linguistic_features': {k: v for k, v in features.items() if k in [
                        'lexical_diversity', 'avg_sentence_length', 'sentiment_compound'
                    ]} if features else {},
                    'ai_patterns': {k: v for k, v in ai_patterns.items() if k in [
                        'bigram_repetition', 'cliche_density', 'sentence_length_uniformity'
                    ]} if ai_patterns else {}
                }
                
                sentence_results.append(sentence_analysis)
                
                if adjusted_ai_prob > threshold:
                    ai_detected_sentences.append({
                        'sentence_number': i + 1,
                        'text': sentence,
                        'ai_probability': adjusted_ai_prob
                    })
                else:
                    human_sentences.append({
                        'sentence_number': i + 1,
                        'text': sentence,
                        'ai_probability': adjusted_ai_prob
                    })
                    
            except Exception as e:
                self.logger.error(f"Error analyzing sentence {i+1}: {str(e)}")
                continue
        
        # Second pass: Context-aware analysis
        # Look at surrounding sentences for context
        context_aware_results = []
        
        for i, sentence_analysis in enumerate(sentence_results):
            # Get surrounding sentence analysis
            prev_sent = sentence_results[i-1] if i > 0 else None
            next_sent = sentence_results[i+1] if i < len(sentence_results)-1 else None
            
            # Current classification
            current_ai_prob = sentence_analysis['ai_probability']
            
            # Adjust based on surrounding context
            if prev_sent and next_sent:
                # If both surrounding sentences are strongly classified the same way
                # but current sentence is weakly classified differently, adjust it
                prev_ai = prev_sent['ai_probability'] > 0.7
                next_ai = next_sent['ai_probability'] > 0.7
                current_weak = 0.4 < current_ai_prob < 0.6
                
                if prev_ai and next_ai and current_weak:
                    # Surrounded by AI text but weakly human - adjust up
                    adjusted_prob = min(0.75, current_ai_prob * 1.3)
                elif not prev_ai and not next_ai and current_weak:
                    # Surrounded by human text but weakly AI - adjust down
                    adjusted_prob = max(0.25, current_ai_prob * 0.7)
                else:
                    adjusted_prob = current_ai_prob
            else:
                adjusted_prob = current_ai_prob
            
            # Update with context-aware probability
            updated_analysis = sentence_analysis.copy()
            if abs(adjusted_prob - current_ai_prob) > 0.05:
                updated_analysis['ai_probability'] = float(adjusted_prob)
                updated_analysis['is_ai_generated'] = adjusted_prob > threshold
                updated_analysis['context_adjusted'] = True
            
            context_aware_results.append(updated_analysis)
        
        # Replace original analysis with context-aware analysis
        sentence_results = context_aware_results
        
        # Recalculate AI/human sentences based on context-aware analysis
        ai_detected_sentences = []
        human_sentences = []
        for result in sentence_results:
            if result['is_ai_generated']:
                ai_detected_sentences.append({
                    'sentence_number': result['sentence_number'],
                    'text': result['sentence_text'],
                    'ai_probability': result['ai_probability'],
                    'context_adjusted': result.get('context_adjusted', False)
                })
            else:
                human_sentences.append({
                    'sentence_number': result['sentence_number'],
                    'text': result['sentence_text'],
                    'ai_probability': result['ai_probability'],
                    'context_adjusted': result.get('context_adjusted', False)
                })
        
        # Calculate statistics
        if sentence_results:
            total_sentences = len(sentence_results)
            ai_sentences_count = len(ai_detected_sentences)
            ai_percentage = (ai_sentences_count / total_sentences) * 100
            avg_ai_probability = np.mean([r['ai_probability'] for r in sentence_results])
            
            # Calculate transition points (possible human-AI boundaries)
            transition_points = []
            for i in range(1, len(sentence_results)):
                prev_is_ai = sentence_results[i-1]['is_ai_generated']
                curr_is_ai = sentence_results[i]['is_ai_generated']
                
                if prev_is_ai != curr_is_ai:
                    transition_points.append({
                        'position': i,
                        'sentence_number': sentence_results[i]['sentence_number'],
                        'transition': 'human-to-ai' if curr_is_ai else 'ai-to-human',
                        'prev_probability': sentence_results[i-1]['ai_probability'],
                        'curr_probability': sentence_results[i]['ai_probability']
                    })
        else:
            total_sentences = ai_sentences_count = 0
            ai_percentage = avg_ai_probability = 0.0
            transition_points = []
        
        return {
            'ai_detected_sentences': ai_detected_sentences,
            'human_sentences': human_sentences,
            'sentence_analysis': sentence_results,
            'transition_points': transition_points,
            'statistics': {
                'total_sentences_analyzed': total_sentences,
                'ai_generated_sentences': ai_sentences_count,
                'human_written_sentences': len(human_sentences),
                'ai_percentage': float(ai_percentage),
                'average_ai_probability': float(avg_ai_probability),
                'transition_count': len(transition_points)
            },
            'threshold_used': threshold
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of all available AI detection models.
        
        Returns:
            List of available model names
        """
        return [
            "roberta-base-openai-detector",
            "roberta-large-openai-detector",
            "chatgpt-detector",
            "mixed-detector",
            "multilingual-detector",
            "distilbert-detector",
            "bert-detector",
            "bart-detector"
        ]
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available AI detection models.
        
        Returns:
            Dict with model names and their loading status
        """
        available_models = self.get_available_models()
        loading_results = {}
        
        for model_name in available_models:
            try:
                success = self.load_model(model_name)
                loading_results[model_name] = success
                if success:
                    self.logger.info(f"Successfully loaded {model_name}")
                else:
                    self.logger.warning(f"Failed to load {model_name}")
            except Exception as e:
                self.logger.error(f"Error loading {model_name}: {str(e)}")
                loading_results[model_name] = False
        
        return loading_results
    
    def detect_all_models(self, text: str) -> Dict:
        """
        Detect AI-generated text using ALL available models.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with results from all models and ensemble
        """
        available_models = self.get_available_models()
        return self.detect_ensemble(text, models=available_models)
    
    def detect_selected_models(self, text: str, selected_models: List[str]) -> Dict:
        """
        Detect AI-generated text using specific selected models.
        
        Args:
            text: Input text to analyze
            selected_models: List of specific model names to use
            
        Returns:
            Dict with results from selected models and ensemble
        """
        # Validate that selected models are available
        available_models = self.get_available_models()
        valid_models = [model for model in selected_models if model in available_models]
        
        if not valid_models:
            raise ValueError(f"None of the selected models are available. Available models: {available_models}")
        
        if len(valid_models) != len(selected_models):
            invalid_models = [model for model in selected_models if model not in available_models]
            self.logger.warning(f"Invalid models ignored: {invalid_models}")
        
        return self.detect_ensemble(text, models=valid_models)
    
    def detect_top_n_models(self, text: str, n: int = 3, criteria: str = "performance") -> Dict:
        """
        Detect AI-generated text using top N models based on specified criteria.
        
        Args:
            text: Input text to analyze
            n: Number of top models to use
            criteria: Selection criteria ('performance', 'speed', 'accuracy')
            
        Returns:
            Dict with results from top N models and ensemble
        """
        # Define model rankings based on different criteria
        model_rankings = {
            "performance": [
                "mixed-detector",
                "roberta-large-openai-detector", 
                "chatgpt-detector",
                "roberta-base-openai-detector",
                "multilingual-detector",
                "distilbert-detector",
                "bert-detector",
                "bart-detector"
            ],
            "speed": [
                "roberta-base-openai-detector",
                "distilbert-detector",
                "chatgpt-detector",
                "mixed-detector",
                "roberta-large-openai-detector",
                "multilingual-detector",
                "bert-detector",
                "bart-detector"
            ],
            "accuracy": [
                "mixed-detector",
                "roberta-large-openai-detector",
                "chatgpt-detector",
                "roberta-base-openai-detector",
                "multilingual-detector",
                "distilbert-detector",
                "bert-detector",
                "bart-detector"
            ]
        }
        
        if criteria not in model_rankings:
            raise ValueError(f"Invalid criteria. Choose from: {list(model_rankings.keys())}")
        
        top_models = model_rankings[criteria][:n]
        
        return self.detect_ensemble(text, models=top_models)

    def _detect_with_bart(self, text: str, model_name: str) -> Dict[str, float]:
        """
        Custom detection method for BART model using perplexity-based approach.
        Since BART is not a classification model, we use perplexity to estimate AI-likeness.
        Enhanced with improved perplexity calculation and better thresholding.
        """
        try:
            from transformers import BartTokenizer, BartForConditionalGeneration
            import torch.nn.functional as F
            
            # Load BART tokenizer and model for generation
            if model_name not in self.models:
                tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
                model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
                model.to(self.device)
                model.eval()
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
            
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            
            # Calculate perplexity using the model's loss
            with torch.no_grad():
                # For BART, we can use the decoder to calculate loss on the same text
                outputs = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            # Enhanced conversion of perplexity to AI probability with dynamic thresholding
            # Use different thresholds for different text lengths
            text_length = len(text.split())
            
            if text_length < 30:
                # For short text, we use a different scale
                low_thresh, high_thresh = 4.0, 15.0
            elif text_length < 100:
                # Medium text
                low_thresh, high_thresh = 3.5, 12.0
            else:
                # Long text typically has lower perplexity
                low_thresh, high_thresh = 3.0, 10.0
            
            # Map perplexity to probability
            if perplexity <= low_thresh:
                # Very low perplexity strongly indicates AI-generated text
                ai_probability = 0.9
            elif perplexity >= high_thresh:
                # High perplexity suggests human-written text
                ai_probability = 0.1
            else:
                # Linear interpolation for values in between
                ai_probability = 0.9 - 0.8 * ((perplexity - low_thresh) / (high_thresh - low_thresh))
            
            human_probability = 1.0 - ai_probability
            
            # Feature-based adjustments for short texts
            if text_length < 50:
                # For short texts, extract additional features
                features = self.feature_extractor.extract_features(text)
                if features:
                    # Use lexical diversity as an additional signal
                    lex_diversity = features.get('lexical_diversity', 0.5)
                    # Higher diversity typically indicates human text
                    if lex_diversity > 0.8 and ai_probability > 0.5:
                        ai_probability = max(0.1, ai_probability - 0.2)
                    elif lex_diversity < 0.4 and ai_probability < 0.5:
                        ai_probability = min(0.9, ai_probability + 0.2)
                    
                    human_probability = 1.0 - ai_probability
            
            return {
                'ai_probability': float(ai_probability),
                'human_probability': float(human_probability),
                'model_used': model_name,
                'perplexity': float(perplexity)
            }
            
        except Exception as e:
            self.logger.error(f"Error in BART detection: {str(e)}")
            return {
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'model_used': model_name,
                'error': str(e)
            }
            
    def detect_with_perplexity(self, text: str) -> Dict:
        """
        Detect AI-generated text using perplexity-based approach with multiple models.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with perplexity-based detection results
        """
        # Models that work well with perplexity-based detection
        perplexity_models = ["bart-detector"]
        
        results = {}
        perplexities = []
        ai_probs = []
        
        for model_name in perplexity_models:
            try:
                result = self._detect_with_bart(text, model_name)
                results[model_name] = result
                
                if 'perplexity' in result and 'error' not in result:
                    perplexities.append(result['perplexity'])
                    ai_probs.append(result['ai_probability'])
            except Exception as e:
                self.logger.error(f"Error with {model_name} perplexity: {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'ai_probability': 0.5,
                    'human_probability': 0.5
                }
        
        # Calculate combined perplexity-based result
        if perplexities:
            avg_perplexity = np.mean(perplexities)
            avg_ai_prob = np.mean(ai_probs)
            
            # Extract linguistic features
            features = self.feature_extractor.extract_features(text)
            ai_patterns = self.feature_extractor.analyze_ai_patterns(text)
            
            # Adjust probability based on linguistic features (subtle adjustment)
            if features and ai_patterns:
                # Features that affect perplexity interpretation
                lex_diversity = features.get('lexical_diversity', 0.5)
                sent_uniformity = ai_patterns.get('sentence_length_uniformity', 0.5)
                repetition = ai_patterns.get('bigram_repetition', 0)
                
                # Calculate adjustment
                adjustment = (((1.0 - lex_diversity) * 0.3) + 
                             (sent_uniformity * 0.3) + 
                             (repetition * 0.4)) / 5.0  # Max effect of 0.2
                
                # Apply adjustment
                adjusted_ai_prob = min(0.95, max(0.05, avg_ai_prob + adjustment))
            else:
                adjusted_ai_prob = avg_ai_prob
            
            return {
                'perplexity_ai_probability': float(adjusted_ai_prob),
                'perplexity_human_probability': float(1.0 - adjusted_ai_prob),
                'average_perplexity': float(avg_perplexity),
                'prediction': 'AI-generated' if adjusted_ai_prob > 0.5 else 'Human-written',
                'individual_results': results,
                'models_used': perplexity_models
            }
        else:
            return {
                'error': "No valid perplexity results",
                'perplexity_ai_probability': 0.5,
                'perplexity_human_probability': 0.5,
                'prediction': 'Uncertain',
                'individual_results': results,
                'models_used': perplexity_models
            }