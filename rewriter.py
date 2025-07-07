import os
import json
import random
import string
import time
import re
from typing import List, Dict, Optional, Tuple
import logging
from functools import lru_cache

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer # Moved import to top

# spaCy imports with fallback
try:
    import spacy
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("spaCy/TextBlob not available. For advanced features, install with: pip install spacy textblob && python -m spacy download en_core_web_sm")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NLTK Data Management ---
def download_nltk_data():
    """Downloads required NLTK data packages if they are not found."""
    required_nltk_data = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('omw-1.4', 'corpora/omw-1.4')
        # FIX: Removed 'punkt_tab' as it is not a valid NLTK package.
    ]
    
    for data_package, path in required_nltk_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading required NLTK package: {data_package}...")
            try:
                nltk.download(data_package, quiet=True)
            except Exception as e:
                print(f"Error downloading {data_package}: {e}. Please try manual download.")

# Call the function at startup
download_nltk_data()


# --- Class Definitions ---

class LocalRefinementRepository:
    """Advanced local text refinement using spaCy, TextBlob, and NLTK"""
    
    def __init__(self):
        self.nlp = None
        self.advanced_features = ADVANCED_NLP_AVAILABLE
        
        if self.advanced_features:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for advanced text processing.")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.advanced_features = False
        
        if not self.advanced_features:
            logger.info("Using NLTK-based text refinement as spaCy is not available.")
    
    def refine_text(self, text: str) -> Tuple[str, Optional[str]]:
        """Refine text using the best available local NLP tools."""
        try:
            if self.advanced_features and self.nlp:
                return self._advanced_refinement(text), None
            else:
                return self._nltk_refinement(text), None
        except Exception as e:
            logger.error(f"Error in text refinement: {str(e)}")
            return self._basic_refinement(text), f"Refinement Error: {e}"

    def _advanced_refinement(self, text: str) -> str:
        """Advanced refinement using spaCy and TextBlob."""
        try:
            # Note: TextBlob's grammar correction can be slow and sometimes inaccurate.
            corrected_text = str(TextBlob(text).correct())
            
            doc = self.nlp(corrected_text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            refined_sentences = [self._improve_sentence_advanced(s) for s in sentences]
            return " ".join(refined_sentences)
        except Exception as e:
            logger.warning(f"Advanced refinement failed, falling back to NLTK: {str(e)}")
            return self._nltk_refinement(text)

    def _improve_sentence_advanced(self, sentence: str) -> str:
        """Improve sentence using advanced NLP with an academic tone."""
        if not sentence.strip():
            return sentence
        
        sentence = sentence.strip()
        sentence = sentence[0].upper() + sentence[1:]
        
        transition_words = {
            "Also": ["Furthermore", "Additionally", "Moreover"],
            "But": ["However", "Nevertheless", "Conversely"],
            "So": ["Therefore", "Consequently", "Thus"],
        }
        
        for original, alternatives in transition_words.items():
            if sentence.startswith(original + " ") and random.random() < 0.25:
                sentence = sentence.replace(original, random.choice(alternatives), 1)
                break
        return sentence

    def _nltk_refinement(self, text: str) -> str:
        """Refinement using NLTK."""
        try:
            sentences = sent_tokenize(text)
            refined_sentences = [self._improve_sentence_nltk(s) for s in sentences]
            return " ".join(refined_sentences)
        except Exception as e:
            logger.warning(f"NLTK refinement failed, using basic refinement: {str(e)}")
            return self._basic_refinement(text)

    def _improve_sentence_nltk(self, sentence: str) -> str:
        """Improve a single sentence using NLTK and WordNet with contextual checks."""
        if not sentence.strip():
            return sentence
            
        sentence = sentence.strip()
        sentence = sentence[0].upper() + sentence[1:]
        
        words = word_tokenize(sentence)
        improved_words = []
        
        for word in words:
            # Attempt synonym replacement with a low probability
            if word.isalpha() and len(word) > 4 and random.random() < 0.1:
                synonym = self._get_wordnet_synonym(word)
                # FIX: Use the dead code to make synonym replacement smarter
                if (synonym and synonym != word.lower() and
                    self._check_semantic_similarity(word, synonym) and
                    self._is_contextually_appropriate(word, synonym, sentence)):
                        
                    # Preserve original capitalization
                    if word.isupper():
                        synonym = synonym.upper()
                    elif word[0].isupper():
                        synonym = synonym.capitalize()
                    improved_words.append(synonym)
                else:
                    improved_words.append(word)
            else:
                improved_words.append(word)
        
        return TreebankWordDetokenizer().detokenize(improved_words)

    def _get_wordnet_synonym(self, word: str) -> Optional[str]:
        """Get a suitable synonym from WordNet."""
        synonyms = []
        try:
            for syn in wordnet.synsets(word.lower())[:2]:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word.lower() and len(synonym.split()) == 1 and synonym.isalpha():
                        synonyms.append(synonym)
            if synonyms:
                return random.choice(synonyms)
        except Exception:
            return None
        return None

    def _check_semantic_similarity(self, original_word: str, synonym: str) -> bool:
        """(Previously Unused) Check if a synonym is semantically close to the original word."""
        try:
            original_synsets = wordnet.synsets(original_word.lower())
            synonym_synsets = wordnet.synsets(synonym.lower())
            
            if not original_synsets or not synonym_synsets:
                return False

            # Use path similarity to check how close they are in the WordNet hierarchy
            similarity = original_synsets[0].path_similarity(synonym_synsets[0])
            return similarity is not None and similarity > 0.3
        except Exception:
            return False

    def _is_contextually_appropriate(self, original_word: str, synonym: str, sentence: str) -> bool:
        """(Previously Unused) Check if a synonym is appropriate in the given context."""
        # Avoid changing proper nouns
        if original_word[0].isupper():
            return False
        # Avoid changing words in common phrases
        common_phrases = ["in order to", "due to", "as well as", "in spite of"]
        for phrase in common_phrases:
            if original_word.lower() in phrase and phrase in sentence.lower():
                return False
        return True

    def _basic_refinement(self, text: str) -> str:
        """Basic text cleanup without external libraries."""
        text = re.sub(r'\s+', ' ', text.strip())
        # Capitalize the start of the first sentence
        if text:
            text = text[0].upper() + text[1:]
        return text

class LocalSynonymRepository:
    """Enhanced local synonym repository using NLTK WordNet."""
    def get_synonym(self, word: str) -> Tuple[str, Optional[str]]:
        """Get a synonym for a word using WordNet."""
        try:
            clean_word = word.lower().strip()
            if len(clean_word) < 3:
                return "", "Word too short"
            
            all_synonyms = set()
            for synset in wordnet.synsets(clean_word)[:3]:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if (synonym != clean_word and len(synonym.split()) == 1 and synonym.isalpha()):
                        all_synonyms.add(synonym)
            
            if not all_synonyms:
                return "", "No suitable synonyms found"
            
            return random.choice(list(all_synonyms)), None
        except Exception as e:
            return "", f"Error fetching synonym: {str(e)}"

class TextRewriteService:
    """Service for rewriting and humanizing text."""
    def __init__(self, refinement_repo: LocalRefinementRepository, synonym_repo: LocalSynonymRepository):
        self.refinement_repo = refinement_repo
        self.synonym_repo = synonym_repo
        self.filler_sentences = self._load_fillers()
        random.seed(time.time())

    def rewrite_text_with_modifications(self, text: str) -> Tuple[str, Optional[str]]:
        """Rewriting with modifications to make text more human-like."""
        base_result, err = self.refinement_repo.refine_text(text)
        if err: return text, err

        sentences = self._split_sentences(base_result)
        transformed = []
        
        for sentence in sentences:
            if random.random() < 0.7: sentence = self._vary_sentence_structure(sentence)
            if random.random() < 0.6: sentence = self._replace_synonyms(sentence)
            if random.random() < 0.5: sentence = self._add_natural_noise(sentence)
            transformed.append(sentence)

        return " ".join(transformed), None
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with a fallback."""
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _vary_sentence_structure(self, sentence: str) -> str:
        """Vary sentence structure by adding transitions or expanding contractions."""
        if len(sentence.split()) < 4:
            return sentence
        
        # Randomly choose a transformation
        if random.random() < 0.5:
            return self._add_transition_word(sentence)
        else:
            return self._convert_contractions(sentence)
            
    def _add_transition_word(self, sentence: str) -> str:
        """Add academic transition words to sentences."""
        transitions = [
            "Furthermore, ", "Additionally, ", "Moreover, ", "Notably, ",
            "Significantly, ", "Importantly, ", "Specifically, ", "Indeed, ",
            "Particularly, ", "Evidently, ", "Consequently, ", "Subsequently, ", # FIX: Removed syntax error (blank line)
            "Interestingly, ", "Remarkably, ", "Essentially, ", "Ultimately, ",
            "Clearly, ", "Obviously, ", "Undoubtedly, ", "Certainly, "
        ]
        
        if random.random() < 0.3:
            return random.choice(transitions) + sentence[0].lower() + sentence[1:]
        return sentence

    def _convert_contractions(self, sentence: str) -> str:
        """Expand common contractions for a more formal tone."""
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "it's": "it is", "that's": "that is"
        }
        for contraction, expansion in contractions.items():
            sentence = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, sentence, flags=re.IGNORECASE)
        return sentence

    def _replace_synonyms(self, sentence: str) -> str:
        """Replace words with synonyms."""
        words = sentence.split()
        max_modifications = max(1, len(words) // 5)
        modifications = 0
        
        for i, word in enumerate(words):
            if modifications >= max_modifications: break
            
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if len(clean_word) > 3 and not self._is_common_word(clean_word) and random.random() < 0.25:
                synonym, err = self.synonym_repo.get_synonym(clean_word)
                if not err and synonym:
                    words[i] = self._preserve_word_format(word, synonym)
                    modifications += 1
        return " ".join(words)

    def _preserve_word_format(self, original: str, replacement: str) -> str:
        """Preserve capitalization and punctuation of the original word."""
        if original.isupper(): return replacement.upper()
        if original.istitle(): return replacement.title()
        
        # Handle punctuation
        prefix = ''.join(c for c in original if not c.isalnum())
        suffix = ''.join(c for c in original if not c.isalnum())
        return prefix + replacement + suffix

    def _add_natural_noise(self, sentence: str) -> str:
        """Replace common words with more descriptive alternatives."""
        replacements = {
             " and ": [" as well as ", " along with "], " but ": [" however, ", " nevertheless, "],
             " use ": [" utilize ", " employ "], " show ": [" demonstrate ", " illustrate "],
             " help ": [" facilitate ", " assist "], " get ": [" obtain ", " acquire "],
             " very ": [" significantly ", " considerably "], " good ": [" effective ", " beneficial "],
             " new ": [" novel ", " innovative ", " contemporary "]
        }
        for old, new_options in replacements.items():
            if old in sentence and random.random() < 0.4:
                sentence = sentence.replace(old, random.choice(new_options), 1)
                break # Only one replacement per sentence to avoid over-stuffing
        return sentence

    def _is_common_word(self, word: str) -> bool:
        """Check if a word is too common or a reserved academic term."""
        common_words = {
            "the", "and", "that", "this", "with", "have", "will", "been", "from",
            "research", "study", "analysis", "data", "method", "result",
            "conclusion", "evidence", "theory", "findings", "literature",
            "significant", "demonstrate", "indicate", "suggest", "examine"
        }
        return word.lower() in common_words

    def _load_fillers(self) -> List[str]:
        """Load academic-appropriate filler sentences."""
        return [
            "This analysis provides valuable insights.",
            "These considerations merit further attention.",
            "The findings contribute to the existing body of knowledge.",
        ]

# --- Service Instantiation (Memoized for Efficiency) ---
# FIX: Use @lru_cache to create a single, shared instance of each service/repository.
# This prevents reloading models and data on every function call.

@lru_cache(maxsize=None)
def get_refinement_repository() -> LocalRefinementRepository:
    return LocalRefinementRepository()

@lru_cache(maxsize=None)
def get_synonym_repository() -> LocalSynonymRepository:
    return LocalSynonymRepository()

@lru_cache(maxsize=None)
def get_rewrite_service() -> TextRewriteService:
    return TextRewriteService(get_refinement_repository(), get_synonym_repository())

# --- Public API Functions ---

def rewrite_text(text: str, enhanced: bool = True) -> Tuple[str, Optional[str]]:
    """
    Main function to rewrite text.
    
    Args:
        text: Input text to rewrite.
        enhanced: Whether to use enhanced modifications for humanization.
        
    Returns:
        A tuple of (rewritten_text, error_message).
    """
    try:
        service = get_rewrite_service()
        if enhanced:
            return service.rewrite_text_with_modifications(text)
        else:
            # For non-enhanced, just use the base refinement
            return get_refinement_repository().refine_text(text)
    except Exception as e:
        logger.error(f"Error in rewrite_text: {str(e)}")
        return text, f"Rewrite error: {str(e)}"

def rewrite_text_extreme(text: str) -> Tuple[str, Optional[str]]:
    """
    Applies multiple rewriting passes for maximum textual transformation.
    Note: Multiple passes can sometimes degrade quality. Use with caution.
    """
    try:
        service = get_rewrite_service()
        result, err = text, None
        
        # Apply 3 passes for extreme transformation
        for _ in range(3):
            if err: return result, err # Stop if an error occurs
            result, err = service.rewrite_text_with_modifications(result)
            
        return result, None
    except Exception as e:
        logger.error(f"Error in rewrite_text_extreme: {str(e)}")
        return text, f"Extreme rewrite error: {str(e)}"

def get_synonym(word: str) -> Tuple[str, Optional[str]]:
    """Gets a synonym for a single word."""
    return get_synonym_repository().get_synonym(word)

def refine_text(text: str) -> Tuple[str, Optional[str]]:
    """Refines text using grammar correction and basic NLP improvements."""
    return get_refinement_repository().refine_text(text)