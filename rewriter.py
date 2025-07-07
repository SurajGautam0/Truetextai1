import os
import json
import random
import string
import time
import re
from typing import List, Dict, Optional, Tuple
import logging

# NLTK is a powerful library for natural language processing.
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet

# spaCy offers more advanced, context-aware NLP features.
try:
    import spacy
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("spaCy/TextBlob not available. For better results, install with: pip install spacy textblob && python -m spacy download en_core_web_sm")

# --- NLTK Data Download ---
def download_nltk_data():
    """
    Downloads necessary NLTK data packages if they are not already present.
    Handles potential download errors gracefully.
    """
    required_nltk_data = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('omw-1.4', 'corpora/omw-1.4')
    ]
    
    for package_id, path in required_nltk_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK data package '{package_id}' not found. Downloading...")
            try:
                nltk.download(package_id, quiet=True)
                print(f"Successfully downloaded '{package_id}'.")
            except Exception as e:
                print(f"Error downloading '{package_id}': {e}. The script might not function optimally.")

# Automatically download required data on startup.
download_nltk_data()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Classes ---

class LocalRefinementRepository:
    """
    Handles initial text cleaning and refinement using the best available tools (spaCy or NLTK).
    This class focuses on grammar, punctuation, and basic structural improvements.
    """
    
    def __init__(self):
        """Initializes the repository, loading spaCy if available."""
        self.nlp = None
        self.advanced_features = ADVANCED_NLP_AVAILABLE
        
        if self.advanced_features:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for advanced text processing.")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Falling back to NLTK. For better performance, run: python -m spacy download en_core_web_sm")
                self.advanced_features = False
        
        if not self.advanced_features:
            logger.info("Using NLTK-based text refinement as spaCy is not available.")
    
    def refine_text(self, text: str) -> str:
        """
        Refines text using the most powerful available local NLP tools.
        
        Args:
            text: The input text to refine.
            
        Returns:
            The refined text.
        """
        try:
            if self.advanced_features and self.nlp:
                return self._advanced_refinement(text)
            else:
                return self._nltk_refinement(text)
        except Exception as e:
            logger.error(f"Error during text refinement: {e}", exc_info=True)
            return self._basic_refinement(text) # Fallback to basic regex cleaning

    def _advanced_refinement(self, text: str) -> str:
        """Refines text using spaCy for robust sentence detection and structure."""
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            refined_sentences = [self._improve_sentence_structure(s) for s in sentences]
            return " ".join(refined_sentences)
        except Exception as e:
            logger.warning(f"Advanced refinement failed, falling back to NLTK: {e}")
            return self._nltk_refinement(text)

    def _nltk_refinement(self, text: str) -> str:
        """Refines text using NLTK for sentence tokenization."""
        try:
            sentences = sent_tokenize(text)
            refined_sentences = [self._improve_sentence_structure(s) for s in sentences]
            return " ".join(refined_sentences)
        except Exception as e:
            logger.warning(f"NLTK refinement failed, using basic refinement: {e}")
            return self._basic_refinement(text)

    def _improve_sentence_structure(self, sentence: str) -> str:
        """Ensures basic sentence structure like capitalization."""
        sentence = sentence.strip()
        if not sentence:
            return ""
        # Capitalize the first letter of the sentence.
        return sentence[0].upper() + sentence[1:]

    def _basic_refinement(self, text: str) -> str:
        """A fallback method that uses regular expressions for basic text cleanup."""
        text = re.sub(r'\s+', ' ', text).strip()
        # Ensure space after sentence-ending punctuation.
        text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
        # Capitalize the start of the text.
        if text:
            text = text[0].upper() + text[1:]
        return text


class LocalSynonymRepository:
    """
    Provides context-aware synonyms using NLTK's WordNet.
    Focuses on finding synonyms that are appropriate for the original word's context.
    """

    def get_synonym(self, word: str, pos_tag: Optional[str] = None) -> Optional[str]:
        """
        Finds a suitable synonym for a word, optionally filtered by part-of-speech.
        
        Args:
            word: The word to find a synonym for.
            pos_tag: The NLTK part-of-speech tag (e.g., 'v' for verb, 'n' for noun).
            
        Returns:
            A suitable synonym string or None if none is found.
        """
        clean_word = word.lower().strip()
        if len(clean_word) < 4:
            return None

        wordnet_pos = self._get_wordnet_pos(pos_tag)

        synsets = wordnet.synsets(clean_word, pos=wordnet_pos)
        if not synsets:
            return None

        synonyms = []
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != clean_word and len(synonym.split()) == 1 and synonym.isalpha():
                    synonyms.append(synonym)
        
        if synonyms:
            word_len = len(clean_word)
            filtered_synonyms = [s for s in synonyms if abs(len(s) - word_len) <= 3]
            return random.choice(filtered_synonyms) if filtered_synonyms else random.choice(synonyms)
        
        return None

    def _get_wordnet_pos(self, treebank_tag: Optional[str]) -> Optional[str]:
        """Converts NLTK POS tags to WordNet's format (e.g., 'JJ' -> 'a')."""
        if not treebank_tag: return None
        if treebank_tag.startswith('J'): return wordnet.ADJ
        if treebank_tag.startswith('V'): return wordnet.VERB
        if treebank_tag.startswith('N'): return wordnet.NOUN
        if treebank_tag.startswith('R'): return wordnet.ADV
        return None


class TextRewriteService:
    """
    Orchestrates the text rewriting process with different strategies.
    Contains a specialized mode for academic writing with adjustable strength.
    """
    
    def __init__(self):
        self.refinement_repo = LocalRefinementRepository()
        self.synonym_repo = LocalSynonymRepository()
        self.detokenizer = TreebankWordDetokenizer()
        logger.info("TextRewriteService initialized for academic rewriting.")

    def rewrite_for_academic(self, text: str, strength: str = 'default') -> str:
        """
        Rewrites text using a strategy specifically tailored for academic papers.
        
        Args:
            text: The input text.
            strength: The intensity of rewriting ('default' or 'strong').
            
        Returns:
            The rewritten text with an academic focus.
        """
        try:
            is_strong = strength == 'strong'
            
            refined_text = self.refinement_repo.refine_text(text)
            sentences = sent_tokenize(refined_text)
            
            transformed_sentences = []
            for sentence in sentences:
                # Apply transformations based on strength
                sentence = self._apply_academic_transformations(sentence, is_strong)
                transformed_sentences.append(sentence)

            if len(transformed_sentences) > 2:
                transformed_sentences = self._vary_sentence_complexity(transformed_sentences, is_strong)

            return " ".join(s for s in transformed_sentences if s)

        except Exception as e:
            logger.error(f"Error in academic rewriting: {e}", exc_info=True)
            return text

    def _apply_academic_transformations(self, sentence: str, is_strong: bool) -> str:
        """Applies a chain of academic transformations to a single sentence."""
        
        # Determine probability based on strength
        prob_structure = 0.9 if is_strong else 0.6
        prob_synonym = 0.9 if is_strong else 0.7
        prob_phrasing = 0.7 if is_strong else 0.5
        prob_paraphrase = 0.5 if is_strong else 0.0 # Only for strong

        if random.random() < prob_structure:
            sentence = self._vary_sentence_structure_academic(sentence, is_strong)
        
        if random.random() < prob_synonym:
            sentence = self._replace_synonyms_academic(sentence, is_strong)
        
        if random.random() < prob_phrasing:
            sentence = self._apply_formal_phrasing(sentence)

        if is_strong and random.random() < prob_paraphrase:
             sentence = self._paraphrase_sentence(sentence)
        
        return sentence

    def _vary_sentence_structure_academic(self, sentence: str, is_strong: bool) -> str:
        """Applies structural variations suitable for formal writing."""
        sentence = self._expand_contractions(sentence)

        transitions = [
            "Furthermore,", "Additionally,", "Moreover,", "Notably,",
            "Consequently,", "Therefore,", "Thus,", "Subsequently,"
        ]
        
        transition_prob = 0.25 if is_strong else 0.15
        if random.random() < transition_prob and len(sentence.split()) > 5:
            if not sentence.lower().startswith(tuple(t.lower() for t in transitions)):
                transition = random.choice(transitions)
                sentence = f"{transition} {sentence[0].lower()}{sentence[1:]}"
        
        return sentence

    def _expand_contractions(self, sentence: str) -> str:
        """Expands common contractions to their full form."""
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
            "it's": "it is", "that's": "that is", "there's": "there is",
            "you're": "you are", "we're": "we are", "they're": "they are"
        }
        for contraction, expansion in contractions.items():
            sentence = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, sentence, flags=re.IGNORECASE)
        return sentence

    def _replace_synonyms_academic(self, sentence: str, is_strong: bool) -> str:
        """Replaces words with context-aware synonyms."""
        try:
            words = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
        except Exception:
            return sentence

        modified_words = list(words)
        # Modify a higher percentage of words in 'strong' mode
        modification_ratio = 0.35 if is_strong else 0.20
        max_modifications = max(1, int(len(words) * modification_ratio))
        modifications = 0
        
        replacement_prob = 0.40 if is_strong else 0.25

        for i, (word, tag) in enumerate(pos_tags):
            if modifications >= max_modifications: break
            
            if tag.startswith(('J', 'V', 'N', 'R')) and len(word) > 3 and not self._is_common_academic_word(word):
                if random.random() < replacement_prob:
                    synonym = self.synonym_repo.get_synonym(word, pos_tag=tag)
                    if synonym:
                        modified_words[i] = self._preserve_word_format(words[i], synonym)
                        modifications += 1
        
        return self.detokenizer.detokenize(modified_words)

    def _apply_formal_phrasing(self, sentence: str) -> str:
        """Replaces simple words/phrases with more formal, academic alternatives."""
        replacements = {
            r'\b(use|uses)\b': "utilize", r'\b(show|shows)\b': "demonstrate",
            r'\b(help|helps)\b': "facilitate", r'\b(get|gets)\b': "obtain",
            r'\b(make|makes)\b': "establish", r'\b(think|thinks)\b': "consider",
            r'\bvery\b': "considerably", r'\b(big|large)\b': "substantial",
            r'\b(good)\b': "effective", r'\b(bad)\b': "detrimental",
            r'\b(a lot of)\b': "numerous",
        }
        for pattern, replacement in replacements.items():
            if re.search(pattern, sentence, re.IGNORECASE) and random.random() < 0.5:
                return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
        return sentence
        
    def _paraphrase_sentence(self, sentence: str) -> str:
        """(Strong mode only) Attempts to paraphrase by reordering clauses."""
        if ', ' in sentence:
            parts = sentence.split(', ', 1)
            # Check for a dependent clause that can be moved
            if len(parts) == 2 and len(parts[0].split()) > 3 and len(parts[1].split()) > 3:
                # Simple check for introductory phrases
                if parts[0].lower().startswith(('although', 'while', 'if', 'because', 'since')):
                    # Reorder: "Clause, main." -> "Main clause."
                    return f"{parts[1][0].upper()}{parts[1][1:]} {parts[0].lower()}"
        return sentence


    def _vary_sentence_complexity(self, sentences: List[str], is_strong: bool) -> List[str]:
        """Varies sentence length by combining short sentences or splitting long ones."""
        if len(sentences) < 2: return sentences
            
        output_sentences = []
        i = 0
        
        combine_prob = 0.3 if is_strong else 0.2
        split_prob = 0.4 if is_strong else 0.0 # Only split in strong mode

        while i < len(sentences):
            current_sentence = sentences[i]
            
            # Attempt to split long sentences in strong mode
            if is_strong and len(current_sentence.split()) > 25 and random.random() < split_prob:
                split_point = self._find_split_point(current_sentence)
                if split_point:
                    part1 = current_sentence[:split_point].strip() + '.'
                    part2 = current_sentence[split_point:].strip()
                    part2 = part2[0].upper() + part2[1:] # Capitalize new sentence
                    output_sentences.append(part1)
                    output_sentences.append(part2)
                    i += 1
                    continue

            # Attempt to combine with the next sentence
            if i + 1 < len(sentences):
                next_sentence = sentences[i+1]
                if len(current_sentence.split()) < 12 and len(next_sentence.split()) < 12 and random.random() < combine_prob:
                    conjunctions = ["and", "while", "whereas", "in addition to which"]
                    combined = f"{current_sentence.rstrip('.')} {random.choice(conjunctions)} {next_sentence[0].lower()}{next_sentence[1:]}"
                    output_sentences.append(combined)
                    i += 2
                    continue
            
            output_sentences.append(current_sentence)
            i += 1
        return output_sentences

    def _find_split_point(self, sentence: str) -> Optional[int]:
        """Finds a suitable point (like after a conjunction) to split a long sentence."""
        words = sentence.split()
        mid_point = len(words) // 2
        
        # Look for conjunctions or commas around the middle of the sentence
        for i in range(mid_point, len(words) - 3):
            if words[i].endswith(',') or words[i].lower() in ['and', 'but', 'however', 'therefore']:
                # Return the character index
                return len(" ".join(words[:i+1])) + 1
        return None

    def _preserve_word_format(self, original: str, replacement: str) -> str:
        """Preserves capitalization of the original word."""
        if original.isupper(): return replacement.upper()
        if original.istitle(): return replacement.title()
        return replacement

    def _is_common_academic_word(self, word: str) -> bool:
        """Checks if a word is a common term that should not be replaced."""
        common_words = {
            "the", "and", "that", "this", "with", "for", "from", "was", "are", "is",
            "research", "study", "analysis", "data", "method", "results",
            "conclusion", "evidence", "theory", "model", "figure", "table",
            "system", "process", "approach", "framework", "concept",
            "significant", "demonstrate", "indicate", "suggest", "establish"
        }
        return word.lower() in common_words

# --- Public API Functions ---

def rewrite_for_academic(text: str, strength: str = 'default') -> str:
    """
    High-level function to rewrite text for academic purposes.
    
    Args:
        text: The input text to rewrite.
        strength: The intensity of rewriting ('default' or 'strong').
        
    Returns:
        The rewritten text, tailored for an academic context.
    """
    if not text or not isinstance(text, str):
        logger.warning("Input text is invalid. Returning an empty string.")
        return ""
    
    if strength not in ['default', 'strong']:
        logger.warning(f"Invalid strength '{strength}'. Using 'default'.")
        strength = 'default'

    start_time = time.time()
    logger.info(f"Starting academic rewrite with strength: {strength}...")
    
    service = TextRewriteService()
    rewritten_text = service.rewrite_for_academic(text, strength)
    
    duration = time.time() - start_time
    logger.info(f"Academic rewrite completed in {duration:.2f} seconds.")
    
    return rewritten_text

# --- Example Usage ---
if __name__ == '__main__':
    original_text = """
    The study showed that the new method is very good. We think it's a big improvement. 
    Researchers get better results with this approach. But, it's not perfect and has some problems. 
    We can't ignore these issues. We'll work on them later. Although the system is complex, it offers many benefits.
    """
    
    print("--- Original Text ---")
    print(original_text)
    print("\n" + "="*50 + "\n")
    
    print("--- Academic Rewrite (Default Strength) ---")
    default_version = rewrite_for_academic(original_text, strength='default')
    print(default_version)
    print("\n" + "="*50 + "\n")

    print("--- Academic Rewrite (Strong Strength) ---")
    strong_version = rewrite_for_academic(original_text, strength='strong')
    print(strong_version)
