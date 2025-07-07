import os
import json
import random
import string
import time
import re
from typing import List, Dict, Optional, Tuple
import logging

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer

# spaCy imports with fallback
try:
    import spacy
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("spaCy/TextBlob not available. Install with: pip install spacy textblob")

class TextRewriter:
    """Enhanced text rewriting service with optimized parameters to reduce unnecessary words"""
    
    def __init__(self):
        self._initialize_nltk()
        self._initialize_services()
        self._load_resources()
        
    def _initialize_nltk(self):
        """Initialize NLTK with required data"""
        required_nltk_data = [
            ('punkt', 'tokenizers/punkt'),
            ('wordnet', 'corpora/wordnet'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('omw-1.4', 'corpora/omw-1.4')
        ]
        
        for data_package, path in required_nltk_data:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(data_package, quiet=True)
                except Exception as e:
                    logging.warning(f"Error downloading {data_package}: {e}")

    def _initialize_services(self):
        """Initialize NLP services"""
        self.nlp = None
        self.advanced_features = ADVANCED_NLP_AVAILABLE
        
        if self.advanced_features:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("Loaded spaCy model for advanced text processing")
            except OSError:
                logging.warning("spaCy model not found. Using NLTK-based processing")
                self.advanced_features = False
        
        self.stemmer = SnowballStemmer('english')
        self.detokenizer = TreebankWordDetokenizer()
    
    def _load_resources(self):
        """Load transition words and other resources"""
        self.transition_words = {
            "Also": ["Furthermore", "Additionally", "Moreover"],
            "But": ["However", "Nevertheless", "Nonetheless"],
            "So": ["Therefore", "Consequently", "Thus"],
            "And": ["Furthermore", "Additionally"],
            "First": ["Initially", "Primarily"],
            "Finally": ["In conclusion", "Ultimately"]
        }
        
        self.common_phrases = [
            "in fact", "as well", "as if", "as though", "in order to",
            "due to", "because of", "in spite of", "in addition to",
            "such as", "for example", "for instance", "in other words"
        ]
        
        self.filler_sentences = [
            "This analysis provides valuable insights.",
            "These findings merit further consideration.",
            "The implications are noteworthy.",
            "This approach yields meaningful results.",
            "The research demonstrates important aspects."
        ]
    
    def rewrite_text(self, text: str, mode: str = 'balanced') -> Tuple[str, Optional[str]]:
        """
        Rewrite text with different modes of modification intensity
        
        Args:
            text: Input text to rewrite
            mode: 'conservative', 'balanced', or 'aggressive' (default: 'balanced')
            
        Returns:
            Tuple of (rewritten_text, error_message)
        """
        try:
            # First apply basic refinement
            refined_text = self._basic_refinement(text)
            
            # Apply mode-specific transformations
            if mode == 'conservative':
                return self._conservative_rewrite(refined_text), None
            elif mode == 'balanced':
                return self._balanced_rewrite(refined_text), None
            elif mode == 'aggressive':
                return self._aggressive_rewrite(refined_text), None
            else:
                return refined_text, "Invalid mode specified"
                
        except Exception as e:
            logging.error(f"Error in text rewriting: {str(e)}")
            return text, f"Rewrite error: {str(e)}"
    
    def _conservative_rewrite(self, text: str) -> str:
        """Minimal modifications for light rewriting"""
        sentences = self._split_sentences(text)
        transformed = []
        
        for sentence in sentences:
            # Apply conservative transformations
            if random.random() < 0.3:
                sentence = self._vary_sentence_structure(sentence, max_transformations=1)
            if random.random() < 0.2:
                sentence = self._replace_keywords(sentence, max_replacements=1)
            
            transformed.append(sentence)
        
        return " ".join(transformed)
    
    def _balanced_rewrite(self, text: str) -> str:
        """Balanced modifications for natural rewriting"""
        sentences = self._split_sentences(text)
        transformed = []
        
        for sentence in sentences:
            # Apply balanced transformations
            if random.random() < 0.5:
                sentence = self._vary_sentence_structure(sentence, max_transformations=2)
            if random.random() < 0.4:
                sentence = self._replace_keywords(sentence, max_replacements=2)
            if random.random() < 0.3:
                sentence = self._improve_academic_tone(sentence)
            
            transformed.append(sentence)
        
        # Occasionally add contextual filler
        if len(transformed) > 1 and random.random() < 0.3:
            filler = self._get_contextual_filler(transformed)
            if filler:
                insert_pos = random.randint(1, len(transformed))
                transformed.insert(insert_pos, filler)
        
        return " ".join(transformed)
    
    def _aggressive_rewrite(self, text: str) -> str:
        """More extensive modifications for heavy rewriting"""
        sentences = self._split_sentences(text)
        transformed = []
        
        for sentence in sentences:
            # Apply aggressive transformations
            if random.random() < 0.7:
                sentence = self._vary_sentence_structure(sentence, max_transformations=3)
            if random.random() < 0.6:
                sentence = self._replace_keywords(sentence, max_replacements=3)
            if random.random() < 0.5:
                sentence = self._improve_academic_tone(sentence)
            if random.random() < 0.4:
                sentence = self._add_conversational_elements(sentence)
            
            transformed.append(sentence)
        
        # More likely to add contextual filler
        if len(transformed) > 1 and random.random() < 0.5:
            filler = self._get_contextual_filler(transformed)
            if filler:
                insert_pos = random.randint(1, len(transformed))
                transformed.insert(insert_pos, filler)
        
        return " ".join(transformed)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def _vary_sentence_structure(self, sentence: str, max_transformations: int = 2) -> str:
        """Vary sentence structure with limited transformations"""
        if len(sentence.split()) < 4:
            return sentence
        
        transformations = [
            self._add_transition_word,
            self._rearrange_clauses,
            self._convert_contractions,
        ]
        
        # Apply up to max_transformations
        transformations_applied = 0
        while transformations_applied < max_transformations and transformations:
            transformation = random.choice(transformations)
            new_sentence = transformation(sentence)
            if new_sentence != sentence:
                sentence = new_sentence
                transformations_applied += 1
                # Remove used transformation to avoid duplicates
                transformations.remove(transformation)
        
        return sentence
    
    def _add_transition_word(self, sentence: str) -> str:
        """Add academic transition words to sentences"""
        if not sentence[0].isupper():
            return sentence
        
        # Reduced probability from original
        if random.random() < 0.3:
            transitions = [
                "Furthermore, ", "Additionally, ", "Moreover, ",
                "However, ", "Nevertheless, ", "Therefore, ",
                "Consequently, ", "Interestingly, ", "Notably, "
            ]
            
            transition = random.choice(transitions)
            
            # Handle quoted sentences
            if sentence.strip().startswith('"'):
                match = re.match(r'(\s*")', sentence)
                quote_char = match.group(1)
                rest_of_sentence = sentence[len(quote_char):]
                return f'{quote_char}{transition}{rest_of_sentence.lower()}'
            else:
                return transition + sentence[0].lower() + sentence[1:]
        
        return sentence
    
    def _rearrange_clauses(self, sentence: str) -> str:
        """Simple clause rearrangement with quality checks"""
        if ', ' in sentence and sentence.count(',') == 1:
            parts = sentence.split(', ', 1)
            if len(parts) == 2 and random.random() < 0.3:
                part1, part2 = parts
                # Only rearrange if it makes sense
                if (len(part1.split()) > 2 and len(part2.split()) > 2 and
                    not any(w in part1.lower() for w in ['although', 'while', 'because'])):
                    return f"{part2.capitalize()}, {part1.lower()}"
        
        return sentence
    
    def _convert_contractions(self, sentence: str) -> str:
        """Expand contractions for academic formality"""
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
            "it's": "it is", "that's": "that is", "there's": "there is",
            "what's": "what is", "you're": "you are", "we're": "we are",
            "they're": "they are"
        }
        
        # Lower probability than original
        if random.random() < 0.6:
            for contraction, expansion in contractions.items():
                if contraction in sentence.lower():
                    # Case-sensitive replacement
                    sentence = re.sub(re.escape(contraction), expansion, sentence, flags=re.IGNORECASE)
                    break
        
        return sentence
    
    def _replace_keywords(self, sentence: str, max_replacements: int = 2) -> str:
        """Replace keywords with synonyms more conservatively"""
        words = word_tokenize(sentence)
        modifications = 0
        
        for i, word in enumerate(words):
            if modifications >= max_replacements:
                break
                
            clean_word = re.sub(r'[^\w]', '', word).lower()
            
            # Skip if too short, common, or part of phrase
            if (len(clean_word) < 4 or 
                self._is_common_word(clean_word) or
                self._is_part_of_phrase(clean_word, sentence)):
                continue
            
            # Lower probability than original
            if random.random() < 0.3:
                synonym = self._get_best_synonym(clean_word, sentence)
                if synonym:
                    new_word = self._preserve_word_format(word, synonym)
                    words[i] = new_word
                    modifications += 1
        
        return self.detokenizer.detokenize(words)
    
    def _get_best_synonym(self, word: str, context: str) -> Optional[str]:
        """Get the most appropriate synonym for a word in context"""
        try:
            synsets = wordnet.synsets(word)
            if not synsets:
                return None
            
            # Collect all suitable synonyms
            synonyms = []
            for syn in synsets[:2]:  # Only check first 2 synsets
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if (synonym != word and 
                        len(synonym.split()) == 1 and
                        synonym.isalpha() and
                        abs(len(synonym) - len(word)) <= 3 and
                        not self._is_common_word(synonym)):
                        synonyms.append(synonym)
            
            if not synonyms:
                return None
            
            # Prefer synonyms that maintain similar meaning
            for syn in synonyms:
                if self._check_semantic_similarity(word, syn):
                    return syn
            
            return random.choice(synonyms)
            
        except Exception:
            return None
    
    def _check_semantic_similarity(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically similar"""
        try:
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)

            # Check path similarity
            max_sim = max(
                (s1.path_similarity(s2) or 0)
                for s1 in synsets1[:2]
                for s2 in synsets2[:2]
            )
            return max_sim > 0.4  # Higher threshold than original

        except Exception:
            return False
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is too common for replacement"""
        common_words = {
            "the", "and", "that", "this", "with", "have", "will", "been", 
            "from", "they", "know", "want", "good", "much", "some", "time",
            "very", "when", "come", "here", "just", "like", "long", "make",
            "many", "over", "such", "take", "than", "them", "well", "were",
            "work", "about", "could", "would", "there", "their", "which",
            "should", "think", "where", "through", "because", "between",
            "important", "different", "following", "around", "though",
            "without", "another", "example", "however", "therefore"
        }
        return word.lower() in common_words
    
    def _is_part_of_phrase(self, word: str, sentence: str) -> bool:
        """Check if word is part of a common phrase"""
        sentence_lower = sentence.lower()
        for phrase in self.common_phrases:
            if word in phrase and phrase in sentence_lower:
                return True
        return False
    
    def _preserve_word_format(self, original: str, replacement: str) -> str:
        """Preserve capitalization and punctuation of original word"""
        # Extract non-alphabetic characters
        prefix = ''.join(c for c in original if not c.isalpha())
        suffix = ''
        core_word = original[len(prefix):]
        
        # Find suffix
        if core_word:
            suffix_start = len(core_word.rstrip(string.ascii_letters))
            suffix = core_word[suffix_start:]
            core_word = core_word[:suffix_start]
        
        # Apply capitalization
        if core_word:
            if core_word[0].isupper():
                replacement = replacement.capitalize()
            elif core_word.isupper():
                replacement = replacement.upper()
        
        return prefix + replacement + suffix
    
    def _improve_academic_tone(self, sentence: str) -> str:
        """Improve academic tone with precise word replacements"""
        replacements = {
            " and ": [" as well as ", " along with "],
            " but ": [" however, ", " nevertheless, "],
            " because ": [" since ", " as "],
            " so ": [" therefore, ", " consequently, "],
            " also ": [" additionally, ", " moreover, "],
            " use ": [" utilize ", " employ "],
            " show ": [" demonstrate ", " illustrate "],
            " help ": [" facilitate ", " assist "],
            " get ": [" obtain ", " acquire "],
            " make ": [" create ", " produce "],
            " find ": [" discover ", " identify "],
            " very ": [" significantly ", " considerably "],
            " big ": [" substantial ", " significant "],
            " small ": [" minimal ", " limited "],
            " good ": [" effective ", " beneficial "],
            " bad ": [" detrimental ", " problematic "],
            " new ": [" novel ", " recent "],
            " old ": [" traditional ", " previous "],
            " many ": [" numerous ", " several "],
            " few ": [" limited ", " scarce "]
        }
        
        replacements_made = 0
        max_replacements = 2  # Fewer than original
        
        for old, new_options in replacements.items():
            if replacements_made >= max_replacements:
                break
                
            if old in sentence.lower() and random.random() < 0.3:  # Lower probability
                new_phrase = random.choice(new_options)
                sentence = re.sub(re.escape(old), new_phrase, sentence, count=1, flags=re.IGNORECASE)
                replacements_made += 1
        
        return sentence
    
    def _add_conversational_elements(self, sentence: str) -> str:
        """Add natural conversational elements sparingly"""
        if len(sentence.split()) < 5:
            return sentence
            
        # Reduced set of conversational elements
        elements = [
            ("You know, ", 0.1),  # phrase, probability
            ("Well, ", 0.1),
            ("Actually, ", 0.1),
            (" I mean, ", 0.08),
            (" basically, ", 0.08),
            (" essentially, ", 0.08)
        ]
        
        for phrase, prob in elements:
            if random.random() < prob:
                if phrase.endswith(", "):  # Starter phrase
                    if not sentence.startswith(phrase.strip()):
                        sentence = phrase + sentence[0].lower() + sentence[1:]
                else:  # Middle phrase
                    words = sentence.split()
                    if len(words) > 4:
                        insert_pos = random.randint(2, len(words) - 2)
                        words.insert(insert_pos, phrase.strip())
                        sentence = " ".join(words)
                break  # Only add one per sentence
        
        return sentence
    
    def _get_contextual_filler(self, sentences: List[str]) -> str:
        """Generate contextual filler that's actually relevant"""
        if not sentences:
            return ""
        
        # Extract meaningful keywords
        keywords = self._extract_keywords(" ".join(sentences))
        if not keywords:
            return ""
        
        templates = [
            "This highlights the importance of {keyword}.",
            "The aspect of {keyword} deserves attention.",
            "{keyword} plays a significant role here.",
            "Regarding {keyword}, several factors apply.",
            "The consideration of {keyword} is relevant."
        ]
        
        template = random.choice(templates)
        keyword = random.choice(keywords[:2])  # Use top keywords only
        return template.format(keyword=keyword.capitalize())
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        filtered = [w for w in words if not self._is_common_word(w)]
        return sorted(set(filtered), key=lambda x: -text.lower().count(x))[:3]  # Top 3
    
    def _basic_refinement(self, text: str) -> str:
        """Basic text cleanup and formatting"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common punctuation issues
        fixes = [
            (r'[\s\r\n]+([,.!?;:])', r'\1'),  # Remove space before punctuation
            (r'([.!?])\s*([a-z])', r'\1 \2'),  # Ensure space after sentence endings
            (r'\bi\b', 'I'),  # Capitalize standalone 'i'
            (r'\s+([)\]}])', r'\1'),  # Remove space before closing brackets
            (r'([(\[{])\s+', r'\1'),  # Remove space after opening brackets
            (r'\s{2,}', ' '),  # Replace multiple spaces
        ]
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)
        
        # Capitalize sentences
        sentences = re.split(r'([.!?]+)', text)
        for i in range(0, len(sentences), 2):
            if sentences[i]:
                sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        
        return ''.join(sentences)

# Public interface functions
def rewrite_text(text: str, mode: str = 'balanced') -> Tuple[str, Optional[str]]:
    """
    Rewrite text with specified intensity
    
    Args:
        text: Input text to rewrite
        mode: 'conservative', 'balanced', or 'aggressive'
        
    Returns:
        Tuple of (rewritten_text, error_message)
    """
    rewriter = TextRewriter()
    return rewriter.rewrite_text(text, mode)

def refine_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Basic text refinement without extensive rewriting
    
    Args:
        text: Input text to refine
        
    Returns:
        Tuple of (refined_text, error_message)
    """
    rewriter = TextRewriter()
    return rewriter._basic_refinement(text), None

def get_synonym(word: str) -> Tuple[str, Optional[str]]:
    """
    Get synonym for a word with context checks
    
    Args:
        word: Word to find synonym for
        
    Returns:
        Tuple of (synonym, error_message)
    """
    rewriter = TextRewriter()
    synonym = rewriter._get_best_synonym(word.lower(), "")
    return synonym if synonym else "", "No suitable synonym found"