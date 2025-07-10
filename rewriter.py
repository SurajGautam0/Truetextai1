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

# spaCy imports with fallback
try:
    import spacy
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("spaCy/TextBlob not available. Install with: pip install spacy textblob")

# Download required NLTK data with better error handling
def download_nltk_data():
    required_nltk_data = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'), 
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('omw-1.4', 'corpora/omw-1.4')
    ]
    
    for data_package, path in required_nltk_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading {data_package}...")
            try:
                nltk.download(data_package, quiet=True)
            except Exception as e:
                print(f"Error downloading {data_package}: {e}")
                # Try alternative approach
                if data_package == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
        except Exception as e:
            print(f"Error with {data_package}: {e}")
            try:
                nltk.download(data_package, quiet=True)
            except:
                pass

# Call the function
download_nltk_data()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize stemmer
stemmer = SnowballStemmer('english')

class LocalRefinementRepository:
    """Advanced local text refinement using spaCy, TextBlob, and NLTK"""
    
    def __init__(self):
        self.nlp = None
        self.advanced_features = ADVANCED_NLP_AVAILABLE
        
        if self.advanced_features:
            try:
                # Try to load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for advanced text processing")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.advanced_features = False
        
        if not self.advanced_features:
            logger.info("Using NLTK-based text refinement")
    
    def refine_text(self, text: str) -> Tuple[str, Optional[str]]:
        """Refine text using best available local NLP tools"""
        try:
            if self.advanced_features and self.nlp:
                return self._advanced_refinement(text), None
            else:
                return self._nltk_refinement(text), None
                
        except Exception as e:
            logger.error(f"Error in text refinement: {str(e)}")
            return self._basic_refinement(text), None
    
    def _advanced_refinement(self, text: str) -> str:
        """Advanced refinement using spaCy and TextBlob"""
        try:
            # Grammar correction with TextBlob
            blob = TextBlob(text)
            corrected_text = str(blob.correct())
            
            # Process with spaCy
            doc = self.nlp(corrected_text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            refined_sentences = []
            for sentence in sentences:
                refined = self._improve_sentence_advanced(sentence)
                refined_sentences.append(refined)
            
            return " ".join(refined_sentences)
            
        except Exception as e:
            logger.warning(f"Advanced refinement failed, falling back to NLTK: {str(e)}")
            return self._nltk_refinement(text)
    
    def _improve_sentence_advanced(self, sentence: str) -> str:
        """Improve sentence using advanced NLP with academic tone"""
        if not sentence.strip():
            return sentence
        
        # Ensure proper capitalization
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        # Academic-appropriate transition words
        transition_words = {
            "Also": ["Furthermore", "Additionally", "Moreover", "In addition"],
            "But": ["However", "Nevertheless", "Nonetheless", "Conversely"],
            "So": ["Therefore", "Consequently", "Thus", "Hence"],
            "And": ["Furthermore", "Additionally", "Moreover"],
            "First": ["Initially", "Primarily", "To begin with"],
            "Finally": ["In conclusion", "Ultimately", "Lastly"]
        }
        
        for original, alternatives in transition_words.items():
            if sentence.startswith(original + " ") and random.random() < 0.25:
                replacement = random.choice(alternatives)
                sentence = sentence.replace(original, replacement, 1)
                break
        
        return sentence
    
    def _nltk_refinement(self, text: str) -> str:
        """Refinement using NLTK"""
        try:
            sentences = sent_tokenize(text)
            refined_sentences = []
            
            for sentence in sentences:
                refined = self._improve_sentence_nltk(sentence)
                refined_sentences.append(refined)
            
            return " ".join(refined_sentences)
            
        except Exception as e:
            logger.warning(f"NLTK refinement failed, using basic refinement: {str(e)}")
            return self._basic_refinement(text)
    
    def _improve_sentence_nltk(self, sentence: str) -> str:
        """Improve sentence using NLTK"""
        if not sentence.strip():
            return sentence
        
        # Basic improvements
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        # Word-level improvements using WordNet
        words = word_tokenize(sentence)
        improved_words = []
        
        for word in words:
            if word.isalpha() and len(word) > 4 and random.random() < 0.1:
                synonym = self._get_wordnet_synonym(word)
                if synonym and synonym != word.lower():
                    # Preserve original capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    improved_words.append(synonym)
                else:
                    improved_words.append(word)
            else:
                improved_words.append(word)
        
        # Reconstruct sentence with proper spacing using NLTK's detokenizer approach
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        detokenizer = TreebankWordDetokenizer()
        return detokenizer.detokenize(improved_words)
    
    def _get_wordnet_synonym(self, word: str) -> Optional[str]:
        """Get synonym from WordNet"""
        try:
            synsets = wordnet.synsets(word.lower())
            if synsets:
                synonyms = []
                for syn in synsets[:2]:  # Check first 2 synsets
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if (synonym != word.lower() and 
                            len(synonym.split()) == 1 and  # Single word only
                            synonym.isalpha()):
                            synonyms.append(synonym)
                
                if synonyms:
                    return random.choice(synonyms)
            return None
        except Exception:
            return None
    
    def _check_semantic_similarity(self, original_word: str, synonym: str) -> bool:
        """Check if synonym is semantically similar to original word"""
        try:
            original_synsets = wordnet.synsets(original_word.lower())
            synonym_synsets = wordnet.synsets(synonym.lower())
            
            if not original_synsets or not synonym_synsets:
                return False
            
            # Check if they share any synsets (exact match)
            for orig_synset in original_synsets:
                for syn_synset in synonym_synsets:
                    if orig_synset == syn_synset:
                        return True
            
            # Check path similarity (how close they are in the WordNet hierarchy)
            max_similarity = 0
            for orig_synset in original_synsets[:2]:  # Check first 2 synsets
                for syn_synset in synonym_synsets[:2]:
                    try:
                        similarity = orig_synset.path_similarity(syn_synset)
                        if similarity and similarity > max_similarity:
                            max_similarity = similarity
                    except:
                        continue
            
            # Return True if similarity is above threshold
            return max_similarity > 0.3
            
        except Exception:
            return False
    
    def _is_contextually_appropriate(self, original_word: str, synonym: str, sentence: str) -> bool:
        """Check if synonym is contextually appropriate in the sentence"""
        try:
            # Skip if words are too different in length (might change meaning)
            if abs(len(original_word) - len(synonym)) > 4:
                return False
            
            # Skip if original word is part of a common phrase
            common_phrases = [
                "in fact", "as well", "as if", "as though", "in order to",
                "due to", "because of", "in spite of", "in addition to",
                "such as", "for example", "for instance", "in other words"
            ]
            
            sentence_lower = sentence.lower()
            for phrase in common_phrases:
                if original_word.lower() in phrase and phrase in sentence_lower:
                    return False
            
            # Check if word is part of a technical term or proper noun
            if original_word[0].isupper() and len(original_word) > 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _basic_refinement(self, text: str) -> str:
        """Basic text refinement without external libraries"""
        # Clean up text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues - MORE COMPREHENSIVE
        replacements = {
            r'[\s\r\n]+([,.!?;:])': r'\1',  # Remove space before punctuation
            r'([.!?])\s*([a-z])': r'\1 \2',  # Ensure space after sentence endings
            r'\bi\b': 'I',  # Capitalize standalone 'i'
            r'\s+([)\]}])': r'\1',  # Remove space before closing brackets
            r'([(\[{])\s+': r'\1',  # Remove space after opening brackets
            r'\s{2,}': ' ',  # Replace multiple spaces with single space
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Ensure sentences start with capital letters
        sentences = re.split(r'([.!?]+)', text)
        result = []
        
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part.strip():  # Sentence content
                part = part.strip()
                if part:
                    part = part[0].upper() + part[1:]
                result.append(part)
            else:  # Punctuation
                result.append(part)
        
        # Join and apply final cleanup passes
        final_text = ''.join(result)
        
        # Multiple cleanup passes to ensure all spacing issues are fixed
        final_text = re.sub(r'\s+([,.!?;:])', r'\1', final_text)  # Remove spaces before punctuation
        # Apply multiple passes to ensure no spaces are left before punctuation
        for _ in range(2):  # Multiple passes to catch nested cases
            final_text = re.sub(r'\s+([,.!?;:])', r'\1', final_text)
        final_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', final_text)  # Ensure space after sentence endings
        final_text = re.sub(r'\s{2,}', ' ', final_text)  # Replace multiple spaces with single space
        final_text = re.sub(r'\s+$', '', final_text)  # Remove trailing spaces
        final_text = re.sub(r'^\s+', '', final_text)  # Remove leading spaces
        
        return final_text

class LocalSynonymRepository:
    """Enhanced local synonym repository using NLTK WordNet"""
    
    def __init__(self):
        # Ensure WordNet is available
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def get_synonym(self, word: str) -> Tuple[str, Optional[str]]:
        """Get synonym for a word using WordNet"""
        try:
            clean_word = word.lower().strip()
            if len(clean_word) < 3:
                return "", "Word too short for synonym replacement"
            
            synsets = wordnet.synsets(clean_word)
            if not synsets:
                return "", "No synonyms found for the word"
            
            # Collect synonyms from multiple synsets
            all_synonyms = []
            for synset in synsets[:3]:  # Check first 3 synsets
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if (synonym != clean_word and 
                        len(synonym.split()) == 1 and  # Single word only
                        synonym.isalpha() and
                        len(synonym) >= 3):
                        all_synonyms.append(synonym)
            
            if not all_synonyms:
                return "", "No suitable synonyms found"
            
            # Filter by similarity (prefer words of similar length)
            word_len = len(clean_word)
            filtered_synonyms = [
                syn for syn in all_synonyms 
                if abs(len(syn) - word_len) <= 3
            ]
            
            if filtered_synonyms:
                return random.choice(filtered_synonyms), None
            elif all_synonyms:
                return random.choice(all_synonyms), None
            else:
                return "", "No valid synonyms found"
                
        except Exception as e:
            return "", f"Error fetching synonym: {str(e)}"

class TextRewriteService:
    """Enhanced service for rewriting and humanizing text"""
    
    def __init__(self):
        self.refinement_repo = LocalRefinementRepository()
        self.synonym_repo = LocalSynonymRepository()
        self.filler_sentences = self._load_fillers()
        random.seed(time.time())
        logger.info("TextRewriteService initialized with local refinement")
    
    def rewrite_text(self, text: str) -> Tuple[str, Optional[str]]:
        """Main rewriting function using local refinement"""
        try:
            # Apply local refinement
            refined, err = self.refinement_repo.refine_text(text)
            return refined if refined else text, err
        except Exception as e:
            logger.error(f"Error in text rewriting: {str(e)}")
            return text, f"Rewriting error: {str(e)}"
    
    def rewrite_text_with_modifications(self, text: str) -> Tuple[str, Optional[str]]:
        """Enhanced rewriting with comprehensive modifications to make text more human-like"""
        try:
            # Start with base rewriting
            base_result, err = self.rewrite_text(text)
            if err:
                return text, err
            
            # Apply additional enhancements with HIGHER probability
            sentences = self._split_sentences(base_result)
            transformed = []
            
            for i, sentence in enumerate(sentences):
                # Apply various transformations with INCREASED probability
                if random.random() < 0.95:
                    sentence = self._vary_sentence_structure(sentence)
                if random.random() < 0.85:
                    sentence = self._replace_synonyms(sentence)
                if random.random() < 0.75:
                    sentence = self._add_natural_noise(sentence)
                if random.random() < 0.6:
                    sentence = self._add_conversational_elements(sentence)
                if random.random() < 0.5:
                    sentence = self._add_natural_imperfections(sentence)
                
                transformed.append(sentence)
            
            # More aggressive sentence reordering
            if len(transformed) > 2 and random.random() < 0.6:  # Increased from 0.5
                if len(transformed) > 3:
                    middle = transformed[1:-1]
                    random.shuffle(middle)
                    transformed = [transformed[0]] + middle + [transformed[-1]]
            
            # More frequent contextual filler addition
            if len(transformed) > 1 and random.random() < 0.6:  # Increased from 0.5
                filler = self._get_contextual_filler(transformed)
                if filler:
                    # Insert at random position (not just end)
                    insert_pos = random.randint(1, len(transformed))
                    transformed.insert(insert_pos, filler)
            
            # New: Add personal opinions and subjective language
            if random.random() < 0.4: # Increased from 0.3
                transformed = self._add_personal_touch(transformed)
            
            # New: Vary sentence length and complexity
            transformed = self._vary_sentence_complexity(transformed)
            
            return " ".join(transformed), None
            
        except Exception as e:
            logger.error(f"Error in enhanced rewriting: {str(e)}")
            return text, f"Enhanced rewriting error: {str(e)}"
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def _vary_sentence_structure(self, sentence: str) -> str:
        """Intelligently vary sentence structure"""
        if len(sentence.split()) < 4:
            return sentence
        
        transformations = [
            self._add_transition_word,
            self._rearrange_clauses,
            self._convert_contractions,
        ]
        
        transformation = random.choice(transformations)
        return transformation(sentence)
    
    def _add_transition_word(self, sentence: str) -> str:
        """Add academic transition words to sentences"""
        transitions = [
            "Furthermore, ", "Additionally, ", "Moreover, ", "Notably, ",
            "Significantly, ", "Importantly, ", "Specifically, ", "Indeed, ",
            "Particularly, ", "Evidently, ",

            "Consequently, ", "Subsequently, ",
            "Interestingly, ", "Remarkably, ", "Essentially, ", "Ultimately, ",
            "Clearly, ", "Obviously, ", "Undoubtedly, ", "Certainly, "
        ]
        
        if not sentence[0].isupper():
            return sentence
        
        # Increased probability from 0.2 to 0.5
        if random.random() < 0.6: # Increased from 0.5
            transition = random.choice(transitions)
            # Check if the sentence starts with a quote, and if so, place the transition before it.
            if sentence.strip().startswith('"'):
                match = re.match(r'(\s*")', sentence)
                quote_char = match.group(1)
                rest_of_sentence = sentence[len(quote_char):]
                return f'{quote_char}{transition}{rest_of_sentence.lower()}'
            else:
                return transition + sentence[0].lower() + sentence[1:]
        
        return sentence
    
    def _rearrange_clauses(self, sentence: str) -> str:
        """Simple clause rearrangement"""
        if ', ' in sentence and sentence.count(',') == 1:
            parts = sentence.split(', ', 1)
            if len(parts) == 2 and random.random() < 0.4: # Increased from 0.3
                part1, part2 = parts
                # Avoid rearranging if it creates an awkward sentence
                if len(part1.split()) > 2 and len(part2.split()) > 2:
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
        
        # Always expand contractions for academic tone (increased probability)
        if random.random() < 0.9: # Increased from 0.8
            for contraction, expansion in contractions.items():
                if contraction in sentence.lower():
                    # Case-sensitive replacement
                    sentence = re.sub(re.escape(contraction), expansion, sentence, flags=re.IGNORECASE)
                    break
        
        return sentence
    
    def _replace_synonyms(self, sentence: str) -> str:
        """Intelligently replace words with synonyms - MORE AGGRESSIVE"""
        words = sentence.split()
        modifications = 0
        max_modifications = max(2, len(words) // 3)  # Allow more modifications, up from //4
        
        for i, word in enumerate(words):
            if modifications >= max_modifications:
                break
                
            # Extract clean word
            clean_word = re.sub(r'[^\w]', '', word).lower()
            
            # Skip if too short or too common
            if (len(clean_word) < 3 or  # Reduced from 4 to 3
                self._is_common_word(clean_word)):
                continue
            
            # INCREASED probability from 0.15 to 0.4
            if random.random() < 0.5: # Increased from 0.4
                synonym, err = self.synonym_repo.get_synonym(clean_word)
                if not err and synonym:
                    # Preserve original word formatting
                    new_word = self._preserve_word_format(word, synonym)
                    words[i] = new_word
                    modifications += 1
        
        return " ".join(words)
    
    def _preserve_word_format(self, original: str, replacement: str) -> str:
        """Preserve capitalization and punctuation of original word"""
        # Extract prefix and suffix punctuation
        prefix = ""
        suffix = ""
        core_word = original
        
        # Get leading punctuation
        start = 0
        while start < len(original) and not original[start].isalpha():
            prefix += original[start]
            start += 1
        
        # Get trailing punctuation
        end = len(original) - 1
        while end >= 0 and not original[end].isalpha():
            suffix = original[end] + suffix
            end -= 1
        
        if start <= end:
            core_word = original[start:end+1]
        
        # Apply capitalization pattern
        if core_word and core_word[0].isupper():
            replacement = replacement.capitalize()
        elif core_word.isupper():
            replacement = replacement.upper()
        elif core_word.islower():
            replacement = replacement.lower()
        
        return prefix + replacement + suffix
    
    def _add_natural_noise(self, sentence: str) -> str:
        """Add natural linguistic variations - MORE AGGRESSIVE"""
        # More comprehensive academic-appropriate replacements
        replacements = {
            " and ": [" as well as ", " along with ", " in addition to ", " together with "],
            " but ": [" however, ", " nevertheless, ", " nonetheless, ", " conversely, "],
            " because ": [" due to the fact that ", " given that ", " since ", " as "],
            " so ": [" therefore, ", " consequently, ", " thus, ", " hence, "],
            " also ": [" furthermore, ", " additionally, ", " moreover, ", " likewise, "],
            " use ": [" utilize ", " employ ", " implement ", " apply "],
            " show ": [" demonstrate ", " illustrate ", " reveal ", " display "],
            " help ": [" facilitate ", " assist ", " aid ", " support "],
            " get ": [" obtain ", " acquire ", " achieve ", " secure "],
            " make ": [" create ", " establish ", " generate ", " produce "],
            " find ": [" discover ", " identify ", " determine ", " locate "],
            " think ": [" consider ", " believe ", " suggest ", " propose "],
            " very ": [" significantly ", " considerably ", " substantially ", " remarkably "],
            " big ": [" substantial ", " significant ", " considerable ", " extensive "],
            " small ": [" minimal ", " limited ", " modest ", " slight "],
            " good ": [" excellent ", " effective ", " beneficial ", " advantageous "],
            " bad ": [" detrimental ", " problematic ", " unfavorable ", " adverse "],
            " new ": [" novel ", " innovative ", " contemporary ", " recent "],
            " old ": [" traditional ", " established ", " conventional ", " previous "],
            " many ": [" numerous ", " multiple ", " various ", " several "],
            " few ": [" limited ", " minimal ", " sparse ", " scarce "]
        }
        
        # Apply multiple replacements per sentence with higher probability
        replacements_made = 0
        max_replacements = 3  # Allow up to 3 replacements per sentence
        
        for old, new_options in replacements.items():
            if replacements_made >= max_replacements:
                break
                
            if old in sentence.lower() and random.random() < 0.4:  # Increased from 0.3
                new_phrase = random.choice(new_options)
                # Case-sensitive replacement
                sentence = re.sub(re.escape(old), new_phrase, sentence, count=1, flags=re.IGNORECASE)
                replacements_made += 1
        
        return sentence
    
    def _add_conversational_elements(self, sentence: str) -> str:
        """Add conversational elements to make text more human-like"""
        if len(sentence.split()) < 5:
            return sentence
            
        # Add conversational phrases
        conversational_starters = [
            "You know, ", "Well, ", "Actually, ", "Basically, ", "Essentially, ",
            "I think ", "I believe ", "I feel like ", "It seems like ", "Apparently, ",
            "Obviously, ", "Clearly, ", "Of course, ", "Naturally, ", "Obviously, ",
            "In my opinion, ", "From what I understand, ", "As far as I can tell, "
        ]
        
        # Add conversational connectors
        conversational_connectors = [
            " you see, ", " I mean, ", " like, ", " sort of, ", " kind of, ",
            " actually, ", " basically, ", " essentially, ", " obviously, ",
            " clearly, ", " naturally, ", " of course, ", " well, "
        ]
        
        # Add starter phrase
        if random.random() < 0.25 and not sentence.startswith(("You know,", "Well,", "Actually,")): # Increased from 0.2
            starter = random.choice(conversational_starters)
            sentence = starter + sentence.lower()
        
        # Add connector phrase in the middle
        if random.random() < 0.2 and len(sentence.split()) > 8: # Increased from 0.15
            words = sentence.split()
            if len(words) > 4:
                insert_pos = random.randint(2, len(words) - 2)
                connector = random.choice(conversational_connectors)
                words.insert(insert_pos, connector.strip())
                sentence = " ".join(words)
        
        return sentence
    
    def _add_natural_imperfections(self, sentence: str) -> str:
        """Add natural imperfections that humans make"""
        if len(sentence.split()) < 4:
            return sentence
            
        # Add filler words
        filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]
        
        # Add repetition (humans sometimes repeat words)
        if random.random() < 0.15: # Increased from 0.1
            words = sentence.split()
            if len(words) > 3:
                repeat_word = random.choice([w for w in words if len(w) > 3 and w.isalpha()])
                if repeat_word in words:
                    insert_pos = words.index(repeat_word) + 1
                    words.insert(insert_pos, repeat_word)
                    sentence = " ".join(words)
        
        # Add incomplete thoughts (with ellipsis)
        if random.random() < 0.08: # Increased from 0.05
            if not sentence.endswith("..."):
                sentence = sentence.rstrip(".") + "..."
        
        # Add parenthetical asides
        if random.random() < 0.15: # Increased from 0.1
            asides = [
                " (which is interesting)", " (I think)", " (you know)", 
                " (obviously)", " (basically)", " (essentially)",
                " (more or less)", " (sort of)", " (kind of)"
            ]
            aside = random.choice(asides)
            # Insert before the last word
            words = sentence.split()
            if len(words) > 2:
                words.insert(-1, aside)
                sentence = " ".join(words)
        
        return sentence
    
    def _add_personal_touch(self, sentences: List[str]) -> List[str]:
        """Add personal opinions and subjective language"""
        if not sentences:
            return sentences
            
        personal_phrases = [
            "I find this quite interesting because",
            "What's really fascinating to me is",
            "I think this is particularly noteworthy since",
            "From my perspective, this suggests",
            "I believe this demonstrates",
            "It seems to me that",
            "I would argue that",
            "In my experience, this indicates",
            "I've noticed that",
            "Personally, I think"
        ]
        
        # Add personal phrase to a random sentence
        if random.random() < 0.4: # Increased from 0.3
            target_idx = random.randint(0, len(sentences) - 1)
            personal_phrase = random.choice(personal_phrases)
            sentences[target_idx] = personal_phrase + " " + sentences[target_idx].lower()
        
        # Add opinion markers
        opinion_markers = [
            " in my opinion", " I believe", " I think", " I feel",
            " it seems to me", " from what I can see", " as I see it"
        ]
        
        for i, sentence in enumerate(sentences):
            if random.random() < 0.2: # Increased from 0.15
                marker = random.choice(opinion_markers)
                # Insert before the last word
                words = sentence.split()
                if len(words) > 2:
                    words.insert(-1, marker)
                    sentences[i] = " ".join(words)
        
        return sentences
    
    def _vary_sentence_complexity(self, sentences: List[str]) -> List[str]:
        """Vary sentence length and complexity to be more human-like"""
        if len(sentences) < 2:
            return sentences
            
        # Sometimes combine short sentences
        i = 0
        while i < len(sentences) - 1:
            if (len(sentences[i].split()) < 8 and 
                len(sentences[i+1].split()) < 8 and 
                random.random() < 0.3): # Increased from 0.2
                # Combine with a conjunction
                conjunctions = [" and", " but", " however,", " moreover,", " furthermore,"]
                conjunction = random.choice(conjunctions)
                sentences[i] = sentences[i].rstrip(".") + conjunction + " " + sentences[i+1].lower()
                sentences.pop(i+1)
            else:
                i += 1
        
        # Sometimes break long sentences
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 20 and random.random() < 0.4: # Increased from 0.3
                words = sentence.split()
                if len(words) > 15:
                    # Find a good break point (after a comma or conjunction)
                    break_point = len(words) // 2
                    for j, word in enumerate(words):
                        if j > 10 and word.endswith(","):
                            break_point = j + 1
                            break
                    
                    first_part = " ".join(words[:break_point])
                    second_part = " ".join(words[break_point:])
                    
                    # Ensure proper capitalization
                    if second_part and second_part[0].isalpha():
                        second_part = second_part[0].upper() + second_part[1:]
                    
                    sentences[i] = first_part
                    sentences.insert(i+1, second_part)
        
        return sentences
    
    def _get_contextual_filler(self, sentences: List[str]) -> str:
        """Generate academic contextual filler sentence"""
        if not sentences:
            return ""
        
        # Extract themes from the text
        all_text = " ".join(sentences)
        keywords = self._extract_keywords(all_text)
        
        if keywords and len(keywords) > 0:
            # Academic templates
            templates = [
                "This analysis underscores the significance of {keyword}.",
                "The examination of {keyword} reveals important insights.",
                "Such findings regarding {keyword} warrant further consideration.",
                "The implications of {keyword} are particularly noteworthy.",
                "This investigation into {keyword} provides valuable understanding.",
                "The study of {keyword} demonstrates considerable importance.",
                "These observations concerning {keyword} merit attention."
            ]
            
            template = random.choice(templates)
            keyword = random.choice(keywords[:3])  # Use top 3 keywords
            return template.format(keyword=keyword)
        
        # Fallback to academic transitions
        return random.choice(self.filler_sentences)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        
        # Filter out common words
        filtered_words = [
            word for word in words 
            if not self._is_common_word(word)
        ]
        
        # Return unique keywords
        return list(dict.fromkeys(filtered_words))
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is too common for replacement in academic context"""
        # Expanded list including academic terms to preserve
        common_words = {
            "the", "and", "that", "this", "with", "have", "will", "been", 
            "from", "they", "know", "want", "been", "good", "much", "some",
            "time", "very", "when", "come", "here", "just", "like", "long",
            "make", "many", "over", "such", "take", "than", "them", "well",
            "were", "work", "about", "could", "would", "there", "their",
            "which", "should", "think", "where", "through", "because",
            "between", "important", "different", "following", "around",
            "though", "without", "another", "example", "however", "therefore",
            # Academic terms to preserve
            "research", "study", "analysis", "data", "method", "result",
            "conclusion", "evidence", "theory", "hypothesis", "findings",
            "literature", "methodology", "framework", "approach", "concept",
            "significant", "substantial", "considerable", "demonstrate",
            "indicate", "suggest", "reveal", "establish", "examine", "AI", "IoT", "ML", "NLP", 
            "deep learning", "blockchain", "cloud computing", "big data", "cybersecurity", "data science", 
            "augmented reality", "virtual reality", "edge computing", "quantum computing", "natural language processing",
            "machine learning", "artificial intelligence", "internet of things", "data analytics", "digital transformation",
            "automation", "smart technology", "sustainability", "innovation", "disruption", "technology"
        }
        return word.lower() in common_words
    
    def _load_fillers(self) -> List[str]:
        """Load academic-appropriate filler sentences"""
        return [
            "This analysis provides valuable insights into the subject matter.",
            "Such examination proves particularly enlightening for understanding the topic.", 
            "These considerations merit further scholarly attention.",
            "The implications of this research become increasingly evident.",
            "This methodological approach yields meaningful academic results.",
            "The findings contribute significantly to the existing body of knowledge.",
            "This investigation enhances our understanding of the phenomenon.",
            "The research demonstrates the complexity of the underlying issues."
        ]

def rewrite_text(text: str, enhanced: bool = True) -> Tuple[str, Optional[str]]:
    """
    Main function to rewrite text with academic style
    
    Args:
        text: Input text to rewrite
        enhanced: Whether to use enhanced modifications (default: True for better humanization)
        
    Returns:
        Tuple of (rewritten_text, error_message)
    """
    try:
        service = TextRewriteService()
        if enhanced:
            result, err = service.rewrite_text_with_modifications(text)
            if err:
                return text, err
            # Apply academic transformations to make it more scholarly
            result = _apply_academic_transformations(result)
            return result, None
        else:
            result, err = service.rewrite_text(text)
            if err:
                return text, err
            # Apply academic transformations even in basic mode
            result = _apply_academic_transformations(result)
            return result, None
    except Exception as e:
        logger.error(f"Error in rewrite_text: {str(e)}")
        return text, f"Rewrite error: {str(e)}"

def rewrite_text_ultra(text: str) -> Tuple[str, Optional[str]]:
    """
    extreme-enhanced rewriting for maximum humanization
    
    Args:
        text: Input text to rewrite
        
    Returns:
        Tuple of (rewritten_text, error_message)
    """
    try:
        service = TextRewriteService()
        # Apply multiple passes for maximum humanization
        result, err = service.rewrite_text_with_modifications(text)
        if err:
            return text, err
        
        # Apply a second pass for even more humanization
        result2, err2 = service.rewrite_text_with_modifications(result)
        if err2:
            return result, err2
        
        # Apply a third pass for final polish and variation
        result3, err3 = service.rewrite_text_with_modifications(result2)
        if err3:
            return result2, err3

        return result3, None
    except Exception as e:
        logger.error(f"Error in rewrite_text_ultra: {str(e)}")
        return text, f"Ultra rewrite error: {str(e)}"

def get_synonym(word: str) -> Tuple[str, Optional[str]]:
    """
    Get synonym for a word
    
    Args:
        word: Word to find synonym for
        
    Returns:
        Tuple of (synonym, error_message)
    """
    repo = LocalSynonymRepository()
    return repo.get_synonym(word)

def refine_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Refine text using NLP tools
    
    Args:
        text: Text to refine
        
    Returns:
        Tuple of (refined_text, error_message)
    """
    repo = LocalRefinementRepository()
    return repo.refine_text(text)

def rewrite_text_academic(text: str) -> Tuple[str, Optional[str]]:
    """
    Rewrite text with academic tone and style
    
    Args:
        text: Input text to rewrite with academic style
        
    Returns:
        Tuple of (academically_rewritten_text, error_message)
    """
    try:
        service = TextRewriteService()
        
        # First apply base rewriting
        result, err = service.rewrite_text(text)
        if err:
            return text, err
        
        # Apply academic transformations
        result = _apply_academic_transformations(result)
        
        return result, None
    except Exception as e:
        logger.error(f"Error in academic rewrite: {str(e)}")
        return text, f"Academic rewrite error: {str(e)}"

def _apply_academic_transformations(text: str) -> str:
    """Apply academic writing transformations to text"""
    
    # Academic word replacements for more formal tone
    academic_replacements = {
        r'\bshow\b': 'demonstrate',
        r'\bget\b': 'obtain',
        r'\bfind\b': 'discover',
        r'\bmake\b': 'establish',
        r'\buse\b': 'utilize',
        r'\bhelp\b': 'facilitate',
        r'\bbig\b': 'significant',
        r'\bsmall\b': 'minimal',
        r'\bgood\b': 'favorable',
        r'\bbad\b': 'adverse',
        r'\bthing\b': 'element',
        r'\bstuff\b': 'material',
        r'\bway\b': 'method',
        r'\bstart\b': 'commence',
        r'\bend\b': 'conclude',
        r'\btry\b': 'attempt',
        r'\blook at\b': 'examine',
        r'\bcheck\b': 'verify',
        r'\btell\b': 'indicate',
        r'\blet\b': 'permit',
        r'\bput\b': 'place',
        r'\btake\b': 'consider',
        r'\bgive\b': 'provide',
        r'\bkeep\b': 'maintain',
        r'\bask\b': 'inquire',
        r'\bsay\b': 'state',
        r'\bthink\b': 'postulate',
        r'\bknow\b': 'understand',
        r'\bsee\b': 'observe',
        r'\bdo\b': 'conduct',
        r'\bgo\b': 'proceed',
        r'\bcome\b': 'emerge',
        r'\bturn\b': 'transform',
        r'\bwork\b': 'function',
        r'\bplay\b': 'serve',
        r'\brun\b': 'operate',
        r'\bmove\b': 'transition',
        r'\bchange\b': 'modify',
        r'\bbring\b': 'introduce',
        r'\bhold\b': 'maintain',
        r'\bmean\b': 'signify',
        r'\bseem\b': 'appear',
        r'\bfeel\b': 'perceive'
    }
    
    # Academic phrase transformations
    academic_phrases = {
        r'\bI think\b': 'It is proposed that',
        r'\bI believe\b': 'The evidence suggests that',
        r'\bwe can see\b': 'it becomes evident that',
        r'\bwe know\b': 'research indicates that',
        r'\bthis shows\b': 'this demonstrates',
        r'\bthis means\b': 'this implies',
        r'\bin conclusion\b': 'consequently',
        r'\bto sum up\b': 'in synthesis',
        r'\bfirst of all\b': 'primarily',
        r'\bsecond\b': 'furthermore',
        r'\bthird\b': 'additionally',
        r'\bfinally\b': 'ultimately',
        r'\balso\b': 'moreover',
        r'\bbut\b': 'however',
        r'\bso\b': 'therefore',
        r'\bbecause\b': 'due to the fact that',
        r'\bsince\b': 'given that',
        r'\bwhen\b': 'during the period when',
        r'\bif\b': 'in the event that',
        r'\bfor example\b': 'for instance',
        r'\blike\b': 'such as',
        r'\babout\b': 'regarding',
        r'\bvery\b': 'particularly',
        r'\breally\b': 'significantly',
        r'\bquite\b': 'considerably',
        r'\bmostly\b': 'predominantly',
        r'\bmainly\b': 'principally',
        r'\boften\b': 'frequently',
        r'\busually\b': 'typically',
        r'\balways\b': 'invariably',
        r'\bnever\b': 'under no circumstances',
        r'\bmaybe\b': 'potentially',
        r'\bprobably\b': 'presumably',
        r'\bcertainly\b': 'undoubtedly'
    }
    
    # Apply word replacements
    for pattern, replacement in academic_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Apply phrase transformations
    for pattern, replacement in academic_phrases.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Add academic sentence starters
    sentences = sent_tokenize(text)
    if sentences:
        academic_starters = [
            "The research demonstrates that ",
            "Analysis reveals that ",
            "Studies indicate that ",
            "Evidence suggests that ",
            "Investigations show that ",
            "The findings establish that ",
            "Research indicates that ",
            "The data demonstrates that "
        ]
        
        # Randomly enhance some sentences with academic starters
        enhanced_sentences = []
        for i, sentence in enumerate(sentences):
            if i < 3 and random.random() < 0.3:  # First few sentences, 30% chance
                starter = random.choice(academic_starters)
                # Make first word lowercase if adding starter
                if sentence:
                    sentence = sentence[0].lower() + sentence[1:]
                sentence = starter + sentence
            enhanced_sentences.append(sentence)
        
        text = " ".join(enhanced_sentences)
    
    # Add transitional phrases between sentences
    sentences = sent_tokenize(text)
    if len(sentences) > 1:
        transitions = [
            "Furthermore, ",
            "Additionally, ",
            "Moreover, ",
            "Consequently, ",
            "Subsequently, ",
            "In addition, ",
            "Nevertheless, ",
            "Nonetheless, ",
            "Therefore, ",
            "Thus, "
        ]
        
        enhanced_sentences = [sentences[0]]  # Keep first sentence as is
        for i in range(1, len(sentences)):
            if random.random() < 0.4:  # 40% chance to add transition
                transition = random.choice(transitions)
                enhanced_sentences.append(transition + sentences[i])
            else:
                enhanced_sentences.append(sentences[i])
        
        text = " ".join(enhanced_sentences)
    
    return text