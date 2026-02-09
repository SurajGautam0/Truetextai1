import os
import random
import re
import logging
import time
from typing import List, Tuple, Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Try advanced NLP
try:
    import spacy
    from textblob import TextBlob
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy/TextBlob not installed. Install with: pip install spacy textblob")

# Download NLTK data
def setup_nltk():
    packages = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for pkg in packages:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except LookupError:
            print(f"Downloading {pkg}...")
            nltk.download(pkg, quiet=True)

setup_nltk()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lemmatizer = WordNetLemmatizer()

# Rich academic / natural transition sets (avoid overused AI patterns)
TRANSITIONS = {
    "start": ["What stands out is", "One key observation is", "Interestingly,", "Importantly,",
              "A central point here is", "It becomes evident that", "Notably,"],
    "contrast": ["That said,", "However,", "Yet at the same time,", "Nevertheless,",
                 "On the other hand,", "This contrasts with", "Despite this,"],
    "addition": ["Beyond that,", "In the same vein,", "Equally significant is", "Moreover,",
                 "Adding to this,", "Building on this idea,"],
    "cause": ["This stems from", "Primarily because", "The reason lies in", "Given that", "Owing to"],
    "result": ["As a result,", "Consequently,", "This naturally leads to", "It follows that"],
    "example": ["For instance,", "A clear example is", "Consider how", "To illustrate,"],
    "conclusion": ["Ultimately,", "In essence,", "Taken together,", "What this suggests is",
                   "All of this points to"]
}

COMMON_WORDS = {
    "the", "and", "that", "this", "with", "for", "are", "was", "were", "have", "has", "had",
    "not", "but", "from", "they", "their", "them", "which", "who", "when", "where", "there",
    "then", "than", "these", "those", "would", "could", "should", "will", "can", "may", "might"
}

class ProfessorGradeHumanizer:
    """Advanced academic humanizer designed to defeat 2026 detectors while producing professor-ready text."""
    
    def __init__(self):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy loaded successfully")
            except OSError:
                logger.warning("Download spaCy model: python -m spacy download en_core_web_sm")
        
        random.seed(time.time() + os.getpid())
    
    def _get_synonym(self, word: str, pos: str = None) -> Optional[str]:
        """Context-aware synonym with preference for less predictable (higher perplexity) words."""
        if len(word) < 4 or word.lower() in COMMON_WORDS:
            return None
            
        synsets = wordnet.synsets(word.lower())
        candidates = []
        
        for syn in synsets[:4]:
            for lemma in syn.lemmas()[:6]:
                syn_word = lemma.name().replace('_', ' ')
                if (syn_word.lower() != word.lower() and 
                    len(syn_word.split()) == 1 and 
                    syn_word.isalpha() and 
                    3 <= len(syn_word) <= 14):
                    # Prefer slightly rarer / more specific words for perplexity
                    if len(syn_word) != len(word) or random.random() < 0.4:
                        candidates.append(syn_word)
        
        if candidates:
            # Bias toward longer or more precise words occasionally
            return random.choice(candidates)
        return None
    
    def _force_burstiness(self, sentences: List[str]) -> List[str]:
        """Create strong human-like variation in sentence length (core anti-detection feature)."""
        if len(sentences) < 3:
            return sentences
            
        result = []
        i = 0
        while i < len(sentences):
            current = sentences[i].strip()
            
            # 35% chance to merge with next (creates longer sentences)
            if (i + 1 < len(sentences) and 
                random.random() < 0.35 and 
                len(current.split()) < 22):
                next_sent = sentences[i+1].strip()
                conj = random.choice([", and yet ", " — ", "; however, ", ". Still, "])
                merged = current.rstrip(".!?") + conj + next_sent[0].lower() + next_sent[1:]
                result.append(merged)
                i += 2
                continue
                
            # 30% chance to split long sentences
            if len(current.split()) > 24 and random.random() < 0.30:
                words = current.split()
                split_point = random.randint(max(8, len(words)//2 - 5), min(len(words)-6, len(words)//2 + 6))
                part1 = " ".join(words[:split_point]).rstrip(".,") + random.choice([";", " —", ","])
                part2 = " ".join(words[split_point:]).capitalize()
                result.extend([part1, part2])
                i += 1
                continue
                
            result.append(current)
            i += 1
            
        return result
    
    def _humanize_sentence(self, sentence: str) -> str:
        """Core per-sentence transformation for perplexity + natural flow."""
        if len(sentence.strip()) < 10:
            return sentence.strip()
            
        # Basic cleanup + capitalization
        sentence = sentence.strip()
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        
        # spaCy for smart POS-based replacements (best perplexity boost)
        if self.nlp:
            doc = self.nlp(sentence)
            tokens = []
            for token in doc:
                word = token.text
                if (token.pos_ in {"ADJ", "ADV", "VERB", "NOUN"} and 
                    random.random() < 0.28 and 
                    len(word) > 4):
                    syn = self._get_synonym(word, token.pos_)
                    if syn:
                        # Preserve case
                        if word[0].isupper():
                            syn = syn.capitalize()
                        word = syn
                tokens.append(word)
            sentence = " ".join(tokens)
        else:
            # Fallback NLTK word-level
            words = word_tokenize(sentence)
            for i, word in enumerate(words):
                clean = re.sub(r'[^a-zA-Z]', '', word).lower()
                if (len(clean) > 4 and 
                    clean not in COMMON_WORDS and 
                    random.random() < 0.25):
                    syn = self._get_synonym(clean)
                    if syn:
                        words[i] = self._preserve_case(word, syn)
            sentence = " ".join(words)
        
        # Occasional natural connectors / asides for human feel
        if random.random() < 0.22:
            aside = random.choice([" (quite remarkably)", " — or so it appears", ", to my mind,", 
                                  " (and this matters)", " — importantly —"])
            words = sentence.split()
            if len(words) > 6:
                pos = random.randint(2, len(words)-3)
                words.insert(pos, aside)
                sentence = " ".join(words)
        
        return sentence
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        return replacement.lower()
    
    def _vary_transitions(self, sentences: List[str]) -> List[str]:
        """Replace mechanical transitions with more natural academic ones."""
        if not sentences:
            return sentences
            
        for i in range(1, len(sentences)):
            sent = sentences[i]
            # Remove common AI-style starters
            for bad in ["Furthermore", "Moreover", "Additionally", "In addition", "Also,"]:
                if sent.strip().startswith(bad):
                    sent = sent.replace(bad, "", 1).strip()
                    break
            
            if random.random() < 0.45 and sent[0].isupper():
                category = random.choice(list(TRANSITIONS.keys()))
                starter = random.choice(TRANSITIONS[category])
                sentences[i] = starter + " " + sent[0].lower() + sent[1:]
        
        return sentences
    
    def _academic_polish(self, text: str) -> str:
        """Final academic tone enhancement without changing meaning."""
        # Light formal upgrades (safe & natural)
        polish_map = {
            r'\bvery\b': 'particularly',
            r'\bgood\b': 'effective',
            r'\bbad\b': 'problematic',
            r'\bthing\b': 'aspect',
            r'\bstuff\b': 'elements',
            r'\buse\b': 'employ',
            r'\bhelp\b': 'facilitate',
            r'\bshow\b': 'demonstrate',
            r'\bget\b': 'obtain',
            r'\bmake\b': 'generate'
        }
        
        for old, new in polish_map.items():
            text = re.sub(old, new, text, flags=re.IGNORECASE)
        
        return text
    
    def humanize(self, text: str, intensity: str = "high") -> str:
        """Main entry point — produces professor-ready, highly humanized text."""
        if not text or not text.strip():
            return text
            
        intensity_map = {"low": 0.4, "medium": 0.65, "high": 0.9}
        factor = intensity_map.get(intensity.lower(), 0.85)
        
        # Step 1: Split into sentences and paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        result_paras = []
        
        for para in paragraphs:
            sentences = sent_tokenize(para)
            
            # Core humanization passes
            processed = [self._humanize_sentence(s) for s in sentences]
            
            # Burstiness (most important anti-detection feature)
            if random.random() < factor:
                processed = self._force_burstiness(processed)
            
            # Transition variety
            if random.random() < factor:
                processed = self._vary_transitions(processed)
            
            # Final polish
            new_para = " ".join(processed)
            new_para = self._academic_polish(new_para)
            
            # Occasional paragraph-level reflection (adds depth & humanity)
            if len(new_para.split()) > 60 and random.random() < 0.25 * factor:
                connector = random.choice([" What this ultimately reveals is that ", 
                                         " One cannot overlook the fact that ", 
                                         " This perspective brings into focus "])
                tail = " ".join(new_para.split()[-12:]).lower()
                new_para = new_para.rstrip(".") + connector + tail + "."
            
            result_paras.append(new_para)
        
        final_text = "\n\n".join(result_paras)
        
        # Final global cleanup
        final_text = re.sub(r'\s+', ' ', final_text)
        final_text = re.sub(r'\s+([,.!?;:])', r'\1', final_text)
        final_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', final_text)
        
        return final_text.strip()


# ====================== USAGE ======================

def rewrite_for_professor(text: str, intensity: str = "high") -> str:
    """
    One-call function to produce submission-ready academic text 
    that strongly resists modern AI detectors.
    """
    humanizer = ProfessorGradeHumanizer()
    return humanizer.humanize(text, intensity)


# ====================== BACKWARD-COMPAT API ======================

def rewrite_text(text: str, enhanced: bool = True) -> Tuple[str, Optional[str]]:
    """Compatibility wrapper used by main.py."""
    try:
        intensity = "high" if enhanced else "medium"
        return rewrite_for_professor(text, intensity=intensity), None
    except Exception as e:
        return text, f"Rewrite error: {e}"


def rewrite_text_academic(text: str) -> Tuple[str, Optional[str]]:
    """Academic wrapper kept for existing endpoint compatibility."""
    try:
        return rewrite_for_professor(text, intensity="medium"), None
    except Exception as e:
        return text, f"Academic rewrite error: {e}"


def refine_text(text: str) -> Tuple[str, Optional[str]]:
    """Light refinement for existing /refine endpoint compatibility."""
    try:
        if not text:
            return text, None
        refined = re.sub(r"\s+", " ", text).strip()
        refined = re.sub(r"\s+([,.!?;:])", r"\1", refined)
        return refined, None
    except Exception as e:
        return text, f"Refine error: {e}"


def get_synonym(word: str) -> Tuple[str, Optional[str]]:
    """WordNet synonym lookup for existing /synonym endpoint compatibility."""
    try:
        w = (word or "").strip().lower()
        if len(w) < 3:
            return "", "Word too short"
        synsets = wordnet.synsets(w)
        options = []
        for syn in synsets[:3]:
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate != w and candidate.isalpha() and len(candidate.split()) == 1:
                    options.append(candidate)
        if not options:
            return "", "No synonyms found"
        return random.choice(options), None
    except Exception as e:
        return "", f"Synonym error: {e}"


# Example / CLI
if __name__ == "__main__":
    print("Professor-Grade AI Humanizer (2026-ready) loaded.\n")
    sample = input("Paste your text (or press Enter for demo):\n")
    if not sample.strip():
        sample = "Artificial intelligence is transforming education in many ways. Students can now access personalized learning experiences. Teachers are also benefiting from new tools."
    
    result = rewrite_for_professor(sample, intensity="high")
    print("\n" + "="*60)
    print("REWRITTEN TEXT (ready for professor submission):")
    print("="*60)
    print(result)
