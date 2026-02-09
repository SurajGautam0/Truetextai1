import logging
import re
from typing import Dict, List, Optional, Tuple

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import sent_tokenize, word_tokenize
except Exception:
    nltk = None
    wordnet = None
    sent_tokenize = None
    word_tokenize = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

logger = logging.getLogger(__name__)

# Keep replacements conservative to preserve meaning.
CONTRACTIONS: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
}

ACADEMIC_PHRASES: Dict[str, str] = {
    r"\bin order to\b": "to",
    r"\ba lot of\b": "many",
    r"\bkind of\b": "somewhat",
    r"\bsort of\b": "somewhat",
    r"\bget\b": "obtain",
    r"\bshow\b": "demonstrate",
    r"\bhelp\b": "support",
    r"\buse\b": "utilize",
    r"\bfind out\b": "determine",
}

COMMON_WORDS = {
    "the", "and", "that", "this", "with", "from", "their", "there", "would",
    "could", "should", "about", "which", "because", "however", "therefore",
}

PROTECTED_PATTERNS = [
    r"https?://\\S+",
    r"\[[^\]]+\]",            # [1], [Smith, 2020]
    r"\([^()]*\d{4}[a-z]?[^()]*\)",
]


def _safe_sent_tokenize(text: str) -> List[str]:
    if not text.strip():
        return []
    if sent_tokenize is not None:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _safe_word_tokenize(text: str) -> List[str]:
    if word_tokenize is not None:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\w+|[^\w\s]", text)


def _detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:)\]])", r"\1", text)
    text = re.sub(r"([([\"'])\s+", r"\1", text)
    return text


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement.lower()


def _protect_spans(text: str) -> Tuple[str, Dict[str, str]]:
    spans: Dict[str, str] = {}
    protected = text
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        key = f"__PROTECTED_{idx}__"
        spans[key] = match.group(0)
        idx += 1
        return key

    for pattern in PROTECTED_PATTERNS:
        protected = re.sub(pattern, repl, protected)
    return protected, spans


def _restore_spans(text: str, spans: Dict[str, str]) -> str:
    for key, value in spans.items():
        text = text.replace(key, value)
    return text


def _expand_contractions(sentence: str) -> str:
    out = sentence
    for old, new in CONTRACTIONS.items():
        out = re.sub(rf"\b{re.escape(old)}\b", new, out, flags=re.IGNORECASE)
    return out


def _apply_academic_phrasing(sentence: str) -> str:
    out = sentence
    for pattern, replacement in ACADEMIC_PHRASES.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _wordnet_synonym(word: str) -> Optional[str]:
    if wordnet is None:
        return None
    try:
        synsets = wordnet.synsets(word.lower())
        candidates: List[str] = []
        for syn in synsets[:3]:
            for lemma in syn.lemmas()[:6]:
                value = lemma.name().replace("_", " ")
                if (
                    value.lower() != word.lower()
                    and value.isalpha()
                    and len(value.split()) == 1
                    and 3 <= len(value) <= 14
                ):
                    candidates.append(value)
        if not candidates:
            return None
        # Prefer close-length words to keep meaning stable.
        candidates.sort(key=lambda w: abs(len(w) - len(word)))
        return candidates[0]
    except Exception:
        return None


def _replace_synonyms(sentence: str, max_changes: int = 1) -> str:
    tokens = _safe_word_tokenize(sentence)
    changes = 0

    for i, token in enumerate(tokens):
        if changes >= max_changes:
            break
        if not token.isalpha():
            continue
        lower = token.lower()
        if len(lower) < 5 or lower in COMMON_WORDS:
            continue

        synonym = _wordnet_synonym(lower)
        if not synonym:
            continue

        tokens[i] = _preserve_case(token, synonym)
        changes += 1

    return _detokenize(tokens)


def _basic_cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s*([A-Za-z])", r"\1 \2", text)
    return text


def _sentence_polish(sentence: str, academic: bool, enhanced: bool) -> str:
    s = sentence.strip()
    if not s:
        return s

    # Optional grammar correction; keep lightweight.
    if TextBlob is not None and len(s.split()) <= 40:
        try:
            corrected = str(TextBlob(s).correct())
            if corrected:
                s = corrected
        except Exception:
            pass

    s = _expand_contractions(s)
    if academic:
        s = _apply_academic_phrasing(s)

    if enhanced:
        s = _replace_synonyms(s, max_changes=1)

    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return _basic_cleanup(s)


def _rewrite_core(text: str, enhanced: bool, academic: bool) -> str:
    protected_text, spans = _protect_spans(text)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", protected_text) if p.strip()]
    out_paragraphs: List[str] = []

    for paragraph in paragraphs:
        sentences = _safe_sent_tokenize(paragraph)
        if not sentences:
            out_paragraphs.append(_basic_cleanup(paragraph))
            continue

        polished = [_sentence_polish(s, academic=academic, enhanced=enhanced) for s in sentences]
        out_paragraphs.append(" ".join(polished))

    rebuilt = "\n\n".join(out_paragraphs)
    rebuilt = _restore_spans(rebuilt, spans)
    return _basic_cleanup(rebuilt)


def rewrite_text(text: str, enhanced: bool = True) -> Tuple[str, Optional[str]]:
    """Rewrite with clarity and flow improvements while preserving meaning."""
    try:
        if not text or not text.strip():
            return text, None
        return _rewrite_core(text, enhanced=enhanced, academic=False), None
    except Exception as e:
        logger.exception("rewrite_text failed")
        return text, f"Rewrite error: {e}"


def rewrite_text_academic(text: str) -> Tuple[str, Optional[str]]:
    """Rewrite with formal academic phrasing and readability improvements."""
    try:
        if not text or not text.strip():
            return text, None
        return _rewrite_core(text, enhanced=True, academic=True), None
    except Exception as e:
        logger.exception("rewrite_text_academic failed")
        return text, f"Academic rewrite error: {e}"


def refine_text(text: str) -> Tuple[str, Optional[str]]:
    """Apply light refinement without paraphrasing."""
    try:
        if not text:
            return text, None
        return _basic_cleanup(text), None
    except Exception as e:
        logger.exception("refine_text failed")
        return text, f"Refine error: {e}"


def get_synonym(word: str) -> Tuple[str, Optional[str]]:
    """Return one suitable WordNet synonym for a word."""
    try:
        w = (word or "").strip()
        if len(w) < 3:
            return "", "Word too short"

        synonym = _wordnet_synonym(w)
        if not synonym:
            return "", "No synonyms found"
        return synonym, None
    except Exception as e:
        logger.exception("get_synonym failed")
        return "", f"Synonym error: {e}"


if __name__ == "__main__":
    sample = "AI tools can help students a lot, but they don't always improve understanding."
    rewritten, err = rewrite_text_academic(sample)
    print("Error:", err)
    print("Output:", rewritten)
