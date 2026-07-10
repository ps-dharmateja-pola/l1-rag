"""
Helper utilities for SDS RAG Assistant.
"""

import re
import time
import unicodedata
from typing import Any, Dict, List


# ── Confidence ────────────────────────────────────────────────────────────────

def calculate_confidence_level(scores: List[float]) -> str:
    """
    Map ChromaDB L2 distances to High / Medium / Low confidence.

    ChromaDB with cosine space returns distances where 0 = identical, 2 = opposite.
    Typical good matches are < 0.5; weak matches are > 1.0.
    """
    if not scores:
        return "Low"
    best = min(scores)
    if best < 0.4:
        return "High"
    elif best < 0.7:
        return "Medium"
    else:
        return "Low"


# ── Source formatting ─────────────────────────────────────────────────────────

def format_source_citation(metadata: Dict[str, Any]) -> str:
    page     = metadata.get("page_number", "?")
    section  = metadata.get("section",     "General")
    doc_name = metadata.get("document_name", "Unknown")
    return f"{doc_name} — Page {page}, Section: {section}"


def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "No sources available"
    return "\n".join(
        f"{i}. {format_source_citation(s.get('metadata', {}))}"
        for i, s in enumerate(sources, 1)
    )


# ── Language detection ────────────────────────────────────────────────────────

# ISO-639-1 code → human-readable name
_LANG_NAMES = {
    "en": "English",   "ru": "Russian",   "pt": "Portuguese",
    "de": "German",    "fr": "French",    "es": "Spanish",
    "it": "Italian",   "nl": "Dutch",     "pl": "Polish",
    "ar": "Arabic",    "zh-cn": "Chinese","zh-tw": "Chinese",
    "ja": "Japanese",  "ko": "Korean",    "hi": "Hindi",
    "tr": "Turkish",   "sv": "Swedish",   "da": "Danish",
    "fi": "Finnish",   "no": "Norwegian", "cs": "Czech",
    "sk": "Slovak",    "ro": "Romanian",  "hu": "Hungarian",
    "el": "Greek",     "he": "Hebrew",    "uk": "Ukrainian",
    "bg": "Bulgarian", "hr": "Croatian",  "sr": "Serbian",
    "lt": "Lithuanian","lv": "Latvian",   "et": "Estonian",
    "vi": "Vietnamese","th": "Thai",      "id": "Indonesian",
    "ms": "Malay",     "ca": "Catalan",   "gl": "Galician",
    "eu": "Basque",    "af": "Afrikaans",
}

# ISO-639-1 code → deep-translator source code
_TRANSLATE_CODES = {
    "en": "en",  "ru": "ru",  "pt": "pt",  "de": "de",  "fr": "fr",
    "es": "es",  "it": "it",  "nl": "nl",  "pl": "pl",  "ar": "ar",
    "zh-cn": "zh-CN", "zh-tw": "zh-TW", "ja": "ja", "ko": "ko",
    "hi": "hi",  "tr": "tr",  "sv": "sv",  "da": "da",  "fi": "fi",
    "no": "no",  "cs": "cs",  "sk": "sk",  "ro": "ro",  "hu": "hu",
    "el": "el",  "he": "iw",  "uk": "uk",  "bg": "bg",  "hr": "hr",
    "vi": "vi",  "th": "th",  "id": "id",  "ms": "ms",  "ca": "ca",
}


def detect_language(pages_data: list) -> str:
    """
    Detect the primary language of a document using statistical text analysis.

    Strategy (in order of reliability):
      1. langdetect — statistical model covering 55 languages.
         Takes a sample of text, runs it through a Naive Bayes classifier
         trained on Wikipedia. Handles Portuguese vs English vs Spanish etc.
      2. Fallback: Unicode script counting for non-Latin scripts (Cyrillic,
         Arabic, Chinese, etc.) — langdetect handles these but this is instant.
      3. Last resort: return "Unknown".

    Returns a human-readable name: "English", "Portuguese", "Russian", etc.
    """
    # Collect a representative sample from across the whole document
    sample = " ".join(p["text"] for p in pages_data)

    # --- Fast path: non-Latin scripts via Unicode ---
    # Count a small sample first; if a non-Latin script dominates, skip langdetect
    script_sample = sample[:2000]
    non_latin: Dict[str, int] = {"cyrillic": 0, "arabic": 0, "chinese": 0,
                                  "greek": 0, "hebrew": 0, "devanagari": 0}
    latin_count = 0
    for ch in script_sample:
        if not ch.isalpha():
            continue
        uname = unicodedata.name(ch, "")
        if   "CYRILLIC"   in uname: non_latin["cyrillic"]   += 1
        elif "ARABIC"     in uname: non_latin["arabic"]     += 1
        elif "CJK"        in uname: non_latin["chinese"]    += 1
        elif "GREEK"      in uname: non_latin["greek"]      += 1
        elif "HEBREW"     in uname: non_latin["hebrew"]     += 1
        elif "DEVANAGARI" in uname: non_latin["devanagari"] += 1
        elif "LATIN"      in uname: latin_count += 1

    total = sum(non_latin.values()) + latin_count
    if total > 0:
        dominant_non_latin = max(non_latin, key=non_latin.get)
        if non_latin[dominant_non_latin] / total > 0.4:
            # Non-Latin script dominates — map directly, no need for langdetect
            return {
                "cyrillic":   "Russian",
                "arabic":     "Arabic",
                "chinese":    "Chinese",
                "greek":      "Greek",
                "hebrew":     "Hebrew",
                "devanagari": "Hindi",
            }[dominant_non_latin]

    # --- Statistical detection via langdetect ---
    # Use up to 3000 chars; enough for accurate detection, not too slow
    detect_sample = sample[:3000].strip()
    if not detect_sample:
        return "Unknown"

    try:
        from langdetect import detect, DetectorFactory
        # Seed for determinism — langdetect is probabilistic by default
        DetectorFactory.seed = 0
        lang_code = detect(detect_sample)
        return _LANG_NAMES.get(lang_code.lower(), f"Unknown ({lang_code})")
    except Exception:
        # langdetect can fail on very short or symbol-heavy texts
        return "Unknown"


def get_language_code(pages_data: list) -> str:
    """
    Return the ISO-639-1 code used by deep-translator.
    Returns 'en' if English (skips translation), 'auto' if unknown.
    """
    sample = " ".join(p["text"] for p in pages_data)[:3000].strip()
    if not sample:
        return "auto"

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang_code = detect(sample).lower()
        # 'en' → skip translation. everything else → translate.
        return _TRANSLATE_CODES.get(lang_code, "auto")
    except Exception:
        return "auto"


# ── Translation ───────────────────────────────────────────────────────────────

def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate *text* to English using Google Translate (free, no API key).

    Called at ingestion time so every chunk is stored in English.
    This means English queries always match semantically, regardless of
    the source PDF language.

    Args:
        text:        The chunk text (any language).
        source_lang: ISO-639-1 code ('ru', 'zh-CN', …) or 'auto'.

    Returns:
        English translation, or the original text on failure.
    """
    if not text.strip():
        return text

    # Skip if already English (Latin-dominant, short texts often have mixed
    # Latin acronyms even in Russian SDS — skip short chunks to avoid noise)
    if len(text) < 20:
        return text

    # Quick check: if overwhelmingly Latin already, skip translation
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        latin_ratio = sum(
            1 for c in alpha_chars
            if "LATIN" in unicodedata.name(c, "")
        ) / len(alpha_chars)
        if latin_ratio > 0.85:
            return text           # already mostly Latin/English

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(
            source=source_lang, target="en"
        ).translate(text)
        return translated or text
    except Exception:
        # Translation is best-effort — never block ingestion
        return text


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving newlines so LangChain's
    RecursiveCharacterTextSplitter can use \\n\\n / \\n separators.
    """
    lines  = text.split("\n")
    lines  = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result.strip()


def extract_section_from_text(text: str) -> str:
    """Extract section title from the beginning of a text chunk."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if len(line) < 120 and line.isupper():
            return line
        if len(line) < 120 and re.search(
            r"\b(SECTION|HAZARD|FIRST AID|FIRE|HANDLING|STORAGE|TRANSPORT|DISPOSAL)\b",
            line, re.IGNORECASE,
        ):
            return line
    return "General"


# ── Misc ──────────────────────────────────────────────────────────────────────

def measure_time(func):
    """Decorator that returns (result, elapsed_seconds)."""
    def wrapper(*args, **kwargs):
        t0     = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - t0
    return wrapper
