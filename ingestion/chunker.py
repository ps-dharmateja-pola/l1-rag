"""
Text chunker for SDS RAG Assistant.

Multilingual support:
  After splitting, each chunk is optionally translated to English before
  being stored. This ensures English queries always match semantically,
  even when the source PDF is in Russian, Chinese, Arabic, etc.

  The original (untranslated) text is preserved in chunk["original_text"]
  so the UI can display it if desired. The chunk["text"] field always
  contains English (used for embedding + LLM context).
"""

from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import config
from utilities.logger import get_logger
from utilities.helpers import translate_to_english


class TextChunker:
    """Chunk pages and optionally translate each chunk to English."""

    def __init__(
        self,
        chunk_size:    Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size    = chunk_size    or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.logger        = get_logger(__name__)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size      = self.chunk_size,
            chunk_overlap   = self.chunk_overlap,
            length_function = len,
            separators      = ["\n\n", "\n", ". ", " ", ""],
        )

    # ── public API ────────────────────────────────────────────────────────────

    def chunk_documents(
        self,
        pages_data:   List[Dict[str, Any]],
        source_lang:  str = "auto",      # ISO-639-1 code or 'auto'
        translate:    bool = True,        # False for English-only PDFs (faster)
    ) -> List[Dict[str, Any]]:
        """
        Split pages into chunks, translate to English, return chunk list.

        Each chunk dict:
          chunk_id      – sequential string ID
          text          – English text (used for embedding + retrieval)
          original_text – original language text (for display)
          metadata      – page_number, section, document_name, translated (bool)
        """
        self.logger.info(
            f"Chunking {len(pages_data)} pages "
            f"(translate={translate}, source_lang={source_lang})"
        )
        chunks   = []
        chunk_id = 0

        for page_data in pages_data:
            raw_text = page_data["text"]
            metadata = {
                "page_number":   page_data["page_number"],
                "section":       page_data["section"],
                "document_name": page_data["document_name"],
            }

            for raw_chunk in self._splitter.split_text(raw_text):
                chunk_id += 1

                if translate:
                    en_text    = translate_to_english(raw_chunk, source_lang)
                    translated = en_text != raw_chunk
                else:
                    en_text    = raw_chunk
                    translated = False

                chunk_meta = {**metadata, "translated": translated}

                chunks.append({
                    "chunk_id":      str(chunk_id),
                    "text":          en_text,        # always English
                    "original_text": raw_chunk,      # native language
                    "metadata":      chunk_meta,
                })

        self.logger.info(
            f"Created {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c['metadata']['translated'])} translated)"
        )
        return chunks
