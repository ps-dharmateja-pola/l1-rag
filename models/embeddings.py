"""
Embedding model for SDS RAG Assistant.

Uses /api/embed (Ollama 0.3+) with parallel batched requests.
Splits large payloads into smaller sub-batches and embeds them
concurrently with ThreadPoolExecutor to cut wall-clock time.
"""

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time

from config.settings import config
from utilities.logger import get_logger

# How many chunks to send per HTTP call.
# Ollama processes the whole batch sequentially inside one call anyway,
# so smaller batches + parallel calls is faster than one giant batch.
_BATCH_SIZE = 8
_MAX_WORKERS = 4   # parallel HTTP connections to Ollama


class EmbeddingModel:
    """Generates embeddings using Ollama's mxbai-embed-large model."""

    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.base_url   = base_url   or config.OLLAMA_BASE_URL
        self.logger     = get_logger(__name__)
        self.api_url    = f"{self.base_url}/api/embed"
        self.logger.info(f"Initialized embedding model: {self.model_name}")

    # ── public API ────────────────────────────────────────────────────────────
    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed all texts using parallel sub-batches.

        Splits `texts` into chunks of _BATCH_SIZE, fires them all in
        parallel via ThreadPoolExecutor, then re-assembles in order.
        """
        if not texts:
            return []

        # Build ordered sub-batches: [(start_idx, [text, ...]), ...]
        batches = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batches.append((i, texts[i: i + _BATCH_SIZE]))

        results: dict[int, List[List[float]]] = {}

        def _embed_batch(start: int, batch: List[str]):
            payload = {"model": self.model_name, "input": batch}
            resp = requests.post(self.api_url, json=payload, timeout=300)
            resp.raise_for_status()
            embs = resp.json().get("embeddings", [])
            if len(embs) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, got {len(embs)}"
                )
            return start, embs

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_embed_batch, start, batch): start
                for start, batch in batches
            }
            for fut in as_completed(futures):
                start, embs = fut.result()   # raises on error
                results[start] = embs

        # Reassemble in original order
        ordered: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            ordered.extend(results[i])

        self.logger.info(
            f"Parallel-embedded {len(texts)} texts in "
            f"{time.time()-t0:.2f}s (dim={len(ordered[0]) if ordered else 'n/a'})"
        )
        return ordered
