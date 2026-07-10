"""
Vector store — ChromaDB 0.5.x with cosine similarity and per-document collections.

Each uploaded PDF gets its own ChromaDB collection stored under data/chroma_db/.
Queries on one document never touch another document's vectors.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config.settings import config
from utilities.logger import get_logger


def _safe_name(name: str) -> str:
    """Convert a filename to a valid ChromaDB collection name (3-63 chars, alphanumeric + hyphens)."""
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", Path(name).stem)[:60]
    safe = safe.strip("-")
    if len(safe) < 3:
        safe = "sds-" + safe
    # Must start and end with alphanumeric
    if not safe[0].isalnum():
        safe = "s" + safe
    if not safe[-1].isalnum():
        safe = safe.rstrip("-") + "0"
    return safe or "sds-default"


class VectorStore:
    """
    ChromaDB-backed vector store. One collection per PDF document.

    Document isolation:
        call switch_collection(pdf_name) before ingesting a new PDF.
        All searches run ONLY against the active collection.
        SDS-A results can never appear when querying SDS-B.
    """

    def __init__(self, collection_name: str = "sds-default"):
        self.logger = get_logger(__name__)

        # Persistent client — stores data in data/chroma_db/, survives restarts
        # anonymized_telemetry=False silences the "Failed to send telemetry" noise
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._collection_name = _safe_name(collection_name)
        self._col = self._get_or_create(self._collection_name)
        self.logger.info(
            f"ChromaDB collection '{self._collection_name}' ready "
            f"({self._col.count()} chunks)"
        )

    # ── collection management ─────────────────────────────────────────────────

    def _get_or_create(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def switch_collection(self, new_name: str) -> None:
        """Switch active collection to a different document. Call before each new PDF."""
        self._collection_name = _safe_name(new_name)
        self._col = self._get_or_create(self._collection_name)
        self.logger.info(f"Switched to collection '{self._collection_name}'")

    def clear_collection(self) -> None:
        """Wipe the active collection."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._col = self._get_or_create(self._collection_name)
        self.logger.info(f"Collection '{self._collection_name}' cleared")

    # ── write ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Upsert chunks + embeddings into the active ChromaDB collection."""
        if not chunks:
            return

        ids        = [c["chunk_id"] for c in chunks]
        documents  = [c["text"] for c in chunks]
        metadatas  = [
            {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
             for k, v in c["metadata"].items()}
            for c in chunks
        ]

        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self.logger.info(
            f"Upserted {len(chunks)} chunks → '{self._collection_name}' "
            f"(total {self._col.count()})"
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        n_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return top-n similar chunks from the ACTIVE collection only."""
        if self._col.count() == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        n = min(n_results or config.TOP_K_RESULTS, self._col.count())
        return self._col.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_collection_count(self) -> int:
        return self._col.count()

    def collection_exists(self) -> bool:
        return self._col.count() > 0
