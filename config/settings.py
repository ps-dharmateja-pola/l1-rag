"""
config/settings.py — Central configuration for SDS RAG Assistant.

All values are hardcoded here. To change a setting, edit this file directly.
No .env file is needed.
"""

from pathlib import Path


class Config:

    # ── Paths ──────────────────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = PROJECT_ROOT / "data"          # PDF temp files + vector store
    LOGS_DIR     = PROJECT_ROOT / "logs"          # sds_rag.log
    VECTOR_DIR   = DATA_DIR / "vector_store"      # per-document .npy + .json files (NumPy fallback)
    CHROMA_DIR   = DATA_DIR / "chroma_db"         # ChromaDB persistent storage

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Ollama (fully local — no API key needed) ───────────────────────────────
    OLLAMA_BASE_URL  = "http://localhost:11434"
    EMBEDDING_MODEL  = "mxbai-embed-large"   # 1024-dim embeddings
    GENERATION_MODEL = "llama3.2:3b"         # answer generation

    # ── Chunking ───────────────────────────────────────────────────────────────
    CHUNK_SIZE    = 600   # characters per chunk
    CHUNK_OVERLAP = 80    # overlap between consecutive chunks

    # ── Retrieval ──────────────────────────────────────────────────────────────
    TOP_K_RESULTS        = 10     # number of chunks retrieved per query
    SIMILARITY_THRESHOLD = 0.75   # cosine distance cutoff (0=identical, 1=unrelated)
                                  # chunks with distance > 0.75 are ignored

    # ── Generation ─────────────────────────────────────────────────────────────
    TEMPERATURE = 0.05   # low = more focused, deterministic answers
    MAX_TOKENS  = 600    # max tokens in the generated answer

    # ── System prompt ──────────────────────────────────────────────────────────
    SYSTEM_PROMPT = (
        "You are a Safety Data Sheet (SDS) reading assistant.\n"
        "Answer ONLY from the retrieved context provided in the prompt.\n"
        "Rules:\n"
        "1. Never use outside knowledge. Never hallucinate.\n"
        "2. If the answer is in the context, state it clearly and cite the page number.\n"
        "3. The context may have been translated from another language — treat it as authoritative.\n"
        "4. If the answer is absent from the context reply exactly: "
        "\"I couldn't find enough information in the uploaded Safety Data Sheet "
        "to answer this question.\"\n"
        "5. ALWAYS respond in English only.\n"
        "6. For safety/precaution questions be thorough — list all hazards, "
        "first-aid steps, PPE requirements, and precautionary measures you find.\n"
        "Be concise. No disclaimers."
    )

    # ── UI ─────────────────────────────────────────────────────────────────────
    APP_TITLE = "SDS RAG Assistant"

    EXAMPLE_QUESTIONS = [
        "What language is the SDS in?",
        "What regulations does this SDS follow?",
        "What compliance details are mentioned?",
        "What jurisdiction does this SDS apply to?",
    ]

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_FILE   = LOGS_DIR / "sds_rag.log"
    LOG_LEVEL  = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


config = Config()
