# SDS RAG Assistant

Upload a Safety Data Sheet PDF and ask questions about it in plain English.
Supports English, Russian, and other languages. Runs fully locally via Ollama.

---

## Setup

```bash
pip install -r requirements.txt

ollama serve
ollama pull mxbai-embed-large
ollama pull llama3.2:3b

streamlit run ui/app.py
```

---

## Project Structure

```
l1-rag/
├── ui/
│   └── app.py                  Entry point. Streamlit interface.
│
├── ingestion/
│   ├── pdf_extractor.py        Extracts text from PDF page by page.
│   └── chunker.py              Splits text into chunks. Translates non-English to English.
│
├── models/
│   ├── embeddings.py           Converts text to vectors via Ollama mxbai-embed-large.
│   └── generator.py            Sends context + question to Ollama llama3.2:3b, streams answer.
│
├── retrieval/
│   ├── vector_store.py         Stores vectors on disk. Searches with NumPy cosine similarity.
│   └── retriever.py            Embeds query, retrieves top-k chunks, assigns confidence score.
│
├── config/
│   └── settings.py             All settings in one place. Edit here to change models or params.
│
├── utilities/
│   ├── helpers.py              Language detection, translation, text cleaning, formatting.
│   └── logger.py               File + console logging. UTF-8 safe on Windows.
│
├── data/
│   └── vector_store/           Auto-created. One subfolder per uploaded PDF.
│       └── <document_name>/
│           ├── embeddings.npy  NumPy matrix of chunk vectors (float32, shape: n×1024)
│           ├── chunks.json     Chunk text, page number, section, translation flag
│           └── ids.json        Chunk IDs matching embeddings matrix rows
│
├── logs/
│   └── sds_rag.log             Runtime log. Latency, retrieval scores, errors.
│
├── requirements.txt
└── README.md
```

---

## Ingestion Pipeline  *(what happens when you click PROCESS PDF)*

```
process_pdf()                               ui/app.py

  PDFExtractor.extract_text()               ingestion/pdf_extractor.py
    fitz.open(pdf)                          PyMuPDF — opens the file
    page.get_text()                         extracts raw text per page
    clean_text()                            removes extra whitespace, keeps newlines
    extract_section_from_text()             detects section heading (HAZARDS, STORAGE, etc.)
    → returns list of pages: [{text, page_number, section, document_name}]

  detect_language()                         utilities/helpers.py
    counts Unicode script chars             Cyrillic? → Russian. Latin? → check keywords.
    keyword matching                        "safety","hazard" → English; "sicherheit" → German
    → stores result in session_state.doc_language

  VectorStore.switch_collection(pdf_name)   retrieval/vector_store.py
    loads data/vector_store/<pdf_name>/     or creates empty if first time
    → all future searches use ONLY this document's vectors

  TextChunker.chunk_documents()             ingestion/chunker.py
    RecursiveCharacterTextSplitter          splits at \n\n → \n → ". " → " "
    chunk_size=600, overlap=80
    if language != English:
      translate_to_english()               utilities/helpers.py
        GoogleTranslator (deep-translator)  free, no key, online
        skips chunks already >85% Latin    fast-path skip
    → each chunk: {chunk_id, text (English), original_text (native), metadata}

  EmbeddingModel.embed_texts()              models/embeddings.py
    batches of 8 chunks, 4 parallel workers ThreadPoolExecutor
    POST /api/embed  to Ollama              model: mxbai-embed-large
    → list of 1024-dim float vectors, one per chunk

  VectorStore.add_chunks()                  retrieval/vector_store.py
    np.vstack → embeddings.npy              saved to disk immediately
    json.dump → chunks.json + ids.json
    → document is now searchable
```

---

## Query Pipeline  *(what happens when you ask a question)*

```
render_chat_interface()                     ui/app.py

  FAST PATH — language questions only
    if "language" in query:
      return session_state.doc_language     no LLM, no embedding, instant answer
      (detected at ingestion, stored in session state)

  NORMAL PATH — all other questions

    EmbeddingModel.embed_query()            models/embeddings.py
      POST /api/embed to Ollama             same model as ingestion
      → 1024-dim vector for the question

    Retriever.retrieve_with_confidence()    retrieval/retriever.py
      VectorStore.search()                  retrieval/vector_store.py
        cosine_similarity(query, all_chunks)  NumPy dot product
        top 10 chunks by similarity
        distance = 1 − similarity           0 = identical, 1 = unrelated
      threshold check: min_distance ≤ 0.75?
        YES → pass chunks to LLM
        NO  → return refusal message
      confidence: <0.4 → High  <0.7 → Medium  else → Low

    GeneratorModel.generate_stream()        models/generator.py
      builds prompt:
        system prompt (answer only from context, English only, be thorough)
        === SDS CONTEXT === + retrieved chunk texts
        Question: <user query>
        Answer (in English):
      POST /api/generate  stream=True       Ollama llama3.2:3b
      keep_alive=10m                        model stays loaded — no cold-start penalty
      yields tokens as they arrive          streamed to UI

    UI renders answer token by token
    shows 🟢🟡🔴 confidence badge
    shows source citations (document, page, section)
```

---

## Configuration  —  `config/settings.py`

| Setting | Default | What it controls |
|---|---|---|
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Ollama model used for embeddings |
| `GENERATION_MODEL` | `llama3.2:3b` | Ollama model used for generation |
| `CHUNK_SIZE` | `600` | Max characters per chunk |
| `CHUNK_OVERLAP` | `80` | Characters shared between consecutive chunks |
| `TOP_K_RESULTS` | `10` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.75` | Max cosine distance to accept a chunk |
| `TEMPERATURE` | `0.05` | Generation randomness (0 = deterministic) |
| `MAX_TOKENS` | `600` | Max answer length in tokens |

---

## Document Isolation

Each PDF gets its own folder: `data/vector_store/<document_name>/`

Calling `VectorStore.switch_collection(pdf_name)` loads only that document's vectors.
Queries on Document A never search Document B's vectors.

---
