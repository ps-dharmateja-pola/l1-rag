"""
Streamlit UI for SDS RAG Assistant.
Industrial / brutalist design cloned from the HTML reference.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import config
from ingestion.pdf_extractor import PDFExtractor
from ingestion.chunker import TextChunker
from models.embeddings import EmbeddingModel
from models.generator import GeneratorModel
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from utilities.logger import get_logger
from utilities.helpers import format_sources_for_display, detect_language, get_language_code

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@700;900&family=IBM+Plex+Mono:wght@500;700&family=Inter:wght@400;500&family=Archivo+Black&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #fbf9f9;
    color: #1b1c1c;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar: full black ── */
[data-testid="stSidebar"] {
    background-color: #000 !important;
    border-right: 2px solid #000 !important;
    min-width: 320px !important;
    max-width: 320px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Sidebar scrollbar ── */
[data-testid="stSidebar"]::-webkit-scrollbar { width: 6px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track { background: #000; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: #fff; }

/* ── Sidebar native widgets (uploader, buttons) ── */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1px dashed rgba(255,255,255,0.35) !important;
    border-radius: 0 !important;
    color: #fff !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #fff !important; }

[data-testid="stSidebar"] button {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    width: 100% !important;
    transition: background 0.1s, color 0.1s !important;
}
[data-testid="stSidebar"] button:hover {
    background: #333 !important;
    color: #fff !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background: #fff !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── Top app bar ── */
[data-testid="stHeader"] { display: none !important; }

/* ── Main content area ── */
div[data-testid="stAppViewBlockContainer"] {
    max-width: 900px;
    padding: 0 2rem 8rem 2rem;
    margin: 0 auto;
}

/* ── Custom scrollbar for main ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #fbf9f9; border-left: 1px solid #c4c7c7; }
::-webkit-scrollbar-thumb { background: #000; }

/* ── Hero wordmark ── */
.sds-hero { text-align: center; padding: 5rem 0 3rem; }
.sds-hero-solid {
    font-family: 'Archivo Black', sans-serif;
    font-size: 72px;
    line-height: 1;
    letter-spacing: -0.03em;
    color: #000;
    display: inline;
}
.sds-hero-outline {
    font-family: 'Archivo Black', sans-serif;
    font-size: 72px;
    line-height: 1;
    letter-spacing: -0.03em;
    -webkit-text-stroke: 2.5px #000;
    color: transparent;
    display: inline;
}
.sds-hero-subtitle {
    font-family: 'Archivo Narrow', sans-serif;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.2;
    color: #000;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 3rem;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
}

/* ── Bento suggestion cards ── */
div[data-testid="stAppViewBlockContainer"] button {
    text-align: left !important;
    background: #fbf9f9 !important;
    color: #747878 !important;
    border: 1px solid #c4c7c7 !important;
    border-top: none !important;
    border-radius: 0 !important;
    padding: 10px 28px 24px 28px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    display: block !important;
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    transition: background 0.07s, color 0.07s !important;
    box-shadow: none !important;
    margin-top: -1px !important;
}
div[data-testid="stAppViewBlockContainer"] button:hover {
    background: #000 !important;
    color: #fff !important;
    border-color: #000 !important;
}
/* Remove top margin between the HTML card-header and the button */
div[data-testid="stAppViewBlockContainer"] div[data-testid="stButton"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid #efeded !important;
    padding: 20px 0 !important;
    margin: 0 !important;
}

/* ── Chat input bar: full-width dark strip ── */
[data-testid="stBottom"] {
    background: #0a0a0a !important;
    border-top: 2px solid #000 !important;
    padding: 16px 24px !important;
}
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background: #0a0a0a !important;
}
/* Target every possible wrapper Streamlit puts around the textarea */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: #0a0a0a !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:active {
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 15px !important;
    padding: 0 !important;
    caret-color: #ffffff !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.25) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.25) !important;
}
/* Send button inside chat input */
[data-testid="stChatInput"] button {
    background: #fff !important;
    color: #000 !important;
    border-radius: 0 !important;
    border: none !important;
}
[data-testid="stChatInput"] button:hover {
    background: #e4e2e2 !important;
}
/* Override any Streamlit focus-ring that resets the color */
div[class*="stChatInput"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "pdf_extractor" not in st.session_state:
        st.session_state.pdf_extractor = PDFExtractor()
    if "text_chunker" not in st.session_state:
        st.session_state.text_chunker = TextChunker()
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()
    if "generator_model" not in st.session_state:
        st.session_state.generator_model = GeneratorModel()
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(st.session_state.vector_store)
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "doc_language" not in st.session_state:
        st.session_state.doc_language = None   # set during ingestion
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "history"


def reset_document():
    st.session_state.vector_store.clear_collection()
    st.session_state.document_loaded = False
    st.session_state.document_name   = None
    st.session_state.doc_language    = None
    st.session_state.chat_history    = []
    st.rerun()


def reset_chat():
    st.session_state.chat_history = []
    st.rerun()


# ── PDF processing ────────────────────────────────────────────────────────────
def process_pdf(pdf_file):
    """
    Full ingestion pipeline:
      1. Extract text (PyMuPDF)
      2. Detect language (character script analysis — instant)
      3. Chunk with RecursiveCharacterTextSplitter
      4. Translate non-English chunks to English (deep-translator)
      5. Embed English chunks (Ollama mxbai-embed-large, parallel batches)
      6. Store in ChromaDB (persistent, per-document collection)
    """
    temp_path = None
    try:
        # Save upload temporarily
        temp_path = Path(config.DATA_DIR) / pdf_file.name
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.session_state.document_name = pdf_file.name

        # Switch ChromaDB to a collection named after this document
        st.session_state.vector_store.switch_collection(pdf_file.name)
        # Re-bind retriever to the new collection
        st.session_state.retriever = Retriever(st.session_state.vector_store)

        with st.status("Processing Safety Data Sheet...", expanded=True) as status:

            # 1 — Extract
            status.write("📄 Extracting text from PDF...")
            pages_data = st.session_state.pdf_extractor.extract_text(temp_path)
            status.write(f"✓ Extracted {len(pages_data)} pages")

            # 2 — Detect language (instant, no network)
            lang_name = detect_language(pages_data)
            lang_code = get_language_code(pages_data)
            st.session_state.doc_language = lang_name
            needs_translation = lang_code not in ("en", "auto")
            status.write(f"✓ Detected language: **{lang_name}**")

            # 3+4 — Chunk + translate
            translate_label = (
                f"✂️ Chunking & translating ({lang_name} → English)..."
                if needs_translation else
                "✂️ Chunking text..."
            )
            status.write(translate_label)
            chunks = st.session_state.text_chunker.chunk_documents(
                pages_data,
                source_lang=lang_code,
                translate=needs_translation,
            )
            translated_count = sum(1 for c in chunks if c["metadata"].get("translated"))
            status.write(
                f"✓ {len(chunks)} chunks"
                + (f" ({translated_count} translated)" if translated_count else "")
            )

            # 5 — Embed (parallel batches)
            status.write("🔮 Generating embeddings...")
            texts      = [c["text"] for c in chunks]
            embeddings = st.session_state.embedding_model.embed_texts(texts)
            status.write(f"✓ {len(embeddings)} embeddings generated")

            # 6 — Store in ChromaDB
            status.write("💾 Storing in ChromaDB...")
            st.session_state.vector_store.add_chunks(chunks, embeddings)
            status.write(
                f"✓ {st.session_state.vector_store.get_collection_count()} chunks indexed"
            )

            status.update(
                label="✅ Document ready — ask your questions!",
                state="complete",
                expanded=False,
            )

        st.session_state.document_loaded = True
        st.session_state.processing      = False
        return True

    except Exception as e:
        import traceback
        st.error(f"Error processing PDF: {str(e)}")
        st.error(traceback.format_exc())
        st.session_state.processing      = False
        st.session_state.document_loaded = False
        return False
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # ── Wordmark ──
        st.markdown("""
<div style="padding:24px 24px 0 24px; background:#000;">
  <div style="display:flex; align-items:baseline; gap:6px;">
    <span style="background:#fff; color:#000; padding:4px 10px;
                 font-family:'Archivo Black',sans-serif; font-size:36px;
                 line-height:1; display:inline-block;">SDS</span>
    <span style="font-family:'Archivo Black',sans-serif; font-size:36px;
                 line-height:1; -webkit-text-stroke:1.5px #fff;
                 color:transparent; display:inline-block;">RAG</span>
  </div>

  <!-- Hazard stripe -->
  <div style="height:6px; margin-top:16px; width:100%;
              background: repeating-linear-gradient(
                45deg, #fff, #fff 4px, #000 4px, #000 8px
              );"></div>

  <p style="font-family:'IBM Plex Mono',monospace; font-size:9px;
            letter-spacing:0.22em; color:rgba(255,255,255,0.5);
            text-transform:uppercase; margin:12px 0 0 0;">
    RETRIEVAL-AUGMENTED-GENERATION
  </p>
</div>
""", unsafe_allow_html=True)

        # ── Navigation ──
        st.markdown("""
<div style="padding:24px 24px 0 24px;">
  <p style="font-family:'IBM Plex Mono',monospace; font-size:10px;
            letter-spacing:0.18em; color:rgba(255,255,255,0.35);
            text-transform:uppercase; margin:0 0 14px 0;">Navigation</p>
</div>
""", unsafe_allow_html=True)

        nav_items = [
            ("description",  "Safety Data Sheets", "sheets"),
            ("history",      "Analysis History",   "history"),
            ("terminal",     "System Status",      "status"),
        ]
        active = st.session_state.active_nav
        for icon, label, key in nav_items:
            is_active = active == key
            bg    = "rgba(255,255,255,1)" if is_active else "transparent"
            color = "#000" if is_active else "rgba(255,255,255,0.75)"
            border = "border-bottom:2px solid #000;" if is_active else ""
            st.markdown(f"""
<a href="#" style="display:flex; align-items:center; gap:14px;
   background:{bg}; color:{color}; padding:12px 24px;
   text-decoration:none; {border}
   font-family:'IBM Plex Mono',monospace; font-size:11px;
   letter-spacing:0.1em; text-transform:uppercase;">
  <span class="material-symbols-outlined" style="font-size:20px;">{icon}</span>
  {label}
</a>""", unsafe_allow_html=True)

        # ── Document Status ──
        st.markdown("""
<div style="margin:24px 24px 0 24px;">
  <p style="font-family:'IBM Plex Mono',monospace; font-size:10px;
            letter-spacing:0.15em; color:#fff;
            text-transform:uppercase; margin:0 0 14px 0;">Document Status</p>
""", unsafe_allow_html=True)

        if st.session_state.document_loaded:
            count = st.session_state.vector_store.get_collection_count()
            name  = st.session_state.document_name or ""
            lang  = st.session_state.doc_language or "Unknown"
            st.markdown(f"""
  <div style="border:1px solid rgba(255,255,255,0.25); padding:16px;">
    <p style="font-family:'IBM Plex Mono',monospace; font-size:10px;
              color:#fff; letter-spacing:0.08em; margin:0 0 4px 0;
              text-transform:uppercase;">✓ LOADED</p>
    <p style="font-family:'Inter',sans-serif; font-size:12px;
              color:rgba(255,255,255,0.8); margin:0 0 4px 0;
              word-break:break-all;">{name}</p>
    <p style="font-family:'IBM Plex Mono',monospace; font-size:10px;
              color:rgba(255,255,255,0.45); margin:0 0 2px 0;">{count} chunks indexed</p>
    <p style="font-family:'IBM Plex Mono',monospace; font-size:10px;
              color:rgba(255,255,255,0.35); margin:0;">Language: {lang}</p>
  </div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
  <div style="border:2px dashed rgba(255,255,255,0.25); padding:24px 16px;
              display:flex; flex-direction:column; align-items:center;
              justify-content:center; text-align:center; gap:10px;">
    <span class="material-symbols-outlined"
          style="color:rgba(255,255,255,0.35); font-size:32px;">upload_file</span>
    <p style="font-family:'IBM Plex Mono',monospace; font-size:10px;
              color:rgba(255,255,255,0.5); letter-spacing:0.05em; margin:0;
              line-height:1.6;">Upload an SDS PDF to ask questions.</p>
  </div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Upload + Process ──
        st.markdown("""
<div style="padding:0 24px; margin-top:auto; border-top:1px solid rgba(255,255,255,0.15);
            padding-top:16px; margin-top:24px;">
""", unsafe_allow_html=True)

        pdf_file = st.file_uploader(
            "Drop PDF here",
            type=["pdf"],
            label_visibility="collapsed",
            disabled=st.session_state.processing,
        )

        if pdf_file:
            if st.session_state.document_loaded:
                # Allow replacing the current document
                st.markdown(
                    "<p style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                    "color:rgba(255,255,255,0.5);margin:8px 0 4px;'>NEW FILE DETECTED</p>",
                    unsafe_allow_html=True,
                )
            if st.button("PROCESS PDF", type="primary", use_container_width=True, key="process_btn"):
                # Clear old data before processing new doc
                st.session_state.vector_store.clear_collection()
                st.session_state.document_loaded = False
                st.session_state.doc_language    = None
                st.session_state.chat_history    = []
                st.session_state.processing      = True
                process_pdf(pdf_file)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Utility buttons ──
        st.markdown("<div style='padding:0 24px 24px 24px; margin-top:12px;'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("CLEAR CHAT", use_container_width=True, key="clear_chat"):
                reset_chat()
        with c2:
            if st.button("RESET APP", use_container_width=True, key="reset_app"):
                reset_document()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Top app bar (rendered in main) ───────────────────────────────────────────
def render_top_bar():
    st.markdown("""
<div style="position:sticky; top:0; z-index:100;
            height:48px; background:#fbf9f9;
            border-bottom:2px solid #000;
            display:flex; align-items:center;
            justify-content:space-between;
            padding:0 24px; margin-bottom:0;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:11px;
               font-weight:700; letter-spacing:0.15em;
               text-transform:uppercase; color:#000;">SDS RAG SYSTEM</span>
  <div style="display:flex; align-items:center; gap:16px;">
    <span class="material-symbols-outlined" style="color:#000; font-size:18px; cursor:pointer;">settings</span>
    <span class="material-symbols-outlined" style="color:#000; font-size:18px; cursor:pointer;">help_center</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Bento card HTML helper ────────────────────────────────────────────────────
BENTO_CARDS = [
    ("language",        "SDS Language",       "What language is the SDS in?"),
    ("gavel",           "Regulations",        "What regulations does this SDS follow?"),
    ("verified_user",   "Compliance Details", "What compliance details are mentioned?"),
    ("public",          "Jurisdiction",       "What jurisdiction does this SDS apply to?"),
]


# ── Chat interface ────────────────────────────────────────────────────────────
def render_chat_interface():
    # Replay existing chat history
    for message in st.session_state.chat_history:
        avatar = "🧑" if message["role"] == "user" else "⚗️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                confidence = message.get("confidence", "Unknown")
                dot = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
                st.caption(f"{dot} Confidence: {confidence}")
                sources = message.get("sources", "")
                if sources and sources != "No relevant information found":
                    with st.expander("Sources"):
                        st.markdown(sources)

    # Process pending query
    if "pending_query" in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        del st.session_state.pending_query

        # ── Fast-path: language questions answered directly from detected metadata ──
        _lang_kw = ["language", "written in", "what language", "язык", "langue", "sprache"]
        if any(kw in query.lower() for kw in _lang_kw) and st.session_state.doc_language:
            lang     = st.session_state.doc_language
            doc_name = st.session_state.document_name or "the document"
            answer = (
                f"The Safety Data Sheet is written in **{lang}**.\n\n"
                f"This was determined by scanning the character script and vocabulary "
                f"of the extracted text from `{doc_name}`."
            )
            with st.chat_message("assistant", avatar="⚗️"):
                st.markdown(answer)
                st.caption("🟢 Confidence: High")
            st.session_state.chat_history.append({
                "role":       "assistant",
                "content":    answer,
                "confidence": "High",
                "sources":    f"Script + vocabulary analysis of {doc_name}",
                "latency":    0,
            })
            st.rerun()
            return
        with st.chat_message("assistant", avatar="⚗️"):
            with st.status("Searching Safety Data Sheet...", expanded=False) as status:
                try:
                    query_embedding = st.session_state.embedding_model.embed_query(query)
                    chunks, confidence, latency = st.session_state.retriever.retrieve_with_confidence(
                        query_embedding
                    )
                    above_threshold = (
                        chunks and min(chunk["score"] for chunk in chunks) <= config.SIMILARITY_THRESHOLD
                    )
                    status.update(label="✓ Context retrieved", state="complete")
                except Exception as e:
                    status.update(label="Retrieval failed", state="error")
                    st.error(f"Retrieval error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, retrieval error: {str(e)}",
                        "confidence": "Low",
                        "sources": "Error",
                        "latency": 0,
                    })
                    st.rerun()
                    return

            if above_threshold and chunks:
                context = "\n\n".join([c["text"] for c in chunks])
                placeholder = st.empty()
                full_response = ""
                start = time.time()
                try:
                    for chunk in st.session_state.generator_model.generate_stream(query, context):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    gen_latency = time.time() - start
                    sources = format_sources_for_display(chunks)
                    from utilities.logger import Logger
                    Logger.log_generation(query, full_response, gen_latency)
                    dot = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "⚪")
                    st.caption(f"{dot} Confidence: {confidence}")
                    with st.expander("Sources"):
                        st.markdown(sources)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "confidence": confidence,
                        "sources": sources,
                        "latency": latency,
                    })
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Generation error: {str(e)}",
                        "confidence": "Low",
                        "sources": "Error",
                        "latency": 0,
                    })
            else:
                placeholder = st.empty()
                refusal = st.session_state.generator_model.generate_refusal()
                full_response = ""
                for word in refusal.split(" "):
                    full_response += word + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.04)
                placeholder.markdown(refusal)
                st.caption("🔴 Confidence: Low")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": refusal,
                    "confidence": "Low",
                    "sources": "No relevant information found",
                    "latency": latency,
                })
        st.rerun()

    # ── Empty state: hero + bento grid ──
    if not st.session_state.chat_history:
        # Hero
        st.markdown("""
<div class="sds-hero">
  <div>
    <span class="sds-hero-solid">SDS</span>&nbsp;<span class="sds-hero-outline">RAG</span>
  </div>
</div>
<p class="sds-hero-subtitle">
  Upload a Safety Data Sheet PDF to ground your assistant in regulatory facts.
</p>
""", unsafe_allow_html=True)

        # 2 × 2 bento grid
        col1, col2 = st.columns(2, gap="small")
        cols = [col1, col2, col1, col2]

        for idx, (icon, title, question) in enumerate(BENTO_CARDS):
            with cols[idx]:
                # Styled card header (HTML, read-only)
                st.markdown(f"""
<div style="border:1px solid #c4c7c7; padding:28px 28px 12px 28px; margin-bottom:0;
            background:#fbf9f9;">
  <span class="material-symbols-outlined"
        style="font-size:28px; color:#000; display:block; margin-bottom:10px;">{icon}</span>
  <p style="font-family:'Archivo Narrow',sans-serif; font-size:18px; font-weight:700;
            color:#000; margin:0 0 6px 0; line-height:1.2;">{title}</p>
</div>""", unsafe_allow_html=True)
                # Plain-text button sits flush below the card header
                if st.button(f'"{question}"', key=f"bento_{idx}", use_container_width=True):
                    if not st.session_state.document_loaded:
                        st.warning("Please upload and process a Safety Data Sheet PDF first.")
                    else:
                        st.session_state.pending_query = question
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.rerun()

    # ── Chat input ──
    if prompt := st.chat_input("Ask a question about the SDS..."):
        if not st.session_state.document_loaded:
            st.warning("Please upload a Safety Data Sheet first.")
        else:
            st.session_state.pending_query = prompt
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    initialize_session_state()
    render_sidebar()
    render_top_bar()
    render_chat_interface()


if __name__ == "__main__":
    main()
