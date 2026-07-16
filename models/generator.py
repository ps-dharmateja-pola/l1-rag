"""
Generation model for SDS RAG Assistant — Ollama (local, no API key needed).

Uses llama3.2:3b via the Ollama /api/generate endpoint with streaming.
keep_alive=10m keeps the model loaded in RAM between requests so subsequent
queries don't pay the cold-start penalty.
"""

from typing import Iterator
import json
import time
import requests

from config.settings import config
from utilities.logger import get_logger, Logger


class GeneratorModel:
    """Generates answers via local Ollama llama3.2:3b."""

    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name    = model_name or config.GENERATION_MODEL
        self.base_url      = base_url   or config.OLLAMA_BASE_URL
        self.logger        = get_logger(__name__)
        self.system_prompt = config.SYSTEM_PROMPT
        self.api_url       = f"{self.base_url}/api/generate"
        self.logger.info(f"Initialized Ollama generator: {self.model_name}")

    # ── streaming (used by UI) ────────────────────────────────────────────────
    def generate_stream(self, query: str, context: str) -> Iterator[str]:
        """Yield text tokens as they arrive from Ollama."""
        payload = self._build_payload(query, context, stream=True)
        try:
            resp = requests.post(self.api_url, json=payload, stream=True, timeout=180)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    text  = chunk.get("response", "")
                    if text:
                        yield text
        except Exception as e:
            self.logger.error(f"Ollama stream error: {e}")
            yield "I encountered an error while generating a response."

    # ── non-streaming (used by tests / debug) ─────────────────────────────────
    def generate(self, query: str, context: str) -> str:
        """Return the full answer as a single string."""
        start   = time.time()
        payload = self._build_payload(query, context, stream=False)
        try:
            resp    = requests.post(self.api_url, json=payload, timeout=180)
            resp.raise_for_status()
            text    = resp.json().get("response", "").strip()
            latency = time.time() - start
            Logger.log_generation(query, text, latency)
            self.logger.info(f"Generated response in {latency:.2f}s")
            return text
        except Exception as e:
            self.logger.error(f"Ollama generation error: {e}")
            return "I encountered an error while generating a response."

    def generate_refusal(self) -> str:
        return (
            "I couldn't find enough information in the uploaded "
            "Safety Data Sheet to answer this question."
        )

    # ── internals ─────────────────────────────────────────────────────────────
    def _build_payload(self, query: str, context: str, stream: bool) -> dict:
        lang_hint = ""
        if any(kw in query.lower() for kw in ["language", "written in", "what language"]):
            lang_hint = (
                "\n[LANGUAGE NOTE] Examine the script of the CONTEXT text. "
                "Cyrillic = Russian. Latin = English"
                "Report what you actually observe.\n"
            )

        prompt = (
            "=== SDS CONTEXT — your ONLY source of truth ===\n"
            f"{context}\n"
            "=== END OF CONTEXT ===\n"
            f"{lang_hint}\n"
            "Using ONLY the context above, answer the question below in English. "
            "If the answer is absent from the context, say so explicitly. "
            "For safety/precaution questions be thorough — list every relevant "
            "hazard, first-aid step, PPE requirement, and precautionary measure "
            "you find.\n\n"
            f"Question: {query}\n\n"
            "Answer (in English):"
        )

        return {
            "model":      self.model_name,
            "prompt":     prompt,
            "system":     self.system_prompt,
            "stream":     stream,
            "keep_alive": "10m",
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS,
                "num_ctx":     2048,
                "top_k":       20,
                "top_p":       0.85,
            },
        }
