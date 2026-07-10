"""
Retrieval module for SDS RAG Assistant.

This module handles retrieval of relevant chunks with similarity threshold filtering.
"""

from typing import List, Dict, Any, Tuple
import time

from config.settings import config
from retrieval.vector_store import VectorStore
from utilities.logger import get_logger, Logger
from utilities.helpers import calculate_confidence_level


class Retriever:
    """Retrieves relevant chunks based on query embeddings."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance for searching
        """
        self.vector_store = vector_store
        self.logger = get_logger(__name__)
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
    
    def retrieve(self, query_embedding: List[float], 
                 n_results: int = None) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        Retrieve relevant chunks based on query embedding.
        
        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved_chunks, latency, above_threshold)
        """
        start_time = time.time()
        
        # Search vector store
        results = self.vector_store.search(query_embedding, n_results)
        
        # Process results
        chunks = []
        scores = []
        chunk_ids = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                chunk_data = {
                    "chunk_id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i]
                }
                chunks.append(chunk_data)
                scores.append(results['distances'][0][i])
                chunk_ids.append(results['ids'][0][i])
        
        latency = time.time() - start_time
        
        # Log retrieval metrics
        Logger.log_retrieval(
            query="embedded_query",
            scores=scores,
            chunk_ids=chunk_ids,
            latency=latency
        )
        
        # Scores are cosine DISTANCES (lower = more relevant).
        # A chunk passes if its distance is <= threshold (i.e. similar enough).
        above_threshold = False
        if scores:
            best_score = min(scores)  # lowest distance = best match
            above_threshold = best_score <= self.similarity_threshold
            self.logger.info(
                f"Best distance: {best_score:.3f}, threshold: {self.similarity_threshold}, "
                f"passes: {above_threshold}"
            )
        
        self.logger.info(f"Retrieved {len(chunks)} chunks in {latency:.3f}s")
        
        return chunks, latency, above_threshold
    
    def retrieve_with_confidence(self, query_embedding: List[float], 
                                 n_results: int = None) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Retrieve chunks and calculate confidence level.
        
        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved_chunks, confidence_level, latency)
        """
        chunks, latency, above_threshold = self.retrieve(query_embedding, n_results)
        
        scores = [chunk["score"] for chunk in chunks]
        confidence = calculate_confidence_level(scores)
        
        return chunks, confidence, latency
    
    def set_threshold(self, threshold: float):
        """
        Update the similarity threshold.
        
        Args:
            threshold: New similarity threshold (0-1)
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
            self.logger.info(f"Updated similarity threshold to {threshold}")
        else:
            self.logger.warning(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
