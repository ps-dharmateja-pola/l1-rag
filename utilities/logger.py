"""
Logging utilities for SDS RAG Assistant.

This module provides a centralized logging configuration for the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import config


class Logger:
    """Centralized logger for SDS RAG Assistant."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Name of the logger (typically __name__ of the calling module)
            
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, config.LOG_LEVEL))
            
            # File handler — UTF-8 encoding to support any language in logged text
            file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
            file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
            file_formatter = logging.Formatter(config.LOG_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler — UTF-8 safe on Windows
            try:
                console_stream = open(
                    sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False
                )
            except Exception:
                console_stream = sys.stdout
            console_handler = logging.StreamHandler(console_stream)
            console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
            console_formatter = logging.Formatter(config.LOG_FORMAT)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def log_retrieval(cls, query: str, scores: list, chunk_ids: list, latency: float):
        """
        Log retrieval metrics for evaluation.
        
        Args:
            query: The user's query
            scores: List of similarity scores
            chunk_ids: List of retrieved chunk IDs
            latency: Retrieval latency in seconds
        """
        logger = cls.get_logger("retrieval")
        logger.info(f"Query: {query}")
        logger.info(f"Scores: {scores}")
        logger.info(f"Chunk IDs: {chunk_ids}")
        logger.info(f"Latency: {latency:.3f}s")
        logger.info(f"Average Score: {sum(scores)/len(scores) if scores else 0:.3f}")
    
    @classmethod
    def log_generation(cls, query: str, response: str, latency: float):
        """
        Log generation metrics for evaluation.
        
        Args:
            query: The user's query
            response: The generated response
            latency: Generation latency in seconds
        """
        logger = cls.get_logger("generation")
        logger.info(f"Query: {query}")
        logger.info(f"Response Length: {len(response)} characters")
        logger.info(f"Generation Latency: {latency:.3f}s")


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    return Logger.get_logger(name)
