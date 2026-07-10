"""
PDF text extraction module for SDS RAG Assistant.

This module handles extraction of text from PDF files using PyMuPDF,
preserving page numbers and section information.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import uuid

from utilities.logger import get_logger
from utilities.helpers import clean_text, extract_section_from_text


class PDFExtractor:
    """Extracts text and metadata from PDF files."""
    
    def __init__(self):
        """Initialize the PDF extractor."""
        self.logger = get_logger(__name__)
    
    def extract_text(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-by-page metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text and metadata for each page
        """
        self.logger.info(f"Extracting text from: {pdf_path}")
        
        document = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(document)):
            page = document[page_num]
            text = page.get_text()
            
            if text.strip():
                cleaned_text = clean_text(text)
                section = extract_section_from_text(cleaned_text)
                
                page_data = {
                    "text": cleaned_text,
                    "page_number": page_num + 1,  # 1-indexed for display
                    "section": section,
                    "document_name": pdf_path.name,
                }
                pages_data.append(page_data)
        
        document.close()
        self.logger.info(f"Extracted {len(pages_data)} pages from PDF")
        
        return pages_data
    
    def extract_text_with_progress(self, pdf_path: Path, progress_callback=None) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with progress reporting.
        
        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of dictionaries containing text and metadata for each page
        """
        self.logger.info(f"Extracting text from: {pdf_path}")
        
        document = fitz.open(pdf_path)
        total_pages = len(document)
        pages_data = []
        
        for page_num in range(total_pages):
            page = document[page_num]
            text = page.get_text()
            
            if text.strip():
                cleaned_text = clean_text(text)
                section = extract_section_from_text(cleaned_text)
                
                page_data = {
                    "text": cleaned_text,
                    "page_number": page_num + 1,
                    "section": section,
                    "document_name": pdf_path.name,
                }
                pages_data.append(page_data)
            
            # Report progress
            if progress_callback:
                progress = (page_num + 1) / total_pages
                progress_callback(progress, f"Processing page {page_num + 1}/{total_pages}")
        
        document.close()
        self.logger.info(f"Extracted {len(pages_data)} pages from PDF")
        
        return pages_data
