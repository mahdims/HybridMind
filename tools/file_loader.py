"""
File Loader - Local PDF Paper Loader
Loads and chunks local PDF papers for context in ideation.
"""

import os
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_local_papers(directory: str = "./data/papers", chunk_size: int = 2000) -> List[str]:
    """
    Load and chunk local PDF papers.

    Args:
        directory: Path to directory containing PDF files
        chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks from all papers
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Creating it...")
            os.makedirs(directory, exist_ok=True)
            return []

        # Check if directory has PDF files
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []

        print(f"Loading {len(pdf_files)} PDF files from {directory}...")

        # Load PDFs
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader,
            show_progress=True
        )
        docs = loader.load()

        if not docs:
            print("No documents loaded")
            return []

        # Simple chunking - take first chunk_size characters from each document
        chunks = []
        for doc in docs:
            content = doc.page_content[:chunk_size]
            if content.strip():
                chunks.append(content)

        print(f"Loaded {len(chunks)} chunks from papers")
        return chunks

    except Exception as e:
        print(f"Error loading local papers: {str(e)}")
        return []


def load_paper_chunks(directory: str = "./data/papers",
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200) -> List[Dict[str, str]]:
    """
    Load papers and create structured chunks with metadata.

    Args:
        directory: Path to directory containing PDF files
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of dictionaries containing chunk text and metadata
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return []

        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []

        print(f"Loading and chunking {len(pdf_files)} PDF files...")

        # Load PDFs
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader
        )
        docs = loader.load()

        if not docs:
            return []

        # Use text splitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Split documents
        chunks = []
        for doc in docs:
            splits = text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                chunks.append({
                    "text": split,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": i
                })

        print(f"Created {len(chunks)} chunks from papers")
        return chunks

    except Exception as e:
        print(f"Error loading paper chunks: {str(e)}")
        return []


def get_papers_summary(directory: str = "./data/papers") -> str:
    """
    Get a summary of available papers.

    Args:
        directory: Path to directory containing PDF files

    Returns:
        Summary string of available papers
    """
    try:
        if not os.path.exists(directory):
            return "No papers directory found."

        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        if not pdf_files:
            return "No PDF files found in papers directory."

        summary = f"Found {len(pdf_files)} papers:\n"
        for i, pdf in enumerate(pdf_files, 1):
            summary += f"{i}. {pdf}\n"

        return summary

    except Exception as e:
        return f"Error getting papers summary: {str(e)}"


if __name__ == "__main__":
    # Test the file loader
    print("Testing file loader...")
    print(get_papers_summary())
    chunks = load_local_papers()
    print(f"\nLoaded {len(chunks)} chunks")
    if chunks:
        print(f"First chunk preview: {chunks[0][:200]}...")
