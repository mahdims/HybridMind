"""
File Loader - Local PDF Paper Loader
Loads and chunks local PDF papers for context in ideation.
Includes LLM-based summarization using Gemini 2.5 Flash for comprehensive paper understanding.
"""

import os
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


def summarize_paper_with_gemini(paper_content: str, paper_name: str) -> str:
    """
    Summarize a research paper using Gemini 2.0 Flash with focus on algorithmic mechanisms.

    Args:
        paper_content: Full text content of the paper (up to ~70K tokens / 280K chars)
        paper_name: Name of the paper file for logging

    Returns:
        Algorithm-focused summary (target: 3500-4000 chars, max: 4000) with detailed
        component descriptions, interactions, and pseudocode
    """
    try:
        # Initialize Gemini 2.0 Flash (fast and cost-effective for summarization)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Using Gemini 2.0 Flash
            temperature=0.1,  # Low temperature for factual summarization
            max_output_tokens=5000  # Target 4000 chars (~1000 tokens)
        )

        # Create summarization prompt focused on algorithmic mechanisms and details
        summarization_prompt = f"""You are an algorithm analysis expert extracting algorithmic details from research papers.

Your task: Extract and explain the ALGORITHMIC MECHANISMS from this paper in detail.

**TARGET LENGTH: 3500-4000 characters (STRICT MAXIMUM: 4000 characters)**

**WHAT TO FOCUS ON (Priority Order):**

1. **Algorithm Components & Mechanisms** (HIGHEST PRIORITY - ~40% of content)
   - What are the key algorithmic components/modules?
   - How does each component work internally (mechanism)?
   - What specific operations does each component perform?
   - What data structures are used in each component?

2. **Component Interactions** (CRITICAL - ~25% of content)
   - How do components interact with each other?
   - What information/data flows between components?
   - What is the execution order and control flow?
   - How do components communicate or coordinate?

3. **Pseudocode** (ESSENTIAL - ~20% of content)
   - Extract or reconstruct the main algorithm pseudocode
   - Include critical subroutines if essential
   - Show actual algorithmic steps with clarity

4. **Technical Details** (~10% of content)
   - Key parameters and their roles
   - Mathematical formulations (only if essential to understanding mechanism)
   - Complexity analysis (one sentence)

5. **Performance Observation** (~5% of content)
   - ONE general sentence: "outperforms X using Y mechanism" or "comparable to X"

**WHAT TO MINIMIZE OR SKIP:**

- ❌ Problem description (we already know the problem)
- ❌ Detailed experimental setup
- ❌ Benchmark instance descriptions
- ❌ Detailed performance numbers and tables
- ❌ Literature review or background
- ❌ Author names, paper metadata, citations

**OUTPUT STRUCTURE:**

## Main Algorithm
[Algorithm name and 1-sentence description of main idea]

## Core Components
[For each component: name, what it does, how it works internally, what operations it performs]

## Component Interactions & Flow
[How components work together, data/information flow between them, execution sequence, coordination mechanisms]

## Pseudocode
```
[Main algorithm pseudocode - be detailed and specific]
[Include key subroutines if critical]
```

## Technical Mechanisms
[Data structures, key parameters, essential mathematical details]

## Performance Note
[One general sentence about performance]

---

**PAPER CONTENT:**

{paper_content[:280000]}

---

**CRITICAL CONSTRAINTS:**
- Length: 3500-4000 characters (STRICT MAXIMUM: 4000)
- Focus: 90% on algorithmic mechanisms, components, and their interactions
- Include: Detailed pseudocode with specific steps
- Minimize: problem description, experiments, performance details"""

        print(f"    Summarizing with Gemini 2.5 Flash...", end=" ")

        # Get summary from Gemini
        response = llm.invoke(summarization_prompt)
        summary = response.content

        summary_length = len(summary)
        print(f"✓ ({summary_length:,} chars)")

        # Warn if summary is significantly shorter than target
        if summary_length < 2500:
            print(f"    ⚠️  WARNING: Summary shorter than expected (target: 3500-4000 chars)")

        return summary

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Summarization failed: {error_msg[:100]}")

        # Fallback: return original content truncated
        print(f"    📋 Falling back to raw content extraction")
        return paper_content[:20000]


def load_local_papers(
    directory: str = "./data/papers",
    chunk_size: int = 280000,
    use_summarization: bool = True
) -> List[str]:
    """
    Load local PDF papers with comprehensive content extraction and optional LLM summarization.

    Args:
        directory: Path to directory containing PDF files
        chunk_size: Maximum characters per paper (default: 280000 for ~70K tokens)
        use_summarization: If True, use Gemini 2.5 Flash to summarize papers (default: True)

    Returns:
        List of paper contents (summaries if use_summarization=True, raw content otherwise)
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

        if use_summarization:
            print(f"Loading {len(pdf_files)} PDF files from {directory}...")
            print("📚 Using Gemini 2.5 Flash for intelligent summarization (algorithm-focused)")
        else:
            print(f"Loading {len(pdf_files)} PDF files from {directory} (raw extraction)...")

        # Load PDFs individually to handle failures gracefully
        chunks = []
        successful = 0
        failed = 0
        failed_files = []

        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(directory, pdf_file)
                print(f"  Loading {pdf_file}...", end=" ")

                loader = UnstructuredPDFLoader(pdf_path)
                docs = loader.load()

                if docs:
                    # Concatenate all document parts (for multi-page PDFs)
                    full_content = ""
                    for doc in docs:
                        full_content += doc.page_content + "\n\n"

                    # Take substantial content (up to chunk_size, default 280K chars for ~70K tokens)
                    raw_content = full_content[:chunk_size].strip()
                    char_count = len(raw_content)
                    print(f"✓ ({char_count:,} chars extracted)")

                    if raw_content:
                        # Use LLM summarization if enabled
                        if use_summarization:
                            summary = summarize_paper_with_gemini(raw_content, pdf_file)
                            chunks.append(summary)
                            successful += 1
                        else:
                            # Use raw content
                            chunks.append(raw_content)
                            successful += 1
                    else:
                        print("⚠️ No content extracted")
                        failed += 1
                        failed_files.append((pdf_file, "No content extracted"))
                else:
                    print("⚠️ No content extracted")
                    failed += 1
                    failed_files.append((pdf_file, "No content extracted"))

            except Exception as e:
                print(f"❌ Failed: {str(e)[:50]}")
                failed += 1
                failed_files.append((pdf_file, str(e)))

        print(f"\n✓ Successfully loaded {successful}/{len(pdf_files)} PDF files")
        if use_summarization:
            print(f"✓ Generated {len(chunks)} AI-powered summaries (algorithm-focused)")
        else:
            print(f"✓ Extracted {len(chunks)} raw content chunks from papers")

        if failed > 0:
            print(f"\n⚠️ Failed to load {failed} files:")
            for filename, error in failed_files:
                print(f"  - {filename}: {error[:60]}")

            if "tesseract" in str(failed_files).lower():
                print(f"\n💡 TIP: Some PDFs require tesseract (OCR tool)")
                print(f"   Install: https://github.com/tesseract-ocr/tesseract")
                print(f"   Or skip problematic PDFs and use what loaded successfully")

        return chunks

    except Exception as e:
        print(f"Error in PDF loading system: {str(e)}")
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
