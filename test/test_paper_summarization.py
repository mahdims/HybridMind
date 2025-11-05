"""
Test script for paper summarization feature using Gemini 2.5 Flash.
This script loads PDFs from data/papers/ and generates AI-powered summaries.
"""

import os
from dotenv import load_dotenv
from tools.file_loader import load_local_papers

def test_paper_summarization():
    """Test the paper summarization feature."""
    print("="*80)
    print("TESTING PAPER SUMMARIZATION WITH GEMINI 2.5 FLASH")
    print("="*80)

    # Load environment variables
    load_dotenv()

    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY not found in .env file")
        print("Please set GOOGLE_API_KEY in your .env file to use Gemini summarization")
        return

    print("\n✓ Google API key found")

    # Test with summarization enabled
    print("\n" + "="*80)
    print("TEST 1: Load papers WITH summarization (Gemini 2.5 Flash)")
    print("="*80)

    summaries = load_local_papers(use_summarization=True)

    if summaries:
        print(f"\n✓ Generated {len(summaries)} summaries")

        # Display details of each summary
        for i, summary in enumerate(summaries, 1):
            print(f"\n{'='*80}")
            print(f"SUMMARY {i}:")
            print(f"{'='*80}")
            print(f"Length: {len(summary):,} characters")
            print(f"\nFirst 500 characters:")
            print("-" * 80)
            print(summary[:500])
            print("-" * 80)

            # Check for key sections
            has_pseudocode = "pseudocode" in summary.lower() or "algorithm" in summary.lower()
            has_complexity = "complexity" in summary.lower() or "O(" in summary
            has_method = "method" in summary.lower() or "approach" in summary.lower()

            print(f"\nContent analysis:")
            print(f"  ✓ Contains pseudocode/algorithm: {has_pseudocode}")
            print(f"  ✓ Contains complexity analysis: {has_complexity}")
            print(f"  ✓ Contains method description: {has_method}")

            if len(summary) < 5000:
                print(f"  ⚠️ WARNING: Summary is shorter than expected ({len(summary)} chars)")
            elif len(summary) >= 7000:
                print(f"  ✓ Summary meets length requirement (≥7000 chars)")
    else:
        print("\n❌ No summaries generated")
        print("Possible issues:")
        print("  1. No PDF files in data/papers/ directory")
        print("  2. PDF loading failed")
        print("  3. Gemini API error")

    # Test with summarization disabled (for comparison)
    print("\n" + "="*80)
    print("TEST 2: Load papers WITHOUT summarization (raw extraction)")
    print("="*80)

    raw_contents = load_local_papers(use_summarization=False)

    if raw_contents:
        print(f"\n✓ Extracted {len(raw_contents)} raw content chunks")
        for i, content in enumerate(raw_contents, 1):
            print(f"  Raw content {i}: {len(content):,} characters")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Summaries generated: {len(summaries) if summaries else 0}")
    print(f"Raw contents extracted: {len(raw_contents) if raw_contents else 0}")

    if summaries and raw_contents:
        avg_summary_len = sum(len(s) for s in summaries) / len(summaries)
        avg_raw_len = sum(len(c) for c in raw_contents) / len(raw_contents)
        print(f"Average summary length: {avg_summary_len:,.0f} chars")
        print(f"Average raw content length: {avg_raw_len:,.0f} chars")
        print(f"Compression ratio: {avg_raw_len/avg_summary_len:.1f}x")

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    test_paper_summarization()
