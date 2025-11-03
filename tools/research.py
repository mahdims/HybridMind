"""
Research Tool - GPT Researcher Wrapper
Searches academic papers and online resources for algorithm-related content.
"""

from gpt_researcher import GPTResearcher
import asyncio
from typing import Optional


async def research_topic(query: str, max_results: int = 5, report_type: str = "research_report") -> str:
    """
    Search academic papers and online resources using GPT Researcher.

    Args:
        query: The research query/problem statement
        max_results: Maximum number of results to retrieve
        report_type: Type of report to generate (default: "research_report")

    Returns:
        Research context as a string containing relevant papers and information
    """
    try:
        # Initialize GPT Researcher
        researcher = GPTResearcher(query, report_type)

        # Conduct research
        await researcher.conduct_research()

        # Get research context
        context = researcher.get_research_context()

        return context if context else "No relevant research found."

    except Exception as e:
        print(f"Error during research: {str(e)}")
        return f"Research failed: {str(e)}"


async def research_with_sources(query: str, max_results: int = 5) -> dict:
    """
    Search and return both context and sources.

    Args:
        query: The research query/problem statement
        max_results: Maximum number of results to retrieve

    Returns:
        Dictionary containing 'context' and 'sources'
    """
    try:
        researcher = GPTResearcher(query, "research_report")
        await researcher.conduct_research()

        context = researcher.get_research_context()
        sources = researcher.get_research_sources()

        return {
            "context": context if context else "No relevant research found.",
            "sources": sources if sources else []
        }

    except Exception as e:
        print(f"Error during research: {str(e)}")
        return {
            "context": f"Research failed: {str(e)}",
            "sources": []
        }


def run_research(query: str, max_results: int = 5) -> str:
    """
    Synchronous wrapper for research_topic.

    Args:
        query: The research query/problem statement
        max_results: Maximum number of results to retrieve

    Returns:
        Research context as a string
    """
    return asyncio.run(research_topic(query, max_results))


if __name__ == "__main__":
    # Test the research tool
    test_query = "efficient algorithms for finding k-th smallest element in sorted arrays"
    print("Testing research tool...")
    result = run_research(test_query)
    print(f"\nResearch Results:\n{result[:500]}...")
