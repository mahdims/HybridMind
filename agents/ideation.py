"""
Ideation Agent - Generates algorithm ideas with self-reflection.
Each agent uses a different SOTA language model for diverse perspectives.
Supports up to 4 agents with varying capabilities including web search.
"""

import os
import re
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from utils import load_and_format_prompt
from config import Config


def parse_reflection_output(reflection_text: str) -> Dict[str, str]:
    """
    Parse reflection output to extract flaws and revised algorithm separately.

    Expected format:
        ---FLAWS---
        [flaws text]
        ---REVISED_ALGORITHM---
        [revised algorithm text]

    Args:
        reflection_text: Raw reflection output from LLM

    Returns:
        Dictionary with 'flaws' and 'revised_algorithm' keys
    """
    # Try to find the delimiters
    flaws_marker = "---FLAWS---"
    revised_marker = "---REVISED_ALGORITHM---"

    # Strategy 1: Try exact delimiter matching
    if flaws_marker in reflection_text and revised_marker in reflection_text:
        # Split by markers
        parts = reflection_text.split(revised_marker)
        if len(parts) >= 2:
            # Get the revised algorithm (everything after the marker)
            revised = parts[1].strip()

            # Get the flaws (between flaws marker and revised marker)
            flaws_part = parts[0].split(flaws_marker)
            flaws = flaws_part[1].strip() if len(flaws_part) > 1 else "No flaws section found"

            return {
                "flaws": flaws,
                "revised_algorithm": revised
            }

    # Strategy 2: Try case-insensitive and flexible matching
    flaws_pattern = r"---\s*FLAWS?\s*---"
    revised_pattern = r"---\s*REVISED[_\s]ALGORITHM\s*---"

    flaws_match = re.search(flaws_pattern, reflection_text, re.IGNORECASE)
    revised_match = re.search(revised_pattern, reflection_text, re.IGNORECASE)

    if flaws_match and revised_match:
        flaws_start = flaws_match.end()
        revised_start = revised_match.end()

        flaws = reflection_text[flaws_start:revised_match.start()].strip()
        revised = reflection_text[revised_start:].strip()

        return {
            "flaws": flaws,
            "revised_algorithm": revised
        }

    # Strategy 3: Fallback - look for common section headers
    # Try to find "improved version", "revised", "enhanced", etc.
    improved_patterns = [
        r"(?:improved|revised|enhanced|updated)\s+(?:version|algorithm|approach)",
        r"here\s+is\s+the\s+(?:improved|revised|enhanced)",
        r"(?:^|\n)#{1,3}\s*(?:improved|revised|enhanced)",
    ]

    for pattern in improved_patterns:
        match = re.search(pattern, reflection_text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Take everything after this marker
            revised = reflection_text[match.start():].strip()
            flaws = reflection_text[:match.start()].strip()

            print(f"  [Parser] ⚠️ Using fallback pattern: '{pattern}'")
            return {
                "flaws": flaws,
                "revised_algorithm": revised
            }

    # Strategy 4: Last resort - assume entire text is the revision
    print("  [Parser] ❌ WARNING: Could not parse reflection format!")
    print("  [Parser] Using entire text as revised algorithm (no separation)")

    return {
        "flaws": "Could not extract flaws section",
        "revised_algorithm": reflection_text.strip()
    }


def get_llm_for_agent(agent_id: int, temperature: float = 0.7):
    """
    Get configured LLM instance based on agent ID.

    Args:
        agent_id: 1 (OpenAI), 2 (Gemini), 3 (Claude), or 4 (Qwen)
        temperature: Temperature setting for the model

    Returns:
        Configured LLM instance
    """
    if agent_id == 1:
        # Agent 1: OpenAI GPT
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

    elif agent_id == 2:
        # Agent 2: Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )

    elif agent_id == 3:
        # Agent 3: Anthropic Claude
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

    elif agent_id == 4:
        # Agent 4: Alibaba Qwen with web search capability
        api_key = os.getenv("QWEN_API_KEY")
        model = os.getenv("QWEN_MODEL", "qwen-plus")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        enable_search = Config.QWEN_ENABLE_SEARCH

        if not api_key:
            raise ValueError("QWEN_API_KEY not found in environment variables")

        # Qwen uses OpenAI-compatible API with custom base URL
        # Enable web search if configured
        if enable_search:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                extra_body={"enable_search": True}
            )
        else:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url
            )

    else:
        raise ValueError(f"Invalid agent_id: {agent_id}. Must be 1, 2, 3, or 4.")


def ideation_agent(state: Dict, agent_id: int, num_reflections: int = 3) -> Dict:
    """
    Single agent: generate idea + reflection loops.
    Uses different LLM provider based on agent_id for diverse perspectives.

    Args:
        state: Current state containing problem, research_context, local_papers
               Can also contain custom prompts for evolutionary operators:
               - crossover_prompt: For crossover operation
               - custom_ideation_prompt: For fresh generation with variants
               - mutation_context: For mutation operation
        agent_id: Unique identifier for this agent (1=OpenAI, 2=Gemini, 3=Claude, 4=Qwen)
        num_reflections: Number of self-reflection iterations

    Returns:
        Dictionary containing agent_id, initial_idea, reflections, and final_idea
    """
    # Get the appropriate LLM for this agent
    llm = get_llm_for_agent(agent_id)

    # Display agent initialization
    agent_labels = {
        1: "SOTA Agent 1",
        2: "SOTA Agent 2 (with search)",
        3: "SOTA Agent 3",
        4: "SOTA Agent 4 (with search)"
    }
    print(f"[SOTA Agent {agent_id}] Initializing {agent_labels.get(agent_id, f'Agent {agent_id}')}")

    # Check for crossover operation (no reflection loops needed)
    if "crossover_prompt" in state:
        print(f"[SOTA Agent {agent_id}] Performing crossover operation...")
        response = llm.invoke(state["crossover_prompt"])
        return {
            "agent_id": agent_id,
            "initial_idea": response.content,
            "reflections": [],
            "final_idea": response.content
        }

    # Prepare context
    problem = state.get("problem", "No problem specified")
    research_context = state.get("research_context", "No research context available")
    local_papers = state.get("local_papers", [])

    # Use ALL context without truncation - modern LLMs can handle 100K+ tokens
    # Research context (online research if enabled)
    research_summary = research_context[:5000] if research_context else "No research available"

    # ALL paper summaries (each paper is ~4000 chars, so 5 papers = ~20K chars)
    # CRITICAL: Do NOT truncate! These are already summarized by Gemini to focus on algorithms
    if local_papers:
        papers_summary = "\n\n---PAPER---\n\n".join(local_papers)
        print(f"[SOTA Agent {agent_id}] Using {len(local_papers)} paper summaries (~{len(papers_summary):,} chars)")
    else:
        papers_summary = "No local papers available"

    # Check for custom ideation prompt (for fresh generation variants)
    if "custom_ideation_prompt" in state:
        idea_prompt = state["custom_ideation_prompt"]
    # Check for mutation context
    elif "mutation_context" in state:
        idea_prompt = state["mutation_context"]
    else:
        # Load initial ideation prompt from template
        idea_prompt = load_and_format_prompt(
            "ideation",
            agent_id=agent_id,
            problem=problem,
            research_context=research_summary,
            local_papers=papers_summary
        )

    print(f"\n[SOTA Agent {agent_id}] Generating initial idea...")
    initial_response = llm.invoke(idea_prompt)
    initial_idea = initial_response.content

    # Reflection loops
    reflections = []
    current_idea = initial_idea

    for i in range(num_reflections):
        print(f"[SOTA Agent {agent_id}] Reflection loop {i + 1}/{num_reflections}...")

        # Load reflection prompt from template
        reflection_prompt = load_and_format_prompt(
            "reflection",
            current_idea=current_idea
        )

        reflection_response = llm.invoke(reflection_prompt)
        reflection_raw = reflection_response.content

        # Parse reflection to extract flaws and revised algorithm separately
        parsed = parse_reflection_output(reflection_raw)

        # Store the full reflection (flaws + revised) for tracking
        reflections.append({
            "iteration": i + 1,
            "flaws": parsed["flaws"],
            "revised_algorithm": parsed["revised_algorithm"],
            "raw_output": reflection_raw  # Keep raw for debugging if needed
        })

        # IMPORTANT: Use ONLY the revised algorithm for next iteration
        # This ensures we don't carry forward critique/analysis text
        current_idea = parsed["revised_algorithm"]

        print(f"  ✓ Extracted clean revised algorithm ({len(current_idea)} chars)")

    print(f"[SOTA Agent {agent_id}] Completed {num_reflections} reflection loops")

    return {
        "agent_id": agent_id,
        "initial_idea": initial_idea,
        "reflections": reflections,
        "final_idea": current_idea  # This is now clean, without critique pollution
    }


def format_agent_output(agent_result: Dict) -> str:
    """
    Format agent output for display.

    Args:
        agent_result: Result dictionary from ideation_agent

    Returns:
        Formatted string representation
    """
    output = f"""
{'='*80}
AGENT {agent_result['agent_id']} OUTPUT
{'='*80}

INITIAL IDEA:
{agent_result['initial_idea']}

REFLECTION PROCESS:
"""
    for reflection in agent_result['reflections']:
        output += f"\n--- Reflection {reflection['iteration']} ---\n"
        output += f"IDENTIFIED FLAWS:\n{reflection['flaws']}\n\n"
        output += f"REVISED VERSION:\n{reflection['revised_algorithm']}\n"

    output += f"""
FINAL REFINED IDEA (Clean, without critique):
{agent_result['final_idea']}

{'='*80}
"""
    return output


if __name__ == "__main__":
    # Test the ideation agent
    from dotenv import load_dotenv
    load_dotenv()

    test_state = {
        "problem": "Design an efficient algorithm for finding the k-th smallest element in two sorted arrays.",
        "research_context": "Binary search techniques can be applied. Merge-based approaches have O(k) complexity.",
        "local_papers": ["Paper 1: Advanced sorting algorithms...", "Paper 2: Binary search optimizations..."]
    }

    result = ideation_agent(test_state, agent_id=1, num_reflections=2)
    print(format_agent_output(result))
