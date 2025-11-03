"""
Ideation Agent - Generates algorithm ideas with self-reflection.
Each agent uses a different SOTA language model for diverse perspectives.
Supports up to 4 agents with varying capabilities including web search.
"""

import os
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from utils import load_and_format_prompt
from config import Config


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

    # Prepare context
    problem = state.get("problem", "No problem specified")
    research_context = state.get("research_context", "No research context available")
    local_papers = state.get("local_papers", [])

    # Truncate context to avoid token limits
    research_summary = research_context[:1500] if research_context else "No research available"
    papers_summary = "\n".join(local_papers[:2])[:1000] if local_papers else "No local papers available"

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
        reflection = reflection_response.content
        reflections.append({
            "iteration": i + 1,
            "analysis": reflection
        })

        # Update current idea with improvements
        current_idea = reflection

    print(f"[SOTA Agent {agent_id}] Completed {num_reflections} reflection loops")

    return {
        "agent_id": agent_id,
        "initial_idea": initial_idea,
        "reflections": reflections,
        "final_idea": current_idea
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
        output += f"{reflection['analysis']}\n"

    output += f"""
FINAL REFINED IDEA:
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
