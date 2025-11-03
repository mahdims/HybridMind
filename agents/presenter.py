"""
Presenter Agent - Generates structured markdown presentations
Creates comprehensive reports from agent ideas and evaluations.
Uses Google Gemini for fast, cost-effective presentation generation.
"""

import os
from typing import Dict, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from .evaluator import parse_evaluation_scores, calculate_weighted_scores
from utils import load_and_format_prompt


def get_llm(temperature: float = 0.5, model: str = None):
    """Get configured LLM instance for presentation generation (uses Gemini)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


def generate_presentation(state: Dict) -> str:
    """
    Create markdown presentation of results.

    Args:
        state: Current state containing problem, agent_ideas, evaluations

    Returns:
        Formatted markdown presentation
    """
    llm = get_llm()

    problem = state.get("problem", "No problem specified")
    agent_ideas = state.get("agent_ideas", [])
    evaluations = state.get("evaluations", [])

    if not agent_ideas:
        return "# Error\n\nNo agent ideas found to present."

    # Prepare summaries
    agent_summaries = ""
    for idea in agent_ideas:
        agent_id = idea.get("agent_id", "Unknown")
        initial = idea.get("initial_idea", "N/A")[:500]
        final = idea.get("final_idea", "N/A")[:800]
        num_reflections = len(idea.get("reflections", []))

        agent_summaries += f"""
### Agent {agent_id}
**Reflections Completed:** {num_reflections}

**Final Proposal:**
{final}

---
"""

    # Prepare evaluation summary with weighted scores
    eval_summary = ""
    if evaluations:
        # Get weighted results
        weighted_results = calculate_weighted_scores(evaluations)
        scores = parse_evaluation_scores(evaluations)

        # Overall weighted scores table
        eval_summary += "\n### Overall Weighted Scores (1-5 Likert Scale)\n\n"
        eval_summary += "| Agent | Weighted Score | Percentage | Simple Avg |\n"
        eval_summary += "|-------|----------------|------------|------------|\n"

        for agent_id in sorted(weighted_results.keys()):
            result = weighted_results[agent_id]
            eval_summary += f"| Agent {agent_id} | {result['weighted_score']:.2f}/{result['max_possible']:.2f} | **{result['percentage']:.1f}%** | {result['simple_average']:.2f}/5 |\n"

        # Detailed scores by dimension
        eval_summary += "\n### Scores by Dimension\n\n"
        eval_summary += "| Agent | Correctness<br>(30%) | SOTA Competitive<br>(30%) | Low Time Complexity<br>(20%) | Clear Pseudocode<br>(10%) | Novelty<br>(10%) |\n"
        eval_summary += "|-------|---------|------------------|---------------------|-------------|----------|\n"

        for agent_id in sorted(scores.keys()):
            correctness = scores[agent_id].get('correctness', 0)
            sota_comp = scores[agent_id].get('sota_competitiveness', 0)
            time_comp = scores[agent_id].get('time_complexity', 0)
            pseudo = scores[agent_id].get('pseudocode_quality', 0)
            novelty = scores[agent_id].get('novelty', 0)

            eval_summary += f"| Agent {agent_id} | {correctness}/5 | {sota_comp}/5 | {time_comp}/5 | {pseudo}/5 | {novelty}/5 |\n"

        # Detailed feedback for each dimension
        eval_summary += "\n### Detailed Evaluation Feedback\n\n"
        for eval_result in evaluations:
            aspect = eval_result['aspect']
            weight = eval_result.get('weight', 0)
            eval_summary += f"#### {aspect.upper().replace('_', ' ')} (Weight: {weight*100:.0f}%)\n\n"

            parsed_scores = eval_result.get('parsed_scores', [])
            for score_data in parsed_scores:
                agent_id = score_data['agent_id']
                score = score_data['score']
                confidence = score_data['confidence']
                justification = score_data['justification']

                eval_summary += f"**Agent {agent_id}**: {score}/5 (Confidence: {confidence})  \n"
                eval_summary += f"{justification}\n\n"

    else:
        eval_summary = "No evaluations available."

    # Load presentation prompt from template
    presentation_prompt = load_and_format_prompt(
        "presentation",
        problem=problem,
        agent_summaries=agent_summaries,
        evaluation_summary=eval_summary,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    print("\n[Presenter] Generating final presentation...")
    response = llm.invoke(presentation_prompt)
    presentation = response.content

    print("[Presenter] Presentation complete")

    return presentation


def save_presentation(presentation: str, filename: str = None) -> str:
    """
    Save presentation to file.

    Args:
        presentation: Markdown presentation content
        filename: Output filename (default: presentation_TIMESTAMP.md)

    Returns:
        Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"presentation_{timestamp}.md"

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(presentation)

    print(f"\n[Presenter] Presentation saved to: {filepath}")
    return filepath


def generate_quick_summary(state: Dict) -> str:
    """
    Generate a quick text summary without LLM (for testing/logging).

    Args:
        state: Current state

    Returns:
        Plain text summary
    """
    summary = "\n" + "="*80 + "\n"
    summary += "QUICK SUMMARY\n"
    summary += "="*80 + "\n\n"

    summary += f"Problem: {state.get('problem', 'N/A')}\n\n"

    agent_ideas = state.get("agent_ideas", [])
    summary += f"Agents: {len(agent_ideas)}\n"

    for idea in agent_ideas:
        agent_id = idea.get("agent_id", "?")
        reflections = len(idea.get("reflections", []))
        summary += f"  - Agent {agent_id}: {reflections} reflections\n"

    evaluations = state.get("evaluations", [])
    if evaluations:
        summary += f"\nEvaluations: {len(evaluations)} aspects\n"
        scores = parse_evaluation_scores(evaluations)
        for agent_id in sorted(scores.keys()):
            avg = sum(scores[agent_id].values()) / len(scores[agent_id]) if scores[agent_id] else 0
            summary += f"  - Agent {agent_id} avg: {avg:.1f}/5\n"

    summary += "\n" + "="*80 + "\n"

    return summary


if __name__ == "__main__":
    # Test the presenter
    from dotenv import load_dotenv
    load_dotenv()

    test_state = {
        "problem": "Find k-th smallest element in two sorted arrays",
        "agent_ideas": [
            {
                "agent_id": 1,
                "initial_idea": "Binary search approach",
                "final_idea": "Optimized binary search on smaller array with partitioning. O(log(min(m,n))) time complexity.",
                "reflections": [{"iteration": 1, "analysis": "Improved edge cases"}]
            },
            {
                "agent_id": 2,
                "initial_idea": "Merge approach",
                "final_idea": "Partial merge until k-th element found. O(k) time complexity with early termination.",
                "reflections": [{"iteration": 1, "analysis": "Added optimization"}]
            },
            {
                "agent_id": 3,
                "initial_idea": "Heap approach",
                "final_idea": "Min-heap with selective insertion from both arrays. O(k log k) complexity.",
                "reflections": [{"iteration": 1, "analysis": "Refined heap operations"}]
            }
        ],
        "evaluations": [
            {"aspect": "novelty", "scores": "Agent 1: 8/10\nAgent 2: 6/10\nAgent 3: 7/10"},
            {"aspect": "correctness", "scores": "Agent 1: 9/10\nAgent 2: 8/10\nAgent 3: 8/10"},
            {"aspect": "efficiency", "scores": "Agent 1: 9/10\nAgent 2: 7/10\nAgent 3: 7/10"},
            {"aspect": "clarity", "scores": "Agent 1: 8/10\nAgent 2: 9/10\nAgent 3: 7/10"}
        ]
    }

    presentation = generate_presentation(test_state)
    print(presentation)
    save_presentation(presentation, "test_presentation.md")
