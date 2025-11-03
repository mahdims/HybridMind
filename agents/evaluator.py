"""
Evaluator Agent - Multi-aspect LLM Judge
Evaluates agent ideas across multiple dimensions using LLM-as-a-judge.
Uses Google Gemini for fast, cost-effective evaluation.
"""

import os
import re
from typing import Dict, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import load_and_format_prompt
from config import Config


def get_llm(temperature: float = 0.3, model: str = None):
    """Get configured LLM instance for evaluation (uses Gemini for cost-effectiveness)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


def parse_evaluation_response(response_text: str) -> List[Dict]:
    """
    Parse evaluation response in format:
    Agent X: score/5 | Confidence: High/Medium/Low | Justification: ...

    Returns:
        List of dicts with agent_id, score, confidence, justification
    """
    results = []

    # Pattern to match the format
    pattern = r"Agent (\d+):\s*(\d+)/5\s*\|\s*Confidence:\s*(High|Medium|Low)\s*\|\s*Justification:\s*(.+?)(?=Agent \d+:|$)"

    matches = re.finditer(pattern, response_text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        agent_id = int(match.group(1))
        score = int(match.group(2))
        confidence = match.group(3).strip()
        justification = match.group(4).strip()

        results.append({
            "agent_id": agent_id,
            "score": score,
            "confidence": confidence,
            "justification": justification
        })

    # If parsing failed, try fallback simpler pattern
    if not results:
        simple_pattern = r"Agent (\d+):\s*(\d+)/5"
        matches = re.finditer(simple_pattern, response_text)
        for match in matches:
            results.append({
                "agent_id": int(match.group(1)),
                "score": int(match.group(2)),
                "confidence": "Medium",
                "justification": "Could not parse justification"
            })

    return results


def evaluate_ideas(state: Dict) -> List[Dict]:
    """
    Multi-aspect evaluation of all agent ideas using robust metrics.

    Args:
        state: Current state containing agent_ideas

    Returns:
        List of evaluation dictionaries with scores, confidence, and weighted results
    """
    llm = get_llm()

    agent_ideas = state.get("agent_ideas", [])

    if len(agent_ideas) < 1:
        print(f"Warning: No agent ideas to evaluate")
        return []

    print(f"[Evaluator] Evaluating {len(agent_ideas)} agent ideas")

    # Use evaluation aspects from config
    aspects = Config.EVALUATION_ASPECTS

    evaluations = []

    # Prepare agent summaries (use final ideas)
    agent_summaries = []
    for idea in agent_ideas:
        agent_id = idea.get("agent_id", "Unknown")
        final_idea = idea.get("final_idea", idea.get("initial_idea", "No idea provided"))
        # Truncate for token management
        agent_summaries.append({
            "agent_id": agent_id,
            "summary": final_idea[:1500]
        })

    # Evaluate each aspect
    for aspect in aspects:
        print(f"\n[Evaluator] Assessing {aspect['name'].upper()} (weight: {aspect['weight']*100:.0f}%)...")

        # Build agent proposals dynamically
        agent_proposals = ""
        for summary in agent_summaries:
            agent_id = summary['agent_id']
            agent_proposals += f"\n--- AGENT {agent_id} PROPOSAL ---\n"
            agent_proposals += f"{summary['summary']}\n"

        # Load evaluation prompt from template
        eval_prompt = load_and_format_prompt(
            "evaluation",
            aspect_name=aspect['name'].upper(),
            aspect_description=aspect['description'],
            aspect_criteria=aspect['criteria'],
            agent_proposals=agent_proposals
        )

        response = llm.invoke(eval_prompt)
        evaluation_result = response.content

        # Parse the structured response
        parsed_scores = parse_evaluation_response(evaluation_result)

        evaluations.append({
            "aspect": aspect['name'],
            "description": aspect['description'],
            "weight": aspect['weight'],
            "raw_response": evaluation_result,
            "parsed_scores": parsed_scores
        })

    print(f"[Evaluator] Completed evaluation across {len(aspects)} aspects")

    return evaluations


def calculate_weighted_scores(evaluations: List[Dict]) -> Dict[int, Dict]:
    """
    Calculate weighted scores and aggregate metrics for each agent.

    Args:
        evaluations: List of evaluation dictionaries with parsed scores

    Returns:
        Dictionary mapping agent_id to scores, weighted score, and breakdown
    """
    # Dynamically detect agent IDs from evaluations
    agent_ids = set()
    for eval_result in evaluations:
        parsed_scores = eval_result.get('parsed_scores', [])
        for score_data in parsed_scores:
            agent_ids.add(score_data['agent_id'])

    # Initialize agent scores for all detected agents
    agent_scores = {agent_id: {} for agent_id in agent_ids}

    # Collect all scores by aspect
    for eval_result in evaluations:
        aspect = eval_result['aspect']
        weight = eval_result['weight']
        parsed_scores = eval_result.get('parsed_scores', [])

        for score_data in parsed_scores:
            agent_id = score_data['agent_id']
            score = score_data['score']
            confidence = score_data['confidence']

            if agent_id in agent_scores:
                agent_scores[agent_id][aspect] = {
                    'score': score,
                    'weight': weight,
                    'confidence': confidence,
                    'weighted_score': score * weight
                }

    # Calculate overall scores for each agent
    results = {}
    for agent_id in sorted(agent_ids):
        aspect_scores = agent_scores[agent_id]

        # Calculate weighted average
        total_weighted = sum(data['weighted_score'] for data in aspect_scores.values())
        total_weight = sum(data['weight'] for data in aspect_scores.values())

        # Simple average
        simple_avg = sum(data['score'] for data in aspect_scores.values()) / len(aspect_scores) if aspect_scores else 0

        results[agent_id] = {
            'aspects': aspect_scores,
            'weighted_score': total_weighted,
            'simple_average': simple_avg,
            'max_possible': 5 * total_weight,  # Since scale is 1-5
            'percentage': (total_weighted / (5 * total_weight) * 100) if total_weight > 0 else 0
        }

    return results


def parse_evaluation_scores(evaluations: List[Dict]) -> Dict[int, Dict[str, float]]:
    """
    Parse evaluation scores into structured format (legacy compatibility).

    Args:
        evaluations: List of evaluation dictionaries

    Returns:
        Dictionary mapping agent_id to aspect scores
    """
    # Dynamically detect agent IDs
    agent_ids = set()
    for eval_result in evaluations:
        parsed_scores = eval_result.get('parsed_scores', [])
        for score_data in parsed_scores:
            agent_ids.add(score_data['agent_id'])

    scores = {agent_id: {} for agent_id in agent_ids}

    for eval_result in evaluations:
        aspect = eval_result['aspect']
        parsed_scores = eval_result.get('parsed_scores', [])

        for score_data in parsed_scores:
            agent_id = score_data['agent_id']
            score = score_data['score']

            if agent_id in scores:
                scores[agent_id][aspect] = score

    return scores


def format_evaluation_summary(evaluations: List[Dict]) -> str:
    """
    Format evaluations into a readable summary with weighted scores.

    Args:
        evaluations: List of evaluation dictionaries

    Returns:
        Formatted string
    """
    summary = "\n" + "="*80 + "\n"
    summary += "EVALUATION RESULTS (1-5 Likert Scale)\n"
    summary += "="*80 + "\n\n"

    # Get weighted scores
    weighted_results = calculate_weighted_scores(evaluations)

    # Summary table
    summary += "OVERALL SCORES (Weighted)\n"
    summary += "-" * 80 + "\n"
    for agent_id in sorted(weighted_results.keys()):
        result = weighted_results[agent_id]
        summary += f"\nAgent {agent_id}:\n"
        summary += f"  Weighted Score: {result['weighted_score']:.2f} / {result['max_possible']:.2f}\n"
        summary += f"  Percentage: {result['percentage']:.1f}%\n"
        summary += f"  Simple Average: {result['simple_average']:.2f} / 5.00\n"

    # Detailed breakdown by aspect
    summary += "\n" + "="*80 + "\n"
    summary += "DETAILED BREAKDOWN BY DIMENSION\n"
    summary += "="*80 + "\n\n"

    for eval_result in evaluations:
        aspect = eval_result['aspect']
        weight = eval_result['weight']
        summary += f"\n{aspect.upper().replace('_', ' ')} (Weight: {weight*100:.0f}%)\n"
        summary += f"Description: {eval_result['description']}\n"
        summary += "-" * 80 + "\n"

        parsed_scores = eval_result.get('parsed_scores', [])
        for score_data in parsed_scores:
            agent_id = score_data['agent_id']
            score = score_data['score']
            confidence = score_data['confidence']
            justification = score_data['justification']

            summary += f"\nAgent {agent_id}: {score}/5 | Confidence: {confidence}\n"
            summary += f"Justification: {justification}\n"

        summary += "\n"

    return summary


if __name__ == "__main__":
    # Test the evaluator
    from dotenv import load_dotenv
    load_dotenv()

    test_state = {
        "agent_ideas": [
            {
                "agent_id": 1,
                "final_idea": "Use binary search on the smaller array to partition both arrays efficiently. Time: O(log(min(m,n)))"
            },
            {
                "agent_id": 2,
                "final_idea": "Merge arrays partially until k-th element is found. Time: O(k)"
            },
            {
                "agent_id": 3,
                "final_idea": "Use min-heap with selective insertion. Time: O(k log k)"
            }
        ]
    }

    results = evaluate_ideas(test_state)
    print(format_evaluation_summary(results))
