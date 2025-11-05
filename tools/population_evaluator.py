"""
Population Evaluator - Evaluate individuals in population using AI judge.

Wraps the existing evaluation system to work with population individuals.
"""

from typing import List, Dict, Any
from tools.population_manager import Individual
from agents.evaluator import evaluate_ideas


def evaluate_individual(
    individual: Individual,
    problem: str,
    generation: int
) -> Individual:
    """
    Evaluate a single individual using AI judge.

    Args:
        individual: Individual to evaluate
        problem: Problem statement
        generation: Current generation number

    Returns:
        Individual with updated scores, feedback, confidence, and fitness
    """
    # Create state for evaluation
    state = {
        "problem": problem,
        "agent_ideas": [
            {
                "agent_id": individual.metadata["source_agent_id"],
                "initial_idea": "",  # Not needed for evaluation
                "final_idea": individual.content,
                "reflections": []  # Not needed for evaluation
            }
        ]
    }

    # Evaluate using existing evaluator
    evaluations = evaluate_ideas(state)

    if evaluations and len(evaluations) > 0:
        # The evaluator returns a list of evaluations (one per aspect)
        # Each evaluation has: {"aspect": "...", "parsed_scores": [...]}
        # We need to extract scores/feedback/confidence for our agent

        agent_id = individual.metadata["source_agent_id"]

        # Initialize dictionaries
        individual.scores = {}
        individual.feedback = {}
        individual.confidence = {}

        # Extract data from each aspect evaluation
        for eval_result in evaluations:
            aspect = eval_result.get("aspect", "unknown")
            parsed_scores = eval_result.get("parsed_scores", [])

            # Find the score for our agent_id
            agent_score_data = None
            for score_data in parsed_scores:
                if score_data.get("agent_id") == agent_id:
                    agent_score_data = score_data
                    break

            if agent_score_data:
                # Extract score, confidence, and justification for this aspect
                individual.scores[aspect] = agent_score_data.get("score", 0)
                individual.confidence[aspect] = agent_score_data.get("confidence", "Medium")
                individual.feedback[aspect] = agent_score_data.get("justification", "")
            else:
                # Fallback if agent not found in parsed scores
                print(f"    ⚠️ Warning: No score found for agent {agent_id} in aspect '{aspect}'")
                individual.scores[aspect] = 0
                individual.confidence[aspect] = "Low"
                individual.feedback[aspect] = "No evaluation available"

        # Ensure all expected aspects are present (from config)
        expected_aspects = ["correctness", "sota_competitiveness", "time_complexity", "pseudocode_quality", "novelty"]
        for aspect in expected_aspects:
            if aspect not in individual.scores:
                print(f"    ⚠️ Warning: Missing aspect '{aspect}' in evaluation results")
                individual.scores[aspect] = 0
                individual.confidence[aspect] = "Low"
                individual.feedback[aspect] = "Aspect not evaluated"

        # Calculate weighted fitness
        from config import Config
        weights = {
            "correctness": 0.30,
            "sota_competitiveness": 0.30,
            "time_complexity": 0.20,
            "pseudocode_quality": 0.10,
            "novelty": 0.10
        }

        fitness = sum(
            individual.scores.get(aspect, 0) * weight
            for aspect, weight in weights.items()
        )
        individual.fitness = round(fitness, 2)

        # Update last evaluated generation
        individual.last_evaluated_generation = generation

    return individual


def evaluate_population(
    individuals: List[Individual],
    problem: str,
    generation: int,
    only_modified: bool = True,
    batch_size: int = 20,
    save_callback = None
) -> List[Individual]:
    """
    Evaluate all individuals in population using BATCH evaluation.

    Args:
        individuals: List of individuals to evaluate
        problem: Problem statement
        generation: Current generation number
        only_modified: Only evaluate individuals that haven't been evaluated
                      or were modified since last evaluation
        batch_size: Number of individuals to evaluate in one API call (default: 20)
        save_callback: Optional callback function to save progress after each batch

    Returns:
        List of evaluated individuals
    """
    # Separate individuals that need evaluation from those that don't
    to_evaluate = []
    already_evaluated = []

    for individual in individuals:
        # Check if evaluation is needed
        if only_modified:
            needs_eval = (
                individual.last_evaluated_generation < 0 or  # Never evaluated
                individual.last_evaluated_generation < individual.metadata["generation_last_modified"]  # Modified since last eval
            )
        else:
            needs_eval = True

        if needs_eval:
            to_evaluate.append(individual)
        else:
            already_evaluated.append(individual)

    if not to_evaluate:
        print(f"  ✓ No individuals need evaluation (all up-to-date)")
        return individuals

    print(f"  Evaluating {len(to_evaluate)} individuals in batches of {batch_size}...")

    # Batch evaluate individuals
    evaluated = []
    for i in range(0, len(to_evaluate), batch_size):
        batch = to_evaluate[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(to_evaluate) + batch_size - 1) // batch_size

        print(f"\n  [Batch {batch_num}/{total_batches}] Evaluating {len(batch)} individuals...")

        # Evaluate batch together
        evaluated_batch = evaluate_batch(batch, problem, generation)
        evaluated.extend(evaluated_batch)

        print(f"  [Batch {batch_num}/{total_batches}] ✓ Completed")

        # Save progress after each batch if callback provided
        if save_callback:
            save_callback(evaluated + already_evaluated)
            print(f"  💾 Progress saved (batch {batch_num}/{total_batches})")

    # Combine evaluated and already-evaluated individuals
    all_individuals = evaluated + already_evaluated

    print(f"\n  ✓ Total: {len(evaluated)} newly evaluated, {len(already_evaluated)} reused")

    return all_individuals


def evaluate_batch(
    individuals: List[Individual],
    problem: str,
    generation: int
) -> List[Individual]:
    """
    Evaluate a batch of individuals together in a single API call.

    This allows the LLM to compare individuals relatively for more consistent scoring.

    Args:
        individuals: List of individuals to evaluate together
        problem: Problem statement
        generation: Current generation number

    Returns:
        List of evaluated individuals
    """
    if not individuals:
        return []

    # Create state with ALL individuals in the batch
    agent_ideas = []
    individual_map = {}  # Map agent_id -> individual

    for idx, individual in enumerate(individuals):
        # Use index as temporary agent_id for evaluation
        temp_agent_id = idx + 1
        individual_map[temp_agent_id] = individual

        agent_ideas.append({
            "agent_id": temp_agent_id,
            "initial_idea": "",
            "final_idea": individual.content,
            "reflections": []
        })

    state = {
        "problem": problem,
        "agent_ideas": agent_ideas
    }

    # Evaluate all individuals together
    evaluations = evaluate_ideas(state)

    if not evaluations or len(evaluations) == 0:
        print(f"    ⚠️ Warning: No evaluations returned for batch")
        return individuals

    # Extract scores for each individual
    for temp_agent_id, individual in individual_map.items():
        # Initialize dictionaries
        individual.scores = {}
        individual.feedback = {}
        individual.confidence = {}

        # Extract data from each aspect evaluation
        for eval_result in evaluations:
            aspect = eval_result.get("aspect", "unknown")
            parsed_scores = eval_result.get("parsed_scores", [])

            # Find the score for this individual's temp_agent_id
            agent_score_data = None
            for score_data in parsed_scores:
                if score_data.get("agent_id") == temp_agent_id:
                    agent_score_data = score_data
                    break

            if agent_score_data:
                # Extract score, confidence, and justification for this aspect
                individual.scores[aspect] = agent_score_data.get("score", 0)
                individual.confidence[aspect] = agent_score_data.get("confidence", "Medium")
                individual.feedback[aspect] = agent_score_data.get("justification", "")
            else:
                # Fallback if not found
                individual.scores[aspect] = 0
                individual.confidence[aspect] = "Low"
                individual.feedback[aspect] = "No evaluation available"

        # Ensure all expected aspects are present
        expected_aspects = ["correctness", "sota_competitiveness", "time_complexity", "pseudocode_quality", "novelty"]
        for aspect in expected_aspects:
            if aspect not in individual.scores:
                individual.scores[aspect] = 0
                individual.confidence[aspect] = "Low"
                individual.feedback[aspect] = "Aspect not evaluated"

        # Calculate weighted fitness
        weights = {
            "correctness": 0.30,
            "sota_competitiveness": 0.30,
            "time_complexity": 0.20,
            "pseudocode_quality": 0.10,
            "novelty": 0.10
        }

        fitness = sum(
            individual.scores.get(aspect, 0) * weight
            for aspect, weight in weights.items()
        )
        individual.fitness = round(fitness, 2)

        # Update last evaluated generation
        individual.last_evaluated_generation = generation

        print(f"    ✓ Individual {individual.id[:8]}... → Fitness: {individual.fitness:.2f}/5.0")

    return list(individual_map.values())


