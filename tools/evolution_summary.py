"""
Evolution Summary Generator - Create markdown summary of evolutionary search results.

Generates top-K summary with detailed feedback for best algorithm ideas.
"""

from datetime import datetime
from typing import List, Dict, Any
from tools.population_manager import Population, Individual


def _normalize_content(content) -> str:
    """
    Normalize individual content to string format.

    Handles cases where content might be a list, dict, or other type.

    Args:
        content: Content to normalize (can be str, list, dict, or other)

    Returns:
        String representation of content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # If content is a list, join items with newlines
        return '\n'.join(str(item) for item in content)
    elif isinstance(content, dict):
        # If content is a dict, format it nicely
        return '\n'.join(f"{k}: {v}" for k, v in content.items())
    else:
        # Fallback: convert to string
        return str(content)


def generate_evolution_summary(
    population: Population,
    top_k: int = 5,
    include_statistics: bool = True
) -> str:
    """
    Generate markdown summary of evolutionary search results.

    Args:
        population: Final population
        top_k: Number of top ideas to show in detail
        include_statistics: Whether to include evolution statistics

    Returns:
        Markdown formatted summary
    """
    # Get top K individuals
    top_individuals = population.get_best(top_k)

    # Build markdown
    md = []

    # Header
    md.append(f"# Evolutionary Search Results - {population.metadata.run_id}\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("---\n")

    # Problem
    md.append("## Problem Statement\n")
    md.append(f"{population.metadata.problem_statement}\n")
    md.append("---\n")

    # Evolution Summary
    md.append("## Evolution Summary\n")
    md.append(f"- **Total Generations:** {population.metadata.current_generation}\n")
    md.append(f"- **Final Population Size:** {len(population.individuals)}\n")

    if top_individuals:
        best_fitness = top_individuals[0].fitness
        avg_fitness = sum(ind.fitness for ind in population.individuals) / len(population.individuals)
        md.append(f"- **Best Fitness:** {best_fitness}/5.0 ({best_fitness/5.0*100:.1f}%)\n")
        md.append(f"- **Average Fitness:** {avg_fitness:.2f}/5.0 ({avg_fitness/5.0*100:.1f}%)\n")

    md.append("\n**Parameters Used:**\n")
    params = population.metadata.parameters
    md.append(f"- Population Size: {params.get('max_population_size', 'N/A')}\n")
    md.append(f"- Mutation/Crossover/Fresh: {params.get('mutation_ratio', 0)*100:.0f}%/{params.get('crossover_ratio', 0)*100:.0f}%/{params.get('fresh_ratio', 0)*100:.0f}%\n")
    md.append(f"- Reflections per Operation: {params.get('num_reflections', 'N/A')}\n")
    md.append(f"- Elite Preservation: Top {params.get('elite_ratio', 0)*100:.0f}%\n")

    md.append("\n---\n")

    # Top K Ideas
    md.append(f"## Top {len(top_individuals)} Algorithm Ideas\n")

    for i, individual in enumerate(top_individuals, 1):
        md.append(f"\n### {_get_rank_emoji(i)} Rank {i}: {_generate_idea_title(individual)}\n")
        md.append(f"**Fitness: {individual.fitness}/5.0 ({individual.fitness/5.0*100:.0f}%)**\n")

        # Genealogy
        md.append("\n**Genealogy:**\n")
        md.append(f"- Created: Generation {individual.metadata['generation_created']} (SOTA Agent {individual.metadata['source_agent_id']})\n")
        if individual.metadata['generation_last_modified'] > individual.metadata['generation_created']:
            md.append(f"- Last Modified: Generation {individual.metadata['generation_last_modified']} (SOTA Agent {individual.metadata['last_modified_by_agent_id']})\n")

        # Operation history
        history = individual.metadata.get('operation_history', [])
        if len(history) > 1:
            operations_summary = _summarize_operations(history)
            md.append(f"- Operations: {operations_summary}\n")

        # Parents (if any)
        parent_ids = individual.metadata.get('parent_ids', [])
        if parent_ids:
            md.append(f"- Parent IDs: {', '.join(parent_ids)}\n")

        # Algorithm content
        md.append("\n**Algorithm Description:**\n")
        # Normalize content to string (handles list, dict, str, etc.)
        content = _normalize_content(individual.content)
        md.append(f"{content}\n")

        # Evaluation scores
        md.append("\n**Evaluation Scores:**\n")
        md.append(_format_scores_table(individual))

        # Detailed feedback
        md.append("\n**Judge Feedback Summary:**\n")
        md.append(_format_feedback(individual))

        md.append("\n---\n")

    # Statistics (if requested)
    if include_statistics:
        md.append("\n## Evolution Statistics\n")
        md.append(_generate_statistics(population))
        md.append("\n---\n")

    # Footer
    md.append("\n## Population File\n")
    md.append(f"**Saved to:** `population/population_{population.metadata.run_id}.json`\n")
    md.append("\nTo continue evolving this population:\n")
    md.append(f"```bash\npython main.py --continue {population.metadata.run_id}\n```\n")

    return "".join(md)


def _get_rank_emoji(rank: int) -> str:
    """Get emoji for rank."""
    emojis = {1: "🥇", 2: "🥈", 3: "🥉"}
    return emojis.get(rank, "🏅")


def _generate_idea_title(individual: Individual) -> str:
    """Generate a short title for the idea based on content."""
    # Normalize content to string (handles list, dict, str, etc.)
    content = _normalize_content(individual.content)

    # Try to extract first meaningful line from content
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and not line.startswith('#'):
            # Truncate if too long
            if len(line) > 60:
                return line[:57] + "..."
            return line

    return f"Algorithm Idea (Gen {individual.metadata['generation_created']})"


def _summarize_operations(history: List[Dict[str, Any]]) -> str:
    """Summarize operation history."""
    mutation_count = sum(1 for op in history if op['operation'] == 'mutation')
    crossover_count = sum(1 for op in history if op['operation'] == 'crossover')

    parts = []
    if mutation_count > 0:
        parts.append(f"{mutation_count} mutation{'s' if mutation_count > 1 else ''}")
    if crossover_count > 0:
        parts.append(f"{crossover_count} crossover{'s' if crossover_count > 1 else ''}")

    return ", ".join(parts) if parts else "fresh generation"


def _format_scores_table(individual: Individual) -> str:
    """Format evaluation scores as markdown table."""
    scores = individual.scores
    confidence = individual.confidence

    # Get emoji for score
    def score_emoji(score):
        if score >= 4.5:
            return "✅"
        elif score >= 3.5:
            return "⚠️"
        else:
            return "❌"

    lines = []
    lines.append("| Aspect | Score | Confidence | Status |\n")
    lines.append("|--------|-------|------------|--------|\n")

    aspects = [
        ("Correctness", "correctness"),
        ("SOTA Competitiveness", "sota_competitiveness"),
        ("Time Complexity", "time_complexity"),
        ("Pseudocode Quality", "pseudocode_quality"),
        ("Novelty", "novelty")
    ]

    for display_name, key in aspects:
        score = scores.get(key, 0)
        conf = confidence.get(key, "Medium")
        emoji = score_emoji(score)
        lines.append(f"| {display_name} | {score}/5 | {conf} | {emoji} |\n")

    return "".join(lines)


def _format_feedback(individual: Individual) -> str:
    """Format detailed feedback."""
    feedback = individual.feedback

    aspects = [
        ("Correctness", "correctness"),
        ("SOTA Competitiveness", "sota_competitiveness"),
        ("Time Complexity", "time_complexity"),
        ("Pseudocode Quality", "pseudocode_quality"),
        ("Novelty", "novelty")
    ]

    lines = []
    for display_name, key in aspects:
        text = feedback.get(key, "No feedback available")
        if text:
            # Truncate very long feedback
            if len(text) > 300:
                text = text[:297] + "..."
            lines.append(f"- **{display_name}:** {text}\n")

    return "".join(lines)


def _generate_statistics(population: Population) -> str:
    """Generate evolution statistics."""
    individuals = population.individuals

    # Count operations
    mutation_count = 0
    crossover_count = 0
    fresh_count = 0

    for ind in individuals:
        history = ind.metadata.get('operation_history', [])
        for op in history:
            if op['operation'] == 'mutation':
                mutation_count += 1
            elif op['operation'] == 'crossover':
                crossover_count += 1
            elif op['operation'] == 'fresh':
                fresh_count += 1

    # Fitness progression (if we had per-generation data)
    # For now, just show final stats

    lines = []
    lines.append("\n**Operator Usage:**\n")
    lines.append(f"- Mutations: {mutation_count} total\n")
    lines.append(f"- Crossovers: {crossover_count} total\n")
    lines.append(f"- Fresh Generations: {fresh_count} total\n")

    # Elite count
    elite_count = sum(1 for ind in individuals if ind.is_elite)
    lines.append(f"\n**Elite Individuals:** {elite_count}\n")

    # Agent distribution
    agent_counts = {}
    for ind in individuals:
        agent_id = ind.metadata['source_agent_id']
        agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

    lines.append("\n**Ideas by Source Agent:**\n")
    for agent_id in sorted(agent_counts.keys()):
        lines.append(f"- SOTA Agent {agent_id}: {agent_counts[agent_id]} ideas\n")

    return "".join(lines)


def save_evolution_summary(
    population: Population,
    top_k: int = 5,
    output_dir: str = "output"
) -> str:
    """
    Generate and save evolution summary to markdown file.

    Args:
        population: Final population
        top_k: Number of top ideas to show
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    import os
    from pathlib import Path

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate summary
    summary = generate_evolution_summary(population, top_k)

    # Save to file
    filename = f"evolution_summary_{population.metadata.run_id}.md"
    filepath = output_path / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)

    return str(filepath)
