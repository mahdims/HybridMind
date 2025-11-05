"""
Selection Mechanisms - Parent and Survivor Selection

Implements various selection strategies for evolutionary algorithm.
"""

import random
from typing import List
from tools.population_manager import Individual


# ============================================================================
# Tournament Selection
# ============================================================================

def tournament_selection(
    population: List[Individual],
    tournament_size: int = 3
) -> Individual:
    """
    Select an individual using tournament selection.

    Args:
        population: Pool of individuals
        tournament_size: Number of individuals in each tournament

    Returns:
        Selected individual (winner of tournament)
    """
    # Randomly sample tournament_size individuals
    tournament = random.sample(population, min(tournament_size, len(population)))

    # Return the one with highest fitness
    return max(tournament, key=lambda x: x.fitness)


def tournament_selection_batch(
    population: List[Individual],
    num_selections: int,
    tournament_size: int = 3
) -> List[Individual]:
    """
    Perform multiple tournament selections.

    Args:
        population: Pool of individuals
        num_selections: Number of selections to make
        tournament_size: Number of individuals in each tournament

    Returns:
        List of selected individuals (can contain duplicates)
    """
    return [
        tournament_selection(population, tournament_size)
        for _ in range(num_selections)
    ]


# ============================================================================
# Roulette Wheel Selection
# ============================================================================

def roulette_wheel_selection(population: List[Individual]) -> Individual:
    """
    Select an individual using roulette wheel (fitness-proportional) selection.

    Args:
        population: Pool of individuals

    Returns:
        Selected individual
    """
    # Calculate total fitness
    total_fitness = sum(ind.fitness for ind in population)

    if total_fitness == 0:
        # If all fitness is 0, select randomly
        return random.choice(population)

    # Generate random number
    pick = random.uniform(0, total_fitness)

    # Accumulate fitness until we exceed pick
    current = 0
    for individual in population:
        current += individual.fitness
        if current >= pick:
            return individual

    # Fallback (shouldn't reach here)
    return population[-1]


def roulette_wheel_selection_batch(
    population: List[Individual],
    num_selections: int
) -> List[Individual]:
    """
    Perform multiple roulette wheel selections.

    Args:
        population: Pool of individuals
        num_selections: Number of selections to make

    Returns:
        List of selected individuals
    """
    return [
        roulette_wheel_selection(population)
        for _ in range(num_selections)
    ]


# ============================================================================
# Rank-Based Selection
# ============================================================================

def rank_based_selection(population: List[Individual]) -> Individual:
    """
    Select an individual using rank-based selection.

    Higher ranked individuals have higher probability, but probability
    is based on rank not absolute fitness (more diverse selection).

    Args:
        population: Pool of individuals (should be sorted by fitness)

    Returns:
        Selected individual
    """
    n = len(population)

    # Assign selection probabilities based on rank
    # Linear ranking: P(rank) = (2 - s + 2*(s-1)*(rank-1)/(n-1)) / n
    # where s is selection pressure (using s=2 for standard linear ranking)
    # Simplifies to: higher rank = higher probability

    # Create rank-based weights (reversed so rank 1 = highest weight)
    weights = [n - i for i in range(n)]

    return random.choices(population, weights=weights, k=1)[0]


def rank_based_selection_batch(
    population: List[Individual],
    num_selections: int
) -> List[Individual]:
    """
    Perform multiple rank-based selections.

    Args:
        population: Pool of individuals
        num_selections: Number of selections to make

    Returns:
        List of selected individuals
    """
    return [
        rank_based_selection(population)
        for _ in range(num_selections)
    ]


# ============================================================================
# Parent Selection (Generic)
# ============================================================================

def select_parents(
    population: List[Individual],
    num_parents: int,
    method: str = "tournament",
    tournament_size: int = 3
) -> List[Individual]:
    """
    Select parent individuals for reproduction.

    Args:
        population: Pool of individuals
        num_parents: Number of parents to select
        method: Selection method ('tournament', 'roulette', 'rank')
        tournament_size: Tournament size (only used for tournament selection)

    Returns:
        List of selected parents
    """
    if method == "tournament":
        return tournament_selection_batch(population, num_parents, tournament_size)
    elif method == "roulette":
        return roulette_wheel_selection_batch(population, num_parents)
    elif method == "rank":
        return rank_based_selection_batch(population, num_parents)
    else:
        raise ValueError(f"Unknown selection method: {method}")


# ============================================================================
# Elite Selection
# ============================================================================

def select_elites(
    population: List[Individual],
    elite_ratio: float
) -> List[Individual]:
    """
    Select elite individuals (top N%).

    Args:
        population: Pool of individuals
        elite_ratio: Ratio of population to preserve as elites (0.0-1.0)

    Returns:
        List of elite individuals
    """
    num_elites = max(1, int(len(population) * elite_ratio))

    # Sort by fitness (descending) and take top N
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

    elites = sorted_pop[:num_elites]

    # Mark as elite
    for elite in elites:
        elite.is_elite = True

    return elites


# ============================================================================
# Survivor Selection
# ============================================================================

def select_survivors(
    combined_population: List[Individual],
    target_size: int
) -> List[Individual]:
    """
    Select survivors from combined population (elites + offspring).

    Uses simple truncation selection: keep top N by fitness.

    Args:
        combined_population: Combined pool of individuals
        target_size: Target population size

    Returns:
        Selected survivors
    """
    # Sort by fitness (descending)
    sorted_pop = sorted(combined_population, key=lambda x: x.fitness, reverse=True)

    # Keep top target_size individuals
    survivors = sorted_pop[:target_size]

    return survivors


# ============================================================================
# Diversity-Based Selection (Future Extension)
# ============================================================================

def select_diverse_individuals(
    population: List[Individual],
    num_to_select: int,
    diversity_weight: float = 0.5
) -> List[Individual]:
    """
    Select individuals balancing fitness and diversity.

    This is a future extension for diversity maintenance.

    Args:
        population: Pool of individuals
        num_to_select: Number of individuals to select
        diversity_weight: Weight for diversity (0-1, higher = more diversity)

    Returns:
        Selected individuals
    """
    # TODO: Implement diversity-based selection
    # For now, just use fitness-based selection
    return select_survivors(population, num_to_select)


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_diversity(population: List[Individual]) -> float:
    """
    Calculate diversity metric for population.

    Higher diversity = more different ideas.

    Args:
        population: Pool of individuals

    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    # TODO: Implement meaningful diversity metric
    # Options:
    # 1. Edit distance between idea contents
    # 2. Variance in scores across evaluation dimensions
    # 3. Number of unique operation histories

    # Placeholder: use variance in fitness
    if len(population) < 2:
        return 0.0

    fitnesses = [ind.fitness for ind in population]
    mean_fitness = sum(fitnesses) / len(fitnesses)
    variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)

    # Normalize to 0-1 range (rough approximation)
    diversity = min(1.0, variance / 2.0)

    return diversity


def get_population_statistics(population: List[Individual]) -> dict:
    """
    Calculate statistics for a population.

    Args:
        population: Pool of individuals

    Returns:
        Dictionary with statistics
    """
    if not population:
        return {
            "size": 0,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
            "worst_fitness": 0.0,
            "diversity": 0.0
        }

    fitnesses = [ind.fitness for ind in population]

    return {
        "size": len(population),
        "best_fitness": max(fitnesses),
        "avg_fitness": sum(fitnesses) / len(fitnesses),
        "worst_fitness": min(fitnesses),
        "diversity": calculate_diversity(population),
        "num_elites": sum(1 for ind in population if ind.is_elite)
    }
