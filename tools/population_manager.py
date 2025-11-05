"""
Population Manager - Handle loading, saving, and managing evolved algorithm populations.

This module provides core functionality for the evolutionary search system to persist
and manage populations of algorithm ideas across multiple runs.
"""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Individual:
    """Represents a single algorithm idea in the population."""
    id: str
    content: str  # Full algorithm description after final reflection
    metadata: Dict[str, Any]  # Genealogy and operation history
    scores: Dict[str, float]  # Evaluation scores for each aspect
    feedback: Dict[str, str]  # Detailed feedback from judge
    confidence: Dict[str, str]  # Confidence levels for each aspect
    fitness: float  # Weighted overall score
    rank: int  # Position in population (1 = best)
    is_elite: bool  # Whether this individual is preserved as elite
    last_evaluated_generation: int  # Last generation when evaluated


@dataclass
class PopulationMetadata:
    """Metadata for a population."""
    run_id: str
    problem_statement: str
    problem_hash: str
    created_at: str
    last_updated: str
    current_generation: int
    parameters: Dict[str, Any]


class Population:
    """Container for population of algorithm ideas."""

    def __init__(self, metadata: PopulationMetadata, individuals: List[Individual]):
        self.metadata = metadata
        self.individuals = individuals

    def to_dict(self) -> Dict[str, Any]:
        """Convert population to dictionary for JSON serialization."""
        return {
            "metadata": asdict(self.metadata),
            "population": [
                {
                    "id": ind.id,
                    "content": ind.content,
                    "metadata": ind.metadata,
                    "scores": ind.scores,
                    "feedback": ind.feedback,
                    "confidence": ind.confidence,
                    "fitness": ind.fitness,
                    "rank": ind.rank,
                    "is_elite": ind.is_elite,
                    "last_evaluated_generation": ind.last_evaluated_generation
                }
                for ind in self.individuals
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Population':
        """Create population from dictionary."""
        metadata = PopulationMetadata(**data["metadata"])
        individuals = [
            Individual(
                id=ind["id"],
                content=ind["content"],
                metadata=ind["metadata"],
                scores=ind["scores"],
                feedback=ind["feedback"],
                confidence=ind["confidence"],
                fitness=ind["fitness"],
                rank=ind["rank"],
                is_elite=ind["is_elite"],
                last_evaluated_generation=ind["last_evaluated_generation"]
            )
            for ind in data["population"]
        ]
        return cls(metadata, individuals)

    def get_best(self, k: int = 1) -> List[Individual]:
        """Get top k individuals by fitness."""
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:k]

    def get_elites(self, elite_ratio: float) -> List[Individual]:
        """Get elite individuals (top N%)."""
        num_elites = max(1, int(len(self.individuals) * elite_ratio))
        return self.get_best(num_elites)

    def update_ranks(self):
        """Update rank field for all individuals based on fitness."""
        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )
        for rank, individual in enumerate(sorted_individuals, 1):
            individual.rank = rank


class PopulationManager:
    """Manages population files and operations."""

    def __init__(self, population_dir: str = "population"):
        """
        Initialize population manager.

        Args:
            population_dir: Directory to store population files
        """
        self.population_dir = Path(population_dir)
        self.population_dir.mkdir(exist_ok=True)

    def generate_run_id(self) -> str:
        """
        Generate unique run ID based on timestamp.

        Returns:
            Run ID in format: run_YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def hash_problem(self, problem: str) -> str:
        """
        Generate hash of problem statement for matching.

        Args:
            problem: Problem statement

        Returns:
            Hash string
        """
        return hashlib.md5(problem.strip().encode()).hexdigest()[:16]

    def get_population_path(self, run_id: str) -> Path:
        """Get file path for a population by run ID."""
        return self.population_dir / f"population_{run_id}.json"

    def save_population(self, population: Population) -> str:
        """
        Save population to JSON file.

        Args:
            population: Population to save

        Returns:
            Path to saved file
        """
        # Update last_updated timestamp
        population.metadata.last_updated = datetime.now().isoformat()

        # Save to file
        filepath = self.get_population_path(population.metadata.run_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(population.to_dict(), f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load_population(self, run_id: str) -> Population:
        """
        Load population from JSON file.

        Args:
            run_id: Run ID to load

        Returns:
            Loaded population

        Raises:
            FileNotFoundError: If population file doesn't exist
        """
        filepath = self.get_population_path(run_id)

        if not filepath.exists():
            raise FileNotFoundError(f"Population file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return Population.from_dict(data)

    def list_populations(self) -> List[Dict[str, Any]]:
        """
        List all available population files with metadata.

        Returns:
            List of dicts with run_id, timestamp, generation, etc.
        """
        populations = []

        for filepath in self.population_dir.glob("population_run_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                metadata = data["metadata"]
                populations.append({
                    "run_id": metadata["run_id"],
                    "created_at": metadata["created_at"],
                    "last_updated": metadata["last_updated"],
                    "current_generation": metadata["current_generation"],
                    "population_size": len(data["population"]),
                    "problem_preview": metadata["problem_statement"][:100] + "...",
                    "filepath": str(filepath)
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by last_updated (most recent first)
        populations.sort(key=lambda x: x["last_updated"], reverse=True)

        return populations

    def get_latest_population(self) -> Optional[Tuple[str, Population]]:
        """
        Get the most recently updated population.

        Returns:
            Tuple of (run_id, population) or None if no populations exist
        """
        populations = self.list_populations()

        if not populations:
            return None

        latest = populations[0]
        run_id = latest["run_id"]
        population = self.load_population(run_id)

        return (run_id, population)

    def verify_problem_match(
        self,
        population: Population,
        problem: str
    ) -> bool:
        """
        Verify if problem matches the population's problem.

        Args:
            population: Population to check
            problem: Problem statement to match

        Returns:
            True if problems match, False otherwise
        """
        problem_hash = self.hash_problem(problem)
        return population.metadata.problem_hash == problem_hash

    def create_individual(
        self,
        content: str,
        generation: int,
        agent_id: int,
        operation: str = "fresh",
        parent_ids: List[str] = None
    ) -> Individual:
        """
        Create a new individual (idea).

        Args:
            content: Algorithm description
            generation: Generation when created
            agent_id: Agent that created this idea
            operation: Type of operation (fresh, mutation, crossover)
            parent_ids: List of parent idea IDs (empty for fresh)

        Returns:
            New Individual instance
        """
        # Generate unique ID
        idea_id = f"idea_{generation}_{agent_id}_{datetime.now().strftime('%H%M%S%f')}"

        # Initialize metadata
        metadata = {
            "generation_created": generation,
            "generation_last_modified": generation,
            "source_agent_id": agent_id,
            "last_modified_by_agent_id": agent_id,
            "parent_ids": parent_ids if parent_ids else [],
            "operation_history": [
                {
                    "generation": generation,
                    "operation": operation,
                    "agent_id": agent_id
                }
            ]
        }

        return Individual(
            id=idea_id,
            content=content,
            metadata=metadata,
            scores={},
            feedback={},
            confidence={},
            fitness=0.0,
            rank=0,
            is_elite=False,
            last_evaluated_generation=-1  # Not evaluated yet
        )

    def create_initial_population(
        self,
        problem: str,
        parameters: Dict[str, Any]
    ) -> Population:
        """
        Create a new empty population for a problem.

        Args:
            problem: Problem statement
            parameters: Evolutionary parameters

        Returns:
            New Population with empty individuals list
        """
        run_id = self.generate_run_id()
        problem_hash = self.hash_problem(problem)

        metadata = PopulationMetadata(
            run_id=run_id,
            problem_statement=problem,
            problem_hash=problem_hash,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            current_generation=0,
            parameters=parameters
        )

        return Population(metadata, [])

    def find_population_by_problem(self, problem: str) -> Optional[Tuple[str, Population]]:
        """
        Find most recent population for a given problem.

        Args:
            problem: Problem statement

        Returns:
            Tuple of (run_id, population) or None if not found
        """
        problem_hash = self.hash_problem(problem)

        populations = self.list_populations()

        for pop_info in populations:
            try:
                population = self.load_population(pop_info["run_id"])
                if population.metadata.problem_hash == problem_hash:
                    return (pop_info["run_id"], population)
            except Exception:
                continue

        return None
