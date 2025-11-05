"""
Evolutionary Engine - Main evolutionary loop for algorithm ideation.

Orchestrates the complete evolutionary search process including:
- Population initialization
- Selection
- Reproduction (mutation, crossover, fresh generation)
- Evaluation
- Survivor selection
"""

from typing import Dict, Any, Optional, List
from tools.population_manager import PopulationManager, Population, Individual
from tools.evolutionary_operators import (
    create_mutation_offspring,
    create_crossover_offspring,
    create_fresh_offspring
)
from tools.selection import select_parents, select_elites, select_survivors, get_population_statistics
from tools.population_evaluator import evaluate_population
from tools.evolution_summary import save_evolution_summary
from config import Config
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from pathlib import Path
import json


# ============================================================================
# Research Preprocessing
# ============================================================================

def load_or_run_research(
    problem: str,
    problem_name: str,
    enable_research: bool = False,
    enable_local_papers: bool = True
) -> Dict[str, Any]:
    """
    Load research from cache or run research and save to cache.

    Args:
        problem: Problem statement
        problem_name: Short name for problem (e.g., 'cvrp') used for cache filename
        enable_research: Whether to enable online research
        enable_local_papers: Whether to load local papers

    Returns:
        Dictionary with research_context and local_papers
    """
    # Check for cached research
    cache_path = Path(f"{problem_name}_research.txt")

    if cache_path.exists():
        print(f"\n✓ Found cached research: {cache_path}")
        print("  Loading from cache (delete file to re-run research)...")

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            print(f"  ✓ Loaded research_context: {len(cached_data.get('research_context', ''))} characters")
            print(f"  ✓ Loaded local_papers: {len(cached_data.get('local_papers', []))} chunks")

            return cached_data

        except Exception as e:
            print(f"  ⚠️ Error loading cache: {e}")
            print("  Running research fresh...")

    # Run research fresh
    print(f"\n{'='*80}")
    print("PREPROCESSING: RESEARCH & CONTEXT GATHERING")
    print(f"{'='*80}")

    research_context = ""
    local_papers = []

    # Online research
    if enable_research:
        try:
            print("\n[Research] Searching for relevant papers and information...")
            from tools.research import research_topic
            context = asyncio.run(research_topic(problem))
            research_context = context
            print(f"[Research] ✓ Retrieved {len(research_context)} characters of context")
        except Exception as e:
            print(f"[Research] ⚠️ Error: {str(e)}")
            research_context = "Research unavailable due to error."
    else:
        print("\n[Research] Online research disabled")
        research_context = "Online research disabled."

    # Local papers
    if enable_local_papers:
        try:
            print("\n[Research] Loading local papers from data/papers/...")
            from tools.file_loader import load_local_papers
            # Use Gemini 2.5 Flash for intelligent summarization focused on algorithms/methods
            papers = load_local_papers(use_summarization=True)
            local_papers = papers

            if not local_papers:
                print(f"[Research] ⚠️ No papers loaded!")
                print(f"[Research] Make sure:")
                print(f"  1. PDF files exist in data/papers/ directory")
                print(f"  2. langchain_community is installed: pip install langchain-community")
                print(f"  3. unstructured[pdf] is installed: pip install unstructured[pdf]")
                print(f"  4. Google API key is configured for Gemini summarization")
            else:
                print(f"[Research] ✓ Loaded {len(local_papers)} paper summaries")
                # Show sample of first summary
                if local_papers and len(local_papers[0]) > 0:
                    print(f"[Research] Sample: {local_papers[0][:200]}...")

        except ModuleNotFoundError as e:
            print(f"[Research] ❌ Missing required module: {str(e)}")
            print(f"[Research] To enable PDF loading, install:")
            print(f"  pip install langchain-community")
            print(f"  pip install unstructured[pdf]")
            local_papers = []
        except Exception as e:
            print(f"[Research] ⚠️ Error loading papers: {str(e)}")
            print(f"[Research] Full error details:")
            import traceback
            traceback.print_exc()
            local_papers = []
    else:
        print("\n[Research] Local papers disabled")
        local_papers = []

    # Save to cache
    cache_data = {
        "research_context": research_context,
        "local_papers": local_papers,
        "problem": problem,
        "timestamp": str(Path.ctime(cache_path)) if cache_path.exists() else ""
    }

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"\n✓ Research cached to: {cache_path}")
    except Exception as e:
        print(f"\n⚠️ Could not save cache: {e}")

    return cache_data


class EvolutionaryEngine:
    """Main engine for evolutionary algorithm ideation."""

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize evolutionary engine.

        Args:
            parameters: Evolutionary parameters from parameters.txt
        """
        self.params = parameters
        self.pop_manager = PopulationManager()
        self.active_agents = Config.get_active_agents()

        if not self.active_agents:
            raise ValueError("No active agents available! Configure at least one API key.")

        print(f"\n✓ Active agents for evolution: {self.active_agents}")

    def _check_missing_evaluations(
        self,
        population: Population,
        target_generation: int
    ) -> List[Individual]:
        """
        Check for individuals with missing or incomplete evaluations.

        Args:
            population: Population to check
            target_generation: Generation to check for

        Returns:
            List of individuals with missing evaluations
        """
        expected_aspects = ["correctness", "sota_competitiveness", "time_complexity", "pseudocode_quality", "novelty"]
        missing = []

        for individual in population.individuals:
            # Check if individual belongs to target generation
            if individual.metadata.get("generation_created", -1) > target_generation:
                continue

            # Check if evaluation is missing or incomplete
            is_missing = False

            # Check if never evaluated
            if individual.last_evaluated_generation < 0:
                is_missing = True
            # Check if scores are missing
            elif not individual.scores or len(individual.scores) == 0:
                is_missing = True
            # Check if all aspects are present
            elif not all(aspect in individual.scores for aspect in expected_aspects):
                is_missing = True
            # Check if fitness is invalid
            elif individual.fitness <= 0:
                is_missing = True

            if is_missing:
                missing.append(individual)

        return missing

    def initialize_population(
        self,
        problem: str,
        state: Dict[str, Any]
    ) -> Population:
        """
        Initialize population with fresh individuals using PARALLEL execution.

        Args:
            problem: Problem statement
            state: System state (with research context, etc.)

        Returns:
            Initial population
        """
        print(f"\n{'='*80}")
        print("INITIALIZING POPULATION (PARALLEL MODE)")
        print(f"{'='*80}")

        population = self.pop_manager.create_initial_population(problem, self.params)

        # Generate initial ideas using all active agents IN PARALLEL
        initial_size = self.params["initial_population_size"]
        num_reflections = self.params["num_reflections"]

        print(f"\nGenerating {initial_size} initial ideas using {len(self.active_agents)} agents IN PARALLEL...")
        print(f"Reflections per idea: {num_reflections}")

        # Distribute ideas across agents
        ideas_per_agent = initial_size // len(self.active_agents)
        remainder = initial_size % len(self.active_agents)

        # Create task list: (agent_id, num_ideas_for_agent)
        agent_tasks = []
        for agent_idx, agent_id in enumerate(self.active_agents):
            num_ideas = ideas_per_agent + (1 if agent_idx < remainder else 0)
            agent_tasks.append((agent_id, num_ideas))

        print(f"Task distribution: {[(f'Agent {aid}', n) for aid, n in agent_tasks]}")

        # Helper function to generate ideas for one agent
        def generate_agent_ideas(agent_id: int, num_ideas: int):
            """Generate multiple ideas for a single agent."""
            agent_individuals = []
            for i in range(num_ideas):
                print(f"  [Agent {agent_id}] Generating idea {i+1}/{num_ideas}...")

                # Generate fresh individual
                individuals = create_fresh_offspring(
                    state=state,
                    available_agents=[agent_id],  # Use specific agent
                    num_reflections=num_reflections,
                    generation=0,
                    num_offspring=1
                )

                if individuals:
                    agent_individuals.append(individuals[0])
                    print(f"  [Agent {agent_id}] Idea {i+1}/{num_ideas} ✓")

            return agent_individuals

        # Execute in parallel using ThreadPoolExecutor
        all_individuals = []
        print(f"\n🚀 Starting PARALLEL execution with {len(agent_tasks)} agents...")

        with ThreadPoolExecutor(max_workers=len(self.active_agents)) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(generate_agent_ideas, agent_id, num_ideas): agent_id
                for agent_id, num_ideas in agent_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    individuals = future.result()
                    all_individuals.extend(individuals)
                    print(f"\n✓ Agent {agent_id} completed: {len(individuals)} ideas generated")
                except Exception as e:
                    print(f"\n❌ Agent {agent_id} failed: {e}")

        population.individuals = all_individuals
        print(f"\n✓ Generated {len(population.individuals)} initial ideas (PARALLEL)")

        # Evaluate initial population
        print("\nEvaluating initial population...")

        # Create save callback
        def save_progress(individuals):
            population.individuals = individuals
            population.update_ranks()
            self.pop_manager.save_population(population)

        population.individuals = evaluate_population(
            population.individuals,
            problem,
            generation=0,
            only_modified=False,  # Evaluate all since they're new
            batch_size=self.params["batch_size"],
            save_callback=save_progress
        )

        # Update ranks
        population.update_ranks()

        # Save initial population
        filepath = self.pop_manager.save_population(population)
        print(f"\n✓ Initial population saved: {filepath}")

        return population

    def evolve_generation(
        self,
        population: Population,
        state: Dict[str, Any],
        generation: int
    ) -> Population:
        """
        Evolve population for one generation.

        Args:
            population: Current population
            state: System state
            generation: Current generation number

        Returns:
            Evolved population
        """
        print(f"\n{'='*80}")
        print(f"GENERATION {generation}")
        print(f"{'='*80}")

        problem = population.metadata.problem_statement

        # 1. Selection - Get elites and parents
        print("\n[1/5] Selection...")
        elites = select_elites(population.individuals, self.params["elite_ratio"])
        print(f"  ✓ Preserved {len(elites)} elite individuals")

        # Select parents for reproduction
        num_parents_needed = self.params["max_population_size"]
        parents = select_parents(
            population.individuals,
            num_parents_needed,
            method=self.params["parent_selection_method"],
            tournament_size=self.params["tournament_size"]
        )
        print(f"  ✓ Selected {len(parents)} parents for reproduction")

        # 2. Reproduction - Create offspring
        print("\n[2/5] Reproduction...")

        offspring = []

        # Calculate number of offspring for each operation
        pop_size = self.params["max_population_size"]
        num_mutations = int(pop_size * self.params["mutation_ratio"])
        num_crossovers = int(pop_size * self.params["crossover_ratio"])
        num_fresh = pop_size - num_mutations - num_crossovers  # Remainder goes to fresh

        # Mutation
        if num_mutations > 0:
            print(f"  Creating {num_mutations} mutation offspring...")
            mutations = create_mutation_offspring(
                parents=parents,
                state=state,
                available_agents=self.active_agents,
                num_reflections=self.params["num_reflections"],
                generation=generation,
                num_offspring=num_mutations,
                enforce_diversity=self.params["enforce_agent_diversity"]
            )
            offspring.extend(mutations)
            print(f"  ✓ {len(mutations)} mutations created")

            # Save progress after mutations
            temp_population = Population(
                metadata=population.metadata,
                individuals=population.individuals + offspring
            )
            self.pop_manager.save_population(temp_population)
            print(f"  💾 Progress saved (mutations added)")

        # Crossover
        if num_crossovers > 0:
            print(f"  Creating {num_crossovers} crossover offspring...")
            crossovers = create_crossover_offspring(
                parents=parents,
                state=state,
                available_agents=self.active_agents,
                generation=generation,
                num_offspring=num_crossovers,
                enforce_diversity=self.params["enforce_agent_diversity"]
            )
            offspring.extend(crossovers)
            print(f"  ✓ {len(crossovers)} crossovers created")

            # Save progress after crossovers
            temp_population = Population(
                metadata=population.metadata,
                individuals=population.individuals + offspring
            )
            self.pop_manager.save_population(temp_population)
            print(f"  💾 Progress saved (crossovers added)")

        # Fresh
        if num_fresh > 0:
            print(f"  Creating {num_fresh} fresh offspring...")
            fresh = create_fresh_offspring(
                state=state,
                available_agents=self.active_agents,
                num_reflections=self.params["num_reflections"],
                generation=generation,
                num_offspring=num_fresh
            )
            offspring.extend(fresh)
            print(f"  ✓ {len(fresh)} fresh ideas created")

            # Save progress after fresh
            temp_population = Population(
                metadata=population.metadata,
                individuals=population.individuals + offspring
            )
            self.pop_manager.save_population(temp_population)
            print(f"  💾 Progress saved (fresh ideas added)")

        print(f"\n  ✓ Total offspring: {len(offspring)}")

        # 3. Evaluation - Evaluate new offspring
        print("\n[3/5] Evaluation...")

        # Create save callback that saves evaluated offspring + existing population
        def save_eval_progress(evaluated_offspring):
            temp_population = Population(
                metadata=population.metadata,
                individuals=population.individuals + evaluated_offspring
            )
            temp_population.update_ranks()
            self.pop_manager.save_population(temp_population)

        offspring = evaluate_population(
            offspring,
            problem,
            generation=generation,
            only_modified=True,
            batch_size=self.params["batch_size"],
            save_callback=save_eval_progress
        )
        print(f"  ✓ Evaluated {len(offspring)} offspring")

        # 4. Survivor Selection - Combine and select best
        print("\n[4/5] Survivor Selection...")
        combined = elites + offspring
        print(f"  Combined pool: {len(elites)} elites + {len(offspring)} offspring = {len(combined)}")

        survivors = select_survivors(
            combined,
            self.params["max_population_size"]
        )
        print(f"  ✓ Selected {len(survivors)} survivors")

        # Update population
        population.individuals = survivors
        population.metadata.current_generation = generation
        population.update_ranks()

        # 5. Statistics
        print("\n[5/5] Statistics...")
        stats = get_population_statistics(population.individuals)
        print(f"  Best Fitness: {stats['best_fitness']:.2f}/5.0 ({stats['best_fitness']/5.0*100:.0f}%)")
        print(f"  Avg Fitness:  {stats['avg_fitness']:.2f}/5.0 ({stats['avg_fitness']/5.0*100:.0f}%)")
        print(f"  Diversity:    {stats['diversity']:.2f}")
        print(f"  Elites:       {stats['num_elites']}")

        # Save population
        filepath = self.pop_manager.save_population(population)
        print(f"\n✓ Population saved: {filepath}")

        return population

    def run_evolution(
        self,
        problem: str,
        state: Dict[str, Any],
        continue_run_id: Optional[str] = None
    ) -> Population:
        """
        Run complete evolutionary search.

        Args:
            problem: Problem statement
            state: System state (with research context)
            continue_run_id: Optional run ID to continue from

        Returns:
            Final evolved population
        """
        # Load or initialize population
        if continue_run_id:
            print(f"\n{'='*80}")
            print(f"CONTINUING EVOLUTION: {continue_run_id}")
            print(f"{'='*80}")

            population = self.pop_manager.load_population(continue_run_id)

            # Verify problem match
            if not self.pop_manager.verify_problem_match(population, problem):
                raise ValueError(
                    f"Problem mismatch! Run {continue_run_id} was for a different problem.\n"
                    f"Use --reset to start fresh evolution for this problem."
                )

            print(f"\n✓ Loaded population from generation {population.metadata.current_generation}")
            print(f"  Population size: {len(population.individuals)}")

            # Check for missing evaluations in current generation
            current_gen = population.metadata.current_generation
            missing_evals = self._check_missing_evaluations(population, current_gen)

            if missing_evals:
                print(f"\n⚠️  Found {len(missing_evals)} individuals with missing/incomplete evaluations")
                print(f"  Running evaluation for generation {current_gen}...")

                # Create save callback for continue mode
                def save_continue_progress(evaluated_individuals):
                    eval_map = {ind.id: ind for ind in evaluated_individuals}
                    for i, ind in enumerate(population.individuals):
                        if ind.id in eval_map:
                            population.individuals[i] = eval_map[ind.id]
                    population.update_ranks()
                    self.pop_manager.save_population(population)

                # Evaluate only individuals with missing evaluations
                evaluated_individuals = evaluate_population(
                    missing_evals,
                    problem,
                    generation=current_gen,
                    only_modified=False,  # Force evaluation
                    batch_size=self.params["batch_size"],
                    save_callback=save_continue_progress
                )

                # Update individuals in population (final update)
                eval_map = {ind.id: ind for ind in evaluated_individuals}
                for i, ind in enumerate(population.individuals):
                    if ind.id in eval_map:
                        population.individuals[i] = eval_map[ind.id]

                # Update ranks and save
                population.update_ranks()
                self.pop_manager.save_population(population)

                print(f"  ✓ Completed missing evaluations")

            print(f"  Best fitness: {population.get_best(1)[0].fitness:.2f}/5.0")

            start_generation = population.metadata.current_generation + 1

            # Check if evolution is already complete
            if start_generation > self.params["max_generations"]:
                print(f"\n✓ Evolution already complete!")
                print(f"  Current generation: {population.metadata.current_generation}")
                print(f"  Max generations: {self.params['max_generations']}")

                # Check if evolution summary is missing and generate if needed
                from pathlib import Path
                summary_filename = f"evolution_summary_{population.metadata.run_id}.md"
                summary_path = Path("output") / summary_filename

                if not summary_path.exists():
                    print(f"\n⚠️  Evolution summary not found: {summary_path}")
                    print(f"📝 Generating evolution summary for top {self.params['top_k_ideas']} solutions...")

                    # Generate and save summary
                    top_k = self.params["top_k_ideas"]
                    generated_path = save_evolution_summary(population, top_k=top_k)

                    print(f"✓ Summary generated: {generated_path}")
                    print(f"✓ Best fitness: {population.get_best(1)[0].fitness:.2f}/5.0")
                else:
                    print(f"✓ Evolution summary exists: {summary_path}")

                print(f"\nTo extend evolution, increase max_generations in parameters.txt")
                return population
        else:
            # Initialize new population
            population = self.initialize_population(problem, state)
            start_generation = 1

        # Display initial stats
        stats = get_population_statistics(population.individuals)
        print(f"\nStarting Evolution:")
        print(f"  Best Fitness: {stats['best_fitness']:.2f}/5.0")
        print(f"  Avg Fitness:  {stats['avg_fitness']:.2f}/5.0")

        # Evolution loop
        max_generations = self.params["max_generations"]
        # Always end at max_generations (not relative to start_generation)
        # This ensures --continue completes up to max_generations, not adding more
        end_generation = max_generations

        print(f"\nEvolution Plan: Running generations {start_generation} to {end_generation}")
        if continue_run_id:
            remaining = end_generation - start_generation + 1
            print(f"  (Continuing: {remaining} generations remaining to reach max_generations={max_generations})")

        for generation in range(start_generation, end_generation + 1):
            population = self.evolve_generation(population, state, generation)

        # Generate final summary
        print(f"\n{'='*80}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*80}")

        top_k = self.params["top_k_ideas"]
        summary_path = save_evolution_summary(population, top_k=top_k)

        print(f"\n✓ Evolution completed after {population.metadata.current_generation} generations")
        print(f"✓ Final population: {len(population.individuals)} individuals")
        print(f"✓ Best fitness: {population.get_best(1)[0].fitness:.2f}/5.0")
        print(f"✓ Summary saved: {summary_path}")
        print(f"✓ Population saved: population/population_{population.metadata.run_id}.json")

        return population


def run_evolutionary_ideation(
    problem: str,
    problem_name: str = "problem",
    enable_research: bool = False,
    enable_local_papers: bool = True,
    continue_run_id: Optional[str] = None
) -> Population:
    """
    Run evolutionary algorithm ideation system.

    Args:
        problem: Problem statement
        problem_name: Short name for problem (e.g., 'cvrp') used for research cache filename
        enable_research: Enable online research
        enable_local_papers: Enable local paper loading
        continue_run_id: Optional run ID to continue from

    Returns:
        Final evolved population
    """
    # Load evolutionary parameters
    params = Config.load_evolutionary_parameters()

    print(f"\n{'='*80}")
    print("EVOLUTIONARY ALGORITHM IDEATION SYSTEM")
    print(f"{'='*80}")

    print("\nEvolutionary Parameters:")
    print(f"  Population Size: {params['max_population_size']}")
    print(f"  Generations: {params['max_generations']}")
    print(f"  Mutation/Crossover/Fresh: {params['mutation_ratio']*100:.0f}%/{params['crossover_ratio']*100:.0f}%/{params['fresh_ratio']*100:.0f}%")
    print(f"  Reflections: {params['num_reflections']}")
    print(f"  Elite Ratio: {params['elite_ratio']*100:.0f}%")
    print(f"  Selection: {params['parent_selection_method']}")
    print(f"  Agent Diversity: {params['enforce_agent_diversity']}")

    # Prepare state with research context
    if enable_research or enable_local_papers:
        # Load or run research (with caching)
        research_data = load_or_run_research(
            problem=problem,
            problem_name=problem_name,
            enable_research=enable_research,
            enable_local_papers=enable_local_papers
        )

        state = {
            "problem": problem,
            "research_context": research_data.get("research_context", ""),
            "local_papers": research_data.get("local_papers", [])
        }
    else:
        state = {
            "problem": problem,
            "research_context": "",
            "local_papers": []
        }

    # Create and run evolutionary engine
    engine = EvolutionaryEngine(params)
    population = engine.run_evolution(problem, state, continue_run_id=continue_run_id)

    return population
