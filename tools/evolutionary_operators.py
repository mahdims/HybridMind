"""
Evolutionary Operators - Mutation, Crossover, and Fresh Generation

Implements the three core evolutionary operations with agent diversity constraints.
Multiple prompt variants are used for crossover and fresh generation to increase diversity.
"""

import random
from typing import List, Dict, Any, Optional
from tools.population_manager import Individual
from agents.ideation import ideation_agent


# ============================================================================
# Agent Diversity Management
# ============================================================================

def select_different_agent(
    excluded_agent_ids: List[int],
    available_agents: List[int]
) -> int:
    """
    Select a different agent than the excluded ones.

    Args:
        excluded_agent_ids: Agent IDs to exclude (e.g., original creator)
        available_agents: List of all active agent IDs

    Returns:
        Different agent ID
    """
    other_agents = [a for a in available_agents if a not in excluded_agent_ids]

    if not other_agents:
        # Fallback: if only excluded agents available, use random from available
        return random.choice(available_agents)

    return random.choice(other_agents)


# ============================================================================
# Crossover Prompt Variants
# ============================================================================

CROSSOVER_PROMPTS = [
    # Variant 1: Hybrid strengths
    """You are tasked with creating a HYBRID algorithm by combining the strengths of two existing approaches.

PARENT ALGORITHM 1:
{parent1}

PARENT ALGORITHM 2:
{parent2}

Your task:
1. Identify the key strengths and innovative aspects of each parent algorithm
2. Create a NEW hybrid algorithm that combines these strengths synergistically
3. Ensure the hybrid is coherent and not just a concatenation
4. Address any potential conflicts between the two approaches
5. Explain how the combination leads to better performance than either parent alone

Provide a complete algorithmic description with pseudocode.""",

    # Variant 2: Take best components
    """You are an algorithm designer tasked with creating an IMPROVED algorithm by selecting the best components from two existing solutions.

SOLUTION A:
{parent1}

SOLUTION B:
{parent2}

Your task:
1. Analyze which specific components/mechanisms are strongest in each solution
2. Design a new algorithm that takes the best parts from both
3. Ensure smooth integration between components from different sources
4. Add any necessary "glue" logic to make the hybrid work cohesively
5. Optimize the overall design for CVRP problem characteristics

Provide detailed pseudocode and explain your design choices.""",

    # Variant 3: Complementary fusion
    """You are creating a FUSION algorithm that leverages complementary aspects of two different approaches.

APPROACH 1:
{parent1}

APPROACH 2:
{parent2}

Your task:
1. Identify where one approach excels and the other is weak (complementary strengths)
2. Design a unified algorithm that uses each approach where it's strongest
3. Create adaptive mechanisms to switch or blend between approaches
4. Ensure the fusion is more than the sum of parts
5. Maintain computational efficiency

Provide complete algorithm specification with pseudocode.""",

    # Variant 4: Evolutionary crossover
    """You are performing ALGORITHMIC CROSSOVER inspired by genetic algorithms.

PARENT GENOME 1:
{parent1}

PARENT GENOME 2:
{parent2}

Your task:
1. Break down each algorithm into key "genes" (search operators, data structures, strategies)
2. Select genes from each parent to create offspring
3. Ensure genetic compatibility (components work together)
4. Apply "recombination" to create novel mechanisms not in either parent
5. Validate the offspring is algorithmically sound

Provide the complete offspring algorithm with detailed pseudocode.""",

    # Variant 5: Synergistic combination
    """You are designing a SYNERGISTIC algorithm where two methods work together cooperatively.

METHOD 1:
{parent1}

METHOD 2:
{parent2}

Your task:
1. Design how these two methods can cooperate (parallel, sequential, hierarchical)
2. Create communication/information-sharing mechanisms between methods
3. Define when and how each method contributes to the search
4. Ensure synergy: combined performance > individual performance
5. Optimize for large-scale CVRP instances

Provide comprehensive algorithm design with pseudocode."""
]


# ============================================================================
# Fresh Generation Prompt Variants
# ============================================================================

FRESH_GENERATION_PROMPTS = [
    # Variant 1: Innovation focus
    """Design a NOVEL algorithm for the following problem. Focus on innovation and creativity.

Problem Statement:
{problem}

Research Context:
{research_context}

Related Papers (Algorithm Details):
{local_papers}

Your task:
1. Think of an approach NOT commonly used for this problem
2. Draw inspiration from other domains or recent research (review papers above)
3. Focus on a unique mechanism or strategy
4. Ensure algorithmic correctness and feasibility
5. Provide clear pseudocode

Be creative and avoid standard textbook solutions!""",

    # Variant 2: SOTA competitive focus
    """Design a COMPETITIVE algorithm that can potentially outperform state-of-the-art methods.

Problem Statement:
{problem}

Research Context:
{research_context}

Related Papers (Algorithm Details):
{local_papers}

Your task:
1. Review the papers above to understand SOTA mechanisms
2. Identify weaknesses in current SOTA algorithms (HGS, SISR, FILO2, LKH3, AILS-II)
3. Design mechanisms that address these weaknesses
4. Focus on computational efficiency for large instances
5. Include sophisticated local search or perturbation mechanisms
6. Provide detailed pseudocode

Aim for publication-worthy ideas!""",

    # Variant 3: Hybrid metaheuristic focus
    """Design a HYBRID metaheuristic combining multiple search paradigms.

Problem Statement:
{problem}

Research Context:
{research_context}

Related Papers (Algorithm Details):
{local_papers}

Your task:
1. Review papers to identify successful metaheuristic paradigms
2. Select 2-3 complementary metaheuristic paradigms that work well together
3. Design how they work together (portfolio, sequential, parallel)
4. Create adaptive mechanisms for parameter/strategy selection
5. Ensure balance between exploration and exploitation
6. Provide comprehensive pseudocode

Focus on creating synergies between different search methods!""",

    # Variant 4: Decomposition focus
    """Design an algorithm using PROBLEM DECOMPOSITION or hierarchical strategies.

Problem Statement:
{problem}

Research Context:
{research_context}

Related Papers (Algorithm Details):
{local_papers}

Your task:
1. Review papers for decomposition strategies used in similar problems
2. Identify how to decompose the problem into subproblems
3. Design algorithms for each subproblem
4. Create mechanisms to combine/merge subproblem solutions
5. Consider multi-level or iterative refinement approaches
6. Provide detailed pseudocode

Think about divide-and-conquer or hierarchical decomposition!""",

    # Variant 5: Adaptive/learning focus
    """Design an ADAPTIVE algorithm that learns during search execution.

Problem Statement:
{problem}

Research Context:
{research_context}

Related Papers (Algorithm Details):
{local_papers}

Your task:
1. Review papers for adaptive mechanisms used in SOTA algorithms
2. Identify what patterns/information to learn during search
3. Design adaptive mechanisms (parameter tuning, operator selection, etc.)
4. Create feedback loops that improve search over time
5. Avoid ML/neural networks (use algorithmic learning only)
6. Provide complete pseudocode

Focus on self-tuning and search trajectory exploitation!""",

    # Variant 6: Efficiency focus
    """Design an EFFICIENT algorithm optimized for large-scale instances.

Problem Statement:
{problem}

Research Context:
{research_context}

Related Papers (Algorithm Details):
{local_papers}

Your task:
1. Review papers for efficiency techniques (data structures, incremental updates)
2. Prioritize computational efficiency (fast neighbor evaluation, incremental updates)
3. Design sophisticated data structures for speed
4. Use smart pruning or candidate list strategies
5. Balance solution quality with computational cost
6. Provide detailed pseudocode with complexity analysis

Aim for algorithms that scale to 1000+ customers!"""
]


# ============================================================================
# Mutation Operator
# ============================================================================

def mutate_individual(
    individual: Individual,
    state: Dict[str, Any],
    agent_id: int,
    num_reflections: int,
    generation: int
) -> Individual:
    """
    Mutate an individual by refining it through reflection loops.

    This is essentially running the ideation agent's reflection process
    on an existing idea using a DIFFERENT agent than the creator.

    Args:
        individual: Individual to mutate
        state: Current system state (with problem context)
        agent_id: Agent to use for mutation (must be different from creator)
        num_reflections: Number of reflection loops
        generation: Current generation number

    Returns:
        New mutated individual
    """
    # Create a modified state that uses the existing idea as the starting point
    mutation_state = state.copy()
    mutation_state["existing_idea_to_refine"] = individual.content
    mutation_state["mutation_context"] = f"""
You are refining an existing algorithm idea through critical reflection.

EXISTING ALGORITHM:
{individual.content}

Your task:
1. Critically analyze the existing algorithm for weaknesses, inefficiencies, or gaps
2. Propose specific improvements to address these issues
3. Refine the algorithm while maintaining its core strengths
4. Ensure the refined version is better than the original
5. Provide complete pseudocode for the improved algorithm

Focus on INCREMENTAL IMPROVEMENT while preserving what works well.
"""

    # Run ideation with the mutation context
    result = ideation_agent(
        mutation_state,
        agent_id=agent_id,
        num_reflections=num_reflections
    )

    # Create new individual with mutation metadata
    from tools.population_manager import PopulationManager
    pm = PopulationManager()

    mutated = pm.create_individual(
        content=result["final_idea"],
        generation=generation,
        agent_id=agent_id,
        operation="mutation",
        parent_ids=[individual.id]
    )

    # Update metadata to track mutation
    mutated.metadata["last_modified_by_agent_id"] = agent_id
    mutated.metadata["operation_history"].append({
        "generation": generation,
        "operation": "mutation",
        "agent_id": agent_id,
        "parent_id": individual.id
    })

    return mutated


# ============================================================================
# Crossover Operator
# ============================================================================

def crossover_individuals(
    parent1: Individual,
    parent2: Individual,
    state: Dict[str, Any],
    agent_id: int,
    generation: int
) -> Individual:
    """
    Create offspring by combining two parent individuals.

    Uses randomly selected prompt variant to encourage diversity.
    Agent must be different from both parents.

    Args:
        parent1: First parent individual
        parent2: Second parent individual
        state: Current system state
        agent_id: Agent to use for crossover (different from both parents)
        generation: Current generation number

    Returns:
        New offspring individual
    """
    # Randomly select crossover prompt variant
    crossover_prompt = random.choice(CROSSOVER_PROMPTS)

    # Format prompt with parent contents
    formatted_prompt = crossover_prompt.format(
        parent1=parent1.content,
        parent2=parent2.content
    )

    # Create state for crossover
    crossover_state = state.copy()
    crossover_state["crossover_prompt"] = formatted_prompt

    # Use the ideation agent with custom prompt
    # We'll create a simple wrapper that uses the crossover prompt
    from langchain_core.messages import HumanMessage

    # Get the LLM for this agent
    from agents.ideation import get_llm_for_agent
    llm = get_llm_for_agent(agent_id)

    # Generate crossover offspring
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    offspring_content = response.content

    # Create new individual with crossover metadata
    from tools.population_manager import PopulationManager
    pm = PopulationManager()

    offspring = pm.create_individual(
        content=offspring_content,
        generation=generation,
        agent_id=agent_id,
        operation="crossover",
        parent_ids=[parent1.id, parent2.id]
    )

    # Update metadata to track crossover
    offspring.metadata["last_modified_by_agent_id"] = agent_id
    offspring.metadata["operation_history"].append({
        "generation": generation,
        "operation": "crossover",
        "agent_id": agent_id,
        "parent_ids": [parent1.id, parent2.id]
    })

    return offspring


# ============================================================================
# Fresh Generation Operator
# ============================================================================

def generate_fresh_individual(
    state: Dict[str, Any],
    agent_id: int,
    num_reflections: int,
    generation: int
) -> Individual:
    """
    Generate a completely fresh individual from scratch.

    Uses randomly selected prompt variant to encourage diversity.
    Includes full research context and paper summaries for informed generation.

    Args:
        state: Current system state (with problem, research_context, local_papers)
        agent_id: Agent to use for generation
        num_reflections: Number of reflection loops
        generation: Current generation number

    Returns:
        New fresh individual
    """
    # Randomly select fresh generation prompt variant
    fresh_prompt = random.choice(FRESH_GENERATION_PROMPTS)

    # Extract context from state (same as ideation agent does)
    problem = state.get("problem", "")
    research_context = state.get("research_context", "No research context available")
    local_papers = state.get("local_papers", [])

    # Prepare context (NO TRUNCATION - use all papers)
    research_summary = research_context[:5000] if research_context else "No research available"

    if local_papers:
        papers_summary = "\n\n---PAPER---\n\n".join(local_papers)
        print(f"  [Fresh Generation] Using {len(local_papers)} paper summaries (~{len(papers_summary):,} chars)")
    else:
        papers_summary = "No local papers available"

    # Format prompt with problem AND full research context
    formatted_prompt = fresh_prompt.format(
        problem=problem,
        research_context=research_summary,
        local_papers=papers_summary
    )

    # Create modified state with fresh generation prompt
    fresh_state = state.copy()
    fresh_state["fresh_generation_prompt"] = formatted_prompt
    fresh_state["custom_ideation_prompt"] = formatted_prompt

    # Run ideation agent with reflections
    result = ideation_agent(
        fresh_state,
        agent_id=agent_id,
        num_reflections=num_reflections
    )

    # Create new individual
    from tools.population_manager import PopulationManager
    pm = PopulationManager()

    individual = pm.create_individual(
        content=result["final_idea"],
        generation=generation,
        agent_id=agent_id,
        operation="fresh",
        parent_ids=[]
    )

    return individual


# ============================================================================
# Batch Operations
# ============================================================================

def create_mutation_offspring(
    parents: List[Individual],
    state: Dict[str, Any],
    available_agents: List[int],
    num_reflections: int,
    generation: int,
    num_offspring: int,
    enforce_diversity: bool = True
) -> List[Individual]:
    """
    Create multiple mutation offspring.

    Args:
        parents: Pool of parent individuals to mutate
        state: System state
        available_agents: List of active agent IDs
        num_reflections: Reflection loops per mutation
        generation: Current generation
        num_offspring: Number of offspring to create
        enforce_diversity: Whether to enforce agent diversity

    Returns:
        List of mutated individuals
    """
    offspring = []

    for _ in range(num_offspring):
        # Select random parent
        parent = random.choice(parents)

        # Select agent (different from parent creator if enforcing diversity)
        if enforce_diversity:
            excluded = [parent.metadata["last_modified_by_agent_id"]]
            agent_id = select_different_agent(excluded, available_agents)
        else:
            agent_id = random.choice(available_agents)

        # Mutate
        child = mutate_individual(parent, state, agent_id, num_reflections, generation)
        offspring.append(child)

    return offspring


def create_crossover_offspring(
    parents: List[Individual],
    state: Dict[str, Any],
    available_agents: List[int],
    generation: int,
    num_offspring: int,
    enforce_diversity: bool = True
) -> List[Individual]:
    """
    Create multiple crossover offspring.

    Args:
        parents: Pool of parent individuals for crossover
        state: System state
        available_agents: List of active agent IDs
        generation: Current generation
        num_offspring: Number of offspring to create
        enforce_diversity: Whether to enforce agent diversity

    Returns:
        List of crossover offspring
    """
    offspring = []

    for _ in range(num_offspring):
        # Select two different parents
        parent1, parent2 = random.sample(parents, 2)

        # Select agent (different from both parents if enforcing diversity)
        if enforce_diversity:
            excluded = [
                parent1.metadata["last_modified_by_agent_id"],
                parent2.metadata["last_modified_by_agent_id"]
            ]
            agent_id = select_different_agent(excluded, available_agents)
        else:
            agent_id = random.choice(available_agents)

        # Crossover
        child = crossover_individuals(parent1, parent2, state, agent_id, generation)
        offspring.append(child)

    return offspring


def create_fresh_offspring(
    state: Dict[str, Any],
    available_agents: List[int],
    num_reflections: int,
    generation: int,
    num_offspring: int
) -> List[Individual]:
    """
    Create multiple fresh individuals.

    Args:
        state: System state
        available_agents: List of active agent IDs
        num_reflections: Reflection loops per individual
        generation: Current generation
        num_offspring: Number of fresh individuals to create

    Returns:
        List of fresh individuals
    """
    offspring = []

    for _ in range(num_offspring):
        # Randomly select agent for each fresh individual
        agent_id = random.choice(available_agents)

        # Generate fresh individual
        individual = generate_fresh_individual(state, agent_id, num_reflections, generation)
        offspring.append(individual)

    return offspring
