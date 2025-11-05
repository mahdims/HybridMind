"""
Standalone script for CVRP (Capacitated Vehicle Routing Problem) optimization ideation.

This script uses the EVOLUTIONARY SEARCH system to evolve innovative algorithms
for solving large-scale CVRP problems using heuristic and metaheuristic methods.

Usage:
    python run_cvrp_ideation.py              # Start new evolution
    python run_cvrp_ideation.py --continue   # Continue latest evolution

The system will:
1. Initialize population of 20 innovative CVRP algorithms
2. Evolve for 10 generations using mutation, crossover, and fresh generation
3. Use different SOTA agents to ensure diverse perspectives
4. Evaluate ideas across correctness, SOTA competitiveness, and efficiency
5. Produce top 5 best ideas with detailed feedback and genealogy
6. Save population for future continuation
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the evolutionary system
from evolutionary_engine import run_evolutionary_ideation


def main():
    """Run CVRP optimization ideation."""

    print("\n" + "="*80)
    print("CVRP OPTIMIZATION - INNOVATIVE METAHEURISTICS IDEATION")
    print("="*80)
    print("\nObjective: Generate breakthrough ideas for large-scale CVRP solving")
    print("Focus Areas:")
    print("  - Hybrid approaches combining SOTA algorithms (HGS, ALNS, AILS, SISR, LKH3)")
    print("  - Novel search mechanisms for exploration-exploitation balance")
    print("  - Computational efficiency for 1000+ customer instances")
    print("  - Innovative metaheuristic components")
    print("="*80)

    # Comprehensive CVRP problem description
    problem = """
    Design an innovative algorithm or hybrid approach for solving large-scale
    Capacitated Vehicle Routing Problems (CVRP) with 1000+ customers using
    heuristic and metaheuristic methods.

    CONTEXT:
    The CVRP is a classical NP-hard combinatorial optimization problem where the
    goal is to find optimal routes for a fleet of vehicles with limited capacity
    to serve a set of customers with known demands, minimizing total distance/cost.

    STATE-OF-THE-ART ALGORITHMS (for reference and potential combination):
    - HGS (Hybrid Genetic Search): Population-based with advanced crossover operators
    - ALNS (Adaptive Large Neighborhood Search): Destroy-and-repair with adaptive weights
    - AILS (Adaptive Iterated Local Search): Perturbation-based with adaptive parameters
    - SISR (Slack Induction by String Removals): Route optimization via string removals
    - LKH3 (Lin-Kernighan-Helsgaun): Advanced k-opt moves with sophisticated data structures
    - FILO and FILO2: 

    OBJECTIVE:
    Develop novel algorithmic ideas that either:

    1. HYBRID APPROACHES: Intelligently combine strengths of multiple SOTA algorithms
       - Example: Integrate HGS population management with ALNS repair operators
       - Example: Use SISR string removal insights within LKH3 k-opt framework
       - Example: Adaptive switching between AILS perturbations and ALNS destroy operators
       - Consider: How can different algorithms complement each other?
       - Consider: When should the system switch between different methods?

    2. INNOVATIVE SEARCH MECHANISMS: Create new metaheuristic components
       - Better balance between exploration (diversification) and exploitation (intensification)
       - Adaptive parameter tuning mechanisms that learn from search history
       - Novel neighborhood structures beyond traditional 2-opt, Or-opt, cross-exchange
       - Multi-level or hierarchical search strategies
       - Dynamic restart strategies based on search progress indicators

    3. COMPUTATIONAL EFFICIENCY: Optimize for large-scale instances (1000+ customers)
       - Efficient data structures for fast neighbor evaluation (delta evaluation, move caching)
       - Parallel or GPU-accelerated components (route evaluation, population management)
       - Incremental evaluation techniques (avoid full recalculation)
       - Smart pruning strategies (granular neighborhoods, candidate lists)
       - Memory-efficient representations for large solution populations

    INNOVATION AREAS TO EXPLORE:

    - Portfolio-Based Approaches:
      * Maintain diverse solution strategies simultaneously
      * Automatically select best-performing method per instance
      * Parallel search with different algorithm configurations

    - Multi-Objective Considerations:
      * Solution robustness (stability under demand uncertainty)
      * Route balance and fairness
      * Sustainability metrics (fuel consumption, emissions)

    - Problem Structure Exploitation:
      * Spatial clustering of customers (hierarchical decomposition)
      * Route template extraction and reuse
      * Problem reduction techniques

    - Adaptive Memory Mechanisms:
      * Advanced tabu structures (attribute-based, long-term memory)
      * Elite solution pool management (diversity preservation)
      * Solution path relinking
      * Frequency-based intensification/diversification

    - Dynamic Parameter Adaptation:
      * Self-tuning based on search landscape characteristics
      * Population size adaptation in genetic algorithms
      * Temperature schedules in simulated annealing
      * Adaptive destroy/repair ratios in ALNS

    EVALUATION CRITERIA:
    - Solution Quality: Gap to best-known solutions on standard benchmarks (Uchoa, CVRPLIB)
    - Computational Efficiency: Time to reach high-quality solutions (anytime performance)
    - Scalability: Performance degradation on instances with varying sizes (100 to 1000+ customers)
    - Robustness: Consistent performance across different instance types (clustered, random, etc.)
    - Novelty: Originality of approach and theoretical soundness
    - Practical Implementability: Code complexity, ease of tuning, dependencies

    DELIVERABLES:
    Propose a detailed algorithmic framework including:
    1. High-level algorithm structure and pseudocode
    2. Key components and their interactions (with diagrams if helpful)
    3. Parameter adaptation strategies (how parameters change during search)
    4. Complexity analysis (time and space for each major operation)
    5. Expected advantages over existing methods (why should this work better?)
    6. Potential limitations and mitigation strategies
    7. Suggested benchmark instances for validation
    8. Implementation recommendations (data structures, key optimizations)

    CONSTRAINTS AND CONSIDERATIONS:
    - Algorithm should be deterministic or use controlled randomness
    - Should handle edge cases (single customer, vehicle capacity constraints)
    - Should work only for Euclidean distances
    - Consider only homogeneous fleet scenarios
    - NO Machine Learning Integration
    - We ONLY have vehcile capacity 
    - CVRP instances has different charectristics (random or cluster customer, central depot or on edge) and we can have adpative behaviour for each category.

    Be creative and think beyond incremental improvements - we want breakthrough ideas
    that could advance the state-of-the-art in large-scale CVRP solving!
    Consider unconventional approaches, cross-pollination from other optimization domains,
    and novel combinations of existing techniques.
    """

    # Check for continue argument
    continue_run_id = None
    if len(sys.argv) > 1 and sys.argv[1] == "--continue":
        # Continue latest CVRP evolution
        from tools.population_manager import PopulationManager
        pm = PopulationManager()

        # Find most recent CVRP evolution
        populations = pm.list_populations()
        for pop in populations:
            loaded_pop = pm.load_population(pop['run_id'])
            if "CVRP" in loaded_pop.metadata.problem_statement or "Vehicle Routing" in loaded_pop.metadata.problem_statement:
                continue_run_id = pop['run_id']
                print(f"\n✓ Found CVRP evolution to continue: {continue_run_id}")
                print(f"  Last updated: {pop['last_updated']}")
                print(f"  Generation: {pop['current_generation']}")
                break

        if not continue_run_id:
            print("\n⚠️  No previous CVRP evolution found. Starting new evolution.")

    print("\nStarting EVOLUTIONARY SEARCH for CVRP optimization...")
    if continue_run_id:
        print(f"Continuing from generation {populations[0]['current_generation']}...")
    else:
        print("Initializing population of 20 innovative CVRP algorithms...")
    print("This will take approximately 30-60 minutes for 10 generations.\n")

    # Run the evolutionary ideation system
    try:
        population = run_evolutionary_ideation(
            problem=problem,
            problem_name="cvrp",       # Cache research as cvrp_research.txt
            enable_research=False,     # Disable online research (Tavily errors)
            enable_local_papers=True,  # Include any local CVRP papers from data/papers/
            continue_run_id=continue_run_id
        )

        # Display results
        print("\n" + "="*80)
        print("CVRP OPTIMIZATION EVOLUTION COMPLETE!")
        print("="*80)
        print("\n✓ Population evolved through", population.metadata.current_generation, "generations")
        print("✓ Final population: 20 innovative CVRP algorithms")
        print("✓ Evolutionary operators used:")
        print("    - Mutation (70%): Refined ideas using different agents")
        print("    - Crossover (20%): Combined parent ideas with 5 prompt variants")
        print("    - Fresh (10%): Generated new ideas with 6 prompt variants")
        print("✓ Evaluated across 5 dimensions with weighted scoring:")
        print("    - Correctness (30%) - CRITICAL")
        print("    - SOTA Competitiveness (30%) - CRITICAL")
        print("    - Low Time Complexity (20%) - IMPORTANT")
        print("    - Clear Pseudocode (10%)")
        print("    - Novelty (10%)")

        best = population.get_best(1)[0]
        print(f"\n✓ Best fitness achieved: {best.fitness:.2f}/5.0 ({best.fitness/5.0*100:.0f}%)")
        print(f"✓ Top 5 summary saved to: output/evolution_summary_{population.metadata.run_id}.md")
        print(f"✓ Population saved to: population/population_{population.metadata.run_id}.json")

        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print(f"1. Review top 5 ideas in: output/evolution_summary_{population.metadata.run_id}.md")
        print("2. Analyze the best algorithms:")
        print("   - Check fitness scores and rankings")
        print("   - Review judge feedback for each evaluation dimension")
        print("   - Examine genealogy (how ideas evolved from parents)")
        print("   - Look for novel hybrid combinations from crossover")
        print("3. Continue evolution if needed:")
        print(f"   python run_cvrp_ideation.py --continue")
        print("   OR")
        print(f"   python main.py --mode evolution --continue {population.metadata.run_id}")
        print("4. Prototype promising algorithms:")
        print("   - Start with highest-ranked ideas")
        print("   - Combine complementary strengths from multiple ideas")
        print("   - Test on CVRP benchmarks (Uchoa, CVRPLIB)")

        print("\n" + "="*80)
        print("EVOLUTIONARY INSIGHTS:")
        print("="*80)
        print("- Review operation history in population file to see how ideas evolved")
        print("- Compare ideas from different generations to track improvement")
        print("- Identify which operators (mutation/crossover/fresh) produced best ideas")
        print("- Look for emergent patterns in successful algorithms")
        print("- Consider running longer evolution (edit parameters.txt: max_generations)")
        print("="*80)

    except Exception as e:
        print("\n❌ Evolution failed. Error:", str(e))
        print("\nPlease check:")
        print("  - At least one API key is configured in .env")
        print("  - API keys are valid and have sufficient credits")
        print("  - Network connection is working")
        print("  - parameters.txt is properly configured")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
