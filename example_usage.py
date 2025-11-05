"""
Example usage of the Algorithmic Multi-Agent Ideation System.
This script demonstrates various ways to use the system programmatically.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_1_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)

    from main import run_ideation_system

    problem = """
    Design an efficient algorithm for merging K sorted linked lists
    into one sorted linked list. Optimize for time complexity.
    """

    result = run_ideation_system(
        problem=problem,
        enable_research=False,  # Disable for faster execution
        enable_local_papers=True,
        interactive=False
    )

    if result:
        print("\n✓ Success! Check output/ directory for results.")


def example_2_custom_workflow():
    """Example 2: Building and running custom workflow."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Workflow")
    print("="*80)

    from graph import build_workflow

    # Build custom workflow
    app = build_workflow(
        enable_research=False,
        enable_local_papers=False
    )

    # Define problem
    problem = "Design a cache eviction policy that minimizes cache misses."

    # Initial state
    initial_state = {
        "problem": problem,
        "research_context": "LRU and LFU are common strategies.",
        "local_papers": [],
        "agent_ideas": [],
        "evaluations": [],
        "presentation": ""
    }

    # Run workflow
    config = {"configurable": {"thread_id": "example_2"}}
    result = app.invoke(initial_state, config=config)

    print(f"\n✓ Generated presentation ({len(result['presentation'])} chars)")


def example_3_step_by_step():
    """Example 3: Running agents step by step."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("="*80)

    from agents.ideation import ideation_agent
    from agents.evaluator import evaluate_ideas, format_evaluation_summary
    from agents.presenter import generate_quick_summary

    # Step 1: Define problem and context
    state = {
        "problem": "Find the longest increasing subsequence in an array.",
        "research_context": "Dynamic programming is a common approach.",
        "local_papers": []
    }

    # Step 2: Run agents individually
    print("\nRunning Agent 1...")
    agent1_result = ideation_agent(state, agent_id=1, num_reflections=2)

    print("\nRunning Agent 2...")
    agent2_result = ideation_agent(state, agent_id=2, num_reflections=2)

    print("\nRunning Agent 3...")
    agent3_result = ideation_agent(state, agent_id=3, num_reflections=2)

    # Step 3: Collect ideas
    state["agent_ideas"] = [agent1_result, agent2_result, agent3_result]

    # Step 4: Evaluate
    print("\nEvaluating ideas...")
    evaluations = evaluate_ideas(state)
    state["evaluations"] = evaluations

    # Step 5: Display summary
    print("\n" + format_evaluation_summary(evaluations))
    print(generate_quick_summary(state))


def example_4_batch_processing():
    """Example 4: Process multiple problems in batch."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing")
    print("="*80)

    from main import run_ideation_system

    problems = [
        "Design an algorithm to detect cycles in a directed graph.",
        "Design an algorithm to find the median of two sorted arrays.",
        "Design an algorithm for efficient pattern matching in text."
    ]

    results = []
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"Processing Problem {i}/{len(problems)}")
        print(f"{'='*80}")

        result = run_ideation_system(
            problem=problem,
            enable_research=False,
            enable_local_papers=False,
            interactive=False
        )

        results.append({
            "problem": problem,
            "success": result is not None
        })

    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    for i, res in enumerate(results, 1):
        status = "✓" if res["success"] else "✗"
        print(f"{status} Problem {i}: {res['problem'][:60]}...")


def example_5_custom_evaluation():
    """Example 5: Custom evaluation aspects."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Evaluation")
    print("="*80)

    from langchain_openai import ChatOpenAI
    import os

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Sample agent ideas
    ideas = [
        "Use binary search tree for O(log n) lookup",
        "Use hash table for O(1) average lookup",
        "Use sorted array with binary search for O(log n) lookup"
    ]

    # Custom evaluation aspect: "Practicality"
    eval_prompt = f"""
    Evaluate these three data structure choices for practicality:

    1. {ideas[0]}
    2. {ideas[1]}
    3. {ideas[2]}

    Score each 1-10 on PRACTICALITY considering:
    - Implementation complexity
    - Real-world performance
    - Maintenance burden
    - Common use cases

    Format:
    Option 1: X/10 - reason
    Option 2: Y/10 - reason
    Option 3: Z/10 - reason
    """

    response = llm.invoke(eval_prompt)
    print("\nCustom Evaluation Result:")
    print(response.content)


def example_6_accessing_results():
    """Example 6: Programmatically access results."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Accessing Results")
    print("="*80)

    from graph import build_workflow
    from agents.evaluator import parse_evaluation_scores

    app = build_workflow(enable_research=False, enable_local_papers=False)

    problem = "Design an algorithm for topological sorting."

    initial_state = {
        "problem": problem,
        "research_context": "",
        "local_papers": [],
        "agent_ideas": [],
        "evaluations": [],
        "presentation": ""
    }

    config = {"configurable": {"thread_id": "example_6"}}
    result = app.invoke(initial_state, config=config)

    # Access different parts of the result
    print(f"\nProblem: {result['problem']}")
    print(f"\nNumber of agents: {len(result['agent_ideas'])}")

    for idea in result['agent_ideas']:
        agent_id = idea['agent_id']
        reflections = len(idea['reflections'])
        print(f"  Agent {agent_id}: {reflections} reflections")

    # Parse scores
    if result['evaluations']:
        scores = parse_evaluation_scores(result['evaluations'])
        print("\nScores by agent:")
        for agent_id, aspect_scores in scores.items():
            avg = sum(aspect_scores.values()) / len(aspect_scores)
            print(f"  Agent {agent_id}: {avg:.1f}/10 average")


def example_7_cvrp_optimization():
    """Example 7: Large-scale CVRP optimization using EVOLUTIONARY SEARCH."""
    print("\n" + "="*80)
    print("EXAMPLE 7: CVRP Optimization - Evolutionary Search")
    print("="*80)
    print("\nNOTE: This example uses EVOLUTIONARY MODE to evolve a population")
    print("of CVRP algorithms over 10 generations (configurable in parameters.txt)")
    print("\nFor better results, use the standalone script:")
    print("  python run_cvrp_ideation.py\n")

    from evolutionary_engine import run_evolutionary_ideation

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

    OBJECTIVE:
    Develop novel algorithmic ideas that either:
    1. HYBRID APPROACHES: Intelligently combine strengths of multiple SOTA algorithms
       - Example: Integrate HGS population management with ALNS repair operators
       - Example: Use SISR string removal insights within LKH3 k-opt framework
       - Example: Adaptive switching between AILS perturbations and ALNS destroy operators

    2. INNOVATIVE SEARCH MECHANISMS: Create new metaheuristic components
       - Better balance between exploration (diversification) and exploitation (intensification)
       - Adaptive parameter tuning mechanisms that learn from search history
       - Novel neighborhood structures beyond traditional 2-opt, Or-opt, cross-exchange
       - Multi-level or hierarchical search strategies

    3. COMPUTATIONAL EFFICIENCY: Optimize for large-scale instances (1000+ customers)
       - Efficient data structures for fast neighbor evaluation
       - Parallel or GPU-accelerated components
       - Incremental evaluation techniques
       - Smart pruning strategies

    INNOVATION AREAS TO EXPLORE:
    - Machine learning integration (predict good destroy operators, guide search)
    - Portfolio-based approaches (maintain diverse solution strategies)
    - Multi-objective considerations (robustness, fairness, sustainability)
    - Problem structure exploitation (clustering, decomposition)
    - Adaptive memory mechanisms (tabu lists, elite solutions)
    - Dynamic parameter adaptation based on search landscape

    EVALUATION CRITERIA:
    - Solution quality (gap to best-known solutions on benchmarks)
    - Computational efficiency (time to reach good solutions)
    - Scalability (performance on instances with varying sizes)
    - Robustness (consistent performance across different instance types)
    - Novelty and theoretical soundness
    - Practical implementability

    DELIVERABLES:
    Propose a detailed algorithmic framework including:
    - High-level algorithm structure and pseudocode
    - Key components and their interactions
    - Parameter adaptation strategies
    - Complexity analysis (time and space)
    - Expected advantages over existing methods
    - Potential limitations and mitigation strategies

    Be creative and think beyond incremental improvements - we want breakthrough ideas!
    """

    population = run_evolutionary_ideation(
        problem=problem,
        problem_name="cvrp",  # Cache research as cvrp_research.txt
        enable_research=False,  # Disabled due to Tavily API issues (can enable if you have valid key)
        enable_local_papers=True,  # Include any local CVRP papers
        continue_run_id=None  # Start new evolution
    )

    if population:
        print("\n" + "="*80)
        print("CVRP OPTIMIZATION EVOLUTION COMPLETE")
        print("="*80)
        print(f"✓ Population evolved through {population.metadata.current_generation} generations")
        print("✓ 20 innovative CVRP algorithms in final population")
        print("✓ Evolutionary operators: 70% mutation, 20% crossover, 10% fresh")
        print("✓ Agent diversity enforced (different agent for mutation/crossover)")
        print(f"✓ Top 5 summary saved to: output/evolution_summary_{population.metadata.run_id}.md")
        print(f"✓ Population saved to: population/population_{population.metadata.run_id}.json")

        best = population.get_best(1)[0]
        print(f"\nBest fitness achieved: {best.fitness:.2f}/5.0 ({best.fitness/5.0*100:.0f}%)")
        print("\nTo continue evolution:")
        print(f"  python main.py --mode evolution --continue {population.metadata.run_id}")
        print("\nReview the top 5 ideas to identify:")
        print("  - Most promising hybrid approaches")
        print("  - Novel search mechanisms from crossover operations")
        print("  - Innovative exploration-exploitation balance strategies")
        print("  - Ideas with best SOTA competitiveness scores")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ALGORITHMIC MULTI-AGENT IDEATION SYSTEM")
    print("EXAMPLE USAGE DEMONSTRATIONS")
    print("="*80)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Workflow", example_2_custom_workflow),
        ("Step-by-Step", example_3_step_by_step),
        ("Batch Processing", example_4_batch_processing),
        ("Custom Evaluation", example_5_custom_evaluation),
        ("Accessing Results", example_6_accessing_results),
        ("CVRP Optimization - Innovative Metaheuristics", example_7_cvrp_optimization),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nNote: Most examples require at least one API key for SOTA Agents (see .env.example for options).")
    print("\nTo run all examples, uncomment the function calls below.")
    print("Or run individual examples by calling their functions.")
    print("\nRECOMMENDED: Try Example 7 (CVRP) for advanced optimization ideation!")

    # Uncomment to run examples:
    example_1_basic_usage()
    # example_2_custom_workflow()
    # example_3_step_by_step()
    # example_4_batch_processing()
    # example_5_custom_evaluation()
    # example_6_accessing_results()
    # example_7_cvrp_optimization()  # <-- NEW: Advanced CVRP optimization


if __name__ == "__main__":
    main()
