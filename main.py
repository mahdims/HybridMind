"""
Main Entry Point for Algorithmic Multi-Agent Ideation System
Run this file to execute the complete workflow.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

from graph import build_workflow
from agents.presenter import save_presentation


def print_banner():
    """Print system banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         ALGORITHMIC MULTI-AGENT IDEATION SYSTEM v1.0                        ║
║                                                                              ║
║         A LangGraph-based system for collaborative algorithm design         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("\n⚠️  ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with the required variables.")
        print("See .env.example for template.\n")
        return False

    return True


def get_problem_from_user() -> str:
    """Get problem statement from user input."""
    print("\n" + "="*80)
    print("PROBLEM DEFINITION")
    print("="*80)
    print("\nDescribe the algorithmic problem you want to solve.")
    print("(Press Enter twice or Ctrl+D to finish)\n")

    lines = []
    print("Problem: ", end="")

    try:
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
    except EOFError:
        pass

    problem = " ".join(lines).strip()

    if not problem:
        print("\n⚠️  No problem provided. Using default example problem.")
        problem = """
        Design an efficient algorithm for finding the k-th smallest
        element in the union of two sorted arrays without merging them.
        Optimize for time complexity.
        """

    return problem


def get_default_problem() -> str:
    """Return a default problem for quick testing."""
    return """
    Design an efficient algorithm for finding the k-th smallest
    element in the union of two sorted arrays without merging them.
    The algorithm should optimize for time complexity and handle edge cases.
    """


def run_ideation_system(problem: str,
                       enable_research: bool = True,
                       enable_local_papers: bool = True,
                       interactive: bool = True):
    """
    Run the complete ideation system.

    Args:
        problem: Problem statement
        enable_research: Enable online research
        enable_local_papers: Enable local paper loading
        interactive: Whether to pause between phases

    Returns:
        Final state dictionary
    """
    print_banner()

    # Build workflow
    print("\n[System] Building workflow graph...")
    app = build_workflow(
        enable_research=enable_research,
        enable_local_papers=enable_local_papers
    )

    # Prepare initial state
    initial_state = {
        "problem": problem,
        "research_context": "",
        "local_papers": [],
        "agent_ideas": [],
        "evaluations": [],
        "presentation": ""
    }

    # Run workflow
    print("\n[System] Starting multi-agent ideation process...")
    print(f"\nProblem: {problem}\n")

    config = {"configurable": {"thread_id": "1"}}

    try:
        # Execute workflow
        result = app.invoke(initial_state, config=config)

        # Display results
        print("\n" + "="*80)
        print("FINAL PRESENTATION")
        print("="*80)
        print(result["presentation"])

        # Save presentation
        filepath = save_presentation(result["presentation"])

        print("\n" + "="*80)
        print("EXECUTION COMPLETE")
        print("="*80)
        num_agents = len(result.get("agent_ideas", []))
        print(f"\n✓ Total agents: {num_agents}")
        print(f"✓ Evaluation aspects: {len(result.get('evaluations', []))}")
        print(f"✓ Presentation saved: {filepath}")
        print(f"✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return result

    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def list_all_populations():
    """List all available population files."""
    from tools.population_manager import PopulationManager

    pm = PopulationManager()
    populations = pm.list_populations()

    if not populations:
        print("\nNo population files found.")
        return

    print(f"\n{'='*100}")
    print(f"AVAILABLE POPULATIONS ({len(populations)} total)")
    print(f"{'='*100}\n")

    for i, pop in enumerate(populations, 1):
        print(f"{i}. Run ID: {pop['run_id']}")
        print(f"   Created: {pop['created_at']}")
        print(f"   Last Updated: {pop['last_updated']}")
        print(f"   Generation: {pop['current_generation']}")
        print(f"   Population Size: {pop['population_size']}")
        print(f"   Problem: {pop['problem_preview']}")
        print(f"   File: {pop['filepath']}")
        print()


def main():
    """Main function."""
    # Load environment variables
    load_dotenv()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Algorithmic Multi-Agent Ideation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode (single run)
  python main.py --problem "Design algorithm for CVRP..."

  # Evolution mode (start new evolution)
  python main.py --mode evolution --problem "Design algorithm for CVRP..."

  # Continue most recent evolution
  python main.py --mode evolution --continue

  # Continue specific evolution
  python main.py --mode evolution --continue run_20250103_143052

  # List all populations
  python main.py --list-populations
        """
    )

    parser.add_argument(
        "--mode",
        choices=["standard", "evolution"],
        default="standard",
        help="Run mode: 'standard' for single run, 'evolution' for evolutionary search (default: standard)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Problem statement (or use interactive mode)"
    )
    parser.add_argument(
        "--continue",
        type=str,
        nargs='?',
        const='latest',
        dest='continue_run',
        metavar='RUN_ID',
        help="Continue evolution from run ID (use without argument for latest, or specify run_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--list-populations",
        action="store_true",
        help="List all available population files and exit"
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        help="Disable online research"
    )
    parser.add_argument(
        "--no-papers",
        action="store_true",
        help="Disable local paper loading"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without pauses"
    )
    parser.add_argument(
        "--problem-name",
        type=str,
        default="problem",
        help="Short name for problem (e.g., 'cvrp') used for research cache filename (default: 'problem')"
    )

    args = parser.parse_args()

    # Handle list-populations command
    if args.list_populations:
        list_all_populations()
        return

    # Get problem (not needed for continue mode)
    if args.continue_run and args.continue_run != 'latest':
        # Continuing specific run - load problem from population
        from tools.population_manager import PopulationManager
        pm = PopulationManager()
        try:
            pop = pm.load_population(args.continue_run)
            problem = pop.metadata.problem_statement
            print(f"\n[System] Continuing from {args.continue_run}")
            print(f"[System] Problem: {problem[:100]}...")
        except FileNotFoundError:
            print(f"\n❌ Error: Population '{args.continue_run}' not found.")
            print("Use --list-populations to see available populations.")
            sys.exit(1)
    elif args.continue_run == 'latest':
        # Continuing latest run
        from tools.population_manager import PopulationManager
        pm = PopulationManager()
        latest = pm.get_latest_population()
        if not latest:
            print("\n❌ Error: No populations found to continue.")
            print("Start a new evolution with: python main.py --mode evolution --problem \"...\"")
            sys.exit(1)
        run_id, pop = latest
        problem = pop.metadata.problem_statement
        args.continue_run = run_id
        print(f"\n[System] Continuing from latest: {run_id}")
        print(f"[System] Problem: {problem[:100]}...")
    elif args.problem:
        problem = args.problem
    else:
        # Use default for non-interactive mode
        problem = get_default_problem()
        print(f"\n[System] Using default problem:\n{problem}\n")

    # Route to appropriate mode
    if args.mode == "evolution":
        from evolutionary_engine import run_evolutionary_ideation

        # Determine continue_run_id
        continue_run_id = args.continue_run if args.continue_run and args.continue_run != 'latest' else None
        if args.continue_run == 'latest':
            continue_run_id = run_id

        run_evolutionary_ideation(
            problem=problem,
            problem_name=args.problem_name,
            enable_research=not args.no_research,
            enable_local_papers=not args.no_papers,
            continue_run_id=continue_run_id
        )
    else:
        # Standard mode
        run_ideation_system(
            problem=problem,
            enable_research=not args.no_research,
            enable_local_papers=not args.no_papers,
            interactive=not args.non_interactive
        )


if __name__ == "__main__":
    main()
