"""
LangGraph Workflow - Orchestrates the multi-agent ideation system
Defines the state graph and execution flow.
Dynamically activates agents based on available API keys.
"""

import asyncio
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

from config import Config
from tools.research import research_topic
from tools.file_loader import load_local_papers
from agents.ideation import ideation_agent
from agents.evaluator import evaluate_ideas
from agents.presenter import generate_presentation


# Define the state schema
class IdeationState(TypedDict):
    """State schema for the ideation workflow."""
    problem: str
    research_context: str
    local_papers: List[str]
    agent_ideas: Annotated[List[Dict], operator.add]  # Accumulator for parallel execution
    evaluations: List[Dict]  # List of evaluation dictionaries
    presentation: str


def build_workflow(enable_research: bool = True, enable_local_papers: bool = True):
    """
    Build the LangGraph workflow.

    Args:
        enable_research: Whether to enable online research (requires API)
        enable_local_papers: Whether to load local papers

    Returns:
        Compiled LangGraph application
    """

    # Define workflow nodes
    def research_node(state: IdeationState) -> Dict:
        """Node 1: Gather research context and local papers."""
        print("\n" + "="*80)
        print("PHASE 1: RESEARCH & CONTEXT GATHERING")
        print("="*80)

        research_context = ""
        local_papers = []

        # Online research
        if enable_research:
            try:
                print("\n[Research] Searching for relevant papers and information...")
                context = asyncio.run(research_topic(state['problem']))
                research_context = context
                print(f"[Research] Retrieved {len(research_context)} characters of context")
            except Exception as e:
                print(f"[Research] Error: {str(e)}")
                research_context = "Research unavailable due to error."
        else:
            research_context = "Online research disabled."

        # Local papers
        if enable_local_papers:
            try:
                print("\n[Research] Loading local papers...")
                papers = load_local_papers()
                local_papers = papers
                print(f"[Research] Loaded {len(local_papers)} paper chunks")
            except Exception as e:
                print(f"[Research] Error loading papers: {str(e)}")
                local_papers = []
        else:
            print("\n[Research] Local papers disabled")

        return {
            "research_context": research_context,
            "local_papers": local_papers
        }

    # Get active agents from configuration
    active_agents = Config.get_active_agents()

    if not active_agents:
        raise ValueError("No API keys configured! Please configure at least one agent API key.")

    print(f"\n[Graph] Active agents: {active_agents}")

    # Agent name mapping
    agent_names = {
        1: "SOTA Agent 1",
        2: "SOTA Agent 2 (with search)",
        3: "SOTA Agent 3",
        4: "SOTA Agent 4 (with search)"
    }

    # Dynamically create agent node functions for parallel execution
    def create_agent_node(agent_id: int):
        """Factory function to create agent nodes dynamically."""
        agent_name = agent_names.get(agent_id, f"Agent {agent_id}")

        def agent_node(state: IdeationState) -> Dict:
            print("\n" + "="*80)
            print(f"AGENT {agent_id} IDEATION - RUNNING IN PARALLEL ({agent_name})")
            print("="*80)

            idea = ideation_agent(state, agent_id=agent_id)

            # Return as list - operator.add will combine with other agents
            return {"agent_ideas": [idea]}

        return agent_node

    # Create agent nodes dynamically
    agent_nodes = {}
    for agent_id in active_agents:
        agent_nodes[f"agent{agent_id}"] = create_agent_node(agent_id)

    def evaluation_node(state: IdeationState) -> Dict:
        """Evaluate all agent ideas."""
        print("\n" + "="*80)
        print(f"PHASE 3: MULTI-ASPECT EVALUATION")
        print(f"Evaluating {len(state.get('agent_ideas', []))} agent proposals")
        print("="*80)

        evals = evaluate_ideas(state)

        return {"evaluations": evals}

    def presentation_node(state: IdeationState) -> Dict:
        """Generate final presentation."""
        print("\n" + "="*80)
        print(f"PHASE 4: PRESENTATION GENERATION")
        print("="*80)

        pres = generate_presentation(state)

        return {"presentation": pres}

    # Build the graph
    workflow = StateGraph(IdeationState)

    # Add research node
    workflow.add_node("research", research_node)

    # Add agent nodes dynamically
    for node_name, node_func in agent_nodes.items():
        workflow.add_node(node_name, node_func)

    # Add evaluation and presentation nodes
    workflow.add_node("evaluate", evaluation_node)
    workflow.add_node("present", presentation_node)

    # Define edges for PARALLEL execution
    workflow.set_entry_point("research")

    print(f"[Graph] Configuring PARALLEL execution for {len(active_agents)} agents")

    # Connect research to ALL agents in parallel (fan-out)
    for agent_id in active_agents:
        agent_node_name = f"agent{agent_id}"
        workflow.add_edge("research", agent_node_name)
        print(f"[Graph] Research -> Agent {agent_id} (parallel)")

    # Connect ALL agents to evaluation (fan-in)
    # LangGraph will wait for all parallel branches to complete
    for agent_id in active_agents:
        agent_node_name = f"agent{agent_id}"
        workflow.add_edge(agent_node_name, "evaluate")

    # Connect evaluation to presentation
    workflow.add_edge("evaluate", "present")
    workflow.add_edge("present", END)

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    print(f"\n[Graph] Workflow compiled successfully")
    print(f"[Graph] Active agents: {len(active_agents)} - RUNNING IN PARALLEL")
    print(f"[Graph] Execution flow: Research -> [{', '.join([f'Agent {i}' for i in active_agents])}] -> Evaluation -> Presentation")

    return app


def visualize_workflow():
    """
    Generate a visual representation of the workflow.
    Requires graphviz and pygraphviz.
    """
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        app = build_workflow(enable_research=False, enable_local_papers=False)

        # Generate mermaid diagram
        print("\nWorkflow Diagram (Mermaid):")
        print(app.get_graph().draw_mermaid())

    except ImportError:
        print("Install pygraphviz for workflow visualization: pip install pygraphviz")
    except Exception as e:
        print(f"Visualization error: {str(e)}")


if __name__ == "__main__":
    # Test workflow building
    print("Building workflow...")
    app = build_workflow(enable_research=False, enable_local_papers=False)
    print("Workflow built successfully!")

    # Try to visualize
    visualize_workflow()
