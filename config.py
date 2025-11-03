"""
Configuration settings for the Algorithmic Multi-Agent Ideation System.
"""

import os
from typing import Dict, Any, List


class Config:
    """System configuration settings."""

    # Multi-Model Configuration
    # SOTA Agent 1
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # SOTA Agent 2
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

    # SOTA Agent 3
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    CLAUDE_TEMPERATURE: float = float(os.getenv("CLAUDE_TEMPERATURE", "0.7"))

    # SOTA Agent 4 (with search capability)
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-plus")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    QWEN_TEMPERATURE: float = float(os.getenv("QWEN_TEMPERATURE", "0.7"))
    QWEN_ENABLE_SEARCH: bool = os.getenv("QWEN_ENABLE_SEARCH", "true").lower() == "true"

    # Agent Settings
    MAX_AGENTS: int = 4  # Maximum possible agents
    NUM_REFLECTIONS: int = 3
    AGENT_SPECIALIZATIONS = {
        1: "OpenAI GPT perspective",
        2: "Google Gemini perspective",
        3: "Anthropic Claude perspective",
        4: "Alibaba Qwen perspective with web search"
    }

    # Agent Model Mapping (agents will be activated based on API key availability)
    AGENT_MODELS = {
        1: {
            "provider": "openai",
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "api_key_env": "OPENAI_API_KEY",
            "supports_search": False
        },
        2: {
            "provider": "google",
            "model": GEMINI_MODEL,
            "temperature": GEMINI_TEMPERATURE,
            "api_key_env": "GOOGLE_API_KEY",
            "supports_search": True  # Gemini supports search via grounding
        },
        3: {
            "provider": "anthropic",
            "model": CLAUDE_MODEL,
            "temperature": CLAUDE_TEMPERATURE,
            "api_key_env": "ANTHROPIC_API_KEY",
            "supports_search": False
        },
        4: {
            "provider": "qwen",
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "api_key_env": "QWEN_API_KEY",
            "base_url": QWEN_BASE_URL,
            "enable_search": QWEN_ENABLE_SEARCH,
            "supports_search": True
        }
    }

    # Evaluation Settings (1-5 Likert Scale)
    # Weights sum to 1.0 - Correctness and SOTA competitiveness are MOST critical
    EVALUATION_ASPECTS = [
        {
            "name": "correctness",
            "description": "Algorithmic correctness and logical soundness (CRITICAL - 30%)",
            "criteria": "Evaluate logical correctness, edge case handling, termination guarantees, and algorithmic soundness. This is a top priority.",
            "weight": 0.30
        },
        {
            "name": "sota_competitiveness",
            "description": "Competitiveness against state-of-the-art (CRITICAL - 30%)",
            "criteria": "Evaluate if the proposed method has enough 'bone' to outperform SOTA CVRP heuristics (HGS, SISR, FILO2, LKH3, AILS-II). Consider: innovative mechanisms that could provide competitive advantage, addressing limitations of current SOTA, practical feasibility of achieving better solution quality or speed, theoretical soundness and empirical promise. Does this approach have real potential to advance the state-of-the-art? This is a top priority.",
            "weight": 0.30
        },
        {
            "name": "time_complexity",
            "description": "Low time complexity and efficiency (IMPORTANT - 20%)",
            "criteria": "Assess Big-O time complexity analysis, optimality, and comparison to known solutions. Critical for scalability to large instances.",
            "weight": 0.20
        },
        {
            "name": "pseudocode_quality",
            "description": "Clear pseudocode and explanation (10%)",
            "criteria": "Assess clarity, structure, completeness of pseudocode, variable naming, and step-by-step logic",
            "weight": 0.10
        },
        {
            "name": "novelty",
            "description": "Originality and innovation (10%)",
            "criteria": "Consider uniqueness of approach, creative problem-solving, and departure from standard solutions",
            "weight": 0.10
        }
    ]

    # Evaluation scale definitions (1-5 Likert)
    EVALUATION_SCALE = {
        1: "Poor/Incorrect - Critical flaws, fundamentally wrong approach",
        2: "Below Average - Significant issues that undermine the solution",
        3: "Adequate - Meets basic requirements but needs improvement",
        4: "Good - Solid solution with only minor issues",
        5: "Excellent/Optimal - Outstanding solution, publication-worthy"
    }

    # Aggregation method for scores
    AGGREGATION_METHOD: str = os.getenv("AGGREGATION_METHOD", "weighted_average")  # Options: weighted_average, simple_average, remove_outliers

    # Directory Settings
    PAPERS_DIR: str = os.getenv("PAPERS_DIR", "./data/papers")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
    PROMPTS_DIR: str = "./prompts"

    # Research Settings
    MAX_RESEARCH_RESULTS: int = 5
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_active_agents(cls) -> List[int]:
        """
        Get list of active agent IDs based on available API keys.

        Returns:
            List of agent IDs that have API keys configured
        """
        active_agents = []

        for agent_id, config in cls.AGENT_MODELS.items():
            api_key_env = config['api_key_env']
            api_key = os.getenv(api_key_env, "")

            if api_key and api_key not in [
                "your_openai_api_key_here",
                "your_google_api_key_here",
                "your_anthropic_api_key_here",
                "your_qwen_api_key_here"
            ]:
                active_agents.append(agent_id)

        return sorted(active_agents)

    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings.
        At least one agent API key must be configured.
        """
        active_agents = cls.get_active_agents()

        if not active_agents:
            print("ERROR: No API keys configured!")
            print("\nPlease configure at least ONE agent API key in .env file")
            print("See .env.example for configuration options")
            return False

        print(f"\n✓ Active agents: {active_agents}")
        for agent_id in active_agents:
            print(f"  SOTA Agent {agent_id}: Ready")

        return True

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        active_agents = cls.get_active_agents()
        return {
            "openai_model": cls.OPENAI_MODEL,
            "openai_temperature": cls.OPENAI_TEMPERATURE,
            "num_agents": len(active_agents),
            "active_agents": active_agents,
            "max_agents": cls.MAX_AGENTS,
            "num_reflections": cls.NUM_REFLECTIONS,
            "papers_dir": cls.PAPERS_DIR,
            "output_dir": cls.OUTPUT_DIR,
        }

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        active_agents = cls.get_active_agents()
        print("\n" + "="*80)
        print("SYSTEM CONFIGURATION - MULTI-AGENT SETUP")
        print("="*80)
        print("\nAgent Configuration:")
        print(f"  SOTA Agent 1: {cls.OPENAI_MODEL} (temp: {cls.OPENAI_TEMPERATURE})")
        print(f"  SOTA Agent 2: {cls.GEMINI_MODEL} (temp: {cls.GEMINI_TEMPERATURE})")
        print(f"  SOTA Agent 3: {cls.CLAUDE_MODEL} (temp: {cls.CLAUDE_TEMPERATURE})")
        print(f"  SOTA Agent 4: {cls.QWEN_MODEL} (temp: {cls.QWEN_TEMPERATURE}, search: {cls.QWEN_ENABLE_SEARCH})")
        print(f"\nSystem Settings:")
        print(f"  Max Agents: {cls.MAX_AGENTS}")
        print(f"  Active Agents: {len(active_agents)} - {active_agents}")
        print(f"  Reflections per Agent: {cls.NUM_REFLECTIONS}")
        print(f"  Evaluation Aspects: {len(cls.EVALUATION_ASPECTS)}")
        print(f"  Papers Directory: {cls.PAPERS_DIR}")
        print(f"  Output Directory: {cls.OUTPUT_DIR}")
        print("="*80 + "\n")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    Config.print_config()

    if Config.validate():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration is invalid")
