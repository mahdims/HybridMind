"""
Utility functions for the Algorithmic Multi-Agent Ideation System.
"""

import os
from typing import Dict, Any


def load_prompt_template(template_name: str, prompts_dir: str = "./prompts") -> str:
    """
    Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (without .txt extension)
        prompts_dir: Directory containing prompt templates

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_path = os.path.join(prompts_dir, f"{template_name}.txt")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with provided variables.

    Args:
        template: Template string with {variable} placeholders
        **kwargs: Variables to substitute in template

    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)


def load_and_format_prompt(template_name: str, prompts_dir: str = "./prompts", **kwargs: Any) -> str:
    """
    Load and format a prompt template in one step.

    Args:
        template_name: Name of the template file (without .txt extension)
        prompts_dir: Directory containing prompt templates
        **kwargs: Variables to substitute in template

    Returns:
        Formatted prompt string
    """
    template = load_prompt_template(template_name, prompts_dir)
    return format_prompt(template, **kwargs)


if __name__ == "__main__":
    # Test prompt loading
    try:
        ideation_template = load_prompt_template("ideation")
        print("[OK] Successfully loaded ideation template")
        print(f"Template length: {len(ideation_template)} characters")

        # Test formatting
        formatted = format_prompt(
            ideation_template,
            agent_id=1,
            problem="Test problem",
            research_context="Test context",
            local_papers="Test papers"
        )
        print("[OK] Successfully formatted template")
        print(f"Formatted length: {len(formatted)} characters")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
