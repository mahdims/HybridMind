"""
Test script to verify system installation and basic functionality.
Run this to ensure all components are properly installed.
"""

import os
import sys
from dotenv import load_dotenv


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    tests = {
        "langgraph": lambda: __import__("langgraph"),
        "langchain": lambda: __import__("langchain"),
        "langchain_openai": lambda: __import__("langchain_openai"),
        "langchain_google_genai": lambda: __import__("langchain_google_genai"),
        "langchain_anthropic": lambda: __import__("langchain_anthropic"),
        "langchain_community": lambda: __import__("langchain_community"),
        "openai": lambda: __import__("openai"),
        "anthropic": lambda: __import__("anthropic"),
        "pydantic": lambda: __import__("pydantic"),
        "dotenv": lambda: __import__("dotenv"),
    }

    results = {}
    for name, import_func in tests.items():
        try:
            import_func()
            results[name] = "✓"
            print(f"  ✓ {name}")
        except ImportError as e:
            results[name] = "✗"
            print(f"  ✗ {name} - {str(e)}")

    return all(v == "✓" for v in results.values())


def test_environment():
    """Test environment configuration for all three API keys."""
    print("\nTesting environment configuration...")

    load_dotenv()

    required_keys = {
        "OPENAI_API_KEY": "Agent 1 (OpenAI)",
        "GOOGLE_API_KEY": "Agent 2 (Gemini)",
        "ANTHROPIC_API_KEY": "Agent 3 (Claude)"
    }

    results = {}
    for key, description in required_keys.items():
        api_key = os.getenv(key)
        if api_key and api_key not in ["your_openai_api_key_here", "your_google_api_key_here", "your_anthropic_api_key_here"]:
            print(f"  ✓ {key} is set ({description})")
            results[key] = True
        else:
            print(f"  ✗ {key} is not configured ({description})")
            results[key] = False

    if not all(results.values()):
        print("\n  Please set all API keys in .env file")
        print("  All three API keys are required for the multi-model setup")

    return all(results.values())


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        "agents",
        "tools",
        "prompts",
        "data/papers",
        "output"
    ]

    results = {}
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        results[dir_path] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")

    return all(results.values())


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")

    modules = [
        "config",
        "graph",
        "tools.research",
        "tools.file_loader",
        "agents.ideation",
        "agents.evaluator",
        "agents.presenter",
    ]

    results = {}
    for module_name in modules:
        try:
            __import__(module_name)
            results[module_name] = "✓"
            print(f"  ✓ {module_name}")
        except Exception as e:
            results[module_name] = "✗"
            print(f"  ✗ {module_name} - {str(e)}")

    return all(v == "✓" for v in results.values())


def test_workflow_build():
    """Test that the workflow can be built."""
    print("\nTesting workflow build...")

    try:
        from graph import build_workflow
        app = build_workflow(enable_research=False, enable_local_papers=False)
        print("  ✓ Workflow built successfully")
        return True
    except Exception as e:
        print(f"  ✗ Workflow build failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SYSTEM INSTALLATION TEST")
    print("="*80 + "\n")

    tests = [
        ("Package Imports", test_imports),
        ("Environment Config", test_environment),
        ("Directory Structure", test_directories),
        ("Custom Modules", test_modules),
        ("Workflow Build", test_workflow_build),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - System is ready to use!")
        print("\nRun 'python main.py' to start the ideation system.")
    else:
        print("✗ SOME TESTS FAILED - Please fix the issues above")
        print("\nRefer to README.md for setup instructions.")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
