# Population Directory

This directory stores evolved algorithm idea populations for the Evolutionary Search system.

## Structure

Each evolutionary run creates a separate population file named by timestamp:
```
population_run_YYYYMMDD_HHMMSS.json
```

Example: `population_run_20250103_143052.json`

## Population File Contents

Each file contains:
- **Metadata**: Problem statement, run info, parameters used
- **Population**: Array of algorithm ideas with scores and feedback
- **Evolution history**: Tracking of generations and improvements

## File Management

- **Multiple runs**: Each run creates a new population file
- **Never deleted**: Old populations are preserved for comparison
- **Continue mode**: Can resume evolution of any previous population
- **Problem-specific**: Each problem gets its own evolution

## Example Workflow

**Start new evolution:**
```bash
python main.py --problem "Your problem..." --mode evolution
# Creates: population_run_20250103_143052.json
```

**Continue most recent:**
```bash
python main.py --continue
# Continues latest population file
```

**Continue specific run:**
```bash
python main.py --continue run_20250103_143052
# Continues that specific population
```

**List available populations:**
```bash
python main.py --list-populations
```

## Population JSON Schema

```json
{
  "metadata": {
    "run_id": "run_20250103_143052",
    "problem_statement": "Original problem...",
    "problem_hash": "abc123",
    "created_at": "2025-01-03T14:30:52",
    "last_updated": "2025-01-03T16:45:30",
    "current_generation": 15,
    "parameters": {
      "population_size": 20,
      "mutation_ratio": 0.70,
      ...
    }
  },
  "population": [
    {
      "id": "idea_001",
      "content": "Full algorithm description...",
      "metadata": {
        "generation_created": 1,
        "generation_last_modified": 5,
        "source_agent_id": 2,
        "last_modified_by_agent_id": 3,
        "parent_ids": [],
        "operation_history": [...]
      },
      "scores": {...},
      "feedback": {...},
      "confidence": {...},
      "fitness": 4.3,
      "rank": 1,
      "is_elite": true
    }
  ]
}
```

## Benefits

1. **Persistence**: Evolution continues across sessions
2. **Comparison**: Compare different evolutionary runs
3. **Reproducibility**: Complete record of evolution process
4. **Experimentation**: Try different parameters on same problem
5. **Knowledge base**: Best ideas preserved and reusable

## Configuration

Edit `parameters.txt` to control:
- Population size
- Mutation/crossover/fresh ratios
- Number of generations
- Selection strategy
- And more...
