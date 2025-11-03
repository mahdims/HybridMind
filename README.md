# Algorithmic Multi-Agent Ideation System v1.0

A LangGraph-based multi-agent system where up to 4 independent agents, **each powered by a different LLM provider**, collaboratively ideate algorithm solutions through self-reflection, get evaluated by LLM judges, and produce comprehensive presentations.

## 🎯 Overview

This system implements a novel approach to algorithm design using multiple AI agents that:
- **Use diverse AI models** (4 different SOTA language models) for varied perspectives
- **Execute in parallel** for maximum speed - 4 agents complete in the time of 1!
- Generate diverse algorithmic solutions to complex problems
- Self-reflect and refine ideas through iterative loops (3 iterations per agent)
- Get evaluated across multiple dimensions using 1-5 Likert scale with weighted scoring
- Produce structured presentations with actionable insights

### ⚡ Performance
- **Parallel Execution**: All agents run simultaneously using separate API calls
- **3-5 minutes total** instead of 12-20 minutes sequential
- **Scalable**: Add or remove agents by simply configuring API keys

## 🤖 Multi-Model Architecture

**Each ideation agent uses a different SOTA language model for maximum diversity:**

- **Agent 1**: SOTA Agent 1 - Generates unique algorithm perspective
- **Agent 2**: SOTA Agent 2 - Generates unique algorithm perspective (supports search)
- **Agent 3**: SOTA Agent 3 - Generates unique algorithm perspective
- **Agent 4**: SOTA Agent 4 - Generates unique algorithm perspective (with web search capability)

**Evaluation & Presentation:**
- **Evaluator**: Specialized evaluation model - Fast, cost-effective multi-aspect evaluation
- **Presenter**: Specialized presentation model - Efficient presentation generation

**Flexible Agent Activation:**
- Agents are activated based on available API keys
- System requires at least ONE API key to function
- Run with any combination of 1-4 agents depending on your API keys

This multi-model approach ensures diverse perspectives from ideation agents while keeping evaluation and presentation cost-effective.

## 🏗️ Architecture

```
Input (Problem)
    ↓
[Phase 1: Research & Context]
    ↓
    ├──────────┬──────────┬──────────┐
    ↓          ↓          ↓          ↓
[Agent 1]  [Agent 2]  [Agent 3]  [Agent 4]
SOTA-1     SOTA-2+S   SOTA-3     SOTA-4+S
  3 loops    3 loops    3 loops    3 loops
    │          │          │          │
    └──────────┴──────────┴──────────┘
              ↓
[Phase 2: PARALLEL AGENT IDEATION]
⚡ All agents run simultaneously!
              ↓
[Phase 3: Multi-Aspect Evaluation]
(1-5 Likert Scale, Weighted Scoring)
              ↓
[Phase 4: Presentation Generation]
              ↓
Output (Markdown Report)

⏱️ Parallel execution: 3-5 min total
📊 Sequential would take: 12-20 min
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- **At least ONE of the following API keys**:
  - **API key for SOTA Agent 1** - Optional
  - **API key for SOTA Agent 2** (with search) - Optional
  - **API key for SOTA Agent 3** - Optional
  - **API key for SOTA Agent 4** (with search) - Optional
- (Optional) Tavily API key for enhanced research

**Note**: The system requires at least ONE agent API key to function. Agents without API keys will be automatically disabled. More agents = more diverse perspectives!

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd MAS_AlgIdeation
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add at least ONE API key
# Add as many as you want - more agents = more perspectives!
#   - OPENAI_API_KEY (for SOTA Agent 1)
#   - GOOGLE_API_KEY (for SOTA Agent 2 with search)
#   - ANTHROPIC_API_KEY (for SOTA Agent 3)
#   - QWEN_API_KEY (for SOTA Agent 4 with search)
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Google (Gemini): https://makersuite.google.com/app/apikey
- Anthropic: https://console.anthropic.com/
- Qwen: https://help.aliyun.com/zh/dashscope/

5. **Run the system**
```bash
python main.py
```

## 📁 Project Structure

```
MAS_AlgIdeation/
├── main.py                 # Entry point
├── graph.py                # LangGraph workflow orchestration
├── config.py               # System configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── agents/
│   ├── ideation.py         # Ideation agent with reflection
│   ├── evaluator.py        # Multi-aspect LLM judge
│   └── presenter.py        # Presentation generator
├── tools/
│   ├── research.py         # GPT Researcher wrapper
│   └── file_loader.py      # Local PDF paper loader
├── prompts/
│   ├── ideation.txt        # Ideation prompt template
│   ├── reflection.txt      # Reflection prompt template
│   ├── evaluation.txt      # Evaluation prompt template
│   └── presentation.txt    # Presentation prompt template
├── data/
│   └── papers/             # Place PDF papers here
└── output/                 # Generated presentations
```

## 🎮 Usage

### Basic Usage

```bash
python main.py
```

This will run the system with the default problem and generate a presentation in the `output/` directory.

### Advanced Example: CVRP Optimization

**NEW!** We've included a comprehensive example for generating innovative algorithms for large-scale Capacitated Vehicle Routing Problems (CVRP):

```bash
# Run the standalone CVRP ideation script
python run_cvrp_ideation.py
```

This will generate innovative ideas for:
- Hybrid metaheuristic approaches combining HGS, ALNS, AILS, SISR, LKH3
- Novel exploration-exploitation balance strategies
- Computational efficiency improvements for 1000+ customers
- Machine learning integration and adaptive mechanisms

**Or run via example_usage.py:**
```bash
python -c "from example_usage import example_7_cvrp_optimization; example_7_cvrp_optimization()"
```

See [run_cvrp_ideation.py](run_cvrp_ideation.py) for the detailed problem specification!

### Advanced Usage

```bash
# Provide custom problem
python main.py --problem "Design an algorithm for finding the longest palindromic substring"

# Disable online research (faster, uses only local papers)
python main.py --no-research

# Disable local paper loading
python main.py --no-papers

# Run in non-interactive mode
python main.py --non-interactive
```

### Adding Local Papers

1. Place PDF papers in the `data/papers/` directory
2. The system will automatically load and chunk them
3. Papers will be used as context for agent ideation

## 🧪 System Components

### 1. Research Tool
- Uses GPT Researcher to gather relevant papers and information
- Searches academic databases and online resources
- Provides context for agent ideation

### 2. Ideation Agents (1-4 Agents with Different Models)
Each agent uses a different SOTA language model to naturally provide diverse perspectives:
- **Agent 1**: SOTA Agent 1 (Optional)
- **Agent 2**: SOTA Agent 2 with search support (Optional)
- **Agent 3**: SOTA Agent 3 (Optional)
- **Agent 4**: SOTA Agent 4 with web search (Optional)

**Flexible Agent Activation:**
- System adapts to available API keys
- Minimum 1 agent required, maximum 4 agents
- More agents = more diverse perspectives

Each agent:
1. Generates an initial algorithm proposal using its assigned LLM
2. Performs 3 self-reflection loops with the same model
3. Refines the idea based on critical analysis
4. Can access web search (Agents 2 & 4) if enabled

**Why different models?**
- Each LLM has different training data, reasoning patterns, and strengths
- Natural diversity without forcing agents into predefined approaches
- Reduces groupthink and model-specific biases
- Produces more creative and varied solutions

### 3. Evaluation Judge
Evaluates all agent proposals using **robust 1-5 Likert scale** across 5 dimensions:

**Evaluation Dimensions (Weighted):**

**TOP PRIORITY (60% - Agents are instructed to excel here):**

- **Correctness** (30%): Algorithmic correctness and logical soundness - **CRITICAL**
  - Logical correctness, edge case handling, termination guarantees
- **SOTA Competitiveness** (30%): Can it outperform state-of-the-art? - **CRITICAL**
  - Evaluates if the approach has enough "bone" to beat HGS, SISR, FILO2, LKH3, AILS-II
  - Considers innovative mechanisms, addressing SOTA limitations, practical feasibility
  - Does this have real potential to advance the state-of-the-art?

**IMPORTANT (20%):**

- **Low Time Complexity** (20%): Big-O analysis and computational efficiency
  - Critical for scalability to large instances (1000+ customers)

**SUPPORTING (20%):**

- **Clear Pseudocode** (10%): Clarity and completeness of explanation
- **Novelty** (10%): Originality and innovation

**Why 1-5 Likert Scale?**
- More reliable and consistent than 1-10
- Clear distinction between levels
- Industry and academic standard
- Psychologically proven optimal range

**Scoring:**
- 5 = Excellent/Optimal (publication-worthy)
- 4 = Good (solid with minor issues)
- 3 = Adequate (needs improvement)
- 2 = Below Average (significant issues)
- 1 = Poor/Incorrect (critical flaws)

Each score includes confidence level (High/Medium/Low) and detailed justification.

See `EVALUATION_SYSTEM.md` for complete details.

### 4. Presentation Generator
Creates comprehensive markdown reports with:
- Problem statement
- Agent proposals and evolution
- Comparative evaluation tables
- Synthesis and recommendations
- Implementation roadmap
- Future work suggestions

## 📊 Output

The system generates a markdown presentation saved to `output/presentation_[timestamp].md` containing:

- Problem analysis
- All agent proposals with reflection history (1-4 agents depending on configuration)
- Comparative evaluation scores with weighted rankings
- Recommended approach
- Implementation steps
- Open questions

Example output structure:
```markdown
# Algorithm Ideation Results

## 1. Problem Statement
[Your problem description]

## 2. Agent Proposals
### Agent 1
[Proposal with reflections]

### Agent 2
[Proposal with reflections]

### Agent 3
[Proposal with reflections]

### Agent 4 (if configured)
[Proposal with reflections]

## 3. Comparative Evaluation (1-5 Likert Scale with Weighted Scoring)
| Agent | Weighted Score | Percentage | Correctness | Time | Space | Pseudocode | Novelty |
|-------|----------------|------------|-------------|------|-------|------------|---------|
| Agent 1 | 4.2/5 | 84% | 5/5 | 5/5 | 3/5 | 4/5 | 4/5 |
| Agent 2 | 3.8/5 | 76% | 4/5 | 4/5 | 4/5 | 4/5 | 3/5 |
| Agent 3 | 3.5/5 | 70% | 4/5 | 3/5 | 4/5 | 3/5 | 4/5 |
| Agent 4 | 3.9/5 | 78% | 4/5 | 4/5 | 3/5 | 4/5 | 4/5 |

## 4. Synthesis & Recommendations
[Best approach and hybrid solutions]

## 5. Implementation Roadmap
[Concrete next steps]

## 6. Open Questions
[Areas for future investigation]
```

## ⚙️ Configuration

Edit `config.py` or `.env` to customize:

```python
# Maximum possible agents (system auto-detects active agents based on API keys)
MAX_AGENTS = 4

# Reflections per agent
NUM_REFLECTIONS = 3

# Model configurations (via .env)
OPENAI_MODEL = "gpt-4"
GEMINI_MODEL = "gemini-1.5-pro"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
QWEN_MODEL = "qwen-plus"

# Enable search for Qwen
QWEN_ENABLE_SEARCH = true
```

**Dynamic Agent Activation:**
- System automatically detects which API keys are configured
- Only agents with valid API keys will be activated
- Minimum 1 agent, maximum 4 agents

## 🧩 Extending the System

### Adding New Evaluation Aspects

Edit `config.py`:
```python
EVALUATION_ASPECTS.append({
    "name": "scalability",
    "description": "How well does the algorithm scale?",
    "criteria": "Consider performance with large inputs"
})
```

### Customizing Prompts

Edit prompt templates in `prompts/` directory:
- `ideation.txt` - Agent idea generation
- `reflection.txt` - Self-reflection process
- `evaluation.txt` - Evaluation criteria
- `presentation.txt` - Final report format

See `PROMPT_CUSTOMIZATION.md` for detailed guide.

### Agent Configuration

The system supports up to 4 agents with automatic activation:
- Agents are enabled/disabled based on API key availability
- No code changes needed to add/remove agents
- Just add or remove API keys in `.env` file
- System dynamically adjusts workflow based on active agents

## 🐛 Troubleshooting

### Common Issues

**Issue**: `No API keys configured!`
- **Solution**: Create `.env` file with at least ONE API key:
  ```
  OPENAI_API_KEY=sk-...        # Optional - for SOTA Agent 1
  GOOGLE_API_KEY=AI...         # Optional - for SOTA Agent 2
  ANTHROPIC_API_KEY=sk-ant-... # Optional - for SOTA Agent 3
  QWEN_API_KEY=sk-...          # Optional - for SOTA Agent 4
  ```
  Note: You only need ONE key minimum, but more agents = more perspectives!

**Issue**: No PDF files loaded
- **Solution**: Add PDF papers to `data/papers/` directory

**Issue**: `EMBEDDING_PROVIDER not found` or `match os.environ["EMBEDDING_PROVIDER"]` error
- **Solution**: Add to your `.env` file:
  ```
  EMBEDDING_PROVIDER=openai
  ```
  This is required by the gpt-researcher library.

**Issue**: GPT Researcher errors
- **Solution**: Install additional dependencies or disable research with `--no-research`

**Issue**: Token limit exceeded
- **Solution**: Reduce `NUM_REFLECTIONS` or use shorter problem statements

## 📈 Performance

### Parallel Execution Times (4 agents)
- **Phase 1** - Research: 30-60 seconds
- **Phase 2** - All 4 agents in parallel: 2-3 minutes (longest agent sets the pace)
- **Phase 3** - Evaluation: 1-2 minutes
- **Phase 4** - Presentation: 30-60 seconds

**Total with parallel execution**: ~3-5 minutes ⚡

### Sequential Execution (legacy comparison)
If agents ran sequentially:
- Research: 30-60 seconds
- Agent 1: 2-3 minutes
- Agent 2: 1-2 minutes
- Agent 3: 2-3 minutes
- Agent 4: 2-3 minutes
- Evaluation: 1-2 minutes
- Presentation: 30-60 seconds

**Total sequential**: ~12-20 minutes 🐌

**Speedup: 75% faster with parallel execution!**

**Cost per run (approximate):**
- SOTA Agent 1: ~$0.40
- SOTA Agent 2: ~$0.03
- SOTA Agent 3: ~$0.10
- SOTA Agent 4: ~$0.05
- Evaluator: ~$0.02
- Presenter: ~$0.01
- **Total**: ~$0.55-0.60 per run with all 4 agents

**Note**: Performance varies by model. Different models have different speed, cost, and reasoning characteristics.

## 🔮 Future Enhancements

- [ ] Parallel agent execution (currently sequential)
- [ ] Ensemble evaluation judges
- [ ] Human-in-the-loop feedback
- [ ] LaTeX pseudocode generation
- [ ] Flowchart visualization
- [ ] Interactive web UI
- [ ] Benchmark comparison mode
- [ ] Code generation from algorithms

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Generated by**: Algorithmic Multi-Agent Ideation System v1.0
**Powered by**: LangGraph, Multi-Model Architecture
