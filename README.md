## Step 1: Integration

- Convert problem formats to correct format like A2A
- Implement dataset loaders & interfaces
- Add quality checks for correctness & reproducibility

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from green_agent import GreenAgent

agent = GreenAgent()
agent.run_evaluation()
```

## Project Structure

```
green-white-agent/
├── green_agent/
│   ├── __init__.py
│   ├── agent.py              # Main agent implementation
│   ├── dataset_loaders/      # Dataset loading utilities
│   ├── interfaces/           # Agent interfaces
│   ├── quality_checks/       # Quality validation
│   └── converters/           # Format converters (A2A)
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```
