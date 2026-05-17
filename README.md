# Dataset Agent Project

A modular dataset-chat agent with:

- FastAPI backend
- Vanilla HTML/CSS/JS frontend
- Code generation and execution component
- Persistent execution state using CSV + JSON
- Follow-up query handling using previous results
- Plotly visualization component
- Optional OpenAI LLM integration

## Setup

```bash
cd dataset_agent_project
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

Optional LLM setup:

Create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

The project still works without an API key using rule-based fallback logic.

## Run backend

```bash
uvicorn backend.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

## Run frontend

Open:

```text
frontend/index.html
```

or use VS Code Live Server.

## Test questions

Try:

```text
What is the average median house value by ocean proximity?
Now show top 5
Sort the previous result by median_house_value and plot it as a bar chart
Compute the standard deviation of this result
Filter previous result where median_house_value is greater than 200000
```

## Persistent state

Results are stored in:

```text
results/results.csv
results/states/<session_id>_<turn_id>.json
```

Each JSON state contains the reusable `result_table`, so visualization and follow-up queries can operate without directly using the raw dataset.
