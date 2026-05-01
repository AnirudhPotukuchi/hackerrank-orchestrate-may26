# Support Triage Agent

This directory contains the AI agent for triaging support tickets.

## Architecture

The agent is built using **Raw Python** and **OpenAI's GPT-4o**.
- **`retriever.py`**: Ingests all markdown files from the `data/` directory, chunks them, and stores them in a local `chromadb` instance. Uses OpenAI's embeddings for semantic search.
- **`agent.py`**: Constructs the prompt using the retrieved context and forces a strict structured JSON output using Pydantic, ensuring exactly the 5 required columns are outputted without hallucinated formatting.
- **`main.py`**: Reads `support_tickets.csv`, processes each row sequentially, and writes predictions to `output.csv`.

## Setup

1. Copy `.env.example` to `.env` in the root repository.
2. Add your OpenAI API key: `OPENAI_API_KEY=sk-yourkey`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Agent

To process the `support_tickets.csv` file, run:
```bash
python main.py
```
This will automatically ingest the `data/` folder into a local ChromaDB on the first run, and then process the CSV.
