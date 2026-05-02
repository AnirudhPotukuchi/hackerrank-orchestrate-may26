# HackerRank Orchestrate: AI Support Triage Agent

This repository contains my solution for the **HackerRank Orchestrate** 24-hour hackathon. 
I built a fully local, terminal-based AI agent that triages real support tickets across three product ecosystems: **HackerRank**, **Claude**, and **Visa**.

## Approach & Architecture Overview

My solution prioritizes **100% local execution, zero-cost scaling, and privacy-first data handling**. By moving away from rate-limited cloud APIs (like Google Gemini or OpenAI) to a local model via Ollama, I completely eliminated `429 Resource Exhausted` errors and ensured uninterrupted, high-speed batch processing for the support tickets.

### Core Components
1. **Inference Engine (`Ollama` + `llama3.2:1b`)**: I use the `llama3.2:1b` model running locally via Ollama. It provides fast, cost-free generation and enforces strict Pydantic JSON schema constraints natively.
2. **Retrieval-Augmented Generation (RAG) (`ChromaDB`)**: I process the provided markdown corpus into a persistent local ChromaDB instance. The system uses local embeddings (`all-MiniLM-L6-v2`) to accurately retrieve the most relevant documentation based on the user's issue.
3. **Orchestrator (`Pandas`)**: A central `main.py` script orchestrates the pipeline, reading the input CSV, routing tickets through the vector database and the LLM, and constructing the finalized `output.csv` matching the strict schema constraints.

### Escalation Logic
The agent uses a direct and balanced system prompt to prevent over-escalation and hallucinations. It accurately classifies a ticket as `escalated` if:
* The user asks for a refund, account deletion, or deals with PII/Security.
* The user is hostile or angry.
* The exact answer is **not** present in the retrieved ChromaDB context.

Otherwise, it routes the ticket as `replied` and generates a safe, grounded response based exclusively on the retrieved documentation.

---

## Setup Instructions

### Prerequisites
1. **Python 3.10+** (Developed and tested on Python 3.13)
2. **Ollama**: You must have [Ollama](https://ollama.com/) installed on your machine.

### 1. Start the Local Model
Before running the agent, you must pull and start the `llama3.2:1b` model:
```bash
ollama run llama3.2:1b
```
Keep this running or ensure the Ollama background service is active.

### 2. Install Dependencies
Navigate to the project root and install the required Python packages:
```bash
pip install -r code/requirements.txt
```

---

## Running the Agent

To execute the triage pipeline:
```bash
python code/main.py
```

**What happens during execution?**
1. **Data Ingestion**: If the `code/chroma_db` directory doesn't exist, `retriever.py` will read the entire `data/` folder, chunk the markdown, and ingest it into the local Chroma vector database.
2. **Ticket Processing**: It will iterate over all tickets in `support_tickets/support_tickets.csv`.
3. **Export**: It saves the structured results into `support_tickets/output.csv` strictly matching the required column order.

*(Note: Ensure `output.csv` is CLOSED in your text editor/Excel before running, or Windows will throw a `PermissionError`!)*

---

## Repository Structure

```text
.
├── AGENTS.md                       # Rules for AI coding tools + transcript logging
├── problem_statement.md            # Full task description and I/O schema
├── README.md                       # Setup and Approach Documentation (You are here)
├── log.txt                         # AI Assistant Chat Transcript for submission
├── code/                           # Core Agent Logic
│   ├── main.py                     # Entry point and data orchestrator
│   ├── agent.py                    # LLM Prompting and Structured Output logic
│   ├── retriever.py                # ChromaDB RAG and local embedding pipeline
│   ├── config.py                   # Global configuration and paths
│   ├── requirements.txt            # Python dependencies
│   └── chroma_db/                  # (Generated) Persistent Vector DB storage
├── data/                           # Local-only support corpus
└── support_tickets/
    ├── sample_support_tickets.csv  # Inputs + expected outputs (for development)
    ├── support_tickets.csv         # Target inputs 
    └── output.csv                  # The finalized, agent-generated outputs
```

---

## Hackathon Submission Details

If you are evaluating this submission:
1. **Code**: Found entirely in the `code/` directory.
2. **Predictions CSV**: The `support_tickets/output.csv` has been successfully generated and formatted.
3. **Chat Transcript**: The chat logs with the AI assistant have been accurately maintained in the local `log.txt` file (as well as the global path specified in `AGENTS.md`).

For detailed rubric information, see [`evalutation_criteria.md`](./evalutation_criteria.md).