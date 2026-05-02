import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# No API key needed for local Ollama

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TICKETS_DIR = BASE_DIR / "support_tickets"

INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"
SAMPLE_CSV = TICKETS_DIR / "sample_support_tickets.csv"

# Vector DB Path
CHROMA_DB_DIR = BASE_DIR / "code" / "chroma_db"

# LLM Configuration
# We use local llama3.2:1b via Ollama for fast, local inference with JSON structured output
CHAT_MODEL = "llama3.2:1b"
