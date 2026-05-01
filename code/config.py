import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"  # Fast, highly capable of structured output
