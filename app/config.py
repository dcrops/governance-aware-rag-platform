import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "data/index")

DEFAULT_CLIENT_NAME = os.getenv(
    "RAG_DEFAULT_CLIENT_NAME",
    "demo_client",
)

DEFAULT_TOP_K = int(
    os.getenv("RAG_DEFAULT_TOP_K", "5")
)

DEFAULT_MIN_SCORE = float(
    os.getenv("RAG_DEFAULT_MIN_SCORE", "0.35")
)