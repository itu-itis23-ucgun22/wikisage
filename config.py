"""
Central configuration for the Wikipedia RAG Assistant.
All tuneable parameters live here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "wiki_rag.db")

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500         # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
MAX_ARTICLE_LENGTH = 15000  # max chars to keep per article (intro + key sections)

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K = 5                 # number of chunks returned per query
COLLECTION_NAME = "wiki_chunks"

# ─── Entities to ingest ─────────────────────────────────────────────────────
PEOPLE = [
    # Required by spec
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Additional 10
    "Cleopatra",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Wolfgang Amadeus Mozart",
    "Isaac Newton",
    "Charles Darwin",
    "Pablo Picasso",
    "Aristotle",
    "Napoleon Bonaparte",
    "Amelia Earhart",
]

PLACES = [
    # Required by spec
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    # Additional 10
    "Stonehenge",
    "Petra",
    "Angkor Wat",
    "Great Barrier Reef",
    "Niagara Falls",
    "Santorini",
    "Galápagos Islands",
    "Chichen Itza",
    "Mount Fuji",
    "Amazon rainforest",
]

# ─── Generation prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using ONLY the provided context from Wikipedia articles.

Rules:
1. Base your answer strictly on the context provided below.
2. If the context does not contain enough information to answer, say "I don't have enough information to answer that question."
3. Do not make up facts or hallucinate information.
4. When referencing information, mention which entity (person or place) it comes from.
5. Keep answers clear, concise, and well-structured.
6. For comparison questions, organize your answer with clear sections for each entity."""
