# 🔍 WikiSage — Local Wikipedia RAG Assistant

https://youtu.be/raf2PFvBf1Q

https://github.com/itu-itis23-ucgun22/wikisage

A privacy-first, fully offline question-answering system that draws on Wikipedia content to answer queries about notable people and locations. It combines vector-based retrieval with a locally-running language model — no internet connection or cloud API required after the initial setup.

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  Streamlit   │  Streaming chat interface
│  Frontend    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────────┐
│  Retriever   │────▶│  Custom Vector Store  │  Cosine similarity search
│  (classify   │     │  (NumPy + JSON)       │  with metadata-based filtering
│   + search)  │     └──────────────────────┘
└──────┬──────┘
       │ top-K chunks
       ▼
┌─────────────┐     ┌──────────────┐
│  Generator   │────▶│  Ollama       │  On-device LLM inference
│  (prompt +   │     │  llama3.2:3b  │
│   stream)    │     └──────────────┘
└─────────────┘
```

**Vector store design (Option B):** A single collection with metadata filtering rather than two separate stores. This supports cross-category queries (e.g., "Which famous place is in Turkey?") while keeping the system simple. The `type` field (`"person"` or `"place"`) narrows retrieval when the query classification is confident.

**Why a custom vector store instead of ChromaDB:** The assignment encourages using language-native functionality over fully featured libraries. The custom store (NumPy cosine similarity + JSON persistence) implements the same interface with no external C dependencies, making it portable across all platforms without native binary issues.

## 📋 Prerequisites

- **Python 3.11+**
- **Ollama** — Download from [ollama.com](https://ollama.com)

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd aia_hw3
pip install -r requirements.txt
```

### 2. Download the required Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Make sure Ollama is running before proceeding:
```bash
ollama serve
```

### 3. Load Wikipedia data

```bash
python ingest.py
```

This step will:
- Retrieve 20 people + 20 places from Wikipedia
- Split each article into overlapping chunks (1500 chars, 200-char overlap)
- Produce embeddings using `nomic-embed-text` with `search_document:` prefix
- Persist everything in `data/vector_store/` as JSON + NumPy files

To wipe and re-ingest from the beginning:
```bash
python ingest.py --reset
```

### 4. Launch the application

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## 💬 Example Queries

### People
- "Who was Albert Einstein and what is he known for?"
- "What did Marie Curie discover?"
- "Why is Nikola Tesla famous?"
- "Compare Lionel Messi and Cristiano Ronaldo"
- "What is Frida Kahlo known for?"

### Places
- "Where is the Eiffel Tower located?"
- "Why is the Great Wall of China important?"
- "What was the Colosseum used for?"
- "What is Machu Picchu?"
- "Where is Mount Everest?"

### Mixed
- "Which famous place is located in Turkey?"
- "Which person is associated with electricity?"
- "Compare Albert Einstein and Nikola Tesla"
- "Compare the Eiffel Tower and the Statue of Liberty"

### Out-of-scope (should return "I don't know")
- "Who is the president of Mars?"
- "Tell me about John Doe"

## 📁 Project Structure

```
aia_hw3/
├── app.py                    # Streamlit chat UI (main entry point)
├── ingest.py                 # Wikipedia data ingestion CLI
├── config.py                 # Centralized configuration values
├── requirements.txt          # Python package dependencies
├── README.md                 # This document
├── product_prd.md            # Product Requirements Document
├── recommendation.md         # Production deployment guide
├── project_description.txt   # Original assignment specification
│
├── core/
│   ├── __init__.py
│   ├── wikipedia_fetcher.py  # Retrieves Wikipedia article text
│   ├── chunker.py            # Splits documents into overlapping chunks
│   ├── embedder.py           # Generates embeddings via Ollama
│   ├── vector_store.py       # Custom vector store (NumPy cosine similarity + JSON)
│   ├── retriever.py          # Query classification and hybrid retrieval
│   ├── generator.py          # LLM answer generation with streaming
│   └── database.py           # SQLite for ingestion tracking and chat history
│
└── data/                     # Created automatically at runtime
    ├── vector_store/         # Embeddings (.npy) and metadata (.json)
    └── wiki_rag.db           # SQLite database file
```

## ⚙️ Configuration

All adjustable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2:3b` | Ollama model used for answer generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for embeddings |
| `CHUNK_SIZE` | 1500 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Character overlap between adjacent chunks |
| `TOP_K` | 5 | Number of chunks fetched per query |

## 🧠 Technical Details

### Chunking Strategy
Articles are split into fixed-size character chunks (1500 chars) with a 200-character overlap. The chunker attempts to break at sentence boundaries to avoid cutting sentences mid-way, preserving semantic coherence while ensuring context continuity between chunks.

### Query Classification
A rule-based three-tier pipeline:
1. **Entity name matching** — checks whether known entity names (or their last names) appear in the query
2. **Keyword heuristics** — words like "who", "born" signal a person; "where", "located" signal a place
3. **Fallback** — searches across both categories when the query is ambiguous

### Retrieval — Hybrid Approach
The retriever uses two strategies depending on whether named entities are detected:

**When entity names are found in the query** (e.g., "Compare Messi and Ronaldo"):
- Each named entity is looked up directly by `entity_name` metadata filter
- Cosine similarity ranks the most relevant chunks *within* that entity's document set
- This combines the precision of rule-based lookup with the ranking power of vector similarity

**When no entity name is found** (e.g., "Which place is in Turkey?"):
- **Keyword search**: content-specific terms from the query (excluding classification signal words) are matched against stored chunk text; rarer geographic/topical terms score higher
- **Semantic search**: cosine similarity against all chunks of the appropriate type
- Both result sets are merged (keyword hits prioritized) and deduplicated

**Why this hybrid design:** Pure semantic similarity over 1500-character Wikipedia chunks performed poorly — all biography chunks clustered near the same region of the embedding space, making entity discrimination unreliable. The hybrid approach preserves the vector store infrastructure while making retrieval robust for the fixed entity set.

### Embedding
`nomic-embed-text` requires task-specific prefixes for asymmetric retrieval:
- Documents are embedded with `search_document: <text>` during ingestion
- Queries are embedded with `search_query: <text>` at runtime

### Generation
The language model is given a strict system prompt instructing it to answer solely from the provided context. When the context is insufficient, it responds with "I don't have enough information." Responses are streamed token-by-token for real-time display.

## 🎥 Demo Video

[Link to demo video]

## 📄 License

Built for educational purposes as part of a university course assignment.
