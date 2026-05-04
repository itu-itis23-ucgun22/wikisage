# Product Requirements Document (PRD)
## WikiSage — Local Wikipedia RAG Assistant

### 1. Overview

WikiSage is an offline Retrieval-Augmented Generation (RAG) system designed to answer user questions about well-known people and locations. It retrieves relevant passages from Wikipedia and produces coherent, grounded responses using a locally hosted language model. The entire stack runs on the user's machine with no dependency on external APIs.

### 2. Problem Statement

Users require a fast, reliable way to get sourced answers about notable entities without depending on cloud-based AI services. Popular tools such as ChatGPT or Gemini require an internet connection and transmit data to remote servers. WikiSage addresses this gap by running entirely on localhost, offering privacy-preserving and citation-backed responses.

### 3. Target Users

- Students conducting research on historical figures and landmarks
- Educators building lesson materials and reference content
- Privacy-conscious individuals who need AI assistance without sending data off-device
- Developers studying RAG system design and implementation

### 4. User Stories

| # | As a... | I want to... | So that... |
|---|---|---|---|
| 1 | User | Ask about a notable person | I receive an accurate, sourced answer |
| 2 | User | Ask about a famous location | I learn key facts with supporting citations |
| 3 | User | Compare two entities | I get a structured side-by-side overview |
| 4 | User | Ask an out-of-scope question | I get an honest "I don't know" response |
| 5 | User | Inspect the sources behind an answer | I can independently verify the information |
| 6 | User | Reset my conversation | I can start a fresh session at any time |
| 7 | Admin | Add new entities to the knowledge base | The system's coverage grows over time |
| 8 | Admin | Wipe and rebuild the knowledge base | I can perform a clean re-ingestion from scratch |

### 5. Functional Requirements

#### 5.1 Data Ingestion
- **FR-1**: Retrieve complete Wikipedia articles for at least 20 people and 20 places
- **FR-2**: Split articles into overlapping chunks with configurable size and overlap
- **FR-3**: Produce embeddings for each chunk using a locally hosted model
- **FR-4**: Persist chunks, embeddings, and metadata in a durable vector database
- **FR-5**: Track which entities have been ingested to prevent duplicate entries
- **FR-6**: Support a full reset that clears all stored data before re-ingestion

#### 5.2 Query Processing
- **FR-7**: Classify each incoming query as targeting a person, a place, or both
- **FR-8**: Embed the user query using the same model used during ingestion
- **FR-9**: Retrieve the top-K most similar chunks from the vector store
- **FR-10**: Apply metadata filters to the vector search based on query classification

#### 5.3 Answer Generation
- **FR-11**: Generate responses using a locally running LLM — no external API calls
- **FR-12**: Constrain all answers strictly to the retrieved context
- **FR-13**: Return an "I don't know" style response when the context is insufficient
- **FR-14**: Stream the response token-by-token for a real-time display experience
- **FR-15**: Incorporate a limited chat history to support multi-turn conversations

#### 5.4 User Interface
- **FR-16**: Provide a browser-based chat interface built with Streamlit
- **FR-17**: Display system statistics including entity counts and vector DB chunk totals
- **FR-18**: Show the source passages that were used to generate each answer
- **FR-19**: Report per-query response latency
- **FR-20**: Offer clickable example questions to help users get started
- **FR-21**: Allow users to clear the current chat and manage session state

### 6. Non-Functional Requirements

- **NFR-1**: The entire system runs on localhost — no internet access needed after initial setup
- **NFR-2**: End-to-end response time should stay below 30 seconds on consumer hardware
- **NFR-3**: All stored data must survive application restarts
- **NFR-4**: Modular architecture where each component can be tested independently
- **NFR-5**: Documentation must be sufficient to allow setup from scratch

### 7. Technical Architecture

#### Components
1. **Wikipedia Fetcher** — Pulls article text via the `wikipedia-api` library
2. **Chunker** — Fixed-size character chunking with configurable overlap and sentence boundary detection
3. **Embedder** — Local embedding generation via Ollama (`nomic-embed-text`)
4. **Vector Store** — Custom implementation using NumPy (cosine similarity) and JSON (persistence), with metadata-based filtering. Chosen over ChromaDB to align with the assignment's "language-native functionality" requirement and to eliminate native C library dependencies.
5. **Retriever** — Hybrid retrieval: entity-targeted lookup (metadata filter + cosine ranking) for named-entity queries; keyword search combined with semantic search for general queries
6. **Generator** — Streaming LLM response generation via Ollama (`llama3.2:3b`)
7. **Database** — SQLite for ingestion tracking and persistent chat history
8. **UI** — Streamlit-based chat interface

#### Data Flow
```
Wikipedia → Fetch → Chunk → Embed (search_document:) → Vector Store (NumPy + JSON)
                                                                ↑
User Query → Classify → Embed (search_query:) → Hybrid Search ─┘
                                                      │
                                               Top-K Chunks → LLM → Streamed Answer
```

### 8. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector store | Custom NumPy + JSON (Option B, single collection) | Aligns with "language-native" requirement; no C dependencies; supports cross-category metadata filtering |
| Retrieval strategy | Hybrid: entity-targeted + keyword + semantic | Pure semantic similarity clustered all Wikipedia chunks too closely; hybrid approach restores precision |
| Embedding prefixes | `search_document:` / `search_query:` | Required by `nomic-embed-text` for asymmetric (query vs. document) retrieval quality |
| Chunking approach | Fixed-size with overlap | Predictable chunk sizes; overlap prevents loss of context at boundaries |
| Query classification | Rule-based (keywords + entity name/token matching) | Fast, transparent, and sufficient for the fixed entity set |
| LLM model | llama3.2:3b | Balances response quality with speed on consumer hardware |
| Embedding model | nomic-embed-text | Native Ollama support, 768-dimensional vectors |
| UI framework | Streamlit | Rapid development with native streaming chat components |

### 9. Success Criteria

- [ ] All 10 required people are ingested and queryable
- [ ] All 10 required places are ingested and queryable
- [ ] At least 10 additional entities (5 people, 5 places) are present in the knowledge base
- [ ] All example queries from the spec return relevant, grounded answers
- [ ] Out-of-scope queries consistently produce "I don't know" style responses
- [ ] Source passages are visible for every assistant response
- [ ] The system can be set up from scratch using only the README

### 10. Future Enhancements

- Full-context multi-turn conversation memory
- Cross-encoder re-ranking for improved retrieval precision
- Support for additional entity categories (events, organizations, concepts)
- Query-level response caching to reduce repeated computation
- Side-by-side model comparison mode across different LLMs
