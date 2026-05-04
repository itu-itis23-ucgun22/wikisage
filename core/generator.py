"""
Answer generation using a local LLM via Ollama.
Supports streaming responses for real-time display.
"""

import requests
import json
import logging
from config import OLLAMA_BASE_URL, LLM_MODEL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    if not chunks:
        return "No relevant context was found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        entity = meta.get("entity_name", "Unknown")
        entity_type = meta.get("type", "unknown")
        parts.append(
            f"[Source {i}: {entity} ({entity_type})]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate(
    query: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """
    Generate an answer using Ollama (non-streaming).

    Args:
        query: the user's question.
        chunks: retrieved context chunks from the vector store.
        chat_history: optional list of previous messages [{role, content}, ...].

    Returns:
        The generated answer string.
    """
    context = _build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history if available
    if chat_history:
        messages.extend(chat_history[-6:])  # Keep last 3 exchanges

    # User message with context
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above."
    )
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to Ollama. Please make sure Ollama is running "
            f"(`ollama serve`) at {OLLAMA_BASE_URL}."
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"❌ Generation error: {e}"


def generate_stream(
    query: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
):
    """
    Generate an answer using Ollama with streaming.

    Yields:
        str: incremental token strings as they arrive.
    """
    context = _build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history[-6:])

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above."
    )
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            },
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
                if data.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        yield (
            "❌ Cannot connect to Ollama. Please make sure Ollama is running "
            f"(`ollama serve`) at {OLLAMA_BASE_URL}."
        )
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield f"❌ Generation error: {e}"
