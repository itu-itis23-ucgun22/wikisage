"""
Document chunking with fixed-size character windows and configurable overlap.
Designed to handle large documents efficiently.
"""

import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.

    Tries to break at sentence boundaries within the chunk window to avoid
    cutting mid-sentence. Falls back to hard split if no boundary is found.

    Args:
        text: the full document text.
        chunk_size: target number of characters per chunk.
        chunk_overlap: number of overlapping characters between chunks.

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a sentence boundary (., !, ?) near the end
        if end < text_len:
            # Look backwards from `end` for a sentence-ending character
            search_start = max(start + chunk_size // 2, start)
            best_break = -1
            for i in range(end, search_start, -1):
                if text[i - 1] in ".!?\n" and (i >= text_len or text[i] == " " or text[i] == "\n"):
                    best_break = i
                    break

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = end - chunk_overlap if end < text_len else text_len

    return chunks


def chunk_article(article: dict, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Chunk a fetched article and attach metadata to each chunk.

    Args:
        article: dict with keys {title, text, url, type}.

    Returns:
        List of dicts: {text, metadata: {entity_name, type, chunk_index, source_url}}.
    """
    raw_chunks = chunk_text(article["text"], chunk_size, chunk_overlap)

    result = []
    for i, chunk in enumerate(raw_chunks):
        result.append({
            "id": f"{article['title'].replace(' ', '_')}_{i}",
            "text": chunk,
            "metadata": {
                "entity_name": article["title"],
                "type": article["type"],
                "chunk_index": i,
                "source_url": article["url"],
            },
        })

    logger.info(
        f"Chunked '{article['title']}' into {len(result)} chunks "
        f"(avg {sum(len(c['text']) for c in result) // max(len(result), 1)} chars)"
    )
    return result
