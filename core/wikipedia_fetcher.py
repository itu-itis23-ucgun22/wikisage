"""
Fetches full Wikipedia articles for a given list of entity names.
Uses the wikipedia-api library for clean text extraction.
"""

import wikipediaapi
import logging
from config import MAX_ARTICLE_LENGTH

logger = logging.getLogger(__name__)


def create_wiki_client() -> wikipediaapi.Wikipedia:
    """Create a Wikipedia API client with a proper user-agent."""
    return wikipediaapi.Wikipedia(
        user_agent="WikiSageAssistant/1.0 (university project)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def fetch_article(wiki: wikipediaapi.Wikipedia, title: str) -> dict | None:
    """
    Fetch a single Wikipedia article by title.

    Returns:
        dict with keys: title, text, url  — or None if not found.
    """
    page = wiki.page(title)

    if not page.exists():
        logger.warning(f"Page not found: '{title}'")
        return None

    text = page.text
    if not text or len(text.strip()) < 100:
        logger.warning(f"Page too short or empty: '{title}'")
        return None

    # Truncate to keep only the most important intro + early sections
    if len(text) > MAX_ARTICLE_LENGTH:
        text = text[:MAX_ARTICLE_LENGTH]
        # Try to end at a sentence boundary
        last_period = text.rfind(".")
        if last_period > MAX_ARTICLE_LENGTH * 0.8:
            text = text[: last_period + 1]

    logger.info(f"Fetched '{page.title}' — {len(text):,} chars")
    return {
        "title": page.title,
        "text": text,
        "url": page.fullurl,
    }


def fetch_all(
    titles: list[str], entity_type: str
) -> list[dict]:
    """
    Fetch multiple Wikipedia articles.

    Args:
        titles: list of article titles to fetch.
        entity_type: "person" or "place" — attached as metadata.

    Returns:
        List of dicts: {title, text, url, type}.
    """
    wiki = create_wiki_client()
    results = []

    for title in titles:
        article = fetch_article(wiki, title)
        if article:
            article["type"] = entity_type
            results.append(article)
        else:
            logger.error(f"FAILED to fetch '{title}' — skipping")

    logger.info(
        f"Fetched {len(results)}/{len(titles)} {entity_type} articles"
    )
    return results
