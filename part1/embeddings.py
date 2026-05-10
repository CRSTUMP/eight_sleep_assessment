"""Ticket loading and local embedding with disk caching.

Embeddings are computed locally via sentence-transformers (no API calls, no rate limits).
Results cached to .cache/ — subsequent runs load in ~1 second.
Cache is invalidated automatically if the ticket list changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from shared.config import settings
from shared.embedder import embed
from shared.models import Ticket

logger = logging.getLogger(__name__)

_EMBED_CACHE = settings.cache_dir / "embeddings.npy"
_ID_CACHE = settings.cache_dir / "embedding_ids.json"


def load_tickets(path: Optional[Path] = None) -> list[Ticket]:
    data_path = path or settings.data_path
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    tickets: list[Ticket] = []
    for item in raw:
        try:
            tickets.append(Ticket.model_validate(item))
        except Exception as e:
            logger.debug(f"Skipping malformed ticket {item.get('conversation_id', '?')}: {e}")

    logger.info(f"Loaded {len(tickets)} tickets from {data_path}")
    return tickets


def get_or_create_embeddings(
    tickets: list[Ticket],
    force: bool = False,
) -> np.ndarray:
    """Return (N, D) embedding matrix, computing and caching if needed."""
    settings.cache_dir.mkdir(exist_ok=True)
    ticket_ids = [t.id for t in tickets]

    if not force and _EMBED_CACHE.exists() and _ID_CACHE.exists():
        if json.loads(_ID_CACHE.read_text()) == ticket_ids:
            logger.info("Loading embeddings from cache")
            return np.load(_EMBED_CACHE)

    logger.info(f"Computing embeddings for {len(tickets)} tickets (local model, no API calls)")
    texts = [t.to_embedding_text() for t in tickets]
    embeddings = embed(texts, show_progress=True)

    np.save(_EMBED_CACHE, embeddings)
    _ID_CACHE.write_text(json.dumps(ticket_ids))
    logger.info(f"Embeddings cached: shape={embeddings.shape}")
    return embeddings
