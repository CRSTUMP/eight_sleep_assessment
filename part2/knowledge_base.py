"""RAG knowledge base: index resolved tickets in ChromaDB.

Design decisions:
- Embeddings use the same local sentence-transformers model as Part 1.
  Consistent model means clustering embeddings and retrieval embeddings are
  directly comparable — no task-type mismatch.
- We reuse Part 1's cached embedding matrix when available (no re-embedding).
- ChromaDB stores pre-computed embeddings directly (no EmbeddingFunction),
  giving us full control and avoiding double-computation.
- Cosine similarity is configured via hnsw:space="cosine" on the collection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

from shared.config import settings
from shared.embedder import embed
from shared.models import Ticket

logger = logging.getLogger(__name__)

COLLECTION_NAME = "resolved_tickets"

_EMBED_CACHE = settings.cache_dir / "embeddings.npy"
_ID_CACHE = settings.cache_dir / "embedding_ids.json"


def build(tickets: list[Ticket], all_embeddings: np.ndarray | None = None) -> chromadb.Collection:
    """Build (or rebuild) the ChromaDB collection for resolved tickets.

    Parameters
    ----------
    tickets:        All tickets (resolved + open).
    all_embeddings: Optional pre-computed embedding matrix aligned with tickets.
                    Reuses Part 1 cache if None and cache is present.
    """
    resolved, resolved_embs = _filter_resolved(tickets, all_embeddings)
    logger.info(f"Indexing {len(resolved)} resolved tickets in ChromaDB")

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in tqdm(range(0, len(resolved), batch_size), desc="Indexing ChromaDB"):
        batch_t = resolved[i : i + batch_size]
        batch_e = resolved_embs[i : i + batch_size]
        collection.add(
            ids=[t.id for t in batch_t],
            embeddings=batch_e.tolist(),
            documents=[t.to_embedding_text() for t in batch_t],
            metadatas=[
                {
                    "category": t.category,
                    "issue_type": t.issue_type,
                    "product": t.product,
                    "priority": t.priority,
                    "resolution_type": t.resolution_type or "",
                }
                for t in batch_t
            ],
        )

    logger.info(f"ChromaDB '{COLLECTION_NAME}': {collection.count()} docs indexed")
    return collection


def load() -> chromadb.Collection:
    """Load an existing ChromaDB collection (no rebuild)."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    col = client.get_collection(COLLECTION_NAME)
    logger.info(f"ChromaDB '{COLLECTION_NAME}' loaded: {col.count()} docs")
    return col


def retrieve(
    query: str,
    collection: chromadb.Collection,
    tickets: list[Ticket],
    n: int | None = None,
) -> list[tuple[Ticket, float]]:
    """Return top-n (ticket, similarity_score) pairs for a query string."""
    k = n or settings.top_k_retrieval
    query_emb = embed([query])[0]
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=k,
        include=["distances", "metadatas"],
    )
    ticket_map = {t.id: t for t in tickets}
    pairs: list[tuple[Ticket, float]] = []
    for tid, dist in zip(results["ids"][0], results["distances"][0]):
        t = ticket_map.get(tid)
        if t:
            pairs.append((t, round(1.0 - dist, 3)))  # cosine distance → similarity
    return pairs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_resolved(
    tickets: list[Ticket],
    all_embeddings: np.ndarray | None,
) -> tuple[list[Ticket], np.ndarray]:
    resolved_idxs = [i for i, t in enumerate(tickets) if t.status == "resolved"]
    resolved = [tickets[i] for i in resolved_idxs]

    if all_embeddings is not None:
        return resolved, all_embeddings[resolved_idxs]

    # Try reusing Part 1 cache
    if _EMBED_CACHE.exists() and _ID_CACHE.exists():
        cached_ids = json.loads(_ID_CACHE.read_text())
        id_to_idx = {tid: i for i, tid in enumerate(cached_ids)}
        cached_embs = np.load(_EMBED_CACHE)
        idxs = [id_to_idx[t.id] for t in resolved if t.id in id_to_idx]
        if len(idxs) == len(resolved):
            logger.info("Reusing Part 1 embedding cache for ChromaDB indexing")
            return resolved, cached_embs[idxs]

    # Fallback: compute embeddings for resolved tickets only
    logger.info(f"Computing embeddings for {len(resolved)} resolved tickets")
    resolved_embs = embed([t.to_embedding_text() for t in resolved], show_progress=True)
    return resolved, resolved_embs
