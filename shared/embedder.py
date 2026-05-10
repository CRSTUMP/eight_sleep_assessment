"""Local sentence-transformers embedding module.

Model: all-mpnet-base-v2
  - 768-dimensional embeddings
  - Strong semantic quality; standard benchmark choice for clustering + retrieval
  - ~420 MB download, cached by HuggingFace Hub after first run
  - ~1,000 sentences/second on CPU; no API rate limits, no cost

Single global model instance is loaded lazily and reused across calls.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Windows long-path workaround: sentence-transformers may have been
        # installed in a short-path venv (C:\ev) due to Windows path length limits.
        import pathlib
        fallback = pathlib.Path(r"C:\ev\Lib\site-packages")
        if fallback.exists():
            import sys
            sys.path.insert(0, str(fallback))
            from sentence_transformers import SentenceTransformer
        else:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Windows: python -m venv C:\\ev && C:\\ev\\Scripts\\pip install sentence-transformers\n"
                "Other:   pip install sentence-transformers"
            )
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
    """Embed texts locally. Returns (N, 768) float32 array.

    Parameters
    ----------
    texts:         List of strings to embed.
    batch_size:    Internal batch size for the model (affects memory, not API calls).
    show_progress: Show tqdm progress bar (useful for large batches).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise so cosine sim = dot product
    )
    return embeddings.astype(np.float32)
