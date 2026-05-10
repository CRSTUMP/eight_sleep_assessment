"""Dimensionality reduction → HDBSCAN clustering → LLM cluster labeling.

Pipeline:
  raw embeddings (768D)
    → L2 normalise
    → PCA (→ 100D, fast linear pre-reduction)
    → UMAP (→ 15D, non-linear structure preservation)
    → HDBSCAN (variable-density clustering, handles noise)
    → noise reassignment to nearest centroid
    → LLM labels top clusters (metadata fallback for smaller ones)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import normalize

from shared.config import settings
from shared.llm import safe_generate_json
from shared.models import ClusterLabel, Ticket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — context engineering: specific, constrained, format-enforced
# ---------------------------------------------------------------------------
_LABEL_PROMPT = """\
You are analysing customer support tickets for Eight Sleep, a smart mattress/sleep-tech company.
Below are {n} representative tickets from ONE cluster. Identify the precise, shared problem pattern.

{tickets}

Return a JSON object with EXACTLY these keys:
{{
  "label": "3-5 word specific label — name the actual failure, not the product category",
  "description": "One sentence: what is the customer experiencing and what causes it?",
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"]
}}

Good label examples: "Pod Temperature Regulation Failure", "App Sleep Score Not Updating",
"Subscription Renewal Billing Error", "Mattress Cover Zipper Defect".
Bad examples: "App Issue", "Hardware Problem", "Billing"."""


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    normed = normalize(embeddings)

    # PCA pre-reduction (fast, avoids UMAP curse-of-dimensionality on 768D)
    try:
        from sklearn.decomposition import PCA
        pca_dims = min(100, normed.shape[0] - 1, normed.shape[1])
        pca = PCA(n_components=pca_dims, random_state=42)
        pca_reduced = pca.fit_transform(normed)
        logger.info(f"PCA: {normed.shape} → {pca_reduced.shape} ({pca.explained_variance_ratio_.sum():.1%} variance)")
    except Exception as e:
        logger.warning(f"PCA failed: {e}, using raw normed embeddings")
        pca_reduced = normed

    # UMAP non-linear reduction
    try:
        from umap import UMAP
        reducer = UMAP(
            n_components=settings.umap_n_components,
            n_neighbors=settings.umap_n_neighbors,
            min_dist=0.0,
            metric="euclidean",
            random_state=42,
            low_memory=True,
        )
        reduced = reducer.fit_transform(pca_reduced)
        logger.info(f"UMAP: {pca_reduced.shape} → {reduced.shape}")
        return reduced
    except ImportError:
        logger.warning("umap-learn not installed — using PCA output for clustering")
        return pca_reduced


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_tickets(reduced: np.ndarray) -> np.ndarray:
    """Return integer label array; -1 = noise (reassigned downstream)."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=settings.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels: np.ndarray = clusterer.fit_predict(reduced)
    except ImportError:
        logger.warning("hdbscan not installed — falling back to KMeans with silhouette selection")
        labels = _kmeans_fallback(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    logger.info(f"Clustering: {n_clusters} clusters, {noise_pct:.1f}% noise")
    return labels


def _kmeans_fallback(reduced: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score, best_labels = 10, -1.0, None
    for k in range(5, 30):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(reduced)
        score = silhouette_score(reduced, lbls, sample_size=min(2000, len(reduced)))
        if score > best_score:
            best_k, best_score, best_labels = k, score, lbls
    logger.info(f"KMeans selected k={best_k} (silhouette={best_score:.3f})")
    return best_labels  # type: ignore[return-value]


def assign_noise_to_nearest(labels: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    """Reassign noise points (-1) to the nearest cluster centroid."""
    if -1 not in labels:
        return labels
    labels = labels.copy()
    cluster_ids = sorted(c for c in set(labels) if c != -1)
    centroids = np.stack([reduced[labels == c].mean(axis=0) for c in cluster_ids])
    noise_mask = labels == -1
    dists = np.linalg.norm(reduced[noise_mask, None] - centroids[None, :], axis=2)
    labels[noise_mask] = np.array(cluster_ids)[dists.argmin(axis=1)]
    logger.info(f"Reassigned {noise_mask.sum()} noise points")
    return labels


# ---------------------------------------------------------------------------
# LLM cluster labeling
# ---------------------------------------------------------------------------

def label_clusters(
    tickets: list[Ticket],
    labels: np.ndarray,
    samples_per_cluster: int = 7,
    delay_between_calls: float = 2.0,
) -> dict[int, ClusterLabel]:
    """Label clusters with LLM (Groq Llama 3.3-70b).

    Only the top `max_clusters_to_label` clusters (by size) get full LLM
    labels. 2s inter-call delay keeps us well under Groq's 30 RPM limit.
    Smaller clusters receive a metadata-derived fallback label.
    """
    cluster_to_tickets: dict[int, list[Ticket]] = defaultdict(list)
    for ticket, label in zip(tickets, labels):
        cluster_to_tickets[int(label)].append(ticket)

    # Sort by size descending so the most important clusters are labeled first
    sorted_by_size = sorted(cluster_to_tickets.keys(), key=lambda c: len(cluster_to_tickets[c]), reverse=True)
    to_label = sorted_by_size[: settings.max_clusters_to_label]
    to_fallback = sorted_by_size[settings.max_clusters_to_label :]


    cluster_labels: dict[int, ClusterLabel] = {}
    logger.info(f"LLM labeling {len(to_label)} clusters; {len(to_fallback)} use metadata fallback")

    for cid in to_label:
        c_tickets = cluster_to_tickets[cid]
        sample = c_tickets[:samples_per_cluster]
        ticket_text = "\n\n".join(
            f"Ticket {i+1}:\n{t.to_display_text()}" for i, t in enumerate(sample)
        )
        prompt = _LABEL_PROMPT.format(n=len(sample), tickets=ticket_text)
        result = safe_generate_json(prompt, fallback={})
        if result:
            cluster_labels[cid] = ClusterLabel(
                cluster_id=cid,
                label=result.get("label", _metadata_label(c_tickets)),
                description=result.get("description", ""),
                keywords=result.get("keywords", []),
                ticket_count=len(c_tickets),
            )
        else:
            cluster_labels[cid] = _fallback_label(cid, c_tickets)
        time.sleep(delay_between_calls)

    for cid in to_fallback:
        cluster_labels[cid] = _fallback_label(cid, cluster_to_tickets[cid])

    return cluster_labels


def _metadata_label(tickets: list[Ticket]) -> str:
    from collections import Counter
    issue = Counter(t.issue_type for t in tickets).most_common(1)[0][0]
    return issue.replace("_", " ").title()


def _fallback_label(cid: int, tickets: list[Ticket]) -> ClusterLabel:
    from collections import Counter
    issue = Counter(t.issue_type for t in tickets).most_common(1)[0][0]
    category = Counter(t.category for t in tickets).most_common(1)[0][0]
    label = issue.replace("_", " ").title()
    return ClusterLabel(
        cluster_id=cid,
        label=label,
        description=f"{category.title()} issue: {issue.replace('_', ' ')}",
        keywords=[issue, category],
        ticket_count=len(tickets),
    )
