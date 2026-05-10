"""Part 1 main pipeline: embed → cluster → label → detect anomalies."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from shared.models import Anomaly, ClusterLabel, DayStats, Ticket
from .anomaly import compute_day_stats, detect_anomalies
from .clustering import (
    assign_noise_to_nearest,
    cluster_tickets,
    label_clusters,
    reduce_dimensions,
)
from .embeddings import get_or_create_embeddings, load_tickets

logger = logging.getLogger(__name__)


def run_pipeline(force_recompute: bool = False) -> dict:
    # 1. Load data
    tickets = load_tickets()

    # 2. Embed (cached)
    embeddings = get_or_create_embeddings(tickets, force=force_recompute)

    # 3. Dimensionality reduction
    reduced = reduce_dimensions(embeddings)

    # 4. Cluster
    labels = cluster_tickets(reduced)
    labels = assign_noise_to_nearest(labels, reduced)

    # 5. LLM cluster labels
    cluster_labels = label_clusters(tickets, labels)

    # 6. Day stats
    day_stats = compute_day_stats(tickets, labels)
    _log_day_summary(day_stats, cluster_labels)

    # 7. Anomaly detection
    # Day 2 anomalies: baseline = Day 1
    anomalies_day2 = detect_anomalies(
        day_stats, cluster_labels, tickets, labels,
        baseline_days=[1], target_day=2,
    )
    # Day 3 anomalies: baseline = Days 1+2 (richer signal)
    anomalies_day3 = detect_anomalies(
        day_stats, cluster_labels, tickets, labels,
        baseline_days=[1, 2], target_day=3,
    )

    logger.info(f"Anomalies — Day 2: {len(anomalies_day2)}, Day 3: {len(anomalies_day3)}")

    return {
        "clusters": {str(cid): cl.model_dump() for cid, cl in cluster_labels.items()},
        "day_stats": {str(d): ds.model_dump() for d, ds in day_stats.items()},
        "anomalies_day2": [a.model_dump() for a in anomalies_day2],
        "anomalies_day3": [a.model_dump() for a in anomalies_day3],
        # raw labels for Part 2 reuse
        "_ticket_ids": [t.id for t in tickets],
        "_labels": labels.tolist(),
    }


def _log_day_summary(
    day_stats: dict[int, DayStats],
    cluster_labels: dict[int, ClusterLabel],
) -> None:
    for day, stats in sorted(day_stats.items()):
        top = sorted(stats.cluster_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(
            f"{cluster_labels.get(cid, ClusterLabel(cluster_id=cid, label=str(cid), description='', keywords=[])).label}({cnt})"
            for cid, cnt in top
        )
        logger.info(f"Day {day}: {stats.total_tickets} tickets | top clusters: {top_str}")
