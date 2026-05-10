"""Day-over-day anomaly detection for ticket clusters.

Approach:
  1. Compute per-cluster ticket counts for each day.
  2. Build a baseline from the specified baseline days (avg counts).
  3. Flag clusters whose Day N count deviates significantly from baseline:
       - spike    : magnitude_pct >= threshold (50% default)
       - drop     : magnitude_pct <= -threshold
       - new_cluster : 0 in baseline but >= min_new_cluster_tickets on target day
  4. Severity is function of magnitude AND absolute count (small noisy clusters
     shouldn't trigger critical alerts even at large % change).
  5. Each anomaly gets an LLM-generated Slack alert — with a safe template fallback.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from shared.config import settings
from shared.llm import safe_generate_json
from shared.models import Anomaly, ClusterLabel, DayStats, Ticket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — tight, actionable, format-enforced
# ---------------------------------------------------------------------------
_SLACK_PROMPT = """\
Generate a Slack alert for Eight Sleep's CX operations team about this ticket volume anomaly.

Anomaly details:
- Issue cluster: {cluster_label}
- Pattern: {cluster_description}
- Day {day}: {current} tickets  ({magnitude:+.0f}% vs {baseline_desc} of {baseline} tickets)
- Severity: {severity}
- Sample issues:
{sample_summaries}

Write a Slack message of exactly 2-3 sentences:
  Sentence 1 — name the specific issue and state the anomaly magnitude.
  Sentence 2 — give the baseline comparison and day context.
  Sentence 3 — one concrete next step for the on-call CX team.

Use *bold* for key terms. Be direct. No fluff. No emoji outside of ⚠️/🚨 for high/critical.

Return JSON: {{"slack_message": "..."}}"""


# ---------------------------------------------------------------------------
# Day statistics
# ---------------------------------------------------------------------------

def compute_day_stats(
    tickets: list[Ticket],
    labels: np.ndarray,
) -> dict[int, DayStats]:
    day_cluster: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    day_total: dict[int, int] = defaultdict(int)

    for ticket, label in zip(tickets, labels):
        d = ticket.day
        day_cluster[d][int(label)] += 1
        day_total[d] += 1

    stats: dict[int, DayStats] = {}
    for d in sorted(day_total):
        total = day_total[d]
        counts = dict(day_cluster[d])
        stats[d] = DayStats(
            day=d,
            total_tickets=total,
            cluster_counts=counts,
            cluster_percentages={cid: cnt / total * 100 for cid, cnt in counts.items()},
        )
    return stats


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    day_stats: dict[int, DayStats],
    cluster_labels: dict[int, ClusterLabel],
    tickets: list[Ticket],
    labels: np.ndarray,
    baseline_days: list[int],
    target_day: int,
    min_magnitude_pct: float = 50.0,
    min_absolute_count: int = 10,
) -> list[Anomaly]:
    """Detect anomalies on target_day relative to baseline_days."""
    target_stats = day_stats.get(target_day)
    if not target_stats:
        logger.warning(f"No data for Day {target_day}")
        return []

    valid_baseline = [d for d in baseline_days if d in day_stats]
    if not valid_baseline:
        logger.warning("No valid baseline days")
        return []

    # Baseline: average counts per cluster across baseline days
    all_cluster_ids: set[int] = set()
    for d in valid_baseline:
        all_cluster_ids.update(day_stats[d].cluster_counts.keys())
    all_cluster_ids.update(target_stats.cluster_counts.keys())

    baseline_counts: dict[int, float] = {
        cid: np.mean([day_stats[d].cluster_counts.get(cid, 0) for d in valid_baseline])
        for cid in all_cluster_ids
    }

    # Cluster → ticket IDs on target day (for sample summaries)
    cluster_target_ids: dict[int, list[str]] = defaultdict(list)
    ticket_map = {t.id: t for t in tickets}
    for ticket, label in zip(tickets, labels):
        if ticket.day == target_day:
            cluster_target_ids[int(label)].append(ticket.id)

    baseline_desc = (
        f"Day {valid_baseline[0]} baseline"
        if len(valid_baseline) == 1
        else f"Day {valid_baseline[0]}–{valid_baseline[-1]} average"
    )

    anomalies: list[Anomaly] = []

    for cid in all_cluster_ids:
        current = target_stats.cluster_counts.get(cid, 0)
        baseline = baseline_counts[cid]
        label_obj = cluster_labels.get(cid)
        cluster_label = label_obj.label if label_obj else f"Cluster {cid}"
        cluster_desc = label_obj.description if label_obj else ""

        # Classify anomaly type
        if baseline < 1:
            if current < settings.min_new_cluster_tickets:
                continue
            anomaly_type = "new_cluster"
            magnitude = 100.0
        else:
            magnitude = (current - baseline) / baseline * 100
            if magnitude >= min_magnitude_pct and current >= min_absolute_count:
                anomaly_type = "spike"
            elif magnitude <= -min_magnitude_pct and int(baseline) >= min_absolute_count:
                anomaly_type = "drop"
            else:
                continue

        severity = _severity(magnitude, current)
        sample_ids = cluster_target_ids.get(cid, [])[:5]
        sample_lines = "\n".join(
            f"  • {ticket_map[tid].to_display_text()[:120]}"
            for tid in sample_ids
            if tid in ticket_map
        )

        slack = _make_slack_alert(
            cluster_label=cluster_label,
            cluster_desc=cluster_desc,
            day=target_day,
            baseline=int(baseline),
            current=current,
            magnitude=magnitude,
            severity=severity,
            sample_lines=sample_lines,
            baseline_desc=baseline_desc,
        )

        anomalies.append(
            Anomaly(
                cluster_id=cid,
                cluster_label=cluster_label,
                cluster_description=cluster_desc,
                anomaly_type=anomaly_type,
                day_detected=target_day,
                baseline_count=int(baseline),
                current_count=current,
                magnitude_pct=round(magnitude, 1),
                severity=severity,
                description=(
                    f"{cluster_label}: {current} tickets on Day {target_day} "
                    f"vs {int(baseline)} baseline ({magnitude:+.0f}%)"
                ),
                slack_message=slack,
                sample_ticket_ids=sample_ids,
            )
        )

    return sorted(anomalies, key=lambda a: abs(a.magnitude_pct), reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity(magnitude: float, count: int) -> str:
    abs_mag = abs(magnitude)
    if abs_mag >= 200 or (abs_mag >= 100 and count >= 50):
        return "critical"
    if abs_mag >= 100 or (abs_mag >= 50 and count >= 30):
        return "high"
    if abs_mag >= 50:
        return "medium"
    return "low"


def _make_slack_alert(
    cluster_label: str,
    cluster_desc: str,
    day: int,
    baseline: int,
    current: int,
    magnitude: float,
    severity: str,
    sample_lines: str,
    baseline_desc: str,
) -> str:
    prompt = _SLACK_PROMPT.format(
        cluster_label=cluster_label,
        cluster_description=cluster_desc,
        day=day,
        current=current,
        magnitude=magnitude,
        baseline_desc=baseline_desc,
        baseline=baseline,
        severity=severity,
        sample_summaries=sample_lines or "  (no sample tickets)",
    )
    result = safe_generate_json(prompt, fallback={})
    if "slack_message" in result:
        return result["slack_message"]
    # Template fallback — always valid
    prefix = "🚨" if severity == "critical" else "⚠️" if severity == "high" else "📊"
    return (
        f"{prefix} *{cluster_label}* volume {'+' if magnitude > 0 else ''}{magnitude:.0f}% on Day {day}. "
        f"*{current} tickets* vs {baseline} baseline ({baseline_desc}). "
        f"Review recent tickets in this cluster for a common root cause."
    )
