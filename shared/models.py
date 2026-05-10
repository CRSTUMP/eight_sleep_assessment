"""Pydantic models for Eight Sleep ticket data.

The raw JSON is nested (metadata, resolution sub-objects).
A model_validator flattens it into a clean Ticket so all downstream code
uses simple attribute access without nested dicts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: str
    content: str  # mapped from raw 'text' field
    timestamp: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: dict) -> dict:
        data = dict(data)
        if "text" in data and "content" not in data:
            data["content"] = data.pop("text")
        if "created_at" in data and "timestamp" not in data:
            data["timestamp"] = data["created_at"]
        return data


class Ticket(BaseModel):
    # Core identity
    id: str
    customer_id: str = ""

    # Timing — day is provided directly in the data (1/2/3)
    day: int = 1
    created_at: Optional[datetime] = None

    # Classification
    status: str
    category: str
    issue_type: str
    product: str
    priority: str

    # Conversation
    messages: list[Message]

    # Resolution (only for resolved tickets)
    resolution_type: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: dict) -> dict:
        """Accept both flat and the raw nested JSON format."""
        if "metadata" not in data:
            return data  # already flat

        meta = data.get("metadata") or {}
        res = data.get("resolution") or {}

        return {
            "id": data.get("conversation_id") or data.get("id", ""),
            "customer_id": data.get("customer_id", ""),
            "day": meta.get("day", 1),
            "created_at": meta.get("created_at"),
            "status": meta.get("status", "open"),
            "category": meta.get("category", ""),
            "issue_type": meta.get("issue_type", ""),
            "product": meta.get("product", ""),
            "priority": meta.get("priority", "medium"),
            "messages": data.get("messages", []),
            "resolution_type": res.get("resolution_type") if res else None,
            "resolution_notes": res.get("resolution_notes") if res else None,
            "resolved_at": res.get("resolved_at") if res else None,
        }

    # ------------------------------------------------------------------
    # Text representations
    # ------------------------------------------------------------------

    def _customer_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role in ("customer", "user")]

    def _agent_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role in ("agent", "support", "agent_bot")]

    def to_embedding_text(self) -> str:
        """Representative text optimised for semantic clustering / retrieval."""
        parts = [
            f"Category: {self.category} | Issue: {self.issue_type} | "
            f"Product: {self.product} | Priority: {self.priority}"
        ]
        customer_msgs = self._customer_messages()
        if customer_msgs:
            parts.append(f"Problem: {customer_msgs[0].content[:600]}")
            if len(customer_msgs) > 1 and len(customer_msgs[-1].content) > 50:
                parts.append(f"Follow-up: {customer_msgs[-1].content[:200]}")
        if self.resolution_notes:
            parts.append(f"Resolution: {self.resolution_notes[:400]}")
        return "\n".join(parts)

    def to_display_text(self) -> str:
        """Concise one-liner for LLM cluster-labeling prompts."""
        customer_msgs = self._customer_messages()
        first = customer_msgs[0].content[:250] if customer_msgs else "No message"
        return (
            f"[{self.id}] {self.category}/{self.issue_type} | {self.product} | {self.priority}\n"
            f"  '{first}'"
        )

    def to_chat_context(self) -> str:
        """Full context block for the RAG chatbot display."""
        lines = [
            f"Ticket ID: {self.id}",
            f"Category: {self.category} / {self.issue_type}",
            f"Product: {self.product} | Priority: {self.priority} | Status: {self.status}",
            "Customer messages:",
        ]
        for i, m in enumerate(self._customer_messages()[:5], 1):
            lines.append(f"  [{i}] {m.content[:400]}")
        if self.resolution_notes:
            lines.append(f"Resolution: {self.resolution_notes[:500]}")
        if self.resolution_type:
            lines.append(f"Resolution type: {self.resolution_type}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class ClusterLabel(BaseModel):
    cluster_id: int
    label: str
    description: str
    keywords: list[str]
    ticket_count: int = 0


class DayStats(BaseModel):
    day: int
    total_tickets: int
    cluster_counts: dict[int, int]
    cluster_percentages: dict[int, float]


class Anomaly(BaseModel):
    cluster_id: int
    cluster_label: str
    cluster_description: str
    anomaly_type: Literal["spike", "new_cluster", "drop"]
    day_detected: int
    baseline_count: int
    current_count: int
    magnitude_pct: float
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    slack_message: str
    sample_ticket_ids: list[str] = Field(default_factory=list)
