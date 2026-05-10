"""RAG chatbot for Part 2.

Context engineering strategy:
  System prompt   — role, constraints, Eight Sleep domain, output format
  Context block   — top-k retrieved resolved tickets (formatted with IDs + similarity)
  Conversation    — last N turns of chat history (sliding window)
  User message    — current query / open ticket

Context window budget (Groq llama-3.3-70b, 128k ctx):
  System:      ~600 tokens
  Retrieved:   5 tickets × ~400 tokens = ~2,000 tokens
  History:     last 6 turns × ~150 tokens = ~900 tokens
  Query:       ~200 tokens
  Total:       ~3,700 tokens — well within budget, leaves room for long tickets
"""

from __future__ import annotations

import logging

import chromadb

from part2.knowledge_base import retrieve
from shared.llm import groq_stream
from shared.models import Ticket

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6  # pairs (user + assistant)

# ---------------------------------------------------------------------------
# System prompt — the core context engineering asset
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an Eight Sleep CX Intelligence assistant embedded in the support agent console.
Your job: given an open customer ticket, surface proven resolution paths grounded in
our historical resolved tickets. You do not guess — you cite.

ABOUT EIGHT SLEEP:
Eight Sleep makes smart mattress covers ("the Pod") that regulate sleep temperature.
Common problem areas: Pod temperature regulation, app connectivity, sleep tracking
accuracy, firmware updates, hardware installation/setup, subscription billing, and
physical product defects (cover, hub, water pipes).

YOUR RESPONSIBILITIES:
1. Identify the core issue(s) in the customer's message.
2. Match to the most relevant resolved tickets provided in the KNOWLEDGE BASE block.
3. Synthesise a step-by-step resolution path, citing the source ticket IDs.
4. Flag when an issue has no match in the knowledge base — do not invent steps.
5. For multi-issue tickets, address each issue in its own section.

RESPONSE FORMAT (use exactly these markdown headers):
## Issue(s) Identified
[1-2 sentences. Name the specific failure mode, not just the category.]

## Resolution Path
*Source tickets: [ID-1], [ID-2]*
1. [Actionable step — specific enough for an agent to execute immediately]
2. [Next step]
...

## If the Above Doesn't Work
[Alternative path from a different resolved ticket, if one exists. Otherwise omit.]

## Escalate When
[Specific triggers drawn from historical patterns — e.g. "if customer reports X
after step 3, escalate to Tier-2 hardware team".]

CONSTRAINTS:
- Cite at least one ticket ID for every step you recommend.
- If the knowledge base contains no relevant resolved ticket, say so explicitly
  and suggest the agent open a new escalation path.
- Keep the total response under 350 words unless the issue is genuinely complex.
- Plain language — agents relay these steps to customers verbally."""

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_retrieved(pairs: list[tuple[Ticket, float]]) -> str:
    if not pairs:
        return "(No relevant resolved tickets found in knowledge base.)"
    blocks = []
    for rank, (ticket, sim) in enumerate(pairs, 1):
        customer_msgs = [m for m in ticket.messages if m.role in ("customer", "user")]
        agent_msgs = [m for m in ticket.messages if m.role in ("agent", "support", "agent_bot")]
        first_customer = customer_msgs[0].content[:350] if customer_msgs else "(none)"
        last_agent = agent_msgs[-1].content[:200] if agent_msgs else "(none)"
        resolution = ticket.resolution_notes or "(see agent messages above)"
        blocks.append(
            f"--- Resolved Ticket #{rank} (similarity {sim:.2f}) ---\n"
            f"ID: {ticket.id}\n"
            f"Category: {ticket.category} / {ticket.issue_type} | Product: {ticket.product}\n"
            f"Customer: {first_customer}\n"
            f"Agent final: {last_agent}\n"
            f"Resolution: {resolution[:400]}\n"
            f"Resolution type: {ticket.resolution_type or 'N/A'}"
        )
    return "\n\n".join(blocks)


def _format_open_ticket(ticket: Ticket) -> str:
    customer_msgs = [m for m in ticket.messages if m.role in ("customer", "user")]
    lines = [
        f"Ticket ID: {ticket.id}",
        f"Category: {ticket.category} / {ticket.issue_type}",
        f"Product: {ticket.product} | Priority: {ticket.priority} | Status: {ticket.status}",
        "Customer messages:",
    ]
    for i, m in enumerate(customer_msgs[:6], 1):
        lines.append(f"  [{i}] {m.content[:400]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chatbot session
# ---------------------------------------------------------------------------

class CXChatbot:
    """Stateful RAG chatbot session for one support agent conversation."""

    def __init__(
        self,
        collection: chromadb.Collection,
        all_tickets: list[Ticket],
    ) -> None:
        self._collection = collection
        self._all_tickets = all_tickets
        self._history: list[dict[str, str]] = []

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        # Sliding window: keep last MAX_HISTORY_TURNS pairs
        trimmed_history = self._history[-(MAX_HISTORY_TURNS * 2):]
        return [{"role": "system", "content": _SYSTEM_PROMPT}] + trimmed_history + [
            {"role": "user", "content": user_content}
        ]

    def chat(self, query: str, open_ticket: Ticket | None = None) -> str:
        """Single turn. Streams to stdout, returns full response string."""
        # Build retrieval query: ticket text + agent query
        retrieval_query = query
        if open_ticket:
            customer_msgs = [m for m in open_ticket.messages if m.role in ("customer", "user")]
            retrieval_query = (
                f"{open_ticket.category} {open_ticket.issue_type} "
                + " ".join(m.content[:200] for m in customer_msgs[:2])
                + f" {query}"
            )

        pairs = retrieve(retrieval_query, self._collection, self._all_tickets)

        # Compose the user turn: open ticket block + knowledge base + agent question
        context_parts: list[str] = []
        if open_ticket:
            context_parts.append(f"OPEN TICKET:\n{_format_open_ticket(open_ticket)}")
        context_parts.append(f"KNOWLEDGE BASE (top {len(pairs)} resolved tickets):\n{_format_retrieved(pairs)}")
        if open_ticket:
            context_parts.append(f"AGENT QUESTION:\n{query}")
        else:
            context_parts.append(f"QUESTION:\n{query}")

        user_content = "\n\n".join(context_parts)
        messages = self._build_messages(user_content)

        # Stream response
        full_response = ""
        for chunk in groq_stream(messages):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()  # newline after streaming completes

        # Update history with compact versions (save context budget)
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": full_response})

        return full_response

    def reset(self) -> None:
        self._history.clear()
