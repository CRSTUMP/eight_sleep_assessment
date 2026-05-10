# Eight Sleep CX Intelligence Assessment

Two-part AI system: (1) ticket clustering + anomaly detection, (2) RAG chatbot for resolution guidance.

## Output
My long form notes are stored in INSIGHTS.md
The final write up is in final.docx


## Setup

### 1. API Keys
Copy `.env.example` to `.env` and fill in your keys:
```
GEMINI_API_KEY=...   # Google AI Studio — aistudio.google.com (free)
GROQ_API_KEY=...     # Groq console — console.groq.com (free)
```

### 2. Install dependencies

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows** — `sentence-transformers` requires long path support. Two options:

*Option A — Enable long paths (recommended, one-time):*
```
# Run as Administrator in PowerShell:
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
# Then restart terminal, then:
pip install -r requirements.txt
```

*Option B — Short-path venv (no admin needed):*
```powershell
python -m venv C:\ev
C:\ev\Scripts\pip install -r requirements.txt
# Then run scripts with C:\ev\Scripts\python instead of python
# (the fallback is automatic — standard `python` also works after this)
```

### 3. Data
Place the ticket data file in the project root. The default expected filename is `sample_tickets_v6.json_` — if yours differs, update `data_path` in `shared/config.py`.

---

## Part 1 — Clustering & Anomaly Detection

```bash
python run_part1.py
```

**Options:**
```
--force-recompute   Ignore embedding cache, recompute from scratch
--output PATH       JSON results file (default: results_part1.json)
```

**What it does:**
1. Loads 10,000 tickets
2. Embeds with `all-mpnet-base-v2` (local, cached to `.cache/` after first run — ~3 min first time)
3. Reduces dimensions: PCA (100D) → UMAP (15D)
4. Clusters with HDBSCAN (min_cluster_size=250 → ~11 clusters)
5. Labels each cluster with Groq Llama 3.3-70b
6. Detects Day 2 anomalies (vs Day 1 baseline) and Day 3 anomalies (vs Day 1+2 average)
7. Generates Slack-ready alerts for each anomaly
8. Prints a Rich table + anomaly panels, saves `results_part1.json`

**Runtime:** ~3 min first run (local embedding). Subsequent runs: ~1 min (cached).

---

## Part 2 — Ticket Resolution Chatbot

Run Part 1 first (builds the embedding cache). Then:

```bash
# First run — builds ChromaDB index of resolved tickets
python run_part2.py --rebuild

# Subsequent runs — loads existing index
python run_part2.py

# Anchor to a specific open ticket
python run_part2.py --ticket TICKET_ID_HERE
```

**Commands in chat:**
- `reset` — clear conversation history
- `exit` / `quit` — end session

**Example:**
```
Agent: My customer's hub is very hot to touch and smells like burning
Assistant: [streams resolution path grounded in similar resolved tickets, cites ticket IDs]

Agent: They already unplugged it — what's next?
Assistant: [follow-up steps from knowledge base]
```

---

## Architecture

```
sample_tickets_v6.json_
       │
       ▼
shared/embedder.py        ← all-mpnet-base-v2 (local, 768D, cached)
       │
       ├──► part1/clustering.py  ← PCA → UMAP → HDBSCAN → Groq labels
       │           │
       │    part1/anomaly.py     ← day-over-day spike detection + Groq alerts
       │
       └──► part2/knowledge_base.py  ← ChromaDB (5,340 resolved tickets)
                                            │
                                     part2/chatbot.py  ← Groq Llama 3.3-70b (streaming)
```

| Decision | Rationale |
|---|---|
| Local embeddings (`all-mpnet-base-v2`) | No API rate limits; consistent task type for index + query |
| PCA → UMAP → HDBSCAN | PCA speeds UMAP; HDBSCAN handles variable density, no k required |
| Groq Llama 3.3-70b for generation | 30 RPM free tier; 128k context; fast streaming |
| ChromaDB persistent | No server process; reloads in ~1s; cosine similarity via hnsw:space |
| Noise reassignment | Every ticket in a cluster for accurate day-over-day counting |
| Sliding window history (6 turns) | Keeps context budget tight while preserving conversational coherence |

## Results

See `INSIGHTS.md` for:
- All 11 discovered clusters with labels, descriptions, and resolution/priority rates
- Day-by-day volume trends
- All 6 detected anomalies with Slack alerts
- Part 2 resolution paths, RAG design, and tool-calling extension discussion
