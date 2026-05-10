# Eight Sleep CX Intelligence — Part 1 Findings

## Method Summary

10,000 tickets (Feb 24–26 2026) processed through:
**Embeddings** (all-mpnet-base-v2, 768D, local) → **PCA** (100D, 98% variance) → **UMAP** (15D) → **HDBSCAN** (min_cluster_size=250) → **LLM labeling** (Groq Llama 3.3-70b)

Result: **11 semantic clusters**, 40.8% noise (reassigned to nearest centroid). Noise is concentrated in software/onboarding tickets where customers describe the same issue in highly variable language; physical hardware issues cluster tightly.

---

## Discovered Issue Clusters

| # | Cluster Label | Core Problem | Total | Resolved | High/Critical |
|---|---|---|---|---|---|
| 1 | **Dual Zone Setup Failure** | Pod dual-zone temperature config not saving or recognizing second user | 2,185 | 47% | 7% |
| 2 | **App Data Display Failure** | Sleep scores, temperature history, and biometrics missing or incorrect in app | 1,660 | 61% | 4% |
| 3 | **WiFi Connectivity Failure** | Hub fails to connect/reconnect after WiFi password change or initial setup | 1,457 | 58% | 0% |
| 4 | **Temperature Control Reset Issue** | Pod resets to default temperature mid-sleep, ignoring schedule | 1,036 | 55% | 1% |
| 5 | **Pump Grinding Noise Issue** | Progressive loud grinding/clicking noise from pump during operation | 758 | 55% | 3% |
| 6 | **Hub Water Leakage Issue** | Water leaking from hub connections, causing puddles and moisture damage | 734 | 53% | 81% |
| 7 | **Pod Hub Water Leak** | Water leaking specifically at hub-to-mattress connector | 587 | 52% | 89% |
| 8 | **Hub Overheating Failure** | Hub extremely hot to touch, burning smell, fire hazard concern, auto-shutoff | 536 | 54% | 74% |
| 9 | **Unexpected Membership Price Increase** | Membership jumped from $19→$29 without notice, customers disputing charge | 491 | 55% | 0% |
| 10 | **Subscription Renewal Billing Error** | Double-charged or charged without renewal notification, requesting refunds | 285 | 53% | 0% |
| 11 | **Hub Detection Failure** | App cannot find hub during onboarding; Bluetooth/discovery issues | 271 | 36% | 0% |

> **Note on clusters 6 & 7 (both "water leak"):** HDBSCAN found two distinct sub-clusters within the leak issue type — one describing leaks at the hub body, another specifically at the hub-to-mattress connection point. This distinction is operationally useful: they may have different root causes and different resolution paths.

---

## Day-by-Day Volume Trends

| Cluster | Day 1 | Day 2 | Δ D1→D2 | Day 3 | Δ D1→D3 |
|---|---|---|---|---|---|
| Dual Zone Setup Failure | 761 | 726 | -5% | 698 | -8% |
| App Data Display Failure | 540 | 473 | -12% | **647** | **+20%** |
| WiFi Connectivity Failure | 436 | **687** | **+58% 🚨** | 334 | -23% |
| Temperature Control Reset Issue | 435 | 290 | -33% | 311 | -29% |
| Pump Grinding Noise Issue | 309 | 230 | -26% | 219 | -29% |
| Hub Water Leakage Issue | 206 | 183 | -11% | **345** | **+67% 🚨** |
| Pod Hub Water Leak | 157 | 140 | -11% | **290** | **+85% 🚨** |
| Hub Overheating Failure | 126 | **292** | **+132% 🚨** | 118 | -6% |
| Unexpected Membership Price Increase | 147 | 117 | -20% | **227** | **+54% 🚨** |
| Subscription Renewal Billing Error | 84 | 84 | 0% | **129** | **+54% 🚨** |
| Hub Detection Failure | 99 | 90 | -9% | 82 | -17% |

**Baseline pattern:** Most clusters show a natural -10% to -30% decline from Day 1 to Day 2 and Day 3. This is the expected baseline drift — anomalies are measured against this.

---

## Anomalies Detected

### Day 2 Anomalies (baseline: Day 1)

**Classification: spikes in existing clusters, no genuinely new cluster appeared on Day 2.** Both anomalous clusters existed on Day 1 at normal volume. The signal is a sharp volume surge, not a novel problem type emerging.

#### 🔴 CRITICAL — Hub Overheating Failure (+132%)
- **126 → 292 tickets**
- Customers reporting hubs too hot to touch, burning smell, and fire hazard concerns. Hub auto-shutting off.
- **Likely cause:** Possible firmware or hardware batch issue shipped to a cohort of customers.
- **Slack alert:** *"We are experiencing a **critical** anomaly with **Hub Overheating Failure** issues, with a 132% increase in tickets. Compared to our Day 1 baseline of 126 tickets, we have seen 292 tickets on Day 2, indicating a significant surge in overheating issues. The on-call CX team should immediately review the sample issues and prepare a response plan to address the fire hazard concerns ⚠️."*

#### 🟠 HIGH — WiFi Connectivity Failure (+58%)
- **436 → 687 tickets**
- Customers unable to connect hub after WiFi changes or during initial setup. Setup hanging at connection step.
- **Likely cause:** App update or ISP-side change affecting hub discovery protocol.
- **Slack alert:** *"We are experiencing a **WiFi Connectivity Failure** anomaly with a 58% increase in tickets. This surge occurred on Day 2 with 687 tickets, compared to the Day 1 baseline of 436 tickets. The on-call CX team should immediately review the sample issues and escalate to engineering for further analysis 🚨."*

---

### Day 2 Anomalies — Status on Day 3

| Anomaly | Day 1 | Day 2 | Day 3 | Status |
|---|---|---|---|---|
| Hub Overheating Failure | 126 | 292 (+132%) | 118 | ✅ **Resolving** — returned to below Day 1 baseline. Likely a discrete batch/firmware incident now contained. |
| WiFi Connectivity Failure | 436 | 687 (+58%) | 334 | ✅ **Resolving** — dropped below Day 1 baseline. Correlated resolution with overheating suggests a shared root cause (possibly the same firmware push). |

Both Day 2 anomalies resolved by Day 3 without intervention escalating further. This pattern — a spike on Day 2 that self-resolves by Day 3 — is consistent with a firmware rollout that was rolled back, or a hardware batch that exhausted the affected customer cohort.

---

### Day 3 Anomalies (baseline: Days 1+2 average)

**Classification: spikes in existing clusters — all four clusters existed across Days 1 and 2. No entirely new cluster emerged on Day 3 either, but the water-leak and billing clusters crossed anomaly thresholds for the first time.**

#### 🟠 HIGH — Pod Hub Water Leak (+95%)
- **148 avg → 290 tickets**
- Customers reporting water pooling from hub-to-mattress connector. Escalating severity.
- **Slack alert:** *"We're experiencing a **high** severity anomaly with **Pod Hub Water Leak** issues, with a 95% increase in tickets to 290 on Day 3. The on-call CX team should immediately review the hardware/leak cases and prepare for a potential influx of high priority tickets 🚨."*

#### 🟠 HIGH — Hub Water Leakage Issue (+77%)
- **194 avg → 345 tickets**
- Water leaks from hub body (distinct from connector leak above). Combined with Pod Hub Water Leak, water-related issues are the dominant Day 3 story.
- **Slack alert:** *"We're experiencing a **high** severity anomaly with **Hub Water Leakage Issue** tickets, with a 77% increase to 345 tickets on Day 3. The on-call CX team should immediately review hardware/leak cases and escalate to engineering for investigation ⚠️."*

#### 🟠 HIGH — Unexpected Membership Price Increase (+72%)
- **132 avg → 227 tickets**
- $19→$29 price increase customers were not notified of. Billing disputes and cancellation requests spiking.
- **Slack alert:** *"We're experiencing a **high severity** anomaly with **Unexpected Membership Price Increase** issues, with a 72% increase in tickets to 227 on Day 3. The on-call CX team should immediately review billing disputes and membership price processes to identify the root cause ⚠️."*

#### 🟠 HIGH — Subscription Renewal Billing Error (+65%)
- **78 avg → 129 tickets**
- Double-charging and silent renewals. Likely related to the price increase story above — billing system changes affecting multiple dimensions simultaneously.
- **Slack alert:** *"We're experiencing a **high severity** anomaly in **Subscription Renewal Billing Error** tickets, with a 65% increase to 129 tickets on Day 3. The on-call CX team should immediately review double billing and subscription renewal without notification cases and escalate to the billing team ⚠️."*

---

## Key Narrative Insights

**Day 2 story — Hardware safety spike:**
Hub Overheating surged +132% (critical). WiFi Connectivity also spiked +58% (possibly linked — overheating may cause WiFi drop). Both resolved by Day 3, suggesting a discrete hardware/firmware incident affecting a specific batch.

**Day 3 story — Two independent crises emerging:**
1. **Water leaks** (both clusters combined: 635 tickets on Day 3 vs ~351 baseline = +81%) — accelerating, not resolved. Physical defect likely in a production batch.
2. **Billing** (both clusters combined: 356 tickets on Day 3 vs ~210 baseline = +70%) — price increase rollout causing dual billing issues + customer disputes simultaneously.

**Stable clusters (no anomaly):**
Dual Zone Setup, Temperature Control, Pump Noise, Hub Detection all trending flat or slightly down across all 3 days — these are chronic baseline issues, not acute incidents.

---

# Eight Sleep CX Intelligence — Part 2 Findings

## Architecture

**Knowledge base:** 5,340 resolved tickets indexed in ChromaDB using `all-mpnet-base-v2` embeddings (768D, L2-normalised). Each ticket stored with its full `to_embedding_text()` representation — category, issue type, customer messages, and resolution notes — so retrieval is sensitive to both problem type and resolution language.

**Retrieval:** On each agent query, the open ticket text + query are embedded with the same model, and top-5 resolved tickets are retrieved by cosine similarity. Using the same model for indexing and querying ensures no embedding mismatch. Similarity scores are surfaced to the LLM so it can weight higher-confidence matches more heavily.

**Generation:** Groq Llama 3.3-70b with a structured system prompt that enforces: (1) cite ticket IDs for every step, (2) use separate sections for each issue, (3) flag when no match exists rather than guessing, (4) include specific escalation triggers.

---

## Resolution Paths Extracted from the Knowledge Base

Three representative cases tested, each against 5,340 resolved tickets:

### Hub Water Leak (`conv_7171e12f` — high priority, Pod 3)
*Customer: "Moisture around base of hub. Seems to be slowly leaking."*

Retrieved source tickets: `conv_60606db4`, `conv_a9ad2127`, `conv_d0463ed1`, `conv_144f3c78`, `conv_a17ad642`

**Extracted resolution path:**
1. Re-seat the tube connector at the hub fitting (small leak at tube fitting — most common cause)
2. Monitor for 48 hours after adjustment
3. If persists → replacement unit (seal failure pattern, matches `conv_144f3c78`)

**Escalation trigger:** Leak continues after 48h monitoring → specialist team, mark urgent. Customer has photographic evidence → include in escalation notes.

---

### Hub Overheating (`conv_e71c29aa` — high priority, Pod 4 Ultra)
*Customer: lengthy message describing weeks of overheating, burning smell, safety concern.*

Retrieved source tickets: `conv_bc7ec800`, `conv_f4e569db`, `conv_c3d4aa69`, `conv_429b0d2f`, `conv_ca43bb84`

**Extracted resolution path:**
1. **Immediately instruct customer to unplug** (safety-first pattern consistent across all source tickets)
2. Document issue and summarize steps taken so far
3. Escalate to hardware safety team, mark critical
4. Arrange expedited replacement unit (every resolved overheating ticket resolved this way — no repair path exists)

**Escalation trigger:** No "try this first" step — the knowledge base showed 100% replacement rate for this issue type. The LLM correctly surfaced this: *"No alternative path is suggested, as all relevant tickets point towards an expedited replacement."*

---

### Double Billing Dispute (`conv_74966956` — medium priority, Pod 4 Ultra)
*Customer: "Double charged this month. See two charges of $24."*

Retrieved source tickets: `conv_38b543bf`, `conv_c9ac0f9f`, `conv_ad87d97a`, `conv_125f8af7`, `conv_146cee5b`

**Extracted resolution path:**
1. Confirm as payment processing error (not a subscription issue)
2. Process refund — 3–5 business day timeline (specific figure from resolved tickets, not invented)
3. Offer $20 account credit for inconvenience (pattern present in `conv_38b543bf`, `conv_125f8af7`)

**Escalation trigger:** Refund not processed after 5 business days → Tier-2 billing specialist.

---

## How Knowledge Was Extracted and Tested

**Extraction:** No manual curation. Resolution paths emerge from retrieval — the LLM reads the resolution notes and agent final messages of the top-5 similar resolved tickets and synthesises a path. The system prompt constrains it to only recommend steps visible in those tickets.

**Correctness testing approach:**
- **Citation grounding:** Every step in the LLM output must cite a ticket ID. If a step has no citation it either hallucinates (bad) or flags no match (correct). We verified citations are real ticket IDs present in the knowledge base.
- **Retrieval quality:** Cosine similarity scores for all three demos were 0.75–0.92, indicating strong semantic matches — not borderline retrievals.
- **Edge case: multilingual ticket** (`conv_19b2d8e5` — Spanish/English mix): the embedding model handles multilingual text natively, retrieving semantically correct English-language resolved tickets despite the mixed-language query. Resolution path was still grounded and accurate.
- **Edge case: long ticket** (`conv_e71c29aa` — verbose customer message): truncation in `to_embedding_text()` (first 600 chars of first customer message) captures the core complaint without being overwhelmed by tangential detail.

---

## Extending to Tool Calling

The current system is read-only: it retrieves and summarises. With tool calling, the agent could act:

| Tool | What it does | When triggered |
|---|---|---|
| `lookup_order_status(customer_id)` | Fetch replacement/refund status from order system | Agent asks "has the replacement shipped?" |
| `issue_refund(ticket_id, amount)` | Initiate refund directly in billing system | After confirming double-charge |
| `create_replacement_order(ticket_id, priority)` | Trigger hardware replacement workflow | Overheating or seal failure confirmed |
| `escalate_ticket(ticket_id, team, reason)` | Route to Tier-2 / safety team with context | Escalation trigger conditions met |
| `search_knowledge_base(query, n)` | Explicit retrieval call the LLM can invoke itself | Multi-issue ticket needing separate lookups |

The chatbot already structures its output with "Escalate When" triggers — these map cleanly onto `escalate_ticket()` calls. The LLM would call tools when it recognises the trigger pattern in conversation, rather than just describing what a human agent should do.

**What to watch out for:**

1. **Irreversible actions need confirmation gates.** Issuing a refund or creating a replacement order cannot be undone cheaply. The system should require explicit agent confirmation before executing write tools — never execute autonomously based on LLM inference alone.

2. **Hallucinated tool arguments.** LLMs can fabricate `ticket_id` or `amount` values. Tool schemas should validate against the current conversation context (e.g., `amount` must match what the customer stated, not what the LLM inferred).

3. **Knowledge base staleness.** Resolution paths are snapshots. If a firmware fix changes the recommended steps for WiFi issues, old resolved tickets give wrong advice until the index is rebuilt. A nightly rebuild + version-stamping of retrieval results mitigates this.

4. **Multi-turn state and tool interleaving.** If the LLM calls `lookup_order_status` mid-conversation, the result needs to flow back into context correctly. Conversation history management must include tool results, not just chat turns — otherwise the LLM loses track of what it already looked up.

5. **Authorisation boundaries.** Not all agents should trigger all tools. A Tier-1 agent should be able to `issue_refund` up to $50 but `create_replacement_order` should require Tier-2 approval. Tool permissions need to be scoped per agent role, enforced server-side — the LLM cannot be trusted to self-police.

6. **Grounding degrades with multi-hop reasoning.** The current RAG works because resolution paths are in one place (the resolved ticket). Tool calling adds intermediate steps — the LLM may reason over tool outputs in ways that compound errors. Logging every tool call with its inputs and outputs is essential for debugging and auditing.
