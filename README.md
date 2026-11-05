# Customer Intelligence & Support Copilot — Teaching Kit

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or network call is needed.

## Learning outcomes
Students learn to (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.

## Setup and run
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run cells from top to bottom. The notebook calls the same modules used by the scripts.

## Repository map
- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers and agent tool calls.
- `src/data_pipeline.py`: audit, clean, SQLite load, validation.
- `src/customer_models.py`: RFM, K-means, PCA, lapse model with train/test separation.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, routing.
- `src/policy_rag.py`: chunking, TF-IDF retrieval and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger and bounded tool loop.
- `sql/queries.sql`: seven SQL learning patterns.
- `outputs/`: generated artifacts; `state/`: agent checkpoint examples.

## How students should learn the data-fetching flow
1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

## Part A — Data foundation
The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing and imputes only completed-order delivery durations by product category. Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

## Part B — Segments and lapse
RFM uses completed orders and a fixed snapshot date. K-means diagnostics compare K=2..8; the notebook starts with K=4 for interpretable teaching. Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes. The pipeline uses a preprocessing Pipeline fitted on training rows, GridSearchCV and a saved joblib model. The Premium retention list is illustrative only because the Premium label uses the whole RFM window and is post-hoc; it is not a leakage-free evaluation.

## Part C — Ticket triage and privacy
Empty and overlong tickets are skipped before model calls. Email, phone and order ID patterns are redacted before prompting. Retries handle simulated transient errors. JSON category values are allow-listed; malformed outputs are accepted only when exactly one allowed category is detectable using `re.findall`. Human review receives guardrail/invalid/budget cases, fraud goes to security, premium customers go to the premium desk, and the rest go to the support queue.

## Part D — Policy retrieval and refund tools
Policy documents are chunked with overlap, represented with `TfidfVectorizer`, and ranked using cosine similarity. The offline mock returns context-derived sentences. Refund rules are implemented in Python tools backed by SQLite: completed orders delivered in over 45 minutes qualify for a full refund; orders over INR 1000 escalate instead of auto-issuing. This is a teaching simulation, not a payment integration.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

## Limitations and next steps
- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric; compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks and human approval workflows.
## Part D — Policy RAG Assistant and Refund Agent

### Policy document chunking and retrieval

The five policy documents were split into 40-word chunks with an overlap of 10 words. This produced 15 chunks:

| Policy document | Chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

Retrieval uses `TfidfVectorizer()` and cosine similarity to rank the most relevant chunks. Term Frequency (TF) measures how often a term appears in a document, while Inverse Document Frequency (IDF) gives less weight to terms that appear in many documents. A limitation of TF-IDF is that it relies on word overlap, so different phrasings with the same meaning may not retrieve the same information.

### Retrieval and grounding evaluation

The evaluation used all 14 questions in `data/raw/policy_eval.csv`.

| Evaluation metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Total chunks | 15 | 13 |
| Expected document retrieved (answerable questions) | 12/12 | 12/12 |
| Expected fact found in answer | 12/12 | 10/12 |
| Fallback for unanswerable questions | 2/2 | 2/2 |

With zero overlap, two expected facts were not found:

- **Refund eligibility:** the phrase "more than 45 minutes" was split across a chunk boundary.
- **Failed payment refund:** the phrase "48 hours" was split across a chunk boundary.

This comparison demonstrates how chunk overlap can preserve information that would otherwise be separated across chunks.

Evaluation outputs:
- `outputs/policy_eval_overlap10.csv`
- `outputs/policy_eval_overlap0.csv`

### Refund agent scenarios

The agent uses database lookup, policy retrieval, refund calculation, and either refund issuance or human escalation.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 859 |
| S2 | REFUND | 135.00 | 5 | 859 |
| S3 | ESCALATE | 1999.00 | 5 | 876 |
| S4 | DENIED | 0.00 | 4 | 583 |
| S5 | DENIED | 0.00 | 3 | 322 |
| S6 | NOT_FOUND | 0.00 | 2 | 70 |
| S7 | FALLBACK_MAX_ITERATIONS | — | 6 | 2988 |

### Safety and recovery demonstrations

| Demonstration | Result |
|---|---|
| S1 with token limit 200 | FALLBACK_TOKEN_LIMIT |
| S1 with time limit 0 seconds | FALLBACK_TIMEOUT |
| S7 with default token limit 6000 | FALLBACK_MAX_ITERATIONS |

The S1 crash-and-resume test successfully recovered from a simulated crash after the second tool execution. The exact tool sequence across the original run and resume was:

`get_order → search_policy → calculate_refund → issue_refund`

The final result was REFUND for INR 128.25. The refund ledger contains one issuance entry for S1, confirming that the refund was not repeated during recovery.

### ReAct versus plan-and-execute

The ReAct design makes decisions after observing each tool result. In this implementation, S1 required five model calls, S5 required three, and S6 required two. This uses more model calls and tokens than a plan-and-execute approach that requests a complete plan once and replans only when necessary.

However, ReAct can respond to unexpected observations such as a missing order or a cancelled order before attempting the next action. A fixed plan may need additional replanning and stronger safeguards when tool results differ from expectations. For this refund workflow, I would choose ReAct because refund issuance and human escalation require decisions based on verified order data and policy results.

### Outputs

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `outputs/policy_eval_overlap10.csv`
- `outputs/policy_eval_overlap0.csv`
- `state/` — saved agent checkpoints
