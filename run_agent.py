
"""Run and evaluate all Part D refund-agent scenarios."""

import json

from src.refund_agent import run_agent, resume_agent

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

LEDGER = OUT / "refund_ledger.jsonl"
RUNS = OUT / "agent_runs.jsonl"

# Start with clean demonstration output files.
for path in [LEDGER, RUNS]:
    if path.exists():
        path.unlink()

scenarios = [
    ("S1", "Please refund order ORD-100048."),
    ("S2", "Please refund order ORD-100067."),
    ("S3", "Please refund order ORD-100145."),
    ("S4", "Please refund order ORD-100001."),
    ("S5", "Please refund order ORD-100020."),
    ("S6", "Please refund order ORD-999999."),
    ("S7", "Please process [LOOP].")
]

results = []

for scenario_id, message in scenarios:
    print(f"\nRunning {scenario_id}...")

    if scenario_id == "S1":
        # Simulate a crash after the second tool observation.
        run_id = "crash-resume-S1"

        try:
            run_agent(
                message,
                scenario_id=scenario_id,
                run_id=run_id,
                crash_after=2
            )
        except RuntimeError:
            print("Simulated crash. Resuming from checkpoint...")

        result = resume_agent(run_id)
        result["crash_recovery"] = True

    else:
        result = run_agent(
            message,
            scenario_id=scenario_id
        )

    result["scenario_id"] = scenario_id
    results.append(result)

    print("Status:", result["status"])
    print("Model calls:", result["llm_calls"])
    print("Tokens:", result["tokens_used"])

# Token-limit demonstration.
print("\nRunning token-limit demonstration...")
token_demo = run_agent(
    "Please refund order ORD-100048.",
    scenario_id="TOKEN_DEMO",
    token_limit=200
)
print("Token demo:", token_demo["status"])

# Timeout demonstration.
print("\nRunning timeout demonstration...")
timeout_demo = run_agent(
    "Please refund order ORD-100048.",
    scenario_id="TIMEOUT_DEMO",
    time_limit_seconds=0
)
print("Timeout demo:", timeout_demo["status"])

# Save scenario results.
with RUNS.open("w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

    f.write(json.dumps({
        "scenario_id": "TOKEN_DEMO",
        **token_demo
    }) + "\n")

    f.write(json.dumps({
        "scenario_id": "TIMEOUT_DEMO",
        **timeout_demo
    }) + "\n")

print("\n========== PART D SUMMARY ==========")

for result in results:
    print(
        result["scenario_id"],
        result["status"],
        "calls =", result["llm_calls"],
        "tokens =", result["tokens_used"]
    )

print("\nToken demo:", token_demo["status"])
print("Timeout demo:", timeout_demo["status"])

print("\nSaved:", RUNS)
print("Saved:", LEDGER)
print("Checkpoints:", ROOT / "state")
