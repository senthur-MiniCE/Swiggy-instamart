from pathlib import Path
import pandas as pd
from src.policy_rag import PolicyRAG, evaluate

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "policy_eval.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

questions = pd.read_csv(RAW, keep_default_na=False)

for overlap in (10, 0):
    print(f"\n{'=' * 12} OVERLAP = {overlap} {'=' * 12}")

    rag = PolicyRAG(chunk_size=40, overlap=overlap)
    results = evaluate(rag)

    # Attach expected answers so failures can be identified.
    results["expected_doc"] = questions["expected_doc"]
    results["expected_fact"] = questions["expected_fact"]

    answerable = results["expected_doc"] != "NONE"
    unanswerable = ~answerable

    retrieval_hits = int(results.loc[answerable, "retrieved_doc"].sum())
    fact_hits = int(results.loc[answerable, "fact_found"].sum())
    fallback_hits = int(results.loc[unanswerable, "fallback"].sum())

    print("Total chunks:", len(rag.chunks))
    print("Chunks by document:")
    print(pd.Series([c["source"] for c in rag.chunks]).value_counts().to_string())

    print(f"Retrieval: {retrieval_hits}/12")
    print(f"Expected facts found: {fact_hits}/12")
    print(f"Unanswerable fallback: {fallback_hits}/2")

    results.to_csv(OUT / f"policy_eval_overlap{overlap}.csv", index=False)

    failures = results[
        answerable & (~results["retrieved_doc"] | ~results["fact_found"])
    ]

    if failures.empty:
        print("Failed answerable questions: none")
    else:
        print("\nFailed answerable questions:")
        for _, row in failures.iterrows():
            print("Question:", row["question"])
            print("Expected document:", row["expected_doc"])
            print("Expected fact:", row["expected_fact"])
            print(
                "Retrieval hit:", row["retrieved_doc"],
                "| Fact found:", row["fact_found"]
            )

print("\nSaved both evaluation CSV files in outputs/.")