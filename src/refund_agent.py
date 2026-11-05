
"""Part D: database-backed refund tools and resumable ReAct agent."""

from pathlib import Path
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import mock_llm
from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"

rag = PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            """SELECT order_id, status, order_amount, delivery_minutes
               FROM orders WHERE order_id = ?""",
            (order_id,)
        ).fetchone()

    if not row:
        return {"error": "order not found"}

    return dict(zip(
        ["order_id", "status", "order_amount", "delivery_minutes"],
        row
    ))


def search_policy(query):
    return {"results": rag.retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes, status="completed"):
    if status != "completed":
        return {"refund_inr": 0.0, "reason": "Order is not completed"}

    amount = (
        float(order_amount)
        if delivery_minutes is not None and float(delivery_minutes) > 45
        else 0.0
    )

    return {
        "refund_inr": round(amount, 2),
        "approval_required": amount > 1000
    }


def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(exist_ok=True)
    path = OUT / "refund_ledger.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    if float(refund_inr) > 1000:
        return {
            "status": "ESCALATE",
            "message": "Human support lead approval required"
        }

    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    result = _ledger("escalate_to_human", order_id, refund_inr)
    return {
        "status": "ESCALATE",
        "refund_inr": result["refund_inr"],
        "message": "Human support lead approval required"
    }


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": name,
        "description": f"Execute the {name} refund workflow operation.",
        "parameters": {"type": "object", "properties": {}}
    }
    for name in TOOL_MAP
]


def _checkpoint(run_id, messages, tokens, steps, executed, llm_calls,
                scenario_id):
    STATE.mkdir(exist_ok=True)

    obj = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens,
        "steps": steps,
        "tools_executed": executed,
        "llm_calls": llm_calls
    }

    path = STATE / f"{run_id}.json"
    path.write_text(json.dumps(obj, default=str), encoding="utf-8")


def _continue(run_id, scenario_id, messages, tokens, steps, executed,
              llm_calls, max_iterations, token_limit,
              time_limit_seconds, crash_after=None):

    start = time.monotonic()

    for iteration in range(max_iterations):
        if time.monotonic() - start >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: hand off to a human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        response = mock_llm.chat(messages)
        llm_calls += 1
        tokens += response.usage.total_tokens

        if tokens > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: hand off to a human agent.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            return {
                "status": content.split(":", 1)[0],
                "final_answer": content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        tc = msg.tool_calls[0]
        args = dict(tc.arguments)

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": args
            }]
        })

        # Pass the verified database status into refund calculation.
        if tc.name == "calculate_refund":
            order = next(
                (
                    step["observation"]
                    for step in reversed(steps)
                    if step["tool"] == "get_order"
                ),
                {}
            )
            args["status"] = order.get("status", "unknown")

        observation = TOOL_MAP[tc.name](**args)
        executed.append(tc.name)

        steps.append({
            "tool": tc.name,
            "arguments": args,
            "observation": observation
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(observation)
        })

        # Save after every tool observation.
        _checkpoint(
            run_id, messages, tokens, steps, executed,
            llm_calls, scenario_id
        )

        if crash_after is not None and len(executed) == crash_after:
            raise RuntimeError("Simulated crash after checkpoint")

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: hand off to a human agent.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens
    }


def run_agent(user_message, scenario_id="demo", max_iterations=6,
              token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after=None):

    run_id = run_id or str(uuid.uuid4())

    messages = [
        {
            "role": "system",
            "content": "AGENT: Use tools to verify orders and follow refund policy."
        },
        {"role": "user", "content": user_message}
    ]

    return _continue(
        run_id, scenario_id, messages, 0, [], [], 0,
        max_iterations, token_limit, time_limit_seconds, crash_after
    )


def resume_agent(run_id, max_iterations=6, token_limit=6000,
                 time_limit_seconds=10):

    path = STATE / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for {run_id}")

    saved = json.loads(path.read_text(encoding="utf-8"))

    return _continue(
        run_id,
        saved["scenario_id"],
        saved["messages"],
        saved["tokens_used"],
        saved["steps"],
        saved["tools_executed"],
        saved["llm_calls"],
        max_iterations,
        token_limit,
        time_limit_seconds
    )
