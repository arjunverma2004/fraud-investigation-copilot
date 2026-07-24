"""
FastAPI backend: owns the concurrent transaction stream and exposes
endpoints the Streamlit UI polls.
"""

import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import polars as pl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command
from history_store import init_db, insert_transaction, get_recent_transactions, get_pending_investigations
from graph import graph


CONCURRENCY = 1  # tune down if you hit Gemini free-tier rate limits
executor = ThreadPoolExecutor(max_workers=CONCURRENCY)
_stream_lock = threading.Lock()
_streaming = {"active": False}



def run_one_transaction(txn: dict):
    txn_id = txn.get("transaction_id", "unknown")
    try:
        insert_transaction(txn) 

        initial_state = {
            "transaction": txn,
            "fraud_score": {},
            "account_checks": [],
            "retrieved_patterns": [],
            "account_feedback_history": [],
            "investigation_report": "",
            "recommendation": None,
            "human_decision": None,
            "human_notes": None,
            "messages": [],
            "iteration_count": 0,
        }
        config = {"configurable": {"thread_id": txn_id}}
        result = graph.invoke(initial_state, config=config)

        if result.get("fraud_score", {}).get("is_flagged"):
            if result.get("investigation_report"):
                print(f"[stream] {txn_id}: investigated -> recommendation={result.get('recommendation')}")
            else:
                print(f"[stream] {txn_id}: flagged, investigation did not complete (no report set)")
        else:
            print(f"[stream] {txn_id}: scored clean, not flagged")
    except Exception:
        import traceback
        print(f"[stream] transaction {txn_id} FAILED:")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class StreamRequest(BaseModel):
    csv_path: str = str(Path(__file__).parent / "final_df.csv")
    delay_seconds: float = 0.5
    limit: int | None = None


def _stream_worker(csv_path: str, delay_seconds: float, limit: int | None):
    """Runs on a background thread so the /stream/start request returns
    immediately instead of blocking for the whole batch."""
    try:
        lazy_df = pl.scan_csv(csv_path)
        if limit:
            lazy_df = lazy_df.limit(limit)

        for row in lazy_df.collect().iter_rows(named=True):
            if not _streaming["active"]:
                break
            txn = {**row, "transaction_id": str(uuid.uuid4())}
            executor.submit(run_one_transaction, txn)
            time.sleep(delay_seconds)
    except Exception:
        import traceback
        print("[stream_worker] FAILED:")
        traceback.print_exc()
    finally:
        with _stream_lock:
            _streaming["active"] = False


@app.post("/stream/start")
def start_stream(req: StreamRequest):
    with _stream_lock:
        if _streaming["active"]:
            return {"status": "already_running"}
        _streaming["active"] = True

    threading.Thread(
        target=_stream_worker,
        args=(req.csv_path, req.delay_seconds, req.limit),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.post("/stream/stop")
def stop_stream():
    with _stream_lock:
        _streaming["active"] = False
    return {"status": "stopping"}


@app.get("/stream/status")
def stream_status():
    return {"active": _streaming["active"]}


@app.get("/dashboard/transactions")
def dashboard_transactions(limit: int = 200):
    return get_recent_transactions(limit=limit)


@app.get("/investigations/pending")
def pending_investigations():
    return get_pending_investigations()


@app.get("/investigations/{transaction_id}")
def get_investigation_state(transaction_id: str):
    config = {"configurable": {"thread_id": transaction_id}}
    snapshot = graph.get_state(config)

    return {
        "values": snapshot.values,
        "next": snapshot.next,
        "is_interrupted": bool(snapshot.next),
    }


class DecisionRequest(BaseModel):
    decision: str  # approve | reject | need_more_info
    notes: str = ""


def run_resume(transaction_id: str, decision: str, notes: str):
    config = {"configurable": {"thread_id": transaction_id}}
    try:
        graph.invoke(Command(resume={"decision": decision, "notes": notes}), config=config)
        print(f"[decision] {transaction_id} resumed successfully with decision={decision}")
    except Exception:
        import traceback
        print(f"[decision] {transaction_id} FAILED to resume:")
        traceback.print_exc()


@app.post("/investigations/{transaction_id}/decision")
def submit_decision(transaction_id: str, req: DecisionRequest):
    executor.submit(run_resume, transaction_id, req.decision, req.notes)
    return {"status": "submitted"}