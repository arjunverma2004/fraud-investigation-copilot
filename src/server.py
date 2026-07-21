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

CONCURRENCY = 4  # tune down if you hit Gemini free-tier rate limits
executor = ThreadPoolExecutor(max_workers=CONCURRENCY)
_stream_lock = threading.Lock()
_streaming = {"active": False}



def run_one_transaction(txn: dict):
    insert_transaction(txn)  # <-- create the row FIRST, so later UPDATEs have something to match

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
    config = {"configurable": {"thread_id": txn["transaction_id"]}}
    try:
        graph.invoke(initial_state, config=config)
    except Exception as e:
        print(f"[stream] transaction {txn['transaction_id']} failed: {e}")


def stream_csv(path: str, delay_seconds: float, limit: int | None):
    """Simulates real-time transaction arrival, submitting each row to
    the thread pool and moving on immediately."""
    lazy_df = pl.scan_csv(path)
    if limit:
        lazy_df = lazy_df.limit(limit)

    # Note: State setting (_streaming["active"] = True) was removed from here 
    # and moved to the start_stream endpoint to fix the race condition.

    # FIX: Use iter_dicts() so row can be unpacked as a dictionary
    for row in lazy_df.collect().iter_dicts():
        if not _streaming["active"]:
            break
        txn = {**row, "transaction_id": str(uuid.uuid4())}
        executor.submit(run_one_transaction, txn)
        time.sleep(delay_seconds)

    with _stream_lock:
        _streaming["active"] = False


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


@app.post("/stream/start")
def start_stream(req: StreamRequest):
    # FIX: Wrap the check and assignment in the thread lock
    with _stream_lock:
        if _streaming["active"]:
            return {"status": "already_running"}
        
        # FIX: Synchronously set the active flag to True before thread creation
        _streaming["active"] = True

    threading.Thread(
        target=stream_csv,
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
    except Exception as e:
        print(f"[decision] {transaction_id} FAILED to resume: {e}")


@app.post("/investigations/{transaction_id}/decision")
def submit_decision(transaction_id: str, req: DecisionRequest):
    executor.submit(run_resume, transaction_id, req.decision, req.notes)
    return {"status": "submitted"}