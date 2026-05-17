import json
import re
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import RESULT_CSV, STATE_DIR, RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

SAFE_SESSION_RE = re.compile(r"[^a-zA-Z0-9_-]")

def clean_session_id(session_id: str) -> str:
    return SAFE_SESSION_RE.sub("_", session_id)


def get_next_turn_id(session_id: str) -> int:
    session_id = clean_session_id(session_id)
    files = list(STATE_DIR.glob(f"{session_id}_*.json"))
    return len(files) + 1


def state_path(session_id: str, turn_id: int) -> Path:
    session_id = clean_session_id(session_id)
    return STATE_DIR / f"{session_id}_{turn_id:04d}.json"


def save_result(
    session_id: str,
    dataset_name: str,
    question: str,
    generated_code: str,
    final_answer: str,
    execution_result: pd.DataFrame,
    visualization: Optional[str],
    visualization_decision: Optional[dict],
) -> dict:
    turn_id = get_next_turn_id(session_id)

    state = {
        "session_id": session_id,
        "turn_id": turn_id,
        "dataset_name": dataset_name,
        "question": question,
        "generated_code": generated_code,
        "final_answer": final_answer,
        "result_table": execution_result.to_dict(orient="records"),
        "columns": list(execution_result.columns),
        "visualization": visualization,
        "visualization_decision": visualization_decision,
    }

    with open(state_path(session_id, turn_id), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    row = pd.DataFrame([
        {
            "session_id": session_id,
            "turn_id": turn_id,
            "dataset_name": dataset_name,
            "question": question,
            "generated_code": generated_code,
            "final_answer": final_answer,
        }
    ])

    if RESULT_CSV.exists():
        old = pd.read_csv(RESULT_CSV)
        new = pd.concat([old, row], ignore_index=True)
    else:
        new = row

    new.to_csv(RESULT_CSV, index=False)
    return state


def get_previous_result(session_id: Optional[str]) -> Optional[dict]:
    if not session_id:
        return None
    session_id = clean_session_id(session_id)
    files = sorted(STATE_DIR.glob(f"{session_id}_*.json"))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_session_states(session_id: str) -> list[dict]:
    session_id = clean_session_id(session_id)
    states = []
    for path in sorted(STATE_DIR.glob(f"{session_id}_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            states.append(json.load(f))
    return states
