from __future__ import annotations

import uuid
import pandas as pd

from .config import DATASET_PATH, DATASET_NAME
from .agent import CodeExecutionAgent
from .result_store import save_result, get_previous_result
from .visualizer import create_visualization


def _is_follow_up_question(question: str) -> bool:
    q = question.lower().strip()

    follow_up_phrases = [
        "now",
        "previous",
        "last result",
        "this result",
        "that result",
        "from above",
        "same result",
        "sort this",
        "filter this",
        "plot this",
        "visualize this",
        "show top",
        "top 5 of this",
        "top 5 from previous",
        "compute standard deviation of this",
    ]

    return any(phrase in q for phrase in follow_up_phrases)


def _handle_follow_up(question: str, previous_result: dict | None):
    """
    Handles true follow-up questions only.
    A full new question like 'Identify the top 5 most expensive geographic areas'
    should NOT be treated as a follow-up.
    """
    if previous_result is None:
        return None

    if not _is_follow_up_question(question):
        return None

    q = question.lower()

    previous_table = previous_result.get("result_table")
    if not previous_table:
        return None

    previous_df = pd.DataFrame(previous_table)

    if previous_df.empty:
        return None

    if "show top 5" in q or "top 5" in q:
        return previous_df.head(5), "Showing the top 5 rows from the previous result."

    if "show top 10" in q or "top 10" in q:
        return previous_df.head(10), "Showing the top 10 rows from the previous result."

    if "sort" in q:
        numeric_cols = previous_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            sort_col = numeric_cols[-1]
            sorted_df = previous_df.sort_values(sort_col, ascending=False)
            return sorted_df, f"Sorted the previous result by {sort_col} in descending order."

    if "standard deviation" in q or "std" in q:
        numeric_df = previous_df.select_dtypes(include="number")
        result_df = numeric_df.std().reset_index()
        result_df.columns = ["column", "standard_deviation"]
        return result_df, "Computed the standard deviation of numeric columns from the previous result."

    return None


def run_agent_turn(session_id: str | None, question: str) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if session_id is None:
        session_id = str(uuid.uuid4())

    previous_result = get_previous_result(session_id)

    follow_up_result = _handle_follow_up(question, previous_result)

    if follow_up_result is not None:
        execution_result, final_answer = follow_up_result
        generated_code = "# Follow-up query handled using previous stored result."

    else:
        agent = CodeExecutionAgent()
        agent_output = agent.run(
            question=question,
            csv_path=DATASET_PATH,
        )

        generated_code = agent_output["generated_code"]
        final_answer = agent_output["final_answer"]

        if not hasattr(agent.node, "last_result_df"):
            raise RuntimeError("Agent did not produce a reusable DataFrame result.")

        execution_result = agent.node.last_result_df

    visualization_html, visualization_decision = create_visualization(
        question=question,
        execution_result=execution_result,
    )

    state = save_result(
        session_id=session_id,
        dataset_name=DATASET_NAME,
        question=question,
        generated_code=generated_code,
        final_answer=final_answer,
        execution_result=execution_result,
        visualization=visualization_html,
        visualization_decision=visualization_decision,
    )

    return {
        "session_id": session_id,
        "turn_id": state["turn_id"],
        "answer": final_answer,
        "generated_code": generated_code,
        "result_table": execution_result.to_dict(orient="records"),
        "visualization": visualization_html,
        "visualization_decision": visualization_decision,
        "error": None,
    }