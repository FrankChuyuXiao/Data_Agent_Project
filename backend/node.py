import code
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from openai import OpenAI


@dataclass
class NodeRunResult:
    question: str
    csv_path: str
    code: str
    execution_result: Dict[str, Any]
    evaluation: str
    answer: str


class AIAgentNode:
    def __init__(self, model: str = "gpt-5.4", api_key: Optional[str] = None) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _response_text(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text.strip()

    def _build_code_prefix(self, csv_path: str) -> str:
        csv_path_str = str(csv_path)
        return (
            'import pandas as pd\n\n'
            f'df = pd.read_csv(r"{csv_path_str}", engine="python", on_bad_lines="skip")\n'
            'df.columns = df.columns.str.strip().str.replace("\\\"", "", regex=False)\n'
            'column_map = {col.lower(): col for col in df.columns}\n'
        )

    def _base_codegen_prompt(self, question: str, csv_path: str) -> str:
        prefix = self._build_code_prefix(csv_path)
        return (
            "You are a Python data analysis code generator.\n\n"
            "You must generate only the analysis body of a pandas script.\n"
            "The CSV loading and column normalization have already been written exactly as shown below.\n"
            "Your code must work with that existing setup and must not redefine df, reload the CSV, or use any other filename.\n\n"
            "Existing code:\n"
            f"{prefix}\n"
            "Requirements:\n"
            "1. Use the existing DataFrame df.\n"
            "2. Use column_map for case-insensitive column lookup.\n"
            "3. Match actual dataset columns from df.columns.\n"
            "4. Use pd.to_numeric(..., errors=\"coerce\") for numeric columns when needed.\n"
            "5. Store the final output in a variable named result.\n"
            "6. result must be a pandas DataFrame.\n"
            "7. Do not print anything.\n"
            "8. Do not define functions or classes.\n"
            "9. Do not use markdown fences.\n"
            "10. If required columns are missing, set result to a DataFrame with columns error and available_columns.\n"
            "11. If computation yields no valid values, set result to a DataFrame with an error column.\n"
            "12. Ensure result values are JSON-serializable. Convert intervals or other complex values to strings if needed.\n"
            "13. Return only executable Python code for the analysis body.\n\n"
            "Question:\n"
            f"{question}"
        )

    def _retry_codegen_prompt(
        self,
        question: str,
        csv_path: str,
        previous_code: str,
        execution_result: Dict[str, Any],
    ) -> str:
        safe_execution_result = json.dumps(
            self._make_json_safe(execution_result), ensure_ascii=False, indent=2
        )
        prefix = self._build_code_prefix(csv_path)
        return (
            "The previous generated code failed or produced an invalid result.\n"
            "Generate corrected Python code for only the analysis body.\n"
            "Do not redefine df, reload the CSV, or use any other filename.\n\n"
            "Existing code prefix:\n"
            f"{prefix}\n"
            "Original question:\n"
            f"{question}\n\n"
            "Previous generated body:\n"
            f"{previous_code}\n\n"
            "Execution result:\n"
            f"{safe_execution_result}\n\n"
            "Fix the problem.\n"
            "Requirements:\n"
            "- Use the existing df and column_map.\n"
            "- Use actual columns from df.columns.\n"
            "- Use pd.to_numeric(..., errors=\"coerce\") for numeric columns when needed.\n"
            "- Keep the final output in a pandas DataFrame named result.\n"
            "- If columns are missing, return a DataFrame with error and available_columns.\n"
            "- Ensure result values are JSON-serializable.\n"
            "- Return only corrected executable Python code for the analysis body."
        )

    def generate_code(
        self,
        question: str,
        csv_path: str,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if retry_context is None:
            prompt = self._base_codegen_prompt(question=question, csv_path=csv_path)
        else:
            prompt = self._retry_codegen_prompt(
                question=question,
                csv_path=csv_path,
                previous_code=retry_context["previous_code"],
                execution_result=retry_context["execution_result"],
            )

        body = self._response_text(prompt)
        body = self._strip_code_fences(body)
        code = self._build_code_prefix(csv_path) + "\n" + body.strip() + "\n"
        return self._validate_generated_code(code=code, csv_path=csv_path)

    def _validate_generated_code(self, code: str, csv_path: str) -> str:
        csv_path_str = str(csv_path)

        if csv_path_str not in code:
            raise ValueError(f"Generated code did not use the required csv path: {csv_path_str}")

        if "result" not in code:
            raise ValueError("Generated code does not define `result`.")

        return code

    def execute_code(self, code: str) -> Dict[str, Any]:
        env: Dict[str, Any] = {}

        try:
            exec(code, env)
        except Exception as exc:
            return {
                "status": "execution_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "result_preview": None,
                "result_is_dataframe": False,
            }

        if "result" not in env:
            return {
                "status": "execution_error",
                "error_type": "MissingResultError",
                "error": "Generated code did not define `result`.",
                "result_preview": None,
                "result_is_dataframe": False,
            }

        result = env["result"]

        self.last_result_df = result

        if not isinstance(result, pd.DataFrame):
            return {
                "status": "execution_error",
                "error_type": "InvalidResultType",
                "error": f"`result` is {type(result).__name__}, not a pandas DataFrame.",
                "result_preview": repr(result),
                "result_is_dataframe": False,
            }

        preview = result.head(10).copy()
        preview_records = self._make_json_safe(preview.to_dict(orient="records"))

        payload: Dict[str, Any] = {
            "status": "ok",
            "result_type": "DataFrame",
            "result_is_dataframe": True,
            "shape": list(result.shape),
            "columns": result.columns.tolist(),
            "result_preview": preview_records,
        }

        if "error" in result.columns:
            payload["status"] = "logical_error"

        return payload

    def evaluate_correctness(self, question: str, execution_result: Dict[str, Any]) -> str:
        safe_result = self._make_json_safe(execution_result)

        if execution_result.get("status") in {"execution_error", "logical_error"}:
            return "FAIL"
        if not execution_result.get("result_is_dataframe", False):
            return "FAIL"
        if not execution_result.get("result_preview"):
            return "FAIL"

        prompt = (
            "Return PASS or FAIL.\n"
            "FAIL if:\n"
            "- execution error\n"
            "- logical error\n"
            "- incorrect or invalid result\n"
            "Question:\n"
            f"{question}\n"
            "Result:\n"
            f"{json.dumps(safe_result, ensure_ascii=False, indent=2)}\n\n"
            "Return only PASS or FAIL."
        )

        evaluation = self._response_text(prompt).upper()
        return "PASS" if "PASS" in evaluation and "FAIL" not in evaluation else "FAIL"

    def generate_answer(self, question: str, execution_result: Dict[str, Any]) -> str:
        safe_result = self._make_json_safe(execution_result)

        prompt = (
            "Generate a clear answer based only on the execution result.\n"
            "If the result indicates an error or invalid output, explain that clearly.\n"
            "Question:\n"
            f"{question}\n"
            "Result:\n"
            f"{json.dumps(safe_result, ensure_ascii=False, indent=2)}"
        )

        return self._response_text(prompt)

    def run(self, question: str, csv_path: str) -> NodeRunResult:
        code = self.generate_code(question=question, csv_path=csv_path)
        execution_result = self.execute_code(code)
        evaluation = self.evaluate_correctness(question=question, execution_result=execution_result)
        answer = self.generate_answer(question=question, execution_result=execution_result)

        return NodeRunResult(
            question=question,
            csv_path=csv_path,
            code=code,
            execution_result=execution_result,
            evaluation=evaluation,
            answer=answer,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return text.strip()

    @staticmethod
    def _make_json_safe(obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, dict):
            return {str(k): AIAgentNode._make_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [AIAgentNode._make_json_safe(v) for v in obj]

        if isinstance(obj, pd.Interval):
            return str(obj)

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        if isinstance(obj, pd.Series):
            return AIAgentNode._make_json_safe(obj.to_dict())

        if isinstance(obj, pd.DataFrame):
            return AIAgentNode._make_json_safe(obj.to_dict(orient="records"))

        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)


if __name__ == "__main__":
    csv_path = "your_dataset.csv"
    question = "What are the first 5 rows of the dataset?"

    node = AIAgentNode()
    output = node.run(question=question, csv_path=csv_path)

    print("=== Generated Code ===")
    print(output.code)
    print("\n=== Execution Result ===")
    print(json.dumps(output.execution_result, indent=2, ensure_ascii=False, default=str))
    print("\n=== Evaluation ===")
    print(output.evaluation)
    print("\n=== Answer ===")
    print(output.answer)
