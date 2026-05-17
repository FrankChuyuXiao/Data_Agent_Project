import json
from typing import Any, Dict, Optional

from .node import AIAgentNode


class CodeExecutionAgent:
    """
    Workflow:
    Input -> Codegen -> Execute -> Evaluate -> (Retry once) -> Respond
    """

    def __init__(self, model: str = "gpt-5.4", api_key: Optional[str] = None) -> None:
        self.node = AIAgentNode(model=model, api_key=api_key)

    def run(self, question: str, csv_path: str) -> Dict[str, Any]:
        attempts = []
        retry_context = None

        for attempt_index in range(2):
            generated_code = self.node.generate_code(
                question=question,
                csv_path=csv_path,
                retry_context=retry_context,
            )
            execution_result = self.node.execute_code(generated_code)
            evaluation = self.node.evaluate_correctness(
                question=question,
                execution_result=execution_result,
            )

            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "generated_code": generated_code,
                    "execution_result": execution_result,
                    "evaluation": evaluation,
                }
            )

            if evaluation == "PASS" or attempt_index == 1:
                final_answer = self.node.generate_answer(
                    question=question,
                    execution_result=execution_result,
                )
                return {
                    "generated_code": generated_code,
                    "execution_result": execution_result,
                    "evaluation": evaluation,
                    "final_answer": final_answer,
                    "attempts": attempts,
                }

            retry_context = {
                "previous_code": generated_code,
                "execution_result": execution_result,
            }

        raise RuntimeError("Unexpected workflow termination.")


if __name__ == "__main__":
    csv_path = "housing.csv"
    question = "What is the average median house value across the dataset?"

    agent = CodeExecutionAgent()
    result = agent.run(question=question, csv_path=csv_path)

    print("=== Agent Workflow Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
