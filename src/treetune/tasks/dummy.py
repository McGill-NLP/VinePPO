from typing import Dict, Any

from treetune.tasks.base_task import Task
from treetune.tasks.gsm8k import GSM8K


@Task.register("dummy", exist_ok=True)
class DummyTask(GSM8K):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.constant_solution = "a" * 250
        self.constant_final_answer = "b" * 20

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "_solution": self.constant_solution,
            "_final_answer": self.constant_final_answer,
        }

    def extract_gold_answer_from_text(self, text: str) -> str:
        return self.constant_final_answer
