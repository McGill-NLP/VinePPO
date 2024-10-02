from datasets import DatasetDict

from treetune.tasks.base_task import Task


@Task.register("anthropic_hh", exist_ok=True)
class AnthropicHH(Task):
    def build_dataset(self) -> DatasetDict:
        datasets = super().build_dataset()
        return datasets.remove_columns(["chosen", "rejected"])
