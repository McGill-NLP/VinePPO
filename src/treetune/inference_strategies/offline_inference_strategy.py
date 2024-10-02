from datasets import Dataset, load_from_disk

from treetune.inference_strategies import InferenceStrategy


@InferenceStrategy.register("offline")
class OfflineInferenceStrategy(InferenceStrategy):
    def __init__(self, run_id: str, **kwargs):
        super().__init__(**kwargs)
        self.run_id = run_id

    def generate(self, dataset: Dataset) -> Dataset:
        from treetune.common import wandb_utils

        inference_ds = wandb_utils.load_inference_result_from_run_id(self.run_id)
        return inference_ds


@InferenceStrategy.register("local")
class LocalInferenceStrategy(InferenceStrategy):
    def __init__(self, inference_path: str, **kwargs):
        super().__init__(**kwargs)
        self.inference_path = inference_path

    def generate(self, dataset: Dataset) -> Dataset:
        inference_ds = load_from_disk(self.inference_path)
        return inference_ds
