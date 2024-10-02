import random
from typing import Dict, Any, Optional

from datasets import DatasetDict

from treetune.tasks.base_task import Task
from treetune.tokenization_utils import Tokenizer


@Task.register("imdb", exist_ok=True)
class IMDB(Task):
    def __init__(
        self,
        chop_reviews: bool = False,
        min_length: int = 2,
        max_length: int = 8,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chop_reviews = chop_reviews
        self.min_length = min_length
        self.max_length = max_length
        self.tokenizer = tokenizer

    def build_dataset(self) -> DatasetDict:
        datasets = super().build_dataset()
        datasets = datasets.filter(
            lambda x: len(x["text"]) > 200,
            num_proc=self.hf_num_proc,
            desc="Filtering short texts",
        )
        if self.chop_reviews:
            datasets = datasets.map(
                self._randomly_chop_text,
                num_proc=self.hf_num_proc,
                desc="Chopping text",
            )
        else:
            datasets = datasets.map(
                lambda x: {"query": x["text"]}, num_proc=self.hf_num_proc
            )

        return datasets

    def _randomly_chop_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly chop a text into chunks of size max_len
        """
        rng = random.Random(example["_treetune__idx"])
        num_query_tokens = rng.randint(self.min_length, self.max_length)
        text = example["text"]

        tokens = self.tokenizer.encode(text)[:num_query_tokens]
        query = self.tokenizer.decode(tokens)

        return {"query": query}
