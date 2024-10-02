from typing import Any, Optional, Dict, List, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from treetune import logging_utils
from treetune.common import Registrable

logger = logging_utils.get_logger(__name__)


class Task(Registrable):
    def __init__(
        self,
        hf_dataset_args: Optional[List[str]] = None,
        hf_dataset_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_dict: bool = True,
        dataset_dict_path: Optional[str] = None,
        hf_num_proc: int = 4,
        field_map: Optional[Dict[str, str]] = None,
        remove_mapped_fields: bool = False,
    ):
        """
        Params:
            hf_dataset_args: The arguments to pass to load_dataset
            hf_dataset_kwargs: The keyword arguments to pass to load_dataset

        Example:
            # Load a dataset from the HuggingFace datasets library
            >>> task = Task(
            >>>    hf_dataset_args=["gsm8k", "main"],
            >>> )

            # Load a dataset from a local file
            >>> task = Task(
            >>>     hf_dataset_args=["json"],
            >>>     hf_dataset_kwargs={
            >>>         "data_files": {
            >>>             "train": "data/train.jsonl",
            >>>             "validation": "data/valid.jsonl",
            >>>             "test": "data/test.jsonl",
            >>>         },
            >>>         "field": None,
            >>>     },
            >>> )
        """
        self.hf_dataset_args = hf_dataset_args
        self.hf_dataset_kwargs = hf_dataset_kwargs or {}

        self.load_dataset_dict = load_dataset_dict
        self.dataset_dict_path = dataset_dict_path
        if self.load_dataset_dict:
            assert self.dataset_dict_path is not None

        # Make sure either hf_dataset_args or load_dataset_dict is provided
        if self.hf_dataset_args is None and self.load_dataset_dict is None:
            raise ValueError(
                "Either hf_dataset_args or load_dataset_dict must be provided"
            )

        self.hf_num_proc = hf_num_proc
        self.field_map = field_map
        self.remove_mapped_fields = remove_mapped_fields

        self._ds_cache: Optional[DatasetDict] = None

    @property
    def name(self) -> str:
        """
        The name of this task
        """
        return f"{self.__class__.__name__}"

    def build_dataset(self) -> DatasetDict:
        """
        Build the datasets for this task
        """
        data_source = self._load_data_source()
        if not isinstance(data_source, DatasetDict):
            raise ValueError(
                f"Expected a DatasetDict, but got {type(data_source)} instead"
            )

        if self.field_map is not None:

            def map_fields(example):
                return {self.field_map[k]: v for k, v in example.items()}

            data_source = data_source.map(
                map_fields,
                desc="Mapping fields",
                num_proc=self.hf_num_proc,
                remove_columns=list(self.field_map.keys())
                if self.remove_mapped_fields
                else None,
            )

        # Add idx column
        data_source = data_source.map(
            lambda example, idx: {"_treetune__idx": idx},
            with_indices=True,
            num_proc=self.hf_num_proc,
            desc="Adding idx column",
        )

        return data_source

    def get_datasets(
        self, split: Optional[str] = None, no_cache: bool = False
    ) -> Union[DatasetDict, Dataset]:
        """
        Get the datasets for this task
        Params:
            split: The split to get, if None, return the entire dataset
            no_cache: If True, do not use the cached dataset
        """
        if self._ds_cache is None or no_cache:
            self._ds_cache = self.build_dataset()

        if split is None:
            return self._ds_cache
        else:
            return self._ds_cache[split]

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references
        Params:
            predictions: A list of predictions, each prediction is a list of candidate answers
            references: The reference dataset

        Returns:
            A dictionary of metrics
        """
        raise NotImplementedError

    def extract_predicted_answer_from_text(self, text: str) -> str:
        """
        Extract the predicted answer from the text
        """
        raise NotImplementedError

    def _load_data_source(
        self,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        if self.load_dataset_dict:
            return DatasetDict.load_from_disk(self.dataset_dict_path)
        return load_dataset(*self.hf_dataset_args, **self.hf_dataset_kwargs)


Task.register("default")(Task)
