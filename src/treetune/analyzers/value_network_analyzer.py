import json
import shutil
import tempfile
from datetime import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from treetune import logging_utils
from treetune.analyzers import Analyzer
from treetune.models import PreTrainedModelForValueNetwork
from treetune.trainers import ValueNetworkTrainer
from transformers.trainer_utils import seed_worker, speed_metrics

logger = logging_utils.get_logger(__name__)


class ValueDataCollator:
    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates the given data instances into a batch.
        Every data instance should have the following keys:
        - "query_token_ids" (List[int]): The token ids of the query.

        Args:
            data_instances (List[Dict[str, Any]]):
                The data instances to collate.
        Returns:
            Dict[str, Any]:
                The collated batch.
                It contains the following keys:
                "input_ids": The input ids of the accepted responses.
                "attention_mask": The attention mask of the accepted responses.
                "original_lengths": The original lengths of the queries.
        """

        # Get the maximum sequence length
        max_seq_len = max(
            len(instance["query_token_ids"])
            for instance in data_instances
        )

        # Create the batch
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "original_length": [],
        }

        # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_token_id = 0

        def get_padded_input_ids_attention_mask_value_targets_and_value_loss_mask(
            query_tok_ids, max_seq_len
        ):
            # Create the input ids and attention mask
            input_ids = query_tok_ids.copy()
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_len - len(input_ids)

            # Pad the input ids and attention mask at the end to the maximum sequence length
            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "original_length": len(query_tok_ids)
            }

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]

            # Create the input ids, attention mask and labels for the accepted response
            response_batch = (
                get_padded_input_ids_attention_mask_value_targets_and_value_loss_mask(
                    query_token_ids,
                    max_seq_len,
                )
            )

            for k, v in response_batch.items():
                batch[k].append(v)

        # Convert the lists to tensors
        for k in batch:
            batch[k] = torch.tensor(batch[k])

        return batch


@Analyzer.register("value_network", exist_ok=True)
class ValueNetworkAnalyzer(Analyzer):
    def __init__(self,
                 d_star_dataset_dir: str,
                 batch_size: int,
                 **kwargs):
        super().__init__(**kwargs)

        self.d_star_dataset_dir = d_star_dataset_dir
        self.batch_size = batch_size
        self.data_collator = ValueDataCollator()

    def analyze(
        self, every_n_checkpoints: int = 1, force_rerun: bool = False, **kwargs
    ):
        super().analyze()

        logger.info(f"Loading the dataset from {self.d_star_dataset_dir}...")
        d_star_dataset = Dataset.load_from_disk(self.d_star_dataset_dir)

        d_star_dataset = d_star_dataset.map(  # todo: dirtiest hack ever to put the format of MATH directly in here
            lambda x: {
                "query": f"[MATH_TASK] Problem:\n{x['problem']}\n\nSolution:\n{x['partial_solution']}"
            }
        )

        tokenizer = self.runtime.tokenizer

        def tokenize_query(x):
            token_ids = tokenizer(x["query"]).input_ids
            assert token_ids[0] == tokenizer.bos_token_id
            assert token_ids[-1] != tokenizer.eos_token_id
            if x['is_last_step']:
                token_ids.append(tokenizer.eos_token_id)

            return token_ids

        d_star_dataset = d_star_dataset.map(
            lambda x: {
                "query_token_ids": tokenize_query(x)
            },
            num_proc=8,
        )

        analysis_root = self.get_analysis_root_dir()
        analysis_root.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (analysis_root / "done").exists():
            logger.warning(
                f"Analysis directory {self.get_analysis_root_dir()} already exists."
            )
            return

        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        # checkpoint_dir.mkdir(exist_ok=True, parents=True) # todo: really? if it does not exist it means we messed up, no? so we should not create it

        ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints
        )
        ckpts = [
            ckpt
            for ckpt in ckpts
            if (ckpt / "hf_pretrained" / "pytorch_model.bin").exists()
        ]
        logger.info(
            f"Computing value network performances of {len(ckpts)} checkpoints: {ckpts}"
        )

        ckpts = ckpts[::-1]
        #ckpts = [ckpts[-1]] # todo: remove milad, just for now that we care about the last checkpoint

        model = self.runtime.model_lazy.construct()

        if torch.cuda.is_available():
            model = model.to("cuda")

        metrics = {}
        for ckpt in tqdm(ckpts, desc="Computing value network performances"):
            result_file = analysis_root / f"{ckpt.name}.json"
            if result_file.exists():
                mse_error = json.loads(result_file.read_text())["mse_error"]
                metrics[ckpt.name] = mse_error
                logger.info(f"Skipping {ckpt} as it has already been analyzed.")
                continue

            model.load_state_dict(
                torch.load(ckpt / "hf_pretrained" / "pytorch_model.bin")
            )
            model.eval()

            mse_error, dataset_with_predicted_value = self.evaluate_value_network(model, d_star_dataset)

            # save the dataset with predicted values
            self._save_dataset_with_predicted_values(dataset_with_predicted_value, ckpt.name)
            self._save_dataset_with_predicted_values_to_cloud(dataset_with_predicted_value, ckpt.name)

            metrics[ckpt.name] = mse_error
            with result_file.open("w") as f:
                json.dump({"mse_error": mse_error}, f)


        for ckpt_name, mse_error in metrics.items():
            global_step = ValueNetworkTrainer.parse_checkpoint_name(ckpt_name)[-1]
            if self.cloud_logger is not None and self.plot_prefix is not None:
                self.cloud_logger.log(
                    {
                        f"{self.plot_prefix}/{self.__class__.__name__}": mse_error,
                        "train/global_step": global_step,
                    }
                )

        self.log_metrics(metrics)

        all_ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )
        all_ckpts = [ckpt for ckpt in all_ckpts if (ckpt / "hf_pretrained").exists()]
        all_ckpts_are_done = all(
            (analysis_root / f"{ckpt.name}.json").exists() for ckpt in all_ckpts
        )
        if all_ckpts_are_done:
            (analysis_root / "done").touch()

    def _save_dataset_with_predicted_values(self, dataset: Dataset, ckpt_name: str):
        save_dir = self.get_analysis_root_dir() / f"dataset_with_predicted_values_{ckpt_name}"
        logger.info(f"Saving the dataset with predicted values to {save_dir}")
        dataset.save_to_disk(save_dir)

    def _save_dataset_with_predicted_values_to_cloud(self, dataset: Dataset, ckpt_name: str):
        if self.cloud_logger is None:
            logger.warning(
                "Cloud logger is not set. "
                "Cannot save the dataset with predicted values to the cloud."
                "This command has no effect."
            )
            return

        output_dir = self.get_analysis_root_dir() / f"dataset_with_predicted_values_{ckpt_name}"
        if not output_dir.exists():
            logger.warning(
                f"Output directory {output_dir} does not exist. "
                "First saving inference results to local disk so we can zip it later."
            )
            self._save_dataset_with_predicted_values(dataset, ckpt_name)

        # First, create a zip file of the dataset into output_dir
        # This is because the cloud logger only accepts files.
        temp_dir = self.get_analysis_root_dir() / next(tempfile._get_candidate_names())
        temp_dir.mkdir(parents=True, exist_ok=True)

        dataset_zip = temp_dir / f"dataset_with_predicted_values_on__{self.metrics_prefix}__{ckpt_name}.zip"
        logger.info(f"Creating zip file {dataset_zip}")
        shutil.make_archive(
            str(dataset_zip.with_suffix("")), "zip", output_dir
        )

        # Then, upload the zip file to the cloud.
        self.cloud_logger.save(str(dataset_zip.absolute()))

    def evaluate_value_network(
        self, model: PreTrainedModelForValueNetwork, dataset: Dataset
    ):

        # make a dataloader over the dataset
        dataloader = self._create_dataloader(dataset)

        all_predicted_values = []

        device = model.value_head.weight.device
        for batch in tqdm(dataloader, desc="Computing predicted values"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            original_lengths = batch["original_length"].to(device)

            # forward pass
            with torch.no_grad():
                predicted_values = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # grab the last value from the predicted values according to the original lengths
            predicted_values = predicted_values[torch.arange(predicted_values.size(0)), original_lengths - 1]
            all_predicted_values.extend(predicted_values.cpu().tolist())

        logger.info("Creating the dataset with predicted values")
        dataset = dataset.add_column("predicted_value", all_predicted_values)

        if "_treetune__candidate_answers" in dataset.column_names:
            dataset = dataset.remove_columns("_treetune__candidate_answers")

        # compute the mse error
        if "gt_value" not in dataset.column_names:
            mse_error = float(1e9) # a crazy high number
        else:
            mse_error = (torch.tensor(dataset["predicted_value"]) - torch.tensor(dataset["gt_value"])**2).mean().item()

        return mse_error, dataset

    def _create_dataloader(self, d_star_dataset: Dataset) -> DataLoader:
        if d_star_dataset is None:
            raise ValueError(
                "Value Network Analyzer requires a D-star dataset to analyze."
            )

        # Filter out the episodes that are too long
        print(f"Size of the dataset: {len(d_star_dataset)}")

        data_loader = DataLoader(
            dataset=d_star_dataset,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            shuffle=False,  # super important to keep the same order as we will add this column to the dataset
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        return data_loader
