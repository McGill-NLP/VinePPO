import gc
import json
import random
import subprocess
import time
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Optional

import psutil
import torch
from accelerate.utils import release_memory
from datasets import Dataset
from deepspeed import get_accelerator
from deepspeed.runtime.utils import torch_memory_reserved, torch_max_memory_reserved
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    pipeline,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)

from treetune.common import Lazy
from treetune.episode_generators.base_episode_generator import (
    EpisodeGenerator,
    Episode,
)
from treetune.episode_generators.on_policy_episode_generator import (
    OnPolicyEpisodeGenerator,
)
from treetune.episode_generators.tree_episode_generator import TreeEpisodeUtils
from treetune.logging_utils import get_logger
from treetune.models import Model
from treetune.tokenization_utils import Tokenizer

logger = get_logger(__name__)


def see_memory_usage(message, force=False):
    if not force:
        return

    from deepspeed import comm as dist

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    device_index = dist.get_rank()
    logger.info(
        f"MA {round(get_accelerator().memory_allocated(device_index=device_index) / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(get_accelerator().max_memory_allocated(device_index=device_index) / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved(device_index=device_index) / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved(device_index=device_index) / (1024 * 1024 * 1024))} GB "
    )

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(
        f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%"
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats(device_index=device_index)


def get_gpu_memory():
    # Run the nvidia-smi command to get GPU memory usage
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    # Parse the output to get memory usage as a list of integers (in MB)
    memory_usage = [int(x) for x in result.stdout.strip().split("\n")]
    return memory_usage


def wait_for_memory_release(target_gpu_index, threshold_mb=1024):
    # Wait until memory usage for the specified GPU is below the threshold
    while True:
        memory_usage = get_gpu_memory()
        if memory_usage[target_gpu_index] < threshold_mb:
            print(
                f"GPU {target_gpu_index} has less than {threshold_mb} MB used. Continuing..."
            )
            break
        else:
            print(
                f"GPU {target_gpu_index} memory used: {memory_usage[target_gpu_index]} MB. Waiting..."
            )
            time.sleep(2)


@EpisodeGenerator.register("episode_generator_with_reward_model")
class EpisodeGeneratorWithRewardModel(OnPolicyEpisodeGenerator, TreeEpisodeUtils):
    def __init__(
        self,
        reward_model: Optional[Lazy[Model]] = None,
        reward_model_tokenizer: Optional[Tokenizer] = None,
        reward_model_padding_side: str = "right",
        reward_pipline_model_name: Optional[str] = None,
        reward_pipeline_task: str = "sentiment-analysis",
        reward_inference_per_device_batch_size: int = 128,
        append_bos_to_query: Union[str, bool] = "auto",
        append_eos_to_response: Union[str, bool] = "auto",
        chop_response: bool = False,
        min_response_length: int = 4,
        max_response_length: int = 16,
        cache_reward_model: bool = True,
        cache_reward_model_on_cpu: bool = False,
        temp_cache_dir: Optional[str] = None,
        penalty_for_unfinished_response: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # `reward_model` and `reward_pipline_model_name` are mutually exclusive
        if reward_model is not None and reward_pipline_model_name is not None:
            raise ValueError(
                "Only one of `reward_model` and `reward_pipline_model_name` should be provided."
            )
        if reward_model is None and reward_pipline_model_name is None:
            raise ValueError(
                "Either `reward_model` or `reward_pipline_model_name` should be provided."
            )

        if reward_model is not None:
            assert reward_model_tokenizer is not None
            assert reward_model_padding_side in ["left", "right"]

        self.reward_model_lazy = reward_model
        self.reward_model_tokenizer = reward_model_tokenizer
        self.reward_model_padding_side = reward_model_padding_side
        self.reward_pipline_model_name = reward_pipline_model_name
        self.reward_pipline_task = reward_pipeline_task
        self.reward_inf_per_device_batch_size = reward_inference_per_device_batch_size
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_to_response
        self.chop_response = chop_response
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.cache_reward_model = cache_reward_model
        self.cache_reward_model_on_cpu = cache_reward_model_on_cpu
        self.penalty_for_unfinished_response = penalty_for_unfinished_response

        if temp_cache_dir is not None:
            self.temp_model_cache_dir = Path(temp_cache_dir)
        else:
            from treetune.common.notebook_utils import get_repo_dir

            self.temp_model_cache_dir = get_repo_dir() / "temp_model_cache_dir"
            logger.info(
                f"No temporary model cache directory provided. Using {self.temp_model_cache_dir}"
            )
        self.temp_model_cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_reward_model_pipeline(
        self,
    ) -> Pipeline:
        device = self.distributed_state.process_index
        sentiment_pipe = pipeline(
            self.reward_pipline_task,
            model=self.reward_pipline_model_name,
            device=device,
        )
        return sentiment_pipe

    def _init_reward_model(self) -> PreTrainedModel:
        this_process_device = self.distributed_state.device

        if hasattr(self, "_reward_model"):
            self._reward_model.to(this_process_device)
            return self._reward_model

        t0 = time.time()

        # Load the reward model into GPU
        cache_path = self.temp_model_cache_dir / ".reward_model"
        if not cache_path.exists():
            cache_path = None

        # noinspection PyTypeChecker
        reward_model: PreTrainedModel = self.reward_model_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
            runtime_hf_model_name=cache_path,
        )
        reward_model.to(this_process_device)
        self._cloud_log(
            {"timing/episode_generation/reward_model_construct": time.time() - t0}
        )

        if self.cache_reward_model and cache_path is None and self.is_main_process():
            # Since the reward model is used in every iteration, it makes
            # sense to cache it on the fast disk to avoid loading it from network storage
            cache_path = self.temp_model_cache_dir / ".reward_model"
            reward_model.save_pretrained(cache_path, safe_serialization=False)

        if self.cache_reward_model_on_cpu:
            self._reward_model = reward_model

        return reward_model

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        episodes_without_rewards = []
        for instance in inference_results:
            episodes = self._convert_to_episodes(instance)
            episodes_without_rewards.extend(episodes)

        episodes = self._compute_rewards(episodes_without_rewards)

        if self.penalty_for_unfinished_response is not None:
            num_unfinished = sum(
                1 for e in episodes if e.scores == self.penalty_for_unfinished_response
            )
            num_total = len(episodes)
            self._cloud_log(
                {
                    "episodes_metric/is_unfinished_response": (
                        num_unfinished / num_total
                    ),
                    "train/global_iteration": iteration,
                }
            )

        return episodes

    def _convert_to_episodes(self, instance: Dict[str, Any]) -> List[Episode]:
        tree = json.loads(instance["_treetune__reasoning_tree"])
        paths = self.extract_paths_from_tree(tree)

        episodes = []
        for path in paths:
            episodes.extend(self._convert_path_to_episodes(instance, path))

        return episodes

    def _compute_rewards_from_pipeline(self, sequences: List[str]) -> List[float]:
        reward_pipeline = self._init_reward_model_pipeline()
        pipe_outputs = reward_pipeline(
            sequences, return_all_scores=True, function_to_apply="none", batch_size=128
        )
        rewards = [
            (
                pos["score"].cpu().item()
                if isinstance(pos["score"], torch.Tensor)
                else pos["score"]
            )
            for _, pos in pipe_outputs
        ]
        del reward_pipeline
        release_memory()
        return rewards

    def _compute_rewards_from_model(self, sequences: List[str]) -> List[float]:
        reward_model = self._init_reward_model()
        reward_model.eval()

        # noinspection PyTypeChecker
        tokenizer: PreTrainedTokenizer = self.tokenizer
        tokenizer.padding_side = self.reward_model_padding_side

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0

        def collate_fn(examples: List[Dict[str, Any]]) -> BatchEncoding:
            return tokenizer(
                [e["text"] for e in examples],
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

        dataloader = DataLoader(
            Dataset.from_dict({"text": sequences}),
            batch_size=self.reward_inf_per_device_batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        has_logged = False

        all_rewards = []
        for inputs in tqdm(dataloader, desc="Computing rewards"):
            with torch.no_grad():
                if self.is_main_process() and not has_logged:
                    decoded = tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=False
                    )
                    logger.info(f"Decoded input: {decoded}")
                    has_logged = True

                inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                outputs = reward_model(**inputs)

                # Extract the rewards from the last token
                if self.reward_model_padding_side == "right":
                    assert torch.all(
                        inputs["attention_mask"][:, 0] == 1
                    ), "Reward model expect the padding to be done on the right side."
                    # Compute the index of last token in the sequence lengths
                    seq_lengths = inputs["attention_mask"].sum(dim=1)
                    last_token_indices = seq_lengths - 1
                    rewards = outputs[range(outputs.shape[0]), last_token_indices]
                elif self.reward_model_padding_side == "left":
                    assert torch.all(
                        inputs["attention_mask"][:, -1] == 1
                    ), "Reward model expect the padding to be done on the left side."
                    rewards = outputs[:, -1]
                else:
                    raise ValueError(
                        f"Invalid padding side: {self.reward_model_padding_side}"
                    )

                all_rewards.extend(rewards.float().cpu().numpy().tolist())

        assert len(all_rewards) == len(sequences)

        if self.cache_reward_model_on_cpu:
            reward_model.to("cpu")

        return all_rewards

    def _compute_rewards(self, episodes: List[Episode]) -> List[Episode]:
        sequences = [
            self.tokenizer.decode(e.query_token_ids + e.response_token_ids)
            for e in episodes
        ]

        if self.reward_model_lazy is not None:
            rewards = self._compute_rewards_from_model(sequences)
        else:
            rewards = self._compute_rewards_from_pipeline(sequences)
        release_memory()

        episodes_with_reward = [
            Episode(
                query_token_ids=e.query_token_ids,
                response_token_ids=e.response_token_ids,
                scores=reward if e.scores == -10000.0 else e.scores,
            )
            for reward, e in zip(rewards, episodes)
        ]

        return episodes_with_reward

    def _convert_path_to_episodes(
        self, instance: Dict[str, Any], path: Dict[str, Any]
    ) -> List[Episode]:
        query_text = path["node_chain"][0]["text"]
        assert len(path["node_chain"]) == 2
        full_text = path["node_chain"][-1]["full_text"]
        response_text = full_text[len(query_text) :]
        finish_reason = path["node_chain"][-1]["finish_reason"]
        is_chopped = finish_reason == "length"
        reward = -10000.0
        if self.penalty_for_unfinished_response is not None and is_chopped:
            assert self.penalty_for_unfinished_response != -10000.0
            reward = self.penalty_for_unfinished_response

        try:
            query_token_ids, response_token_ids = self._tokenize_query_and_response(
                query_text, response_text
            )
        except Exception as e:
            logger.error(
                f"Failed to tokenize query and response for instance {instance['_treetune__idx']}"
            )
            logger.error(f"Query: {query_text}")
            logger.error(f"Response: {response_text}")
            return []

        if self.chop_response:
            rng = random.Random(instance["_treetune__idx"])
            response_length = rng.randint(
                self.min_response_length, self.max_response_length
            )
            response_token_ids = response_token_ids[:response_length]
            is_chopped = True

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if not is_chopped and self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        episode = Episode(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            scores=reward,
        )
        return [episode]

    def _tokenize_query_and_response(
        self, query: str, response: str
    ) -> Tuple[List[int], List[int]]:
        instance_text = f"{query}{response}"
        instance_encoding = self.tokenizer(
            instance_text,
            add_special_tokens=False,  # We already added BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = instance_encoding["input_ids"]
        offsets = instance_encoding["offset_mapping"]

        # Find the index where the response starts.
        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query)
        )

        # Split the token IDs into query and response parts
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        # Check that the decoded text matches the original text
        if not getattr(self, "has_warned_about_decoding_mismatch", False):
            decoded_instance = self.tokenizer.decode(
                query_token_ids + response_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            has_warned = False
            if decoded_instance != instance_text:
                logger.warning(
                    f"Decoded instance does not match original instance.\n"
                    f"Original instance: {instance_text}\n"
                    f"Decoded instance: {decoded_instance}"
                )
                has_warned = True

            decoded_query = self.tokenizer.decode(
                query_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if decoded_query != query:
                logger.warning(
                    f"Decoded query does not match original query.\n"
                    f"Original query: {query}\n"
                    f"Decoded query: {decoded_query}"
                )
                has_warned = True

            decoded_response = self.tokenizer.decode(
                response_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if decoded_response != response:
                logger.warning(
                    f"Decoded response does not match original response.\n"
                    f"Original response: {response}\n"
                    f"Decoded response: {decoded_response}"
                )
                has_warned = True

            self.has_warned_about_decoding_mismatch = has_warned

        return query_token_ids, response_token_ids

    def _should_append_bos_to_query(self) -> bool:
        """
        Determine whether to append BOS to the query based on the tokenizer
        """
        if self.append_bos_to_query != "auto":
            return self.append_bos_to_query

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.bos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append BOS for tokenizer {self.tokenizer.name_or_path}"
            )

    def _should_append_eos_to_response(self) -> bool:
        """
        Determine whether to append EOS to the response based on the tokenizer
        """
        if self.append_eos_to_response != "auto":
            return self.append_eos_to_response

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.eos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append EOS for tokenizer {self.tokenizer.name_or_path}"
            )
