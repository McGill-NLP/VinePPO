import copy
import json
from dataclasses import asdict
from pathlib import Path
import random
from typing import Union, Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import torch
from accelerate.utils import release_memory
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel
from wandb.apis.public import Run

from treetune.analyzers.analyzer import Analyzer
from treetune.common import Lazy
from treetune.common.deepspeed_utils import prepare_data_loader_for_inference
from treetune.common.vllm_server import VLLMServer
from treetune.episode_generators import TreeEpisodeUtils
from treetune.episode_generators.base_episode_generator import Episode
from treetune.episode_generators.episode_generator_with_reward_function import RewardFunction
from treetune.inference_strategies import InferenceStrategy
from treetune.tasks import Task
from treetune import logging_utils
from treetune.trainers.data_collator import PPODataCollator, COLUMN_REF_SHIFTED_LOGPS, COLUMN_ACTOR_SHIFTED_LOGPS
from treetune.tokenization_utils import Tokenizer
from treetune.trainers.deepspeed_policy_trainer import DeepSpeedPolicyTrainer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("kl_with_reference")
class KLWithReferenceAnalyzer(Analyzer):
    def __init__(
        self,
        task: Task,
        inference_strategy: Lazy[InferenceStrategy],
        vllm_server: Lazy[VLLMServer],
        cloud_logger: Run,
        runtime,
        tokenizer: Tokenizer,
        reward_function: Lazy[RewardFunction],
        append_bos_to_query: bool,
        append_eos_to_response: bool,
        actor_deepspeed_config: dict,
        max_sequence_length: Optional[int] = None,
        **kwargs
    ):
        from treetune.runtime.policy_iteration_runtime import PolicyIterationRuntime
        self.runtime: PolicyIterationRuntime = runtime
        super().__init__(cloud_logger, runtime, **kwargs)

        self.task = task
        self.inference_strategy_lazy = inference_strategy
        self.vllm_server_lazy = vllm_server

        trainer = getattr(self.runtime, "trainer", None)
        if trainer is None or isinstance(trainer, Lazy):
            trainer = self.runtime._construct_trainer(init_model_only=False)
        else:
            logger.info("Using the existing trainer. Is this correct though(?)")
        assert isinstance(trainer, DeepSpeedPolicyTrainer)
        self.trainer: DeepSpeedPolicyTrainer = trainer
        self.trainer.cache_deepspeed_engines = False
        self.trainer.actor_deepspeed_config = actor_deepspeed_config
        self.trainer.total_num_training_steps = 0  # dummy, defensive programming so it messes things if actually used later on.
        per_device_batch_size = self.trainer.args.per_device_train_batch_size
        if actor_deepspeed_config.get("train_batch_size", None) == "auto":
            actor_deepspeed_config["train_batch_size"] = per_device_batch_size
        actor_deepspeed_config['gradient_accumulation_steps'] = 1
        actor_deepspeed_config['train_micro_batch_size_per_gpu'] = per_device_batch_size

        self.max_sequence_length = max_sequence_length
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_to_response
        self.tokenizer = tokenizer
        self.reward_function = reward_function.construct(tokenizer=self.tokenizer)

    def analyze(self, every_n_checkpoints: int = 1, force_rerun: bool = False):
        if not self.distributed_state.is_main_process:
            logger.info("Not main process in analyzer, skipping. It is a waste of time though. ")
            return

        super().analyze()
        self.get_analysis_root_dir().mkdir(parents=True, exist_ok=True)

        if not force_rerun and (self.get_analysis_root_dir() / "done").exists():
            return

        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        ckpts = self.runtime._get_list_of_evaluation_checkpoints(checkpoint_dir, every_n_checkpoints=every_n_checkpoints, ignore_worker_vars=False)

        for ckpt in ckpts:
            if (self.get_analysis_root_dir() / f"{ckpt.name}_kl_with_reference.json").exists() and not force_rerun:
                # read the file and get the kl
                with open(self.get_analysis_root_dir() / f"{ckpt.name}_kl_with_reference.json", "r") as f:
                    kl = json.load(f)['kl']
                logger.info(f"Skipping {ckpt} as it is already processed. {kl:.8f}")
                continue

            logger.info(f"Processing {ckpt.name}")
            kl = self._compute_kl_with_reference(ckpt, seed=42)
            logger.info(f"KL with reference: {kl:.8f}")
            # update the file
            with open(self.get_analysis_root_dir() / f"{ckpt.name}_kl_with_reference.json", "w") as f:
                json.dump({"kl": kl}, f)

        # ----- check if all checkpoints are processed and mark as done -----

        all_ckpts_for_all_process = self.runtime._get_list_of_evaluation_checkpoints(checkpoint_dir, every_n_checkpoints=every_n_checkpoints, ignore_worker_vars=True)
        def are_all_checkpoint_kls_computed():
            for _ckpt in all_ckpts_for_all_process:
                if not (self.get_analysis_root_dir() / f"{_ckpt.name}_kl_with_reference.json").exists():
                    return False
            return True

        if are_all_checkpoint_kls_computed():
            # read all of them, and gather to a single dict, then upload to cloud
            logger.info("All checkpoints are processed. Gathering all KLs and uploading to cloud.")
            all_kl_metrics = {}
            for ckpt in tqdm(all_ckpts_for_all_process, desc='parsing all checkpoint KLs.'):
                with open(self.get_analysis_root_dir() / f"{ckpt.name}_kl_with_reference.json", "r") as f:
                    all_kl_metrics[ckpt.name] = json.load(f)
            with open(self.get_analysis_root_dir() / "all_kl_with_reference.json", "w") as f:
                json.dump(all_kl_metrics, f)
            # log to cloud
            self.cloud_logger.save(str((self.get_analysis_root_dir() / f"all_kl_with_reference.json").absolute()), policy="now")
            (self.get_analysis_root_dir() / "done").touch()
            logger.info("All checkpoints are processed and uploaded to cloud.")



    def _compute_kl_with_reference(self, ckpt, seed: int) -> float:

        # ------ STEP 1: generate the results using the vLLM server -------
        vllm_server = self.vllm_server_lazy.construct(seed=seed)
        hf_ckpt_path = ckpt / "hf_pretrained"

        ckpt_eval_root_dir = self.get_analysis_root_dir() / ckpt.name
        ckpt_eval_root_dir.mkdir(exist_ok=True, parents=True)
        results_path = ckpt_eval_root_dir / "inference_results"

        # save the tokenizer for the vllm if it does not exist
        if not (hf_ckpt_path / "tokenizer.json").exists():
            logger.info(f"Tokenizer does not exist in {hf_ckpt_path}, saving it for vllm.")
            self.tokenizer.save_pretrained(hf_ckpt_path)

        server_url = vllm_server.start_server(
            hf_ckpt_path_or_model=str(hf_ckpt_path),
            wait_for_response=True,
            log_path=results_path.parent / f"{results_path.stem}.vllm_log",
            timeout=800,
        )

        guidance_llm_kwargs = {
            "api_base": server_url,
            "model": str(hf_ckpt_path),
        }

        # initialize the inference strategy with the vLLM server URL
        inference_strategy_lazy = copy.deepcopy(self.inference_strategy_lazy)
        inference_strategy_lazy._params['guidance_llm'].update(guidance_llm_kwargs)
        infer_strategy = inference_strategy_lazy.construct(
            result_dir=results_path.parent / f"{results_path.stem}.infer_strategy",
            seed=seed,
            cloud_logger=self.cloud_logger,
        )

        ds = self.task.get_datasets(split="train", no_cache=True)
        logger.info(f"Original dataset size: {len(ds)}, limiting to 500 examples for inference.")
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(500, len(ds)))
        ds = ds.select(indices)
        logger.info(f"Dataset size after sampling: {len(ds)}")

        results = infer_strategy.generate(ds)
        results.save_to_disk(str(results_path))
        vllm_server.stop_server()

        ds = Dataset.load_from_disk(str(results_path))

        # ------ STEP 2: convert the dataset to input ids -------
        episodes = self.extract_episodes_from_ds(
            inference_results=ds,
        )
        episodes_lst = [self._convert_to_dict(e) for e in episodes]
        ds = Dataset.from_list(episodes_lst)
        # drop advantages, if not the datacollator will complain
        ds = ds.remove_columns(["advantages"])

        # ------ STEP 3: compute the KL divergence with the reference dataset with deep speed engines -------
        actor = self._init_actor(ckpt)
        device = actor.device
        actor.eval()
        if isinstance(actor, DeepSpeedEngine):
            assert actor.zero_optimization_stage() == 0, "Zero stage must be 0"  # todo(milad): why?

        ds = self.trainer._update_episodes_with_log_probs(model_engine=actor,
                                                          dataset=ds,
                                                          column_name=COLUMN_ACTOR_SHIFTED_LOGPS,
                                                          )

        self.trainer._destroy_ds_engine(actor)
        del actor
        release_memory()

        ref = self.trainer._init_reference_model()
        ref.eval()
        if isinstance(ref, DeepSpeedEngine):
            assert ref.zero_optimization_stage() == 0, "Zero stage must be 0"

        ds = self.trainer._update_episodes_with_log_probs(model_engine=ref,
                                                          dataset=ds,
                                                          column_name=COLUMN_REF_SHIFTED_LOGPS,
                                                          )
        self.trainer._destroy_reference_engine(ref)
        del ref
        release_memory()

        # ------ STEP 4: compute the KL divergence -------

        data_loader = prepare_data_loader_for_inference(
            ds,
            per_device_batch_size=self.trainer.args.per_device_train_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.trainer.args.dataloader_num_workers,
                "pin_memory": self.trainer.args.dataloader_pin_memory,
            },
        )

        total_kls = []
        for inputs in tqdm(
            data_loader, desc="Computing KL", disable=not self.trainer._is_main_process()
        ):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
            attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
            labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)

            shifted_labels = labels[..., 1:].contiguous()  # Shape: (batch_size, max_seq_len-1)
            shifted_labels_mask = (shifted_labels != -100).to(attention_mask.dtype)  # Shape: (batch_size, max_seq_len-1
            shifted_actor_logprobs = inputs[COLUMN_ACTOR_SHIFTED_LOGPS]
            shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            with torch.no_grad():
                ref_kl = self.trainer._compute_kl_penalty(
                    shifted_actor_logprobs,
                    shifted_ref_logprobs,
                    estimation_type='kl',
                )

            total_kls += (ref_kl * shifted_labels_mask).sum(dim=1).tolist()

        return sum(total_kls) / len(total_kls)

    def _init_actor(
        self, ckpt: Optional[Path]
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if ckpt is not None:
            # Patch the trainer to construct the model from the checkpoint
            assert "hf_model_name" in self.trainer.actor_lazy._params
            self.trainer.actor_lazy._params["hf_model_name"] = str(
                ckpt / "hf_pretrained"
            )
        else:
            init_model_name = self.trainer.actor_lazy._params["hf_model_name"]
            logger.info(f"Initializing actor model from {init_model_name}")

        actor = self.trainer._init_actor_model()
        actor.train()

        return actor

    def extract_episodes_from_ds(
        self,
        inference_results: Dataset,
    ):
        # taken from MathRestEMEpisodeGenerator
        episodes_dict = {}
        encountered_question_indices = []
        for instance in tqdm(inference_results, desc="Extracting episodes"):

            tree = json.loads(instance["_treetune__reasoning_tree"])

            idx = instance["_treetune__idx"]
            assert idx not in encountered_question_indices, f"Question {idx} is encountered more than once."
            encountered_question_indices.append(idx)

            paths = TreeEpisodeUtils().extract_paths_from_tree(tree)
            for path in paths:
                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths. just query and response."

                finish_reason = path["node_chain"][-1]["finish_reason"]
                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text):]

                if finish_reason != "length":
                    # Generation stopped because the model hit <eos>
                    reward, is_unfinished_response = self.reward_function(
                        query_text, response_text, instance
                    )
                else:
                    # Generation stopped because the model hit the `max_tokens` limit
                    reward = self.reward_function.get_unfinished_response_penalty()
                    is_unfinished_response = True

                try:
                    query_token_ids, response_token_ids = (
                            self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            # Only append EOS token if the response is complete
                            allow_append_eos=not is_unfinished_response,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to tokenize query and response for instance {instance['_treetune__idx']}: {e}"
                    )
                    logger.error(f"Query: {query_text}")
                    logger.error(f"Response: {response_text}")
                    continue

                if self.max_sequence_length is not None:
                    seq_len = len(query_token_ids) + len(response_token_ids)
                    if seq_len > self.max_sequence_length:
                        logger.warning(
                            f"Sequence length {seq_len} is greater than "
                            f"max sequence length {self.max_sequence_length}."
                        )

                        # Truncate the response
                        response_token_ids = response_token_ids[
                                             : self.max_sequence_length - len(query_token_ids)
                                             ]
                        reward = self.reward_function.get_unfinished_response_penalty()
                        is_unfinished_response = True

                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=reward,
                )

                episodes_dict.setdefault(idx, []).append(episode)

        episodes = []
        for idx, eps in episodes_dict.items():
            episodes += eps

        return episodes

    # I copied and pasted the following methods from MathEpisodeGenerator as I have never heard of the DRY principle.
    # life is too short to follow DRY principle everywhere.
    # As Udi Dahan once said in 97 Things Every Programmer Should Know:
    # The fact that two wildly different parts of the system performed some logic in the same way meant less than I thought.
    # Up until I had pulled out those libraries of shared code, these parts were not dependent on each other.
    # Each could evolve independently. Each could change its logic to suit the needs of the system’s changing business environment.
    # Those four lines of similar code were accidental—a temporal anomaly, a coincidence.

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

    def _tokenize_query_and_response(
        self, query: str, response: str, allow_append_eos: bool = True
    ) -> Tuple[List[int], List[int]]:
        # This a legacy method that is not used anymore. It is kept here for reference.
        return self._tokenize_trajectory(
            {"query_text": query, "response_text": response},
            is_unfinished_response=not allow_append_eos,
            return_offsets=False,
        )

    def _tokenize_trajectory(
        self,
        trajectory: Dict[str, Any],
        is_unfinished_response: bool = False,
        return_offsets: bool = False,
        safety_check_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tuple[List[int], List[int]],
        Tuple[List[int], List[int], List[Tuple[int, int]]],
    ]:
        safety_check_kwargs = safety_check_kwargs or {}
        query_text = trajectory["query_text"]
        response_text = trajectory["response_text"]

        episode_text = f"{query_text}{response_text}"
        episode_encoding = self.tokenizer(
            episode_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = episode_encoding["input_ids"]
        offsets = episode_encoding["offset_mapping"]

        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query_text)
        )
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        self._safety_check_tokenization(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            query=query_text,
            response=response_text,
            episode_text=episode_text,
            **safety_check_kwargs,
        )

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if not is_unfinished_response and self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        if return_offsets:
            return query_token_ids, response_token_ids, offsets
        else:
            return query_token_ids, response_token_ids

    def _safety_check_tokenization(
        self,
        query_token_ids: List[str],
        response_token_ids: List[str],
        query: str,
        response: str,
        episode_text: str,
        check_query_reconstruction: bool = True,
        check_response_reconstruction: bool = True,
    ):
        decoding_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        decoded_instance = self.tokenizer.decode(
            query_token_ids + response_token_ids, **decoding_kwargs
        )
        assert decoded_instance == episode_text, (
            f"Decoded instance does not match original instance.\n"
            f"Original instance: {episode_text}\n"
            f"Decoded instance: {decoded_instance}"
        )

        if check_query_reconstruction:
            decoded_query = self.tokenizer.decode(query_token_ids, **decoding_kwargs)
            assert decoded_query == query, (
                f"Decoded query does not match original query.\n"
                f"Original query: {query}\n"
                f"Decoded query: {decoded_query}"
            )

        if check_response_reconstruction:
            decoded_response = self.tokenizer.decode(
                response_token_ids, **decoding_kwargs
            )
            assert decoded_response == response, (
                f"Decoded response does not match original response.\n"
                f"Original response: {response}\n"
                f"Decoded response: {decoded_response}"
            )

    def _convert_to_dict(self, episode_obj) -> Dict[str, Any]:
        if isinstance(episode_obj, dict):
            return episode_obj

        return asdict(episode_obj)
