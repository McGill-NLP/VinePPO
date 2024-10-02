from typing import Any, Dict, Optional

import torch
from accelerate.data_loader import BatchSamplerShard
from datasets import Dataset
from deepspeed import comm as dist
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, RandomSampler


def prepare_data_loader_for_training(
    dataset: Dataset,
    per_device_batch_size: int,
    seed: int,
    drop_last: bool = True,
    even_batches: bool = True,
    data_loader_kwargs: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    data_loader_kwargs = data_loader_kwargs or {}

    generator = torch.Generator()
    generator.manual_seed(seed)

    non_dist_batch_sampler = BatchSampler(
        RandomSampler(dataset, generator=generator),
        batch_size=per_device_batch_size,
        drop_last=drop_last,
    )
    dist_batch_sampler = BatchSamplerShard(
        non_dist_batch_sampler,
        num_processes=dist.get_world_size(),
        process_index=dist.get_rank(),
        split_batches=False,
        even_batches=even_batches,
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=dist_batch_sampler,
        **data_loader_kwargs,
    )
    return data_loader


def prepare_data_loader_for_inference(
    dataset: Dataset, per_device_batch_size: int, data_loader_kwargs: Dict[str, Any]
) -> DataLoader:
    non_dist_batch_sampler = BatchSampler(
        SequentialSampler(dataset), per_device_batch_size, False
    )
    dist_batch_sampler = BatchSamplerShard(
        non_dist_batch_sampler,
        num_processes=dist.get_world_size(),
        process_index=dist.get_rank(),
        split_batches=False,
        even_batches=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=dist_batch_sampler,
        **data_loader_kwargs,
    )
    return data_loader


def all_gather_different_sizes(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.size(0)], device=tensor.device)
    size_list = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max(size_list).item()

    pad_size = max_size - tensor.size(0)
    if pad_size > 0:
        padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)
        tensor_padded = pad(tensor, padding, "constant", 0)
    else:
        tensor_padded = tensor

    gathered_tensors_padded = [
        torch.zeros(
            max_size, *tensor.size()[1:], device=tensor.device, dtype=tensor.dtype
        )
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_tensors_padded, tensor_padded)

    gathered_tensors = []
    for i, size_tensor in enumerate(size_list):
        original_size = size_tensor.item()
        gathered_tensor = gathered_tensors_padded[i][:original_size]
        gathered_tensors.append(gathered_tensor)

    return gathered_tensors
