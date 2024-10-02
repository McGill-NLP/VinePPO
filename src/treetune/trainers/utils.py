from typing import Tuple, Iterator, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, DataLoader


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


@torch.no_grad()
def get_global_statistics(
    accelerator, xs: torch.Tensor, mask=None, device="cpu"
) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor(
        [xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device
    )
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.to(device)


@torch.no_grad()
def get_global_statistics_no_move(
    dist, xs: torch.Tensor, mask=None, unbiased=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """

    if mask is not None:
        sum_and_count = torch.tensor([(xs * mask).sum(), mask.sum()], device=xs.device)
    else:
        sum_and_count = [xs.sum(), xs.numel()]

    dist.all_reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    dist.all_reduce(sum_var)
    global_var = sum_var / count

    if unbiased:
        if count == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = count / (count - 1)
        global_var = global_var * bessel_correction

    return global_mean, global_var


class RunningMoments:
    def __init__(self, accelerator, force_no_sync=False):
        """
        Calculates the running mean and standard deviation of a data stream. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.accelerator = accelerator
        self.force_no_sync = force_no_sync

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed and not self.force_no_sync:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return (
            xs_mean.item(),
            (xs_var * xs_count / (xs_count - 1)).float().sqrt().item(),
        )


class DeepSpeedRunningMoments:
    def __init__(self, force_no_sync=False):
        """
        Calculates the running mean and standard deviation of a data stream. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.force_no_sync = force_no_sync

        if self.force_no_sync:
            from deepspeed import comm as dist

            self.dist = dist
        else:
            self.dist = None

        self.device = None

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if not self.force_no_sync and self.dist.is_initialized():
            raise NotImplementedError()
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return (
            xs_mean.item(),
            (xs_var * xs_count / (xs_count - 1)).float().sqrt().item(),
        )

    @torch.no_grad()
    def get_global_statistics(
        self, xs: torch.Tensor, mask=None, device="cpu"
    ) -> Tuple[float, float, int]:
        """
        Computes element-wise mean and variance of the tensor across processes. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
        """
        if getattr(self, "device", None) is None:
            from deepspeed import get_accelerator

            self.device = torch.device(
                get_accelerator().device_name(), self.dist.get_local_rank()
            )

        xs = xs.to(self.device)
        sum_and_count = torch.tensor(
            [xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device
        )
        sum_and_count = self.dist.reduce(sum_and_count)
        global_sum, count = sum_and_count
        global_mean = global_sum / count

        sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
        sum_var = self.dist.reduce.reduce(sum_var)
        global_var = sum_var / count

        return global_mean.to(device), global_var.to(device), count.to(device)

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class MergedBatchSampler(Sampler):
    def __init__(self, *batch_samplers):
        super().__init__(None)
        self.batch_samplers = batch_samplers

    def __iter__(self):
        for batch_sampler in self.batch_samplers:
            for batch in batch_sampler:
                yield batch

    def __len__(self):
        return sum(len(batch_sampler) for batch_sampler in self.batch_samplers)


def logprobs_from_logits(logits, labels, gather=True):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values, mask, shift_mean=True, distributed=False, unbiased_variance=False
):
    """Whiten values with masked values."""
    from deepspeed import comm as dist

    if distributed and dist.is_initialized():
        mean, var = get_global_statistics_no_move(
            dist, values, mask=mask, unbiased=unbiased_variance
        )
    else:
        mean, var = masked_mean(values, mask), masked_var(
            values, mask, unbiased=unbiased_variance
        )
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_rescale_by_std(values, mask, distributed=False, unbiased_variance=False):
    """Whiten values with masked values."""
    from deepspeed import comm as dist

    if distributed and dist.is_initialized():
        mean, var = get_global_statistics_no_move(
            dist, values, mask=mask, unbiased=unbiased_variance
        )
    else:
        mean, var = masked_mean(values, mask), masked_var(
            values, mask, unbiased=unbiased_variance
        )
    whitened = values * torch.rsqrt(var + 1e-8)
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def skip_first_batches(dataloader_iter: Iterator[DataLoader], num_batches: int) -> iter:
    """
    Skips the first `num_batches` batches in a dataloader.
    """
    num_skipped = 0
    while num_skipped < num_batches:
        for _ in dataloader_iter:
            num_skipped += 1
            if num_skipped >= num_batches:
                break

    assert num_skipped == num_batches

    return dataloader_iter


def monitor_tensor_anomalies(
    tensor: torch.Tensor, mask: torch.Tensor, z_threshold: float = 3
) -> Dict[str, int]:
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    # Mask NaN and Inf values to compute mean and std of the finite values only
    finite_tensor = tensor[~nan_mask & ~inf_mask & mask]

    # Compute the mean and standard deviation of the finite values
    if finite_tensor.numel() == 0:
        mean = torch.tensor(float("nan"), device=tensor.device)
        std = torch.tensor(float("nan"), device=tensor.device)
    else:
        mean, std = masked_mean(tensor, mask), masked_var(tensor, mask, unbiased=False)

    # Handle case where std is 0
    if std == 0 or std.isnan():
        # If std is 0, all values are the same. No anomalies based on z-score.
        z_scores = torch.zeros_like(tensor)
    else:
        # Compute z-scores for finite values
        z_scores = (tensor - mean) / std

    # Check for NaN values
    num_nan = (nan_mask * mask).sum().item()

    # Check for infinity values
    num_inf = (inf_mask * mask).sum().item()

    # Check for values with high z-scores (anomalies)
    high_z_mask = z_scores.abs() > z_threshold
    num_high_z = (high_z_mask * mask).sum().item()

    # Total anomalies
    total_anomalies = num_nan + num_inf + num_high_z

    return {
        "num_nan": num_nan,
        "num_inf": num_inf,
        "num_high_z": num_high_z,
        "total_anomalies": total_anomalies,
    }


def close_to_zero_percentage(
    tensor: torch.Tensor, mask: torch.Tensor, threshold: float = 1e-8
) -> torch.Tensor:
    """
    Computes the percentage of values in the tensor that are close to zero.
    """
    close_to_zero_mask = torch.abs(tensor) < threshold
    num_close_to_zero = (close_to_zero_mask * mask).sum()
    total_values = mask.sum()
    return num_close_to_zero / total_values
