"""Batch-wise normalization of calibrated step values."""

import numpy as np
import torch


def actor_row_weights(response_mask, *, micro_batch_size, world_size, loss_agg_mode):
    counts = np.asarray(response_mask).sum(axis=1).astype(np.float64)
    if np.any(counts <= 0) or len(counts) % (micro_batch_size * world_size):
        raise ValueError("Expected nonempty responses in complete actor microbatches")
    if loss_agg_mode == "token-mean":
        chunks = counts.reshape(-1, micro_batch_size)
        return (chunks / chunks.sum(axis=1, keepdims=True)).reshape(-1)
    if loss_agg_mode == "seq-mean-token-mean":
        return np.ones(len(counts), dtype=np.float64)
    if loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        return counts
    raise ValueError(f"Unknown actor aggregation: {loss_agg_mode}")


@torch.no_grad()
def common_scale_step_norm(repaired_raw, response_mask, groups, mode,
                           transport_weights, native_micro=None):
    from gigpo import core_gigpo

    if native_micro is None:
        native_micro = core_gigpo.step_norm_reward(
            repaired_raw, response_mask, groups, remove_std=mode == "mean_norm")
    if mode == "mean_norm":
        return native_micro.detach(), 1.0
    if mode != "mean_std_norm":
        raise ValueError(f"Unknown normalization: {mode}")

    centered = core_gigpo.step_norm_reward(repaired_raw, response_mask, groups, remove_std=True)
    first = response_mask.long().argmax(dim=1, keepdim=True)
    reference = native_micro.gather(1, first).squeeze(1).cpu().numpy().astype(np.float64)
    centered = centered.gather(1, first).squeeze(1).cpu().numpy().astype(np.float64)
    rms = lambda x: float(np.sqrt(np.average(x * x, weights=transport_weights)))
    centered_rms = rms(centered)
    coefficient = rms(reference) / centered_rms if centered_rms > 0 else 0.0
    scaled = (coefficient * centered).astype(np.float32)
    result = torch.as_tensor(scaled, device=repaired_raw.device).unsqueeze(1) * response_mask
    return result, coefficient
