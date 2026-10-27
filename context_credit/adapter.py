"""Use ConCal estimates in GiGPO's advantage computation."""

from time import perf_counter

import numpy as np
import torch

from .common_scale import actor_row_weights, common_scale_step_norm
from .core import compute_context_return
from .transport import physical_rows


def validate_runtime_config(config):
    block = config.algorithm.get("context_credit", {})
    if not block.get("enabled", False):
        return None
    if config.algorithm.adv_estimator != "gigpo" or config.algorithm.gigpo.enable_similarity:
        raise ValueError("ConCal requires GiGPO with exact state groups")
    if block.get("pooling_mode", "actual_prior") != "actual_prior":
        raise ValueError("ConCal uses actual_prior continuation estimates")
    normalization = block.get("local_normalization", "common_scale")
    if normalization not in ("native", "common_scale"):
        raise ValueError(f"Unknown step normalization: {normalization}")
    if normalization == "common_scale" and config.algorithm.gigpo.mode == "mean_std_norm":
        actor = config.actor_rollout_ref.actor
        if (actor.use_dynamic_bsz or actor.shuffle or actor.ulysses_sequence_parallel_size != 1
                or actor.ppo_micro_batch_size_per_gpu is None):
            raise ValueError("Common-scale actor weights require static, unshuffled microbatches")
    return block


@torch.no_grad()
def apply_context_credit(data, *, gamma, context_config, context_max_steps,
                         gigpo_mode="mean_norm", step_advantage_w=1.0,
                         context_invalid_penalty=0.0, context_global_step=0,
                         context_loss_agg_mode="token-mean", context_micro_batch_size=8,
                         context_world_size=4, **kwargs):
    from gigpo import core_gigpo

    if not context_config.get("enabled", False):
        return data
    started = perf_counter()
    nt, batch = data.non_tensor_batch, data.batch
    rows, inverse = physical_rows(data)
    result = compute_context_return(
        task_uid=nt["uid"][rows], traj_uid=nt["traj_uid"][rows],
        step_idx=nt["context_step_idx"][rows], anchor_obs=nt["anchor_obs"][rows],
        executed_actions=nt["context_executed_action"][rows], rewards=nt["rewards"][rows],
        env_done=nt["context_env_done"][rows], truncated=nt["context_truncated"][rows],
        max_steps=context_max_steps, gamma=gamma)

    old_raw, mask = batch["step_rewards"], batch["response_mask"]
    penalty = context_invalid_penalty * (~np.asarray(nt["is_action_valid"][rows], dtype=bool))
    if not np.allclose(old_raw.cpu().numpy(), result.mc_returns[inverse] - penalty[inverse],
                       atol=2e-5, rtol=3e-6):
        raise ValueError("Native step returns do not match the recorded trajectories")
    correction = torch.as_tensor(result.delta[inverse], dtype=old_raw.dtype, device=old_raw.device)
    repaired = old_raw + correction
    groups = core_gigpo.build_step_group(nt["anchor_obs"], nt["uid"], False)
    step_advantages = core_gigpo.step_norm_reward(
        repaired, mask, groups, remove_std=gigpo_mode == "mean_norm")
    normalization = context_config.get("local_normalization", "common_scale")
    coefficient = 1.0
    if normalization == "common_scale" and gigpo_mode == "mean_std_norm":
        weights = actor_row_weights(mask.cpu().numpy(), micro_batch_size=context_micro_batch_size,
                                    world_size=context_world_size, loss_agg_mode=context_loss_agg_mode)
        step_advantages, coefficient = common_scale_step_norm(
            repaired, mask, groups, gigpo_mode, weights, native_micro=step_advantages)

    native = batch["advantages"]
    if step_advantage_w == 0:
        advantages = native
    else:
        episode = core_gigpo.episode_norm_reward(
            batch["token_level_rewards"], mask, nt["uid"], nt["traj_uid"],
            remove_std=gigpo_mode == "mean_norm")
        advantages = episode + float(step_advantage_w) * step_advantages
        if normalization != "common_scale" or gigpo_mode == "mean_norm":
            # Preserve native FP32 targets in groups whose raw values did not change.
            changed = set(groups[(repaired != old_raw).cpu().numpy()])
            changed = torch.as_tensor([g in changed for g in groups], device=mask.device)
            advantages = torch.where(changed[:, None], advantages, native)
    if not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite ConCal advantage")
    batch["advantages"], batch["returns"] = advantages, advantages.clone()
    data.meta_info["context_credit_metrics"] = {
        "concal/physical_visits": float(len(rows)),
        "concal/mean_abs_correction": float(np.abs(result.delta).mean()),
        "concal/common_scale": coefficient,
        "concal/seconds": perf_counter() - started,
    }

    interval = context_config.get("dump_every", 0)
    if interval > 0 and context_global_step % interval == 0:
        from .diagnostics import save_batch
        save_batch(data, result, rows, inverse, repaired, step_advantages, coefficient,
                   context_config["audit_dir"], context_global_step)
    return data
