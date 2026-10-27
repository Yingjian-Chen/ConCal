"""Optional rollout snapshots for offline analysis."""

from pathlib import Path

import numpy as np


def save_batch(data, result, rows, inverse, repaired, step_advantages, coefficient,
               directory, step):
    nt, batch = data.non_tensor_batch, data.batch
    payload = {
        "q": result.q, "mc_returns": result.mc_returns,
        "context_id": result.context_id, "context_source_count": result.context_source_count,
        "physical_index": rows, "inverse_mapping": inverse,
        "common_scale_coefficient": coefficient,
    }
    for source, target in [("uid", "task_uid"), ("traj_uid", "traj_uid"),
                           ("anchor_obs", "anchor_obs"), ("rewards", "rewards"),
                           ("context_step_idx", "step_idx"), ("context_env_done", "env_done"),
                           ("context_truncated", "truncated"),
                           ("context_executed_action", "executed_actions")]:
        values = np.asarray(nt[source][rows].tolist())
        payload[target] = values
    for key in ("responses", "response_mask", "step_rewards", "token_level_rewards", "advantages", "returns"):
        payload[key] = batch[key].detach().cpu().numpy()
    payload["repaired_raw"] = repaired.detach().cpu().numpy()
    payload["step_advantages"] = step_advantages.detach().cpu().numpy()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"step_{step:06d}.npz", **payload)
