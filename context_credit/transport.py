"""Map actor rows to the physical visits used for continuation sharing."""

import numpy as np


def record_transition(batch, step, dones, infos, max_steps, env_name):
    raw_done = np.asarray(dones, dtype=bool)
    terminal = np.zeros(len(infos), dtype=bool)
    truncated = np.zeros(len(infos), dtype=bool)
    for i, info in enumerate(infos):
        finished = bool(info.get("won", False)) or bool(info.get("lost", False))
        timeout = bool(info.get("TimeLimit.truncated", info.get("truncated", False)))
        budget_end = step + 1 == max_steps and (not raw_done[i] or env_name.lower().startswith("alfworld"))
        truncated[i] = ((timeout and raw_done[i]) or budget_end) and not finished
        terminal[i] = raw_done[i] and not truncated[i]
    batch.non_tensor_batch.update(
        context_step_idx=np.full(len(infos), step, dtype=np.int32),
        context_env_done=terminal, context_truncated=truncated,
        context_executed_action=np.asarray([info["context_executed_action"] for info in infos], dtype=object))


def physical_rows(data):
    nt = data.non_tensor_batch
    fields = ("uid", "traj_uid", "anchor_obs", "rewards", "context_step_idx",
              "context_env_done", "context_truncated", "context_executed_action")
    selected, seen = [], {}
    inverse = np.empty(len(nt["uid"]), dtype=np.int64)
    tensors = [data.batch[k].detach().cpu().numpy()
               for k in ("responses", "response_mask", "step_rewards", "token_level_rewards")]
    for row, (trajectory, step) in enumerate(zip(nt["traj_uid"], nt["context_step_idx"])):
        key = str(trajectory), int(step)
        if key not in seen:
            seen[key] = len(selected)
            selected.append(row)
        else:
            first = selected[seen[key]]
            if (any(not np.array_equal(nt[k][row], nt[k][first]) for k in fields)
                    or any(not np.array_equal(x[row], x[first]) for x in tensors)):
                raise ValueError("Conflicting copies of the same physical visit")
        inverse[row] = seen[key]
    return np.asarray(selected, dtype=np.int64), inverse
