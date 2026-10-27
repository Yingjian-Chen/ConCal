"""Recursive continuation calibration over recorded visits."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class ContextReturnResult:
    q: np.ndarray
    mc_returns: np.ndarray
    delta: np.ndarray
    all_row_values: np.ndarray
    next_index: np.ndarray
    remaining: np.ndarray
    context_id: np.ndarray
    context_source_count: np.ndarray
    context_weight: np.ndarray


def _key(value):
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__, value


def compute_context_return(*, task_uid, traj_uid, step_idx, anchor_obs,
                           rewards, env_done, truncated, max_steps,
                           executed_actions, gamma=0.95,
                           pooling_mode="actual_prior"):
    """Estimate Q from each action's reward and calibrated actual successor."""
    if pooling_mode != "actual_prior":
        raise ValueError("ConCal uses the actual_prior estimator")
    if max_steps < 1 or not 0 <= gamma <= 1:
        raise ValueError("Expected a positive interaction limit and gamma in [0, 1]")

    r = np.asarray(rewards, dtype=np.float64)
    steps = np.asarray(step_idx, dtype=np.int64)
    done = np.asarray(env_done, dtype=bool)
    cut = np.asarray(truncated, dtype=bool)
    n = len(r)
    arrays = (task_uid, traj_uid, anchor_obs, executed_actions, steps, done, cut)
    if not n or any(len(a) != n for a in arrays):
        raise ValueError("Visit arrays must have the same nonzero length")
    if not np.isfinite(r).all() or np.any(steps < 0) or np.any(steps >= max_steps):
        raise ValueError("Invalid reward or visit step")
    if np.any(done & cut):
        raise ValueError("A visit cannot be both terminal and truncated")

    tasks = [_key(x) for x in task_uid]
    trajectories = [_key(x) for x in traj_uid]
    observations = [_key(x) for x in anchor_obs]
    chains = defaultdict(list)
    for i in range(n):
        chains[tasks[i], trajectories[i]].append(i)

    trajectory_id = np.empty(n, dtype=np.int64)
    next_index = np.full(n, -1, dtype=np.int64)
    previous = np.full(n, -1, dtype=np.int64)
    mc = np.zeros(n, dtype=np.float64)
    for tid, key in enumerate(sorted(chains, key=repr)):
        rows = sorted(chains[key], key=lambda i: steps[i])
        if not np.array_equal(steps[rows], np.arange(len(rows))):
            raise ValueError("Each trajectory must contain consecutive visits from step 0")
        if np.any(done[rows[:-1]] | cut[rows[:-1]]):
            raise ValueError("A stopped trajectory has a later visit")
        last = rows[-1]
        if not (done[last] or cut[last] or steps[last] == max_steps - 1):
            raise ValueError("A trajectory must end at a terminal or collection boundary")
        trajectory_id[rows] = tid
        next_index[rows[:-1]] = rows[1:]
        previous[rows[1:]] = rows[:-1]
        for i in reversed(rows):
            successor = next_index[i]
            mc[i] = r[i] + (gamma * mc[successor] if successor >= 0 else 0.0)

    contexts = []
    for i, p in enumerate(previous):
        history = ("start",) if p < 0 else ("past", observations[p], executed_actions[p])
        contexts.append((tasks[i], observations[i], history))
    pools = {key: c for c, key in enumerate(sorted(set(contexts), key=repr))}
    context_id = np.array([pools[key] for key in contexts], dtype=np.int64)
    visits = np.bincount(context_id, minlength=len(pools))
    sources = [set() for _ in pools]
    for i, c in enumerate(context_id):
        sources[c].add(trajectory_id[i])
    source_count = np.array([len(s) for s in sources], dtype=np.int64)
    alpha = source_count / (source_count + 1.0)

    # Keep a canonical reduction order when the transported rows are reordered.
    order = np.array(sorted(range(n), key=lambda i: (
        repr(tasks[i]), repr(observations[i]), trajectory_id[i], steps[i])), dtype=np.int64)
    advancing = next_index >= 0
    values = np.zeros((max_steps + 1, n), dtype=np.float64)
    for b in range(1, max_steps + 1):
        own = r.copy()
        own[advancing] += gamma * values[b - 1, next_index[advancing]]
        mean = np.bincount(context_id[order], weights=own[order],
                           minlength=len(pools)) / visits
        values[b] = own + alpha[context_id] * (mean[context_id] - own)

    remaining = max_steps - steps
    continuation = np.zeros(n, dtype=np.float64)
    continuation[advancing] = values[remaining[advancing] - 1, next_index[advancing]]
    q = r + gamma * continuation
    if not np.isfinite(values).all() or not np.isfinite(q).all() or not np.isfinite(mc).all():
        raise ValueError("Nonfinite continuation estimate")
    return ContextReturnResult(q, mc, q - mc, values, next_index, remaining,
                               context_id, source_count, alpha)
