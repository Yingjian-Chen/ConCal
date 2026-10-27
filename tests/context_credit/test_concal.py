"""Estimator and advantage tests using GiGPO's native normalization functions."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
import uuid

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from context_credit.core import compute_context_return
from context_credit.common_scale import actor_row_weights, common_scale_step_norm
from context_credit.transport import physical_rows, record_transition
from context_credit.adapter import apply_context_credit


@pytest.fixture
def native(monkeypatch):
    # Load the unchanged functions without importing the distributed trainer.
    path = ROOT / "gigpo/core_gigpo.py"
    names = {"to_hashable", "summarize_group_size", "are_similar", "build_step_group",
             "step_norm_reward", "episode_norm_reward", "compute_gigpo_outcome_advantage"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ModuleType("gigpo.core_gigpo")
    module.__dict__.update(torch=torch, np=np, defaultdict=defaultdict, Counter=Counter,
        SequenceMatcher=SequenceMatcher, uuid=uuid, List=List, Dict=Dict, Any=Any)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    package = ModuleType("gigpo")
    package.core_gigpo = module
    monkeypatch.setitem(sys.modules, "gigpo", package)
    monkeypatch.setitem(sys.modules, "gigpo.core_gigpo", module)
    return module


def visits():
    return dict(task_uid=np.array(["task"] * 4), traj_uid=np.array(["a", "a", "b", "b"]),
        step_idx=np.array([0, 1, 0, 1]), anchor_obs=np.array(["start", "room", "start", "room"]),
        executed_actions=np.array(["enter", "finish", "enter", "wait"]),
        rewards=np.array([0., 10., 0., 0.]), env_done=np.array([False, True, False, True]),
        truncated=np.zeros(4, dtype=bool), max_steps=2, gamma=0.95)


def test_visit_specific_sharing():
    result = compute_context_return(**visits())
    assert result.q[1] == 10. and result.q[3] == 0.
    np.testing.assert_allclose(result.q[[0, 2]], 0.95 * np.array([20/3, 10/3]))
    assert result.all_row_values[1, [1, 3]].mean() == pytest.approx(5.)
    assert result.all_row_values[1, 1] - result.all_row_values[1, 3] == pytest.approx(10/3)


def test_incoming_actions_define_pools():
    args = visits()
    args["executed_actions"][2] = "other"
    result = compute_context_return(**args)
    np.testing.assert_array_equal(result.q, result.mc_returns)


def test_terminal_and_collection_boundaries():
    args = visits()
    args["env_done"][3] = False
    args["truncated"][3] = True
    result = compute_context_return(**args)
    assert result.next_index[1] == result.next_index[3] == -1
    np.testing.assert_array_equal(result.q, compute_context_return(**visits()).q)
    args["env_done"][0] = True
    with pytest.raises(ValueError, match="later visit"):
        compute_context_return(**args)


def test_row_order_does_not_change_values():
    args = visits()
    expected = compute_context_return(**args)
    order = np.array([3, 0, 2, 1])
    shuffled = {k: v[order] if isinstance(v, np.ndarray) else v for k, v in args.items()}
    result = compute_context_return(**shuffled)
    np.testing.assert_array_equal(result.q, expected.q[order])
    np.testing.assert_array_equal(result.all_row_values, expected.all_row_values[:, order])


def frame(native):
    args = visits()
    result = compute_context_return(**args)
    mask = torch.tensor([[0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.float32)
    token_rewards = torch.tensor([[0., 10., 0.], [0., 10., 0.], [0., 0., 0.], [0., 0., 0.]])
    raw = torch.from_numpy(result.mc_returns.astype(np.float32))
    native_adv, _ = native.compute_gigpo_outcome_advantage(token_rewards, raw, mask,
        args["anchor_obs"], args["task_uid"], args["traj_uid"], mode="mean_norm")
    nt = dict(uid=args["task_uid"], traj_uid=args["traj_uid"], anchor_obs=args["anchor_obs"],
        rewards=args["rewards"], context_step_idx=args["step_idx"],
        context_env_done=args["env_done"], context_truncated=args["truncated"],
        context_executed_action=args["executed_actions"], is_action_valid=np.ones(4, dtype=bool))
    batch = dict(responses=torch.ones((4, 3), dtype=torch.long), response_mask=mask,
        step_rewards=raw, token_level_rewards=token_rewards, advantages=native_adv, returns=native_adv.clone())
    return SimpleNamespace(batch=batch, non_tensor_batch=nt, meta_info={})


def test_transport_copies_are_counted_once(native):
    data = frame(native)
    order = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    data.batch = {k: v[order] for k, v in data.batch.items()}
    data.non_tensor_batch = {k: v[order] for k, v in data.non_tensor_batch.items()}
    rows, inverse = physical_rows(data)
    np.testing.assert_array_equal(rows, np.arange(4))
    np.testing.assert_array_equal(inverse, order)
    data.non_tensor_batch["rewards"][4] = 3.
    with pytest.raises(ValueError, match="Conflicting copies"):
        physical_rows(data)


@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
def test_common_scale_preserves_gaps_and_reference_rms(native, mode):
    raw = torch.tensor([-1., 1., -5., 5.])
    mask = torch.tensor([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 0.]])
    groups = np.array(["a", "a", "b", "b"])
    weights = actor_row_weights(mask.numpy(), micro_batch_size=2, world_size=1, loss_agg_mode="token-mean")
    result, coefficient = common_scale_step_norm(raw, mask, groups, mode, weights)
    centered = native.step_norm_reward(raw, mask, groups, remove_std=True)
    reference = native.step_norm_reward(raw, mask, groups, remove_std=mode == "mean_norm")
    assert torch.count_nonzero(result[mask == 0]) == 0
    assert result[2, 0] / result[0, 0] == pytest.approx(5.)
    rms = lambda x: np.sqrt(np.average(x[:, 0].numpy().astype(float)**2, weights=weights))
    assert rms(result) == pytest.approx(rms(reference), rel=1e-7)
    expected = torch.from_numpy((coefficient * centered[:, 0].numpy().astype(float)).astype(np.float32))[:, None] * mask
    assert torch.equal(result, expected)


def test_zero_step_credit(native):
    raw = torch.zeros(4)
    mask = torch.ones((4, 2))
    groups = np.array(["a", "a", "b", "b"])
    result, coefficient = common_scale_step_norm(raw, mask, groups, "mean_std_norm", np.ones(4))
    assert coefficient == 0. and torch.count_nonzero(result) == 0


def test_disabled_and_zero_weight_keep_native_advantages(native):
    data = frame(native)
    expected = data.batch["advantages"].clone()
    apply_context_credit(data, gamma=.95, context_max_steps=2, context_config={"enabled": False})
    assert torch.equal(expected, data.batch["advantages"])
    expected, _ = native.compute_gigpo_outcome_advantage(
        data.batch["token_level_rewards"], data.batch["step_rewards"], data.batch["response_mask"],
        data.non_tensor_batch["anchor_obs"], data.non_tensor_batch["uid"],
        data.non_tensor_batch["traj_uid"], mode="mean_norm", step_advantage_w=0.)
    data.batch["advantages"] = expected.clone()
    apply_context_credit(data, gamma=.95, context_max_steps=2, step_advantage_w=0.,
        context_micro_batch_size=2, context_world_size=1,
        context_config={"enabled": True, "local_normalization": "common_scale"})
    assert torch.equal(expected, data.batch["advantages"])


@pytest.mark.parametrize("env", ["alfworld/AlfredTWEnv", "Webshop"])
def test_recorded_boundary_matches_previous_collector(env):
    infos = [{"won": True, "context_executed_action": "finish"},
             {"TimeLimit.truncated": True, "context_executed_action": "wait"},
             {"context_executed_action": "wait"}]
    data = SimpleNamespace(non_tensor_batch={})
    record_transition(data, 1, [True, True, False], infos, 2, env)
    np.testing.assert_array_equal(data.non_tensor_batch["context_env_done"], [True, False, False])
    np.testing.assert_array_equal(data.non_tensor_batch["context_truncated"], [False, True, True])
