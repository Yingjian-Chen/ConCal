# ConCal

Continuation calibration for group-based agent reinforcement learning.

ConCal matches visits by task and incoming transition, recursively combines
each visit's transition estimate with its sharing-pool mean, and uses the
calibrated successor values in GiGPO's step advantages.

## Implementation

| File | Role |
| --- | --- |
| `context_credit/core.py` | Matching, recursive continuation values and action values |
| `context_credit/adapter.py` | Integration with GiGPO's state groups and episode advantages |
| `context_credit/common_scale.py` | Shared batch normalization scale |
| `context_credit/transport.py` | Recorded transitions and physical-visit mapping |
| `context_credit/diagnostics.py` | Optional frozen rollout snapshots |

The actor, rollout engine, PPO objective, KL loss and optimizer use verl-agent.
ALFWorld and WebShop share the same estimator. With WebShop's mean-only
reference normalization, the common coefficient is one.

## Setup

Follow the [verl-agent environment setup](https://github.com/langfengQ/verl-agent)
for ALFWorld and WebShop, and install this checkout with `pip install -e .`.
The reproduced runs use vLLM 0.11.0 for ALFWorld and 0.8.4 for WebShop in
separate environments. Prepare the Qwen2.5-1.5B-Instruct model and benchmark
datasets following the upstream instructions.

The root `requirements.txt` lists common dependencies. Benchmark-specific
environment dependencies follow the upstream installation instructions.

Set `PROJECT_ROOT` to the directory containing your `models`, `data` and `envs`
directories. `MODEL_DIR`, `DATA_DIR`, `CONDA_ENV` and `RUN_DIR` can also be set
individually. To use an already activated environment, set
`SKIP_CONDA_ACTIVATE=1`.

```bash
PROJECT_ROOT=/path/to/project SEED=123 bash run_concal_alfworld.sh
PROJECT_ROOT=/path/to/project SEED=123 bash run_concal_webshop.sh
```

Both scripts retain the experiment's 150 updates, 16 tasks and eight
trajectories per task. `N_GPUS` defaults to four. Use a new `RUN_DIR` for each
run. The WebShop entry uses the GraphGPO text-rich setting.

To resolve and check a configuration before training:

```bash
PROJECT_ROOT=/path/to/project PREPARE_VALIDATE_ONLY=1 bash run_concal_alfworld.sh
```

Set `DUMP_EVERY=5` to save a frozen batch every five updates under
`$RUN_DIR/rollouts`. Standard training metrics are logged through GiGPO.
Authenticate W&B through `wandb login` or `WANDB_API_KEY`, or set
`WANDB_MODE=offline` for offline logging.

## Tests

```bash
python -m pytest tests/context_credit/test_concal.py -q
```

## Acknowledgments

Built on [verl-agent/GiGPO](https://github.com/langfengQ/verl-agent), commit
`20bd331bdbc9026a5668e11362178e10ab7400c8`. Upstream copyright notices remain in the source files. The existing WebShop Ray dependency
initialization is retained for the experiment environments.
