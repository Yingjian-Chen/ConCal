# ConCal
Code for **Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning**.

In long-horizon agentic RL, methods such as GiGPO compare actions at shared states, but value estimates based on a single sampled continuation remain sensitive to later decisions. ConCal (Continuation Calibration) addresses this by sharing continuation evidence within a rollout batch. It matches same-task visits by their incoming state-action-state transitions, combines shared evidence with each visit’s own transition estimate, and recursively propagates calibrated values backward to refine earlier action-value estimates. This reduces dependence on individual continuations and mitigates local credit distortions without additional rollouts.

## 1. Setup

Use separate Conda environments for ALFWorld and WebShop. Run the following commands from the repository root.

### ALFWorld

Install the training environment:

```bash
conda create -n concal-alfworld python=3.12 -y
conda activate concal-alfworld

pip install vllm==0.11.0 "ray[default]==2.50.0" \
  tensordict==0.10.0 transformers==4.57.3
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
pip install -e . --no-deps
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld==0.4.2
```

Download ALFWorld data and prepare the trainer inputs:

```bash
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download -f

mkdir -p data/verl-agent/text
python examples/data_preprocess/prepare.py --mode text \
  --local_dir ./data/verl-agent \
  --train_data_size 16 --val_data_size 128
```

### WebShop

Install WebShop and download the 1,000-product dataset:

```bash
conda create -n concal-webshop python=3.10 -y
conda activate concal-webshop

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pushd agent_system/environments/env_package/webshop/webshop
bash setup.sh -d small
popd
```

Install the training dependencies and prepare the trainer inputs:

```bash
pip install vllm==0.8.4 "ray[default]==2.58.0" \
  tensordict==0.8.3 transformers==4.51.1 typer==0.9.4
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e . --no-deps

mkdir -p data/verl-agent/webshop/text
python examples/data_preprocess/prepare.py --mode text \
  --local_dir ./data/verl-agent/webshop \
  --train_data_size 16 --val_data_size 256
```

The generated Parquet files provide text placeholders for the trainer. Actual tasks and observations come from the environments.

Download the base model from either configured environment:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
```

## 2. Training
The scripts default to four GPUs and seed 123. Run `wandb login` before training, or set `WANDB_MODE=offline`.

### ALFWorld

```bash
conda activate concal-alfworld
export SKIP_CONDA_ACTIVATE=1
export ALFWORLD_DATA="$HOME/.cache/alfworld"
bash run_concal_alfworld.sh
```

### WebShop

```bash
conda activate concal-webshop
export SKIP_CONDA_ACTIVATE=1
bash run_concal_webshop.sh
```

To use another seed, run `SEED=456 bash run_concal_alfworld.sh` or `SEED=456 bash run_concal_webshop.sh`.

## 3. Results

Success rates are reported as mean ± sample standard deviation across three training runs.

| Algorithm | Task | Model | Success Rate (%) | Training Log |
| --- | --- | --- | ---: | :---: |
| ConCal | ALFWorld | Qwen2.5-1.5B-Instruct | 94.01 ± 0.45 | [![W&B][wandb-badge]][alfworld-log] |
| ConCal | ALFWorld | Qwen2.5-7B-Instruct | - | ongoing |
| ConCal | WebShop | Qwen2.5-1.5B-Instruct | 79.69 ± 3.41 | [![W&B][wandb-badge]][webshop-log] |
| ConCal | WebShop | Qwen2.5-7B-Instruct | - | ongoing |

[wandb-badge]: https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb
[alfworld-log]: https://forge.coreweave.com/wandb/yingjian-test/gigpo/reports/Training-Logs-on-ALFWorld--VmlldzoxODE4OTg3Ng?accessToken=1jdis1u5i74tnkj55kadoeoacv73l5medwvmfk1ie4s3sbct14v99uijx9rex3u3
[webshop-log]: https://forge.coreweave.com/wandb/yingjian-test/gigpo/reports/Training-Logs-on-WebShop--VmlldzoxODE4OTk2OA?accessToken=gr0obj06bxn3roumxhecixocj33gu6v5szy0whxuukextaw4txgi8v0t512qxw77

Built on [verl-agent/GiGPO](https://github.com/langfengQ/verl-agent), with environments from [ALFWorld](https://github.com/alfworld/alfworld) and [WebShop](https://github.com/princeton-nlp/WebShop).
