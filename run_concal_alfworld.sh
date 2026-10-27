#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_DIR}"
SEED="${SEED:-123}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/concal-main-alfworld-seed${SEED}}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/concal-alfworld-${SEED}-$$}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/verl-agent/text}"
N_GPUS="${N_GPUS:-4}"
DUMP_EVERY="${DUMP_EVERY:-0}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != 1 ]]; then
  set +u
  source "${CONDA_SH:-$PROJECT_ROOT/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-$PROJECT_ROOT/envs/gigpo-alfworld}"
  set -u
fi
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_num_server_call_thread=2 RAY_worker_num_grpc_internal_threads=1
export ALFWORLD_DATA="${ALFWORLD_DATA:-$PROJECT_ROOT/data/alfworld}"
unset RAY_ADDRESS
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
unset VLLM_ATTENTION_BACKEND

args=(
  algorithm.adv_estimator=gigpo
  algorithm.context_credit.enabled=true
  algorithm.context_credit.pooling_mode=actual_prior
  algorithm.context_credit.local_normalization=common_scale
  algorithm.context_credit.audit_dir="$RUN_DIR/rollouts"
  algorithm.context_credit.dump_every="$DUMP_EVERY"
  algorithm.use_kl_in_reward=False
  algorithm.filter_groups.enable=False
  algorithm.gamma=0.95
  algorithm.gigpo.mode=mean_std_norm
  algorithm.gigpo.step_advantage_w=1.0
  +ray_init.address=local
  +ray_init._temp_dir="$RAY_TEMP_DIR"
  ray_init.num_cpus=64
  "+ray_init.runtime_env.env_vars.VLLM_USE_V1='1'"
  "+ray_init.runtime_env.env_vars.VLLM_ENABLE_V1_MULTIPROCESSING='0'"

  "+ray_init.runtime_env.env_vars.OMP_NUM_THREADS='1'"
  "+ray_init.runtime_env.env_vars.OPENBLAS_NUM_THREADS='1'"
  "+ray_init.runtime_env.env_vars.MKL_NUM_THREADS='1'"
  "+ray_init.runtime_env.env_vars.NUMEXPR_NUM_THREADS='1'"
  "+ray_init.runtime_env.env_vars.RAY_num_server_call_thread='2'"
  "+ray_init.runtime_env.env_vars.RAY_worker_num_grpc_internal_threads='1'"
  data.train_files="$DATA_DIR/train.parquet"
  data.val_files="$DATA_DIR/test.parquet"
  data.train_batch_size=16
  ++data.seed="$SEED"
  data.val_batch_size=128
  data.max_prompt_length=2048
  data.max_response_length=512
  data.filter_overlong_prompts=True
  data.truncation=error
  data.return_raw_chat=True
  actor_rollout_ref.model.path="$MODEL_DIR"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.ppo_micro_batch_size=null
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=False
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size=256
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.actor.use_invalid_action_penalty=True
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
  env.env_name=alfworld/AlfredTWEnv
  env.alfworld.eval_dataset=eval_in_distribution
  env.seed="$SEED"
  env.max_steps=50
  env.history_length=2
  env.rollout.n=8
  env.resources_per_worker.num_cpus=0.1
  trainer.critic_warmup=0
  trainer.logger='[console,wandb]'
  trainer.project_name=gigpo
  trainer.experiment_name="concal_main_alfworld_qwen25_15b_seed${SEED}"
  trainer.n_gpus_per_node="$N_GPUS"
  trainer.nnodes=1
  trainer.ray_wait_register_center_timeout=1200
  trainer.default_local_dir="$RUN_DIR/checkpoints"
  trainer.save_freq=5
  trainer.max_actor_ckpt_to_keep=3
  trainer.resume_mode=disable
  trainer.test_freq=5
  trainer.total_epochs=150
  trainer.total_training_steps=150
  trainer.val_before_train=True
)
args+=("$@")
python scripts/validate_concal.py --benchmark alfworld "${args[@]}"
if [[ "${PREPARE_VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
mkdir -p "$RUN_DIR"
python -u -m verl.trainer.main_ppo "${args[@]}" 2>&1 | tee "$RUN_DIR/train.log"
