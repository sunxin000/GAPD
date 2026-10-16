#!/usr/bin/env bash
# Modified 2026-10-16 for the GAPD release candidate; see CHANGELOG.md.
set -euo pipefail

# KBQA-R1 S-Expression Training (DAPO / retool recipe)
# Adapts the run_qwen2_7b_dapo configuration to KBQA datasets and reward setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# -----------------------------------------------------------------------------
# GPU discovery helpers (reuse logic from the legacy script)
# -----------------------------------------------------------------------------
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "Detected ${GPU_COUNT} GPUs using nvidia-smi"
        return 0
    fi
    if command -v python &> /dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "Detected ${GPU_COUNT} GPUs using PyTorch"
        return 0
    fi
    if command -v python3 &> /dev/null; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "Detected ${GPU_COUNT} GPUs using PyTorch (python3)"
        return 0
    fi
    echo "Warning: Could not detect GPU count, defaulting to 8"
    GPU_COUNT=8
    return 1
}

detect_gpu_count

# Detect GPU type (H20 vs A100)
detect_gpu_type() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        if [[ "${GPU_NAME}" == *"H20"* ]]; then
            echo "H20"
        elif [[ "${GPU_NAME}" == *"A100"* ]]; then
            echo "A100"
        else
            echo "UNKNOWN"
        fi
    else
        echo "UNKNOWN"
    fi
}

GPU_TYPE=$(detect_gpu_type)
echo "Detected GPU Type: ${GPU_TYPE}"

# export VLLM_USE_V1=${VLLM_USE_V1:-0}
# export VLLM_ATTENTION_BACKEND=XFORMERS
# FIXED: Enable XFORMERS for better performance (was commented out)
# export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# -----------------------------------------------------------------------------
# Dataset, model, and bookkeeping configuration
# -----------------------------------------------------------------------------
DATASET_TYPE=${DATASET_TYPE:-'webqsp'}

# Set total epochs based on dataset type
if [[ "${DATASET_TYPE}" == "grailqa" ]]; then
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
else
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-12}
fi
# Mirror the sexpr_generation script naming: DATA_DIR and BASE_MODEL
# Dataset-specific default data directories:
#   WebQSP: data/webqsp_rl_dataset (contains gold_trace)
#   GrailQA: data/grailqa_rl_dataset_with_gold_trace (separate dir, gold_trace injected)
#   GraphQ: data/graphq_rl_dataset
if [ -n "${DATA_DIR:-}" ]; then
    # User override via env var -- keep as-is
    :
elif [ "${DATASET_TYPE}" = "grailqa" ]; then
    DATA_DIR="${REPO_ROOT}/data/grailqa_rl_dataset_with_gold_trace"
else
    DATA_DIR="${REPO_ROOT}/data/${DATASET_TYPE}_rl_dataset"
fi
# Train/val file names differ between datasets (some use *_clean.parquet, some plain *.parquet)
if [ "${DATASET_TYPE}" = "grailqa" ]; then
    TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
    VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test.parquet"}
else
    TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train_clean.parquet"}
    VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test_clean.parquet"}
fi

# Supply the dataset-specific SFT warm-start checkpoint explicitly.
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: Set MODEL_PATH to a HuggingFace SFT checkpoint directory or model ID." >&2
    exit 1
fi
export BASE_MODEL="${MODEL_PATH}"
if [[ "${MODEL_PATH}" == /* && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: Missing checkpoint config.json: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
# allow script to be called without positional args while running with `set -u`
EXP_ARG="${1:-}"
if [[ -n "${EXP_ARG}" ]]; then
    EXPERIMENT_NAME="${EXP_ARG}__${TIMESTAMP}"
else
    EXPERIMENT_NAME="${DATASET_TYPE}-kbqa-r1-dapo/${TIMESTAMP}"
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
LOG_DIR_ROOT="${LOG_DIR_ROOT:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR_ROOT}/${DATASET_TYPE}/${EXPERIMENT_NAME}.log}"
if [[ "${GAPD_DRY_RUN:-false}" != "true" ]]; then
    mkdir -p "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" "$(dirname "${LOG_FILE}")"
fi

WANDB_MODE=${WANDB_MODE:-offline}
WANDB_PROJECT=${WANDB_PROJECT:-'KBQA-R1-SExpr-DAPO'}
export WANDB_MODE
export WANDB_PROJECT
export WANDB_API_KEY=${WANDB_API_KEY:-''}
export TENSORBOARD_DIR="${REPO_ROOT}/tensorboard_log/${DATASET_TYPE}/${EXPERIMENT_NAME}"
VALIDATION_DUMP_DIR="${LOG_DIR_ROOT}/${DATASET_TYPE}/${EXPERIMENT_NAME}/validation"
if [[ "${GAPD_DRY_RUN:-false}" != "true" ]]; then mkdir -p "${VALIDATION_DUMP_DIR}"; fi
# -----------------------------------------------------------------------------
# Training hyper-parameters (aligned with run_qwen2_7b_dapo defaults)
# -----------------------------------------------------------------------------
adv_estimator=${adv_estimator:-grpo}
use_kl_in_reward=${use_kl_in_reward:-false}
kl_coef=${kl_coef:-0.0}
use_kl_loss=${use_kl_loss:-false}
kl_loss_coef=${kl_loss_coef:-0.001}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}

# -----------------------------------------------------------------------------
# Rollout Importance Sampling (mismatch correction) configuration
# These env vars let you toggle mismatch_helper.py features without editing code.
# Set rollout_is_threshold to a float (e.g. 3.0) to ENABLE; leave empty or "" to disable.
# Recommended initial settings:
#   token-level + truncate: stable baseline
#   threshold 2.5~5.0; monitor mismatch/rollout_is_eff_sample_size & mismatch/rollout_is_p95
# To just observe metrics first: set rollout_is=false but keep threshold.
# To switch to mask mode: set rollout_is_mode=mask and optionally rollout_is_threshold_lower (default 1/upper).
# To disable veto: set rollout_is_veto_threshold="null" (Hydra will parse null) or very small (1e-6).
# -----------------------------------------------------------------------------
rollout_is=${rollout_is:-false}                       # apply weights to policy loss (false = metrics only)
rollout_is_level=${rollout_is_level:-token}          # token | sequence | geometric
rollout_is_mode=${rollout_is_mode:-truncate}         # truncate | mask
rollout_is_threshold=${rollout_is_threshold:-4.0}    # float => enable; "" or null => disable
rollout_is_threshold_lower=${rollout_is_threshold_lower:-null}  # optional; only meaningful for mask mode
rollout_is_veto_threshold=${rollout_is_veto_threshold:-null} # per-token veto (null to disable)

max_turns=${max_turns:-6}
max_prompt_length=${max_prompt_length:-14336}
# max_response_length is the total multi-turn response padding target (not per-turn limit).
# Per-turn vLLM generation is limited by multi_turn.max_tokens_per_turn (default 1024).
# Multi-turn responses can span multiple turns and exceed 1024 tokens, so we set this
# to 10240 to accommodate full trajectories (December runs showed max ~8700 tokens).
max_response_length=${max_response_length:-10240}
actor_lr=${actor_lr:-1e-6}

train_batch_size=${train_batch_size:-256}
# FIXED: Increase mini batch size from 16 to 64 for better throughput
ppo_mini_batch_size=${ppo_mini_batch_size:-128}
# FIXED: Reduce responses per prompt to match original (was 8, now 5)
n_resp_per_prompt=${n_resp_per_prompt:-5}
n_resp_per_prompt_val=${n_resp_per_prompt_val:-1}
data_val_max_samples=${DATA_VAL_MAX_SAMPLES:--1}

# PERFORMANCE FIX: Increase tensor parallel from 2 to 4 to match original script
# This improves inference throughput significantly on 8-GPU setup
infer_tp=${infer_tp:-4}
offload=${offload:-true}

enable_overlong_buffer=${enable_overlong_buffer:-false}
overlong_buffer_len=${overlong_buffer_len:-$((256 * 4))}
overlong_penalty_factor=${overlong_penalty_factor:-1.0}

# -----------------------------------------------------------------------------
# GAPD (Gold-Action Policy Distillation) configuration
# -----------------------------------------------------------------------------
gapd_enable=${GAPD_ENABLE:-true}
gapd_jsd_coef=${GAPD_JSD_COEF:-0.1}
gapd_pg_coef=${GAPD_PG_COEF:-1.0}
gapd_jsd_beta=${GAPD_JSD_BETA:-0.0}
gapd_jsd_temperature=${GAPD_JSD_TEMPERATURE:-1.0}
gapd_jsd_token_clip=${GAPD_JSD_TOKEN_CLIP:-2.0}
gapd_jsd_log_prob_min_clamp=${GAPD_JSD_LOG_PROB_MIN_CLAMP:--10.0}
gapd_jsd_top_k=${GAPD_JSD_TOP_K:-20}
gapd_use_pg=${GAPD_USE_PG:-true}
gapd_kl_mode=${GAPD_KL_MODE:-k1}
gapd_teacher_max_length=${GAPD_TEACHER_MAX_LENGTH:-16384}
# GAPD turn limit: -1 = all turns, 1 = first turn only, 2 = first two turns, etc.
gapd_teacher_max_turns=${GAPD_TEACHER_MAX_TURNS:--1}
# Enable state-anchored per-turn GAPD using gold_trace from dataset.
gapd_use_state_matching=${GAPD_USE_STATE_MATCHING:-true}
# Conservative GAPD supervision gates for all-turn ablations.
gapd_skip_same_next_state=${GAPD_SKIP_SAME_NEXT_STATE:-false}
gapd_skip_expression_id_risk=${GAPD_SKIP_EXPRESSION_ID_RISK:-false}
# Restrict GAPD/GAPD supervision to <action>...</action> spans. This avoids
# distilling teacher-hint style into <think> text.
gapd_supervise_action_only=${GAPD_SUPERVISE_ACTION_ONLY:-false}
# Skip GAPD/GAPD when the rollout already has high reward mid_f1. Negative disables.
gapd_skip_high_mid_f1_threshold=${GAPD_SKIP_HIGH_MID_F1_THRESHOLD:--1.0}

# Validation-time GAPD teacher diagnostics. This is file-only debug output used
# to align validation rollouts with teacher token supervision; it does not feed
# teacher tensors into validation metrics or training loss.
gapd_val_teacher_debug=${GAPD_VAL_TEACHER_DEBUG:-false}
gapd_val_teacher_debug_dir=${GAPD_VAL_TEACHER_DEBUG_DIR:-"${VALIDATION_DUMP_DIR}/teacher_debug"}
gapd_val_teacher_debug_max_samples=${GAPD_VAL_TEACHER_DEBUG_MAX_SAMPLES:-100}
gapd_val_teacher_debug_only_wrong=${GAPD_VAL_TEACHER_DEBUG_ONLY_WRONG:-true}
gapd_val_teacher_debug_mid_f1_threshold=${GAPD_VAL_TEACHER_DEBUG_MID_F1_THRESHOLD:-0.999}

# Token-level dump configuration for offline GAPD analysis
gapd_token_dump_dir=${GAPD_TOKEN_DUMP_DIR:-""}
gapd_token_dump_max_samples=${GAPD_TOKEN_DUMP_MAX_SAMPLES:-100}
gapd_token_dump_prob=${GAPD_TOKEN_DUMP_PROB:-0.05}
gapd_token_dump_strategy=${GAPD_TOKEN_DUMP_STRATEGY:-representative}
gapd_token_dump_min_valid_tokens=${GAPD_TOKEN_DUMP_MIN_VALID_TOKENS:-1}
if [[ -n "${GAPD_TOKEN_DUMP_STUDENT_TOP_K:-}" ]]; then
    gapd_token_dump_student_top_k=${GAPD_TOKEN_DUMP_STUDENT_TOP_K}
elif [[ -n "${gapd_token_dump_dir}" ]]; then
    gapd_token_dump_student_top_k=16
else
    gapd_token_dump_student_top_k=0
fi

# vLLM rollout resource knobs. Keep defaults aligned with prior runs, but allow
# quick recovery from EngineCore startup OOM/resource failures in shared clusters.
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-28672}

# Trainer control knobs. Eval-only wrappers set TRAINER_VAL_ONLY=true so the
# trainer runs the initial validation pass and returns before entering training.
trainer_val_before_train=${TRAINER_VAL_BEFORE_TRAIN:-true}
trainer_val_only=${TRAINER_VAL_ONLY:-false}
trainer_log_val_generations=${TRAINER_LOG_VAL_GENERATIONS:-10}
trainer_save_freq=${TRAINER_SAVE_FREQ:-20}
trainer_test_freq=${TRAINER_TEST_FREQ:-20}
trainer_resume_mode=${TRAINER_RESUME_MODE:-auto}
trainer_resume_from_path=${TRAINER_RESUME_FROM_PATH:-null}

# -----------------------------------------------------------------------------
# OPD (On-Policy Distillation with hindsight) configuration
# -----------------------------------------------------------------------------
opd_enable=${OPD_ENABLE:-false}
opd_max_obs_chars=${OPD_MAX_OBS_CHARS:-2048}
opd_strip_info_tags=${OPD_STRIP_INFO_TAGS:-false}


loss_agg_mode="token-mean"

# With max_response_length=10240, total sequence length can be up to
# prompt_length + response_length = 14336 + 10240 = 24576.
# Increase token budget to accommodate longer sequences.
actor_max_token_len_per_gpu=28672
log_prob_max_token_len_per_gpu=$((actor_max_token_len_per_gpu))

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPU_COUNT}}
TRAINING_GPUS=${TRAINING_GPUS:-${NGPUS_PER_NODE}}

use_dynamic_bsz=true

if (( NGPUS_PER_NODE < 1 )); then
    NGPUS_PER_NODE=1
fi
if (( TRAINING_GPUS < 1 )); then
    TRAINING_GPUS=1
fi
if (( TRAINING_GPUS > NGPUS_PER_NODE )); then
    TRAINING_GPUS=${NGPUS_PER_NODE}
fi


if (( infer_tp > NGPUS_PER_NODE )); then
    echo "Adjusting infer_tp (${infer_tp}) down to available GPUs per node (${NGPUS_PER_NODE})."
    infer_tp=${NGPUS_PER_NODE}
fi
if (( infer_tp < 1 )); then
    infer_tp=1
fi

echo "========================================"
echo "KBQA-R1 DAPO (retool) Configuration"
echo "========================================"
echo "Dataset Type : ${DATASET_TYPE}"
echo "Train File   : ${TRAIN_FILE}"
echo "Val File     : ${VAL_FILE}"
echo "Model Path   : ${MODEL_PATH}"
echo "Experiment   : ${EXPERIMENT_NAME}"
echo "Checkpoints  : ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
echo "Log File     : ${LOG_FILE} (only main run log; internal sexpr file logs disabled)"
echo "GPUs / node  : ${NGPUS_PER_NODE} (using ${TRAINING_GPUS} for training)"
echo "========================================"

# -----------------------------------------------------------------------------
# KBQA specific toggles
# -----------------------------------------------------------------------------
export SEXPR_MODE=true
export ENABLE_ACTION_REASONING=true
export ENABLE_RELATION_RETRIEVAL=true
export SEXPR_MAX_TURNS=${max_turns}
export SEXPR_VALIDATION_LEVEL=${SEXPR_VALIDATION_LEVEL:-STANDARD}
export PYTHON_LOG_LEVEL=${PYTHON_LOG_LEVEL:-INFO}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VERL_PPO_LOGGING_LEVEL=${VERL_PPO_LOGGING_LEVEL:-INFO}

export PYTHONUNBUFFERED=1
export DISABLE_SEXPR_FILE_LOGS=1  # 禁用内部写盘 (sexpr_executor / logging manager)

# CUDA forward-compat: the compat layer (driver 550) supports CUDA 12.8 runtime
# Set LD_LIBRARY_PATH in the cluster environment if its CUDA driver requires it.

# Register kbqa_sexpr agent loop in all Ray worker processes via verl's external module mechanism
export VERL_USE_EXTERNAL_MODULES=kbqa_r1.agent_loop
train_command=(python3 -m kbqa_r1.trainer.main_dapo_kbqa \
    data.val_max_samples=${data_val_max_samples} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.rollout.dtype=float16 \
    +algorithm.rollout_is=${rollout_is} \
    +algorithm.rollout_is_level=${rollout_is_level} \
    +algorithm.rollout_is_mode=${rollout_is_mode} \
    +algorithm.rollout_is_threshold=${rollout_is_threshold} \
    +algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold} \
    +algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    +algorithm.filter_groups.enable=false \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=true \
    data.truncation='left' \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    +data.max_start_length=2048 \
    +data.max_obs_length=3072 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    ++actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.3 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-1.0} \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    +actor_rollout_ref.actor.gapd_enable=${gapd_enable} \
    +actor_rollout_ref.actor.gapd_jsd_coef=${gapd_jsd_coef} \
    +actor_rollout_ref.actor.gapd_pg_coef=${gapd_pg_coef} \
    +actor_rollout_ref.actor.gapd_jsd_beta=${gapd_jsd_beta} \
    +actor_rollout_ref.actor.gapd_jsd_temperature=${gapd_jsd_temperature} \
    +actor_rollout_ref.actor.gapd_jsd_token_clip=${gapd_jsd_token_clip} \
    +actor_rollout_ref.actor.gapd_jsd_log_prob_min_clamp=${gapd_jsd_log_prob_min_clamp} \
    +actor_rollout_ref.actor.gapd_jsd_top_k=${gapd_jsd_top_k} \
    +actor_rollout_ref.actor.gapd_use_pg=${gapd_use_pg} \
    +actor_rollout_ref.actor.gapd_kl_mode=${gapd_kl_mode} \
    +actor_rollout_ref.actor.gapd_teacher_max_length=${gapd_teacher_max_length} \
    +actor_rollout_ref.rollout.gapd_enable=${gapd_enable} \
    +actor_rollout_ref.rollout.gapd_jsd_top_k=${gapd_jsd_top_k} \
    +actor_rollout_ref.rollout.gapd_teacher_max_length=${gapd_teacher_max_length} \
    +actor_rollout_ref.rollout.gapd_teacher_max_turns=${gapd_teacher_max_turns} \
    +actor_rollout_ref.rollout.gapd_use_state_matching=${gapd_use_state_matching} \
    +actor_rollout_ref.rollout.gapd_skip_same_next_state=${gapd_skip_same_next_state} \
    +actor_rollout_ref.rollout.gapd_skip_expression_id_risk=${gapd_skip_expression_id_risk} \
    +actor_rollout_ref.rollout.gapd_supervise_action_only=${gapd_supervise_action_only} \
    +actor_rollout_ref.rollout.gapd_skip_high_mid_f1_threshold=${gapd_skip_high_mid_f1_threshold} \
    +actor_rollout_ref.rollout.gapd_val_teacher_debug=${gapd_val_teacher_debug} \
    +actor_rollout_ref.rollout.gapd_val_teacher_debug_dir=${gapd_val_teacher_debug_dir} \
    +actor_rollout_ref.rollout.gapd_val_teacher_debug_max_samples=${gapd_val_teacher_debug_max_samples} \
    +actor_rollout_ref.rollout.gapd_val_teacher_debug_only_wrong=${gapd_val_teacher_debug_only_wrong} \
    +actor_rollout_ref.rollout.gapd_val_teacher_debug_mid_f1_threshold=${gapd_val_teacher_debug_mid_f1_threshold} \
    +actor_rollout_ref.actor.gapd_token_dump_dir=${gapd_token_dump_dir} \
    +actor_rollout_ref.actor.gapd_token_dump_max_samples=${gapd_token_dump_max_samples} \
    +actor_rollout_ref.actor.gapd_token_dump_prob=${gapd_token_dump_prob} \
    +actor_rollout_ref.actor.gapd_token_dump_strategy=${gapd_token_dump_strategy} \
    +actor_rollout_ref.actor.gapd_token_dump_min_valid_tokens=${gapd_token_dump_min_valid_tokens} \
    +actor_rollout_ref.actor.gapd_token_dump_student_top_k=${gapd_token_dump_student_top_k} \
    +actor_rollout_ref.actor.opd_enable=${opd_enable} \
    +actor_rollout_ref.actor.opd_max_obs_chars=${opd_max_obs_chars} \
    +actor_rollout_ref.actor.opd_strip_info_tags=${opd_strip_info_tags} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    +actor_rollout_ref.rollout.multi_turn.max_tokens_per_turn=1024 \
    actor_rollout_ref.rollout.agent.default_agent_loop=kbqa_sexpr \
    trainer.logger=['console','tensorboard'] \
    trainer.validation_data_dir="${VALIDATION_DUMP_DIR}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME//\//-}" \
    trainer.n_gpus_per_node=${TRAINING_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.resume_mode=${trainer_resume_mode} \
    trainer.resume_from_path="${trainer_resume_from_path}" \
    trainer.val_before_train=${trainer_val_before_train} \
    trainer.val_only=${trainer_val_only} \
    trainer.log_val_generations=${trainer_log_val_generations} \
    trainer.save_freq=${trainer_save_freq} \
    trainer.test_freq=${trainer_test_freq} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" \
    +sexpr_config.enable_sexpr_mode=true \
    +sexpr_config.enable_action_reasoning=true \
    +sexpr_config.enable_relation_retrieval=true \
    +sexpr_config.validation_level=${SEXPR_VALIDATION_LEVEL} \
    +sexpr_config.max_function_calls=10 \
    +sexpr_config.enable_entity_linking=true \
    +sexpr_config.enable_semantic_validation=true \
    +sexpr_config.use_complete_sparql_converter=true \
    +sexpr_config.enable_logging=false \
    +max_turns=${max_turns} \
    +use_odbc=true \
    +use_aioodbc=false \
    +odbc_config.driver_path="${FREEBASE_ODBC_DRIVER:-Virtuoso}" \
    +odbc_config.host="${FREEBASE_ODBC_HOST:-localhost}" \
    +odbc_config.port=${FREEBASE_ODBC_PORT:-13001} \
    +odbc_config.uid="${FREEBASE_ODBC_UID:-dba}" \
    +odbc_config.pwd="${FREEBASE_ODBC_PWD:-dba}" \
    +odbc_config.pool_size=${FREEBASE_ODBC_POOL_SIZE:-12} \
    +odbc_config.max_pool_size=${FREEBASE_ODBC_MAX_POOL_SIZE:-12} \
    +odbc_config.pool_timeout=30 \
    +odbc_config.query_timeout=600 \
    +odbc_config.max_concurrent=8 \
    +odbc_config.max_retries=2 \
    +odbc_config.retry_delay=1.0 \
    +sparql_batch_size=128 \
    +sparql_max_concurrent=8 \
    +sparql.url="${SPARQL_URL:-http://localhost:8000/execute}" \
    reward_model.reward_manager=dapo \
    +custom_reward_function.reward_kwargs.mid_f1_weight=1.0 \
    +custom_reward_function.reward_kwargs.structure_format_score=0.1 \
    +custom_reward_function.reward_kwargs.function_reward_weight=0.5 \
    +custom_reward_function.val_reward_kwargs.mid_f1_weight=1.0 \
    +custom_reward_function.val_reward_kwargs.structure_format_score=0.0 \
    custom_reward_function.path="${REPO_ROOT}/kbqa_custom_reward.py" \
    custom_reward_function.name=compute_score \
    "${@:2}"
)
if [[ "${GAPD_DRY_RUN:-false}" == "true" ]]; then
    for arg in "${train_command[@]}"; do
        case "${arg}" in
            +odbc_config.pwd=*) printf '%s\n' '+odbc_config.pwd=<redacted>' ;;
            *) printf '%s\n' "${arg}" ;;
        esac
    done
    exit 0
fi
"${train_command[@]}" 2>&1 | tee "${LOG_FILE}"

echo "========================================"
echo "KBQA-R1 DAPO training finished (exit code $?)"
echo "Logs        : ${LOG_FILE}"
echo "Checkpoints : ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
echo "========================================"
