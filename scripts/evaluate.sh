#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAINER_VAL_ONLY=true
export TRAINER_VAL_BEFORE_TRAIN=true
export TRAINER_TEST_FREQ=0
export TRAINER_SAVE_FREQ=0
export GAPD_ENABLE=false
export OPD_ENABLE=false
export GAPD_VAL_TEACHER_DEBUG=false
export n_resp_per_prompt="${n_resp_per_prompt:-1}"
export n_resp_per_prompt_val="${n_resp_per_prompt_val:-1}"
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
    if [[ ! -d "${RESUME_CKPT_PATH}/actor" ]]; then
        echo "ERROR: RESUME_CKPT_PATH must contain an actor/ directory." >&2
        exit 1
    fi
    export TRAINER_RESUME_MODE=resume_path
    export TRAINER_RESUME_FROM_PATH="${RESUME_CKPT_PATH}"
else
    # MODEL_PATH must be a merged HuggingFace checkpoint in this mode.
    export TRAINER_RESUME_MODE=disable
    export TRAINER_RESUME_FROM_PATH=null
fi
exec bash "${ROOT}/scripts/train.sh" "$@"
