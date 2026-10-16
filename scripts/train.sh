#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DATASET_TYPE="${DATASET_TYPE:-webqsp}"
case "${DATASET_TYPE}" in webqsp|grailqa|graphq) ;; *) echo 'DATASET_TYPE must be webqsp, grailqa or graphq' >&2; exit 2;; esac
: "${MODEL_PATH:?Set MODEL_PATH to a dataset-specific SFT checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE explicitly}"
: "${VAL_FILE:?Set VAL_FILE explicitly to the intended validation split}"
export GAPD_ENABLE="${GAPD_ENABLE:-true}"
export GAPD_JSD_COEF="${GAPD_JSD_COEF:-0.5}"
export GAPD_PG_COEF="${GAPD_PG_COEF:-1.0}"
export GAPD_USE_PG="${GAPD_USE_PG:-true}"
export GAPD_KL_MODE="${GAPD_KL_MODE:-k1}"
export GAPD_JSD_TOKEN_CLIP="${GAPD_JSD_TOKEN_CLIP:-2.0}"
export GAPD_TEACHER_MAX_TURNS="${GAPD_TEACHER_MAX_TURNS:--1}"
export GAPD_SKIP_SAME_NEXT_STATE="${GAPD_SKIP_SAME_NEXT_STATE:-true}"
export GAPD_SKIP_EXPRESSION_ID_RISK="${GAPD_SKIP_EXPRESSION_ID_RISK:-true}"
export GAPD_SKIP_HIGH_MID_F1_THRESHOLD="${GAPD_SKIP_HIGH_MID_F1_THRESHOLD:-0.9}"
# Turn dumps remain opt-in, as they contain dataset text and generated traces.
if [[ "${GAPD_DRY_RUN:-false}" != "true" ]]; then
    python3 tools/preflight.py
    if [[ "${TRAINER_VAL_ONLY:-false}" != "true" && "${GAPD_ENABLE}" == "true" ]]; then
        python3 tools/check_gold_trace.py "${TRAIN_FILE}" "${VAL_FILE}"
    fi
fi
exec bash scripts/train/train_kbqa_sexpr_generation_dapo_gapd.sh "gapd-${DATASET_TYPE}" "$@"
