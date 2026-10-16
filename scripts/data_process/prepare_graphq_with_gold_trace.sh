#!/usr/bin/env bash
# Modified 2026-10-16 for the GAPD release candidate; see CHANGELOG.md.
set -euo pipefail

# Build GraphQ RL data with precomputed gold_trace for GAPD/GAPD.
#
# This mirrors prepare_webqsp_with_gold_trace.sh but uses GraphQ defaults.
# It intentionally requires an explicit precomputed gold_trace JSON and never
# uses prepare_rl_dataset.py --compute_gold_trace.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET="graphq"

pick_file() {
    local env_value="$1"
    shift
    if [[ -n "${env_value}" ]]; then
        echo "${env_value}"
        return 0
    fi
    local path
    for path in "$@"; do
        if [[ -f "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done
    echo ""
}

BASE_RL_DIR="${BASE_RL_DIR:-${REPO_ROOT}/data/graphq_rl_dataset}"
BASE_TRAIN="$(pick_file "${BASE_TRAIN:-}" \
    "${BASE_RL_DIR}/train.parquet" \
    "${BASE_RL_DIR}/train_clean.parquet")"
BASE_TEST="$(pick_file "${BASE_TEST:-}" \
    "${BASE_RL_DIR}/test.parquet" \
    "${BASE_RL_DIR}/test_clean.parquet")"

RAW_TRAIN_JSON="$(pick_file "${RAW_TRAIN_JSON:-}" \
    "${REPO_ROOT}/dataset/GraphQ/processed/GraphQ_train.json" \
    "${REPO_ROOT}/dataset/GraphQ/processed/graphq_train.json" \
    "${REPO_ROOT}/dataset/GraphQ/processed/train.json" \
    "${REPO_ROOT}/dataset/GraphQuestions/processed/GraphQ_train.json" \
    "${REPO_ROOT}/dataset/GraphQuestions/processed/GraphQuestions_train.json")"
RAW_TEST_JSON="$(pick_file "${RAW_TEST_JSON:-}" \
    "${REPO_ROOT}/dataset/GraphQ/processed/GraphQ_test.json" \
    "${REPO_ROOT}/dataset/GraphQ/processed/graphq_test.json" \
    "${REPO_ROOT}/dataset/GraphQ/processed/test.json" \
    "${REPO_ROOT}/dataset/GraphQuestions/processed/GraphQ_test.json" \
    "${REPO_ROOT}/dataset/GraphQuestions/processed/GraphQuestions_test.json")"

HINT_DIR="${HINT_DIR:-${REPO_ROOT}/data/graphq_rl_dataset_sft}"
TRACE_DIR="${TRACE_DIR:-${REPO_ROOT}/data/graphq_gold_trace}"
FINAL_DIR="${FINAL_DIR:-${REPO_ROOT}/data/graphq_rl_dataset_with_gold_trace}"

HINT_STYLE="${HINT_STYLE:-reference}"
MIN_GOLD_TRACE_SUCCESS_RATE="${MIN_GOLD_TRACE_SUCCESS_RATE:-0.9}"
SPARQL_URL="${SPARQL_URL:-http://localhost:8890/sparql}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TRACE_LIMIT="${TRACE_LIMIT:-}"

mkdir -p "${HINT_DIR}" "${TRACE_DIR}" "${FINAL_DIR}"

check_file() {
    local path="$1"
    local label="$2"
    if [[ -z "${path}" || ! -f "${path}" ]]; then
        echo "ERROR: ${label} not found." >&2
        echo "       Set the corresponding env var if your path is different." >&2
        echo "       BASE_TRAIN=${BASE_TRAIN:-}" >&2
        echo "       BASE_TEST=${BASE_TEST:-}" >&2
        echo "       RAW_TRAIN_JSON=${RAW_TRAIN_JSON:-}" >&2
        echo "       RAW_TEST_JSON=${RAW_TEST_JSON:-}" >&2
        exit 1
    fi
}

check_file "${BASE_TRAIN}" "base train parquet"
check_file "${BASE_TEST}" "base test parquet"
check_file "${RAW_TRAIN_JSON}" "raw train json"
check_file "${RAW_TEST_JSON}" "raw test json"

echo "========================================"
echo "Prepare GraphQ RL Dataset With Gold Trace"
echo "========================================"
echo "Repo root                  : ${REPO_ROOT}"
echo "Base train parquet          : ${BASE_TRAIN}"
echo "Base test parquet           : ${BASE_TEST}"
echo "Raw train json              : ${RAW_TRAIN_JSON}"
echo "Raw test json               : ${RAW_TEST_JSON}"
echo "Hint dir                    : ${HINT_DIR}"
echo "Trace dir                   : ${TRACE_DIR}"
echo "Final dir                   : ${FINAL_DIR}"
echo "Hint style                  : ${HINT_STYLE}"
echo "Min gold trace success rate : ${MIN_GOLD_TRACE_SUCCESS_RATE}"
echo "SPARQL URL                  : ${SPARQL_URL}"
echo "========================================"

build_hint_split() {
    local input_parquet="$1"
    local output_parquet="$2"
    local split_name="$3"

    echo ""
    echo "[1/4] Building ${split_name} action-hint parquet..."
    local cmd=(
        python3 scripts/data_process/prepare_rejection_sampling_dataset.py
        --input_path "${input_parquet}"
        --output_path "${output_parquet}"
        --hint_style "${HINT_STYLE}"
        --skip_no_hint
    )
    if [[ -n "${NUM_SAMPLES}" ]]; then
        cmd+=(--num_samples "${NUM_SAMPLES}")
    fi
    "${cmd[@]}"
}

build_trace_split() {
    local hint_parquet="$1"
    local trace_json="$2"
    local split_name="$3"

    echo ""
    echo "[2/4] Building ${split_name} gold_trace JSON..."
    local cmd=(
        python3 scripts/data_process/build_gold_trace.py
        --input "${hint_parquet}"
        --output "${trace_json}"
        --dataset "${DATASET}"
        --sparql_url "${SPARQL_URL}"
        --log-interval "${LOG_INTERVAL}"
    )
    if [[ -n "${TRACE_LIMIT}" ]]; then
        cmd+=(--limit "${TRACE_LIMIT}")
    fi
    "${cmd[@]}"
}

prepare_final_split() {
    local raw_json="$1"
    local hint_parquet="$2"
    local trace_json="$3"
    local split_name="$4"

    echo ""
    echo "[3/4] Writing final ${split_name} RL parquet with strict gold_trace check..."
    python3 scripts/data_process/prepare_rl_dataset.py \
        --data_path "${raw_json}" \
        --output_dir "${FINAL_DIR}" \
        --dataset "${DATASET}" \
        --use_odbc \
        --with_hint_path "${hint_parquet}" \
        --gold_trace_path "${trace_json}" \
        --min_gold_trace_success_rate "${MIN_GOLD_TRACE_SUCCESS_RATE}"
}

validate_final_split() {
    local final_parquet="$1"
    local split_name="$2"

    echo ""
    echo "[4/4] Validating final ${split_name} parquet..."
    python3 - "${final_parquet}" "${MIN_GOLD_TRACE_SUCCESS_RATE}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
threshold = float(sys.argv[2])
df = pd.read_parquet(path)

present = 0
sample_nonempty = 0
steps_total = 0
steps_nonempty = 0
for _, row in df.iterrows():
    extra = row.get("extra_info", {}) or {}
    trace = extra.get("gold_trace")
    if isinstance(trace, list) and trace:
        present += 1
        has_nonempty = False
        steps_total += len(trace)
        for step in trace:
            if not isinstance(step, dict):
                continue
            mids = step.get("next_mid_set") or []
            if len(mids) > 0:
                has_nonempty = True
                steps_nonempty += 1
        if has_nonempty:
            sample_nonempty += 1

total = len(df)
present_rate = present / total if total else 0.0
sample_rate = sample_nonempty / total if total else 0.0
step_rate = steps_nonempty / steps_total if steps_total else 0.0
print(f"[CHECK] split={path}")
print(f"[CHECK] total={total} present={present} ({present_rate:.2%})")
print(f"[CHECK] samples_with_nonempty_step={sample_nonempty} ({sample_rate:.2%})")
print(f"[CHECK] steps_nonempty={steps_nonempty}/{steps_total} ({step_rate:.2%})")
if min(present_rate, sample_rate, step_rate) < threshold:
    raise SystemExit(
        f"gold_trace quality below threshold {threshold:.2%}: "
        f"present={present_rate:.2%}, sample_nonempty={sample_rate:.2%}, step_nonempty={step_rate:.2%}"
    )
PY
}

TRAIN_HINT="${HINT_DIR}/train_with_hints.parquet"
TEST_HINT="${HINT_DIR}/test_with_hints.parquet"
TRAIN_TRACE="${TRACE_DIR}/train_gold_trace.json"
TEST_TRACE="${TRACE_DIR}/test_gold_trace.json"

build_hint_split "${BASE_TRAIN}" "${TRAIN_HINT}" "train"
build_hint_split "${BASE_TEST}" "${TEST_HINT}" "test"

build_trace_split "${TRAIN_HINT}" "${TRAIN_TRACE}" "train"
build_trace_split "${TEST_HINT}" "${TEST_TRACE}" "test"

prepare_final_split "${RAW_TRAIN_JSON}" "${TRAIN_HINT}" "${TRAIN_TRACE}" "train"
prepare_final_split "${RAW_TEST_JSON}" "${TEST_HINT}" "${TEST_TRACE}" "test"

validate_final_split "${FINAL_DIR}/train.parquet" "train"
validate_final_split "${FINAL_DIR}/test.parquet" "test"

echo ""
echo "Done."
echo "Final train: ${FINAL_DIR}/train.parquet"
echo "Final test : ${FINAL_DIR}/test.parquet"
