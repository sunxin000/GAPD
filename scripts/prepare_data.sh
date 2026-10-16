#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo 'Usage: bash scripts/prepare_data.sh DATASET SPLIT BASE_RL_PARQUET RAW_JSON OUTPUT_DIR' >&2
    echo 'DATASET: webqsp|grailqa|graphq; SPLIT: train|dev|test' >&2
    exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
dataset="$1"; split="$2"; base_rl="$3"; raw_json="$4"; output_dir="$5"
case "${dataset}" in webqsp|grailqa|graphq) ;; *) echo 'Invalid dataset' >&2; exit 2;; esac
case "${split}" in train|dev|test) ;; *) echo 'Invalid split' >&2; exit 2;; esac
for input in "${base_rl}" "${raw_json}"; do
    if [[ ! -f "${input}" ]]; then echo "ERROR: Input missing: ${input}" >&2; exit 1; fi
done
mkdir -p "${output_dir}/hints" "${output_dir}/traces"
python3 scripts/data_process/prepare_rejection_sampling_dataset.py \
    --input_path "${base_rl}" --output_path "${output_dir}/hints/${split}.parquet" \
    --hint_style reference --skip_no_hint
python3 scripts/data_process/build_gold_trace.py \
    --input "${output_dir}/hints/${split}.parquet" --output "${output_dir}/traces/${split}.json" \
    --dataset "${dataset}" --sparql_url "${SPARQL_URL:-http://localhost:8890/sparql}"
python3 scripts/data_process/prepare_rl_dataset.py \
    --data_path "${raw_json}" --output_dir "${output_dir}" --dataset "${dataset}" --split "${split}" \
    --use_odbc --with_hint_path "${output_dir}/hints/${split}.parquet" \
    --gold_trace_path "${output_dir}/traces/${split}.json" \
    --min_gold_trace_success_rate "${MIN_GOLD_TRACE_SUCCESS_RATE:-0.9}"
python3 tools/check_gold_trace.py "${output_dir}/${split}.parquet" \
    --minimum "${MIN_GOLD_TRACE_SUCCESS_RATE:-0.9}"
