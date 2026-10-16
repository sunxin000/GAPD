# Data preparation and external resources

No benchmark dataset, Freebase dump, checkpoint, cached prediction or experiment dump is included in this candidate.

## Inputs

The original README uses the [KBQA-o1](https://github.com/LHRLAB/KBQA-o1) preprocessing pipeline for WebQSP, GrailQA and GraphQ. The retained GAPD preparation route needs:

- Processed JSON for each chosen split, with question IDs, questions, answer MIDs, logical forms and function lists.
- Base RL parquet with canonical response/action content and `extra_info`. `prepare_rejection_sampling_dataset.py` extracts reference action hints from it.
- The Freebase backend, ontology files and SimCSE checkpoint described in `SETUP.md`.

The base parquet files are external bootstrap inputs. The candidate does not yet provide a fully self-contained route from raw benchmark downloads to these bootstrap inputs. Their version, download/distribution location and preprocessing provenance remain release tasks.

Typical processed JSON fields inherited from KBQA-R1 are:

```json
{
  "ID": "synthetic-example-001",
  "question": "A synthetic question for schema illustration",
  "answer": ["m.example_answer"],
  "sexpr": "(JOIN example.relation m.example_entity)",
  "function_list": [
    "expression0 = START('m.example_entity')",
    "expression1 = JOIN('example.relation', expression0)"
  ]
}
```

This is a schema illustration, not an executable Freebase example.

## Gold traces

Run the same preparation command once per split:

```bash
bash scripts/prepare_data.sh grailqa train \
  /path/to/base_train.parquet /path/to/processed_train.json data/grailqa
bash scripts/prepare_data.sh grailqa dev \
  /path/to/base_dev.parquet /path/to/processed_dev.json data/grailqa
```

The command extracts action hints, executes the reference actions to collect MID sets, writes a trace JSON map and merges it into the final RL parquet. Each trace step contains `prev_mid_set`, `next_mid_set`, `per_turn_hint`, `step_index` and `action_function`.

The explicit split argument controls the output name. The original Python script also retains path-based inference when `--split` is omitted. Use `--split` in new workflows to avoid accidental split selection from a parent-directory name.

`tools/check_gold_trace.py` checks nonempty coverage at sample and step level and rejects malformed trace fields. The default minimum is 0.9, matching the preparation script's threshold. Initial empty previous states can be valid; nonempty next-state coverage is still required. No trace or hint is generated from a test question unless that split is explicitly supplied.

## SFT warm-start

If you have rejection-sampling validation dumps, the retained converter can build SFT data:

```bash
python scripts/data_process/build_sft_from_dumps.py \
  --dump_dir /path/to/validation-dumps \
  --output_file data/webqsp_rl_dataset_sft/train_sft_new.parquet \
  --min_mid_f1 0.9
BASE_MODEL=/path/to/base-model DATASET_TYPE=webqsp \
  bash scripts/train/run_sft_from_rejection.sh
```

The SFT launcher's retained source default is `SFT_LR=1e-5`; the paper appendix states `5e-6`. Set and record the intended value. A portable rejection-sampling generation recipe and the final dataset-specific SFT checkpoint IDs remain to be finalized. The main GAPD entry point can use an existing SFT checkpoint directly.

## Split and evaluation records

Keep train, development/validation and final test provenance separate. GrailQA level-based reporting needs `extra_info.level`. This candidate requires explicit `TRAIN_FILE` and `VAL_FILE`; it does not select a test file as validation automatically. Document any training subset used for validation on datasets without a development split.

For a reproducibility release, record original dataset URLs and revisions, raw/processed file hashes, preprocessing commands, split IDs, successful trace rates and entity-linking assumptions. Distribute third-party data and models separately under their own terms.
