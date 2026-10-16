# GAPD: Gold-Action Policy Distillation

Research implementation for **GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering**.

[Code](https://github.com/sunxin000/GAPD) · [Paper](https://arxiv.org/abs/2605.29584)

GAPD augments outcome-based reinforcement learning with token-level guidance derived from gold logical forms. Student and gold execution states are aligned through intermediate Freebase entity sets. The current policy, conditioned on the aligned gold-action hint, supplies teacher probabilities for student-generated tokens. Teacher hints are used during training; the evaluation entry point disables distillation and teacher diagnostics.

The implementation builds on KBQA-R1 and includes a customized VERL fork.

## Implementation map

| Component | Location |
|---|---|
| Gold execution traces | `scripts/data_process/build_gold_trace.py` |
| Entity-state alignment | `kbqa_r1/gapd/state_matcher.py` |
| Teacher prompts and probabilities | `kbqa_r1/gapd/teacher_prompt_builder.py`, `gapd_rollout_teacher.py` |
| Multi-turn rollout and supervision gates | `kbqa_r1/agent_loop/kbqa_sexpr_agent_loop.py` |
| Token-level distillation and policy loss | `kbqa_r1/gapd/gapd_engine.py`, `gapd_loss.py` |
| Training entry point | `kbqa_r1/trainer/main_dapo_kbqa.py` |
| S-expression execution and SPARQL | `kbqa_r1/sexpr/`, `kbqa_r1/sparql/` |

Supported KBQA datasets are WebQSP, GrailQA and GraphQ.

## Install

Use Linux, Python 3.12+, NVIDIA GPUs and a compatible CUDA/PyTorch/vLLM stack. The paper describes training on 8 A100 GPUs with 80 GB memory each. Freebase/Virtuoso, its ODBC driver, ontology files, processed benchmark inputs and model checkpoints are separate prerequisites.

```bash
git clone https://github.com/sunxin000/GAPD.git
cd GAPD
conda create -n gapd python=3.12 -y
conda activate gapd
# First prepare the CUDA stack and unixODBC as described in docs/SETUP.md.
python -m pip install -r requirements.txt
```

Use the bundled `verl/` fork. Installing the stock VERL package will omit GAPD integration changes. The editable installation keeps source-relative scripts and Hydra configuration paths available.

## Prepare data

The retained preparation pipeline starts from processed benchmark JSON and base RL parquet containing canonical action responses/function lists. Download or create these inputs separately; see [data requirements](docs/DATA.md).

```bash
bash scripts/prepare_data.sh webqsp train \
  /path/to/base_train.parquet /path/to/WebQSP_train.json data/webqsp
bash scripts/prepare_data.sh webqsp dev \
  /path/to/base_dev.parquet /path/to/WebQSP_dev.json data/webqsp
```

Choose the intended benchmark split explicitly. For benchmarks without a development split, use and document your training/validation partition. GrailQA evaluation should use its designated development data with level labels.

## Train

Start from a dataset-specific SFT checkpoint, then configure the data and backend. [The configuration example](configs/train.env.example) lists the available environment settings.

```bash
export DATASET_TYPE=webqsp
export MODEL_PATH=/path/to/webqsp-sft-checkpoint
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/dev.parquet
export NGPUS_PER_NODE=8
export SIMCSE_MODEL_PATH=/path/to/sup-simcse-roberta-large
bash scripts/train.sh
```

To inspect arguments without importing the training stack or launching workers:

```bash
GAPD_DRY_RUN=true bash scripts/train.sh
```

The public wrapper selects the coefficient and gates from the main-run launchers. Additional Hydra overrides can be appended to `scripts/train.sh`; see [the configuration reference](docs/REPRODUCIBILITY.md) for defaults and paper settings.

## Evaluate

For a merged HuggingFace model, set `MODEL_PATH` to the trained model and select `VAL_FILE`, then run:

```bash
bash scripts/evaluate.sh
```

For a VERL FSDP checkpoint, additionally set `RESUME_CKPT_PATH=/path/to/global_step_120`. The shared trainer also loads `TRAIN_FILE`, so provide it during evaluation. This entry point runs validation and returns before any policy updates. It disables GAPD/OPD and teacher diagnostics. Validation dumps are written under `logs/`.

## Local checks

These checks need Python 3.12+ and the standard library, without GPUs, model downloads or KB access:

```bash
python -m unittest discover -s tests -v
python tools/audit_release.py
```

Training preflight runs automatically during actual launches. It checks local modules, ontology resources, input paths and CUDA availability.

## Citation

```bibtex
@article{sun2026gapd,
  title={GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering},
  author={Sun, Xin and Xie, Jianan and Chen, Zhongqi and Liu, Qiang and Wu, Shu and Song, Bowen and Wang, Weiqiang and Wang, Zilei and Wang, Liang},
  journal={arXiv preprint arXiv:2605.29584},
  year={2026},
  doi={10.48550/arXiv.2605.29584}
}
```

## Licensing and acknowledgments

GAPD is released under the [Apache License 2.0](LICENSE), following the license declared in the original source README. Third-party components retain their original license terms and copyright notices. See [third-party notices](THIRD_PARTY_NOTICES.md).

We acknowledge [VERL](https://github.com/volcengine/verl), [KBQA-o1](https://github.com/LHRLAB/KBQA-o1), [RNG-KBQA](https://github.com/salesforce/rng-kbqa), [GrailQA](https://github.com/dki-lab/GrailQA) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1).
