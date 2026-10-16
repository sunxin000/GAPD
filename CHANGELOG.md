# Changelog

## 0.1.1 — 2026-10-16

- Unified GAPD naming across package paths, modules, classes, configuration fields, environment variables and metrics.
- Simplified the README navigation to Code and Paper.

## 0.1.0 — 2026-10-16

- Prepared the source release from the selected working tree, retaining the customized VERL integration.
- Replaced private repository/model/output defaults in selected launchers with portable paths and explicit model configuration; removed the SFT launcher's cluster daemon restart.
- Made SimCSE configuration public and overridable through `SIMCSE_MODEL_PATH`; retained the existing model-load fallback.
- Made state-matcher trace typing independent of the KB runtime for CPU checks; matching semantics are unchanged.
- Added explicit data split selection, gold-trace validation, train/evaluate wrappers and redacted argument dry runs.
- Added packaging, setup/reproducibility documentation, citation metadata, third-party license copies and local release checks.
- Excluded Git history, external/broken soft links, logs, checkpoints, papers, private documents and legacy platform-specific launchers.
- Omitted unrelated inherited VLA examples whose worker imports are absent from this source snapshot; GAPD uses the text-agent FSDP/vLLM route.
- Normalized constant regex literals in `webqsp_sparql.py` without changing their string values, removing Python 3.12 syntax warnings.

- Added the root Apache-2.0 license, following the original source README, and prepared the initial GitHub source release.
