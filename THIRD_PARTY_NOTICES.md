# Third-party attribution inventory

Existing copyright/license headers are preserved. Original GAPD contributions follow the root Apache-2.0 LICENSE. Third-party code retains its respective license terms and notices.

| Component | Evidence / source | Included notice |
|---|---|---|
| Customized VERL fork and inherited utilities | Apache-2.0 headers; [upstream license](https://github.com/volcengine/verl/blob/main/LICENSE) | `LICENSES/Apache-2.0.txt`; source headers |
| KBQA-o1-derived execution/retrieval code | Source docstrings; [upstream MIT license](https://github.com/LHRLAB/KBQA-o1/blob/main/LICENSE), Copyright (c) 2025 Haoran Luo | `LICENSES/MIT-KBQA-o1.txt` |
| RNG-KBQA-derived SPARQL executor | BSD-3-Clause header in `kbqa_r1/sparql/sparql_executor.py`; [upstream license](https://github.com/salesforce/rng-kbqa/blob/main/LICENSE.txt) | `LICENSES/BSD-3-Clause-RNG-KBQA.txt`, `LICENSES/RNG-KBQA-NOTICE.txt` |
| GrailQA code inherited through RNG-KBQA | [RNG-KBQA notice](https://github.com/salesforce/rng-kbqa/blob/main/NOTICE) identifies GrailQA/executor ancestry | Retained upstream notice and Apache license |
| VERL's bundled Torch/vLLM and other adaptations | Embedded copyright and license text under `verl/third_party/` and individual source files | Source headers and embedded notices retained |

License texts for KBQA-o1 and RNG-KBQA were retrieved from the linked upstream pages during preparation; rendered source whitespace may differ. Apache-2.0 text was copied from a local VERL checkout and cross-checked against upstream.

This is an initial attribution inventory. The selected source does not record exact upstream commits for every inherited module. Maintainers should complete that provenance mapping before publication, including any additional notice requirements found in the actual upstream revisions.

Datasets, Freebase, model weights and optional installed packages are external resources with separate terms; none is redistributed in this release.
