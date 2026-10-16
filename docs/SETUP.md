# Environment and backend setup

## Python and GPU dependencies

Use Python 3.12 or later. The bundled VERL tree uses Python 3.12 f-string syntax; the original README's Python 3.10 instruction does not cover this source snapshot.

The source setup notes recorded CUDA 12.8, PyTorch 2.8.0, vLLM 0.10.2, flash-attn 2.8.3 and xformers 0.0.32.post1. See `requirements/training-recorded.txt` for the recorded environment versions.

On a Linux GPU host:

1. Prepare the CUDA driver/runtime, PyTorch, vLLM and matching flash-attn builds.
2. Install unixODBC and development headers through the host's package manager, and register the Virtuoso driver.
3. From the repository root, run `python -m pip install -r requirements.txt`.
4. Set the model, input and backend variables, then run `python tools/preflight.py`.

`peft==0.10.0` retains the source setup note's compatibility choice. It must still be validated against the selected stack. Optional FAISS acceleration requires a compatible FAISS build; the source supports NumPy similarity search when FAISS is unavailable. The editable install includes the bundled VERL fork and Hydra YAML files.

## Freebase and Virtuoso

Provision a Freebase Virtuoso instance separately. The original project refers to [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) and the [GrailQA ontology](https://github.com/dki-lab/GrailQA/tree/main/ontology). Place ontology resources here:

```text
dataset/Freebase/fb_roles
dataset/Freebase/fb_types
dataset/Freebase/reverse_properties
```

Register your installed ODBC driver, for example:

```ini
[Virtuoso]
Description = Virtuoso ODBC Driver
Driver = /path/to/virtodbc_r.so
Threading = 0
```

The public training launcher accepts `FREEBASE_ODBC_DRIVER`, `FREEBASE_ODBC_HOST`, `FREEBASE_ODBC_PORT`, `FREEBASE_ODBC_UID`, `FREEBASE_ODBC_PWD`, `FREEBASE_ODBC_POOL_SIZE` and `FREEBASE_ODBC_MAX_POOL_SIZE`. Local defaults are `Virtuoso`, `localhost`, port `13001`, and the development credentials `dba`/`dba`.

`SPARQL_URL` configures an HTTP endpoint. Gold-trace preparation uses the standard SPARQL endpoint, normally `http://localhost:8890/sparql`. Training selects ODBC; its legacy HTTP fallback expects a batch `/execute` service at `http://localhost:8000/execute`. These two interfaces are distinct. Use ODBC for the documented training route.

The legacy standalone `sparql_executor.py` still has its own localhost endpoint constants; the main training route uses `SPARQLConfig` and ODBC settings instead.

## Models

Provide the Llama-3.1-8B-Instruct backbone and/or dataset-specific SFT checkpoint under its applicable model terms. No weights are bundled. The public launchers require an explicit model path or model ID instead of selecting a private filesystem location.

Relation retrieval defaults to `princeton-nlp/sup-simcse-roberta-large`; set `SIMCSE_MODEL_PATH` to a local copy to avoid downloads. The retained source falls back to `sentence-transformers/all-MiniLM-L6-v2` if the chosen SimCSE model fails to load. Inspect logs and use the intended model for reproducibility.

## Verification scope

This implementation targets text-agent training with FSDP and vLLM. Unrelated inherited VLA examples were omitted because their worker imports are incomplete in the selected source snapshot. Other optional VERL backends are not covered by this release recipe.

CPU unit tests cover matching, reward semantics, data validation and launcher argument construction. Static auditing covers Python syntax, shell syntax and selected release hygiene patterns.
