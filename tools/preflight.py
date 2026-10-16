"""Check local training prerequisites without launching Ray or downloading models."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    if sys.version_info < (3, 12):
        errors.append("Use Python 3.12+ for the bundled VERL source syntax.")
    for name in ("torch", "transformers", "vllm", "ray", "tensordict", "torchdata",
                 "hydra", "omegaconf", "pyodbc", "aioodbc", "SPARQLWrapper",
                 "pandas", "pyarrow", "sklearn", "tensorboard", "flash_attn"):
        if importlib.util.find_spec(name) is None:
            errors.append(f"Missing module: {name}")
    for name in ("TRAIN_FILE", "VAL_FILE"):
        if not os.environ.get(name) or not Path(os.environ[name]).is_file():
            errors.append(f"Set {name} to an existing parquet file.")
    for name in ("fb_roles", "fb_types", "reverse_properties"):
        if not (ROOT / "dataset" / "Freebase" / name).is_file():
            errors.append(f"Missing ontology resource: dataset/Freebase/{name}")
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA is unavailable; distributed training requires NVIDIA GPUs.")
        except Exception as exc:
            errors.append(f"PyTorch import failed ({type(exc).__name__}).")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Local prerequisites found. Verify the Virtuoso service separately before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
