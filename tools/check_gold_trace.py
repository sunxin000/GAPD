"""Validate gold-trace coverage in prepared parquet files before GAPD training."""
import argparse
from collections.abc import Mapping
import json


def normalize(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def trace_coverage(rows):
    total = present = sample_nonempty = steps = steps_nonempty = malformed = 0
    for row in rows:
        total += 1
        extra = normalize(row.get("extra_info"))
        trace = normalize(extra.get("gold_trace")) if isinstance(extra, Mapping) else None
        if not isinstance(trace, list) or not trace:
            continue
        present += 1
        found = False
        for raw_step in trace:
            steps += 1
            step = normalize(raw_step)
            if not isinstance(step, Mapping):
                malformed += 1
                continue
            prev = normalize(step.get("prev_mid_set"))
            mids = normalize(step.get("next_mid_set"))
            if (not isinstance(prev, (list, set, tuple)) or
                    not isinstance(mids, (list, set, tuple)) or
                    not all(isinstance(mid, str) for mid in [*prev, *mids]) or
                    not isinstance(step.get("per_turn_hint"), str) or
                    not step["per_turn_hint"].strip()):
                malformed += 1
                continue
            if mids:
                steps_nonempty += 1
                found = True
        sample_nonempty += int(found)
    return {"rows": total, "traces": present, "samples_nonempty": sample_nonempty,
            "steps": steps, "steps_nonempty": steps_nonempty, "malformed_steps": malformed,
            "present_rate": present / total if total else 0.0,
            "sample_nonempty_rate": sample_nonempty / total if total else 0.0,
            "step_nonempty_rate": steps_nonempty / steps if steps else 0.0}


def is_acceptable(stats, minimum=0.9):
    return (stats["rows"] > 0 and stats["malformed_steps"] == 0 and
            min(stats[k] for k in ("present_rate", "sample_nonempty_rate", "step_nonempty_rate")) >= minimum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--minimum", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 <= args.minimum <= 1:
        parser.error("--minimum must be between 0 and 1")
    import pandas as pd
    passed = True
    for path in args.paths:
        stats = trace_coverage(pd.read_parquet(path).to_dict("records"))
        print(json.dumps({"path": path, **stats}, ensure_ascii=False))
        passed &= is_acceptable(stats, args.minimum)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
