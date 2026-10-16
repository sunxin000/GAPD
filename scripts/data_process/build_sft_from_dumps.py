#!/usr/bin/env python3
"""
Convert validation/rollout dumps (JSONL) from rejection sampling into an SFT dataset (parquet).

Input dump format (each JSON line):
{
  "input": "<user prompt text>",
  "output": "<assistant response text>",
  "gts": { ... ground truth ... },
  "score": 0.92,
  "step": 0,
  ... optional extra keys ...
}

Output parquet schema (multi-turn friendly):
- messages: list[dict] with two turns: user -> assistant
- prompt: str (redundant, for quick inspection)
- response: str (redundant, for verification scripts)
- reward: float
- reward_model: dict (wraps original gts if available)
- extra_info: dict (step, and all passthrough fields if any)

Usage:
  python scripts/data_process/build_sft_from_dumps.py \
    --dump_dir data/webqsp_runs/validation \
    --output_file data/webqsp_sft_dataset/train.parquet \
    --reward_threshold 0.8 \
        --limit 0 \
        --min_mid_f1 0.9 \
        --structure_reward_eq 0.1 \
        [--info_role tool|user|assistant_masked]

Notes:
- Works for both validation_data_dir and rollout_data_dir produced by `main_ppo_kbqa.py`.
- Filters samples by score >= reward_threshold.
- Merges multiple *.jsonl files in the directory (sorted by step).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import datasets


@dataclass
class DumpEntry:
    input: str
    output: str
    score: float
    step: int | None
    gts: Any | None
    passthrough: Dict[str, Any]

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "DumpEntry":
        input_text = d.get("input", "")
        output_text = d.get("output", "")
        score = d.get("score", None)
        step = d.get("step", None)
        gts = d.get("gts", None)

        passthrough = {k: v for k, v in d.items() if k not in {"input", "output", "score", "step", "gts"}}
        return DumpEntry(
            input=input_text,
            output=output_text,
            score=float(score) if score is not None else None,
            step=step,
            gts=gts,
            passthrough=passthrough,
        )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip broken lines; could log if needed
                continue


def load_dump_entries(dump_dir: Path) -> List[DumpEntry]:
    files = sorted(dump_dir.glob("*.jsonl"), key=lambda p: p.stem)
    entries: List[DumpEntry] = []
    for fp in files:
        for obj in iter_jsonl(fp):
            try:
                entries.append(DumpEntry.from_json(obj))
            except Exception:
                continue
    return entries


def to_sft_row(e: DumpEntry, info_role: str = "tool") -> Dict[str, Any]:
    def clean_prompt_hints(text: str) -> str:
        """Remove action hints and post-question guidance added during rejection sampling.

        Patterns removed:
        - Reference hint block starting with "**Reference Action Sequence (for guidance):**" up to a line with only '---'.
        - Example block starting with "**Example Solution Approach:**" up to the next blank line or '---'.
        - Hidden hint placeholder lines like "[HINT_PLACEHOLDER: ...]".
        - Post-question guidance sentence appended after Question.
        """

        s = text

        # Remove Reference Action Sequence block (up to --- delimiter)
        s = re.sub(
            r"\*\*Reference Action Sequence \(for guidance\):\*\*[\s\S]*?\n---\n",
            "",
            s,
            flags=re.IGNORECASE,
        )

        # Remove Example block
        s = re.sub(
            r"\*\*Example Solution Approach:\*\*[\s\S]*?(?:\n\n|\n---\n)",
            "",
            s,
            flags=re.IGNORECASE,
        )

        # Remove hidden placeholder
        s = re.sub(r"\[HINT_PLACEHOLDER:.*?\]", "", s, flags=re.IGNORECASE)

        # Remove known post-question guidance sentence (inserted after Question: ...)
        post_q = (
            "The reference already provides a complete action sequence to solve the problem. "
            "You only need to provide concise, effective reasoning. Immediately after <think>, issue the next step "
            "using <action> so the environment can execute it. Keep your reasoning brief—just explain why you will "
            "take the next action."
        )
        s = s.replace("\n" + post_q, "")

        # Also handle cases where trailing spaces or missing leading newline exist
        s = s.replace(post_q, "")

        # Collapse any excessive blank lines created
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def parse_response_to_messages(resp: str) -> list[dict]:
        """Split a single response string into multi-turn messages.

        - Assistant segments (think/action/answer) are kept as assistant messages.
        - Environment responses within <information>...</information> become tool messages (masked by dataset).
        - Keeps original tag wrappers to preserve structure learning.
        """
        if not resp or not isinstance(resp, str):
            return [{"role": "assistant", "content": resp or ""}]

        parts = []
        pattern = re.compile(r"<(think|action|information|answer)>([\s\S]*?)</\1>", re.IGNORECASE)
        pos = 0
        cur_assistant = []

        for m in pattern.finditer(resp):
            # Text before the tag
            if m.start() > pos:
                pre = resp[pos : m.start()]
                if pre.strip():
                    cur_assistant.append(pre)

            tag = m.group(1).lower()
            content = m.group(2)

            if tag in {"think", "action", "answer"}:
                cur_assistant.append(f"<{tag}>{content}</{tag}>")
            elif tag == "information":
                # Flush current assistant segment
                if cur_assistant:
                    parts.append({"role": "assistant", "content": "".join(cur_assistant).strip()})
                    cur_assistant = []
                info_text = f"<information>{content}</information>"
                if info_role == "tool":
                    parts.append({"role": "tool", "content": info_text.strip()})
                elif info_role == "user":
                    parts.append({"role": "user", "content": info_text.strip()})
                elif info_role == "assistant_masked":
                    parts.append({"role": "assistant", "content": info_text.strip(), "loss_mask": 0})
                else:
                    # default fallback
                    parts.append({"role": "tool", "content": info_text.strip()})

            pos = m.end()

        # Trailing text after last tag
        if pos < len(resp):
            tail = resp[pos:]
            if tail.strip():
                cur_assistant.append(tail)

        if cur_assistant:
            parts.append({"role": "assistant", "content": "".join(cur_assistant).strip()})

        # Fallback: if nothing parsed into parts, return single assistant message
        if not parts:
            parts = [{"role": "assistant", "content": resp}]

        return parts

    cleaned_input = clean_prompt_hints(e.input)
    assistant_parts = parse_response_to_messages(e.output)

    messages = [{"role": "user", "content": cleaned_input}] + assistant_parts

    reward_model = None
    if e.gts is not None:
        reward_model = {"style": "rule", "ground_truth": e.gts}

    extra = {"step": e.step}
    # Merge passthrough fields for traceability
    extra.update(e.passthrough or {})

    return {
        "messages": messages,
        # Redundant columns for tooling/inspection
        "prompt": cleaned_input,
        "response": e.output,
        "reward": e.score,
        "reward_model": reward_model,
        "extra_info": extra,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT parquet from validation/rollout dumps (JSONL)"
    )
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing *.jsonl dump files")
    parser.add_argument("--output_file", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--reward_threshold", type=float, default=0.0, help="Keep samples with score >= threshold")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of kept samples (0 = all)")
    parser.add_argument(
        "--info_role",
        type=str,
        default="tool",
        choices=["tool", "user", "assistant_masked"],
        help="How to represent <information> segments in messages (all are masked during training except assistant_masked which uses loss_mask=0)",
    )
    parser.add_argument(
        "--min_mid_f1",
        type=float,
        default=None,
        help="Keep samples with mid_f1 >= this value (requires 'mid_f1' field in dumps)",
    )
    parser.add_argument(
        "--structure_reward_eq",
        type=float,
        default=None,
        help="Keep samples with structure_reward == this value (epsilon=1e-6), requires 'structure_reward' field in dumps",
    )
    parser.add_argument(
        "--save_json_sample",
        action="store_true",
        help="Also save a random JSONL sample for manual inspection",
    )
    parser.add_argument(
        "--json_sample_num",
        type=int,
        default=100,
        help="Number of samples to save when --save_json_sample is set",
    )
    parser.add_argument(
        "--json_sample_out",
        type=str,
        default=None,
        help="Output JSONL path (default: <output_file>_sample.jsonl)",
    )

    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    assert dump_dir.exists() and dump_dir.is_dir(), f"Dump dir not found: {dump_dir}"

    entries = load_dump_entries(dump_dir)
    if not entries:
        raise SystemExit(f"No entries found in {dump_dir}")

    # Metric-based filtering (mid_f1, structure_reward) using passthrough fields in each entry
    def pass_metric_filters(e: DumpEntry) -> bool:
        # Default pass
        ok = True
        # mid_f1 >= min_mid_f1
        if args.min_mid_f1 is not None:
            mid_f1_val: Optional[float] = None
            # try common keys
            for key in ("mid_f1", "mid_f1_score", "mid_f1_mean"):
                if key in e.passthrough and isinstance(e.passthrough[key], (int, float)):
                    mid_f1_val = float(e.passthrough[key])
                    break
            if mid_f1_val is None:
                return False
            if not (mid_f1_val >= float(args.min_mid_f1)):
                return False

        # structure_reward equality with tolerance
        if args.structure_reward_eq is not None:
            sr_val: Optional[float] = None
            for key in ("structure_reward", "structure_reward_score", "structure_reward_mean"):
                if key in e.passthrough and isinstance(e.passthrough[key], (int, float)):
                    sr_val = float(e.passthrough[key])
                    break
            if sr_val is None:
                return False
            if abs(sr_val - float(args.structure_reward_eq)) > 1e-6:
                return False
        return ok

    # Filter by metrics first, then reward threshold
    metric_filtered = [e for e in entries if pass_metric_filters(e)]

    # Filter by reward threshold
    filtered = [e for e in metric_filtered if (e.score is not None and e.score >= args.reward_threshold)]
    if args.limit and args.limit > 0:
        filtered = filtered[: args.limit]

    if not filtered:
        raise SystemExit(
            f"No samples after filtering with threshold={args.reward_threshold}. Try lowering the threshold."
        )

    rows = [to_sft_row(e, info_role=args.info_role) for e in filtered]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(out_path))

    print(f"Saved {len(rows)} samples to {out_path}")
    print("Columns: messages, prompt, response, reward, reward_model, extra_info")

    # Optionally save a JSONL sample for quick manual inspection
    if args.save_json_sample:
        sample_n = min(int(args.json_sample_num), len(rows))
        # basic deterministic sample using step ordering
        sampled = rows[:sample_n] if sample_n == len(rows) else rows[:: max(1, len(rows)//sample_n)][:sample_n]
        json_out = (
            Path(args.json_sample_out)
            if args.json_sample_out
            else out_path.with_suffix(out_path.suffix + "").with_name(out_path.name + "_sample.jsonl")
        )
        with open(json_out, "w", encoding="utf-8") as f:
            for rec in sampled:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        print(f"Also wrote {sample_n} JSON samples to {json_out}")


if __name__ == "__main__":
    main()
