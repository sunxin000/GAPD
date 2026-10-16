"""Static source-release checks; prints locations, never potential secret values."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {"__pycache__", ".git", ".venv", "build", "dist"}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "token": re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"),
    "private-path": re.compile(r"/(?:ossfs|aml/share|nas)/|/" + r"Users/[^/\s]+/|/" + r"home/[^/\s]+/"),
    "internal-domain": re.compile(r"(?:code\.alipay\.com|yuque\.antfin\.com|[\w.-]+\.antgroup\.com)"),
    "credential-assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"']([^\"'\n]{8,})[\"']"),
}


def audit(root=ROOT):
    findings = []
    counts = {"files": 0, "python": 0, "shell": 0, "syntax_warnings": 0}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            findings.append({"path": str(rel), "kind": "symlink"})
            continue
        if not path.is_file():
            continue
        counts["files"] += 1
        if path.stat().st_size > 2 * 1024 * 1024:
            findings.append({"path": str(rel), "kind": "large-file"})
        if path.suffix.lower() in {".pt", ".pth", ".safetensors", ".parquet", ".pdf", ".docx", ".xlsx", ".zip"}:
            findings.append({"path": str(rel), "kind": "non-source-artifact"})
        if path.name == ".env" or any(x in {"data", "dataset", "checkpoints", "logs", "gapd_dump", "paper"} for x in rel.parts[:1]):
            findings.append({"path": str(rel), "kind": "private-or-generated-directory"})
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"path": str(rel), "kind": "binary-file"})
            continue
        if path.suffix == ".py":
            counts["python"] += 1
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", SyntaxWarning)
                    ast.parse(source, filename=str(rel))
                counts["syntax_warnings"] += len(captured)
            except SyntaxError as exc:
                findings.append({"path": str(rel), "line": exc.lineno, "kind": "python-syntax"})
        if path.suffix == ".sh":
            counts["shell"] += 1
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True)
            if result.returncode:
                findings.append({"path": str(rel), "kind": "shell-syntax"})
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(source):
                findings.append({"path": str(rel), "line": source.count("\n", 0, match.start()) + 1, "kind": kind})
    return {"counts": counts, "findings": findings,
            "limitations": "Pattern-based source checks; excludes local build/cache metadata. Does not audit Git history, dependencies or execute GPU/KB integration."}


def main():
    if sys.version_info < (3, 12):
        raise SystemExit("Use Python 3.12+ to validate this source snapshot.")
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(result["findings"]))


if __name__ == "__main__":
    raise SystemExit(main())
