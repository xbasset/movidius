"""Verify published evidence, local documentation links and experiment boundaries."""

import argparse
import ast
import hashlib
from importlib.util import resolve_name
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from movidius_lab.paths import ROOT


def module_exists(name):
    path = ROOT.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--history", action="store_true", help="Compare artifacts with their original Git commits")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "docs/evidence.json").read_text())
    errors = []
    paths = set()
    for entry in manifest["artifacts"]:
        path = ROOT / entry["path"]
        paths.add(path)
        if not path.is_file():
            errors.append(f"Missing evidence: {entry['path']}")
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            errors.append(f"Changed evidence: {entry['path']}")
        if path.suffix == ".json":
            try:
                json.loads(data, parse_constant=reject_nonfinite)
            except ValueError as error:
                errors.append(f"Invalid JSON: {entry['path']}: {error}")
        if args.history:
            original = subprocess.run(["git", "show", f"{entry['recorded_at_commit']}:{entry['original_path']}"],
                                      cwd=ROOT, capture_output=True)
            if original.returncode or original.stdout != data:
                errors.append(f"History mismatch: {entry['path']}")
    actual = {p for p in (ROOT / "experiments").glob("*/results/**/*")
              if p.is_file() and "private" not in p.parts and "raw" not in p.parts}
    if paths != actual:
        errors.append(f"Evidence inventory differs: {sorted(str(p.relative_to(ROOT)) for p in paths ^ actual)}")

    docs = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md")) + list((ROOT / "experiments").rglob("*.md"))
    link_count = 0
    for path in docs:
        source = path.read_text()
        for target in re.findall(r"\]\(([^)\s]+)\)", source):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            link_count += 1
            if not (path.parent / unquote(parsed.path)).exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
        for name in re.findall(r"\bpython(?:3)?\s+(?:-u\s+)?-m\s+([\w.]+)", source):
            if not module_exists(name):
                errors.append(f"Missing documented module in {path.relative_to(ROOT)}: {name}")
        for name in re.findall(r"\bbash\s+([\w/.-]+\.sh)", source):
            if not (ROOT / name).is_file():
                errors.append(f"Missing documented shell script in {path.relative_to(ROOT)}: {name}")

    sources = list((ROOT / "movidius_lab").glob("*.py")) + list((ROOT / "experiments").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    for path in sources:
        owner = path.relative_to(ROOT).parts[1] if path.is_relative_to(ROOT / "experiments") else None
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            if isinstance(node, ast.ImportFrom) and node.level:
                package = ".".join(path.relative_to(ROOT).parent.parts)
                names = [resolve_name("." * node.level + (node.module or alias.name), package)
                         for alias in node.names]
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if name.split(".")[0] not in {"experiments", "movidius_lab", "scripts"}:
                    continue
                if not module_exists(name):
                    errors.append(f"Missing local import in {path.relative_to(ROOT)}: {name}")
                if name.startswith("experiments.") and name.split(".")[1] != owner:
                    errors.append(f"Cross-experiment import in {path.relative_to(ROOT)}: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(paths)} unchanged evidence files, {link_count} local links, {len(sources)} Python sources; documented entry points and import boundaries valid.")
    if args.history:
        print("OK: every evidence file matches its original Git commit.")


if __name__ == "__main__":
    main()
