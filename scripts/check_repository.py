"""Check repository links, Markdown fences, and the first-week code entry point."""

from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SKIP = {
    ".git", ".pnpm-store", "node_modules", ".vitepress", "__pycache__",
    ".venv", "build",
}


def markdown_files(directory):
    for path in sorted(directory.iterdir()):
        if path.name in SKIP or path == ROOT / "references/local":
            continue
        if path.is_symlink():
            continue
        if path.is_dir():
            yield from markdown_files(path)
        elif path.suffix == ".md":
            yield path


def main():
    failures = []
    links_checked = 0
    pages_checked = 0
    for path in markdown_files(ROOT):
        # VitePress routes and anchors have a dedicated post-build checker.
        if any(ROOT / folder in path.parents for folder in
               ("cuda-docs/docs", "cuda-docs/notes")):
            continue
        text = path.read_text(encoding="utf-8")
        pages_checked += 1
        fences = re.findall(r"^```.*$", text, re.MULTILINE)
        if len(fences) % 2:
            failures.append(f"Unclosed code fence: {path.relative_to(ROOT)}")
        prose = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text,
                       flags=re.MULTILINE | re.DOTALL)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", prose):
            target = target.strip().strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            destination = path.parent / unquote(url.path)
            if not destination.exists():
                failures.append(
                    f"Missing link: {path.relative_to(ROOT)} -> {target}"
                )
            links_checked += 1

    note = ROOT / "notebook/第一周-硬件介绍与入门代码.md"
    source = ROOT / "operators/cuda/vector_add/vector_add.cu"
    blocks = re.findall(r"```cpp\n(.*?)\n```", note.read_text(encoding="utf-8"),
                        flags=re.DOTALL)
    complete = [block for block in blocks if "#include <cuda_runtime.h>" in block]
    if len(complete) != 1 or source.read_text(encoding="utf-8").rstrip() != complete[0].rstrip():
        failures.append("First-week article and vector_add.cu are out of sync")

    report_path = ROOT / "operators/cuda/vector_add/validation/leetgpu-functional.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "FUNCTIONAL":
        failures.append("Historical LeetGPU report is missing its execution mode")
    for name in ("baseline", "boundaries", "random", "events", "roundtrip"):
        output = report.get("cases", {}).get(name, "")
        if "Exit status: 0" not in output:
            failures.append(f"Incomplete historical validation log: {name}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"PASS: {pages_checked} Markdown files, {links_checked} local links, "
          "article/source consistency and historical validation metadata.")
    print("This command does not compile or execute CUDA code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
