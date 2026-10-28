"""Generate the notes site from notebook/ without maintaining duplicate articles."""

from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import json
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook"
SITE = ROOT / "cuda-docs/notes"
ARTICLES = SITE / "articles"
ASSETS = SITE / "public/notebook"
REPO = "https://github.com/husichao666/cuda-learning"
IMAGES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}


def route(path):
    return "/articles/" + path.relative_to(NOTEBOOK).with_suffix("").as_posix()


def rewrite_link(match, source):
    target = match[2].strip().strip("<>")
    url = urlsplit(target)
    if url.scheme or url.netloc or not url.path:
        return match[0]
    destination = (source.parent / unquote(url.path)).resolve()
    relative = destination.relative_to(ROOT)
    if not destination.exists():
        raise ValueError(f"Missing article link: {source} -> {target}")
    if destination.is_relative_to(NOTEBOOK) and destination.suffix == ".md":
        link = "/articles/" if destination.name == "README.md" else route(destination)
    elif match[1].startswith("!") and destination.suffix.lower() in IMAGES:
        asset = ASSETS / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, asset)
        link = "/notebook/" + relative.as_posix()
    else:
        kind = "tree" if destination.is_dir() else "blob"
        link = f"{REPO}/{kind}/main/{relative.as_posix()}"
    link = quote(link, safe="/:@-._~")
    if url.query:
        link += "?" + url.query
    if url.fragment:
        link += "#" + url.fragment
    return match[1] + "(" + link + ")"


def main():
    for generated in (ARTICLES, ASSETS):
        if generated.exists():
            shutil.rmtree(generated)
        generated.mkdir(parents=True)
    sources = sorted(p for p in NOTEBOOK.rglob("*.md") if p.name != "README.md")
    if not sources:
        raise ValueError("No notebook articles found")
    sidebar = []
    for source in sources:
        body = source.read_text(encoding="utf-8")
        heading = re.search(r"^# (.+)$", body, flags=re.MULTILINE)
        if heading is None:
            raise ValueError(f"Article needs an H1 title: {source}")
        title = heading[1]
        sidebar.append({"text": title, "link": route(source)})
        # Code examples are copied verbatim; only prose links are rewritten.
        parts = re.split(r"(^```[^\n]*\n.*?^```[^\n]*$)", body,
                         flags=re.MULTILINE | re.DOTALL)
        for index in range(0, len(parts), 2):
            parts[index] = re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)",
                                  lambda m: rewrite_link(m, source), parts[index])
        output = ARTICLES / source.relative_to(NOTEBOOK)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(parts), encoding="utf-8")
    (ARTICLES / "sidebar.json").write_text(
        json.dumps(sidebar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 学习笔记", "", "从 CUDA 基础开始，按学习进度持续补充文章。", ""]
    lines.extend(f"- [{item['text']}]({quote(item['link'])})" for item in sidebar)
    lines.extend(["", f"[配套算子代码]({REPO}/tree/main/operators)", ""])
    (ARTICLES / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Prepared {len(sources)} notebook article(s).")


if __name__ == "__main__":
    main()
