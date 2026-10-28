"""Check both Pages sites, including subpath links, anchors, JS, CSS and images."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json


ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "build/pages"
BASE = "/cuda-learning/"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in {"a", "link"} and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in {"img", "script"} and attrs.get("src"):
            self.links.append(attrs["src"])


def main():
    assert (BUILT / "index.html").is_file(), "Missing learning notes homepage"
    assert (BUILT / "guide/index.html").is_file(), "Missing independent guide homepage"
    sidebar = json.loads((ROOT / "cuda-docs/notes/articles/sidebar.json").read_text())
    for article in sidebar:
        assert (BUILT / (article["link"].lstrip("/") + ".html")).is_file(), article
    pages = {path.resolve(): Page(path) for path in BUILT.rglob("*.html")}
    checked = 0
    for path, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = unquote(url.path)
            if not target:
                destination = path
            elif target.startswith("/"):
                assert target.startswith(BASE), f"Link escapes Pages base: {href} in {path}"
                destination = BUILT / target[len(BASE):]
            else:
                destination = path.parent / target
            if url.path:
                if url.path.endswith("/"):
                    destination /= "index.html"
                elif not destination.suffix:
                    destination = destination.with_suffix(".html")
            destination = destination.resolve()
            assert destination.is_relative_to(BUILT), f"Link escapes output: {href}"
            assert destination.is_file(), f"Missing {href} in {path.relative_to(BUILT)}"
            if url.fragment and destination in pages:
                assert unquote(url.fragment) in pages[destination].ids, f"Missing anchor: {href}"
            checked += 1
    print(f"PASS: {len(sidebar)} articles, {len(pages)} HTML pages, "
          f"{checked} local links, anchors and assets across both Pages sites.")


if __name__ == "__main__":
    main()
