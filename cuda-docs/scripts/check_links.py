"""Validate built local links, section coverage, and source image preservation."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse, ast, hashlib, json, re

project=Path(__file__).resolve().parents[1]
legacy=project.parent/'references/cuda-programming-guide/legacy'
docs=project/'docs'
parser=argparse.ArgumentParser()
parser.add_argument('--built', type=Path, default=docs/'.vitepress/dist')
parser.add_argument('--base', default='/')
args=parser.parse_args()
built=args.built.resolve()
assert built.is_dir(), 'Run the VitePress build first.'
class Page(HTMLParser):
    def __init__(self,path):
        super().__init__();self.ids=set();self.links=[]
        self.feed(path.read_text())
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.add(attrs['id'])
        if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])
        if tag=='img' and 'src' in attrs:self.links.append(attrs['src'])
pages={p:Page(p) for p in built.rglob('*.html')}
checked=0
for path,page in pages.items():
    for href in page.links:
        url=urlsplit(href)
        if url.scheme or url.netloc:continue
        target=unquote(url.path)
        if target.startswith('/'):
            assert target.startswith(args.base), f'Link escapes site base: {href}'
            target=target[len(args.base):]
            base=built/target
        else:base=path.parent/target
        if not url.path:base=path
        elif url.path.endswith('/'):base=base/'index.html'
        elif not base.suffix:base=base.with_suffix('.html')
        assert base.is_file(), f'Missing target {href} from {path.relative_to(built)}'
        if url.fragment and base in pages:
            assert unquote(url.fragment) in pages[base].ids, f'Missing anchor {href} from {path.relative_to(built)}'
        checked+=1
source_nums=[]
source_chapters=list(legacy.glob('CUDA编程指南-第[1-6]*-*.md'))
assert len(source_chapters)==6, 'Missing legacy chapter sources.'
for path in source_chapters:
    chapter=re.search(r'第(\d+)',path.name)[1]
    source_nums.append(chapter)
    source_nums.extend(re.findall(r'^#{2,6} ([\d.]+) ',path.read_text(),re.M))
actual=[]
for path in docs.glob('chapter-*/*.md'):
    chapter=path.parent.name.split('-')[1]
    if path.name=='index.md':actual.append(chapter)
    else:actual.extend(re.findall(r'^#{1,6} ([\d.]+) ',path.read_text(),re.M))
assert sorted(source_nums)==sorted(actual), 'Section numbering mismatch'
source_images=list((legacy/'images/chapter-01').glob('*.png'))
assert source_images, 'Missing legacy source images.'
for image in source_images:
    copied=docs/'public/images/chapter-01'/image.name
    assert hashlib.sha256(image.read_bytes()).digest()==hashlib.sha256(copied.read_bytes()).digest()
sections=sum(len(g['items']) for g in json.loads((docs/'.vitepress/sidebar.json').read_text()))
python_blocks=0
for path in (docs/'chapter-2').glob('section-*.md'):
    body=path.read_text()
    assert '正文尚未翻译' not in body and '{{C' not in body and '{{FIG' not in body, path
    assert not any(c in body for c in ('\uffff','\ufffd')), path
    for snippet in re.findall(r'```python\n([\s\S]*?)\n```',body):
        ast.parse(snippet)
        python_blocks+=1
images=list((docs/'public/images').rglob('*.png'))
print(f'PASS: {sections} section pages, {len(actual)} original numbered headings, {checked} local links/anchors, {len(images)} image assets, {python_blocks} Python snippets parsed.')
