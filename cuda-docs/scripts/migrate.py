"""One-time migration from chapter Markdown to section pages; refuses to overwrite pages."""
from pathlib import Path
from urllib.parse import unquote
import re, json, shutil

project = Path(__file__).resolve().parents[1]
source = project.parent / 'references/cuda-programming-guide/legacy'
docs = project / 'docs'
chapters = sorted(source.glob('CUDA编程指南-第[1-6]*-*.md'), key=lambda p: int(re.search(r'第(\d+)', p.name)[1]))
page_map, documents, groups = {}, [], []
original_headings = 0
for path in chapters:
    text = path.read_text()
    chapter = re.search(r'第(\d+)', path.name)[1]
    title = text.splitlines()[0].lstrip('# ')
    chapter_route = f'/chapter-{chapter}/'
    group = {'text': title, 'link': chapter_route, 'collapsed': chapter != '1', 'items': []}
    matches = list(re.finditer(r'^## (\d+\.\d+) (.+)$', text, re.M))
    footnotes = ''
    if '\n## 脚注\n' in text:
        text, footnotes = text.split('\n## 脚注\n', 1)
        matches = list(re.finditer(r'^## (\d+\.\d+) (.+)$', text, re.M))
    def start(m):
        prefix = text[:m.start()]
        anchor = re.search(r'<a id="sec-[\d-]+"></a>\s*$', prefix)
        return anchor.start() if anchor else m.start()
    intro = text[:start(matches[0])].strip() if matches else text.strip()
    for i, match in enumerate(matches):
        num, label = match.groups()
        route = f'/chapter-{chapter}/section-{num.replace(".", "-")}'
        chunk = text[start(match):start(matches[i+1]) if i+1 < len(matches) else len(text)].strip()
        # Page title becomes h1; retain original numbering and stable anchors.
        chunk = re.sub(r'^(#{2,6}) ', lambda m: m[1][1:]+' ', chunk, flags=re.M)
        if footnotes:
            used = set(re.findall(r'\[\^(\d+)\]', chunk))
            definitions = [line for line in footnotes.strip().splitlines() if any(line.startswith(f'[^{n}]:') for n in used)]
            if definitions: chunk += '\n\n'+'\n\n'.join(definitions)
        translated = chapter == '1'
        front = '---\ntitle: '+json.dumps(num+' '+label,ensure_ascii=False)+'\ndescription: '+json.dumps('CUDA 编程指南 · '+title,ensure_ascii=False)+'\n---\n\n'
        # Place the section status under its page title, before deeper headings.
        if not translated:
            chunk = re.sub(r'^(# .+)$',r'\1\n\n::: info 待翻译\n本节已建立目录框架，正文尚未翻译。\n:::',chunk,count=1,flags=re.M)
        documents.append((route,front+chunk+'\n'))
        group['items'].append({'text':num+' '+label,'link':route})
        for sec in re.findall(r'^#{1,6} ([\d.]+) ',chunk,re.M):
            page_map[(path.name,'sec-'+sec.replace('.','-'))]=route
            original_headings += 1
        for anchor in re.findall(r'<a id="(figure-\d+)"></a>',chunk):
            page_map[(path.name,anchor)]=route
    page_map[(path.name,'')]=chapter_route
    intro += '\n\n## 本章目录\n\n'+'\n'.join(f'- [{item["text"]}]({item["link"]})' for item in group['items'])+'\n'
    if chapter!='1':
        intro=intro.split('\n\n## 本章目录')[0]+'\n\n本章正文待翻译，点击下方小节进入对应页面。\n\n## 本章目录\n\n'+'\n'.join(f'- [{item["text"]}]({item["link"]})' for item in group['items'])+'\n'
    documents.append((chapter_route,intro))
    groups.append(group)

first_chapter=chapters[0].name
for route, content in documents:
    def rewrite(m):
        label, target=m.groups()
        if target.startswith(('https://','http://','/chapter-')):return m[0]
        if target.startswith('images/'):
            return f'{label}(/'+target+')'
        filename,_,anchor=unquote(target).partition('#')
        key=(filename or first_chapter,anchor)
        if key in page_map:
            return f'{label}({page_map[key]}'+('#'+anchor if anchor else '')+')'
        raise ValueError(f'Unmapped link: {target} in {route}')
    content=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',rewrite,content)
    content=content.replace('第 2 章及之后的章节目前仅有标题框架，尚未翻译正文。','第 2 章及之后的章节目前仅有标题框架，尚未翻译正文。')
    dest=docs/(route.strip('/')+'/index.md' if route.endswith('/') else route.strip('/')+'.md')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(content)
shutil.copytree(source/'images',docs/'public'/'images',dirs_exist_ok=True)
(docs/'.vitepress'/'sidebar.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2)+'\n')
print(f'Migrated {len(groups)} chapter overviews and {sum(len(g["items"]) for g in groups)} section pages; preserved {original_headings+len(groups)} numbered headings.')
