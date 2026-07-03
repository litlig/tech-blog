---
name: notebook-to-post
description: Convert a Jupyter notebook (.ipynb, usually a GitHub URL) into a Hugo + Blowfish blog post for this repo. Use when the user asks to turn a notebook into a blog post / companion post, or says "convert this notebook".
---

# Notebook → Blog post

Turn a Jupyter notebook into a page-bundle post under `content/posts/<slug>/index.md`,
following this blog's Blowfish conventions. Do it with the Python **standard library only** —
no Jupyter, no `pip install`, no PIL.

## Editorial rules (do not violate)

- **Follow the notebook's prose closely.** Fix only grammar, typos, and descriptions that are
  *logically inaccurate*. Do not rewrite the author's voice or restructure their argument.
- **Add almost nothing of your own** — at most a sentence or two, and only where a figure or
  transition genuinely needs a caption. When in doubt, add nothing.
- **Skip large code blocks** (plotting, training loops, boilerplate). Keep only the few *core,
  interesting* snippets that carry the idea — trim them to the essential lines.

## Steps

### 1. Fetch the notebook

```bash
curl -sL <raw-github-url> -o /tmp/nb.ipynb
```
For a GitHub blob URL, use the `raw.githubusercontent.com` form
(`https://raw.githubusercontent.com/<user>/<repo>/<branch>/<path>.ipynb`).

### 2. Dump cells to read them

Run a stdlib script to print each cell's type + source and flag image outputs, so you can read
the prose and decide which code/figures to keep:

```python
import json
nb = json.load(open('/tmp/nb.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f"\n===== CELL {i} [{c['cell_type']}] =====\n{src}")
    if c['cell_type'] == 'code':
        n = sum(1 for o in c.get('outputs', []) for k in o.get('data', {}) if k.startswith('image'))
        if n: print(f"[-> {n} image output(s)]")
```

### 3. Extract figures into the page bundle

Create `content/posts/<slug>/` and write each `image/png` output to disk, then rename to something
meaningful (e.g. `data.png`, `cfg-weights.png`):

```python
import json, base64
nb = json.load(open('/tmp/nb.ipynb'))
out = 'content/posts/<slug>'
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        v = o.get('data', {}).get('image/png')
        if v:
            b = base64.b64decode(''.join(v) if isinstance(v, list) else v)
            open(f'{out}/cell{i}.png', 'wb').write(b)
```

### 4. Write `index.md`

Front matter (YAML), `draft: false` to publish:

```yaml
---
title: "<title>"
date: <today, YYYY-MM-DD>
draft: false
categories: ["learning"]
tags: ["diffusion", "notebook", ...]
summary: "<one-line summary>"
---
```

Then the body, in this order:

1. `{{< katex >}}` — **required** for any math. Blowfish only loads KaTeX when this shortcode is
   present; front-matter `math: true` is ignored.
2. A blockquote intro noting the code is trimmed, linking the original notebook, and the Colab badge:
   ```md
   > The code below is trimmed to the essentials; the full version lives in the original notebook:
   > [**github.com/<user>/<repo>/<file>.ipynb**](<blob-url>)
   >
   > <a href="<colab-url>" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
   ```
3. If it's a follow-up to another post, cross-link with `{{< ref "posts/<other-slug>" >}}`.
4. The prose + kept code snippets + figures (`![alt](figure.png)`).

**Math conversion (important):** convert notebook `$...$` / `$$...$$` to Blowfish's delimiters —
inline `\(...\)`, display `\[...\]`. Do **not** rely on single `$...$` inline math.

### 5. Verify the build

```bash
hugo --gc --minify
```
Then confirm on `public/posts/<slug>/index.html`:
- KaTeX is referenced (`grep -c katex`),
- figures are wired (`grep 'src=.*\.png'`) and exist on disk,
- any `{{< ref >}}` link resolved (no build error).

Note: local builds show `localhost:1313` URLs; production (`build.sh`) uses the real baseURL.

## Reference

- Config lives in `config/_default/` (Blowfish TOML). Posts are page bundles under `content/posts/`.
- The two existing examples to match for tone/structure: `content/posts/diffusion-2d-notebook/`
  and `content/posts/guided-2d-diffusion/`.
