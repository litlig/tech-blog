---
name: notebook-to-post
description: Convert a Jupyter notebook (.ipynb, usually a GitHub URL) into a Hugo + Blowfish blog post for this repo. Use when the user asks to turn a notebook into a blog post / companion post, or says "convert this notebook".
---

# Notebook → Blog post

Turn a Jupyter notebook into a page-bundle post under `content/posts/<slug>/index.md`,
following this blog's Blowfish conventions. Do it with the Python **standard library only** —
no Jupyter, no `pip install`, no PIL.

## Editorial doctrine: you are the mechanic, not the author

The whole value of these posts is that a **human did the work and is writing about it in their own
voice**. Readers now reflexively detect and dismiss AI-shaped prose, and they are right to — smooth,
templated writing reads as slop even when the underlying work is real and original. Your job is to
*transport* the author's notebook prose into the blog and fix what is broken. It is **not** to write,
improve, polish, or "make it flow." The author's words stay the author's words.

**Rules (do not violate):**

- **Preserve the author's voice verbatim.** Carry sentences across as written. Keep their phrasing,
  rhythm, informality, and idiosyncrasies. A slightly awkward-but-clear human sentence is *better*
  than a smooth one you rewrote — the awkwardness is the fingerprint that proves a person wrote it.
- **Narrow the fix scope to real defects only.** Change a word *only* when it is (a) an outright
  spelling/grammar error, or (b) factually or logically wrong. If a sentence is clunky, casual, or
  imperfect but clear and correct, **leave it exactly as written.** "Could be smoother" is never a
  reason to edit. When unsure, don't touch it.
- **Add nothing in your own voice.** No invented captions, no connective transitions, no summarizing
  "takeaways," no framing sentences, no closing questions. If a figure genuinely needs a caption, use
  the author's own words from the notebook; if there are none, write a flat factual alt text (e.g.
  "training loss vs. steps"), not a crafted sentence.
- **Never impose AI-tell structure.** Do not add: the "I built/implemented X from scratch to see what
  it actually does" framing, tidy rule-of-three lists, balanced em-dash clauses, bulleted "key
  takeaways," or rhetorical engagement-bait questions at the end. If the author didn't write it,
  it doesn't go in.
- **Skip large code blocks** (plotting, training loops, boilerplate). Keep only the few *core,
  interesting* snippets that carry the idea — trim them to the essential lines. (This is the one
  place you actively cut; everything else is preserve-in-place.)

**Litmus test before you write a line:** would the author read this sentence and recognize it as their
own? If you generated it and it isn't in the notebook, delete it unless it is a pure mechanical
necessity. When the notebook prose is thin, the post is allowed to be short — a short post in the
author's voice beats a padded one in yours.

Front matter (`summary`, `tags`) may be plainly functional and keyword-bearing; that's metadata, not
voice. The **title and body prose must be the author's.** If the author wants a different title,
that's their call, not yours to invent.

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
4. The prose (the author's own, carried over per the editorial doctrine above) + kept code
   snippets + figures (`![alt](figure.png)`).

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
- Match the existing posts (`content/posts/diffusion-2d-notebook/`, `content/posts/guided-2d-diffusion/`)
  for **mechanics only** — front matter shape, `{{< katex >}}`, the trimmed-code blockquote, Colab
  badge, figure wiring, cross-links. Do **not** copy their prose voice: those were produced by the
  earlier "polish it" version of this skill and read as AI-generated. The doctrine above supersedes
  their tone.
