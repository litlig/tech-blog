# tech-blog

Technical blog at **blog.strayforge.com**, built with [Hugo](https://gohugo.io)
+ the [Blowfish](https://blowfish.page) theme. Sibling to the personal blog
(cat.strayforge.com).

## Local development

```bash
hugo server -D        # http://localhost:1313, includes drafts
```

## Writing

```bash
hugo new posts/my-post/index.md
```

Supported in posts (see `content/posts/hello-world/`):
- **Math** — add `params: { math: true }` to frontmatter, then `$inline$` / `$$block$$`.
- **Code** — fenced blocks with syntax highlighting + copy button.
- **Diagrams** — fenced ` ```mermaid ` blocks.
- **Charts** — `{{< chart >}} ... {{< /chart >}}` shortcode (Chart.js).

## Deploy

Hosted on Cloudflare Workers (static assets). Pushing to `main` triggers a build:

- Build command: `hugo --gc --minify`
- Deploy command: `npx wrangler deploy`
- Config: `wrangler.jsonc` (`workers_dev: false`, serves `./public`)

The theme is a git submodule; clone with `--recurse-submodules`.
