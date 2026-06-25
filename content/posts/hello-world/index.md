---
title: "Hello, World — Feature Test"
date: 2026-06-25
draft: false
categories: ["meta"]
tags: ["setup"]
# Enable KaTeX math rendering on this page.
params:
  math: true
summary: "A scaffold post that exercises math, code, diagrams, and charts."
---

This first post exists to verify the blog renders **math**, **code**, **diagrams**,
and **charts**. Delete it once you've confirmed everything looks right.

## Math (KaTeX)

Inline math like $e^{i\pi} + 1 = 0$ renders mid-sentence. Block math:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$

## Code

Syntax highlighting with a copy button (hover the block):

```python
def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
```

## Diagram (Mermaid)

```mermaid
flowchart LR
    A[Write post] --> B{Push to main}
    B --> C[Cloudflare build]
    C --> D[blog.strayforge.com]
```

## Chart (Chart.js)

{{< chart >}}
type: 'line',
data: {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
  datasets: [{
    label: 'Commits',
    data: [12, 19, 7, 22, 30],
    borderColor: '#3b82f6',
    tension: 0.3
  }]
}
{{< /chart >}}

That's the full toolkit — write away.
