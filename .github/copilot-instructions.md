# Copilot Instructions for ludwigwinkler.github.io

## Project Overview
This is a Jekyll-based static site, primarily for technical blogging and documentation. Posts are written in Markdown and reside in `blogposts/_posts/`. The site uses MathJax for rendering mathematical notation and often includes code snippets (Python, Torch, etc.) for machine learning and statistical mechanics topics.

## Key Directories & Files
- `blogposts/_posts/`: Main content, Markdown blog posts with embedded math/code
- `.github/copilot-instructions.md`: This file, for agent guidance
- `_layouts/`, `_includes/`: Jekyll HTML templates and partials
- `assets/`, `images/`: Static resources

## Authoring & Editing Patterns
- Posts use extended Markdown, with MathJax blocks for equations and `<div style="overflow-x: auto;">` for wide math/code
- Python code is often shown for ML examples; use `torch` for softmax/logsumexp demos
- When editing Markdown, preserve MathJax formatting and code block styles
- Use line comments `// ...existing code...` when editing files to avoid repeating unchanged content
- For new files, place them inside `/Users/ludwigwinkler/Work/ludwigwinkler.github.io`

## Conventions
- Mathematical derivations are detailed and stepwise, often with physical intuition
- Posts may include inline explanations, code, and visualizations
- Numerical stability tricks (e.g., subtracting max in logsumexp) are explained and demonstrated
- Use clear section headers (`###`) and keep math blocks scrollable for readability

## Build & Preview
- Standard Jekyll workflow: `bundle exec jekyll serve` to build/preview locally
- MathJax is loaded via CDN in post headers for math rendering

## Integration Points
- No backend/server code; all content is static
- MathJax for math, Torch for ML code snippets
- Posts may reference external images/assets in `/blog/` or `/assets/`

## Example: Editing a Post
When updating a post, insert new content using:
```markdown
// ...existing code...
New content here
// ...existing code...
```
This keeps diffs minimal and readable for agents and humans.

## Example: Adding a MathJax Block
```markdown
<div style="overflow-x: auto;">
$$
\begin{align*}
S[p] = \mathbb{E_{p(x)}} [ - \ln p(x)]
\end{align*}
$$
</div>
```

## Agent Workflow
- Always use concise, context-aware edits
- Avoid repeating unchanged code
- Place new files in the main project directory
- Use project-specific math/code conventions

---

If any section is unclear or missing, please provide feedback to improve these instructions.