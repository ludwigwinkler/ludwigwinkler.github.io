# Claude Instructions for ludwigwinkler.github.io

## Project Overview

This is a Jekyll-based static site for technical blogging, explainers, and interactive notes.

## Explainer Formatting

- `explainers/_posts/` supports two valid authoring patterns.
- Fragment explainers: start with Jekyll front matter, then write HTML or Markdown content only. Do not include `<!DOCTYPE html>`, `<html>`, `<head>`, or `<body>`. These render through the shared `explainerpost` layout.
- Standalone interactive explainers: start with Jekyll front matter, then provide a complete HTML document if the page needs its own CSS, scripts, and document structure. The shared explainer layout detects full HTML documents and renders them raw.
- Do not mix fragment content and a nested full HTML document in the same explainer.

## Editing Guidance

- Preserve MathJax formatting and existing code block styles.
- Keep edits minimal and consistent with the current visual design.
- For wide equations, prefer scrollable wrappers when using Markdown-based content.

## Preview

- Use `bundle exec jekyll serve` to preview changes locally.
