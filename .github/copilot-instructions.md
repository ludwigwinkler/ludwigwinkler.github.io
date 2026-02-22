# Copilot Instructions for ludwigwinkler.github.io

## Build & Preview

```bash
bundle exec jekyll serve   # Build and serve locally at http://127.0.0.1:4000/
```

No test suite or linter exists.

## Architecture

Jekyll static site (Massively theme) for technical blog posts on machine learning, stochastic processes, and statistical mechanics. All content is static Markdown with embedded MathJax and occasional Python code.

- **Posts** live in `blogposts/_posts/` (not the standard Jekyll `_posts/`).
- **Post images** go in `blog/<PostName>/` (e.g., `blog/FlowMatching/sample_flow.png`). Thumbnails go in `blog/blogthumbnails/`.
- **Legacy images** (pre-2019) are in `images/`.
- Permalinks follow `/blog/:title/`.
- Kramdown with GFM input and Rouge syntax highlighting.

## Blog Post Structure

Every post follows this template:

```markdown
---
layout: post
title:  "Post Title"
category: blog
date:   YYYY-MM-DD
excerpt: "Short tagline"
image: "/blog/blogthumbnails/thumbnail.png"
---
<head>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
       <script type="text/x-mathjax-config">
         MathJax.Hub.Config({
          TeX: {
                equationNumbers: { autoNumber: "all" },
                extensions: ["AMSmath.js", "AMSsymbols.js", "cancel.js"]
            },
           tex2jax: {
             inlineMath: [ ['$','$'], ["\\(","\\)"] ],
             displayMath: [['$$','$$']],
             processEscapes: true
           }
         });
       </script>
       <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</head>

Post body here...
```

Key points:
- The MathJax `<head>` block is embedded **per-post**, not in the layout. Copy it exactly from a recent post (e.g., `2025-07-12-FeynmanKac.md`).
- `category` is always `blog`.
- `image` is the cover thumbnail shown on the blog index page.

## Writing Style

The blog has a distinctive voice — conversational, pedagogical, and mathematically rigorous but approachable. When writing or editing posts, match these patterns:

### Tone & Voice
- **Conversational academic**: Speaks directly to the reader ("Let's assume...", "We can observe...", "The question naturally arises..."). Uses first person plural ("we") to walk the reader through derivations.
- **Witty excerpt lines**: Post excerpts are playful puns or cultural references, not dry summaries. Examples: "Is it Katz, Kak, Kaz, Katsch? Anyway, it's nice math.", "Do you have a minute to talk about our lord and saviour, Thomas Bayes...?", "'If I Could Turn Back Time' by Cher (1989)", "Optional Derivatives", "Warning: May contain traces of nuts (and matrices)".
- **Historical/cultural color**: Posts often open with jokes, anecdotes, or references to scientists' nationalities ("A Dutch, a German and a Russian walk into a bar..."). Occasionally references pop culture, Netflix, or Berlin life.
- **Honest about difficulty**: Admits when things are confusing ("I struggled for quite some time..."), references personal learning journey, and signals when a simpler alternative exists.

### Mathematical Exposition
- **First principles**: Derivations start from the very basics (e.g., what is a differential equation? what is differentiability?) and build up systematically. Never skips intermediate algebra steps.
- **Physical intuition before formalism**: Concepts are motivated with physical analogies before equations appear — coffee mugs cooling, soccer balls being kicked, marbles on staircases, balls rolling downhill.
- **Step-by-step with narration**: Every line of a derivation is accompanied by prose explaining *why* that step was taken, not just *what* was done. Key terms get highlighted in **bold** or $\LaTeX$ inline.
- **"Computational evidence"**: Analytical results are always validated with code and plots. The phrase "computational evidence" (attributed to a friend) is used to describe numerical verification.
- **Custom LaTeX macros**: Recent posts define macros at the top for readability (e.g., `\newcommand{\Efunc}[1]{\mathbb{E}\left[ #1\right]}`).

### Topic Domain
The blog covers a coherent arc of mathematical topics centered on **stochastic processes and probabilistic machine learning**:
- **Core**: SDEs, Wiener processes, Ito calculus, Fokker-Planck equations, Ornstein-Uhlenbeck processes, geometric Brownian motion
- **Generative models**: Diffusion models, flow matching, score-based methods, reverse-time SDEs, discrete diffusion, Feynman-Kac steering
- **Bayesian/probabilistic ML**: Gaussian processes, variational inference, reparameterization trick, VAEs, importance sampling, annealed importance sampling, sequential Monte Carlo
- **Applied math**: Optimal transport (Sinkhorn), FFT, trace estimation, RSA cryptography, Black-Scholes, Poisson processes, continuous-time Markov chains
- **Sampling**: MCMC, Hamiltonian Monte Carlo, Metropolis-Hastings, particle filtering

### Structural Patterns
- Posts often build on earlier posts with explicit cross-references ("I explained the reparameterization trick in detail in another blog post...").
- Sections use `###` headers for major topic shifts.
- Long derivations alternate between equation blocks and explanatory paragraphs — never long stretches of equations without prose.
- Posts end with either a forward-looking remark ("topics for another time..."), a practical summary, or a link to code/slides.

## Math & Equation Conventions

- Inline math: `$...$`
- Display math: `$$...$$`, typically inside `\begin{align}...\end{align}` (numbered) or `\begin{align*}...\end{align*}` (unnumbered)
- Wide equations must be wrapped for horizontal scroll:
  ```html
  <div style="overflow-x: auto;">
  $$
  \begin{align*}
  ...long equation...
  \end{align*}
  $$
  </div>
  ```
- Use `\underbrace{...}_{\text{label}}` and `\overbrace{...}^{\text{label}}` to annotate equation terms inline.
- Use `\cancel{}` for terms that cancel (requires AMSmath cancel extension, loaded in recent MathJax headers).
- Common notation: $\mathbb{E}[\cdot]$ for expectation, $\mathbb{V}[\cdot]$ for variance, $\mathbb{C}[\cdot, \cdot]$ for covariance, $\mathcal{N}(\mu, \sigma)$ for Normal distribution, $dW_t$ for Wiener process differentials.

## Images in Posts

Inline images use raw HTML (not Markdown `![]()` syntax) in recent posts:

```html
<img src="/blog/PostName/image.png" alt="Description" style="width: 100%; height: auto;" />
```

Legacy posts use Markdown image syntax with Kramdown attributes:
```markdown
![](/blog/HMC/HMC_2D01.png){: .align="center" height="50%" width="50%"}
```

## Code Blocks

- Python code uses PyTorch (`torch`) for ML demonstrations and NumPy/Matplotlib for numerical experiments.
- Some posts use JAX (`jax.jit`, `jax.vmap`) for parallel computation examples.
- Code is shown alongside mathematical derivations to provide "computational evidence" — i.e., numerical verification of analytical results.
- Code style: functional or class-based (`torch.nn.Module`), with inline comments explaining the mathematical correspondence.