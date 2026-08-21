# Explainer authoring

Create one directory for each section and one subdirectory for each explainer:

```text
explainers/
  probability/
    central-limit-theorem/
      index.md
      content.html
```

`index.md` supplies the card and page metadata:

```markdown
---
title: The Central Limit Theorem
description: Why sums of independent random variables become normally distributed.
section: Probability
date: 2026-08-21
embed: content.html
---

Optional short introduction shown above the embedded explainer.
```

Write the standalone explainer in `content.html`. It is embedded in the titled
site page, and its parent directory determines the section organization.
