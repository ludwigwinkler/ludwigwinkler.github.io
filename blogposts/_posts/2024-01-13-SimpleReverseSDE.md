---
layout: post
title:  "Simple Reverse-Time SDE Derivation for Diffusion Models"
date:   2024-01-13
excerpt: "Save yourself a lot of Bayes with a linear function"
image: "../../blog/blogthumbnails/simple_reverse_sde.png"
---
<head>
<!-- <script type="text/x-mathjax-config">  -->
  <!-- MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script> -->
<!-- uncomment two lines above and remove the html css to svg lines -->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "all" } },
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [['$$','$$'], ['\[' , '\]'], ['\\[', '\\]']],
      processEscapes: true
    },
    "HTML-CSS": { linebreaks: { automatic: true } },
    CommonHTML: { linebreaks: { automatic: true } },
    SVG: { linebreaks: { automatic: true } }
    });
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
</head>
$$
\newcommand{\Efunc}[1]{\mathbb{E}\left[ #1\right]}
\newcommand{\Vfunc}[1]{\mathbb{V}\left[ #1\right]}
\newcommand{\KL}[2]{\text{KL}\left[ #1 \ || \ #2 \right]}
\newcommand{\denom}[1]{\frac{1}{#1}}
\newcommand{\drift}{\mu(X_t, t)}
\newcommand{\diff}{\sigma(X_t, t)}
$$

We start out with the Fokker-Planck equation (FPE) which relates the change over time for the probability for a specific value of $x$ with a diffusion term $\sigma(t)$ which is only dependent on the time,
$$
\begin{align}
	\partial_t \ p(x,t) = & - \partial_x \left[ \drift \ p(x, t) \right] + \denom{2} \partial_x^2 \left[ \sigma(t)^2 \ p(x, t) \right]  \\
	=                     & - \partial_x \left[ \drift \ p(x, t) \right] + \denom{2} \sigma(t)^2 \partial_x^2 \left[ \ p(x, t) \right].
\end{align}
$$

Now we consider a time reversion $\tau(t) = 1 - t$ and are interested in what the change of the probability distribution is under this reversed time index,
$$
\begin{align}
	\partial_{t} \ p(x,\tau(t)) = & - \partial_x \left[ \mu(X(\tau(t)), \tau(t)) \ p(x, \tau(t)) \right]       \\
	                              & + \denom{2} \partial_x^2 \left[ \sigma(\tau(t))^2 \ p(x, \tau(t)) \right].
\end{align}
$$
With the time transformation $\tau(t)$, we apply the chain rule on the time transformation on the left hand side to obtain
$$
\begin{align}
	\frac{\partial p(x,\tau(t))}{\partial t}
	= \frac{ \partial p(x,\tau(t))}{\partial \tau} \ \frac{\partial \tau(t)}{\partial t}
	= \frac{ \partial p(x,\tau(t))}{\partial \tau} \
	\underbrace{ \frac{\partial \tau(t)}{\partial t}}_{-1}
	= -\frac{ \partial p(x,\tau(t))}{\partial \tau}.
\end{align}
$$
Then, we pull the negative factor from the chain rule to the right hand side and combine the drift and diffusion term into a single derivative via the distributive property of the partial derivative,
$$
\begin{align}
	\frac{ \partial p(x,\tau(t))}{\partial \tau} = & \partial_x \left[ \mu(X(\tau(t)), \tau(t)) \ p(x, \tau(t)) \right]         \nonumber                              \\
	                                               & - \denom{2} \sigma(\tau(t))^2 \partial_x^2 \left[ \ p(x, \tau(t)) \right]  \label{eq:app_reversetime_derivation1} \\
	=                                              & - \partial_x \Bigg[ -\mu(X(\tau(t)), \tau(t)) \ p(x, \tau(t))              \nonumber                              \\
	                                               & \qquad \quad + \denom{2} \sigma(\tau(t))^2 \partial_x \left[ \ p(x, \tau(t)) \right] \Bigg]
\end{align}
$$
Applying the log derivative identity $\partial_x \log p(x) = \frac{1}{p(x)} \partial_x p(x)$, which rearranged yields $ \partial_x p(x) = p(x) \partial_x \log p(x)$, we obtain
$$
\begin{align}
	\frac{ \partial p(x,\tau(t))}{\partial \tau} = & - \partial_x \Bigg[ -\mu(X(\tau(t)), \tau(t)) \ p(x, \tau(t))                            \nonumber                                                                             \\
	                                               & \qquad \quad + \denom{2} \sigma(\tau(t))^2 \partial_x \log p(x, \tau(t)) p(x, \tau(t) ) \Bigg]                                                                                 \\
	=                                              & - \partial_x \Bigg[ \Big( \underbrace{-\mu(X(\tau(t)), \tau(t)) + \denom{2} \sigma(\tau(t))^2 \partial_x \log p(x, \tau(t))}_{\text{reverse drift}} \Big) p(x, \tau(t))\Bigg].
\end{align}
$$

The equation above states that the inverted drift with an additional scaled score term of the forward distribution will invert the stochastic process.
Interestingly, there is no diffusion term occuring in this formulation of the reverse FPE.
We can in fact derive a more flexible reverse drift by returning to a slightly rearranged equation \ref{eq:app_reversetime_derivation1} with an additional scaling factor $\alpha^2$,
$$
\begin{align}
	\frac{ \partial p(x,\tau(t))}{\partial \tau} = & - \partial_x \left[ - \mu(X(\tau(t)), \tau(t)) \ p(x, \tau(t)) \right]         \nonumber                                                                                                            \\
	                                               & - \denom{2} \sigma(\tau(t))^2 \partial_x^2 \left[ \ p(x, \tau(t)) \right] \underbrace{\pm \ \alpha^2 \ \sigma(\tau(t))^2 \partial_x^2 \left[ \ p(x, \tau(t)) \right]}_{\text{additional diffusion}} \\
	=                                              & - \partial_x \left[ - \mu(X(\tau(t)), \tau(t)) \ p(x, \tau(t)) \right]         \nonumber                                                                                                            \\
	                                               & - \sigma(\tau(t))^2 \left( \frac{1}{2} + \alpha^2 \right) \partial_x^2 \left[ \ p(x, \tau(t)) \right]                                                                                           \\
	                                               & \underbrace{ + \ \frac{\alpha^2}{2} \ \sigma(\tau(t))^2 \partial_x^2 \left[ \ p(x, \tau(t)) \right]}_{\text{additional diffusion}}                                                                  \\
	=                                              & - \partial_x \Bigg[ \Big( - \mu(X(\tau(t)), \tau(t))                                                                                                                                                \\
	                                               & \qquad \qquad + \sigma(\tau(t))^2 \left( \frac{1}{2} + \alpha^2 \right) \partial_x \log p(x, \tau(t)) \Big) \ p(x, \tau(t)) \Bigg]         \nonumber                                            \\
	                                               & + \ \alpha^2 \ \sigma(\tau(t))^2 \partial_x^2 \left[ \ p(x, \tau(t)) \right].
\end{align}
$$

from which we can infer the reverse drift consisting of the inverted original drift with the additionally scaled score with $\alpha$ and the additional diffusion,
$$
\begin{align}
	dX(\tau) = & ( - \mu(X(\tau(t)), \tau(t)) + \sigma(\tau(t))^2 \left( \frac{1}{2} + \alpha^2 \right) \partial_x \log p(x, \tau(t)) ) dt \\
	           & + \alpha \ \sigma(\tau(t)) dW(\tau)
\end{align}
$$