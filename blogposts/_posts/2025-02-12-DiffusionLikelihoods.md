---
layout: post
title:  "Likelihood Calculations in Diffusion Models"
category: blog
date:   2025-02-12
excerpt: "Mr. Fokker and Mr. Planck, meet Ito-San"
image: "/blog/ItoDensityEstimator.png"
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

Let's start out this blog post with the one equation that is at the heart of diffusion models, the Fokker-Planck equation,
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ p(x, t) 
&= - \sum_{i=1}^D \partial_{x_i} \left[ \mu_i(x, t) p(x, t) \right] + \frac{1}{2} \sum_{i=1}^D \sum_{j=1}^D \partial_{x_i} \partial_{x_j} \left[ \sigma_{ij}^2(x, t) p(x, t) \right] \\ 
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \frac{1}{2} \nabla \cdot \left[ \nabla \cdot \left[ \sigma^2(x, t) p(x, t) \right] \right] \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) \ p(x, t) \right] + \frac{1}{2} \Delta \cdot \left[ \sigma^2(x, t) \ p(x, t) \right] \\
\end{align}
$$
</div>
which describes the evolution of the probability density function of a stochastic process that follows a stochastic differential equation
<div style="overflow-x: auto;">
$$
\begin{align}
dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t.
\end{align}
$$
</div>

These two equations allow us to work with the underlying stochastic process in two different ways.
Either we integrate the stochastic differential equation to obtain the sample path of the process, or we solve the Fokker-Planck equation to obtain the probability density function of the process.

<!-- <span style="color:red;">Add image to visualize a trajectory of $X_t$ vs $p(x,t)$.</span> -->

The FPE describes the evolution of the probability density function of a stochastic process.
The first term on the right-hand side is the drift term, and the second term is the diffusion term.
The drift term is the divergence of the drift vector field $\mathbf{\mu}(x, t)$, and the diffusion term is the Laplacian of the diffusion matrix field $\mathbf{\sigma}(x, t)$. 
The Fokker-Planck equation is a partial differential equation that describes the evolution of the probability density function of a stochastic process.
It is a generalization of the heat equation, which describes the diffusion of heat in a medium.

Following these two 'representations' of a stochastic process, we can now ask the question: How can we calculate the likelihood of the observed data given a stochastic process?
In the following, we will derive two different ways to calculate the likelihood of the observed data given a diffusion model: the probability flow formulation and Ito's density estimator.

# Probability Flows

The first way to compute likelihoods of observed data given a diffusion model is to use the probability flow formulation.
**This approach rewrites the FPE such that we "pull in" in the diffusive component $\sigma^2(t)$ into the drift component, thereby making the stochastic part "disappear".**
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ p(x, t) 
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \frac{1}{2} \Delta \cdot \left[ \sigma^2(t) p(x, t) \right] \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \nabla \cdot \left[\frac{1}{2} \sigma^2(t)  \nabla p(x, t) \right] \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \nabla \cdot \left[ \frac{1}{2} \sigma^2(t) \ p(x, t) \ \nabla \log p(x, t) \right]  \\
&= - \nabla \cdot \left[ \left( \mathbf{\mu}(x, t) - \frac{1}{2} \sigma^2(t) \nabla \log p(x, t) \right) p(x, t) \right] \\
&= - \nabla \cdot \left[ \ \tilde{\mu}(x, t) \ p(x, t) \right] \\
\end{align}
$$
</div>

In fact, both the distribution $p(x,t)$ and the sample $x$ are a function of time.
To make this really explicit, we can write out the probability distribution as $p(x(t), t)$ and taking its total derivative with respect to time gives us
<div style="overflow-x: auto;">
$$
\begin{align}
d_t \ p(x(t), t) &= \partial_t \ p(x(t), t) + \nabla_x p(x(t), t)^\top \ \partial_t x(t) \\
 &= \partial_t \ p(x(t), t) + \nabla_x p(x(t), t)^\top \ \mu(x,t) \\
\end{align}
$$
</div>

Essentially, the time $t$ occurs in two places: in the probability distribution $p(x,t)$ and nested in the sample $x(t)$.
If we want to derive with respect to the time $t$, we have to consider both the explicit dependence of the probability distribution on the time $p( \cdot , t)$ and the dependence of time in the sample $\partial_t p(x(t), t) = \partial_x p(x(t), t) \ \partial_t x(t)$.

Let's circle back to the FPE in which we relate the change in time $\partial_t p(x,t)$ to the change in space $\nabla \cdot \mathbf{\mu}(x, t) \ p(x, t)$.
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ p(x, t) 
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) \ p(x, t) \right] \\
&= - \sum_i \partial_{x_i} \left[ \mathbf{\mu}_i(x, t) \ p(x, t) \right] \\
&= - \sum_i \left[ \partial_{x_i} \ \mathbf{\mu}_i(x, t) \ p(x, t) + \mathbf{\mu}_i(x, t) \ \partial_{x_i} \ p(x, t) \right] \\
&= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t) - \mathbf{\mu}(x, t)^\top \ \nabla_{x} \ p(x, t)
\end{align}
$$
</div>

and substituting the FPE into the total derivative yields
<div style="overflow-x: auto;">
$$
\begin{align}
d_t \ p(x(t), t) &= \partial_t \ p(x(t), t) + \nabla_x \ p(x(t), t) \ \partial_t x(t) \\
&= \partial_t \ p(x(t), t) + \nabla_x \ p(x(t), t) \ \mu(x, t) \\
&= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t) - \cancel{\mathbf{\mu}(x, t)^\top \ \nabla_x \ p(x, t)} + \cancel{\nabla_x p(x(t), t)^\top \ \mu(x,t)} \\
&= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t)
\end{align}
$$
</div>

Pulling $p(x,t)$ over to the left side then yields the total derivative of the log-likelihood,
<div style="overflow-x: auto;">
$$
\begin{align}
d_t \ p(x(t), t) &= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t) \\
\frac{d_t \ p(x(t), t)}{p(x(t),t)} &=- \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] \\
d_t \log p(x(t), t) &= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right].
\end{align}
$$
</div>

The result above is remarkable as it allows us to compute the likelihood of the observed data given a diffusion model by integrating an ordinary differential equation.
This is in fact a continuous normalizing flow.
To apply this way of computing log-likelihoods, we need to identify the deterministic drift, which we saw in equations (5) - (9), and integrate the ODE to obtain the log-likelihood of the observed data given a diffusion model.

Computing the log-likelihood of a sample $x(T)$ is then given by integrating the change in the log-likelihood from time $0$ to time $T$:
<div style="overflow-x: auto;">
$$
\begin{align}
\log p(x(T), T) &= \log p(x(0), 0) - \int_0^T d_t \log p(x(s), s) \ ds \\
&= \log p(x(0), 0) - \int_0^T \nabla_x \cdot \left[ \mathbf{\mu}(x, s) \right] \ ds \\
\end{align}
$$
</div>

A small disclaimer: the tricky part is computing the divergence of the drift vector field $\mathbf{\mu}(x, t)$ efficiently.
The problem here is that for $D$ dimensions, we have to compute $D$ partial derivatives for each dimension, which can be computationally expensive in higher dimensions.
To alleviate this problem, Hutchinson's stochastic trace estimator is commonly used.

What does the trace have to do with the divergence?
Well, essentially the divergence sums over the diagonal elements of the Jacobian matrix $J_\mu \in \mathbb{R}^{D \times D}$ of the drift vector field $\mathbf{\mu}(x, t)$, and $J_{\mu, ij} = \partial_{x_i} \mu_j(x,t)$ by its definition 
<div style="overflow-x: auto;">
$$
\begin{align}
\nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] = \sum_{i}^D \partial_{x_i} \mu_i(x, t) =  \text{Tr}(J_\mu).
\end{align}
$$
</div>

Hutchinson's stochastic trace estimator approximates the trace of a matrix by sampling random vectors and computing the inner product of the matrix with the random vectors, like so
<div style="overflow-x: auto;">
$$
\begin{align}
\nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] &= \text{Tr}[J_\mu] \\
&= \text{Tr}[J_\mu \ \text{I}] \quad \quad \quad \ \ \  | \quad \text{identity matrix I}\\
&= \mathbb{E}\left[ \text{Tr}[J_\mu \ \epsilon \epsilon^T ] \right] \quad | \quad \text{identity matrix I} = \mathbb{E}[\epsilon \epsilon^T], \epsilon \sim \mathcal{N}(0,I), \epsilon \in \mathbb{R}^D  \\
&= \mathbb{E}\left[ \text{Tr}[\epsilon^T J_\mu \ \epsilon ] \right] \quad | \quad \text{circularity of trace}: \text{Tr}[ABC] = \text{Tr}[CAB]\\
&= \mathbb{E}\left[ \epsilon^T J_\mu \ \epsilon  \right] \quad \quad \ \ \ | \quad \epsilon^T J_\mu \epsilon \in \mathbb{R} \text{ is scalar, so drop trace operator}\\
\end{align}
$$
</div>

Furthemore the term $J_\mu \epsilon$ is a Jacobian-vector product and can be computed efficiently using automatic differentiation.
Effectively, we're backpropagating the vector $\epsilon$ through the neural network evaluation ( in pseudo-code `torch.autograd.grad(outputs=mu, inputs=x, grad_outputs=epsilon)`) and contract it with `epsilon` again.
We do that for multiple samples of $\epsilon$ and average the results to obtain an unbiased estimate of the trace of the Jacobian matrix.
Taking a single sample of $\epsilon$ and computing the Jacobian-vector product is computationally more efficient than computing the full Jacobian matrix but comes with a higher estimator variance.

# Ito Density Estimators

The probability flow log-likelihood estimator is derived through the Fokker-Planck equation.
Here we will compute the log-likelihood of the observed data given a diffusion model using Ito's lemma based on the SDE formulation.

We start out by splitting the Laplace operator $\Delta$ into two divergences,
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ p(x, t) 
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \frac{1}{2} \Delta \left[ \sigma^2(t) p(x, t) \right] \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \nabla \cdot \left[\frac{1}{2} \sigma^2(t)  \nabla p(x, t) \right].
\end{align}
$$
</div>

Working with the change in the probability $\partial_t \ p(x,t)$ is nice per se but working with the log-likelihood $\partial_t \ \log p(x,t)$ is even nicer as it is numerically more stable.
The goal is therefore to express the change in probability not in the probability space $p(x,t)$ itself but instead in the log-probability space $\log p(x,t)$.
To achieve that, we'll use the chain rule for the log-likelihood which is also known as the log-derivative trick which also applies to the Laplace operator:
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ \log p(x, t) &= \frac{1}{p(x, t)} \partial_t \ p(x, t) \\
& \downarrow \\
\partial_t \ p(x, t) &= p(x,t) \ \partial_t \ \log p(x, t) \\
\nabla_x \ p(x, t) &= p(x,t) \ \nabla_x \ \log p(x, t) \\
\Delta \ p(x,t)
&= \nabla_x \cdot \left[ \nabla_x p(x, t) \right] \\
&= \nabla \cdot \left[ p(x, t) \ \nabla_x \log p(x, t) \right] \\
% &= \sum_i^D \partial_{x_i} \left[ p(x, t) \ \nabla_x \log p(x, t) \right] \\
% &= \sum_i^D \partial_{x_i} p_i(x, t) \ \nabla_x \log p(x, t) + p(x,t) \\
&= \nabla p(x, t) \cdot \nabla_x \log p(x, t) + p(x, t) \ \nabla_x^2 \log p(x, t) \\
\end{align}
$$
</div>

The first step is to write out the FPE in terms of the derivatives,
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ p(x, t) &= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \frac{1}{2} \Delta \cdot \left[ \sigma^2(t) p(x, t) \right] \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) p(x, t) \right] + \frac{1}{2} \sigma^2(t) \ \Delta \cdot p(x, t) \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t) - \mathbf{\mu}(x, t)^\top \ \nabla p(x, t) + \frac{1}{2} \sigma^2(t) \ \Delta \cdot p(x, t) \\
&= - \nabla \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t) - \mathbf{\mu}(x, t)^\top \ \nabla p(x, t) + \frac{1}{2} \sigma^2(t) \ \nabla_x \cdot \left[ \nabla_x p(x, t) \right] \\
\end{align}
$$
</div>

Step two is then to express these derivatives with derivatives of the log-likelihood,
<div style="overflow-x: auto;">
$$
\begin{align}
p(x,t) \ \partial_t \ \log p(x, t) &= - \nabla \cdot \left[ \mathbf{\mu}(x, t) \right] \ p(x, t) - \mathbf{\mu}(x, t)^\top \ p(x,t) \ \nabla_x \ \log p(x, t) \\
& \quad + \frac{1}{2} \sigma^2(t) \ \left( \nabla_x p(x, t) \cdot \nabla_x \log p(x, t) + p(x, t) \ \Delta \log p(x, t)\right) \\
\partial_t \ \log p(x, t) 
&= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] - \mathbf{\mu}(x, t)^\top \ \nabla_x \ \log p(x, t) \\
& \quad + \frac{1}{2} \sigma^2(t) \ \left( \frac{\nabla_x p(x, t)}{p(x,t)} \cdot \nabla_x \log p(x, t) + \Delta \log p(x, t)\right) \\
&= - \nabla_x \cdot \left[ \mathbf{\mu}(x, t) \right] - \mathbf{\mu}(x, t)^\top \ \nabla_x \ \log p(x, t) \\
& \quad + \frac{1}{2} \sigma^2(t) \ \left( \left\| \nabla_x \log p(x, t) \right\| ^2 + \Delta \log p(x, t)\right)
\end{align}
$$
</div>

Thus we have obtained an ODE for the log-likelihood (instead of the likelihood $p(x,t)$) for a sample of the diffusion model.
This is essentially the log transformed version of the FPE.
This equation is actually closely related to the Hamilton-Jacobi-Bellman equation in the context of optimal control theory and was first showcased by Lorenz Richter et al.

This is still a PDE and describes the evolution of the log-probability.
But we're not interested in the overall evolution of the log-probability, but rather in the likelihood of the observed data given a diffusion model.
Intuitively, we're only interested in the change of probability of a particular sample $x_t$ which has its own additional dynamics expressed through the sampling SDE.

In order to obtain the total derivative $d \log p(x,t)$ which takes into account the change in probability as well as the change in the sample $x_t$, we have to consider the chain rule for the log-likelihood.
For that we will take Ito's lemma.

For a function function $f(X_t)$ Ito's lemma calculates the change in the function $df(X_t)$ as a function of the change in the sample $X_t$.

The real application of log-likelihood calculations in diffusion models is to compute the likelihood of the observed data given a diffusion model where the flow of time is reversed.
To achieve that we 'simply' introduce a new time index $\tau = 1 - t$ and respectively $t = 1 - \tau$.
Now we can express the time in terms of the reversed time $\tau$,
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_t \ \log p(x, 1-t) =  \partial_\tau \ \log p(x, 1-t) \ \partial_t \ \tau = - \ \partial_\tau \ \log p(x, \tau)
\end{align}
$$
</div>

Applying the time reversion to the log transformed FPE, the evolution of the probability density function in terms of the reversed time $\tau$ is given by:
<div style="overflow-x: auto;">
$$
\begin{align}
\partial_\tau \ \log p(x, \tau) = \nabla_x \cdot \mathbf{\mu}(x, \tau) + \mathbf{\mu}(x, \tau) \ \nabla_x \ \log p(x, \tau) - \frac{1}{2} \sigma^2(\tau) \ \left( \left(\nabla_x \log p(x, \tau) \right)^2 + \Delta \cdot \log p(x, \tau)\right)
\end{align}
$$
</div>

Now, we'll hand over the calculations to Ito-san to actually compute the likelihood of the observed data given a diffusion model.
<div style="overflow-x: auto;">
$$
\begin{align}
df(X_t, t) &= \left( \partial_t f(X_t, t) + \nabla_x f(X_t, t)^T  \ \mu(X_t, t) + \frac{1}{2} \Delta \cdot f(X_t, t) \ \sigma(X_t, t)^2 \right) dt + \nabla_x f(X_t, t)^T \sigma(X_t, t) dW_t \\
& \downarrow f = \log p(x,t)\\
d \log p(x_t, t) &= \left( \partial_t \log p(x_t, t) + \nabla_x \log p(x_t, t)^T  \ \mu(X_t, t) + \frac{1}{2} \Delta \cdot \log p(x_t, t) \sigma(t)^2 \right) dt + \nabla_x \log p(x_t, t)^T \sigma(t) dW_t \\
\end{align}
$$
</div>

The essential step in Skreta et al's paper is to combine the log-transformed FPE with Ito's lemma to obtain the likelihood of the observed data given a diffusion model.
This expresses the total change in log-likelihood originating both from the particle dynamics $dX_t$ and the evolution of the probability density function $d \log p(x_t, t)$.

We have
<div style="overflow-x: auto;">
$$
\begin{align}
d \log p(x_t, \tau) &= \left( \color{blue}{\partial_\tau \log p(x_\tau, \tau)} + \nabla_x \log p(x_\tau, \tau)^T  \ \mu(x_\tau, \tau) + \frac{1}{2} \Delta \cdot \log p(x_\tau, \tau) \ \sigma(t)^2 \right) d\tau + \nabla_x \log p(x_\tau, \tau)^T \sigma(\tau) dW_\tau \\
&= \Big( \color{blue}{\nabla_x \cdot \mu(x, \tau) + \mu(x, \tau) \ \nabla_x \ \log p(x, \tau) - \frac{1}{2} \sigma^2(\tau) \ \left( \left(\nabla_x \log p(x, \tau) \right)^2 + \cancel{\Delta \cdot \log p(x, \tau)} \right) } \\
& \quad \quad + \nabla_x \log p(x_\tau, \tau)^T  \ \mu(X_\tau, \tau) + \frac{1}{2} \cancel{\Delta \cdot \log p(x_\tau, \tau) \ \sigma(\tau)^2} \Big) d\tau \\
& \quad + \nabla_x \log p(x_\tau, \tau)^T \sigma(\tau) dW_\tau \\
&= \left( \nabla_x \cdot \mu(x, \tau) + \nabla_x \ \log p(x, \tau) \left( ~~\mathbf{\mu}(x, \tau)~~ - \frac{1}{2} \sigma^2(\tau) \nabla \log p(x, \tau) \right) \right) d\tau \\
& \quad + \nabla_x \log p(x_\tau, \tau) \left(\mu(x, \tau) d\tau + \sigma(\tau) dW_\tau \right) \\
&= \left( \nabla_x \cdot \mu(x, \tau) + \nabla_x \ \log p(x, \tau) \left( \mathbf{\mu}(x, \tau) - \frac{1}{2} \sigma^2(\tau) \nabla \log p(x, \tau) \right) \right) d\tau \\
& \quad + \nabla_x \log p(x_\tau, \tau) \ dx_\tau \\
\end{align}
$$
</div>

The result above stands in contrast to the probability flow formulation for calculating the likelihood of the observed data given a diffusion model.
Whereas the probability flow formulation relies on integrating an ODE, here we can directly compute the log-likelihood of the observed data given a diffusion model by solving a SDE.