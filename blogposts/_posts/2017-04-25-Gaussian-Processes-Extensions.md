---
layout: post
title:  "Gaussian Processes - Extensions"
category: blog
date:   2018-04-23
excerpt: "Extensions to Gaussian Processes"
image: "/images/GP_4Obs_Deriv.png"
---
<head>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
       <script type="text/x-mathjax-config">
         MathJax.Hub.Config({
           tex2jax: {
             inlineMath: [ ['$','$'], ["\\(","\\)"] ],
             displayMath: [['$$','$$']],
             processEscapes: true
           }
         });
       </script>
       <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</head>

## Noise

In the real world, the observations onto which the Gaussian process is fitted are often influenced and distorted by noise.
This noise is modeled as a independent, identically distributed normal distribution around zero with an error variance $\sigma^2\_{\varepsilon}$:

$$
\begin{align}
     y &= f(x) + \varepsilon, \quad \quad \text{i.i.d.} \ \varepsilon \sim \mathcal{N}(0, \sigma^2_{\varepsilon})
\end{align}
$$

The covariance matrix between the respective observations with noise is modified on the diagonal entries.
The linear covariance operator can be applied independently to both the objective function evaluation and the noise, yet the noise variance can only be included for the diagonal entries of the covariance matrix.
This is due to the assumption of independent, identical distributed noise, which is uncorrelated between observations.

$$
\begin{align}
     \mathbb{C}[y,y'] &= k(x,x') + \mathbb{1}_{y=y'}  \mathbb{V}[\varepsilon]\\
     &= k(x,x') + \mathbb{1}_{y=y'}  \sigma^2_{\varepsilon}
\end{align}
$$

Gaussian process without noisy observations:

![](Noise_nonoise.png){:height="75%" width="75%"}

Gaussian process with noise only on the mean of the posterior distribution:

![](Noise_meannoise.png){:height="75%" width="75%"}

Gaussian process with noise only on the variance of the posterior distribution:

![](Noise_varnoise.png){:height="75%" width="75%"}

Gaussian process with noisy observations affecting both the mean and the variance:

![](Noise_bothnoise.png){:height="75%" width="75%"}

This can be realized with the addition of the noise's variance to the diagonal entries of the covariance matrix of the observation kernel matrix $K\_{XX}$:

$$
\begin{align}
     \mathbf{K} =
     \begin{bmatrix}
          K_{XX}+\sigma^2_{\varepsilon} \cdot I  & K_{XX_*} \\
          K_{X_*X} & K_{X_*X_*}
     \end{bmatrix} =
     \begin{bmatrix}
          k(X, X) + \sigma_{\varepsilon} \cdot I & k(X, X_*) \\
          k(X_*, X) & k(X_*, X_*)
     \end{bmatrix}
\end{align}
$$

where $I$ is an identity matrix $I\in \mathbb{R}^{N \times N}$.
While the noise itself decreases the precision with which we can fit the GP to the observations, it has convenient numerical properties.
The Gramian block matrix $K_{XX}$ has to be inverted during the computation of the mean and covariance function.
Due to possible rank defincencies, $K_{XX}$ can become singular which prohibits its inversion.
Rank definencies in the covariance matrix can arise when two observations are numerically almost identical.
Incorporating the noise variance into the covariance matrix can be thus regarded as a regularization of the Gaussian process.
This opens the possibility of different regularization themes as both the mean and variance can be independently regularized with respect to the inverse of $K_{XX}$ in $\mu(y\_* )$ and $\Sigma(y\_* )$.


## Model Selection

The optimization of hyperparameters in machine learning is a pivotal process which can influence the performance significantly.
In this regard, Bayesian methods offer a substantial advantage over non-Bayesian methods as the optimal hyperparameters can be automatically recovered from the Bayesian model.
For a supervised learning task, the objective is to maximize the likelihood probability of the targets $p(\mathcal{D})$.

A central aspect of Bayesian methods is the placement of a prior $p(\theta)$ over possible values of $\theta$ which encodes the prior belief what values of $\theta$ are regarded as probable.
Instead of considering a single value for $\theta$ a probability distribution is used that assigns a different weighting to different values of $\theta$.
This is especially important in tasks with small datasets where the likelihood is sensitive to the variability in the data.

The prior can be marginalized to evaluate its influence on the data likelihood.
The objective is therefore to find suitable distributions for $\theta$ which increase the likelihood of the data, ie.

$$
\begin{align}
     p(\mathcal{D}) = \int p(\mathcal{D}, \theta) \ p(\theta) \ d\theta
\end{align}
$$

In the case of Gaussian processes with the squared exponential kernel, the hyperparameters are $\theta = \\{ \alpha, \sigma \\}$ for which we seek values that maximize the probability of the data, i.e.

$$
\begin{align}
     \max_{\theta} \ p(\mathcal{D}; \theta)
     &= \max_{\theta} \ p(y, X ; \theta) \\
     &= \max_{\theta} \ p(y, X  | \theta) \ p(\theta) \\
     &= \max_{\theta}
     \frac{1}{\sqrt{(2 \pi)^{N} |K_{XX}|^2}}
     \exp \left[
     -\frac{1}{2}
          y ^T
          {K_{XX}}^{-1}
          y
     \right]
     \\
     &= \max_{\theta}
     \frac{1}{\sqrt{(2 \pi)^{N} |k(X, X; \theta)|^2}}
     \exp \left[
     -\frac{1}{2}
          y ^T
          k(X, X;\theta)^{-1}
          y
     \right]
\end{align}
$$

where the parameters $\theta$ determine the Gramian matrix $k(XX;\theta)$.
The maximization of the data likelihood is commonly reformulated as a minimzation of the negative log-likelihood.
Working with the log-probability offers a higher numerical stability with respect to floating-point arithmetic of modern computers.

$$
\begin{align}
     \min_{\theta}  -\log{p(\mathcal{D};\theta)}
     &= \min_{\theta} \ \frac{N}{2} \log\left[ 2 \pi \right] + \log\left[ |K_{XX}|\right] + \frac{1}{2} y^TK_{XX}^{-1}y \\
     &= \min_{\theta} \ \frac{N}{2} \log\left[ 2 \pi \right] + \log\left[ |k(XX;\theta)|\right] + \frac{1}{2} y^Tk(X,X;\theta)^{-1}y
\end{align}
$$

The optimization of the log-likelihood can be done with regular optimization algorithms such as limited memory BFGS.

## Derivative Information

Gaussian processes in their traditional definition are described as a Gaussian distribution over possibly infinite observations.
A Gaussian process computes a predictive distribution for $y\_*$ such that predictions are close to observations in their vicinity.
We can expand the Gaussian process by including derivative observations into the set of observations which enforces a similarity in the gradients of the predictions with respect to observations in their vicinity:

$$
\begin{align}
     \begin{bmatrix}
          y \\
          \nabla y \\
          y_*
     \end{bmatrix}
     \sim
     \mathcal{N}\left(\ \cdot \ | \mathbf{0}, \mathbf{K}^{\nabla}\right)
\end{align}
$$

The joint distribution over predictions, derivative observations and observations can be modeled as a Gaussian over all three types of observations:

$$
\begin{align}
p(y_*, \nabla y, y, X_*, X)
&\propto
\exp \left[
-\frac{1}{2}
\begin{bmatrix}
     y \\
     \nabla y \\
     y_*
\end{bmatrix}^T
\left[
\begin{array}{c c | c}
     K_{XX} & K^{\nabla}_{XX} & K_{XX_*} \\
     K^{\nabla T}_{XX} & K^{\nabla\nabla}_{XX} & K^{\nabla}_{XX_*} \\
     \hline
     K_{X_*X} & K^{\nabla}_{X_*X} & K_{X_*X_*}
\end{array}
\right]^{-1}
\begin{bmatrix}
     y \\
     \nabla y \\
     y_*
\end{bmatrix}
\right]
\end{align}
$$

with the expanded covariance matrix which now includes similarity measures between predictions, observations and derivative observations:
$$
\begin{align}
     \mathbf{K}^{\nabla}
     &=
     \left[
     \begin{array}{c | c}
          K^{\nabla, \nabla \nabla}_{X X} & K^{\nabla}_{X X_*} \\
          \hline
          K^{\nabla}_{X_*X} & K_{X_* X_*}
     \end{array}
     \right]\\
     &=
     \left[
          \begin{array}{c c | c}
               K_{XX} & K^{\nabla}_{XX} & K_{XX_*} \\
               K^{\nabla T}_{XX} & K^{\nabla\nabla}_{XX} & K^{\nabla}_{XX_*} \\
               \hline
               K_{X_*X} & K^{\nabla}_{X_*X} & K_{X_*X_*}
          \end{array}
     \right] \\
     &=
     \left[
     \renewcommand*{\array_*tretch}{1.5}
          \begin{array}{c c | c}
               k_{y,y}(X, X) & k_{y, \nabla y}(X, X) & k_{y,y_*}(X, X_*) \\
               k_{\nabla y, y}(X, X) & k_{ \nabla y, \nabla y}(X, X) & k_{\nabla y, y_*}(X, X_*) \\
               \hline
               k_{y_*, y}(X_*, X) & k_{y_*, \nabla y}(X_*, X) & k_{y_*, y_*}(X_*, X_*)
          \end{array}
     \right] \label{eq:derivobs_kernelmatrix-1}
\end{align}
$$

The posterior distribution including derivative observations can be derived from the joint distribution with the matrix inversion lemma in the same manner as seen above.
The mean and covariance of the posterior distribution with derivative observations can be computed with the expanded kernel matrices:

$$
\begin{align*}
     p(y_* | \nabla y, y, X_*, X) &= \mathcal{N} \big( K^{\nabla}_{X_*X} {K^{\nabla, \nabla \nabla}_{XX}}^{-1} y, K_{X_*X_*} - K^{\nabla}_{X_*X} {K^{\nabla, \nabla \nabla}_{XX}}^{-1} K^{\nabla}_{XX_*} \big)
\end{align*}
$$

The Gramian block matrices between predictions, observations and derivative observations can be computed with updated kernels with incorporate the derivative observations.
More precisely, the covariance between two any entries in the observation respectively prediction vector are defined as

$$
\begin{align}
     \mathbb{C}[y, y'] &= k_{y, y'}(x, x') \\
     \mathbb{C}[y, \nabla y'] &= k_{y, \nabla y'}(x, x') \\
     \mathbb{C}[\nabla y, y'] &= k_{\nabla y, y'}(x, x') \\
     \mathbb{C}[\nabla y, \nabla y'] &= k_{\nabla y, \nabla y'}(x, x')
\end{align}
$$

These updated kernels can be derived in a fairly straightforward manner since the covariance with the zero mean assumption is a linear operator.
In order to expand the Gaussian process with derivative observations we have to take the derivative of the kernel and expand the covariance matrix with the respective entries:

$$
\begin{align}
     \mathbb{C}[y, y'] &= \frac{1}{N}\sum_{i=0}^N y_i \cdot y_i' \\
          &= k(x,x') \\
     \mathbb{C}[y, \nabla_{x'}y'] &= \frac{1}{N} \sum_{i=0}^N y_i \cdot \nabla_{x'}y_i' \\
          &= \nabla_{x'} \frac{1}{N} \sum_{i=0}^N y_i \cdot y_i' \\
          &= \nabla_{x'} \mathbb{C}[y,y'] \\
          &= \nabla_{x'} k(x,x')\\
     \mathbb{C}[\nabla_{x}y, \nabla_{x'}y'] &= \frac{1}{N} \sum_{i=0}^N \nabla_{x} y_i \cdot \nabla_{x'}y_i' \\
          &= \nabla_{x} \nabla_{x'} \frac{1}{N}\sum_{i=0}^N y_i \cdot y_i' \\
          &= \nabla_{x}\nabla_{x'} \mathbb{C}[y, y'] \\
          &= \nabla_{x}\nabla_{x'} k(x,x')
\end{align}
$$

While derivative observations themselves are usually hard to come by for computationally expensive functions $f(x)$, derivative observations are of numerical advantage in cases where observations lie very close to each other.
In these cases the inversion can become unstable or even impossible due to the rank definciency.
Derivative observations pose a useful way to circumvent such rank definciencies for very close observations by combining two observations into one observation and a derivative observation.

A Gaussian process:

![](GP_4Obs.png){:height="75%" width="75%"}

The same Gaussian process with derivative observations. The GP is able to fit the true function considerably better:

![](GP_4Obs_Deriv.png){:height="75%" width="75%"}
