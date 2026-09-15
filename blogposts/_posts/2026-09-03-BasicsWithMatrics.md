---
layout: post
title:  "The Basics with Matric<span style='color:#9aa0a6;'>e</span>s"
category: blog
date:   2026-09-03
excerpt: "Correct your steps."
highlighter: rouge
# image: "/blog/FKC/steered_diffusion.gif"
---
{% include mathjax3.html %}




### Enter the Matrix

{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="directions" %}

Before defining a matrix, it helps to give it a physical interpretation.
Imagine leaving a bar in a perfectly rectangular street grid and asking for directions home.
Someone tells you to walk two blocks to the right and one block up, which we can collect into the direction vector

$$
\mathbf{x}=
\begin{bmatrix}2\\1\end{bmatrix}.
$$

A matrix $\mathbf{A}$ acts like the map through which those directions are interpreted.
If $\mathbf{A}=\mathbf{I}$, nothing changes and you arrive exactly where the instructions point.
If the streets lean with the slope, moving upward may also make you drift sideways: a shear turns the same instructions into a slightly different route.
A rotation matrix can be more dramatic and turn the complete direction by $90^\circ$.

In the animation above, drag the direction and swap the map.
The teal arrow is the direction $\mathbf{x}$ that you were given; the gold arrow is where the transformed direction $\mathbf{A}\mathbf{x}$ actually takes you.

A matrix is little else than plain linear transformation but written in the most space and time saving way.
Take the equation $$y = a \cdot x$$ which scales a number $x$ by a factor of $a$.
What happens if we have multiple numbers and want to scale them all at once?
For that we introduce little subscripts to indicate the first $x$ as $x_1$ and the second $x$ as $x_2$. Similarly, we denote the corresponding outputs as $y_1$ and $y_2$ and the scaling turning $x_1$ into $y_1$ as $a_1$ and the scaling turning $x_2$ into $y_2$ as $a_2$.
So for example we would have $y_1 = a_1 \cdot x_1$ and $y_2 = a_2 \cdot x_2$ and so on.
Stacking the $y_1$ and $y_2$ equations, we can it out as

$$
\begin{align*}
y_1 = a_1 \cdot x_1 \\
y_2 = a_2 \cdot x_2
\end{align*}
$$

In the equations above we're clearly using $x_i$ as a single input to obtain a corresponding output $y_i$.
Since Indian mathematicians came up with the concept of $0$ a long time ago, we can also theoretically add all the $x_i$'s to both equations by properly multiplying them by zero:

$$
\begin{align*}
y_1 = a_1 \cdot x_1 + 0 \cdot x_2 \\
y_2 = 0 \cdot x_1 + a_2 \cdot x_2
\end{align*}
$$

But now we for each $y_i$ have an expression that involves all the $x_i$'s, not just a single one.
If we now want to distinguish that a particular $a_i$ is the scaling we use to scale a particular $x_j$, we need a way to indicate both what input $x_i$ is scaled how for each output $y_i$.
So we have to denote each $a$ by it's input and output indices $a_{\text{output} \leftarrow {\text{input}}}$.
So we get

$$
\begin{align*}
y_1 = a_{1 \leftarrow 1} \cdot x_1 + \overbrace{a_{1 \leftarrow 2}}^{=0} \cdot x_2 \\
y_2 = \underbrace{a_{2 \leftarrow 1}}_{=0} \cdot x_1 + a_{2 \leftarrow 2} \cdot x_2
\end{align*}
$$

Writing arrows is cumbersome, especially for larger matrices, which is why mathematicians prefer the compact notation and we simply write $a_{ij}$ instead of $a_{i \leftarrow j}$ how input $x_j$ affects output $y_i$.
Since this is just a sum, we can also write it as such:

$$
y_1 = \sum_{j} a_{1j} \cdot x_j \\
y_2 = \sum_{j} a_{2j} \cdot x_j
$$

Mathematicians are a lazy bunch, sorry, efficient bunch and they quickly become bored at writing out all these little subscripts $_1$ and $_2$ all the time, so they thought hard about how to write the two equations in a more compact form.
The $y$'s, $a$'s and $x$'s only differ in their subscripts so we should be able to collect all the $y$'s, $a$'s and $x$'s into some sort of bundle up object that represents all of them at once, which leads us to the concept of vectors and matrices.

Let's first collect the $y$'s and $x$'s and introduce introduce a collective object $\mathbf{x} = ( x_1 , x_2)$ and $\mathbf{y} = ( y_1 , y_2)$.
This is straight forward as both $\mathbf{x}$ and $\mathbf{y}$ only have one dimension which is denoted by its single subscript.
The $a$'s form a two-dimensional array since they have two subscripts which we call a matrix, and we denote it by $\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$.

This brings us to the workhorse of linear algebra, the matrix-vector multiplication:

$$
\begin{align*}
\mathbf{y} &= \mathbf{A} \mathbf{x} \\
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} &= \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \\
 &= \begin{bmatrix} a_{11} \cdot x_1 & + & a_{12} \cdot x_2 \\  a_{21} \cdot x_1 & + & a_{22} \cdot x_2 \end{bmatrix} \\
\end{align*}
$$

As a simple starting point, consider the diagonal matrix

$$
\mathbf{A} = \begin{bmatrix} a_{11} & 0 \\ 0 & a_{22} \end{bmatrix}.
$$

But wait a second, if we squint really closely, we can rediscover the stacking we did with the vector $\mathbf{x}$ and $\mathbf{y}$ in the matrix $\mathbf{A}$ by interpreting the columns of the matrix as vectors as well:

$$
\mathbf{A}
= \begin{bmatrix} a_{11} & 0 \\ 0 & a_{22} \end{bmatrix}
= \begin{bmatrix}
\begin{bmatrix} a_{11} \\ 0 \end{bmatrix} &
\begin{bmatrix} 0 \\ a_{22} \end{bmatrix}
\end{bmatrix}.
$$

so our matrix $\mathbf{A}$ can be seen as a collection of vectors as well!

With this representation of a matrix, we can now reinterpret matrix-vector multiplication as a combination of the matrix's column vectors:

$$
\begin{align*}
\mathbf{y} &= \mathbf{A} \mathbf{x} \\
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} &= \begin{bmatrix}
\begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} &
\begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \\
&= \begin{bmatrix} \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} x_2
\end{bmatrix} \\
&= \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} x_2
\end{align*}
$$

This now opens up a new interpretation as we've initially considered $\mathbf{y} = \mathbf{Ax}$ as the scaling of the input $\mathbf{x}$ by their respective $a$ coefficients. Now, we can also see it as forming the output $\mathbf{y}$ by combining the column vectors of $\mathbf{A}$, each scaled by the corresponding component of $\mathbf{x}$.
Initially we thought about scaling $x$ with $a$, but now we're flipping the script and scaling the column vectors in the matrix $\mathbf{A}$ by the components of $\mathbf{x}$.

The animation below highlights both of these approaches: We either interpret it as a scaling of the input $x$ by the values in $A$ through its column space. Alternatively, we interpret the linear transformation as a combination of the column space of $A$ through $x$.

{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="scaling" %}

The column space view shifts the analysis from the input $x$ onto the matrix $A$ itself, opening up some interesting options which we will see below.

### Rank of a Matrix

When inspecting the column space of a matrix

$$
\begin{align*}
\mathbf{y} &= \mathbf{A} \mathbf{x} \\
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} &= \begin{bmatrix}
\begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} &
\begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \\
&= \begin{bmatrix} \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} x_2
\end{bmatrix} \\
&= \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} x_2
\end{align*}
$$

we can see that the resulting vector $\mathbf{y}$ is a linear combination of the columns of $\mathbf{A}$.

Consequently, we can ask what kind of vectors $\mathbf{y}$ we can generate from $\mathbf{A}$ for a given $x$. The answer to that is closely related to the rank of a matrix Let's consider the matrix

$$
A = \begin{bmatrix} \begin{bmatrix} 2.5 \\ 0 \end{bmatrix} & \begin{bmatrix} 0 \\ 0.5 \end{bmatrix} \end{bmatrix}
$$

which coincidentally can be interpreted as the basis vectors of a two dimensional coordinate system. A vector $\mathbf{x} = (x_1, x_2)$ scales the first column by $x_1$ and the second column by $x_2$

$$
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 2.5 \\ 0 \end{bmatrix} x_1 +  \begin{bmatrix} 0 \\ 0.5 \end{bmatrix} x_2
$$

The column space of $\mathbf{A}$ consists of two different vectors as they both point into different directions. Essentially both vectors stay out of each other's lanes and don't interfere across dimension with each other. But what happens if we flip the entries in the second column of $\mathbf{A}$?

$$
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} 2.5 \\ 0 \end{bmatrix} x_1 +  \begin{bmatrix} 0.5 \\ 0 \end{bmatrix} x_2
$$

In this case regardless of what we use as input $\mathbf{x}$, the second dimension of our output vector $y_2$ will always be zero. This collapses the second dimension and in effect we can only generate a variable output in the first dimension. The important test is whether we have independent columns in the matrix $\mathbf{A}$ or whether the column space consists of collinear vectors.

To make it more interesting (as in not squashing things to zero) we can purposefully construct a column space where $\mathbf{a}_2$ is a multiple of $\mathbf{a}_1$, namely $\mathbf{a}_2=1/5 \cdot  \mathbf{a}_1$.

$$
\begin{align*}
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}
=
\underbrace{\begin{bmatrix} 2.5 \\ 1 \end{bmatrix}}_{\mathbf a_1}x_1
+
\underbrace{\begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}}_{\mathbf a_2} x_2.
\end{align*}
$$

Both columns have two nonzero components, but the second remains a multiple of the first.
From the column space view of a matrix, we're essentially creating an output vector $\mathbf{y}$ as a linear combination of vectors that all point into the same direction.
They therefore provide only one independent direction, and every output remains on the line to which the columns point, just with possibly different lengths.
We say that the matrix $\mathbf{A}$ has rank 1 even if it has more than 1 dimension, as it provides only one independent direction in its column space.



In the widget below, you can see that we can move the output vector $\mathbf{y}$ by adjusting the components of the input vector $\mathbf{x}$.
In the $\text{rank}{A}=1$ case on the right, the output vector $\mathbf{y}$ is constrained to move along a single line, reflecting the fact that the column space of $\mathbf{A}$ has only one independent direction.

{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="rank" %}

### Matrix Inverse

The rank discussion gives us a natural way to understand why some matrices have an inverse and others do not.
The rank-one transformation can be viewed as a many-to-one map. Drag either input point below along the labeled same-output line: every point on this line maps to the same fixed $\mathbf{y}$. Moving in the null direction changes the input without changing the output, so no unique inverse can recover the original input.



Recall our rank-one matrix

$$
\mathbf{A}
=
\begin{bmatrix}
2.5 & 0.5 \\
1 & 0.2
\end{bmatrix},
$$

whose second column is a multiple of its first. Its matrix-vector multiplication can be written as

$$
\begin{align*}
\mathbf{A}\mathbf{x}
&=
\begin{bmatrix} 2.5 \\ 1 \end{bmatrix}x_1
+
\begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}x_2 \\
&=
\begin{bmatrix} 2.5 \\ 1 \end{bmatrix}x_1
+
\frac{1}{5}\begin{bmatrix} 2.5 \\ 1 \end{bmatrix}x_2 \\
&=
\begin{bmatrix} 2.5 \\ 1 \end{bmatrix}
\underbrace{\left(x_1+\frac{1}{5}x_2\right)}_{\in \mathbb{R}}.
\end{align*}
$$

Although we multiplied a two-dimensional input, $x_1$ and $x_2$, into the matrix, the output is determined by the direction of the column space of the matrix scaled by a linear combination of the input components.
Whatever two values $x_1$ and $x_2$ we choose, the direction of the resulting output vector $\mathbf{y}$ remains the same and the input only appears for determining the length of the vector.
The entire two-dimensional input plane is squashed onto the one-dimensional line spanned by the vector $\begin{bmatrix}2.5 & 1\end{bmatrix}^{\mathsf T}$.

This creates a kind of mathematical trapdoor: many different inputs fall onto the same output.
For example, consider the two inputs

$$
\mathbf{x}^{(1)}
=
\begin{bmatrix} 1 \\ 0 \end{bmatrix},
\qquad
\mathbf{x}^{(2)}
=
\begin{bmatrix} 0 \\ 5 \end{bmatrix}.
$$

They are clearly different, but both produce exactly the same output:

$$
\mathbf{A}\mathbf{x}^{(1)}
=
\mathbf{A}\mathbf{x}^{(2)}
=
\begin{bmatrix} 2.5 \\ 1 \end{bmatrix}.
$$

Now imagine that we are only given the output $\mathbf{y}=(2.5, 1)$ and are asked to recover the input.
Was the input $\mathbf{x}^{(1)}$, $\mathbf{x}^{(2)}$, or one of infinitely many other possibilities?
The problem is not merely that recovering the input is difficult. The matrix has discarded the information required to choose a **unique** answer.

We can see the lost information by subtracting the two inputs to obtain the null space of the matrix $\mathbf{A}$. The null space are all the vectors which get mapped to the zero vector by the matrix.

$$
\mathbf{A}
\left(
\mathbf{x}^{(2)}-\mathbf{x}^{(1)}
\right)
=
\begin{bmatrix}
2.5 & 0.5 \\
1 & 0.2
\end{bmatrix}
\begin{bmatrix} -1 \\ 5 \end{bmatrix}
=
\begin{bmatrix} 0 \\ 0 \end{bmatrix}.
$$

The direction $(-1, 5)$ belongs to the null space of $\mathbf{A}$.
Moving the input in this direction does not move the output at all:

$$
\mathbf{A}
\left(
\mathbf{x}
+t\begin{bmatrix}-1 \\ 5\end{bmatrix}
\right)
=
\mathbf{A} x + t \underbrace{\mathbf{A} \begin{bmatrix}-1 \\ 5\end{bmatrix}}_{=\mathbf{0}}
=
\mathbf{A}\mathbf{x}
$$

for any value of $t$.
All inputs along this line are indistinguishable after applying $\mathbf{A}$.

An inverse matrix is supposed to undo a matrix transformation.
If

$$
\mathbf{y}=\mathbf{A}\mathbf{x},
$$

then we would like to apply another matrix $\mathbf{A}^{-1}$ that recovers the original input. This is nothing else than the scalar case of $y =ax$ and getting $x = a^{-1} y$:
$$
\mathbf{x}=\mathbf{A}^{-1}\mathbf{y}.
$$

Doing something and then undoing it should be equivalent to doing nothing, so an inverse must satisfy

$$
\mathbf{A}^{-1}\mathbf{A}
=
\mathbf{A}\mathbf{A}^{-1}
=
\mathbf{I}.
$$

But our rank-one matrix cannot have such an inverse. If it did, applying $\mathbf{A}^{-1}$ to the collision above would give

$$
\mathbf{x}^{(1)}
=
\mathbf{A}^{-1} y =
\mathbf{A}^{-1}\mathbf{A} \underbrace{\mathbf{x}^{(1)}}_{=y}
=
\mathbf{A}^{-1} \underbrace{\mathbf{A}\mathbf{x}^{(2)}}_{=y}
=
\mathbf{x}^{(2)},
$$

even though the two inputs are different.

{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="rank-projection" %}

This gives us the essential condition for invertibility:

> **A square matrix is invertible exactly when it has full rank.**


### Determinants of Matrices

The previous sections asked whether a matrix preserves all directions or squashes some of them together.
The determinant packages the same geometric story into a single number: it tells us how a matrix scales area.

We begin with the simplest case, a diagonal matrix:

$$
\begin{align*}
\mathbf{A}
&=
\begin{bmatrix}
a_{11} & 0 \\
0 & a_{22}
\end{bmatrix}
=
\underset{
\begin{array}{cc}
\underbrace{\hphantom{\begin{bmatrix} a_{11} \\ 0 \end{bmatrix}}}_{\mathbf{a}_1}
&
\underbrace{\hphantom{\begin{bmatrix} 0 \\ a_{22} \end{bmatrix}}}_{\mathbf{a}_2}
\end{array}
}{
\begin{bmatrix}
\begin{bmatrix} a_{11} \\ 0 \end{bmatrix}
&
\begin{bmatrix} 0 \\ a_{22} \end{bmatrix}
\end{bmatrix}
}.
\end{align*}
$$

The first column stays on the horizontal axis and the second stays on the vertical axis.
Consequently, the unit square becomes a rectangle with side lengths $\lvert a_{11}\rvert$ and $\lvert a_{22}\rvert$, and therefore with area $\lvert a_{11}\rvert\lvert a_{22}\rvert=\lvert a_{11}a_{22}\rvert$.
The corresponding signed area is

$$
\det(\mathbf{A})=\sqrt{a_{11}^2 + 0^2} \sqrt{0^2 + a_{22}^2} = \sqrt{a_{11}^2} \sqrt{a_{22}^2} = |a_{11}| |a_{22}|.
$$

Drag the two column vectors below.
They are constrained to their respective axes, making it visible that each diagonal entry scales one side independently and that their product scales the area.

{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="determinant-diagonal" %}

Now we let the columns point in arbitrary directions:

$$
\begin{align*}
\mathbf{A}
&=
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
=
\begin{bmatrix}
\begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix}
&
\begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix}
\end{bmatrix}.
\end{align*}
$$

The transformed unit square is still spanned by the two columns, but its right angles can now lean into a general parallelogram.
Its signed area is

$$
\det(\mathbf{A})=a_{11}a_{22}-a_{12}a_{21}.
$$

This is a deceptively simple equation which you will read in most text books. In order to understand how this simple subtractions comes to be, play around with the widget below which shows how we subtract the triangles from the parallelogram to obtain the term above. The determinant can be calculated by computing the outer bounding box and subtracting all triangles between the outer most bounding box and the actual parallelogram.


{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="determinant" %}

The magnitude $\lvert\det(\mathbf{A})\rvert$ is the area-scaling factor.
A determinant of $2$ doubles every area, while a determinant of $1/2$ halves it.
The sign keeps track of orientation: a positive determinant preserves the ordering of the two directions, while a negative determinant flips it, like turning a sheet of paper over.

This zero-area case is exactly the rank collapse from before.
When the columns are collinear (try it out by making both vectors of the parallelogram identical in their direction), the matrix maps the plane onto a line, so $\det(\mathbf{A})=0$ and the lost direction makes the matrix non-invertible.
When $\det(\mathbf{A})\neq 0$, the columns span the plane and the matrix is invertible.
In higher dimensions the same idea remains: the determinant is the signed scaling factor for $n$-dimensional volume.

The determinant is associative in the sense if we want to know how a matrix $\mathbf{A}$ and a matrix $\mathbf{B}$ change a vector, we can chain the operations and compute each determinant separately, and then multiply them.

$$
\det[AB] = \det[A] \det[B]
$$

### Trace of a Matrix

We already interpreted $a_{ij}$ as how input component $x_j$ contributes to output component $y_i$.
The diagonal entry $a_{jj}$ is special: it measures how much the $j$-th input direction contributes back to that same output direction, rather than being mixed into another one.
Adding these direct contributions gives the **trace**:

$$
\operatorname{Tr}(\mathbf{A})
=
\sum_j a_{jj}
=
\sum_j \mathbf{e}_j^\mathsf{T}\mathbf{A}\mathbf{e}_j.
$$

The trace is not an area or volume.
Instead, it measures the first-order expansion or contraction caused by the tiny transformation $\mathbf{I}+\delta\mathbf{A}$:

$$
\det(\mathbf{I}+\delta\mathbf{A})
=
1+\delta\operatorname{Tr}(\mathbf{A})+\mathnormal{O}(\delta^2).
$$

Let's derive this from scratch in two dimensions. We start with the identity matrix and perturb every entry by a small amount $\delta$:

$$
\mathbf{I}+\delta\mathbf{A}
=
\begin{bmatrix}
1+\delta a_{11} & \delta a_{12} \\
\delta a_{21} & 1+\delta a_{22}
\end{bmatrix}.
$$

Its determinant expands line by line as

$$
\begin{align*}
\det(\mathbf{I}+\delta\mathbf{A})
&=
(1+\delta a_{11})(1+\delta a_{22})
-\delta^2 a_{12}a_{21} \\
&=
1+\delta(a_{11}+a_{22})
+\delta^2(a_{11}a_{22}-a_{12}a_{21}).
\end{align*}
$$

The constant term is $1$, the coefficient of $\delta$ is $a_{11}+a_{22}=\operatorname{Tr}(\mathbf{A})$, and everything involving two perturbations belongs to $\mathnormal{O}(\delta^2)$.
In other words, if we define $f(\delta)=\det(\mathbf{I}+\delta\mathbf{A})$, its Taylor expansion around $\delta=0$ is

$$
f(\delta)=f(0)+\delta f'(0)+\mathnormal{O}(\delta^2),
\qquad
f(0)=1,
\qquad
f'(0)=\operatorname{Tr}(\mathbf{A}).
$$

The same reasoning works in $n$ dimensions.
Each column of $\mathbf{I}+\delta\mathbf{A}$ is an identity column plus $\delta$ times a column of $\mathbf{A}$.
By the determinant's multilinearity, choosing no perturbed column gives $\det(\mathbf{I})=1$, while choosing only the $j$-th perturbed column gives $\delta a_{jj}$.
Choosing two or more perturbed columns introduces at least $\delta^2$.
Adding the possible single-column perturbations therefore gives $\delta\sum_j a_{jj}=\delta\operatorname{Tr}(\mathbf{A})$ at first order.

This first-principles expansion fits into a more general identity. For arbitrary square matrices $\mathbf{A}$ and $\mathbf{B}$ of the same size,

<div style="overflow-x: auto;">
$$
\det(\mathbf{B}+\delta\mathbf{A})
=
\det(\mathbf{B})
+\delta\,\operatorname{Tr}\!\left[\operatorname{adj}(\mathbf{B})\mathbf{A}\right]
+\mathnormal{O}(\delta^2).
$$
</div>

If $\mathbf{B}$ is invertible, then $\operatorname{adj}(\mathbf{B})=\det(\mathbf{B})\mathbf{B}^{-1}$, so this becomes

<div style="overflow-x: auto;">
$$
\det(\mathbf{B}+\delta\mathbf{A})
=
\det(\mathbf{B})\left(
1+\delta\,\operatorname{Tr}\!\left[\mathbf{B}^{-1}\mathbf{A}\right]
+\mathnormal{O}(\delta^2)
\right).
$$
</div>

Taking $\mathbf{B}=\mathbf{I}$ recovers the original identity.
If $\mathbf{B}$ is singular, the formula using $\operatorname{adj}(\mathbf{B})$ still applies.
In particular, when $\operatorname{rank}(\mathbf{B})\leq n-2$, $\operatorname{adj}(\mathbf{B})=0$, so the linear term vanishes.

### The "Eigenheiten" of Matrices

What happens to individual directions under the transformation $\mathbf{A}$?
Is there a nonzero vector whose direction survives without being turned, changing only in length or orientation?








{% include_relative html/2026-09-03-BasicsWithMatrics.html widget="eigenvector" %}


Such a vector satisfies

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}.
$$

We call $\mathbf{v}$ an **eigenvector** and $\lambda$ its **eigenvalue**.
Rearranging gives

$$
(\mathbf{A}-\lambda\mathbf{I})\mathbf{v}=0.
$$

Since $\mathbf{v}\neq 0$, the matrix $\mathbf{A}-\lambda\mathbf{I}$ must squash a nonzero direction to zero.
From our discussion of rank, inverses, and determinants, this means it is non-invertible and therefore

$$
\det(\mathbf{A}-\lambda\mathbf{I})=0.
$$

Geometrically, an eigenvector marks a line preserved by the transformation, while its eigenvalue tells us whether that line is stretched, compressed, flipped, or collapsed.
These directions are the natural axes of the transformation.

Drag $\mathbf{x}$ below and hunt for a direction that the matrix does not turn.


We can make the relationship between eigenvectors and the column space precise using our rank-one matrix from before:

$$
\mathbf{A}
=
\begin{bmatrix}
2.5 & 0.5 \\
1 & 0.2
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{a}_1 & \mathbf{a}_2
\end{bmatrix},
\qquad
\mathbf{a}_2=0.2\mathbf{a}_1,
\qquad
\mathbf{a}_1=
\begin{bmatrix}2.5\\1\end{bmatrix}.
$$

Because the second column is a multiple of the first, every output is

$$
\mathbf{A}\mathbf{x}
=
x_1\mathbf{a}_1+x_2\mathbf{a}_2
=
(x_1+0.2x_2)\mathbf{a}_1.
$$

Therefore, the column space contains only one direction:

$$
\operatorname{Col}(\mathbf{A})
=
\operatorname{span}\{\mathbf{a}_1\}.
$$

Now let us apply the matrix to that direction:

$$
\begin{align*}
\mathbf{A}\mathbf{a}_1
&=
2.5\mathbf{a}_1+1\mathbf{a}_2 \\
&=
2.5\mathbf{a}_1+0.2\mathbf{a}_1 \\
&=
2.7\mathbf{a}_1.
\end{align*}
$$

Thus, the direction spanning the column space is an eigenvector with eigenvalue $2.7$.
Since the column space is one-dimensional, every nonzero vector on that line is scaled by the same factor:

$$
\operatorname{Col}(\mathbf{A})=E_{2.7},
$$

where $E_{2.7}$ denotes the eigenspace associated with $\lambda=2.7$.

The forgotten direction from the inverse section gives the other eigenspace:

$$
\mathbf{A}
\begin{bmatrix}-1\\5\end{bmatrix}
=
-\mathbf{a}_1+5\mathbf{a}_2
=
\mathbf{0}.
$$

It is therefore an eigenvector with eigenvalue $0$, and

$$
\ker(\mathbf{A})=E_0.
$$

For this rank-one matrix, the eigenspaces separate the plane into the direction that survives and the direction that is forgotten.
More generally, if $\lambda\neq0$, then

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}
\quad\Longrightarrow\quad
\mathbf{v}
=
\mathbf{A}\left(\frac{\mathbf{v}}{\lambda}\right)
\quad\Longrightarrow\quad
\mathbf{v}\in\operatorname{Col}(\mathbf{A}).
$$

Every eigenvector with a nonzero eigenvalue must therefore lie in the column space.
The converse is not true for every matrix, but it is true here because the column space contains only one direction.

### Decompositions of a Matrix

A complicated linear map becomes easier to understand when we split it into a sequence of simpler maps.
If the square matrix $\mathbf{A}$ has enough linearly independent eigenvectors to form the columns of $\mathbf{V}$, then it can be decomposed as

$$
\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}.
$$

Read from right to left, $\mathbf{V}^{-1}$ changes into eigenvector coordinates, $\boldsymbol{\Lambda}$ scales each eigendirection independently, and $\mathbf{V}$ changes back.
This is the **eigendecomposition**, and it exists when $\mathbf{A}$ is diagonalizable.

The **singular value decomposition** extends the same intuition to every real matrix:

$$
\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\mathsf T}.
$$

Here, orthogonal changes of coordinates surround independent scalings.
The number of nonzero singular values in $\boldsymbol{\Sigma}$ is exactly the rank of the matrix.

### Matrix Exponentials

A matrix can describe not only one transformation, but also a transformation applied continuously.
Consider the linear differential equation

$$
\frac{d\mathbf{x}}{dt}=\mathbf{A}\mathbf{x}.
$$

Over a tiny interval $\Delta t$, the state changes approximately as

$$
\mathbf{x}(t+\Delta t)
\approx
\left(\mathbf{I}+\Delta t\,\mathbf{A}\right)\mathbf{x}(t).
$$

Dividing a duration $t$ into $n$ tiny updates gives the matrix analogue of repeated scalar compounding:

<div style="overflow-x: auto;">
$$
\mathbf{x}(t)
=
\lim_{n\to\infty}
\left(\mathbf{I}+\frac{t}{n}\mathbf{A}\right)^n
\mathbf{x}(0)
=
e^{t\mathbf{A}}\mathbf{x}(0).
$$
</div>

Expanding this limit defines the **matrix exponential**:

$$
e^{t\mathbf{A}}
=
\mathbf{I}
+t\mathbf{A}
+\frac{t^2\mathbf{A}^2}{2!}
+\frac{t^3\mathbf{A}^3}{3!}
+\cdots.
$$

Thus, $\mathbf{A}$ specifies the infinitesimal change, while $e^{t\mathbf{A}}$ accumulates it into the finite transformation after time $t$.
If $\mathbf{A}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ is diagonalizable, then

$$
e^{t\mathbf{A}}
=
\mathbf{V}e^{t\boldsymbol{\Lambda}}\mathbf{V}^{-1},
$$

so each eigendirection evolves independently by $e^{t\lambda_i}$.
The power-series definition remains valid even when $\mathbf{A}$ is not diagonalizable.

This connection also gives

$$
\det\!\left(e^{\mathbf{A}}\right)
=
e^{\operatorname{Tr}(\mathbf{A})}.
$$

### Einstein Notation: You don't have to be Einstein to take derivatives of matrices

For one component of a matrix-vector multiplication, we write

$$
y_1=A_{11}x_1+A_{12}x_2+\cdots.
$$

Every output component follows the same pattern: fix the output index $i$ and sum over all input indices $j$,

$$
y_i=\sum_j A_{ij}x_j.
$$

Here, $i$ is a **free index**: it selects which component of $\mathbf{y}$ we are computing.
The index $j$ is a **repeated**, or **dummy**, index: it runs over all input components and is summed away.
Since a repeated index means “sum over this index,” we can drop the summation sign and write

$$
y_i=A_{ij}x_j.
$$

This is the Einstein summation convention: less notation, but exactly the same matrix-vector multiplication.