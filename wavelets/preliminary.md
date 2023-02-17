# Preliminary

## Euler's formula

The Euler's formula refers to
$$
e^{i\theta} = \operatorname{cos}\theta + i\operatorname{sin}\theta .
$$

The proof of this formula can be introduced by investigating the function 
$$ f(\theta) = e^{-i\theta} \left( \operatorname{cos}\theta + i\operatorname{sin}\theta\right) $$
with the facts that

* $f'(\theta) = 0$
* $f(0) = 1$

which conclude that $\forall \theta\in\mathbb{R}, f(\theta) = 1$.

Beware of the following facts,

* $e^{i\theta}$ is $2\pi$ periode function,
* $|e^{i\theta}| = 1$.

they will be the key to step into the Fourier kingdom.


## Inner product

For $x,y\in l^2(\mathbb{Z})$, the inner product defined as
$$
\langle x, y \rangle = \sum_{j=-\infty}^{\infty} x_j \bar y_j.
$$
and for $x,y\in \mathbb{C}^N$, the inner product defined as
$$
\langle x, y \rangle = \sum_{j=1}^N x_j \bar y_j.
$$


For $f,g \in L^2(\mathbb{R})$, the inner product defined as
$$
\langle f, g \rangle = \int f \bar g.
$$
and for $f,g \in L^2(a,b)$, the inner product defined as
$$
\langle f, g \rangle = \int_a^b f \bar g.
$$


## Orthogonal Basis

#### Discrete case, the space $\mathbb{C}^n$

If $ \{v_k\}_0^{n-1} = (v_0, v_1, \dots, v_{n-1})$
is an orthogonal basis of space $\mathbb{C}^n$, which means
$$
 \langle v_k, v_l \rangle = ||v_k||\cdot||v_l||\cdot\delta_{k,l},
$$
where 
$$
||v_k|| = 
\sqrt{
\langle v_k, v_k \rangle
}
=
\left(
\sum_{j=0}^{n-1} v_{k,j}
\right)^{\frac{1}{2}}
$$

<!--
#v_{k,0}^2 + v_{k,1}^2 + \cdots + v_{k,n-1}^2
-->

> Note that $ v_k = (v_{k,0}, v_{k,1},\dots v_{k,n-1}) \in\mathbb{C}^n $ and $ v_{k,j}\in\mathbb{C}$.

then $\forall f \in \mathbb{C}^n$, 
$$
f 
= \sum_{k = 0}^{n-1} \frac{\langle f,v_k\rangle}{\langle v_k, v_k\rangle} v_k
= \sum_{k = 0}^{n-1} \frac{\langle f,v_k\rangle}{||v_k||} \cdot \frac{v_k}{||v_k||}
.
$$

> It is an orthonormal bases if $\forall k, ||v_k|| = 1$.

Suppose $\forall k\in\mathbb{Z}, ||v_k||=||v||$, the above formula can be rewritten in matrix form,
$$
f = 

\frac{1}{||v||^2}
\begin{bmatrix}
v_0 \, v_1 \, \cdots \, v_{n-1}
\end{bmatrix}
\begin{bmatrix}
\langle f, v_0 \rangle \\
\langle f, v_1 \rangle \\
\vdots \\
\langle f, v_{n-1} \rangle \\
\end{bmatrix}
=
\frac{1}{||v||^2}
\begin{bmatrix}
v_0 \, v_1 \, \cdots \, v_{n-1}
\end{bmatrix}
\begin{bmatrix}
\bar v_0 \, \bar v_1 \, \cdots \, \bar v_{n-1}
\end{bmatrix}^T
f
$$
that is
$$
\begin{bmatrix}
 f_0 \\
 f_1 \\
\\
 f_{n-1} \\
\end{bmatrix}
=
\frac{1}{||v||^2}
\begin{bmatrix}
v_{0,0} & v_{1,0}  & \cdots & v_{n-1,0} \\
v_{0,1} & v_{1,1}  & \cdots & v_{n-1,1} \\
\vdots & \vdots & \vdots & \vdots \\
v_{0,n-1} & v_{1,n-1} & \cdots & v_{n-1,n-1} \\
\end{bmatrix}
\begin{bmatrix}
\bar v_{0,0} & \bar v_{0,1}  & \cdots & \bar v_{0,n-1} \\
\bar v_{1,0} & \bar v_{1,1}  & \cdots & \bar v_{1,n-1} \\
\vdots & \vdots & \vdots & \vdots \\
\bar v_{n-1,0} & \bar v_{n-1,1} & \cdots & \bar v_{n-1,n-1} \\
\end{bmatrix}
\begin{bmatrix}
 f_0 \\
 f_1 \\
\\
 f_{n-1} \\
\end{bmatrix}
.
$$

<!--
$$
\begin{bmatrix}
 f_0 \\
 f_1 \\
\\
 f_{n-1} \\
\end{bmatrix}
=
\frac{1}{||v||^2}
\begin{bmatrix}
\begin{bmatrix}
 v_{0,0} \\
 v_{0,1} \\
\\
 v_{0,n-1} \\
\end{bmatrix}
\begin{bmatrix}
 v_{1,0} \\
 v_{1,1} \\
\\
 v_{1,n-1} \\
\end{bmatrix}
\cdots
\begin{bmatrix}
 v_{n-1,0} \\
 v_{n-1,1} \\
\\
 v_{n-1,n-1} \\
\end{bmatrix}
\end{bmatrix}

\cdot

\begin{bmatrix}
\begin{bmatrix}
 \bar v_{0,0} \\
 \bar v_{0,1} \\
\\
 \bar v_{0,n-1} \\
\end{bmatrix}
\begin{bmatrix}
 \bar v_{1,0} \\
 \bar v_{1,1} \\
\\
 \bar v_{1,n-1} \\
\end{bmatrix}
\cdots
\begin{bmatrix}
 \bar v_{n-1,0} \\
 \bar v_{n-1,1} \\
\\
 \bar v_{n-1,n-1} \\
\end{bmatrix}
\end{bmatrix}^T
\begin{bmatrix}
 f_0 \\
 f_1 \\
\\
 f_{n-1} \\
\end{bmatrix}
$$
-->

> The construction of orthogonal bases in space $\mathbb{C}^n$ equals to find square matrix $V$ such that $V\bar V^T=||v||^2$.

The following shows the intuition to extend $ k \in \mathbb{N}$, i.e. orthogonal bases $ \{v_k\}_{0}^{\infty}$,

#### the space $L^1(a,b)$

For $f \in L^1(a,b)$, 
extending to $k\in\mathbb{Z}$, $v_k(x)\in L^1(a,b)$, 
respect to
$$
\langle v_k(x), v_l(x) \rangle = \int_{a}^{b} v_k(x) \bar v_l(x) \operatorname{d}\! x = ||v_k|| \cdot ||v_l|| \cdot \delta_{k,l},
$$
where $\forall k\in \mathbb{Z}$,
$$
||v_k|| = \left( \int_a^b v_k(x)\bar v_k(x)\operatorname{d}\! x \right)^{\frac{1}{2}}.
$$
Then
$$
f(x) 
= \sum_{k = -\infty}^{\infty} \frac{\langle f(x), v_k(x)\rangle }{||v_k||}\cdot \frac{v_k(x)}{||v_k||}
= \sum_{k = -\infty}^{\infty} \frac{1}{||v_k||}\left( \int_{a}^{b} f(x) \bar v_k(x)\operatorname{d}\! x \right) \cdot \frac{v_k(x)}{||v_k||}.
$$

#### the space $L^1(\mathbb{R})$
For $f \in L^1(\mathbb{R})$, 
extending to $t\in\mathbb{R}$, which is $v(x,t)\in L^1(\mathbb{R})$,

$$
f(x) 
= \int_{-\infty}^{+\infty} \frac{1}{||v(t)||} \langle f(x), v(x,t)\rangle \cdot \frac{v(x,t)}{||v(t)||} \operatorname{d}\! t
= \int_{-\infty}^{+\infty} \frac{1}{||v(t)||} \left( \int_{-\infty}^{+\infty} f(x)\bar v(x,t)\operatorname{d}\! x \right) \cdot \frac{v(x,t)}{||v(t)||} \operatorname{d}\! t.
$$
where $\forall t\in\mathbb{R}$,
$$
||v(t)|| = \left(\int_{-\infty}^{+\infty} v(x,t)\bar v(x,t)\operatorname{d}\! x\right)^{\frac{1}{2}}.
$$

