# Multiresolution

#### Definition. $L^2(\mathbb{R})$

$$
L^2(\mathbb{R}) = \{

f : \mathbb{R} \mapsto \mathbb{C} \, | \, \int_{\mathbb{R}} |f(x)|^2 \operatorname{d}x < \infty
    \}
$$

#### Definition. Multiresolution Analysis

A multiresolution analysis for $L^2(\mathbb{R})$ consists of 

* a sequence of closed subspaces $(V_j)_{j\in\mathbb{Z}} $ of $L^2(\mathbb{R})$, and
* a scaling function $ \phi \in V_0$ ,

such that the following conditions holds, 

* $\{\phi(\cdot-k)\}_{k\in\mathbb{Z}}$ constitute an orthonomal basis of $V_0$.
* $\forall i\in\mathbb{Z}, V_{j-1} \sub V_{j} $,
* $\overline{\cup V_j} = L^2(\mathbb{R}) $,
* $\cap V_j = \{0\} $,
* $f \in V_{0} \iff f(2^{j}\cdot) \in V_j $,

> The scaling function $\phi$ in multiresolution analysis determines the spaces $V_j$ uniquely.

#### Definition. Wavelet space

For any $j\in\mathbb{Z}$, let $W_j$ denote the orthogonal complement of $V_j$
with respect to $V_{j+1}$, i.e.

$$
W_j = \{f\in V_{j+1}\, | \, \langle f,g \rangle = 0, \forall g\in V_j \}.
$$

For each $j\in\mathbb{Z}$, the closed subspaces, 

$$
V_j = V_{j-1} \oplus W_{j-1} = \cdots \oplus W_{j-2} \oplus W_{j-1}
$$

Any wavelet generate a direct sum decomposition of $L^2(\mathbb{R})$.

Basis for $V_j$, $\{\phi_k^j = 2^{j/2}\phi(2^jx-k)\}_{k\in\mathbb{Z}}$

$\phi(x) = \sum_{k\in\mathbb{Z}} p_k \phi(2x-k)$

$\phi_l^{j-1} = 2^{-1/2} \sum_{k\in\mathbb{Z}} p_{k-2l} \phi_k^j$

Basis for $W_j$, $\{\psi_k^j = 2^{j/2}\psi(2^jx-k)\}_{k\in\mathbb{Z}}$

$\psi(x) = \sum_{k\in\mathbb{Z}} (-1)^k \overline{p_{l-k}} \psi(2x-k)$

$\psi_l^{j-1} = 2^{-1/2} \sum_{k\in\mathbb{Z}} (-1)^k \overline{p_{1-(k-2l)}} \psi_k^j$

#### Decomposition formulas

Denote $ f_k^j = \langle f, \phi_k^{j}\rangle $ and $ g_k^j = \langle f, \psi_k^{j}\rangle  $,

$$
\begin{cases}
f_l^{j-1} = 2^{-1/2} \sum_k \overline{p_{k-2l}} f_k^j,\\
g_l^{j-1} = 2^{-1/2} \sum_k (-1)^k p_{1-(k-2l)} f_k^j,
\end{cases}
$$

#### Reconstruction formulas

$$
f_k^j = \sum_l p_{k-2l} f_l^{j-1} + \sum_l (-1)^k \overline{p_{1-(k-2l)}} g_l^{j-1}
$$

---

#### Proposition.

The function $\phi\in L^2(\mathbb{R})$ generate a multiresolution analysis.
Let $\psi\in L^2(\mathbb{R})$ and $\{\psi(\cdot-k)\}_{k\in\mathbb{Z}}$
is an orthonormal basis for $W_0$, then

* the functions $\{\psi(2^j\cdot-k)\}_{k\in\mathbb{Z}}$ form an orthonormal basis for $W_j$,
* $\psi$ is wavelet, i.e., the functions $\{\psi(2^j\cdot-k)\}_{j,k\in\mathbb{Z}}$ form an orthonormal basis for $W_j$,
* the functions $\{\phi(\cdot-k)\}_{k\in\mathbb{Z}}\cup\{\psi(2^j\cdot-k)\}_{j\in\mathbb{N},k\in\mathbb{Z}} $
  form an orthonormal basis for $L^2(\mathbb{R})$.

#### Proposition.

The function $\phi\in L^2(\mathbb{R})$ generate a multiresolution analysis.
Then there exist a 1-periodic function $H\in L^2(0,1)$ such that

$$
\hat \phi(2t) = H(t) \hat \phi(t), t\in\mathbb{R}.
$$

where

$$
H(t) = \sum_{k\in\mathbb{R}} c_k e^{2\pi i kt}
$$

Let 

$$
G(t) = \overline{H(t+\frac{1}{2})} e^{-2\pi i t}
$$

define the function $\psi$

$$
\hat \psi(2t) = G(t)\hat \phi(t)
$$

then

* the functions $\{\psi(2^j\cdot-k)\}_{k\in\mathbb{Z}}$ form an orthonormal basis for $W_j$,
* $\psi$ is wavelet, i.e., the functions $\{\psi(2^j\cdot-k)\}_{j,k\in\mathbb{Z}}$ form an orthonormal basis for $W_j$,

$$
G(t) = \overline{H(t+\frac{1}{2})} e^{-2\pi i t}
= \sum_{k\in\mathbb{R}} \bar c_k e^{-2\pi i k(t+\frac{1}{2}) -2\pi i t}
$$

#### Proposition

If

$$
G(t) = \sum_{k\in\mathbb{R}} d_k e^{2\pi i kt}
$$

then

$$
\psi(x) = 2\sum_{k\in\mathbb{Z}} d_k\phi(2x+k), x\in\mathbb{R}.
$$

Compactly supported wavelet means that only few coeficients of $d_k$ are non-zeros.

#### Theorem

Let $\phi\in L^2(\mathbb{R})$, 
$\{\phi(\cdot-k)\}_{k\in\mathbb{Z}}$
is an orthonormal system iff.

$$
\sum_{k\in\mathbb{Z}} |\hat\phi(t+k)|^2 = 1, t\in\mathbb{R}.
$$
