#### Conjugate Descent

#### Definition

Let $Q\in\mathbb{R}^{n\times n}$ be a symmetric matrix.
$v_i,v_j\in\mathbb{R}^n,v_i\ne v_j \ne0$ are Q-orthogonal w.r.t. $Q$ if 

$$
v_i^T Q v_j = 0.
$$

#### Proposition

Let $Q$ be positive definite. 
If ${v_1,v_2,\dots,v_k}$ are Q-orthogonal, then they are linear independent.

#### Proof

If $v_k=\alpha_1v_1+\cdots+\alpha_{k-1}v_{k-1}$, then
$$v_k^TQv_k=v_k^TQ(\alpha_1v_1+\cdots+\alpha_{k-1}v_{k-1})=\alpha_1v_k^TQv_1+\cdots+\alpha_{k-1}v_k^TQv_{k-1}=0$$
controversy with postive definite.

#### Proposition

Let $v_1,v_2,\dots,v_n\in\mathbb{R}^n$ vectors Q-orthogonal, 
$x_0\in\mathbb{R}^n$ be arbitrary starting point.

Assume 

$$
\alpha_k = -\frac{g^T_k v_k}{v^T_kQv_k} 
$$

$$
x_{k+1} = x_k + \alpha_k v_k 
$$

$$
g_k = Qx_k - b
$$

then $x_n$ is the solution of $Qx=b$.

$$
\underset{x\in\mathbb{R}^n}{min} \frac{1}{2} x^TQx - b^T x
$$

Matrix $Q\in\mathbb{R}^{n\times n}$ is positive definite.

$$
    Qx = b, \quad x\in\mathbb{R}^n
$$

Let $v_1,v_2,\dots,v_n\in\mathbb{R}^n$ vectors Q-orthogonal, which is linear independent,
then $\forall x^* \in\mathbb{R}^n$ there exist ${\alpha_1,\alpha_2,\dots,\alpha_n}\in\mathbb{R}$,

$$
    x^* = \sum_{i=1}^n \alpha_iv_i = \alpha_1v_1+\alpha_2v_2+\cdots+\alpha_nv_n
$$

$$
    v_i^TQx^* = v_i^TQ(\alpha_1v_1+\alpha_2v_2+\cdots+\alpha_nv_n) = \alpha_iv_i^TQv_i
$$

$$
    \alpha_i = \frac{v_i^TQx^*}{v_i^TQv_i} = \frac{v_i^Tb}{v_i^TQv_i}
$$

$$
    x^* = \sum_{i=1}^n \alpha_iv_i = \sum_{i=1}^n \frac{v_i^Tb}{v_i^TQv_i}v_i 
$$

#### Proposition

Let $x_0\in\mathbb{R}^n$ arbitrary, 

$$
\begin{aligned}
v_0 &= b-Qx_0 \\
\alpha_k &= -\frac{g^T_k v_k}{v^T_kQv_k} \\
x_{k+1} &= x_k + \alpha_k v_k \\
g_k &= Qx_k - b \\
v_{k+1} &= -g_{k+1} + \beta_k v_k \\
\beta_k &= \frac{g^T_{k+1}Qv_k}{v_k^TQv_k}
\end{aligned}
$$
