#### Principal Component Analysis
In order to maximize variance, the first weight vector $w_{1}$ thus has to satisfy
$$
{\mathbf  {w}}_{{1}}=
{\underset  {\Vert {\mathbf  {w}}\Vert =1}{\operatorname {\arg \,max}}}\,\left\{\sum _{i}\left({\mathbf  {x}}_{{(i)}}\cdot {\mathbf  {w}}\right)^{2}\right\}
={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {\arg \,max} }}\,\{\Vert \mathbf {Xw} \Vert ^{2}\}={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {\arg \,max} }}\,\left\{\mathbf {w} ^{T}\mathbf {X^{T}} \mathbf {Xw} \right\}
$$
Since $w_1$ has been defined to be a unit vector, it equivalently also satisfies
$$
\mathbf {w} _{1}={\operatorname {\arg \,max} }\,\left\{{\frac {\mathbf {w} ^{T}\mathbf {X^{T}} \mathbf {Xw} }{\mathbf {w} ^{T}\mathbf {w} }}\right\}
$$
The quantity to be maximised can be recognised as a Rayleigh quotient. 
A standard result for a positive semidefinite matrix such as $X^TX$ is that 
the quotient's maximum possible value is the largest eigenvalue of the matrix, which occurs when w is the corresponding eigenvector.

Let $x\in\mathbb{R}^n$ be origin samples, projection vector $w$ satisfies
$$
w^Tw = 1.
$$
Denote $\Sigma =\frac{1}{N} (X-\bar X)(X-\bar X)^T $ the variance of origin samples,
variance of projected samples, 
$$
\sigma(X,w) = \frac{1}{N} (w^TX-w^T\bar X)(w^TX-w^T\bar X)^T = w^T \Sigma w
$$

Lagrangian multiplier form,
$$
    \operatorname{max} w^T\Sigma w + \lambda (1-w^Tw),
$$
leads to
$$
    \Sigma w = \lambda w.
$$
the variance $\sigma(X,w)$ is maximized with $w$ is the eigenvector of $\Sigma$,
the maximum value is the related eigenvalue $\lambda$.


PCA is either done by 
* singular value decomposition of a design matrix;
* eigenvalue decomposition on the covariance matrix.

#### Eigenvalue Decomposition
A (non-zero) vector $v\in\mathbb{R}^n$ is an eigenvector of a square matrix $A\in\mathbb{R}^n\times\mathbb{R}^n$ if it satisfies the linear equation
$$
{A} {v} =\lambda {v} 
$$
where $\lambda\in\mathbb{R}$ termed the eigenvalue corresponding to $v$. 


Since 
$\forall i\in\mathbb{N}$, 

$$
{A} {v_i} =\lambda {v_i} ,
$$

$$
A
\begin{bmatrix}
v_1 & v_2 & \cdots & v_n 
\end{bmatrix}
=
\begin{bmatrix}
v_1 & v_2 & \cdots & v_n 
\end{bmatrix}
\begin{bmatrix}
\lambda_1 & & & \\
& \lambda_2 & & \\
& & \ddots & & \\
& & & \lambda_n & \\
\end{bmatrix}
$$

$$
A V = V\Lambda
$$

Diagonalisable matrice $A$ can be factorized as
$$
A = V\Lambda V^{-1}
$$
where $\Lambda_{ii}=\lambda_i$ and $v_i$ is the $i$-th column of $V$.


#### Singualar Value Decomposition
The singular value decomposition (svd) of matrix $A\in\mathbb{R}^n\times\mathbb{R}^m$
$$
A = U\Sigma W^T
$$
where $U\in\mathbb{R}^n\times\mathbb{R}^n$,
$W\in\mathbb{R}^m\times\mathbb{R}^m$ and
$\Sigma\in\mathbb{R}^n\times\mathbb{R}^m$ is diagonal with positive number.
$$
A^TA = W\Sigma^TU^T U\Sigma W^T = W\Sigma^2 W^T
$$
The right singular vectors $W$ of $A$ the eigenvectors of $X^TX$,
while the singular values of $A$ are equal to the square-root of the eigenvalues of $X^TX$.

