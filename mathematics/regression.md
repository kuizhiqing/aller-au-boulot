#### Regression
#### Linear Regression
Objective function
$$
{ {\underset w{\operatorname {min} }}(\|y-x\cdot w \|^{2}_2})
$$
Ordinary least square Lagrangian function
$$
L(w) = ||y - x\cdot w ||^2_2 = (y - x\cdot w )^T(y - x\cdot w ) = y^Ty - w^Tx^T y - y^Tx w + w^Tx^T xw 
$$
$$
\frac{\partial L(w)}{\partial w} = 2 x^T x w - 2 x^T y 
$$
Let $ {\partial L(w)}/{\partial w} = 0 $ leads to 
$$
w = (x^T x)^{-1} x^T y 
$$

With respect to different constraints,
* $||w||_1<t$ for lasso,
* $||w||_2<t$ for ridge.

#### Ridge Regression : L1 Regularization
Least absolute shrinkage and selection operator
$$
{ {\underset w{\operatorname {min} }}(\|y-x\cdot w \|_2^{2}+\lambda \|w \|_{1}).}
$$

* When $\lambda = 0$ , no parameters are eliminated. The estimate is equal to the one found with linear regression.
* As $\lambda$ increases, more and more coefficients are set to zero and eliminated (theoretically, when $\lambda=\infty$, all coefficients are eliminated).
* As $\lambda$ increases, bias increases.
* As $\lambda$ decreases, variance increases.

#### Lasso Regression : L2 Regularization

$$
{ {\underset w{\operatorname {min} }}(\|y-x\cdot w \|_2^{2}+\lambda \|w \|_2^{2}).}
$$
$$
L(w) = w^Tx^T xw - w^Tx^T y - y^Tx w + y^Ty + \lambda w^Tw
$$
$$
\frac{\partial L(w)}{\partial w} = 2 x^T x w - 2 x^T y + 2\lambda w 
$$
Let $ {\partial L(w)}/{\partial w} = 0 $ leads to 
$$
w = (x^T x+\lambda I)^{-1} x^T y 
$$

#### Elastic net Regularization

$$
{ {\underset w{\operatorname {min} }}(\|y-x\cdot w \|_2^{2}+\lambda _{2}\|w \|_2^{2}+\lambda _{1}\|w \|_{1}).}
$$

#### Logistic Regression
For $p\in (0,1)$,
$$
    \operatorname{logit} p 
     = \operatorname{ln}(\operatorname{odds})
     = \operatorname{ln}\frac{p}{1-p}
$$
Logistic function is inverse logit, sigmoid function defined as follows, 
for $q\in (-\infty,\infty)$,
$$
    p = s(q) = 
    \operatorname{logit}^{-1} q
     ={\frac {e^{q}}{e^{q}+1}}={\frac {1}{1+e^{-q}}}
$$
Suppose $\operatorname{logit} y$ linear with $x$, i.e.
$$
     \operatorname{ln}\frac{y}{1-y} = \beta_0 + \beta_1 x
$$
then 
$$
    y = s(x) = {\frac {1}{1+e^{-(\beta_0 + \beta_1 x)}}}
$$


