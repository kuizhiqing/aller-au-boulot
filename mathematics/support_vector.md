#### Support Vector Machine
Suppose dataset $x_i\in\mathbb{R}^n$ related to label $y_i\in\{+1,-1\}$,
find $w\in\mathbb{R}^n$ and bias $b\in\mathbb{R}$ such that

$$
\begin{cases}
        x_i \cdot w + b \ge +1, \quad y_i &= +1 \\
        x_i \cdot w + b \le -1, \quad y_i &= -1 
\end{cases}
$$

which leads to

$$
y_i(x_i \cdot w + b) \ge +1.
$$

The distance between optimal hyperplanes is $2/||w||$, so maximizing the margin equals to minimizing $||w||$.

Primary problem : min $L_p(w,\alpha_i)$,

$$
L_p(w,\alpha_i) = \frac{1}{2} ||w||^{2} - \sum_{i}^n \alpha_i y_i(x_i\cdot w+b) + \sum_i^n \alpha_i
$$

$$
\frac{\partial L_p(w,\alpha_i)}{\partial w} = 0
$$

leads to

$$
w = \sum_i^n \alpha_i y_i x_i
$$

and $\forall i,\alpha_i \ge 0$,

$$
\frac{\partial L_p(w,\alpha_i)}{\partial b} = 0
$$

leads to

$$
\sum_i^n \alpha_i y_i = 0.
$$

Dual problem : max $L_d(\alpha_i)$,

$$
L_d(\alpha_i) =\sum_i^n \alpha_i - \frac{1}{2} \sum_{i,j}^n \alpha_i \alpha_j y_i y_j (x_i\cdot x_j) 
$$

s.t. $\forall i,\alpha_i \ge 0$,

$$
\sum_i^n \alpha_i y_i = 0.
$$

Only *support vectors* have nonzero coefficients $\alpha_i$.

Approaches : sub-gradient descent and coordinate descent.

