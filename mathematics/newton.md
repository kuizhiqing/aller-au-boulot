#### Newton's Method

Given a twice differentiable function $ f:\mathbb {R} \to \mathbb {R} $, solve the optimization problem

$$
\min _{x\in \mathbb {R} }f(x).
$$

Consider the second-order Taylor expansion of $f$ around $x_{k}$

$$
f(x_{k}+t)\approx f(x_{k})+f'(x_{k})t+{\frac {1}{2}}f''(x_{k})t^{2}.
$$

Setting the derivative to zero,

$$
0={\frac {\rm {d}}{{\rm {d}}t}}\left(f(x_{k})+f'(x_{k})t+{\frac {1}{2}}f''(x_{k})t^{2}\right)=f'(x_{k})+f''(x_{k})t,
$$

the minimum is achieved for

$$
t=-{\frac {f'(x_{k})}{f''(x_{k})}},
$$

then

$$
x_{k+1}=x_{k}+t=x_{k}-{\frac {f'(x_{k})}{f''(x_{k})}}.
$$
