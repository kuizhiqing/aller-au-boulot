# kmeans

Let

$$
x_1, x_2, \dots x_n \in\mathbb{R}^m
$$

find 
$ \hat x \in \mathbb{R}^m $

such that 

$$
\operatorname{min} \sum_{k=1}^n |\theta_k|
$$

where

$$
\operatorname{cos} \theta_k = \frac{x_k\cdot \hat x}{||x_k||\cdot||\hat x||}
$$

cos defines $\forall x_j,x_k \in\mathbb{R}^m$

$$
\operatorname{cos} \theta = \frac{x_j}{||x_j||}\cdot \frac{x_k}{||x_k||}
$$

Equivalent form

$$
\operatorname{min} -\sum_{k=1}^n \frac{x_k\cdot \hat x}{||x_k||\cdot||\hat x||} 
=
\operatorname{min} -\frac{1}{c}\sum_{k=1}^n \frac{1}{||x_k||} \left(\sum_{l=1}^m x_{kl}\cdot \hat x_l\right)
$$

under condition

$$
\sum_{l=1}^{m} (\hat x_l)^2 = c^2
$$

where $c\in\mathbb{R}, c>0.$

$$
\mathcal{L}(\hat x_1,\hat x_2,\dots,\hat x_m,\lambda) 
= -\frac{1}{c}\sum_{k=1}^n \frac{1}{||x_k||} \left(\sum_{l=1}^m x_{kl}\cdot \hat x_l\right)
+\lambda\left(
\sum_{l=1}^{m} (\hat x_l)^2 - c^2
\right)
$$

For $1\le l \le n$, let

$$
\frac{\partial \mathcal{L}(\hat x_1,\hat x_2,\dots,\hat x_m,\lambda)}{\partial \hat x_l}
=
= -\frac{1}{c}\sum_{k=1}^n \frac{x_{kl}}{||x_k||} 
 +2\lambda \hat x_l = 0 
$$

then

$$
\hat x_l = \frac{1}{2\lambda c}\sum_{k=1}^n \frac{x_{kl}}{||x_k||} 
$$

replace $\hat x_l$ with above expression

$$
\sum_{l=1}^{m} \left(
\frac{1}{2\lambda c}\sum_{k=1}^n \frac{x_{kl}}{||x_k||} 
\right)^2 = c^2
$$

then

$$
2\lambda c =
\frac{1}{c}
\left(
\sum_{l=1}^{m} 
\left(
\sum_{k=1}^n \frac{x_{kl}}{||x_k||} 
\right)^2 
\right)^{1/2}
$$

$$
\hat x_l = c\cdot 
\left(
\sum_{k=1}^n \frac{x_{kl}}{||x_k||} 
\right)
\left(
\sum_{l=1}^{m} 
\left(
\sum_{k=1}^n \frac{x_{kl}}{||x_k||} 
\right)^2 
\right)^{-1/2}
$$

For $||x_k|| = 1 $ and $ c= 1$,

$$
\hat x_l = 
\left(
\sum_{k=1}^n {x_{kl}} 
\right)
\left(
\sum_{l=1}^{m} 
\left(
\sum_{k=1}^n {x_{kl}} 
\right)^2 
\right)^{-1/2}
=
\bar x_l
\left(
\sum_{l=1}^{m} 
{\bar x_l}^2 
\right)^{-1/2}
= \frac{\bar x_l}{||\bar x||}
$$

$$
\hat x 
= \frac{\bar x}{||\bar x||}
$$

$$
||x_j - x_k||^2 = x_j^Tx_j - 2 x_j^Tx_k + x_k^Tx_k
$$

$$
||x_j - x_k||^2 = 2(1 - x_j^Tx_k)
$$
