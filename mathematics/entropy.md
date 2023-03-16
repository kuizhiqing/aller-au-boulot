
## Entropy

### Definition
For a discrete probability distribution $p$ on the finite set $\{x_1,x_2,\dots,x_N\}$  with $p_i=p(x_i)$,
the entropy of $p$ is defined as 

$$
h(p) = -\sum_{i=1}^{N} p_i \operatorname{log} p_i.
$$

For a continuous probability density function $p$ on an interval $[a,b]$, 
the entropy of $p$ is defined as

$$
h(p) = -\int_{a}^{b} p(x) \operatorname{log} p(x) \operatorname{d} x.
$$

### Theorem

For a probability density function $p$ on a finite set $\{x_1,x_2,\dots,x_N\}$,
then

$$
h(p) \le \operatorname{log} n,
$$

with equality iff. $p$ is uniform, i.e. $\forall i \le N, p(x_i)=1/n$.

> Uniform probability yields maximum uncertainty and therefore maximum entropy.

### Theorem

For a continuous probability density function $p$ on $\mathbb{R}$ with variance $\sigma^2$, then

$$
h(p) \le \frac{1}{2} (1 + \operatorname{log}(2\pi\sigma^2))
$$

with equality iff. $p$ if Gaussian with variance $\sigma^2$, i.e. for some $\mu$ we have 

$$
p(x)= \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
$$

### Theorem

For a continuous probability density function $p$ on $(0,\infty)$ with mean $\lambda$, then

$$
h(p) \le 1 + \operatorname{\lambda},
$$

with equality iff. $p$ is exponential with mean $\lambda$, i.e. 

$$
p(x) = \frac{1}{\lambda} e^{-\frac{1}{\lambda}x}.
$$

---

<!--
$$
    H(X) = -\sum_{x\in X} P(x)\operatorname{log} P(x)
$$
-->


#### Cross entropy
The cross entropy of the distribution $q$ relative to a distribution $p$ over a given set is defined as follows:

$$
    H(p,q)=-\operatorname{E} _{p}[\log q] = -\sum p\log q =  H(p)+D_{\mathrm {KL} }(p\|q)
$$

#### Kullback-Leibler divergence

The Kullback-Leibler divergence (relative entropy) was introduced as the directed divergence between two distributions

$$
    D_{\text{KL}}(P\parallel Q)=-\sum _{x\in {\mathcal {X}}}P(x)\log \left({\frac {Q(x)}{P(x)}}\right)
$$

> The Kullback-Leibler divergence is then interpreted as the average difference of the number of bits required for encoding samples of $P$ using a code optimized for $Q$ rather than one optimized for $P$.


#### Jensen-Shannon divergence

$$
  D_{\text{JS}}(P\parallel Q)= 
  \frac{1}{2} D_{\text{KL}}(P\parallel M) +
  \frac{1}{2} D_{\text{KL}}(Q\parallel M)
$$

where $M = \frac{1}{2} (P+Q)$


---

### Mutual Information

Let $(X,Y)$ be a pair of random variables with values over the space 
${\mathcal {X}}\times {\mathcal {Y}}$. If their joint distribution is 
${\displaystyle P_{(X,Y)}}$ and the marginal distributions are 
$P_X$ and 
${\displaystyle P_{Y}}$, the **mutual information** is defined as

$$
{\displaystyle I(X;Y)=D_{\mathrm {KL} }(P_{(X,Y)}\|P_{X}\otimes P_{Y})}
$$

where $D_{{{\mathrm  {KL}}}}$ is the Kullback–Leibler divergence.

#### PMFs for discrete distributions

The mutual information of two jointly discrete random variables
$X$ and
$Y$ is calculated as a double sum:

$$
{\displaystyle \operatorname {I} (X;Y)=\sum _{y\in {\mathcal {Y}}}\sum _{x\in {\mathcal {X}}}{P_{(X,Y)}(x,y)\log \left({\frac {P_{(X,Y)}(x,y)}{P_{X}(x)\,P_{Y}(y)}}\right)},}
$$

where
${\displaystyle P_{(X,Y)}}$ is the joint probability mass function of
$X$ and
$Y$, and
$P_X$ and
${\displaystyle P_{Y}}$ are the marginal probability mass functions of
$X$ and
$Y$ respectively.

#### PDFs for continuous distributions

In the case of jointly continuous random variables, the double sum is replaced by a double integral:

$$
{\displaystyle \operatorname {I} (X;Y)=\int _{\mathcal {Y}}\int _{\mathcal {X}}{P_{(X,Y)}(x,y)\log {\left({\frac {P_{(X,Y)}(x,y)}{P_{X}(x)\,P_{Y}(y)}}\right)}}\;dx\,dy,}
$$

where
${\displaystyle P_{(X,Y)}}$ is now the joint probability density function of
$X$ and
$Y$, and
$P_X$ and
${\displaystyle P_{Y}}$ are the marginal probability density functions of
$X$ and
$Y$ respectively.

#### Mutual information and Kullback–Leibler divergence

Mutual information is the Kullback–Leibler divergence from the product of the marginal distributions

$$
{\displaystyle \operatorname {I} (X;Y)=D_{\text{KL}}\left(p_{(X,Y)}\parallel p_{X}p_{Y}\right)}
$$

$$
{\displaystyle \operatorname {I} (X;Y)=\mathbb {E} _{Y}\left[D_{\text{KL}}\!\left(p_{X\mid Y}\parallel p_{X}\right)\right]}
$$


