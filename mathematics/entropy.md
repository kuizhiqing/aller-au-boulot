
## Entropy

#### Definition
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

#### Theorem

For a probability density function $p$ on a finite set $\{x_1,x_2,\dots,x_N\}$,
then
$$
h(p) \le \operatorname{log} n,
$$
with equality iff. $p$ is uniform, i.e. $\forall i \le N, p(x_i)=1/n$.

#### Theorem

For a continuous probability density function $p$ on $\mathbb{R}$ with variance $\sigma^2$, then
$$
h(p) \le \frac{1}{2} (1 + \operatorname{log}(2\pi\sigma^2))
$$
with equality iff. $p$ if Gaussian with variance $\sigma^2$, i.e. for some $\mu$ we have 
$$
p(x)= \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
$$

#### Theorem

For a continuous probability density function $p$ on $(0,\infty)$ with mean $\lambda$, then
$$
h(p) \le 1 + \operatorname{\lambda},
$$
with equality iff. $p$ is exponential with mean $\lambda$, i.e. 
$$
p(x) = \frac{1}{\lambda} e^{-\frac{1}{\lambda}x}.
$$

---

#### Probability density function

#### Entropy 
$$
    H(X) = -\sum_{x\in X} P(x)\operatorname{log} P(x)
$$
Uniform probability yields maximum uncertainty and therefore maximum entropy.

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



