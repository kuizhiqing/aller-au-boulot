# Fourier Analysis

## Discrete Fourier Transform

In the space $\mathbb{C}^n$, fourier provide an orthogonal basis

$$
\{ v_k = (\omega^0, \omega^k, \omega^{2k}, \dots, \omega^{(n-1)k})\}_{0\le k\le n-1}
$$

with $ \omega = e^{2\pi i/n},$
the $n$-th root of $1$. 

This proposition is valable thanks to

$$
\langle v_k, v_l \rangle 
= n\cdot \delta_{k,l}.
$$

and $\forall k\in\mathbb{Z}, ||v_k||^2 = n$.

As a consequence,

$$
I_n
=
\frac{1}{n}
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{(n-1)}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{n-1} & \omega^{2(n-1)} & \cdots & \omega^{(n-1)(n-1)} 
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \bar\omega & \bar\omega^2 & \cdots & \bar\omega^{(n-1)}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \bar\omega^{n-1} & \bar\omega^{2(n-1)} & \cdots & \bar\omega^{(n-1)(n-1)} 
\end{bmatrix}
$$

The discrete fourier transform can be represented as follows,

$$
\begin{bmatrix}
\hat f_0 \\
\hat f_1 \\
\vdots \\
\hat f_{n-1} \\
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \bar\omega & \bar\omega^2 & \cdots & \bar\omega^{(n-1)}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \bar\omega^{n-1} & \bar\omega^{2(n-1)} & \cdots & \bar\omega^{(n-1)(n-1)} 
\end{bmatrix}
\begin{bmatrix}
 f_0 \\
 f_1 \\
\vdots \\
 f_{n-1} \\
\end{bmatrix}
$$

then the inverse fourier transform would be

$$
\begin{bmatrix}
 f_0 \\
 f_1 \\
\vdots \\
 f_{n-1} \\
\end{bmatrix}
=
\frac{1}{n}
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{(n-1)}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{n-1} & \omega^{2(n-1)} & \cdots & \omega^{(n-1)(n-1)} 
\end{bmatrix}
\begin{bmatrix}
\hat f_0 \\
\hat f_1 \\
\vdots \\
\hat f_{n-1} \\
\end{bmatrix}
$$

Equivalently, 
the fourier transform element

$$
\hat f_k = \sum_{j=0}^{n-1} \bar\omega^{jk} f_j, \quad 0\le k \le n-1, 
$$

and the inverse fourier transform element recovered by

$$
f_l = \sum_{k=0}^{n-1} \omega^{kl} \hat f_k, \quad 0\le l \le n-1.
$$

### Fast Fourier Transform (FFT)

Note that 
$\forall k\in\mathbb{Z}, e^{2\pi k} = 1$, 
the elements in matrix above are massively in common, $n$ elements disinct, precisely.

With the orthogonality,
rewrite the formula as follows,

$$
\begin{aligned}
\hat f_k = & \sum_{j=0}^{n-1} \omega^{jk} f_j \\
= & \sum_{j=0}^{n-1} \omega^{2jk} f_{2j} + \sum_{j=0}^{n-1} \omega^{2jk+k} f_{2j+1} \\
= & \sum_{j=0}^{n-1} \omega^{2jk} f_{2j} + \omega^{k}\sum_{j=0}^{n-1} \omega^{2jk} f_{2j+1} \\
\end{aligned}
$$

The above result shows that the coefficients of fourier transform can be calculated recursively using the method of divide and conquer
which reduce the calculation complexity from $O(n^2)$ to $O(n\operatorname{log}n)$.

Denote the transform matrix as $\Omega$, then

$$
\Omega_{2n} = 
\begin{bmatrix}
I_n & D_n \\
I_n & -D_n \\
\end{bmatrix}
\begin{bmatrix}
\Omega_n & 0 \\
0 & \Omega_n \\
\end{bmatrix}
P
$$

with $P$ is a permutation matrix and

$$
D_n = 
\begin{bmatrix}
\omega^0 & & & & \\
& \omega^1 & & & \\
& & \omega^2 & & \\
& & & \ddots & \\
\end{bmatrix}.

$$

## Fourier series

Consider the space $L^1(-\frac{T}{2},\frac{T}{2})$, 
the functions $\{\psi_k = e^{ikt\cdot 2\pi/T}, t\in\mathbb{R}\}_{k\in\mathbb{Z}} $, 
form a orthogonal basis 
since

$$
\langle \psi_k,\psi_l \rangle = 
\int_{-T/2}^{T/2}e^{ikt\cdot 2\pi/T} e^{-ilt\cdot 2\pi/T}dt =
T\cdot\delta_{k,l}
$$

#### Complex form

For function $f \in L^1(-\frac{T}{2},\frac{T}{2}) $,

$$
f(t) = \sum_{k=-\infty}^{\infty} c_k e^{ikt\cdot 2\pi/T}
$$

where

$$
c_k = \frac{1}{T} \langle f(t), e^{ikt\cdot 2\pi/T} \rangle
=  \frac{1}{T} \int_{-T/2}^{T/2}f(t)e^{-ikt\cdot 2\pi/T}dt 
$$

#### From complex to real

Since $e^{i\theta} = \operatorname{cos}\theta + i\operatorname{sin}\theta $,
suppose $c_k= a_k + ib_k,\, a_k,b_k\in\mathbb{R}$, then

$$
\begin{aligned}
f(t) =& \sum_{k=-\infty}^{\infty} (a_k + ib_k) (\operatorname{cos}(kt\cdot 2\pi/T) + i\operatorname{sin}(kt\cdot 2\pi/T)) \\
= & (a_0+ib_0) + \sum_{k=1}^{\infty} (a_k + ib_k) (\operatorname{cos}(kt\cdot 2\pi/T) + i\operatorname{sin}(kt\cdot 2\pi/T)) \\
& +\sum_{k=1}^{\infty} (a_{-k} + ib_{-k}) (\operatorname{cos}(-kt\cdot 2\pi/T) + i\operatorname{sin}(-kt\cdot 2\pi/T)) \\ 
\end{aligned}
$$

if $f(t)\in\mathbb{R}$,
$a_k=a_{-k}$ and $b_k=-b_{-k}$ which means $c_k=\bar{c_k}$, then

$$
f(t) = a_0 + \sum_{k=1}^{\infty} (2a_k) \operatorname{cos}(kt\cdot 2\pi/T) + (-2b_k)\operatorname{sin}(kt\cdot 2\pi/T)) 
$$

#### Real value, conventional form

For $2\pi$-periodic function $f$,

$$
f(t) = \frac{a_0}{2} + \sum_{k=1}^{\infty} (a_k \operatorname{cos}(kt) + b_k\operatorname{sin}(kt))
$$

where

$$
a_k 
= \frac{1}{||\operatorname{cos}(kt)||^2} \langle f(t), \operatorname{cos}(kt)\rangle
= \frac{1}{\pi} \int_{-\pi}^{\pi} f(t) \operatorname{cos}(kt) dt
$$

$$
b_k 
= \frac{1}{||\operatorname{sin}(kt)||^2} \langle f(t), \operatorname{sin}(kt)\rangle
= \frac{1}{\pi} \int_{-\pi}^{\pi} f(t) \operatorname{sin}(kt) dt
$$

## Fourier Transform

#### From fourier series to fourier transform

$$
f(t) = \sum_{k=-\infty}^{\infty} 
\left(
\frac{1}{T} \int_{-T/2}^{T/2}f(x)e^{-ikx\cdot 2\pi/T}dx 
\right)
\cdot
e^{ikt\cdot 2\pi/T}
$$

Since

$$
\int_{-\infty}^{\infty} 
{f}(t) dt = 
\lim_{T\to\infty}
\sum_{k=-\infty}^{\infty}
\frac{2\pi}{T} f(k\cdot\frac{2\pi}{T})
$$

which leads,

$$
\begin{aligned}
f(t) 
&= 
\lim_{T\to\infty}\sum_{k=-\infty}^{\infty} 
\left(
\frac{1}{T} \int_{-T/2}^{T/2}f(x)e^{-ikx\cdot 2\pi/T}
\cdot
e^{ikt\cdot 2\pi/T}
dx 
\right) \\
&=
\frac{1}{2\pi}
\lim_{T\to\infty}\sum_{k=-\infty}^{\infty} 
\frac{2\pi}{T} 
\left(
\int_{-T/2}^{T/2}f(x)e^{-ikx\cdot 2\pi/T}
\cdot
e^{ikt\cdot 2\pi/T}
dx 
\right) \\
&=
\frac{1}{2\pi}
\int_{-\infty}^{\infty}
\left(
\int_{-\infty}^{\infty}f(x)e^{-ix\cdot \xi}
\cdot
e^{it\cdot \xi}
dx
\right)
d\xi \\ 
&=
\frac{1}{2\pi}
\int_{-\infty}^{\infty}
\left(
\int_{-\infty}^{\infty}f(x)e^{-ix\cdot \xi}
dx
\right) 
e^{it\cdot \xi}
d\xi \\
\end{aligned}
$$

Denote the Fourier transform as follows,

$$
\hat {f}(\xi )
=
\int_{-\infty}^{\infty}f(x)e^{-ix\cdot \xi}
dx
$$

then the Inverse Fourier transform

$$
f(t)
=
\frac{1}{2\pi}
\int_{-\infty}^{\infty}
\hat {f}(\xi )
e^{it\cdot \xi}
d\xi
$$

this is non-unitary, angular frequency form.

#### Fourier transform (unitary, oridnay frequency form)

For function $f\in L^1(\mathbb{R})$, 
i.e. $\int_{-\infty}^{\infty}|f(t)|dt < \infty$,

$$
\hat {f}(\xi )
=\int_{-\infty }^{\infty } f(t)e^{-2\pi i\xi t }\,dt
$$

with 
$\hat f\in L^1(\mathbb{R})$, i.e.
$\int_{-\infty}^{\infty}|f(\xi)|d\xi < \infty$.

Inverse Fourier transform

$$
{f}(t) = \int_{-\infty }^{\infty } \hat f(\xi)e^{2\pi it\xi }\,d\xi
$$

<!--
$\mathcal{F}(f)=\hat f$
-->

## More

If function $f$ is frequency band-limited, i.e. 
$ \exists \Omega \le 0, \forall |\xi| \ge \Omega, \hat f(\xi) = 0.  $
Then

$$
\hat f(\xi) = \sum_{k=-\infty}^{\infty} c_k e^{ik\xi\cdot \pi/\Omega}
$$

where

$$
c_k = \frac{1}{2\Omega} \langle \hat f(\xi), e^{ik\xi\cdot \pi/\Omega} \rangle
=  \frac{1}{2\Omega} \int_{-\Omega}^{\Omega}\hat f(\xi)e^{-ik\xi\cdot \pi/\Omega}d\xi 
$$

Further

$$
c_k 
=  \frac{2\pi}{2\Omega}\frac{1}{2\pi} \int_{-\infty}^{\infty}\hat f(\xi)e^{i(-k\pi/\Omega)\xi}d\xi
=  \frac{2\pi}{2\Omega} f(-k\pi/\Omega)
$$

and

$$
\hat f(\xi) 
= \sum_{k=-\infty}^{\infty} \frac{2\pi}{2\Omega} f(-k\pi/\Omega) e^{ik\xi\cdot \pi/\Omega}
= \sum_{k=-\infty}^{\infty} \frac{2\pi}{2\Omega} f(k\pi/\Omega) e^{-ik\xi\cdot \pi/\Omega}
$$

Therefore,

$$
\begin{aligned}
f(t)
&= \frac{1}{2\pi} \int_{-\Omega}^{\Omega} \hat {f}(\xi ) e^{it\cdot \xi} d\xi\\
&= \frac{1}{2\pi} \int_{-\Omega}^{\Omega} \left( \sum_{k=-\infty}^{\infty} 
\frac{2\pi}{2\Omega} f(k\pi/\Omega) e^{-ik\xi\cdot \pi/\Omega} \right) e^{it\cdot \xi} d\xi \\
&= \frac{1}{2\Omega} \sum_{k=-\infty}^{\infty} f(k\pi/\Omega) 
\int_{-\Omega}^{\Omega} e^{-i\xi(k\cdot \pi/\Omega-t)}  d\xi \\
&= \frac{1}{2\Omega} \sum_{k=-\infty}^{\infty} f(k\pi/\Omega) 
\left[
\frac{1}{-i(k\cdot \pi/\Omega-t)}
e^{-i\xi(k\cdot \pi/\Omega-t)}  
\right]_{-\Omega}^{\Omega} \\
&= \sum_{k=-\infty}^{\infty} f(k\pi/\Omega) 
\frac{\operatorname{sin}(k\pi-\Omega t)}{k\pi- \Omega t}
\end{aligned}
$$

#### Derivatives.

$$
\int_{-\infty}^{\infty}f'(t)e^{-it\cdot \xi} dt
=
\left[ f(t)e^{-it\cdot \xi}\right]_{-\infty}^{\infty} - 
\int_{-\infty}^{\infty}f(t)(-i\xi)e^{-it\cdot \xi} dt
=
(i\xi)\int_{-\infty}^{\infty}f(t)e^{-it\cdot \xi} dt
$$

$$
\mathcal{F}(\frac{df(t)}{dt})
=
(i\xi)
\mathcal{F}(f(t))
$$

<!--
#### wave function
$$
u_{tt} = 
c^2 
u_{xx}
$$
-->

#### Convolution

Definition

$$
(f*g)(x) = \int_{-\infty}^{\infty}f(\xi)g(x-\xi) d \xi
$$

$$
\mathcal{F}(f*g) = \mathcal{F}(f)\cdot\mathcal{F}(g) 
$$

$$
f*g = g*f
$$

$$
\begin{aligned}
\mathcal{F}(f*g) 
= &
\int_{-\infty}^{\infty}
\int_{-\infty}^{\infty}f(\xi)g(x-\xi) d\xi
e^{-ix\cdot t}
dx \\

= &
\int_{-\infty}^{\infty}
f(\xi)
\left(
\int_{-\infty}^{\infty}
g(x-\xi) 
e^{-ix\cdot t}
dx
\right)
d\xi \\

= &
\int_{-\infty}^{\infty}
f(\xi)
\left(
\int_{-\infty}^{\infty}
g(x-\xi) 
e^{-i(x-\xi)\cdot t}
d(x-\xi)
\right)
e^{-i\xi\cdot t}
d\xi \\

= &
\hat g(t)
\int_{-\infty}^{\infty}
f(\xi)
e^{-i\xi\cdot t}
d\xi \\

= &
\mathcal{F}(f)\cdot\mathcal{F}(g) \\
\end{aligned}
$$

#### Parseval's theorem

$$
\int_{-\infty}^{\infty}|\hat f(\xi)|^2d\xi =
2\pi \int_{-\infty}^{\infty}|f(t)|^2dt
$$
