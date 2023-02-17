# A wavelet tour of signal processing the Sparse way

Stephane Mallat

#### history

* 1910 - Haar - a piecewise constant function
* 1980 - Stromberg - a piecewise linear function - an orthonormal basis, better approximation of smooth function
* Morlet, Grossmann - continuous wavelet transform 
* Meyer - a family of orthonormal wavelet bases with function infinitely continuously differentiable 
* Daubechies - wavelets of compact support

#### Time-Frequency localisation trade-off

Keeping $ \|f_s\|^2 = \|f\|^2$, by denoting

$$
f_s (t) = \frac{1}{\sqrt{s}} f\left(\frac{t}{s}\right)
$$

then

$$
\hat f_s(\omega) = \sqrt{s} \hat f(s\omega)
$$

it shows that we lose in frequency while we gain in time.

> $\mathcal{F}\left( \frac{1}{\sqrt{s}} e^{-\frac{t^2}{s}} \right) = \frac{1}{\sqrt{2}} e^{-\frac{s\omega^2}{4}}$
