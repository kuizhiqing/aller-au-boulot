# differentiation

$$
\mathbf{J} f
= \begin{bmatrix}
\frac{\partial f_0}{\partial x_0} & \frac{\partial f_0}{\partial x_1} & \cdots & \frac{\partial f_0}{\partial x_{n-1}} \\
\frac{\partial f_1}{\partial x_0} & \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_{n-1}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_{m-1}}{\partial x_0} & \frac{\partial f_{m-1}}{\partial x_1} & \cdots & \frac{\partial f_{m-1}}{\partial x_{n-1}} \\
\end{bmatrix}
$$

$$
\frac{\partial }{\partial x} (f\cdot g)= \frac{\partial f}{\partial x}\cdot g + f\cdot \frac{\partial g}{\partial x}
$$

$$
\frac{\partial }{\partial x} (f + g)= \frac{\partial f}{\partial x} + \frac{\partial g}{\partial x}
$$

$$
\frac{\partial }{\partial x} (f\circ g)= \frac{\partial f}{\partial g}\frac{\partial g}{\partial x}
$$

$$
\frac{\partial }{\partial x} (f\circ g)= \frac{\partial f}{\partial g}\frac{\partial g}{\partial x}
=\left[ \frac{\partial f}{\partial g_1} \, \frac{\partial f}{\partial g_2} \right]
\left[\frac{\partial g_1}{\partial x} \, \frac{\partial g_2}{\partial x}\right]^T
=\frac{\partial f}{\partial g_1} \frac{\partial g_1}{\partial x}
+\frac{\partial f}{\partial g_2} \frac{\partial g_2}{\partial x}
$$

$$
f(x_1,x_2) = x_1(x_1+x_2)
$$

$$
\begin{matrix}
g_1 =&  x_1 \\
g_2 =&  x_2 \\
g_3 =&  g_1 + g_2 \\
g_4 =& g_1\cdot g_3 \\
\end{matrix}
$$

$$
\frac{\partial f}{\partial x_1}
=\frac{\partial g_4}{\partial x_1}
=\frac{\partial }{\partial x_1}(g_1\cdot g_3)
=\frac{\partial g_1}{\partial x_1} g_3
+g_1 \frac{\partial g_3}{\partial x_1}
=\frac{\partial g_1}{\partial x_1} g_3
+\left(
g_1 \frac{\partial g_1}{\partial x_1}
+g_1 \frac{\partial g_2}{\partial x_1}\right)
$$

$$
\frac{\partial f}{\partial x_1}
= \frac{\partial f}{\partial g_4}
\frac{\partial g_4}{\partial x_1}
= \frac{\partial g_4}{\partial g_1}
\frac{\partial g_1}{\partial x_1}
+\frac{\partial g_4}{\partial g_3}
\frac{\partial g_3}{\partial x_1}
= \frac{\partial g_4}{\partial g_1}
\frac{\partial g_1}{\partial x_1}
+\frac{\partial g_4}{\partial g_3}
\left(
\frac{\partial g_3}{\partial g_1}
\frac{\partial g_1}{\partial x_1}
+\frac{\partial g_3}{\partial g_2}
\frac{\partial g_2}{\partial x_1}
\right)
$$
