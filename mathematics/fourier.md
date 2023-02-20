# 傅立叶分析 Fourier Analysis

## 基础知识

掌握傅立叶分析需要了解以下概念：

* 向量空间 (vector space)
* 线性 (linearity)
* 向量基 (base/basis)

#### 内积

离散向量
$x,y\in \mathbb{C}^N$
的内积定义为

$$
\langle x, y \rangle = \sum_{j=1}^N x_j \bar y_j.
$$

连续函数
$f,g \in L^2(\mathbb{R})$
的内积定义为

$$
\langle f, g \rangle = \int f \bar g.
$$

> 注意理解离散求和符号 $\sum$ 和连续求和符号 $\int$ 在概念上的一致性。

#### 正交基

如果 $B$ 是向量空间 $H$ 的一组正交基，那么对于所有向量 $\forall x \in H$, 

$$
x = \sum_{b\in B} \frac{\langle x,b\rangle}{\langle b, b\rangle} b
= \sum_{b\in B} \frac{\langle x,b\rangle}{||b||^2} b.
$$

一般将 $\frac{\langle x,b\rangle}{||b||^2}$ 部分叫做向量的坐标。

> 注意同一向量空间可能有不同的基，所以对应不同的坐标。

在二维平面 $\mathbb{R}^2$ 中，经典基向量为
$v_1 = (1, 0)$ 和 $v_2=(0,1)$，他们组成的变换矩阵为单位矩阵 (identity matrix)

$$
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
\end{bmatrix}
$$

假设我们选取 $u_1 =(1,1)$ 和 $u_2=(-1,1)$ 作为新的一组基，
因为 $\lang u_1, u_2\rang = 0$, 这是一组正交基，他们组成的变换矩阵为

$$
T=
\frac{1}{2}
\begin{bmatrix}
1 & 1 \\
-1 & 1 \\
\end{bmatrix}
$$

> 这里我们并没有对向量进行归一化处理，所以在变换矩阵或逆变换矩阵里一般会有一个系数，
> 这也是在傅立叶变换的定义中有不同系数版本的原因。

任意二维平面上的向量 $(x,y) \in\mathbb{R}^2$ 都可以通过基 $u_1,u_2$ 的线性组合表示，
向量在该坐标系中的坐标可以通过变换矩阵得到

$$
\frac{1}{2}
\begin{bmatrix}
1 & 1 \\
-1 & 1 \\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
\end{bmatrix}
$$

> 可以看出， $(1,1)$ 变换后的坐标为 $(1,0)$，因为 $(1,1) = \frac{1}{2}(1\cdot u_1 + 0\cdot u_2)$，
> $(0,1)$ 变换后的坐标为 $(1,1)$，因为 $(0,1) = \frac{1}{2}(1\cdot u_1 + 1\cdot u_2)$.

从坐标系 $(u_1,u_2)$ 变换回 $(v_1,v_2)$ 的逆变换矩阵也即矩阵 $T$ 的逆矩阵为

$$
T^{-1}=
\begin{bmatrix}
1 & -1 \\
1 & 1 \\
\end{bmatrix}
$$

总结而言，矩阵 $T$ 可以进行坐标变换，同时其逆矩阵 $T^{-1}$ 可以实现坐标逆变换，

$$
I= T^{-1} T
=\left(
\begin{bmatrix}
1 & -1 \\
1 & 1 \\
\end{bmatrix}
\right)
\cdot
\left(
\frac{1}{2}
\begin{bmatrix}
1 & 1 \\
-1 & 1 \\
\end{bmatrix}
\right)
$$

## 离散傅立叶变换

现实场景下的信号处理一般针对离散数据而言，
长度为 $n$ 的离散数据可以当作
 $n$ 维向量空间中的向量,
假定它在经典基向量下的坐标为
$ x = (x_0, x_1, x_2, \dots, x_{n-1}) \in\mathbb{C}^n$.

这里我们考虑一组新的基向量
$(v_0, v_1, \dots, v_{n-1})$，
注意 $v_k\in\mathbb{C}^n$ 是一个 $n$ 维向量。
假定这是一组正交基，那么记

$$
V =
\begin{bmatrix}
v_0 \\
v_1 \\
\vdots \\
v_{n-1} \\
\end{bmatrix}
\in\mathbb{C}^{n\times n}
$$

则 $ V\cdot x $ 表示向量 $x$ 在变换后的向量。
注意这里的 $V$ 是 $n\times n$ 的矩阵。 

假设矩阵 $V$ 是可逆的，它的逆矩阵记做 $V^{-1}$，那么我们有

$$
I = V^{-1} V
$$

对于所有的向量 $x\in\mathbb{C}^n$，

$$
(V^{-1} V)\cdot x = V^{-1}\cdot (Vx) = x
$$

这样我们可以将 $x' = Vx$ 叫做 $x$ 的 *～变换*，而 $x = V^{-1}x'$ 叫做 $x$ 的 *～逆变换*。

当采用如下的基向量时，我们得到了离散条件下的 *傅立叶变换* 和 *傅立叶逆变换*，

$$
v_k = (e^{-2\pi i k \cdot 0/n}, e^{-2\pi i k \cdot 1/n}, \dots, e^{-2\pi i k\cdot (n-1)/n})
$$

> 这里 $e$ 为欧拉 Euler 常数，$e^{i\theta} = \operatorname{cos}\theta + i\operatorname{sin}\theta$，所以
$k$ 为整数时 $e^{2\pi i k} = 1$，当 $k=l/n$ 为时可以理解为 1 的 $n$ 次方根的第 $l$ 个根。 

$k$ 的取值为从 $0$ 到 $n-1$ 共 $n$ 个向量，注意到

$$
\lang v_k, v_l\rang = n\cdot  \delta_{k,l}
$$

也即这是一组正交向量，证明可以参考内积的定义进行展开计算。

> 基向量 $v_k$ 的选取非常巧妙，也是傅立叶变换的精髓，可以这样理解，
> 对于 $n$ 维空间，选取 1 的 $n$ 个 $n$ 次方根作为一个基向量，然后通过分别取 $0$ 到 $n-1$ 次方构建出一组正交基。

记 $\omega = e^{-2\pi i/n}$, 
基向量 $v_k = (\omega^0, \omega^k, \omega^{2k}\dots, \omega^{(n-1)k})$，
傅立叶变换的矩阵为

$$
F_n =
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{(n-1)}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{n-1} & \omega^{2(n-1)} & \cdots & \omega^{(n-1)(n-1)} 
\end{bmatrix}
$$

傅立叶逆变换的矩阵为

$$
\hat F_n =
\frac{1}{n}
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \bar\omega & \bar\omega^2 & \cdots & \bar\omega^{(n-1)}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \bar\omega^{n-1} & \bar\omega^{2(n-1)} & \cdots & \bar\omega^{(n-1)(n-1)} 
\end{bmatrix}
$$

由此可见，傅立叶变换的矩阵或者说基向量的精髓之处在于它的逆矩阵可以如此轻而易举地计算或者说构造出来。

至此，对于任意长度为 $n$ 的向量 $x$，如果它表示的是时域的信号，
那么可以通过 $F_n x$ 计算它的傅立叶变换 $x'$，得到频域上的信号表示，
在频域上对信号进行适当处理之后再通过 $\hat F_n x'$ 重新得到时域上被修改过的信号。
频域上的修改会对时域上的信号有怎样的影响已经超出了本文的范畴，推荐信号处理相关书籍。

通过如下的计算可以看出在计算傅立叶变换时可通过分治的思想将
计算复杂度从 $O(n^2)$ 降到 $O(\operatorname{log} n)$，这也快速傅立叶变换 (FFT) 在工程上如此成功的理论基础。

$$
\hat x_k = \sum_{j=0}^{n-1} \omega^{jk} x_j =  
\sum_{j=0}^{n-1} \omega^{2jk} x_{2j} + \sum_{j=0}^{n-1} \omega^{2jk+k} x_{2j+1} 
= \sum_{j=0}^{n-1} \omega^{2jk} x_{2j} + \omega^{k}\sum_{j=0}^{n-1} \omega^{2jk} x_{2j+1} 
$$

至此我们从向量空间的角度理解了离散傅立叶变换，后续的文章我们继续从离散情况出发推广字连续有界
（傅立叶级数 Fourier Series）和连续（傅立叶变换 Fourier Transform）的情况。
