# 1

Let $p' = [u',v',w]$, $P' = [X_W,Y_W,Z_W,1]$
$$
P_C = \begin{bmatrix}
^WR_C^T & -^WR_C^Tt \\
\mathbf{0}^T & 1
\end{bmatrix}
P'
$$

$$
p' = \begin{bmatrix}
fx & 0 &cx\\
0 & fy & cy\\
0 & 0& 1
\end{bmatrix}
P_{C[0:2]}
$$

$$
u = \frac{u'}{w}\\
v = \frac{v'}{w}
$$

# 2

$$
u = \frac{u'}{w} (1 + k_1r^2 + k_2r^4 +k_3r^6) + 2p_1\frac{u'}{w}\frac{v'}{w} +p_2(r^2+2(\frac{u'}{w})^2) \\
v = \frac{v'}{w} (1 + k_1r^2 + k_2r^4 +k_3r^6) + 2p_2\frac{u'}{w}\frac{v'}{w} +p_1(r^2+2(\frac{v'}{w})^2)\\
r = \sqrt{(\frac{u'}{w}-c_x)^2+ (\frac{v'}{w}-c_y)^2}
$$

# 3

