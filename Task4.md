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

$$
u' = u(1 + k_1r^2 + k_2r^4 +k_3r^6) + 2p_1uv +p_2(r^2+2u^2)\\
v' =v(1 + k_1r^2 + k_2r^4 +k_3r^6) + 2p_2uv +p_1(r^2+2v^2)
$$

$$
\begin{bmatrix}
u'\\v'
\end{bmatrix}
=
\begin{bmatrix}
ur^2 & ur^4&ur^6 &2uv &(r^2+2u^2) & u\\
vr^2 & vr^4&vr^6  &(r^2+2v^2)&2uv & v
\end{bmatrix}
\begin{bmatrix}
k_1\\k_2\\k_3\\p_1\\p_2\\1
\end{bmatrix}
$$

Denote:
$$
f(p') = 
\begin{bmatrix}
f_1(u')-u\\
f_2(v’)-v
\end{bmatrix}
$$
Where 
$$
f_1(u') = u'(1 + k_1r^2 + k_2r^4 +k_3r^6) + 2p_1u'v' +p_2(r^2+2u'^2)\\
f_2(v') = v'(1 + k_1r^2 + k_2r^4 +k_3r^6) + 2p_2u'v' +p_1(r^2+2v'^2)\\
r = \sqrt{(u'-c_x)^2+ (v'-c_y)^2}
$$
The Jacobian is 
$$
J(p') = \begin{bmatrix}
\frac{\partial f_1}{\partial u'} & \frac{\partial f_1}{\partial v'}\\
\frac{\partial f_2}{\partial u'} & \frac{\partial f_2}{\partial v'}
\end{bmatrix}\\
\frac{\partial f_1}{\partial u'} = 1+k_1r^2+2k_1u'(u'-c_x)+k_2r^4+4k_2u'r^2(u'-c_x)+k_3r^6+6k_3u'r^4(u'-c_x)+2p_1v'+2p_2(u'-c_x)+4p_2u'\\
\frac{\partial f_1}{\partial v'} = 2u'k_1(v'-c_y)+4u'k_2r^2(v'-c_y)+6u'k_3r^4(v'-c_y)+2p_1u'+2p_2(v'-c_y)\\
\frac{\partial f_2}{\partial u'} = 2v'k_1(u'-c_x)+4v'k_2r^2(u'-c_x)+6v'k_3r^4(u'-c_x)+2p_2v'+2p_1(u'-c_x)\\
\frac{\partial f_2}{\partial v'} = 1+k_1r^2+2k_1v'(v'-c_y)+k_2r^4+4k_2v'r^2(v'-c_y)+k_3r^6+6k_3v'r^4(v'-c_y)+2p_2u'+2p_1(v'-c_y)+4p_1v'\\
$$


Using Gaussian-Newton to get $u'$, $ v'$

1. Initialization: $u' = u$,  $v' = v$
2. For the $k^{th}$ iteration, calculate $J(p'_k)$ and $f(p'_k)$

3. Calculate $$\Delta p_k'$$ , where  $$J^T(p'_k)J(p'_k)\Delta p_k' = -J^T(p'_k)f(p_k')$$

4. If $$\Delta p_k'$$ is small enough, then stop; otherwise, $$p'_{k+1} = p'_k + \Delta p'_k$$

